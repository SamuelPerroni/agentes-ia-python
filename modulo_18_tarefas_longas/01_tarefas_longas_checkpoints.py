"""
============================================================
MÓDULO 18.1 - TAREFAS LONGAS E CHECKPOINTS
============================================================
Neste módulo, aprendemos a lidar com processos que levam
minutos ou horas: como salvar o progresso, como retomar
execuções interrompidas e como acompanhar o avanço em tempo real.

CONCEITO CHAVE:
Um agente que processa 1.000 boletos tem um problema real:
o que acontece se ele travar no boleto 647? Sem checkpoints,
ele precisa recomeçar do zero. Com checkpoints, retoma do 648.

POR QUE CHECKPOINTS SÃO ESSENCIAIS?
- Falhas são inevitáveis: rede cai, API retorna erro 429,
  o servidor é reiniciado, a janela de tempo do Lambda fecha
- Reprocessamento caro: se cada boleto custa 1s de LLM,
  reprocessar 1.000 custa ~16min e dinheiro desnecessário
- Auditoria: o checkpoint registra o estado exato em que
  o processo foi interrompido, facilitando análise post-mortem

ANALOGIA:
Games salvam o progresso em checkpoints para que o jogador
não precise refazer fases inteiras quando morre. O mesmo
princípio se aplica a processos de automação.

ESTADOS DE UM ITEM NUM LOTE:

  ┌──────────┐  processar  ┌──────────────┐  sucesso  ┌────────────┐
  │ pendente │────────────▶│ em_andamento │──────────▶│ concluido  │
  └──────────┘             └──────────────┘           └────────────┘
                                  │
                            erro  │
                                  ▼
                           ┌──────────────┐  retry   ┌──────────┐
                           │    falhou    │──────────▶│ pendente │
                           └──────────────┘           └──────────┘
                                  │
                          max retry│
                                  ▼
                           ┌──────────────┐
                           │  abandonado  │
                           └──────────────┘

FLUXO DO MÓDULO:

  ┌─────────────────────────────────────────────────────────┐
  │  1. Carrega ou cria checkpoint (JSON)                   │
  │  2. Filtra itens JÁ processados (não reprocessa)        │
  │  3. Para cada item pendente:                            │
  │     ├── Marca como em_andamento                        │
  │     ├── Processa (LLM ou tool)                         │
  │     ├── Se ok: marca concluido + salva checkpoint      │
  │     └── Se erro: incrementa tentativas / abandona      │
  │  4. Ao final: exibe relatório de progresso             │
  └─────────────────────────────────────────────────────────┘

Tópicos cobertos:
1. Estrutura de checkpoint persistente em JSON
2. Estados dos itens (pendente, concluido, falhou)
3. Retomada automática após interrupção
4. Retry com limite de tentativas
5. Barra de progresso com Rich
============================================================
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Console do Rich para output formatado
console = Console()


# ============================================================
# 1. ESTRUTURA DO CHECKPOINT
# ============================================================
# O checkpoint é um arquivo JSON no disco que salva o estado
# exato do lote: quais itens foram processados, quais falharam,
# quantas tentativas cada um teve.
#
# QUANDO SALVAR:
# - Após CADA item processado (mais seguro, mais I/O)
# - A cada N itens (balance entre segurança e performance)
# - Em produção: após cada item é o padrão recomendado
#
# ESTRUTURA DO ARQUIVO checkpoint.json:
# {
#   "lote_id": "lote_abril_2026",
#   "criado_em": "2026-04-03T10:00:00",
#   "ultimo_update": "2026-04-03T10:45:00",
#   "itens": {
#     "boleto_001": {
#       "status": "concluido",
#       "tentativas": 1,
#       "resultado": {...},
#       "processado_em": "2026-04-03T10:01:23"
#     },
#     "boleto_002": {
#       "status": "falhou",
#       "tentativas": 3,
#       "ultimo_erro": "TimeoutError: API lenta"
#     },
#     ...
#   }
# }
# ============================================================

@dataclass
class EstadoItem:
    """
    Estado de processamento de um item do lote.

    STATUS:
    - pendente: ainda não processado
    - em_andamento: processamento iniciado (detecta crashes)
    - concluido: processado com sucesso
    - falhou: excedeu o número máximo de tentativas
    """

    item_id: str
    status: str = "pendente"  # pendente | em_andamento | concluido | falhou
    tentativas: int = 0
    resultado: dict[str, Any] = field(default_factory=dict)
    ultimo_erro: str = ""
    processado_em: str = ""


class GerenciadorCheckpoint:
    """
    Persiste e gerencia o estado de um lote de processamento.

    OPERAÇÕES PRINCIPAIS:
    - carregar() → lê o checkpoint existente ou cria novo
    - itens_pendentes() → lista itens que ainda precisam ser processados
    - marcar_em_andamento(id) → evita reprocessamento duplicado
    - marcar_concluido(id, resultado) → salva resultado e persiste
    - marcar_falhou(id, erro) → registra falha e persiste
    """

    def __init__(
        self,
        caminho_arquivo: str,
        lote_id: str,
        max_tentativas: int = 3,
    ) -> None:
        self.caminho = caminho_arquivo
        self.lote_id = lote_id
        self.max_tentativas = max_tentativas
        self.itens: dict[str, EstadoItem] = {}
        self._criado_em = datetime.now().isoformat()
        self._carregar()

    def _carregar(self) -> None:
        """Carrega checkpoint existente ou inicializa vazio."""
        if not os.path.exists(self.caminho):
            return
        with open(self.caminho, encoding="utf-8") as f:
            dados = json.load(f)
        self._criado_em = dados.get("criado_em", self._criado_em)
        for item_id, estado in dados.get("itens", {}).items():
            self.itens[item_id] = EstadoItem(**estado)

        concluidos = sum(
            1 for i in self.itens.values() if i.status == "concluido"
        )
        if concluidos > 0:
            console.print(
                f"  [green]✓ Checkpoint carregado: "
                f"{concluidos} itens já processados[/green]"
            )

    def _persistir(self) -> None:
        """Salva estado atual no arquivo JSON."""
        dados = {
            "lote_id": self.lote_id,
            "criado_em": self._criado_em,
            "ultimo_update": datetime.now().isoformat(),
            "itens": {
                k: asdict(v) for k, v in self.itens.items()
            },
        }
        with open(self.caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

    def registrar_lote(self, item_ids: list[str]) -> None:
        """
        Registra todos os IDs do lote como pendentes.

        Se um ID já existe no checkpoint, não o sobrescreve
        (mantém o estado atual — ex: já concluído).

        Parâmetros:
        - item_ids: lista com todos os IDs do lote completo
        """
        for item_id in item_ids:
            if item_id not in self.itens:
                self.itens[item_id] = EstadoItem(item_id=item_id)
        self._persistir()

    def itens_pendentes(self) -> list[str]:
        """
        Retorna IDs dos itens que ainda precisam ser processados.

        INCLUI:
        - status == "pendente"
        - status == "em_andamento" (possível crash anterior)
          com tentativas < max_tentativas

        EXCLUI:
        - status == "concluido"
        - status == "falhou" (tentativas esgotadas)
        """
        pendentes = []
        for item_id, estado in self.itens.items():
            if estado.status in ("pendente", "em_andamento"):
                if estado.tentativas < self.max_tentativas:
                    pendentes.append(item_id)
        return pendentes

    def marcar_em_andamento(self, item_id: str) -> None:
        """
        Marca item como em andamento antes de processar.

        Isso permite detectar itens que ficaram "presos"
        caso o processo seja interrompido abruptamente.
        """
        item = self.itens[item_id]
        item.status = "em_andamento"
        item.tentativas += 1
        self._persistir()

    def marcar_concluido(
        self,
        item_id: str,
        resultado: dict[str, Any],
    ) -> None:
        """
        Marca item como concluído e salva o resultado.

        Parâmetros:
        - item_id: ID do item processado
        - resultado: dicionário com o resultado do processamento
        """
        item = self.itens[item_id]
        item.status = "concluido"
        item.resultado = resultado
        item.processado_em = datetime.now().isoformat()
        self._persistir()

    def marcar_falhou(self, item_id: str, erro: str) -> None:
        """
        Registra falha e marca como abandonado se esgotou tentativas.

        Parâmetros:
        - item_id: ID do item que falhou
        - erro: descrição do erro para diagnóstico
        """
        item = self.itens[item_id]
        item.ultimo_erro = erro
        if item.tentativas >= self.max_tentativas:
            item.status = "falhou"
            console.print(
                f"  [red]✗ {item_id}: abandonado após "
                f"{item.tentativas} tentativas[/red]"
            )
        else:
            item.status = "pendente"  # volta para retry
        self._persistir()

    def relatorio(self) -> None:
        """Exibe tabela com o estado final do processamento."""
        contadores: dict[str, int] = {}
        for item in self.itens.values():
            contadores[item.status] = (
                contadores.get(item.status, 0) + 1
            )

        tabela = Table(title=f"Relatório do Lote: {self.lote_id}")
        tabela.add_column("Status", style="bold")
        tabela.add_column("Quantidade")
        tabela.add_column("%")
        total = len(self.itens)
        for status, qtd in sorted(contadores.items()):
            cor = {
                "concluido": "green",
                "falhou": "red",
                "pendente": "yellow",
                "em_andamento": "cyan",
            }.get(status, "white")
            tabela.add_row(
                f"[{cor}]{status}[/{cor}]",
                str(qtd),
                f"{qtd / total * 100:.1f}%" if total else "0%",
            )
        tabela.add_row("[bold]TOTAL[/bold]", str(total), "100%")
        console.print(tabela)


# ============================================================
# 2. PROCESSADOR COM CHECKPOINT
# ============================================================

def _processar_boleto_simulado(boleto_id: str) -> dict[str, Any]:
    """
    Simula o processamento de um boleto pelo agente.

    Em produção, substitua por:
        from modulo_06_agente_boletos.agente_boletos import AgenteBoletos
        agente = AgenteBoletos()
        return agente.processar_mensagem(texto_do_boleto)

    Simula falhas aleatórias (20% de chance) para demonstrar
    o mecanismo de retry.

    Parâmetros:
    - boleto_id: identificador do boleto a processar

    Retorna:
    - Dicionário com resultado da extração

    Levanta:
    - RuntimeError com 20% de probabilidade (simula falha de API)
    """
    time.sleep(0.05)  # simula latência do LLM
    if random.random() < 0.20:  # 20% de chance de falha
        raise RuntimeError(f"API timeout ao processar {boleto_id}")
    return {
        "boleto_id": boleto_id,
        "valor": round(random.uniform(100, 5000), 2),
        "banco": random.choice(["Bradesco", "Itaú", "Santander"]),
        "status": "extraido_com_sucesso",
    }


def processar_lote_com_checkpoint(
    lote_id: str,
    boleto_ids: list[str],
    caminho_checkpoint: str,
    max_tentativas: int = 3,
) -> GerenciadorCheckpoint:
    """
    Processa um lote completo com checkpoint e barra de progresso.

    Se o checkpoint já existir (execução anterior interrompida),
    retoma automaticamente do ponto onde parou.

    Parâmetros:
    - lote_id: nome do lote (ex: "boletos_abril_2026")
    - boleto_ids: lista completa de IDs a processar
    - caminho_checkpoint: onde salvar o arquivo de checkpoint
    - max_tentativas: máximo de tentativas por item antes de abandonar

    Retorna:
    - GerenciadorCheckpoint com o estado final do lote
    """
    checkpoint = GerenciadorCheckpoint(
        caminho_checkpoint, lote_id, max_tentativas
    )
    checkpoint.registrar_lote(boleto_ids)

    pendentes = checkpoint.itens_pendentes()
    total_pendentes = len(pendentes)

    if total_pendentes == 0:
        console.print("[green]✅ Todos os itens já processados![/green]")
        return checkpoint

    console.print(
        f"\n[bold]{total_pendentes} itens pendentes para processar"
        f"[/bold]\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        tarefa = progress.add_task(
            f"Processando {lote_id}...",
            total=total_pendentes,
        )

        for boleto_id in pendentes:
            checkpoint.marcar_em_andamento(boleto_id)
            try:
                resultado = _processar_boleto_simulado(boleto_id)
                checkpoint.marcar_concluido(boleto_id, resultado)
            except (RuntimeError, OSError) as exc:
                checkpoint.marcar_falhou(boleto_id, str(exc))
            progress.advance(tarefa)

    return checkpoint


# ============================================================
# DEMO COMPLETA — Processamento com checkpoint e retomada
# ============================================================

def demo_checkpoints() -> None:
    """
    Demonstra checkpoints com simulação de interrupção e retomada.

    ETAPAS:
    1. Cria lote de 20 boletos e processa a primeira metade
       (simula processo interrompido na metade)
    2. Salva o checkpoint
    3. Retoma o processamento — só processa os pendentes
    4. Exibe relatório final com status de cada item

    OBSERVE NO OUTPUT:
    - Na retomada, os itens já concluídos são pulados
    - Itens que falharam são retentados até max_tentativas
    - A barra de progresso mostra apenas os pendentes reais

    EXERCÍCIO SUGERIDO:
    1. Ajuste a probabilidade de falha para 50% e veja os retries
    2. Interrompa o script com Ctrl+C e rode novamente
       (o checkpoint garante que não perde o progresso)
    3. Adicione um campo "custo_tokens" ao resultado e some
       o custo total no relatório
    """
    console.print(Panel.fit(
        "[bold]Tarefas Longas com Checkpoints[/bold]\n"
        "Processamento de lote com retomada automática",
        title="💾 Módulo 18 — Checkpoints",
        border_style="yellow",
    ))

    # Arquivo de checkpoint temporário para a demo
    caminho_ckpt = os.path.join(
        os.path.dirname(__file__),
        "demo_checkpoint.json",
    )

    # Remove checkpoint anterior para a demo ser reprodutível
    if os.path.exists(caminho_ckpt):
        os.remove(caminho_ckpt)

    # ── FASE 1: Processa parcialmente (simula interrupção) ────
    console.print("\n[bold]Fase 1: Processamento parcial[/bold]")
    boleto_ids = [f"boleto_{i:03d}" for i in range(1, 21)]

    # Só registra o lote mas processa apenas os primeiros 10
    checkpoint = GerenciadorCheckpoint(
        caminho_ckpt, "lote_demo_abril", 3
    )
    checkpoint.registrar_lote(boleto_ids)

    # Processa manualmente os primeiros 10 (simula interrupção)
    for bid in boleto_ids[:10]:
        checkpoint.marcar_em_andamento(bid)
        try:
            resultado = _processar_boleto_simulado(bid)
            checkpoint.marcar_concluido(bid, resultado)
        except (RuntimeError, OSError) as exc:
            checkpoint.marcar_falhou(bid, str(exc))

    concluidos = sum(
        1 for i in checkpoint.itens.values()
        if i.status == "concluido"
    )
    console.print(
        f"\n[yellow]⚡ Processo interrompido. "
        f"{concluidos} itens processados.[/yellow]"
    )

    # ── FASE 2: Retomada automática ───────────────────────────
    console.print("\n[bold]Fase 2: Retomada automática[/bold]")
    checkpoint_final = processar_lote_com_checkpoint(
        lote_id="lote_demo_abril",
        boleto_ids=boleto_ids,
        caminho_checkpoint=caminho_ckpt,
        max_tentativas=3,
    )

    # ── RELATÓRIO FINAL ───────────────────────────────────────
    console.print("\n[bold]Relatório final do lote:[/bold]")
    checkpoint_final.relatorio()

    console.print("\n💡 [bold yellow]Dica:[/bold yellow]")
    console.print(
        f"  O checkpoint está em: {caminho_ckpt}\n"
        "  Abra o JSON e veja o estado exato de cada boleto."
    )


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    demo_checkpoints()
