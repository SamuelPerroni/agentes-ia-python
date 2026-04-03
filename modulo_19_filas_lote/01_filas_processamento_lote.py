"""
============================================================
MÓDULO 19.1 - FILAS E PROCESSAMENTO EM LOTE
============================================================
Neste módulo, aprendemos a processar grandes volumes de boletos
com controle de prioridade, rate limiting e paralelismo seguro.

CONCEITO CHAVE:
Em automação de processos, o agente raramente processa uma
mensagem por vez em produção. A realidade é um fluxo contínuo:
100 boletos chegam ao mesmo tempo, a API de LLM aceita no máximo
30 requisições por minuto, e alguns boletos são urgentes.

FILA vs. PROCESSAMENTO SEQUENCIAL:
- Sequencial: processa um, espera concluir, processa o próximo
  Simples, mas o tempo total = N × tempo_por_item
- Fila: desacopla a chegada dos itens do processamento
  Permite: prioridade, rate limiting, retry, dead letter queue

PADRÕES COBERTOS:
1. Fila de prioridade — boletos urgentes na frente
2. Rate limiter — respeita o limite de chamadas por minuto da API
3. Worker pool — processa N itens em paralelo com controle
4. Dead letter queue — itens que falharam após N tentativas

ARQUITETURA DE FILAS PARA AGENTES:

  ┌──────────────────────────────────────────────────────────┐
  │  Produtores                  Consumidores                │
  │  (fontes de dados)           (agente workers)            │
  │                                                          │
  │  E-mail ──▶ ┌──────────────┐                             │
  │  API    ──▶ │ Fila         │ ──▶ Worker 1 ──▶ LLM        │
  │  Portal ──▶ │ Prioridade   │ ──▶ Worker 2 ──▶ LLM        │
  │  Upload ──▶ │ Alta:Urgente │ ──▶ Worker 3 ──▶ LLM        │
  │             │ Média:Normal │                  │          │
  │             │ Baixa:Batch  │              Resultado      │
  │             └──────────────┘                  │          │
  │                    │                          ▼          │
  │              Itens falhos ──▶ Dead Letter Queue          │
  └──────────────────────────────────────────────────────────┘

FILAS EM PRODUÇÃO:
- Redis (via rq ou celery)   → simples, rápido, self-hosted
- Azure Service Bus          → gerenciado, com DLQ nativa
- AWS SQS                    → gerenciado, escalável
- RabbitMQ                   → potente, complexo de operar
- In-memory (este módulo)    → suficiente para ensinar o conceito

Tópicos cobertos:
1. Fila de prioridade (heapq) com níveis Alta/Média/Baixa
2. Rate limiter token bucket
3. Processamento paralelo com ThreadPoolExecutor
4. Dead letter queue para itens que excederam retries
5. Métricas de processamento em lote
============================================================
"""

from __future__ import annotations

import heapq
import random
import time
import concurrent.futures as _cf  # type: ignore
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from threading import Lock
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Console do Rich para output formatado
console = Console()


# ============================================================
# 1. PRIORIDADE E ITEM DE FILA
# ============================================================
# Usamos IntEnum para que a comparação numérima funcione
# com heapq de forma natural: valores menores = mais urgente.
# ============================================================

class Prioridade(IntEnum):
    """
    Nível de prioridade para itens na fila.

    ALTA (1) é processada antes de MEDIA (2) e BAIXA (3).
    Números menores têm prioridade em heapq (min-heap).

    QUANDO USAR CADA NÍVEL:
    - ALTA: boletos vencendo hoje, reclamações de cliente
    - MEDIA: boletos com vencimento próximo (< 5 dias)
    - BAIXA: processamento em lote noturno, reconciliação
    """
    ALTA = 1
    MEDIA = 2
    BAIXA = 3


@dataclass(order=True)
class ItemFila:
    """
    Representa um item na fila de processamento.

    O campo `prioridade` controla a ordem no heap.
    Os demais campos são ignorados na comparação (compare=False).

    ATRIBUTOS:
    - prioridade: Prioridade.ALTA/MEDIA/BAIXA
    - item_id: identificador único do boleto/documento
    - payload: dados do item a ser processado
    - tentativas: contador de tentativas (para retry)
    - criado_em: timestamp de entrada na fila
    """

    prioridade: Prioridade
    item_id: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False)
    tentativas: int = field(default=0, compare=False)
    criado_em: str = field(
        default_factory=lambda: datetime.now().isoformat(),
        compare=False,
    )


# ============================================================
# 2. FILA DE PRIORIDADE THREAD-SAFE
# ============================================================
# heapq não é thread-safe por padrão. Usamos Lock para garantir
# que múltiplas threads possam acessar a fila sem race conditions.
# ============================================================

class FilaPrioridade:
    """
    Fila de prioridade thread-safe usando heapq + Lock.

    OPERAÇÕES:
    - enfileirar(item): adiciona à fila respeitando prioridade
    - desenfileirar(): remove e retorna o item de maior prioridade
    - tamanho(): número de itens na fila
    - vazia(): True se não há itens
    """

    def __init__(self) -> None:
        self._heap: list[ItemFila] = []
        self._lock = Lock()

    def enfileirar(self, item: ItemFila) -> None:
        """
        Adiciona item à fila com prioridade correta.

        Thread-safe: usa lock para evitar race conditions
        em ambientes com múltiplos workers.

        Parâmetros:
        - item: ItemFila a adicionar
        """
        with self._lock:
            heapq.heappush(self._heap, item)

    def desenfileirar(self) -> ItemFila | None:
        """
        Remove e retorna o item de maior prioridade.

        Retorna None se a fila estiver vazia.

        IMPORTANTE: o item sai da fila imediatamente.
        Se o processamento falhar, re-enfileire o item
        (com tentativas + 1) para retry.
        """
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)

    def tamanho(self) -> int:
        """Retorna o número de itens na fila."""
        with self._lock:
            return len(self._heap)

    def vazia(self) -> bool:
        """Retorna True se a fila estiver vazia."""
        return self.tamanho() == 0


# ============================================================
# 3. RATE LIMITER — Token Bucket
# ============================================================
# O algoritmo Token Bucket é o mais usado para rate limiting:
# - Um "balde" tem capacidade de N tokens
# - Cada chamada à API consome 1 token
# - O balde é reabastecido a uma taxa constante (ex: 30/min)
# - Se o balde estiver vazio, a chamada espera
#
# EQUIVALENTE NA PRÁTICA:
# Groq API tem limite de 30 req/min no plano gratuito.
# Com o rate limiter, o agente nunca excede esse limite
# independente de quantos workers estão rodando em paralelo.
# ============================================================

class RateLimiter:
    """
    Rate limiter usando algoritmo Token Bucket.

    PARÂMETROS DE CONFIGURAÇÃO (exemplos reais):
    - Groq gratuito:     30 req/min  → max_tokens=30, janela_s=60
    - OpenAI Tier 1:    500 req/min  → max_tokens=500, janela_s=60
    - Azure OpenAI:    1000 req/min  → max_tokens=1000, janela_s=60

    USO:
        limiter = RateLimiter(max_tokens=30, janela_s=60)
        limiter.consumir()  # bloqueia se necessário
        resultado = chamar_api()
    """

    def __init__(
        self,
        max_tokens: int,
        janela_s: float = 60.0,
    ) -> None:
        self.max_tokens = max_tokens
        self.janela_s = janela_s
        self._tokens_disponiveis = float(max_tokens)
        self._ultimo_reabastecimento = time.monotonic()
        self._lock = Lock()

    def _reabastecer(self) -> None:
        """
        Reabastece tokens proporcionalmente ao tempo decorrido.

        Chamado internamente antes de cada consumo.
        Taxa = max_tokens / janela_s tokens por segundo.
        """
        agora = time.monotonic()
        decorrido = agora - self._ultimo_reabastecimento
        taxa = self.max_tokens / self.janela_s
        novos = decorrido * taxa
        self._tokens_disponiveis = min(
            self.max_tokens,
            self._tokens_disponiveis + novos,
        )
        self._ultimo_reabastecimento = agora

    def consumir(self, tokens: int = 1) -> float:
        """
        Consome tokens da reserva, bloqueando se necessário.

        Se não há tokens suficientes, calcula o tempo de espera
        e dorme até reabastecimento suficiente.

        Parâmetros:
        - tokens: número de tokens a consumir (default 1)

        Retorna:
        - Tempo de espera em segundos (0 se não precisou esperar)
        """
        with self._lock:
            self._reabastecer()
            if self._tokens_disponiveis >= tokens:
                self._tokens_disponiveis -= tokens
                return 0.0
            # Calcula espera necessária
            deficit = tokens - self._tokens_disponiveis
            taxa = self.max_tokens / self.janela_s
            espera = deficit / taxa
        # Dorme FORA do lock para não bloquear outras threads
        time.sleep(espera)
        with self._lock:
            self._reabastecer()
            self._tokens_disponiveis -= tokens
        return espera


# ============================================================
# 4. WORKER E DEAD LETTER QUEUE
# ============================================================

@dataclass
class ResultadoProcessamento:
    """
    Resultado do processamento de um item da fila.

    STATUS:
    - sucesso: item processado corretamente
    - falha_transitoria: erro recuperável (retry)
    - falha_permanente: enviado para dead letter queue
    """

    item_id: str
    status: str
    resultado: dict[str, Any] = field(default_factory=dict)
    erro: str = ""
    tempo_espera_rate_ms: float = 0.0
    tentativa: int = 1


def _processar_item_simulado(
    item: ItemFila,
) -> dict[str, Any]:
    """
    Simula o processamento de um boleto pelo agente.

    Em produção, substitua por chamada real ao agente.
    Simula 15% de chance de falha transitória.

    Parâmetros:
    - item: ItemFila com os dados do boleto

    Retorna:
    - Dicionário com resultado da extração

    Levanta:
    - RuntimeError com 15% de probabilidade
    """
    time.sleep(random.uniform(0.02, 0.08))  # simula latência LLM
    if random.random() < 0.15:  # 15% de chance de falha
        raise RuntimeError(
            f"Timeout ao processar {item.item_id}"
        )
    return {
        "item_id": item.item_id,
        "valor": round(random.uniform(100, 8000), 2),
        "prioridade_original": item.prioridade.name,
        "processado_em": datetime.now().isoformat(),
    }


class ProcessadorEmLote:
    """
    Processa itens de uma FilaPrioridade com rate limiting e DLQ.

    CONFIGURAÇÃO:
    - max_workers: paralelismo (threads simultâneas)
    - rate_limiter: controla chamadas por minuto
    - max_tentativas: antes de ir para dead letter queue

    DEAD LETTER QUEUE (DLQ):
    Itens que falharam após max_tentativas são movidos para
    a DLQ para análise manual. Em produção, persista a DLQ
    em banco ou arquivo para investigação posterior.
    """

    def __init__(
        self,
        max_workers: int = 3,
        rate_limiter: RateLimiter | None = None,
        max_tentativas: int = 3,
    ) -> None:
        self.max_workers = max_workers
        self.rate_limiter = rate_limiter or RateLimiter(
            max_tokens=60, janela_s=60
        )
        self.max_tentativas = max_tentativas
        self.dead_letter_queue: list[ItemFila] = []
        self._resultados: list[ResultadoProcessamento] = []
        self._lock = Lock()

    def processar(
        self,
        fila: FilaPrioridade,
        funcao_processar: Callable[[ItemFila], dict[str, Any]],
    ) -> list[ResultadoProcessamento]:
        """
        Processa todos os itens da fila com workers paralelos.

        FLUXO POR ITEM:
        1. Rate limiter: espera se necessário
        2. Chama funcao_processar(item)
        3. Sucesso: adiciona a resultados
        4. Erro + tentativas < max: re-enfileira com +1 tentativa
        5. Erro + tentativas >= max: move para DLQ

        Parâmetros:
        - fila: FilaPrioridade com os itens a processar
        - funcao_processar: função que processa um ItemFila

        Retorna:
        - Lista de ResultadoProcessamento com todos os resultados
        """
        itens_snapshot: list[ItemFila] = []
        while not fila.vazia():
            item = fila.desenfileirar()
            if item:
                itens_snapshot.append(item)

        with _cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futuros = {
                pool.submit(
                    self._processar_com_retry, item, funcao_processar
                ): item
                for item in itens_snapshot
            }
            for futuro in _cf.as_completed(futuros):
                resultado = futuro.result()
                if resultado:
                    with self._lock:
                        self._resultados.append(resultado)

        return self._resultados

    def _processar_com_retry(
        self,
        item: ItemFila,
        funcao: Callable[[ItemFila], dict[str, Any]],
    ) -> ResultadoProcessamento | None:
        """
        Processa um item com retry automático e rate limiting.

        Parâmetros:
        - item: ItemFila a processar
        - funcao: função de processamento

        Retorna:
        - ResultadoProcessamento ou None se movido para DLQ
        """
        for tentativa in range(1, self.max_tentativas + 1):
            espera = self.rate_limiter.consumir()
            try:
                resultado = funcao(item)
                return ResultadoProcessamento(
                    item_id=item.item_id,
                    status="sucesso",
                    resultado=resultado,
                    tempo_espera_rate_ms=espera * 1000,
                    tentativa=tentativa,
                )
            except (RuntimeError, OSError) as exc:
                if tentativa >= self.max_tentativas:
                    # Esgotou tentativas → Dead Letter Queue
                    with self._lock:
                        self.dead_letter_queue.append(item)
                    return ResultadoProcessamento(
                        item_id=item.item_id,
                        status="falha_permanente",
                        erro=str(exc),
                        tentativa=tentativa,
                    )
                # Backoff antes de retry
                time.sleep(0.1 * tentativa)
        return None

    def exibir_relatorio(self) -> None:
        """Exibe tabela com resultado do processamento em lote."""
        sucessos = sum(
            1 for r in self._resultados if r.status == "sucesso"
        )
        falhas = len(self.dead_letter_queue)
        total = len(self._resultados)

        console.print(Panel.fit(
            f"[green]✓ Sucesso:[/green] {sucessos}\n"
            f"[red]✗ Dead Letter Queue:[/red] {falhas}\n"
            f"[bold]Total:[/bold] {total}",
            title="📊 Resultado do Processamento em Lote",
            border_style="green" if falhas == 0 else "yellow",
        ))

        if self.dead_letter_queue:
            tabela_dlq = Table(title="Dead Letter Queue — Revisar")
            tabela_dlq.add_column("Item ID", style="red")
            tabela_dlq.add_column("Prioridade")
            tabela_dlq.add_column("Tentativas")
            for item in self.dead_letter_queue:
                tabela_dlq.add_row(
                    item.item_id,
                    item.prioridade.name,
                    str(item.tentativas),
                )
            console.print(tabela_dlq)


# ============================================================
# DEMO COMPLETA — Fila de prioridade com rate limiting
# ============================================================

def demo_filas() -> None:
    """
    Demonstra processamento em lote com fila de prioridade.

    ETAPAS:
    1. Cria 15 boletos com diferentes prioridades
    2. Enfileira todos (itens ALTA chegam na frente)
    3. Processa com 3 workers paralelos + rate limiter
    4. Mostra relatório com sucessos e DLQ

    OBSERVE NO OUTPUT:
    - Itens ALTA são processados antes mesmo de chegarem depois
    - O rate limiter registra o tempo de espera quando ativo
    - Itens que falharam 3 vezes vão para a DLQ

    EXERCÍCIO SUGERIDO:
    1. Aumente max_tokens para 60/min e veja o impacto na latência
    2. Mude max_tentativas para 1 e observe mais itens na DLQ
    3. Substitua _processar_item_simulado por uma chamada real
       ao agente de boletos do módulo 6
    """
    console.print(Panel.fit(
        "[bold]Filas e Processamento em Lote[/bold]\n"
        "Prioridade + Rate Limiting + Workers Paralelos",
        title="📬 Módulo 19 — Filas",
        border_style="green",
    ))

    # Cria e preenche a fila com boletos de diferentes prioridades
    fila = FilaPrioridade()

    boletos = [
        # 5 urgentes (vencendo hoje)
        *[
            ItemFila(
                prioridade=Prioridade.ALTA,
                item_id=f"urgente_{i:02d}",
                payload={"descricao": "Vence hoje"},
            )
            for i in range(1, 6)
        ],
        # 7 normais
        *[
            ItemFila(
                prioridade=Prioridade.MEDIA,
                item_id=f"normal_{i:02d}",
                payload={"descricao": "Vence em 3 dias"},
            )
            for i in range(1, 8)
        ],
        # 3 em lote (baixa prioridade)
        *[
            ItemFila(
                prioridade=Prioridade.BAIXA,
                item_id=f"lote_{i:02d}",
                payload={"descricao": "Processamento noturno"},
            )
            for i in range(1, 4)
        ],
    ]

    # Embaralha para mostrar que a fila reordena por prioridade
    random.shuffle(boletos)
    for boleto in boletos:
        fila.enfileirar(boleto)

    console.print(
        f"\n[bold]{fila.tamanho()} boletos na fila[/bold] "
        f"(adicionados em ordem aleatória, processados por prioridade)"
    )

    # Configura o processador
    # Rate limiter: 60 req/min (suficiente para a demo ser rápida)
    processador = ProcessadorEmLote(
        max_workers=3,
        rate_limiter=RateLimiter(max_tokens=60, janela_s=60),
        max_tentativas=3,
    )

    # Processa toda a fila
    console.print(
        "\n[dim]Processando com 3 workers paralelos...[/dim]"
    )
    inicio = time.monotonic()
    processador.processar(fila, _processar_item_simulado)
    duracao = time.monotonic() - inicio

    console.print(f"\n[dim]Tempo total: {duracao:.1f}s[/dim]")
    processador.exibir_relatorio()

    console.print("\n💡 [bold yellow]Dica:[/bold yellow]")
    console.print(
        "  Em produção, substitua FilaPrioridade por Redis + rq\n"
        "  ou Azure Service Bus para ter fila persistente e\n"
        "  processamento em múltiplos servidores."
    )


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    demo_filas()
