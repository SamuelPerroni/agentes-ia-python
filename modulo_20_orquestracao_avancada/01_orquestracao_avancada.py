"""
============================================================
MÓDULO 20.1 - ORQUESTRAÇÃO AVANÇADA MULTI-AGENTE
============================================================
Neste módulo, aprendemos a coordenar múltiplos agentes
especializados com um supervisor que gerencia o fluxo,
handoff de contexto e recuperação de falhas.

CONCEITO CHAVE:
No módulo 8, vimos o padrão multiagente de forma introdutória.
Aqui aprofundamos: como um supervisor decide QUAL agente chamar,
como o contexto é passado de um agente para o próximo (handoff),
e como a orquestração garante coerência mesmo quando um agente falha.

POR QUE MÚLTIPLOS AGENTES?
Um agente único sofre de "cognitive overload": quanto mais
responsabilidades tem, pior o desempenho em cada uma.
Especialização é o princípio do desenvolvimento em equipe aplicado
a agentes de IA.

ANALOGIA:
Um departamento financeiro tem:
- Recepcionista: triagem e direcionamento
- Analista de boletos: extração e validação
- Calculista: cálculos financeiros complexos
- Compliance: verificação de regras e limites
- Supervisor: aprovação de valores altos

Cada papel tem escopo reduzido = melhor desempenho e auditoria.

PADRÃO SUPERVISOR + WORKERS:

  ┌─────────────────────────────────────────────────────────┐
  │                     SUPERVISOR                          │
  │  ┌──────────────────────────────────────────────────┐  │
  │  │  1. Analisa a tarefa                             │  │
  │  │  2. Decide qual worker tem competência           │  │
  │  │  3. Envia com contexto (handoff)                 │  │
  │  │  4. Recebe resultado + valida                    │  │
  │  │  5. Se erro → tenta outro worker ou escala       │  │
  │  └──────────────────────────────────────────────────┘  │
  │         │                                               │
  │    ┌────▼────┐   ┌──────────┐   ┌──────────┐          │
  │    │ Worker  │   │  Worker  │   │  Worker  │          │
  │    │Extração │   │ Cálculo  │   │Compliance│          │
  │    └─────────┘   └──────────┘   └──────────┘          │
  └─────────────────────────────────────────────────────────┘

HANDOFF — Transferência de Contexto:
O handoff garante que cada agente receba TODO o contexto
necessário sem precisar buscar informações por conta própria.
É um "prontuário de contexto" que acompanha a tarefa.

  Tarefa inicial
       │
       ▼
  Supervisor cria Contexto {tarefa, histórico, metadados}
       │
       ├──▶ Worker A recebe Contexto + adiciona resultado_a
       │
       └──▶ Worker B recebe Contexto (atualizado) + processa

Tópicos cobertos:
1. Agente Worker com papel e capacidades definidas
2. Contexto de handoff estruturado
3. Supervisor que decide o roteamento dinamicamente
4. Recuperação de falha: fallback para worker alternativo
5. Auditoria: histórico de decisões do supervisor
============================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# Console do Rich para output formatado
console = Console()


# ============================================================
# 1. CONTEXTO DE HANDOFF
# ============================================================
# O contexto percorre TODA a cadeia de agentes, acumulando
# o trabalho de cada um. É o "prontuário" da tarefa.
#
# CAMPOS CHAVE:
# - tarefa_original: a solicitação inicial (não muda)
# - dados_coletados: resultados acumulados dos workers
# - historico_decisoes: rastro de auditoria do supervisor
# - status: estado atual da tarefa na orquestração
# ============================================================

@dataclass
class ContextoHandoff:
    """
    Contexto compartilhado que percorre a cadeia de agentes.

    IMUTÁVEL: tarefa_original nunca é alterada após criação.
    MUTÁVEL: dados_coletados e historico_decisoes crescem a cada worker.

    BOAS PRÁTICAS:
    - Não remova dados antigos (auditoria)
    - Adicione timestamp em cada atualização
    - Propague erros de workers anteriores
    """

    tarefa_id: str
    tarefa_original: str
    dados_coletados: dict[str, Any] = field(default_factory=dict)
    historico_decisoes: list[dict[str, Any]] = field(
        default_factory=list
    )
    status: str = "iniciado"  # iniciado | em_andamento | concluido | falhou
    criado_em: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def registrar_decisao(
        self,
        agente: str,
        acao: str,
        detalhes: str = "",
    ) -> None:
        """
        Registra uma decisão no histórico de auditoria.

        Cada entrada contém quem decidiu, o quê e quando.
        Este histórico é fundamental para compliance e debugging.

        Parâmetros:
        - agente: nome do agente que tomou a decisão
        - acao: descrição curta da ação (ex: "roteou para worker")
        - detalhes: informações adicionais opcionais
        """
        self.historico_decisoes.append({
            "timestamp": datetime.now().isoformat(),
            "agente": agente,
            "acao": acao,
            "detalhes": detalhes,
        })


# ============================================================
# 2. AGENTE WORKER BASE
# ============================================================
# Cada worker tem:
# - papel: descrição do que ele faz
# - capacidades: lista de tipos de tarefa que sabe tratar
# - processar(): implementa a lógica especializada
#
# EM PRODUÇÃO:
# Cada worker pode ter seu próprio system_prompt, modelo LLM,
# ferramentas e guardrails. O supervisor não precisa saber
# os detalhes internos — só chama processar().
# ============================================================

class WorkerBase:
    """
    Classe base para todos os workers especializados.

    CONTRATO:
    - processar(contexto) deve retornar o contexto atualizado
    - processar NUNCA deve lançar exceção para falhas esperadas
      → use contexto.status = "falhou" para comunicar erros
    - processar PODE lançar exceção para falhas inesperadas
      (o supervisor captura e tenta outro worker)
    """

    def __init__(self, nome: str, papel: str) -> None:
        self.nome = nome
        self.papel = papel

    def processar(
        self,
        contexto: ContextoHandoff,
    ) -> ContextoHandoff:
        """
        Processa a tarefa e atualiza o contexto.

        Subclasses devem sobrescrever este método.
        """
        raise NotImplementedError


# ============================================================
# 3. WORKERS ESPECIALIZADOS
# ============================================================

class WorkerExtracao(WorkerBase):
    """
    Worker especializado em extração de dados de boletos.

    RESPONSABILIDADE:
    - Lê o texto do boleto (da tarefa_original)
    - Extrai: valor, banco, vencimento, pagador, beneficiário
    - Armazena em dados_coletados["extracao"]
    - NÃO calcula juros, NÃO valida regras de negócio

    EM PRODUÇÃO:
    Este worker teria um system_prompt de extração estruturada
    e usaria structured output (Pydantic) para garantir o JSON.
    """

    def __init__(self) -> None:
        super().__init__(
            nome="WorkerExtracao",
            papel="Extrai dados estruturados do texto do boleto",
        )

    def processar(
        self,
        contexto: ContextoHandoff,
    ) -> ContextoHandoff:
        """
        Extrai dados do texto do boleto na tarefa original.

        Simula chamada ao LLM com structured output.
        Em produção: usa Groq + schema Pydantic BoletoExtraido.
        """
        console.print(
            f"  [cyan]→ {self.nome}: extraindo dados...[/cyan]"
        )
        time.sleep(0.05)  # simula latência LLM

        # Simula extração — em produção: chamada real ao LLM
        tarefa = contexto.tarefa_original.lower()
        valor = 0.0
        if "r$" in tarefa:
            # Extrai o valor da string (simplificado para demo)
            partes = tarefa.split("r$")
            if len(partes) > 1:
                try:
                    valor_str = (
                        partes[1].strip().split()[0]
                        .replace(".", "")
                        .replace(",", ".")
                    )
                    valor = float(valor_str)
                except (ValueError, IndexError):
                    valor = 0.0

        banco = "Desconhecido"
        for b in ["bradesco", "itaú", "santander", "caixa"]:
            if b in tarefa:
                banco = b.title()
                break

        vencido = "vencido" in tarefa or "venceu" in tarefa

        contexto.dados_coletados["extracao"] = {
            "valor": valor or 1500.00,
            "banco": banco,
            "vencido": vencido,
            "dias_atraso": 7 if vencido else 0,
        }
        contexto.registrar_decisao(
            agente=self.nome,
            acao="dados_extraidos",
            detalhes=f"valor={valor or 1500.00}, banco={banco}",
        )
        return contexto


class WorkerCalculo(WorkerBase):
    """
    Worker especializado em cálculos financeiros.

    RESPONSABILIDADE:
    - Lê dados_coletados["extracao"]
    - Calcula multa (2%), juros (0,033%/dia), total atualizado
    - Armazena em dados_coletados["calculo"]
    - Depende do WorkerExtracao ter rodado antes

    EM PRODUÇÃO:
    Este worker usaria a tool calcular_multa_juros do módulo 3.
    """

    def __init__(self) -> None:
        super().__init__(
            nome="WorkerCalculo",
            papel="Calcula multa e juros para boletos vencidos",
        )

    def processar(
        self,
        contexto: ContextoHandoff,
    ) -> ContextoHandoff:
        """
        Calcula valores atualizados com base nos dados extraídos.

        Pré-requisito: WorkerExtracao deve ter rodado antes.
        """
        extracao = contexto.dados_coletados.get("extracao", {})
        if not extracao:
            contexto.registrar_decisao(
                agente=self.nome,
                acao="erro_dependencia",
                detalhes="extracao não disponível, abortando",
            )
            return contexto

        console.print(
            f"  [cyan]→ {self.nome}: calculando valores...[/cyan]"
        )
        time.sleep(0.03)  # simula processamento

        valor = extracao.get("valor", 0.0)
        dias = extracao.get("dias_atraso", 0)

        if dias > 0:
            multa = valor * 0.02
            juros = valor * 0.00033 * dias
            total = valor + multa + juros
        else:
            multa = 0.0
            juros = 0.0
            total = valor

        contexto.dados_coletados["calculo"] = {
            "valor_original": valor,
            "multa": round(multa, 2),
            "juros": round(juros, 2),
            "total_atualizado": round(total, 2),
            "dias_atraso": dias,
        }
        contexto.registrar_decisao(
            agente=self.nome,
            acao="calculo_concluido",
            detalhes=f"total={total:.2f}",
        )
        return contexto


class WorkerCompliance(WorkerBase):
    """
    Worker especializado em regras de negócio e compliance.

    RESPONSABILIDADE:
    - Verifica se o valor exige aprovação humana
    - Checa se o pagador está em lista de restrições
    - Aplica políticas de cobrança específicas
    - Armazena em dados_coletados["compliance"]

    EM PRODUÇÃO:
    Este worker integraria com sistemas de análise de crédito,
    listas de CNPJ bloqueados e limites por alçada.
    """

    def __init__(self) -> None:
        super().__init__(
            nome="WorkerCompliance",
            papel="Verifica regras de negócio e limites de alçada",
        )

    def processar(
        self,
        contexto: ContextoHandoff,
    ) -> ContextoHandoff:
        """
        Aplica regras de compliance ao resultado já calculado.

        Regras verificadas:
        - Valor total > R$ 5.000 → exige aprovação humana (HITL)
        - Banco desconhecido → requer verificação manual
        """
        calculo = contexto.dados_coletados.get("calculo", {})
        extracao = contexto.dados_coletados.get("extracao", {})

        console.print(
            f"  [cyan]→ {self.nome}: verificando compliance...[/cyan]"
        )
        time.sleep(0.02)

        total = calculo.get("total_atualizado", 0.0)
        banco = extracao.get("banco", "Desconhecido")

        alertas: list[str] = []
        exige_aprovacao = False

        if total > 5000:
            alertas.append(
                f"Valor R$ {total:.2f} acima do limite de alçada"
            )
            exige_aprovacao = True

        if banco == "Desconhecido":
            alertas.append("Banco não identificado — verificar manualmente")

        contexto.dados_coletados["compliance"] = {
            "aprovado": not exige_aprovacao,
            "exige_aprovacao_humana": exige_aprovacao,
            "alertas": alertas,
        }
        contexto.registrar_decisao(
            agente=self.nome,
            acao="compliance_verificado",
            detalhes=f"aprovado={not exige_aprovacao}, alertas={len(alertas)}",
        )
        return contexto


# ============================================================
# 4. SUPERVISOR — Orquestrador central
# ============================================================
# O supervisor conhece todos os workers disponíveis e toma
# decisões dinâmicas sobre o roteamento.
#
# DIFERENÇA DO ROTEADOR SIMPLES (módulo 8):
# O roteador escolhe UM caminho. O supervisor pode:
# - Executar múltiplos workers em sequência
# - Decidir em tempo de execução baseado em resultados anteriores
# - Tratar falhas e tentar alternativas
# - Publicar decisões para auditoria
# ============================================================

class Supervisor:
    """
    Orquestrador que coordena workers especializados.

    RESPONSABILIDADES:
    - Manter o registro de workers disponíveis
    - Decidir qual sequência de workers executar
    - Tratar falhas com fallback
    - Manter histórico de decisões para auditoria
    - Produzir a resposta final consolidada
    """

    def __init__(self) -> None:
        self._workers: dict[str, WorkerBase] = {}
        self._historico_tarefas: list[ContextoHandoff] = []

    def registrar_worker(self, worker: WorkerBase) -> None:
        """
        Registra um worker no supervisor.

        Semelhante ao REGISTRY de tools do módulo 3,
        mas para agentes workers completos.

        Parâmetros:
        - worker: instância de WorkerBase a registrar
        """
        self._workers[worker.nome] = worker
        console.print(
            f"  [dim]Worker registrado: {worker.nome} "
            f"({worker.papel})[/dim]"
        )

    def _decidir_sequencia(
        self,
        contexto: ContextoHandoff,
    ) -> list[str]:
        """
        Decide a sequência de workers para esta tarefa.

        LÓGICA DE ROTEAMENTO:
        - Sempre extrai primeiro (ExtrAtração é o passo base)
        - Aplica cálculo se há dados de extração disponíveis
        - Sempre verifica compliance ao final

        EM PRODUÇÃO:
        Esta lógica pode usar uma LLM para decidir dinamicamente,
        baseada no tipo de tarefa, urgência e dados disponíveis.
        É o "cérebro" da orquestração.

        Parâmetros:
        - contexto: contexto atual (pode ter dados de passos anteriores)

        Retorna:
        - Lista de nomes de workers na ordem de execução
        """
        # Sequência padrão para processamento de boleto
        sequencia = ["WorkerExtracao", "WorkerCalculo", "WorkerCompliance"]
        contexto.registrar_decisao(
            agente="Supervisor",
            acao="sequencia_definida",
            detalhes=" → ".join(sequencia),
        )
        return sequencia

    def processar_tarefa(
        self,
        tarefa: str,
        tarefa_id: str | None = None,
    ) -> ContextoHandoff:
        """
        Orquestra o processamento completo de uma tarefa.

        FLUXO:
        1. Cria contexto de handoff
        2. Decide sequência de workers
        3. Executa cada worker em ordem, passando o contexto
        4. Trata falhas com fallback (skipa worker com erro)
        5. Marca tarefa como concluída ou falhou
        6. Armazena no histórico

        Parâmetros:
        - tarefa: texto da tarefa a processar
        - tarefa_id: ID opcional (gerado automaticamente se omitido)

        Retorna:
        - ContextoHandoff com todo o resultado e histórico
        """
        id_gerado = tarefa_id or (
            f"tarefa_{len(self._historico_tarefas) + 1:03d}"
        )
        contexto = ContextoHandoff(
            tarefa_id=id_gerado,
            tarefa_original=tarefa,
        )
        contexto.status = "em_andamento"

        console.print(
            f"\n[bold]Supervisor processando: {id_gerado}[/bold]"
        )
        sequencia = self._decidir_sequencia(contexto)

        for nome_worker in sequencia:
            worker = self._workers.get(nome_worker)
            if worker is None:
                contexto.registrar_decisao(
                    agente="Supervisor",
                    acao="worker_nao_encontrado",
                    detalhes=nome_worker,
                )
                continue

            try:
                contexto = worker.processar(contexto)
            except (RuntimeError, ValueError, KeyError) as exc:
                # Worker falhou: registra e continua com os demais
                contexto.registrar_decisao(
                    agente="Supervisor",
                    acao="worker_falhou_continuando",
                    detalhes=f"{nome_worker}: {exc}",
                )
                console.print(
                    f"  [yellow]⚠ {nome_worker} falhou, "
                    f"continuando: {exc}[/yellow]"
                )

        contexto.status = "concluido"
        self._historico_tarefas.append(contexto)
        return contexto

    def exibir_resultado(self, contexto: ContextoHandoff) -> None:
        """
        Exibe o resultado consolidado e o histórico de decisões.

        SEÇÕES:
        1. Dados extraídos pelo WorkerExtracao
        2. Cálculos do WorkerCalculo
        3. Status do WorkerCompliance
        4. Árvore de decisões do Supervisor
        """
        console.print(Panel.fit(
            f"[bold]Tarefa:[/bold] {contexto.tarefa_id}\n"
            f"[bold]Status:[/bold] [green]{contexto.status}[/green]",
            title="📋 Resultado da Orquestração",
            border_style="cyan",
        ))

        # Tabela de dados coletados
        tabela = Table(title="Dados coletados pelos workers")
        tabela.add_column("Worker", style="bold")
        tabela.add_column("Dado")
        tabela.add_column("Valor")

        extracao = contexto.dados_coletados.get("extracao", {})
        for k, v in extracao.items():
            tabela.add_row("Extração", k, str(v))

        calculo = contexto.dados_coletados.get("calculo", {})
        for k, v in calculo.items():
            tabela.add_row("Cálculo", k, str(v))

        compliance = contexto.dados_coletados.get("compliance", {})
        aprovado = compliance.get("aprovado", True)
        cor = "green" if aprovado else "yellow"
        for alerta in compliance.get("alertas", []):
            tabela.add_row(
                "Compliance",
                "alerta",
                f"[{cor}]{alerta}[/{cor}]",
            )
        tabela.add_row(
            "Compliance",
            "aprovado",
            (
                "[green]✓ Sim[/green]"
                if aprovado
                else "[yellow]⚠ Requer aprovação humana[/yellow]"
            ),
        )
        console.print(tabela)

        # Árvore de decisões (auditoria)
        arvore = Tree(
            "[bold]Histórico de decisões do Supervisor[/bold]"
        )
        for decisao in contexto.historico_decisoes:
            ts = decisao["timestamp"][11:19]  # só hora
            arvore.add(
                f"[dim]{ts}[/dim] [bold]{decisao['agente']}[/bold]"
                f" → {decisao['acao']}"
                + (
                    f" ({decisao['detalhes']})"
                    if decisao["detalhes"]
                    else ""
                )
            )
        console.print(arvore)


# ============================================================
# DEMO COMPLETA — Orquestração multi-agente
# ============================================================

def demo_orquestracao() -> None:
    """
    Demonstra o padrão Supervisor + Workers para boletos.

    ETAPAS:
    1. Cria e registra 3 workers especializados
    2. Processa 3 tarefas diferentes:
       - Boleto normal (< R$ 5.000)
       - Boleto vencido com multa
       - Boleto alto valor (> R$ 5.000 → exige aprovação)
    3. Exibe resultado e histórico de decisões de cada um

    OBSERVE NO OUTPUT:
    - Cada worker registra suas decisões no contexto
    - O Supervisor nunca "sabe" o que cada worker faz internamente
    - O histórico de decisões mostra o rastro completo de auditoria
    - Boleto > R$ 5.000 aciona o alerta de aprovação humana

    EXERCÍCIO SUGERIDO:
    1. Crie um WorkerResumo que gera a resposta final em linguagem
       natural para o usuário usando a LLM (Groq)
    2. Adicione um WorkerFraudeDetection que verifica padrões
       suspeitos nos dados extraídos
    3. Implemente um fallback: se WorkerCalculo falhar,
       o Supervisor redireciona para um WorkerCalculoSimples
    """
    console.print(Panel.fit(
        "[bold]Orquestração Avançada Multi-Agente[/bold]\n"
        "Padrão Supervisor + Workers Especializados",
        title="🎭 Módulo 20 — Orquestração",
        border_style="magenta",
    ))

    # Cria e configura o supervisor
    supervisor = Supervisor()
    console.print("\n[bold]Registrando workers...[/bold]")
    supervisor.registrar_worker(WorkerExtracao())
    supervisor.registrar_worker(WorkerCalculo())
    supervisor.registrar_worker(WorkerCompliance())

    # ── Tarefa 1: Boleto normal ───────────────────────────────
    console.print("\n" + "─" * 50)
    console.print("[bold]Tarefa 1: Boleto normal[/bold]")
    ctx1 = supervisor.processar_tarefa(
        tarefa=(
            "Preciso pagar o boleto do Bradesco, "
            "valor R$ 1.500,00, vence amanhã."
        ),
        tarefa_id="tarefa_001",
    )
    supervisor.exibir_resultado(ctx1)

    # ── Tarefa 2: Boleto vencido ──────────────────────────────
    console.print("\n" + "─" * 50)
    console.print("[bold]Tarefa 2: Boleto vencido[/bold]")
    ctx2 = supervisor.processar_tarefa(
        tarefa=(
            "Tenho um boleto do Itaú de R$ 800,00 que venceu "
            "há 7 dias, quero saber o valor atualizado."
        ),
        tarefa_id="tarefa_002",
    )
    supervisor.exibir_resultado(ctx2)

    # ── Tarefa 3: Alto valor ──────────────────────────────────
    console.print("\n" + "─" * 50)
    console.print("[bold]Tarefa 3: Alto valor (> R$ 5.000)[/bold]")
    ctx3 = supervisor.processar_tarefa(
        tarefa=(
            "Preciso processar o boleto da fornecedora "
            "R$ 12.000,00, banco Santander, vence hoje."
        ),
        tarefa_id="tarefa_003",
    )
    supervisor.exibir_resultado(ctx3)

    console.print("\n💡 [bold yellow]Dica:[/bold yellow]")
    console.print(
        "  Em produção, substitua os workers simulados por\n"
        "  agentes reais com chamadas ao Groq (módulo 6).\n"
        "  O Supervisor pode usar uma LLM para decidir a\n"
        "  sequência dinamicamente (ReAct + tools = roteamento)."
    )


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    demo_orquestracao()
