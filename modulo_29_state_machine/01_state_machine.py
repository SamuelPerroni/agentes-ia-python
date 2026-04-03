"""
============================================================
MÓDULO 29.1 - STATE MACHINE PARA FLUXOS DE APROVAÇÃO
============================================================
Neste módulo, aprendemos a modelar processos de negócio
como máquinas de estado, onde o agente gerencia transições,
validações e notificações em cada etapa.

CONCEITO CHAVE:
Workflows corporativos têm estágios bem definidos e regras
de transição (quem pode aprovar, qual valor exige dupla
aprovação, o que acontece no prazo). Uma state machine
torna essas regras explícitas, auditáveis e testáveis.

POR QUE STATE MACHINE?
- Processos sem estado explícito viram "código spaghetti"
- Auditoria exige saber em que estado estava cada item
- Testes: cada transição é verificável independentemente
- Sem state machine, o agente pode pular etapas importantes

ESTADOS DO FLUXO DE APROVAÇÃO DE PAGAMENTO:

  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │   RASCUNHO ──▶ PENDENTE ──▶ EM_ANÁLISE                 │
  │                    │            │                       │
  │                    │       ┌────┴────┐                  │
  │                    │       ▼         ▼                  │
  │                    │   APROVADO   REJEITADO             │
  │                    │       │                            │
  │                    │       ▼                            │
  │                    │  AGUARD_PAGTO                      │
  │                    │       │                            │
  │                    │       ▼                            │
  │                EXPIRADO  PAGO                           │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

TRANSIÇÕES PERMITIDAS:

  RASCUNHO      → PENDENTE      (submeter)
  PENDENTE      → EM_ANÁLISE    (analista assume)
  PENDENTE      → EXPIRADO      (prazo vencido)
  EM_ANÁLISE    → APROVADO      (aprovação)
  EM_ANÁLISE    → REJEITADO     (rejeição com motivo)
  APROVADO      → AGUARD_PAGTO  (encaminha para pagamento)
  AGUARD_PAGTO  → PAGO          (pagamento confirmado)

REGRAS DE NEGÓCIO:
- Valor > R$ 10.000 exige dupla aprovação
- RASCUNHO expira em 3 dias sem submissão
- PENDENTE expira em 5 dias sem análise
- Toda transição gera registro de auditoria

Tópicos cobertos:
1. Enum de estados e definição de transições válidas
2. Dataclass SolicitacaoPagamento com histórico de eventos
3. Máquina de estado com validação de transições
4. Regras de negócio: dupla aprovação, expiração
5. Log de auditoria imutável (append-only)
6. Agente que pilota o workflow automaticamente
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. DEFINIÇÃO DE ESTADOS
# ============================================================

class Estado(str, Enum):
    """Estados possíveis de uma solicitação de pagamento."""
    RASCUNHO = "RASCUNHO"
    PENDENTE = "PENDENTE"
    EM_ANALISE = "EM_ANALISE"
    APROVADO = "APROVADO"
    REJEITADO = "REJEITADO"
    AGUARD_PAGTO = "AGUARD_PAGTO"
    PAGO = "PAGO"
    EXPIRADO = "EXPIRADO"


# Mapa de transições permitidas: estado_atual → [destinos válidos]
TRANSICOES: dict[Estado, list[Estado]] = {
    Estado.RASCUNHO:      [Estado.PENDENTE, Estado.EXPIRADO],
    Estado.PENDENTE:      [Estado.EM_ANALISE, Estado.EXPIRADO],
    Estado.EM_ANALISE:    [Estado.APROVADO, Estado.REJEITADO],
    Estado.APROVADO:      [Estado.AGUARD_PAGTO],
    Estado.AGUARD_PAGTO:  [Estado.PAGO],
    Estado.REJEITADO:     [],
    Estado.PAGO:          [],
    Estado.EXPIRADO:      [],
}

# Estados terminais — não aceitam mais transições
ESTADOS_TERMINAIS = {
    Estado.REJEITADO,
    Estado.PAGO,
    Estado.EXPIRADO,
}


# ============================================================
# 2. MODELO DE DADOS
# ============================================================

@dataclass
class EventoAuditoria:
    """Registro imutável de uma transição de estado."""
    timestamp: str
    estado_anterior: Estado
    estado_novo: Estado
    responsavel: str
    motivo: str = ""


@dataclass
class SolicitacaoPagamento:
    """
    Entidade principal do workflow de aprovação.
    Contém o estado atual, dados do pagamento e histórico.
    """
    id: str
    descricao: str
    valor: float
    fornecedor: str
    solicitante: str
    # Estado inicial sempre é RASCUNHO
    estado: Estado = Estado.RASCUNHO
    # Aprovações coletadas (para fluxo de dupla aprovação)
    aprovacoes: list[str] = field(default_factory=list)
    # Histórico append-only de eventos
    historico: list[EventoAuditoria] = field(
        default_factory=list
    )

    @property
    def requer_dupla_aprovacao(self) -> bool:
        """Valores acima de R$ 10.000 exigem 2 aprovadores."""
        return self.valor > 10_000.0


# ============================================================
# 3. MÁQUINA DE ESTADOS
# ============================================================

class ErroDeBloqueio(Exception):
    """Transição inválida tentada."""


class MaquinaAprovacao:
    """
    Gerencia as transições de estado de uma solicitação.
    Valida regras de negócio antes de cada transição.
    """

    def __init__(self, solicitacao: SolicitacaoPagamento) -> None:
        self._s = solicitacao

    def _registrar(
        self,
        destino: Estado,
        responsavel: str,
        motivo: str = "",
    ) -> None:
        """Grava evento de auditoria e muda o estado."""
        evento = EventoAuditoria(
            timestamp=datetime.now().isoformat(
                timespec="seconds"
            ),
            estado_anterior=self._s.estado,
            estado_novo=destino,
            responsavel=responsavel,
            motivo=motivo,
        )
        self._s.historico.append(evento)
        self._s.estado = destino
        console.print(
            f"  [dim]{evento.timestamp}[/] "
            f"[bold]{evento.estado_anterior}[/] → "
            f"[bold green]{evento.estado_novo}[/] "
            f"por [cyan]{responsavel}[/]"
            + (f" — {motivo}" if motivo else "")
        )

    def _validar_transicao(self, destino: Estado) -> None:
        """Lança ErroDeBloqueio se a transição for inválida."""
        permitidos = TRANSICOES.get(self._s.estado, [])
        if destino not in permitidos:
            raise ErroDeBloqueio(
                f"Transição inválida: "
                f"{self._s.estado} → {destino}. "
                f"Permitidos: {permitidos}"
            )

    # ---- Ações de workflow ----

    def submeter(self, solicitante: str) -> None:
        """RASCUNHO → PENDENTE."""
        self._validar_transicao(Estado.PENDENTE)
        self._registrar(Estado.PENDENTE, solicitante)

    def assumir_analise(self, analista: str) -> None:
        """PENDENTE → EM_ANALISE."""
        self._validar_transicao(Estado.EM_ANALISE)
        self._registrar(Estado.EM_ANALISE, analista)

    def aprovar(
        self, aprovador: str, motivo: str = ""
    ) -> None:
        """
        EM_ANALISE → APROVADO.
        Se dupla aprovação é exigida, aguarda segundo
        aprovador antes de mudar o estado.
        """
        self._validar_transicao(Estado.APROVADO)
        if aprovador in self._s.aprovacoes:
            raise ErroDeBloqueio(
                f"{aprovador} já aprovou esta solicitação."
            )
        self._s.aprovacoes.append(aprovador)
        if self._s.requer_dupla_aprovacao and (
            len(self._s.aprovacoes) < 2
        ):
            console.print(
                f"  [yellow]Dupla aprovação: "
                f"{len(self._s.aprovacoes)}/2 "
                f"({aprovador} aprovou). "
                f"Aguardando segundo aprovador.[/]"
            )
            return
        self._registrar(
            Estado.APROVADO, aprovador, motivo
        )

    def rejeitar(
        self, aprovador: str, motivo: str
    ) -> None:
        """EM_ANALISE → REJEITADO."""
        self._validar_transicao(Estado.REJEITADO)
        if not motivo:
            raise ErroDeBloqueio(
                "Motivo obrigatório para rejeição."
            )
        self._registrar(Estado.REJEITADO, aprovador, motivo)

    def encaminhar_pagamento(
        self, responsavel: str
    ) -> None:
        """APROVADO → AGUARD_PAGTO."""
        self._validar_transicao(Estado.AGUARD_PAGTO)
        self._registrar(Estado.AGUARD_PAGTO, responsavel)

    def confirmar_pagamento(
        self, responsavel: str
    ) -> None:
        """AGUARD_PAGTO → PAGO."""
        self._validar_transicao(Estado.PAGO)
        self._registrar(Estado.PAGO, responsavel)

    def expirar(self) -> None:
        """Marca a solicitação como expirada por prazo."""
        self._validar_transicao(Estado.EXPIRADO)
        self._registrar(
            Estado.EXPIRADO,
            "sistema",
            "Prazo de análise excedido",
        )


# ============================================================
# 4. EXIBIÇÃO DE HISTÓRICO
# ============================================================

def exibir_historico(
    solicitacao: SolicitacaoPagamento,
) -> None:
    """Exibe o log de auditoria da solicitação."""
    tabela = Table(
        title=f"Histórico — {solicitacao.id}",
        header_style="bold magenta",
        show_lines=True,
    )
    tabela.add_column("Timestamp")
    tabela.add_column("De")
    tabela.add_column("Para")
    tabela.add_column("Responsável")
    tabela.add_column("Motivo")
    for ev in solicitacao.historico:
        tabela.add_row(
            ev.timestamp,
            ev.estado_anterior.value,
            ev.estado_novo.value,
            ev.responsavel,
            ev.motivo,
        )
    console.print(tabela)


# ============================================================
# 5. DEMO
# ============================================================

def demo_state_machine() -> None:
    """Demonstração do agente de state machine para fluxos de aprovação."""
    console.print(
        Panel(
            "[bold]Módulo 29 — State Machine para Fluxos "
            "de Aprovação[/]\n"
            "Workflow completo: rascunho → aprovação → pagamento",
            style="bold blue",
        )
    )

    # --- Cenário 1: fluxo feliz (valor baixo) ---
    console.rule("[yellow]Cenário 1 — Fluxo normal (< R$ 10k)")
    sol1 = SolicitacaoPagamento(
        id="PAG-001",
        descricao="Licença de Software",
        valor=3_500.0,
        fornecedor="SoftCo Ltda",
        solicitante="ana.silva",
    )
    m1 = MaquinaAprovacao(sol1)
    m1.submeter("ana.silva")
    m1.assumir_analise("jose.gestor")
    m1.aprovar("jose.gestor", "Documentação completa")
    m1.encaminhar_pagamento("jose.gestor")
    m1.confirmar_pagamento("financeiro")
    console.print(
        f"\n  Estado final: [bold green]{sol1.estado}[/]\n"
    )
    exibir_historico(sol1)

    # --- Cenário 2: dupla aprovação (valor alto) ---
    console.rule(
        "[yellow]Cenário 2 — Dupla aprovação (> R$ 10k)"
    )
    sol2 = SolicitacaoPagamento(
        id="PAG-002",
        descricao="Servidores Cloud — contrato anual",
        valor=48_000.0,
        fornecedor="CloudPro S.A.",
        solicitante="pedro.ti",
    )
    m2 = MaquinaAprovacao(sol2)
    m2.submeter("pedro.ti")
    m2.assumir_analise("diretora.financeira")
    m2.aprovar("diretora.financeira", "Dentro do orçamento")
    m2.aprovar(
        "ceo", "Aprovado na reunião de diretoria"
    )
    m2.encaminhar_pagamento("diretora.financeira")
    m2.confirmar_pagamento("financeiro")
    console.print(
        f"\n  Estado final: [bold green]{sol2.estado}[/]\n"
    )
    exibir_historico(sol2)

    # --- Cenário 3: rejeição ---
    console.rule("[yellow]Cenário 3 — Rejeição com motivo")
    sol3 = SolicitacaoPagamento(
        id="PAG-003",
        descricao="Consultoria sem contrato",
        valor=7_200.0,
        fornecedor="Consultor Avulso",
        solicitante="marcos.op",
    )
    m3 = MaquinaAprovacao(sol3)
    m3.submeter("marcos.op")
    m3.assumir_analise("jose.gestor")
    m3.rejeitar(
        "jose.gestor",
        "Fornecedor não cadastrado e sem contrato vigente",
    )
    console.print(
        f"\n  Estado final: "
        f"[bold red]{sol3.estado}[/]\n"
    )
    exibir_historico(sol3)

    # --- Cenário 4: transição inválida ---
    console.rule(
        "[yellow]Cenário 4 — Transição inválida bloqueada"
    )
    sol4 = SolicitacaoPagamento(
        id="PAG-004",
        descricao="Teste de bloqueio",
        valor=1_000.0,
        fornecedor="Teste",
        solicitante="test",
    )
    m4 = MaquinaAprovacao(sol4)
    try:
        # Tenta confirmar pagamento sem approval → erro
        m4.confirmar_pagamento("financeiro")
    except ErroDeBloqueio as exc:
        console.print(
            f"  [red]Transição bloqueada: {exc}[/]"
        )


if __name__ == "__main__":
    demo_state_machine()
