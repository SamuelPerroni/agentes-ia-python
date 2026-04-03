"""
============================================================
MÓDULO 37.1 - ESCALAÇÃO COM SLA E REVISÃO HUMANA AVANÇADA
============================================================
Neste módulo, implementamos HITL (Human-in-the-Loop)
avançado: em vez de apenas perguntar ao humano, o agente
gerencia prazos com SLA, escala para gestores quando o
prazo vence e notifica a cadeia de responsáveis.

FLUXO DE ESCALAÇÃO:

  Caso criado
       │
       ▼
  [N1: Analista] ──── SLA 2h ────► [N2: Gestor]
                                        │
                                   SLA 4h adicional
                                        │
                                        ▼
                                  [N3: Diretoria]
                                        │
                                   SLA 2h adicional
                                        │
                                        ▼
                                  [Alerta Crítico]

CONCEITOS COBERTOS:
1. SLA por nível — cada nível tem prazo diferente
2. Escalação automática — quando prazo vence sem ação
3. Cadeia de notificação — e-mail, Slack, SMS por nível
4. Ação humana que fecha o caso em qualquer nível
5. Indicadores de SLA — cumprido, violado, em risco
6. Relatório de SLA ao final do período
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. ENUMERAÇÕES
# ============================================================

class NivelEscalacao(Enum):
    """Níveis de escalação hierárquica para casos de revisão."""
    N1_ANALISTA = "N1 - Analista"
    N2_GESTOR = "N2 - Gestor"
    N3_DIRETORIA = "N3 - Diretoria"
    CRITICO = "CRÍTICO"


class StatusCaso(Enum):
    """Status de um caso de revisão,
    desde a criação até a resolução ou expiração."""
    ABERTO = "Aberto"
    EM_REVISAO = "Em Revisão"
    RESOLVIDO = "Resolvido"
    EXPIRADO = "Expirado"


class StatusSLA(Enum):
    """Status do SLA baseado no tempo decorrido em relação ao prazo definido:
    - OK: dentro do prazo
    - EM_RISCO: > 80% do prazo consumido
    - VIOLADO: prazo ultrapassado
    - CUMPRIDO: resolvido dentro do prazo"""
    OK = "OK"
    EM_RISCO = "Em Risco"       # > 80% do prazo consumido
    VIOLADO = "Violado"         # prazo ultrapassado
    CUMPRIDO = "Cumprido"       # resolvido dentro do prazo


# ============================================================
# 2. CONFIGURAÇÃO DE SLA
# ============================================================

# SLA por nível: horas máximas para resposta naquele nível
SLA_HORAS: dict[NivelEscalacao, float] = {
    NivelEscalacao.N1_ANALISTA: 2.0,
    NivelEscalacao.N2_GESTOR: 4.0,
    NivelEscalacao.N3_DIRETORIA: 2.0,
    NivelEscalacao.CRITICO: 1.0,   # urgência máxima
}

# Responsáveis simulados por nível
RESPONSAVEIS: dict[NivelEscalacao, list[str]] = {
    NivelEscalacao.N1_ANALISTA: [
        "ana.souza@empresa.com",
        "pedro.lima@empresa.com",
    ],
    NivelEscalacao.N2_GESTOR: ["gestor.fintech@empresa.com"],
    NivelEscalacao.N3_DIRETORIA: ["diretoria@empresa.com"],
    NivelEscalacao.CRITICO: [
        "diretoria@empresa.com",
        "+55 11 99999-9999",   # SMS
    ],
}


# ============================================================
# 3. EVENTO DE ESCALAÇÃO
# ============================================================

@dataclass
class EventoEscalacao:
    """Registra cada mudança de nível ou ação humana."""
    timestamp: str
    nivel_anterior: Optional[NivelEscalacao]
    nivel_novo: NivelEscalacao
    motivo: str
    responsavel: Optional[str] = None


# ============================================================
# 4. CASO DE REVISÃO
# ============================================================

@dataclass
class CasoRevisao:
    """
    Representa um item que requer decisão humana
    antes de continuar o processo APA.
    """
    id: str
    descricao: str
    dados: dict
    criado_em: datetime = field(
        default_factory=datetime.now
    )
    nivel_atual: NivelEscalacao = NivelEscalacao.N1_ANALISTA
    status: StatusCaso = StatusCaso.ABERTO
    resolucao: Optional[str] = None
    resolvido_em: Optional[datetime] = None
    historico: list[EventoEscalacao] = field(
        default_factory=list
    )
    nivel_desde: datetime = field(
        default_factory=datetime.now
    )

    def __post_init__(self) -> None:
        self.historico.append(EventoEscalacao(
            timestamp=self.criado_em.isoformat(
                timespec="seconds"
            ),
            nivel_anterior=None,
            nivel_novo=self.nivel_atual,
            motivo="Criação do caso",
        ))

    # ---- Consultas de SLA ----

    @property
    def sla_limite(self) -> timedelta:
        """Retorna o prazo máximo para resposta no nível atual."""
        return timedelta(
            hours=SLA_HORAS[self.nivel_atual]
        )

    @property
    def tempo_no_nivel(self) -> timedelta:
        """Calcula o tempo decorrido desde a última mudança de nível."""
        return datetime.now() - self.nivel_desde

    @property
    def percentual_sla(self) -> float:
        """Calcula o percentual do SLA consumido no nível atual."""
        return (
            self.tempo_no_nivel.total_seconds()
            / self.sla_limite.total_seconds()
        )

    @property
    def status_sla(self) -> StatusSLA:
        """Determina o status do SLA com base no tempo decorrido."""
        if self.resolvido_em:
            return StatusSLA.CUMPRIDO
        p = self.percentual_sla
        if p > 1.0:
            return StatusSLA.VIOLADO
        if p > 0.8:
            return StatusSLA.EM_RISCO
        return StatusSLA.OK


# ============================================================
# 5. GERENCIADOR DE SLA
# ============================================================

class GerenciadorSLA:
    """
    Verifica prazos, escalona casos e registra ações
    humanas. Em produção, seria executado via cron ou
    APScheduler a cada ~5 minutos.
    """

    def __init__(self) -> None:
        self._casos: list[CasoRevisao] = []

    def criar_caso(
        self,
        caso_id: str,
        descricao: str,
        dados: dict,
        offset_criacao: timedelta = timedelta(0),
    ) -> CasoRevisao:
        """
        Cria caso de revisão.
        offset_criacao permite simular casos criados
        no passado para demonstrar escalação.
        """
        caso = CasoRevisao(
            id=caso_id,
            descricao=descricao,
            dados=dados,
        )
        # Simula criação no passado para demo
        caso.criado_em = datetime.now() - offset_criacao
        caso.nivel_desde = caso.criado_em
        self._casos.append(caso)
        return caso

    def verificar_slas(self) -> list[CasoRevisao]:
        """
        Verifica todos os casos abertos:
        - Escalona quando SLA violado
        - Notifica quando EM_RISCO
        Retorna lista de casos que foram escalados.
        """
        escalados = []
        for caso in self._casos:
            if caso.status in (
                StatusCaso.RESOLVIDO,
                StatusCaso.EXPIRADO,
            ):
                continue

            sla = caso.status_sla
            if sla == StatusSLA.VIOLADO:
                self._escalonar(caso)
                escalados.append(caso)
            elif sla == StatusSLA.EM_RISCO:
                self._notificar_risco(caso)
        return escalados

    def resolver_caso(
        self,
        caso_id: str,
        resolucao: str,
        responsavel: str,
    ) -> None:
        """Registra ação humana que finaliza o caso."""
        caso = self._buscar(caso_id)
        caso.status = StatusCaso.RESOLVIDO
        caso.resolucao = resolucao
        caso.resolvido_em = datetime.now()
        caso.historico.append(EventoEscalacao(
            timestamp=caso.resolvido_em.isoformat(
                timespec="seconds"
            ),
            nivel_anterior=caso.nivel_atual,
            nivel_novo=caso.nivel_atual,
            motivo=f"Resolvido: {resolucao}",
            responsavel=responsavel,
        ))
        console.print(
            f"  [green]✓ Caso {caso.id} resolvido "
            f"por {responsavel}[/]"
        )

    # ---- internos ----

    def _proximo_nivel(
        self, atual: NivelEscalacao
    ) -> NivelEscalacao:
        ordem = [
            NivelEscalacao.N1_ANALISTA,
            NivelEscalacao.N2_GESTOR,
            NivelEscalacao.N3_DIRETORIA,
            NivelEscalacao.CRITICO,
        ]
        idx = ordem.index(atual)
        return (
            ordem[idx + 1]
            if idx + 1 < len(ordem)
            else NivelEscalacao.CRITICO
        )

    def _escalonar(self, caso: CasoRevisao) -> None:
        anterior = caso.nivel_atual
        proximo = self._proximo_nivel(anterior)
        caso.nivel_atual = proximo
        caso.nivel_desde = datetime.now()
        caso.historico.append(EventoEscalacao(
            timestamp=datetime.now().isoformat(
                timespec="seconds"
            ),
            nivel_anterior=anterior,
            nivel_novo=proximo,
            motivo=f"SLA violado em {anterior.value}",
        ))
        responsaveis = RESPONSAVEIS.get(proximo, [])
        console.print(
            f"  [red]▲ Escalado {caso.id}:[/] "
            f"{anterior.value} → {proximo.value}"
        )
        for dest in responsaveis:
            console.print(
                f"    [yellow]📧 Notificação: {dest}[/]"
            )

    def _notificar_risco(self, caso: CasoRevisao) -> None:
        p = caso.percentual_sla * 100
        console.print(
            f"  [yellow]⚠ Em risco {caso.id} "
            f"({p:.0f}% do SLA consumido)[/]"
        )

    def _buscar(self, caso_id: str) -> CasoRevisao:
        for c in self._casos:
            if c.id == caso_id:
                return c
        raise KeyError(f"Caso {caso_id} não encontrado")

    # ---- relatório ----

    def exibir_dashboard(self) -> None:
        """Exibe tabela com status de SLA de todos os casos abertos."""
        tabela = Table(
            title="Dashboard de SLA",
            header_style="bold magenta",
        )
        tabela.add_column("ID")
        tabela.add_column("Descrição")
        tabela.add_column("Nível")
        tabela.add_column("SLA")
        tabela.add_column("% Tempo")
        tabela.add_column("Status")

        cores_sla = {
            StatusSLA.OK: "green",
            StatusSLA.EM_RISCO: "yellow",
            StatusSLA.VIOLADO: "red",
            StatusSLA.CUMPRIDO: "blue",
        }
        for caso in self._casos:
            sla = caso.status_sla
            cor = cores_sla[sla]
            pct = f"{caso.percentual_sla:.0%}"
            tabela.add_row(
                caso.id,
                caso.descricao[:35],
                caso.nivel_atual.value,
                f"{SLA_HORAS[caso.nivel_atual]}h",
                pct,
                f"[{cor}]{sla.value}[/]",
            )
        console.print(tabela)

    def exibir_historico_caso(
        self, caso_id: str
    ) -> None:
        """Exibe o histórico de eventos de um caso específico."""
        caso = self._buscar(caso_id)
        console.print(
            f"\n  [bold]Histórico do caso {caso_id}[/]"
        )
        for ev in caso.historico:
            anterior = (
                ev.nivel_anterior.value
                if ev.nivel_anterior else "—"
            )
            console.print(
                f"  {ev.timestamp}  "
                f"{anterior} → {ev.nivel_novo.value}  "
                f"[dim]{ev.motivo}[/]"
            )


# ============================================================
# 6. DEMO
# ============================================================

def demo_escalacao_sla() -> None:
    """Demonstração do gerenciador de SLA e escalação automática."""
    console.print(
        Panel(
            "[bold]Módulo 37 — Escalação com SLA[/]\n"
            "Revisão humana com prazo, notificação "
            "automática e cadeia de escalação",
            style="bold blue",
        )
    )

    gerenciador = GerenciadorSLA()

    # Caso 1: resolvido rapidamente (SLA OK)
    gerenciador.criar_caso(
        caso_id="RV-001",
        descricao="Boleto com CNPJ divergente",
        dados={
            "cnpj_doc": "12.345.678/0001-99",
            "cnpj_cadastro": "12.345.678/0001-09",
        },
        offset_criacao=timedelta(minutes=30),
    )

    # Caso 2: SLA violado em N1, deve escalar para N2
    gerenciador.criar_caso(
        caso_id="RV-002",
        descricao="Valor suspeito: R$ 980.000,00",
        dados={"valor": 980000.00, "beneficiario": "XYZ"},
        offset_criacao=timedelta(hours=3),   # > 2h SLA N1
    )

    # Caso 3: já em N2 e violando N2 também
    caso3 = gerenciador.criar_caso(
        caso_id="RV-003",
        descricao="NF duplicada possível fraude",
        dados={"nf_1": "NF-4521", "nf_2": "NF-4521"},
        offset_criacao=timedelta(hours=7),   # >2h N1 +4h N2
    )
    # Simulamos escalada prévia para N2
    caso3.nivel_atual = NivelEscalacao.N2_GESTOR
    caso3.nivel_desde = (
        datetime.now() - timedelta(hours=4, minutes=30)
    )

    console.rule("[yellow]Estado inicial")
    gerenciador.exibir_dashboard()

    console.rule("[yellow]Verificação de SLAs (ciclo 1)")
    gerenciador.verificar_slas()

    console.rule("[yellow]Ação humana: resolver caso 1")
    gerenciador.resolver_caso(
        caso_id="RV-001",
        resolucao="CNPJ correto confirmado — aprovar",
        responsavel="ana.souza@empresa.com",
    )

    console.rule("[yellow]Estado após escalações")
    gerenciador.exibir_dashboard()

    console.rule("[yellow]Histórico do caso 2")
    gerenciador.exibir_historico_caso("RV-002")

    console.rule("[yellow]Histórico do caso 3")
    gerenciador.exibir_historico_caso("RV-003")


if __name__ == "__main__":
    demo_escalacao_sla()
