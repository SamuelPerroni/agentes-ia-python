"""
============================================================
MÓDULO 46.1 - MULTI-AGENT DEBATE
============================================================
Em decisões de alto risco, um único agente pode ter viés
ou ponto cego. Multi-Agent Debate cria múltiplos agentes
com perspectivas diferentes que argumentam e votam.

ARQUITETURA:

  ┌─────────────────────────────────────────┐
  │             SISTEMA DE DEBATE           │
  │                                         │
  │  ┌──────────┐  ┌──────────┐  ┌───────┐  │
  │  │Agente    │  │Agente    │  │Agente │  │
  │  │Conserv.  │  │Neutro    │  │Liberal│  │
  │  └────┬─────┘  └────┬─────┘  └───┬───┘  │
  │       │             │            │      │
  │       └─────────────┴────────────┘      │
  │                     │                   │
  │                     ▼                   │
  │              ┌─────────────┐            │
  │              │  Moderador  │            │
  │              │  Agrega +   │            │
  │              │  Voto final │            │
  │              └─────────────┘            │
  └─────────────────────────────────────────┘

PERSPECTIVAS:

  CONSERVADOR — minimiza risco, exige todas as regras
  NEUTRO      — segue política por padrão
  LIBERAL     — favorece negócio, aceita risco calculado

DINÂMICA DO DEBATE:
  Rodada 1: cada agente analisa o caso independentemente
  Rodada 2: cada agente vê os argumentos dos outros e
            pode reforçar ou mudar seu voto
  Decisão: maioria simples; empate → decide o Neutro

QUANDO USAR:
  ✓ Aprovações de alto valor (acima de R$ 50k)
  ✓ Fornecedores novos ou sinalizados
  ✓ Casos limítrofes que política não cobre
  ✗ Operações rotineiras (custo de 3× o LLM)

Tópicos cobertos:
1. AgentePerspectiva — gera voto e argumento com viés
2. Rodada de debate — os agentes veem argumentos alheios
3. Moderador — agrega votos e decide
4. Demo: 3 casos (unanimidade, maioria, empate)
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. TIPOS
# ============================================================

class Voto(Enum):
    """Voto possível de um agente: aprovar, rejeitar ou escalar."""
    APROVAR = "aprovar"
    REJEITAR = "rejeitar"
    ESCALAR = "escalar"


class Perspectiva(Enum):
    """Perspectiva ou viés de um agente no debate."""
    CONSERVADOR = "conservador"
    NEUTRO = "neutro"
    LIBERAL = "liberal"


@dataclass
class CasoDebate:
    """Representa um caso a ser debatido pelos agentes."""
    caso_id: str
    descricao: str
    valor: float
    cnpj_valido: bool
    fornecedor_novo: bool
    possui_nfe: bool
    historico_pagamentos: int  # 0=novo, >0=num. de pagtos ok


@dataclass
class ArgumentoAgente:
    """Argumento gerado por um agente com base
    em sua perspectiva e análise do caso."""
    perspectiva: Perspectiva
    voto: Voto
    confianca: float   # 0.0 – 1.0
    justificativa: str


@dataclass
class ResultadoDebate:
    """Resultado final do debate, incluindo votos de cada agente,
    decisão final do moderador, se houve consenso e um resumo."""
    caso_id: str
    votos: list[ArgumentoAgente]
    decisao_final: Voto
    consenso: bool
    resumo: str


# ============================================================
# 2. AGENTES COM PERSPECTIVAS
# ============================================================

class AgentePerspectiva:
    """
    Simula um agente com viés deliberado.
    Em produção: system prompt define a perspectiva,
    o LLM gera o argumento real.
    """

    # Limiares e pesos por perspectiva
    _CONFIG = {
        Perspectiva.CONSERVADOR: {
            "limite_valor": 5_000,
            "exige_nfe": True,
            "penaliza_novo": True,
            "bonus_historico": 0.05,
        },
        Perspectiva.NEUTRO: {
            "limite_valor": 10_000,
            "exige_nfe": True,
            "penaliza_novo": False,
            "bonus_historico": 0.10,
        },
        Perspectiva.LIBERAL: {
            "limite_valor": 50_000,
            "exige_nfe": False,
            "penaliza_novo": False,
            "bonus_historico": 0.15,
        },
    }

    def __init__(self, perspectiva: Perspectiva) -> None:
        """Inicializa o agente com a perspectiva definida, carregando
        os parâmetros de decisão correspondentes."""
        self.perspectiva = perspectiva
        self._cfg = self._CONFIG[perspectiva]

    def analisar(
        self,
        caso: CasoDebate,
        argumentos_anteriores: Optional[list[ArgumentoAgente]]
        = None,
    ) -> ArgumentoAgente:
        """Analisa o caso e gera um argumento com voto,
        confiança e justificativa.
        Se argumentos_anteriores for fornecido,
        leva em conta os votos dos outros agentes."""
        score = 0.5  # score base neutro
        motivos: list[str] = []

        # CNPJ inválido → rejeição imediata em todos
        if not caso.cnpj_valido:
            return ArgumentoAgente(
                perspectiva=self.perspectiva,
                voto=Voto.REJEITAR,
                confianca=0.99,
                justificativa="CNPJ inválido — rejeição "
                "imediata por política.",
            )

        # Valor
        if caso.valor <= self._cfg["limite_valor"]:
            score += 0.25
            motivos.append(
                f"valor R$ {caso.valor:,.0f} "
                f"dentro do limite"
            )
        elif caso.valor <= self._cfg["limite_valor"] * 3:
            motivos.append(f"valor elevado R$ {caso.valor:,.0f}")
        else:
            score -= 0.30
            motivos.append(
                f"valor R$ {caso.valor:,.0f} muito alto"
            )

        # NF-e
        if caso.possui_nfe:
            score += 0.15
            motivos.append("NF-e presente")
        elif self._cfg["exige_nfe"]:
            score -= 0.25
            motivos.append("NF-e ausente (obrigatória)")

        # Fornecedor novo
        if caso.fornecedor_novo and self._cfg["penaliza_novo"]:
            score -= 0.20
            motivos.append("fornecedor sem histórico")
        elif not caso.fornecedor_novo:
            hist_bonus = (
                caso.historico_pagamentos
                * self._cfg["bonus_historico"]
            )
            score += min(hist_bonus, 0.20)
            motivos.append(
                f"{caso.historico_pagamentos} pagtos ok"
            )

        # Influência dos outros agentes (rodada 2)
        if argumentos_anteriores:
            votos_outros = [a.voto for a in argumentos_anteriores]
            unanime_rejeitar = all(
                v == Voto.REJEITAR for v in votos_outros
            )
            unanime_aprovar = all(
                v == Voto.APROVAR for v in votos_outros
            )
            if unanime_rejeitar:
                score -= 0.15
                motivos.append("outros agentes rejeitaram")
            elif unanime_aprovar:
                score += 0.10
                motivos.append("outros agentes aprovaram")

        # Decisão
        score = max(0.0, min(1.0, score))
        if score >= 0.65:
            voto = Voto.APROVAR
        elif score <= 0.35:
            voto = Voto.REJEITAR
        else:
            voto = Voto.ESCALAR

        justificativa = (
            f"Score {score:.2f}: " + "; ".join(motivos)
        )
        return ArgumentoAgente(
            perspectiva=self.perspectiva,
            voto=voto,
            confianca=abs(score - 0.5) * 2,
            justificativa=justificativa,
        )


# ============================================================
# 3. MODERADOR
# ============================================================

class Moderador:
    """
    Agrega votos e determina decisão final.
    Empate → voto do agente NEUTRO prevalece.
    """

    def decidir(
        self, votos: list[ArgumentoAgente]
    ) -> tuple[Voto, bool, str]:
        """Retorna (voto_final, consenso, resumo)."""
        contagem: dict[Voto, int] = {
            Voto.APROVAR: 0,
            Voto.REJEITAR: 0,
            Voto.ESCALAR: 0,
        }
        for a in votos:
            contagem[a.voto] += 1

        maximo = max(contagem.values())
        vencedores = [
            v for v, c in contagem.items() if c == maximo
        ]
        consenso = len(vencedores) == 1 and maximo == len(votos)

        if len(vencedores) == 1:
            decisao = vencedores[0]
            resumo = (
                f"{'Consenso' if consenso else 'Maioria'}: "
                f"{decisao.value} "
                f"({maximo}/{len(votos)} votos)"
            )
        else:
            # Empate → voto do NEUTRO
            neutro = next(
                a for a in votos
                if a.perspectiva == Perspectiva.NEUTRO
            )
            decisao = neutro.voto
            resumo = (
                f"Empate — voto do agente neutro prevalece: "
                f"{decisao.value}"
            )

        return decisao, consenso, resumo


# ============================================================
# 4. SISTEMA DE DEBATE
# ============================================================

class SistemaDebate:
    """
    Orquestra o debate em duas rodadas:
    Rodada 1 — análise independente
    Rodada 2 — análise com argumentos dos outros
    """

    def __init__(self) -> None:
        self._agentes = [
            AgentePerspectiva(Perspectiva.CONSERVADOR),
            AgentePerspectiva(Perspectiva.NEUTRO),
            AgentePerspectiva(Perspectiva.LIBERAL),
        ]
        self._moderador = Moderador()

    def debater(self, caso: CasoDebate) -> ResultadoDebate:
        """Executa o debate para um caso, retornando o resultado final."""
        # Rodada 1: análise independente
        votos_r1 = [a.analisar(caso) for a in self._agentes]

        # Rodada 2: com visibilidade dos outros
        votos_r2 = [
            a.analisar(caso, argumentos_anteriores=votos_r1)
            for a in self._agentes
        ]

        decisao, consenso, resumo = self._moderador.decidir(
            votos_r2
        )
        return ResultadoDebate(
            caso_id=caso.caso_id,
            votos=votos_r2,
            decisao_final=decisao,
            consenso=consenso,
            resumo=resumo,
        )


# ============================================================
# 5. DEMO
# ============================================================

_CASOS: list[CasoDebate] = [
    CasoDebate(
        caso_id="caso-01",
        descricao="Fornecedor conhecido, NF-e ok, valor baixo",
        valor=2_500.0,
        cnpj_valido=True,
        fornecedor_novo=False,
        possui_nfe=True,
        historico_pagamentos=12,
    ),
    CasoDebate(
        caso_id="caso-02",
        descricao="Fornecedor novo, valor alto, sem NF-e",
        valor=45_000.0,
        cnpj_valido=True,
        fornecedor_novo=True,
        possui_nfe=False,
        historico_pagamentos=0,
    ),
    CasoDebate(
        caso_id="caso-03",
        descricao="Valor limítrofe, NF-e presente, 3 pagtos ok",
        valor=9_800.0,
        cnpj_valido=True,
        fornecedor_novo=False,
        possui_nfe=True,
        historico_pagamentos=3,
    ),
]

_COR_VOTO = {
    Voto.APROVAR: "green",
    Voto.REJEITAR: "red",
    Voto.ESCALAR: "yellow",
}


def demo_debate_multiagente() -> None:
    """Executa a demonstração do debate multiagente."""
    console.print(
        Panel(
            "[bold]Módulo 46 — Multi-Agent Debate[/]\n"
            "3 agentes com perspectivas diferentes debatem "
            "em 2 rodadas e votam na decisão final",
            style="bold blue",
        )
    )

    sistema = SistemaDebate()
    resultados: list[ResultadoDebate] = []

    for caso in _CASOS:
        console.rule(
            f"[yellow]{caso.caso_id}: {caso.descricao}"
        )
        resultado = sistema.debater(caso)
        resultados.append(resultado)

        tabela = Table(header_style="bold magenta")
        tabela.add_column("Agente")
        tabela.add_column("Voto")
        tabela.add_column("Confiança", justify="right")
        tabela.add_column("Justificativa")

        for arg in resultado.votos:
            cor = _COR_VOTO[arg.voto]
            tabela.add_row(
                arg.perspectiva.value.capitalize(),
                f"[{cor}]{arg.voto.value}[/]",
                f"{arg.confianca:.0%}",
                arg.justificativa[:55],
            )
        console.print(tabela)

        cor_final = _COR_VOTO[resultado.decisao_final]
        console.print(
            Panel(
                f"[{cor_final}][bold]{resultado.resumo}[/]",
                title="Decisão Final",
                style=cor_final,
            )
        )

    # Resumo geral
    console.rule("[yellow]Resumo Geral")
    tabela_res = Table(header_style="bold magenta")
    tabela_res.add_column("Caso")
    tabela_res.add_column("Decisão")
    tabela_res.add_column("Consenso?", justify="center")

    for r in resultados:
        cor = _COR_VOTO[r.decisao_final]
        tabela_res.add_row(
            r.caso_id,
            f"[{cor}]{r.decisao_final.value}[/]",
            "✓" if r.consenso else "—",
        )
    console.print(tabela_res)


if __name__ == "__main__":
    demo_debate_multiagente()
