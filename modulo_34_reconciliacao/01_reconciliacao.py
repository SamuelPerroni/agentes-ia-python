"""
============================================================
MÓDULO 34.1 - RECONCILIAÇÃO E CONCILIAÇÃO FINANCEIRA
============================================================
Neste módulo, aprendemos a construir um agente que compara
automaticamente o extrato bancário com os lançamentos do
ERP, identifica divergências e classifica cada diferença.

CONCEITO CHAVE:
Conciliação bancária é uma das tarefas financeiras que mais
consome tempo de analistas — tipicamente 2 a 4 horas por
dia em empresas médias. O agente APA automatiza 80-90%
dos casos e escala para revisão humana apenas as exceções.

TIPOS DE DIVERGÊNCIA:

  ┌──────────────────────────────────────────────────────────┐
  │  Tipo               │ Descrição                         │
  │─────────────────────┼───────────────────────────────────│
  │  DUPLICIDADE        │ Mesmo lançamento 2x no ERP        │
  │  AUSENTE_ERP        │ Débito no banco sem lançamento    │
  │  AUSENTE_BANCO      │ Lançamento no ERP sem débito      │
  │  DIFERENCA_VALOR    │ Valores não batem                 │
  │  DIFERENCA_DATA     │ Datas diferentes (D+1, D+2)       │
  │  OK                 │ Conciliado                        │
  └──────────────────────────────────────────────────────────┘

FLUXO DO AGENTE:

  Extrato Bancário   Lançamentos ERP
        │                  │
        └──────┬───────────┘
               ▼
  [ Normalizador ] → padroniza datas, valores, descrições
               │
               ▼
  [ Matcher ] → cruza por valor+data (tolerância ±1 dia)
               │
               ▼
  [ Classificador ] → rotula cada divergência
               │
               ▼
  [ LLM ] → descreve divergências em linguagem natural
               │
               ▼
  [ Relatório ] → tabela + sumário executivo

Tópicos cobertos:
1. Dataclasses para lançamento bancário e ERP
2. Normalizador de dados (datas, valores, descrições)
3. Algoritmo de matching com tolerância de data
4. Classificação de divergências em 5 categorias
5. Geração de sumário executivo pelo LLM (simulado)
6. Relatório final com estatísticas de conciliação
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. MODELOS DE DADOS
# ============================================================

@dataclass
class LancamentoBanco:
    """Linha do extrato bancário."""
    id: str
    data: date
    descricao: str
    valor: float          # negativo = débito, positivo = crédito
    documento: str = ""   # número do documento se disponível


@dataclass
class LancamentoERP:
    """Lançamento contábil no ERP."""
    id: str
    data: date
    descricao: str
    valor: float
    fornecedor: str = ""
    documento: str = ""


class TipoDivergencia(str, Enum):
    """Tipos de divergência encontrados na conciliação bancária."""
    OK = "OK"
    DUPLICIDADE = "DUPLICIDADE"
    AUSENTE_ERP = "AUSENTE_ERP"
    AUSENTE_BANCO = "AUSENTE_BANCO"
    DIFERENCA_VALOR = "DIFERENCA_VALOR"
    DIFERENCA_DATA = "DIFERENCA_DATA"


@dataclass
class ResultadoConciliacao:
    """Resultado do cruzamento entre um par de lançamentos."""
    banco_id: Optional[str]
    erp_id: Optional[str]
    tipo: TipoDivergencia
    valor_banco: Optional[float]
    valor_erp: Optional[float]
    data_banco: Optional[date]
    data_erp: Optional[date]
    descricao: str
    requer_revisao: bool


# ============================================================
# 2. NORMALIZADOR
# ============================================================

def _normalizar_valor(valor: float) -> float:
    """Arredonda para 2 casas decimais."""
    return round(abs(valor), 2)


def _normalizar_descricao(desc: str) -> str:
    """
    Remove caracteres de ruído para melhorar o matching.
    ERP pode ter "PAG FORNEC 001" enquanto banco tem
    "PAG FORN 001" — heurística simples aqui.
    """
    return (
        desc.upper()
        .replace("FORNEC", "FORN")
        .replace("PAGAMENTO", "PAG")
        .replace("  ", " ")
        .strip()
    )


# ============================================================
# 3. ENGINE DE CONCILIAÇÃO
# ============================================================

class EngineConciliacao:
    """
    Cruza lançamentos bancários com lançamentos do ERP
    e classifica divergências.
    """

    # Tolerância de ±N dias para matching por data
    TOLERANCIA_DIAS = 1
    # Tolerância de valor em R$ (para arredondamentos)
    TOLERANCIA_VALOR = 0.05

    def conciliar(
        self,
        extrato: list[LancamentoBanco],
        erp: list[LancamentoERP],
    ) -> list[ResultadoConciliacao]:
        """Realiza a conciliação entre extrato bancário e lançamentos ERP."""
        resultados: list[ResultadoConciliacao] = []
        erp_nao_casados: set[str] = {e.id for e in erp}
        banco_nao_casados: set[str] = {b.id for b in extrato}

        # Detecta duplicidade no ERP (mesmo valor+data+desc)
        for i, a in enumerate(erp):
            for b in erp[i + 1:]:
                if (
                    _normalizar_valor(a.valor)
                    == _normalizar_valor(b.valor)
                    and abs((a.data - b.data).days) == 0
                ):
                    resultados.append(ResultadoConciliacao(
                        banco_id=None,
                        erp_id=f"{a.id}+{b.id}",
                        tipo=TipoDivergencia.DUPLICIDADE,
                        valor_banco=None,
                        valor_erp=_normalizar_valor(a.valor),
                        data_banco=None,
                        data_erp=a.data,
                        descricao=(
                            f"Lançamento duplicado no ERP: "
                            f"{a.descricao}"
                        ),
                        requer_revisao=True,
                    ))

        # Cruzamento principal: banco × ERP
        for lb in extrato:
            val_b = _normalizar_valor(lb.valor)
            casado = False
            for le in erp:
                if le.id not in erp_nao_casados:
                    continue
                val_e = _normalizar_valor(le.valor)
                diff_val = abs(val_b - val_e)
                diff_dias = abs((lb.data - le.data).days)

                if diff_val <= self.TOLERANCIA_VALOR:
                    if diff_dias == 0:
                        # Match perfeito
                        resultados.append(
                            ResultadoConciliacao(
                                banco_id=lb.id,
                                erp_id=le.id,
                                tipo=TipoDivergencia.OK,
                                valor_banco=val_b,
                                valor_erp=val_e,
                                data_banco=lb.data,
                                data_erp=le.data,
                                descricao=lb.descricao,
                                requer_revisao=False,
                            )
                        )
                        erp_nao_casados.discard(le.id)
                        banco_nao_casados.discard(lb.id)
                        casado = True
                        break
                    elif diff_dias <= self.TOLERANCIA_DIAS:
                        # Match com diferença de data
                        resultados.append(
                            ResultadoConciliacao(
                                banco_id=lb.id,
                                erp_id=le.id,
                                tipo=TipoDivergencia.DIFERENCA_DATA,
                                valor_banco=val_b,
                                valor_erp=val_e,
                                data_banco=lb.data,
                                data_erp=le.data,
                                descricao=lb.descricao,
                                requer_revisao=False,
                            )
                        )
                        erp_nao_casados.discard(le.id)
                        banco_nao_casados.discard(lb.id)
                        casado = True
                        break
                elif diff_dias <= self.TOLERANCIA_DIAS:
                    # Valores diferentes
                    resultados.append(
                        ResultadoConciliacao(
                            banco_id=lb.id,
                            erp_id=le.id,
                            tipo=TipoDivergencia.DIFERENCA_VALOR,
                            valor_banco=val_b,
                            valor_erp=val_e,
                            data_banco=lb.data,
                            data_erp=le.data,
                            descricao=lb.descricao,
                            requer_revisao=True,
                        )
                    )
                    erp_nao_casados.discard(le.id)
                    banco_nao_casados.discard(lb.id)
                    casado = True
                    break
            if not casado and lb.id in banco_nao_casados:
                # Débito no banco sem lançamento no ERP
                resultados.append(ResultadoConciliacao(
                    banco_id=lb.id,
                    erp_id=None,
                    tipo=TipoDivergencia.AUSENTE_ERP,
                    valor_banco=val_b,
                    valor_erp=None,
                    data_banco=lb.data,
                    data_erp=None,
                    descricao=lb.descricao,
                    requer_revisao=True,
                ))

        # Lançamentos no ERP sem débito bancário
        for le in erp:
            if le.id in erp_nao_casados:
                resultados.append(ResultadoConciliacao(
                    banco_id=None,
                    erp_id=le.id,
                    tipo=TipoDivergencia.AUSENTE_BANCO,
                    valor_banco=None,
                    valor_erp=_normalizar_valor(le.valor),
                    data_banco=None,
                    data_erp=le.data,
                    descricao=le.descricao,
                    requer_revisao=True,
                ))

        return resultados


# ============================================================
# 4. GERADOR DE SUMÁRIO (SIMULADO LLM)
# ============================================================

def gerar_sumario_llm(
    resultados: list[ResultadoConciliacao],
) -> str:
    """
    Gera sumário executivo em linguagem natural.
    Em produção, envia o relatório ao LLM com instrução:
    'Resuma as divergências de conciliação abaixo em
     linguagem de negócio para o controller financeiro.'
    """
    total = len(resultados)
    ok = sum(
        1 for r in resultados if r.tipo == TipoDivergencia.OK
    )
    divergencias = total - ok
    revisao = sum(1 for r in resultados if r.requer_revisao)

    linhas = [
        f"Conciliação do período analisou {total} "
        f"lançamentos.",
        f"{ok} lançamentos conciliados automaticamente "
        f"({ok/total:.0%} taxa de conciliação).",
    ]
    if divergencias:
        linhas.append(
            f"{divergencias} divergência(s) encontrada(s), "
            f"sendo {revisao} que requer(em) revisão humana."
        )

    tipos = {}
    for r in resultados:
        if r.tipo != TipoDivergencia.OK:
            tipos[r.tipo.value] = tipos.get(
                r.tipo.value, 0
            ) + 1
    for tipo, qtd in sorted(tipos.items()):
        linhas.append(f"  • {tipo}: {qtd} ocorrência(s)")

    return "\n".join(linhas)


# ============================================================
# 5. DEMO
# ============================================================

def demo_reconciliacao() -> None:
    """Demonstration do processo de conciliação bancária automatizada."""
    console.print(
        Panel(
            "[bold]Módulo 34 — Reconciliação e Conciliação "
            "Financeira[/]\n"
            "Extrato bancário × ERP: identificação "
            "automática de divergências",
            style="bold blue",
        )
    )

    hoje = date(2026, 4, 3)

    # Extrato bancário
    extrato: list[LancamentoBanco] = [
        LancamentoBanco(
            "B001", hoje - timedelta(3),
            "PAG FORN ALPHA", 1500.0,
        ),
        LancamentoBanco(
            "B002", hoje - timedelta(2),
            "PAG FORN BETA", 890.0,
        ),
        LancamentoBanco(
            "B003", hoje - timedelta(1),
            "PAG FORN GAMMA", 3200.0,
        ),
        LancamentoBanco(
            "B004", hoje,
            "PAG FORN DELTA", 2100.0,
        ),
        LancamentoBanco(
            "B005", hoje,
            "TARIFA BANCARIA", 45.0,
        ),
    ]

    # Lançamentos ERP
    erp: list[LancamentoERP] = [
        LancamentoERP(
            "E001", hoje - timedelta(3),
            "Pagamento Fornecedor Alpha",
            1500.0, "Alpha Ltda",
        ),
        LancamentoERP(
            "E002", hoje - timedelta(1),
            "Pagamento Fornecedor Beta",
            890.0, "Beta S.A.",  # data D-1
        ),
        LancamentoERP(
            "E003", hoje - timedelta(1),
            "Pagamento Fornecedor Gamma",
            3100.0, "Gamma ME",  # valor diferente
        ),
        LancamentoERP(
            "E004", hoje,
            "Pagamento Fornecedor Delta",
            2100.0, "Delta Tech",
        ),
        # Duplicidade:
        LancamentoERP(
            "E005", hoje,
            "Pagamento Fornecedor Delta",
            2100.0, "Delta Tech",
        ),
        # Ausente no banco:
        LancamentoERP(
            "E006", hoje - timedelta(5),
            "Seguro Empresarial",
            750.0, "Seguradora XYZ",
        ),
    ]

    engine = EngineConciliacao()
    resultados = engine.conciliar(extrato, erp)

    # Exibe resultados
    cores = {
        TipoDivergencia.OK:              "[green]",
        TipoDivergencia.DIFERENCA_DATA:  "[yellow]",
        TipoDivergencia.DIFERENCA_VALOR: "[red]",
        TipoDivergencia.AUSENTE_ERP:     "[red]",
        TipoDivergencia.AUSENTE_BANCO:   "[red]",
        TipoDivergencia.DUPLICIDADE:     "[magenta]",
    }

    tabela = Table(
        title="Resultado da Conciliação",
        header_style="bold magenta",
        show_lines=True,
    )
    tabela.add_column("Banco ID")
    tabela.add_column("ERP ID")
    tabela.add_column("Tipo")
    tabela.add_column("Val.Banco", justify="right")
    tabela.add_column("Val.ERP", justify="right")
    tabela.add_column("Revisão")
    tabela.add_column("Descrição")

    for r in resultados:
        cor = cores.get(r.tipo, "")
        tabela.add_row(
            r.banco_id or "—",
            r.erp_id or "—",
            f"{cor}{r.tipo.value}[/]",
            f"{r.valor_banco:,.2f}" if r.valor_banco else "—",
            f"{r.valor_erp:,.2f}" if r.valor_erp else "—",
            "[red]Sim[/]" if r.requer_revisao
            else "[green]Não[/]",
            r.descricao[:35],
        )
    console.print(tabela)

    # Sumário executivo
    console.rule("[yellow]Sumário Executivo (LLM)")
    sumario = gerar_sumario_llm(resultados)
    console.print(
        Panel(sumario, title="Sumário", style="cyan")
    )


if __name__ == "__main__":
    demo_reconciliacao()
