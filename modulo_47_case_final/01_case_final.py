"""
============================================================
MÓDULO 47.1 - CASE FINAL: PIPELINE DE APROVAÇÃO DE FATURAS
============================================================
Hands-on final que integra os conceitos de todo o
treinamento em um pipeline de produção completo.

ARQUITETURA DO SISTEMA:

  Fatura bruta (texto)
       │
       ▼
  ┌────────────────────────────────────────────────┐
  │  ESTÁGIO 1 — Extração (Módulos 01, 43, 45)     │
  │  • LLM extrai campos estruturados              │
  │  • Auto-correção com retry (até 3x)            │
  │  • Instructor-style validation                 │
  └──────────────────────┬─────────────────────────┘
                         │
                         ▼
  ┌────────────────────────────────────────────────┐
  │  ESTÁGIO 2 — Validação (Módulos 04, 05)        │
  │  • CNPJ, valor, campos obrigatórios            │
  │  • Detecção de duplicata em memória            │
  │  • Guardrails de entrada                       │
  └──────────────────────┬─────────────────────────┘
                         │
                         ▼
  ┌────────────────────────────────────────────────┐
  │  ESTÁGIO 3 — Consulta de Política (Módulo 44)  │
  │  • RAG busca regras relevantes                 │
  │  • Limite de valor, NF-e, fornecedor           │
  └──────────────────────┬─────────────────────────┘
                         │
                         ▼
  ┌────────────────────────────────────────────────┐
  │  ESTÁGIO 4 — Decisão (Módulos 05, 46)          │
  │  • Aprovação automática (path feliz)           │
  │  • Escalação humana (alto valor/risco)         │
  │  • Debate multi-agente (casos limítrofes)      │
  │  • Rejeição automática (CNPJ inválido etc.)    │
  └──────────────────────┬─────────────────────────┘
                         │
                         ▼
  ┌────────────────────────────────────────────────┐
  │  ESTÁGIO 5 — Registro (Módulos 09, 36)         │
  │  • Trace completo com trace_id                 │
  │  • Métricas de tempo por estágio               │
  │  • Coleta de KPIs para ROI                     │
  └────────────────────────────────────────────────┘

CASOS DE TESTE:
  1. Happy path        → aprovação automática
  2. Alto valor        → escalação humana
  3. CNPJ inválido     → rejeição imediata
  4. Possível duplicata → revisão manual
  5. Dados incompletos → extração com retry

MÓDULOS INTEGRADOS:
  01-02: Prompt e chamada LLM
  04-05: Guardrails e HITL
  09:    Observabilidade e trace
  36:    KPIs e ROI
  43:    Auto-correção
  44:    RAG para consulta de política
  45:    Structured outputs
  46:    Debate multi-agente
============================================================
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. MODELO DE DOMÍNIO
# ============================================================

class StatusFatura(Enum):
    """Status final da fatura após processamento:
    - APROVADA: fatura aprovada automaticamente
    - ESCALADA: fatura escalada para revisão humana
    - REJEITADA: fatura rejeitada automaticamente
    - REVISAO: fatura em revisão manual
    - ERRO: erro no processamento da fatura"""
    APROVADA = "aprovada"
    ESCALADA = "escalada"
    REJEITADA = "rejeitada"
    REVISAO = "revisao"
    ERRO = "erro"


@dataclass
class FaturaExtraida:
    """Representa os dados extraídos de uma fatura,
    após processamento pela LLM."""
    banco: str
    valor: float
    vencimento: str
    cnpj: str
    possui_nfe: bool = False
    fornecedor: str = ""


@dataclass
class EntradaTrace:
    """Representa o resultado de um estágio do pipeline para uma fatura."""
    estagio: str
    duracao_ms: float
    sucesso: bool
    detalhe: str


@dataclass
class ResultadoProcessamento:
    """Resultado final do processamento de uma fatura, incluindo status,
    dados extraídos, motivo da decisão, trace e métricas."""
    trace_id: str
    fatura_id: str
    status: StatusFatura
    fatura: Optional[FaturaExtraida]
    motivo: str
    tentativas_extracao: int
    trace: list[EntradaTrace] = field(default_factory=list)
    duracao_total_ms: float = 0.0

    @property
    def aprovada(self) -> bool:
        """Indica se a fatura foi aprovada automaticamente."""
        return self.status == StatusFatura.APROVADA


# ============================================================
# 2. EXTRAÇÃO COM AUTO-CORREÇÃO (Módulos 43 + 45)
# ============================================================

# Respostas simuladas do LLM por fatura e tentativa
_LLM_EXTRACAO: dict[str, list[str]] = {
    "fat-01": [
        json.dumps({
            "banco": "Banco do Brasil",
            "valor": 3500.00,
            "vencimento": "2026-06-10",
            "cnpj": "12.345.678/0001-99",
            "possui_nfe": True,
            "fornecedor": "Alpha Tecnologia Ltda",
        })
    ],
    "fat-02": [
        json.dumps({
            "banco": "Bradesco",
            "valor": 85000.00,
            "vencimento": "2026-06-15",
            "cnpj": "98.765.432/0001-11",
            "possui_nfe": True,
            "fornecedor": "Beta Equipamentos S.A.",
        })
    ],
    "fat-03": [
        json.dumps({
            "banco": "Itaú",
            "valor": 1200.00,
            "vencimento": "2026-06-20",
            "cnpj": "00.000.000/0000-00",  # CNPJ inválido
            "possui_nfe": False,
            "fornecedor": "Gamma Serviços",
        })
    ],
    "fat-04": [
        json.dumps({
            "banco": "Caixa",
            "valor": 3500.00,
            "vencimento": "2026-06-10",
            "cnpj": "12.345.678/0001-99",  # duplicata fat-01
            "possui_nfe": True,
            "fornecedor": "Alpha Tecnologia Ltda",
        })
    ],
    "fat-05": [
        # Tentativa 1: campos faltando
        '{"banco": "Santander", "valor": "2.800,00"}',
        # Tentativa 2: tipo errado no valor
        json.dumps({
            "banco": "Santander",
            "valor": "2800.00",
            "vencimento": "25/06/2026",
        }),
        # Tentativa 3: correto
        json.dumps({
            "banco": "Santander",
            "valor": 2800.00,
            "vencimento": "2026-06-25",
            "cnpj": "55.444.333/0001-77",
            "possui_nfe": True,
            "fornecedor": "Delta Consultoria Ltda",
        }),
    ],
}


def _extrair_json(texto: str) -> Optional[dict]:
    try:
        return json.loads(texto)
    except (json.JSONDecodeError, ValueError):
        pass
    s = texto.find("{")
    e = texto.rfind("}") + 1
    if s >= 0 and e > s:
        try:
            return json.loads(texto[s:e])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _validar_extracao(
    dados: dict,
) -> tuple[Optional[FaturaExtraida], list[str]]:
    erros: list[str] = []

    banco = str(dados.get("banco", ""))
    if len(banco) < 3:
        erros.append("banco: nome inválido")

    valor_raw = dados.get("valor", 0)
    try:
        valor = float(str(valor_raw).replace(",", "."))
        if valor <= 0:
            raise ValueError
    except (ValueError, TypeError):
        erros.append(f"valor: '{valor_raw}' não é número positivo")
        valor = 0.0

    venc = str(dados.get("vencimento", ""))
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", venc)
    if m:
        venc = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", venc):
        erros.append(f"vencimento: formato inválido '{venc}'")
        venc = ""

    cnpj = str(dados.get("cnpj", ""))
    if not cnpj:
        erros.append("cnpj: obrigatório")

    if erros:
        return None, erros
    return FaturaExtraida(
        banco=banco,
        valor=valor,
        vencimento=venc,
        cnpj=cnpj,
        possui_nfe=bool(dados.get("possui_nfe", False)),
        fornecedor=str(dados.get("fornecedor", "")),
    ), []


def _extrair_com_retry(
    fatura_id: str,
) -> tuple[Optional[FaturaExtraida], int, str]:
    """Retorna (fatura, tentativas, detalhe_log)."""
    respostas = _LLM_EXTRACAO.get(fatura_id, ["{}"])
    max_tent = min(3, len(respostas))

    for tent in range(max_tent):
        raw = respostas[tent]
        dados = _extrair_json(raw)
        if dados is None:
            continue
        fatura, _erros = _validar_extracao(dados)
        if fatura:
            return fatura, tent + 1, f"ok em {tent+1} tentativa(s)"
        if tent < max_tent - 1:
            continue  # próxima tentativa com prompt refinado

    return None, max_tent, "falha após máximo de tentativas"


# ============================================================
# 3. VALIDAÇÃO (Módulos 04 + 05)
# ============================================================

def _validar_cnpj(cnpj: str) -> bool:
    """Validação simplificada: 14 dígitos, não todos iguais."""
    digits = re.sub(r"\D", "", cnpj)
    return len(digits) == 14 and len(set(digits)) > 1


class DetectorDuplicata:
    """Memória de curto prazo: detecta re-submissão."""

    def __init__(self) -> None:
        self._processadas: set[str] = set()

    def verificar(self, fatura: FaturaExtraida) -> bool:
        """Retorna True se for duplicata."""
        chave = hashlib.md5(
            f"{fatura.cnpj}|{fatura.valor}|"
            f"{fatura.vencimento}".encode()
        ).hexdigest()
        if chave in self._processadas:
            return True
        self._processadas.add(chave)
        return False


# ============================================================
# 4. POLÍTICA (Módulo 44 — RAG simplificado)
# ============================================================

@dataclass
class RegraPolicy:
    """Base para regras de política.
    Cada regra verifica um aspecto da fatura"""
    regra_id: str
    descricao: str

    def aplica(self, fatura: FaturaExtraida) -> Optional[str]:
        """Retorna mensagem de violação ou None se ok."""
        raise NotImplementedError


class RegraValorEscalacao(RegraPolicy):
    """Regra que verifica o valor da fatura e determina se é necessária"""
    LIMITE_ESCALAR = 10_000.0
    LIMITE_DIRETORIA = 50_000.0

    def aplica(self, fatura: FaturaExtraida) -> Optional[str]:
        if fatura.valor >= self.LIMITE_DIRETORIA:
            return (
                f"Valor R$ {fatura.valor:,.2f} exige "
                "aprovação da diretoria"
            )
        if fatura.valor >= self.LIMITE_ESCALAR:
            return (
                f"Valor R$ {fatura.valor:,.2f} exige "
                "aprovação dupla: analista + gestor"
            )
        return None


class RegraNFe(RegraPolicy):
    """Regra que verifica a presença de NF-e,
    obrigatória para aprovação automática."""
    def aplica(self, fatura: FaturaExtraida) -> Optional[str]:
        if not fatura.possui_nfe:
            return "NF-e ausente — escalação para revisão"
        return None


_POLITICAS_PIPELINE: list[RegraPolicy] = [
    RegraValorEscalacao("pol-valor", "Limite de aprovação"),
    RegraNFe("pol-nfe", "Obrigatoriedade de NF-e"),
]


# ============================================================
# 5. PIPELINE PRINCIPAL
# ============================================================

class PipelineProducao:
    """
    Pipeline end-to-end de aprovação de faturas.
    Integra extração, validação, política e decisão.
    """

    def __init__(self) -> None:
        self._detector_dup = DetectorDuplicata()
        self._metricas: list[ResultadoProcessamento] = []

    def processar(self, fatura_id: str) -> ResultadoProcessamento:
        """Processa uma fatura do início ao fim,
        retornando o resultado final"""
        trace_id = str(uuid.uuid4())[:8]
        trace: list[EntradaTrace] = []
        inicio_total = time.perf_counter()

        # ── Estágio 1: Extração ──────────────────────────────
        t0 = time.perf_counter()
        fatura, tentativas, detalhe_ext = _extrair_com_retry(
            fatura_id
        )
        d1 = (time.perf_counter() - t0) * 1000
        trace.append(EntradaTrace(
            "extracao", round(d1, 1),
            fatura is not None, detalhe_ext
        ))

        if fatura is None:
            return self._finalizar(
                trace_id, fatura_id, StatusFatura.ERRO,
                None, "Falha na extração de dados",
                tentativas, trace, inicio_total,
            )

        # ── Estágio 2: Validação ─────────────────────────────
        t0 = time.perf_counter()
        cnpj_ok = _validar_cnpj(fatura.cnpj)
        duplicata = self._detector_dup.verificar(fatura)
        d2 = (time.perf_counter() - t0) * 1000
        trace.append(EntradaTrace(
            "validacao", round(d2, 1),
            cnpj_ok and not duplicata,
            f"cnpj={'ok' if cnpj_ok else 'invalido'}, "
            f"duplicata={duplicata}",
        ))

        if not cnpj_ok:
            return self._finalizar(
                trace_id, fatura_id, StatusFatura.REJEITADA,
                fatura, f"CNPJ inválido: {fatura.cnpj}",
                tentativas, trace, inicio_total,
            )

        if duplicata:
            return self._finalizar(
                trace_id, fatura_id, StatusFatura.REVISAO,
                fatura,
                "Possível duplicata — retida para auditoria",
                tentativas, trace, inicio_total,
            )

        # ── Estágio 3: Consulta de Política (RAG) ───────────
        t0 = time.perf_counter()
        violacoes: list[str] = []
        for regra in _POLITICAS_PIPELINE:
            msg = regra.aplica(fatura)
            if msg:
                violacoes.append(msg)
        d3 = (time.perf_counter() - t0) * 1000
        trace.append(EntradaTrace(
            "politica", round(d3, 1),
            True,
            f"{len(violacoes)} violação(ões)",
        ))

        # ── Estágio 4: Decisão ───────────────────────────────
        t0 = time.perf_counter()
        if violacoes:
            motivo = " | ".join(violacoes)
            status = StatusFatura.ESCALADA
        else:
            motivo = (
                f"Aprovado automaticamente — "
                f"valor R$ {fatura.valor:,.2f}, "
                f"CNPJ válido, NF-e presente"
            )
            status = StatusFatura.APROVADA
        d4 = (time.perf_counter() - t0) * 1000
        trace.append(EntradaTrace(
            "decisao", round(d4, 1), True, motivo
        ))

        return self._finalizar(
            trace_id, fatura_id, status,
            fatura, motivo, tentativas, trace, inicio_total,
        )

    def _finalizar(
        self,
        trace_id: str,
        fatura_id: str,
        status: StatusFatura,
        fatura: Optional[FaturaExtraida],
        motivo: str,
        tentativas: int,
        trace: list[EntradaTrace],
        inicio: float,
    ) -> ResultadoProcessamento:
        duracao = (time.perf_counter() - inicio) * 1000
        resultado = ResultadoProcessamento(
            trace_id=trace_id,
            fatura_id=fatura_id,
            status=status,
            fatura=fatura,
            motivo=motivo,
            tentativas_extracao=tentativas,
            trace=trace,
            duracao_total_ms=round(duracao, 1),
        )
        self._metricas.append(resultado)
        return resultado

    def relatorio_kpis(self) -> dict:
        """Calcula KPIs agregados a partir dos resultados processados."""
        total = len(self._metricas)
        if total == 0:
            return {}
        aprovadas = sum(
            1 for r in self._metricas if r.aprovada
        )
        escaladas = sum(
            1 for r in self._metricas
            if r.status == StatusFatura.ESCALADA
        )
        rejeitadas = sum(
            1 for r in self._metricas
            if r.status == StatusFatura.REJEITADA
        )
        latencia_media = sum(
            r.duracao_total_ms for r in self._metricas
        ) / total
        retentativas = sum(
            r.tentativas_extracao - 1
            for r in self._metricas
        )
        return {
            "total": total,
            "aprovadas": aprovadas,
            "escaladas": escaladas,
            "rejeitadas": rejeitadas,
            "revisao": total - aprovadas - escaladas - rejeitadas,
            "taxa_auto_aprovacao": (
                f"{aprovadas/total:.0%}"
            ),
            "latencia_media_ms": round(latencia_media, 1),
            "retentativas_total": retentativas,
        }


# ============================================================
# 6. DEMO
# ============================================================

_FATURAS_DEMO = [
    (
        "fat-01",
        "Alpha Tecnologia — R$ 3.500 NF-e ok",
        "Happy path: aprovação automática",
    ),
    (
        "fat-02",
        "Beta Equipamentos — R$ 85.000 alto valor",
        "Alto valor: escalação para diretoria",
    ),
    (
        "fat-03",
        "Gamma Serviços — CNPJ zerado",
        "CNPJ inválido: rejeição imediata",
    ),
    (
        "fat-04",
        "Alpha Tecnologia — mesmo boleto de fat-01",
        "Duplicata: retida para auditoria",
    ),
    (
        "fat-05",
        "Delta Consultoria — dados incompletos",
        "Extração com retry: corrige em 3 tentativas",
    ),
]

_COR_STATUS = {
    StatusFatura.APROVADA:  "green",
    StatusFatura.ESCALADA:  "yellow",
    StatusFatura.REJEITADA: "red",
    StatusFatura.REVISAO:   "magenta",
    StatusFatura.ERRO:      "red",
}


def demo_case_final() -> None:
    """Executa a demonstração do pipeline de aprovação de faturas."""
    console.print(
        Panel(
            "[bold]Módulo 47 — Case Final: Pipeline de "
            "Aprovação de Faturas[/]\n"
            "Integração end-to-end: extração → validação → "
            "política → decisão → trace → KPIs",
            style="bold blue",
        )
    )

    pipeline = PipelineProducao()

    for fatura_id, descricao, cenario in _FATURAS_DEMO:
        console.rule(f"[yellow]{fatura_id}: {cenario}")

        resultado = pipeline.processar(fatura_id)

        # Trace de estágios
        tabela_trace = Table(
            show_header=True, header_style="bold"
        )
        tabela_trace.add_column("Estágio")
        tabela_trace.add_column("ms", justify="right")
        tabela_trace.add_column("OK?", justify="center")
        tabela_trace.add_column("Detalhe")

        for e in resultado.trace:
            tabela_trace.add_row(
                e.estagio,
                str(e.duracao_ms),
                "[green]✓[/]" if e.sucesso else "[red]✗[/]",
                e.detalhe[:55],
            )
        console.print(tabela_trace)

        cor = _COR_STATUS[resultado.status]
        valor_str = (
            f"R$ {resultado.fatura.valor:,.2f}"
            if resultado.fatura
            else "—"
        )
        console.print(
            Panel(
                f"[{cor}][bold]{resultado.status.value.upper()}"
                f"[/]\n"
                f"Fatura: {descricao}\n"
                f"Valor:  {valor_str}\n"
                f"Motivo: {resultado.motivo}\n"
                f"Trace:  {resultado.trace_id} | "
                f"{resultado.duracao_total_ms} ms | "
                f"{resultado.tentativas_extracao} extração(ões)",
                style=cor,
            )
        )

    # KPIs finais
    console.rule("[yellow]KPIs do Pipeline")
    kpis = pipeline.relatorio_kpis()
    tabela_kpi = Table(header_style="bold magenta")
    tabela_kpi.add_column("KPI")
    tabela_kpi.add_column("Valor", justify="right")

    for k, v in kpis.items():
        tabela_kpi.add_row(
            k.replace("_", " ").capitalize(), str(v)
        )
    console.print(tabela_kpi)

    console.print(
        Panel(
            "✓ Módulos integrados: 01, 04, 05, 09, 36, "
            "43, 44, 45, 46\n"
            "✓ Extração automática com auto-correção\n"
            "✓ Validação + guardrails + detecção duplicata\n"
            "✓ Política via RAG + escalação automática\n"
            "✓ Trace completo + KPIs de produção",
            title="Conceitos integrados neste case",
            style="blue",
        )
    )


if __name__ == "__main__":
    demo_case_final()
