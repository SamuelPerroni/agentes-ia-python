"""
============================================================
MÓDULO 36.1 - KPIs E RELATÓRIO DE ROI DO PROCESSO APA
============================================================
Neste módulo, aprendemos a medir o que realmente importa
em APA: quantos documentos foram processados, qual a taxa
de acerto, quanto tempo foi economizado e qual o ROI.

CONCEITO CHAVE:
Sem métricas, APA é um projeto de TI sem retorno visível
para a gestão. Com KPIs bem definidos, o agente justifica
seu custo, detecta degradação de qualidade e comprova
valor para stakeholders de negócio.

MÉTRICAS DE APA:

  ┌──────────────────────────────────────────────────────────┐
  │  VOLUME              │  QUALIDADE                        │
  │  ─────────────────── │  ──────────────────────────────   │
  │  docs processados/dia│  taxa de acerto (precision)       │
  │  docs com erro       │  taxa de revisão humana           │
  │  docs em fila        │  confiança média dos campos       │
  │                      │                                   │
  │  TEMPO               │  CUSTO                            │
  │  ─────────────────── │  ──────────────────────────────   │
  │  tempo médio/doc     │  custo por documento              │
  │  percentil p95       │  custo total de tokens (LLM)      │
  │  horas economizadas  │  ROI vs. processo manual          │
  └──────────────────────────────────────────────────────────┘

FÓRMULA DE ROI:

  Horas economizadas = docs_processados × tempo_manual
  Custo evitado = horas × custo_hora_analista
  ROI = (Custo evitado − Custo APA) / Custo APA × 100%

Tópicos cobertos:
1. Dataclass de registro de execução com campos de KPI
2. Coletor de métricas: acumula execuções, calcula stats
3. Cálculo de ROI com parâmetros configuráveis
4. Identificação de documentos de baixa confiança
5. Relatório executivo em Rich (tabelas + painel)
6. Exportação de métricas como dict (para Prometheus/API)
============================================================
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. REGISTRO DE EXECUÇÃO
# ============================================================

@dataclass
class RegistroExecucao:
    """
    Captura todos os dados relevantes de uma execução
    do agente para cálculo posterior de KPIs.
    """
    id: str
    timestamp: str
    tipo_documento: str        # boleto, nfe, contrato
    duracao_ms: float
    campos_extraidos: int
    campos_corretos: int       # validados por humano ou regra
    confianca: float           # 0-1 reportada pelo agente
    requer_revisao: bool
    custo_tokens_usd: float
    erro: Optional[str] = None

    @property
    def sucesso(self) -> bool:
        return self.erro is None

    @property
    def precisao(self) -> float:
        if self.campos_extraidos == 0:
            return 0.0
        return self.campos_corretos / self.campos_extraidos


# ============================================================
# 2. PARÂMETROS DE ROI
# ============================================================

@dataclass
class ParametrosROI:
    """
    Parâmetros para cálculo de ROI.
    Customizar conforme realidade do cliente.
    """
    # Tempo manual por documento (minutos)
    tempo_manual_min: float = 8.0
    # Custo hora do analista (R$)
    custo_hora_analista: float = 80.0
    # Custo fixo mensal da infra APA (R$)
    custo_infra_mensal: float = 500.0


# ============================================================
# 3. COLETOR DE MÉTRICAS
# ============================================================

class ColetorKPI:
    """
    Acumula registros de execução e calcula KPIs agregados.
    Thread-safe para uso em produção com lock se necessário.
    """

    def __init__(
        self, params: Optional[ParametrosROI] = None
    ) -> None:
        self._params = params or ParametrosROI()
        self._registros: list[RegistroExecucao] = []

    def registrar(self, exec_: RegistroExecucao) -> None:
        self._registros.append(exec_)

    # ---- KPIs de volume ----

    @property
    def total_processados(self) -> int:
        return len(self._registros)

    @property
    def total_erros(self) -> int:
        return sum(1 for r in self._registros if not r.sucesso)

    @property
    def total_revisao_humana(self) -> int:
        return sum(
            1 for r in self._registros if r.requer_revisao
        )

    # ---- KPIs de qualidade ----

    @property
    def taxa_sucesso(self) -> float:
        if not self._registros:
            return 0.0
        return 1 - self.total_erros / self.total_processados

    @property
    def taxa_revisao(self) -> float:
        if not self._registros:
            return 0.0
        return self.total_revisao_humana / self.total_processados

    @property
    def confianca_media(self) -> float:
        vals = [
            r.confianca for r in self._registros
            if r.sucesso
        ]
        return statistics.mean(vals) if vals else 0.0

    @property
    def precisao_media(self) -> float:
        vals = [
            r.precisao for r in self._registros
            if r.sucesso
        ]
        return statistics.mean(vals) if vals else 0.0

    # ---- KPIs de tempo ----

    @property
    def duracao_media_ms(self) -> float:
        vals = [
            r.duracao_ms for r in self._registros
            if r.sucesso
        ]
        return statistics.mean(vals) if vals else 0.0

    @property
    def duracao_p95_ms(self) -> float:
        vals = sorted(
            r.duracao_ms for r in self._registros
            if r.sucesso
        )
        if not vals:
            return 0.0
        idx = int(len(vals) * 0.95)
        return vals[min(idx, len(vals) - 1)]

    # ---- KPIs de custo e ROI ----

    @property
    def custo_total_tokens_usd(self) -> float:
        return sum(
            r.custo_tokens_usd for r in self._registros
        )

    def calcular_roi(self) -> dict[str, float]:
        """Calcula retorno sobre investimento do período."""
        p = self._params
        docs = self.total_processados
        # Documentos que não precisaram de revisão humana
        docs_automatizados = docs - self.total_revisao_humana

        horas_economizadas = (
            docs_automatizados * p.tempo_manual_min / 60
        )
        custo_evitado_brl = (
            horas_economizadas * p.custo_hora_analista
        )
        # Converte tokens USD → BRL (taxa aproximada)
        custo_apa_brl = (
            self.custo_total_tokens_usd * 5.8  # USD→BRL
            + p.custo_infra_mensal
        )
        roi_pct = (
            (custo_evitado_brl - custo_apa_brl)
            / custo_apa_brl * 100
            if custo_apa_brl > 0 else 0.0
        )
        return {
            "docs_automatizados": float(docs_automatizados),
            "horas_economizadas": round(horas_economizadas, 1),
            "custo_evitado_brl": round(custo_evitado_brl, 2),
            "custo_apa_brl": round(custo_apa_brl, 2),
            "roi_percentual": round(roi_pct, 1),
        }

    def documentos_baixa_confianca(
        self, limiar: float = 0.6
    ) -> list[RegistroExecucao]:
        """Retorna documentos abaixo do limiar de confiança."""
        return [
            r for r in self._registros
            if r.sucesso and r.confianca < limiar
        ]

    def exportar_dict(self) -> dict:
        """Exporta KPIs como dict para Prometheus/API."""
        roi = self.calcular_roi()
        return {
            "total_processados": self.total_processados,
            "total_erros": self.total_erros,
            "taxa_sucesso": round(self.taxa_sucesso, 4),
            "taxa_revisao": round(self.taxa_revisao, 4),
            "confianca_media": round(self.confianca_media, 4),
            "precisao_media": round(self.precisao_media, 4),
            "duracao_media_ms": round(self.duracao_media_ms, 1),
            "duracao_p95_ms": round(self.duracao_p95_ms, 1),
            "custo_tokens_usd": round(
                self.custo_total_tokens_usd, 6
            ),
            **roi,
        }


# ============================================================
# 4. RELATÓRIO EXECUTIVO
# ============================================================

def exibir_relatorio(coletor: ColetorKPI) -> None:
    """Exibe relatório completo em Rich."""

    # KPIs de volume e qualidade
    tabela_kpi = Table(
        title="KPIs do Período",
        header_style="bold magenta",
    )
    tabela_kpi.add_column("Indicador")
    tabela_kpi.add_column("Valor", justify="right")
    tabela_kpi.add_column("Meta")
    tabela_kpi.add_column("Status")

    def _linha(
        nome: str,
        valor: str,
        meta: str,
        ok: bool,
    ) -> None:
        tabela_kpi.add_row(
            nome, valor, meta,
            "[green]✓[/]" if ok else "[red]✗[/]"
        )

    _linha(
        "Total processados",
        str(coletor.total_processados), "—", True
    )
    _linha(
        "Taxa de sucesso",
        f"{coletor.taxa_sucesso:.1%}", "> 95%",
        coletor.taxa_sucesso >= 0.95,
    )
    _linha(
        "Taxa de revisão humana",
        f"{coletor.taxa_revisao:.1%}", "< 20%",
        coletor.taxa_revisao <= 0.20,
    )
    _linha(
        "Confiança média",
        f"{coletor.confianca_media:.1%}", "> 80%",
        coletor.confianca_media >= 0.80,
    )
    _linha(
        "Precisão média",
        f"{coletor.precisao_media:.1%}", "> 90%",
        coletor.precisao_media >= 0.90,
    )
    _linha(
        "Latência média",
        f"{coletor.duracao_media_ms:.0f} ms", "< 2000 ms",
        coletor.duracao_media_ms < 2000,
    )
    _linha(
        "Latência p95",
        f"{coletor.duracao_p95_ms:.0f} ms", "< 5000 ms",
        coletor.duracao_p95_ms < 5000,
    )
    console.print(tabela_kpi)

    # ROI
    roi = coletor.calcular_roi()
    console.print(
        Panel(
            f"Documentos automatizados: "
            f"[bold]{roi['docs_automatizados']:.0f}[/]\n"
            f"Horas economizadas:       "
            f"[bold]{roi['horas_economizadas']}h[/]\n"
            f"Custo evitado (R$):       "
            f"[bold green]R$ {roi['custo_evitado_brl']:,.2f}[/]\n"
            f"Custo APA (R$):           "
            f"R$ {roi['custo_apa_brl']:,.2f}\n"
            f"ROI:                      "
            f"[bold green]{roi['roi_percentual']:.1f}%[/]",
            title="Retorno sobre Investimento",
            style="green",
        )
    )

    # Alertas
    baixa_confianca = coletor.documentos_baixa_confianca()
    if baixa_confianca:
        console.print(
            f"\n  [yellow]⚠ {len(baixa_confianca)} documento(s) "
            f"com confiança < 60% — verificar:[/]"
        )
        for r in baixa_confianca[:3]:
            console.print(
                f"    • {r.id} | {r.tipo_documento} | "
                f"conf {r.confianca:.0%}"
            )


# ============================================================
# 5. DEMO
# ============================================================

def _gerar_registros_simulados() -> list[RegistroExecucao]:
    """Cria conjunto de execuções simuladas para o demo."""
    import random
    random.seed(42)

    tipos = ["boleto", "nfe", "contrato"]
    registros = []
    for i in range(50):
        tipo = tipos[i % 3]
        confianca = random.uniform(0.5, 1.0)
        campos = random.randint(3, 6)
        corretos = int(
            campos * random.uniform(0.8, 1.0)
        )
        registros.append(RegistroExecucao(
            id=f"DOC-{i+1:04d}",
            timestamp=datetime.now().isoformat(
                timespec="seconds"
            ),
            tipo_documento=tipo,
            duracao_ms=random.uniform(400, 3000),
            campos_extraidos=campos,
            campos_corretos=corretos,
            confianca=confianca,
            requer_revisao=confianca < 0.65,
            custo_tokens_usd=random.uniform(
                0.0001, 0.002
            ),
            erro=(
                "Timeout na API"
                if i % 15 == 0
                else None
            ),
        ))
    return registros


def demo_kpis_roi() -> None:
    console.print(
        Panel(
            "[bold]Módulo 36 — KPIs e Relatório de ROI[/]\n"
            "Métricas de qualidade, tempo e retorno "
            "financeiro do processo APA",
            style="bold blue",
        )
    )

    coletor = ColetorKPI(
        ParametrosROI(
            tempo_manual_min=8.0,
            custo_hora_analista=80.0,
            custo_infra_mensal=500.0,
        )
    )

    for reg in _gerar_registros_simulados():
        coletor.registrar(reg)

    console.rule("[yellow]Relatório Executivo")
    exibir_relatorio(coletor)

    console.rule("[yellow]Exportação (API / Prometheus)")
    import json
    console.print(
        json.dumps(coletor.exportar_dict(), indent=2)
    )


if __name__ == "__main__":
    demo_kpis_roi()
