"""
============================================================
MÓDULO 16.1 - MONITORAMENTO DE AGENTES EM PRODUÇÃO
============================================================
Neste módulo, aprendemos a instrumentar o agente com métricas
de desempenho, verificações de SLA e alertas automáticos.

CONCEITO CHAVE:
Observabilidade (módulo 9) diz "o que aconteceu nesta execução".
Monitoramento diz "como o agente está se comportando AO LONGO
DO TEMPO": latência média, taxa de sucesso, volume de requisições,
violações de SLA e tendências de degradação.

POR QUE MONITORAMENTO É DIFERENTE DE LOGGING?
- Logging registra EVENTOS individuais (trace por trace)
- Monitoramento agrega SÉRIES TEMPORAIS (p50, p95, p99 de latência)
- Alertas disparam quando LIMITES são ultrapassados (ex: p95 > 3s)
- Dashboards mostram TENDÊNCIAS antes que vire problema crítico

ANALOGIA:
Logging é o prontuário de cada consulta (módulo 9).
Monitoramento é o relatório mensal do hospital:
"Esta semana tivemos 1.200 atendimentos, tempo médio 4min,
2 casos críticos que superaram o SLA de 10min."

MÉTRICAS ESSENCIAIS PARA AGENTES DE IA:

  ┌─────────────────────────────────────────────────────┐
  │  LATÊNCIA                                           │
  │  p50 / p95 / p99 por etapa do agente                │
  │  (entrada → guardrail → LLM → tools → saída)        │
  ├─────────────────────────────────────────────────────┤
  │  TAXA DE SUCESSO                                    │
  │  % de execuções sem erro / com guardrail bloqueado  │
  │  / com HITL ativado / com fallback acionado         │
  ├─────────────────────────────────────────────────────┤
  │  USO DE TOKENS                                      │
  │  tokens_prompt + tokens_completion por chamada      │
  │  (impacta custo diretamente)                        │
  ├─────────────────────────────────────────────────────┤
  │  SLA COMPLIANCE                                     │
  │  % de execuções dentro do tempo máximo acordado     │
  │  (ex: 95% das respostas em menos de 5 segundos)     │
  └─────────────────────────────────────────────────────┘

FERRAMENTAS DE MONITORAMENTO EM PRODUÇÃO:
- Langfuse          → dashboard dedicado para LLMs (gratuito/OSS)
- OpenTelemetry     → padrão aberto, integra com Datadog/Grafana
- Azure App Insights → integra com stack Azure (ver módulo AppInsights)
- CloudWatch (AWS)   → se já usa AWS
- Prometheus+Grafana → self-hosted, controle total

FLUXO DESTE MÓDULO:
  Execução ──▶ Coletor de Métricas ──▶ Agregador
                                           │
                               ┌───────────┴───────────┐
                               ▼                       ▼
                         Verificador SLA         Dashboard (Rich)
                               │
                         SLA violado? ──▶ Alerta (log/webhook/email)

Tópicos cobertos:
1. Coleta de métricas por execução (latência, tokens, resultado)
2. Agregação de percentis (p50, p95, p99)
3. Verificação automática de SLAs
4. Dashboard de monitoramento com Rich
5. Integração conceitual com Langfuse
============================================================
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Console do Rich para output formatado
console = Console()


# ============================================================
# 1. ESTRUTURA DE MÉTRICA POR EXECUÇÃO
# ============================================================
# Cada execução do agente gera uma MetricaExecucao.
# Esta estrutura é simples intencionalmente: cada campo
# corresponde a uma métrica que coletamos no agente.
#
# EM PRODUÇÃO: envie cada MetricaExecucao para um sistema
# de coleta (Langfuse, OpenTelemetry, etc.) ao final
# de cada execução do agente.
# ============================================================

@dataclass
class MetricaExecucao:
    """
    Representa as métricas de uma única execução do agente.

    CAMPOS OBRIGATÓRIOS para monitoramento útil:
    - trace_id: liga a métrica ao trace detalhado (módulo 9)
    - timestamp: quando aconteceu (para tendências temporais)
    - latencia_total_ms: experiência do usuário final
    - resultado: "sucesso", "guardrail", "erro", "hitl"
    - tokens_total: custo direto (prompt + completion)
    - latencia_llm_ms: isola a latência da LLM do resto
    """

    trace_id: str
    timestamp: datetime
    latencia_total_ms: float
    latencia_llm_ms: float
    resultado: str  # "sucesso" | "guardrail" | "erro" | "hitl"
    tokens_total: int
    modulo: str


# ============================================================
# 2. COLETOR E AGREGADOR DE MÉTRICAS
# ============================================================
# O ColetorMetricas acumula execuções e calcula estatísticas.
# Em produção, este coletor seria substituído por uma lib
# de telemetria, mas a lógica de negócio é a mesma.
#
# PERCENTIS:
# - p50 (mediana): metade das execuções é mais rápida que isso
# - p95: 95% das execuções é mais rápida que isso
#       → use para definir SLA (ex: SLA = p95 < 3000ms)
# - p99: quase o pior caso — mostra os outliers extremos
# ============================================================

class ColetorMetricas:
    """
    Coleta, agrega e verifica SLAs de execuções do agente.

    COMO USAR:
    1. Crie uma instância com os SLAs do contrato
    2. Registre cada execução com .registrar()
    3. Chame .dashboard() para ver o estado atual
    4. Chame .verificar_sla() para checar violações
    """

    def __init__(
        self,
        sla_latencia_ms: float = 5000.0,
        sla_taxa_sucesso_pct: float = 95.0,
    ) -> None:
        """
        Parâmetros:
        - sla_latencia_ms: latência máxima acordada (default 5s)
        - sla_taxa_sucesso_pct: % mínima de sucesso (default 95%)
        """
        self.sla_latencia_ms = sla_latencia_ms
        self.sla_taxa_sucesso_pct = sla_taxa_sucesso_pct
        self._execucoes: list[MetricaExecucao] = []

    def registrar(self, metrica: MetricaExecucao) -> None:
        """
        Registra uma execução no coletor.

        QUANDO CHAMAR:
        Ao final de CADA execução do agente, independente
        do resultado (sucesso, erro, guardrail bloqueado).
        Execuções bloqueadas por guardrail também são dados
        valiosos para monitoramento!
        """
        self._execucoes.append(metrica)

    def _calcular_percentil(
        self,
        valores: list[float],
        percentil: float,
    ) -> float:
        """
        Calcula o percentil de uma lista de valores.

        EXEMPLO:
        percentil([100, 200, 300, 400, 500], 95) → 480.0

        Usa interpolação linear (método padrão em sistemas
        de monitoramento como Prometheus e Datadog).
        """
        if not valores:
            return 0.0
        ordenado = sorted(valores)
        idx = (percentil / 100) * (len(ordenado) - 1)
        inferior = int(idx)
        fracao = idx - inferior
        if inferior + 1 >= len(ordenado):
            return ordenado[-1]
        return ordenado[inferior] + fracao * (
            ordenado[inferior + 1] - ordenado[inferior]
        )

    def estatisticas(self) -> dict[str, Any]:
        """
        Calcula estatísticas agregadas de todas as execuções.

        Retorna:
        - total: número de execuções registradas
        - taxa_sucesso_pct: % de execuções com resultado "sucesso"
        - latencia_{p50,p95,p99}_ms: percentis de latência
        - tokens_media: média de tokens por execução
        - por_resultado: contagem agrupada por tipo de resultado
        """
        if not self._execucoes:
            return {"total": 0}

        latencias = [e.latencia_total_ms for e in self._execucoes]
        sucessos = sum(
            1 for e in self._execucoes if e.resultado == "sucesso"
        )
        tokens = [e.tokens_total for e in self._execucoes]

        por_resultado: dict[str, int] = {}
        for exe in self._execucoes:
            por_resultado[exe.resultado] = (
                por_resultado.get(exe.resultado, 0) + 1
            )

        return {
            "total": len(self._execucoes),
            "taxa_sucesso_pct": (
                sucessos / len(self._execucoes) * 100
            ),
            "latencia_p50_ms": self._calcular_percentil(
                latencias, 50
            ),
            "latencia_p95_ms": self._calcular_percentil(
                latencias, 95
            ),
            "latencia_p99_ms": self._calcular_percentil(
                latencias, 99
            ),
            "latencia_media_ms": statistics.mean(latencias),
            "tokens_media": statistics.mean(tokens),
            "por_resultado": por_resultado,
        }

    def verificar_sla(self) -> list[str]:
        """
        Verifica violações de SLA definidas na instância.

        Retorna lista de violações (vazia se tudo OK).

        SLAs VERIFICADOS:
        1. p95 de latência ≤ sla_latencia_ms
        2. taxa de sucesso ≥ sla_taxa_sucesso_pct

        QUANDO USAR EM PRODUÇÃO:
        Execute esta verificação a cada N minutos via cron
        ou após cada janela de 100 execuções. Se a lista
        não estiver vazia, dispare alertas (email, Slack,
        PagerDuty, etc.).
        """
        stats = self.estatisticas()
        if stats.get("total", 0) == 0:
            return []

        violacoes: list[str] = []

        # Verifica SLA de latência (p95)
        p95 = stats["latencia_p95_ms"]
        if p95 > self.sla_latencia_ms:
            violacoes.append(
                f"⚠️  LATÊNCIA p95={p95:.0f}ms > "
                f"SLA={self.sla_latencia_ms:.0f}ms"
            )

        # Verifica SLA de taxa de sucesso
        taxa = stats["taxa_sucesso_pct"]
        if taxa < self.sla_taxa_sucesso_pct:
            violacoes.append(
                f"⚠️  SUCESSO {taxa:.1f}% < "
                f"SLA={self.sla_taxa_sucesso_pct:.0f}%"
            )

        return violacoes

    def dashboard(self) -> None:
        """
        Exibe dashboard de monitoramento no terminal com Rich.

        SEÇÕES DO DASHBOARD:
        1. Resumo geral (total, taxa de sucesso, tokens médios)
        2. Latências por percentil (p50 / p95 / p99)
        3. Distribuição de resultados
        4. Status de SLA (verde OK / vermelho VIOLADO)

        EM PRODUÇÃO: substitua este Rich por Grafana/Langfuse.
        Este dashboard é útil em desenvolvimento e para demos.
        """
        stats = self.estatisticas()
        if stats.get("total", 0) == 0:
            console.print("[yellow]Nenhuma execução registrada.[/yellow]")
            return

        # Tabela de latências
        t_latencia = Table(title="Latência por Percentil")
        t_latencia.add_column("Percentil", style="bold")
        t_latencia.add_column("Latência (ms)")
        t_latencia.add_column("SLA")

        sla_p95 = self.sla_latencia_ms
        for label, valor in [
            ("p50 (mediana)", stats["latencia_p50_ms"]),
            ("p95 (SLA ref.)", stats["latencia_p95_ms"]),
            ("p99 (pior caso)", stats["latencia_p99_ms"]),
        ]:
            status = (
                "[green]✓ OK[/green]"
                if valor <= sla_p95
                else "[red]✗ VIOLADO[/red]"
            )
            t_latencia.add_row(label, f"{valor:.0f} ms", status)
        console.print(t_latencia)

        # Tabela de resultados
        t_resultados = Table(title="Distribuição de Resultados")
        t_resultados.add_column("Resultado", style="bold")
        t_resultados.add_column("Contagem")
        t_resultados.add_column("%")
        total = stats["total"]
        for resultado, contagem in sorted(
            stats["por_resultado"].items()
        ):
            cor = "green" if resultado == "sucesso" else "yellow"
            t_resultados.add_row(
                f"[{cor}]{resultado}[/{cor}]",
                str(contagem),
                f"{contagem / total * 100:.1f}%",
            )
        console.print(t_resultados)

        # Status de SLA
        violacoes = self.verificar_sla()
        if violacoes:
            for v in violacoes:
                console.print(f"[bold red]{v}[/bold red]")
            console.print(
                "\n[bold red]🚨 AÇÃO NECESSÁRIA: SLA violado!"
                "[/bold red]\n"
                "  → Notifique o time e investigue as execuções "
                "com maior latência.\n"
                "  → Em produção: dispare webhook para Slack/"
                "PagerDuty."
            )
        else:
            console.print(
                "\n[bold green]✅ Todos os SLAs dentro do limite"
                "[/bold green]"
            )


# ============================================================
# 3. INTEGRAÇÃO CONCEITUAL COM LANGFUSE
# ============================================================
# Langfuse é a ferramenta mais popular para monitoramento de LLMs.
# É open-source, tem dashboard web e SDK para Python/TS.
#
# COMO INTEGRAR (em produção):
#   pip install langfuse
#
#   from langfuse import Langfuse
#   lf = Langfuse(
#       public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
#       secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
#       host="https://cloud.langfuse.com",
#   )
#   trace = lf.trace(name="processar_boleto", id=trace_id)
#   span = trace.span(name="guardrail_entrada")
#   span.end(output={"resultado": "ok"})
#   trace.update(output={"status": "sucesso"}, level="DEFAULT")
#
# O DASHBOARD DO LANGFUSE OFERECE:
# - Latência p50/p95/p99 por trace name
# - Custo em $ por token (integra com preços dos modelos)
# - Feedback do usuário (👍 / 👎) por execução
# - Rastreabilidade span-a-span (como o trace_utils do módulo 9)
# ============================================================

def mostrar_integracao_langfuse() -> None:
    """
    Exibe o código de integração com Langfuse para referência.

    Não executa código real — é uma referência educacional.
    """
    console.print(Panel(
        "[bold]Integração Langfuse (produção):[/bold]\n\n"
        "[dim]pip install langfuse[/dim]\n\n"
        "[cyan]from langfuse import Langfuse\n"
        "lf = Langfuse(\n"
        '    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),\n'
        '    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),\n'
        ")\n"
        "trace = lf.trace(name='processar_boleto', id=trace_id)\n"
        "span = trace.span(name='guardrail_entrada')\n"
        "span.end(output={'resultado': 'ok'})[/cyan]\n\n"
        "[bold]Alternativas:[/bold] OpenTelemetry, Datadog APM, "
        "Azure Application Insights (ver skill appinsights)",
        title="🔗 Langfuse — Monitoramento em Produção",
        border_style="blue",
    ))


# ============================================================
# DEMO COMPLETA — Simulação de monitoramento em produção
# ============================================================

def _gerar_execucoes_simuladas(n: int = 50) -> list[MetricaExecucao]:
    """
    Gera N execuções simuladas com distribuição realista.

    Distribui resultados e latências para simular um agente
    em produção processando boletos ao longo do dia.
    """
    resultados_possiveis = [
        ("sucesso", 78),    # 78% das execuções terminam bem
        ("guardrail", 12),  # 12% bloqueadas por guardrail
        ("hitl", 7),        # 7% vão para aprovação humana
        ("erro", 3),        # 3% com erro real
    ]

    exec_list: list[MetricaExecucao] = []
    for i in range(n):
        # Escolhe resultado com peso
        r = random.random() * 100
        acumulado = 0.0
        resultado = "sucesso"
        for nome, peso in resultados_possiveis:
            acumulado += peso
            if r < acumulado:
                resultado = nome
                break

        # Latência depende do resultado (erros costumam ser lentos)
        base_ms = {
            "sucesso": random.gauss(1800, 400),
            "guardrail": random.gauss(200, 50),
            "hitl": random.gauss(2500, 500),
            "erro": random.gauss(4800, 800),
        }[resultado]
        latencia = max(100.0, base_ms)

        exec_list.append(MetricaExecucao(
            trace_id=f"tr_{i:04d}",
            timestamp=datetime.now(),
            latencia_total_ms=latencia,
            latencia_llm_ms=latencia * 0.7,
            resultado=resultado,
            tokens_total=random.randint(800, 2000),
            modulo="agente_boletos",
        ))
    return exec_list


def demo_monitoramento() -> None:
    """
    Demonstra o sistema de monitoramento com execuções simuladas.

    ETAPAS:
    1. Simula 50 execuções do agente (distribuição realista)
    2. Registra cada execução no ColetorMetricas
    3. Exibe dashboard com latências, resultados e SLA
    4. Verifica e exibe violações de SLA (se houver)
    5. Mostra integração conceitual com Langfuse

    OBSERVE NO OUTPUT:
    - Os percentis p50/p95/p99 mostram a distribuição de latência
    - Execuções de "erro" elevam o p99 (outliers extremos)
    - A taxa de sucesso ~78% está abaixo do SLA padrão (95%)
      → isso dispara o alerta de SLA no dashboard

    EXERCÍCIO SUGERIDO:
    1. Ajuste os pesos dos resultados para 95% de sucesso
    2. Adicione uma métrica de "custo_usd" e exiba no dashboard
    3. Integre o ColetorMetricas ao agente_boletos.py do módulo 6
    """
    console.print(Panel.fit(
        "[bold]Monitoramento em Produção — Dashboard ao vivo[/bold]\n"
        "Simulando 50 execuções do agente de boletos",
        title="📊 Módulo 16 — Monitoramento",
        border_style="blue",
    ))

    # Configura SLAs (conforme definição com o negócio)
    coletor = ColetorMetricas(
        sla_latencia_ms=5000.0,    # máximo 5s (p95)
        sla_taxa_sucesso_pct=95.0,  # mínimo 95% de sucesso
    )

    # Simula execuções e registra
    console.print("\n[dim]Simulando execuções...[/dim]")
    execucoes = _gerar_execucoes_simuladas(50)
    for exec_ in execucoes:
        coletor.registrar(exec_)

    console.print(
        f"[green]{len(execucoes)} execuções registradas[/green]\n"
    )

    # Dashboard
    coletor.dashboard()

    # Referência Langfuse
    console.print()
    mostrar_integracao_langfuse()


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    demo_monitoramento()
