"""
============================================================
MÓDULO 32.1 - AGENTE COMO MICROSSERVIÇO (FASTAPI)
============================================================
Neste módulo, aprendemos a expor o agente de APA como um
endpoint REST — tornando-o invocável por ERPs, RPAs,
dashboards e qualquer sistema corporativo via HTTP.

CONCEITO CHAVE:
Um agente que roda só como script local não é Automação
de Processo — é um experimento. Para ser APA de verdade,
o agente precisa ser invocável via API, retornar JSON
estruturado e ser monitorável como qualquer microsserviço.

POR QUE FASTAPI?
- Geração automática de documentação (Swagger / OpenAPI)
- Validação automática de entrada com Pydantic
- Async/await nativo (não bloqueia no I/O da LLM)
- Leve: inicia em < 1s, ideal para containers

ARQUITETURA:

  ┌────────────────────────────────────────────────────────┐
  │  ERP / RPA / Dashboard                                 │
  │       │  POST /processar-boleto                        │
  │       ▼                                                │
  │  [ FastAPI App ]                                       │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │  POST /processar-boleto  → AgenteBoletosAPI     │   │
  │  │  POST /processar-lote    → processa lista       │   │
  │  │  GET  /saude             → health check         │   │
  │  │  GET  /metricas          → KPIs do agente       │   │
  │  └─────────────────────────────────────────────────┘   │
  │       │                                                │
  │  [ Agente IA ] → Groq → resposta JSON                  │
  └────────────────────────────────────────────────────────┘

ENDPOINTS:
  POST /processar-boleto   → analisa um boleto
  POST /processar-lote     → analisa lista de boletos
  GET  /saude              → health check (uptime, versão)
  GET  /metricas           → requisições, erros, latência

SEGURANÇA (produção):
  - API Key no header X-API-Key
  - Rate limiting (slowapi)
  - HTTPS obrigatório (proxy reverso nginx/caddy)
  - Autenticação OAuth2 com Bearer token

Tópicos cobertos:
1. Modelos Pydantic de entrada e saída
2. Simulação do servidor ASGI sem instalar FastAPI
3. Handler de requisição com validação e resposta JSON
4. Health check e endpoint de métricas
5. Processamento em lote via endpoint
6. Como executar o servidor FastAPI real (comentado)
============================================================
"""

from __future__ import annotations

import time
import uuid
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. MODELOS DE ENTRADA E SAÍDA (PYDANTIC EM PRODUÇÃO)
# ============================================================
# Em produção use Pydantic v2:
#
#   from pydantic import BaseModel, Field
#
#   class RequisicaoBoleto(BaseModel):
#       texto: str = Field(..., min_length=10)
#       id_externo: str | None = None
#
#   class RespostaBoleto(BaseModel):
#       id_processamento: str
#       valor: float | None
#       vencimento: str | None
#       banco: str | None
#       vencido: bool
#       confianca: float
#       tempo_ms: float
#
# Aqui usamos dataclasses para simular sem instalar Pydantic.
# ============================================================

@dataclass
class RequisicaoBoleto:
    """Payload de entrada do endpoint POST /processar-boleto."""
    texto: str
    id_externo: Optional[str] = None


@dataclass
class RespostaBoleto:
    """Payload de saída do endpoint."""
    id_processamento: str
    valor: Optional[float]
    vencimento: Optional[str]
    banco: Optional[str]
    vencido: bool
    confianca: float
    tempo_ms: float
    erro: Optional[str] = None


@dataclass
class RespostaSaude:
    """Payload do endpoint GET /saude."""
    status: str          # "ok" | "degradado" | "offline"
    versao: str
    uptime_segundos: float
    timestamp: str


@dataclass
class RespostaMetricas:
    """Payload do endpoint GET /metricas."""
    total_requisicoes: int
    erros: int
    latencia_media_ms: float
    taxa_sucesso: float


# ============================================================
# 2. AGENTE DE BOLETOS (NÚCLEO DE NEGÓCIO)
# ============================================================
# Este é o mesmo agente dos módulos anteriores — aqui ele
# é embrulhado pela camada HTTP, mas a lógica não muda.
# ============================================================


def _analisar_boleto_core(texto: str) -> dict:
    """
    Lógica central de análise de boleto.
    Em produção chame o LLM (Groq) aqui.
    Simulamos por regex para o treinamento.
    """
    # Extrai valor
    m_val = re.search(
        r"R\$\s*([\d.,]+)", texto, re.IGNORECASE
    )
    valor = None
    if m_val:
        try:
            valor = float(
                m_val.group(1)
                .replace(".", "")
                .replace(",", ".")
            )
        except ValueError:
            pass

    # Extrai vencimento
    m_venc = re.search(r"\d{2}/\d{2}/\d{4}", texto)
    vencimento = m_venc.group() if m_venc else None

    # Detecta banco
    banco = None
    for nome in ["Bradesco", "Itaú", "Santander", "Caixa"]:
        if nome.lower() in texto.lower():
            banco = nome
            break

    # Verifica vencimento
    vencido = False
    if vencimento:
        try:
            venc_dt = datetime.strptime(
                vencimento, "%d/%m/%Y"
            )
            vencido = venc_dt < datetime.today()
        except ValueError:
            pass

    campos = sum([
        valor is not None,
        vencimento is not None,
        banco is not None,
    ])
    confianca = campos / 3

    return {
        "valor": valor,
        "vencimento": vencimento,
        "banco": banco,
        "vencido": vencido,
        "confianca": confianca,
    }


# ============================================================
# 3. SERVIDOR FASTAPI SIMULADO
# ============================================================
# Simula o comportamento de um servidor FastAPI sem
# instalação. Em produção, substitua por:
#
#   from fastapi import FastAPI, HTTPException, Header
#   from fastapi.responses import JSONResponse
#   import uvicorn
#
#   app = FastAPI(title="Agente APA", version="1.0.0")
#
#   @app.post("/processar-boleto")
#   async def processar_boleto(
#       req: RequisicaoBoleto,
#       x_api_key: str = Header(...),
#   ) -> RespostaBoleto:
#       if x_api_key != settings.api_key:
#           raise HTTPException(status_code=401)
#       ...
#
#   if __name__ == "__main__":
#       uvicorn.run(app, host="0.0.0.0", port=8000)
# ============================================================

@dataclass
class _Metricas:
    """Acumula métricas do servidor."""
    total: int = 0
    erros: int = 0
    latencias: list[float] = field(default_factory=list)

    def registrar(self, latencia_ms: float, erro: bool) -> None:
        """Registra uma nova requisição para cálculo de métricas.
        """
        self.total += 1
        if erro:
            self.erros += 1
        self.latencias.append(latencia_ms)

    def resumo(self) -> RespostaMetricas:
        """Calcula o resumo das métricas acumuladas."""
        media = (
            sum(self.latencias) / len(self.latencias)
            if self.latencias else 0.0
        )
        taxa = (
            (self.total - self.erros) / self.total
            if self.total else 0.0
        )
        return RespostaMetricas(
            total_requisicoes=self.total,
            erros=self.erros,
            latencia_media_ms=round(media, 1),
            taxa_sucesso=round(taxa, 3),
        )


class ServidorAgenteSimulado:
    """
    Simula as rotas FastAPI sem instalar o framework.
    Cada método representa um endpoint HTTP.
    """

    VERSAO = "1.0.0"

    def __init__(self) -> None:
        self._inicio = time.monotonic()
        self._metricas = _Metricas()

    # GET /saude
    def get_saude(self) -> RespostaSaude:
        """Retorna o status de saúde do serviço. Em produção, pode incluir"""
        return RespostaSaude(
            status="ok",
            versao=self.VERSAO,
            uptime_segundos=round(
                time.monotonic() - self._inicio, 2
            ),
            timestamp=datetime.now().isoformat(
                timespec="seconds"
            ),
        )

    # GET /metricas
    def get_metricas(self) -> RespostaMetricas:
        """Retorna as métricas acumuladas do serviço."""
        return self._metricas.resumo()

    # POST /processar-boleto
    def post_processar_boleto(
        self, req: RequisicaoBoleto
    ) -> RespostaBoleto:
        """
        Processa um boleto a partir do texto de entrada.
        A lógica de análise é a mesma do agente de boletos,
        mas aqui também registramos métricas de latência e erros.
        """
        t0 = time.monotonic()
        erro = None
        resultado: dict = {}
        try:
            if len(req.texto.strip()) < 10:
                raise ValueError(
                    "Texto muito curto para análise."
                )
            resultado = _analisar_boleto_core(req.texto)
        except ValueError as exc:
            erro = str(exc)
        finally:
            ms = (time.monotonic() - t0) * 1000
            self._metricas.registrar(ms, erro is not None)

        return RespostaBoleto(
            id_processamento=str(uuid.uuid4())[:8],
            valor=resultado.get("valor"),
            vencimento=resultado.get("vencimento"),
            banco=resultado.get("banco"),
            vencido=resultado.get("vencido", False),
            confianca=resultado.get("confianca", 0.0),
            tempo_ms=round(ms, 1),
            erro=erro,
        )

    # POST /processar-lote
    def post_processar_lote(
        self, textos: list[str]
    ) -> list[RespostaBoleto]:
        """Processa uma lista de boletos em lote. Em produção, pode ser
        otimizado para processamento paralelo ou assíncrono."""
        return [
            self.post_processar_boleto(
                RequisicaoBoleto(texto=t)
            )
            for t in textos
        ]


# ============================================================
# 4. DEMO
# ============================================================

def demo_microsservico_fastapi() -> None:
    """Demonstração do agente de boletos exposto como microsserviço
    via FastAPI. Simula requisições e mostra respostas e métricas."""
    console.print(
        Panel(
            "[bold]Módulo 32 — Agente como Microsserviço "
            "(FastAPI)[/]\n"
            "Expõe o agente APA como endpoint REST",
            style="bold blue",
        )
    )

    servidor = ServidorAgenteSimulado()

    # --- Health check ---
    console.rule("[yellow]GET /saude")
    saude = servidor.get_saude()
    console.print(
        f"  status:  [green]{saude.status}[/]\n"
        f"  versão:  {saude.versao}\n"
        f"  uptime:  {saude.uptime_segundos}s\n"
        f"  time:    {saude.timestamp}"
    )

    # --- Requisições individuais ---
    console.rule("[yellow]POST /processar-boleto")
    boletos = [
        "Boleto Bradesco R$ 1.500,00 venc. 10/04/2026",
        "Itaú R$ 890,00 vencimento 15/03/2026",
        "curto",   # deve gerar erro
        "Santander pagamento R$ 3.200,00 data 05/05/2026",
    ]

    tabela = Table(
        title="Respostas individuais",
        header_style="bold green",
    )
    tabela.add_column("ID")
    tabela.add_column("Banco")
    tabela.add_column("Valor (R$)", justify="right")
    tabela.add_column("Venc.")
    tabela.add_column("Vencido")
    tabela.add_column("Conf.")
    tabela.add_column("Ms", justify="right")
    tabela.add_column("Erro")

    for txt in boletos:
        r = servidor.post_processar_boleto(
            RequisicaoBoleto(texto=txt)
        )
        tabela.add_row(
            r.id_processamento,
            r.banco or "—",
            f"{r.valor:,.2f}" if r.valor else "—",
            r.vencimento or "—",
            "[red]Sim[/]" if r.vencido else "[green]Não[/]",
            f"{r.confianca:.0%}",
            str(r.tempo_ms),
            f"[red]{r.erro}[/]" if r.erro else "",
        )
    console.print(tabela)

    # --- Lote ---
    console.rule("[yellow]POST /processar-lote")
    lote = servidor.post_processar_lote(boletos[:3])
    console.print(
        f"  {len(lote)} boletos processados em lote."
    )

    # --- Métricas ---
    console.rule("[yellow]GET /metricas")
    m = servidor.get_metricas()
    console.print(
        f"  total:        {m.total_requisicoes}\n"
        f"  erros:        {m.erros}\n"
        f"  latência avg: {m.latencia_media_ms} ms\n"
        f"  taxa sucesso: {m.taxa_sucesso:.1%}"
    )

    # --- Como usar FastAPI real ---
    console.print(
        Panel(
            "# Instalação\n"
            "pip install fastapi uvicorn\n\n"
            "# main.py\n"
            "from fastapi import FastAPI\n"
            "app = FastAPI(title='Agente APA')\n\n"
            "@app.post('/processar-boleto')\n"
            "async def processar(req: RequisicaoBoleto):\n"
            "    return _analisar_boleto_core(req.texto)\n\n"
            "# Iniciar servidor\n"
            "uvicorn main:app --host 0.0.0.0 --port 8000\n\n"
            "# Documentação automática\n"
            "http://localhost:8000/docs",
            title="FastAPI Real — Como Usar",
            style="dim",
        )
    )


if __name__ == "__main__":
    demo_microsservico_fastapi()
