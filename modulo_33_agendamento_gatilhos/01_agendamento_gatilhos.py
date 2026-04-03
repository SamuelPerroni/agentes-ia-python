"""
============================================================
MÓDULO 33.1 - AGENDAMENTO E GATILHOS DE PROCESSO (APA)
============================================================
Neste módulo, aprendemos como processos de APA são
*iniciados*: agendamento por cron, gatilhos por evento
(nova nota fiscal chegou) e webhooks de sistemas externos.

CONCEITO CHAVE:
Automação de Processo sem gatilho não é automação —
é um script que alguém precisa lembrar de rodar.
Em APA real, o processo inicia sozinho quando a condição
é satisfeita: horário, evento ou mensagem externa.

TIPOS DE GATILHO:

  ┌──────────────────────────────────────────────────────────┐
  │  CRON (tempo)                                            │
  │  └── "todo dia às 8h": extrai boletos do portal         │
  │  └── "toda sexta às 17h": fecha conciliação da semana   │
  │                                                          │
  │  EVENTO (fila / mensagem)                               │
  │  └── "nova NF-e no S3": processa imediatamente          │
  │  └── "boleto criado no ERP": agenda pagamento           │
  │                                                          │
  │  WEBHOOK (HTTP push)                                     │
  │  └── ERP chama POST /webhook/boleto quando cria         │
  │  └── Banco chama POST /webhook/pagamento quando baixa   │
  └──────────────────────────────────────────────────────────┘

PADRÃO: DISPATCHER DE GATILHOS

       Gatilho recebido
            │
            ▼
  [ GatilhoDispatcher ]
  ├── identifica tipo
  ├── valida payload
  └── enfileira tarefa → [ Agente ]

EM PRODUÇÃO:
  CRON     → APScheduler / Celery Beat / cron do SO
  EVENTO   → SQS / Service Bus / RabbitMQ consumer
  WEBHOOK  → FastAPI endpoint (já visto no módulo 32)

Tópicos cobertos:
1. Enum de tipos de gatilho com configuração
2. Simulação de scheduler de cron jobs
3. Simulação de consumer de fila de eventos
4. Handler de webhook com validação de assinatura HMAC
5. Dispatcher que roteia cada gatilho para o handler
6. Histórico de execuções com status e horário
============================================================
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. TIPOS DE GATILHO
# ============================================================

class TipoGatilho(str, Enum):
    """Enum para tipos de gatilho de processo."""
    CRON = "cron"
    EVENTO = "evento"
    WEBHOOK = "webhook"
    MANUAL = "manual"


@dataclass
class ConfigGatilho:
    """Configuração de um gatilho de processo."""
    nome: str
    tipo: TipoGatilho
    # Para CRON: expressão cron ("0 8 * * *")
    # Para EVENTO: nome da fila ou tópico
    # Para WEBHOOK: path do endpoint
    parametro: str
    ativo: bool = True


# ============================================================
# 2. EXECUÇÃO DE PROCESSO
# ============================================================

@dataclass
class ExecucaoProcesso:
    """Registro de uma execução disparada por gatilho."""
    id: str
    gatilho: str
    tipo: TipoGatilho
    inicio: str
    fim: Optional[str] = None
    status: str = "executando"   # executando|ok|erro
    resultado: str = ""
    erro: Optional[str] = None


# ============================================================
# 3. HANDLERS DE PROCESSO
# ============================================================
# Cada handler recebe o payload do gatilho e executa
# a lógica de negócio. Em APA, os handlers chamam
# o agente de IA para processar o documento ou tarefa.
# ============================================================

def handler_extracao_boletos(payload: dict) -> str:
    """
    Handler cron: extrai boletos do portal todo dia às 8h.
    Em produção: chama AgenteRPA (módulo 26).
    """
    data = payload.get("data", datetime.today().strftime(
        "%Y-%m-%d"
    ))
    # Simula extração
    qtd = 5
    console.print(
        f"  [handler] Extração de boletos em {data}: "
        f"{qtd} boletos encontrados."
    )
    return f"{qtd} boletos extraídos para {data}"


def handler_nova_nfe(payload: dict) -> str:
    """
    Handler de evento: processa NF-e assim que chega na fila.
    Em produção: chama AgenteDocumentos (módulo 28).
    """
    arquivo = payload.get("arquivo", "nfe_desconhecida.pdf")
    console.print(
        f"  [handler] Nova NF-e detectada: {arquivo}"
    )
    return f"NF-e {arquivo} processada com sucesso"


def handler_webhook_pagamento(payload: dict) -> str:
    """
    Handler de webhook: banco notifica confirmação de pagamento.
    Em produção: atualiza ERP via API.
    """
    boleto_id = payload.get("boleto_id", "?")
    valor = payload.get("valor", 0.0)
    console.print(
        f"  [handler] Pagamento confirmado: "
        f"boleto {boleto_id} R$ {valor:,.2f}"
    )
    return f"Boleto {boleto_id} baixado no sistema"


# ============================================================
# 4. VALIDADOR DE ASSINATURA HMAC (WEBHOOK)
# ============================================================
# Webhooks de sistemas externos (bancos, ERPs) assinam o
# payload com HMAC-SHA256 para garantir autenticidade.
# Sem verificação, qualquer um pode chamar o endpoint.
# ============================================================

def verificar_assinatura_hmac(
    payload_bytes: bytes,
    assinatura_recebida: str,
    segredo: str,
) -> bool:
    """
    Verifica assinatura HMAC-SHA256 de um webhook.
    Usa comparação de tempo constante para evitar
    timing attack.
    """
    esperada = hmac.new(
        segredo.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(
        f"sha256={esperada}", assinatura_recebida
    )


# ============================================================
# 5. DISPATCHER DE GATILHOS
# ============================================================

class GatilhoDispatcher:
    """
    Recebe qualquer tipo de gatilho, valida e executa
    o handler correspondente.
    """

    def __init__(self) -> None:
        # Mapeia nome do gatilho → função handler
        self._handlers: dict[str, Callable[[dict], str]] = {}
        self._historico: list[ExecucaoProcesso] = []
        self._contador = 0

    def registrar(
        self,
        nome: str,
        handler: Callable[[dict], str],
    ) -> None:
        """Registra um handler para um nome de gatilho específico."""
        self._handlers[nome] = handler
        console.print(
            f"  [dim]Dispatcher: '{nome}' registrado.[/]"
        )

    def disparar(
        self,
        gatilho: ConfigGatilho,
        payload: dict,
    ) -> ExecucaoProcesso:
        """Executa o handler do gatilho e registra resultado."""
        self._contador += 1
        exec_id = f"EXEC-{self._contador:04d}"
        exec_ = ExecucaoProcesso(
            id=exec_id,
            gatilho=gatilho.nome,
            tipo=gatilho.tipo,
            inicio=datetime.now().isoformat(
                timespec="seconds"
            ),
        )
        if not gatilho.ativo:
            exec_.status = "ignorado"
            exec_.resultado = "Gatilho desativado"
            self._historico.append(exec_)
            return exec_

        handler = self._handlers.get(gatilho.nome)
        if handler is None:
            exec_.status = "erro"
            exec_.erro = f"Handler '{gatilho.nome}' não encontrado"
            self._historico.append(exec_)
            return exec_

        try:
            exec_.resultado = handler(payload)
            exec_.status = "ok"
        except Exception as exc:  # noqa: BLE001
            exec_.status = "erro"
            exec_.erro = str(exc)
        finally:
            exec_.fim = datetime.now().isoformat(
                timespec="seconds"
            )

        self._historico.append(exec_)
        return exec_

    def exibir_historico(self) -> None:
        """Exibe o histórico de execuções em formato de tabela."""
        tabela = Table(
            title="Histórico de Execuções",
            header_style="bold magenta",
        )
        tabela.add_column("ID")
        tabela.add_column("Gatilho")
        tabela.add_column("Tipo")
        tabela.add_column("Status")
        tabela.add_column("Resultado")
        for e in self._historico:
            status_fmt = {
                "ok": "[green]ok[/]",
                "erro": "[red]erro[/]",
                "ignorado": "[yellow]ignorado[/]",
                "executando": "[cyan]...[/]",
            }.get(e.status, e.status)
            tabela.add_row(
                e.id,
                e.gatilho,
                e.tipo.value,
                status_fmt,
                (e.resultado or e.erro or "")[:50],
            )
        console.print(tabela)


# ============================================================
# 6. SIMULADOR DE SCHEDULER (CRON)
# ============================================================

class SchedulerSimulado:
    """
    Simula o disparo de jobs agendados (cron).
    Em produção use APScheduler:
      from apscheduler.schedulers.blocking import (
          BlockingScheduler
      )
      scheduler = BlockingScheduler()
      scheduler.add_job(
          handler_extracao_boletos,
          'cron', hour=8, args=[{}]
      )
      scheduler.start()
    """

    def __init__(
        self, dispatcher: GatilhoDispatcher
    ) -> None:
        self._dispatcher = dispatcher
        self._jobs: list[tuple[ConfigGatilho, dict]] = []

    def agendar(
        self, gatilho: ConfigGatilho, payload: dict
    ) -> None:
        """Agenda um job para execução futura."""
        self._jobs.append((gatilho, payload))

    def executar_todos(self) -> None:
        """Simula disparo imediato de todos os jobs."""
        console.print(
            f"  [cyan]Scheduler: disparando "
            f"{len(self._jobs)} job(s)...[/]"
        )
        for gatilho, payload in self._jobs:
            self._dispatcher.disparar(gatilho, payload)


# ============================================================
# 7. DEMO
# ============================================================

def demo_agendamento_gatilhos() -> None:
    """Demonstração de agendamento e gatilhos de processo."""
    console.print(
        Panel(
            "[bold]Módulo 33 — Agendamento e Gatilhos "
            "de Processo[/]\n"
            "Cron, eventos e webhooks como ponto de "
            "entrada da APA",
            style="bold blue",
        )
    )

    dispatcher = GatilhoDispatcher()
    dispatcher.registrar(
        "extracao_boletos_diaria", handler_extracao_boletos
    )
    dispatcher.registrar("nova_nfe_fila", handler_nova_nfe)
    dispatcher.registrar(
        "webhook_pagamento_banco",
        handler_webhook_pagamento,
    )

    scheduler = SchedulerSimulado(dispatcher)

    # --- Cron: diário às 8h ---
    console.rule("[yellow]Gatilho CRON — diário às 8h")
    g_cron = ConfigGatilho(
        nome="extracao_boletos_diaria",
        tipo=TipoGatilho.CRON,
        parametro="0 8 * * *",
    )
    scheduler.agendar(g_cron, {"data": "2026-04-03"})
    scheduler.executar_todos()

    # --- Evento: nova NF-e na fila ---
    console.rule("[yellow]Gatilho EVENTO — nova NF-e")
    g_evento = ConfigGatilho(
        nome="nova_nfe_fila",
        tipo=TipoGatilho.EVENTO,
        parametro="fila/nfe-entrada",
    )
    dispatcher.disparar(
        g_evento, {"arquivo": "nfe_001234.pdf"}
    )

    # --- Webhook: banco confirma pagamento ---
    console.rule("[yellow]Gatilho WEBHOOK — pagamento")

    # Valida assinatura antes de processar
    payload_dict = {"boleto_id": "BOL-042", "valor": 1500.0}
    payload_bytes = json.dumps(payload_dict).encode()
    segredo = "segredo_compartilhado_banco"
    assinatura = (
        "sha256="
        + hmac.new(
            segredo.encode(),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()
    )
    if verificar_assinatura_hmac(
        payload_bytes, assinatura, segredo
    ):
        console.print(
            "  [green]Assinatura HMAC válida.[/]"
        )
        g_webhook = ConfigGatilho(
            nome="webhook_pagamento_banco",
            tipo=TipoGatilho.WEBHOOK,
            parametro="/webhook/pagamento",
        )
        dispatcher.disparar(g_webhook, payload_dict)
    else:
        console.print(
            "  [red]Assinatura HMAC inválida — "
            "requisição rejeitada.[/]"
        )

    # --- Gatilho desativado ---
    console.rule(
        "[yellow]Gatilho MANUAL desativado"
    )
    g_inativo = ConfigGatilho(
        nome="extracao_boletos_diaria",
        tipo=TipoGatilho.MANUAL,
        parametro="manual",
        ativo=False,
    )
    dispatcher.disparar(g_inativo, {})

    # --- Histórico ---
    console.rule("[yellow]Histórico de Execuções")
    dispatcher.exibir_historico()


if __name__ == "__main__":
    demo_agendamento_gatilhos()
