"""
============================================================
MÓDULO 31.1 - INTEGRAÇÃO COM SISTEMAS DE NOTIFICAÇÃO
============================================================
Neste módulo, aprendemos a fechar o loop de automação:
após processar um documento ou completar uma tarefa,
o agente notifica as pessoas certas no canal certo.

CONCEITO CHAVE:
Um agente que processa boletos e aprova pagamentos, mas
não notifica ninguém, não fecha o ciclo de trabalho.
Notificações conectam a automação com os usuários de negócio.

CANAIS SUPORTADOS:
  - E-mail        → SMTP ou SendGrid API
  - Microsoft Teams → Incoming Webhook (JSON card)
  - Slack           → Incoming Webhook (Block Kit)

ARQUITETURA:

  ┌─────────────────────────────────────────────────────────┐
  │  Agente                                                 │
  │     │                                                   │
  │     ▼                                                   │
  │  [ EventoBus ]  → publica evento                        │
  │     │                                                   │
  │  ┌──┴───────────────────────────────────────┐           │
  │  │  NotificadorEmail  NotificadorTeams      │           │
  │  │  NotificadorSlack  (extensível)          │           │
  │  └──────────────────────────────────────────┘           │
  │     │                                                   │
  │  Canal externo (SMTP / Webhook)                         │
  └─────────────────────────────────────────────────────────┘

PADRÃO: OBSERVER (pub-sub interno)
  Eventos publicados pelo agente são roteados para os
  notificadores registrados — sem acoplamento direto.

BOAS PRÁTICAS:
- Notificações assíncronas: não bloqueiam o fluxo principal
- Retry com backoff: webhooks podem falhar temporariamente
- Template de mensagem separado da lógica de envio
- Logs de entrega para auditoria

Tópicos cobertos:
1. Dataclass de eventos de negócio (aprovação, vencimento)
2. Notificador base (interface) + implementações mock
3. Formatação de card JSON para Teams (Adaptive Card)
4. Formatação de Block Kit para Slack
5. EventoBus com pub-sub e múltiplos canais
6. Agente que publica eventos ao completar tarefas
============================================================
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()


# ============================================================
# 1. MODELOS DE EVENTOS
# ============================================================
# Eventos são fatos imutáveis que descrevem o que aconteceu.
# Cada evento carrega os dados necessários para formatar
# a mensagem em qualquer canal.
# ============================================================

@dataclass
class EventoBase:
    """Dados comuns a todos os eventos de negócio."""
    id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(
            timespec="seconds"
        )
    )


@dataclass
class EventoBoletoVencido(EventoBase):
    """Boleto está vencido e aguarda providência."""
    sacado: str = ""
    valor: float = 0.0
    vencimento: str = ""
    dias_atraso: int = 0


@dataclass
class EventoPagamentoAprovado(EventoBase):
    """Pagamento foi aprovado e encaminhado ao financeiro."""
    descricao: str = ""
    valor: float = 0.0
    aprovador: str = ""
    solicitante: str = ""


@dataclass
class EventoFalhaAgente(EventoBase):
    """Erro crítico no pipeline do agente."""
    modulo: str = ""
    mensagem_erro: str = ""
    severidade: str = "ERROR"  # WARNING, ERROR, CRITICAL


# ============================================================
# 2. INTERFACE DE NOTIFICADOR
# ============================================================

@dataclass
class ResultadoNotificacao:
    """Resultado do envio de uma notificação."""
    canal: str
    evento_id: str
    sucesso: bool
    detalhe: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(
            timespec="seconds"
        )
    )


class NotificadorBase(ABC):
    """Interface que todo notificador deve implementar."""

    @property
    @abstractmethod
    def canal(self) -> str:
        """Nome do canal de notificação."""

    @abstractmethod
    def enviar(
        self, evento: EventoBase
    ) -> ResultadoNotificacao:
        """Envia notificação formatada para o canal."""


# ============================================================
# 3. NOTIFICADOR DE E-MAIL
# ============================================================
# Em produção use smtplib ou SendGrid:
#
#   import smtplib
#   from email.mime.text import MIMEText
#   msg = MIMEText(corpo, 'html')
#   msg['Subject'] = assunto
#   msg['From']    = remetente
#   msg['To']      = destinatario
#   with smtplib.SMTP(host, port) as s:
#       s.starttls()
#       s.login(user, password)
#       s.send_message(msg)
# ============================================================

class NotificadorEmail(NotificadorBase):
    """Envia notificações por e-mail (simulado)."""

    def __init__(
        self,
        destinatarios: list[str],
        remetente: str = "agente@empresa.com.br",
    ) -> None:
        self._destinatarios = destinatarios
        self._remetente = remetente

    @property
    def canal(self) -> str:
        return "email"

    def _formatar(self, evento: EventoBase) -> tuple[str, str]:
        """Retorna (assunto, corpo_html)."""
        if isinstance(evento, EventoBoletoVencido):
            assunto = (
                f"[AÇÃO NECESSÁRIA] Boleto vencido — "
                f"R$ {evento.valor:,.2f}"
            )
            corpo = (
                f"<b>Boleto vencido há "
                f"{evento.dias_atraso} dia(s)</b><br>"
                f"Sacado: {evento.sacado}<br>"
                f"Valor: R$ {evento.valor:,.2f}<br>"
                f"Vencimento: {evento.vencimento}"
            )
        elif isinstance(evento, EventoPagamentoAprovado):
            assunto = (
                f"Pagamento aprovado — "
                f"{evento.descricao}"
            )
            corpo = (
                f"<b>Pagamento aprovado</b><br>"
                f"Descrição: {evento.descricao}<br>"
                f"Valor: R$ {evento.valor:,.2f}<br>"
                f"Aprovador: {evento.aprovador}"
            )
        elif isinstance(evento, EventoFalhaAgente):
            assunto = (
                f"[{evento.severidade}] Falha no agente "
                f"— {evento.modulo}"
            )
            corpo = (
                f"<b>Erro no módulo {evento.modulo}</b>"
                f"<br>{evento.mensagem_erro}"
            )
        else:
            assunto = f"Evento: {type(evento).__name__}"
            corpo = str(evento)
        return assunto, corpo

    def enviar(
        self, evento: EventoBase
    ) -> ResultadoNotificacao:
        assunto, corpo = self._formatar(evento)
        # Simula envio
        console.print(
            f"  [EMAIL] Para: {', '.join(self._destinatarios)}\n"
            f"          Assunto: {assunto}\n"
            f"          Corpo: {corpo[:80]}…"
        )
        return ResultadoNotificacao(
            canal="email",
            evento_id=evento.id,
            sucesso=True,
            detalhe=f"Enviado para {len(self._destinatarios)} destinatário(s)",
        )


# ============================================================
# 4. NOTIFICADOR TEAMS (ADAPTIVE CARD)
# ============================================================
# Em produção:
#   import requests
#   requests.post(
#       webhook_url,
#       json=card,
#       headers={"Content-Type": "application/json"},
#   )
# ============================================================

class NotificadorTeams(NotificadorBase):
    """Envia notificações para canal Microsoft Teams."""

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    @property
    def canal(self) -> str:
        return "teams"

    def _montar_card(
        self, evento: EventoBase
    ) -> dict:
        """Monta Adaptive Card para Teams."""
        if isinstance(evento, EventoBoletoVencido):
            titulo = "⚠️ Boleto Vencido"
            cor = "attention"
            fatos = [
                {"title": "Sacado",    "value": evento.sacado},
                {"title": "Valor",
                 "value": f"R$ {evento.valor:,.2f}"},
                {"title": "Vencimento", "value": evento.vencimento},
                {"title": "Atraso",
                 "value": f"{evento.dias_atraso} dia(s)"},
            ]
        elif isinstance(evento, EventoPagamentoAprovado):
            titulo = "✅ Pagamento Aprovado"
            cor = "good"
            fatos = [
                {"title": "Descrição",
                 "value": evento.descricao},
                {"title": "Valor",
                 "value": f"R$ {evento.valor:,.2f}"},
                {"title": "Aprovador", "value": evento.aprovador},
            ]
        else:
            titulo = f"🔔 {type(evento).__name__}"
            cor = "default"
            fatos = [
                {"title": "ID", "value": evento.id}
            ]
        return {
            "type": "message",
            "attachments": [{
                "contentType":
                    "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema":
                        "http://adaptivecards.io/schemas/"
                        "adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": titulo,
                            "weight": "Bolder",
                            "size": "Medium",
                            "color": cor,
                        },
                        {
                            "type": "FactSet",
                            "facts": fatos,
                        },
                    ],
                },
            }],
        }

    def enviar(
        self, evento: EventoBase
    ) -> ResultadoNotificacao:
        card = self._montar_card(evento)
        card_json = json.dumps(card, indent=2)
        console.print(
            Syntax(
                card_json[:400] + "\n…",
                "json",
                theme="monokai",
            )
        )
        return ResultadoNotificacao(
            canal="teams",
            evento_id=evento.id,
            sucesso=True,
            detalhe=f"Card enviado para {self._webhook_url[:40]}…",
        )


# ============================================================
# 5. NOTIFICADOR SLACK (BLOCK KIT)
# ============================================================

class NotificadorSlack(NotificadorBase):
    """Envia notificações para canal Slack."""

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    @property
    def canal(self) -> str:
        return "slack"

    def _montar_blocks(
        self, evento: EventoBase
    ) -> dict:
        """Monta Block Kit para Slack."""
        if isinstance(evento, EventoBoletoVencido):
            texto = (
                f"*⚠️ Boleto vencido há "
                f"{evento.dias_atraso} dia(s)*\n"
                f"Sacado: {evento.sacado}\n"
                f"Valor: R$ {evento.valor:,.2f} | "
                f"Vencimento: {evento.vencimento}"
            )
        elif isinstance(evento, EventoPagamentoAprovado):
            texto = (
                f"*✅ Pagamento aprovado*\n"
                f"{evento.descricao} — "
                f"R$ {evento.valor:,.2f}\n"
                f"Aprovado por {evento.aprovador}"
            )
        else:
            texto = f"*{type(evento).__name__}* — ID: {evento.id}"
        return {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": texto,
                    },
                },
                {"type": "divider"},
                {
                    "type": "context",
                    "elements": [{
                        "type": "mrkdwn",
                        "text": f"🤖 Agente APA | {evento.timestamp}",
                    }],
                },
            ]
        }

    def enviar(
        self, evento: EventoBase
    ) -> ResultadoNotificacao:
        blocks = self._montar_blocks(evento)
        console.print(
            Syntax(
                json.dumps(blocks, indent=2)[:300] + "\n…",
                "json",
                theme="monokai",
            )
        )
        return ResultadoNotificacao(
            canal="slack",
            evento_id=evento.id,
            sucesso=True,
            detalhe="Block Kit enviado",
        )


# ============================================================
# 6. EVENTOBÚS (PUB-SUB)
# ============================================================

class EventoBus:
    """
    Barramento de eventos interno.
    Notificadores registrados recebem todos os eventos
    publicados — sem acoplamento com quem publica.
    """

    def __init__(self) -> None:
        self._notificadores: list[NotificadorBase] = []
        self._log: list[ResultadoNotificacao] = []

    def registrar(
        self, notificador: NotificadorBase
    ) -> None:
        """Registra um notificador para receber eventos publicados."""
        self._notificadores.append(notificador)
        console.print(
            f"  [dim]EventoBus: {notificador.canal} "
            f"registrado.[/]"
        )

    def publicar(self, evento: EventoBase) -> None:
        """Publica um evento para todos os notificadores registrados."""
        console.print(
            f"\n[bold magenta]📢 Evento:[/] "
            f"[cyan]{type(evento).__name__}[/] "
            f"(ID: {evento.id})"
        )
        for notif in self._notificadores:
            resultado = notif.enviar(evento)
            self._log.append(resultado)

    def exibir_log(self) -> None:
        """Exibe um resumo das notificações enviadas."""
        tabela = Table(
            title="Log de Notificações",
            header_style="bold magenta",
        )
        tabela.add_column("Canal")
        tabela.add_column("Evento ID")
        tabela.add_column("Status")
        tabela.add_column("Detalhe")
        tabela.add_column("Timestamp")
        for r in self._log:
            status = (
                "[green]OK[/]" if r.sucesso
                else "[red]FALHA[/]"
            )
            tabela.add_row(
                r.canal,
                r.evento_id,
                status,
                r.detalhe[:40],
                r.timestamp,
            )
        console.print(tabela)


# ============================================================
# 7. DEMO
# ============================================================

def demo_notificacoes() -> None:
    """Demonstração do módulo de notificações com eventos simulados."""
    console.print(
        Panel(
            "[bold]Módulo 31 — Integração com Sistemas "
            "de Notificação[/]\n"
            "E-mail, Teams e Slack a partir de eventos "
            "do agente",
            style="bold blue",
        )
    )

    # Configura o barramento
    bus = EventoBus()
    bus.registrar(
        NotificadorEmail(
            ["gestor@empresa.com.br", "financeiro@empresa.com.br"]
        )
    )
    bus.registrar(
        NotificadorTeams(
            "https://empresa.webhook.office.com/webhookb2/xxx"
        )
    )
    bus.registrar(
        NotificadorSlack(
            "https://hooks.slack.com/services/T00/B00/xxx"
        )
    )

    # Evento 1: boleto vencido
    console.rule("[yellow]Evento 1 — Boleto Vencido")
    bus.publicar(
        EventoBoletoVencido(
            id="EVT-001",
            sacado="Empresa Alpha Ltda",
            valor=1_500.0,
            vencimento="25/03/2026",
            dias_atraso=9,
        )
    )

    # Evento 2: pagamento aprovado
    console.rule("[yellow]Evento 2 — Pagamento Aprovado")
    bus.publicar(
        EventoPagamentoAprovado(
            id="EVT-002",
            descricao="Licença de Software",
            valor=3_500.0,
            aprovador="jose.gestor",
            solicitante="ana.silva",
        )
    )

    # Evento 3: falha crítica
    console.rule("[yellow]Evento 3 — Falha no Agente")
    bus.publicar(
        EventoFalhaAgente(
            id="EVT-003",
            modulo="modulo_06_agente_boletos",
            mensagem_erro="Timeout ao acessar API Groq após 3 tentativas",
            severidade="CRITICAL",
        )
    )

    # Resumo
    console.rule("[yellow]Resumo de Entregas")
    bus.exibir_log()


if __name__ == "__main__":
    demo_notificacoes()
