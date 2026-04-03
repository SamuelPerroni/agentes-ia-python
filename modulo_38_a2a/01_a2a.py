"""
============================================================
MÓDULO 38.1 - PROTOCOLO A2A (AGENT-TO-AGENT)
============================================================
O protocolo A2A (Agent-to-Agent), proposto pelo Google em
2025 e adotado por SAP, Salesforce e outros, padroniza a
comunicação entre agentes de IA independentes.

PROBLEMA QUE RESOLVE:
Antes do A2A, cada integração entre agentes era ad-hoc:
chamadas HTTP diretas, formatos incompatíveis, sem
descoberta automática de capacidades.

ARQUITETURA A2A:

  ┌─────────────────────────────────────────────────────┐
  │  DISCOVERY — AgentCard                              │
  │  Cada agente publica um "cartão" que descreve:      │
  │  • Capacidades (skills)                             │
  │  • Endpoint de comunicação                          │
  │  • Versão e autenticação aceita                     │
  └──────────────┬──────────────────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────────────────┐
  │  TASK — unidade de trabalho                         │
  │  TaskRequest  → enviada pelo agente cliente         │
  │  TaskResponse → retornada pelo agente servidor      │
  │  TaskStatus   → SUBMITTED, WORKING, DONE, FAILED    │
  └──────────────┬──────────────────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────────────────┐
  │  ORQUESTRADOR                                       │
  │  Agente mestre que:                                 │
  │  • Descobre agentes via AgentCard                   │
  │  • Delega tarefas ao especialista correto           │
  │  • Consolida resultados                             │
  └─────────────────────────────────────────────────────┘

FLUXO REAL (produção):
  1. Orquestrador lê AgentCards de um registro central
  2. Seleciona agente com skill adequada
  3. Envia Task via HTTP (POST /tasks)
  4. Polling ou streaming do status
  5. Recebe TaskResponse e continua o pipeline

Tópicos cobertos:
1. AgentCard — descoberta de capacidades
2. TaskRequest / TaskResponse / TaskStatus
3. AgentServer — agente servidor de tarefas
4. AgentClient — agente cliente/orquestrador
5. Roteamento de tarefas por skill match
6. Demo com 3 agentes especializados
============================================================
"""

from __future__ import annotations

import uuid
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. ENUMERAÇÕES
# ============================================================

class TaskStatus(Enum):
    """Status de uma tarefa A2A."""
    SUBMITTED = "submitted"
    WORKING = "working"
    DONE = "done"
    FAILED = "failed"


# ============================================================
# 2. AGENT CARD — DESCOBERTA
# ============================================================

@dataclass
class AgentSkill:
    """Uma capacidade específica do agente."""
    id: str
    name: str
    description: str
    input_schema: dict   # tipos de entrada aceitos
    output_schema: dict  # tipos de saída produzidos


@dataclass
class AgentCard:
    """
    Cartão de identidade do agente.
    Publicado em /.well-known/agent.json em produção.
    """
    agent_id: str
    name: str
    description: str
    version: str
    endpoint: str        # URL base do agente
    skills: list[AgentSkill]
    auth_schemes: list[str] = field(
        default_factory=lambda: ["bearer"]
    )

    def supports_skill(self, skill_id: str) -> bool:
        """Verifica se o agente suporta a skill solicitada."""
        return any(s.id == skill_id for s in self.skills)


# ============================================================
# 3. TASK — UNIDADE DE TRABALHO
# ============================================================

@dataclass
class TaskRequest:
    """Tarefa enviada do cliente para o agente servidor."""
    task_id: str
    skill_id: str
    input_data: dict
    sender_agent_id: str
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(
            timespec="seconds"
        )
    )
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskResponse:
    """Resultado retornado pelo agente servidor."""
    task_id: str
    status: TaskStatus
    output_data: dict
    agent_id: str
    completed_at: str = field(
        default_factory=lambda: datetime.now().isoformat(
            timespec="seconds"
        )
    )
    error: Optional[str] = None


# ============================================================
# 4. AGENTE SERVIDOR
# ============================================================

class AgentServer:
    """
    Agente que expõe skills e executa tarefas recebidas.
    Em produção: FastAPI + POST /tasks.
    """

    def __init__(self, card: AgentCard) -> None:
        self.card = card
        self._handlers: dict[
            str, Callable[[dict], dict]
        ] = {}

    def register_handler(
        self,
        skill_id: str,
        handler: Callable[[dict], dict],
    ) -> None:
        """Registra uma função handler para uma skill específica."""
        self._handlers[skill_id] = handler

    def execute(self, request: TaskRequest) -> TaskResponse:
        """Executa a tarefa e retorna o resultado."""
        handler = self._handlers.get(request.skill_id)
        if handler is None:
            return TaskResponse(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                output_data={},
                agent_id=self.card.agent_id,
                error=(
                    f"Skill '{request.skill_id}' "
                    f"não suportada"
                ),
            )
        try:
            output = handler(request.input_data)
            return TaskResponse(
                task_id=request.task_id,
                status=TaskStatus.DONE,
                output_data=output,
                agent_id=self.card.agent_id,
            )
        except Exception as exc:  # pylint: disable=broad-except
            return TaskResponse(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                output_data={},
                agent_id=self.card.agent_id,
                error=str(exc),
            )


# ============================================================
# 5. REGISTRO DE AGENTES (DISCOVERY)
# ============================================================

class AgentRegistry:
    """
    Registro central de AgentCards.
    Em produção: Apigee Hub, serviço REST ou arquivo YAML.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentServer] = {}

    def register(self, server: AgentServer) -> None:
        """Registra um agente servidor no registro."""
        self._agents[server.card.agent_id] = server
        console.print(
            f"  [dim]Registry: agente "
            f"'{server.card.name}' registrado.[/]"
        )

    def find_by_skill(
        self, skill_id: str
    ) -> Optional[AgentServer]:
        """Encontra um agente que suporte a skill solicitada."""
        for server in self._agents.values():
            if server.card.supports_skill(skill_id):
                return server
        return None

    def list_cards(self) -> list[AgentCard]:
        """Lista os AgentCards de todos os agentes registrados."""
        return [s.card for s in self._agents.values()]


# ============================================================
# 6. AGENTE ORQUESTRADOR (CLIENTE A2A)
# ============================================================

class OrchestratorAgent:
    """
    Agente mestre que usa o registro para delegar tarefas
    aos especialistas certos via protocolo A2A.
    """

    def __init__(
        self,
        agent_id: str,
        registry: AgentRegistry,
    ) -> None:
        self.agent_id = agent_id
        self._registry = registry

    def delegate(
        self,
        skill_id: str,
        input_data: dict,
        metadata: Optional[dict] = None,
    ) -> TaskResponse:
        """Encontra o agente certo e delega a tarefa."""
        server = self._registry.find_by_skill(skill_id)
        if server is None:
            task_id = str(uuid.uuid4())[:8]
            return TaskResponse(
                task_id=task_id,
                status=TaskStatus.FAILED,
                output_data={},
                agent_id="orchestrator",
                error=(
                    f"Nenhum agente encontrado para "
                    f"skill '{skill_id}'"
                ),
            )
        request = TaskRequest(
            task_id=str(uuid.uuid4())[:8],
            skill_id=skill_id,
            input_data=input_data,
            sender_agent_id=self.agent_id,
            metadata=metadata or {},
        )
        console.print(
            f"  [cyan]A2A:[/] {self.agent_id} → "
            f"{server.card.name} (skill: {skill_id})"
        )
        return server.execute(request)


# ============================================================
# 7. AGENTES ESPECIALIZADOS (HANDLERS)
# ============================================================

def _handler_ler_boleto(data: dict) -> dict:
    """Agente especializado em leitura de boletos."""
    texto = data.get("texto", "")
    valor = None
    m = re.search(r"R\$\s*([\d.,]+)", texto)
    if m:
        try:
            valor = float(
                m.group(1).replace(".", "").replace(",", ".")
            )
        except ValueError:
            pass
    return {
        "valor": valor,
        "banco": "Banco Simulado",
        "vencimento": "2026-05-10",
        "confianca": 0.92,
    }


def _handler_validar_cnpj(data: dict) -> dict:
    """Agente especializado em validação de CNPJ."""
    cnpj = data.get("cnpj", "")
    numeros = re.sub(r"\D", "", cnpj)
    valido = len(numeros) == 14 and len(set(numeros)) > 1
    return {
        "cnpj": cnpj,
        "valido": valido,
        "situacao": "ATIVA" if valido else "INVÁLIDA",
    }


def _handler_classificar_doc(data: dict) -> dict:
    """Agente especializado em classificação de documentos."""
    texto = data.get("texto", "").lower()
    if "boleto" in texto or "vencimento" in texto:
        tipo = "boleto"
    elif "nota fiscal" in texto or "nfe" in texto:
        tipo = "nfe"
    elif "contrato" in texto:
        tipo = "contrato"
    else:
        tipo = "desconhecido"
    return {"tipo": tipo, "confianca": 0.88}


# ============================================================
# 8. DEMO
# ============================================================

def demo_a2a() -> None:
    """Demonstração do protocolo A2A com 3 agentes especializados"""
    console.print(
        Panel(
            "[bold]Módulo 38 — Protocolo A2A "
            "(Agent-to-Agent)[/]\n"
            "Descoberta, delegação e comunicação "
            "padronizada entre agentes",
            style="bold blue",
        )
    )

    # --- Cria agentes especializados ---
    agente_boleto = AgentServer(AgentCard(
        agent_id="agent-boleto-v1",
        name="AgenteLeitorBoletos",
        description="Extrai dados de boletos bancários",
        version="1.0.0",
        endpoint="https://agentes.empresa.com/boletos",
        skills=[AgentSkill(
            id="ler_boleto",
            name="Leitura de Boleto",
            description="Extrai valor, vencimento e banco",
            input_schema={"texto": "string"},
            output_schema={
                "valor": "float",
                "banco": "string",
                "vencimento": "string",
            },
        )],
    ))
    agente_boleto.register_handler(
        "ler_boleto", _handler_ler_boleto
    )

    agente_cnpj = AgentServer(AgentCard(
        agent_id="agent-cnpj-v1",
        name="AgenteValidadorCNPJ",
        description="Valida CNPJs via Receita Federal",
        version="1.0.0",
        endpoint="https://agentes.empresa.com/cnpj",
        skills=[AgentSkill(
            id="validar_cnpj",
            name="Validação de CNPJ",
            description="Verifica dígitos e situação",
            input_schema={"cnpj": "string"},
            output_schema={
                "valido": "bool",
                "situacao": "string",
            },
        )],
    ))
    agente_cnpj.register_handler(
        "validar_cnpj", _handler_validar_cnpj
    )

    agente_doc = AgentServer(AgentCard(
        agent_id="agent-doc-v1",
        name="AgenteClassificador",
        description="Classifica tipo de documento",
        version="1.0.0",
        endpoint="https://agentes.empresa.com/docs",
        skills=[AgentSkill(
            id="classificar_documento",
            name="Classificação de Documento",
            description=(
                "Identifica boleto, NF-e, contrato, etc."
            ),
            input_schema={"texto": "string"},
            output_schema={
                "tipo": "string",
                "confianca": "float",
            },
        )],
    ))
    agente_doc.register_handler(
        "classificar_documento", _handler_classificar_doc
    )

    # --- Registro ---
    registry = AgentRegistry()
    registry.register(agente_boleto)
    registry.register(agente_cnpj)
    registry.register(agente_doc)

    # --- Exibe AgentCards ---
    console.rule("[yellow]AgentCards disponíveis")
    tabela = Table(header_style="bold magenta")
    tabela.add_column("Agent ID")
    tabela.add_column("Nome")
    tabela.add_column("Skills")
    for card in registry.list_cards():
        tabela.add_row(
            card.agent_id,
            card.name,
            ", ".join(s.id for s in card.skills),
        )
    console.print(tabela)

    # --- Orquestrador delega tarefas ---
    console.rule("[yellow]Delegação via A2A")
    orch = OrchestratorAgent("orchestrator-main", registry)

    tarefas: list[tuple[str, dict]] = [
        (
            "classificar_documento",
            {"texto": "Boleto bancário vencimento 10/05"},
        ),
        (
            "ler_boleto",
            {"texto": "Boleto R$ 1.250,00 venc 10/05/2026"},
        ),
        (
            "validar_cnpj",
            {"cnpj": "11.222.333/0001-81"},
        ),
        (
            "skill_inexistente",
            {"texto": "teste"},
        ),
    ]

    resultados_tabela = Table(header_style="bold magenta")
    resultados_tabela.add_column("Skill")
    resultados_tabela.add_column("Status")
    resultados_tabela.add_column("Output")

    for skill_id, input_data in tarefas:
        resp = orch.delegate(skill_id, input_data)
        status_fmt = (
            "[green]DONE[/]"
            if resp.status == TaskStatus.DONE
            else "[red]FAILED[/]"
        )
        output_str = (
            str(resp.output_data)[:60]
            if resp.status == TaskStatus.DONE
            else resp.error or ""
        )
        resultados_tabela.add_row(
            skill_id, status_fmt, output_str
        )

    console.rule("[yellow]Resultados")
    console.print(resultados_tabela)


if __name__ == "__main__":
    demo_a2a()
