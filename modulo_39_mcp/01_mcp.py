"""
============================================================
MÓDULO 39.1 - MCP (MODEL CONTEXT PROTOCOL)
============================================================
O MCP (Model Context Protocol), criado pela Anthropic em
2024 e adotado amplamente em 2025, padroniza como agentes
de IA se conectam a ferramentas e fontes de dados externas.

PROBLEMA QUE RESOLVE:
Antes do MCP, cada ferramenta tinha sua própria interface.
Um agente que precisasse de 10 ferramentas tinha 10
integrações diferentes. Com MCP, o agente fala o mesmo
protocolo para qualquer ferramenta.

ARQUITETURA MCP:

  ┌──────────────────────────────────────────────────────┐
  │  MCP HOST (ex: Claude Desktop, VS Code, seu agente)  │
  │  └── MCPClient → protocolo padrão                    │
  └────────────────┬─────────────────────────────────────┘
                   │  JSON-RPC 2.0 (stdio ou HTTP/SSE)
  ┌────────────────▼─────────────────────────────────────┐
  │  MCP SERVER (expõe recursos ao agente)               │
  │  ├── Tools     → funções que o LLM pode chamar       │
  │  ├── Resources → arquivos, DBs (leitura)             │
  │  └── Prompts   → templates reutilizáveis             │
  └──────────────────────────────────────────────────────┘

DIFERENÇA vs TOOL CALLING TRADICIONAL:
  Tool calling ad-hoc: agente define e chama tools
    diretamente — acoplado, difícil de reutilizar.
  MCP: servidor independente expõe tools — o agente
    descobre e usa sem conhecer a implementação.

FLUXO:
  1. MCPClient conecta ao servidor
  2. Chama tools/list → descobre ferramentas disponíveis
  3. LLM decide qual tool usar (igual ao tool calling)
  4. MCPClient chama tools/call com argumentos
  5. MCPServer executa e retorna resultado

Tópicos cobertos:
1. MCPTool — definição de ferramenta com schema JSON
2. MCPResource — recurso somente leitura
3. MCPServer — servidor que expõe tools e resources
4. MCPClient — cliente que descobre e chama tools
5. Loop de agente integrado com MCP
6. Demo com servidor de boletos e servidor de CNPJ
============================================================
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. DEFINIÇÕES DE TOOL E RESOURCE
# ============================================================

@dataclass
class MCPTool:
    """
    Definição de uma ferramenta MCP.
    Equivale ao JSON Schema passado para o LLM no
    tool calling convencional.
    """
    name: str
    description: str
    input_schema: dict    # JSON Schema { type, properties }
    handler: Callable[[dict], Any]

    def to_dict(self) -> dict:
        """Serializa no formato do protocolo MCP."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPResource:
    """
    Recurso somente leitura exposto pelo servidor.
    Ex: arquivo de políticas, schema do banco.
    """
    uri: str          # mcp://empresa/politicas/cobranca
    name: str
    mime_type: str
    content: str      # texto do recurso


# ============================================================
# 2. MCP SERVER
# ============================================================

class MCPServer:
    """
    Servidor MCP que expõe tools e resources.
    Em produção: processo separado comunicando via stdio
    ou HTTP/SSE com JSON-RPC 2.0.
    """

    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}

    def add_tool(self, tool: MCPTool) -> None:
        """Adiciona uma tool ao servidor. O handler é a função"""
        self._tools[tool.name] = tool

    def add_resource(self, resource: MCPResource) -> None:
        """Adiciona um recurso somente leitura ao servidor."""
        self._resources[resource.uri] = resource

    # ---- Handlers de protocolo ----

    def handle_tools_list(self) -> list[dict]:
        """tools/list — lista todas as tools disponíveis."""
        return [t.to_dict() for t in self._tools.values()]

    def handle_tools_call(
        self, name: str, arguments: dict
    ) -> dict:
        """tools/call — executa uma tool pelo nome."""
        tool = self._tools.get(name)
        if tool is None:
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Tool '{name}' não encontrada"
                        ),
                    }
                ],
            }
        try:
            result = tool.handler(arguments)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            json.dumps(result, ensure_ascii=False)
                            if isinstance(result, dict)
                            else str(result)
                        ),
                    }
                ]
            }
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "isError": True,
                "content": [
                    {"type": "text", "text": str(exc)}
                ],
            }

    def handle_resources_read(self, uri: str) -> dict:
        """resources/read — lê um recurso pelo URI."""
        resource = self._resources.get(uri)
        if resource is None:
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": f"Resource '{uri}' não encontrado",
                    }
                ],
            }
        return {
            "contents": [
                {
                    "uri": resource.uri,
                    "mimeType": resource.mime_type,
                    "text": resource.content,
                }
            ]
        }


# ============================================================
# 3. MCP CLIENT
# ============================================================

class MCPClient:
    """
    Cliente MCP usado pelo agente/host para se comunicar
    com um MCPServer. Simula o transporte JSON-RPC.
    """

    def __init__(self, server: MCPServer) -> None:
        self._server = server
        self._tools_cache: Optional[list[dict]] = None

    def list_tools(self) -> list[dict]:
        """Descobre as tools disponíveis no servidor."""
        if self._tools_cache is None:
            self._tools_cache = (
                self._server.handle_tools_list()
            )
        return self._tools_cache

    def call_tool(
        self, name: str, arguments: dict
    ) -> dict:
        """Chama uma tool no servidor."""
        return self._server.handle_tools_call(
            name, arguments
        )

    def read_resource(self, uri: str) -> dict:
        """Lê um resource do servidor."""
        return self._server.handle_resources_read(uri)


# ============================================================
# 4. LOOP DE AGENTE COM MCP
# ============================================================

def executar_agente_mcp(
    pergunta: str,
    client: MCPClient,
) -> str:
    """
    Simula um loop de agente que usa MCP para descobrir
    e chamar tools. Em produção, o LLM decide qual tool
    chamar com base nas definições retornadas por
    list_tools().
    """
    tools_disponiveis = client.list_tools()
    nomes = [t["name"] for t in tools_disponiveis]
    console.print(
        f"  [dim]Tools MCP disponíveis: {nomes}[/]"
    )

    # Simulação de decisão do LLM:
    # Em produção → passa tools para o LLM e ele decide
    resultado_final = []
    pergunta_lower = pergunta.lower()

    if "boleto" in pergunta_lower:
        resp = client.call_tool(
            "extrair_boleto", {"texto": pergunta}
        )
        resultado_final.append(
            resp["content"][0]["text"]
        )

    if "cnpj" in pergunta_lower:
        m = re.search(r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}",
                      pergunta)
        cnpj = m.group(0) if m else "00.000.000/0000-00"
        resp = client.call_tool(
            "validar_cnpj", {"cnpj": cnpj}
        )
        resultado_final.append(
            resp["content"][0]["text"]
        )

    if "politica" in pergunta_lower or "prazo" in pergunta_lower:
        resp = client.read_resource(
            "mcp://empresa/politicas/cobranca"
        )
        if "contents" in resp:
            resultado_final.append(
                resp["contents"][0]["text"][:200]
            )

    return (
        "\n".join(resultado_final)
        if resultado_final
        else "Nenhuma tool aplicável para esta pergunta."
    )


# ============================================================
# 5. DEMO
# ============================================================

def _criar_servidor_financeiro() -> MCPServer:
    """Cria um MCPServer com tools financeiras."""
    server = MCPServer(
        name="mcp-servidor-financeiro",
        version="1.0.0",
    )

    # Tool: extrair boleto
    def _extrair_boleto(args: dict) -> dict:
        texto = args.get("texto", "")
        m = re.search(r"R\$\s*([\d.,]+)", texto)
        valor = None
        if m:
            try:
                valor = float(
                    m.group(1)
                    .replace(".", "")
                    .replace(",", ".")
                )
            except ValueError:
                pass
        return {
            "valor": valor,
            "banco": "Banco Simulado",
            "vencimento": "2026-05-10",
        }

    server.add_tool(MCPTool(
        name="extrair_boleto",
        description=(
            "Extrai valor, banco e vencimento de um "
            "boleto bancário a partir do texto"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "texto": {
                    "type": "string",
                    "description": "Texto do boleto",
                }
            },
            "required": ["texto"],
        },
        handler=_extrair_boleto,
    ))

    # Tool: validar CNPJ
    def _validar_cnpj(args: dict) -> dict:
        cnpj = args.get("cnpj", "")
        nums = re.sub(r"\D", "", cnpj)
        valido = len(nums) == 14 and len(set(nums)) > 1
        return {
            "cnpj": cnpj,
            "valido": valido,
            "situacao": "ATIVA" if valido else "INVÁLIDA",
        }

    server.add_tool(MCPTool(
        name="validar_cnpj",
        description="Valida um CNPJ e retorna sua situação",
        input_schema={
            "type": "object",
            "properties": {
                "cnpj": {
                    "type": "string",
                    "description": "CNPJ a validar",
                }
            },
            "required": ["cnpj"],
        },
        handler=_validar_cnpj,
    ))

    # Resource: política de cobrança
    server.add_resource(MCPResource(
        uri="mcp://empresa/politicas/cobranca",
        name="Política de Cobrança",
        mime_type="text/plain",
        content=(
            "POLÍTICA DE COBRANÇA v2.0\n"
            "Prazo de pagamento: 30 dias após emissão.\n"
            "Multa por atraso: 2% ao mês.\n"
            "Desconto pontualidade: 5% se pago em 10 dias."
        ),
    ))

    return server


def demo_mcp() -> None:
    """Demo completa do MCP com servidor financeiro e cliente agente."""
    console.print(
        Panel(
            "[bold]Módulo 39 — MCP "
            "(Model Context Protocol)[/]\n"
            "Protocolo padronizado para conectar agentes "
            "a ferramentas e fontes de dados",
            style="bold blue",
        )
    )

    server = _criar_servidor_financeiro()
    client = MCPClient(server)

    # Exibe tools disponíveis
    console.rule("[yellow]Descoberta de Tools (tools/list)")
    tabela = Table(header_style="bold magenta")
    tabela.add_column("Tool")
    tabela.add_column("Descrição")
    tabela.add_column("Parâmetros")
    for tool in client.list_tools():
        params = ", ".join(
            tool["inputSchema"]
            .get("properties", {})
            .keys()
        )
        tabela.add_row(
            tool["name"],
            tool["description"][:50],
            params,
        )
    console.print(tabela)

    # Loop de agente com 3 perguntas
    console.rule("[yellow]Loop de Agente com MCP")
    perguntas = [
        "Qual o valor do boleto R$ 1.750,00 venc 10/05?",
        "Valide o CNPJ 11.222.333/0001-81 do fornecedor.",
        "Qual o prazo e multa da política de cobrança?",
    ]
    for pergunta in perguntas:
        console.print(f"\n  [bold cyan]Q:[/] {pergunta}")
        resposta = executar_agente_mcp(pergunta, client)
        console.print(f"  [bold green]A:[/] {resposta}")


if __name__ == "__main__":
    demo_mcp()
