"""
============================================================
MÓDULO 3.2 - TOOL CALLING (LLM + FERRAMENTAS)
============================================================
Agora conectamos as Tools à LLM!
A LLM decide quando chamar cada tool e com quais argumentos.

FLUXO COMPLETO:
1. Usuário faz pergunta
2. LLM analisa e decide se precisa de uma tool
3. Se sim: retorna tool_call (nome + args)
4. Nosso código executa a ferramenta
5. Resultado volta para a LLM
6. LLM formula resposta final
7. Pode repetir (múltiplas tools em sequência)

COMO A LLM SABE QUAIS TOOLS EXISTEM?
Nós passamos TOOLS_SCHEMA no parâmetro tools= da API.
A LLM lê as descrições e decide se precisa chamar alguma.

PROTOCOLO DE MENSAGENS (como a conversa fica no histórico):
  {"role": "user",      "content": "Pergunta do usuário"}
  {"role": "assistant",  "tool_calls": [{"name": "...", "args": {...}}]}
  {"role": "tool",       "tool_call_id": "...", "content": "resultado"}
  {"role": "assistant",  "content": "Resposta final usando o resultado"}
============================================================
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console

# Importar tools e schemas do módulo base (compartilhado)
# TOOLS_SCHEMA = lista de descrições JSON que a LLM recebe
# TOOLS_REGISTRY = dict nome → função para executar
from modulo_03_tools.tools_base import (
    TOOLS_SCHEMA,
    TOOLS_REGISTRY,
)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
console = Console()


# ============================================================
# SYSTEM PROMPT - INSTRUÇÕES PARA O AGENTE
# ============================================================
# O system prompt informa a LLM sobre suas capacidades (tools)
# e define REGRAS de comportamento. A LLM só sabe que tem tools
# porque passamos no parâmetro tools= E mencionamos aqui.

SYSTEM_PROMPT = (
    "Você é um assistente especializado em boletos bancários "
    "brasileiros.\n\n"
    "Você tem acesso a ferramentas para:\n"
    "- Calcular multas e juros de boletos vencidos\n"
    "- Validar linhas digitáveis\n"
    "- Verificar se boletos estão vencidos\n"
    "- Identificar bancos pelo código\n\n"
    "REGRAS:\n"
    "- Use as ferramentas sempre que precisar de dados precisos\n"
    "- Não invente valores - use a ferramenta de cálculo\n"
    "- Explique os resultados de forma clara ao usuário\n"
    "- Se não tiver informação suficiente, pergunte ao usuário"
)


def executar_tool(nome: str, argumentos: dict) -> str:
    """
    Executa uma tool pelo nome com os argumentos fornecidos.

    TEORIA:
    Esta função é o "executor" - ela recebe o que a LLM pediu
    e chama a função Python correspondente.

    FLUXO:
      LLM retorna: {"name": "calcular_multa_juros", "args": {"valor": 800}}
      → TOOLS_REGISTRY["calcular_multa_juros"] retorna a função
      → func(**{"valor": 800}) executa a função com os args
      → resultado é convertido para JSON string e devolvido à LLM

    POR QUE RETORNA STRING JSON (e não dict)?
    Porque o protocolo de tool calling exige que o campo "content"
    da mensagem com role="tool" seja uma string.
    """
    if nome not in TOOLS_REGISTRY:
        return json.dumps({"erro": f"Tool '{nome}' não encontrada"})

    try:
        func = TOOLS_REGISTRY[nome]
        resultado = func(**argumentos)
        return json.dumps(resultado, ensure_ascii=False)
    except (KeyError, TypeError, ValueError) as e:
        return json.dumps({"erro": str(e)})


def agente_com_tools(pergunta: str, verbose: bool = True) -> str:
    """
    AGENTE v3 - Com Tool Calling.

    TEORIA - O LOOP DO AGENTE COM TOOLS:

    O agente funciona em um LOOP porque uma única pergunta pode
    precisar de múltiplas tools. Exemplo:
      "Meu boleto de R$ 500 venceu em 01/03/2025, quanto pago?"
      → Tool 1: verificar_vencimento("01/03/2025")
      → Tool 2: calcular_multa_juros(500, dias_atraso)
      → Resposta final com o valor atualizado

    PSEUDOCÓDIGO DO LOOP:
    while True:
        resposta = LLM(mensagens + tools)

        if resposta tem tool_calls:
            para cada tool_call:
                resultado = executar_tool(tool_call)
                adicionar resultado às mensagens
            continuar loop (LLM vê os resultados)
        else:
            retornar resposta de texto (fim)
    """
    mensagens = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pergunta},
    ]

    if verbose:
        console.print(f"\n👤 Pergunta: {pergunta}", style="white")
        console.print("─" * 50)

    # Loop do agente: pode precisar de múltiplas iterações
    # Cada iteração = 1 chamada à LLM
    # Segurança: evitar loop infinito
    max_iteracoes = 5
    for iteracao in range(max_iteracoes):
        if verbose:
            console.print(f"\n🔄 Iteração {iteracao + 1}", style="dim")

        # Chamar LLM com as tools disponíveis
        # tools= envia os schemas para a LLM saber o que pode chamar
        resposta = client.chat.completions.create(
            model=MODEL,
            messages=mensagens,
            tools=TOOLS_SCHEMA,
            # "auto" = LLM decide se usa tool ou responde direto
            # "none" = força resposta sem tools
            # {"type": "function", "function": {"name": "..."}}
            #   = força uma tool específica
            tool_choice="auto",
            # Baixa temperatura para cálculos precisos
            temperature=0.2,
            max_tokens=1024,
        )

        mensagem_resposta = resposta.choices[0].message

        # Caso 1: LLM quer chamar uma ou mais tools
        # A LLM retorna tool_calls pedindo para executarmos funções.
        # Nós executamos e passamos o resultado de volta.
        if mensagem_resposta.tool_calls:
            # PASSO CRÍTICO: Adicionar a mensagem da LLM ao histórico
            # A API EXIGE que a mensagem do assistant com tool_calls
            # apareça ANTES das mensagens de role="tool"
            mensagens.append({
                "role": "assistant",
                "content": mensagem_resposta.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in mensagem_resposta.tool_calls
                ]
            })

            # Executar cada tool chamada
            for tool_call in mensagem_resposta.tool_calls:
                nome_tool = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if verbose:
                    console.print(
                        f"   🔧 Tool: {nome_tool}", style="bold yellow",
                    )
                    args_str = json.dumps(args, ensure_ascii=False)
                    console.print(
                        f"      Args: {args_str}", style="dim",
                    )

                # Executar a tool
                resultado = executar_tool(nome_tool, args)

                if verbose:
                    console.print(
                        f"      Resultado: {resultado}", style="cyan",
                    )

                # PASSO CRÍTICO: Adicionar resultado ao histórico
                # role="tool" = resultado de uma tool
                # tool_call_id vincula ao pedido específico da LLM
                mensagens.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": resultado,
                })

        # Caso 2: LLM respondeu com texto (sem tool call) = FIM
        # Se a LLM tem toda a informação que precisa, ela responde em texto.
        # Isso pode ser após 0 ou mais chamadas de tools.
        else:
            texto_final = mensagem_resposta.content
            if verbose:
                console.print(
                    f"\n🤖 Resposta Final: {texto_final}",
                    style="bold cyan",
                )
            return texto_final

    return "⚠️ Número máximo de iterações atingido."


# ============================================================
# DEMONSTRAÇÃO
# ============================================================
# Testamos vários cenários para ver o agente usando tools automaticamente.
# Cada pergunta ativa tools diferentes — observe no output!
def demo_tool_calling():
    """Demonstra o agente usando tools automaticamente."""
    console.print("\n🎓 DEMO: Agente com Tool Calling", style="bold yellow")
    console.print("=" * 60)

    perguntas = [
        # Pergunta 1: usa calcular_multa_juros (1 tool)
        "Meu boleto de R$ 800,00 venceu há 10 dias. "
        "Multa de 2% e juros de 0,033% ao dia. Quanto pago?",

        # Pergunta 2: usa validar_linha_digitavel + buscar_banco (2 tools)
        "Valide esta linha digitável: "
        "34191.09065 44830.136706 00000.000178 1 70060000125000",

        # Pergunta 3: usa verificar_vencimento + calcular_multa_juros
        "Tenho um boleto com vencimento em 01/03/2026, "
        "valor R$ 2.500,00. Está vencido? Se sim, quanto pago "
        "com multa de 2% e juros de 0,033%/dia?",

        # Pergunta 4: sem tool (LLM responde da base de conhecimento)
        "O que significa o código de barras de um boleto?",
    ]

    for i, pergunta in enumerate(perguntas, 1):
        console.print(f"\n{'='*60}", style="bold")
        console.print(f"📝 TESTE {i}", style="bold blue")
        agente_com_tools(pergunta, verbose=True)
        console.print()


# ============================================================
# MODO INTERATIVO
# ============================================================
# Aqui o usuário pode conversar livremente com o agente.
# O agente decide automaticamente quando usar tools.
def agente_interativo():
    """Modo interativo do agente com tools."""
    console.print("\n🤖 Agente de Boletos v3 - Com Tools", style="bold blue")
    console.print("Digite 'sair' para encerrar\n", style="dim")

    while True:
        user_input = input("👤 Você: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("sair", "exit"):
            console.print("👋 Até logo!", style="bold blue")
            break

        try:
            agente_com_tools(user_input, verbose=True)
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            console.print(f"❌ Erro: {e}", style="bold red")


if __name__ == "__main__":
    console.print("🎓 MÓDULO 3.2 - TOOL CALLING", style="bold blue")
    console.print("=" * 60)

    demo_tool_calling()

    # Modo interativo - descomente para testar
    # agente_interativo()

    console.print("\n✅ Módulo 3.2 concluído!", style="bold green")
    console.print(
        "\n💡 PRÓXIMO: Adicionar GUARDRAILS de segurança (Módulo 4) →",
        style="yellow",
    )
