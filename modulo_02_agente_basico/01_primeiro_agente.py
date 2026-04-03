"""
============================================================
MÓDULO 2.1 - PRIMEIRO AGENTE
============================================================
Construímos do zero o loop básico de um agente:
1. Receber input do usuário
2. Enviar para a LLM
3. Receber resposta
4. Repetir

CONCEITO CHAVE - O QUE É UM AGENTE?
Um agente é diferente de uma simples chamada à LLM porque:
- Roda em LOOP (não é uma chamada única)
- Mantém ESTADO (lembra do contexto)
- Toma DECISÕES (escolhe o que fazer baseado na situação)
- Pode AGIR (executar ferramentas/funções)

Neste módulo, construiremos a versão MAIS SIMPLES:
- O loop existe (while True)
- Mas sem memória, sem tools, sem decisões complexas
- Vamos evoluir nos próximos módulos!

ANALOGIA:
Chamar a LLM uma vez = usar uma calculadora
Agente = ter um assistente que conversa com você
============================================================
"""

import os
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console  # Rich: output colorido no terminal

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
console = Console()


# ============================================================
# O AGENTE MAIS SIMPLES POSSÍVEL
# ============================================================
# Um agente é basicamente: LOOP + LLM + DECISÃO
#
# Fluxo deste agente:
#   ┌──────────────────────────────────┐
#   │  Usuário digita pergunta         │
#   └────────────────┬─────────────────┘
#                  │
#                  v
#   ┌──────────────────────────────────┐
#   │  Envia para LLM (sem histórico)  │
#   └────────────────┬─────────────────┘
#                  │
#                  v
#   ┌──────────────────────────────────┐
#   │  Mostra resposta ao usuário      │
#   └────────────────┬─────────────────┘
#                  │
#           volta ao início
#
# PROBLEMA: cada pergunta é isolada (sem memória)!
# Isso será corrigido no próximo arquivo (02_agente_com_memoria.py)

# O SYSTEM PROMPT define a IDENTIDADE do agente.
# É a mensagem mais importante — tudo que o agente faz parte daqui.
# Veja as boas práticas do módulo 1 para construir bons system prompts.
SYSTEM_PROMPT = (
    "Você é um assistente especializado em boletos bancários "
    "brasileiros.\n\n"
    "Você pode ajudar com:\n"
    "- Explicar campos de um boleto\n"
    "- Calcular multas e juros por atraso\n"
    "- Identificar informações do boleto\n\n"
    "Sempre seja claro e objetivo. Se não souber algo, diga que "
    "não sabe.\n"
    "Quando o usuário disser 'sair', 'exit' ou 'tchau', "
    "despeça-se educadamente."
)


def agente_simples():
    """
    AGENTE v1 - Loop básico de conversação.

    TEORIA:
    O loop mais simples de um agente:

    while True:
        input → LLM → output

    Problemas desta versão:
    - Sem memória (cada pergunta é isolada)
    - Sem ferramentas
    - Sem validação

    Vamos melhorar nos próximos módulos!
    """
    console.print("\n🤖 Agente de Boletos v1 - Básico", style="bold blue")
    console.print("Digite 'sair' para encerrar\n", style="dim")

    while True:
        # 1. RECEBER INPUT
        user_input = input("👤 Você: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("sair", "exit", "tchau"):
            console.print("🤖 Até logo! 👋", style="bold blue")
            break

        # 2. ENVIAR PARA LLM
        # NOTA: enviamos APENAS a mensagem atual + system prompt
        # Não há histórico! A LLM não sabe o que foi perguntado antes.
        # Esse é o problema principal desta versão v1.
        mensagens = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        try:
            resposta = client.chat.completions.create(
                model=MODEL,
                messages=mensagens,
                temperature=0.3,
                max_tokens=1024,
            )

            # 3. MOSTRAR RESPOSTA
            texto = resposta.choices[0].message.content
            console.print(f"🤖 Agente: {texto}\n", style="cyan")

        except (ConnectionError, TimeoutError, RuntimeError) as e:
            console.print(f"❌ Erro: {e}", style="bold red")


# ============================================================
# DEMONSTRAÇÃO NÃO-INTERATIVA (para entender o fluxo)
# ============================================================
# No modo demo, executamos perguntas automáticas para mostrar:
# 1. O fluxo de input → LLM → output
# 2. O PROBLEMA de não ter memória (cada pergunta é isolada)
# Use este modo na apresentação do treinamento.
def demo_agente_simples():
    """Demonstração automática do agente para fins didáticos."""
    console.print("\n🎓 DEMO: Fluxo do Agente Simples", style="bold yellow")
    console.print("=" * 50)

    # Perguntas-exemplo para demonstração automática
    # Note que a 3ª pergunta ("Qual a diferença entre boleto e PIX?")
    # não se conecta às anteriores — porque o agente não tem memória
    perguntas = [
        "O que é a linha digitável de um boleto?",
        "Meu boleto de R$ 500 venceu há 5 dias. Quanto pago de "
        "multa com 2% de multa e 0,033% de juros ao dia?",
        "Qual a diferença entre boleto e PIX?",
    ]

    for i, pergunta in enumerate(perguntas, 1):
        console.print(f"\n--- Iteração {i} do Loop ---", style="dim")
        console.print(f"👤 Input: {pergunta}", style="white")

        mensagens = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pergunta},
        ]

        resposta = client.chat.completions.create(
            model=MODEL,
            messages=mensagens,
            temperature=0.3,
            max_tokens=512,
        )

        texto = resposta.choices[0].message.content
        console.print(f"🤖 Output: {texto}", style="cyan")

    console.print(
        "\n⚠️  PROBLEMA: O agente não lembra das perguntas anteriores!",
        style="bold red",
    )
    console.print("   Cada pergunta é tratada de forma isolada.", style="red")
    console.print(
        "   Vamos resolver isso no próximo arquivo! →", style="yellow",
    )


if __name__ == "__main__":
    console.print("🎓 MÓDULO 2.1 - PRIMEIRO AGENTE", style="bold blue")
    console.print("=" * 60)

    # Modo demo (não-interativo) - bom para apresentação
    demo_agente_simples()

    # Modo interativo - descomente para testar
    # agente_simples()

    console.print("\n✅ Módulo 2.1 concluído!", style="bold green")
