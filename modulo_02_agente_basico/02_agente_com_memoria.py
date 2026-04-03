"""
============================================================
MÓDULO 2.2 - AGENTE COM MEMÓRIA
============================================================
Evoluímos o agente para manter histórico de conversação.

CONCEITO CHAVE:
A "memória" de um agente baseado em LLM é simplesmente a
lista de mensagens anteriores enviada junto ao prompt.
A LLM não "lembra" de nada sozinha — somos NÓS que reenviamos
o histórico a cada chamada.

COMO A MEMÓRIA FUNCIONA POR DENTRO:
  Chamada 1: [system, user_msg_1]
  Chamada 2: [system, user_msg_1, assistant_msg_1, user_msg_2]
  Chamada 3: [system, user_msg_1, assistant_msg_1, user_msg_2,
             assistant_msg_2, user_msg_3]
  ... e assim por diante

A cada nova mensagem, reenviamos TODA a conversa anterior.
Isso é mais simples do que parece, mas tem UM GRANDE PROBLEMA:
LLMs têm limite de contexto (ex: 8k, 32k, 128k tokens).
Se o histórico ficar grande demais, precisamos TRUNCAR.

EVOLUÇÃO:
  Módulo 2.1 (sem memória):  input → LLM → output  (isolado)
  Módulo 2.2 (com memória): input + histórico → LLM → output (contínuo)
============================================================
"""

import os
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
console = Console()


# O SYSTEM PROMPT do agente v2 é mais detalhado que o v1.
# Adicionamos a regra "Referencie informações anteriores da conversa"
# para que a LLM USE ativamente a memória/histórico.
SYSTEM_PROMPT = (
    "Você é um assistente especializado em boletos bancários "
    "brasileiros.\n\n"
    "CAPACIDADES:\n"
    "- Explicar campos e informações de boletos\n"
    "- Calcular multas e juros por atraso\n"
    "- Responder dúvidas sobre pagamentos\n\n"
    "REGRAS:\n"
    "- Seja claro e objetivo\n"
    "- Use linguagem acessível\n"
    "- Se não souber, diga que não sabe\n"
    "- Referencie informações anteriores da conversa quando relevante"
)


class AgenteComMemoria:
    """
    AGENTE v2 - Com memória de conversação.

    TEORIA - TIPOS DE MEMÓRIA:

    1. Memória de Curto Prazo (esta implementação):
       - Histórico da conversa atual
       - Perdida ao encerrar o programa

    2. Memória de Longo Prazo (futuro):
       - Banco de dados / vetores
       - Persiste entre sessões

    3. Memória de Trabalho:
       - Contexto da tarefa atual
       - Resultados intermediários

    ATENÇÃO:
    - LLMs têm limite de contexto (ex: 8k, 32k, 128k tokens)
    - Precisamos gerenciar o tamanho do histórico
    """

    def __init__(self, system_prompt: str, max_historico: int = 20):
        self.system_prompt = system_prompt
        self.max_historico = max_historico
        # A "memória" é simplesmente a lista de mensagens
        self.historico: list[dict] = []

    def adicionar_mensagem(self, role: str, content: str):
        """
        Adiciona mensagem ao histórico com controle de tamanho.

        Cada mensagem é um dict:
        {"role": "user"/"assistant", "content": "texto"}
        O histórico cresce a cada turno da conversação.
        """
        self.historico.append({"role": role, "content": content})

        # GESTÃO DA JANELA DE CONTEXTO:
        # LLMs têm limite de tokens. Se o histórico for grande demais,
        # a chamada falha ou fica cara. Precisamos truncar.
        #
        # ESTRATÉGIA: mantemos as 2 primeiras mensagens (geralmente contêm
        # contexto importante ("Tenho um boleto do Itaú...") + as últimas N.
        # Assim preservamos o contexto inicial e o contexto recente.
        #
        # ALTERNATIVAS mais sofisticadas (não implementadas aqui):
        # - Resumir mensagens antigas com a LLM
        # - Usar embedding + banco vetorial para memória de longo prazo
        # - Comprimir histórico mantendo fatos-chave
        if len(self.historico) > self.max_historico:
            # Estratégia: manter as primeiras 2 + últimas (max-2) mensagens
            # As primeiras geralmente contêm contexto importante
            self.historico = (
                self.historico[:2]
                + self.historico[-(self.max_historico - 2):]
            )
            console.print(
                "⚠️  Histórico truncado para caber no contexto",
                style="dim yellow",
            )

    def montar_mensagens(self) -> list[dict]:
        """
        Monta a lista completa de mensagens para enviar à LLM.

        A estrutura final enviada à LLM é:
        [system_prompt, msg_user_1, msg_assistant_1, msg_user_2, ...]

        O system prompt SEMPRE vai primeiro (não faz parte do histórico).
        Depois, todo o histórico da conversa em ordem cronológica.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            *self.historico,  # Desempacota toda a lista de mensagens
        ]

    def processar(self, user_input: str) -> str:
        """
        Processa uma mensagem do usuário e retorna a resposta.

        FLUXO COMPLETO:
        1. Salva a mensagem do usuário na memória
        2. Monta TODAS as mensagens (system + histórico completo)
        3. Envia tudo para a LLM
        4. Salva a resposta da LLM na memória (para próximos turnos)
        5. Retorna o texto da resposta

        O segredo está no passo 2: a LLM recebe TODA a conversa,
        então ela consegue referenciar mensagens anteriores.
        """
        # 1. Adicionar input do usuário à memória
        self.adicionar_mensagem("user", user_input)

        # 2. Montar mensagens com TODO o histórico
        mensagens = self.montar_mensagens()

        # 3. Chamar a LLM
        resposta = client.chat.completions.create(
            model=MODEL,
            messages=mensagens,
            temperature=0.3,
            max_tokens=1024,
        )

        texto = resposta.choices[0].message.content

        # 4. Adicionar resposta à memória (a LLM "lembra" do que disse)
        self.adicionar_mensagem("assistant", texto)

        return texto

    def mostrar_historico(self):
        """Mostra o histórico de conversação (debug)."""
        console.print("\n📜 HISTÓRICO DA CONVERSA:", style="bold yellow")
        for i, msg in enumerate(self.historico):
            role = "👤" if msg["role"] == "user" else "🤖"
            conteudo = msg["content"]
            texto = (
                conteudo[:100] + "..." if len(conteudo) > 100 else conteudo
            )
            console.print(
                f"   {i+1}. {role} [{msg['role']}]: {texto}",
                style="dim",
            )
        total = len(self.historico)
        console.print(f"   Total: {total} mensagens\n", style="dim")


# ============================================================
# DEMONSTRAÇÃO COM MEMÓRIA
# ============================================================
# A demo usa 3 perguntas onde CADA UMA DEPENDE DA ANTERIOR:
# 1. "Tenho um boleto do Itaú de R$ 1.200" → estabelece contexto
# 2. "Calcule multa e juros" → só faz sentido com o contexto do #1
# 3. "E se eu pagar semana que vem?" → depende do #1 e #2
#
# Sem memória (módulo 2.1), a LLM não saberia de qual boleto estamos falando.
# Com memória, ela "lembra" porque reenviamos todo o histórico.
def demo_agente_com_memoria():
    """Demonstra que o agente agora lembra do contexto anterior."""
    console.print("\n🎓 DEMO: Agente COM Memória", style="bold yellow")
    console.print("=" * 50)

    agente = AgenteComMemoria(SYSTEM_PROMPT)

    # Conversação onde cada pergunta depende da anterior
    perguntas = [
        "Tenho um boleto do Itaú no valor de R$ 1.200,00 "
        "que venceu dia 01/03/2026.",
        "Considerando multa de 2% e juros de 0,033% ao dia, "
        "quanto pago hoje (16/03/2026)?",
        "E se eu pagar só semana que vem, no dia 23/03?",
    ]

    for i, pergunta in enumerate(perguntas, 1):
        console.print(f"\n--- Turno {i} ---", style="dim")
        console.print(f"👤 Você: {pergunta}", style="white")

        resposta = agente.processar(pergunta)
        console.print(f"🤖 Agente: {resposta}", style="cyan")

    # Mostrar que a memória foi mantida
    agente.mostrar_historico()

    console.print(
        "✅ O agente lembra do boleto mencionado na 1ª pergunta!",
        style="bold green",
    )
    console.print(
        "   Na 3ª pergunta, ele sabe do que estamos falando.",
        style="green",
    )


# ============================================================
# MODO INTERATIVO
# ============================================================
# Neste modo, o aluno pode conversar livremente com o agente.
# Comandos especiais:
# - 'historico': mostra toda a conversa salva na memória
# - 'limpar': reseta a memória (como começar uma conversa nova)
# - 'sair': encerra o agente
def agente_interativo():
    """Modo interativo do agente com memória."""
    console.print("\n🤖 Agente de Boletos v2 - Com Memória", style="bold blue")
    console.print("Comandos: 'historico' | 'limpar' | 'sair'\n", style="dim")

    agente = AgenteComMemoria(SYSTEM_PROMPT)

    while True:
        user_input = input("👤 Você: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("sair", "exit"):
            console.print("🤖 Até logo! 👋", style="bold blue")
            break
        if user_input.lower() == "historico":
            agente.mostrar_historico()
            continue
        if user_input.lower() == "limpar":
            agente.historico = []
            console.print("🗑️  Memória limpa!", style="yellow")
            continue

        try:
            resposta = agente.processar(user_input)
            console.print(f"🤖 Agente: {resposta}\n", style="cyan")
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            console.print(f"❌ Erro: {e}", style="bold red")


if __name__ == "__main__":
    console.print("🎓 MÓDULO 2.2 - AGENTE COM MEMÓRIA", style="bold blue")
    console.print("=" * 60)

    # Demo automática
    demo_agente_com_memoria()

    # Modo interativo - descomente para testar
    # agente_interativo()

    console.print("\n✅ Módulo 2.2 concluído!", style="bold green")
    console.print(
        "\n💡 PRÓXIMO PASSO: Adicionar FERRAMENTAS ao agente (Módulo 3)",
        style="yellow",
    )
