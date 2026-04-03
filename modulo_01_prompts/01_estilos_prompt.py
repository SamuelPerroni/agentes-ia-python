"""
============================================================
MÓDULO 1.1 - ESTILOS DE PROMPT
============================================================
Neste módulo, vamos explorar os diferentes estilos de prompt
e entender quando usar cada um.

O QUE É UM PROMPT?
Prompt é a instrução/mensagem que enviamos à LLM (Large Language Model).
A qualidade da resposta depende DIRETAMENTE da qualidade do prompt.
Dominar estilos de prompt é a PRIMEIRA habilidade para construir agentes.

POR QUE ESTILOS DIFERENTES?
Cada estilo resolve um tipo de problema:
- Problemas simples → Zero-shot (direto ao ponto)
- Problemas que precisam de consistência → Few-shot (com exemplos)
- Problemas que exigem raciocínio → Chain-of-Thought (passo a passo)
- Agentes especializados → Role-playing (persona)
- Automação/integração → Structured Output (JSON)
- Agentes com ferramentas → ReAct (pensar + agir)

Estilos cobertos:
1. Zero-shot   - Sem exemplos, instrução direta
2. Few-shot    - Com exemplos demonstrativos
3. Chain-of-Thought (CoT) - Raciocínio passo a passo
4. Role-playing - Atribuição de papel/persona via system prompt
5. Structured Output - Saída estruturada (JSON) para automação
6. ReAct       - Reasoning + Acting (padrão #1 para agentes com tools)
============================================================
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

# Carrega variáveis do arquivo .env (GROQ_API_KEY, GROQ_MODEL)
load_dotenv()

# Inicializa o cliente Groq — a API gratuita que usaremos no treinamento
# Groq oferece modelos de alta performance (Llama 3.3 70B) sem custo
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def chamar_llm(mensagens: list[dict], temperature: float = 0.3) -> str:
    """
    Função auxiliar para chamar a LLM.

    Parâmetros:
    - mensagens: lista de dicts no formato OpenAI
      ({"role": ..., "content": ...})
      Roles: "system" (instruções), "user" (pergunta), "assistant" (resposta)
    - temperature: controla criatividade (0.0 = determinístico, 1.0 = criativo)
      Para dados/cálculos use 0.1-0.3, para texto criativo use 0.7-1.0
    """
    resposta = client.chat.completions.create(
        model=MODEL,
        messages=mensagens,
        temperature=temperature,
        max_tokens=1024,  # Limite de tokens na resposta (não no prompt)
    )
    return resposta.choices[0].message.content


# ============================================================
# 1. ZERO-SHOT - Sem nenhum exemplo
# ============================================================
# A LLM recebe apenas a instrução, sem nenhum exemplo.
# É o estilo mais simples: "faça X" — e a LLM tenta fazer.
#
# QUANDO USAR:
# - Tarefas genéricas que a LLM já conhece (classificação, resumo, tradução)
# - Prototipagem rápida (testar se a LLM entende o que você quer)
# - Quando não temos exemplos disponíveis
#
# LIMITAÇÕES:
# - Para tarefas específicas do seu domínio, pode errar o formato/critério
# - A LLM "advinha" o que você espera — pode não ser o que você quer

def exemplo_zero_shot():
    """
    ZERO-SHOT: Pedimos à LLM para classificar sem dar exemplos.

    TEORIA:
    - Funciona bem para tarefas que a LLM já conhece
    - Mais simples, porém menos preciso para tarefas específicas
    - Ideal para prototipagem rápida
    """
    print("\n" + "="*50)
    print("1. ZERO-SHOT")
    print("="*50)

    mensagens = [
        {
            "role": "user",
            "content": (
                "Classifique o seguinte texto como "
                "POSITIVO, NEGATIVO ou NEUTRO:\n\n"
                "Texto: 'O boleto venceu ontem e agora vou ter "
                "que pagar multa, que absurdo!'"
            )
        }
    ]

    resposta = chamar_llm(mensagens)
    print("Prompt: Classificar sentimento sem exemplos")
    print(f"Resposta: {resposta}")
    return resposta


# ============================================================
# 2. FEW-SHOT - Com exemplos demonstrativos
# ============================================================
# Fornecemos exemplos ("shots") dentro do próprio prompt.
# A LLM aprende o PADRÃO dos exemplos e aplica ao caso novo.
#
# QUANDO USAR:
# - Quando precisa de formato consistente (sempre igual)
# - Classificações com critérios específicos do seu domínio
# - Quando zero-shot não dá o resultado esperado
#
# DICAS:
# - 2 a 5 exemplos geralmente bastam (mais que 5 raramente ajuda)
# - Cubra casos variados nos exemplos (positivo, negativo, neutro)
# - Mantenha o formato IDÊNTICO entre os exemplos
# - Os exemplos ensinam tanto o FORMATO quanto os CRITÉRIOS

def exemplo_few_shot():
    """
    FEW-SHOT: Damos exemplos para a LLM seguir o padrão.

    TEORIA:
    - Exemplos ensinam formato e critérios esperados
    - 2-5 exemplos geralmente são suficientes
    - Exemplos devem cobrir casos variados
    """
    print("\n" + "="*50)
    print("2. FEW-SHOT")
    print("="*50)

    mensagens = [
        {
            "role": "user",
            "content": (
                "Classifique os boletos pelo status. "
                "Responda APENAS com o status.\n\n"
                "Exemplo 1:\n"
                "Boleto: Vencimento 15/01/2026, Valor R$ 150,00, "
                "Data atual 10/01/2026\n"
                "Status: DENTRO DO PRAZO\n\n"
                "Exemplo 2:\n"
                "Boleto: Vencimento 01/01/2026, Valor R$ 200,00, "
                "Data atual 10/01/2026\n"
                "Status: VENCIDO\n\n"
                "Exemplo 3:\n"
                "Boleto: Vencimento 10/01/2026, Valor R$ 300,00, "
                "Data atual 10/01/2026\n"
                "Status: VENCE HOJE\n\n"
                "Agora classifique:\n"
                "Boleto: Vencimento 20/03/2026, Valor R$ 450,00, "
                "Data atual 16/03/2026\n"
                "Status:"
            )
        }
    ]

    resposta = chamar_llm(mensagens)
    print("Prompt: Classificar boleto com exemplos")
    print(f"Resposta: {resposta}")
    return resposta


# ============================================================
# 3. CHAIN-OF-THOUGHT (CoT) - Raciocínio passo a passo
# ============================================================
# Pedimos para a LLM "pensar em voz alta" antes de responder.
# A LLM decompõe o problema em etapas intermediárias.
#
# POR QUE FUNCIONA?
# - LLMs são autorregressivas: cada token depende dos anteriores
# - Ao gerar passos intermediários, a LLM "se dá contexto" para acertar
# - Sem CoT, a LLM tenta ir direto à resposta e pode errar
#
# QUANDO USAR:
# - Cálculos matemáticos (ex: multa + juros de boleto)
# - Lógica e raciocínio (ex: "o boleto está vencido? se sim, quanto pagar?")
# - Comparações complexas
#
# COMO ATIVAR:
# - Simples: adicione "Pense passo a passo" no prompt
# - Detalhado: numere os passos que a LLM deve seguir

def exemplo_chain_of_thought():
    """
    CHAIN-OF-THOUGHT: LLM raciocina passo a passo.

    TEORIA:
    - Melhora drasticamente tarefas de raciocínio
    - A LLM decompõe o problema em etapas
    - Adicionar "Pense passo a passo" já ajuda
    - Ideal para cálculos, lógica e análise
    """
    print("\n" + "="*50)
    print("3. CHAIN-OF-THOUGHT")
    print("="*50)

    mensagens = [
        {
            "role": "user",
            "content": (
                "Um boleto tem valor original de R$ 1.500,00, "
                "venceu há 15 dias.\n"
                "A multa é de 2% sobre o valor original e os "
                "juros são de 0,033% ao dia.\n"
                "Qual o valor total a pagar?\n\n"
                "Pense passo a passo antes de dar a resposta final."
            )
        }
    ]

    resposta = chamar_llm(mensagens)
    print("Prompt: Calcular multa de boleto (com raciocínio)")
    print(f"Resposta: {resposta}")
    return resposta


# ============================================================
# 4. ROLE-PLAYING - Atribuição de Persona/Papel
# ============================================================
# Definimos um papel/persona para a LLM através do system prompt.
# É assim que criamos agentes ESPECIALIZADOS.
#
# COMO FUNCIONA:
# - O "system prompt" é a primeira mensagem da conversa (role: "system")
# - Define QUEM a LLM é, o que faz, e o que NÃO faz
# - Muda o vocabulário, tom, profundidade e foco das respostas
#
# PARA AGENTES, SEMPRE DEFINA:
# - Quem é o agente ("Você é um assistente de boletos...")
# - O que pode fazer (capacidades)
# - O que NÃO pode fazer (limites — evita alucinações)
# - Regras de comportamento (formato, idioma, etc.)
#
# DICA: O system prompt é a peça MAIS IMPORTANTE de um agente

def exemplo_role_playing():
    """
    ROLE-PLAYING: Definimos uma persona via system prompt.

    TEORIA:
    - System prompt define o comportamento base
    - Persona influencia tom, vocabulário e foco
    - Sempre defina: quem é, o que faz, o que NÃO faz
    - Fundamental para agentes especializados
    """
    print("\n" + "="*50)
    print("4. ROLE-PLAYING (System Prompt)")
    print("="*50)

    mensagens = [
        {
            "role": "system",
            "content": (
                "Você é um assistente financeiro especializado "
                "em boletos bancários brasileiros.\n\n"
                "SUAS RESPONSABILIDADES:\n"
                "- Ajudar usuários a entender informações de boletos\n"
                "- Calcular multas e juros por atraso\n"
                "- Explicar campos do boleto "
                "(linha digitável, código de barras, etc.)\n\n"
                "REGRAS:\n"
                "- Sempre responda em português brasileiro\n"
                "- Use linguagem clara e acessível\n"
                "- Quando houver cálculos, mostre o passo a passo\n"
                "- NUNCA forneça conselhos de investimento\n"
                "- NUNCA processe pagamentos diretamente"

            )
        },
        {
            "role": "user",
            "content": "O que é a linha digitável de um boleto?"
        }
    ]

    resposta = chamar_llm(mensagens)
    print("Prompt: Pergunta com persona de especialista")
    print(f"Resposta: {resposta}")
    return resposta


# ============================================================
# 5. STRUCTURED OUTPUT - Saída em formato JSON
# ============================================================
# Pedimos para a LLM retornar dados em formato estruturado (JSON, XML, CSV...).
# ESSENCIAL para agentes, pois precisamos parsear a resposta no código.
#
# POR QUE É IMPORTANTE PARA AGENTES?
# - Agentes precisam PROCESSAR a resposta da LLM (não apenas exibir)
# - JSON é fácil de parsear com json.loads() em Python
# - Com Pydantic (módulo 4), podemos validar o JSON automaticamente
#
# TÉCNICAS PARA FORÇAR JSON:
# 1. Dizer "Responda APENAS com JSON" no prompt
# 2. Fornecer o schema/template esperado
# 3. Usar response_format={"type": "json_object"} (quando a API suporta)
# 4. Validar com Pydantic e fazer retry se falhar (módulo 4)

def exemplo_structured_output():
    """
    STRUCTURED OUTPUT: Forçamos saída em JSON.

    TEORIA:
    - Agentes precisam parsear respostas da LLM
    - JSON é o formato mais comum
    - Sempre forneça o schema esperado
    - Use "response_format" quando disponível
    - Valide a saída com Pydantic (veremos no módulo 4)
    """
    print("\n" + "="*50)
    print("5. STRUCTURED OUTPUT (JSON)")
    print("="*50)

    mensagens = [
        {
            "role": "system",
            "content": (
                "Você extrai dados de boletos. Responda APENAS "
                "com JSON válido, sem texto adicional."
            )
        },
        {
            "role": "user",
            "content": """Extraia as informações do seguinte boleto:

Banco: Itaú (341)
Beneficiário: Empresa ABC Ltda
CNPJ: 12.345.678/0001-90
Valor: R$ 1.250,00
Vencimento: 25/03/2026
Linha digitável: 34191.09065 44830.136706 00000.000178 1 70060000125000

Retorne no formato JSON:
{
    "banco": "nome (código)",
    "beneficiario": "nome",
    "cnpj": "número",
    "valor": numero_decimal,
    "vencimento": "YYYY-MM-DD",
    "linha_digitavel": "número",
    "status": "DENTRO DO PRAZO ou VENCIDO"
}"""
        }
    ]

    resposta = chamar_llm(mensagens, temperature=0.1)
    print("Prompt: Extrair dados de boleto em JSON")
    print(f"Resposta: {resposta}")

    # Tentar parsear o JSON — isto é o que um agente faria
    # É normal a LLM retornar JSON envolto em ```json...``` — limpar
    try:
        json_str = resposta.strip()
        # Remover marcadores de código (```json ... ``` ou ``` ... ```)
        if json_str.startswith("```"):
            json_str = json_str.split("\n", 1)[1]
            json_str = json_str.rsplit("```", 1)[0]

        # json.loads() converte a string JSON em um dicionário Python
        dados = json.loads(json_str)
        print("\n✅ JSON parseado com sucesso!")
        print(f"   Banco: {dados.get('banco')}")
        print(f"   Valor: {dados.get('valor')}")
        print(f"   Status: {dados.get('status')}")
    except json.JSONDecodeError as e:
        # Se a LLM não retornou JSON válido — módulo 4 mostra como fazer retry
        print(f"\n❌ Erro ao parsear JSON: {e}")

    return resposta


# ============================================================
# 6. ReAct - REASONING + ACTING
# ============================================================
# Padrão FUNDAMENTAL para agentes que usam ferramentas (tools).
# A LLM alterna entre PENSAR → AGIR → OBSERVAR → repetir.
#
# O CICLO ReAct:
#   Thought:     "Preciso verificar se o boleto está vencido"
#   Action:      verificar_vencimento("01/03/2026")
#   Observation: {"vencido": true, "dias_atraso": 15}
#   Thought:     "Está vencido, preciso calcular multa e juros"
#   Action:      calcular_multa(1500.00, 15)
#   Observation: {"total": 1537.43}
#   Final Answer: "O boleto está vencido. O valor atualizado é R$ 1.537,43"
#
# POR QUE É O PADRÃO #1 PARA AGENTES:
# - Combina raciocínio (CoT) com capacidade de agir (tools)
# - O "Thought" evita que a LLM tome ações precipitadas
# - O "Observation" traz dados REAIS (não alucinados)
# - No módulo 3 implementaremos isso com tool calling real

def exemplo_react():
    """
    ReAct: A LLM raciocina, decide uma ação, observa o resultado, e repete.

    TEORIA:
    - Padrão mais usado em agentes com tools
    - Ciclo: Thought → Action → Observation → ... → Final Answer
    - "Thought" evita ações precipitadas (a LLM pensa antes)
    - "Observation" traz dados reais (resultado de ferramentas)
    - Combina Chain-of-Thought com capacidade de agir
    """
    print("\n" + "="*50)
    print("6. ReAct (REASONING + ACTING)")
    print("="*50)

    mensagens = [
        {
            "role": "system",
            "content": (
                "Você é um agente que processa boletos. "
                "Para cada solicitação, use o formato ReAct:\n\n"
                "Thought: [seu raciocínio sobre o que precisa fazer]\n"
                "Action: [ação a tomar - ex: extrair_dados, "
                "calcular_multa, validar_codigo]\n"
                "Observation: [resultado da ação]\n"
                "... (repita Thought/Action/Observation quantas "
                "vezes necessário)\n"
                "Final Answer: [resposta final ao usuário]\n\n"
                "Ferramentas disponíveis:\n"
                "- extrair_dados: Extrai campos do texto do boleto\n"
                "- validar_vencimento: Verifica se o boleto está "
                "vencido (data atual: 16/03/2026)\n"
                "- calcular_encargos: Calcula multa (2%) e "
                "juros (0,033%/dia)"
            )
        },
        {
            "role": "user",
            "content": (
                "Analise este boleto e me diga quanto pagar:\n"
                "Banco Itaú (341), Valor R$ 2.300,00, Vencimento: 01/03/2026"
            )
        }
    ]

    resposta = chamar_llm(mensagens)
    print("Prompt: Processar boleto com padrão ReAct")
    print(f"Resposta:\n{resposta}")
    return resposta


# ============================================================
# EXECUÇÃO DOS EXEMPLOS
# ============================================================
if __name__ == "__main__":
    print("🎓 MÓDULO 1.1 - ESTILOS DE PROMPT")
    print("=" * 60)

    # Descomente os exemplos que quiser executar:
    exemplo_zero_shot()
    exemplo_few_shot()
    exemplo_chain_of_thought()
    exemplo_role_playing()
    exemplo_structured_output()
    exemplo_react()

    print("\n" + "=" * 60)
    print("✅ Módulo 1.1 concluído!")
    print("\n💡 RESUMO DOS ESTILOS:")
    print("   Zero-shot    → Rápido, sem exemplos")
    print("   Few-shot     → Com exemplos, mais consistente")
    print("   CoT          → Raciocínio passo a passo")
    print("   Role-playing → Persona via system prompt")
    print("   Structured   → Saída em JSON/formato definido")
    print("   ReAct        → Think → Act → Observe (agentes com tools)")
