"""
============================================================
MÓDULO 1.2 - BOAS PRÁTICAS DE PROMPTS
============================================================
Técnicas comprovadas para escrever prompts melhores.
Comparamos prompts RUINS vs BONS para cada prática.

POR QUE BOAS PRÁTICAS?
A diferença entre um prompt bom e um ruim pode ser:
- Resposta genérica vs. resposta exata
- JSON inválido vs. JSON perfeito
- Alucinação vs. "não sei"

METODOLOGIA DESTE MÓDULO:
Para cada prática, mostramos um prompt RUIM e um BOM,
e executamos ambos para ver a diferença na prática.
Isso é o método mais eficaz de ensinar prompt engineering.

PRÁTICAS COBERTAS:
1. Seja específico (evite prompts vagos)
2. Use delimitadores (separe dados de instruções)
3. Peça formato de saída (JSON, Markdown, etc.)
4. Saída de emergência (evite alucinações)
5. Instruções em passos (decomponha tarefas complexas)
6. Padrão ReAct (para agentes com tools)
============================================================
"""

import os
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console  # Rich: output colorido no terminal
from rich.table import Table  # Rich: tabelas formatadas

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
console = Console()


def chamar_llm(mensagens: list[dict], temperature: float = 0.3) -> str:
    """Chama a LLM e retorna o texto da resposta."""
    resposta = client.chat.completions.create(
        model=MODEL,
        messages=mensagens,
        temperature=temperature,
        max_tokens=1024,
    )
    return resposta.choices[0].message.content


def comparar_prompts(titulo: str, prompt_ruim: str, prompt_bom: str):
    """
    Executa e compara um prompt ruim vs um bom lado a lado.

    Essa é a técnica didática central deste módulo:
    ao ver as duas respostas, o aluno entende visceralmente
    a diferença que um prompt bem escrito faz.
    """
    console.print(f"\n{'='*60}", style="bold")
    console.print(f"📌 {titulo}", style="bold yellow")
    console.print(f"{'='*60}", style="bold")

    # Prompt ruim
    console.print("\n❌ PROMPT RUIM:", style="bold red")
    console.print(f"   {prompt_ruim}", style="dim")
    resp_ruim = chamar_llm([{"role": "user", "content": prompt_ruim}])
    if len(resp_ruim) > 200:
        console.print(f"   Resposta: {resp_ruim[:200]}...\n")
    else:
        console.print(f"   Resposta: {resp_ruim}\n")

    # Prompt bom
    console.print("✅ PROMPT BOM:", style="bold green")
    console.print(f"   {prompt_bom}", style="dim")
    resp_bom = chamar_llm([{"role": "user", "content": prompt_bom}])
    if len(resp_bom) > 200:
        console.print(f"   Resposta: {resp_bom[:200]}...\n")
    else:
        console.print(f"   Resposta: {resp_bom}\n")

    return resp_ruim, resp_bom


# ============================================================
# PRÁTICA 1: Seja Específico
# ============================================================
# A prática MAIS IMPORTANTE. Prompts vagos geram respostas vagas.
#
# REGRA DE OURO: Sempre defina:
#   - O QUE você quer (tarefa)
#   - COMO você quer (formato, estilo)
#   - PARA QUEM (contexto do público)
#
# RUIM: "Me fale sobre boletos" → resposta genérica de 500 palavras
# BOM:  "Explique em 3 bullets..." → resposta focada e útil
def pratica_1_seja_especifico():
    """
    TEORIA:
    - Prompts vagos geram respostas vagas
    - Quanto mais contexto, melhor a resposta
    - Defina: O QUE quer, COMO quer, PARA QUEM
    """
    comparar_prompts(
        titulo="PRÁTICA 1: Seja Específico",
        prompt_ruim="Me fale sobre boletos.",
        prompt_bom=(
            "Explique em 3 bullets o que é um boleto bancário brasileiro, "
            "focando nos campos principais (beneficiário, valor, "
            "vencimento e linha digitável). "
            "Use linguagem simples para alguém que nunca viu um boleto."
        )
    )


# ============================================================
# PRÁTICA 2: Use Delimitadores
# ============================================================
# Delimitadores separam INSTRUÇÕES dos DADOS no prompt.
# Sem delimitadores, a LLM confunde o que é instrução e o que é dado.
#
# BENEFÍCIOS:
# - Clareza: a LLM sabe onde começam e terminam os dados
# - Segurança: previne injeção de prompt (dados viram "dados", não instruções)
# - Reprodutibilidade: facilita automatizar com templates
#
# DELIMITADORES COMUNS:
#   ``` (code blocks)   --- (linhas)   <xml> (tags)
#   ### (seções)       """ (aspas)    {{}} (templates)
def pratica_2_use_delimitadores():
    """
    TEORIA:
    - Delimitadores separam instrução dos dados
    - Previne injeção de prompt (o dado vira "dado", não "instrução")
    - Use: ```, ---, <>, ###, ou XML tags
    """
    comparar_prompts(
        titulo="PRÁTICA 2: Use Delimitadores",
        prompt_ruim=(
            "Extraia o valor e vencimento do boleto Banco Itaú beneficiário "
            "ACME Ltda valor R$ 500,00 vencimento 20/03/2026"
        ),
        prompt_bom=(
            "Extraia o valor e vencimento do boleto abaixo.\n\n"
            "Responda no formato:\n"
            "- Valor: R$ XX,XX\n"
            "- Vencimento: DD/MM/AAAA\n\n"
            "---INÍCIO DO BOLETO---\n"
            "Banco: Itaú\n"
            "Beneficiário: ACME Ltda\n"
            "Valor: R$ 500,00\n"
            "Vencimento: 20/03/2026\n"
            "---FIM DO BOLETO---"
        )
    )


# ============================================================
# PRÁTICA 3: Peça Formato de Saída
# ============================================================
# SEMPRE diga à LLM o FORMATO da resposta que você espera.
# Sem formato definido, cada resposta vem diferente.
#
# FORMATOS COMUNS:
# - JSON: para agentes e automação (fácil de parsear com json.loads)
# - Markdown: para usuários humanos (legível e bonito)
# - Bullets: para listas e resumos
# - Tabela: para comparações
#
# DICA: Forneça o TEMPLATE/SCHEMA da saída no prompt.
# Ex: {"banco": "...", "valor": 0.00} → a LLM segue o formato.
def pratica_3_formato_de_saida():
    """
    TEORIA:
    - Sempre especifique o formato esperado
    - Para agentes: JSON é melhor (fácil de parsear)
    - Para humanos: Markdown, bullets, tabelas
    - Ajuda a LLM a ser concisa e focada
    """
    comparar_prompts(
        titulo="PRÁTICA 3: Especifique o Formato de Saída",
        prompt_ruim=(
            "Analise esse boleto: Valor R$ 1000, vencido há 10 dias, "
            "juros 1% ao mês, multa 2%."
        ),
        prompt_bom=(
            "Analise o boleto abaixo e retorne APENAS um JSON "
            "no formato especificado.\n\n"
            "<boleto>\n"
            "Valor original: R$ 1.000,00\n"
            "Dias em atraso: 10\n"
            "Juros: 1% ao mês (pro-rata)\n"
            "Multa: 2% sobre valor original\n"
            "</boleto>\n\n"
            "Formato de resposta:\n"
            "```json\n"
            '{"valor_original": 0.00, "multa": 0.00, '
            '"juros": 0.00, "total": 0.00}\n'
            "```"
        )
    )


# ============================================================
# PRÁTICA 4: Dê ao Modelo uma "Saída de Emergência"
# ============================================================
# Permita que a LLM diga "não sei" ou "dados insuficientes".
# Sem isso, a LLM INVENTA informações (alucinação).
#
# O QUE É ALUCINAÇÃO?
# Quando a LLM gera informação que PARECE real mas é INVENTADA.
# Em agentes financeiros (boletos), isso é CRÍTICO:
# um valor errado pode causar prejuízo real.
#
# COMO EVITAR:
# - Diga: "Se a informação não estiver disponível, retorne {erro: ...}"
# - A LLM prefere inventar a admitir ignorância — precisamos explicitá-lo
# - Combine com guardrails de saída (módulo 4) para dupla proteção
def pratica_4_saida_de_emergencia():
    """
    TEORIA:
    - Permita que a LLM diga "não sei" ou "dados insuficientes"
    - Evita alucinações (inventar dados)
    - Importante para agentes que tomam decisões
    """
    comparar_prompts(
        titulo='PRÁTICA 4: "Saída de Emergência" (evitar alucinação)',
        prompt_ruim=(
            "Qual o CNPJ do beneficiário deste boleto: "
            "Valor R$ 200,00, Vencimento 10/04/2026"
        ),
        prompt_bom=(
            "Qual o CNPJ do beneficiário deste boleto?\n\n"
            "<boleto>\n"
            "Valor: R$ 200,00\n"
            "Vencimento: 10/04/2026\n"
            "</boleto>\n\n"
            "Se a informação solicitada NÃO estiver presente no boleto, "
            'responda: {"erro": "informação não disponível", '
            '"campo": "nome_do_campo"}'
        )
    )


# ============================================================
# PRÁTICA 5: Instruções em Passos
# ============================================================
# Tarefas complexas devem ser decompostas em passos numerados.
# Cada passo = uma ação clara e verificar.
#
# POR QUE FUNCIONA:
# - A LLM segue os passos na ORDEM (não pula etapas)
# - Cada passo produz resultado intermediário (mais fácil de debugar)
# - Combina com Chain-of-Thought: os passos são o "raciocínio guiado"
#
# DICA PARA AGENTES:
# - Seus tools/funções devem corresponder aos passos
#   Ex: Passo 1 → extrair_dados(), Passo 2 → verificar_vencimento(), etc.
def pratica_5_instrucoes_em_passos():
    """
    TEORIA:
    - Tarefas complexas devem ser divididas em passos
    - Numere os passos para clareza
    - Cada passo = uma ação clara
    - Ajuda a LLM a manter o foco
    """
    comparar_prompts(
        titulo="PRÁTICA 5: Divida em Passos Numerados",
        prompt_ruim="Processe esse boleto e me diga se está ok para pagar.",
        prompt_bom=(
            "Analise o boleto seguindo EXATAMENTE estes passos:\n\n"
            "Passo 1: Identifique o banco emissor pelo código "
            "(341=Itaú, 033=Santander, 001=BB)\n"
            "Passo 2: Verifique se o boleto está vencido "
            "(data atual: 16/03/2026)\n"
            "Passo 3: Se vencido, calcule multa (2%) + juros (0,033%/dia)\n"
            "Passo 4: Dê o veredito FINAL: PAGAR (se valor < R$ 5000) "
            "ou SOLICITAR APROVAÇÃO\n\n"
            "<boleto>\n"
            "Banco: 341\n"
            "Valor: R$ 850,00\n"
            "Vencimento: 10/03/2026\n"
            "</boleto>"
        )
    )


# ============================================================
# PRÁTICA 6: Técnica de Prompt para Agentes - ReAct
# ============================================================
# O padrão ReAct (Reasoning + Acting) é o padrão mais usado em agentes.
# Aqui mostramos a VERSÃO DE PROMPT do ReAct (sem tools reais).
# No módulo 3, veremos a implementação com tool calling real.
#
# DIFERENÇA ENTRE ReAct-PROMPT e TOOL CALLING:
# - ReAct-prompt: a LLM "faz de conta" que executa ações (tudo é texto)
# - Tool calling: a LLM REALMENTE chama funções Python (módulo 3)
def pratica_6_react_prompt():
    """
    TEORIA - PADRÃO ReAct (Reasoning + Acting):
    - O agente PENSA, depois AGE, depois OBSERVA
    - Formato: Thought → Action → Observation → ... → Final Answer
    - Fundamental para agentes com ferramentas
    - Veremos a implementação completa no Módulo 3
    """
    console.print(f"\n{'='*60}", style="bold")
    console.print(
        "📌 PRÁTICA 6: Padrão ReAct para Agentes",
        style="bold yellow",
    )
    console.print(f"{'='*60}", style="bold")

    mensagens = [
        {
            "role": "system",
            "content": (
                "Você é um agente que processa boletos. "
                "Use o formato ReAct para raciocinar:\n\n"
                "Thought: [seu raciocínio sobre o que fazer]\n"
                "Action: [ação a tomar - ex: extrair_dados, "
                "calcular_multa, validar_codigo]\n"
                "Observation: [resultado da ação]\n"
                "... (repita quantas vezes necessário)\n"
                "Final Answer: [resposta final ao usuário]\n\n"
                "Ferramentas disponíveis:\n"
                "- extrair_dados: Extrai campos do texto do boleto\n"
                "- validar_vencimento: Verifica se o boleto está vencido\n"
                "- calcular_encargos: Calcula multa e juros"
            )
        },
        {
            "role": "user",
            "content": (
                "Analise este boleto:\n"
                "Banco Itaú (341), Valor R$ 2.300,00, Vencimento: 01/03/2026\n"
                "Preciso saber se está vencido e quanto pagar."
            )
        }
    ]

    resposta = chamar_llm(mensagens)
    console.print(f"\nResposta ReAct:\n{resposta}", style="cyan")
    return resposta


# ============================================================
# RESUMO VISUAL DAS BOAS PRÁTICAS
# ============================================================
# Tabela consolidada para consulta rápida durante o treinamento.
# Use como "cheat sheet" ao construir seus próprios prompts.
def mostrar_resumo():
    """
    Mostra uma tabela resumo das boas práticas de prompts.

    Essa tabela é um recurso visual para reforçar as práticas ensinadas.
    Use-a como referência rápida ao criar seus próprios prompts.
    """
    table = Table(title="📋 Resumo: Boas Práticas de Prompts")
    table.add_column("Prática", style="bold")
    table.add_column("Quando Usar", style="cyan")
    table.add_column("Dica", style="green")

    table.add_row(
        "Seja Específico", "Sempre", "Defina O QUE, COMO e PARA QUEM",
    )
    table.add_row(
        "Use Delimitadores",
        "Dados externos no prompt",
        "```, ---,  <xml>, ###",
    )
    table.add_row(
        "Formato de Saída",
        "Agentes / automação",
        "JSON para agentes, MD para humanos",
    )
    table.add_row(
        "Saída de Emergência",
        "Dados podem estar ausentes",
        'Permita "não sei"',
    )
    table.add_row(
        "Passos Numerados", "Tarefas complexas", "1 passo = 1 ação clara",
    )
    table.add_row(
        "ReAct Pattern",
        "Agentes com tools",
        "Think → Act → Observe → Answer",
    )

    console.print(table)


if __name__ == "__main__":
    console.print("🎓 MÓDULO 1.2 - BOAS PRÁTICAS DE PROMPTS", style="bold blue")
    console.print("=" * 60)

    # Descomente os exemplos que quiser executar:
    pratica_1_seja_especifico()
    pratica_2_use_delimitadores()
    pratica_3_formato_de_saida()
    pratica_4_saida_de_emergencia()
    pratica_5_instrucoes_em_passos()
    pratica_6_react_prompt()

    mostrar_resumo()

    console.print("\n✅ Módulo 1.2 concluído!", style="bold green")
