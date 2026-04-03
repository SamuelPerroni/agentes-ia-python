"""
============================================================
MÓDULO 10.1 - MEMÓRIA DE LONGO PRAZO E RAG SIMPLES
============================================================
Neste módulo, vamos além da memória de curto prazo (últimas mensagens)
e implementamos uma memória consultável de políticas e fatos.

CONCEITO CHAVE — Os Dois Tipos de Memória:
┌─────────────────────────────────────────────────────────┐
│                MEMÓRIA DO AGENTE                         │
├────────────────────────┬────────────────────────────────┤
│   CURTO PRAZO          │   LONGO PRAZO (este módulo)    │
│                        │                                │
│ - Últimas N mensagens  │ - Base de conhecimento         │
│ - Perdida ao reiniciar │ - Persistente (arquivo/banco)  │
│ - Já implementada no   │ - Consultada por relevância    │
│   módulo 02            │ - Injetada no prompt da LLM    │
│ - Limitada pelo        │ - Escalável (milhares de docs) │
│   contexto da LLM      │                                │
└────────────────────────┴────────────────────────────────┘

O QUE É RAG (Retrieval-Augmented Generation)?
RAG é o padrão onde, antes de chamar a LLM, você BUSCA informações
relevantes em uma base de dados e INJETA no prompt. Isso permite que
a LLM responda com base em conhecimento que ela NÃO TEM no treino.

  Pergunta do usuário
       ↓
  RETRIEVAL: buscar documentos relevantes na base
       ↓
  AUGMENTATION: adicionar esses documentos ao prompt
       ↓
  GENERATION: LLM gera resposta usando a pergunta + documentos

POR QUE RAG EM VEZ DE COLOCAR TUDO NO SYSTEM PROMPT?
- System prompts têm LIMITE DE TOKENS (4K-8K tipicamente)
- A empresa pode ter centenas de políticas — não cabem no prompt
- RAG busca só o RELEVANTE, economizando tokens e melhorando foco
- Atualizar a base é mais fácil que reescrever prompts

ANALOGIA:
É como um advogado consultando jurisprudência:
- Ele não memoriza TODAS as leis (system prompt limitado)
- Quando recebe um caso, PESQUISA as leis relevantes (retrieval)
- Monta a argumentação com base nas leis encontradas (generation)

NESTE MÓDULO:
Usamos busca por palavras-chave (a versão mais simples de RAG).
Em produção, a evolução é usar embeddings e busca vetorial.

Tópicos cobertos:
1. Carregamento da base de conhecimento (JSON de políticas)
2. Busca por relevância (keyword matching com scoring)
3. Exibição dos resultados em tabela Rich
4. Caminho de evolução para RAG com embeddings
============================================================
"""

import json
import os

from rich.console import Console
from rich.table import Table

# Console do Rich para output formatado com cores e tabelas
console = Console()


# ============================================================
# 1. CARREGAMENTO DA BASE DE CONHECIMENTO
# ============================================================
# A base é o arquivo politicas_cobranca.json que contém as regras
# de negócio da empresa para cobrança de boletos.
#
# ESTRUTURA:
# [
#   {"id": "POL-001", "titulo": "...", "tags": [...], "conteudo": "..."},
#   {"id": "POL-002", ...},
#   ...
# ]
#
# Tags facilitam a busca — são palavras-chave que descrevem o tema.
# Em produção, as tags seriam substituídas por embeddings vetoriais.
# ============================================================

def carregar_base_conhecimento() -> list[dict]:
    """
    Carrega a base de conhecimento (políticas de cobrança) do arquivo JSON.

    COMO FUNCIONA:
    - Encontra o arquivo politicas_cobranca.json no diretório deste módulo
    - Lê e converte o JSON em uma lista de dicionários Python
    - Cada dicionário representa uma política/regra de negócio

    RETORNO:
    Lista de dicts com campos: id, titulo, tags, conteudo
    """
    caminho = os.path.join(
        os.path.dirname(__file__),
        "politicas_cobranca.json"
    )
    with open(caminho, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


# ============================================================
# 2. BUSCA POR RELEVÂNCIA — Recuperação de Memórias
# ============================================================
# Dado uma pergunta, buscamos os documentos mais relevantes.
#
# ALGORITMO SIMPLIFICADO:
# 1. Quebrar a pergunta em palavras
# 2. Para cada documento, contar quantas palavras da pergunta
#    aparecem no texto do documento (tags + titulo + conteudo)
# 3. Ordenar por contagem (score) e retornar os top_k
#
# LIMITAÇÕES desta abordagem:
# - "boleto vencido" e "boleto em atraso" = mesma intenção,
#   mas "atraso" e "vencido" são palavras diferentes
# - Não pondera importância (a palavra "de" vale tanto quanto "multa")
#
# EVOLUÇÃO PARA PRODUÇÃO:
# Usar embeddings (vetores numéricos que capturam SIGNIFICADO):
# - "vencido" e "atraso" teriam vetores próximos no espaço vetorial
# - Bibliotecas: sentence-transformers, OpenAI Embeddings
# - Bancos vetoriais: Pinecone, Weaviate, Chroma, pgvector
# ============================================================

def recuperar_memorias(
        pergunta: str,
        base: list[dict],
        top_k: int = 2
) -> list[dict]:
    """
    Busca documentos relevantes na base de conhecimento para uma pergunta.

    Parâmetros:
    - pergunta: texto do usuário (ex: "boleto em atraso com multa")
    - base: lista de documentos da base de conhecimento
    - top_k: quantos documentos retornar (default: 2)

    COMO FUNCIONA:
    1. Cria um set de termos da pergunta (lowercase, sem duplicatas)
    2. Para cada documento, faz score = soma de termos encontrados
    3. Ordena por score decrescente
    4. Retorna os top_k com score > 0 (ignora irrelevantes)

    RETORNO:
    Lista de até top_k dicts, ordenados por relevância.
    """
    # Set de palavras da pergunta — set() elimina duplicatas
    termos = set(palavra.lower() for palavra in pergunta.split())

    pontuados = []
    for documento in base:
        score = 0
        # Juntamos tags + titulo + conteudo num texto único para busca
        texto_busca = " ".join(
            documento["tags"] + [documento["titulo"], documento["conteudo"]]
        ).lower()
        # Cada termo da pergunta encontrado no documento soma 1 ponto
        for termo in termos:
            if termo in texto_busca:
                score += 1
        pontuados.append((score, documento))

    # Ordena do mais relevante (maior score) para o menos relevante
    pontuados.sort(key=lambda item: item[0], reverse=True)

    # Filtra documentos irrelevantes (score == 0) e limita ao top_k
    return [documento for score, documento in pontuados if score > 0][:top_k]


# ============================================================
# 3. DEMONSTRAÇÃO — Busca na Base de Conhecimento
# ============================================================
# Simulamos uma pergunta do usuário e mostramos quais políticas
# o agente recuperaria da memória de longo prazo.
#
# FLUXO:
# 1. Pergunta do "usuário" (hardcoded para demo)
# 2. Carrega a base de conhecimento (3 políticas)
# 3. Busca as mais relevantes (keyword matching)
# 4. Exibe em tabela Rich
#
# NA PRÁTICA (dentro do agente):
# Essas memórias recuperadas seriam INJETADAS no prompt da LLM
# como contexto adicional antes de gerar a resposta.
#
# EXEMPLO DE PROMPT COM MEMÓRIA INJETADA:
# "Você é um assistente de boletos.
#  CONTEXTO DA BASE DE CONHECIMENTO:
#  - POL-001: Boletos vencidos recebem multa fixa de 2%...
#  - POL-002: Pagamentos acima de R$ 5.000 exigem aprovação...
#  PERGUNTA DO USUÁRIO: {pergunta}"
# ============================================================

def demo_memoria_longo_prazo() -> None:
    """
    Demonstra o ciclo completo de memória de longo prazo (RAG simples).

    ETAPAS:
    1. Define uma pergunta de exemplo sobre boletos
    2. Carrega a base de conhecimento (politicas_cobranca.json)
    3. Busca as políticas mais relevantes para a pergunta
    4. Exibe os resultados em uma tabela formatada com Rich

    OBSERVE NO OUTPUT:
    - Apenas políticas RELEVANTES são retornadas (não toda a base)
    - O scoring de relevância é simples mas eficaz para poucos documentos
    - A próxima evolução seria usar embeddings para semântica

    EXERCÍCIO SUGERIDO:
    1. Adicione mais políticas ao arquivo politicas_cobranca.json
    2. Teste com perguntas diferentes e veja quais políticas aparecem
    3. Tente uma pergunta que não match nenhuma política (lista vazia)
    """
    # Pergunta de exemplo — em produção, viria do usuário real
    pergunta = (
        "Quais políticas devo lembrar para boleto em atraso com alto valor?"
    )

    # Carrega a base de conhecimento do JSON
    base = carregar_base_conhecimento()

    # Busca os documentos mais relevantes (top_k=2 por padrão)
    memorias = recuperar_memorias(pergunta, base)

    # Monta tabela Rich para exibição formatada
    tabela = Table(
        title="📚 Memórias Recuperadas (RAG Simples)",
        header_style="bold magenta"
    )
    tabela.add_column("ID", style="cyan")
    tabela.add_column("Título")
    tabela.add_column("Conteúdo")

    for memoria in memorias:
        tabela.add_row(memoria["id"], memoria["titulo"], memoria["conteudo"])

    console.print(f"\n🔍 Pergunta: {pergunta}", style="bold blue")
    console.print(f"📊 Documentos na base: {len(base)}", style="dim")
    console.print(f"✅ Documentos recuperados: {len(memorias)}", style="dim")
    console.print(tabela)

    # Mostra como ficaria a injeção no prompt
    console.print(
        "\n💡 Como isso seria injetado no prompt da LLM:",
        style="bold yellow"
    )
    for memoria in memorias:
        console.print(
            f"  📌 [{memoria['id']}] {memoria['conteudo']}",
            style="dim"
        )

    console.print("\n🚀 Próxima evolução:", style="bold green")
    console.print("  Trocar palavras-chave por EMBEDDINGS e busca VETORIAL.")
    console.print(
        "  Isso captura SIGNIFICADO ('vencido' ≈ 'atraso') "
        "em vez de texto exato."
    )


# ============================================================
# PONTO DE ENTRADA — Execução direta do módulo
# ============================================================
# Ao rodar `python 01_memoria_rag.py`, o aluno verá a busca
# por relevância aplicada às políticas de cobrança.
#
# EXERCÍCIO EXTRA:
# 1. Crie uma nova política POL-004 sobre "desconto para pagamento antecipado"
# 2. Pergunte "meu cliente quer desconto" e veja se a política aparece
# 3. Implemente um campo "prioridade" nas políticas para desempate
# ============================================================

if __name__ == "__main__":
    demo_memoria_longo_prazo()
