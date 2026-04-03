"""
============================================================
MÓDULO 10 - UTILITÁRIOS DE MEMÓRIA (memory_utils.py)
============================================================
Este arquivo contém as funções REUTILIZÁVEIS de memória de longo prazo.
É importado pelo agente de boletos (módulo 06) e pelo demo (01_memoria_rag.py).

O QUE É MEMÓRIA DE LONGO PRAZO?
Nos módulos anteriores, nosso agente tinha apenas memória de CURTO prazo
(as últimas mensagens da conversa). Isso significa que, ao iniciar uma
nova conversa, o agente "esquece" tudo.

Memória de longo prazo é uma BASE CONSULTÁVEL de conhecimento que o
agente pode acessar a qualquer momento — políticas da empresa, regras
de negócio, procedimentos, etc.

POR QUE MEMÓRIA DE LONGO PRAZO?
- System prompts têm LIMITE DE TOKENS — não cabe toda a política da empresa
- Regras mudam — é mais fácil atualizar um JSON do que reescrever prompts
- Relevância: em vez de enviar TUDO para a LLM, buscamos apenas o que
  é relevante para a pergunta (isso economiza tokens e melhora a resposta)

ANALOGIA:
Pense em um atendente humano:
- Memória de curto prazo = o que o cliente acabou de falar
- Memória de longo prazo = o manual de procedimentos na gaveta
O atendente não lê o manual inteiro a cada ligação — ele consulta
a seção relevante para o problema do momento.

TÉCNICA USADA — Busca por Palavras-Chave (Keyword Search):
Neste módulo, usamos busca simples por palavras-chave para manter
o treinamento acessível. Em produção, a evolução natural é:

  Palavras-chave (aqui) → TF-IDF → Embeddings → Busca Vetorial (RAG completo)

COMPONENTES:
1. carregar_base_conhecimento() - lê o JSON de políticas de cobrança
2. recuperar_memorias()         - busca os documentos mais relevantes
============================================================
"""

from __future__ import annotations

import json
import os


# ============================================================
# 1. CARREGAMENTO DA BASE DE CONHECIMENTO
# ============================================================
# A base é um arquivo JSON simples (politicas_cobranca.json) que
# contém documentos com: id, titulo, tags e conteudo.
#
# ESTRUTURA DE CADA DOCUMENTO:
# {
#   "id": "POL-001",
#   "titulo": "Política padrão de multa",
#   "tags": ["multa", "juros", "atraso"],
#   "conteudo": "Boletos vencidos recebem multa fixa de 2%..."
# }
#
# POR QUE JSON E NÃO BANCO DE DADOS?
# - Para treinamento, JSON é simples e não requer infraestrutura
# - Em produção, use banco vetorial (Pinecone, Weaviate, pgvector)
# - O princípio é o mesmo: carregar → buscar → injetar no prompt
# ============================================================

def carregar_base_conhecimento() -> list[dict]:
    """
    Carrega a base de conhecimento (políticas de cobrança) do arquivo JSON.

    COMO FUNCIONA:
    1. Encontra o arquivo politicas_cobranca.json no mesmo diretório
    2. Lê e faz parse do JSON
    3. Retorna uma lista de dicionários (um por documento/política)

    RETORNO:
    Lista de dicts, cada um com: id, titulo, tags, conteudo

    NOTA:
    Em produção, esta função seria substituída por uma consulta a um
    banco de dados vetorial ou um serviço de busca (Elasticsearch, etc.)
    """
    caminho = os.path.join(
        os.path.dirname(__file__),
        "politicas_cobranca.json"
    )
    with open(caminho, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


# ============================================================
# 2. BUSCA POR RELEVÂNCIA (Keyword Search)
# ============================================================
# Dado uma pergunta, queremos encontrar os documentos MAIS
# RELEVANTES da base de conhecimento. Usamos uma técnica simples:
# contar quantas palavras da pergunta aparecem no documento.
#
# COMO FUNCIONA O SCORING:
# 1. Quebramos a pergunta em palavras (split)
# 2. Para cada documento, combinamos tags + titulo + conteudo
# 3. Contamos quantos termos da pergunta aparecem no texto combinado
# 4. Ordenamos por score (maior = mais relevante)
# 5. Retornamos os top_k documentos com score > 0
#
# EXEMPLO:
# Pergunta: "boleto em atraso com multa"
# Documento POL-001 (tags: multa, juros, atraso) → score = 2 (multa + atraso)
# Documento POL-002 (tags: hitl, risco, aprovacao) → score = 0
# → Retorna POL-001
#
# LIMITAÇÕES:
# - Não entende sinônimos ("atrasado" vs "vencido")
# - Não pondera importância das palavras (todas valem 1)
# - Em produção, use embeddings para capturar significado semântico
#
# EVOLUÇÃO PARA PRODUÇÃO:
#   ┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
#   │ Keyword Search   │ →→  │ TF-IDF        │ →→  │ Embeddings/  │
#   │ (este módulo)    │     │ (pondera      │     │ Vetorial     │
#   │                  │     │  palavras)     │     │ (semântico)  │
#   └─────────────────┘     └──────────────┘     └──────────────┘
# ============================================================

def recuperar_memorias(
        pergunta: str,
        base: list[dict],
        top_k: int = 2
) -> list[dict]:
    """
    Busca os documentos mais relevantes na base para a pergunta dada.

    Parâmetros:
    - pergunta: texto free-form do usuário (ex: "boleto em atraso")
    - base: lista de documentos carregados da base de conhecimento
    - top_k: quantidade máxima de documentos a retornar (default: 2)

    COMO FUNCIONA:
    1. Quebra a pergunta em termos (palavras) e converte para minúsculas
    2. Para cada documento, cria um texto de busca (tags + titulo + conteudo)
    3. Conta quantos termos da pergunta aparecem no texto de busca
    4. Ordena por score decrescente e retorna os top_k com score > 0

    RETORNO:
    Lista de até top_k dicionários (documentos relevantes), ordenados
    por relevância. Se nenhum documento tiver score > 0, retorna lista vazia.

    EXEMPLO:
    >>> base = carregar_base_conhecimento()
    >>> docs = recuperar_memorias("multa de boleto em atraso", base)
    >>> docs[0]["titulo"]  # → "Política padrão de multa"
    """
    # Quebramos a pergunta em palavras únicas (set elimina duplicatas)
    termos = set(palavra.lower() for palavra in pergunta.split())

    # Para cada documento, calculamos um score de relevância
    pontuados = []
    for documento in base:
        score = 0
        # Combinamos tags + titulo + conteudo em um texto único para busca
        texto_busca = " ".join(
            documento["tags"] + [documento["titulo"], documento["conteudo"]]
        ).lower()
        # Contamos quantos termos da pergunta aparecem no texto do documento
        for termo in termos:
            if termo in texto_busca:
                score += 1
        pontuados.append((score, documento))

    # Ordenamos por score decrescente (mais relevante primeiro)
    pontuados.sort(key=lambda item: item[0], reverse=True)

    # Filtramos documentos com score > 0 e limitamos a top_k
    return [documento for score, documento in pontuados if score > 0][:top_k]
