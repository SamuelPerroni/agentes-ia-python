"""
============================================================
MÓDULO 44.1 - RAG AVANÇADO COM RERANKING
============================================================
RAG básico retorna os k documentos mais próximos por
embedding. RAG avançado combina múltiplos sinais e
re-ordena com um modelo mais preciso (reranker).

PIPELINE COMPLETO:

  Query
    │
    ├── BM25 (léxico, rápido)       ──► top-10
    │
    └── Embedding (semântico)       ──► top-10
                  │
                  ▼
    RRF (Reciprocal Rank Fusion)    ──► top-20 unificado
                  │
                  ▼
    Cross-Encoder Reranker          ──► top-3 precisos
                  │
                  ▼
    Geração LLM com contexto top-3

POR QUE RRF?
  BM25 pontua em escala absoluta (ex: 12.3).
  Embedding pontua em escala diferente (ex: 0.87).
  Não são comparáveis diretamente.
  RRF usa apenas o rank:  score = Σ 1 / (k + rank_i)
  Isso elimina a necessidade de normalizar scores.

POR QUE RERANKER?
  Embedding busca por aproximação em alta dimensão.
  Reranker avalia o par (query, doc) como sequência
  única — mais preciso, mas mais lento.
  Roda apenas sobre os ~20 candidatos do RRF.

QUANDO VALE A PENA:
  ✓ Base > 500 documentos
  ✓ Queries complexas ou multi-conceito
  ✓ Custo de erro alto (financeiro, jurídico)
  ✗ Base pequena (< 50 docs) — BM25 simples já resolve

Tópicos cobertos:
1. Tokenização e indexação de chunks
2. BM25 simplificado (TF com normalização)
3. Cosine similarity com vetores esparsos simulados
4. RRF — fusão sem reescala de pontuações
5. Cross-encoder reranker simulado
6. Comparativo: BM25 only vs Hybrid + Rerank
============================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. TOKENIZAÇÃO E VETORIZAÇÃO (SIMULADAS)
# ============================================================

_STOPWORDS = {
    "o", "a", "de", "do", "da", "em", "e", "para",
    "com", "que", "os", "as", "um", "uma", "ao", "ou",
}


def _tokenizar(texto: str) -> list[str]:
    """Lower + split + remoção de stopwords e pontuação."""
    tokens = texto.lower().split()
    return [
        t.strip(".,;:()[]")
        for t in tokens
        if t.strip(".,;:()[]") not in _STOPWORDS
        and len(t.strip(".,;:()[]")) > 2
    ]


# Vocabulário de domínio financeiro com pesos semânticos.
# Em produção: sentence-transformers ou OpenAI embeddings.
_VOCAB: dict[str, float] = {
    "pagamento": 0.80, "multa": 0.85, "desconto": 0.75,
    "prazo": 0.70,     "boleto": 0.90, "vencimento": 0.85,
    "cnpj": 0.90,      "fornecedor": 0.70, "aprovacao": 0.80,
    "limite": 0.75,    "duplicata": 0.85, "nota": 0.70,
    "fiscal": 0.70,    "validacao": 0.80, "politica": 0.90,
    "atraso": 0.80,    "antecipado": 0.75, "bloqueio": 0.85,
    "auditoria": 0.75, "valor": 0.70, "banco": 0.70,
    "diretoria": 0.80, "gestor": 0.75, "analista": 0.70,
}


def _vetorizar(texto: str) -> dict[str, float]:
    """Vetor esparso simulado a partir do vocab de domínio."""
    lower = texto.lower()
    return {t: p for t, p in _VOCAB.items() if t in lower}


def _cosine_sim(
    v1: dict[str, float], v2: dict[str, float]
) -> float:
    """Cosine similarity entre dois vetores esparsos."""
    if not v1 or not v2:
        return 0.0
    dot = sum(v1[k] * v2.get(k, 0.0) for k in v1)
    n1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    n2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0


# ============================================================
# 2. ESTRUTURAS DE DADOS
# ============================================================

@dataclass
class Chunk:
    """Unidade básica de informação indexada."""
    chunk_id: str
    fonte: str
    conteudo: str
    termos: list[str] = field(default_factory=list)
    vetor: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.termos:
            self.termos = _tokenizar(self.conteudo)
        if not self.vetor:
            self.vetor = _vetorizar(self.conteudo)


@dataclass
class ResultadoBusca:
    """Resultado de busca com múltiplos sinais para reranking."""
    chunk: Chunk
    score_bm25: float = 0.0
    score_embedding: float = 0.0
    rank_bm25: int = 999
    rank_embedding: int = 999
    score_rrf: float = 0.0
    score_reranker: float = 0.0


# ============================================================
# 3. BM25 SIMPLIFICADO
# ============================================================

class BM25:
    """
    BM25 simplificado sem IDF.
    k1=1.5 controla saturação de frequência.
    b=0.75 controla normalização por comprimento.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._avg_len = (
            sum(len(c.termos) for c in chunks) / len(chunks)
            if chunks else 1
        )

    def _tf(self, termo: str, chunk: Chunk) -> float:
        tf = chunk.termos.count(termo)
        dl = len(chunk.termos)
        num = tf * (self.K1 + 1)
        den = tf + self.K1 * (
            1 - self.B + self.B * dl / self._avg_len
        )
        return num / den if den > 0 else 0.0

    def buscar(
        self, query: str, top_k: int = 10
    ) -> list[tuple[Chunk, float]]:
        """Busca BM25 simplificada: pontua cada chunk e retorna top_k."""
        termos = _tokenizar(query)
        scores = [
            (c, sum(self._tf(t, c) for t in termos))
            for c in self._chunks
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================
# 4. BUSCA POR EMBEDDING
# ============================================================

class BuscaEmbedding:
    """Busca por similaridade de embedding."""
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def buscar(
        self, query: str, top_k: int = 10
    ) -> list[tuple[Chunk, float]]:
        """Vetoriza a query e calcula cosine similarity com cada chunk."""
        vq = _vetorizar(query)
        scores = [
            (c, _cosine_sim(vq, c.vetor))
            for c in self._chunks
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================
# 5. RECIPROCAL RANK FUSION
# ============================================================

def rrf_fusion(
    res_bm25: list[tuple[Chunk, float]],
    res_emb: list[tuple[Chunk, float]],
    k: int = 60,
) -> list[ResultadoBusca]:
    """
    RRF: score(d) = Σ_retriever 1 / (k + rank(d))
    Elimina a necessidade de normalizar scores heterogêneos.
    """
    rank_b = {c.chunk_id: i + 1 for i, (c, _) in enumerate(res_bm25)}
    rank_e = {c.chunk_id: i + 1 for i, (c, _) in enumerate(res_emb)}

    todos: dict[str, ResultadoBusca] = {}
    for c, s in res_bm25:
        todos[c.chunk_id] = ResultadoBusca(
            chunk=c, score_bm25=s
        )
    for c, s in res_emb:
        if c.chunk_id not in todos:
            todos[c.chunk_id] = ResultadoBusca(chunk=c)
        todos[c.chunk_id].score_embedding = s

    for cid, res in todos.items():
        rb = rank_b.get(cid, 999)
        re = rank_e.get(cid, 999)
        res.rank_bm25 = rb
        res.rank_embedding = re
        res.score_rrf = 1 / (k + rb) + 1 / (k + re)

    return sorted(
        todos.values(), key=lambda x: x.score_rrf, reverse=True
    )


# ============================================================
# 6. CROSS-ENCODER RERANKER (SIMULADO)
# ============================================================

class CrossEncoderReranker:
    """
    Simula cross-encoder que avalia (query, doc) como par.
    Em produção: sentence-transformers cross-encoder/ms-marco.

    A simulação usa: sobreposição de termos + alinhamento
    semântico de vetores — score normalizado em [0, 1].
    """

    def pontuar(self, query: str, chunk: Chunk) -> float:
        """Pontua o par (query, chunk) com base em:
        1. Overlap de termos (BM25-like)
        2. Bônus por presença de termos no conteúdo
        3. Similaridade semântica de vetores"""
        tq = set(_tokenizar(query))
        tc = set(chunk.termos)
        if not tq:
            return 0.0
        overlap = len(tq & tc) / len(tq)
        bonus = sum(
            0.08 for t in tq
            if t in chunk.conteudo.lower()
        )
        semantico = _cosine_sim(_vetorizar(query), chunk.vetor)
        return min(1.0, overlap + bonus + semantico * 0.3)

    def rerankar(
        self,
        query: str,
        candidatos: list[ResultadoBusca],
        top_k: int = 3,
    ) -> list[ResultadoBusca]:
        """Reranka os candidatos usando o cross-encoder simulado."""
        for r in candidatos:
            r.score_reranker = self.pontuar(query, r.chunk)
        return sorted(
            candidatos,
            key=lambda x: x.score_reranker,
            reverse=True,
        )[:top_k]


# ============================================================
# 7. PIPELINE RAG AVANÇADO
# ============================================================

class PipelineRAGAvancado:
    """Pipeline de busca híbrida RAG avançado com BM25,
    embeddings e reranking."""
    def __init__(self, chunks: list[Chunk]) -> None:
        self._bm25 = BM25(chunks)
        self._emb = BuscaEmbedding(chunks)
        self._reranker = CrossEncoderReranker()

    def buscar(
        self, query: str, top_k_final: int = 3
    ) -> list[ResultadoBusca]:
        """Busca híbrida com BM25 + embedding + RRF + reranking."""
        res_b = self._bm25.buscar(query, top_k=10)
        res_e = self._emb.buscar(query, top_k=10)
        fusao = rrf_fusion(res_b, res_e)[:20]
        return self._reranker.rerankar(query, fusao, top_k_final)

    def buscar_apenas_bm25(
        self, query: str, top_k: int = 3
    ) -> list[tuple[Chunk, float]]:
        """Busca apenas com BM25."""
        return self._bm25.buscar(query, top_k)


# ============================================================
# 8. BASE DE CONHECIMENTO DE DEMO
# ============================================================

_POLITICAS = [
    (
        "pol-01", "Política de Pagamento",
        "Pagamentos de boletos devem ser realizados até a "
        "data de vencimento. Atraso incorre em multa de 2% "
        "ao mês mais juros de 1% ao dia.",
    ),
    (
        "pol-02", "Política de Desconto",
        "Boletos pagos com antecipação de 10 dias ou mais "
        "recebem desconto de 5% sobre o valor total.",
    ),
    (
        "pol-03", "Política de Aprovação",
        "Pagamentos acima de R$ 10.000 exigem aprovação "
        "dupla: analista e gestor. Valores acima de "
        "R$ 50.000 requerem aprovação da diretoria.",
    ),
    (
        "pol-04", "Validação de CNPJ",
        "Todo fornecedor deve ter CNPJ válido e ativo. "
        "CNPJs com pendências geram bloqueio automático "
        "do pagamento.",
    ),
    (
        "pol-05", "Prevenção de Duplicatas",
        "O sistema verifica duplicatas por CNPJ + valor + "
        "vencimento. Pagamentos duplicados são retidos "
        "para auditoria.",
    ),
    (
        "pol-06", "Nota Fiscal",
        "Toda NF-e deve ser validada contra o pedido de "
        "compra antes do pagamento. Divergência acima de "
        "5% bloqueia o fluxo.",
    ),
]


def _criar_base() -> list[Chunk]:
    return [
        Chunk(chunk_id=cid, fonte=fonte, conteudo=conteudo)
        for cid, fonte, conteudo in _POLITICAS
    ]


# ============================================================
# 9. DEMO
# ============================================================

def demo_rag_avancado() -> None:
    """Demonstração do pipeline RAG avançado com
    BM25 + embedding + RRF + reranking."""
    console.print(
        Panel(
            "[bold]Módulo 44 — RAG Avançado com Reranking[/]\n"
            "Hybrid search (BM25 + embedding) + RRF + "
            "cross-encoder reranker",
            style="bold blue",
        )
    )

    chunks = _criar_base()
    pipeline = PipelineRAGAvancado(chunks)

    queries = [
        "Quais são as regras de multa por atraso?",
        "Como aprovar pagamentos de alto valor?",
        "O que acontece se o CNPJ estiver irregular?",
    ]

    for query in queries:
        console.rule(f"[yellow]{query}")

        # BM25 puro (baseline)
        res_bm25 = pipeline.buscar_apenas_bm25(query, top_k=3)
        console.print("[dim]BM25 only (baseline):[/]")
        for i, (c, s) in enumerate(res_bm25, 1):
            console.print(
                f"  {i}. [score={s:.2f}] {c.fonte} — "
                f"{c.conteudo[:55]}..."
            )

        # Híbrido + rerank
        top = pipeline.buscar(query, top_k_final=3)
        tabela = Table(
            show_header=True, header_style="bold magenta"
        )
        tabela.add_column("#")
        tabela.add_column("Fonte")
        tabela.add_column("BM25 rank", justify="center")
        tabela.add_column("Emb rank", justify="center")
        tabela.add_column("RRF", justify="right")
        tabela.add_column("Reranker", justify="right")
        for i, r in enumerate(top, 1):
            tabela.add_row(
                str(i),
                r.chunk.fonte,
                f"{r.rank_bm25}°",
                f"{r.rank_embedding}°",
                f"{r.score_rrf:.4f}",
                f"[green]{r.score_reranker:.3f}[/]",
            )
        console.print(tabela)
        console.print(
            f"  [bold]Contexto gerado para o LLM:[/] "
            f"{top[0].chunk.conteudo[:70]}..."
        )


if __name__ == "__main__":
    demo_rag_avancado()
