"""
============================================================
MÓDULO 42.1 - PROMPT CACHING
============================================================
Prompt caching armazena partes fixas do contexto no lado
do provedor de LLM para reutilização em requisições
subsequentes. Reduz custo em até 90% e latência em 85%.

COMO FUNCIONA:

  Sem cache:
  ──────────
  Req 1: [system longo] + [histórico] + [novo input]
         ↳ cobra todos os tokens
  Req 2: [system longo] + [histórico] + [novo input]
         ↳ cobra todos os tokens DE NOVO

  Com cache:
  ──────────
  Req 1: [system longo ← ESCREVE CACHE] + [novo input]
         ↳ cobra todos os tokens (cache miss)
  Req 2: [system longo ← LÊ CACHE]     + [novo input]
         ↳ cobra só os novos tokens (cache hit!)

PROVEDORES QUE SUPORTAM:
  Anthropic Claude  → cache_control: {"type": "ephemeral"}
  OpenAI GPT-4o     → automático para system > 1024 tokens
  Groq              → suporte previsto em 2025

QUANDO O CACHE VALE MAIS:
  ✓ System prompt com políticas/regras longas
  ✓ Documentos de referência no prompt (PDFs, schemas)
  ✓ Histórico de conversa longo
  ✓ Many-shot prompting (50+ exemplos no prompt)

ECONOMIA TÍPICA:
  Prompt base: 2.000 tokens (system + políticas)
  Input novo:    100 tokens por requisição
  100 requisições → 200.000 tokens sem cache
                  →  12.000 tokens com cache (-94%)

Tópicos cobertos:
1. CacheManager — rastreia hits, misses e economia
2. Simulação de requisição com e sem cache
3. Cálculo de economia de tokens e custo
4. Estratégias de invalidação de cache
5. Benchmark: 50 requisições com/sem caching
6. Quando NÃO usar cache (conteúdo dinâmico)
============================================================
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. ENTRADA E SAÍDA
# ============================================================

@dataclass
class MensagemLLM:
    """Representa uma mensagem enviada para a LLM."""
    role: str      # system | user | assistant
    content: str
    cache: bool = False   # marcar para cache no provider


@dataclass
class RespostaLLM:
    """Representa a resposta da LLM,
    incluindo métricas de tokens e latência."""
    conteudo: str
    tokens_input: int
    tokens_input_cached: int   # quantos vieram do cache
    tokens_output: int
    latencia_ms: float

    @property
    def tokens_cobrados(self) -> int:
        """Tokens que foram efetivamente cobrados."""
        return self.tokens_input - self.tokens_input_cached


# ============================================================
# 2. CACHE MANAGER
# ============================================================

@dataclass
class EstatisticasCache:
    """Rastreia estatísticas de cache para análise de economia."""
    hits: int = 0
    misses: int = 0
    tokens_economizados: int = 0
    custo_economizado_usd: float = 0.0

    @property
    def taxa_hit(self) -> float:
        """Calcula a taxa de acerto do cache."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheManager:
    """Gerencia o cache de prompts do lado do provedor.
    Simula o comportamento de cache de prompt do lado
    do provedor. Em produção, o cache é gerenciado
    automaticamente pela API (Anthropic/OpenAI).
    """

    # Custo de tokens cacheados é ~10% do normal
    CUSTO_INPUT_NORMAL_POR_1K = 0.003   # USD
    CUSTO_INPUT_CACHED_POR_1K = 0.0003  # USD (≈10%)

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}
        self.stats = EstatisticasCache()

    def _chave(self, conteudo: str) -> str:
        return hashlib.sha256(
            conteudo.encode()
        ).hexdigest()[:16]

    def consultar(
        self, mensagens_cacheadas: list[str]
    ) -> tuple[bool, int]:
        """
        Retorna (cache_hit, tokens_cacheados).
        """
        conteudo_combinado = "".join(mensagens_cacheadas)
        chave = self._chave(conteudo_combinado)
        tokens = len(conteudo_combinado) // 4

        if chave in self._cache:
            self.stats.hits += 1
            self.stats.tokens_economizados += tokens
            economia = (
                tokens / 1000
                * (
                    self.CUSTO_INPUT_NORMAL_POR_1K
                    - self.CUSTO_INPUT_CACHED_POR_1K
                )
            )
            self.stats.custo_economizado_usd += economia
            return True, tokens

        # Cache miss: armazena para próximas requisições
        self._cache[chave] = conteudo_combinado
        self.stats.misses += 1
        return False, 0


# ============================================================
# 3. CLIENTE LLM COM CACHE
# ============================================================

def _estimar_tokens(texto: str) -> int:
    return max(1, len(texto) // 4)


def chamar_llm_com_cache(
    mensagens: list[MensagemLLM],
    cache_manager: CacheManager,
    latencia_base_ms: float = 500.0,
) -> RespostaLLM:
    """
    Simula uma chamada ao LLM com suporte a cache.
    Mensagens marcadas com cache=True são candidatas
    ao cache do provedor.
    """
    # Separa mensagens cacheáveis das dinâmicas
    msgs_cache = [
        m.content for m in mensagens if m.cache
    ]
    msgs_dinamicas = [
        m.content for m in mensagens if not m.cache
    ]

    tokens_total_input = sum(
        _estimar_tokens(m.content) for m in mensagens
    )

    cache_hit, tokens_cacheados = (
        cache_manager.consultar(msgs_cache)
        if msgs_cache
        else (False, 0)
    )

    # Latência reduzida em cache hit (sem reprocessar prefix)
    latencia = (
        latencia_base_ms * 0.15
        if cache_hit
        else latencia_base_ms
    )

    # Simula resposta
    ultimo_input = msgs_dinamicas[-1] if msgs_dinamicas else ""
    resposta_sim = (
        f"Processado: {ultimo_input[:60]}..."
        if len(ultimo_input) > 60
        else f"Processado: {ultimo_input}"
    )

    return RespostaLLM(
        conteudo=resposta_sim,
        tokens_input=tokens_total_input,
        tokens_input_cached=tokens_cacheados,
        tokens_output=_estimar_tokens(resposta_sim),
        latencia_ms=latencia,
    )


# ============================================================
# 4. DEMO
# ============================================================

_SYSTEM_LONGO = """
Você é um agente especializado em Automação de Processos
com IA (APA) para o setor financeiro.

CONTEXTO DA EMPRESA:
A empresa processa 500 boletos/dia com prazo de pagamento
de 30 dias. Multa por atraso: 2% ao mês. Desconto para
pagamento antecipado de 5% se pago em 10 dias.

POLÍTICA DE VALIDAÇÃO:
1. Todo CNPJ deve ser validado antes do processamento.
2. Boletos acima de R$ 10.000 requerem dupla aprovação.
3. Fornecedores com dívidas ativas são automaticamente
   bloqueados.
4. Notas fiscais devem corresponder aos pedidos de compra.

REGRAS DE EXTRAÇÃO:
- Valor: sempre em BRL, duas casas decimais
- CNPJ: formato XX.XXX.XXX/XXXX-XX
- Data: formato YYYY-MM-DD
- Banco: nome completo sem abreviações

EXEMPLOS DE SAÍDA:
{"tipo": "boleto", "valor": 1250.00, "vencimento":
"2026-05-10", "banco": "Banco do Brasil", "cnpj":
"12.345.678/0001-99"}
""".strip()


def demo_prompt_caching() -> None:
    """Demonstração do impacto do prompt caching em um cenário de
    processamento de boletos, comparando tokens cobrados e latência"""
    console.print(
        Panel(
            "[bold]Módulo 42 — Prompt Caching[/]\n"
            "Reduza custo em até 90% reutilizando partes "
            "fixas do contexto no provedor de LLM",
            style="bold blue",
        )
    )

    n_requisicoes = 20
    inputs_usuario = [
        f"Processe o boleto R$ {i * 100 + 500:.2f} "
        f"venc 10/0{(i % 9) + 1}/2026"
        for i in range(n_requisicoes)
    ]

    # --- SEM cache ---
    console.rule("[yellow]Sem Cache")
    tokens_sem_cache_total = 0
    latencias_sem: list[float] = []
    for inp in inputs_usuario:
        tokens = (
            _estimar_tokens(_SYSTEM_LONGO)
            + _estimar_tokens(inp)
        )
        tokens_sem_cache_total += tokens
        latencias_sem.append(500.0)

    custo_sem = (
        tokens_sem_cache_total / 1000
        * CacheManager.CUSTO_INPUT_NORMAL_POR_1K
    )

    # --- COM cache ---
    console.rule("[yellow]Com Cache")
    cache_mgr = CacheManager()
    tokens_com_cache_total = 0
    latencias_com: list[float] = []

    for inp in inputs_usuario:
        msgs = [
            MensagemLLM("system", _SYSTEM_LONGO, cache=True),
            MensagemLLM("user", inp, cache=False),
        ]
        resp = chamar_llm_com_cache(msgs, cache_mgr)
        tokens_com_cache_total += resp.tokens_cobrados
        latencias_com.append(resp.latencia_ms)

    custo_com = (
        tokens_com_cache_total / 1000
        * CacheManager.CUSTO_INPUT_NORMAL_POR_1K
    )

    # --- Relatório ---
    console.rule("[yellow]Comparativo")
    tabela = Table(header_style="bold magenta")
    tabela.add_column("Métrica")
    tabela.add_column("Sem Cache", justify="right")
    tabela.add_column("Com Cache", justify="right")
    tabela.add_column("Economia", justify="right")

    lat_media_sem = sum(latencias_sem) / len(latencias_sem)
    lat_media_com = sum(latencias_com) / len(latencias_com)

    tabela.add_row(
        "Requisições",
        str(n_requisicoes),
        str(n_requisicoes),
        "—",
    )
    tabela.add_row(
        "Tokens cobrados",
        f"{tokens_sem_cache_total:,}",
        f"{tokens_com_cache_total:,}",
        "[green]-"
        f"{1 - tokens_com_cache_total/tokens_sem_cache_total:.0%}[/]",
    )
    tabela.add_row(
        "Custo USD",
        f"${custo_sem:.4f}",
        f"${custo_com:.4f}",
        f"[green]-${custo_sem - custo_com:.4f}[/]",
    )
    tabela.add_row(
        "Latência média",
        f"{lat_media_sem:.0f} ms",
        f"{lat_media_com:.0f} ms",
        f"[green]-{(1 - lat_media_com/lat_media_sem):.0%}[/]",
    )
    tabela.add_row(
        "Cache hits",
        "—",
        str(cache_mgr.stats.hits),
        f"taxa {cache_mgr.stats.taxa_hit:.0%}",
    )
    console.print(tabela)

    console.print(
        Panel(
            f"Tokens economizados: "
            f"[bold]{cache_mgr.stats.tokens_economizados:,}[/]\n"
            f"Custo economizado:   "
            f"[bold green]"
            f"${cache_mgr.stats.custo_economizado_usd:.4f}[/]",
            title="Resumo de Economia",
            style="green",
        )
    )


if __name__ == "__main__":
    demo_prompt_caching()
