"""
============================================================
MÓDULO 21.1 - GESTÃO DE CUSTOS E OTIMIZAÇÃO DE TOKENS
============================================================
Neste módulo, aprendemos a monitorar, estimar e reduzir o custo
de operação de agentes de IA em produção.

CONCEITO CHAVE:
Em APA (Automação de Processos com Agentes), o custo de tokens
é a principal métrica financeira. Um agente processando 10.000
boletos por dia com prompt ineficiente pode custar 10× mais
do que o necessário.

POR QUE ISSO IMPORTA EM APA?
- Prompts longos = mais tokens de entrada = mais $
- Histórico de conversação cresce sem controle
- Modelos maiores custam 10-50× mais que modelos menores
- Sem monitoramento, o custo é imprevisível

ESTRATÉGIAS DE OTIMIZAÇÃO:

  ┌──────────────────────────────────────────────────────────┐
  │           PIRÂMIDE DE OTIMIZAÇÃO DE TOKENS               │
  │                                                          │
  │              ┌───────────────┐                           │
  │              │ Modelo certo  │  ← Use o menor adequado   │
  │              │  por tarefa   │                           │
  │           ┌──┴───────────────┴──┐                        │
  │           │   Semantic cache    │  ← Reutilize respostas │
  │           │  perguntas iguais   │                        │
  │        ┌──┴─────────────────────┴──┐                     │
  │        │   Compressão de prompt    │  ← Remova ruído     │
  │        │   e histórico             │                     │
  │     ┌──┴───────────────────────────┴──┐                  │
  │     │   Monitoramento + alertas de $  │  ← Visibilidade  │
  │     └─────────────────────────────────┘                  │
  └──────────────────────────────────────────────────────────┘

SELEÇÃO DINÂMICA DE MODELO:

  ┌──────────┐     classifica      ┌──────────────────────┐
  │  Tarefa  │ ──────────────────▶ │ Simples? → llama-8b  │
  │          │                     │ Média?   → llama-70b │
  └──────────┘                     │ Complexa?→ gpt-4o    │
                                   └──────────────────────┘
  Diferença de custo: até 50× entre o menor e o maior modelo.

SEMANTIC CACHE:
Perguntas semanticamente iguais não precisam chamar o LLM.
Um cache de embeddings (ou hash exato) evita chamadas repetidas.

  "Qual o vencimento do boleto XYZ?" → cache HIT  → sem custo
  "Me diz quando vence o boleto XYZ" → cache MISS → chama LLM

Tópicos cobertos:
1. Estimativa e rastreamento de custo por execução
2. Budget por tarefa com interrupção automática
3. Seleção dinâmica de modelo por complexidade da tarefa
4. Semantic cache por hash exato
5. Compressão de histórico de conversação
6. Relatório de custo por lote
============================================================
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Console do Rich para output formatado
console = Console()


# ============================================================
# 1. TABELA DE PREÇOS E ESTIMATIVA DE CUSTO
# ============================================================
# Preços por 1M de tokens (valores aproximados de abril de 2026).
# Sempre consulte a página do provedor para valores atualizados.
#
# MODELO      | INPUT ($/1M) | OUTPUT ($/1M) | USO TÍPICO
# ------------|-------------|---------------|------------------
# llama-8b    |    0.05     |     0.08      | Triagem, extração
# llama-70b   |    0.59     |     0.79      | Raciocínio geral
# llama-405b  |    3.00     |     3.00      | Casos complexos
# gpt-4o-mini |    0.15     |     0.60      | Balanceado
# gpt-4o      |    5.00     |    15.00      | Alta precisão
#
# DICA: Use llama-8b para extração de campos estruturados
# (onde o formato é fixo) e reserve modelos maiores para
# análise de linguagem natural complexa.
# ============================================================

PRECOS_MODELO: dict[str, dict[str, float]] = {
    # Custo em dólares por 1 milhão de tokens
    "llama-3.1-8b-instant": {
        "input":  0.05,
        "output": 0.08,
        "complexidade_max": 2,   # suporta tarefas de complexidade ≤ 2
    },
    "llama-3.3-70b-versatile": {
        "input":  0.59,
        "output": 0.79,
        "complexidade_max": 4,
    },
    "llama-3.1-405b-reasoning": {
        "input":  3.00,
        "output": 3.00,
        "complexidade_max": 5,
    },
}


def estimar_custo_usd(
    modelo: str,
    tokens_entrada: int,
    tokens_saida: int,
) -> float:
    """
    Calcula custo estimado em USD para uma chamada ao LLM.

    Parâmetros:
    - modelo: nome do modelo (chave de PRECOS_MODELO)
    - tokens_entrada: tokens no prompt + contexto
    - tokens_saida: tokens na resposta gerada

    Retorna:
    - Custo em dólares (float)
    """
    preco = PRECOS_MODELO.get(modelo, {
        "input": 1.00, "output": 1.00,
    })
    custo = (
        tokens_entrada * preco["input"] / 1_000_000
        + tokens_saida * preco["output"] / 1_000_000
    )
    return round(custo, 8)


# ============================================================
# 2. RASTREADOR DE CUSTO POR SESSÃO
# ============================================================
# Mantém o custo acumulado de uma sessão (ex: processamento
# de um lote de boletos) e emite alertas quando o budget
# está sendo comprometido.
#
# EM PRODUÇÃO: persista os dados no banco para dashboards
# financeiros e alertas via e-mail/Slack quando o custo
# ultrapassar o budget mensal.
# ============================================================

@dataclass
class RegistroChamada:
    """Registro de uma única chamada ao LLM com custo detalhado."""

    timestamp: str
    modelo: str
    tarefa: str
    tokens_entrada: int
    tokens_saida: int
    custo_usd: float
    cache_hit: bool = False


@dataclass
class RastreadorCusto:
    """
    Rastreia custos de chamadas ao LLM em tempo real.

    COMO USAR:
        rastreador = RastreadorCusto(budget_usd=0.10)
        rastreador.registrar("llama-3.3-70b", "extração", 500, 150)
        if rastreador.budget_excedido():
            raise RuntimeError("Budget esgotado!")

    DICA DE PRODUÇÃO:
    Crie um rastreador por lote/processo e persista o JSON
    de relatório no S3/Azure Blob para auditoria financeira.
    """

    budget_usd: float = 1.00
    _chamadas: list[RegistroChamada] = field(default_factory=list)

    @property
    def custo_total(self) -> float:
        """Soma de todos os custos registrados."""
        return round(sum(c.custo_usd for c in self._chamadas), 8)

    @property
    def tokens_total_entrada(self) -> int:
        """Soma de tokens de entrada de todas as chamadas."""
        return sum(c.tokens_entrada for c in self._chamadas)

    @property
    def tokens_total_saida(self) -> int:
        """Soma de tokens de saída de todas as chamadas."""
        return sum(c.tokens_saida for c in self._chamadas)

    def registrar(
        self,
        modelo: str,
        tarefa: str,
        tokens_entrada: int,
        tokens_saida: int,
        cache_hit: bool = False,
    ) -> float:
        """
        Registra uma chamada e retorna o custo calculado.

        Parâmetros:
        - modelo: identificador do modelo usado
        - tarefa: descrição curta da tarefa (para relatório)
        - tokens_entrada: tokens consumidos no prompt
        - tokens_saida: tokens gerados na resposta
        - cache_hit: True se a resposta veio do cache (custo $0)

        Retorna:
        - custo_usd dessa chamada específica
        """
        custo = 0.0 if cache_hit else estimar_custo_usd(
            modelo, tokens_entrada, tokens_saida
        )
        self._chamadas.append(RegistroChamada(
            timestamp=datetime.now().isoformat(),
            modelo=modelo,
            tarefa=tarefa,
            tokens_entrada=tokens_entrada,
            tokens_saida=tokens_saida,
            custo_usd=custo,
            cache_hit=cache_hit,
        ))
        return custo

    def budget_excedido(self) -> bool:
        """Retorna True se o custo total ultrapassou o budget."""
        return self.custo_total >= self.budget_usd

    def percentual_budget(self) -> float:
        """Percentual do budget consumido (0-100+)."""
        if self.budget_usd <= 0:
            return 100.0
        return round(self.custo_total / self.budget_usd * 100, 1)

    def relatorio(self) -> None:
        """Exibe tabela detalhada de todas as chamadas."""
        tabela = Table(title="Relatório de Custo de Tokens")
        tabela.add_column("Tarefa", style="cyan")
        tabela.add_column("Modelo", style="dim")
        tabela.add_column("In", justify="right")
        tabela.add_column("Out", justify="right")
        tabela.add_column("Cache", justify="center")
        tabela.add_column("$ USD", justify="right", style="yellow")

        for c in self._chamadas:
            cache_str = "[green]HIT[/green]" if c.cache_hit else "—"
            custo_str = (
                "[dim]$0.00[/dim]" if c.cache_hit
                else f"${c.custo_usd:.6f}"
            )
            tabela.add_row(
                c.tarefa,
                c.modelo.split("-")[0] + "…",
                str(c.tokens_entrada),
                str(c.tokens_saida),
                cache_str,
                custo_str,
            )

        console.print(tabela)

        pct = self.percentual_budget()
        cor_budget = (
            "green" if pct < 50
            else "yellow" if pct < 85
            else "red bold"
        )
        console.print(
            f"\n  Total: [yellow]${self.custo_total:.6f}[/yellow] | "
            f"Budget: ${self.budget_usd:.2f} | "
            f"Uso: [{cor_budget}]{pct}%[/{cor_budget}]"
        )


# ============================================================
# 3. SELEÇÃO DINÂMICA DE MODELO
# ============================================================
# Classifica a complexidade da tarefa e escolha o modelo mais
# barato capaz de executá-la com qualidade.
#
# NÍVEIS DE COMPLEXIDADE (1-5):
# 1 — Extração de campo único (ex: "qual o valor?")
# 2 — Extração multi-campo (ex: "extraia valor, banco, vencimento")
# 3 — Raciocínio simples (ex: "o boleto está vencido?")
# 4 — Análise + decisão (ex: "aprove ou recuse com justificativa")
# 5 — Casos complexos (ex: "analise a fraude com base em 10 sinais")
#
# REGRA: escolha o modelo com complexidade_max >= nível exigido
# e menor custo de input. Economias de até 50× são comuns.
# ============================================================

def classificar_complexidade(mensagem: str) -> int:
    """
    Estima a complexidade da tarefa (1-5) analisando palavras-chave.

    Em produção, substitua por um classificador treinado ou
    use um modelo ultra-leve (llama-8b) para classificar antes
    de chamar o modelo definitivo.

    Parâmetros:
    - mensagem: texto da tarefa/pergunta do usuário

    Retorna:
    - Nível de 1 (simples) a 5 (muito complexo)
    """
    msg = mensagem.lower()

    # Indicadores de alta complexidade
    if any(w in msg for w in [
        "analise", "compare", "investigue", "fraude",
        "justifique", "auditoria", "múltiplos critérios",
    ]):
        return 4

    # Raciocínio e decisão
    if any(w in msg for w in [
        "decide", "aprove", "recuse", "avalie",
        "recomende", "verifique se",
    ]):
        return 3

    # Extração multi-campo
    if any(w in msg for w in [
        "extraia", "liste todos", "quais são", "todos os campos",
    ]):
        return 2

    # Extração simples
    return 1


def selecionar_modelo(complexidade: int) -> str:
    """
    Retorna o modelo mais barato adequado à complexidade.

    Parâmetros:
    - complexidade: nível 1-5

    Retorna:
    - Nome do modelo a ser usado
    """
    for nome, cfg in sorted(
        PRECOS_MODELO.items(),
        key=lambda item: item[1]["input"],  # ordena pelo mais barato
    ):
        if cfg["complexidade_max"] >= complexidade:
            return nome
    # Fallback para o modelo mais capaz
    return "llama-3.1-405b-reasoning"


# ============================================================
# 4. SEMANTIC CACHE POR HASH
# ============================================================
# A forma mais simples de cache: se o texto exato já foi
# respondido antes, reutilize a resposta sem chamar o LLM.
#
# LIMITAÇÃO: só captura perguntas IDÊNTICAS.
# EVOLUÇÃO: use embeddings para capturar perguntas similares.
#   → OpenAI text-embedding-3-small ou sentence-transformers
#   → Armazene vetores no Redis/Pinecone
#   → Busca por cosine similarity com threshold de 0.95
#
# Para demonstrar o conceito, o cache por hash é suficiente
# e não tem dependência de biblioteca de embeddings.
# ============================================================

class CacheSemantico:
    """
    Cache de respostas do LLM por hash SHA-256 do prompt.

    INTERFACE:
        cache = CacheSemantico()
        resp = cache.buscar(prompt)
        if resp is None:
            resp = chamar_llm(prompt)
            cache.armazenar(prompt, resp, tokens_saida=120)
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def _chave(self, prompt: str) -> str:
        """SHA-256 do prompt normalizado (espaços colapsados)."""
        normalizado = " ".join(prompt.split()).lower()
        return hashlib.sha256(normalizado.encode()).hexdigest()[:16]

    def buscar(self, prompt: str) -> str | None:
        """
        Busca resposta em cache.

        Retorna:
        - String com a resposta se encontrada (cache HIT)
        - None se não encontrada (cache MISS)
        """
        entrada = self._store.get(self._chave(prompt))
        if entrada:
            entrada["hits"] += 1
            entrada["ultimo_acesso"] = datetime.now().isoformat()
            return entrada["resposta"]
        return None

    def armazenar(
        self,
        prompt: str,
        resposta: str,
        tokens_saida: int = 0,
    ) -> None:
        """
        Armazena par prompt→resposta no cache.

        Parâmetros:
        - prompt: texto exato enviado ao LLM
        - resposta: texto exato retornado
        - tokens_saida: para estatísticas de economia
        """
        chave = self._chave(prompt)
        self._store[chave] = {
            "prompt_preview": prompt[:80],
            "resposta": resposta,
            "tokens_saida": tokens_saida,
            "hits": 0,
            "criado_em": datetime.now().isoformat(),
            "ultimo_acesso": datetime.now().isoformat(),
        }

    def estatisticas(self) -> dict[str, Any]:
        """Retorna métricas do cache."""
        total_hits = sum(e["hits"] for e in self._store.values())
        tokens_economizados = sum(
            e["tokens_saida"] * e["hits"]
            for e in self._store.values()
        )
        return {
            "entradas": len(self._store),
            "total_hits": total_hits,
            "tokens_economizados": tokens_economizados,
        }


# ============================================================
# 5. COMPRESSÃO DE HISTÓRICO DE CONVERSAÇÃO
# ============================================================
# O histórico de conversa cresce sem controle e consome tokens
# de contexto. Após N mensagens, sumarizamos as mais antigas
# para liberar espaço sem perder o contexto relevante.
#
# ESTRATÉGIA SLIDING WINDOW + SUMÁRIO:
#
#   [msg1, msg2, msg3, msg4, msg5, msg6, msg7] ← janela cheia
#         │
#         ▼ sumarizar msgs 1-4
#   [SUMÁRIO(1-4), msg5, msg6, msg7]           ← janela enxuta
#
# O SUMÁRIO preserva: fatos coletados, decisões tomadas,
# e o estado atual da tarefa — o que é suficiente para
# continuar o processamento.
# ============================================================

def comprimir_historico(
    mensagens: list[dict[str, str]],
    max_mensagens: int = 10,
    manter_recentes: int = 4,
) -> list[dict[str, str]]:
    """
    Comprime o histórico de conversação por sumarização simulada.

    Quando o histórico excede max_mensagens, sumariza as
    mensagens mais antigas e preserva apenas as mais recentes.

    Parâmetros:
    - mensagens: lista de {"role": ..., "content": ...}
    - max_mensagens: limite para acionar compressão
    - manter_recentes: quantas mensagens recentes preservar

    Retorna:
    - Nova lista com sumário + mensagens recentes
    """
    if len(mensagens) <= max_mensagens:
        return mensagens

    # Mensagens a sumarizar (as mais antigas)
    para_sumarizar = mensagens[:-manter_recentes]
    recentes = mensagens[-manter_recentes:]

    # Em produção: chame o LLM com um prompt de sumarização:
    # "Resuma em 3 bullet points o que foi discutido:
    #  {chr(10).join(m['content'] for m in para_sumarizar)}"
    # Aqui simulamos o sumário para a demo.
    trechos = [
        m["content"][:60] + "…"
        for m in para_sumarizar
        if m["role"] == "assistant"
    ]
    sumario = (
        f"[CONTEXTO ANTERIOR RESUMIDO — "
        f"{len(para_sumarizar)} mensagens]\n"
        + "\n".join(f"• {t}" for t in trechos[:3])
    )

    return [
        {"role": "system", "content": sumario},
        *recentes,
    ]


def contar_tokens_estimado(mensagens: list[dict[str, str]]) -> int:
    """
    Estimativa simples: ~1 token por 4 caracteres de texto.

    Em produção use tiktoken (OpenAI) ou a biblioteca do seu
    provedor para contagem exata.

    Parâmetros:
    - mensagens: lista de {"role": ..., "content": ...}

    Retorna:
    - Estimativa de tokens
    """
    total_chars = sum(
        len(m.get("content", "")) + len(m.get("role", ""))
        for m in mensagens
    )
    return total_chars // 4


# ============================================================
# DEMO COMPLETA — Gestão de custos em pipeline de boletos
# ============================================================

def _simular_chamada_llm(
    modelo: str,
    prompt: str,
    tarefa: str,
) -> tuple[str, int, int]:
    """
    Simula uma chamada ao LLM com tokens estimados.

    Em produção, substitua por:
        from groq import Groq
        client = Groq()
        resp = client.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": prompt}],
        )
        tokens_in  = resp.usage.prompt_tokens
        tokens_out = resp.usage.completion_tokens
        return resp.choices[0].message.content, tokens_in, tokens_out

    Retorna:
    - (resposta, tokens_entrada, tokens_saida)
    """
    time.sleep(0.05)  # simula latência de rede
    tokens_entrada = len(prompt) // 4
    tokens_saida = 80 + (len(tarefa) * 2)
    resposta = (
        f"[SIMULADO — {modelo.split('-')[1]}] "
        f"Resultado para: {tarefa[:50]}"
    )
    return resposta, tokens_entrada, tokens_saida


def demo_custos() -> None:
    """
    Demonstra monitoramento e otimização de custos em pipeline.

    ETAPAS:
    1. Processa 6 tarefas simuladas com seleção dinâmica de modelo
    2. Demonstra o benefício do cache semântico
    3. Simula compressão de histórico
    4. Exibe relatório comparativo de custo

    OBSERVE NO OUTPUT:
    - Tarefas simples usam llama-8b (mais barato)
    - Tarefas complexas escalam para llama-70b
    - Cache hit = custo zero na segunda chamada
    - O relatório mostra economia acumulada

    EXERCÍCIO SUGERIDO:
    1. Adicione uma tarefa de complexidade 5 e veja qual modelo
       é selecionado automaticamente
    2. Repita a mesma tarefa e observe o cache HIT
    3. Adicione 12 mensagens ao histórico e veja a compressão
    """
    console.print(Panel.fit(
        "[bold]Gestão de Custos e Otimização de Tokens[/bold]\n"
        "Seleção de modelo, cache semântico e compressão de histórico",
        title="💰 Módulo 21 — Custos e Tokens",
        border_style="yellow",
    ))

    rastreador = RastreadorCusto(budget_usd=0.005)
    cache = CacheSemantico()

    # Tarefas de complexidade variada
    tarefas = [
        ("Qual o valor do boleto?", None),
        ("Extraia valor, banco e vencimento do boleto", None),
        ("O boleto está vencido? Aplique multa e juros", None),
        ("Analise se há indícios de fraude no boleto", None),
        ("Qual o valor do boleto?", None),  # repetida
        ("Extraia valor, banco e vencimento do boleto", None),  # repetida
    ]

    console.print("\n[bold]── Processando tarefas ──[/bold]")

    for prompt, _ in tarefas:
        complexidade = classificar_complexidade(prompt)
        modelo = selecionar_modelo(complexidade)

        # Tenta cache primeiro
        resposta_cache = cache.buscar(prompt)

        if resposta_cache:
            rastreador.registrar(
                modelo=modelo,
                tarefa=prompt[:40],
                tokens_entrada=0,
                tokens_saida=0,
                cache_hit=True,
            )
            console.print(
                f"  [green]CACHE HIT[/green] "
                f"[dim]{prompt[:50]}…[/dim]"
            )
        else:
            resposta, tok_in, tok_out = _simular_chamada_llm(
                modelo, prompt, prompt
            )
            cache.armazenar(prompt, resposta, tok_out)
            custo = rastreador.registrar(
                modelo=modelo,
                tarefa=prompt[:40],
                tokens_entrada=tok_in,
                tokens_saida=tok_out,
            )
            cor_modelo = (
                "green" if "8b" in modelo
                else "yellow" if "70b" in modelo
                else "red"
            )
            console.print(
                f"  [dim]complexidade={complexidade}[/dim] "
                f"modelo=[{cor_modelo}]{modelo.split('-')[1]}…"
                f"[/{cor_modelo}] "
                f"tokens={tok_in}+{tok_out} "
                f"[yellow]${custo:.6f}[/yellow]"
            )

        if rastreador.budget_excedido():
            console.print(
                "\n[bold red]⚠ Budget esgotado! "
                "Parando processamento.[/bold red]"
            )
            break

    # Relatório de custo
    console.print("\n[bold]── Relatório de Custo ──[/bold]")
    rastreador.relatorio()

    # Estatísticas do cache
    stats = cache.estatisticas()
    console.print(
        f"\n  Cache: {stats['entradas']} entrada(s), "
        f"{stats['total_hits']} hit(s), "
        f"{stats['tokens_economizados']} tokens economizados"
    )

    # Demo de compressão de histórico
    console.print("\n[bold]── Compressão de Histórico ──[/bold]")
    historico = []
    for i in range(1, 8):
        historico.append(
            {"role": "user", "content": f"Pergunta {i} sobre o boleto"}
        )
        historico.append({
            "role": "assistant",
            "content": f"Resposta {i}: valor R$ {i * 100:.2f}",
        })

    antes = contar_tokens_estimado(historico)
    comprimido = comprimir_historico(historico, max_mensagens=10)
    depois = contar_tokens_estimado(comprimido)

    console.print(
        f"  Histórico: {len(historico)} msg → "
        f"{len(comprimido)} msg após compressão"
    )
    console.print(
        f"  Tokens estimados: ~{antes} → ~{depois} "
        f"([green]-{antes - depois} tokens[/green])"
    )

    console.print("\n💡 [bold yellow]Dica:[/bold yellow]")
    console.print(
        "  Em produção, use tiktoken para contar tokens exatos\n"
        "  e llm-guard ou similar para compressão automática."
    )


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    demo_custos()
