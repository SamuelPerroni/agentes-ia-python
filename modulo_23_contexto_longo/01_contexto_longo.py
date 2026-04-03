"""
============================================================
MÓDULO 23.1 - GERENCIAMENTO DE JANELA DE CONTEXTO
============================================================
Neste módulo, aprendemos a lidar com documentos que não cabem
na janela de contexto do LLM e com históricos que crescem
indefinidamente.

CONCEITO CHAVE:
Todo LLM tem um limite de tokens (contexto). Documentos longos
como contratos, notas fiscais volumosas e histórico de processos
frequentemente ultrapassam esse limite.

LIMITES TÍPICOS (abril de 2026):
  llama-3.1-8b        →  128.000 tokens (~96.000 palavras)
  llama-3.3-70b       →  128.000 tokens
  gpt-4o              →  128.000 tokens
  claude-3-5-sonnet   →  200.000 tokens

PROBLEMA: mesmo 128k tokens não resolve tudo.
Um lote de 50 contratos de 20 páginas cada = ~1M tokens.

ESTRATÉGIAS (do mais simples ao mais sofisticado):

  ┌──────────────────────────────────────────────────────────┐
  │  ESTRATÉGIA          QUANDO USAR                        │
  │  ─────────────────────────────────────────────────────  │
  │  Truncar             Início/fim do doc tem o relevante  │
  │  Sliding Window      Precisa processar o doc inteiro    │
  │  Chunking + Map      Extração por seção, depois mescla  │
  │  Summarization Loop  Histórico de conversa longo        │
  │  RAG (Módulo 10)     Base de conhecimento grande        │
  └──────────────────────────────────────────────────────────┘

SLIDING WINDOW — processar documentos maiores que o contexto:

  Documento: [████████████████████████████████████████████]
                                                    (6000 tokens)
  Janela 1:  [████████████]                         (2000 tok)
  Janela 2:        [████████████]                   (overlap)
  Janela 3:              [████████████]

  Overlap: 200 tokens de sobreposição evitam perder informação
  na fronteira entre janelas.

MAP-REDUCE PARA EXTRAÇÃO EM LOTE:

  Documento
       │
       ├─▶ Chunk 1 ──▶ LLM ──▶ resultado_1
       ├─▶ Chunk 2 ──▶ LLM ──▶ resultado_2    ← MAP
       └─▶ Chunk 3 ──▶ LLM ──▶ resultado_3
                                      │
                              LLM (merge) ──▶ resultado_final  ← REDUCE

Tópicos cobertos:
1. Contagem e estimativa de tokens sem tiktoken
2. Divisão de texto em chunks com overlap configurável
3. Sliding window para processamento de documentos longos
4. Summarization loop para histórico de conversação
5. Map-reduce para extração em múltiplos chunks
6. Escolha de estratégia por tamanho do documento
============================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Console do Rich para output formatado
console = Console()


# ============================================================
# 1. CONTAGEM E ESTIMATIVA DE TOKENS
# ============================================================
# Em produção, use tiktoken (OpenAI) ou o tokenizer do seu
# provedor para contagem exata. Aqui usamos a regra prática
# de ~4 caracteres por token — suficiente para planejamento.
#
# INSTALAÇÃO PARA PRODUÇÃO:
#   pip install tiktoken
#
#   import tiktoken
#   enc = tiktoken.encoding_for_model("gpt-4o")
#   tokens = len(enc.encode(texto))
# ============================================================

CHARS_POR_TOKEN = 4  # estimativa conservadora


def estimar_tokens(texto: str) -> int:
    """
    Estima o número de tokens em um texto.

    Usa a heurística de 4 caracteres por token — válida para
    textos em português com vocabulário corporativo.

    Parâmetros:
    - texto: string de qualquer comprimento

    Retorna:
    - Estimativa de tokens (inteiro)
    """
    return max(1, len(texto) // CHARS_POR_TOKEN)


def cabe_no_contexto(
    texto: str,
    limite_tokens: int = 4096,
    reserva_saida: int = 512,
) -> bool:
    """
    Verifica se o texto cabe na janela de contexto.

    Parâmetros:
    - texto: conteúdo a ser enviado ao LLM
    - limite_tokens: janela do modelo (padrão conservador: 4096)
    - reserva_saida: tokens reservados para a resposta gerada

    Retorna:
    - True se cabe, False se precisa de estratégia de divisão
    """
    espaco_disponivel = limite_tokens - reserva_saida
    return estimar_tokens(texto) <= espaco_disponivel


# ============================================================
# 2. CHUNKING COM OVERLAP
# ============================================================
# Divide um texto em blocos de tamanho fixo (em tokens),
# com sobreposição configurável entre blocos consecutivos.
#
# OVERLAP: garante que frases que ficam na fronteira entre
# dois chunks sejam vistas em contexto completo por pelo menos
# um dos chunks.
#
# OVERLAP RECOMENDADO:
# - Extração de campos: 10-15% do tamanho do chunk
# - Summarização: 5-10% (contexto menos crítico)
# - Q&A sobre documento: 20-25% (contexto é crítico)
# ============================================================

@dataclass
class Chunk:
    """Bloco de texto com metadados de posição no documento."""

    indice: int
    texto: str
    tokens_estimados: int
    inicio_char: int
    fim_char: int


def dividir_em_chunks(
    texto: str,
    tamanho_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """
    Divide um texto em chunks com overlap por fronteira de parágrafo.

    Prioriza dividir em parágrafos (quebras de linha dupla) para
    manter a coesão semântica. Só corta no meio de parágrafo se
    ele exceder o tamanho máximo.

    Parâmetros:
    - texto: documento a ser dividido
    - tamanho_tokens: máximo de tokens por chunk
    - overlap_tokens: tokens de sobreposição entre chunks

    Retorna:
    - Lista de Chunk ordenada por posição no documento
    """
    tamanho_chars = tamanho_tokens * CHARS_POR_TOKEN
    overlap_chars = overlap_tokens * CHARS_POR_TOKEN

    chunks: list[Chunk] = []
    inicio = 0
    indice = 0

    while inicio < len(texto):
        fim = min(inicio + tamanho_chars, len(texto))

        # Tenta terminar em fim de parágrafo se não for o último chunk
        if fim < len(texto):
            quebra = texto.rfind("\n\n", inicio, fim)
            if quebra != -1 and quebra > inicio + tamanho_chars // 2:
                fim = quebra + 2  # inclui a quebra

        trecho = texto[inicio:fim]
        chunks.append(Chunk(
            indice=indice,
            texto=trecho,
            tokens_estimados=estimar_tokens(trecho),
            inicio_char=inicio,
            fim_char=fim,
        ))

        # Avança com overlap: recua overlap_chars a partir do fim
        inicio = max(inicio + 1, fim - overlap_chars)
        indice += 1

    return chunks


# ============================================================
# 3. SLIDING WINDOW PARA DOCUMENTOS LONGOS
# ============================================================
# Processa um documento grande janela por janela, acumulando
# os resultados parciais em uma lista para processamento final.
#
# USE QUANDO: precisa extrair informação de um documento
# que não cabe no contexto, mas a informação pode estar
# em qualquer parte do texto.
# ============================================================

def processar_com_sliding_window(
    texto: str,
    funcao_processar: Any,
    tamanho_janela_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[str]:
    """
    Processa um documento longo janela por janela.

    Para cada chunk, chama funcao_processar e acumula
    os resultados. Ideal para extração de campos quando
    você não sabe em qual parte do documento eles estão.

    Parâmetros:
    - texto: documento completo
    - funcao_processar: f(chunk_texto, indice, total) → str
    - tamanho_janela_tokens: tokens por janela
    - overlap_tokens: sobreposição entre janelas

    Retorna:
    - Lista de resultados parciais (um por chunk)
    """
    chunks = dividir_em_chunks(texto, tamanho_janela_tokens, overlap_tokens)
    resultados: list[str] = []

    console.print(
        f"  Documento: {estimar_tokens(texto)} tokens → "
        f"{len(chunks)} chunks de ~{tamanho_janela_tokens} tokens"
    )

    for chunk in chunks:
        resultado = funcao_processar(chunk.texto, chunk.indice, len(chunks))
        if resultado:
            resultados.append(resultado)

    return resultados


# ============================================================
# 4. MAP-REDUCE PARA EXTRAÇÃO EM LOTE
# ============================================================
# MAP:    processa cada chunk independentemente → extrações parciais
# REDUCE: consolida todas as extrações em um resultado final
#
# VANTAGEM: os chunks MAP podem ser processados em paralelo
# (ver módulo 19 — ThreadPoolExecutor).
# ============================================================

@dataclass
class ResultadoMapReduce:
    """Resultado final do pipeline Map-Reduce."""

    campos_extraidos: dict[str, Any]
    chunks_processados: int
    tokens_totais: int
    conflitos: list[str]  # campos com valores conflitantes entre chunks


def _mergar_extrações(
    extrações: list[dict[str, Any]],
) -> ResultadoMapReduce:
    """
    Consolida extrações parciais de múltiplos chunks.

    Estratégia:
    - Valor numérico: usa o maior (ex: valor do boleto)
    - String: usa o primeiro não-nulo encontrado
    - Conflito: registra mas não descarta

    Parâmetros:
    - extrações: lista de dicts com campos extraídos por chunk

    Retorna:
    - ResultadoMapReduce com campos consolidados
    """
    consolidado: dict[str, Any] = {}
    conflitos: list[str] = []

    campos_numericos = {"valor", "multa", "juros"}

    for campo in set(k for e in extrações for k in e):
        valores = [
            e[campo] for e in extrações
            if campo in e and e[campo] is not None
        ]
        if not valores:
            consolidado[campo] = None
            continue

        if campo in campos_numericos:
            # Pega o maior valor numérico
            try:
                consolidado[campo] = max(float(v) for v in valores)
            except (ValueError, TypeError):
                consolidado[campo] = valores[0]
        else:
            # Pega o primeiro não-vazio
            consolidado[campo] = valores[0]
            # Registra conflito se há mais de um valor distinto
            if len(set(str(v) for v in valores)) > 1:
                conflitos.append(
                    f"{campo}: {list(set(str(v) for v in valores))}"
                )

    return ResultadoMapReduce(
        campos_extraidos=consolidado,
        chunks_processados=len(extrações),
        tokens_totais=0,
        conflitos=conflitos,
    )


# ============================================================
# 5. SUMMARIZATION LOOP PARA HISTÓRICO
# ============================================================
# Quando o histórico de conversa cresce além do limite,
# sumarizamos as mensagens mais antigas para liberar espaço.
# O sumário preserva o contexto sem consumir toda a janela.
# ============================================================

@dataclass
class GerenciadorHistorico:
    """
    Gerencia o histórico de conversação com compressão automática.

    Monitora o tamanho estimado do histórico em tokens e
    aciona sumarização quando o limite configurado é atingido.

    PARÂMETROS:
    - limite_tokens: janela máxima disponível para o histórico
    - manter_recentes: mínimo de mensagens recentes a preservar
    - funcao_sumarizar: f(mensagens) → str do sumário
    """

    limite_tokens: int = 3000
    manter_recentes: int = 6
    _historico: list[dict[str, str]] = field(default_factory=list)
    _compressoes: int = 0

    def adicionar(self, role: str, content: str) -> None:
        """
        Adiciona mensagem ao histórico e aciona compressão se necessário.

        Parâmetros:
        - role: "user", "assistant" ou "system"
        - content: texto da mensagem
        """
        self._historico.append({"role": role, "content": content})
        self._compactar_se_necessario()

    def _tokens_atuais(self) -> int:
        return sum(
            estimar_tokens(m["content"])
            for m in self._historico
        )

    def _compactar_se_necessario(self) -> None:
        """Comprime o histórico se ultrapassou o limite de tokens."""
        if self._tokens_atuais() <= self.limite_tokens:
            return
        if len(self._historico) <= self.manter_recentes:
            return

        recentes = self._historico[-self.manter_recentes:]
        antigas = self._historico[:-self.manter_recentes]

        # Simula sumarização (em produção: chama o LLM)
        pontos = [
            m["content"][:80]
            for m in antigas
            if m["role"] == "assistant"
        ]
        sumario = (
            f"[SUMÁRIO DE {len(antigas)} mensagens anteriores]\n"
            + "\n".join(f"• {p}" for p in pontos[:4])
        )

        self._historico = [
            {"role": "system", "content": sumario},
            *recentes,
        ]
        self._compressoes += 1

    def obter(self) -> list[dict[str, str]]:
        """Retorna o histórico atual (comprimido se necessário)."""
        return self._historico.copy()

    @property
    def tamanho_tokens(self) -> int:
        """Retorna a estimativa de tokens do histórico atual."""
        return self._tokens_atuais()

    @property
    def compressoes_realizadas(self) -> int:
        """Retorna o número de compressões realizadas até agora."""
        return self._compressoes


# ============================================================
# DEMO COMPLETA — Processamento de contrato longo
# ============================================================

def _gerar_contrato_simulado(paginas: int = 15) -> str:
    """Gera um contrato simulado com múltiplas páginas."""
    secoes = []
    for i in range(1, paginas + 1):
        secoes.append(
            f"\n\nCláusula {i}: Disposições Gerais — Seção {i}\n"
            f"As partes acordam que, nos termos da legislação "
            f"vigente, especialmente a Lei nº 10.406/2002 "
            f"(Código Civil), as obrigações aqui previstas "
            f"deverão ser cumpridas no prazo de 30 dias úteis "
            f"contados da assinatura do presente instrumento. "
            f"O valor de parcela referente à cláusula {i} "
            f"corresponde a R$ {i * 100:.2f}, totalizando "
            f"os compromissos financeiros da parte contratante. "
            f"Eventuais inadimplências sujeitam-se à multa de 2%"
            f" e juros de 1% ao mês sobre o saldo devedor."
        )
    return "CONTRATO DE PRESTAÇÃO DE SERVIÇOS\n" + "".join(secoes)


def demo_contexto_longo() -> None:
    """
    Demonstra estratégias de gerenciamento de janela de contexto.

    ETAPAS:
    1. Gera contrato simulado de 15 páginas (~3.600 tokens)
    2. Divide em chunks com overlap e mostra a estratégia
    3. Demonstra sliding window com extração simulada
    4. Demonstra summarization loop do histórico
    5. Exibe relatório de estratégias por tamanho

    OBSERVE NO OUTPUT:
    - Cada chunk tem overlap com o anterior (nunca perde fronteira)
    - O summarization loop comprime mensagens antigas preservando
      o contexto relevante
    - O relatório sugere a estratégia correta por tamanho

    EXERCÍCIO SUGERIDO:
    1. Aumente para 50 páginas e veja quantos chunks são criados
    2. Reduza o overlap para 0 e observe se a cobertura piora
    3. Adicione 20 mensagens ao histórico e veja quantas compressões
       são acionadas
    """
    console.print(Panel.fit(
        "[bold]Gerenciamento de Janela de Contexto[/bold]\n"
        "Chunking, sliding window e summarization loop",
        title="📄 Módulo 23 — Contexto Longo",
        border_style="cyan",
    ))

    # ── Seção 1: Análise do documento ──────────────────────────
    console.print("\n[bold]── 1. Análise do documento ──[/bold]")
    contrato = _gerar_contrato_simulado(paginas=15)
    tokens_doc = estimar_tokens(contrato)

    console.print(f"  Tamanho: {len(contrato):,} chars | "
                  f"~{tokens_doc:,} tokens")

    cabe = cabe_no_contexto(contrato, limite_tokens=4096, reserva_saida=512)
    status = (
        "[green]cabe no contexto[/green]"
        if cabe else "[yellow]precisa chunking[/yellow]"
    )
    console.print(f"  Janela de 4096 tokens: {status}")

    # ── Seção 2: Chunking ───────────────────────────────────────
    console.print("\n[bold]── 2. Chunking com overlap ──[/bold]")
    chunks = dividir_em_chunks(
        contrato,
        tamanho_tokens=600,
        overlap_tokens=80,
    )
    console.print(f"  {len(chunks)} chunks gerados "
                  f"(~600 tokens cada, 80 overlap)")

    tabela_chunks = Table(show_header=True)
    tabela_chunks.add_column("Chunk", justify="center", style="bold")
    tabela_chunks.add_column("Tokens", justify="right")
    tabela_chunks.add_column("Chars", justify="right")
    tabela_chunks.add_column("Início do texto", style="dim")

    for c in chunks[:5]:
        tabela_chunks.add_row(
            f"#{c.indice + 1}",
            str(c.tokens_estimados),
            str(c.fim_char - c.inicio_char),
            c.texto[:55].replace("\n", " ") + "…",
        )
    if len(chunks) > 5:
        tabela_chunks.add_row(
            f"… +{len(chunks) - 5}", "…", "…", "…",
        )
    console.print(tabela_chunks)

    # ── Seção 3: Sliding window ─────────────────────────────────
    console.print("\n[bold]── 3. Sliding window — extração simulada ──[/bold]")

    extrações_parciais: list[dict[str, Any]] = []

    def _extrair_chunk(texto: str, idx: int, total: int) -> str:
        """Simula extração de valor e multa de um chunk."""
        valores = re.findall(r"R\$\s*([\d.]+,\d{2})", texto)
        resultado = None
        if valores:
            val_float = float(valores[0].replace(".", "").replace(",", "."))
            extrações_parciais.append({
                "valor": val_float,
                "banco": "Banco Simulado",
                "vencimento": "2026-12-31",
            })
            resultado = f"chunk {idx + 1}/{total}: R$ {valores[0]}"
        return resultado or ""

    resultados = processar_com_sliding_window(
        contrato,
        _extrair_chunk,
        tamanho_janela_tokens=600,
        overlap_tokens=80,
    )
    console.print(f"  {len(resultados)} chunks com dados extraídos "
                  f"de {len(chunks)} chunks totais")

    # ── Seção 4: Map-Reduce ─────────────────────────────────────
    if extrações_parciais:
        console.print("\n[bold]── 4. Consolidação Map-Reduce ──[/bold]")
        resultado_final = _mergar_extrações(extrações_parciais)
        console.print(
            f"  chunks processados: {resultado_final.chunks_processados}\n"
            f"  campos consolidados: {resultado_final.campos_extraidos}\n"
            f"  conflitos: {resultado_final.conflitos or 'nenhum'}"
        )

    # ── Seção 5: Summarization loop do histórico ────────────────
    console.print("\n[bold]── 5. Summarization loop de histórico ──[/bold]")
    hist = GerenciadorHistorico(limite_tokens=600, manter_recentes=4)

    for i in range(1, 13):
        hist.adicionar(
            "user",
            f"Pergunta {i}: sobre a cláusula {i} do contrato"
        )
        hist.adicionar(
            "assistant",
            f"Resposta {i}: a cláusula {i} "
            f"estabelece multa de 2% e juros de 1% ao mês",
        )

    console.print(
        f"  {12 * 2} mensagens inseridas → "
        f"{len(hist.obter())} no histórico atual\n"
        f"  compressões realizadas: {hist.compressoes_realizadas}\n"
        f"  tokens atuais: ~{hist.tamanho_tokens}"
    )

    # ── Relatório de estratégias ────────────────────────────────
    console.print("\n[bold]── Guia de escolha de estratégia ──[/bold]")
    guia = Table(show_header=True)
    guia.add_column("Tamanho do doc", style="cyan")
    guia.add_column("Estratégia recomendada")
    guia.add_column("Custos")

    guia.add_row(
        "< 2k tokens",
        "Enviar completo",
        "[green]1 chamada[/green]",
    )
    guia.add_row(
        "2k – 10k tokens",
        "Sliding window sequencial",
        "[yellow]N chamadas[/yellow]",
    )
    guia.add_row(
        "10k – 100k tokens",
        "Map-Reduce paralelo",
        "[yellow]N+1 chamadas[/yellow]",
    )
    guia.add_row(
        "> 100k tokens / base de docs",
        "RAG (ver Módulo 10)",
        "[green]apenas chunks relevantes[/green]",
    )
    console.print(guia)

    console.print("\n💡 [bold yellow]Dica:[/bold yellow]")
    console.print(
        "  Use tiktoken para contar tokens exatos antes de\n"
        "  decidir a estratégia. A estimativa de 4 chars/token\n"
        "  pode ter variação de ±20% dependendo do idioma."
    )


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    demo_contexto_longo()
