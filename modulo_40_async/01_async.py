"""
============================================================
MÓDULO 40.1 - PROCESSAMENTO ASSÍNCRONO COM ASYNCIO
============================================================
Em APA corporativa, processar 1 documento por vez é lento.
asyncio permite processar dezenas de documentos em
paralelo sem criar threads, usando o event loop do Python.

CONCEITO CHAVE:
Uma chamada de LLM leva ~1-3 segundos de I/O. Enquanto
aguarda a resposta, a CPU fica ociosa. Com asyncio,
o event loop usa esse tempo para iniciar outras chamadas.

COMPARAÇÃO:

  SÍNCRONO (sequencial):
  ──────────────────────
  Doc1 ──── espera LLM ────► resposta
                              Doc2 ──── espera LLM ────► resposta
  Tempo total: N × latência

  ASSÍNCRONO (asyncio):
  ─────────────────────
  Doc1 ─── aguarda ───────────────────────────► resposta
  Doc2 ─── aguarda ──────────────────────► resposta
  Doc3 ─── aguarda ───────────────────────────────► resposta
  Tempo total: ≈ max(latências) + overhead mínimo

PADRÕES COBERTOS:

  asyncio.gather()   → paraleliza N corrotinas
  asyncio.Semaphore  → limita concorrência (rate limit)
  asyncio.Queue      → fila produtor/consumidor
  async for          → streaming de respostas

QUANDO USAR:
  ✓ Processamento em lote (boletos, NF-es, contratos)
  ✓ Múltiplas chamadas de API em paralelo
  ✓ Consulta de vários sistemas ao mesmo tempo

QUANDO NÃO USAR:
  ✗ Código CPU-bound (use multiprocessing)
  ✗ Código legado síncrono que não pode ser refatorado

Tópicos cobertos:
1. Função async simulando chamada de LLM
2. asyncio.gather — processa lote em paralelo
3. asyncio.Semaphore — limita rate para não estourar API
4. asyncio.Queue — fila com produtor e consumidores
5. Comparação de tempo sequencial vs paralelo
6. Demo com lote de 10 boletos
============================================================
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. RESULTADO DE PROCESSAMENTO
# ============================================================

@dataclass
class ResultadoDoc:
    """Representa o resultado do processamento de um documento."""
    doc_id: str
    valor: Optional[float]
    duracao_ms: float
    erro: Optional[str] = None

    @property
    def sucesso(self) -> bool:
        """Indica se o processamento do documento foi bem-sucedido."""
        return self.erro is None


# ============================================================
# 2. CHAMADA DE LLM SIMULADA (ASYNC)
# ============================================================

async def _chamar_llm_async(
    texto: str,
    latencia_min: float = 0.3,
    latencia_max: float = 1.5,
) -> dict:
    """
    Simula chamada assíncrona ao LLM.
    Em produção: use httpx.AsyncClient ou o SDK oficial
    do Groq/OpenAI que já suporta async.

    Exemplo real:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/...",
                json={"messages": [...]},
                headers={"Authorization": f"Bearer {key}"},
            )
            return resp.json()
    """
    # Simula latência variável de I/O
    await asyncio.sleep(random.uniform(latencia_min, latencia_max))
    m = re.search(r"R\$\s*([\d.,]+)", texto)
    valor = None
    if m:
        try:
            valor = float(
                m.group(1).replace(".", "").replace(",", ".")
            )
        except ValueError:
            pass
    return {"valor": valor, "banco": "Banco Simulado"}


# ============================================================
# 3. PROCESSAR UM DOCUMENTO
# ============================================================

async def processar_documento(
    doc_id: str,
    texto: str,
    semaforo: asyncio.Semaphore,
) -> ResultadoDoc:
    """
    Processa um único documento com controle de
    concorrência via Semaphore.
    """
    t0 = time.perf_counter()
    async with semaforo:
        # Dentro do semaforo: no máximo N simultâneos
        try:
            dados = await _chamar_llm_async(texto)
            duracao_ms = (time.perf_counter() - t0) * 1000
            return ResultadoDoc(
                doc_id=doc_id,
                valor=dados.get("valor"),
                duracao_ms=round(duracao_ms, 1),
            )
        except Exception as exc:  # pylint: disable=broad-except
            duracao_ms = (time.perf_counter() - t0) * 1000
            return ResultadoDoc(
                doc_id=doc_id,
                valor=None,
                duracao_ms=round(duracao_ms, 1),
                erro=str(exc),
            )


# ============================================================
# 4. PROCESSAR LOTE — GATHER COM SEMAPHORE
# ============================================================

async def processar_lote_async(
    documentos: list[tuple[str, str]],
    max_concorrente: int = 5,
) -> list[ResultadoDoc]:
    """
    Processa N documentos em paralelo, limitando a
    max_concorrente chamadas simultâneas à API.

    Args:
        documentos: lista de (doc_id, texto)
        max_concorrente: limite de concorrência (rate limit)
    """
    semaforo = asyncio.Semaphore(max_concorrente)
    tarefas = [
        processar_documento(doc_id, texto, semaforo)
        for doc_id, texto in documentos
    ]
    # asyncio.gather executa todas as corrotinas de uma vez
    # e aguarda todas terminarem
    return list(await asyncio.gather(*tarefas))


# ============================================================
# 5. PROCESSAR SEQUENCIAL — PARA COMPARAÇÃO
# ============================================================

async def processar_sequencial(
    documentos: list[tuple[str, str]],
) -> list[ResultadoDoc]:
    """Versão sequencial — para medir diferença de tempo."""
    semaforo = asyncio.Semaphore(1)  # só 1 por vez
    resultados = []
    for doc_id, texto in documentos:
        r = await processar_documento(doc_id, texto, semaforo)
        resultados.append(r)
    return resultados


# ============================================================
# 6. FILA PRODUTOR/CONSUMIDOR (PIPELINE)
# ============================================================

async def produtor(
    fila: asyncio.Queue,  # type: ignore[type-arg]
    documentos: list[tuple[str, str]],
) -> None:
    """Enfileira documentos para processamento."""
    for item in documentos:
        await fila.put(item)
    # Sinaliza fim com sentinelas
    for _ in range(3):  # 3 consumidores
        await fila.put(None)


async def consumidor(
    worker_id: int,
    fila: asyncio.Queue,  # type: ignore[type-arg]
    resultados: list[ResultadoDoc],
    semaforo: asyncio.Semaphore,
) -> None:
    """Consome documentos da fila e processa."""
    while True:
        item = await fila.get()
        if item is None:
            break
        doc_id, texto = item
        r = await processar_documento(doc_id, texto, semaforo)
        resultados.append(r)
        console.print(
            f"    [dim]Worker-{worker_id} → "
            f"{doc_id}: R$ {r.valor}[/]"
        )


# ============================================================
# 7. DEMO
# ============================================================

def _gerar_documentos(n: int) -> list[tuple[str, str]]:
    random.seed(7)
    docs = []
    for i in range(n):
        valor = random.randint(500, 9000)
        docs.append((
            f"DOC-{i+1:04d}",
            f"Boleto R$ {valor:,.2f} venc 10/05/2026",
        ))
    return docs


async def _demo_async() -> None:
    documentos = _gerar_documentos(10)

    # --- Sequencial ---
    console.rule("[yellow]Modo Sequencial (1 por vez)")
    t0 = time.perf_counter()
    res_seq = await processar_sequencial(documentos)
    t_seq = time.perf_counter() - t0
    console.print(
        f"  Sequencial: {len(res_seq)} docs em "
        f"[red]{t_seq:.2f}s[/]"
    )

    # --- Paralelo com gather ---
    console.rule("[yellow]Modo Paralelo (gather + semaphore)")
    t0 = time.perf_counter()
    res_par = await processar_lote_async(
        documentos, max_concorrente=5
    )
    t_par = time.perf_counter() - t0
    speedup = t_seq / t_par if t_par > 0 else 1
    console.print(
        f"  Paralelo:   {len(res_par)} docs em "
        f"[green]{t_par:.2f}s[/] "
        f"([bold]{speedup:.1f}× mais rápido[/])"
    )

    # Tabela de resultados
    tabela = Table(header_style="bold magenta")
    tabela.add_column("Doc ID")
    tabela.add_column("Valor", justify="right")
    tabela.add_column("Duração (ms)", justify="right")
    tabela.add_column("Status")
    for r in res_par:
        tabela.add_row(
            r.doc_id,
            f"R$ {r.valor:,.2f}" if r.valor else "—",
            str(r.duracao_ms),
            "[green]OK[/]" if r.sucesso else "[red]ERRO[/]",
        )
    console.print(tabela)

    # --- Fila produtor/consumidor ---
    console.rule("[yellow]Modo Fila (3 workers em paralelo)")
    fila: asyncio.Queue = asyncio.Queue()  # type: ignore[type-arg]
    resultados_fila: list[ResultadoDoc] = []
    semaforo = asyncio.Semaphore(3)
    t0 = time.perf_counter()
    await asyncio.gather(
        produtor(fila, documentos[:6]),
        consumidor(1, fila, resultados_fila, semaforo),
        consumidor(2, fila, resultados_fila, semaforo),
        consumidor(3, fila, resultados_fila, semaforo),
    )
    t_fila = time.perf_counter() - t0
    console.print(
        f"  Fila: {len(resultados_fila)} docs em "
        f"[green]{t_fila:.2f}s[/]"
    )


def demo_async() -> None:
    """Demonstração do processamento assíncrono com asyncio."""
    console.print(
        Panel(
            "[bold]Módulo 40 — Processamento Assíncrono "
            "com asyncio[/]\n"
            "Processe lotes de documentos em paralelo "
            "sem criar threads",
            style="bold blue",
        )
    )
    asyncio.run(_demo_async())


if __name__ == "__main__":
    demo_async()
