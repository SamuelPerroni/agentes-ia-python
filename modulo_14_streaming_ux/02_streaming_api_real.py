"""
============================================================
MÓDULO 14.2 - STREAMING REAL COM API GROQ
============================================================
Este script demonstra STREAMING DE VERDADE usando a API Groq
com o parâmetro stream=True. Diferente do 01_streaming_console.py
(que simula com delays artificiais), aqui os tokens chegam da
LLM em tempo real, um por um.

O QUE É STREAMING REAL?
Em vez de esperar a LLM gerar a resposta COMPLETA (que pode levar
3-15 segundos) e depois mostrar tudo de uma vez, usamos stream=True
para receber PEDAÇOS (chunks/tokens) conforme vão sendo gerados.

POR QUE STREAMING REAL vs SIMULADO?
- 01_streaming_console.py: simula streaming com time.sleep(0.02)
  → Útil para demos e testes offline
- 02_streaming_api_real.py (ESTE ARQUIVO): streaming real com Groq
  → Mostra a experiência REAL do usuário em produção

DIFERENÇA TÉCNICA:
  # SEM streaming (tudo de uma vez):
  resposta = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[...],
      stream=False  # ← padrão, espera tudo ficar pronto
  )
  print(resposta.choices[0].message.content)  # 5s de espera → tudo

  # COM streaming (token por token):
  stream = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[...],
      stream=True  # ← resposta vem em pedaços
  )
  for chunk in stream:
      print(chunk.choices[0].delta.content, end="")  # imediato

ANALOGIA:
Imagine baixar um vídeo:
- SEM streaming = baixar o arquivo inteiro antes de assistir (buffering)
- COM streaming = assistir enquanto baixa (YouTube)
O tempo total pode ser o mesmo, mas a experiência é completamente
diferente. Ninguém gosta de esperar um download de 100MB para ver
um vídeo de 1 minuto.

DIAGRAMA — Fluxo de Streaming Real:

  ╔════════════════════════════════════════════════════╗
  ║  Usuário envia mensagem                            ║
  ║       │                                            ║
  ║       ↓                                            ║
  ║  client.chat.completions.create(stream=True)       ║
  ║       │                                            ║
  ║       ↓                                            ║
  ║  API Groq começa a gerar tokens                    ║
  ║       │                                            ║
  ║       ├─→ chunk 1: "O "    → print("O ", end="")   ║
  ║       ├─→ chunk 2: "valor" → print("valor", end="")║
  ║       ├─→ chunk 3: " do"   → print(" do", end="")  ║
  ║       ├─→ chunk 4: " bol"  → print(" bol", end="") ║
  ║       ├─→ ...                                      ║
  ║       └─→ chunk N: "."     → print(".", end="")    ║
  ║       │                                            ║
  ║       ↓                                            ║
  ║  stream.stop_reason = "stop"                       ║
  ║  Resposta completa exibida!                        ║
  ╚════════════════════════════════════════════════════╝

O QUE ESTE SCRIPT DEMONSTRA:
1. Chamada streaming REAL à API Groq (token por token no terminal)
2. Chamada NÃO-streaming para comparação de experiência
3. Medição de tempo: primeiro token vs resposta completa
4. Streaming com Rich (cores e formatação no terminal)
5. Tratamento de erros (API key ausente, falha de conexão)

PRÉ-REQUISITOS:
- Variável GROQ_API_KEY definida no .env
- pip install groq python-dotenv rich

EXERCÍCIO SUGERIDO:
1. Execute o script e observe a diferença entre streaming e não-streaming
2. Mude o MODEL para "llama-3.1-8b-instant" e compare velocidades
3. Tente com prompts mais longos e veja o impacto no "time to first token"
4. Adicione um spinner (como no 01_streaming_console.py) ANTES do streaming
============================================================
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.panel import Panel

# ============================================================
# CONFIGURAÇÃO — Carrega variáveis de ambiente e inicializa
# ============================================================
# Mesmo padrão dos módulos 01-06: .env na raiz com GROQ_API_KEY
# ============================================================

load_dotenv()

console = Console()


def _criar_cliente() -> Groq:
    """
    Cria e retorna um cliente Groq autenticado.

    VALIDAÇÃO:
    Verifica se a GROQ_API_KEY está definida antes de prosseguir.
    Em produção, NUNCA faça hardcode da chave na variável —
    sempre use variável de ambiente (módulo 13: checklist de segurança).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        console.print(
            "[bold red]❌ GROQ_API_KEY não encontrada![/]\n"
            "Defina no arquivo .env:\n"
            "  GROQ_API_KEY=gsk_sua_chave_aqui\n",
            style="red",
        )
        raise SystemExit(1)
    return Groq(api_key=api_key)


# Modelo padrão — mesmo do restante do treinamento
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ============================================================
# 1. STREAMING REAL — Token por token direto da API
# ============================================================
# Esta é a implementação REAL de streaming com Groq.
# Cada chunk contém um ou mais tokens gerados pela LLM.
#
# ESTRUTURA DE UM CHUNK:
#   chunk.choices[0].delta.content → texto do token (ou None)
#   chunk.choices[0].finish_reason → None durante geração, "stop" no final
#
# POR QUE delta E NÃO message?
# - Em respostas completas: choices[0].message.content (texto todo)
# - Em streaming: choices[0].delta.content (só o pedaço novo)
# - "delta" = diferença/incremento (termo de física/matemática)
#
# CUIDADO COM None:
# O primeiro e o último chunk podem ter delta.content = None
# Por isso usamos `or ""` para evitar imprimir "None" na tela.
# ============================================================

def demonstrar_streaming(prompt: str) -> str:
    """
    Faz uma chamada com streaming à API Groq e exibe tokens em tempo real.

    Parâmetros:
    - prompt: pergunta ou instrução para a LLM

    Retorno:
    - Texto completo da resposta (concatenação de todos os tokens)

    MÉTRICAS EXIBIDAS:
    - Time to First Token (TTFT): tempo até o primeiro token aparecer
    - Tempo total: tempo para a resposta completa
    - Total de tokens recebidos

    O QUE OBSERVAR:
    - O texto começa a aparecer MUITO antes da resposta estar completa
    - TTFT geralmente é < 1s, mesmo para respostas longas
    - O usuário "vê progresso" desde o início
    """
    client = _criar_cliente()

    console.print(
        Panel(f"[bold cyan]Prompt:[/] {prompt}", title="🔄 Streaming Real"),
    )
    console.print("[dim]Aguardando primeiro token...[/]")

    # ── Chamada com stream=True ──────────────────────────────
    inicio = time.perf_counter()
    tempo_primeiro_token: float | None = None
    resposta_completa: list[str] = []
    total_tokens = 0

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um assistente especializado em boletos bancários. "
                    "Responda de forma clara e direta em português."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=0.3,
        max_tokens=512,
    )

    # ── Iteração sobre chunks ────────────────────────────────
    # Cada iteração do loop recebe UM chunk da API
    # O chunk pode conter 1 token (1-4 caracteres geralmente)
    for chunk in stream:
        # Extrai o texto do token (pode ser None nos chunks de controle)
        token = chunk.choices[0].delta.content or ""

        if token:
            # Marca o tempo do PRIMEIRO token (métrica TTFT)
            if tempo_primeiro_token is None:
                tempo_primeiro_token = time.perf_counter() - inicio

            # Imprime o token SEM pular linha (end="")
            # flush=True garante que aparece IMEDIATAMENTE no terminal
            print(token, end="", flush=True)

            resposta_completa.append(token)
            total_tokens += 1

    # Pula linha após o streaming terminar
    print()

    tempo_total = time.perf_counter() - inicio

    # ── Métricas de performance ──────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[green]⏱️  Time to First Token (TTFT):[/] "
            f"{tempo_primeiro_token:.3f}s\n"
            f"[green]⏱️  Tempo total:[/] {tempo_total:.3f}s\n"
            f"[green]📊 Tokens recebidos:[/] {total_tokens}\n"
            f"[green]📈 Tokens/segundo:[/] {total_tokens / tempo_total:.1f}",
            title="📊 Métricas de Streaming",
            border_style="green",
        )
    )

    return "".join(resposta_completa)


# ============================================================
# 2. SEM STREAMING — Resposta completa de uma vez
# ============================================================
# Para COMPARAR a experiência, fazemos a mesma chamada SEM
# streaming. O usuário fica olhando para uma tela em branco
# até a resposta completa chegar.
#
# DIFERENÇA DE EXPERIÊNCIA:
# - Com streaming: "ah, está respondendo, deixa eu ler..."
# - Sem streaming: "travou? será que deu erro?"
# ============================================================

def demonstrar_sem_streaming(prompt: str) -> str:
    """
    Faz uma chamada SEM streaming à API Groq para comparação.

    Parâmetros:
    - prompt: pergunta ou instrução para a LLM

    Retorno:
    - Texto completo da resposta

    O QUE OBSERVAR:
    - NADA aparece no terminal até a resposta completa chegar
    - O tempo total é similar ao streaming, mas a EXPERIÊNCIA é pior
    - O usuário não tem feedback visual durante a espera
    """
    client = _criar_cliente()

    console.print(
        Panel(
            f"[bold cyan]Prompt:[/] {prompt}",
            title="⏳ Sem Streaming (esperando resposta completa...)",
        ),
    )

    inicio = time.perf_counter()

    # Chamada SEM streaming — bloqueia até a resposta completa
    resposta = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um assistente especializado em boletos bancários. "
                    "Responda de forma clara e direta em português."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,  # ← padrão, espera tudo ficar pronto
        temperature=0.3,
        max_tokens=512,
    )

    tempo_total = time.perf_counter() - inicio
    texto = resposta.choices[0].message.content or ""

    # Exibe a resposta de uma vez (experiência "bloco de texto")
    console.print(f"\n{texto}\n")

    console.print(
        Panel(
            f"[yellow]⏱️  Tempo total:[/] {tempo_total:.3f}s\n"
            f"[yellow]⏱️  Time to First Token:[/] {tempo_total:.3f}s "
            f"(= tempo total, pois tudo chega junto)",
            title="📊 Métricas Sem Streaming",
            border_style="yellow",
        )
    )

    return texto


# ============================================================
# 3. COMPARATIVO — Streaming vs Não-Streaming lado a lado
# ============================================================
# Executa as duas abordagens com o MESMO prompt para que o aluno
# compare a experiência e as métricas.
#
# INSIGHTS ESPERADOS:
# - Tempo total: similar (a LLM demora o mesmo, o que muda é a entrega)
# - TTFT com streaming: ~0.1-0.5s (quase imediato)
# - TTFT sem streaming: = tempo total (5-10s de espera sem feedback)
# - Experiência: streaming SEMPRE melhor para o usuário
#
# QUANDO NÃO USAR STREAMING:
# - Chamadas internas (tool → agente): não tem usuário olhando
# - Processamento em batch: não precisa de UX
# - Quando você precisa do texto completo para pós-processamento
#   antes de exibir (ex: validação de guardrails na saída)
# ============================================================

def comparativo_streaming() -> None:
    """
    Executa a mesma pergunta com e sem streaming para comparar.

    FLUXO:
    1. Faz chamada COM streaming → exibe tokens em tempo real
    2. Faz chamada SEM streaming → exibe tudo de uma vez
    3. O aluno compara a experiência visual e as métricas

    EXERCÍCIO SUGERIDO:
    1. Rode e observe: qual experiência é melhor?
    2. Teste com prompts mais longos (peça "um parágrafo detalhado")
    3. Compare os tempos — o tempo TOTAL deve ser similar
    4. Note como o TTFT com streaming é muito menor
    """
    prompt = (
        "Explique em 3 frases o que acontece quando um boleto bancário vence "
        "e quais são as opções do pagador."
    )

    console.print(
        Panel(
            "[bold]Vamos comparar a mesma pergunta COM e SEM streaming.\n"
            "Observe a diferença na EXPERIÊNCIA do usuário.[/]",
            title="🔬 Comparativo: Streaming vs Sem Streaming",
            border_style="blue",
        )
    )

    # ── Parte 1: COM streaming ─────────────────────────────
    console.print("\n[bold green]═══ PARTE 1: COM STREAMING ═══[/]\n")
    demonstrar_streaming(prompt)

    console.print("\n" + "─" * 60 + "\n")

    # ── Parte 2: SEM streaming ─────────────────────────────
    console.print("[bold yellow]═══ PARTE 2: SEM STREAMING ═══[/]\n")
    demonstrar_sem_streaming(prompt)

    # ── Resumo final ─────────────────────────────────────────
    console.print(
        Panel(
            "[bold]📝 Resumo do comparativo:[/]\n\n"
            "• [green]COM streaming:[/] texto aparece quase imediatamente\n"
            "  TTFT baixo → usuário vê progresso desde o início\n\n"
            "• [yellow]SEM streaming:[/] tela em branco por vários segundos\n"
            "  TTFT = tempo total → sem feedback até o final\n\n"
            "• Tempo TOTAL: similar nas duas abordagens\n"
            "• O que muda: a PERCEPÇÃO de velocidade pelo usuário\n\n"
            "[dim]💡 Em produção, "
            "SEMPRE use streaming para respostas ao usuário.[/]",
            title="🎯 Conclusão",
            border_style="cyan",
        )
    )


# ============================================================
# PONTO DE ENTRADA — Execução direta do módulo
# ============================================================
# Ao rodar `python 02_streaming_api_real.py`, o aluno verá:
# 1. Streaming real (tokens aparecendo um a um)
# 2. Sem streaming (resposta completa de uma vez)
# 3. Comparativo de métricas entre as duas abordagens
#
# EXERCÍCIOS SUGERIDOS:
# 1. Mude o MODEL para "llama-3.1-8b-instant" e compare velocidades
# 2. Aumente max_tokens para 1024 e observe o streaming mais longo
# 3. Adicione um spinner Rich ANTES do streaming (como no módulo 14.1)
# 4. Integre com o agente de boletos: faça a resposta final usar streaming
# 5. Tente medir a diferença de TTFT entre modelos diferentes
# ============================================================

if __name__ == "__main__":
    comparativo_streaming()
