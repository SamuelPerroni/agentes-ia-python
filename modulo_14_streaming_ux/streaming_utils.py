"""
============================================================
MÓDULO 14 - UTILITÁRIOS DE STREAMING (streaming_utils.py)
============================================================
Este arquivo contém funções REUTILIZÁVEIS para exibição incremental
de respostas no terminal. É importado pelo agente de boletos (módulo 06)
e pelo demo (01_streaming_console.py).

O QUE É STREAMING DE RESPOSTA?
Em vez de esperar a LLM gerar a resposta COMPLETA e depois exibir
tudo de uma vez (que pode demorar 5-10 segundos), mostramos a resposta
CARACTERE POR CARACTERE (ou chunk por chunk) enquanto ela é gerada.

POR QUE STREAMING MELHORA A UX?
- Tempo percebido: 5s "congelado" parece mais lento que 5s de texto
  aparecendo gradualmente — mesmo que o tempo real seja o mesmo
- Feedback visual: o usuário vê que "algo está acontecendo"
- Interruptibilidade: o usuário pode cancelar se perceber que a resposta
  não é o que ele quer (em vez de esperar 10s para ver "resposta errada")

ANALOGIA:
Pense em um garçom:
- SEM streaming = garçom desaparece por 10 min, volta com o prato pronto
  (você fica ansioso: "será que ele esqueceu?")
- COM streaming = garçom traz entrada, depois o prato, depois a sobremesa
  (mesmo tempo total, mas você se sente atendido o tempo todo)

NESTE MÓDULO:
Implementamos streaming SIMULADO (caractere por caractere) para o terminal.
Em produção com a API Groq, usaríamos o parâmetro `stream=True`:

  # Exemplo com Groq streaming real:
  stream = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[...],
      stream=True  # ← habilita streaming
  )
  for chunk in stream:
      texto = chunk.choices[0].delta.content or ""
      print(texto, end="")

COMPONENTES:
1. gerar_chunks() - divide texto em pedaços de tamanho fixo
2. stream_texto() - exibe texto incrementalmente no console Rich
============================================================
"""

from __future__ import annotations

import time


# ============================================================
# 1. GERAÇÃO DE CHUNKS — Dividir texto em pedaços
# ============================================================
# Para simular streaming, dividimos o texto completo em pedaços
# (chunks) de tamanho fixo e exibimos cada um com um delay.
#
# POR QUE CHUNKS E NÃO CARACTERE POR CARACTERE?
# - Caractere por caractere é lento demais para textos longos
# - Chunks de 10-15 caracteres dão boa sensação de fluidez
# - Em streaming real, a LLM retorna tokens (1-4 chars cada),
#   que são equivalentes a chunks pequenos
#
# COMO FUNCIONA:
# "Valor atualizado: R$ 510" com tamanho=12 gera:
# ["Valor atuali", "zado: R$ 510"]
# ============================================================

def gerar_chunks(texto: str, tamanho: int = 12) -> list[str]:
    """
    Divide um texto em chunks de tamanho fixo para exibição incremental.

    Parâmetros:
    - texto: string completa a ser dividida
    - tamanho: quantidade de caracteres por chunk (default: 12)

    COMO FUNCIONA:
    - Usa slicing com step para criar fatias do texto
    - O último chunk pode ter menos caracteres que o tamanho definido
    - Chunks vazios não são gerados (range para quando > len)

    Retorno:
    Lista de strings, cada uma com até `tamanho` caracteres

    EXEMPLO:
    >>> gerar_chunks("Hello World!", 5)
    ['Hello', ' Worl', 'd!']
    """
    return [
        texto[indice:indice + tamanho]
        for indice in range(0, len(texto), tamanho)
    ]


# ============================================================
# 2. STREAM DE TEXTO — Exibição incremental no terminal
# ============================================================
# Simula o efeito de streaming exibindo chunks com delay entre eles.
# Usa console.print(end="") para não pular linha entre chunks.
#
# PARÂMETROS DE TUNING:
# - delay=0.01: muito rápido (quase instantâneo, bom para testes)
# - delay=0.02: velocidade natural de leitura (bom para demo)
# - delay=0.05: velocidade dramática (bom para apresentações)
#
# EM PRODUÇÃO:
# Não usamos delay artificial — o delay é o tempo real entre
# chunks recebidos da API (stream=True). A função ficaria:
#   for chunk in api_stream:
#       console.print(chunk.text, end="")
# ============================================================

def stream_texto(console, texto: str, delay: float = 0.01) -> None:
    """
    Exibe texto de forma incremental (streaming) no console Rich.

    Parâmetros:
    - console: instância de rich.console.Console para output
    - texto: string completa a ser exibida incrementalmente
    - delay: segundos entre cada chunk (default: 0.01)

    COMO FUNCIONA:
    1. Divide o texto em chunks via gerar_chunks()
    2. Para cada chunk, imprime sem pular linha (end="")
    3. Espera `delay` segundos entre cada chunk
    4. Ao final, imprime uma linha vazia (pula linha)

    EFEITO VISUAL:
    O texto aparece "digitando" no terminal, dando a sensação de
    que a resposta está sendo gerada em tempo real.

    ANALOGIA:
    É como um letreiro eletrônico: as letras vão aparecendo
    gradualmente em vez de piscar tudo de uma vez.
    """
    for chunk in gerar_chunks(texto):
        console.print(chunk, end="")
        time.sleep(delay)
    # Pula linha ao final para não grudar no próximo output
    console.print()
