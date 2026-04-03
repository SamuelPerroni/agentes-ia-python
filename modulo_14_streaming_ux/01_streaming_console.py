"""
============================================================
MÓDULO 14.1 - STREAMING E UX NO TERMINAL
============================================================
Neste módulo, vamos tornar o agente menos "caixa-preta" para o
usuário, mostrando etapas de processamento e resposta incremental.

CONCEITO CHAVE — A Percepção de Velocidade:
O tempo REAL de resposta importa, mas o tempo PERCEBIDO pelo
usuário importa MAIS. Duas estratégias melhoram a percepção:

  1. MOSTRAR ETAPAS: em vez de tela em branco por 5 segundos,
     mostrar "Analisando entrada...", "Chamando tools...", etc.
  2. STREAMING: mostrar a resposta ENQUANTO ela é gerada, caractere
     por caractere, em vez de esperar tudo ficar pronto.

POR QUE UX IMPORTA EM AGENTES?
- Agentes de IA podem levar 3-15 segundos para responder
- Um usuário percebe "lentidão" após ~2 segundos sem feedback
- Mostrar progresso reduz a frustração mesmo sem acelerar o processo
- É a diferença entre "travou?" e "ah, está processando, blz"

ANALOGIA:
Pense em uma barra de progresso no Windows:
- Sem barra: "travou? vou reiniciar" (frustração)
- Com barra: "30%... 60%... 90%... pronto!" (paciência)
O tempo real é o MESMO, mas a experiência é completamente diferente.

TÉCNICAS DEMONSTRADAS NESTE MÓDULO:

  ╔═══════════════════════════════════════════════════╗
  ║  1. STATUS SPINNER (Rich)                         ║
  ║     Mostra "Analisando entrada..." com animação   ║
  ║     rotativa enquanto o agente processa            ║
  ╠═══════════════════════════════════════════════════╣
  ║  2. STREAMING DE RESPOSTA                         ║
  ║     Texto aparece caractere por caractere          ║
  ║     (simulando tokens chegando da API)             ║
  ╠═══════════════════════════════════════════════════╣
  ║  3. ETAPAS NOMEADAS                               ║
  ║     Pipeline visível: entrada → guardrail →        ║
  ║     tools → risco → resposta                       ║
  ╚═══════════════════════════════════════════════════╝

COMO FUNCIONA EM PRODUÇÃO (Groq com stream=True):
  # Na API real, usamos o parâmetro stream=True
  stream = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[...],
      stream=True  # ← a resposta chega em pedaços
  )
  for chunk in stream:
      token = chunk.choices[0].delta.content or ""
      print(token, end="", flush=True)

Tópicos cobertos:
1. Exibição de etapas com spinner (Rich status)
2. Streaming simulado de resposta (caractere por caractere)
3. Pipeline visual do agente no terminal
4. Boas práticas de UX para agentes
============================================================
"""

from __future__ import annotations

import time

from rich.console import Console

# Console do Rich — fornece spinners, painéis, cores e formatação
console = Console()


# ============================================================
# 1. STREAMING DE RESPOSTA — Exibição caractere por caractere
# ============================================================
# Esta função simula o efeito de streaming: o texto vai aparecendo
# gradualmente no terminal, como se a LLM estivesse "digitando".
#
# COMO FUNCIONA:
# - Itera sobre cada caractere do texto
# - Imprime sem pular linha (end="")
# - Espera `delay` segundos entre cada caractere
# - No final, pula uma linha
#
# PARÂMETROS DE DELAY:
# - 0.01s: rápido (quase instantâneo, bom para testes)
# - 0.02s: velocidade natural de "digitação" (bom para demos)
# - 0.05s: velocidade dramática (bom para apresentações)
#
# EM PRODUÇÃO:
# Não usamos delay artificial — o delay é o tempo real entre
# tokens recebidos da API com stream=True.
# ============================================================

def exibir_resposta_streaming(texto: str, delay: float = 0.02) -> None:
    """
    Exibe um texto caractere por caractere no terminal (efeito typewriter).

    Parâmetros:
    - texto: string completa a ser exibida incrementalmente
    - delay: segundos entre cada caractere (default: 0.02)

    EFEITO VISUAL:
    O texto aparece "digitando" no terminal, dando a sensação de
    que a resposta está sendo gerada em tempo real.

    QUANDO USAR:
    - Demos e apresentações (impressiona e mostra profissionalismo)
    - Testes de UX (verificar se o feedback visual funciona)
    - Em produção, substituir por streaming real da API
    """
    for caractere in texto:
        console.print(caractere, end="")
        time.sleep(delay)
    console.print()  # Pula linha ao final


# ============================================================
# 2. DEMONSTRAÇÃO COMPLETA — Pipeline visual do agente
# ============================================================
# Simula o processamento do agente de boletos mostrando cada etapa
# com um spinner (animação rotativa) e a resposta final em streaming.
#
# FLUXO VISUAL QUE O USUÁRIO VÊ:
#
#  ⠋ Analisando entrada...          (spinner girando ~0.4s)
#  ⠋ Executando guardrails...       (spinner girando ~0.4s)
#  ⠋ Chamando tools...              (spinner girando ~0.4s)
#  ⠋ Validando risco...             (spinner girando ~0.4s)
#  ⠋ Montando resposta final...     (spinner girando ~0.4s)
#  📋 Dados extraídos               (streaming caractere a caractere)
#  💰 Valor atualizado: R$ 1.023,30 (streaming)
#  ⚠️ Aprovação humana recomendada  (streaming)
#  ✅ Status final: análise concluída (streaming)
#
# BOAS PRÁTICAS DE UX:
# 1. Use nomes de etapas que o USUÁRIO entenda (não "fn_parse_input")
# 2. Mantenha os labels consistentes com o que o agente realmente faz
# 3. Não mostre etapas que demoram < 100ms (poluição visual)
# 4. Use emojis para tornar o output mais scannable
# ============================================================

def demo_streaming() -> None:
    """
    Demonstra o pipeline visual do agente com spinners e streaming.

    ETAPAS:
    1. Mostra 5 etapas do agente com spinner do Rich (console.status)
    2. Após as etapas, exibe a resposta final com efeito streaming

    OBSERVE NO OUTPUT:
    - Cada etapa mostra um spinner enquanto "processa"
    - A transição entre etapas dá sensação de progresso
    - A resposta final aparece gradualmente (typewriter effect)
    - Emojis facilitam o scan visual (📋 dados, 💰 valor, etc.)

    EXERCÍCIO SUGERIDO:
    1. Adicione uma 6ª etapa (ex: "Registrando trace...")
    2. Mude o delay do streaming para 0.05 e veja a diferença
    3. Tente com textos maiores e observe o efeito
    """
    # Etapas do pipeline — nomes que o USUÁRIO entende
    etapas = [
        "Analisando entrada",
        "Executando guardrails",
        "Chamando tools",
        "Validando risco",
        "Montando resposta final",
    ]

    # console.status() cria um spinner animado enquanto o bloco executa
    # O sleep simula o tempo de processamento de cada etapa
    for etapa in etapas:
        with console.status(f"{etapa}..."):
            time.sleep(0.4)  # Simula tempo de processamento

    # Resposta final em streaming — cada linha com emoji para scan rápido
    # Em produção, isso viria dos tokens reais da API com stream=True
    exibir_resposta_streaming(
        "📋 Dados extraídos\n"
        "💰 Valor atualizado: R$ 1.023,30\n"
        "⚠️ Aprovação humana recomendada\n"
        "✅ Status final: análise concluída"
    )

    # Dicas finais para o aluno
    console.print("\n💡 Dicas de UX para agentes:", style="bold yellow")
    console.print("  1. Sempre mostre FEEDBACK enquanto o agente processa")
    console.print("  2. Use spinners para etapas internas (< 1s cada)")
    console.print("  3. Use streaming para a resposta final (quando > 2s)")
    console.print("  4. Nomes de etapas devem ser claros para o USUÁRIO")
    console.print("  5. Em produção, use stream=True na API da LLM")


# ============================================================
# PONTO DE ENTRADA — Execução direta do módulo
# ============================================================
# Ao rodar `python 01_streaming_console.py`, o aluno verá o
# pipeline visual completo do agente com spinners e streaming.
#
# EXERCÍCIO EXTRA:
# 1. Combine com o TraceRecorder (módulo 09) para registrar o
#    tempo de cada etapa e identificar gargalos
# 2. Implemente streaming real com Groq (stream=True) em vez
#    de delay artificial
# 3. Adicione cores diferentes para etapas OK vs etapas com warning
# ============================================================

if __name__ == "__main__":
    demo_streaming()
