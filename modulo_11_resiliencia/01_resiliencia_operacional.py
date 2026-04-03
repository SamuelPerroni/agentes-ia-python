"""
============================================================
MÓDULO 11.1 - RESILIÊNCIA OPERACIONAL EM AGENTES
============================================================
Neste módulo, vamos entender e demonstrar os padrões de resiliência
que protegem agentes de IA contra falhas de APIs externas.

CONCEITO CHAVE:
Em produção, seu agente depende de SERVIÇOS EXTERNOS (APIs de LLM,
bancos de dados, APIs parceiras). Esses serviços PODEM E VÃO FALHAR.
Resiliência é planejar o que fazer QUANDO (não SE) isso acontecer.

POR QUE RESILIÊNCIA É ESSENCIAL?
- APIs de LLM têm SLA de ~99.9% = ~43 min de downtime por mês
- Rate limits podem bloquear seu agente em horários de pico
- Timeouts acontecem por latência de rede ou sobrecarga do servidor
- Sem resiliência: uma falha temporária = experiência QUEBRADA pro usuário

PADRÕES DE RESILIÊNCIA (do mais simples ao mais sofisticado):

  ┌────────────────────────────────────────────────────┐
  │  1. RETRY — Tentar de novo                         │
  │     Problema: erro transitório (timeout pontual)   │
  │     Solução: repetir a mesma chamada               │
  ├────────────────────────────────────────────────────┤
  │  2. EXPONENTIAL BACKOFF — Esperar mais a cada vez  │
  │     Problema: API sobrecarregada                   │
  │     Solução: dar tempo para o serviço se recuperar │
  ├────────────────────────────────────────────────────┤
  │  3. FALLBACK — Trocar para alternativa             │
  │     Problema: modelo principal indisponível        │
  │     Solução: usar modelo menor/mais rápido         │
  ├────────────────────────────────────────────────────┤
  │  4. DEGRADAÇÃO GRACIOSA — Funcionar parcialmente   │
  │     Problema: todas as LLMs falharam               │
  │     Solução: responder com template fixo + pedido  │
  │     de desculpas, em vez de erro 500               │
  └────────────────────────────────────────────────────┘

ANALOGIA:
Pense em uma loja online:
- Retry = "não carregou? atualize a página"
- Backoff = "muita gente na fila, espere 5 min e tente de novo"
- Fallback = "cartão recusado? tente com Pix"
- Degradação = "estamos fora do ar, mas você pode ligar para 0800"

NESTE MÓDULO:
Usamos o ClienteLLMResiliente (de cliente_resiliente.py) com uma
função FAKE que simula falhas controladas para mostrar o comportamento.

Tópicos cobertos:
1. Simulação de falhas controladas (função fake)
2. Retry com backoff automático
3. Fallback para modelo alternativo
4. Metadados de resultado (modelo usado, tentativas, fallback)
============================================================
"""

from rich.console import Console
from rich.panel import Panel

# Importamos o ClienteLLMResiliente do mesmo módulo
from modulo_11_resiliencia.cliente_resiliente import ClienteLLMResiliente

# Console do Rich para output formatado
console = Console()


# ============================================================
# 1. FUNÇÃO FAKE — Simula falhas controladas
# ============================================================
# Para demonstrar resiliência sem depender de uma API real,
# criamos uma função que "programa" seus comportamentos:
# - "erro" = levanta TimeoutError (simula API lenta)
# - qualquer outro valor = retorna resposta normal
#
# O parâmetro `comportamento` é uma lista que funciona como
# uma FILA: cada chamada consome o primeiro elemento.
#
# EXEMPLO:
# fake_call("modelo", ["erro", "erro", "ok"])
# Chamada 1: levanta TimeoutError ("erro")
# Chamada 2: levanta TimeoutError ("erro")
# Chamada 3: retorna "Resposta estável com modelo" ("ok")
# ============================================================

def fake_call(modelo: str, comportamento: list[str]) -> str:
    """
    Simula uma chamada à LLM com comportamento programado.

    Parâmetros:
    - modelo: nome do modelo (passado pelo ClienteLLMResiliente)
    - comportamento: lista de strings ("erro" ou qualquer outro valor)
      Cada chamada consome o primeiro item da lista (pop)

    COMO FUNCIONA:
    - Se o próximo item é "erro", levanta TimeoutError
    - Se é qualquer outro valor, retorna string de sucesso
    - A lista diminui a cada chamada (simula "retry até funcionar")
    """
    # pop(0) remove e retorna o primeiro elemento da lista
    status = comportamento.pop(0)
    if status == "erro":
        raise TimeoutError(f"Modelo {modelo} demorou demais")
    return f"Resposta estável com {modelo}"


# ============================================================
# 2. DEMONSTRAÇÃO — Retry + Fallback em ação
# ============================================================
# Configuramos o cenário:
# - Modelo primário: llama-3.3-70b (potente mas "instável" na demo)
# - Modelo fallback: llama-3.1-8b (menor mas "estável" na demo)
# - max_retries=1: só 1 retry por modelo (2 tentativas total por modelo)
# - Comportamento: ["erro", "erro", "ok"]
#
# FLUXO ESPERADO:
# 1. Primário, tentativa 1: "erro" → TimeoutError → wait 0.5s
# 2. Primário, tentativa 2: "erro" → TimeoutError → wait 1.0s
# 3. (esgotou retries do primário) → troca para fallback
# 4. Fallback, tentativa 1: "ok" → retorna resposta ✅
#
# RESULTADO:
# - modelo_usado: "llama-3.1-8b-instant" (fallback)
# - tentativas: 1 (no modelo final)
# - fallback_acionado: True
# ============================================================

def demo_resiliencia() -> None:
    """
    Demonstra o ClienteLLMResiliente com falhas simuladas.

    ETAPAS:
    1. Cria cliente com modelo primário e fallback
    2. Define comportamento: 2 erros seguidos, depois sucesso
    3. Executa e observa o retry + fallback em ação
    4. Exibe painel com os metadados do resultado

    OBSERVE NO OUTPUT:
    - O modelo usado NÃO é o primário (fallback foi acionado)
    - As tentativas mostram quantas vezes o modelo FINAL foi chamado
    - O fallback_acionado=True confirma que o primário falhou

    EXERCÍCIO SUGERIDO:
    1. Mude o comportamento para ["ok"] (sucesso na 1ª) e veja o resultado
    2. Mude para ["erro", "erro", "erro", "erro"] (falha total)
    e veja o RuntimeError
    3. Aumente max_retries e observe o tempo total (backoff acumula)
    """
    # Configura o cliente com os dois modelos
    cliente = ClienteLLMResiliente(
        primary_model="llama-3.3-70b-versatile",
        fallback_model="llama-3.1-8b-instant",
        max_retries=1,  # 1 retry = 2 tentativas por modelo
    )

    # Cenário: 2 erros (primário esgota), depois sucesso (no fallback)
    comportamento = ["erro", "erro", "ok"]

    # Executa com a função fake — o cliente faz retry + fallback internamente
    resultado = cliente.executar(fake_call, comportamento)

    # Exibe o resultado em painel formatado
    console.print(Panel.fit(
        f"[bold]Modelo usado:[/bold] {resultado.modelo_usado}\n"
        f"[bold]Tentativas no modelo final:[/bold] {resultado.tentativas}\n"
        f"[bold]Fallback acionado:[/bold] "
        f"{'✅ Sim' if resultado.fallback_acionado else '❌ Não'}\n"
        f"[bold]Conteúdo da resposta:[/bold] {resultado.conteudo}",
        title="🛡️ Resiliência — Resultado da Execução",
        border_style="green",
    ))

    # Explicação do que aconteceu
    console.print("\n📋 O que aconteceu nesta execução:", style="bold yellow")
    console.print(
        "  1. Primário (70b): tentativa 1 → TimeoutError → esperou 0.5s"
    )
    console.print(
        "  2. Primário (70b): tentativa 2 → TimeoutError → esperou 1.0s"
    )
    console.print("  3. Primário esgotou retries → ativou FALLBACK")
    console.print("  4. Fallback (8b): tentativa 1 → SUCESSO ✅")

    console.print("\n💡 Dica:", style="bold green")
    console.print("  Em produção, monitore a TAXA DE FALLBACK:")
    console.print("  - < 1%: normal (erros transitórios)")
    console.print("  - 1-5%: atenção (possível degradação do provedor)")
    console.print("  - > 5%: investigue (problema sistêmico)")


# ============================================================
# PONTO DE ENTRADA — Execução direta do módulo
# ============================================================
# Ao rodar `python 01_resiliencia_operacional.py`, o aluno verá
# o ciclo completo de retry + fallback com falhas simuladas.
#
# EXERCÍCIO EXTRA:
# 1. Crie um cenário onde o fallback TAMBÉM falha (teste RuntimeError)
# 2. Adicione um timer para medir quanto tempo o retry total levou
# 3. Integre com o TraceRecorder para registrar cada tentativa
# ============================================================

if __name__ == "__main__":
    demo_resiliencia()
