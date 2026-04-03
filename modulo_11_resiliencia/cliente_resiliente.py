"""
============================================================
MÓDULO 11 - CLIENTE RESILIENTE (cliente_resiliente.py)
============================================================
Este arquivo contém o ClienteLLMResiliente — uma classe que
encapsula chamadas à LLM com tratamento operacional automático:
retry, exponential backoff e fallback para modelo alternativo.

É importado pelo agente de boletos (módulo 06) e pelo demo
(01_resiliencia_operacional.py).

O QUE É RESILIÊNCIA OPERACIONAL?
É a capacidade do sistema de continuar funcionando mesmo quando
partes dele falham. Em agentes de IA, a principal fonte de falha
é a API da LLM: timeout, rate limit, erro 500, indisponibilidade.

POR QUE RESILIÊNCIA É ESSENCIAL?
- APIs de LLM são SERVIÇOS EXTERNOS — você não controla a disponibilidade
- Em produção, o agente não pode "travar" porque a LLM demorou 30s
- Usuários esperam resposta mesmo em cenários degradados
- Sem resiliência, uma falha temporária = indisponibilidade total

ANALOGIA:
Pense em um GPS no carro:
- Retry = recalcular rota quando o sinal GPS falha momentaneamente
- Backoff = esperar um pouco mais a cada tentativa (evita sobrecarregar)
- Fallback = usar rota salva offline se o GPS não voltar

PADRÕES IMPLEMENTADOS:

1. RETRY (Retentativa):
   Se a chamada falha, tenta novamente. Útil para erros transitórios
   (pico de carga, timeout pontual).

2. EXPONENTIAL BACKOFF:
   A cada retry, espera MAIS TEMPO antes de tentar de novo.
   Isso evita sobrecarregar a API que já está com problemas.
   Fórmula: wait = min(0.5 * tentativa, 2.0) segundos

   Tentativa 1: espera 0.5s
   Tentativa 2: espera 1.0s
   Tentativa 3: espera 1.5s (cap em 2.0s)

3. FALLBACK (Degradação Controlada):
   Se o modelo principal falha TODAS as retentativas, tenta com
   um modelo alternativo (menor, mais rápido, mais barato).
   Melhor dar uma resposta "razoável" do que nenhuma resposta.

DIAGRAMA DO FLUXO:

  ┌──────────────────────────────────────────────┐
  │           ClienteLLMResiliente                │
  ├──────────────────────────────────────────────┤
  │                                              │
  │  Modelo Primário (ex: llama-3.3-70b)         │
  │  ┌────────────────────────────────┐          │
  │  │ Tentativa 1  →  OK? ──→ ✅     │          │
  │  │      │ falhou                   │          │
  │  │ wait 0.5s                       │          │
  │  │ Tentativa 2  →  OK? ──→ ✅     │          │
  │  │      │ falhou                   │          │
  │  │ wait 1.0s                       │          │
  │  │ Tentativa 3  →  OK? ──→ ✅     │          │
  │  │      │ falhou (esgotou retries) │          │
  │  └──────┼──────────────────────────┘          │
  │         ↓                                     │
  │  Modelo Fallback (ex: llama-3.1-8b)          │
  │  ┌────────────────────────────────┐          │
  │  │ Tentativa 1  →  OK? ──→ ✅     │          │
  │  │      │ falhou                   │          │
  │  │ Tentativa 2  →  OK? ──→ ✅     │          │
  │  │      │ falhou                   │          │
  │  │ Tentativa 3  →  FALHA TOTAL ❌ │          │
  │  └─────────────────────────────────┘          │
  │                                               │
  │  → RuntimeError("Falha após retries...")      │
  └───────────────────────────────────────────────┘

COMPONENTES:
1. ResultadoChamada - dataclass com o resultado da chamada
2. ClienteLLMResiliente - classe com retry + backoff + fallback
============================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


# ============================================================
# 1. RESULTADO DA CHAMADA — Struct de retorno
# ============================================================
# Usamos um dataclass para retornar informações ALÉM do conteúdo:
# qual modelo foi usado, quantas tentativas foram necessárias e
# se o fallback foi acionado. Isso é importante para:
# - Observabilidade: logar no trace qual modelo respondeu
# - Métricas: medir taxa de fallback (quando > 5%, investigue)
# - Debugging: saber se a lentidão é por retry ou por modelo
# ============================================================

@dataclass
class ResultadoChamada:
    """
    Resultado de uma chamada resiliente à LLM.

    Campos:
    - conteudo: a resposta da LLM (ou qualquer retorno da call_fn)
    - tentativas: quantas tentativas foram necessárias no modelo que respondeu
    - modelo_usado: nome do modelo que efetivamente gerou a resposta
    - fallback_acionado: True se o modelo primário falhou e o fallback atuou
    """
    conteudo: Any
    tentativas: int
    modelo_usado: str
    fallback_acionado: bool


# ============================================================
# 2. CLIENTE LLM RESILIENTE — Retry + Backoff + Fallback
# ============================================================
# Esta classe encapsula QUALQUER função de chamada à LLM.
# Ela não conhece o Groq nem o OpenAI — recebe uma função genérica.
# Isso a torna reutilizável com qualquer provedor.
#
# COMO USAR:
#   cliente = ClienteLLMResiliente(
#       primary_model="llama-3.3-70b-versatile",
#       fallback_model="llama-3.1-8b-instant",
#       max_retries=2
#   )
#   resultado = cliente.executar(minha_funcao_llm, *args, **kwargs)
#
# A minha_funcao_llm deve aceitar (modelo, *args, **kwargs) como
# primeiro argumento, assim o cliente pode trocar o modelo no fallback.
#
# POR QUE NÃO ACOPLAR AO GROQ?
# - Amanhã você pode usar OpenAI, Anthropic, Azure, local (Ollama)
# - A lógica de retry/fallback é INDEPENDENTE do provedor
# - Princípio de design: separe a "mecânica" do "provedor"
# ============================================================

class ClienteLLMResiliente:
    """
    Encapsula uma função de chamada à LLM com tratamento operacional.

    RESPONSABILIDADES:
    - Retry automático com exponential backoff
    - Fallback para modelo alternativo se o primário falhar
    - Retornar metadados (modelo usado, tentativas, fallback acionado)

    PARÂMETROS:
    - primary_model: modelo principal (ex: "llama-3.3-70b-versatile")
    - fallback_model: modelo de reserva (ex: "llama-3.1-8b-instant"), opcional
    - max_retries: máximo de retentativas POR MODELO (default: 2)
      Total de tentativas = (max_retries + 1) * quantidade de modelos

    ANALOGIA:
    É como ligar para um médico: se não atende na primeira, você
    tenta mais 2 vezes (retry). Se não atender, liga para outro
    médico (fallback). Se nenhum atender, vai ao pronto-socorro (erro).
    """

    def __init__(
        self,
        primary_model: str,
        fallback_model: str | None = None,
        max_retries: int = 2,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.max_retries = max_retries

    def executar(self, call_fn, *args, **kwargs) -> ResultadoChamada:
        """
        Executa a função de chamada com retry, backoff e fallback.

        Parâmetros:
        - call_fn: função chamada como
          call_fn(modelo, *args, **kwargs)
        - *args, **kwargs: argumentos repassados para call_fn

        COMO FUNCIONA:
        1. Monta lista de modelos: [primário, fallback]
           (se fallback configurado)
        2. Para cada modelo, tenta até (max_retries + 1) vezes
        3. Entre tentativas, espera com backoff: min(0.5 * tentativa, 2.0)s
        4. Se conseguir, retorna ResultadoChamada com metadados
        5. Se esgotar TODOS os modelos e retries, levanta RuntimeError

        RETORNO:
        ResultadoChamada com: conteudo, tentativas,
        modelo_usado, fallback_acionado

        EXCEÇÃO:
        RuntimeError se TODAS as tentativas em TODOS os modelos falharem
        """
        ultimo_erro = None

        # Monta a lista de modelos a tentar
        # (primário primeiro, fallback depois)
        modelos = [self.primary_model]
        if self.fallback_model:
            modelos.append(self.fallback_model)

        # Itera sobre cada modelo disponível
        for indice_modelo, modelo in enumerate(modelos):
            # Para cada modelo, tenta até (max_retries + 1) vezes
            # +1: a primeira tentativa não é "retry", é a original
            for tentativa in range(1, self.max_retries + 2):
                try:
                    # Chama a função passando o modelo como primeiro argumento
                    conteudo = call_fn(modelo, *args, **kwargs)
                    return ResultadoChamada(
                        conteudo=conteudo,
                        tentativas=tentativa,
                        modelo_usado=modelo,
                        # Fallback acionado = estamos no 2º+ modelo da lista
                        fallback_acionado=indice_modelo > 0,
                    )
                # pragma: no cover - dependente de API externa
                except (ConnectionError, TimeoutError, OSError) as erro:
                    ultimo_erro = erro
                    # Exponential backoff: espera mais a cada tentativa
                    # min() garante que não ultrapasse 2.0 segundos
                    tempo_espera = min(0.5 * tentativa, 2.0)
                    time.sleep(tempo_espera)

        # Se chegou aqui, TODOS os modelos e retries falharam
        raise RuntimeError(
            f"Falha após retries e fallback: {ultimo_erro}"
        )
