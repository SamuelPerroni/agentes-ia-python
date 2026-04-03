"""
============================================================
MÓDULO 7.2 - DESIGN DE AVALIAÇÃO (Benchmark e Deploy)
============================================================
Neste módulo, aprendemos a PLANEJAR a avaliação do agente de forma
estruturada, definindo baselines, dimensões e critérios de bloqueio.

CONCEITO CHAVE:
O módulo 7.1 (01_avaliacao_resultados.py) ensina a MEDIR. Este módulo
ensina a DECIDIR: "dado essas métricas, o agente está bom o suficiente
para ir para produção?"

POR QUE DESIGN DE AVALIAÇÃO?
- Métricas sem critérios de aceite são inúteis: "acurácia 75% é boa?"
  Depende! Se o baseline é 70%, ótimo. Se é 90%, péssimo.
- Precisamos definir ANTES de testar o que é aceitável e o que bloqueia
- Isso evita decisões subjetivas no calor do momento
  ("ah, tá bom o suficiente")

ANALOGIA:
Pense em um exame médico:
- O exame mede glicose, colesterol, etc. (métricas = módulo 7.1)
- O médico compara com VALORES DE REFERÊNCIA (baselines = módulo 7.2)
- Se algum valor está fora da faixa, BLOQUEIA a alta (bloqueio de deploy)
- Não adianta saber que a glicose é 200 se você não sabe que o normal é < 100

AS 4 DIMENSÕES DE AVALIAÇÃO:

  ┌─────────────────────────────────────────────────────┐
  │          DIMENSÕES DE AVALIAÇÃO DO AGENTE            │
  ├─────────────────────────────────────────────────────┤
  │                                                      │
  │  1. TÉCNICA                                         │
  │     - JSON de saída é válido?                       │
  │     - Campos foram extraídos corretamente?           │
  │     - Cálculos de multa/juros estão corretos?        │
  │                                                      │
  │  2. NEGÓCIO                                         │
  │     - Resposta segue as regras de cobrança?          │
  │     - Classificação de risco está aderente?          │
  │     - HITL foi acionado quando deveria?              │
  │                                                      │
  │  3. SEGURANÇA                                       │
  │     - Prompt injection foi bloqueado?                │
  │     - PII foi mascarada antes de persistir?          │
  │     - Dados sensíveis NÃO vazaram na resposta?       │
  │                                                      │
  │  4. EXPERIÊNCIA DO USUÁRIO                          │
  │     - Resposta é clara e compreensível?              │
  │     - Informação está completa (nada faltando)?      │
  │     - Nota do LLM-as-Judge é aceitável?             │
  │                                                      │
  └─────────────────────────────────────────────────────┘

CRITÉRIOS DE BASELINE (benchmark_template.json):
- precisao_extracao_minima: 0.85 (85% dos campos corretos)
- guardrails_minimos: 0.95 (95% de eficácia em bloqueio)
- nota_llm_geral_minima: 7.5 (nota do LLM-as-Judge, 0-10)

BLOQUEIOS DE DEPLOY (qualquer um impede a nova versão):
- Queda de precisão de extração > 5%
- Qualquer regressão em bloqueio de prompt injection
- Falha em cálculo financeiro

Tópicos cobertos:
1. Carregamento do benchmark template (JSON)
2. Visualização das dimensões e baselines
3. Critérios de bloqueio de deploy
4. Decisão estruturada: deploy ou bloquear?
============================================================
"""

import json
import os

from rich.console import Console
from rich.table import Table

# Console do Rich para output formatado com cores e tabelas
console = Console()


# ============================================================
# 1. CARREGAMENTO DO BENCHMARK TEMPLATE
# ============================================================
# O benchmark é definido em um arquivo JSON separado para facilitar
# a manutenção. Qualquer pessoa pode alterar os thresholds sem
# mexer no código Python.
#
# ESTRUTURA DO benchmark_template.json:
# {
#   "baseline": {
#     "precisao_extracao_minima": 0.85,  → 85% de acurácia mínima
#     "guardrails_minimos": 0.95,         → 95% de eficácia mínima
#     "nota_llm_geral_minima": 7.5        → Nota mínima do LLM-as-Judge
#   },
#   "bloqueios_de_deploy": [
#     "queda de precisão de extração > 5%",
#     "qualquer regressão em bloqueio de prompt injection",
#     "falha em cálculo financeiro"
#   ],
#   "dimensoes": ["tecnica", "negocio", "seguranca", "experiencia"]
# }
# ============================================================

def carregar_benchmark() -> dict:
    """
    Carrega o template de benchmark do arquivo JSON.

    COMO FUNCIONA:
    - Encontra benchmark_template.json no diretório deste módulo
    - Lê e faz parse do JSON
    - Retorna o dicionário completo com baselines, bloqueios e dimensões

    RETORNO:
    Dict com chaves: baseline, bloqueios_de_deploy, dimensoes
    """
    caminho = os.path.join(
        os.path.dirname(__file__), "benchmark_template.json"
    )
    with open(caminho, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


# ============================================================
# 2. VISUALIZAÇÃO DO BENCHMARK — Tabela de Dimensões
# ============================================================
# Mostramos cada dimensão de avaliação com exemplos concretos de
# métricas, seguido dos baselines e critérios de bloqueio.
#
# POR QUE 4 DIMENSÕES?
# Avaliar apenas "acurácia" não é suficiente. Um agente pode extrair
# dados corretamente (técnica ✅) mas violar regras de negócio (negócio ❌)
# ou vazar PII nos logs (segurança ❌).
#
# CADA DIMENSÃO CAPTURA UM ASPECTO DIFERENTE:
# - Técnica = funciona corretamente?
# - Negócio = segue as regras da empresa?
# - Segurança = é seguro e conforme LGPD?
# - Experiência = o usuário entende e gosta da resposta?
# ============================================================

def mostrar_benchmark() -> None:
    """
    Exibe o benchmark completo: dimensões, baselines e bloqueios de deploy.

    ETAPAS:
    1. Carrega o benchmark do JSON
    2. Monta tabela com as 4 dimensões e exemplos de métricas
    3. Exibe os baselines (thresholds mínimos)
    4. Exibe os critérios de bloqueio de deploy

    OBSERVE NO OUTPUT:
    - As 4 dimensões cobrem todos os aspectos do agente
    - Os baselines são MÍNIMOS aceitáveis (abaixo = reprovar)
    - Os bloqueios são AUTOMÁTICOS (se qualquer um for violado, não deploya)

    EXERCÍCIO SUGERIDO:
    1. Adicione uma 5ª dimensão (ex: "latencia" com threshold < 3s)
    2. Ajuste os baselines no JSON e observe como a avaliação muda
    3. Crie um script que lê métricas reais e compara com os baselines
    """
    benchmark = carregar_benchmark()

    # Tabela com as 4 dimensões e exemplos de métricas
    tabela = Table(
        title="📊 Dimensões de Avaliação", header_style="bold magenta"
    )
    tabela.add_column("Dimensão", style="cyan")
    tabela.add_column("Exemplo de métrica")

    # Mapeamento dimensão → exemplos concretos de métricas
    exemplos = {
        "tecnica": "JSON válido, acurácia de extração, cálculo correto",
        "negocio": "aderência às regras de cobrança e risco",
        "seguranca": "prompt injection bloqueado, PII mascarada",
        "experiencia": "clareza, completude, utilidade percebida",
    }

    for dimensao in benchmark["dimensoes"]:
        tabela.add_row(dimensao, exemplos[dimensao])

    console.print(tabela)

    # Exibe os baselines (thresholds mínimos aceitáveis)
    console.print("\n📏 Baselines (mínimos para deploy):", style="bold blue")
    console.print(
        json.dumps(benchmark["baseline"], indent=2, ensure_ascii=False)
    )

    # Exibe os bloqueios de deploy (qualquer um impede a nova versão)
    console.print(
        "\n🚫 Bloqueios de deploy (qualquer um impede release):",
        style="bold red",
    )
    for regra in benchmark["bloqueios_de_deploy"]:
        console.print(f"  ❌ {regra}")

    # Regra prática final
    console.print("\n💡 Regra prática:", style="bold yellow")
    console.print("  1. Defina baselines ANTES de medir (evita viés)")
    console.print("  2. Bloqueios devem ser AUTOMÁTICOS (CI/CD)")
    console.print("  3. Revise baselines trimestralmente (expectativas mudam)")
    console.print(
        "  4. Documente exceções (se aceitou abaixo do baseline, por quê?)"
    )


# ============================================================
# PONTO DE ENTRADA — Execução direta do módulo
# ============================================================
# Ao rodar `python 02_design_avaliacao.py`, o aluno verá as
# dimensões, baselines e bloqueios de deploy formatados.
#
# EXERCÍCIO EXTRA:
# 1. Crie um script que simula resultados de avaliação e compara
#    com os baselines para decidir "deploy" ou "bloquear"
# 2. Integre os bloqueios no pipeline de CI/CD (ex: GitHub Actions)
# 3. Adicione histórico de avaliações para rastrear evolução
# ============================================================

if __name__ == "__main__":
    mostrar_benchmark()
