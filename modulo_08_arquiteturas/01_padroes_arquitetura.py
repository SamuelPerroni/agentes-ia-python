"""
============================================================
MÓDULO 8.1 - PADRÕES DE ARQUITETURA DE AGENTES
============================================================
Neste módulo, vamos comparar as 4 arquiteturas mais comuns para
agentes de IA, sempre usando o mesmo domínio: boletos bancários.

O QUE É ARQUITETURA DE AGENTES?
Arquitetura é a FORMA como organizamos os componentes do agente:
como a entrada chega, quem decide o próximo passo, quantos modelos
participam e como a resposta é montada.

Até agora, no treinamento, construímos um agente "monolítico"
(tudo em uma classe). Isso funciona, mas na vida real você
vai precisar escolher padrões diferentes dependendo do problema.

POR QUE EXISTEM PADRÕES DIFERENTES?
Cada padrão resolve um trade-off:
- Simplicidade vs. flexibilidade
- Custo vs. qualidade
- Latência vs. profundidade de análise
- Manutenção fácil vs. especialização

ANALOGIA:
Pense em um restaurante:
- Fluxo Linear = fast food (pedido fixo, sempre o mesmo fluxo)
- Roteador    = host do restaurante (olha o pedido e direciona à cozinha certa)
- Planner     = chef que planeja o menu antes de cozinhar
- Multiagente = cozinha com estações especializadas
  (carnes, molhos, sobremesas)

ARQUITETURAS COBERTAS:
1. Fluxo Linear        - Sequência fixa de etapas (A -> B -> C -> D)
2. Roteador            - Classificar intenção e despachar ao subfluxo certo
3. Planner-Executor    - LLM planeja os passos e executor cumpre cada um
4. Multiagente         - Vários agentes com papéis independentes se coordenam

DIAGRAMA VISUAL — Comparação lado a lado:

  ┌── FLUXO LINEAR ───┐   ┌──── ROTEADOR ─────┐
  │                   │   │                   │
  │ Entrada           │   │ Entrada           │
  │   ↓               │   │   ↓               │
  │ Etapa 1           │   │ Classificador     │
  │   ↓               │   │  ↙    ↓    ↘      │
  │ Etapa 2           │   │ Sub1  Sub2  Sub3  │
  │   ↓               │   │   ↘   ↓    ↙      │
  │ Etapa 3           │   │   Resposta        │
  │   ↓               │   └───────────────────┘
  │ Resposta          │
  └───────────────────┘

  ┌─ PLANNER-EXECUTOR ─┐   ┌── MULTIAGENTE ────┐
  │                    │   │                   │
  │ Entrada            │   │ Entrada           │
  │   ↓                │   │   ↓               │
  │ LLM planeja        │   │ Agente A (lê)     │
  │   ↓                │   │   ↓               │
  │ Executor step 1    │   │ Agente B (calcula)│
  │   ↓                │   │   ↓               │
  │ Executor step 2    │   │ Agente C (valida) │
  │   ↓                │   │   ↓               │
  │ Resultado          │   │ Agente D (audita) │
  └────────────────────┘   │   ↓               │
                           │ Resposta final    │
                           └───────────────────┘

Tópicos cobertos:
1. Modelagem de cada arquitetura com dataclass
2. Simulação de cada padrão aplicado a boletos
3. Comparativo visual em tabela (Rich)
4. Critérios de decisão para escolher a arquitetura certa
============================================================
"""

from dataclasses import dataclass
from typing import Callable

from rich.console import Console
from rich.table import Table

# Console do Rich para output formatado com cores, tabelas e painéis
console = Console()


# ============================================================
# ESTRUTURA DE DADOS PARA O COMPARATIVO
# ============================================================
# Usamos um dataclass para padronizar o resultado de cada
# arquitetura. Isso facilita a comparação e a exibição em tabela.
#
# Cada arquitetura retorna os mesmos campos: nome, quando usar,
# vantagens, desvantagens, latência esperada, complexidade e
# um exemplo de fluxo mostrando como a pergunta seria processada.
# ============================================================

@dataclass
class ResultadoArquitetura:
    """
    Representa o resultado da simulação de uma arquitetura.

    Campos:
    - arquitetura: nome do padrão (ex: "Fluxo Linear")
    - quando_usar: cenário ideal para este padrão
    - vantagens: pontos fortes
    - desvantagens: pontos fracos e riscos
    - latencia: tempo esperado (Baixa / Média / Alta)
    - complexidade: custo de implementação e manutenção
    - exemplo_fluxo: representação textual do fluxo para a pergunta dada
    """
    arquitetura: str
    quando_usar: str
    vantagens: str
    desvantagens: str
    latencia: str
    complexidade: str
    exemplo_fluxo: str


# ============================================================
# 1. FLUXO LINEAR — Sequência fixa de etapas
# ============================================================
# O padrão mais simples: cada etapa executa na ordem, sem decisões.
# A -> B -> C -> D — como uma linha de montagem.
#
# QUANDO USAR:
# - A tarefa é previsível (sempre os mesmos passos)
# - O domínio é restrito (ex: sempre calcular valor de UM boleto)
# - Velocidade e custo são prioridade
#
# EXEMPLO NO NOSSO DOMÍNIO:
# O agente de boletos que construímos no módulo 06 é essencialmente
# linear: recebe texto → extrai dados → calcula encargos → responde.
#
# LIMITAÇÕES:
# - Não lida bem com perguntas fora do fluxo previsto
# - Se adicionar novas funcionalidades, o fluxo fica frágil
# - Dificuldade de escalar para múltiplos tipos de tarefa
# ============================================================

def fluxo_linear(pergunta: str) -> ResultadoArquitetura:
    """
    Simula o processamento de uma pergunta no padrão Fluxo Linear.

    TEORIA:
    - O fluxo é fixo: Entrada → Extração → Cálculo → Resposta
    - Não há decisão intermediária — todos os passos sempre executam
    - É o padrão de menor custo (1 chamada LLM ou até 0 se usar regras)

    ANALOGIA:
    É como um drive-through: o pedido entra, segue a esteira e sai pronto.
    Não importa o que você pede, o fluxo é sempre o mesmo.
    """
    return ResultadoArquitetura(
        arquitetura="Fluxo Linear",
        quando_usar="Tarefa previsível com sequência fixa de passos.",
        vantagens="Simples, barato, fácil de testar. Ideal para MVP.",
        desvantagens="Pouca flexibilidade — travado na sequência definida.",
        latencia="Baixa",
        complexidade="Baixa",
        # Mostramos o fluxo concreto com trecho da pergunta
        exemplo_fluxo=(
            "Entrada -> extrair_dados -> calcular_encargos"
            f" -> responder: {pergunta[:45]}..."
        ),
    )


# ============================================================
# 2. ROTEADOR — Classificar intenção e despachar
# ============================================================
# Primeiro passo: entender a INTENÇÃO do usuário.
# Depois: direcionar para o subfluxo especializado.
#
# QUANDO USAR:
# - O agente atende MÚLTIPLOS tipos de solicitação (consulta, 2ª via,
#   cálculo de juros, reclamação, etc.)
# - Cada tipo de solicitação tem um fluxo diferente
# - Você quer escalar sem complicar um único fluxo gigante
#
# COMO FUNCIONA:
#   ┌────────────┐
#   │   Entrada   │
#   └──────┬─────┘
#          ↓
#   ┌────────────┐
#   │Classificador│  ← Pode ser LLM, regex ou regra de negócio
#   └──┬───┬───┬─┘
#      ↓   ↓   ↓
#   Sub1  Sub2  Sub3  ← Subfluxos especializados
#      ↘   ↓   ↙
#   ┌────────────┐
#   │  Resposta   │
#   └────────────┘
#
# RISCOS:
# - Se o classificador erra a intenção, TODO o fluxo erra
# - Precisa de boa cobertura de intents e fallback para "não entendi"
# ============================================================

def roteador(pergunta: str) -> ResultadoArquitetura:
    """
    Simula o processamento de uma pergunta no padrão Roteador.

    TEORIA:
    - O roteador é um "despachante" — lê a pergunta e decide
      para qual subagente/subfluxo enviar
    - Aqui usamos regras simples (palavras-chave) para simular
    - Em produção, o classificador pode ser uma LLM com prompt
      específico ou um modelo de classificação de intenção

    ANALOGIA:
    É como o host de um restaurante: ele olha quantas pessoas são,
    se têm reserva, e direciona para a mesa certa. Ele não cozinha.
    """
    # Classificação simples por palavras-chave
    # Em produção, isso seria uma chamada LLM ou modelo de classificação
    if "segunda via" in pergunta.lower():
        destino = "subagente de atendimento"
    elif "venc" in pergunta.lower() or "juros" in pergunta.lower():
        destino = "subagente financeiro"
    else:
        destino = "subagente de leitura"

    return ResultadoArquitetura(
        arquitetura="Roteador",
        quando_usar="Múltiplos tipos de intenção com fluxos diferentes.",
        vantagens="Escalável — cada novo intent vira um subfluxo isolado.",
        desvantagens="Erros de roteamento degradam toda a experiência.",
        latencia="Baixa a média",
        complexidade="Média",
        # Mostramos a classificação que o roteador faria
        exemplo_fluxo=f"Classificar intenção -> enviar para {destino}",
    )


# ============================================================
# 3. PLANNER-EXECUTOR — LLM planeja, código executa
# ============================================================
# A LLM recebe a pergunta e gera um PLANO de passos.
# Depois, um executor (código determinístico) executa cada passo.
#
# QUANDO USAR:
# - A tarefa é composta e os passos VARIAM conforme a pergunta
# - Exemplo: "calcule juros E verifique o beneficiário E gere relatório"
#   tem passos diferentes de "só extraia os dados do boleto"
# - Quando você precisa de ADAPTABILIDADE mas com execução controlada
#
# COMO FUNCIONA:
# 1. LLM recebe: "dada esta pergunta, quais passos são necessários?"
# 2. LLM retorna: ["extrair dados", "verificar vencimento", "calcular juros"]
# 3. Executor roda cada passo chamando as tools certas
# 4. Resultado final é montado a partir dos outputs parciais
#
# TRADE-OFF:
# - MAIS ROBUSTO que linear para tarefas complexas
# - MAIS LENTO e MAIS CARO (chamada LLM extra só para planejar)
# - Risco: o plano pode ser incoerente se o prompt não for bom
#
# ANALOGIA:
# É como um chef que primeiro LÊ o pedido completo, PLANEJA a
# sequência de preparo, e depois EXECUTA prato a prato.
# ============================================================

def planner_executor(_pergunta: str) -> ResultadoArquitetura:
    """
    Simula o processamento de uma pergunta no padrão Planner-Executor.

    TEORIA:
    - Fase 1 (Planner): a LLM decompõe a tarefa em passos numerados
    - Fase 2 (Executor): código determinístico executa cada passo
    - Permite tarefas complexas e variáveis sem reescrever o agente
    - Custo extra: 1 chamada LLM para o plano + N chamadas para execução

    ANALOGIA:
    É como um GPS: primeiro calcula a rota (plano), depois você segue
    curva a curva (execução). Se o plano for ruim, a execução também será.
    """
    # Simulamos o plano que a LLM geraria para esta pergunta
    plano = [
        "1. Extrair dados do boleto",
        "2. Verificar vencimento",
        "3. Calcular valor atualizado se necessário",
        "4. Gerar resumo final",
    ]
    return ResultadoArquitetura(
        arquitetura="Planner-Executor",
        quando_usar="Tarefas compostas ou variáveis que exigem decomposição.",
        vantagens="Robusto em multi-etapas — se adapta à pergunta.",
        desvantagens="Mais lento, mais caro e depende da qualidade do plano.",
        latencia="Média a alta",
        complexidade="Alta",
        # Mostramos os passos do plano gerado
        exemplo_fluxo=" | ".join(plano),
    )


# ============================================================
# 4. MULTIAGENTE — Vários agentes especializados se coordenam
# ============================================================
# Cada "agente" tem uma responsabilidade isolada e um contexto
# próprio. Um orquestrador coordena a comunicação entre eles.
#
# QUANDO USAR:
# - O problema tem PAPÉIS DISTINTOS que exigem especialização
#   (ex: leitura de documentos, cálculo financeiro, análise de risco,
#   auditoria de compliance)
# - Cada papel precisa de um prompt/contexto/modelo diferente
# - Escala: equipes diferentes mantêm agentes diferentes
#
# COMO FUNCIONA:
#   ┌───────────────────────────────────────────┐
#   │             ORQUESTRADOR                   │
#   └──┬──────┬──────────┬──────────┬───────────┘
#      ↓      ↓          ↓          ↓
#   Agente  Agente     Agente     Agente
#   Leitura Financeiro  Risco     Auditor
#   (extrai  (calcula   (avalia   (registra
#    dados)   juros)    perigo)    decisão)
#      ↘      ↓          ↓         ↙
#      └───── Resultado Final ─────┘
#
# RISCOS:
# - Orquestração difícil (quem passa o quê para quem?)
# - Custo mais alto (MÚLTIPLAS chamadas LLM)
# - Debugging complexo (qual agente errou?)
# - Latência alta (execução sequencial ou paralela com merge)
#
# ANALOGIA:
# É como um hospital: cada especialista (cardiologista, ortopedista,
# radiologista) faz sua parte e o médico principal coordena.
# ============================================================

def multiagente(_pergunta: str) -> ResultadoArquitetura:
    """
    Simula o processamento de uma pergunta no padrão Multiagente.

    TEORIA:
    - Cada agente é um "especialista" com seu próprio system prompt,
      ferramentas e, potencialmente, modelo LLM diferente
    - O orquestrador decide a ordem e passa contexto entre agentes
    - Ideal quando a qualidade exige especialização profunda

    ANALOGIA:
    É como uma cozinha profissional: o chef de carnes, o saucier,
    o confeiteiro — cada um faz o que sabe melhor, e o chef
    executivo coordena o timing dos pratos.
    """
    return ResultadoArquitetura(
        arquitetura="Multiagente",
        quando_usar="Problemas com papéis distintos:"
        " leitura, risco, auditoria.",
        vantagens="Especialização profunda e isolamento de contexto.",
        desvantagens="Orquestração difícil, custo e latência mais altos.",
        latencia="Alta",
        complexidade="Alta",
        # Mostramos a cadeia de agentes especializados
        exemplo_fluxo=(
            "Agente de leitura -> Agente financeiro -> "
            "Agente de risco -> Agente auditor"
        ),
    )


# ============================================================
# 5. COMPARATIVO VISUAL — Tabela lado a lado
# ============================================================
# Aqui juntamos todos os padrões e mostramos em uma tabela
# para facilitar a comparação. Cada arquitetura processa a MESMA
# pergunta, mas de forma diferente.
#
# POR QUE UM COMPARATIVO?
# Na prática, a escolha de arquitetura é uma das primeiras decisões
# de design de um agente. Errar aqui = retrabalho caro depois.
#
# CRITÉRIOS DE DECISÃO (regra prática):
# ┌────────────────────┬──────────────────────────────────┐
# │ Situação           │ Arquitetura recomendada          │
# ├────────────────────┼──────────────────────────────────┤
# │ MVP / prova de     │ Fluxo Linear                     │
# │ conceito           │                                  │
# ├────────────────────┼──────────────────────────────────┤
# │ Múltiplos intents  │ Roteador                         │
# │ no mesmo produto   │                                  │
# ├────────────────────┼──────────────────────────────────┤
# │ Tarefas complexas  │ Planner-Executor                 │
# │ e variáveis        │                                  │
# ├────────────────────┼──────────────────────────────────┤
# │ Domínio com papéis │ Multiagente                      │
# │ distintos em prod  │                                  │
# └────────────────────┴──────────────────────────────────┘
# ============================================================

def mostrar_comparativo(pergunta: str) -> None:
    """
    Exibe um comparativo visual das 4 arquiteturas aplicadas à mesma pergunta.

    COMO FUNCIONA:
    1. Recebe uma pergunta de exemplo sobre boletos
    2. Simula como cada arquitetura processaria essa pergunta
    3. Monta uma tabela Rich com colunas comparativas
    4. Exibe regras práticas de quando usar cada padrão

    DICA PARA O ALUNO:
    Rode este módulo com diferentes perguntas para ver como o
    roteador muda o destino e como o fluxo de cada arquitetura
    se adapta (ou não) ao tipo de solicitação.
    """
    console.print("\n🎓 COMPARATIVO DE ARQUITETURAS", style="bold blue")
    console.print(f"Pergunta exemplo: {pergunta}", style="dim")

    # Lista de funções-fábrica — cada uma simula uma arquitetura
    arquiteturas: list[Callable[[str], ResultadoArquitetura]] = [
        fluxo_linear,
        roteador,
        planner_executor,
        multiagente,
    ]

    # Monta a tabela com Rich para output colorido e alinhado
    tabela = Table(show_header=True, header_style="bold magenta")
    tabela.add_column("Arquitetura", style="cyan")
    tabela.add_column("Quando usar")
    tabela.add_column("Latência")
    tabela.add_column("Complexidade")
    tabela.add_column("Resumo do fluxo")

    # Processa a pergunta em cada arquitetura e adiciona à tabela
    for fabrica in arquiteturas:
        resultado = fabrica(pergunta)
        tabela.add_row(
            resultado.arquitetura,
            resultado.quando_usar,
            resultado.latencia,
            resultado.complexidade,
            resultado.exemplo_fluxo,
        )

    console.print(tabela)

    # Regras práticas de evolução — do mais simples ao mais complexo
    console.print("\n💡 Regra prática de evolução:", style="bold yellow")
    console.print("  1. Comece SEMPRE com fluxo linear (prove valor rápido)")
    console.print("  2. Adicione roteamento quando houver intents diferentes")
    console.print(
        "  3. Use planner quando a sequência de passos variar por pergunta"
    )
    console.print(
        "  4. Só use multiagente quando a especialização compensar o custo"
    )
    console.print(
        "\n  ⚠️  Cada nível acima adiciona complexidade, custo e latência."
    )
    console.print(
        "  A regra de ouro: use a arquitetura MAIS SIMPLES"
        " que resolva o problema."
    )


# ============================================================
# PONTO DE ENTRADA — Execução direta do módulo
# ============================================================
# Ao rodar `python 01_padroes_arquitetura.py`, o aluno verá o
# comparativo aplicado a uma pergunta real sobre boletos.
#
# EXERCÍCIO SUGERIDO:
# 1. Troque a pergunta abaixo por outras (ex: "quero segunda via")
#    e veja como o roteador muda o destino
# 2. Adicione uma 5ª arquitetura (ex: "event-driven") como exercício
# 3. Crie um diagrama ASCII próprio para a sua variação
# ============================================================

if __name__ == "__main__":
    mostrar_comparativo(
        "Meu boleto venceu há 12 dias, preciso calcular juros"
        " e validar se o beneficiário está correto."
    )
