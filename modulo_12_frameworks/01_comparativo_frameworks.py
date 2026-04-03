"""
============================================================
MÓDULO 12.1 - COMPARATIVO MANUAL vs FRAMEWORKS (na prática)
============================================================
Neste script, vamos VISUALIZAR lado a lado o que fizemos
manualmente durante o treinamento e como cada framework
popular resolve o mesmo problema.

O QUE ESTE SCRIPT FAZ?
Monta tabelas comparativas interativas usando Rich para que
você veja EXATAMENTE onde cada conceito manual mapeia para
qual abstração de framework.

POR QUE UM SCRIPT E NÃO SÓ UM MARKDOWN?
- Tabelas Rich são mais legíveis no terminal (cores, alinhamento)
- Você pode FILTRAR por framework ou por conceito
- Pode ESTENDER adicionando novos frameworks facilmente
- Serve como base para um "avaliador de frameworks" automatizado

CONCEITO CHAVE — Mapeamento Bidirecional:
Quando você vê um `@tool` no LangChain, precisa saber que POR BAIXO
ele faz a mesma coisa que nosso `tool_registry` do módulo 03.
E quando você vê nosso `tool_registry`, precisa saber que FRAMEWORKS
chamam isso de `@tool`, `function`, `action` ou `skill`.

ANALOGIA:
Pense em idiomas:
- "Água" em português = "Water" em inglês = "Eau" em francês
- O conceito é o MESMO, a palavra muda conforme o "framework" (idioma)
- Quem entende o CONCEITO aprende qualquer "idioma" rápido
- Quem só decorou a palavra, fica perdido ao trocar de idioma

FRAMEWORKS COBERTOS:
1. LangChain / LangGraph  — O mais popular, enorme ecossistema
2. Semantic Kernel         — SDK Microsoft, integração Azure
3. CrewAI                  — Foco em multiagente colaborativo
4. AutoGen                 — Microsoft, conversação entre agentes
5. LlamaIndex              — Especialista em RAG e dados

DIAGRAMA — Fluxo deste script:

  ╔══════════════════════════════════════════════╗
  ║  1. Definir dados de mapeamento (dataclass)  ║
  ║     ↓                                        ║
  ║  2. Montar tabela geral (todos frameworks)   ║
  ║     ↓                                        ║
  ║  3. Análise por framework (pontos fortes)    ║
  ║     ↓                                        ║
  ║  4. Guia de decisão interativo               ║
  ║     ↓                                        ║
  ║  5. Exercícios sugeridos                     ║
  ╚══════════════════════════════════════════════╝

Tópicos cobertos:
1. Mapeamento manual → framework em tabela Rich
2. Análise de pontos fortes e fracos por framework
3. Árvore de decisão para escolher o framework certo
4. Exercícios de reflexão para fixar o aprendizado
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# ============================================================
# CONSOLE RICH — Nosso "pintor" de tabelas e painéis no terminal
# ============================================================
console = Console()


# ============================================================
# 1. ESTRUTURA DE DADOS — Mapeamento conceito → frameworks
# ============================================================
# Usamos dataclass para organizar cada conceito de forma que
# seja fácil adicionar novos frameworks ou conceitos depois.
#
# PADRÃO DE DESIGN:
# Cada MapeamentoConceito tem:
# - conceito_manual: o que FIZEMOS no treinamento (ex: "System prompt")
# - modulo_origem: onde aparece no treinamento (ex: "módulo 01")
# - equivalentes: dicionário framework -> nome da abstração
#
# POR QUE DATACLASS?
# - Validação automática de tipos (Python 3.10+)
# - __repr__ legível para debug
# - Fácil de serializar (JSON, CSV) se precisar exportar
# ============================================================

@dataclass
class MapeamentoConceito:
    """
    Representa UM conceito que implementamos manualmente e seus
    equivalentes em diferentes frameworks.

    Atributos:
    - conceito_manual: o que construímos no treinamento
    - modulo_origem: em qual módulo do treinamento aparece
    - equivalentes: dict com {nome_framework: nome_abstração}

    EXEMPLO:
    MapeamentoConceito(
        conceito_manual="System prompt",
        modulo_origem="módulo 01, 06",
        equivalentes={
            "LangChain":       "PromptTemplate / system message",
            "Semantic Kernel":  "instructions",
            "CrewAI":           "role / backstory",
        }
    )
    """
    conceito_manual: str
    modulo_origem: str
    equivalentes: dict[str, str] = field(default_factory=dict)


@dataclass
class FrameworkInfo:
    """
    Informações resumidas sobre um framework para análise comparativa.

    Atributos:
    - nome: nome do framework
    - foco_principal: em que ele é melhor
    - ponto_forte: maior vantagem
    - ponto_fraco: maior desvantagem
    - quando_usar: cenários ideais
    - quando_evitar: cenários onde não é boa opção
    """
    nome: str
    foco_principal: str
    ponto_forte: str
    ponto_fraco: str
    quando_usar: str
    quando_evitar: str


# ============================================================
# 2. BASE DE DADOS — Todos os mapeamentos do treinamento
# ============================================================
# Esta lista contém CADA conceito que construímos nos módulos
# 01-14 e seus equivalentes nos 5 frameworks principais.
#
# COMO USAR ESTA BASE:
# - Ao estudar um framework novo, procure o conceito manual aqui
# - Ao construir algo novo, veja como cada framework resolveria
# - Para entrevistas técnicas: demonstre que sabe "as duas línguas"
#
# SE QUISER ADICIONAR UM FRAMEWORK:
# Basta adicionar uma nova chave no dicionário `equivalentes`
# de cada conceito. Ex: "MeuFramework": "minha_abstração"
# ============================================================

MAPEAMENTOS: list[MapeamentoConceito] = [
    MapeamentoConceito(
        conceito_manual="System prompt",
        modulo_origem="módulo 01, 06",
        equivalentes={
            "LangChain": "PromptTemplate / system message",
            "Semantic Kernel": "instructions / prompt config",
            "CrewAI": "role / backstory / goal",
            "AutoGen": "system_message",
            "LlamaIndex": "system_prompt / service_context",
        },
    ),
    MapeamentoConceito(
        conceito_manual="Histórico de mensagens",
        modulo_origem="módulo 02",
        equivalentes={
            "LangChain": "ConversationBufferMemory / ChatMessageHistory",
            "Semantic Kernel": "ChatHistory",
            "CrewAI": "memory (built-in)",
            "AutoGen": "conversation state / chat history",
            "LlamaIndex": "ChatMemoryBuffer",
        },
    ),
    MapeamentoConceito(
        conceito_manual="Tool schema + registry",
        modulo_origem="módulo 03",
        equivalentes={
            "LangChain": "@tool decorator / StructuredTool",
            "Semantic Kernel": "@kernel_function / skill",
            "CrewAI": "@tool decorator / Tool class",
            "AutoGen": "register_function / FunctionCall",
            "LlamaIndex": "FunctionTool / QueryEngineTool",
        },
    ),
    MapeamentoConceito(
        conceito_manual="Pipeline de guardrails",
        modulo_origem="módulo 04",
        equivalentes={
            "LangChain": "RunnablePassthrough / callbacks / middleware",
            "Semantic Kernel": "filters / function_invocation_filter",
            "CrewAI": "guardrails (via callback)",
            "AutoGen": "message_transform / middleware",
            "LlamaIndex": "QueryTransform / response_synthesizer",
        },
    ),
    MapeamentoConceito(
        conceito_manual="HITL (Human-in-the-Loop)",
        modulo_origem="módulo 05",
        equivalentes={
            "LangChain": "LangGraph interrupt / human_approval node",
            "Semantic Kernel": "approval filter / function filter",
            "CrewAI": "human_input=True no Task",
            "AutoGen": "HumanProxyAgent / ask_human_input",
            "LlamaIndex": "HumanInputLLM (experimental)",
        },
    ),
    MapeamentoConceito(
        conceito_manual="Avaliação e benchmark",
        modulo_origem="módulo 07",
        equivalentes={
            "LangChain": "LangSmith evaluators / run scoring",
            "Semantic Kernel": "custom evaluators",
            "CrewAI": "task callbacks / output scoring",
            "AutoGen": "evaluation harness (custom)",
            "LlamaIndex": "Ragas / evaluation modules",
        },
    ),
    MapeamentoConceito(
        conceito_manual="Arquitetura de agentes",
        modulo_origem="módulo 08",
        equivalentes={
            "LangChain": "LangGraph (graph-based orchestration)",
            "Semantic Kernel": "Planner / StepwisePlanner",
            "CrewAI": "Crew / Process (sequential, hierarchical)",
            "AutoGen": "GroupChat / ConversableAgent",
            "LlamaIndex": "AgentRunner / QueryPipeline",
        },
    ),
    MapeamentoConceito(
        conceito_manual="Trace e observabilidade",
        modulo_origem="módulo 09",
        equivalentes={
            "LangChain": "LangSmith tracing / callbacks",
            "Semantic Kernel": "OpenTelemetry integration",
            "CrewAI": "verbose=True / callbacks",
            "AutoGen": "logging / message hooks",
            "LlamaIndex": "Phoenix / callback_manager",
        },
    ),
    MapeamentoConceito(
        conceito_manual="Memória de longo prazo (RAG)",
        modulo_origem="módulo 10",
        equivalentes={
            "LangChain": "VectorStoreRetriever / RetrievalQA",
            "Semantic Kernel": "MemoryStore / TextMemoryPlugin",
            "CrewAI": "long_term_memory / knowledge",
            "AutoGen": "Retrieve (rag_agent) / TeachableAgent",
            "LlamaIndex": "VectorStoreIndex / query_engine",
        },
    ),
    MapeamentoConceito(
        conceito_manual="Retry e fallback",
        modulo_origem="módulo 11",
        equivalentes={
            "LangChain": "FallbackLLM / retry with_structured_output",
            "Semantic Kernel": "retry policy / HttpRetryConfig",
            "CrewAI": "max_retry_limit no agente",
            "AutoGen": "retry_wait / max_retry",
            "LlamaIndex": "LiteLLM router / fallback",
        },
    ),
    MapeamentoConceito(
        conceito_manual="Streaming de resposta",
        modulo_origem="módulo 14",
        equivalentes={
            "LangChain": "StreamingCallbackHandler / astream",
            "Semantic Kernel": "streaming_chat_message_content",
            "CrewAI": "streaming (via LLM config)",
            "AutoGen": "stream=True no LLM config",
            "LlamaIndex": "stream_chat / StreamingResponse",
        },
    ),
]


# ============================================================
# 3. FRAMEWORKS — Informações detalhadas para análise
# ============================================================
# Cada framework tem características únicas. Esta base permite
# gerar tabelas de comparação e guias de decisão.
# ============================================================

FRAMEWORKS: list[FrameworkInfo] = [
    FrameworkInfo(
        nome="LangChain / LangGraph",
        foco_principal="Ecossistema completo de agentes e pipelines",
        ponto_forte=(
            "Maior ecossistema: 700+ integrações "
            "(LLMs, bancos vetoriais, APIs)"
        ),
        ponto_fraco=(
            "Abstrações profundas — difícil debugar quando algo dá errado"
        ),
        quando_usar="Projetos que precisam de muitas integrações prontas",
        quando_evitar=(
            "Projetos simples ou quando precisa de controle total do fluxo"
        ),
    ),
    FrameworkInfo(
        nome="Semantic Kernel (Microsoft)",
        foco_principal="Orquestração de IA com foco empresarial",
        ponto_forte=(
            "Integração nativa com Azure OpenAI e ecossistema Microsoft"
        ),
        ponto_fraco=(
            "Comunidade menor, menos exemplos disponíveis"
        ),
        quando_usar="Empresas que já usam stack Microsoft / Azure",
        quando_evitar=(
            "Projetos que precisam de muitas integrações não-Microsoft"
        ),
    ),
    FrameworkInfo(
        nome="CrewAI",
        foco_principal="Agentes multiagente colaborativos",
        ponto_forte=(
            "Conceito intuitivo de 'equipe' com papéis (Agent + Task + Crew)"
        ),
        ponto_fraco="Menos flexível para cenários fora do modelo de 'crew'",
        quando_usar=(
            "Cenários multiagente com papéis bem definidos (como módulo 08)"
        ),
        quando_evitar="Agente único simples ou pipelines sem 'equipe'",
    ),
    FrameworkInfo(
        nome="AutoGen (Microsoft)",
        foco_principal="Conversação multi-turno entre agentes",
        ponto_forte="Foco em diálogo: agentes conversam, debatem e revisam",
        ponto_fraco="API em evolução rápida — breaking changes frequentes",
        quando_usar="Cenários onde agentes precisam 'conversar' entre si",
        quando_evitar=(
            "Projetos que precisam de estabilidade de API a longo prazo"
        ),
    ),
    FrameworkInfo(
        nome="LlamaIndex",
        foco_principal="RAG e pipelines de dados",
        ponto_forte="Melhor suporte a ingestão e indexação de documentos",
        ponto_fraco="Menos foco em agentes, mais em pipeline de dados",
        quando_usar=(
            "Quando o foco é ingestão e consulta de documentos "
            "(como módulo 10)"
        ),
        quando_evitar="Quando o foco é orquestração de agentes e não dados",
    ),
]


# ============================================================
# 4. TABELA GERAL — Todos os conceitos × todos os frameworks
# ============================================================
# Esta função monta a tabela principal de mapeamento.
#
# COMO FUNCIONA:
# 1. Cria uma tabela Rich com colunas: Conceito | Módulo | Framework1 | ...
# 2. Itera sobre cada mapeamento e adiciona uma linha
# 3. Cada célula mostra o nome da abstração naquele framework
#
# POR QUE RICH TABLE?
# - Alinhamento automático de colunas
# - Cores para facilitar leitura
# - Truncamento inteligente de textos longos
# - Funciona em qualquer terminal
#
# EXERCÍCIO SUGERIDO:
# Adicione um novo framework à tabela (ex: Haystack, DSPy)
# e preencha os equivalentes pesquisando na documentação oficial.
# ============================================================

def exibir_tabela_mapeamento() -> None:
    """
    Monta e exibe a tabela geral de mapeamento Manual → Frameworks.

    A tabela mostra:
    - Coluna 1: conceito que implementamos manualmente
    - Coluna 2: módulo de origem no treinamento
    - Colunas 3-7: equivalente em cada framework (LangChain, SK, etc.)

    COMO LER A TABELA:
    Cada linha é um conceito. Leia horizontalmente para ver como
    cada framework chama a mesma coisa. Ex:
    "Tool registry" → LangChain chama "@tool", SK chama "@kernel_function"
    """
    # ── Cabeçalho da tabela ──
    tabela = Table(
        title="📋 MAPEAMENTO COMPLETO: Manual → Frameworks",
        title_style="bold cyan",
        show_lines=True,
        expand=True,
    )

    # Colunas fixas: conceito e módulo de origem
    tabela.add_column("Conceito Manual", style="bold yellow", min_width=20)
    tabela.add_column("Módulo", style="dim", min_width=10)

    # Colunas dinâmicas: uma por framework
    # Usamos cores diferentes para facilitar a diferenciação visual
    cores_frameworks = ["green", "blue", "magenta", "red", "cyan"]
    nomes_frameworks = [
        "LangChain", "Semantic Kernel", "CrewAI", "AutoGen", "LlamaIndex"
    ]

    for nome, cor in zip(nomes_frameworks, cores_frameworks):
        tabela.add_column(nome, style=cor, min_width=15)

    # ── Preenchimento: uma linha por conceito ──
    for mapeamento in MAPEAMENTOS:
        # Busca o equivalente em cada framework (ou "—" se não mapeado)
        valores = [
            mapeamento.equivalentes.get(fw, "—") for fw in nomes_frameworks
        ]
        tabela.add_row(
            mapeamento.conceito_manual,
            mapeamento.modulo_origem,
            *valores,
        )

    console.print()
    console.print(tabela)
    console.print()


# ============================================================
# 5. ANÁLISE POR FRAMEWORK — Detalhes individuais
# ============================================================
# Mostra painel detalhado para cada framework com pontos fortes,
# fracos e quando usar/evitar.
#
# FORMATO DO PAINEL:
# ┌── LangChain / LangGraph ──────────────────────┐
# │ 🎯 Foco: Ecossistema completo de agentes...    │
# │ ✅ Forte: Maior ecossistema: 700+ integrações  │
# │ ❌ Fraco: Abstrações profundas — difícil debug  │
# │ 👍 Usar: Muitas integrações prontas             │
# │ 👎 Evitar: Projetos simples...                  │
# └────────────────────────────────────────────────┘
#
# POR QUE PAINÉIS E NÃO TEXTO PURO?
# - Agrupamento visual ajuda na memorização
# - Emojis tornam o scan mais rápido
# - Bordas delimitam claramente cada framework
# ============================================================

def exibir_analise_frameworks() -> None:
    """
    Exibe um painel Rich detalhado para cada framework.

    Cada painel contém:
    - Foco principal do framework
    - Ponto forte (maior vantagem)
    - Ponto fraco (maior desvantagem)
    - Quando usar (cenário ideal)
    - Quando evitar (cenário ruim)

    COMO USAR:
    Leia todos os painéis e COMPARE mentalmente. Pergunte-se:
    "Para o MEU próximo projeto, qual ponto forte é mais importante?"
    """
    console.print()
    console.print("[bold cyan]🔍 ANÁLISE DETALHADA POR FRAMEWORK[/]")
    console.print()

    for fw in FRAMEWORKS:
        # Monta o conteúdo do painel com emojis para scan rápido
        conteudo = (
            f"🎯 [bold]Foco:[/] {fw.foco_principal}\n"
            f"✅ [green]Forte:[/] {fw.ponto_forte}\n"
            f"❌ [red]Fraco:[/] {fw.ponto_fraco}\n"
            f"👍 [green]Usar quando:[/] {fw.quando_usar}\n"
            f"👎 [red]Evitar quando:[/] {fw.quando_evitar}"
        )

        console.print(Panel(
            conteudo,
            title=f"[bold]{fw.nome}[/]",
            border_style="blue",
            expand=False,
            width=75,
        ))
        console.print()


# ============================================================
# 6. ÁRVORE DE DECISÃO — Qual framework escolher?
# ============================================================
# Monta uma árvore visual (Rich Tree) que guia a decisão de
# quando usar cada framework (ou ficar no manual).
#
# A ÁRVORE SEGUE A LÓGICA:
# Preciso de um agente?
#   ├── É aprendizado? → Manual
#   ├── É produção?
#   │    ├── Muitas integrações? → LangChain
#   │    ├── Stack Microsoft? → Semantic Kernel
#   │    ├── Multiagente? → CrewAI / AutoGen
#   │    └── RAG é o foco? → LlamaIndex
#   └── Não sei → Manual primeiro, migra depois
#
# POR QUE ÁRVORE E NÃO FLUXOGRAMA?
# - Rich Tree renderiza nativamente no terminal
# - Fácil de expandir (basta adicionar branches)
# - Hierarquia visual intuitiva
# ============================================================

def exibir_arvore_decisao() -> None:
    """
    Monta e exibe uma árvore de decisão para escolha de framework.

    A árvore começa com a pergunta "Preciso de um agente?" e ramifica
    em cenários que levam à recomendação de framework ou abordagem manual.

    COMO USAR:
    1. Comece pela raiz (pergunta principal)
    2. Siga o ramo que descreve seu cenário
    3. A folha final é a recomendação

    EXERCÍCIO SUGERIDO:
    Pense no seu próximo projeto e percorra a árvore. O resultado
    faz sentido? Se não, que critério está faltando?
    """
    # Árvore Rich — cada add() cria um ramo, sub-ramos são aninhados
    arvore = Tree(
        "🤔 [bold cyan]Preciso de um agente de IA?[/]",
        guide_style="bold blue",
    )

    # ── Ramo 1: Aprendizado ──
    ramo_aprendizado = arvore.add("📚 É aprendizado / prova de conceito?")
    ramo_aprendizado.add(
        "✅ [green]Faça MANUAL[/] (como neste treinamento) — "
        "entenda a mecânica antes de abstrair"
    )

    # ── Ramo 2: Produção ──
    ramo_producao = arvore.add("🏭 É um produto em produção?")

    ramo_integracoes = ramo_producao.add(
        "🔌 Preciso de muitas integrações (LLMs, bancos, APIs)?"
    )
    ramo_integracoes.add(
        "✅ [green]LangChain / LangGraph[/] — 700+ integrações prontas"
    )

    ramo_microsoft = ramo_producao.add("☁️ Já uso stack Microsoft / Azure?")
    ramo_microsoft.add(
        "✅ [green]Semantic Kernel[/] — integração nativa com Azure OpenAI"
    )

    ramo_multi = ramo_producao.add(
        "👥 Preciso de multiagente com papéis definidos?"
    )
    ramo_multi.add("✅ [green]CrewAI[/] — modelo de equipe intuitivo")
    ramo_multi.add(
        "✅ [green]AutoGen[/] — se preciso de conversação entre agentes"
    )

    ramo_rag = ramo_producao.add("📄 O foco é RAG / ingestão de documentos?")
    ramo_rag.add(
        "✅ [green]LlamaIndex[/] — melhor suporte a indexação e consulta"
    )

    ramo_controle = ramo_producao.add(
        "🎛️ Preciso de controle total sobre o fluxo?"
    )
    ramo_controle.add(
        "✅ [green]MANUAL + libs pontuais[/] (Groq, Pydantic, Rich) — "
        "máximo controle, mínima abstração"
    )

    # ── Ramo 3: Não sei ──
    ramo_nao_sei = arvore.add("🤷 Não sei o que preciso ainda?")
    ramo_nao_sei.add(
        "✅ [green]Manual PRIMEIRO[/] → migre para framework quando "
        "souber exatamente o que precisa"
    )

    console.print()
    console.print("[bold cyan]🌳 ÁRVORE DE DECISÃO: Manual vs Framework[/]")
    console.print()
    console.print(arvore)
    console.print()


# ============================================================
# 7. COMPARATIVO QUANTITATIVO — Tabela de scores
# ============================================================
# Atribui uma nota (1-5) para cada framework em critérios que
# importam na prática: facilidade, debug, comunidade, etc.
#
# AS NOTAS SÃO SUBJETIVAS E BASEADAS NO ESTADO DE 2024-2025
# Frameworks evoluem rápido — revise periodicamente.
#
# COMO INTERPRETAR:
# 5 = Excelente nesse critério
# 4 = Bom
# 3 = Aceitável
# 2 = Fraco
# 1 = Ruim nesse critério
# ============================================================

def exibir_tabela_scores() -> None:
    """
    Exibe tabela comparativa com notas (1-5) para cada framework
    em critérios práticos de seleção.

    Critérios avaliados:
    - Facilidade de início (quão rápido você começa a produzir)
    - Debugabilidade (quão fácil é encontrar e resolver bugs)
    - Comunidade (tamanho da comunidade e disponibilidade de ajuda)
    - Integrações (quantidade de integrações prontas)
    - Estabilidade API (frequência de breaking changes)
    - Documentação (qualidade e completude da documentação)
    """
    # Dados: (framework, facilidade, debug,
    # comunidade, integrações, estabilidade, docs)
    scores: list[tuple[str, int, int, int, int, int, int]] = [
        ("Manual (treinamento)", 3, 5, 0, 1, 5, 0),
        ("LangChain / LangGraph", 4, 2, 5, 5, 3, 4),
        ("Semantic Kernel", 3, 3, 3, 3, 4, 4),
        ("CrewAI", 5, 3, 4, 3, 3, 3),
        ("AutoGen", 3, 2, 3, 3, 2, 3),
        ("LlamaIndex", 4, 3, 4, 4, 3, 4),
    ]

    tabela = Table(
        title="📊 COMPARATIVO QUANTITATIVO (notas 1-5)",
        title_style="bold cyan",
        show_lines=True,
    )

    tabela.add_column("Framework", style="bold yellow", min_width=22)
    tabela.add_column("Início", justify="center", style="green")
    tabela.add_column("Debug", justify="center", style="red")
    tabela.add_column("Comunidade", justify="center", style="blue")
    tabela.add_column("Integrações", justify="center", style="magenta")
    tabela.add_column("Estabilidade", justify="center", style="cyan")
    tabela.add_column("Docs", justify="center", style="white")

    # Converte nota numérica para representação visual com estrelas
    def nota_visual(nota: int) -> str:
        """Converte nota 1-5 em estrelas para facilitar a leitura."""
        if nota == 0:
            return "N/A"
        return "★" * nota + "☆" * (5 - nota)

    for (
        nome, inicio, debug, comunidade, integracoes, estabilidade, docs
    ) in scores:
        tabela.add_row(
            nome,
            nota_visual(inicio),
            nota_visual(debug),
            nota_visual(comunidade),
            nota_visual(integracoes),
            nota_visual(estabilidade),
            nota_visual(docs),
        )

    console.print()
    console.print(tabela)
    console.print()

    # ── Legenda ──
    console.print(
        "[dim]★★★★★ = Excelente | "
        "★★★☆☆ = Aceitável | "
        "★☆☆☆☆ = Fraco | "
        "N/A = Não aplicável[/]"
    )
    console.print(
        "[dim]Notas subjetivas baseadas no estado 2024-2025."
        " Revisar periodicamente.[/]"
    )
    console.print()


# ============================================================
# 8. REGRAS PRÁTICAS — Resumo para consulta rápida
# ============================================================

def exibir_regras_praticas() -> None:
    """
    Exibe as regras práticas de quando usar/evitar frameworks.

    REGRA DE OURO:
    "Aprenda manual → Use framework quando precisar de produtividade
    → Fuja do framework quando ele esconder demais."

    COMO USAR:
    Imprima esta seção e cole na parede (sério). Consulte antes
    de cada novo projeto de agente.
    """
    # ── Quando USAR framework ──
    usar = Panel(
        "• Você precisa de [bold]produtividade[/]"
        " (iteração rápida, muitas integrações)\n"
        "• O framework tem [bold]manutenção ativa[/] e comunidade grande\n"
        "• Você [bold]entende[/] o que o framework faz"
        " (não é caixa-preta pra você)\n"
        "• O projeto vai para [bold]produção[/] "
        "com equipe que mantém o código",
        title="[bold green]✅ QUANDO USAR FRAMEWORK[/]",
        border_style="green",
        expand=False,
        width=75,
    )

    # ── Quando NÃO usar framework ──
    nao_usar = Panel(
        "• Você está [bold]aprendendo[/] (framework esconde a mecânica)\n"
        "• O framework [bold]esconde demais[/] "
        "o fluxo crítico do seu domínio\n"
        "• Você precisa de [bold]controle total[/] "
        "sobre cada decisão do agente\n"
        "• O framework tem [bold]breaking changes[/] "
        "frequentes e sua equipe é pequena",
        title="[bold red]❌ QUANDO NÃO USAR FRAMEWORK[/]",
        border_style="red",
        expand=False,
        width=75,
    )

    # ── Regra de ouro ──
    regra = Panel(
        "[bold]Aprenda manualmente[/] para entender a mecânica.\n"
        "[bold]Use framework[/] quando precisar de produtividade, "
        "integrações e escala.\n"
        "[bold]Fuja do framework[/] quando ele esconder "
        "demais o fluxo crítico do domínio.",
        title="[bold yellow]🎯 REGRA DE OURO[/]",
        border_style="yellow",
        expand=False,
        width=75,
    )

    console.print()
    console.print(usar)
    console.print(nao_usar)
    console.print(regra)
    console.print()


# ============================================================
# 9. PONTO DE ENTRADA — Demonstração completa
# ============================================================
# Executa todas as visualizações em sequência para uma
# experiência de aprendizado completa no terminal.
#
# ORDEM DE EXECUÇÃO:
# 1. Tabela geral (visão macro: todos conceitos × frameworks)
# 2. Tabela de scores (comparativo quantitativo)
# 3. Análise por framework (detalhes individuais)
# 4. Árvore de decisão (guia prático)
# 5. Regras práticas (resumo para consulta rápida)
#
# EXERCÍCIOS SUGERIDOS APÓS RODAR O SCRIPT:
# 1. Adicione um 6º framework (Haystack, DSPy, etc.)
# 2. Mude as notas dos scores e justifique sua avaliação
# 3. Escolha UM framework e reimplemente o agente de boletos
# 4. Compare: mais fácil? Mais difícil de debugar? Por quê?
# 5. Documente sua conclusão: "Para este caso, X foi melhor porque..."
# ============================================================

if __name__ == "__main__":
    console.print(Panel(
        "[bold]MÓDULO 12.1 — COMPARATIVO MANUAL vs FRAMEWORKS[/]\n\n"
        "Este script exibe tabelas comparativas entre os conceitos\n"
        "que construímos manualmente e como frameworks populares\n"
        "resolvem os mesmos problemas.",
        title="[bold cyan]🗺️ MAPEAMENTO COMPLETO[/]",
        border_style="cyan",
        width=75,
    ))

    # 1. Tabela geral de mapeamento
    exibir_tabela_mapeamento()

    # 2. Comparativo quantitativo com scores
    exibir_tabela_scores()

    # 3. Análise detalhada de cada framework
    exibir_analise_frameworks()

    # 4. Árvore de decisão
    exibir_arvore_decisao()

    # 5. Regras práticas
    exibir_regras_praticas()

    console.print(
        "[bold green]✅ Fim do comparativo! "
        "Revise e escolha com sabedoria.[/]"
    )
