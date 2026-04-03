# ============================================================
# MÓDULO 12 - MANUAL vs FRAMEWORKS: MAPEAMENTO COMPLETO
# ============================================================

## O que este módulo ensina?

Durante todo o treinamento (módulos 01 a 11), construímos um agente de IA
**do zero**, sem usar nenhum framework. Fizemos isso de propósito — para que
você entenda **cada peça** que compõe um agente.

Agora, vamos mapear cada conceito que implementamos manualmente para o que
frameworks populares fazem automaticamente. Isso vai te preparar para:
- Avaliar frameworks com **olhar crítico** (saber o que está "por trás")
- Migrar para um framework quando fizer sentido
- Não ficar "preso" a um framework por falta de entendimento

## Por que construímos tudo manualmente primeiro?

> **ANALOGIA:**
> Antes de usar uma calculadora, você aprende a fazer conta no papel.
> Não porque o papel é melhor, mas porque quando a calculadora der um
> resultado estranho, você sabe que algo está errado.

Frameworks são **caixas-pretas produtivas** — ótimas para velocidade,
mas perigosas se você não entende o que acontece dentro delas.

## Mapeamento completo: Manual → Framework

| Conceito manual | Onde aparece neste treinamento | Equivalente típico em framework | Exemplos de frameworks |
|---|---|---|---|
| **System prompt** | módulo 01 (estilos de prompt), módulo 06 (agente boletos) | `PromptTemplate`, `system message`, `instructions` | LangChain, CrewAI, Semantic Kernel |
| **Histórico de mensagens** | módulo 02 (agente com memória) | `memory`, `conversation state`, `chat history` | LangChain Memory, AutoGen |
| **Tool schema + registry** | módulo 03 (tools com JSON Schema) | `@tool` decorator, `function`, `action`, `skill` | LangChain Tools, OpenAI Functions, Semantic Kernel Skills |
| **Pipeline de guardrails** | módulo 04 (validação entrada/saída) | `middleware`, `callback`, `validator`, `guardrail` | Guardrails AI, NeMo Guardrails, LangChain Callbacks |
| **HITL (Human-in-the-Loop)** | módulo 05 (aprovação humana por risco) | `approval node`, `human checkpoint`, `interrupt` | LangGraph interrupt, CrewAI human input |
| **Avaliação e benchmark** | módulo 07 (métricas + LLM-as-Judge) | `evaluator`, `benchmark`, `run comparison` | LangSmith, Ragas, Phoenix |
| **Trace e observabilidade** | módulo 09 (TraceRecorder + JSONL) | `tracing`, `spans`, `observability` | LangSmith, Phoenix, OpenTelemetry |
| **Retry e fallback** | módulo 11 (ClienteLLMResiliente) | `runtime policy`, `resilience middleware`, `retry` | LiteLLM, LangChain FallbackLLM |
| **Memória de longo prazo** | módulo 10 (keyword search + JSON) | `vector store`, `retriever`, `RAG chain` | LangChain VectorStore, LlamaIndex, Chroma |
| **Streaming de resposta** | módulo 14 (exibição incremental) | `stream`, `callback handler`, `on_llm_new_token` | LangChain Streaming, Groq stream=True |

## Frameworks mais relevantes em 2026

### 🔷 LangChain / LangGraph
- **O que é:** Framework Python/JS mais popular para agentes
- **Ponto forte:** Enorme ecossistema de integrações (LLMs, bancos vetoriais, tools)
- **Ponto fraco:** Abstrações muito profundas — difícil debugar quando algo dá errado
- **Quando usar:** Projetos que precisam de muitas integrações prontas

### 🔷 Semantic Kernel (Microsoft)
- **O que é:** SDK da Microsoft para orquestração de IA
- **Ponto forte:** Integração nativa com Azure OpenAI e ecossistema Microsoft
- **Ponto fraco:** Menos exemplos e comunidade menor que LangChain
- **Quando usar:** Empresas que já usam stack Microsoft/Azure

### 🔷 CrewAI
- **O que é:** Framework focado em agentes multiagente colaborativos
- **Ponto forte:** Conceito de "equipe" de agentes com papéis definidos
- **Ponto fraco:** Menos flexível para cenários que não se encaixam no modelo de "crew"
- **Quando usar:** Cenários multiagente com papéis bem definidos (módulo 08)

### 🔷 AutoGen (Microsoft)
- **O que é:** Framework para agentes conversacionais multi-turno
- **Ponto forte:** Foco em conversação entre agentes (ideal para debates/revisão)
- **Ponto fraco:** API em evolução rápida (breaking changes frequentes)
- **Quando usar:** Cenários onde agentes precisam "conversar" entre si

### 🔷 LlamaIndex
- **O que é:** Framework especializado em RAG (Retrieval-Augmented Generation)
- **Ponto forte:** Melhor suporte a fontes de dados e indexação
- **Ponto fraco:** Menos foco em agentes, mais em pipeline de dados
- **Quando usar:** Quando o foco é ingestão e consulta de documentos (módulo 10)

## Diagrama de decisão: Manual vs Framework

```
Preciso de um agente de IA?
       │
       ├── É um prova de conceito / aprendizado?
       │      └── ✅ Faça MANUAL (como neste treinamento)
       │
       ├── É um produto em produção?
       │      ├── Preciso de muitas integrações (LLMs, bancos, APIs)?
       │      │      └── ✅ Use FRAMEWORK (LangChain, Semantic Kernel)
       │      │
       │      ├── O domínio é muito específico e eu preciso de controle total?
       │      │      └── ✅ Faça MANUAL + bibliotecas pontuais (Groq, Pydantic)
       │      │
       │      └── Preciso de multiagente com papéis definidos?
       │             └── ✅ Use CrewAI ou AutoGen
       │
       └── Não sei o que preciso ainda?
              └── ✅ Faça manual PRIMEIRO, migre depois se necessário
```

## Regras práticas

### ✅ Quando USAR framework:
- Você precisa de **produtividade** (iteração rápida, muitas integrações)
- O framework tem **manutenção ativa** e comunidade grande
- Você **entende** o que o framework faz (não é caixa-preta pra você)
- O projeto vai para **produção** com equipe que mantém o código

### ❌ Quando NÃO usar framework:
- Você está **aprendendo** (framework esconde a mecânica)
- O framework **esconde demais** o fluxo crítico do seu domínio
- Você precisa de **controle total** sobre cada decisão do agente
- O framework tem **breaking changes** frequentes e sua equipe é pequena

### 🎯 Regra de ouro:
> **Aprenda manualmente para entender a mecânica.**
> **Use framework quando precisar de produtividade, integrações e escalabilidade.**
> **Fuja de framework quando ele esconder demais o fluxo crítico do seu domínio.**

## Exercício sugerido

1. Escolha UN framework da tabela acima (ex: LangChain)
2. Reimplemente o agente de boletos (módulo 06) usando esse framework
3. Compare: quantas linhas de código? Ficou mais fácil de entender?
4. Tente debugar um erro — foi fácil encontrar onde o problema está?
5. Documente suas conclusões: "para este caso, manual/framework foi melhor porque..."