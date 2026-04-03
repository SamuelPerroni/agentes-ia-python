# 🎓 GUIA DO TREINAMENTO: Agentes de IA com Python (Hands-On)

---

## 📋 Visão Geral

| Item | Detalhe |
| ------ | --------- |
| **Duração estimada** | 6 a 8 horas |
| **Nível** | Intermediário (requer Python básico) |
| **LLM** | Groq API (gratuita) + Llama 3.3 70B |
| **Projeto Final** | Agente de Leitura de Boletos Bancários |

### Objetivo

Construir, **na prática**, um agente de IA completo em Python, evoluindo do conceito mais simples (chamada à LLM) até um agente com ferramentas, guardrails, supervisão humana e avaliação automatizada.

### O que os participantes vão aprender

- Como funciona um agente de IA por dentro
- Engenharia de prompts para agentes
- Implementação de tool calling
- Proteção com guardrails (entrada e saída)
- Camada human-in-the-loop (HITL)
- Avaliação de qualidade do agente
- Observabilidade, resiliência e governança
- Memória de longo prazo e padrões arquiteturais

---

## 🔧 Pré-requisitos e Setup (15 min)

### Antes do treinamento

1. **Python 3.10+** instalado
2. **VS Code** (recomendado)
3. Conta gratuita no **Groq**: <https://console.groq.com/>

### Setup no início do treinamento

```bash
# 1. Abrir o projeto
cd treinamento_agentes_ia

# 2. Criar ambiente virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar API key
copy .env.example .env
# Editar .env e colocar a GROQ_API_KEY obtida em console.groq.com/keys
```

### Verificar que tudo funciona

```bash
python -c "from groq import Groq; print('✅ Groq instalado')"
python -c "from dotenv import load_dotenv; print('✅ dotenv instalado')"
python -c "from pydantic import BaseModel; print('✅ Pydantic instalado')"
python -c "from rich import print; print('✅ Rich instalado')"
```

---

## 📚 Estrutura dos Módulos

```text
modulo_01_prompts/          ← Engenharia de Prompts
modulo_02_agente_basico/    ← Primeiro Agente
modulo_03_tools/            ← Ferramentas (Tool Calling)
modulo_04_guardrails/       ← Segurança e Validação
modulo_05_hitl/             ← Human-in-the-Loop
modulo_06_agente_boletos/   ← Projeto Completo
modulo_07_avaliacao/        ← Avaliação de Resultados
modulo_08_arquiteturas/     ← Padrões de Arquitetura
modulo_09_observabilidade/  ← Tracing e Debugging
modulo_10_memoria_longo_prazo/ ← Memória de Longo Prazo
modulo_11_resiliencia/      ← Retry, fallback e degradação
modulo_12_frameworks/       ← Manual vs Frameworks
modulo_13_seguranca_governanca/ ← Checklist de Produção
modulo_14_streaming_ux/     ← Streaming e Experiência
```

---

## MÓDULO 1: Engenharia de Prompts (45 min)

### 📖 Teoria para apresentar

> **O que é um prompt?**
> É a instrução que enviamos à LLM. A qualidade do prompt determina diretamente a qualidade da resposta.

#### Conceitos-chave para explicar

1. **Tokens**: LLMs processam tokens, não palavras. Cada token ≈ ¾ de uma palavra
2. **Temperature**: Controla criatividade (0 = determinístico, 1 = criativo)
3. **System Prompt**: Instrução que define o comportamento base da LLM
4. **Context Window**: Limite de tokens que a LLM consegue processar

### 🖥️ Arquivo: `modulo_01_prompts/01_estilos_prompt.py`

#### Roteiro de execução

1. **Abrir o arquivo** e explicar a estrutura
2. **Zero-shot** (linha ~45): Executar e discutir a limitação
3. **Few-shot** (linha ~75): Mostrar como exemplos melhoram a consistência
4. **Chain-of-Thought** (linha ~115): Destacar como o raciocínio passo a passo melhora cálculos
5. **Role-playing** (linha ~145): Mostrar o impacto do system prompt na personalidade
6. **Structured Output** (linha ~190): Explicar por que agentes precisam de JSON

> **💡 Ponto de discussão**: "Quando usar cada estilo? Qual é melhor para agentes?"
> R: Agentes geralmente combinam Role-playing (system prompt) + Structured Output (JSON) + CoT (raciocínio)

### 🖥️ Arquivo: `modulo_01_prompts/02_boas_praticas.py`

#### Roteiro — Boas Práticas

1. Para cada prática, **executar o prompt RUIM primeiro**, depois o BOM
2. Deixar os participantes **verem a diferença** na prática
3. Destacar a **Prática 6 (ReAct)** - é o padrão usado por agentes

#### As 6 boas práticas

| # | Prática | Para que serve |
| --- | --------- | -------------- |
| 1 | Seja Específico | Definir O QUE, COMO e PARA QUEM |
| 2 | Use Delimitadores | Separar instrução de dados (segurança!) |
| 3 | Formato de Saída | Agentes precisam parsear = JSON |
| 4 | Saída de Emergência | Permitir "não sei" para evitar alucinação |
| 5 | Passos Numerados | Decomposição de tarefas complexas |
| 6 | Padrão ReAct | Think → Act → Observe para agentes |

> **Exercício rápido**: Peça aos participantes para escreverem um system prompt para um agente de atendimento ao cliente.

---

## MÓDULO 2: Construindo um Agente Básico (30 min)

### 📖 Teoria — Agente Básico

> **O que diferencia um agente de uma simples chamada à LLM?**
>
> - **LLM**: Pergunta → Resposta (uma vez)
> - **Agente**: Loop de Pergunta → Raciocínio → Ação → Observação → Resposta
>

```text
┌─────────────┐
│   Usuário    │
└──────┬───────┘
       │ input
       ▼
┌──────────────┐     ┌────────────┐
│    Agente    │────▶│    LLM     │
│   (Loop)     │◀────│  (Cérebro) │
└──────┬───────┘     └────────────┘
       │ output
       ▼
┌─────────────┐
│   Usuário    │
└─────────────┘
```

### 🖥️ Arquivo: `modulo_02_agente_basico/01_primeiro_agente.py`

1. **Mostrar** a demo automática (`demo_agente_simples`)
2. **Destacar** o problema: o agente NÃO lembra do que foi dito antes
3. **Ponto-chave**: Cada chamada à LLM é isolada (`stateless`)

### 🖥️ Arquivo: `modulo_02_agente_basico/02_agente_com_memoria.py`

1. **Mostrar** a classe `AgenteComMemoria`
2. **Explicar** que memória = lista de mensagens anteriores
3. **Executar** a demo e mostrar que agora ele referencia perguntas anteriores
4. **Destacar** o gerenciamento de contexto (truncar histórico quando muito grande)

> **Ponto de discussão**: "O que acontece quando o histórico fica maior que a context window?"
> R: Precisamos de estratégias: truncar, resumir ou usar memória vetorial

---

## MÓDULO 3: Tools - Ferramentas (45 min)

### 📖 Teoria — Tools

> **Tools são superpoderes do agente.**
> A LLM sozinha não acessa internet, não calcula, não lê banco de dados.
> Tools permitem que ela "aja" no mundo real.

```text
Fluxo de Tool Calling:

Usuário: "Quanto pago de multa num boleto de R$ 1000 vencido há 10 dias?"
                │
                ▼
         ┌──────────┐
         │   LLM    │  "Preciso calcular... vou chamar a tool!"
         └────┬─────┘
              │ tool_call: calcular_multa_juros(valor=1000, dias=10)
              ▼
     ┌─────────────────┐
     │  Código Python   │  Executa a função de verdade
     └────────┬────────┘
              │ resultado: {"multa": 20, "juros": 3.30, "total": 1023.30}
              ▼
         ┌──────────┐
         │   LLM    │  "O valor total a pagar é R$ 1.023,30..."
         └──────────┘
```

### 🖥️ Arquivo: `modulo_03_tools/01_criando_tools.py`

1. **Mostrar** as funções Python: são funções **normais**
2. **Explicar** o SCHEMA: é como a LLM "lê o manual" de cada ferramenta
3. **Destacar** o REGISTRY: mapeia nome → função
4. **Executar** a demo para ver cada tool funcionando isoladamente

#### Anatomia de um Tool Schema

```python
{
    "type": "function",
    "function": {
        "name": "nome_da_funcao",           # Como a LLM vai chamar
        "description": "O que ela faz...",    # Quando a LLM deve usar
        "parameters": {
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "O que é este parâmetro"  # Crucial!
                }
            },
            "required": ["param1"]
        }
    }
}
```

> **Dica**: A `description` da tool é FUNDAMENTAL. É ela que diz à LLM **quando** usar a tool.

### 🖥️ Arquivo: `modulo_03_tools/02_tool_calling.py`

1. **Explicar** o loop do agente com tools (o while com retry)
2. **Executar** os 4 testes e mostrar:
   - Teste 1: LLM chama `calcular_multa_juros`
   - Teste 2: LLM chama `validar_linha_digitavel` + `buscar_banco_por_codigo`
   - Teste 3: LLM encadeia múltiplas tools
   - Teste 4: LLM responde direto (sem tool)
3. **Destacar**: A LLM DECIDE sozinha quando chamar qual tool

> **Exercício**: Peça aos participantes para criar uma nova tool (ex: `consultar_segunda_via`) e adicioná-la ao schema.

---

## MÓDULO 4: Guardrails (45 min)

### 📖 Teoria — Guardrails

> **Guardrails são as cercas de segurança do agente.**
> Sem elas, o agente pode: vazar dados, ser manipulado, calcular errado ou agir fora do escopo.

```text
            GUARDRAILS DE ENTRADA                GUARDRAILS DE SAÍDA
┌─────────┐  ┌───────────────────┐  ┌────┐  ┌──────────────────┐  ┌──────────┐
│ Usuário  │─▶│ Injection? Escopo?│─▶│ LLM│─▶│ Schema? Conteúdo?│─▶│ Resposta │
│          │  │ PII? Tamanho?     │  │    │  │ Alucinação?      │  │          │
└─────────┘  └───────────────────┘  └────┘  └──────────────────┘  └──────────┘
             ❌ Bloqueia se falhar           ❌ Bloqueia se falhar
```

### 🖥️ Arquivo: `modulo_04_guardrails/01_validacao_entrada.py`

#### Guardrails de Entrada (ordem por custo)

| # | Guardrail | Custo | O que faz |
| --- | ----------- | ------- | ----------- |
| 1 | Tamanho | 💚 Mínimo | Limita chars do input |
| 2 | Prompt Injection | 💚 Regex | Detecta manipulação |
| 3 | PII | 💚 Regex | Detecta dados sensíveis |
| 4 | Escopo (keyword) | 💚 Rápido | Verifica se é sobre boletos |
| 5 | Escopo (LLM) | 🔴 Caro | Classificação com LLM |

1. **Executar** a demo mostrando cada caso de teste
2. **Destacar o conceito de pipeline**: verificações baratas primeiro
3. **Mostrar** o mascaramento de PII
4. **Discutir**: "Por que não usar só LLM para tudo?" (custo, latência)

### 🖥️ Arquivo: `modulo_04_guardrails/02_validacao_saida.py`

#### Guardrails de Saída

| # | Guardrail | O que valida |
| --- | ----------- | ------------- |
| 1 | Pydantic Schema | Formato correto (JSON válido, tipos certos) |
| 2 | Filtro de Conteúdo | Termos proibidos na resposta |
| 3 | Retry com Correção | Pede à LLM para corrigir JSON inválido |
| 4 | LLM-as-Judge | Verifica alucinação |

1. **Mostrar** o schema Pydantic `BoletoExtraido`
2. **Executar** teste com JSON válido vs inválido
3. **Destacar** o mecanismo de retry
4. **Explicar** LLM-as-Judge para checagem de alucinação

> **Ponto de discussão**: "Quais guardrails são essenciais para produção?"
> R: Todos de entrada + Pydantic + filtro de conteúdo. LLM-as-judge é bônus.

---

## MÓDULO 5: Human-in-the-Loop (30 min)

### 📖 Teoria — Human-in-the-Loop

> **Nem toda decisão deve ser automática.**
> Em operações financeiras, um erro do agente = prejuízo real.
> HITL é o "freio de mão" do agente.

```text
                        NÃO (risco baixo)
  Classificar Risco ──────────────────────▶ Executar Automaticamente
        │
        │ SIM (risco alto)
        ▼
  ┌──────────────┐    Aprovar    ┌───────────┐
  │   Humano     │─────────────▶│  Executar  │
  │  Operador    │              └───────────┘
  │              │    Rejeitar   ┌───────────┐
  │              │─────────────▶│  Cancelar  │
  └──────────────┘              └───────────┘
```

### 🖥️ Arquivo: `modulo_05_hitl/01_human_in_the_loop.py`

1. **Explicar** os níveis de risco: BAIXO → MÉDIO → ALTO → CRÍTICO
2. **Executar** a demo com 4 cenários diferentes
3. **Mostrar** o ponto de aprovação (como seria em produção)
4. (Opcional) **Executar** modo interativo para os participantes aprovarem/rejeitarem

#### Critérios típicos para HITL

- **Valor** > R$ 5.000 → aprovação obrigatória
- **Dados incompletos** → verificação humana
- **Confiança baixa** → revisão
- **Ações irreversíveis** → sempre aprovar

> **Ponto de discussão**: "Onde colocar o HITL no fluxo de vocês?"

---

## MÓDULO 6: Projeto Completo - Agente de Boletos (45 min)

### 📖 Teoria — Projeto Completo

> **Agora juntamos TUDO!**
> Este módulo mostra como todos os conceitos se conectam num agente real.

```text
                    AGENTE COMPLETO DE BOLETOS
                    
  Input ──▶ [Guardrails Entrada] ──▶ [LLM + Tools] ──▶ [Guardrails Saída]
                                          │
                                          ▼
                                   [Classificar Risco]
                                          │
                                    Alto? │ Baixo?
                                          │
                                    [HITL] │ [Auto]
                                          │
                                          ▼
                                    [Resposta Final]
```

### 🖥️ Arquivos do módulo

| Arquivo | Conteúdo |
| --------- | ---------- |
| `exemplos_boletos.json` | 5 boletos de exemplo com dados esperados |
| `tools_boleto.py` | Ferramentas: extração, cálculo, resumo |
| `guardrails_boleto.py` | Validações e classificação de risco |
| `agente_boletos.py` | Agente completo que integra tudo |

### Roteiro

1. **Abrir** `exemplos_boletos.json` - mostrar a estrutura dos dados de teste
2. **Abrir** `tools_boleto.py` - revisar as 3 tools especializadas
3. **Abrir** `guardrails_boleto.py` - revisar regras de negócio e classificação de risco
4. **Abrir** `agente_boletos.py` - ir pelo fluxo completo:
   - Mostrar o SYSTEM_PROMPT e discutir a estrutura
   - Explicar o método `processar_mensagem` com os 5 passos
   - **Executar** a demo automática
   - Observar: boletos processados, guardrails bloqueando, HITL aprovando
5. (Tempo extra) **Executar** modo interativo para os participantes testarem

> **Exercício**: Peça para adicionarem uma nova regra de negócio em `guardrails_boleto.py` (ex: bloquear CNPJ em lista de bloqueio)

---

## MÓDULO 7: Avaliação de Resultados (30 min)

### 📖 Teoria — Avaliação

> **"Se você não pode medir, não pode melhorar."**
> Avaliação é o que diferencia um protótipo de um produto.

#### Tipos de avaliação para agentes

| Tipo | Método | Automatizável? |
| ------ | -------- | ---------------- |
| Exatidão | Comparar com ground truth | ✅ Sim |
| Cálculos | Verificar matematicamente | ✅ Sim |
| Guardrails | Testes positivos/negativos | ✅ Sim |
| Qualidade | LLM-as-Judge | ⚡ Semi |
| Experiência | Feedback humano | ❌ Não |

### 🖥️ Arquivo: `modulo_07_avaliacao/01_avaliacao_resultados.py`

1. **Explicar** cada tipo de avaliação
2. **Executar** a avaliação completa
3. **Analisar** os resultados em grupo:
   - Quais campos tiveram menor acurácia?
   - Os guardrails pegaram todos os ataques?
   - Qual a nota do LLM-as-Judge?
4. **Discutir**: "O que vocês fariam para melhorar os números?"

#### LLM-as-Judge

```text
LLM avalia em 4 dimensões:
- Precisão (0-10)     → Informações corretas?
- Completude (0-10)   → Tudo foi coberto?
- Clareza (0-10)      → Fácil de entender?
- Utilidade (0-10)    → Realmente ajudou?
```

> **Exercício final**: Peça para os participantes adicionarem um novo caso de teste e executarem a avaliação novamente. Melhorou ou piorou?

---

## MÓDULO 8: Arquiteturas de Agentes (25 min)

### Objetivo — Arquiteturas

Mostrar que existem diferentes formas de compor agentes, e que loop com tools é apenas um padrão.

### Arquivo — Arquiteturas

- `modulo_08_arquiteturas/01_padroes_arquitetura.py`

### O que demonstrar — Arquiteturas

1. Fluxo linear para tarefas previsíveis
2. Roteador para múltiplas intenções
3. Planner-executor para tarefas compostas
4. Multiagente para especialização forte

> **Ponto de discussão**: "Qual padrão resolve o seu caso com menor complexidade possível?"

---

## MÓDULO 9: Observabilidade e Debugging (20 min)

### Objetivo — Observabilidade

Ensinar a diagnosticar falhas do agente por trace, e não por adivinhação.

### Arquivos — Observabilidade

- `modulo_09_observabilidade/trace_utils.py`
- `modulo_09_observabilidade/01_observabilidade_debug.py`

### O que demonstrar — Observabilidade

1. Geração de `trace_id`
2. Registro de estágios do fluxo
3. Persistência em JSONL
4. Sanitização de PII antes de salvar logs

---

## MÓDULO 10: Memória de Longo Prazo e RAG Simples (25 min)

### Objetivo — Memória

Ensinar a diferença entre histórico curto e memória recuperável.

### Arquivos — Memória

- `modulo_10_memoria_longo_prazo/politicas_cobranca.json`
- `modulo_10_memoria_longo_prazo/memory_utils.py`
- `modulo_10_memoria_longo_prazo/01_memoria_rag.py`

### O que demonstrar — Memória

1. Recuperação por palavras-chave
2. Injeção de contexto recuperado no agente
3. Evolução natural para embeddings e vetor store

---

## MÓDULO 11: Resiliência Operacional (20 min)

### Objetivo — Resiliência

Mostrar como o agente se comporta quando a LLM ou uma dependência falha.

### Arquivos — Resiliência

- `modulo_11_resiliencia/cliente_resiliente.py`
- `modulo_11_resiliencia/01_resiliencia_operacional.py`

### O que demonstrar — Resiliência

1. Retry com backoff
2. Fallback de modelo
3. Degradação graciosa
4. Integração do wrapper resiliente no agente de boletos

---

## MÓDULO 12: Manual vs Frameworks (15 min)

### Objetivo — Frameworks

Mapear o que foi construído manualmente para abstrações comuns de framework.

### Arquivo — Frameworks

- `modulo_12_frameworks/01_mapeamento_frameworks.md`

### O que demonstrar — Frameworks

1. Onde prompt, memory, tool e guardrail aparecem na mão
2. O que um framework abstrai
3. Quando vale usar framework e quando ele atrapalha

---

## MÓDULO 13: Segurança e Governança (20 min)

### Objetivo — Segurança

Cobrir o que normalmente separa demo de produção.

### Arquivo — Segurança

- `modulo_13_seguranca_governanca/01_checklist_governanca.md`

### O que demonstrar — Segurança

1. Gestão de segredos
2. Persistência segura de logs
3. Auditoria de decisões humanas
4. Critérios de bloqueio de deploy

---

## MÓDULO 14: Streaming e Experiência de Usuário (15 min)

### Objetivo — Streaming

Melhorar a UX do agente no terminal e tornar o fluxo visível ao usuário.

### Arquivos — Streaming

- `modulo_14_streaming_ux/streaming_utils.py`
- `modulo_14_streaming_ux/01_streaming_console.py`

### O que demonstrar — Streaming

1. Exibição por etapas
2. Resposta incremental
3. Relação entre UX e confiança do usuário

---

---

## MÓDULO 15: Integração com Sistemas Reais (20 min)

### Objetivo — Integração

Conectar o agente a fontes de dados corporativas reais: PDFs, APIs REST autenticadas com OAuth 2 e automação web.

### Arquivos — Integração

- `modulo_15_integracao_sistemas/01_integracao_sistemas_reais.py`

### O que demonstrar — Integração

1. Extração de texto de PDF com `pdfplumber` (com fallback simulado)
2. Gerenciamento de token OAuth 2 com cache e renovação automática
3. Cliente HTTP autenticado (`GET` e `POST`)
4. Stub de automação web com Playwright

---

## MÓDULO 16: Monitoramento em Produção (20 min)

### Objetivo — Monitoramento

Coletar e visualizar métricas de execução do agente (latência, taxa de sucesso, percentis p50/p95/p99) e definir SLAs com alertas.

### Arquivos — Monitoramento

- `modulo_16_monitoramento/01_monitoramento_producao.py`

### O que demonstrar — Monitoramento

1. Coleta estruturada de métricas por execução
2. Cálculo de percentis e verificação de SLAs
3. Dashboard Rich com tabelas e painéis coloridos
4. Referência à integração com Langfuse

---

## MÓDULO 17: CI/CD para Agentes (20 min)

### Objetivo — CI/CD

Versionar prompts, executar golden dataset automaticamente e fazer shadow testing antes de promover uma nova versão de prompt para produção.

### Arquivos — CI/CD

- `modulo_17_cicd_agentes/01_cicd_agentes.py`

### O que demonstrar — CI/CD

1. Registro de versões de prompt com persistência JSON
2. Golden dataset com 5 casos de teste
3. Suite de regressão automatizada
4. Shadow test comparando versão ativa vs. candidata
5. Decisão de promoção ou bloqueio baseada em taxa de sucesso

---

## MÓDULO 18: Tarefas Longas com Checkpoints (20 min)

### Objetivo — Checkpoints

Processar lotes grandes com salvamento de estado intermediário, permitindo retomada após falhas ou interrupções.

### Arquivos — Checkpoints

- `modulo_18_tarefas_longas/01_tarefas_longas_checkpoints.py`

### O que demonstrar — Checkpoints

1. Estrutura de estado por item (`pendente` → `em_andamento` → `concluido` / `falhou`)
2. Persistência JSON automática após cada transição
3. Retomada sem reprocessar itens já concluídos
4. Barra de progresso Rich mostrando apenas itens pendentes reais

---

## MÓDULO 19: Filas e Processamento em Lote (20 min)

### Objetivo — Filas

Implementar fila de prioridade thread-safe com rate limiting e workers paralelos para processamento em lote de alto volume.

### Arquivos — Filas

- `modulo_19_filas_lote/01_filas_processamento_lote.py`

### O que demonstrar — Filas

1. Fila de prioridade com `heapq` e `threading.Lock`
2. Rate limiter por token bucket
3. Workers paralelos com `ThreadPoolExecutor`
4. Dead Letter Queue (DLQ) para itens que esgotam tentativas

---

## MÓDULO 20: Orquestração Avançada Multi-Agente (20 min)

### Objetivo — Orquestração Avançada

Coordenar múltiplos agentes especialistas com um supervisor, passando contexto estruturado via handoff e mantendo trilha de auditoria completa.

### Arquivos — Orquestração Avançada

- `modulo_20_orquestracao_avancada/01_orquestracao_avancada.py`

### O que demonstrar — Orquestração Avancada

1. Padrão Supervisor + Workers com `WorkerBase` abstrato
2. `ContextoHandoff` que acumula resultados e histórico de decisões
3. Workers sequenciais: extração → cálculo → compliance
4. Tratamento de falhas individuais sem interromper o pipeline

---

## MÓDULO 21: Gestão de Custos e Otimização de Tokens (20 min)

### Objetivo — Custos

Monitorar o consumo de tokens, estimar custos por chamada, selecionar dinamicamente o modelo mais econômico e usar cache semântico para eliminar chamadas repetidas.

### Arquivos — Custos

- `modulo_21_custos_tokens/01_custos_tokens.py`

### O que demonstrar — Custos

1. Estimativa de custo por chamada com `estimar_custo_usd`
2. `RastreadorCusto` com alertas de orçamento e relatório
3. `CacheSemantico` via hash SHA-256 (cache HIT/MISS)
4. Compressão de histórico de conversação com `comprimir_historico`

---

## MÓDULO 22: Testes de Agentes com LLM Mockada (20 min)

### Objetivo — Testes Mock

Escrever testes unitários para agentes sem realizar chamadas reais à API — custo $0, execução em milissegundos.

### Arquivos — Testes Mock

- `modulo_22_testes_mock/01_testes_mock.py`

### O que demonstrar — Testes Mock

1. Injeção de dependência do cliente Groq via construtor
2. `MagicMock` com resposta JSON controlada
3. Teste de guardrail de entrada sem chamar a API
4. Suite de 7 testes rodando em < 50 ms, custo $0

---

## MÓDULO 23: Gerenciamento de Janela de Contexto (20 min)

### Objetivo — Contexto Longo

Processar documentos maiores que a janela de contexto usando chunking, sliding window e Map-Reduce, mantendo histórico sem exceder o limite de tokens.

### Arquivos — Contexto Longo

- `modulo_23_contexto_longo/01_contexto_longo.py`

### O que demonstrar — Contexto Longo

1. Heurística de estimativa de tokens (4 chars/token)
2. Chunking com overlap configurável
3. Sliding window para documentos maiores que o contexto
4. Map-Reduce para extração em lote e consolidação

---

## MÓDULO 24: Roteamento Dinâmico de Tarefas (20 min)

### Objetivo — Roteamento

Implementar o padrão Router: classificar automaticamente o tipo de documento e despachar para o agente especialista mais adequado com trilha de auditoria.

### Arquivos — Roteamento

- `modulo_24_roteamento/01_roteamento_dinamico.py`

### O que demonstrar — Roteamento

1. Plugin pattern para registrar agentes especialistas
2. Score de confiança baseado em padrões regex
3. Fallback automático para agente genérico
4. Trilha de auditoria das decisões de roteamento

---

## MÓDULO 25: LGPD e Privacidade de Dados (20 min)

### Objetivo — LGPD

Aplicar controles de privacidade LGPD no pipeline do agente: detectar dados pessoais, anonimizar antes de enviar ao LLM, re-identificar na resposta e manter log de auditoria sem dados reais.

### Arquivos — LGPD

- `modulo_25_lgpd_privacidade/01_lgpd_privacidade.py`

### O que demonstrar — LGPD

1. Detecção de PII: CPF, CNPJ, e-mail, telefone, cartão, CEP
2. Anonimização reversível antes de enviar ao LLM
3. Guardrail de saída que remove PII residual
4. Log de auditoria LGPD sem dados pessoais reais

---

## MÓDULO 26: RPA e Automação de Browser (20 min)

### Objetivo — RPA

Combinar agentes de IA com automação de browser (Playwright) para acessar sistemas legados sem API — o "último milha" da APA.

### Arquivos — RPA

- `modulo_26_rpa_browser/01_rpa_browser.py`

### O que demonstrar — RPA

1. `BrowserSimulado` com modo dry-run para validar fluxo sem abrir browser
2. `AgenteRPA` encapsulando login + extração de boletos
3. Registro de auditoria de cada ação executada
4. Equivalente Playwright real (código comentado)

---

## MÓDULO 27: Agentes com Banco de Dados — Text-to-SQL (20 min)

### Objetivo — Text-to-SQL

Converter perguntas em linguagem natural em queries SQL, validar contra comandos perigosos e executar com segurança em banco relacional.

### Arquivos — Text-to-SQL

- `modulo_27_text_to_sql/01_text_to_sql.py`

### O que demonstrar — Text-to-SQL

1. Banco SQLite em memória com dados de boletos e pagamentos
2. `carregar_schema()` injeta DDL no prompt do LLM
3. `validar_sql()` bloqueia DROP/DELETE/INSERT/UPDATE
4. Teste de bloqueio de SQL perigoso gerado por injeção

---

## MÓDULO 28: Processamento de Documentos Multi-modal (20 min)

### Objetivo — Documentos

Processar PDFs de boletos, NF-e e contratos, detectar o tipo de documento automaticamente e extrair campos estruturados com nível de confiança.

### Arquivos — Documentos

- `modulo_28_documentos_multimodal/01_documentos_multimodal.py`

### O que demonstrar — Documentos

1. Detecção de tipo de documento por padrões regex
2. Extratores estruturados: `extrair_boleto`, `extrair_nfe`, `extrair_contrato`
3. Score de confiança baseado em campos encontrados
4. Integração com pdfplumber real (código comentado)

---

## MÓDULO 29: State Machine para Fluxos de Aprovação (20 min)

### Objetivo — State Machine

Modelar workflows corporativos de aprovação como máquinas de estado com transições explícitas, regras de negócio (dupla aprovação) e log de auditoria imutável.

### Arquivos — State Machine

- `modulo_29_state_machine/01_state_machine.py`

### O que demonstrar — State Machine

1. Enum `Estado` e dict `TRANSICOES` como contrato do fluxo
2. `MaquinaAprovacao` com validação de transições
3. Regra de dupla aprovação para valores > R$ 10.000
4. Log de auditoria append-only com timestamp

---

## MÓDULO 30: Agente em Ambiente Corporativo (20 min)

### Objetivo — Ambiente Corporativo

Configurar agentes para redes com proxy HTTP, certificados SSL internos e autenticação OAuth2 service-to-service — obstáculos reais em produção.

### Arquivos — Ambiente Corporativo

- `modulo_30_ambiente_corporativo/01_ambiente_corporativo.py`

### O que demonstrar — Ambiente Corporativo

1. `ConfigCorporativa` carregada de variáveis de ambiente
2. `ClienteHTTPCorporativo` com proxy + retry + backoff
3. `GerenciadorOAuth2` com cache e renovação automática de token
4. Checklist de onboarding para ambientes corporativos

---

## MÓDULO 31: Integração com Sistemas de Notificação (20 min)

### Objetivo — Notificações

Fechar o loop de automação notificando as pessoas certas via e-mail, Teams e Slack quando o agente completa tarefas ou detecta problemas.

### Arquivos — Notificações

- `modulo_31_notificacoes/01_notificacoes.py`

### O que demonstrar — Notificações

1. Padrão Observer: `EventoBus` desacopla publicador dos canais
2. `NotificadorTeams` com Adaptive Card JSON
3. `NotificadorSlack` com Block Kit
4. Log de entrega de notificações para rastreabilidade

---

## 🎯 Checklist Final

Ao final do treinamento, os participantes devem ser capazes de:

- [ ] Configurar uma LLM gratuita (Groq) em Python
- [ ] Escrever prompts eficazes (zero/few-shot, CoT, structured output)
- [ ] Construir o loop básico de um agente
- [ ] Implementar memória de conversação
- [ ] Criar e registrar ferramentas (tools)
- [ ] Implementar tool calling com a LLM
- [ ] Adicionar guardrails de entrada (injection, PII, escopo)
- [ ] Adicionar guardrails de saída (Pydantic, filtros)
- [ ] Implementar camada Human-in-the-Loop
- [ ] Avaliar a qualidade do agente com métricas
- [ ] Comparar arquiteturas de agentes e escolher a mais simples adequada
- [ ] Instrumentar tracing e auditar execuções com trace_id
- [ ] Aplicar memória de longo prazo além do histórico curto
- [ ] Adicionar retry, fallback e degradação graciosa
- [ ] Executar testes automatizados com `pytest`
- [ ] Aplicar checklist de segurança e governança antes de produção
- [ ] Integrar o agente com PDFs e APIs REST autenticadas (OAuth 2)
- [ ] Coletar métricas e verificar SLAs em produção
- [ ] Versionar prompts e executar suíte de regressão automática
- [ ] Implementar processamento com checkpoints e retomada
- [ ] Processar lotes em paralelo com fila de prioridade e DLQ
- [ ] Estimar e controlar custos de tokens por execução
- [ ] Escrever testes de agente com LLM mockada (custo $0)
- [ ] Aplicar chunking/sliding window para documentos longos
- [ ] Implementar roteamento dinâmico para múltiplos tipos de documento
- [ ] Aplicar controles de privacidade LGPD no pipeline do agente
- [ ] Automatizar acesso a sistemas legados sem API com browser automation
- [ ] Construir agente Text-to-SQL com validação de segurança
- [ ] Processar PDFs e detectar tipo de documento automaticamente
- [ ] Modelar fluxo de aprovação como state machine auditável
- [ ] Configurar agente em ambiente corporativo (proxy, SSL, OAuth2)
- [ ] Integrar notificações de e-mail, Teams e Slack no pipeline
- [ ] Orquestrar múltiplos agentes com handoff e trilha de auditoria

---

## 💡 Dicas para o Instrutor

### Fluxo recomendado

1. **Apresente a teoria** (2-3 min por conceito, com slides ou whiteboard)
2. **Mostre o código** já pronto e explique
3. **Execute junto** com os participantes
4. **Modifique ao vivo** para explorar cenários
5. **Deixe tempo** para perguntas e exercícios

### Pausas sugeridas

- Após Módulo 2 (agente básico montado) - 10 min
- Após Módulo 4 (guardrails) - 10 min
- Após Módulo 6 (projeto completo) - 10 min

### Pontos de destaque para cada módulo

| Módulo | Frase-chave para destacar |
| -------- | -------------------------- |
| 1 | "O prompt é 80% do resultado de um agente" |
| 2 | "Um agente é um loop, não uma chamada" |
| 3 | "Tools transformam texto em ação" |
| 4 | "Sem guardrails, seu agente é um risco" |
| 5 | "Nem tudo deve ser automático" |
| 6 | "Integração é onde a complexidade aparece" |
| 7 | "Se não mede, não sabe se funciona" |

### Se surgir a pergunta: "Por que não usar LangChain/CrewAI/etc?"

> "Frameworks são ótimos para produtividade, mas escondem a mecânica. Hoje estamos aprendendo como os agentes funcionam por dentro. Com esse conhecimento, vocês podem usar qualquer framework sabendo o que está acontecendo nos bastidores."

---

## 🔗 Recursos Adicionais

- **Groq Console**: <https://console.groq.com/>
- **Documentação Groq**: <https://console.groq.com/docs>
- **Pydantic**: <https://docs.pydantic.dev/>
- **Prompt Engineering Guide**: <https://www.promptingguide.ai/pt>
- **ReAct Paper**: <https://arxiv.org/abs/2210.03629>

---

## 📁 Comandos Rápidos para Executar Cada Módulo

```bash
# A partir da raiz do projeto:

# Módulo 1
python modulo_01_prompts/01_estilos_prompt.py
python modulo_01_prompts/02_boas_praticas.py

# Módulo 2
python modulo_02_agente_basico/01_primeiro_agente.py
python modulo_02_agente_basico/02_agente_com_memoria.py

# Módulo 3
python modulo_03_tools/01_criando_tools.py
python -m modulo_03_tools.02_tool_calling

# Módulo 4
python modulo_04_guardrails/01_validacao_entrada.py
python modulo_04_guardrails/02_validacao_saida.py

# Módulo 5
python modulo_05_hitl/01_human_in_the_loop.py

# Módulo 6 (Projeto Completo)
python -m modulo_06_agente_boletos.agente_boletos

# Módulo 7 (Avaliação)
python -m modulo_07_avaliacao.01_avaliacao_resultados
python -m modulo_07_avaliacao.02_design_avaliacao

# Módulo 8
python modulo_08_arquiteturas/01_padroes_arquitetura.py

# Módulo 9
python modulo_09_observabilidade/01_observabilidade_debug.py

# Módulo 10
python modulo_10_memoria_longo_prazo/01_memoria_rag.py

# Módulo 11
python modulo_11_resiliencia/01_resiliencia_operacional.py

# Módulo 14
python modulo_14_streaming_ux/01_streaming_console.py

# Módulo 15
python modulo_15_integracao_sistemas/01_integracao_sistemas_reais.py

# Módulo 16
python modulo_16_monitoramento/01_monitoramento_producao.py

# Módulo 17
python modulo_17_cicd_agentes/01_cicd_agentes.py

# Módulo 18
python modulo_18_tarefas_longas/01_tarefas_longas_checkpoints.py

# Módulo 19
python modulo_19_filas_lote/01_filas_processamento_lote.py

# Módulo 20
python modulo_20_orquestracao_avancada/01_orquestracao_avancada.py

# Módulo 21
python modulo_21_custos_tokens/01_custos_tokens.py

# Módulo 22
python modulo_22_testes_mock/01_testes_mock.py

# Módulo 23
python modulo_23_contexto_longo/01_contexto_longo.py

# Módulo 24
python modulo_24_roteamento/01_roteamento_dinamico.py

# Módulo 25
python modulo_25_lgpd_privacidade/01_lgpd_privacidade.py

# Módulo 26
python modulo_26_rpa_browser/01_rpa_browser.py

# Módulo 27
python modulo_27_text_to_sql/01_text_to_sql.py

# Módulo 28
python modulo_28_documentos_multimodal/01_documentos_multimodal.py

# Módulo 29
python modulo_29_state_machine/01_state_machine.py

# Módulo 30
python modulo_30_ambiente_corporativo/01_ambiente_corporativo.py

# Módulo 31
python modulo_31_notificacoes/01_notificacoes.py

# Testes automatizados
pytest
```
