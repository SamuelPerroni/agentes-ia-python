# MÓDULO 13 - CHECKLIST DE SEGURANÇA E GOVERNANÇA

## O que este módulo ensina?

Antes de colocar um agente de IA em produção, existe uma série de
verificações de **segurança**, **privacidade** e **governança** que
precisam ser feitas. Este checklist resume tudo que aprendemos nos
módulos anteriores em uma lista acionável.

## Por que Governança em Agentes de IA?

> **ANALOGIA:**
> Um carro sem freio pode ser rápido, mas é perigoso.
> Governança são os "freios" e "cintos de segurança" do agente —
> eles não atrapalham o funcionamento normal, mas protegem quando
> algo dá errado.

**Riscos de um agente SEM governança:**

- Vaza dados pessoais de clientes em logs (violação de LGPD)
- Executa ações financeiras sem aprovação humana
- Não tem rastreabilidade — ninguém sabe por que o agente fez X
- Uma versão nova com bug vai para produção sem testes
- Quando a LLM cai, o sistema inteiro para

---

## 1. Segredos e Configuração

### O que verificar

- [ ] **Nunca commitar `.env`** com credenciais reais
  - Use `.env.example` com valores placeholder no repositório
  - Adicione `.env` ao `.gitignore`
- [ ] **Separar chave de API por ambiente**
  - Desenvolvimento: chave com rate limit baixo
  - Produção: chave com rate limit adequado ao tráfego
  - ⚠️ Nunca use a mesma chave em dev e prod
- [ ] **Definir rotação periódica de credenciais**
  - Chaves comprometidas devem ser revogadas imediatamente
  - Defina um calendário (ex: trocar a cada 90 dias)
- [ ] **Usar variáveis de ambiente** (não hardcode)
  - `os.getenv("GROQ_API_KEY")` ✅
  - `api_key = "gsk_abc123..."` ❌

### Onde implementamos isso no treinamento

- `.env` e `.env.example` na raiz do projeto
- `python-dotenv` para carregar variáveis de ambiente
- Módulo 06: agente_boletos usa `os.getenv()` para todas as configurações

---

## 2. Dados Sensíveis (LGPD / Privacidade)

### O que verificar - Dados Sensíveis

- [ ] **Mascarar CPF, email e telefone** antes de salvar em logs/traces
  - Implementado no módulo 09: `redigir_texto()` em `trace_utils.py`
  - Regex detecta CPF, CNPJ, email e substitui por `[TIPO_REDACTED]`
- [ ] **Persistir apenas o mínimo necessário**
  - Pergunte: "preciso MESMO salvar este dado?"
  - Se a resposta é "talvez", não salve
- [ ] **Definir base legal e prazo de retenção** para dados pessoais
  - LGPD Art. 7: qual base legal justifica o tratamento?
  - Defina prazo: "traces de 30 dias, depois apagar automaticamente"
- [ ] **Não enviar PII para a LLM** quando possível
  - Mascare ANTES de incluir no prompt
  - Exemplo: "Cliente [CPF_REDACTED] perguntou sobre boleto..."

### Onde implementamos isso no treinamento - Dados Sensíveis

- Módulo 04: guardrails de entrada que detectam PII
- Módulo 09: `redigir_texto()` mascara antes de persistir
- Módulo 06: agente integra sanitização no fluxo principal

### Diagrama de fluxo de dados sensíveis

```Text
Entrada do usuário (pode conter CPF, email, etc.)
       │
       ↓
[Guardrail de entrada] ← módulo 04
  └── Detecta e mascara PII
       │
       ↓
[Prompt para LLM] ← dados JÁ mascarados
       │
       ↓
[Resposta da LLM]
       │
       ↓
[TraceRecorder] ← módulo 09
  └── Sanitiza payload antes de salvar
       │
       ↓
[Arquivo JSONL] ← SEM dados pessoais
```

---

## 3. Auditoria e Rastreabilidade

### O que verificar - Auditoria

- [ ] **Registrar `trace_id`** por execução
  - Cada interação do agente deve ter um identificador único
  - Implementado no módulo 09: `TraceRecorder.trace_id`
- [ ] **Salvar decisão humana** em operações de alto risco
  - HITL (módulo 05): registrar quem aprovou, quando, e o motivo
  - Exemplo: `{"decisao": "aprovado", "operador": "João", "trace_id": "abc123"}`
- [ ] **Identificar versão do prompt e do código** que gerou a resposta
  - Em produção, versione seus system prompts (v1.0, v1.1, etc.)
  - Registre no trace qual versão do prompt foi usada
- [ ] **Manter histórico de traces** para investigação post-mortem
  - Quando um cliente reclama, o trace permite reconstruir o que aconteceu
  - Defina retenção mínima: 30-90 dias em produção

### Onde implementamos isso no treinamento - Auditoria

- Módulo 05: HITL com decisões humanas registradas
- Módulo 09: TraceRecorder com trace_id, timestamp, stage, payload
- Módulo 06: agente persiste trace JSONL por execução

---

## 4. Política de Deploy (Versionamento e Regressão)

### O que verificar - Deploy

- [ ] **Não promover versão nova sem benchmark**
  - Módulo 07: definimos baselines mínimos (acurácia ≥ 85%, guardrails ≥ 95%)
  - Executar `pytest` antes de cada deploy (módulo 07 + diretório tests/)
- [ ] **Exigir regressão estável** em segurança e cálculos
  - Se a nova versão piorou em QUALQUER métrica, BLOQUEAR o deploy
  - Métricas críticas: precisão de cálculos, eficácia de guardrails
- [ ] **Definir rollback operacional**
  - Em caso de problema, como voltar à versão anterior?
  - Documente o procedimento (não improvise no momento de crise)
- [ ] **Ambiente de staging** antes de produção
  - Teste com dados reais (anonimizados) antes de expor ao usuário final

### Onde implementamos isso no treinamento - Deploy

- Módulo 07: benchmark_template.json com thresholds mínimos
- Módulo 07: deploy_blockers define métricas que impedem deploy
- Diretório tests/: testes automatizados para regressão

### Fluxo de deploy seguro

```Text
Código novo → Testes unitários (pytest)
                    │
                    ↓
              Benchmark (módulo 07)
                    │
                    ↓
              Métricas OK? ──→ ❌ BLOQUEAR deploy
                    │
                    ✅
                    ↓
              Deploy em staging
                    │
                    ↓
              Teste manual / smoke test
                    │
                    ↓
              Deploy em produção
                    │
                    ↓
              Monitorar traces por 24h
```

---

## 5. Resiliência e Degradação

### O que verificar - Resiliência

- [ ] **Retry + backoff** configurados para chamadas à LLM
  - Implementado no módulo 11: `ClienteLLMResiliente`
  - max_retries: quantas vezes tentar antes de desistir
- [ ] **Fallback para modelo alternativo** quando o primário falha
  - Módulo 11: fallback automático (70b → 8b)
  - Registrar no trace quando fallback é acionado
- [ ] **Resposta de degradação** quando TUDO falha
  - Em vez de erro 500, mostrar: "Estamos com intermitência, tente novamente"
  - Definir template de resposta emergencial
- [ ] **Monitorar taxa de fallback** como métrica operacional
  - < 1%: normal | 1-5%: atenção | > 5%: investigar

### Onde implementamos isso no treinamento - Resiliência

- Módulo 11: ClienteLLMResiliente com retry, backoff e fallback
- Módulo 06: agente integra cliente resiliente no fluxo

---

## 6. Perguntas Obrigatórias Antes de Produção

Antes de colocar QUALQUER agente de IA em produção, responda estas
5 perguntas. Se não conseguir responder alguma, o agente NÃO ESTÁ
PRONTO para produção.

### 1️⃣ Que dados pessoais entram no fluxo?

> Liste todos: CPF, email, nome, endereço, dados financeiros.
> Para cada um, defina: é necessário? É mascarado? Onde é armazenado?

### 2️⃣ O que fica salvo em log?

> Detalhe: quais campos são persistidos? Por quanto tempo?
> Há PII nos logs? Se sim, está mascarado?

### 3️⃣ Como uma decisão humana é auditada?

> Quando o HITL é acionado, quem decide? A decisão é registrada?
> É possível reconstituir a cadeia de eventos depois?

### 4️⃣ Qual métrica bloqueia deploy?

> Ex: "Se acurácia de cálculos < 85%, deploy é bloqueado."
> Defina no benchmark_template.json (módulo 07).

### 5️⃣ Como o sistema degrada quando a LLM falha?

> O sistema para completamente? Tem fallback? Tem mensagem amigável?
> Quantas retentativas? Qual o timeout máximo?

---

## Resumo visual — Camadas de Proteção

```Text
┌───────────────────────────────────────────────────────────┐
│                    AGENTE EM PRODUÇÃO                     │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  Camada 1: ENTRADA                                        │
│  └── Guardrails (módulo 04) + Mascaramento PII            │
│                                                           │
│  Camada 2: PROCESSAMENTO                                  │
│  └── Resiliência (módulo 11) + Observabilidade (módulo 09)│
│                                                           │
│  Camada 3: DECISÃO                                        │
│  └── HITL (módulo 05) para operações de alto risco        │
│                                                           │
│  Camada 4: SAÍDA                                          │
│  └── Validação de resposta + Streaming (módulo 14)        │
│                                                           │
│  Camada 5: AVALIAÇÃO                                      │
│  └── Benchmark (módulo 07) + Policy de deploy             │
│                                                           │
│  Camada 6: AUDITORIA                                      │
│  └── Traces JSONL + Versionamento de prompts              │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## Exercício sugerido

1. Copie este checklist para o seu projeto real
2. Marque cada item como ✅ (feito) ou ❌ (pendente)
3. Para cada ❌, defina um plano de ação com prazo
4. Responda as 5 perguntas obrigatórias por escrito
5. Revise o checklist a cada sprint/release
