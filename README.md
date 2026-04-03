# 🤖 Treinamento Hands-On: Agentes de IA com Python

Treinamento prático de construção de Agentes de Inteligência Artificial usando Python e LLMs gratuitas (Groq + Llama).

## Pré-requisitos

- Python 3.10+
- Conta gratuita no [Groq](https://console.groq.com/) (API key)
- VS Code (recomendado)

## Setup Rápido

```bash
# 1. Clonar/abrir o projeto
cd treinamento_agentes_ia

# 2. Criar ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar variáveis de ambiente
copy .env.example .env
# Edite o arquivo .env e coloque sua GROQ_API_KEY
```

## Estrutura do Treinamento

| Módulo | Tema | Arquivo |
| -------- | ------ | --------- |
| 01 | Engenharia de Prompts | `modulo_01_prompts/` |
| 02 | Agente Básico | `modulo_02_agente_basico/` |
| 03 | Tools (Ferramentas) | `modulo_03_tools/` |
| 04 | Guardrails | `modulo_04_guardrails/` |
| 05 | Human-in-the-Loop | `modulo_05_hitl/` |
| 06 | Projeto: Agente de Boletos | `modulo_06_agente_boletos/` |
| 07 | Avaliação de Resultados | `modulo_07_avaliacao/` |
| 08 | Arquiteturas de Agentes | `modulo_08_arquiteturas/` |
| 09 | Observabilidade e Debugging | `modulo_09_observabilidade/` |
| 10 | Memória de Longo Prazo e RAG Simples | `modulo_10_memoria_longo_prazo/` |
| 11 | Resiliência Operacional | `modulo_11_resiliencia/` |
| 12 | Manual vs Frameworks | `modulo_12_frameworks/` |
| 13 | Segurança e Governança | `modulo_13_seguranca_governanca/` |
| 14 | Streaming e UX | `modulo_14_streaming_ux/` |
| 15 | PDF e OCR | `modulo_15_pdf_ocr/` |
| 16 | Integração com APIs REST | `modulo_16_api_rest/` |
| 17 | Métricas e SLA em Produção | `modulo_17_metricas_sla/` |
| 18 | Versionamento de Prompts | `modulo_18_versionamento_prompts/` |
| 19 | Checkpoint e Retomada | `modulo_19_checkpoint/` |
| 20 | Processamento em Lote | `modulo_20_batch/` |
| 21 | Custos e Tokens | `modulo_21_custos_tokens/` |
| 22 | Testes com LLM Mockada | `modulo_22_testes_mock/` |
| 23 | Context Window | `modulo_23_context_window/` |
| 24 | Roteamento Dinâmico | `modulo_24_roteamento/` |
| 25 | Privacidade e LGPD | `modulo_25_lgpd/` |
| 26 | RPA e Browser Automation | `modulo_26_rpa_browser/` |
| 27 | Text-to-SQL | `modulo_27_text_to_sql/` |
| 28 | Documentos Multimodais | `modulo_28_multimodal/` |
| 29 | State Machine | `modulo_29_state_machine/` |
| 30 | Ambiente Corporativo | `modulo_30_ambiente_corporativo/` |
| 31 | Notificações (e-mail, Teams, Slack) | `modulo_31_notificacoes/` |
| 32 | Agente como Microsserviço REST | `modulo_32_microsservico_fastapi/` |
| 33 | Agendamento e Gatilhos de Processo | `modulo_33_agendamento_gatilhos/` |
| 34 | Reconciliação Financeira Automática | `modulo_34_reconciliacao/` |
| 35 | Onboarding de Fornecedores | `modulo_35_onboarding_fornecedor/` |
| 36 | KPIs e ROI do Processo APA | `modulo_36_kpis_roi/` |
| 37 | Escalação com SLA | `modulo_37_escalacao_sla/` |
| 38 | Protocolo A2A (Agent-to-Agent) | `modulo_38_a2a/` |
| 39 | Protocolo MCP (Model Context Protocol) | `modulo_39_mcp/` |
| 40 | Async com asyncio | `modulo_40_async/` |
| 41 | Fine-Tuning para Domínio Específico | `modulo_41_fine_tuning/` |
| 42 | Prompt Caching | `modulo_42_prompt_caching/` |
| 43 | Agente Auto-Corretivo | `modulo_43_agente_autocorretivo/` |

## Guia Completo

Consulte o [GUIA_TREINAMENTO.md](GUIA_TREINAMENTO.md) para o passo a passo detalhado.

## LLM Utilizada

Usamos a API gratuita do **Groq** com o modelo **Llama 3.3 70B**.

- Limite gratuito generoso
- Velocidade excepcional
- Suporte a tool calling
- API compatível com OpenAI

## Expansão para Produção

Além da trilha base (módulos 01–14), o treinamento inclui camadas progressivas de maturidade:

### Camada 2 — Integração e Qualidade (módulos 15–20)

- PDF/OCR, integração REST com OAuth 2, métricas e SLA em produção
- Versionamento de prompts, checkpoints, processamento em lote

### Camada 3 — Operação Avançada (módulos 21–25)

- Controle de custos de tokens, testes com LLM mockada (custo $0)
- Context window, roteamento dinâmico, privacidade e LGPD

### Camada 4 — APA Corporativa (módulos 26–31)

- Browser automation para sistemas legados, Text-to-SQL
- Documentos multimodais, state machine auditável
- Configuração em ambiente corporativo (proxy, SSL, OAuth2)
- Notificações via e-mail, Teams e Slack

### Camada 5 — APA em Produção (módulos 32–37)

- Agente exposto como microsserviço REST (FastAPI)
- Gatilhos automáticos: cron, fila de eventos, webhook (HMAC)
- Reconciliação financeira banco × ERP com classificação de divergências
- Onboarding de fornecedores (CNPJ, Receita Federal, sanções, certidões)
- KPIs e cálculo de ROI para justificativa executiva
- Escalação com SLA: N1 → N2 → N3 com notificação automática

### Camada 6 — Protocolos Avançados e IA Moderna (módulos 38–43)

- Protocolo A2A: descoberta e delegação entre agentes especializados
- Protocolo MCP: integração padronizada de ferramentas e fontes de dados
- Processamento paralelo com asyncio, Semaphore e padrão produtor/consumidor
- Fine-tuning: preparação de dataset, avaliação de qualidade e estimativa de custo
- Prompt caching: redução de custo em até 90% e latência em até 85%
- Agente auto-corretivo: detecção de erros próprios e retry com prompt dinâmico
