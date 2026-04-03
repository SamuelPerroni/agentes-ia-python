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

## Guia Completo

Consulte o [GUIA_TREINAMENTO.md](GUIA_TREINAMENTO.md) para o passo a passo detalhado.

## LLM Utilizada

Usamos a API gratuita do **Groq** com o modelo **Llama 3.3 70B**.

- Limite gratuito generoso
- Velocidade excepcional
- Suporte a tool calling
- API compatível com OpenAI

## Expansão para Produção

Além da trilha base, o treinamento agora inclui uma camada de maturidade com:

- comparação de arquiteturas de agentes
- tracing e logs auditáveis
- memória de longo prazo e recuperação simples
- retry, backoff e fallback de modelo
- governança e segurança operacional
- streaming no terminal
- testes automatizados com `pytest`
