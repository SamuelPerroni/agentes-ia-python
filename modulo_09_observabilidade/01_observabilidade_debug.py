"""
============================================================
MÓDULO 9.1 - OBSERVABILIDADE E DEBUGGING DE AGENTES
============================================================
Neste módulo, aprendemos a instrumentar o agente para que CADA
ETAPA da execução seja registrada, permitindo auditoria e debug.

CONCEITO CHAVE:
Observabilidade é a capacidade de entender o que um sistema está
fazendo INTERNAMENTE, apenas olhando suas saídas (logs, métricas,
traces). É diferente de monitoramento: monitoramento diz "está
funcionando?", observabilidade diz "POR QUE está (ou não) funcionando."

O QUE VAMOS DEMONSTRAR AQUI:
1. Criar um TraceRecorder com trace_id único
2. Registrar eventos em cada etapa do agente (entrada, guardrail,
   tool_call, resposta)
3. Ver como dados sensíveis (CPF) são mascarados automaticamente
4. Persistir o trace em arquivo JSONL para análise posterior

POR QUE OBSERVABILIDADE É CRÍTICA EM AGENTES?
- Agentes de IA não são determinísticos: a mesma entrada pode gerar
  respostas diferentes. Sem trace, você não sabe O QUE aconteceu.
- Erros silenciosos: a LLM pode retornar uma resposta "plausível"
  mas errada. O trace mostra se as tools foram chamadas corretamente.
- Compliance: reguladores podem exigir rastreabilidade de decisões.
- Performance: traces permitem medir latência de cada etapa e
  identificar gargalos (ex: a LLM está demorando 5s para responder?).

ANALOGIA:
Imagine um médico que atende um paciente mas NÃO ANOTA NADA no
prontuário. O próximo médico não sabe o que foi feito. Com observabilidade,
cada "consulta" (execução) do agente tem um prontuário completo.

FLUXO DEMONSTRADO:
  ┌────────────────────────────────────────────────────┐
  │               Demo de Observabilidade               │
  ├────────────────────────────────────────────────────┤
  │ 1. Cria TraceRecorder (trace_id gerado)            │
  │ 2. log_event("entrada", {mensagem com CPF})        │
  │    → CPF é mascarado automaticamente               │
  │ 3. log_event("guardrail", {resultado})             │
  │ 4. log_event("tool_call", {tool + args})           │
  │ 5. log_event("resposta", {texto da resposta})      │
  │ 6. persist() → salva trace_xxxx.jsonl              │
  │ 7. Exibe painel com trace_id, qtd eventos, caminho │
  └────────────────────────────────────────────────────┘

Tópicos cobertos:
1. Criação e uso do TraceRecorder
2. Registro de eventos por estágio
3. Mascaramento automático de PII (CPF)
4. Persistência em JSONL
5. Resumo visual com Rich
============================================================
"""

import os

from rich.console import Console
from rich.panel import Panel

# Importamos o TraceRecorder que criamos no trace_utils.py
# Ele é o componente central de observabilidade do nosso agente
from modulo_09_observabilidade.trace_utils import TraceRecorder

# Console do Rich para output formatado com cores e painéis
console = Console()


# ============================================================
# DEMO DE OBSERVABILIDADE — Ciclo Completo de um Trace
# ============================================================
# Esta função simula o que acontece DENTRO do agente de boletos
# quando ele processa uma mensagem. Em vez de chamar a LLM de
# verdade, registramos os eventos que SERIAM gerados.
#
# OBSERVE:
# - O trace_id é gerado automaticamente (UUID hex de 12 chars)
# - O CPF na mensagem de entrada é mascarado para [CPF_REDACTED]
# - Cada evento tem timestamp, stage e payload sanitizado
# - Ao final, o arquivo JSONL é salvo no diretório "traces/"
#
# QUANDO USAR TRACES EM PRODUÇÃO:
# - Em TODA execução do agente (não apenas em erros)
# - Traces permitem análise post-mortem, métricas e auditoria
# - O custo de armazenamento é baixo (JSONL é leve e comprimível)
# ============================================================

def demo_observabilidade() -> None:
    """
    Demonstra o ciclo completo de observabilidade do agente.

    ETAPAS:
    1. Cria um TraceRecorder apontando para o diretório "traces/"
    2. Simula 4 eventos do agente (entrada, guardrail, tool_call, resposta)
    3. Persiste o trace em um arquivo JSONL
    4. Exibe um painel resumo no terminal

    OBSERVE NO OUTPUT:
    - O trace_id é único por execução (identifica a "sessão")
    - O CPF "123.456.789-00" é automaticamente reduzido a [CPF_REDACTED]
    - O arquivo JSONL é criado no mesmo diretório deste script

    EXERCÍCIO SUGERIDO:
    1. Rode este script e abra o arquivo .jsonl gerado num editor
    2. Verifique que o CPF foi mascarado no payload do evento "entrada"
    3. Adicione um evento de "erro" simulado e veja como fica no trace
    """
    # Passo 1: Criar o TraceRecorder
    # O log_dir aponta para uma pasta "traces/" dentro deste módulo
    tracer = TraceRecorder(
        log_dir=os.path.join(
            os.path.dirname(__file__),
            "traces"
            )
        )

    # Passo 2: Registrar eventos — note que o CPF será mascarado
    # porque o TraceRecorder chama _sanitizar() internamente
    tracer.log_event("entrada", {
        "mensagem": "Meu CPF é 123.456.789-00 e meu boleto venceu ontem."
    })

    # Guardrail detectou e mascarou PII — registramos isso no trace
    tracer.log_event("guardrail", {
        "resultado": "PII detectado e mascarado"
    })

    # Tool chamada com seus argumentos — para debug posterior
    tracer.log_event("tool_call", {
        "tool": "calcular_valor_atualizado",
        "args": "valor=500,dias=1"
    })

    # Resposta final gerada pelo agente
    tracer.log_event("resposta", {
        "texto": "Valor atualizado: R$ 510,17"
    })

    # Passo 3: Persistir o trace em arquivo JSONL
    caminho = tracer.persist()

    # Passo 4: Exibir um painel resumo com Rich
    console.print(Panel.fit(
        f"[bold]Trace ID:[/bold] {tracer.trace_id}\n"
        f"[bold]Eventos registrados:[/bold] {len(tracer.events)}\n"
        f"[bold]Estágios percorridos:[/bold] "
        f"{' → '.join(e['stage'] for e in tracer.events)}\n"
        f"[bold]Arquivo salvo em:[/bold] {caminho}",
        title="🔍 Observabilidade — Trace Completo",
        border_style="blue",
    ))

    # Dica final para o aluno
    console.print("\n💡 Dica:", style="bold yellow")
    console.print("  Abra o arquivo .jsonl gerado e observe:")
    console.print("  - Cada linha é um evento JSON independente")
    console.print("  - O CPF foi substituído por [CPF_REDACTED]")
    console.print("  - Todos os eventos compartilham o mesmo trace_id")
    console.print("  - O timestamp mostra a ordem exata dos eventos")


# ============================================================
# PONTO DE ENTRADA — Execução direta do módulo
# ============================================================
# Ao rodar `python 01_observabilidade_debug.py`, o aluno verá
# o ciclo completo de tracing aplicado ao domínio de boletos.
#
# EXERCÍCIO EXTRA:
# 1. Adicione mais eventos (ex: "validacao_risco", "hitl_decisao")
# 2. Abra o JSONL gerado e analise com: python -m json.tool < arquivo.jsonl
# 3. Crie um script que lê o JSONL e conta quantos eventos de cada stage
# ============================================================

if __name__ == "__main__":
    demo_observabilidade()
