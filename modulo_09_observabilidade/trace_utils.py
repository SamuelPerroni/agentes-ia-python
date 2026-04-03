"""
============================================================
MÓDULO 9 - UTILITÁRIOS DE TRACING (trace_utils.py)
============================================================
Este arquivo contém as ferramentas REUTILIZÁVEIS de observabilidade.
É importado pelo agente de boletos (módulo 06)
e pelo demo (01_observabilidade_debug.py).

O QUE É TRACING?
Tracing é o registro estruturado de TUDO que acontece durante uma
execução do agente: entrada recebida, guardrails aplicados, tools
chamadas, resposta gerada, erros encontrados.

POR QUE TRACING É ESSENCIAL?
- Agentes de IA não são determinísticos — a mesma entrada pode gerar
  respostas diferentes. Sem trace, você não sabe O QUE aconteceu.
- Debugging sem trace = "chutar no escuro". Com trace, você vê cada
  etapa e identifica onde o agente errou.
- Compliance/auditoria: em produção, reguladores podem exigir
  rastreabilidade de TODAS as decisões automatizadas.
- Performance: traces permitem medir latência de cada etapa.

ANALOGIA:
Pense no trace como a "caixa preta" de um avião. Ninguém lê ela
durante o voo normal, mas quando algo dá errado, ela é ESSENCIAL
para entender o que aconteceu.

COMPONENTES DESTE MÓDULO:
1. PADROES_SENSIVEIS - regex para detectar PII (CPF, CNPJ, email)
2. redigir_texto()   - mascara PII antes de persistir logs (LGPD)
3. TraceRecorder     - classe que coleta e persiste eventos em JSONL

FORMATO DE PERSISTÊNCIA — JSONL (JSON Lines):
Cada linha do arquivo é um JSON independente. Isso permite:
- Append sem reler o arquivo todo
- Leitura parcial / streaming
- Compatibilidade com ferramentas de análise (jq, pandas, etc.)

Exemplo de linha JSONL:
{
    "trace_id": "abc123",
    "timestamp": "2026-03-16T10:30:00",
    "stage": "entrada",
    "payload": {...}
}
============================================================
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


# ============================================================
# 1. DETECÇÃO E MASCARAMENTO DE PII (Dados Sensíveis)
# ============================================================
# Antes de salvar qualquer log, precisamos MASCARAR dados pessoais.
# Isso é obrigatório pela LGPD e pela boa prática de segurança.
#
# POR QUE MASCARAR ANTES DE SALVAR?
# - Logs podem ser acessados por equipes de suporte, SRE, devs
# - Se um log contém CPF ou email, qualquer pessoa com acesso ao
#   log tem acesso ao dado pessoal — isso é uma violação de privacidade
# - A regra é: mascare NO PONTO DE COLETA, não depois
#
# PADRÕES DETECTADOS:
# - CPF: 123.456.789-00 (com ou sem pontuação)
# - CNPJ: 12.345.678/0001-00 (com ou sem pontuação)
# - Email: usuario@dominio.com
#
# COMO FUNCIONA:
# O regex encontra o padrão e substitui por [TIPO_REDACTED]
# Ex: "CPF 123.456.789-00" → "CPF [CPF_REDACTED]"
# ============================================================

PADROES_SENSIVEIS = {
    "cpf": r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}",
    "cnpj": r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
}


def redigir_texto(texto: str) -> str:
    """
    Mascara dados sensíveis (PII) em um texto antes de persistir em logs.

    COMO FUNCIONA:
    1. Itera sobre cada padrão regex em PADROES_SENSIVEIS
    2. Para cada match, substitui pelo marcador [TIPO_REDACTED]
    3. Retorna o texto com todos os dados sensíveis mascarados

    EXEMPLOS:
    - "CPF 123.456.789-00"     → "CPF [CPF_REDACTED]"
    - "email: joao@empresa.com" → "email: [EMAIL_REDACTED]"
    - Texto sem PII             → retorna inalterado

    IMPORTANTE:
    - Esta função deve ser chamada ANTES de qualquer persistência
    - Ela é usada internamente pelo TraceRecorder._sanitizar()
    - Novos padrões (telefone, cartão, etc.) podem ser adicionados
      ao dicionário PADROES_SENSIVEIS
    """
    texto_redigido = texto
    # Aplicamos cada regex em sequência — a ordem não importa
    # porque os padrões não se sobrepõem
    for tipo, padrao in PADROES_SENSIVEIS.items():
        texto_redigido = re.sub(
            padrao,
            f"[{tipo.upper()}_REDACTED]",
            texto_redigido
        )
    return texto_redigido


# ============================================================
# 2. TraceRecorder — Coletor de Eventos do Agente
# ============================================================
# O TraceRecorder é a classe central de observabilidade.
# Ele coleta eventos em memória e depois persiste em arquivo JSONL.
#
# CICLO DE VIDA:
#   1. Criar instância (gera trace_id único)
#   2. Chamar log_event() em cada etapa do agente
#   3. Ao final, chamar persist() para salvar no disco
#
# DIAGRAMA DO FLUXO:
#
#   ┌─────────────┐
#   │  TraceRecorder │
#   │  trace_id=abc  │
#   └──────┬──────┘
#          │
#   log_event("entrada", {...})     → evento 1 na lista
#   log_event("guardrail", {...})   → evento 2 na lista
#   log_event("tool_call", {...})   → evento 3 na lista
#   log_event("resposta", {...})    → evento 4 na lista
#          │
#   persist()  → salva trace_abc.jsonl no disco
#          │
#   resumo()   → retorna dict com trace_id + contagem + estágios
#
# CAMPOS DE CADA EVENTO:
# - trace_id:  identificador único da execução (hex de 12 chars)
# - timestamp: momento exato do evento (ISO 8601)
# - stage:     fase do agente (entrada, guardrail, tool_call, resposta, erro)
# - payload:   dados da etapa (JÁ SANITIZADOS — sem PII)
# ============================================================

@dataclass
class TraceRecorder:
    """
    Coletor de eventos de observabilidade para agentes.

    RESPONSABILIDADES:
    - Gerar trace_id único por execução (identificação da "sessão")
    - Registrar eventos com timestamp e payload sanitizado
    - Persistir o trace completo em formato JSONL
    - Fornecer resumo para dashboards ou logs de alto nível

    ATRIBUTOS:
    - log_dir: diretório onde os arquivos JSONL serão salvos
    - trace_id: identificador hex de 12 caracteres (gerado automaticamente)
    - events: lista de eventos coletados em memória (antes de persistir)

    ANALOGIA:
    É como um gravador de voz em uma reunião: ele registra tudo, e
    ao final você pode ouvir (ou ler) o que aconteceu em cada momento.
    """

    log_dir: str
    trace_id: str = field(default_factory=lambda: uuid4().hex[:12])
    events: list[dict] = field(default_factory=list)

    def log_event(self, stage: str, payload: dict) -> None:
        """
        Registra um evento no trace atual.

        Parâmetros:
        - stage: nome da fase
        (ex: "entrada", "guardrail", "tool_call", "resposta")
        - payload: dicionário com os dados daquela etapa

        O payload é AUTOMATICAMENTE sanitizado antes de ser armazenado
        (CPF, CNPJ, email são mascarados via _sanitizar).
        """
        evento = {
            "trace_id": self.trace_id,
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "payload": self._sanitizar(payload),
        }
        self.events.append(evento)

    def persist(self) -> str:
        """
        Salva todos os eventos coletados em um arquivo JSONL no disco.

        COMO FUNCIONA:
        1. Cria o diretório de logs se não existir (os.makedirs)
        2. Cria um arquivo com nome trace_{trace_id}.jsonl
        3. Escreve cada evento como uma linha JSON independente
        4. Retorna o caminho completo do arquivo criado

        FORMATO JSONL (uma linha por evento):
        {
            "trace_id": "abc",
            "timestamp": "...",
            "stage": "entrada",
            "payload": {...}
        }
        {
            "trace_id": "abc",
            "timestamp": "...",
            "stage": "guardrail",
            "payload": {...}
        }
        """
        os.makedirs(self.log_dir, exist_ok=True)
        caminho = os.path.join(self.log_dir, f"trace_{self.trace_id}.jsonl")
        with open(caminho, "w", encoding="utf-8") as arquivo:
            for evento in self.events:
                arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")
        return caminho

    def resumo(self) -> dict:
        """
        Retorna um resumo de alto nível do trace para dashboards/logs.

        Retorno:
        - trace_id: o identificador da execução
        - eventos: quantidade total de eventos registrados
        - estagios: lista ordenada dos estágios percorridos
        """
        return {
            "trace_id": self.trace_id,
            "eventos": len(self.events),
            "estagios": [evento["stage"] for evento in self.events],
        }

    def _sanitizar(self, payload: dict) -> dict:
        """
        Mascara dados sensíveis em TODOS os valores string do payload.

        COMO FUNCIONA:
        - Percorre cada par chave/valor do payload
        - Se o valor é string, aplica redigir_texto() para mascarar PII
        - Se é outro tipo (int, float, bool, etc.), mantém inalterado
        - Retorna um NOVO dicionário (não modifica o original)

        IMPORTANTE:
        Este método é chamado automaticamente por log_event().
        O desenvolvedor NÃO precisa sanitizar manualmente.
        """
        sanitizado = {}
        for chave, valor in payload.items():
            if isinstance(valor, str):
                sanitizado[chave] = redigir_texto(valor)
            else:
                sanitizado[chave] = valor
        return sanitizado
