"""
============================================================
MÓDULO 6 - GUARDRAILS ESPECÍFICOS DO AGENTE DE BOLETOS
============================================================
Guardrails especializados para o contexto de boletos.

DIFERENÇA PARA O MÓDULO 4 (guardrails genéricos):
  Módulo 4: guardrails genéricos (injection, PII, escopo, tamanho)
  Este arquivo: guardrails DE NEGÓCIO específicos para boletos

CONTÉM:
  1. SCHEMAS PYDANTIC  → valida formato dos dados extraídos
  2. REGRAS DE NEGÓCIO → limites de valor, prazo, completude
  3. CLASSIFICAÇÃO DE RISCO → score numérico que determina se
     precisa de aprovação humana (integra com HITL do módulo 5)

FLUXO NO AGENTE:
  Dados extraídos → validar_completude() → validar_regras_negocio()
  → classificar_risco_boleto() → se risco alto → HITL
============================================================
"""

from pydantic import BaseModel, Field, field_validator


# ============================================================
# SCHEMA DE VALIDAÇÃO DO BOLETO (Pydantic)
# ============================================================
# Estes models definem o "contrato" do que é um boleto válido.
# Se a LLM retornar dados que não batem, o Pydantic rejeita.
#
# Campos obrigatórios: banco, beneficiário, valor, vencimento, status
# Campos opcionais: linha_digitável, CNPJ, descrição

class BoletoValidado(BaseModel):
    """Schema Pydantic para boleto processado pelo agente.

    Cada campo tem validações específicas:
    - banco: não pode ser vazio
    - beneficiário: pelo menos 2 caracteres
    - valor: deve ser positivo e ≤ R$ 1.000.000
    - status: só valores predefinidos (evita alucinação)
    """
    banco: str = Field(..., min_length=1)
    beneficiario: str = Field(..., min_length=2)
    valor: float = Field(..., gt=0, le=1_000_000)
    vencimento: str = Field(...)
    status: str = Field(...)
    linha_digitavel: str | None = None
    cnpj: str | None = None
    descricao: str | None = None

    @field_validator("status")
    @classmethod
    def validar_status(cls, v):
        """Valida que o status é um dos valores predefinidos."""
        validos = ["DENTRO DO PRAZO", "VENCIDO", "VENCE HOJE", "DATA_INVALIDA"]
        if v not in validos:
            raise ValueError(f"Status inválido. Esperado: {validos}")
        return v


class ResultadoProcessamento(BaseModel):
    """Schema para resultado completo do processamento.

    Este model valida o output FINAL do agente antes de entregar
    ao usuário. Garante que o agente sempre retorne:
    - Dados extraídos estruturados
    - Valor a pagar calculado
    - Indicação se precisa de aprovação humana
    - Resumo legível
    """
    boleto_id: str | None = None
    dados_extraidos: dict
    valor_a_pagar: float = Field(..., gt=0)
    requer_aprovacao: bool
    nivel_risco: str
    resumo: str


# ============================================================
# GUARDRAILS DE NEGÓCIO
# ============================================================
# Regras de negócio específicas para boletos no Brasil.
# Estas regras vão além da validação de formato (Pydantic)
# e verificam CONSISTÊNCIA e PLAUSIBILIDADE dos dados.

def validar_regras_negocio(dados_boleto: dict) -> dict:
    """
    Valida regras de negócio específicas para boletos.

    5 REGRAS IMPLEMENTADAS:
    1. Valor máximo: > R$ 100.000 → possível erro de extração
    2. Prazo de prescrição: > 365 dias vencido → pode estar prescrito
    3. Valor mínimo: < R$ 1,00 → possível erro de decimal
    4. Beneficiário ausente: dado crítico faltando
    5. Data futura distante: > 365 dias → possível erro de data
    """
    erros = []
    avisos = []

    valor = dados_boleto.get("valor", 0)
    dias_atraso = dados_boleto.get("dias_atraso", 0)

    # Regra 1: Valor máximo (proteção contra erros de extração)
    # Se o regex pegou o número errado, o valor pode estar absurdo
    if valor > 100_000:
        erros.append(f"Valor R$ {valor:,.2f} excede limite de R$ 100.000,00")

    # Regra 2: Boleto muito vencido (prescrição)
    # Após 1 ano, boletos podem estar prescritos (depende do tipo)
    if dias_atraso > 365:
        erros.append(
            f"Boleto vencido há {dias_atraso} dias (mais de 1 ano)."
            f" Pode estar prescrito."
        )

    # Regra 3: Valor muito baixo (possível erro de decimal)
    # R$ 0,50 pode indicar que perdeu os zeros (R$ 500,00)
    if 0 < valor < 1:
        avisos.append(
            f"Valor muito baixo: R$ {valor:.2f}. Verificar se está correto."
        )

    # Regra 4: Beneficiário ausente (dado crítico)
    # Sem beneficiário, não sabemos para quem estamos "pagando"
    if not dados_boleto.get("beneficiario"):
        erros.append(
            "Beneficiário não identificado. Não processar sem esta informação."
        )

    # Regra 5: Data futura muito distante (possível erro de digitação)
    dias_para_vencer = dados_boleto.get("dias_para_vencer", 0)
    if dias_para_vencer > 365:
        avisos.append(
            f"Boleto vence em {dias_para_vencer} dias."
            f" Verificar se a data está correta."
        )

    return {
        "valido": len(erros) == 0,
        "erros": erros,
        "avisos": avisos,
    }


def validar_completude(dados_boleto: dict) -> dict:
    """Verifica se todos os campos obrigatórios foram extraídos.

    Campos OBRIGATÓRIOS: banco, beneficiário, valor, vencimento
    (sem eles, não podemos processar o boleto com segurança)

    Campos DESEJADOS: CNPJ, linha digitável, descrição
    (enriquecem a análise mas não bloqueiam o processamento)

    O score_completude (0.0 a 1.0) indica a qualidade da extração.
    """
    campos_obrigatorios = ["banco", "beneficiario", "valor", "vencimento"]
    campos_desejados = ["cnpj", "linha_digitavel", "descricao"]

    faltantes_obrigatorios = [
        c for c in campos_obrigatorios if not dados_boleto.get(c)
    ]
    faltantes_desejados = [
        c for c in campos_desejados if not dados_boleto.get(c)
    ]

    total_campos = len(campos_obrigatorios) + len(campos_desejados)
    campos_preenchidos = total_campos
    campos_preenchidos -= len(faltantes_obrigatorios)
    campos_preenchidos -= len(faltantes_desejados)

    return {
        "completo": len(faltantes_obrigatorios) == 0,
        "campos_faltantes_obrigatorios": faltantes_obrigatorios,
        "campos_faltantes_desejados": faltantes_desejados,
        "score_completude": round(campos_preenchidos / total_campos, 2),
    }


# ============================================================
# CLASSIFICAÇÃO DE RISCO PARA BOLETOS
# ============================================================
# Sistema de PONTUAÇÃO que acumula fatores de risco.
# Quanto maior o score, maior o risco.
#
# TABELA DE PONTUAÇÃO:
#   Valor > R$ 10k:        +3 pontos
#   Valor > R$ 5k:         +2 pontos
#   Valor > R$ 1k:         +1 ponto
#   Campos faltando:       +2 pontos
#   Regras violadas:       +3 pontos
#
# CLASSIFICAÇÃO:
#   0 pontos  = BAIXO   (automático)
#   1-2 pontos = MÉDIO  (notificação)
#   3-4 pontos = ALTO   (aprovação humana)
#   5+ pontos  = CRÍTICO (aprovação + extra)

def classificar_risco_boleto(dados_boleto: dict) -> dict:
    """Classificação de risco baseada em score numérico acumulativo."""
    score_risco = 0
    fatores = []

    valor = dados_boleto.get("valor", 0)

    # Fator: Valor
    if valor > 10000:
        score_risco += 3
        fatores.append(f"Valor alto: R$ {valor:,.2f}")
    elif valor > 5000:
        score_risco += 2
        fatores.append(f"Valor moderado-alto: R$ {valor:,.2f}")
    elif valor > 1000:
        score_risco += 1

    # Fator: Completude
    completude = validar_completude(dados_boleto)
    if not completude["completo"]:
        score_risco += 2
        fatores.append(
            f"Campos obrigatórios faltando: "
            f"{completude['campos_faltantes_obrigatorios']}"
        )

    # Fator: Regras de negócio
    regras = validar_regras_negocio(dados_boleto)
    if not regras["valido"]:
        score_risco += 3
        fatores.extend(regras["erros"])

    # Determinar nível
    if score_risco >= 5:
        nivel = "CRITICO"
    elif score_risco >= 3:
        nivel = "ALTO"
    elif score_risco >= 1:
        nivel = "MEDIO"
    else:
        nivel = "BAIXO"

    return {
        "nivel": nivel,
        "score": score_risco,
        "fatores": fatores,
        "requer_aprovacao_humana": nivel in ("ALTO", "CRITICO"),
    }
