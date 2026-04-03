"""
Tools base compartilhado - funções e schemas reutilizáveis.
Importado pelos módulos 3, 5, 6 e 7.
"""

from datetime import datetime


# ============================================================
# FUNÇÕES (TOOLS)
# ============================================================

def calcular_multa_juros(
        valor_original: float,
        dias_atraso: int,
        taxa_multa: float = 2.0,
        taxa_juros_dia: float = 0.033) -> dict:
    """
    Calcula multa e juros de um boleto vencido.

    FÓRMULA USADA (padrão brasileiro):
      multa = valor * (taxa_multa / 100)            -> aplicada uma vez
      juros = valor * (taxa_juros_dia / 100) * dias -> proporcional
      total = valor + multa + juros
    """
    if dias_atraso <= 0:
        return {
            "valor_original": valor_original,
            "multa": 0.0,
            "juros": 0.0,
            "total": valor_original,
            "status": "DENTRO DO PRAZO"
        }
    multa = valor_original * (taxa_multa / 100)
    juros = valor_original * (taxa_juros_dia / 100) * dias_atraso
    total = valor_original + multa + juros
    return {
        "valor_original": round(valor_original, 2),
        "multa": round(multa, 2),
        "juros": round(juros, 2),
        "total": round(total, 2),
        "dias_atraso": dias_atraso,
        "status": "VENCIDO"
    }


def validar_linha_digitavel(linha: str) -> dict:
    """
    Valida formato básico de uma linha digitável de boleto.

    SOBRE A LINHA DIGITÁVEL:
      - Boleto de cobrança: 47 dígitos (bancos comuns)
      - Boleto de convênio: 48 dígitos (concessionárias)
      - Os 3 primeiros dígitos identificam o banco emissor
    """
    linha_limpa = linha.replace(".", "").replace(" ", "").replace("-", "")
    resultado = {
        "linha_original": linha,
        "linha_limpa": linha_limpa,
        "total_digitos": len(linha_limpa),
        "apenas_numeros": linha_limpa.isdigit(),
    }
    if not linha_limpa.isdigit():
        resultado["valido"] = False
        resultado["erro"] = "Linha digitável deve conter apenas números"
        return resultado
    if len(linha_limpa) == 47:
        resultado["tipo"] = "Boleto bancário (cobrança)"
        resultado["banco_codigo"] = linha_limpa[:3]
        resultado["valido"] = True
    elif len(linha_limpa) == 48:
        resultado["tipo"] = "Boleto de concessionária (convênio)"
        resultado["valido"] = True
    else:
        resultado["valido"] = False
        resultado["erro"] = (
            f"Esperado 47 ou 48 dígitos, encontrado {len(linha_limpa)}"
        )
    return resultado


def verificar_vencimento(data_vencimento: str) -> dict:
    """
    Verifica se um boleto está vencido comparando com a data atual.

    Retorna a diferença em dias e o status:
    VENCIDO / DENTRO DO PRAZO / VENCE HOJE.
    """
    try:
        vencimento = datetime.strptime(data_vencimento, "%d/%m/%Y")
    except ValueError:
        return {
            "erro": f"Formato de data inválido: {data_vencimento}. "
            "Use DD/MM/AAAA",
        }
    hoje = datetime.now()
    diferenca = (hoje - vencimento).days
    return {
        "data_vencimento": data_vencimento,
        "data_atual": hoje.strftime("%d/%m/%Y"),
        "dias_diferenca": diferenca,
        "vencido": diferenca > 0,
        "status": (
            "VENCIDO" if diferenca > 0
            else "DENTRO DO PRAZO" if diferenca < 0
            else "VENCE HOJE"
        ),
    }


def buscar_banco_por_codigo(codigo: str) -> dict:
    """
    Retorna informações do banco pelo código numérico (3 dígitos).
    Os 3 primeiros dígitos da linha digitável identificam o banco.
    Ex: 001 = Banco do Brasil, 341 = Itaú, 237 = Bradesco.
    """
    bancos = {
        "001": "Banco do Brasil", "033": "Santander",
        "104": "Caixa Econômica Federal", "237": "Bradesco",
        "341": "Itaú Unibanco", "389": "Mercantil do Brasil",
        "422": "Safra", "745": "Citibank", "756": "Sicoob",
    }
    nome = bancos.get(codigo)
    if nome:
        return {"codigo": codigo, "nome": nome, "encontrado": True}
    return {
        "codigo": codigo,
        "encontrado": False,
        "erro": "Banco não encontrado",
    }


# ============================================================
# SCHEMAS PARA TOOL CALLING (OpenAI Function Calling Format)
# ============================================================
# O schema diz à LLM QUAIS tools existem e COMO chamá-las.
#
# FORMATO OBRIGATÓRIO (padrão OpenAI, usado pela Groq também):
#   {
#     "type": "function",          ← sempre "function"
#     "function": {
#       "name": "...",              ← deve bater com o nome no REGISTRY
#       "description": "...",      ← CRUCIAL: a LLM decide se usa a tool
#                                     com base nesta descrição!
#       "parameters": {             ← JSON Schema dos parâmetros
#         "type": "object",
#         "properties": {...},
#         "required": [...]        ← parâmetros obrigatórios
#       }
#     }
#   }
#
# DICA: Descrições ruins = LLM usa a tool na hora errada!
# Seja específico: "Calcula multa e juros" > "Faz cálculos"

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calcular_multa_juros",
            "description": (
                "Calcula multa e juros de um boleto vencido. "
                "Use quando o usuário perguntar quanto pagar "
                "por um boleto atrasado."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "valor_original": {
                        "type": "number",
                        "description": "Valor original do boleto em reais",
                    },
                    "dias_atraso": {
                        "type": "integer",
                        "description": "Número de dias em atraso",
                    },
                    "taxa_multa": {
                        "type": "number",
                        "description": "Taxa de multa em % (padrão: 2.0)",
                    },
                    "taxa_juros_dia": {
                        "type": "number",
                        "description": (
                            "Taxa de juros ao dia em % "
                            "(padrão: 0.033)"
                        ),
                    },
                },
                "required": ["valor_original", "dias_atraso"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validar_linha_digitavel",
            "description": (
                "Valida o formato de uma linha digitável de boleto "
                "bancário."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "linha": {
                        "type": "string",
                        "description": (
                            "Linha digitável do boleto "
                            "(47 ou 48 dígitos)"
                        ),
                    },
                },
                "required": ["linha"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verificar_vencimento",
            "description": (
                "Verifica se um boleto está vencido com base na "
                "data de vencimento."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data_vencimento": {
                        "type": "string",
                        "description": (
                            "Data de vencimento no formato "
                            "DD/MM/AAAA"
                        ),
                    },
                },
                "required": ["data_vencimento"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_banco_por_codigo",
            "description": (
                "Busca o nome do banco pelo código numérico "
                "(3 dígitos)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "codigo": {
                        "type": "string",
                        "description": (
                            "Código do banco com 3 dígitos "
                            "(ex: '341')"
                        ),
                    },
                },
                "required": ["codigo"],
            },
        }
    }
]

# ============================================================
# REGISTRY: MAPEIA NOME (string) → FUNÇÃO (callable)
# ============================================================
# Quando a LLM retorna tool_call com name="calcular_multa_juros",
# o nosso código faz: func = TOOLS_REGISTRY["calcular_multa_juros"]
# e depois chama func(**argumentos).
#
# IMPORTANTE: O nome no registry DEVE ser idêntico ao "name" no schema!
TOOLS_REGISTRY = {
    "calcular_multa_juros": calcular_multa_juros,
    "validar_linha_digitavel": validar_linha_digitavel,
    "verificar_vencimento": verificar_vencimento,
    "buscar_banco_por_codigo": buscar_banco_por_codigo,
}
