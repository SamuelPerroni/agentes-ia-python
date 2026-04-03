"""
============================================================
MÓDULO 6 - TOOLS ESPECÍFICAS DO AGENTE DE BOLETOS
============================================================
Ferramentas especializadas para o agente completo.
Inclui parsing, validação e processamento de boletos.

DIFERENÇA PARA O tools_base.py (módulo 3):
  tools_base.py    → tools genéricas (cálculo, validação, vencimento)
  tools_boleto.py  → tools especializadas (extração de texto, resumo)

TOOLS NESTE ARQUIVO:
  1. extrair_dados_boleto()   → parseia texto de boleto via regex
  2. calcular_valor_atualizado() → calcula multa e juros
  3. gerar_resumo_boleto()    → formata dados extraídos para display

COMO A LLM USA ESTAS TOOLS:
  A LLM recebe o texto bruto do boleto do usuário e decide:
  1º Chamar extrair_dados_boleto(texto) para parsear
  2º Se vencido, chamar calcular_valor_atualizado(valor, dias)
  3º Opcionalmente, chamar gerar_resumo_boleto(dados)
============================================================
"""

import re
from datetime import datetime


def extrair_dados_boleto(texto_boleto: str) -> dict:
    """
    Extrai campos estruturados de um texto de boleto usando regex.

    COMO FUNCIONA:
    O texto do boleto pode vir em vários formatos (copiado de PDF,
    digitado pelo usuário, etc.). Usamos regex para encontrar
    padrões comuns como:
      "Banco: Itaú"            → captura "Itaú"
      "Valor: R$ 1.500,00"     → captura 1500.00
      "Vencimento: 25/03/2026" → captura a data

    LIMITAÇÕES Do REGEX:
    - Só funciona com formatos previsíveis
    - Textos muito bagunçados podem falhar
    - Em produção, usar OCR + NLP seria mais robusto
    """
    dados = {}

    # REGEX: Extrair nome do banco
    # Tenta dois formatos: "Banco: Nome" e "Nome (código)"
    match = re.search(
        r"(?:Banco|banco)[:\s]*([^\n]+?)(?:\s*\((\d{3})\))?$",
        texto_boleto, re.MULTILINE
    )
    if match:
        banco_nome = match.group(1).strip().rstrip("(").strip()
        dados["banco"] = banco_nome
        if match.group(2):
            dados["banco_codigo"] = match.group(2)

    # REGEX: Formato alternativo "Nome (código)" em qualquer lugar do texto
    if "banco_codigo" not in dados:
        match = re.search(r"(\w[\w\s]+?)\s*\((\d{3})\)", texto_boleto)
        if match:
            dados["banco"] = match.group(1).strip()
            dados["banco_codigo"] = match.group(2)

    # REGEX: Extrair beneficiário (quem recebe o pagamento)
    match = re.search(r"[Bb]enefici[áa]rio[:\s]*(.+)", texto_boleto)
    if match:
        dados["beneficiario"] = match.group(1).strip()

    # REGEX: Extrair CNPJ (formato XX.XXX.XXX/XXXX-XX)
    match = re.search(r"CNPJ[:\s]*([\d./-]+)", texto_boleto)
    if match:
        dados["cnpj"] = match.group(1).strip()

    # REGEX: Extrair valor monetário (converte de "1.500,00" para 1500.00)
    match = re.search(r"[Vv]alor[:\s]*R?\$?\s*([\d.,]+)", texto_boleto)
    if match:
        valor_str = match.group(1).replace(".", "").replace(",", ".")
        try:
            dados["valor"] = float(valor_str)
        except ValueError:
            dados["valor_texto"] = match.group(1)

    # REGEX: Extrair data de vencimento (formato DD/MM/AAAA)
    # Também converte para formato ISO (YYYY-MM-DD) para padronização
    match = re.search(r"[Vv]encimento[:\s]*(\d{2}/\d{2}/\d{4})", texto_boleto)
    if match:
        dados["vencimento"] = match.group(1)
        try:
            dt = datetime.strptime(match.group(1), "%d/%m/%Y")
            dados["vencimento_iso"] = dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # REGEX: Extrair linha digitável
    # (sequência de dígitos com pontos e espaços)
    match = re.search(
        r"[Ll]inha\s*[Dd]igit[áa]vel[:\s]*([\d.\s-]+)",
        texto_boleto
    )
    if match:
        dados["linha_digitavel"] = match.group(1).strip()

    # REGEX: Extrair descrição/referência do boleto
    match = re.search(
        r"[Dd]escri[çc][ãa]o[:\s]*(.+)",
        texto_boleto
    )
    if match:
        dados["descricao"] = match.group(1).strip()

    # CALCULAR STATUS: compara data de vencimento com hoje
    # Isso é feito automaticamente para que a LLM já receba
    # a informação de atraso pronta (sem precisar chamar outra tool)
    if "vencimento" in dados:
        try:
            dt_venc = datetime.strptime(dados["vencimento"], "%d/%m/%Y")
            hoje = datetime.now()
            dias_diff = (hoje - dt_venc).days
            if dias_diff > 0:
                dados["status"] = "VENCIDO"
                dados["dias_atraso"] = dias_diff
            elif dias_diff == 0:
                dados["status"] = "VENCE HOJE"
                dados["dias_atraso"] = 0
            else:
                dados["status"] = "DENTRO DO PRAZO"
                dados["dias_para_vencer"] = abs(dias_diff)
        except ValueError:
            dados["status"] = "DATA_INVALIDA"

    # Métrica de qualidade: quantos campos conseguimos extrair
    dados["campos_extraidos"] = len(dados)
    return dados


def calcular_valor_atualizado(
        valor: float, dias_atraso: int,
        taxa_multa: float = 2.0,
        taxa_juros_dia: float = 0.033
) -> dict:
    """
    Calcula valor atualizado com multa e juros.

    FÓRMULA PADRÃO BRASILEIRO:
      multa = valor × (taxa_multa / 100)              → cobrada UMA vez
      juros = valor × (taxa_juros_dia / 100) × dias   → proporcional aos dias
      total = valor + multa + juros

    TAXAS PADRÃO (Código Civil brasileiro):
      multa: 2% (máximo permitido)
      juros: 0,033%/dia (≈ 1%/mês ≈ 12%/ano)
    """
    if dias_atraso <= 0:
        return {
            "valor_original": valor,
            "multa": 0.0,
            "juros": 0.0,
            "total": valor,
            "descricao_calculo": "Boleto dentro do prazo - sem encargos"
        }

    multa = round(valor * (taxa_multa / 100), 2)
    juros = round(valor * (taxa_juros_dia / 100) * dias_atraso, 2)
    total = round(valor + multa + juros, 2)

    return {
        "valor_original": valor,
        "multa": multa,
        "juros": juros,
        "total": total,
        "dias_atraso": dias_atraso,
        "descricao_calculo": (
            f"Original: R$ {valor:,.2f} + "
            f"Multa {taxa_multa}%: R$ {multa:,.2f} + "
            f"Juros {taxa_juros_dia}%/dia x {dias_atraso} "
            f"dias: R$ {juros:,.2f} = "
            f"Total: R$ {total:,.2f}"
        )
    }


def gerar_resumo_boleto(dados: dict) -> str:
    """Gera um resumo formatado dos dados do boleto para exibição.
    Mapeia chaves internas (snake_case) para labels legíveis.
    """
    linhas = ["=" * 40, "RESUMO DO BOLETO", "=" * 40]

    campos_display = {
        "banco": "Banco",
        "banco_codigo": "Código Banco",
        "beneficiario": "Beneficiário",
        "cnpj": "CNPJ",
        "valor": "Valor",
        "vencimento": "Vencimento",
        "status": "Status",
        "linha_digitavel": "Linha Digitável",
        "descricao": "Descrição",
        "dias_atraso": "Dias em Atraso",
        "dias_para_vencer": "Dias p/ Vencer",
    }

    for chave, label in campos_display.items():
        if chave in dados:
            valor = dados[chave]
            if chave == "valor" and isinstance(valor, (int, float)):
                valor = f"R$ {valor:,.2f}"
            linhas.append(f"  {label}: {valor}")

    linhas.append("=" * 40)
    return "\n".join(linhas)


# ============================================================
# SCHEMAS E REGISTRY PARA O AGENTE COMPLETO
# ============================================================
# Mesma estrutura que tools_base.py, mas para as tools específicas.
# O agente_boletos.py importa estes schemas e registry.

# Tools schema para o agente completo
BOLETO_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "extrair_dados_boleto",
            "description": (
                "Extrai dados estruturados de um texto de boleto bancário. "
                "Use quando receber o texto de um boleto para análise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "texto_boleto": {
                        "type": "string",
                        "description": "Texto completo do boleto"
                    }
                },
                "required": ["texto_boleto"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_valor_atualizado",
            "description": (
                "Calcula o valor atualizado de um boleto vencido "
                "com multa e juros. Use quando precisar saber "
                "quanto pagar por boleto em atraso."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "valor": {
                        "type": "number",
                        "description": "Valor original do boleto",
                    },
                    "dias_atraso": {
                        "type": "integer",
                        "description": "Número de dias em atraso",
                    },
                    "taxa_multa": {
                        "type": "number",
                        "description": "Taxa de multa em % (padrão 2.0)",
                    },
                    "taxa_juros_dia": {
                        "type": "number",
                        "description": "Juros ao dia em % (padrão 0.033)",
                    },
                },
                "required": ["valor", "dias_atraso"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "gerar_resumo_boleto",
            "description": (
                "Gera um resumo formatado dos dados extraídos do boleto."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dados": {
                        "type": "object",
                        "description": (
                            "Dicionário com os dados extraídos do boleto"
                        ),
                    }
                },
                "required": ["dados"]
            }
        }
    }
]

# Registry: nome da tool → função Python
# O agente usa: BOLETO_TOOLS_REGISTRY[nome_da_tool](**args)
BOLETO_TOOLS_REGISTRY = {
    "extrair_dados_boleto": extrair_dados_boleto,
    "calcular_valor_atualizado": calcular_valor_atualizado,
    "gerar_resumo_boleto": gerar_resumo_boleto,
}
