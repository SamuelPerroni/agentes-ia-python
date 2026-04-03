"""
============================================================
Guardrails de entrada - módulo importável.
============================================================
Este arquivo é a VERSÃO IMPORTÁVEL dos guardrails de entrada.
Diferente do 01_validacao_entrada.py (que é a demo/treinamento),
este é o módulo que outros arquivos importam:

  from modulo_04_guardrails.validacao_entrada import (
      pipeline_guardrails_entrada
  )

Usado por: modulo_06_agente_boletos/agente_boletos.py

CONTÉM:
  - detectar_prompt_injection(): regex contra injection
  - validar_escopo(): keywords de tópicos permitidos
  - detectar_pii(): regex para CPF, CNPJ, email, etc.
  - mascarar_pii(): substitui dados sensíveis por [MASCARADO]
  - validar_tamanho(): limite de caracteres
  - pipeline_guardrails_entrada(): executa TODOS em sequência
"""

import re
from dotenv import load_dotenv

load_dotenv()

# Padrões de prompt injection (regex)
# Veja 01_validacao_entrada.py para explicação detalhada de cada padrão
PADROES_INJECTION = [
    r"ignore\s+(todas?\s+)?(as\s+)?instru[çc][õo]es",
    r"ignore\s+(all\s+)?previous",
    r"esqueça\s+(tudo|todas)",
    r"forget\s+(all|everything)",
    r"you\s+are\s+now",
    r"voc[eê]\s+agora\s+[eé]",
    r"novo\s+modo",
    r"jailbreak",
    r"DAN\s+mode",
    r"system\s*prompt",
    r"revelar?\s+(seu|suas?)\s+instru",
    r"reveal\s+your\s+instructions",
]

# Tópicos que definem o escopo do agente (domínio de boletos)
TOPICOS_PERMITIDOS = [
    "boleto", "pagamento", "vencimento", "multa", "juros",
    "banco", "linha digitável", "código de barras", "beneficiário",
    "valor", "cobrança", "fatura",
]

# Regex para detectar dados pessoais
# (PII - Personally Identifiable Information)
PADROES_PII = {
    "cpf": r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}",
    "cnpj": r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "telefone": r"\(?\d{2}\)?\s?\d{4,5}-?\d{4}",
    "cartao_credito": r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",
}


def detectar_prompt_injection(texto: str) -> dict:
    """Detecta possíveis tentativas de prompt injection usando regex."""
    texto_lower = texto.lower()
    for padrao in PADROES_INJECTION:
        match = re.search(padrao, texto_lower)
        if match:
            return {
                "bloqueado": True,
                "motivo": "Possível prompt injection",
                "padrao_detectado": match.group(),
            }
    return {"bloqueado": False}


def validar_escopo(texto: str) -> dict:
    """Valida se o texto do usuário está dentro do escopo
    definido pelos tópicos permitidos."""
    texto_lower = texto.lower()
    topicos_encontrados = [t for t in TOPICOS_PERMITIDOS if t in texto_lower]
    if topicos_encontrados:
        return {
            "dentro_do_escopo": True,
            "topicos_detectados": topicos_encontrados,
        }
    return {
        "dentro_do_escopo": False,
        "mensagem": (
            "Só posso ajudar com questões "
            "relacionadas a boletos bancários."
        ),
    }


def detectar_pii(texto: str) -> dict:
    """Detecta se o texto contém informações pessoais (PII) usando regex."""
    pii_encontrado = {}
    for tipo, padrao in PADROES_PII.items():
        matches = re.findall(padrao, texto)
        if matches:
            pii_encontrado[tipo] = {"quantidade": len(matches)}
    return {
        "contem_pii": bool(pii_encontrado),
        "tipos_encontrados": list(pii_encontrado.keys()),
    }


def mascarar_pii(texto: str) -> str:
    """Substitui dados pessoais (PII) por placeholders [TIPO_MASCARADO]."""
    texto_mascarado = texto
    for tipo, padrao in PADROES_PII.items():
        texto_mascarado = re.sub(
            padrao, f"[{tipo.upper()}_MASCARADO]", texto_mascarado
        )
    return texto_mascarado


def validar_tamanho(texto: str, max_chars: int = 2000) -> dict:
    """Valida se o texto não excede o limite de caracteres."""
    return {
        "valido": len(texto) <= max_chars,
        "tamanho": len(texto),
        "limite": max_chars,
    }


def pipeline_guardrails_entrada(texto: str) -> dict:
    """Pipeline completa de guardrails.
    Ordem: tamanho → injection → PII → escopo."""
    resultado = {"aprovado": True, "verificacoes": {}}

    check_tamanho = validar_tamanho(texto)
    resultado["verificacoes"]["tamanho"] = check_tamanho
    if not check_tamanho["valido"]:
        resultado["aprovado"] = False
        resultado["motivo_bloqueio"] = "Mensagem excede limite de tamanho"
        return resultado

    check_injection = detectar_prompt_injection(texto)
    resultado["verificacoes"]["injection"] = check_injection
    if check_injection["bloqueado"]:
        resultado["aprovado"] = False
        resultado["motivo_bloqueio"] = "Possível prompt injection"
        return resultado

    check_pii = detectar_pii(texto)
    resultado["verificacoes"]["pii"] = check_pii
    if check_pii["contem_pii"]:
        resultado["aviso_pii"] = True
        resultado["texto_mascarado"] = mascarar_pii(texto)

    check_escopo = validar_escopo(texto)
    resultado["verificacoes"]["escopo"] = check_escopo
    if not check_escopo.get("dentro_do_escopo", False):
        resultado["aprovado"] = False
        resultado["motivo_bloqueio"] = "Fora do escopo do agente"
        return resultado

    return resultado
