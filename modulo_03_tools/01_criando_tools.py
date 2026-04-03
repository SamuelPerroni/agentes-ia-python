"""
============================================================
MÓDULO 3.1 - CRIANDO TOOLS (FERRAMENTAS)
============================================================
Tools são funções Python que o agente pode chamar.
A LLM decide QUANDO e COM QUAIS ARGUMENTOS chamar cada tool.

CONCEITO CHAVE:
Tools transformam a LLM de "apenas texto" para "capaz de agir".
A LLM não executa a tool — ela pede para o nosso código executar.

POR QUE TOOLS SÃO ESSENCIAIS?
Sem tools, a LLM só consegue GERAR TEXTO. Com tools, ela pode:
  - Fazer cálculos precisos (em vez de "chutar" valores)
  - Consultar dados reais (APIs, bancos de dados)
  - Executar ações no mundo real (enviar email, criar arquivo)
  - Validar informações (verificar CPF, CEP, etc.)

Fluxo:
  User → LLM → "Quero chamar tool X com args Y"
  → Código executa → Resultado volta pra LLM

3 PARTES DE UMA TOOL:
  1. FUNÇÃO PYTHON   → a lógica real (o que faz)
  2. SCHEMA JSON     → a descrição para a LLM (como chamar)
  3. REGISTRY (dict) → o mapeamento nome → função (conexão)
============================================================
"""

import json
from datetime import datetime
from rich.console import Console

console = Console()


# ============================================================
# PASSO 1: DEFINIR FUNÇÕES PYTHON NORMAIS
# ============================================================
# Primeiro, criamos funções Python comuns.
# Depois, criamos o "schema" para a LLM entender.
#
# BOAS PRÁTICAS PARA FUNÇÕES-TOOL:
#   • Retornar SEMPRE um dict (dados estruturados)
#   • Usar type hints (float, int, str) nos parâmetros
#   • Ter valores padrão para parâmetros opcionais
#   • Tratar erros internamente (retornar {"erro": "..."} em vez de raise)
#   • Docstring clara — ajuda a escrever a descrição no schema

def calcular_multa_juros(
        valor_original: float,
        dias_atraso: int,
        taxa_multa: float = 2.0,
        taxa_juros_dia: float = 0.033) -> dict:
    """
    Calcula multa e juros de um boleto vencido.

    Args:
        valor_original: Valor original do boleto em reais
        dias_atraso: Número de dias em atraso
        taxa_multa: Percentual de multa (padrão 2%)
        taxa_juros_dia: Percentual de juros por dia (padrão 0.033%)

    Returns:
        Dicionário com valores calculados
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

    A linha digitável tem 47 dígitos para boletos comuns.
    """
    # Remover pontos e espaços
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
    Verifica se um boleto está vencido.

    Args:
        data_vencimento: Data no formato DD/MM/AAAA
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
    """Retorna informações do banco pelo código."""
    bancos = {
        "001": "Banco do Brasil",
        "033": "Santander",
        "104": "Caixa Econômica Federal",
        "237": "Bradesco",
        "341": "Itaú Unibanco",
        "389": "Mercantil do Brasil",
        "422": "Safra",
        "745": "Citibank",
        "756": "Sicoob",
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
# PASSO 2: CRIAR SCHEMAS PARA A LLM (OpenAI Function Calling Format)
# ============================================================
# A LLM precisa saber:
# - Nome da função (deve bater com o REGISTRY)
# - Descrição (o que ela faz — a LLM DECIDE se chama baseado nisso!)
# - Parâmetros (nome, tipo, descrição, obrigatórios)
#
# FORMATO DO SCHEMA (JSON Schema):
#   "type": "object"          → parâmetros são um objeto
#   "properties": {}          → cada parâmetro com tipo e descrição
#   "required": []            → parâmetros que a LLM DEVE enviar
#
# DICA IMPORTANTE:
# Parâmetros com default no Python (taxa_multa=2.0) NÃO vão em "required".
# Isso permite que a LLM omita esses parâmetros e o Python use o default.
#
# POR QUE A DESCRIÇÃO É TÃO IMPORTANTE?
# A LLM lê a descrição de TODAS as tools antes de decidir qual usar.
# Uma descrição ruim = a LLM usa a tool errada ou não usa quando deveria.
# Inclua QUANDO usar: "Use quando o usuário perguntar..."

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
                        "description": (
                            "Valor original do boleto em reais "
                            "(ex: 1500.00)"
                        ),
                    },
                    "dias_atraso": {
                        "type": "integer",
                        "description": "Número de dias em atraso"
                    },
                    "taxa_multa": {
                        "type": "number",
                        "description": (
                            "Taxa de multa em percentual "
                            "(padrão: 2.0)"
                        ),
                    },
                    "taxa_juros_dia": {
                        "type": "number",
                        "description": (
                            "Taxa de juros ao dia em percentual "
                            "(padrão: 0.033)"
                        ),
                    }
                },
                "required": ["valor_original", "dias_atraso"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validar_linha_digitavel",
            "description": (
                "Valida o formato de uma linha digitável de boleto "
                "bancário. Use quando o usuário fornecer uma linha "
                "digitável para verificação."
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
                    }
                },
                "required": ["linha"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verificar_vencimento",
            "description": (
                "Verifica se um boleto está vencido com base na data "
                "de vencimento. Use quando precisar saber se um boleto "
                "já passou da data."
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
                    }
                },
                "required": ["data_vencimento"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_banco_por_codigo",
            "description": (
                "Busca o nome do banco pelo código numérico (3 dígitos). "
                "Use quando precisar identificar qual banco emitiu o "
                "boleto."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "codigo": {
                        "type": "string",
                        "description": (
                            "Código do banco com 3 dígitos "
                            "(ex: '341' para Itaú)"
                        ),
                    }
                },
                "required": ["codigo"]
            }
        }
    }
]

# Registro: mapeia nome da tool → função Python
# Este dicionário é o "elo" entre o que a LLM pede e o que o código executa:
#   LLM diz: "chamar calcular_multa_juros"
#   → REGISTRY["calcular_multa_juros"] → função Python
TOOLS_REGISTRY = {
    "calcular_multa_juros": calcular_multa_juros,
    "validar_linha_digitavel": validar_linha_digitavel,
    "verificar_vencimento": verificar_vencimento,
    "buscar_banco_por_codigo": buscar_banco_por_codigo,
}


# ============================================================
# DEMONSTRAÇÃO
# ============================================================
# Aqui testamos as tools DIRETAMENTE (sem LLM), para verificar
# que as funções funcionam corretamente antes de conectá-las ao agente.
# Só no próximo módulo (3.2) vamos integrar com a LLM!
def demo_tools():
    """Demonstra as tools funcionando diretamente (sem LLM)."""
    console.print("\n🎓 DEMO: Testando Tools Diretamente", style="bold yellow")
    console.print("=" * 50)

    # Test 1: Calcular multa
    console.print("\n📌 Tool: calcular_multa_juros", style="bold")
    resultado = calcular_multa_juros(1500.00, 15)
    console.print("   Input: valor=1500, dias_atraso=15")
    saida = json.dumps(resultado, indent=2, ensure_ascii=False)
    console.print(f"   Output: {saida}", style="cyan")

    # Test 2: Validar linha digitável
    console.print("\n📌 Tool: validar_linha_digitavel", style="bold")
    linha = "34191.09065 44830.136706 00000.000178 1 70060000125000"
    resultado = validar_linha_digitavel(linha)
    saida = json.dumps(resultado, indent=2, ensure_ascii=False)
    console.print(f"   Output: {saida}", style="cyan")

    # Test 3: Verificar vencimento
    console.print("\n📌 Tool: verificar_vencimento", style="bold")
    resultado = verificar_vencimento("01/03/2026")
    saida = json.dumps(resultado, indent=2, ensure_ascii=False)
    console.print(f"   Output: {saida}", style="cyan")

    # Test 4: Buscar banco
    console.print("\n📌 Tool: buscar_banco_por_codigo", style="bold")
    resultado = buscar_banco_por_codigo("341")
    saida = json.dumps(resultado, indent=2, ensure_ascii=False)
    console.print(f"   Output: {saida}", style="cyan")

    # Mostrar schemas - é isso que a LLM "vê" ao decidir qual tool usar
    console.print(
        "\n📋 SCHEMAS DAS TOOLS (o que a LLM recebe):",
        style="bold yellow",
    )
    console.print(
        "   A LLM lê esses schemas e decide qual tool chamar!",
        style="dim",
    )
    for tool in TOOLS_SCHEMA:
        func = tool["function"]
        console.print(f"\n   🔧 {func['name']}", style="bold")
        desc = func['description'][:80]
        console.print(f"      Descrição: {desc}...", style="dim")
        params = func["parameters"]["properties"]
        required = func["parameters"].get("required", [])
        for pname, pinfo in params.items():
            req = "✅" if pname in required else "  "
            pdesc = pinfo['description'][:60]
            console.print(
                f"      {req} {pname}: {pinfo['type']} - {pdesc}",
                style="dim",
            )


if __name__ == "__main__":
    console.print("🎓 MÓDULO 3.1 - CRIANDO TOOLS", style="bold blue")
    console.print("=" * 60)

    demo_tools()

    console.print("\n✅ Módulo 3.1 concluído!", style="bold green")
    console.print(
        "\n💡 PRÓXIMO: Ver a LLM CHAMANDO estas tools (Módulo 3.2) →",
        style="yellow",
    )
