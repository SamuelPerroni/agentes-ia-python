"""
============================================================
MÓDULO 4.2 - GUARDRAILS: VALIDAÇÃO DE SAÍDA
============================================================
Validamos o que SAI da LLM antes de entregar ao usuário.

CONCEITO CHAVE:
A LLM pode gerar respostas incorretas, em formato errado,
ou com conteúdo inadequado. Guardrails de saída são a última
linha de defesa.

POR QUE VALIDAR A SAÍDA?
  1. LLMs "alucinam" - inventam dados que parecem reais
  2. JSON pode vir malformado (faltando vírgula, chaves, etc.)
  3. A LLM pode vazar o system prompt na resposta
  4. Valores calculados podem estar errados (LLM não é boa em matemática)
  5. Conteúdo fora do escopo pode aparecer (conselhos financeiros, etc.)

3 TIPOS DE GUARDRAIL DE SAÍDA NESTE MÓDULO:
  1. PYDANTIC  - valida formato e regras de negócio do JSON
  2. ALUCINAÇÃO - verifica se a LLM inventou informações
  3. CONTEÚDO  - filtra termos proibidos na resposta
============================================================
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field, field_validator
from rich.console import Console

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
console = Console()


# ============================================================
# GUARDRAIL DE SAÍDA 1: VALIDAÇÃO COM PYDANTIC
# ============================================================
# O QUE É PYDANTIC?
# É uma biblioteca Python que valida dados automaticamente.
# Você define uma classe com tipos e regras, e o Pydantic garante
# que os dados batem com o esperado.
#
# POR QUE USAR PYDANTIC COM LLMs?
# A LLM retorna texto (JSON). Pydantic converte para objetos Python
# e valida tipos, formatos e regras de negócio DE UMA VEZ.
#
# EXEMPLO DO FLUXO:
#   LLM retorna: '{"valor": -100, "banco": "Itaú"}'
#   Pydantic tenta: BoletoExtraido(**dados)
#   Erro!: "valor must be > 0" → capturamos e pedimos correção
#
# RECURSOS USADOS:
#   Field(..., gt=0)        → valor deve ser > 0
#   Field(..., min_length=1)→ string não pode ser vazia
#   Field(..., pattern=...) → string deve bater com regex
#   @field_validator       → validação customizada com lógica Python

class BoletoExtraido(BaseModel):
    """
    Schema Pydantic para validar dados extraídos de um boleto.

    TEORIA:
    - Pydantic valida tipos, formatos e regras de negócio
    - Se a LLM retornar algo fora do schema, capturamos o erro
    - Podemos pedir para a LLM tentar novamente (retry)

    CAMPOS E SUAS VALIDAÇÕES:
    - banco: string não vazia (min_length=1)
    - valor: float positivo (gt=0) - evita valores 0 ou negativos
    - vencimento: formato específico YYYY-MM-DD via regex
    - beneficiario: string com pelo menos 2 chars
    - status: só aceita valores de uma lista pré-definida
    """
    banco: str = Field(..., min_length=1, description="Nome do banco")
    valor: float = Field(..., gt=0, description="Valor do boleto em reais")
    vencimento: str = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Data YYYY-MM-DD"
    )
    beneficiario: str = Field(
        ..., min_length=2, description="Nome do beneficiário"
    )
    status: str = Field(..., description="Status: DENTRO DO PRAZO ou VENCIDO")

    @field_validator("status")
    @classmethod
    def validar_status(cls, v):
        # Validação customizada: só aceita status conhecidos
        # Sem isso, a LLM poderia retornar "ATRASADO", "PENDENTE", etc.
        validos = ["DENTRO DO PRAZO", "VENCIDO", "VENCE HOJE"]
        if v not in validos:
            raise ValueError(f"Status deve ser um de: {validos}")
        return v

    @field_validator("valor")
    @classmethod
    def validar_valor(cls, v):
        # Regra de negócio: valores acima de R$ 1.000.000 são suspeitos
        # Isso pega alucinações onde a LLM inventa valores absurdos
        if v > 1_000_000:
            raise ValueError("Valor suspeito: acima de R$ 1.000.000")
        return v


class CalculoMulta(BaseModel):
    """Schema para validar cálculos de multa/juros.

    A validação cruzada (total >= valor_original) garante consistência
    matemática - se a LLM calcular errado, capturamos aqui.
    """
    valor_original: float = Field(..., gt=0)
    multa: float = Field(..., ge=0)
    juros: float = Field(..., ge=0)
    total: float = Field(..., gt=0)

    @field_validator("total")
    @classmethod
    def validar_total(cls, v, info):
        # VALIDAÇÃO CRUZADA: total nunca pode ser menor que o valor original
        # Isso garante consistência matemática (multa e juros são sempre >= 0)
        valor_orig = info.data.get("valor_original", 0)
        if valor_orig and v < valor_orig:
            raise ValueError("Total não pode ser menor que valor original")
        return v


# ============================================================
# GUARDRAIL DE SAÍDA 2: CHECAGEM DE ALUCINAÇÃO
# ============================================================
# O QUE É ALUCINAÇÃO?
# É quando a LLM gera informação que NÃO está nos dados fornecidos.
# Diferente de "erro": alucinação parece plausível mas é inventada.
#
# EXEMPLOS:
#   Dados: "Banco Itaú, R$ 800,00"
#   Alucinação: "Beneficiário: João Silva" → inventou o nome!
#
# POR QUE É CRÍTICO PARA BOLETOS?
# Valores errados = prejuízo financeiro real.
# Um agente que inventa dados de pagamento é perigoso.
#
# ABORDAGEM: LLM-as-Judge
# Usamos uma segunda chamada à LLM como "juiz" que compara
# a resposta com os dados originais. Não é perfeito, mas é
# a melhor abordagem sem embeddings ou modelos especializados.

def checar_alucinacao(resposta_llm: str, dados_originais: str) -> dict:
    """
    GUARDRAIL: Verifica se a LLM inventou informações.

    TEORIA:
    Alucinação = LLM gera informação que NÃO está nos dados fornecidos.
    Para um agente de boletos, isso é CRÍTICO (valores errados = prejuízo).

    ABORDAGEM: Usar a própria LLM como "juiz" (LLM-as-judge).
    """
    mensagens = [
        {
            "role": "system",
            "content": (
                "Você é um verificador de fatos. "
                "Compare a RESPOSTA com os DADOS ORIGINAIS.\n"
                "Verifique se a resposta contém informações "
                "que NÃO estão nos dados.\n\n"
                "Responda APENAS com JSON:\n"
                "{\n"
                '  "tem_alucinacao": true/false,\n'
                '  "confianca": 0.0-1.0,\n'
                '  "problemas": ["lista de informações '
                'possivelmente inventadas"]\n'
                "}"
            )
        },
        {
            "role": "user",
            "content": (
                f"DADOS ORIGINAIS:\n{dados_originais}"
                f"\n\nRESPOSTA DA LLM:\n{resposta_llm}"
            )
        }
    ]

    resposta = client.chat.completions.create(
        model=MODEL,
        messages=mensagens,
        temperature=0.1,
        max_tokens=300,
    )

    try:
        texto = resposta.choices[0].message.content.strip()
        if texto.startswith("```"):
            texto = texto.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(texto)
    except (json.JSONDecodeError, IndexError):
        return {
            "tem_alucinacao": False,
            "confianca": 0.5,
            "problemas": ["Não foi possível verificar"],
        }


# ============================================================
# GUARDRAIL DE SAÍDA 3: FILTRO DE CONTEÚDO
# ============================================================
# Verifica se a resposta da LLM contém conteúdo inadequado.
# Mesmo com um bom system prompt, a LLM pode:
#   1. Vazar partes do system prompt
#   2. Dar conselhos financeiros não autorizados
#   3. Sugerir ações fora do escopo
#
# Este é o guardrail mais simples: busca por termos proibidos.

# Lista de termos que NÃO devem aparecer na resposta

TERMOS_PROIBIDOS_SAIDA = [
    "system prompt",
    "minhas instruções são",
    "aqui está meu prompt",
    "conselho de investimento",
    "invista em",
    "compre ações",
]


def filtrar_conteudo_saida(resposta: str) -> dict:
    """
    GUARDRAIL: Filtra conteúdo inadequado na saída.

    Verifica se a resposta da LLM contém:
    - Vazamento do system prompt
    - Conselhos financeiros não autorizados
    - Conteúdo fora do escopo
    """
    resposta_lower = resposta.lower()
    termos_encontrados = [
        t for t in TERMOS_PROIBIDOS_SAIDA if t in resposta_lower
    ]

    return {
        "aprovado": len(termos_encontrados) == 0,
        "termos_proibidos": termos_encontrados,
        "mensagem": (
            "Resposta contém conteúdo não permitido"
            if termos_encontrados else None
        ),
    }


# ============================================================
# VALIDAR SAÍDA JSON DA LLM COM RETRY
# ============================================================
# CONCEITO: RETRY COM CORREÇÃO AUTOMÁTICA
# Se a LLM retorna JSON inválido ou fora do schema, podemos:
#   1. Capturar o erro
#   2. Enviar o JSON + erro de volta para a LLM
#   3. Pedir que corrija
#
# Geralmente 1 retry resolve 90% dos casos.
# Limitamos a 2 retries para não gastar tokens infinitamente.

def validar_e_parsear_json(
    resposta_llm: str,
    schema_class: type[BaseModel],
    max_retries: int = 2,
) -> dict:
    """
    Tenta parsear e validar a resposta JSON da LLM.
    Se falhar, pede à LLM para corrigir (retry).

    TEORIA:
    - LLMs nem sempre retornam JSON perfeito
    - Um retry com o erro geralmente resolve
    - Limitar retries para não gastar tokens infinitamente
    """
    # Limpar resposta
    texto = resposta_llm.strip()
    if texto.startswith("```"):
        texto = texto.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    for tentativa in range(max_retries + 1):
        try:
            dados = json.loads(texto)
            objeto_validado = schema_class(**dados)
            return {
                "sucesso": True,
                "dados": objeto_validado.model_dump(),
                "tentativa": tentativa + 1,
            }
        except json.JSONDecodeError as e:
            erro = f"JSON inválido: {e}"
        except (ValueError, TypeError) as e:
            erro = f"Validação falhou: {e}"

        if tentativa < max_retries:
            console.print(
                f"   ⚠️ Tentativa {tentativa + 1} falhou: {erro}",
                style="yellow",
            )
            console.print("   🔄 Pedindo correção à LLM...", style="dim")

            mensagens_correcao = [
                {
                    "role": "system",
                    "content": (
                        "Corrija o JSON abaixo para que seja válido. "
                        "Retorne APENAS o JSON corrigido."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"JSON com erro:\n{texto}\n\nErro: {erro}"
                        f"\n\nSchema esperado: "
                        f"{schema_class.model_json_schema()}"
                    )
                }
            ]

            resp = client.chat.completions.create(
                model=MODEL,
                messages=mensagens_correcao,
                temperature=0.1,
                max_tokens=500,
            )
            texto = resp.choices[0].message.content.strip()
            if texto.startswith("```"):
                texto = texto.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return {
        "sucesso": False,
        "erro": erro,
        "tentativas": max_retries + 1,
    }


# ============================================================
# PIPELINE DE GUARDRAILS DE SAÍDA
# ============================================================
# Executa TODOS os guardrails de saída em sequência.
# Ordem: Conteúdo → Schema → Alucinação
#
# POR QUE ESSA ORDEM?
#   1. Conteúdo: barato (string search) - rejeita rápido
#   2. Schema: médio (Pydantic parse + possíveis retries)
#   3. Alucinação: caro (chamada extra à LLM) - só se passou nos outros

def pipeline_guardrails_saida(
    resposta_llm: str,
    dados_originais: str = None,
    schema: type[BaseModel] = None,
) -> dict:
    """
    Executa TODOS os guardrails de saída.
    """
    resultado = {"aprovado": True, "verificacoes": {}}

    # 1. Filtro de conteúdo
    check_conteudo = filtrar_conteudo_saida(resposta_llm)
    resultado["verificacoes"]["conteudo"] = check_conteudo
    if not check_conteudo["aprovado"]:
        resultado["aprovado"] = False
        resultado["motivo"] = "Conteúdo proibido na resposta"
        return resultado

    # 2. Validação de schema (se fornecido)
    if schema:
        check_schema = validar_e_parsear_json(resposta_llm, schema)
        resultado["verificacoes"]["schema"] = check_schema
        if not check_schema["sucesso"]:
            resultado["aprovado"] = False
            resultado["motivo"] = f"Schema inválido: {check_schema['erro']}"
            return resultado
        resultado["dados_validados"] = check_schema["dados"]

    # 3. Checagem de alucinação (se dados originais fornecidos)
    if dados_originais:
        check_alucinacao = checar_alucinacao(resposta_llm, dados_originais)
        resultado["verificacoes"]["alucinacao"] = check_alucinacao
        if check_alucinacao.get("tem_alucinacao"):
            resultado["aviso_alucinacao"] = True
            resultado["problemas_alucinacao"] = check_alucinacao.get(
                "problemas", []
            )

    return resultado


# ============================================================
# DEMONSTRAÇÃO
# ============================================================
# 4 testes que cobrem cada guardrail de saída:
#   1. JSON válido → Pydantic aceita
#   2. JSON com valor inválido → Pydantic rejeita (valor negativo)
#   3. Filtro de conteúdo → detecta termos proibidos
#   4. Pipeline completa → todos os guardrails juntos
def demo_guardrails_saida():
    """
    Demonstração dos guardrails de saída.
    """
    console.print("\n🎓 DEMO: Guardrails de Saída", style="bold yellow")
    console.print("=" * 60)

    # Teste 1: JSON válido
    console.print("\n📝 Teste 1: JSON válido (Pydantic)", style="bold")
    json_valido = (
        '{"banco": "Itaú", "valor": 1250.00, '
        '"vencimento": "2026-03-25", '
        '"beneficiario": "Empresa ABC", '
        '"status": "DENTRO DO PRAZO"}'
    )
    resultado = validar_e_parsear_json(json_valido, BoletoExtraido)
    console.print(f"   Resultado: {resultado}", style="cyan")

    # Teste 2: JSON com valor inválido
    console.print("\n📝 Teste 2: JSON com valor inválido", style="bold")
    json_invalido = (
        '{"banco": "Itaú", "valor": -100, '
        '"vencimento": "2026-03-25", '
        '"beneficiario": "Empresa ABC", '
        '"status": "DENTRO DO PRAZO"}'
    )
    resultado = validar_e_parsear_json(json_invalido, BoletoExtraido)
    console.print(f"   Resultado: {resultado}", style="cyan")

    # Teste 3: Filtro de conteúdo
    console.print("\n📝 Teste 3: Filtro de conteúdo", style="bold")
    resposta_ok = "O boleto está dentro do prazo. Valor: R$ 500,00."
    resposta_ruim = (
        "O boleto está ok. "
        "Também recomendo que invista em ações da Petrobras."
    )
    res_ok = filtrar_conteudo_saida(resposta_ok)
    res_ruim = filtrar_conteudo_saida(resposta_ruim)
    console.print(f"   Resposta OK: {res_ok}", style="green")
    console.print(f"   Resposta ruim: {res_ruim}", style="red")

    # Teste 4: Pipeline completa
    console.print("\n📝 Teste 4: Pipeline completa de saída", style="bold")
    dados_boleto = "Banco: Itaú, Valor: R$ 800,00, Vencimento: 25/03/2026"
    resposta_llm = json.dumps({
        "banco": "Itaú", "valor": 800.00, "vencimento": "2026-03-25",
        "beneficiario": "Empresa XYZ", "status": "DENTRO DO PRAZO"
    })
    resultado = pipeline_guardrails_saida(
        resposta_llm, dados_boleto, BoletoExtraido
    )
    estilo = "bold green" if resultado["aprovado"] else "bold red"
    console.print(f"   Aprovado: {resultado['aprovado']}", style=estilo)
    if resultado.get("dados_validados"):
        console.print(
            f"   Dados: {resultado['dados_validados']}", style="cyan"
        )


if __name__ == "__main__":
    console.print(
        "🎓 MÓDULO 4.2 - GUARDRAILS: VALIDAÇÃO DE SAÍDA",
        style="bold blue",
    )
    console.print("=" * 60)

    demo_guardrails_saida()

    console.print("\n✅ Módulo 4.2 concluído!", style="bold green")
    console.print("\n💡 RESUMO GUARDRAILS:")
    console.print(
        "   ENTRADA: Injection → Escopo → PII → Tamanho",
        style="yellow",
    )
    console.print("   SAÍDA:   Conteúdo → Schema → Alucinação", style="yellow")
