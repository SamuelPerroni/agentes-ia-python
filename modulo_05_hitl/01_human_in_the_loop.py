"""
============================================================
MÓDULO 5 - HUMAN-IN-THE-LOOP (HITL)
============================================================
Implementamos a camada de supervisão humana no agente.

CONCEITO CHAVE:
Nem toda decisão do agente deve ser automática.
Ações de alto risco precisam de aprovação humana.

O QUE É HITL?
Human-in-the-Loop significa colocar um HUMANO no fluxo de
decisão do agente. O agente processa, mas o humano APROVA.

POR QUE HITL É IMPORTANTE?
  1. CONFIANÇA: LLMs podem errar, especialmente com valores financeiros
  2. RESPONSABILIDADE: quem responde se o agente errar? O humano decide.
  3. REGULAÇÃO: muitos setores exigem supervisão humana (finanças, saúde)
  4. QUALIDADE: feedback humano melhora o agente ao longo do tempo

QUANDO USAR HITL:
- Valores financeiros acima de um limite
- Ações irreversíveis (pagamentos, exclusões)
- Quando a confiança do agente é baixa
- Dados ambíguos ou incompletos

NÍVEIS DE RISCO (implementados aqui):
  ┌──────────────────────────────────────────────┐
  │ BAIXO   → Automático, sem intervenção        │
  │ MÉDIO   → Executa + Notifica humano          │
  │ ALTO    → PARA e pede aprovação humana       │
  │ CRÍTICO → Aprovação + Verificação extra      │
  └──────────────────────────────────────────────┘
============================================================
"""

import os
import json
from enum import Enum
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
console = Console()


# ============================================================
# NÍVEIS DE RISCO
# ============================================================
# Definimos níveis de risco usando Enum para garantir que só
# valores válidos sejam usados (evita erros de digitlação).
#
# Cada nível corresponde a uma ação diferente do sistema:
#   BAIXO   → < R$ 1.000    → processa sozinho
#   MÉDIO   → R$ 1k - 5k   → processa + notifica
#   ALTO    → R$ 5k - 10k  → PARA e espera aprovação
#   CRÍTICO → > R$ 10k      → aprovação + verificação extra

class NivelRisco(str, Enum):
    """Enum para níveis de risco de ações do agente."""
    # Ação automática, sem intervenção
    BAIXO = "BAIXO"
    # Notifica humano, mas executa mesmo assim
    MEDIO = "MEDIO"
    # PARA e requer aprovação humana explícita
    ALTO = "ALTO"
    # Requer aprovação + verificação adicional (ex: MFA)
    CRITICO = "CRITICO"


def classificar_risco(valor: float = 0, dados_completos: bool = True) -> dict:
    """
    Classifica o nível de risco de uma ação do agente.

    TEORIA:
    A classificação de risco determina se o agente pode agir
    sozinho ou precisa de aprovação humana.

    Critérios típicos:
    - Valor da operação
    - Tipo de ação (leitura vs escrita)
    - Completude dos dados
    - Confiança na extração
    """
    # Regras de classificação
    if not dados_completos:
        return {
            "nivel": NivelRisco.ALTO,
            "motivo": "Dados incompletos - requer verificação humana"
        }

    if valor > 10000:
        return {
            "nivel": NivelRisco.CRITICO,
            "motivo": f"Valor alto: R$ {valor:,.2f}"
        }
    elif valor > 5000:
        return {
            "nivel": NivelRisco.ALTO,
            "motivo": f"Valor elevado: R$ {valor:,.2f}"
        }
    elif valor > 1000:
        return {
            "nivel": NivelRisco.MEDIO,
            "motivo": f"Valor moderado: R$ {valor:,.2f}"
        }
    else:
        return {
            "nivel": NivelRisco.BAIXO,
            "motivo": "Operação de baixo risco"
        }


# ============================================================
# PONTO DE APROVAÇÃO HUMANA
# ============================================================
# Este é o "break point" onde o agente PARA e espera decisão.
# Em produção, isso poderia ser:
#   - Um popup na tela do operador
#   - Uma notificação no Slack/Teams
#   - Uma fila de aprovação em um dashboard
#   - Um email com link de aprovação
# Aqui usamos console interativo (Rich Prompt) para demonstrar.

def solicitar_aprovacao_humana(
        acao: str,
        detalhes: dict,
        nivel_risco: NivelRisco
) -> dict:
    """
    HITL: Solicita aprovação humana para uma ação.

    TEORIA:
    O ponto de aprovação deve:
    1. Mostrar claramente O QUE será feito
    2. Mostrar POR QUE precisa de aprovação
    3. Dar opções: Aprovar, Rejeitar, Modificar
    4. Registrar a decisão (auditoria)
    """
    # Montar painel de informações
    # O painel mostra TUDO que o humano precisa para decidir:
    # - O QUE será feito (ação)
    # - POR QUE precisa de aprovação (nível de risco)
    # - DETALHES da operação (valores, beneficiário, etc.)
    info = f"""
[bold]Ação:[/bold] {acao}
[bold]Nível de Risco:[/bold] {nivel_risco.value}

[bold]Detalhes:[/bold]
"""
    for chave, valor in detalhes.items():
        info += f"  • {chave}: {valor}\n"

    console.print(
        Panel(
            info,
            title="🔐 APROVAÇÃO NECESSÁRIA",
            border_style="yellow"
        )
    )

    # Solicitar decisão do humano
    # 3 opções: aprovar (seguir), rejeitar (cancelar), modificar (ajustar)
    # Default é "rejeitar" por segurança (fail-safe)
    decisao = Prompt.ask(
        "Decisão",
        choices=["aprovar", "rejeitar", "modificar"],
        default="rejeitar"
    )

    resultado = {
        "decisao": decisao,
        "acao": acao,
        "nivel_risco": nivel_risco.value,
    }

    if decisao == "modificar":
        modificacao = Prompt.ask("Descreva a modificação")
        resultado["modificacao"] = modificacao
    elif decisao == "rejeitar":
        motivo = Prompt.ask(
            "Motivo da rejeição (Enter para pular)", default=""
        )
        if motivo:
            resultado["motivo_rejeicao"] = motivo

    return resultado


# ============================================================
# FEEDBACK HUMANO
# ============================================================
# O feedback é diferente da aprovação: acontece DEPOIS da resposta.
# Serve para MELHORAR o agente ao longo do tempo.
#
# USOS DO FEEDBACK:
#   1. Ajustar prompts com base em erros recorrentes
#   2. Criar datasets para fine-tuning
#   3. Calcular métricas (satisfação, precisão)
#   4. Identificar padrões de falha

def coletar_feedback(resposta_agente: str) -> dict:
    """
    Coleta feedback humano sobre a resposta do agente.

    TEORIA:
    Feedback humano pode ser usado para:
    1. Melhorar prompts futuros
    2. Treinar/fine-tune modelos
    3. Identificar padrões de erro
    4. Calcular métricas de qualidade
    """
    console.print(
        Panel(
            resposta_agente,
            title="Resposta do Agente",
            border_style="cyan"
        )
    )

    nota = Prompt.ask("Avalie (1-5)", choices=["1", "2", "3", "4", "5"])
    correto = Confirm.ask("A informação estava correta?")

    feedback = {
        "nota": int(nota),
        "correto": correto,
    }

    if not correto:
        correcao = Prompt.ask("Qual seria a resposta correta?")
        feedback["correcao"] = correcao

    if int(nota) < 3:
        sugestao = Prompt.ask("Sugestão de melhoria")
        feedback["sugestao"] = sugestao

    return feedback


# ============================================================
# AGENTE COM HITL - FLUXO COMPLETO
# ============================================================
# Esta é a classe que integra HITL no fluxo do agente.
# O fluxo completo (com todas as camadas) fica:
#
#   Input → Guardrails Entrada → LLM → Classificar Risco
#     → Se risco baixo: processa automaticamente
#     → Se risco alto: PARA → Humano decide → continua ou cancela
#     → Guardrails Saída → Output

class AgenteHITL:
    """
    Agente que integra Human-in-the-Loop no fluxo.

    FLUXO:
    1. Receber input do usuário
    2. Guardrails de entrada
    3. LLM processa
    4. Classificar risco da ação
    5. Se risco alto → HITL (aprovação)
    6. Guardrails de saída
    7. Executar ação (ou não)
    8. Coletar feedback
    """

    def __init__(self):
        # Armazena todas as decisões HITL para auditoria
        # Em produção, isso iria para um banco de dados
        self.historico_decisoes: list[dict] = []
        # Armazena feedbacks para análise posterior
        self.feedbacks: list[dict] = []

    def processar_boleto(
            self,
            dados_boleto: dict,
            modo_demo: bool = False
    ) -> dict:
        """Processa um boleto com camada HITL.

        modo_demo=True simula aprovação automática (para testes).
        modo_demo=False pede aprovação real via console.
        """
        console.print("\n🔄 Processando boleto...", style="bold")

        # Extrair dados
        valor = dados_boleto.get("valor", 0)
        beneficiario = dados_boleto.get("beneficiario", "Desconhecido")
        vencimento = dados_boleto.get("vencimento", "N/A")
        # Verificar completude dos dados
        # Dados incompletos = risco alto (a LLM pode ter falhado na extração)
        dados_completos = all(
            dados_boleto.get(k) for k in [
                "valor",
                "beneficiario",
                "vencimento",
                "banco"
            ]
        )

        # Classificar risco
        risco = classificar_risco(
            valor=valor,
            dados_completos=dados_completos,
        )

        nivel = NivelRisco(risco["nivel"])
        if nivel == NivelRisco.BAIXO:
            estilo_risco = "green"
        elif nivel == NivelRisco.MEDIO:
            estilo_risco = "yellow"
        else:
            estilo_risco = "red"
        console.print(
            f"📊 Risco classificado: {nivel.value} - {risco['motivo']}",
            style=estilo_risco,
        )

        # Decisão baseada no risco
        # Só ALTO e CRÍTICO pausam o fluxo para aprovação
        # BAIXO e MÉDIO continuam automaticamente
        if nivel in (NivelRisco.ALTO, NivelRisco.CRITICO):
            if modo_demo:
                # Em modo demo, simular aprovação automática
                console.print(
                    "   [DEMO] Simulando aprovação humana → APROVADO",
                    style="dim"
                )
                decisao = {"decisao": "aprovar", "demo": True}
            else:
                decisao = solicitar_aprovacao_humana(
                    acao=f"Processar boleto de {beneficiario}",
                    detalhes={
                        "Beneficiário": beneficiario,
                        "Valor": f"R$ {valor:,.2f}",
                        "Vencimento": vencimento,
                        "Banco": dados_boleto.get("banco", "N/A"),
                    },
                    nivel_risco=nivel,
                )

            self.historico_decisoes.append(decisao)

            if decisao["decisao"] == "rejeitar":
                return {
                    "status": "REJEITADO",
                    "motivo": decisao.get(
                        "motivo_rejeicao",
                        "Rejeitado pelo operador"
                    )
                }
            elif decisao["decisao"] == "modificar":
                return {
                    "status": "MODIFICAÇÃO_REQUERIDA",
                    "modificacao": decisao.get("modificacao")
                }

        elif nivel == NivelRisco.MEDIO:
            console.print(
                "   ℹ️ Risco médio - processando com notificação",
                style="yellow"
            )
        else:
            console.print(
                "   ✅ Risco baixo - processando automaticamente",
                style="green"
            )

        # Simular processamento
        resultado = {
            "status": "PROCESSADO",
            "beneficiario": beneficiario,
            "valor": valor,
            "vencimento": vencimento,
            "nivel_risco": nivel.value,
            "aprovacao_humana": nivel in (NivelRisco.ALTO, NivelRisco.CRITICO),
        }

        return resultado

    def mostrar_historico(self):
        """Mostra histórico de decisões HITL."""
        if not self.historico_decisoes:
            console.print("Nenhuma decisão HITL registrada.", style="dim")
            return

        console.print("\n📋 HISTÓRICO DE DECISÕES HITL:", style="bold yellow")
        for i, decisao in enumerate(self.historico_decisoes, 1):
            d = decisao["decisao"]
            if d == "aprovar":
                emoji = "✅"
            elif d == "rejeitar":
                emoji = "❌"
            else:
                emoji = "✏️"
            console.print(
                f"   {i}. {emoji} {d.upper()} "
                f"- {decisao.get('acao', 'N/A')}"
            )


# ============================================================
# DEMONSTRAÇÃO
# ============================================================
# 4 cenários que demonstram cada nível de risco:
#   1. R$ 150    → BAIXO  → automático
#   2. R$ 8.500  → ALTO   → requer aprovação
#   3. Dados incompletos → ALTO (faltam campos)
#   4. R$ 25.000 → CRÍTICO → aprovação + extra
def demo_hitl():
    """Demonstração do agente com camada Human-in-the-Loop (HITL)."""
    console.print("\n🎓 DEMO: Human-in-the-Loop", style="bold yellow")
    console.print("=" * 60)

    agente = AgenteHITL()

    # Cenário 1: Baixo risco (automático)
    console.print(
        "\n📌 CENÁRIO 1: Boleto de baixo valor (automático)",
        style="bold"
    )
    resultado = agente.processar_boleto({
        "banco": "Itaú",
        "beneficiario": "Loja ABC",
        "valor": 150.00,
        "vencimento": "20/03/2026",
    }, modo_demo=True)
    console.print(f"   Resultado: {resultado}", style="cyan")

    # Cenário 2: Alto risco (requer aprovação)
    console.print(
        "\n📌 CENÁRIO 2: Boleto de alto valor (requer aprovação)",
        style="bold"
    )
    resultado = agente.processar_boleto({
        "banco": "Bradesco",
        "beneficiario": "Fornecedor XYZ Ltda",
        "valor": 8500.00,
        "vencimento": "15/03/2026",
    }, modo_demo=True)
    console.print(f"   Resultado: {resultado}", style="cyan")

    # Cenário 3: Crítico (dados incompletos)
    console.print(
        "\n📌 CENÁRIO 3: Dados incompletos (requer aprovação)",
        style="bold"
    )
    resultado = agente.processar_boleto({
        "beneficiario": "Empresa Desconhecida",
        "valor": 3000.00,
    }, modo_demo=True)
    console.print(f"   Resultado: {resultado}", style="cyan")

    # Cenário 4: Valor crítico
    console.print(
        "\n📌 CENÁRIO 4: Valor muito alto (CRÍTICO)",
        style="bold"
    )
    resultado = agente.processar_boleto({
        "banco": "Santander",
        "beneficiario": "Grande Fornecedor SA",
        "valor": 25000.00,
        "vencimento": "30/03/2026",
    }, modo_demo=True)
    console.print(f"   Resultado: {resultado}", style="cyan")

    # Histórico
    agente.mostrar_historico()

    console.print("\n💡 RESUMO HITL:", style="bold yellow")
    console.print("   BAIXO  → Automático", style="green")
    console.print("   MÉDIO  → Automático + Notificação", style="yellow")
    console.print("   ALTO   → Requer Aprovação Humana", style="red")
    console.print(
        "   CRÍTICO → Aprovação + Verificação Extra",
        style="bold red"
    )


def demo_hitl_interativo():
    """Demo interativo - o humano realmente aprova/rejeita."""
    console.print(
        "\n🎓 DEMO INTERATIVA: Human-in-the-Loop",
        style="bold yellow"
    )
    console.print("Você será o humano no loop!\n", style="dim")

    agente = AgenteHITL()

    resultado = agente.processar_boleto({
        "banco": "Itaú",
        "beneficiario": "Mega Fornecedor Ltda",
        "valor": 7500.00,
        "vencimento": "10/03/2026",
    }, modo_demo=False)

    console.print(
        f"\nResultado Final: {json.dumps(
            resultado,
            indent=2,
            ensure_ascii=False
        )}",
        style="cyan"
    )

    # Coletar feedback
    if Confirm.ask("\nDeseja avaliar a resposta do agente?"):
        feedback = coletar_feedback(
            json.dumps(
                resultado,
                indent=2,
                ensure_ascii=False
            )
        )
        console.print(f"Feedback registrado: {feedback}", style="green")


if __name__ == "__main__":
    console.print("🎓 MÓDULO 5 - HUMAN-IN-THE-LOOP (HITL)", style="bold blue")
    console.print("=" * 60)

    # Demo automática
    demo_hitl()

    # Demo interativa - descomente para testar com aprovação manual
    # demo_hitl_interativo()

    console.print("\n✅ Módulo 5 concluído!", style="bold green")
    console.print(
        "\n💡 PRÓXIMO: Projeto completo - Agente de Boletos (Módulo 6) →",
        style="yellow"
    )
