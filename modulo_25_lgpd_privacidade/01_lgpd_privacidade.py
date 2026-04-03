"""
============================================================
MÓDULO 25.1 - LGPD E PRIVACIDADE DE DADOS NO AGENTE
============================================================
Neste módulo, aprendemos a implementar controles de privacidade
compatíveis com a LGPD (Lei Geral de Proteção de Dados —
Lei 13.709/2018) em agentes de IA que processam dados pessoais.

CONCEITO CHAVE:
Agentes de APA financeira processam CPF, nome, endereço,
dados bancários — todos dados pessoais ou sensíveis pela LGPD.
Sem controles adequados, o agente pode vazar dados nos logs,
nos prompts enviados a APIs externas ou nos arquivos de trace.

RISCOS ESPECÍFICOS EM AGENTES DE IA:
1. Dados pessoais em prompts → enviados ao provedor de LLM
2. Logs de trace com CPF/nome → visíveis em Grafana/Langfuse
3. Histórico de conversa persistido sem criptografia
4. Outputs do agente com dados além do necessário

LGPD — PRINCÍPIOS RELEVANTES PARA AGENTES:
┌───────────────────────────────────────────────────────────┐
│  Princípio           Implementação no agente              │
│  ───────────────────────────────────────────────────────  │
│  Finalidade          Prompt especifica uso restrito       │
│  Necessidade         Anonimize antes de enviar ao LLM     │
│  Segurança           Criptografe dados em repouso/trânsito│
│  Prevenção           Guardrail bloqueia PII no output     │
│  Transparência       Log de finalidade da coleta          │
│  Não discriminação   Testes de viés nos outputs           │
└───────────────────────────────────────────────────────────┘

FLUXO DE PRIVACIDADE:

  Texto c/ PII
       │
       ▼
  [ PII Detector ] ──▶ log de auditoria (tipo de dado, não o dado)
       │
       ▼
  [ Anonimizador ] ──▶ substitui CPF, nome, etc. por tokens
       │
       ▼
  [ LLM ] ←── prompt sem dados pessoais reais
       │
       ▼
  [ Des-anonimizador ] ──▶ restitui os dados originais se necessário
       │
       ▼
  Resultado com PII controlado

  O LLM externo NUNCA vê o CPF real — só o token "[CPF_1]"

Tópicos cobertos:
1. Detecção de PII em texto (CPF, CNPJ, e-mail, telefone, nome)
2. Anonimização reversível por token mapeado
3. Guardrail de saída: bloqueia PII no output do agente
4. Log de auditoria LGPD sem dados pessoais
5. Pseudonimização para dados de treinamento
6. Checklist de conformidade LGPD para APA
============================================================
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Console do Rich para output formatado
console = Console()


# ============================================================
# 1. DETECTORES DE PII
# ============================================================
# PII = Personally Identifiable Information
# (Informação de Identificação Pessoal)
#
# DADOS COBERTOS PELA LGPD (relevantes para APA financeira):
# - CPF / CNPJ         → identificação fiscal
# - Nome completo      → dado pessoal direto
# - E-mail             → contato pessoal
# - Telefone/celular   → contato pessoal
# - CEP / endereço     → dado de localização
# - Dados bancários    → conta, agência, cartão
#
# LIMITAÇÃO: regex captura formatos comuns mas não é perfeito.
# Em produção use presidio (Microsoft), spaCy NER ou um
# serviço gerenciado de detecção de PII.
# ============================================================

PADROES_PII: dict[str, str] = {
    # Documento fiscal — formato: 000.000.000-00 ou 00000000000
    "cpf": r"\b\d{3}[.\s]?\d{3}[.\s]?\d{3}[-\s]?\d{2}\b",

    # Pessoa jurídica — formato: 00.000.000/0000-00
    "cnpj": r"\b\d{2}[.\s]?\d{3}[.\s]?\d{3}[/\s]?\d{4}[-\s]?\d{2}\b",

    # E-mail
    "email": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",

    # Telefone brasileiro — (00) 00000-0000 ou variações
    "telefone": r"\(?\d{2}\)?[\s.-]?\d{4,5}[-\s.]?\d{4}\b",

    # Cartão de crédito/débito — 4 grupos de 4 dígitos
    "cartao": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",

    # CEP — 00000-000 ou 00000000
    "cep": r"\b\d{5}-?\d{3}\b",
}


@dataclass
class OcorrenciaPII:
    """Registro de uma ocorrência de PII detectada."""

    tipo: str
    inicio: int
    fim: int
    valor_original: str
    token: str  # identificador para des-anonimização


def detectar_pii(texto: str) -> list[OcorrenciaPII]:
    """
    Detecta todas as ocorrências de PII em um texto.

    Parâmetros:
    - texto: string a ser analisada

    Retorna:
    - Lista de OcorrenciaPII, ordenada por posição no texto
    """
    ocorrencias: list[OcorrenciaPII] = []
    contador: dict[str, int] = {}

    for tipo, padrao in PADROES_PII.items():
        for match in re.finditer(padrao, texto):
            contador[tipo] = contador.get(tipo, 0) + 1
            token = f"[{tipo.upper()}_{contador[tipo]}]"
            ocorrencias.append(OcorrenciaPII(
                tipo=tipo,
                inicio=match.start(),
                fim=match.end(),
                valor_original=match.group(),
                token=token,
            ))

    # Ordena por posição para facilitar substituição
    ocorrencias.sort(key=lambda o: o.inicio)
    return ocorrencias


# ============================================================
# 2. ANONIMIZADOR REVERSÍVEL
# ============================================================
# Substitui PII por tokens no texto antes de enviar ao LLM.
# Armazena o mapeamento token→dado para des-anonimizar depois.
#
# REVERSÍVEL vs. IRREVERSÍVEL:
# - Reversível (este módulo): para processamento → des-anonimiza o output
# - Irreversível (pseudonimização): para logs/treino → hash SHA-256
#
# SEGURANÇA:
# O dicionário de mapeamento NÃO deve ser persistido junto
# com o texto anonimizado. Mantenha-os separados.
# ============================================================

class Anonimizador:
    """
    Anonimiza PII em textos e permite des-anonimização do output.

    CICLO DE USO:
        anon = Anonimizador()
        texto_limpo, mapa = anon.anonimizar(texto_original)
        # Envie texto_limpo ao LLM
        resposta_limpa = chamar_llm(texto_limpo)
        # Restitua os dados originais na resposta
        resposta_final = anon.des_anonimizar(resposta_limpa)
    """

    def anonimizar(
        self, texto: str
    ) -> tuple[str, dict[str, str]]:
        """
        Substitui PII por tokens no texto.

        Parâmetros:
        - texto: texto original com dados pessoais

        Retorna:
        - (texto_anonimizado, mapa_token_para_original)
        """
        ocorrencias = detectar_pii(texto)
        if not ocorrencias:
            return texto, {}

        mapa: dict[str, str] = {}
        resultado = texto
        # Substituição de trás para frente para preservar índices
        for oc in reversed(ocorrencias):
            mapa[oc.token] = oc.valor_original
            resultado = resultado[:oc.inicio] + oc.token + resultado[oc.fim:]

        return resultado, mapa

    def des_anonimizar(
        self,
        texto_anonimizado: str,
        mapa: dict[str, str],
    ) -> str:
        """
        Restitui os dados originais em um texto anonimizado.

        Parâmetros:
        - texto_anonimizado: texto com tokens [CPF_1], [EMAIL_1], etc.
        - mapa: dicionário retornado por anonimizar()

        Retorna:
        - Texto com dados originais restaurados
        """
        resultado = texto_anonimizado
        for token, original in mapa.items():
            resultado = resultado.replace(token, original)
        return resultado


# ============================================================
# 3. PSEUDONIMIZAÇÃO PARA LOGS E TREINO
# ============================================================
# Diferente da anonimização reversível, a pseudonimização
# é ONE-WAY: substitui o dado por um hash que NÃO permite
# recuperar o valor original.
#
# USO: logs de trace, datasets de avaliação, telemetria.
# LGPD Art. 13: dados pseudonimizados têm regime jurídico
# mais flexível para finalidade de pesquisa.
# ============================================================

def pseudonimizar(valor: str, salt: str = "treinamento-agentes") -> str:
    """
    Gera hash SHA-256 de um valor pessoal para logs.

    O hash é determinístico: o mesmo CPF sempre gera o même hash,
    permitindo correlacionar registros sem expor o dado real.

    Parâmetros:
    - valor: dado pessoal (CPF, nome, etc.)
    - salt: segredo para dificultar ataques de dicionário
             Em produção: use um segredo diferente por tenant

    Retorna:
    - Primeiros 16 caracteres do hash hex (suficiente para correlação)
    """
    dados = f"{salt}:{valor}".encode()
    return hashlib.sha256(dados).hexdigest()[:16]


# ============================================================
# 4. GUARDRAIL DE SAÍDA — BLOQUEIA PII NO OUTPUT DO AGENTE
# ============================================================
# Mesmo que o agente receba texto anonimizado, ele pode
# "adivinhar" ou gerar dados pessoais no output.
# O guardrail de saída é a última linha de defesa.
# ============================================================

def guardrail_saida_pii(
    texto_output: str,
    tipos_bloqueados: list[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Verifica e remove PII do output do agente.

    Parâmetros:
    - texto_output: resposta gerada pelo agente
    - tipos_bloqueados: tipos de PII a bloquear
      (padrão: todos — ["cpf","cnpj","email","telefone","cartao","cep"])

    Retorna:
    - (texto_sanitizado, lista_de_pii_encontrada_e_removida)
    """
    if tipos_bloqueados is None:
        tipos_bloqueados = list(PADROES_PII.keys())

    encontrados: list[str] = []
    resultado = texto_output

    for tipo in tipos_bloqueados:
        padrao = PADROES_PII[tipo]
        matches = list(re.finditer(padrao, resultado))
        if matches:
            encontrados.extend(
                f"{tipo}: ***REMOVIDO***" for _ in matches
            )
            resultado = re.sub(padrao, f"[{tipo.upper()}_REMOVIDO]", resultado)

    return resultado, encontrados


# ============================================================
# 5. LOG DE AUDITORIA LGPD
# ============================================================
# Registra QUAIS tipos de dados foram processados, sem registrar
# os dados em si. Exigido pelo Art. 37 da LGPD (registro de
# operações de tratamento) e essencial para responder ao DPO.
#
# CAMPOS OBRIGATÓRIOS (recomendados pelo ANPD):
# - finalidade do tratamento
# - base legal (Art. 7 LGPD)
# - tipos de dados envolvidos (NÃO os valores)
# - timestamp e identificador da operação
# ============================================================

@dataclass
class RegistroAuditoria:
    """Registro de uma operação de tratamento de dados pessoais."""

    timestamp: str
    operacao_id: str
    finalidade: str
    base_legal: str
    tipos_dados: list[str]
    quantidade_ocorrencias: int
    resultado: str  # "processado" | "bloqueado" | "anonimizado"


class RegistradorLGPD:
    """
    Registra operações de tratamento para conformidade LGPD.

    Em produção, persista os registros em banco imutável
    (append-only) e mantenha por no mínimo 5 anos.
    """

    def __init__(self) -> None:
        self._registro: list[RegistroAuditoria] = []

    def registrar(
        self,
        operacao_id: str,
        finalidade: str,
        base_legal: str,
        ocorrencias: list[OcorrenciaPII],
        resultado: str,
    ) -> None:
        """
        Registra uma operação de tratamento.

        Parâmetros:
        - operacao_id: identificador único da operação
        - finalidade: ex "análise de boleto bancário"
        - base_legal: ex "execução de contrato (Art. 7, V)"
        - ocorrencias: lista retornada por detectar_pii()
        - resultado: "processado", "bloqueado" ou "anonimizado"
        """
        tipos = list({o.tipo for o in ocorrencias})
        self._registro.append(RegistroAuditoria(
            timestamp=datetime.now().isoformat(),
            operacao_id=operacao_id,
            finalidade=finalidade,
            base_legal=base_legal,
            tipos_dados=tipos,
            quantidade_ocorrencias=len(ocorrencias),
            resultado=resultado,
        ))

    def exibir_registro(self) -> None:
        """Exibe o registro de auditoria formatado."""
        tabela = Table(title="Registro de Tratamento de Dados — LGPD")
        tabela.add_column("ID Operação", style="dim")
        tabela.add_column("Finalidade", style="cyan")
        tabela.add_column("Base Legal", style="dim")
        tabela.add_column("Tipos de Dados", style="yellow")
        tabela.add_column("Qt.", justify="right")
        tabela.add_column("Resultado")

        for r in self._registro:
            cor = {
                "processado": "green",
                "anonimizado": "blue",
                "bloqueado": "red",
            }.get(r.resultado, "white")
            tabela.add_row(
                r.operacao_id[:20],
                r.finalidade[:35],
                r.base_legal[:30],
                ", ".join(r.tipos_dados),
                str(r.quantidade_ocorrencias),
                f"[{cor}]{r.resultado}[/{cor}]",
            )
        console.print(tabela)


# ============================================================
# DEMO COMPLETA — Pipeline de privacidade LGPD
# ============================================================

def demo_lgpd() -> None:
    """
    Demonstra o pipeline completo de privacidade compatível com LGPD.

    ETAPAS:
    1. Detecta e categoriza PII em um boleto simulado
    2. Anonimiza o texto antes de enviar ao LLM (simulado)
    3. Des-anonimiza o output para o usuário final
    4. Guardrail de saída remove PII residual
    5. Registra a operação no log de auditoria LGPD
    6. Exibe checklist de conformidade

    OBSERVE NO OUTPUT:
    - O LLM (simulado) recebe texto sem CPF/e-mail reais
    - Des-anonimização restaura os dados apenas na resposta final
    - O log não contém nenhum dado pessoal real, só metadados

    EXERCÍCIO SUGERIDO:
    1. Adicione o padrão de "nome completo" (2+ palavras maiúsculas)
       e veja como o detector se comporta
    2. Teste o guardrail de saída com uma resposta que "vaza" um CPF
    3. Implemente um decorator @privacidade que envolve qualquer
       função de agente com o pipeline anonimizar → executar → des-anonimizar
    """
    console.print(Panel.fit(
        "[bold]LGPD e Privacidade de Dados no Agente[/bold]\n"
        "Anonimização, guardrail de saída e log de auditoria",
        title="🔒 Módulo 25 — LGPD e Privacidade",
        border_style="red",
    ))

    anonimizador = Anonimizador()
    registrador = RegistradorLGPD()

    # Boleto simulado com múltiplos dados pessoais
    texto_boleto = (
        "BOLETO BANCÁRIO — Banco Bradesco S.A.\n"
        "Beneficiário: João da Silva | CPF: 123.456.789-09\n"
        "Endereço: Rua das Flores, 100 — CEP: 01310-100 — São Paulo/SP\n"
        "E-mail para notificação: joao.silva@empresa.com.br\n"
        "Telefone: (11) 98765-4321\n"
        "Sacado: Empresa ABC Ltda | CNPJ: 12.345.678/0001-99\n"
        "Valor: R$ 1.250,00 | Vencimento: 05/04/2026"
    )

    # ── Passo 1: Detecção ──────────────────────────────────────
    console.print("\n[bold]── 1. Detecção de PII ──[/bold]")
    ocorrencias = detectar_pii(texto_boleto)

    tabela_pii = Table(show_header=True)
    tabela_pii.add_column("Tipo", style="yellow")
    tabela_pii.add_column("Token")
    tabela_pii.add_column("Valor detectado", style="dim red")

    for oc in ocorrencias:
        tabela_pii.add_row(oc.tipo, oc.token, oc.valor_original)
    console.print(tabela_pii)

    # ── Passo 2: Anonimização ──────────────────────────────────
    console.print("\n[bold]── 2. Anonimização reversível ──[/bold]")
    texto_anonimizado, mapa = anonimizador.anonimizar(texto_boleto)
    console.print(
        "[dim]Texto enviado ao LLM (sem PII real):[/dim]\n"
        f"[green]{texto_anonimizado}[/green]"
    )

    # ── Passo 3: Chamada ao LLM (simulada) ────────────────────
    console.print("\n[bold]── 3. Processamento pelo LLM (simulado) ──[/bold]")
    # O LLM recebe texto com tokens, processa e devolve texto com tokens
    resposta_llm = (
        "Boleto identificado:\n"
        "- Sacado: [CPF_1] (Bradesco)\n"
        "- Empresa pagadora: [CNPJ_1]\n"
        "- Contato: [EMAIL_1] | [TELEFONE_1]\n"
        "- Valor: R$ 1.250,00 | Status: dentro do prazo"
    )
    console.print(f"[dim]Resposta do LLM (com tokens):\n{resposta_llm}[/dim]")

    # ── Passo 4: Des-anonimização ──────────────────────────────
    console.print("\n[bold]── 4. Des-anonimização para o usuário ──[/bold]")
    resposta_final = anonimizador.des_anonimizar(resposta_llm, mapa)
    console.print(f"[green]{resposta_final}[/green]")

    # ── Passo 5: Guardrail de saída ────────────────────────────
    console.print("\n[bold]── 5. Guardrail de saída ──[/bold]")
    texto_suspeito = (
        "Analisei o boleto. O CPF do sacado é 123.456.789-09 "
        "e o e-mail é joao.silva@empresa.com.br. Valor: R$ 1.250,00."
    )
    sanitizado, removidos = guardrail_saida_pii(texto_suspeito)
    console.print(f"  Input  : [dim]{texto_suspeito}[/dim]")
    console.print(f"  Output : [green]{sanitizado}[/green]")
    if removidos:
        console.print(
            f"  [red]⚠ {len(removidos)} ocorrência(s) de PII removida(s) "
            f"do output[/red]"
        )

    # ── Passo 6: Pseudonimização para logs ────────────────────
    console.print("\n[bold]── 6. Pseudonimização para logs ──[/bold]")
    cpf_exemplo = "123.456.789-09"
    hash_cpf = pseudonimizar(cpf_exemplo)
    console.print(
        f"  CPF real  : [red]{cpf_exemplo}[/red]\n"
        f"  Hash log  : [dim]{hash_cpf}[/dim] "
        f"(correlacionável, não reversível)"
    )

    # ── Log de auditoria LGPD ──────────────────────────────────
    registrador.registrar(
        operacao_id="OP-2026-0403-001",
        finalidade="Análise automatizada de boleto bancário",
        base_legal="Execução de contrato (Art. 7, V — LGPD)",
        ocorrencias=ocorrencias,
        resultado="anonimizado",
    )
    registrador.registrar(
        operacao_id="OP-2026-0403-002",
        finalidade="Guardrail: remoção de PII em output",
        base_legal="Segurança e prevenção (Art. 6, VII — LGPD)",
        ocorrencias=ocorrencias[:2],
        resultado="bloqueado",
    )

    console.print("\n[bold]── 7. Log de auditoria LGPD ──[/bold]")
    registrador.exibir_registro()

    # ── Checklist de conformidade ─────────────────────────────
    console.print("\n[bold]── Checklist de conformidade LGPD ──[/bold]")
    checklist = [
        ("Mapeamento de dados pessoais no fluxo do agente", True),
        ("Anonimização antes de enviar ao LLM externo", True),
        ("Guardrail de saída bloqueia PII residual", True),
        ("Log de auditoria sem dados pessoais reais", True),
        ("Logs de trace e observabilidade sem CPF/nome", True),
        ("Dados de treino/avaliação pseudonimizados", True),
        ("Política de retenção de dados (Art. 16 LGPD)", False),
        ("Contrato DPA com provedor de LLM (Art. 26 LGPD)", False),
        ("Canal de atendimento para direitos do titular (Art. 18)", False),
    ]
    tabela_check = Table(show_header=False)
    tabela_check.add_column("Status", justify="center")
    tabela_check.add_column("Item")

    for item, ok in checklist:
        icone = "[green]✓[/green]" if ok else "[red]✗[/red]"
        tabela_check.add_row(icone, item)
    console.print(tabela_check)

    console.print("\n💡 [bold yellow]Dica:[/bold yellow]")
    console.print(
        "  Use a biblioteca microsoft/presidio para detecção\n"
        "  de PII mais robusta em produção (NER + regex combinados).\n"
        "  Consulte o DPO antes de enviar qualquer dado pessoal\n"
        "  a provedores externos de LLM."
    )


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    demo_lgpd()
