"""
============================================================
MÓDULO 24.1 - ROTEAMENTO DINÂMICO DE TAREFAS
============================================================
Neste módulo, aprendemos a implementar um Router Agent que
classifica cada tarefa e a direciona para o agente especialista
mais adequado, sem intervenção humana.

CONCEITO CHAVE:
Em APA, raramente todas as tarefas têm o mesmo tipo ou
complexidade. Um boleto pode ser simples, mas amanhã chega
uma nota fiscal eletrônica, um contrato de fornecedor e um
extrato bancário. O roteador cuida da triagem.

POR QUE ROTEAMENTO DINÂMICO?
- Um agente genérico ("faz tudo") tem prompt enorme e pior desempenho
- Agentes especializados têm prompts pequenos, precisos e baratos
- O roteador pode usar um modelo ultra-leve (llama-8b) para classificar
  e reservar modelos maiores apenas para o processamento real

PADRÃO ROUTER:

  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │   Entrada ──▶ [ ROUTER ] ──┬──▶ Agente Boleto           │
  │                            ├──▶ Agente NF-e             │
  │                            ├──▶ Agente Contrato         │
  │                            └──▶ Agente Fallback         │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

TIPOS DE ROTEADOR:

  1. Por palavra-chave  → rápido, sem custo, mas frágil
  2. Por template match → regex/patterns, determinístico
  3. Por LLM classifier → robusto, custa tokens, mais preciso
  4. Por embedding sim  → busca por similaridade semântica

Neste módulo usamos o tipo 2 (pattern) para a demo principal
e mostramos como substituir por LLM classifier em produção.

ROTEAMENTO COM FALLBACK E RETRY:

  ┌─────────┐  falhou?  ┌──────────────┐  falhou?  ┌──────────┐
  │ Router  │──────────▶│ Agente Prim. │──────────▶│ Fallback │
  └─────────┘           └──────────────┘           └──────────┘
       │                                                  │
       │      confiança baixa?                           │
       └──────────────────────────────────────── HITL ───┘

Tópicos cobertos:
1. Classificação de documentos por padrões
2. Registro de agentes especializados (plugin pattern)
3. Roteamento com score de confiança
4. Fallback quando nenhum agente é adequado
5. Auditoria de decisões do roteador
6. Roteamento com LLM classifier (simulado)
============================================================
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Console do Rich para output formatado
console = Console()


# ============================================================
# 1. DEFINIÇÃO DE AGENTE ESPECIALISTA
# ============================================================
# Cada agente especialista tem:
# - nome: identificador único
# - descricao: para o LLM classifier e logs
# - padroes: regex que indicam que o documento pertence a ele
# - processar: a função de processamento real
#
# PLUGIN PATTERN: novos agentes são registrados sem alterar
# o roteador. Basta criar e registrar — o roteador os usa
# automaticamente.
# ============================================================

@dataclass
class AgenteEspecialista:
    """
    Definição de um agente especialista registrável no roteador.

    Parâmetros:
    - nome: identificador único (ex: "boleto", "nfe", "contrato")
    - descricao: descrição do domínio do agente
    - padroes: lista de regex que identificam documentos deste tipo
    - min_confianca: pontuação mínima para roteamento (0.0 a 1.0)
    - processar: função f(texto, metadados) → ResultadoAgente
    """

    nome: str
    descricao: str
    padroes: list[str]
    min_confianca: float = 0.5
    processar: Callable[[str, dict[str, Any]], "ResultadoAgente"] | None = None


@dataclass
class ResultadoAgente:
    """Resultado padronizado de qualquer agente especialista."""

    agente: str
    sucesso: bool
    dados: dict[str, Any]
    confianca: float
    tempo_ms: float
    mensagem: str = ""


@dataclass
class DecisaoRoteamento:
    """Registro de uma decisão de roteamento para auditoria."""

    timestamp: str
    texto_preview: str
    agente_selecionado: str
    confianca: float
    motivo: str
    scores_todos: dict[str, float]


# ============================================================
# 2. MOTOR DE CLASSIFICAÇÃO POR PADRÕES
# ============================================================
# Calcula um score de confiança (0.0–1.0) para cada agente
# com base nos padrões regex encontrados no texto.
#
# SCORE = matches_encontrados / total_padroes_do_agente
#
# Exemplo: agente boleto tem 5 padrões, encontrou 3 → score=0.60
#
# EVOLUÇÃO PARA LLM CLASSIFIER:
# Em produção, chame um LLM leve com o texto e a lista de
# agentes disponíveis e peça que ele retorne o nome do agente
# mais adequado. Mais robusto para linguagem natural variada.
# ============================================================

def _calcular_score(texto: str, agente: AgenteEspecialista) -> float:
    """
    Calcula score de confiança para um agente dado um texto.

    Parâmetros:
    - texto: documento a ser classificado (normalizado para minúsculas)
    - agente: agente a ser avaliado

    Retorna:
    - Float entre 0.0 e 1.0
    """
    if not agente.padroes:
        return 0.0
    texto_lower = texto.lower()
    matches = sum(
        1 for p in agente.padroes
        if re.search(p, texto_lower)
    )
    return round(matches / len(agente.padroes), 3)


# ============================================================
# 3. ROTEADOR PRINCIPAL
# ============================================================

class Roteador:
    """
    Classifica documentos e direciona para o agente correto.

    CICLO DE VIDA:
    1. Agentes são registrados via registrar()
    2. rotear() recebe um texto e calcula scores
    3. O agente com maior score acima do threshold é selecionado
    4. Se nenhum passa o threshold → agente_fallback é usado
    5. A decisão é auditada em historico_decisoes

    CONFIGURAÇÃO:
    - threshold_confianca: score mínimo para aceitar roteamento (0.0–1.0)
    - agente_fallback: nome do agente usado quando nada casa
    """

    def __init__(
        self,
        threshold_confianca: float = 0.35,
        agente_fallback: str = "generico",
    ) -> None:
        self.threshold_confianca = threshold_confianca
        self.agente_fallback = agente_fallback
        self._agentes: dict[str, AgenteEspecialista] = {}
        self.historico_decisoes: list[DecisaoRoteamento] = []

    def registrar(self, agente: AgenteEspecialista) -> None:
        """
        Registra um agente especialista no roteador.

        Parâmetros:
        - agente: instância de AgenteEspecialista a ser registrada
        """
        self._agentes[agente.nome] = agente
        console.print(
            f"  [dim]Agente registrado: [cyan]{agente.nome}[/cyan] "
            f"({len(agente.padroes)} padrões)[/dim]"
        )

    def classificar(self, texto: str) -> dict[str, float]:
        """
        Calcula scores de todos os agentes registrados.

        Parâmetros:
        - texto: documento a classificar

        Retorna:
        - Dicionário {nome_agente: score} ordenado por score desc
        """
        scores = {
            nome: _calcular_score(texto, ag)
            for nome, ag in self._agentes.items()
        }
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def rotear(
        self,
        texto: str,
        metadados: dict[str, Any] | None = None,
    ) -> ResultadoAgente:
        """
        Classifica o texto e executa o agente correto.

        Parâmetros:
        - texto: documento a processar
        - metadados: informações adicionais (origem, id, etc.)

        Retorna:
        - ResultadoAgente do especialista selecionado
        """
        metadados = metadados or {}
        scores = self.classificar(texto)

        # Seleciona o agente com maior score acima do threshold
        selecionado = None
        motivo = "sem match acima do threshold"

        for nome, score in scores.items():
            ag = self._agentes[nome]
            if score >= max(self.threshold_confianca, ag.min_confianca):
                selecionado = ag
                motivo = (
                    f"score={score:.2f} ≥ threshold={self.threshold_confianca}"
                )
                break

        # Fallback se nenhum agente foi selecionado
        if selecionado is None:
            if self.agente_fallback in self._agentes:
                selecionado = self._agentes[self.agente_fallback]
                motivo = "fallback (nenhum agente acima do threshold)"
            else:
                return ResultadoAgente(
                    agente="nenhum",
                    sucesso=False,
                    dados={},
                    confianca=0.0,
                    tempo_ms=0.0,
                    mensagem="Nenhum agente registrado para este documento",
                )

        # Registra decisão para auditoria
        self.historico_decisoes.append(DecisaoRoteamento(
            timestamp=datetime.now().isoformat(),
            texto_preview=texto[:80].replace("\n", " "),
            agente_selecionado=selecionado.nome,
            confianca=scores.get(selecionado.nome, 0.0),
            motivo=motivo,
            scores_todos=scores,
        ))

        # Executa o agente
        if selecionado.processar is None:
            return ResultadoAgente(
                agente=selecionado.nome,
                sucesso=False,
                dados={},
                confianca=scores.get(selecionado.nome, 0.0),
                tempo_ms=0.0,
                mensagem="Agente sem função de processamento registrada",
            )

        inicio = time.perf_counter()
        resultado = selecionado.processar(texto, metadados)
        resultado.tempo_ms = round(
            (time.perf_counter() - inicio) * 1000, 1
        )
        return resultado

    def exibir_historico(self) -> None:
        """Exibe tabela de decisões de roteamento para auditoria."""
        tabela = Table(title="Histórico de Decisões — Roteador")
        tabela.add_column("Documento", style="dim", max_width=35)
        tabela.add_column("Agente", style="cyan bold")
        tabela.add_column("Confiança", justify="right")
        tabela.add_column("Motivo", style="dim")

        for d in self.historico_decisoes:
            cor = (
                "green" if d.confianca >= 0.6
                else "yellow" if d.confianca >= 0.35
                else "red"
            )
            tabela.add_row(
                d.texto_preview[:34],
                d.agente_selecionado,
                f"[{cor}]{d.confianca:.0%}[/{cor}]",
                d.motivo[:50],
            )
        console.print(tabela)


# ============================================================
# 4. AGENTES ESPECIALIZADOS (SIMULADOS)
# ============================================================
# Em produção, cada função processar_ chamaria o LLM com o
# prompt especializado do seu domínio.
# ============================================================

def processar_boleto(texto: str, _meta: dict[str, Any]) -> ResultadoAgente:
    """Extrai campos financeiros de boletos bancários."""
    valor_match = re.search(r"r\$\s*([\d.,]+)", texto.lower())
    valor = None
    if valor_match:
        try:
            valor = float(
                valor_match.group(1).replace(".", "").replace(",", ".")
            )
        except ValueError:
            pass
    return ResultadoAgente(
        agente="boleto",
        sucesso=True,
        dados={"tipo": "boleto", "valor": valor, "banco": "extraído"},
        confianca=0.9,
        tempo_ms=0.0,
        mensagem="Boleto processado com sucesso",
    )


def processar_nfe(_texto: str, _meta: dict[str, Any]) -> ResultadoAgente:
    """Extrai dados de Notas Fiscais Eletrônicas."""
    return ResultadoAgente(
        agente="nfe",
        sucesso=True,
        dados={"tipo": "nfe", "chave_acesso": "extraída", "cfop": "6101"},
        confianca=0.85,
        tempo_ms=0.0,
        mensagem="NF-e processada com sucesso",
    )


def processar_contrato(_texto: str, _meta: dict[str, Any]) -> ResultadoAgente:
    """Extrai cláusulas e partes contratantes de contratos."""
    return ResultadoAgente(
        agente="contrato",
        sucesso=True,
        dados={
            "tipo": "contrato",
            "partes": "extraídas",
            "vigencia": "extraída"
        },
        confianca=0.75,
        tempo_ms=0.0,
        mensagem="Contrato processado com sucesso",
    )


def processar_generico(texto: str, _meta: dict[str, Any]) -> ResultadoAgente:
    """Agente fallback para documentos não classificados."""
    return ResultadoAgente(
        agente="generico",
        sucesso=True,
        dados={"tipo": "desconhecido", "texto_preview": texto[:100]},
        confianca=0.0,
        tempo_ms=0.0,
        mensagem="Documento não classificado — HITL recomendado",
    )


def _construir_roteador() -> Roteador:
    """Cria e configura o roteador com todos os agentes registrados."""
    roteador = Roteador(threshold_confianca=0.35, agente_fallback="generico")

    roteador.registrar(AgenteEspecialista(
        nome="boleto",
        descricao="Boletos bancários e cobranças",
        padroes=[
            r"boleto",
            r"c[oó]digo de barras",
            r"linha digit[aá]vel",
            r"benefici[aá]rio",
            r"sacado",
            r"cedente",
            r"vencimento",
            r"banco\s+\w+.*r\$",
        ],
        min_confianca=0.30,
        processar=processar_boleto,
    ))

    roteador.registrar(AgenteEspecialista(
        nome="nfe",
        descricao="Notas Fiscais Eletrônicas (NF-e)",
        padroes=[
            r"nota fiscal eletr[oô]nica",
            r"nf-?e",
            r"chave de acesso",
            r"cfop",
            r"icms",
            r"cnpj.*emitente",
            r"danfe",
        ],
        min_confianca=0.30,
        processar=processar_nfe,
    ))

    roteador.registrar(AgenteEspecialista(
        nome="contrato",
        descricao="Contratos e instrumentos jurídicos",
        padroes=[
            r"contrato",
            r"cl[aá]usula",
            r"contratante",
            r"contratada",
            r"partes",
            r"obriga[çc][oã]es",
            r"rescis[aã]o",
        ],
        min_confianca=0.30,
        processar=processar_contrato,
    ))

    roteador.registrar(AgenteEspecialista(
        nome="generico",
        descricao="Fallback para documentos não classificados",
        padroes=[],
        min_confianca=0.0,
        processar=processar_generico,
    ))

    return roteador


# ============================================================
# DEMO COMPLETA — Roteamento de documentos variados
# ============================================================

def demo_roteamento() -> None:
    """
    Demonstra roteamento dinâmico de documentos heterogêneos.

    ETAPAS:
    1. Configura roteador com 3 agentes especializados + fallback
    2. Processa 5 documentos de tipos diferentes
    3. Exibe scores de confiança de cada agente por documento
    4. Exibe histórico de decisões para auditoria

    OBSERVE NO OUTPUT:
    - Boleto → roteado para agente boleto (alta confiança)
    - NF-e  → roteado para agente nfe (alta confiança)
    - Contrato → roteado para agente contrato
    - Documento ambíguo → score baixo, mas casa com um agente
    - Texto sem padrões → fallback genérico

    EXERCÍCIO SUGERIDO:
    1. Reduza o threshold para 0.10 e veja se o genérico deixa
       de ser acionado para documentos ambíguos
    2. Crie um agente "extrato_bancario" com seus próprios padrões
    3. Substitua _calcular_score por uma chamada ao LLM para
       comparar precisão vs. custo
    """
    console.print(Panel.fit(
        "[bold]Roteamento Dinâmico de Tarefas[/bold]\n"
        "Classificação automática e direcionamento para agentes especialistas",
        title="🔀 Módulo 24 — Roteamento",
        border_style="magenta",
    ))

    console.print("\n[bold]── Registrando agentes ──[/bold]")
    roteador = _construir_roteador()

    documentos = [
        (
            "boleto",
            "BOLETO BANCÁRIO — Banco Itaú\n"
            "Beneficiário: Empresa XYZ Ltda | CNPJ: 00.000.000/0001-00\n"
            "Sacado: João da Silva\n"
            "Linha digitável: 34191.09050 37207.727788 61940.450005 5\n"
            "Vencimento: 05/04/2026 | Valor: R$ 1.250,00",
        ),
        (
            "nfe",
            "NOTA FISCAL ELETRÔNICA — DANFE\n"
            "NF-e nº 000123456 | Série 001\n"
            "CNPJ Emitente: 12.345.678/0001-99\n"
            "Chave de Acesso: "
            "4321 2026 0400 0000 0000 0000 0000 0000 0000 0000 0000\n"
            "CFOP: 6101 | ICMS: 12% | Total: R$ 5.400,00",
        ),
        (
            "contrato",
            "CONTRATO DE PRESTAÇÃO DE SERVIÇOS\n"
            "Cláusula 1ª — Das Partes\n"
            "Contratante: ABC Indústria S.A.\n"
            "Contratada: XYZ Serviços Ltda.\n"
            "Cláusula 2ª — Das Obrigações\n"
            "A contratada se compromete a entregar os serviços\n"
            "nos prazos estabelecidos, sob pena de rescisão.",
        ),
        (
            "ambíguo",
            "Documento de pagamento referente à prestação de serviços\n"
            "Valor a pagar: R$ 800,00\n"
            "Data: 03/04/2026\n"
            "Referência: Serviço de consultoria técnica",
        ),
        (
            "desconhecido",
            "Prezado colaborador,\n"
            "Segue o material solicitado para a reunião de amanhã.\n"
            "Atenciosamente, Departamento de RH",
        ),
    ]

    console.print("\n[bold]── Processando documentos ──[/bold]")
    resultados_tabela: list[tuple[str, ResultadoAgente]] = []

    for tipo, texto in documentos:
        resultado = roteador.rotear(texto, {"tipo_esperado": tipo})
        icone = "[green]✓[/green]" if resultado.sucesso else "[red]✗[/red]"
        cor_agente = {
            "boleto": "green",
            "nfe": "cyan",
            "contrato": "blue",
            "generico": "yellow",
        }.get(resultado.agente, "white")
        console.print(
            f"  {icone} [{tipo}] → "
            f"[{cor_agente}]{resultado.agente}[/{cor_agente}] "
            f"[dim]({resultado.confianca:.0%})[/dim] "
            f"— {resultado.mensagem}"
        )
        resultados_tabela.append((tipo, resultado))

    # Scores detalhados
    console.print("\n[bold]── Scores por documento ──[/bold]")
    tabela_scores = Table(show_header=True)
    tabela_scores.add_column("Documento", style="dim")
    tabela_scores.add_column("boleto", justify="right")
    tabela_scores.add_column("nfe", justify="right")
    tabela_scores.add_column("contrato", justify="right")
    tabela_scores.add_column("Selecionado", style="bold")

    for decisao in roteador.historico_decisoes:
        scores = decisao.scores_todos
        tabela_scores.add_row(
            decisao.texto_preview[:30],
            f"{scores.get('boleto', 0):.0%}",
            f"{scores.get('nfe', 0):.0%}",
            f"{scores.get('contrato', 0):.0%}",
            f"[cyan]{decisao.agente_selecionado}[/cyan]",
        )
    console.print(tabela_scores)

    # Histórico de auditoria
    console.print("\n[bold]── Histórico de decisões ──[/bold]")
    roteador.exibir_historico()

    console.print("\n💡 [bold yellow]Dica:[/bold yellow]")
    console.print(
        "  Em produção, use um LLM leve (llama-8b) para classificar\n"
        "  e um modelo maior apenas no agente especialista selecionado.\n"
        "  O custo de classificação é amortizado pela especialização."
    )


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    demo_roteamento()
