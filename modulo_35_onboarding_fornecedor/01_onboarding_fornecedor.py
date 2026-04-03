"""
============================================================
MÓDULO 35.1 - ONBOARDING AUTOMATIZADO DE FORNECEDORES
============================================================
Neste módulo, aprendemos a construir um pipeline APA
multi-etapa que valida automaticamente o cadastro de
novos fornecedores antes da aprovação.

CONCEITO CHAVE:
O cadastro de fornecedor é um processo crítico de
compliance: um fornecedor irregular pode gerar multas,
problemas fiscais e riscos de reputação. Analistas levam
2-3 dias verificando manualmente — o agente faz em segundos.

PIPELINE DE VALIDAÇÃO:

  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  Solicitação de Cadastro                                 │
  │       │                                                  │
  │       ▼                                                  │
  │  [1] Validar CNPJ (formato + dígito verificador)         │
  │       │                                                  │
  │       ▼                                                  │
  │  [2] Consultar Receita Federal (situação cadastral)      │
  │       │  IRREGULAR → REJEITAR                            │
  │       ▼                                                  │
  │  [3] Verificar Lista de Sanções (CEIS/CNEP/OFAC)         │
  │       │  ENCONTRADO → REJEITAR                           │
  │       ▼                                                  │
  │  [4] Verificar Certidões (CND, FGTS, Trabalhista)        │
  │       │  NEGATIVA → APROVAÇÃO CONDICIONAL                │
  │       ▼                                                  │
  │  [5] Decisão Final → APROVADO / CONDICIONAL / REJEITADO  │
  │       │                                                  │
  │       ▼                                                  │
  │  [6] LLM gera parecer em linguagem natural               │
  └──────────────────────────────────────────────────────────┘

CATEGORIAS DE RESULTADO:
  APROVADO     → todas as verificações passaram
  CONDICIONAL  → pendências menores (certidão vencida)
  REJEITADO    → bloqueio crítico (CNPJ irregular, sanção)

Tópicos cobertos:
1. Validação de CNPJ com algoritmo de dígito verificador
2. Simulação de consulta Receita Federal
3. Simulação de consulta a listas de sanções
4. Verificação de certidões com validade
5. Motor de decisão multi-critério
6. Geração de parecer em linguagem natural (LLM simulado)
============================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. VALIDAÇÃO DE CNPJ
# ============================================================

def validar_cnpj(cnpj: str) -> bool:
    """
    Valida CNPJ usando o algoritmo oficial dos dígitos
    verificadores (módulo 11).
    Remove pontuação antes de validar.
    """
    # Remove máscara
    numeros = re.sub(r"\D", "", cnpj)
    if len(numeros) != 14:
        return False
    if len(set(numeros)) == 1:
        return False   # ex.: 00.000.000/0000-00

    def _calcular(n: str, pesos: list[int]) -> int:
        soma = sum(
            int(d) * p for d, p in zip(n, pesos)
        )
        resto = soma % 11
        return 0 if resto < 2 else 11 - resto

    p1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    p2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    d1 = _calcular(numeros[:12], p1)
    d2 = _calcular(numeros[:13], p2)
    return numeros[-2:] == f"{d1}{d2}"


# ============================================================
# 2. MODELOS
# ============================================================

@dataclass
class SolicitacaoCadastro:
    """Dados submetidos para cadastro de fornecedor."""
    razao_social: str
    cnpj: str
    email: str
    segmento: str


class StatusVerificacao(str, Enum):
    """Status de cada etapa de verificação: OK, PENDENTE ou BLOQUEADO."""
    OK = "OK"
    PENDENTE = "PENDENTE"
    BLOQUEADO = "BLOQUEADO"


@dataclass
class ResultadoVerificacao:
    """Resultado de cada etapa da validação."""
    etapa: str
    status: StatusVerificacao
    detalhe: str


class DecisaoFinal(str, Enum):
    """Decisão final do processo de onboarding:
    APROVADO, CONDICIONAL ou REJEITADO."""
    APROVADO = "APROVADO"
    CONDICIONAL = "CONDICIONAL"
    REJEITADO = "REJEITADO"


@dataclass
class ParecerCadastro:
    """Parecer final do pipeline de onboarding."""
    cnpj: str
    razao_social: str
    decisao: DecisaoFinal
    verificacoes: list[ResultadoVerificacao]
    observacoes: list[str]
    parecer_texto: str


# ============================================================
# 3. CONSULTAS SIMULADAS
# ============================================================
# Em produção, cada consulta chama uma API ou scraper:
# - Receita Federal: https://www.receitaws.com.br/v1/cnpj/
# - CEIS/CNEP: Portal da Transparência API
# - CND Federal: https://solucoes.receita.fazenda.gov.br
# - FGTS: Caixa Econômica Federal
# Aqui simulamos com dicionários de exemplo.
# ============================================================

_SITUACAO_RF: dict[str, dict] = {
    "11222333000181": {
        "situacao": "ATIVA",
        "data_abertura": "2010-05-15",
        "atividade": "Desenvolvimento de software",
    },
    "22333444000195": {
        "situacao": "BAIXADA",
        "data_baixa": "2023-01-10",
    },
    "33444555000100": {
        "situacao": "INAPTA",
        "motivo": "Omissão de declarações",
    },
}

_LISTA_SANCOES: set[str] = {
    "33444555000100",   # CEIS
    "44555666000187",   # CNEP
}

_CERTIDOES: dict[str, dict] = {
    "11222333000181": {
        "cnd_federal": date(2026, 9, 30),
        "fgts":        date(2026, 6, 15),
        "trabalhista": date(2026, 3, 1),   # vencida
    },
}


def consultar_receita_federal(
    cnpj: str,
) -> dict:
    """Simula consulta à Receita Federal."""
    numeros = re.sub(r"\D", "", cnpj)
    return _SITUACAO_RF.get(
        numeros,
        {"situacao": "ATIVA", "data_abertura": "2015-01-01"},
    )


def consultar_lista_sancoes(cnpj: str) -> bool:
    """Retorna True se encontrado em lista de sanções."""
    numeros = re.sub(r"\D", "", cnpj)
    return numeros in _LISTA_SANCOES


def consultar_certidoes(
    cnpj: str,
) -> dict[str, date]:
    """Retorna datas de validade das certidões."""
    numeros = re.sub(r"\D", "", cnpj)
    return _CERTIDOES.get(
        numeros,
        {
            "cnd_federal": date(2026, 12, 31),
            "fgts":        date(2026, 12, 31),
            "trabalhista": date(2026, 12, 31),
        },
    )


# ============================================================
# 4. MOTOR DE VALIDAÇÃO
# ============================================================

class MotorOnboarding:
    """
    Executa o pipeline de validação sequencial.
    Para imediatamente se uma etapa crítica falha.
    """

    def avaliar(
        self, solicitacao: SolicitacaoCadastro
    ) -> ParecerCadastro:
        """Avalia a solicitação de cadastro e retorna um parecer final."""
        verificacoes: list[ResultadoVerificacao] = []
        observacoes: list[str] = []

        # --- Etapa 1: formato CNPJ ---
        if validar_cnpj(solicitacao.cnpj):
            verificacoes.append(ResultadoVerificacao(
                "Formato CNPJ",
                StatusVerificacao.OK,
                "Dígitos verificadores válidos",
            ))
        else:
            verificacoes.append(ResultadoVerificacao(
                "Formato CNPJ",
                StatusVerificacao.BLOQUEADO,
                "CNPJ inválido",
            ))
            return self._parecer(
                solicitacao, verificacoes, observacoes,
                DecisaoFinal.REJEITADO,
            )

        # --- Etapa 2: Receita Federal ---
        rf = consultar_receita_federal(solicitacao.cnpj)
        situacao = rf.get("situacao", "DESCONHECIDA")
        if situacao == "ATIVA":
            verificacoes.append(ResultadoVerificacao(
                "Receita Federal",
                StatusVerificacao.OK,
                f"Situação: {situacao}",
            ))
        else:
            verificacoes.append(ResultadoVerificacao(
                "Receita Federal",
                StatusVerificacao.BLOQUEADO,
                f"Situação: {situacao}",
            ))
            return self._parecer(
                solicitacao, verificacoes, observacoes,
                DecisaoFinal.REJEITADO,
            )

        # --- Etapa 3: Lista de sanções ---
        if consultar_lista_sancoes(solicitacao.cnpj):
            verificacoes.append(ResultadoVerificacao(
                "Lista de Sanções",
                StatusVerificacao.BLOQUEADO,
                "Fornecedor consta no CEIS/CNEP",
            ))
            return self._parecer(
                solicitacao, verificacoes, observacoes,
                DecisaoFinal.REJEITADO,
            )
        verificacoes.append(ResultadoVerificacao(
            "Lista de Sanções",
            StatusVerificacao.OK,
            "Não encontrado em listas de sanção",
        ))

        # --- Etapa 4: Certidões ---
        hoje = date.today()
        certidoes = consultar_certidoes(solicitacao.cnpj)
        tem_pendencia = False

        for nome, validade in certidoes.items():
            if validade < hoje:
                verificacoes.append(ResultadoVerificacao(
                    f"Certidão {nome.upper()}",
                    StatusVerificacao.PENDENTE,
                    f"Vencida em {validade.isoformat()}",
                ))
                observacoes.append(
                    f"Certidão {nome.upper()} vencida — "
                    "solicitar regularização em 30 dias"
                )
                tem_pendencia = True
            else:
                verificacoes.append(ResultadoVerificacao(
                    f"Certidão {nome.upper()}",
                    StatusVerificacao.OK,
                    f"Válida até {validade.isoformat()}",
                ))

        decisao = (
            DecisaoFinal.CONDICIONAL
            if tem_pendencia
            else DecisaoFinal.APROVADO
        )
        return self._parecer(
            solicitacao, verificacoes, observacoes, decisao
        )

    def _parecer(
        self,
        sol: SolicitacaoCadastro,
        verificacoes: list[ResultadoVerificacao],
        observacoes: list[str],
        decisao: DecisaoFinal,
    ) -> ParecerCadastro:
        texto = self._gerar_parecer_llm(
            sol, verificacoes, decisao, observacoes
        )
        return ParecerCadastro(
            cnpj=sol.cnpj,
            razao_social=sol.razao_social,
            decisao=decisao,
            verificacoes=verificacoes,
            observacoes=observacoes,
            parecer_texto=texto,
        )

    def _gerar_parecer_llm(
        self,
        sol: SolicitacaoCadastro,
        verificacoes: list[ResultadoVerificacao],
        decisao: DecisaoFinal,
        observacoes: list[str],
    ) -> str:
        """
        Gera parecer em linguagem natural.
        Em produção, envia contexto ao LLM.
        """
        ok = sum(
            1 for v in verificacoes
            if v.status == StatusVerificacao.OK
        )
        total = len(verificacoes)
        if decisao == DecisaoFinal.APROVADO:
            base = (
                f"O fornecedor {sol.razao_social} foi "
                f"APROVADO para cadastro. Todas as "
                f"{total} verificações passaram sem "
                f"restrições."
            )
        elif decisao == DecisaoFinal.CONDICIONAL:
            pend = total - ok
            base = (
                f"O fornecedor {sol.razao_social} pode ser "
                f"cadastrado de forma CONDICIONAL. "
                f"{ok}/{total} verificações passaram, com "
                f"{pend} pendência(s) que devem ser "
                f"regularizadas em até 30 dias."
            )
        else:
            bloq = next(
                (v for v in verificacoes
                 if v.status == StatusVerificacao.BLOQUEADO),
                None,
            )
            motivo = bloq.detalhe if bloq else "restrição"
            base = (
                f"O cadastro de {sol.razao_social} foi "
                f"REJEITADO. Motivo: {motivo}."
            )
        if observacoes:
            base += " Observações: " + "; ".join(observacoes)
        return base


# ============================================================
# 5. DEMO
# ============================================================

def demo_onboarding_fornecedores() -> None:
    """Demonstration do pipeline de onboarding automatizado de fornecedores."""
    console.print(
        Panel(
            "[bold]Módulo 35 — Onboarding Automatizado "
            "de Fornecedores[/]\n"
            "Pipeline multi-etapa: CNPJ → Receita Federal "
            "→ Sanções → Certidões → Decisão",
            style="bold blue",
        )
    )

    motor = MotorOnboarding()

    candidatos = [
        SolicitacaoCadastro(
            razao_social="Fornecedor Alpha Ltda",
            cnpj="11.222.333/0001-81",
            email="alpha@alpha.com.br",
            segmento="TI",
        ),
        SolicitacaoCadastro(
            razao_social="Beta Comércio Baixado",
            cnpj="22.333.444/0001-95",
            email="beta@beta.com.br",
            segmento="Logística",
        ),
        SolicitacaoCadastro(
            razao_social="Gamma Inapta ME",
            cnpj="33.444.555/0001-00",
            email="gamma@gamma.com.br",
            segmento="Consultoria",
        ),
        SolicitacaoCadastro(
            razao_social="Fornecedor CNPJ Inválido",
            cnpj="00.000.000/0000-00",
            email="inv@inv.com",
            segmento="Outros",
        ),
    ]

    cores_decisao = {
        DecisaoFinal.APROVADO:    "[green]",
        DecisaoFinal.CONDICIONAL: "[yellow]",
        DecisaoFinal.REJEITADO:   "[red]",
    }

    for candidato in candidatos:
        console.rule(
            f"[cyan]{candidato.razao_social}"
        )
        parecer = motor.avaliar(candidato)

        # Tabela de verificações
        tabela = Table(
            header_style="bold magenta",
            show_header=True,
        )
        tabela.add_column("Etapa")
        tabela.add_column("Status")
        tabela.add_column("Detalhe")
        cores_status = {
            StatusVerificacao.OK:       "[green]OK[/]",
            StatusVerificacao.PENDENTE: "[yellow]PENDENTE[/]",
            StatusVerificacao.BLOQUEADO: "[red]BLOQUEADO[/]",
        }
        for v in parecer.verificacoes:
            tabela.add_row(
                v.etapa,
                cores_status[v.status],
                v.detalhe,
            )
        console.print(tabela)

        cor = cores_decisao[parecer.decisao]
        console.print(
            f"\n  Decisão: {cor}[bold]"
            f"{parecer.decisao}[/][/]\n"
        )
        console.print(
            Panel(
                parecer.parecer_texto,
                title="Parecer",
                style="dim",
            )
        )


if __name__ == "__main__":
    demo_onboarding_fornecedores()
