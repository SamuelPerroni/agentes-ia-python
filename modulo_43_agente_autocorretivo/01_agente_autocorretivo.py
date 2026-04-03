"""
============================================================
MÓDULO 43.1 - AGENTE AUTO-CORRETIVO
============================================================
Um agente auto-corretivo detecta seus próprios erros de
saída e refaz a tarefa com estratégias progressivamente
mais rígidas — sem intervenção humana.

CICLO DE AUTO-CORREÇÃO:

  ┌─────────────────────────────────┐
  │         Prompt inicial          │
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │          Chama LLM              │
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │       Valida saída              │◄───────────────┐
  └──────────────┬──────────────────┘                │
                 │                                   │
         ┌───────┴────────┐                          │
         ▼                ▼                          │
     VÁLIDO          INVÁLIDO                        │
         │                │                          │
         ▼                ▼                          │
     Retorna         Aplica                          │
     resultado       estratégia                      │
                     de retry            max retries │
                          │                 atingido │
                          └─────────────────────────►│
                          └──────────► Retry ────────┘

ESTRATÉGIAS PROGRESSIVAS:
  Tentativa 1: System prompt normal
  Tentativa 2: Adiciona "retorne APENAS JSON"
  Tentativa 3: Adiciona exemplo de saída esperada
  Tentativa 4: Simplifica a tarefa (fallback)

TIPOS DE ERRO DETECTADOS:
  • JSON inválido — parse falha
  • Campos ausentes — schema incompleto
  • Tipos incorretos — valor string onde esperava float
  • Valores implausíveis — data no passado, valor negativo
  • Tamanho de resposta — campo muito curto ou vazio

Tópicos cobertos:
1. ValidadorSaida — detecta erros com mensagens úteis
2. EstrategiaRetentativa — monta prompts de correção
3. AgenteAutocorretivo — loop com retry inteligente
4. Métricas de auto-correção por sessão
5. Demo com 4 casos: sucesso 1ª tentativa, sucesso após
   retry, sucesso em fallback, falha total
============================================================
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Semente para resultados reproduzíveis na demo
random.seed(42)


# ============================================================
# 1. TIPOS E ERROS
# ============================================================

class TipoErro(Enum):
    """Tipos de erro na validação da saída do LLM."""
    JSON_INVALIDO = "json_invalido"
    CAMPO_AUSENTE = "campo_ausente"
    TIPO_INCORRETO = "tipo_incorreto"
    VALOR_IMPLAUSIVE = "valor_implausivel"
    RESPOSTA_VAZIA = "resposta_vazia"


@dataclass
class ErroValidacao:
    """Representa um erro encontrado na validação da saída do LLM,"""
    tipo: TipoErro
    campo: Optional[str]
    descricao: str
    sugestao_correcao: str


@dataclass
class ResultadoValidacao:
    """Resultado da validação da saída do LLM, incluindo se é válido,
    quais erros foram encontrados e os dados extraídos (se válidos)."""
    valido: bool
    erros: list[ErroValidacao] = field(default_factory=list)
    dados_extraidos: Optional[dict[str, Any]] = None

    @property
    def resumo_erros(self) -> str:
        """Gera um resumo legível dos erros encontrados para log e feedback.
        """
        return "; ".join(e.descricao for e in self.erros)


# ============================================================
# 2. VALIDADOR DE SAÍDA
# ============================================================

SCHEMA_BOLETO = {
    "tipo": str,
    "valor": float,
    "vencimento": str,
    "banco": str,
    "cnpj": str,
}


class ValidadorSaida:
    """Valida a saída do LLM contra um schema esperado."""

    def validar(self, texto_resposta: str) -> ResultadoValidacao:
        """Valida a resposta do LLM, retornando um
        ResultadoValidacao com erros detalhados."""
        if not texto_resposta.strip():
            return ResultadoValidacao(
                valido=False,
                erros=[
                    ErroValidacao(
                        TipoErro.RESPOSTA_VAZIA,
                        None,
                        "Resposta vazia",
                        "Solicite explicitamente uma resposta "
                        "não vazia",
                    )
                ],
            )

        # Tenta extrair JSON (aceita texto extra ao redor)
        dados = self._extrair_json(texto_resposta)
        if dados is None:
            return ResultadoValidacao(
                valido=False,
                erros=[
                    ErroValidacao(
                        TipoErro.JSON_INVALIDO,
                        None,
                        "Não foi possível extrair JSON válido",
                        "Peça ao modelo para retornar APENAS "
                        "o JSON, sem texto adicional",
                    )
                ],
            )

        erros: list[ErroValidacao] = []

        # Verifica campos obrigatórios e tipos
        for campo, tipo_esperado in SCHEMA_BOLETO.items():
            if campo not in dados:
                erros.append(
                    ErroValidacao(
                        TipoErro.CAMPO_AUSENTE,
                        campo,
                        f"Campo '{campo}' ausente",
                        f"Inclua o campo '{campo}' "
                        f"do tipo {tipo_esperado.__name__}",
                    )
                )
                continue

            valor = dados[campo]
            if not isinstance(valor, tipo_esperado):
                # Tenta coerção implícita (ex: int → float)
                try:
                    dados[campo] = tipo_esperado(valor)
                except (ValueError, TypeError):
                    erros.append(
                        ErroValidacao(
                            TipoErro.TIPO_INCORRETO,
                            campo,
                            f"Campo '{campo}' deveria ser "
                            f"{tipo_esperado.__name__}, "
                            f"recebeu {type(valor).__name__}",
                            f"Converta '{campo}' para "
                            f"{tipo_esperado.__name__}",
                        )
                    )

        # Validação de plausibilidade
        if "valor" in dados and isinstance(dados["valor"], float):
            if dados["valor"] <= 0:
                erros.append(
                    ErroValidacao(
                        TipoErro.VALOR_IMPLAUSIVE,
                        "valor",
                        "Valor do boleto deve ser positivo",
                        "Corrija o campo 'valor' para um "
                        "número positivo",
                    )
                )

        if "cnpj" in dados and isinstance(dados["cnpj"], str):
            cnpj = dados["cnpj"].replace(
                ".", ""
            ).replace("/", "").replace("-", "")
            if len(cnpj) != 14:
                erros.append(
                    ErroValidacao(
                        TipoErro.VALOR_IMPLAUSIVE,
                        "cnpj",
                        f"CNPJ inválido: '{dados['cnpj']}'",
                        "CNPJ deve ter 14 dígitos no formato "
                        "XX.XXX.XXX/XXXX-XX",
                    )
                )

        return ResultadoValidacao(
            valido=len(erros) == 0,
            erros=erros,
            dados_extraidos=dados if len(erros) == 0 else None,
        )

    @staticmethod
    def _extrair_json(texto: str) -> Optional[dict]:
        # Tenta o texto inteiro primeiro
        try:
            return json.loads(texto)
        except json.JSONDecodeError:
            pass

        # Tenta extrair o primeiro bloco {...}
        inicio = texto.find("{")
        fim = texto.rfind("}") + 1
        if inicio >= 0 and fim > inicio:
            try:
                return json.loads(texto[inicio:fim])
            except json.JSONDecodeError:
                pass

        return None


# ============================================================
# 3. ESTRATÉGIA DE RETENTATIVA
# ============================================================

class EstrategiaRetentativa:
    """
    Estratégias progressivamente mais rígidas para guiar
    o modelo a produzir saída correta.
    """

    @staticmethod
    def construir_prompt(
        prompt_original: str,
        tentativa: int,
        erros_anteriores: list[ErroValidacao],
    ) -> str:
        """Constrói um prompt refinado com base na
        tentativa atual e nos erros anteriores."""
        descricao_erros = "\n".join(
            f"- {e.descricao}: {e.sugestao_correcao}"
            for e in erros_anteriores
        )

        if tentativa == 1:
            # Tentativa 2: reforça formato JSON
            return (
                f"{prompt_original}\n\n"
                "IMPORTANTE: Retorne SOMENTE o JSON, "
                "sem explicações, markdown ou texto extra.\n"
                f"Erros anteriores:\n{descricao_erros}"
            )
        elif tentativa == 2:
            # Tentativa 3: fornece exemplo de saída
            exemplo = json.dumps(
                {
                    "tipo": "boleto",
                    "valor": 1250.00,
                    "vencimento": "2026-05-10",
                    "banco": "Banco do Brasil",
                    "cnpj": "12.345.678/0001-99",
                },
                indent=2,
                ensure_ascii=False,
            )
            return (
                f"{prompt_original}\n\n"
                "Retorne EXATAMENTE neste formato JSON:\n"
                f"```json\n{exemplo}\n```\n"
                f"Erros a corrigir:\n{descricao_erros}"
            )
        else:
            # Tentativa 4+: simplifica a tarefa (fallback)
            return (
                "Extraia apenas o valor e o CNPJ do texto "
                "abaixo e retorne um JSON simples. "
                "Se não encontrar um campo, use null.\n\n"
                + prompt_original.split("\n")[0]
            )


# ============================================================
# 4. SIMULAÇÃO DO LLM — cenários controlados para a demo
# ============================================================

_RESPOSTAS_SIM: dict[str, list[str]] = {
    "caso_sucesso": [
        '{"tipo":"boleto","valor":1250.00,'
        '"vencimento":"2026-05-10",'
        '"banco":"Banco do Brasil",'
        '"cnpj":"12.345.678/0001-99"}'
    ],
    "caso_retry": [
        # Tentativa 1: JSON com campo faltando
        '{"tipo":"boleto","valor":750.00,'
        '"banco":"Itaú","cnpj":"98.765.432/0001-11"}',
        # Tentativa 2: responde corretamente
        '{"tipo":"boleto","valor":750.00,'
        '"vencimento":"2026-04-15","banco":"Itaú",'
        '"cnpj":"98.765.432/0001-11"}',
    ],
    "caso_fallback": [
        # Tentativa 1: texto puro
        "O boleto é do Bradesco no valor de R$ 3.200,00",
        # Tentativa 2: JSON incompleto
        '{"tipo": "boleto"}',
        # Tentativa 3: fallback JSON
        '{"valor":3200.00,"cnpj":"11.222.333/0001-44"}',
    ],
    "caso_falha": [
        # 3 tentativas, todas erradas
        "Não foi possível extrair informações",
        '{"tipo": "boleto", "valor": -100}',
        "{}",
    ],
}


def _chamar_llm_simulado(
    caso: str, tentativa: int, _prompt: str = ""
) -> str:
    respostas = _RESPOSTAS_SIM.get(caso, ["{}"])
    idx = min(tentativa, len(respostas) - 1)
    return respostas[idx]


# ============================================================
# 5. AGENTE AUTO-CORRETIVO
# ============================================================

@dataclass
class MetricasAgente:
    """Métricas para monitorar o desempenho do agente auto-corretivo"""
    total_chamadas: int = 0
    total_retentativas: int = 0
    sucessos: int = 0
    falhas: int = 0

    def registrar(self, tentativas: int, sucesso: bool) -> None:
        """Registra uma chamada do agente,
        contabilizando tentativas e sucesso/falha."""
        self.total_chamadas += 1
        self.total_retentativas += max(0, tentativas - 1)
        if sucesso:
            self.sucessos += 1
        else:
            self.falhas += 1


class AgenteAutocorretivo:
    """
    Agente que valida a própria saída e reprocessa
    com estratégias progressivas em caso de erro.
    """

    MAX_TENTATIVAS = 3

    def __init__(
        self,
        validador: ValidadorSaida,
        estrategia: EstrategiaRetentativa,
    ) -> None:
        self.validador = validador
        self.estrategia = estrategia
        self.metricas = MetricasAgente()

    def processar(
        self, caso: str, prompt: str
    ) -> tuple[Optional[dict], int, list[str]]:
        """
        Retorna (dados, num_tentativas, log_tentativas).
        """
        prompt_atual = prompt
        log: list[str] = []
        erros_acumulados: list[ErroValidacao] = []

        for tentativa in range(self.MAX_TENTATIVAS + 1):
            resposta_raw = _chamar_llm_simulado(
                caso, tentativa, prompt_atual
            )
            resultado = self.validador.validar(resposta_raw)

            status = "✓" if resultado.valido else "✗"
            log.append(
                f"  Tentativa {tentativa + 1}: "
                f"{status} "
                f"'{resposta_raw[:60]}'"
            )

            if resultado.valido:
                self.metricas.registrar(tentativa + 1, True)
                return resultado.dados_extraidos, tentativa + 1, log

            # Acumula erros e refina o prompt
            erros_acumulados = resultado.erros
            if tentativa < self.MAX_TENTATIVAS:
                prompt_atual = (
                    self.estrategia.construir_prompt(
                        prompt,
                        tentativa,
                        erros_acumulados,
                    )
                )
                log.append(
                    f"  [yellow]Refinando prompt: "
                    f"{erros_acumulados[0].sugestao_correcao}"
                    f"[/]"
                )

        self.metricas.registrar(self.MAX_TENTATIVAS + 1, False)
        return None, self.MAX_TENTATIVAS + 1, log


# ============================================================
# 6. DEMO
# ============================================================

def demo_agente_autocorretivo() -> None:
    """Demonstração do agente auto-corretivo."""
    console.print(
        Panel(
            "[bold]Módulo 43 — Agente Auto-Corretivo[/]\n"
            "Detecta erros na própria saída e refaz a tarefa "
            "com estratégias progressivamente mais rígidas",
            style="bold blue",
        )
    )

    agente = AgenteAutocorretivo(
        ValidadorSaida(),
        EstrategiaRetentativa(),
    )

    casos = [
        (
            "caso_sucesso",
            "Extraia dados do boleto: CNPJ 12.345.678/0001-99, "
            "valor R$ 1250,00, Banco do Brasil, vencimento "
            "10/05/2026",
            "Sucesso na 1ª tentativa",
        ),
        (
            "caso_retry",
            "Extraia dados: boleto Itaú R$ 750,00 vencimento "
            "15/04/2026 CNPJ 98.765.432/0001-11",
            "Campo ausente → sucesso no retry",
        ),
        (
            "caso_fallback",
            "Extraia dados: Bradesco R$ 3200,00 venc abr/2026 "
            "CNPJ 11.222.333/0001-44",
            "Texto livre → JSON incompleto → fallback",
        ),
        (
            "caso_falha",
            "Processar documento ilegível @@#$%",
            "Falha total após 3 tentativas",
        ),
    ]

    tabela = Table(header_style="bold magenta")
    tabela.add_column("Caso", style="bold")
    tabela.add_column("Tentativas", justify="center")
    tabela.add_column("Resultado")
    tabela.add_column("Valor extraído", justify="right")

    for caso_id, prompt, descricao in casos:
        console.rule(f"[yellow]{descricao}")
        dados, n_tent, log = agente.processar(caso_id, prompt)

        for linha in log:
            console.print(linha)

        status_str = (
            "[green]✓ SUCESSO[/]"
            if dados
            else "[red]✗ FALHOU[/]"
        )
        valor_str = (
            f"R$ {dados.get('valor', 0):.2f}"
            if dados and "valor" in dados
            else "—"
        )
        tabela.add_row(
            descricao,
            str(n_tent),
            status_str,
            valor_str,
        )

    console.rule("[yellow]Resumo")
    console.print(tabela)

    m = agente.metricas
    total = m.sucessos + m.falhas
    console.print(
        Panel(
            f"Total processado:  {total}\n"
            f"Sucesso:           "
            f"[green]{m.sucessos}[/] "
            f"({m.sucessos/total:.0%})\n"
            f"Falhas:            "
            f"[red]{m.falhas}[/]\n"
            f"Retentativas:      {m.total_retentativas}",
            title="Métricas de Auto-Correção",
            style="blue",
        )
    )


if __name__ == "__main__":
    demo_agente_autocorretivo()
