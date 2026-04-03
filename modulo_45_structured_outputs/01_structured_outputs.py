"""
============================================================
MÓDULO 45.1 - STRUCTURED OUTPUTS COM INSTRUCTOR
============================================================
`instructor` é uma biblioteca que força o LLM a retornar
saída perfeitamente alinhada a um schema Pydantic.
Elimina regex frágil e parsing manual de texto.

COMO FUNCIONA:

  Sem instructor:
  ───────────────
  Prompt: "extraia o valor em JSON"
  Resposta: "O valor é R$ 1.250,00 conforme solicitado."
  → você parseia manualmente, trata exceções, regex...

  Com instructor:
  ───────────────
  Schema Pydantic → converte para "tool schema" da API
  LLM é forçada a chamar a "tool" com o schema correto
  instructor valida a resposta e retenta automaticamente

CICLO DE VALIDAÇÃO AUTOMÁTICA:

  schema Pydantic
       │
       ▼
  Tool schema JSON (gerado automaticamente)
       │
       ▼ (enviado para API)
  Resposta do LLM
       │
       ├── válida? → retorna objeto Python tipado
       │
       └── inválida? → ValidationError vira contexto
                        de retry automático

VANTAGENS SOBRE PARSING MANUAL:
  ✓ Erros de tipo corijidos automaticamente
     (string "1.250,00" → float 1250.0)
  ✓ Campos ausentes detectados com mensagem clara
  ✓ Validators complexos (CNPJ, datas, ranges)
  ✓ Retry com contexto: LLM sabe exatamente o que errou
  ✓ Schema é fonte única da verdade (code → doc → prompt)

SETUP REAL (comentado — requer chave de API):
  # pip install instructor
  # import instructor
  # from openai import OpenAI
  # client = instructor.from_openai(OpenAI())
  # boleto = client.chat.completions.create(
  #     model="gpt-4o-mini",
  #     response_model=BoletoExtraido,
  #     messages=[{"role": "user", "content": texto}],
  # )

Tópicos cobertos:
1. BoletoExtraido — schema com validators embutidos
2. SchemaInspector — geração automática de prompt
3. InstructorSimulado — retry com ValidationError
4. Comparativo: parsing manual vs instructor
5. Demo com 3 documentos e diferentes falhas
============================================================
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. SCHEMA DO BOLETO
# ============================================================

@dataclass
class ErroValidacao:
    """Representa um erro de validação em um campo específico."""
    campo: str
    mensagem: str
    valor_recebido: str


@dataclass
class BoletoExtraido:
    """
    Schema de extração de boleto.
    Em produção com instructor real: usar Pydantic BaseModel
    com @field_validator e Field(description=...).
    """
    banco: str
    valor: float
    vencimento: str   # formato YYYY-MM-DD
    cnpj: str

    @classmethod
    def from_dict(
        cls, dados: dict
    ) -> tuple[Optional["BoletoExtraido"], list[ErroValidacao]]:
        """
        Cria instância com validação.
        Retorna (objeto, erros). Se erros → objeto é None.
        """
        erros: list[ErroValidacao] = []

        # Banco
        banco = dados.get("banco", "")
        if not banco or len(str(banco)) < 3:
            erros.append(ErroValidacao(
                "banco",
                "Nome do banco obrigatório (min 3 chars)",
                str(banco),
            ))

        # Valor — aceita string com vírgula ou ponto
        valor_raw = dados.get("valor", 0)
        try:
            valor_str = str(valor_raw).replace(",", ".")
            valor = float(valor_str)
            if valor <= 0:
                raise ValueError("valor não positivo")
        except (ValueError, TypeError):
            erros.append(ErroValidacao(
                "valor",
                "Deve ser número positivo (ex: 1250.00)",
                str(valor_raw),
            ))
            valor = 0.0

        # Vencimento — aceita dd/mm/yyyy e converte
        venc_raw = str(dados.get("vencimento", ""))
        vencimento = _normalizar_data(venc_raw)
        if not vencimento:
            erros.append(ErroValidacao(
                "vencimento",
                "Formato inválido. Use YYYY-MM-DD ou "
                "DD/MM/YYYY",
                venc_raw,
            ))
            vencimento = ""

        # CNPJ
        cnpj_raw = str(dados.get("cnpj", ""))
        cnpj_digits = re.sub(r"\D", "", cnpj_raw)
        if len(cnpj_digits) != 14:
            erros.append(ErroValidacao(
                "cnpj",
                "CNPJ deve ter 14 dígitos",
                cnpj_raw,
            ))

        if erros:
            return None, erros
        return cls(
            banco=str(banco),
            valor=valor,
            vencimento=vencimento,
            cnpj=cnpj_raw,
        ), []


def _normalizar_data(texto: str) -> str:
    """Converte dd/mm/yyyy → YYYY-MM-DD."""
    # Já no formato correto
    if re.match(r"^\d{4}-\d{2}-\d{2}$", texto):
        return texto
    # Formato dd/mm/yyyy
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", texto)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return ""


# ============================================================
# 2. SCHEMA INSPECTOR — gera prompt a partir do schema
# ============================================================

_CAMPO_DESCRICOES = {
    "banco":      "string — nome completo do banco",
    "valor":      "number — valor em BRL (ex: 1250.00)",
    "vencimento": "string — data no formato YYYY-MM-DD",
    "cnpj":       "string — CNPJ no formato XX.XXX.XXX/XXXX-XX",
}


def gerar_prompt_schema() -> str:
    """
    Em instructor real, o schema Pydantic é convertido
    automaticamente para o formato de tool calling da API.
    Aqui mostramos o equivalente educacional.
    """
    campos_str = "\n".join(
        f'  "{c}": {d}'
        for c, d in _CAMPO_DESCRICOES.items()
    )
    return (
        "Extraia os dados do boleto e retorne SOMENTE JSON "
        "com a estrutura:\n"
        "{\n"
        f"{campos_str}\n"
        "}"
    )


# ============================================================
# 3. RESPOSTAS SIMULADAS DO LLM — diferentes falhas
# ============================================================

# Cada documento tem até 3 tentativas de resposta.
# Tentativa 0: resposta com alguma falha
# Tentativa 1: resposta melhorada após erro apontado
# Tentativa 2: resposta correta
_RESPOSTAS: dict[str, list[str]] = {
    "doc_01": [
        # Tentativa 1: resposta direta correta
        json.dumps({
            "banco": "Banco do Brasil",
            "valor": 1250.00,
            "vencimento": "2026-05-10",
            "cnpj": "12.345.678/0001-99",
        }),
    ],
    "doc_02": [
        # Tentativa 1: valor como string com vírgula
        json.dumps({
            "banco": "Bradesco",
            "valor": "890,50",
            "vencimento": "15/06/2026",
            "cnpj": "98.765.432/0001-11",
        }),
        # Tentativa 2: corrigido após erro
        json.dumps({
            "banco": "Bradesco",
            "valor": 890.50,
            "vencimento": "2026-06-15",
            "cnpj": "98.765.432/0001-11",
        }),
    ],
    "doc_03": [
        # Tentativa 1: texto livre sem JSON
        "O boleto do Itaú é de R$ 3.200,00 com "
        "vencimento em 20/07/2026 para CNPJ "
        "11.222.333/0001-44.",
        # Tentativa 2: JSON sem campo banco
        json.dumps({
            "valor": 3200.00,
            "vencimento": "2026-07-20",
            "cnpj": "11.222.333/0001-44",
        }),
        # Tentativa 3: correto após dois erros
        json.dumps({
            "banco": "Itaú",
            "valor": 3200.00,
            "vencimento": "2026-07-20",
            "cnpj": "11.222.333/0001-44",
        }),
    ],
}


def _chamar_llm(doc_id: str, tentativa: int) -> str:
    respostas = _RESPOSTAS.get(doc_id, ['{}'])
    return respostas[min(tentativa, len(respostas) - 1)]


# ============================================================
# 4. INSTRUCTOR SIMULADO
# ============================================================

class InstructorSimulado:
    """
    Simula o comportamento da lib `instructor`:
    1. Chama o LLM com o schema embutido no prompt
    2. Tenta parsear e validar a resposta
    3. Se inválida, inclui os erros no retry prompt
    4. Repete até MAX_RETRIES ou sucesso
    """

    MAX_RETRIES = 3

    def extrair(
        self, doc_id: str, _texto_original: str
    ) -> tuple[Optional[BoletoExtraido], int, list[str]]:
        """
        Retorna (objeto, num_tentativas, log).
        """
        log: list[str] = []

        for tentativa in range(self.MAX_RETRIES):
            resposta_raw = _chamar_llm(doc_id, tentativa)

            # Tenta extrair JSON da resposta
            dados = self._extrair_json(resposta_raw)
            if dados is None:
                log.append(
                    f"  Tentativa {tentativa+1}: "
                    f"[red]✗ JSON não encontrado[/]"
                )
                continue

            objeto, erros = BoletoExtraido.from_dict(dados)

            if objeto is not None:
                log.append(
                    f"  Tentativa {tentativa+1}: "
                    f"[green]✓ válido[/]"
                )
                return objeto, tentativa + 1, log

            erros_str = "; ".join(
                f"{e.campo}: {e.mensagem}" for e in erros
            )
            log.append(
                f"  Tentativa {tentativa+1}: "
                f"[yellow]✗ erros=[/] {erros_str}"
            )

        return None, self.MAX_RETRIES, log

    @staticmethod
    def _extrair_json(texto: str) -> Optional[dict]:
        try:
            return json.loads(texto)
        except (json.JSONDecodeError, ValueError):
            pass
        start = texto.find("{")
        end = texto.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(texto[start:end])
            except (json.JSONDecodeError, ValueError):
                pass
        return None


# ============================================================
# 5. DEMO
# ============================================================

def demo_structured_outputs() -> None:
    """Executa a demonstração do uso de structured outputs com instructor."""
    console.print(
        Panel(
            "[bold]Módulo 45 — Structured Outputs com "
            "instructor[/]\n"
            "Schema Pydantic como fonte única da verdade: "
            "validação + retry automático",
            style="bold blue",
        )
    )

    # Mostra o prompt gerado automaticamente do schema
    console.rule("[yellow]Schema → Prompt (automático)")
    console.print(
        Panel(gerar_prompt_schema(), title="Prompt gerado")
    )

    # Processa 3 documentos
    console.rule("[yellow]Processamento com instructor")
    instructor = InstructorSimulado()

    docs = [
        ("doc_01", "Boleto BB R$ 1250,00 venc 10/05/2026"),
        ("doc_02", "Boleto Bradesco R$ 890,50 15/06/2026"),
        ("doc_03", "Boleto Itaú R$ 3.200 venc 20/07/2026"),
    ]

    tabela = Table(header_style="bold magenta")
    tabela.add_column("Doc")
    tabela.add_column("Tentativas", justify="center")
    tabela.add_column("Banco")
    tabela.add_column("Valor", justify="right")
    tabela.add_column("Vencimento")

    for doc_id, texto in docs:
        console.rule(f"[dim]{doc_id}: {texto}")
        obj, n, log = instructor.extrair(doc_id, texto)
        for linha in log:
            console.print(linha)

        if obj:
            tabela.add_row(
                doc_id,
                str(n),
                obj.banco,
                f"R$ {obj.valor:,.2f}",
                obj.vencimento,
            )
        else:
            tabela.add_row(
                doc_id, str(n),
                "[red]FALHA[/]", "—", "—",
            )

    console.rule("[yellow]Resumo")
    console.print(tabela)


if __name__ == "__main__":
    demo_structured_outputs()
