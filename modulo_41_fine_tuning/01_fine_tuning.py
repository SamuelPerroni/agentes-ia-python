"""
============================================================
MÓDULO 41.1 - FINE-TUNING PARA DOMÍNIO ESPECÍFICO
============================================================
Fine-tuning adapta um modelo pré-treinado a um domínio
específico usando exemplos do negócio. Para APA em
finanças, isso reduz alucinações em CNPJ, valores e
terminologia contábil.

QUANDO USAR vs NÃO USAR:

  ┌─────────────────────────────────────────────────────┐
  │  USE fine-tuning quando:                            │
  │  ✓ Prompt engineering já foi esgotado               │
  │  ✓ Há > 50 exemplos de qualidade                    │
  │  ✓ Formato de saída muito específico e repetitivo   │
  │  ✓ Velocidade/custo de inference é crítico          │
  │  ✓ Terminologia de domínio é muito especializada    │
  │                                                     │
  │  NÃO use fine-tuning quando:                        │
  │  ✗ Menos de 50 exemplos rotulado                    │
  │  ✗ Few-shot prompting já funciona bem               │
  │  ✗ O problema muda frequentemente                   │
  │  ✗ Orçamento não permite custos de training         │
  └─────────────────────────────────────────────────────┘

TÉCNICAS:

  Fine-tuning completo (Full)
  └── Todos os pesos ajustados — caro, máxima qualidade

  LoRA (Low-Rank Adaptation)
  └── Adiciona matrizes de baixo rank — 10-100× mais barato
  └── Recomendado para maioria dos casos de APA

  QLoRA (Quantized LoRA)
  └── LoRA + quantização 4-bit — roda em GPU menor

FORMATO DO DATASET:
  OpenAI / Groq (fine-tuning via API):
  {"messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."},
    {"role": "assistant", "content": "..."}
  ]}

CUSTO ESTIMADO (referência 2025):
  Dataset 100 exemplos × Llama 3.1 8B via Groq:
  Treino: ~$2–5 | Inference: 50% mais barato por token

Tópicos cobertos:
1. Análise de quando vale a pena fazer fine-tuning
2. Construção e validação de dataset JSONL
3. Avaliação de qualidade de exemplos
4. Estimativa de custo de treinamento
5. Comparação: base model vs fine-tuned (simulado)
6. Checklist de qualidade do dataset
============================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. ESTRUTURA DO DATASET
# ============================================================

@dataclass
class ExemploTreinamento:
    """
    Um exemplo de treinamento no formato chat.
    Compatível com OpenAI fine-tuning API e Groq.
    """
    system: str
    user: str
    assistant: str       # resposta esperada do modelo
    categoria: str       # boleto | cnpj | nfe | contrato
    fonte: str           # como foi gerado: humano | llm+revisão

    def to_jsonl(self) -> str:
        """Serializa no formato JSONL para upload."""
        obj = {
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
                {
                    "role": "assistant",
                    "content": self.assistant,
                },
            ]
        }
        return json.dumps(obj, ensure_ascii=False)

    @property
    def tokens_estimados(self) -> int:
        """Estimativa grosseira: 1 token ≈ 4 chars."""
        total_chars = (
            len(self.system)
            + len(self.user)
            + len(self.assistant)
        )
        return total_chars // 4


# ============================================================
# 2. QUALIDADE DO EXEMPLO
# ============================================================

class QualidadeExemplo(Enum):
    """Classificação da qualidade de um exemplo de treinamento."""
    EXCELENTE = "excellent"
    BOM = "good"
    FRACO = "weak"
    INVALIDO = "invalid"


def avaliar_qualidade(
    ex: ExemploTreinamento,
) -> tuple[QualidadeExemplo, list[str]]:
    """
    Avalia a qualidade de um exemplo de treinamento.
    Retorna (qualidade, lista_de_problemas).
    """
    problemas: list[str] = []

    if len(ex.user) < 20:
        problemas.append("Input muito curto (< 20 chars)")
    if len(ex.assistant) < 10:
        problemas.append("Output muito curto (< 10 chars)")
    if ex.assistant == ex.user:
        problemas.append("Output idêntico ao input")
    if len(ex.system) < 20:
        problemas.append("System prompt muito vago")
    if ex.tokens_estimados > 2048:
        problemas.append("Exemplo longo demais (> 2048 tokens)")

    if len(problemas) == 0:
        return QualidadeExemplo.EXCELENTE, []
    if len(problemas) == 1:
        return QualidadeExemplo.BOM, problemas
    if len(problemas) == 2:
        return QualidadeExemplo.FRACO, problemas
    return QualidadeExemplo.INVALIDO, problemas


# ============================================================
# 3. DATASET
# ============================================================

@dataclass
class Dataset:
    """Conjunto de exemplos de treinamento."""
    nome: str
    exemplos: list[ExemploTreinamento] = field(
        default_factory=list
    )

    def adicionar(self, ex: ExemploTreinamento) -> None:
        """Adiciona um exemplo ao dataset."""
        self.exemplos.append(ex)

    def validar(self) -> dict:
        """Valida todos os exemplos e retorna relatório."""
        contagens: dict[str, int] = {
            q.value: 0 for q in QualidadeExemplo
        }
        problemas_encontrados: list[str] = []
        for ex in self.exemplos:
            q, probs = avaliar_qualidade(ex)
            contagens[q.value] += 1
            problemas_encontrados.extend(probs)

        total = len(self.exemplos)
        validos = (
            contagens[QualidadeExemplo.EXCELENTE.value]
            + contagens[QualidadeExemplo.BOM.value]
        )
        return {
            "total": total,
            "validos": validos,
            "taxa_validade": (
                validos / total if total > 0 else 0.0
            ),
            "contagens": contagens,
            "pronto": total >= 50 and validos / total >= 0.8,
        }

    def tokens_totais(self) -> int:
        """Calcula o total de tokens estimados no dataset."""
        return sum(e.tokens_estimados for e in self.exemplos)

    def exportar_jsonl(self) -> str:
        """Exporta o dataset completo no formato JSONL."""
        return "\n".join(e.to_jsonl() for e in self.exemplos)


# ============================================================
# 4. ESTIMATIVA DE CUSTO
# ============================================================

def estimar_custo(
    dataset: Dataset,
    epochs: int = 3,
    custo_por_1k_tokens_treino: float = 0.008,  # USD
) -> dict:
    """
    Estimativa de custo de fine-tuning via API.
    Valores de referência para OpenAI gpt-4o-mini (2025).
    """
    tokens_treino = dataset.tokens_totais() * epochs
    custo_treino_usd = (
        tokens_treino / 1000 * custo_por_1k_tokens_treino
    )
    # Inference: modelo fine-tuned custa ~2× o base
    # mas com respostas menores por domínio → break-even
    return {
        "exemplos": len(dataset.exemplos),
        "tokens_treino": tokens_treino,
        "epochs": epochs,
        "custo_treino_usd": round(custo_treino_usd, 4),
        "custo_treino_brl": round(custo_treino_usd * 5.8, 2),
    }


# ============================================================
# 5. COMPARAÇÃO BASE VS FINE-TUNED (SIMULADA)
# ============================================================

_BASE_MODEL_RESPOSTAS = {
    "Qual o valor do boleto?": (
        "O boleto contém um valor monetário que deve ser pago."
    ),
    "Qual o CNPJ do fornecedor?": (
        "O CNPJ é um número de identificação de empresas."
    ),
    "Qual o vencimento?": (
        "O vencimento é a data limite para pagamento."
    ),
}

_FINE_TUNED_RESPOSTAS = {
    "Qual o valor do boleto?": (
        '{"valor": 1250.00, "moeda": "BRL"}'
    ),
    "Qual o CNPJ do fornecedor?": (
        '{"cnpj": "12.345.678/0001-99", "valido": true}'
    ),
    "Qual o vencimento?": (
        '{"vencimento": "2026-05-10", '
        '"formato": "YYYY-MM-DD"}'
    ),
}


# ============================================================
# 6. DEMO
# ============================================================

def _criar_dataset_exemplo() -> Dataset:
    """Cria dataset de demonstração com boletos."""
    ds = Dataset(nome="dataset-boletos-v1")
    system = (
        "Você é um extrator de dados de boletos bancários. "
        "Retorne sempre JSON com os campos: valor, "
        "vencimento, banco, beneficiario."
    )
    exemplos_raw = [
        (
            "Boleto Banco do Brasil R$ 1.250,00 "
            "venc 10/05/2026 beneficiário Alpha Ltda",
            '{"valor": 1250.00, "vencimento": "2026-05-10",'
            ' "banco": "Banco do Brasil",'
            ' "beneficiario": "Alpha Ltda"}',
            "boleto",
        ),
        (
            "Boleto Bradesco R$ 890,50 venc 15/06/2026 "
            "pagamento a Beta S.A.",
            '{"valor": 890.50, "vencimento": "2026-06-15",'
            ' "banco": "Bradesco",'
            ' "beneficiario": "Beta S.A."}',
            "boleto",
        ),
        (
            "Boleto Itaú R$ 3.200,00 venc 20/07/2026 "
            "Gamma ME fornecedor",
            '{"valor": 3200.00, "vencimento": "2026-07-20",'
            ' "banco": "Itaú",'
            ' "beneficiario": "Gamma ME"}',
            "boleto",
        ),
        # Exemplo fraco (curto)
        (
            "bol",
            "{}",
            "boleto",
        ),
        (
            "NF-e 4521 emitida por Delta Tech CNPJ "
            "12.345.678/0001-99 valor R$ 5.000,00",
            '{"tipo": "nfe", "numero": "4521",'
            ' "emitente": "Delta Tech",'
            ' "cnpj": "12.345.678/0001-99",'
            ' "valor": 5000.00}',
            "nfe",
        ),
    ]
    for user, assistant, categoria in exemplos_raw:
        ds.adicionar(ExemploTreinamento(
            system=system,
            user=user,
            assistant=assistant,
            categoria=categoria,
            fonte="humano",
        ))
    return ds


def demo_fine_tuning() -> None:
    """Demo completa de construção, validação e
    análise de dataset para fine-tuning."""
    console.print(
        Panel(
            "[bold]Módulo 41 — Fine-Tuning para Domínio "
            "Específico[/]\n"
            "Quando vale a pena, como preparar dataset "
            "e estimar custo",
            style="bold blue",
        )
    )

    # --- Dataset ---
    console.rule("[yellow]Construção do Dataset")
    ds = _criar_dataset_exemplo()
    relatorio = ds.validar()

    tabela_ds = Table(header_style="bold magenta")
    tabela_ds.add_column("Métrica")
    tabela_ds.add_column("Valor")
    tabela_ds.add_row("Total de exemplos", str(relatorio["total"]))
    tabela_ds.add_row(
        "Válidos",
        f"{relatorio['validos']} "
        f"({relatorio['taxa_validade']:.0%})",
    )
    for q, n in relatorio["contagens"].items():
        tabela_ds.add_row(f"  {q}", str(n))
    tabela_ds.add_row(
        "Pronto para treino?",
        "[green]Sim[/]" if relatorio["pronto"]
        else "[red]Não — adicione mais exemplos[/]",
    )
    console.print(tabela_ds)

    # --- Custo ---
    console.rule("[yellow]Estimativa de Custo")
    custo = estimar_custo(ds, epochs=3)
    console.print(
        Panel(
            f"Exemplos:        {custo['exemplos']}\n"
            f"Tokens treino:   {custo['tokens_treino']:,}\n"
            f"Epochs:          {custo['epochs']}\n"
            f"Custo USD:       ${custo['custo_treino_usd']}\n"
            f"Custo BRL:       R$ {custo['custo_treino_brl']}",
            title="Estimativa (gpt-4o-mini)",
        )
    )

    # --- Base vs Fine-tuned ---
    console.rule("[yellow]Base Model vs Fine-Tuned (simulação)")
    tabela_cmp = Table(header_style="bold magenta")
    tabela_cmp.add_column("Pergunta")
    tabela_cmp.add_column("Base Model")
    tabela_cmp.add_column("Fine-Tuned")
    for pergunta, resp_base in _BASE_MODEL_RESPOSTAS.items():
        tabela_cmp.add_row(
            pergunta,
            f"[yellow]{resp_base[:40]}[/]",
            f"[green]{_FINE_TUNED_RESPOSTAS[pergunta][:40]}[/]",
        )
    console.print(tabela_cmp)

    # --- Exemplo JSONL ---
    console.rule("[yellow]Amostra do Dataset (JSONL)")
    console.print(ds.exemplos[0].to_jsonl())


if __name__ == "__main__":
    demo_fine_tuning()
