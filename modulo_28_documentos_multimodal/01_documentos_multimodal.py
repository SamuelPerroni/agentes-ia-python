"""
============================================================
MÓDULO 28.1 - PROCESSAMENTO DE DOCUMENTOS MULTI-MODAL
============================================================
Neste módulo, aprendemos a processar documentos PDF reais
(boletos, notas fiscais, contratos) combinando extração
de texto estruturado com análise por LLM.

CONCEITO CHAVE:
Documentos corporativos chegam como PDF, imagem ou arquivo
escaneado. O agente precisa "ler" esses arquivos antes de
processar — isso é a camada multi-modal de um pipeline APA.

FERRAMENTAS:
- pdfplumber: extração de texto e tabelas de PDFs nativos
- pytesseract + Pillow: OCR para PDFs escaneados (imagens)
- LLM: interpreta e estrutura o conteúdo extraído

PIPELINE COMPLETO:

  Arquivo PDF
       │
       ▼
  [ Detector de Tipo ]
  ├── PDF nativo      → pdfplumber → texto estruturado
  └── PDF escaneado   → OCR (tesseract) → texto raw
       │
       ▼
  [ Extrator de Metadados ]
  ├── número de páginas, tamanho, criador
       │
       ▼
  [ LLM ] → estrutura campos relevantes (valor, CNPJ, data)
       │
       ▼
  [ Validador Pydantic ] → garante tipos e formato
       │
       ▼
  Resultado estruturado + confiança

TIPOS DE DOCUMENTO:

  ┌──────────────┬──────────────────────────────────────┐
  │ Tipo         │ Campos extraídos                     │
  ├──────────────┼──────────────────────────────────────┤
  │ Boleto       │ valor, vencimento, cedente, CNPJ     │
  │ NF-e         │ número, CNPJ emit/dest, valor total  │
  │ Contrato     │ partes, vigência, valor, objeto      │
  │ Extrato      │ período, saldo inicial/final, mov.   │
  └──────────────┴──────────────────────────────────────┘

Tópicos cobertos:
1. Simulação de extração de PDF (sem arquivo real)
2. Detecção de tipo de documento por padrões
3. Extração estruturada via LLM (simulado)
4. Validação do resultado com dataclass tipada
5. Fluxo completo: PDF → texto → campos → saída
6. Como usar pdfplumber real (código comentado)
============================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. EXTRATOR DE PDF (SIMULADO / REAL)
# ============================================================
# Em produção, substitua PdfSimulado.extrair_texto por:
#
#   import pdfplumber
#   with pdfplumber.open(caminho_pdf) as pdf:
#       texto = "\n".join(
#           pagina.extract_text() or ""
#           for pagina in pdf.pages
#       )
#
# Para PDFs escaneados (imagens), use OCR:
#
#   from pdf2image import convert_from_path
#   import pytesseract
#   imagens = convert_from_path(caminho_pdf)
#   texto = "\n".join(
#       pytesseract.image_to_string(img, lang="por")
#       for img in imagens
#   )
# ============================================================

@dataclass
class PdfInfo:
    """Metadados básicos de um PDF."""
    nome_arquivo: str
    num_paginas: int
    tipo_detectado: str   # 'nativo' ou 'escaneado'
    texto: str


# Textos simulados que representam o conteúdo extraído de PDFs
_PDFS_SIMULADOS: dict[str, PdfInfo] = {
    "boleto_abril.pdf": PdfInfo(
        nome_arquivo="boleto_abril.pdf",
        num_paginas=1,
        tipo_detectado="nativo",
        texto=(
            "BANCO BRADESCO S.A.\n"
            "Boleto Bancário\n"
            "Cedente: Fornecedor XYZ Ltda\n"
            "CNPJ: 12.345.678/0001-90\n"
            "Sacado: Empresa ABC S.A.\n"
            "Valor do Documento: R$ 4.750,00\n"
            "Data de Vencimento: 15/04/2026\n"
            "Linha Digitável: 34191.09008 00000.060004 "
            "67000.100004 4 10010000475000\n"
        ),
    ),
    "nfe_001.pdf": PdfInfo(
        nome_arquivo="nfe_001.pdf",
        num_paginas=2,
        tipo_detectado="nativo",
        texto=(
            "NOTA FISCAL ELETRÔNICA\n"
            "Número: 000001234\n"
            "Data de Emissão: 01/04/2026\n"
            "Emitente: Fornecedor XYZ Ltda\n"
            "CNPJ Emitente: 12.345.678/0001-90\n"
            "Destinatário: Empresa ABC S.A.\n"
            "CNPJ Destinatário: 98.765.432/0001-11\n"
            "Valor Total da Nota: R$ 4.750,00\n"
            "Chave de Acesso: 3526 0104 1234 5678 0001 "
            "55 0010 0000 1234 5 6789 0123 4567 8901\n"
        ),
    ),
    "contrato_servicos.pdf": PdfInfo(
        nome_arquivo="contrato_servicos.pdf",
        num_paginas=8,
        tipo_detectado="nativo",
        texto=(
            "CONTRATO DE PRESTAÇÃO DE SERVIÇOS\n"
            "Contratante: Empresa ABC S.A.\n"
            "CNPJ: 98.765.432/0001-11\n"
            "Contratada: Fornecedor XYZ Ltda\n"
            "CNPJ: 12.345.678/0001-90\n"
            "Objeto: Prestação de serviços de TI e suporte\n"
            "Valor Mensal: R$ 15.000,00\n"
            "Vigência: 01/04/2026 a 31/03/2027\n"
            "Cláusula 1 — Das Obrigações...\n"
            "Cláusula 2 — Do Pagamento...\n"
        ),
    ),
}


def extrair_pdf_simulado(nome_arquivo: str) -> PdfInfo | None:
    """Simula extração de texto de um PDF."""
    return _PDFS_SIMULADOS.get(nome_arquivo)


# ============================================================
# 2. DETECTOR DE TIPO DE DOCUMENTO
# ============================================================

_PADROES_TIPO: dict[str, list[str]] = {
    "boleto": [
        r"boleto banc",
        r"linha digit",
        r"data de vencimento",
        r"cedente",
    ],
    "nfe": [
        r"nota fiscal eletr",
        r"chave de acesso",
        r"cnpj emitente",
        r"emiss",
    ],
    "contrato": [
        r"contrato de",
        r"contratante",
        r"contratada",
        r"vigência",
    ],
}


def detectar_tipo_documento(texto: str) -> str:
    """
    Classifica o documento pelo conteúdo do texto.
    Retorna: 'boleto', 'nfe', 'contrato' ou 'desconhecido'.
    """
    texto_lower = texto.lower()
    pontuacao: dict[str, int] = {}
    for tipo, padroes in _PADROES_TIPO.items():
        pontuacao[tipo] = sum(
            1 for p in padroes
            if re.search(p, texto_lower)
        )
    melhor = max(pontuacao, key=lambda k: pontuacao[k])
    return melhor if pontuacao[melhor] >= 2 else "desconhecido"


# ============================================================
# 3. EXTRATORES ESTRUTURADOS
# ============================================================
# Em produção, o LLM recebe o texto e retorna JSON.
# Aqui usamos regex para simular o resultado.
# ============================================================

@dataclass
class DadosBoleto:
    """Campos extraídos de um boleto."""
    cedente: str
    cnpj: str
    valor: float
    vencimento: str
    linha_digitavel: str
    confianca: float  # 0-1


@dataclass
class DadosNFe:
    """Campos extraídos de uma NF-e."""
    numero: str
    cnpj_emitente: str
    cnpj_destinatario: str
    valor_total: float
    data_emissao: str
    confianca: float


@dataclass
class DadosContrato:
    """Campos extraídos de um contrato."""
    contratante: str
    contratada: str
    objeto: str
    valor_mensal: float
    vigencia_inicio: str
    vigencia_fim: str
    confianca: float


def _extrair_valor(texto: str, padrao: str) -> str:
    """Extrai valor numérico após rótulo."""
    m = re.search(padrao, texto, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extrair_boleto(texto: str) -> DadosBoleto:
    """Extrai campos de um boleto."""
    cedente = _extrair_valor(texto, r"Cedente:\s*(.+)")
    cnpj = _extrair_valor(texto, r"CNPJ:\s*([\d./-]+)")
    venc = _extrair_valor(
        texto, r"Vencimento:\s*(\d{2}/\d{2}/\d{4})"
    )
    linha = _extrair_valor(
        texto, r"Linha Digit[aá]vel:\s*(.+)"
    )
    val_str = _extrair_valor(
        texto,
        r"Valor do Documento:\s*R\$\s*([\d.,]+)",
    )
    try:
        valor = float(
            val_str.replace(".", "").replace(",", ".")
        )
    except ValueError:
        valor = 0.0
    campos_ok = sum([
        bool(cedente), bool(cnpj), bool(venc),
        bool(linha), valor > 0,
    ])
    return DadosBoleto(
        cedente=cedente,
        cnpj=cnpj,
        valor=valor,
        vencimento=venc,
        linha_digitavel=linha,
        confianca=campos_ok / 5,
    )


def extrair_nfe(texto: str) -> DadosNFe:
    """Extrai campos de uma NF-e."""
    numero = _extrair_valor(texto, r"N[uú]mero:\s*(\d+)")
    # Pega todos os CNPJs e distingue emitente/destinatário
    cnpjs = re.findall(
        r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", texto
    )
    cnpj_emit = cnpjs[0] if len(cnpjs) > 0 else ""
    cnpj_dest = cnpjs[1] if len(cnpjs) > 1 else ""
    data_emissao = _extrair_valor(
        texto, r"Data de Emiss[aã]o:\s*(\d{2}/\d{2}/\d{4})"
    )
    val_str = _extrair_valor(
        texto,
        r"Valor Total da Nota:\s*R\$\s*([\d.,]+)",
    )
    try:
        valor_total = float(
            val_str.replace(".", "").replace(",", ".")
        )
    except ValueError:
        valor_total = 0.0
    campos_ok = sum([
        bool(numero), bool(cnpj_emit), bool(cnpj_dest),
        bool(data_emissao), valor_total > 0,
    ])
    return DadosNFe(
        numero=numero,
        cnpj_emitente=cnpj_emit,
        cnpj_destinatario=cnpj_dest,
        valor_total=valor_total,
        data_emissao=data_emissao,
        confianca=campos_ok / 5,
    )


def extrair_contrato(texto: str) -> DadosContrato:
    """Extrai campos de um contrato."""
    contratante = _extrair_valor(
        texto, r"Contratante:\s*(.+)"
    )
    contratada = _extrair_valor(
        texto, r"Contratada:\s*(.+)"
    )
    objeto = _extrair_valor(texto, r"Objeto:\s*(.+)")
    val_str = _extrair_valor(
        texto, r"Valor Mensal:\s*R\$\s*([\d.,]+)"
    )
    try:
        valor_mensal = float(
            val_str.replace(".", "").replace(",", ".")
        )
    except ValueError:
        valor_mensal = 0.0
    datas = re.findall(
        r"\d{2}/\d{2}/\d{4}", texto
    )
    vig_inicio = datas[0] if len(datas) > 0 else ""
    vig_fim = datas[1] if len(datas) > 1 else ""
    campos_ok = sum([
        bool(contratante), bool(contratada),
        bool(objeto), valor_mensal > 0,
        bool(vig_inicio), bool(vig_fim),
    ])
    return DadosContrato(
        contratante=contratante,
        contratada=contratada,
        objeto=objeto,
        valor_mensal=valor_mensal,
        vigencia_inicio=vig_inicio,
        vigencia_fim=vig_fim,
        confianca=campos_ok / 6,
    )


# ============================================================
# 4. AGENTE DE PROCESSAMENTO DE DOCUMENTOS
# ============================================================

class AgenteDocumentos:
    """
    Agente que processa arquivos PDF e retorna campos
    estruturados com nível de confiança.
    """

    def processar(self, nome_arquivo: str) -> None:
        """Processa um arquivo PDF e exibe os resultados."""
        console.rule(
            f"[cyan]Processando: {nome_arquivo}"
        )

        # 1. Extrai texto do PDF
        pdf = extrair_pdf_simulado(nome_arquivo)
        if pdf is None:
            console.print(
                f"[red]Arquivo não encontrado: "
                f"{nome_arquivo}[/]"
            )
            return

        console.print(
            f"  Páginas: {pdf.num_paginas} | "
            f"Tipo PDF: {pdf.tipo_detectado}"
        )

        # 2. Detecta tipo do documento
        tipo = detectar_tipo_documento(pdf.texto)
        console.print(f"  Tipo detectado: [bold]{tipo}[/]")

        # 3. Extrai campos conforme tipo
        if tipo == "boleto":
            dados = extrair_boleto(pdf.texto)
            self._exibir_boleto(dados)
        elif tipo == "nfe":
            dados = extrair_nfe(pdf.texto)
            self._exibir_nfe(dados)
        elif tipo == "contrato":
            dados = extrair_contrato(pdf.texto)
            self._exibir_contrato(dados)
        else:
            console.print(
                "[yellow]Tipo desconhecido — "
                "requer revisão humana.[/]"
            )

    def _exibir_boleto(self, d: DadosBoleto) -> None:
        tabela = Table(
            title="Boleto Extraído",
            header_style="bold green",
        )
        tabela.add_column("Campo")
        tabela.add_column("Valor")
        tabela.add_row("Cedente", d.cedente)
        tabela.add_row("CNPJ", d.cnpj)
        tabela.add_row("Valor", f"R$ {d.valor:,.2f}")
        tabela.add_row("Vencimento", d.vencimento)
        tabela.add_row(
            "Linha Digitável", d.linha_digitavel[:30] + "…"
        )
        tabela.add_row(
            "Confiança", f"{d.confianca:.0%}"
        )
        console.print(tabela)

    def _exibir_nfe(self, d: DadosNFe) -> None:
        tabela = Table(
            title="NF-e Extraída", header_style="bold green"
        )
        tabela.add_column("Campo")
        tabela.add_column("Valor")
        tabela.add_row("Número", d.numero)
        tabela.add_row("CNPJ Emitente", d.cnpj_emitente)
        tabela.add_row("CNPJ Destinatário", d.cnpj_destinatario)
        tabela.add_row("Valor Total", f"R$ {d.valor_total:,.2f}")
        tabela.add_row("Data Emissão", d.data_emissao)
        tabela.add_row("Confiança", f"{d.confianca:.0%}")
        console.print(tabela)

    def _exibir_contrato(self, d: DadosContrato) -> None:
        tabela = Table(
            title="Contrato Extraído",
            header_style="bold green",
        )
        tabela.add_column("Campo")
        tabela.add_column("Valor")
        tabela.add_row("Contratante", d.contratante)
        tabela.add_row("Contratada", d.contratada)
        tabela.add_row("Objeto", d.objeto[:50] + "…")
        tabela.add_row(
            "Valor Mensal", f"R$ {d.valor_mensal:,.2f}"
        )
        tabela.add_row(
            "Vigência",
            f"{d.vigencia_inicio} a {d.vigencia_fim}",
        )
        tabela.add_row("Confiança", f"{d.confianca:.0%}")
        console.print(tabela)


# ============================================================
# 5. DEMO
# ============================================================

def demo_documentos_multimodal() -> None:
    """Demonstração do agente de processamento de documentos."""
    console.print(
        Panel(
            "[bold]Módulo 28 — Processamento de Documentos "
            "Multi-modal[/]\n"
            "PDF → texto → campos estruturados com confiança",
            style="bold blue",
        )
    )

    agente = AgenteDocumentos()
    arquivos = [
        "boleto_abril.pdf",
        "nfe_001.pdf",
        "contrato_servicos.pdf",
    ]
    for arq in arquivos:
        agente.processar(arq)
        console.print()

    # Guia pdfplumber real
    console.print(
        Panel(
            "# Instalação\n"
            "pip install pdfplumber pillow pytesseract\n\n"
            "# PDF nativo (texto selecionável)\n"
            "import pdfplumber\n"
            "with pdfplumber.open('doc.pdf') as pdf:\n"
            "    texto = '\\n'.join(\n"
            "        p.extract_text() or ''\n"
            "        for p in pdf.pages\n"
            "    )\n\n"
            "# Tabelas em PDF\n"
            "with pdfplumber.open('doc.pdf') as pdf:\n"
            "    tabelas = pdf.pages[0].extract_tables()\n\n"
            "# PDF escaneado (OCR)\n"
            "from pdf2image import convert_from_path\n"
            "import pytesseract\n"
            "imgs = convert_from_path('scan.pdf')\n"
            "texto = '\\n'.join(\n"
            "    pytesseract.image_to_string(i, lang='por')\n"
            "    for i in imgs\n"
            ")",
            title="pdfplumber + pytesseract — Uso Real",
            style="dim",
        )
    )


if __name__ == "__main__":
    demo_documentos_multimodal()
