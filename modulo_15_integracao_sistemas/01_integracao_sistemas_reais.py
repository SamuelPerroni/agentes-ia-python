"""
============================================================
MÓDULO 15.1 - INTEGRAÇÃO COM SISTEMAS EXTERNOS REAIS
============================================================
Neste módulo, aprendemos a conectar o agente ao mundo corporativo:
PDFs, APIs com autenticação OAuth2 e automação de interfaces web.

CONCEITO CHAVE:
Agentes de automação precisam CONSUMIR dados de sistemas que não
foram projetados para serem consumidos por IA: PDFs escaneados,
portais bancários sem API, sistemas legados com autenticação
proprietária. A integração é onde a maior parte do trabalho
real acontece.

POR QUE INTEGRAÇÃO É O MAIOR DESAFIO DA APA?
- LLMs só processam TEXTO — qualquer outro formato precisa ser
  convertido antes (PDF → texto, imagem → texto, HTML → texto)
- Sistemas corporativos exigem autenticação: OAuth2, certificados,
  cookies de sessão — o agente precisa gerenciar tokens
- APIs podem mudar: o agente precisa ser resiliente a mudanças
  de schema e versões de API
- Quando não há API: automação de interface (Playwright/Selenium)
  vira a única opção — é lenta, frágil, mas às vezes necessária

ANALOGIA:
Um analista humano que processa um boleto:
1. Abre o e-mail e baixa o PDF → agente: extrai texto do PDF
2. Loga no sistema bancário → agente: OAuth2 ou automação web
3. Consulta o saldo da conta → agente: chama API REST autenticada
4. Confirma os dados com o usuário → agente: HITL se necessário

FLUXO DE INTEGRAÇÃO DEMONSTRADO:

  ┌─────────────────────────────────────────────────────┐
  │              Pipeline de Integração                  │
  ├─────────────────────────────────────────────────────┤
  │                                                      │
  │  PDF/HTML/API  ──▶  Extrator  ──▶  Texto Limpo      │
  │                                         │            │
  │                                         ▼            │
  │                               ┌──────────────┐       │
  │  Token OAuth2  ──▶  Auth      │     LLM      │       │
  │  (cache 55min) ──▶  Header    │   (Processa) │       │
  │                               └──────┬───────┘       │
  │                                      │               │
  │                                      ▼               │
  │                               Resultado estruturado  │
  └─────────────────────────────────────────────────────┘

PADRÕES COBERTOS:
1. Extração de texto de PDF (com stub para demo sem dependência)
2. Gerenciamento de token OAuth2 com cache e renovação automática
3. Cliente HTTP genérico com autenticação injetada
4. Automação web conceitual (Playwright) para sistemas sem API

Tópicos cobertos:
1. Leitura e extração de texto de PDFs
2. Token OAuth2 com cache e expiração
3. Cliente REST autenticado (Authorization: Bearer)
4. Automação de interface web (Playwright conceitual)
5. Integração completa: extração → auth → LLM
============================================================
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


import pdfplumber
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Console do Rich para output formatado
console = Console()


# ============================================================
# 1. EXTRAÇÃO DE TEXTO DE PDF
# ============================================================
# PDFs são o formato mais comum de documentos corporativos.
# Para que o agente possa processar um boleto em PDF, precisamos
# primeiro extrair o texto puro (sem formatação).
#
# BIBLIOTECAS RECOMENDADAS (adicionar ao requirements.txt):
#   pdfplumber   → melhor para PDFs digitais (tabelas, layout)
#   PyMuPDF      → mais rápido, bom para PDFs escaneados
#   pdfminer.six → mais baixo nível, configurável
#
# IMPORTANTE: PDFs escaneados (imagens) precisam de OCR adicional
# (ex: Tesseract + pytesseract).
#
# Na demo abaixo, usamos um PDF simulado (string) para que o
# código rode sem instalar dependências extras.
# ============================================================

def extrair_texto_pdf(caminho_pdf: str) -> str:
    """
    Extrai texto puro de um arquivo PDF.

    Em produção, usa pdfplumber para PDFs digitais.
    Na demo, retorna texto simulado se o arquivo não existir.

    COMO USAR EM PRODUÇÃO:
        import pdfplumber
        with pdfplumber.open(caminho_pdf) as pdf:
            return "\\n".join(
                page.extract_text() or ""
                for page in pdf.pages
            )

    DICA:
    - PDFs escaneados retornam texto vazio → use OCR
    - Tabelas em PDFs perdem formatação → pós-processe com LLM
    - Senhas de PDF: pdfplumber aceita `password=` no open()

    Parâmetros:
    - caminho_pdf: caminho para o arquivo .pdf

    Retorna:
    - Texto extraído como string
    """
    # Tentativa real: usa pdfplumber se disponível
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            paginas = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(paginas).strip()
    except ImportError:
        # Sem pdfplumber: retorna texto simulado para a demo
        pass
    except FileNotFoundError:
        # Arquivo não existe: retorna texto simulado para a demo
        pass

    # STUB de demo — representa um boleto bancário típico em PDF
    # Em produção: substitua por pdfplumber.open(caminho_pdf)
    return """
BOLETO BANCÁRIO — BANCO BRADESCO S.A. (237)
================================================
Beneficiário: Empresa XYZ Serviços Ltda
CNPJ: 12.345.678/0001-99

Pagador: João da Silva
CPF: 123.456.789-00

Nosso Número: 0000002512345
Valor do Documento: R$ 1.250,00
Data de Vencimento: 05/04/2026
Data de Emissão: 01/04/2026

Linha Digitável:
23793.38128 60007.827136 97000.063305 1 10330000125000

Instruções ao Caixa:
- Após vencimento, cobrar multa de 2% + juros de 0,033% ao dia
- Não receber após 30 dias do vencimento
""".strip()


# ============================================================
# 2. GERENCIAMENTO DE TOKEN OAUTH2
# ============================================================
# APIs corporativas modernas usam OAuth2 para autenticação.
# O fluxo mais comum em automação é o "Client Credentials":
#   client_id + client_secret → token de acesso (expira em N min)
#
# PROBLEMA COMUM:
# Chamar /oauth/token a cada requisição é ineficiente e pode
# causar rate limiting. A solução é CACHEAR o token e renovar
# apenas quando estiver próximo do vencimento.
#
# FLUXO OAUTH2 CLIENT CREDENTIALS:
#
#   ┌──────────────────────────────────────────┐
#   │  Agente                                  │
#   │    │                                     │
#   │    ├── 1. Tem token válido? ─── SIM ──▶ usa token
#   │    │                                     │
#   │    └── NÃO ──▶ POST /oauth/token         │
#   │                  client_id               │
#   │                  client_secret           │
#   │                  grant_type=client_cred  │
#   │                       │                  │
#   │                       ▼                  │
#   │               { "access_token": "...",   │
#   │                 "expires_in": 3600 }     │
#   │                       │                  │
#   │                  guarda + calcula exp    │
#   └──────────────────────────────────────────┘
# ============================================================

@dataclass
class GerenciadorOAuth2:
    """
    Gerencia o ciclo de vida de tokens OAuth2.

    Cacheia o token e renova automaticamente quando está
    próximo da expiração (margem de 5 minutos por padrão).

    ATRIBUTOS:
    - endpoint_token: URL do endpoint OAuth2 (ex: /oauth/token)
    - client_id: identificador do cliente (da plataforma parceira)
    - client_secret: segredo do cliente (NUNCA em código, use .env)
    - _token_cache: token atual (None se não autenticado)
    - _expira_em: datetime de expiração do token atual
    """

    endpoint_token: str
    client_id: str
    client_secret: str
    _token_cache: str | None = field(default=None, init=False)
    _expira_em: datetime | None = field(default=None, init=False)

    def _token_esta_valido(self, margem_segundos: int = 300) -> bool:
        """
        Verifica se o token atual ainda é válido.

        Usa uma margem de 5 minutos (300s) para renovar
        ANTES da expiração real, evitando erros 401.

        MARGEM DE SEGURANÇA:
        Se o token expira às 15:00, renovamos às 14:55.
        Isso previne falhas em chamadas longas que começam
        com o token válido mas terminam após a expiração.
        """
        if self._token_cache is None or self._expira_em is None:
            return False
        agora = datetime.now()
        margem = timedelta(seconds=margem_segundos)
        return agora < (self._expira_em - margem)

    def obter_token(self) -> str:
        """
        Retorna um token de acesso válido, renovando se necessário.

        FLUXO:
        1. Se token em cache ainda é válido → retorna direto
        2. Se não → chama o endpoint OAuth2 para obter novo token
        3. Guarda token + calcula expiração → retorna token

        EM PRODUÇÃO (com requests):
            import requests
            resposta = requests.post(
                self.endpoint_token,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=10,
            )
            resposta.raise_for_status()
            dados = resposta.json()
            self._token_cache = dados["access_token"]
            self._expira_em = datetime.now() + timedelta(
                seconds=dados.get("expires_in", 3600)
            )
        """
        if self._token_esta_valido():
            console.print(
                "  [green]✓ Token em cache ainda válido[/green]"
            )
            return self._token_cache  # type: ignore[return-value]

        # Simula chamada ao endpoint OAuth2
        console.print(
            f"  [yellow]→ Buscando novo token em: "
            f"{self.endpoint_token}[/yellow]"
        )
        time.sleep(0.1)  # simula latência de rede

        # Em produção: parse da resposta real do servidor OAuth2
        self._token_cache = (
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiJhZ2VudGUiLCJleHAiOjk5OX0"
            ".assinatura_simulada"
        )
        # Token válido por 1 hora (3600s) — valor típico de OAuth2
        self._expira_em = datetime.now() + timedelta(seconds=3600)

        console.print(
            f"  [green]✓ Novo token obtido "
            f"(expira às {self._expira_em.strftime('%H:%M:%S')})"
            "[/green]"
        )
        return self._token_cache


# ============================================================
# 3. CLIENTE HTTP COM AUTENTICAÇÃO INJETADA
# ============================================================
# Com o GerenciadorOAuth2, criamos um cliente HTTP que injeta
# automaticamente o token em TODA requisição, sem o código de
# negócio precisar se preocupar com autenticação.
#
# PADRÃO: este cliente é o equivalente ao "httpx.AsyncClient"
# ou "requests.Session" configurado com um auth adapter.
#
# VANTAGEM:
# O restante do código (agente, tools) nunca vê tokens JWT.
# Ele só chama `cliente.get("/api/boleto/123")` e pronto.
# A autenticação acontece de forma transparente.
# ============================================================

class ClienteApiAutenticado:
    """
    Cliente HTTP que injeta token OAuth2 automaticamente.

    Encapsula a autenticação para que o código de negócio
    não precise gerenciar tokens diretamente.

    USO:
        cliente = ClienteApiAutenticado(
            base_url="https://api.banco.com.br",
            auth=GerenciadorOAuth2(endpoint, id, secret),
        )
        dados = cliente.get("/v1/boleto/123456")

    EM PRODUÇÃO (com httpx):
        import httpx
        resposta = httpx.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
    """

    def __init__(
        self,
        base_url: str,
        auth: GerenciadorOAuth2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = auth

    def get(self, endpoint: str) -> dict[str, Any]:
        """
        Executa GET autenticado.

        O token é obtido (ou renovado) automaticamente
        antes de cada requisição.

        Parâmetros:
        - endpoint: caminho relativo (ex: "/v1/boleto/123")

        Retorna:
        - Dicionário com a resposta JSON (simulado na demo)
        """
        token = self.auth.obter_token()
        url = f"{self.base_url}{endpoint}"
        console.print(f"  [cyan]GET {url}[/cyan]")
        console.print(
            f"  [dim]Authorization: Bearer "
            f"{token[:30]}...[/dim]"
        )

        # Simula resposta de uma API bancária
        return {
            "status": "ativo",
            "nosso_numero": "0000002512345",
            "valor": 1250.00,
            "vencimento": "2026-04-05",
            "beneficiario": "Empresa XYZ Serviços Ltda",
            "pagador": "João da Silva",
        }

    def post(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Executa POST autenticado com payload JSON.

        Parâmetros:
        - endpoint: caminho relativo
        - payload: dicionário que será serializado como JSON

        Retorna:
        - Dicionário com a resposta JSON (simulado na demo)
        """
        self.auth.obter_token()
        url = f"{self.base_url}{endpoint}"
        console.print(
            f"  [cyan]POST {url}[/cyan]"
        )
        console.print(
            f"  [dim]Payload: {json.dumps(payload, ensure_ascii=False)}"
            "[/dim]"
        )
        # Simula confirmação de pagamento
        return {"protocolo": "PAG-2026-004521", "status": "confirmado"}


# ============================================================
# 4. AUTOMAÇÃO WEB COM PLAYWRIGHT (CONCEITUAL)
# ============================================================
# Quando não existe API, a única saída é automatizar o navegador.
# Playwright é a ferramenta moderna recomendada (mais estável
# que Selenium, suporta async nativamente).
#
# INSTALAÇÃO:
#   pip install playwright
#   playwright install chromium
#
# CASOS DE USO TÍPICOS NA APA:
# - Portal bancário sem API              → login + download de extrato
# - Sistema legado ERP da empresa        → preencher formulário
# - Plataforma de terceiro com 2FA       → aguardar aprovação HITL
#
# AVISO:
# Automação web é frágil: qualquer mudança no HTML da página
# pode quebrar o script. Use XPath/CSS resilientes e considere
# HITL quando o site usa CAPTCHA ou 2FA.
#
# FLUXO TÍPICO COM PLAYWRIGHT:
#
#   async with async_playwright() as pw:
#       browser = await pw.chromium.launch(headless=True)
#       page = await browser.new_page()
#       await page.goto("https://portal.banco.com.br")
#       await page.fill("#usuario", "meu_usuario")
#       await page.fill("#senha", os.environ["PORTAL_SENHA"])
#       await page.click("#btn-login")
#       await page.wait_for_selector(".extrato-table")
#       html = await page.inner_text(".extrato-table")
#       await browser.close()
#   # Passa o HTML para o agente processar
# ============================================================

def extrair_via_automacao_web(url_portal: str) -> str:
    """
    Extrai dados de um portal web sem API usando automação.

    Na demo, retorna HTML simulado.
    Em produção, use Playwright com as instruções acima.

    DICAS DE PRODUÇÃO:
    - Sempre use headless=True em servidores
    - Configure timeouts: page.set_default_timeout(30_000)
    - Salve screenshot em erros: page.screenshot(path="erro.png")
    - Use page.wait_for_selector() em vez de time.sleep()
    - Para sites com 2FA: integre com fluxo HITL (módulo 5)

    Parâmetros:
    - url_portal: URL do portal a ser acessado

    Retorna:
    - Texto extraído da página (para envio ao agente)
    """
    console.print(
        f"  [yellow]→ Conectando ao portal: {url_portal}[/yellow]"
    )
    time.sleep(0.1)  # simula tempo de carregamento

    # Stub da demo — representa o que Playwright extrairia
    return """
Extrato de Pagamentos — Portal Banco XYZ
Período: 01/04/2026 a 05/04/2026

Boleto 001 | Empresa ABC | R$ 500,00   | Venc: 03/04/2026 | PENDENTE
Boleto 002 | Fornec. DEF | R$ 1.250,00 | Venc: 05/04/2026 | PENDENTE
Boleto 003 | Serv. GHI   | R$ 780,00   | Venc: 02/04/2026 | VENCIDO
""".strip()


# ============================================================
# DEMO COMPLETA — Pipeline de Integração
# ============================================================

def demo_integracao() -> None:
    """
    Demonstra o pipeline completo de integração com sistemas reais.

    ETAPAS:
    1. Extração de texto de PDF (stub se pdfplumber não instalado)
    2. Autenticação OAuth2 e obtenção de token com cache
    3. Chamada à API REST autenticada
    4. Automação web para portal sem API
    5. Consolidação dos dados para envio ao agente

    OBSERVE NO OUTPUT:
    - O token é obtido uma vez e cacheado para as próximas chamadas
    - A segunda chamada à API reutiliza o token sem nova requisição
    - O texto do PDF e o HTML do portal são normalizados antes do LLM

    EXERCÍCIO SUGERIDO:
    1. Instale pdfplumber e crie um PDF de boleto para testar
    2. Substitua o stub do OAuth2 por uma chamada real a uma API pública
    3. Instale Playwright e automatize um site simples de sua escolha
    """
    console.print(Panel.fit(
        "[bold]Pipeline de Integração com Sistemas Externos[/bold]\n"
        "PDF → OAuth2 → REST API → Automação Web",
        title="🔌 Módulo 15 — Integração",
        border_style="magenta",
    ))

    # ── ETAPA 1: Extração de PDF ──────────────────────────────
    console.print("\n[bold]Etapa 1: Extração de PDF[/bold]")
    texto_boleto = extrair_texto_pdf("boleto_abril.pdf")
    console.print(Panel(
        texto_boleto,
        title="📄 Texto extraído do PDF",
        border_style="dim",
    ))

    # ── ETAPA 2: OAuth2 — primeira chamada (sem cache) ────────
    console.print("\n[bold]Etapa 2: Autenticação OAuth2[/bold]")
    auth = GerenciadorOAuth2(
        endpoint_token="https://api.banco.com.br/oauth/token",
        client_id="agente_boletos_prod",
        client_secret="s3cr3t_from_env",  # em produção: os.getenv()
    )
    cliente = ClienteApiAutenticado(
        base_url="https://api.banco.com.br",
        auth=auth,
    )

    # ── ETAPA 3: Primeira chamada REST ────────────────────────
    console.print("\n[bold]Etapa 3: Consulta à API REST (1ª chamada)[/bold]")
    dados_boleto = cliente.get("/v1/boleto/0000002512345")

    # ── ETAPA 4: Segunda chamada REST ─────────────────────────
    # O token foi cacheado: NÃO faz nova requisição ao /oauth/token
    console.print(
        "\n[bold]Etapa 4: Segunda chamada (token cacheado)[/bold]"
    )
    confirmacao = cliente.post(
        "/v1/pagamento",
        payload={"nosso_numero": "0000002512345", "acao": "confirmar"},
    )

    # ── ETAPA 5: Automação web ────────────────────────────────
    console.print("\n[bold]Etapa 5: Automação do portal web[/bold]")
    extrato_html = extrair_via_automacao_web(
        "https://portal.banco.com.br/extrato"
    )

    # ── RESULTADO CONSOLIDADO ─────────────────────────────────
    console.print("\n[bold]Resultado consolidado[/bold]")
    tabela = Table(title="Dados capturados de múltiplas fontes")
    tabela.add_column("Fonte", style="bold")
    tabela.add_column("Dado")
    tabela.add_row("PDF", f"Valor: R$ {dados_boleto['valor']:.2f}")
    tabela.add_row("API REST", f"Status: {dados_boleto['status']}")
    tabela.add_row(
        "API REST (POST)",
        f"Protocolo: {confirmacao['protocolo']}",
    )
    tabela.add_row(
        "Portal Web", f"{extrato_html.split(chr(10), maxsplit=1)[0]}"
    )
    console.print(tabela)

    console.print(
        "\n[green]✅ Todos os dados coletados e prontos para o agente"
        "[/green]"
    )
    console.print("\n💡 [bold yellow]Dica:[/bold yellow]")
    console.print(
        "  Passe o texto_boleto + extrato_html como contexto na "
        "mensagem de sistema\n"
        "  da LLM para que ela possa raciocinar sobre os dados reais."
    )


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    demo_integracao()
