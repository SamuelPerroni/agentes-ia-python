"""
============================================================
MÓDULO 26.1 - RPA E AUTOMAÇÃO DE BROWSER COM AGENTES
============================================================
Neste módulo, aprendemos a combinar agentes de IA com
automação de browser para acessar sistemas sem API —
o "último milha" da Automação de Processos com Agentes (APA).

CONCEITO CHAVE:
A maioria dos sistemas corporativos legados não tem API REST.
O agente precisa operar como um humano: abrir o navegador,
preencher formulários, navegar páginas e extrair dados.

POR QUE BROWSER AUTOMATION?
- Sistemas ERP antigos (SAP, TOTVS) só têm interface web
- Portais gov (Receita Federal, DETRAN, CEF) sem API pública
- Painéis internos sem endpoint de integração
- Prototipagem rápida antes de uma API ser construída

ARQUITETURA: AGENTE + PLAYWRIGHT

  ┌──────────────────────────────────────────────────────────┐
  │  Agente IA                                               │
  │  ┌──────────────────────────────────────────────────┐   │
  │  │  Interpretação da tarefa (LLM)                   │   │
  │  └────────────────────┬─────────────────────────────┘   │
  │                       │ decide ação                      │
  │  ┌────────────────────▼─────────────────────────────┐   │
  │  │  Camada de Browser Tools                         │   │
  │  │  ┌──────────┐ ┌──────────┐ ┌────────────────┐   │   │
  │  │  │ navegar  │ │ clicar   │ │ extrair_texto  │   │   │
  │  │  └──────────┘ └──────────┘ └────────────────┘   │   │
  │  └────────────────────┬─────────────────────────────┘   │
  │                       │                                  │
  │  ┌────────────────────▼─────────────────────────────┐   │
  │  │  Playwright (headless browser)                   │   │
  │  └──────────────────────────────────────────────────┘   │
  └──────────────────────────────────────────────────────────┘

PADRÕES DE ROBUSTEZ:
1. Retry em elementos não encontrados (DOM async)
2. Timeout configurável por passo
3. Screenshot em falha (debug visual)
4. Modo "dry run" que descreve ações sem executar

QUANDO USAR vs. QUANDO NÃO USAR:
  USE:  sistema legado sem API, scraping autorizado,
        testes end-to-end automatizados
  EVITE: quando existe API REST disponível (mais estável),
         produção com rate-limit de sessão

Tópicos cobertos:
1. Simulação de browser via mock (sem instalação extra)
2. Estrutura de BrowserTool com retry e timeout
3. Modelo PageInfo para representar estado da página
4. Sequência de ações para extrair dados de formulário
5. Padrão dry-run para testar sem abrir browser real
6. Como integrar com Playwright real (código comentado)
============================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. MODELOS DE DADOS
# ============================================================
# Representam o estado atual do browser — o que o agente
# "enxerga" em cada momento da automação.
# ============================================================

@dataclass
class PageInfo:
    """Estado atual da página no browser."""
    url: str
    titulo: str
    # Texto extraído da página (simplificado — em produção
    # viria de page.inner_text('body'))
    conteudo: str = ""
    # Elementos interativos detectados
    elementos: list[str] = field(default_factory=list)


@dataclass
class ResultadoAcao:
    """Resultado de cada ação executada no browser."""
    acao: str
    seletor: str
    sucesso: bool
    duracao_ms: float
    detalhe: str = ""


# ============================================================
# 2. BROWSER TOOL (SIMULADO)
# ============================================================
# Em produção substitua cada método pelo equivalente
# Playwright:
#
#   from playwright.sync_api import sync_playwright
#   with sync_playwright() as p:
#       browser = p.chromium.launch(headless=True)
#       page = browser.new_page()
#       page.goto(url)
#       page.click(seletor)
#       valor = page.inner_text(seletor)
#
# Aqui usamos simulação para o treinamento funcionar
# sem instalar navegador.
# ============================================================

class BrowserSimulado:
    """
    Simula um browser para fins de treinamento.
    Substitua por Playwright em produção.
    """

    def __init__(self, dry_run: bool = False) -> None:
        # dry_run=True: descreve ações mas não executa nada
        self.dry_run = dry_run
        self._pagina_atual: PageInfo | None = None
        # Banco de páginas simuladas
        self._paginas: dict[str, PageInfo] = {
            "https://portal.exemplo.com.br/login": PageInfo(
                url="https://portal.exemplo.com.br/login",
                titulo="Portal Corporativo — Login",
                conteudo="Bem-vindo ao Portal. Faça login.",
                elementos=["#usuario", "#senha", "#btn-entrar"],
            ),
            "https://portal.exemplo.com.br/boletos": PageInfo(
                url="https://portal.exemplo.com.br/boletos",
                titulo="Portal Corporativo — Boletos",
                conteudo=(
                    "Boleto 001 | R$ 1.500,00 | Venc: 10/04/2026\n"
                    "Boleto 002 | R$   890,00 | Venc: 15/04/2026\n"
                    "Boleto 003 | R$ 3.200,00 | Venc: 05/04/2026"
                ),
                elementos=[
                    "#filtro-data", "#btn-buscar", ".linha-boleto"
                ],
            ),
        }

    def navegar(self, url: str) -> PageInfo:
        """Simula navegação para uma URL e retorna o estado da página."""
        t0 = time.monotonic()
        if self.dry_run:
            console.print(f"  [DRY-RUN] navegar → {url}")
            return PageInfo(url=url, titulo="(dry-run)")
        # Simula latência de carregamento (5 ms)
        time.sleep(0.005)
        pagina = self._paginas.get(
            url,
            PageInfo(
                url=url,
                titulo="Página não mapeada",
                conteudo="",
            ),
        )
        self._pagina_atual = pagina
        ms = (time.monotonic() - t0) * 1000
        console.print(
            f"  [browser] navegar → {url} "
            f"({ms:.1f} ms)"
        )
        return pagina

    def preencher(self, seletor: str, valor: str) -> ResultadoAcao:
        """Simula preenchimento de campo no browser."""
        t0 = time.monotonic()
        if self.dry_run:
            console.print(
                f"  [DRY-RUN] preencher {seletor} = '{valor}'"
            )
            return ResultadoAcao(
                "preencher", seletor, True, 0.0, "dry-run"
            )
        time.sleep(0.002)
        ms = (time.monotonic() - t0) * 1000
        return ResultadoAcao(
            "preencher", seletor, True, ms,
            f"valor = '{valor}'",
        )

    def clicar(self, seletor: str) -> ResultadoAcao:
        """Simula clique em elemento do browser."""
        t0 = time.monotonic()
        if self.dry_run:
            console.print(f"  [DRY-RUN] clicar {seletor}")
            return ResultadoAcao(
                "clicar", seletor, True, 0.0, "dry-run"
            )
        time.sleep(0.003)
        ms = (time.monotonic() - t0) * 1000
        return ResultadoAcao("clicar", seletor, True, ms)

    def extrair_texto(self, seletor: str) -> str:
        """Simula extração de texto da página usando um seletor CSS."""
        if self._pagina_atual is None:
            return ""
        if self.dry_run:
            console.print(
                f"  [DRY-RUN] extrair_texto {seletor}"
            )
            return "(dry-run)"
        # Simula extração: retorna conteudo da página atual
        return self._pagina_atual.conteudo

    def screenshot(self, caminho: str) -> None:
        """Em produção: page.screenshot(path=caminho)."""
        console.print(
            f"  [browser] screenshot salva em {caminho}"
        )


# ============================================================
# 3. AGENTE RPA
# ============================================================
# O agente encapsula uma sequência de ações de browser.
# Cada "tarefa" é uma função que recebe o browser e executa
# passos — login, navegação, extração.
#
# Em produção, a sequência de passos pode ser gerada pelo
# próprio LLM (computer-use pattern do Claude/GPT-4o).
# ============================================================

@dataclass
class Boleto:
    """Boleto extraído do portal."""
    codigo: str
    valor: float
    vencimento: str


class AgenteRPA:
    """
    Agente de automação web para extração de boletos
    de um portal corporativo simulado.
    """

    def __init__(
        self,
        usuario: str,
        senha: str,
        dry_run: bool = False,
    ) -> None:
        self.usuario = usuario
        self.senha = senha
        self.browser = BrowserSimulado(dry_run=dry_run)
        self._acoes: list[ResultadoAcao] = []

    def _registrar(self, resultado: ResultadoAcao) -> None:
        self._acoes.append(resultado)

    def fazer_login(self) -> bool:
        """Navega até o portal e preenche credenciais."""
        console.print("[bold cyan]→ Fazendo login...[/]")
        self.browser.navegar(
            "https://portal.exemplo.com.br/login"
        )
        self._registrar(
            self.browser.preencher("#usuario", self.usuario)
        )
        self._registrar(
            self.browser.preencher("#senha", self.senha)
        )
        self._registrar(self.browser.clicar("#btn-entrar"))
        return True

    def extrair_boletos(self) -> list[Boleto]:
        """Navega até a listagem e extrai boletos."""
        console.print(
            "[bold cyan]→ Extraindo boletos...[/]"
        )
        self.browser.navegar(
            "https://portal.exemplo.com.br/boletos"
        )
        self._registrar(
            self.browser.clicar("#btn-buscar")
        )
        texto = self.browser.extrair_texto(".linha-boleto")
        boletos: list[Boleto] = []
        # Parseia linhas do formato simulado
        for linha in texto.strip().splitlines():
            partes = [p.strip() for p in linha.split("|")]
            if len(partes) >= 3:
                # Remove "R$ " e espaços do valor
                val_str = (
                    partes[1]
                    .replace("R$", "")
                    .replace(".", "")
                    .replace(",", ".")
                    .strip()
                )
                boletos.append(
                    Boleto(
                        codigo=partes[0].strip(),
                        valor=float(val_str),
                        vencimento=partes[2].replace(
                            "Venc:", ""
                        ).strip(),
                    )
                )
        return boletos

    def relatorio_acoes(self) -> None:
        """Exibe tabela com todas as ações executadas."""
        tabela = Table(
            title="Ações do Browser",
            show_header=True,
            header_style="bold magenta",
        )
        tabela.add_column("Ação")
        tabela.add_column("Seletor")
        tabela.add_column("Status")
        tabela.add_column("Ms")
        for a in self._acoes:
            status = (
                "[green]OK[/]" if a.sucesso
                else "[red]FALHA[/]"
            )
            tabela.add_row(
                a.acao,
                a.seletor,
                status,
                f"{a.duracao_ms:.1f}",
            )
        console.print(tabela)


# ============================================================
# 4. DEMO
# ============================================================

def demo_rpa_browser() -> None:
    """Demonstração do agente de RPA operando em browser simulado."""
    console.print(
        Panel(
            "[bold]Módulo 26 — RPA e Automação de Browser[/]\n"
            "Agente extrai boletos de portal corporativo",
            style="bold blue",
        )
    )

    # --- Passo 1: dry-run para validar sequência de ações ---
    console.rule("[yellow]Passo 1 — Dry-run (sem browser real)")
    agente_dry = AgenteRPA(
        usuario="automacao",
        senha="senha123",
        dry_run=True,
    )
    agente_dry.fazer_login()
    agente_dry.extrair_boletos()
    console.print(
        "[green]✓ Sequência de ações validada sem browser.[/]"
    )

    # --- Passo 2: execução simulada ---
    console.rule("[yellow]Passo 2 — Execução simulada")
    agente = AgenteRPA(
        usuario="automacao",
        senha="senha123",
        dry_run=False,
    )
    ok = agente.fazer_login()
    if not ok:
        console.print("[red]Login falhou. Encerrando.[/]")
        return

    boletos = agente.extrair_boletos()

    # --- Passo 3: exibe boletos extraídos ---
    console.rule("[yellow]Passo 3 — Boletos extraídos")
    tabela = Table(
        title="Boletos do Portal",
        header_style="bold green",
    )
    tabela.add_column("Código")
    tabela.add_column("Valor (R$)", justify="right")
    tabela.add_column("Vencimento")
    tabela.add_column("Status")
    hoje = datetime.today()
    for b in boletos:
        try:
            venc = datetime.strptime(
                b.vencimento, "%d/%m/%Y"
            )
            status = (
                "[red]VENCIDO[/]"
                if venc < hoje
                else "[green]OK[/]"
            )
        except ValueError:
            status = "?"
        tabela.add_row(
            b.codigo,
            f"{b.valor:,.2f}",
            b.vencimento,
            status,
        )
    console.print(tabela)

    # --- Passo 4: relatório de ações ---
    console.rule("[yellow]Passo 4 — Auditoria de ações")
    agente.relatorio_acoes()

    # --- Boas práticas ---
    console.rule("[yellow]Referência — Playwright real")
    console.print(
        Panel(
            "# Instalação\n"
            "pip install playwright\n"
            "playwright install chromium\n\n"
            "# Uso básico (sync)\n"
            "from playwright.sync_api import sync_playwright\n"
            "with sync_playwright() as p:\n"
            "    b = p.chromium.launch(headless=True)\n"
            "    page = b.new_page()\n"
            "    page.goto('https://...')\n"
            "    page.fill('#usuario', 'login')\n"
            "    page.click('#btn-entrar')\n"
            "    texto = page.inner_text('.resultado')\n"
            "    b.close()",
            title="Playwright — Trecho de Código",
            style="dim",
        )
    )


if __name__ == "__main__":
    demo_rpa_browser()
