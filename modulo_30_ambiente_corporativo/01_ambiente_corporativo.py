"""
============================================================
MÓDULO 30.1 - AGENTE EM AMBIENTE CORPORATIVO
============================================================
Neste módulo, aprendemos a configurar agentes para operar
dentro de redes corporativas com restrições de segurança:
proxy HTTP, certificados SSL internos, autenticação
service-to-service e deploy on-premises.

CONCEITO CHAVE:
A maioria dos treinamentos assume conectividade direta com
a internet. Na realidade corporativa, toda saída passa por
proxy, certificados são emitidos por CA interna, e tokens
de serviço são renovados via OAuth2 client credentials.

DESAFIOS DO AMBIENTE CORPORATIVO:

  ┌─────────────────────────────────────────────────────────┐
  │  Agente                                                 │
  │     │                                                   │
  │     ▼                                                   │
  │  [ Proxy HTTP ]  ← filtra/inspeciona tráfego SSL        │
  │     │                                                   │
  │     ▼                                                   │
  │  [ Firewall ]    ← bloqueia domínios não autorizados    │
  │     │                                                   │
  │     ▼                                                   │
  │  [ Internet ]  → Groq API / Azure OpenAI               │
  └─────────────────────────────────────────────────────────┘

  Internamente:
  ┌─────────────────────────────────────────────────────────┐
  │  Agente ──▶ [ OAuth2 Token ] ──▶ API Interna           │
  │               (client credentials)                     │
  └─────────────────────────────────────────────────────────┘

SOLUÇÕES:
1. Proxy: requests Session com proxies dict + HTTPS_PROXY env
2. SSL:   requests Session com verify='/path/ca_bundle.pem'
3. OAuth2: client_credentials flow com cache de token
4. Retry: urllib3 Retry com backoff em falhas de rede
5. Secrets: nunca hardcode — use variáveis de ambiente

Tópicos cobertos:
1. Configuração de proxy HTTP/HTTPS no cliente HTTP
2. Verificação de CA interna (bundle de certificados)
3. OAuth2 Client Credentials Flow com cache de token
4. Timeout e retry com backoff exponencial
5. Cliente Groq adaptado para ambiente corporativo
6. Checklist de configuração para onboarding
============================================================
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. CONFIGURAÇÃO DE AMBIENTE
# ============================================================
# Leia SEMPRE de variáveis de ambiente — nunca hardcode
# credenciais ou endereços de proxy no código.
#
# Variáveis esperadas (definir em .env ou vault):
#   HTTP_PROXY   / HTTPS_PROXY  → endereço do proxy
#   REQUESTS_CA_BUNDLE          → caminho do CA bundle
#   OAUTH2_TOKEN_URL            → URL do servidor OAuth2
#   OAUTH2_CLIENT_ID            → client ID do serviço
#   OAUTH2_CLIENT_SECRET        → secret do serviço
#   API_INTERNA_URL             → URL da API corporativa
# ============================================================

@dataclass
class ConfigCorporativa:
    """
    Agrega todas as configurações do ambiente corporativo.
    Carregada das variáveis de ambiente — sem defaults
    para dados sensíveis.
    """
    # Proxy (opcional — None se conexão direta)
    http_proxy: Optional[str] = field(
        default_factory=lambda: os.getenv("HTTP_PROXY")
    )
    https_proxy: Optional[str] = field(
        default_factory=lambda: os.getenv("HTTPS_PROXY")
    )
    # Caminho para CA bundle da empresa (opcional)
    # None = usa CAs padrão do sistema
    ca_bundle: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "REQUESTS_CA_BUNDLE"
        )
    )
    # OAuth2 para APIs internas
    oauth2_token_url: Optional[str] = field(
        default_factory=lambda: os.getenv("OAUTH2_TOKEN_URL")
    )
    oauth2_client_id: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "OAUTH2_CLIENT_ID"
        )
    )
    oauth2_client_secret: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "OAUTH2_CLIENT_SECRET"
        )
    )
    # Timeouts (segundos)
    timeout_connect: float = 10.0
    timeout_read: float = 60.0
    # Retry
    max_retries: int = 3
    retry_backoff: float = 2.0

    def resumo(self) -> dict[str, str]:
        """Resumo sem dados sensíveis (para logging)."""
        return {
            "proxy_http":    "configurado" if self.http_proxy else "direto",
            "proxy_https":   "configurado" if self.https_proxy else "direto",
            "ca_bundle":     self.ca_bundle or "padrão do sistema",
            "oauth2_url":    self.oauth2_token_url or "não configurado",
            "timeout":       f"{self.timeout_connect}s connect / "
                             f"{self.timeout_read}s read",
            "max_retries":   str(self.max_retries),
        }


# ============================================================
# 2. CLIENTE HTTP CORPORATIVO
# ============================================================
# Encapsula proxy, SSL e retry em um único cliente
# reutilizável. Em produção use a biblioteca requests:
#
#   import requests
#   from requests.adapters import HTTPAdapter
#   from urllib3.util.retry import Retry
#
#   session = requests.Session()
#   session.proxies = {
#       "http":  config.http_proxy,
#       "https": config.https_proxy,
#   }
#   session.verify = config.ca_bundle or True
#   retry = Retry(
#       total=config.max_retries,
#       backoff_factor=config.retry_backoff,
#       status_forcelist=[429, 500, 502, 503, 504],
#   )
#   adapter = HTTPAdapter(max_retries=retry)
#   session.mount("http://",  adapter)
#   session.mount("https://", adapter)
# ============================================================

@dataclass
class RequisicaoSimulada:
    """Resultado de uma requisição HTTP simulada."""
    url: str
    status_code: int
    latencia_ms: float
    via_proxy: bool
    corpo: str = ""


class ClienteHTTPCorporativo:
    """
    Cliente HTTP que abstrai proxy, SSL e retry.
    Simulado para o treinamento — substitua por
    requests.Session em produção.
    """

    def __init__(self, config: ConfigCorporativa) -> None:
        self._config = config
        self._tentativas: list[dict[str, str]] = []

    def get(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
    ) -> RequisicaoSimulada:
        """Executa GET com retry e backoff."""
        for tentativa in range(1, self._config.max_retries + 1):
            resultado = self._executar_get(url, headers)
            self._tentativas.append({
                "url": url,
                "tentativa": str(tentativa),
                "status": str(resultado.status_code),
            })
            if resultado.status_code < 500:
                return resultado
            # Backoff exponencial antes de nova tentativa
            if tentativa < self._config.max_retries:
                espera = (
                    self._config.retry_backoff ** tentativa
                )
                console.print(
                    f"  [yellow]Tentativa {tentativa} falhou "
                    f"(HTTP {resultado.status_code}). "
                    f"Aguardando {espera:.1f}s...[/]"
                )
                time.sleep(espera * 0.01)  # reduzido p/ demo
        return resultado

    def _executar_get(
        self,
        url: str,
        _headers: Optional[dict[str, str]],
    ) -> RequisicaoSimulada:
        """Simula execução de requisição HTTP."""
        via_proxy = bool(self._config.https_proxy)
        # Simula latência: proxy adiciona ~30 ms
        latencia = 120.0 + (30.0 if via_proxy else 0.0)
        return RequisicaoSimulada(
            url=url,
            status_code=200,
            latencia_ms=latencia,
            via_proxy=via_proxy,
            corpo='{"status": "ok", "dados": []}',
        )

    def historico(self) -> list[dict[str, str]]:
        """Retorna histórico de tentativas (para diagnóstico)."""
        return list(self._tentativas)


# ============================================================
# 3. OAUTH2 CLIENT CREDENTIALS
# ============================================================
# Padrão para autenticação service-to-service:
# o agente (não um usuário humano) obtém um access_token
# usando client_id + client_secret.
#
# Em produção:
#   resp = requests.post(
#       config.oauth2_token_url,
#       data={
#           "grant_type":    "client_credentials",
#           "client_id":     config.oauth2_client_id,
#           "client_secret": config.oauth2_client_secret,
#           "scope":         "api.leitura",
#       }
#   )
#   token = resp.json()["access_token"]
#   expires_in = resp.json()["expires_in"]  # segundos
# ============================================================

@dataclass
class TokenCache:
    """Cache de access_token com controle de expiração."""
    token: str
    expira_em: datetime

    @property
    def expirado(self) -> bool:
        """Verifica se o token está expirado ou próximo da expiração."""
        # Margem de segurança: renova 60s antes do vencimento
        return datetime.now() >= (
            self.expira_em - timedelta(seconds=60)
        )


class GerenciadorOAuth2:
    """
    Gerencia o ciclo de vida do access_token OAuth2.
    Renova automaticamente quando próximo da expiração.
    """

    def __init__(self, config: ConfigCorporativa) -> None:
        self._config = config
        self._cache: Optional[TokenCache] = None
        self._renovacoes = 0

    def obter_token(self) -> str:
        """Retorna token válido, renovando se necessário."""
        if self._cache is None or self._cache.expirado:
            self._renovar()
        return self._cache.token  # type: ignore[union-attr]

    def _renovar(self) -> None:
        """Simula chamada ao servidor OAuth2."""
        if not self._config.oauth2_token_url:
            # Ambiente sem OAuth2 configurado → token mock
            token_simulado = (
                "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9."
                "mock_payload.mock_signature"
            )
            self._cache = TokenCache(
                token=token_simulado,
                expira_em=datetime.now() + timedelta(hours=1),
            )
            self._renovacoes += 1
            console.print(
                "  [dim]OAuth2: token mock gerado "
                f"(renovação #{self._renovacoes})[/]"
            )
            return

        # Em produção: chamada real ao token_url
        console.print(
            f"  [cyan]OAuth2: renovando token via "
            f"{self._config.oauth2_token_url}[/]"
        )
        # ... chamada requests aqui ...
        raise NotImplementedError(
            "Configure OAUTH2_TOKEN_URL para usar OAuth2 real."
        )

    @property
    def renovacoes_realizadas(self) -> int:
        """Retorna o número de renovações de token realizadas."""
        return self._renovacoes


# ============================================================
# 4. CLIENTE GROQ CORPORATIVO
# ============================================================
# Adapta o cliente Groq para operar via proxy e com
# autenticação OAuth2 para APIs internas adicionais.
# ============================================================

class AgenteClienteCorporativo:
    """
    Agente que opera em ambiente corporativo:
    proxy + SSL + OAuth2 + retry automático.
    """

    def __init__(self, config: ConfigCorporativa) -> None:
        self._config = config
        self._http = ClienteHTTPCorporativo(config)
        self._oauth2 = GerenciadorOAuth2(config)

    def chamar_api_interna(
        self, endpoint: str
    ) -> RequisicaoSimulada:
        """Chama API interna com token OAuth2."""
        token = self._oauth2.obter_token()
        headers = {"Authorization": f"Bearer {token[:20]}…"}
        return self._http.get(endpoint, headers=headers)

    def diagnosticar(self) -> None:
        """Exibe configuração atual (sem dados sensíveis)."""
        resumo = self._config.resumo()
        tabela = Table(
            title="Configuração Corporativa",
            header_style="bold magenta",
        )
        tabela.add_column("Parâmetro")
        tabela.add_column("Valor")
        for k, v in resumo.items():
            tabela.add_row(k, v)
        console.print(tabela)


# ============================================================
# 5. DEMO
# ============================================================

def demo_ambiente_corporativo() -> None:
    """Demonstração do agente operando em ambiente corporativo."""
    console.print(
        Panel(
            "[bold]Módulo 30 — Agente em Ambiente "
            "Corporativo[/]\n"
            "Proxy, SSL, OAuth2 service-to-service e retry",
            style="bold blue",
        )
    )

    # --- Cenário 1: ambiente com proxy ---
    console.rule("[yellow]Cenário 1 — Com proxy corporativo")
    config_proxy = ConfigCorporativa(
        http_proxy="http://proxy.corp.empresa.com.br:8080",
        https_proxy="http://proxy.corp.empresa.com.br:8080",
        ca_bundle=None,
    )
    agente = AgenteClienteCorporativo(config_proxy)
    agente.diagnosticar()
    resultado = agente.chamar_api_interna(
        "https://api.interna.empresa.com.br/boletos"
    )
    console.print(
        f"\n  HTTP {resultado.status_code} | "
        f"{resultado.latencia_ms:.0f} ms | "
        f"via proxy: {resultado.via_proxy}"
    )

    # --- Cenário 2: sem proxy (conexão direta) ---
    console.rule("[yellow]Cenário 2 — Conexão direta")
    config_direto = ConfigCorporativa(
        http_proxy=None,
        https_proxy=None,
    )
    agente2 = AgenteClienteCorporativo(config_direto)
    resultado2 = agente2.chamar_api_interna(
        "https://api.groq.com/health"
    )
    console.print(
        f"\n  HTTP {resultado2.status_code} | "
        f"{resultado2.latencia_ms:.0f} ms | "
        f"via proxy: {resultado2.via_proxy}"
    )

    # --- Checklist ---
    console.print(
        Panel(
            "CHECKLIST DE ONBOARDING CORPORATIVO\n\n"
            "[ ] Verificar URLs liberadas no firewall\n"
            "    - api.groq.com (porta 443)\n"
            "    - openai.azure.com (se Azure OpenAI)\n\n"
            "[ ] Configurar variáveis de ambiente\n"
            "    HTTPS_PROXY=http://proxy:8080\n"
            "    REQUESTS_CA_BUNDLE=/etc/ssl/ca-bundle.crt\n\n"
            "[ ] Criar service account no OAuth2 / AD\n"
            "    Permissões mínimas (princípio de menor privilégio)\n\n"
            "[ ] Validar certificado SSL da CA interna\n"
            "    openssl s_client -connect proxy:8080\n\n"
            "[ ] Testar conectividade\n"
            "    curl -x $HTTPS_PROXY https://api.groq.com\n\n"
            "[ ] Guardar segredos em vault (não em .env)\n"
            "    Azure Key Vault / HashiCorp Vault / AWS SSM",
            title="Checklist Corporativo",
            style="dim",
        )
    )


if __name__ == "__main__":
    demo_ambiente_corporativo()
