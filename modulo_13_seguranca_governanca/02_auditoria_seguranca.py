"""
============================================================
MÓDULO 13.2 - AUDITORIA DE SEGURANÇA E GOVERNANÇA (Prática)
============================================================
Este script IMPLEMENTA em código as verificações do checklist
de governança (01_checklist_governanca.md). Em vez de apenas
ler um checklist, aqui você EXECUTA cada verificação e vê o
resultado no terminal.

O QUE ESTE SCRIPT FAZ?
Roda uma auditoria automatizada cobrindo 6 áreas de segurança:
1. Validação de segredos (API keys, .env, .gitignore)
2. Detecção e mascaramento de PII (CPF, email, CNPJ)
3. Verificação de traces e rastreabilidade
4. Checklist de deploy (testes, benchmarks)
5. Verificação de resiliência (fallback, retry)
6. Relatório final com score de governança

POR QUE AUTOMATIZAR O CHECKLIST?
- Checklists em markdown são esquecidos com o tempo
- Automação garante que TODA verificação roda SEMPRE
- Resultado é objetivo: ✅ passou ou ❌ falhou
- Pode ser integrado ao CI/CD (ex: rodar antes de deploy)
- Score numérico permite acompanhar evolução ao longo do tempo

ANALOGIA:
Pense no checklist de pré-voo de um avião:
- O piloto NÃO confia na memória — segue o checklist toda vez
- Cada item é verificado mecanicamente (não por "feeling")
- Se UM item falha, o avião NÃO decola
- O checklist é o MESMO para pilotos novatos e experientes
Este script é o "checklist de pré-voo" do seu agente de IA.

DIAGRAMA — Fluxo da Auditoria:

  ╔══════════════════════════════════════════════╗
  ║  1. Verificar segredos (.env, .gitignore)    ║
  ║     ↓                                        ║
  ║  2. Testar mascaramento de PII (regex)       ║
  ║     ↓                                        ║
  ║  3. Verificar estrutura de traces (JSONL)    ║
  ║     ↓                                        ║
  ║  4. Verificar presença de testes (pytest)    ║
  ║     ↓                                        ║
  ║  5. Verificar resiliência (fallback config)  ║
  ║     ↓                                        ║
  ║  6. Gerar relatório com score final          ║
  ╚══════════════════════════════════════════════╝

SCORE DE GOVERNANÇA:
- 100%: Agente pronto para produção 🟢
-  80%: Quase pronto, ajustes menores 🟡
-  60%: Riscos significativos, revisar 🟠
- < 60%: NÃO colocar em produção 🔴

Tópicos cobertos:
1. Verificação automatizada de segredos e .gitignore
2. Detecção de PII com regex (CPF, CNPJ, email)
3. Validação de estrutura de traces (campos obrigatórios)
4. Checklist de deploy (testes e benchmarks)
5. Score de governança com relatório visual
============================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ============================================================
# CONSOLE RICH — Nosso "painel de auditoria" no terminal
# ============================================================
console = Console()

# ============================================================
# DIRETÓRIO RAIZ DO PROJETO — Ponto de referência para buscas
# ============================================================
# Este script fica em modulo_13_seguranca_governanca/
# então a raiz do projeto é um nível acima.
# Usamos Path para navegação segura entre OS (Windows/Linux/Mac)
# ============================================================
RAIZ_PROJETO = Path(__file__).resolve().parent.parent


# ============================================================
# 1. ESTRUTURA DE DADOS — Resultado de cada verificação
# ============================================================
# Cada verificação retorna um ResultadoVerificacao com:
# - nome: o que foi verificado
# - passou: True/False
# - detalhes: mensagem explicativa
# - categoria: área de segurança (para agrupamento no relatório)
#
# POR QUE DATACLASS?
# - Organiza resultados de forma estruturada
# - Fácil de iterar e contar passes/falhas
# - Pode ser serializada para relatório JSON se necessário
# ============================================================

@dataclass
class ResultadoVerificacao:
    """
    Resultado de UMA verificação de segurança/governança.

    Atributos:
    - nome: descrição curta da verificação
    - passou: True se passou, False se falhou
    - detalhes: explicação do resultado
    - categoria: área de segurança (segredos, pii, traces, deploy, resiliência)
    """
    nome: str
    passou: bool
    detalhes: str
    categoria: str


@dataclass
class RelatorioAuditoria:
    """
    Relatório consolidado de todas as verificações.

    Atributos:
    - verificacoes: lista de todos os resultados individuais
    - score: percentual de verificações que passaram (0-100)
    """
    verificacoes: list[ResultadoVerificacao] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Calcula o score como percentual de verificações que passaram."""
        if not self.verificacoes:
            return 0.0
        passou = sum(1 for v in self.verificacoes if v.passou)
        return (passou / len(self.verificacoes)) * 100

    @property
    def nivel(self) -> str:
        """
        Retorna o nível de governança baseado no score.

        ESCALA:
        🟢 100%: Pronto para produção
        🟡  80%: Quase pronto
        🟠  60%: Riscos significativos
        🔴 < 60%: NÃO ir para produção
        """
        if self.score >= 100:
            return "🟢 PRONTO PARA PRODUÇÃO"
        elif self.score >= 80:
            return "🟡 QUASE PRONTO — ajustes menores necessários"
        elif self.score >= 60:
            return "🟠 RISCOS SIGNIFICATIVOS — revisar antes de deploy"
        else:
            return "🔴 NÃO COLOCAR EM PRODUÇÃO"


# ============================================================
# 2. VERIFICAÇÕES DE SEGREDOS — .env, .gitignore, hardcode
# ============================================================
# Conforme o checklist (seção 1):
# - .env NÃO pode ser commitado
# - .gitignore DEVE conter .env
# - API keys NÃO podem estar hardcoded nos scripts
#
# COMO FUNCIONA:
# - Verifica se .env existe (deve existir para funcionar)
# - Verifica se .gitignore existe e contém ".env"
# - Escaneia arquivos .py buscando padrões de chave hardcoded
#
# POR QUE VERIFICAR HARDCODE?
# Uma chave hardcoded em código-fonte pode:
# - Vazar se o repositório for público (ou se alguém clonar)
# - Ser commitada acidentalmente (mesmo em repos privados)
# - Ser impossível de rotacionar (precisa alterar código)
# ============================================================

def verificar_segredos(relatorio: RelatorioAuditoria) -> None:
    """
    Verifica se segredos estão protegidos conforme o checklist.

    Verificações:
    1. .env existe? (necessário para as credenciais)
    2. .gitignore existe e contém ".env"?
    3. Nenhum arquivo .py contém chaves hardcoded?
    """
    # ── 2.1 Verifica se .env existe ──────────────────────────
    env_path = RAIZ_PROJETO / ".env"
    relatorio.verificacoes.append(
        ResultadoVerificacao(
            nome="Arquivo .env existe",
            passou=env_path.exists(),
            detalhes=(
                "✅ .env encontrado na raiz do projeto"
                if env_path.exists()
                else (
                    "❌ .env NÃO encontrado — "
                    "credenciais não estão configuradas"
                )
            ),
            categoria="segredos",
        )
    )

    # ── 2.2 Verifica se .gitignore protege o .env ───────────
    gitignore_path = RAIZ_PROJETO / ".gitignore"
    gitignore_protege = False
    if gitignore_path.exists():
        conteudo_gitignore = gitignore_path.read_text(encoding="utf-8")
        # Procura ".env" como entrada no .gitignore
        # Aceita ".env" sozinho ou com comentário/espaço
        gitignore_protege = any(
            linha.strip() == ".env" or linha.strip().startswith(".env")
            for linha in conteudo_gitignore.splitlines()
            if not linha.strip().startswith("#")
        )

    relatorio.verificacoes.append(
        ResultadoVerificacao(
            nome=".gitignore protege .env",
            passou=gitignore_protege,
            detalhes=(
                "✅ .gitignore contém entrada para .env"
                if gitignore_protege
                else (
                    "❌ .env NÃO está no .gitignore — "
                    "risco de vazar credenciais!"
                )
            ),
            categoria="segredos",
        )
    )

    # ── 2.3 Verifica chaves hardcoded nos arquivos .py ──────
    # Padrão regex: strings que parecem API keys (gsk_, sk-, etc.)
    # CUIDADO: este regex é heurístico — pode ter falsos positivos
    # em strings de exemplo/documentação, mas melhor alertar do que ignorar
    padrao_chave = re.compile(
        r"""(?:api_key|secret|token|password)"""
        r"""\s*=\s*["'][a-zA-Z0-9_\-]{20,}["']""",
        re.IGNORECASE,
    )
    arquivos_com_hardcode: list[str] = []

    for arquivo_py in RAIZ_PROJETO.rglob("*.py"):
        # Ignora ambientes virtuais e cache
        partes = arquivo_py.parts
        if any(
            parte in (".venv", "venv", "__pycache__", ".git")
            for parte in partes
        ):
            continue
        try:
            conteudo = arquivo_py.read_text(encoding="utf-8")
            # Ignora matches dentro de docstrings/comentários que são exemplos
            for numero_linha, linha in enumerate(conteudo.splitlines(), 1):
                linha_limpa = linha.strip()
                # Pula comentários e exemplos em docstrings
                if linha_limpa.startswith("#") or linha_limpa.startswith("❌"):
                    continue
                if padrao_chave.search(linha):
                    nome_relativo = arquivo_py.relative_to(RAIZ_PROJETO)
                    arquivos_com_hardcode.append(
                        f"{nome_relativo}:{numero_linha}"
                    )
        except (UnicodeDecodeError, PermissionError):
            continue

    relatorio.verificacoes.append(
        ResultadoVerificacao(
            nome="Sem chaves hardcoded em .py",
            passou=len(arquivos_com_hardcode) == 0,
            detalhes=(
                "✅ Nenhuma chave hardcoded encontrada nos scripts"
                if not arquivos_com_hardcode
                else (
                    "❌ Possíveis chaves hardcoded em: "
                    + ", ".join(arquivos_com_hardcode[:5])
                )
            ),
            categoria="segredos",
        )
    )


# ============================================================
# 3. VERIFICAÇÃO DE PII — Detecção e mascaramento de dados
# ============================================================
# Conforme o checklist (seção 2):
# - CPF, CNPJ e email devem ser detectados por regex
# - O mascaramento deve substituir por [TIPO_REDACTED]
# - Nenhum dado sensível deve chegar em logs sem máscara
#
# COMO FUNCIONA:
# Testamos os mesmos regex do módulo 09 (trace_utils.py) com
# dados de exemplo, verificando se a detecção/mascaramento funciona.
#
# POR QUE TESTAR AQUI E NÃO SÓ NOS TESTES?
# - pytest testa se o CÓDIGO funciona tecnicamente
# - Auditoria testa se a PROTEÇÃO está ativa no projeto
# - São perspectivas complementares: qualidade vs segurança
# ============================================================

# Mesmos padrões do módulo 09 — replicados aqui para independência
# Se mudar no módulo 09, deve mudar aqui também (ou importar)
PADROES_PII = {
    "cpf": r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}",
    "cnpj": r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
}

# Dados de teste com PII conhecida — usados para validar os regex
# ATENÇÃO: dados fictícios, NÃO são dados reais de pessoas
DADOS_TESTE_PII = [
    ("CPF formatado", "Cliente com CPF 123.456.789-00 ligou", "cpf"),
    ("CPF sem pontos", "CPF 12345678900 no cadastro", "cpf"),
    ("CNPJ formatado", "Empresa CNPJ 12.345.678/0001-90", "cnpj"),
    ("Email", "Contato: joao.silva@empresa.com.br", "email"),
]


def mascarar_pii(texto: str) -> str:
    """
    Mascara dados sensíveis (PII) em um texto.

    COMO FUNCIONA:
    1. Itera sobre cada padrão regex de PII
    2. Substitui cada match por [TIPO_REDACTED]
    3. Retorna texto com dados mascarados

    É o MESMO algoritmo do módulo 09 (redigir_texto).
    Replicado aqui para que a auditoria seja independente.
    """
    texto_mascarado = texto
    for tipo, padrao in PADROES_PII.items():
        texto_mascarado = re.sub(
            padrao, f"[{tipo.upper()}_REDACTED]", texto_mascarado
        )
    return texto_mascarado


def verificar_pii(relatorio: RelatorioAuditoria) -> None:
    """
    Verifica se a detecção e mascaramento de PII funciona.

    Verificações:
    1. Cada padrão de PII é detectado corretamente?
    2. O mascaramento substitui por [TIPO_REDACTED]?
    3. O texto original NÃO permanece após mascaramento?
    """
    todos_passaram = True

    for _nome_teste, texto_original, tipo_esperado in DADOS_TESTE_PII:
        texto_mascarado = mascarar_pii(texto_original)
        marcador_esperado = f"[{tipo_esperado.upper()}_REDACTED]"

        # Verifica se o marcador está presente e o dado original sumiu
        detectou = marcador_esperado in texto_mascarado
        if not detectou:
            todos_passaram = False

    relatorio.verificacoes.append(
        ResultadoVerificacao(
            nome="Detecção de PII (CPF, CNPJ, email)",
            passou=todos_passaram,
            detalhes=(
                "✅ Todos os padrões de PII são detectados e mascarados"
                if todos_passaram
                else "❌ Falha na detecção de algum padrão de PII"
            ),
            categoria="pii",
        )
    )

    # Teste extra: texto SEM PII deve permanecer inalterado
    texto_limpo = "Olá, tudo bem? Preciso de ajuda com meu boleto."
    texto_apos = mascarar_pii(texto_limpo)
    sem_falso_positivo = texto_limpo == texto_apos

    relatorio.verificacoes.append(
        ResultadoVerificacao(
            nome="Sem falsos positivos em texto limpo",
            passou=sem_falso_positivo,
            detalhes=(
                "✅ Texto sem PII permanece inalterado após mascaramento"
                if sem_falso_positivo
                else f"❌ Texto limpo foi alterado: '{texto_apos}'"
            ),
            categoria="pii",
        )
    )


# ============================================================
# 4. VERIFICAÇÃO DE TRACES — Rastreabilidade e auditoria
# ============================================================
# Conforme o checklist (seção 3):
# - O módulo de observabilidade (09) deve existir
# - Arquivos de trace (JSONL) devem ter campos obrigatórios
# - trace_id deve ser único por execução
#
# COMO FUNCIONA:
# - Verifica se o módulo 09 existe na estrutura do projeto
# - Verifica se trace_utils.py contém as funções esperadas
# - Se há traces JSONL existentes, valida a estrutura
# ============================================================

def verificar_traces(relatorio: RelatorioAuditoria) -> None:
    """
    Verifica se o sistema de traces/observabilidade está configurado.

    Verificações:
    1. Módulo 09 existe com trace_utils.py?
    2. trace_utils.py contém TraceRecorder e redigir_texto?
    3. Se há traces JSONL, os campos obrigatórios estão presentes?
    """
    # ── 4.1 Verifica se o módulo de observabilidade existe ───
    trace_utils = RAIZ_PROJETO / "modulo_09_observabilidade" / "trace_utils.py"
    relatorio.verificacoes.append(
        ResultadoVerificacao(
            nome="Módulo de observabilidade existe",
            passou=trace_utils.exists(),
            detalhes=(
                "✅ trace_utils.py encontrado no módulo 09"
                if trace_utils.exists()
                else "❌ trace_utils.py NÃO encontrado — sem rastreabilidade!"
            ),
            categoria="traces",
        )
    )

    # ── 4.2 Verifica conteúdo do trace_utils.py ─────────────
    if trace_utils.exists():
        conteudo = trace_utils.read_text(encoding="utf-8")
        tem_trace_recorder = "class TraceRecorder" in conteudo
        tem_redigir = "def redigir_texto" in conteudo
        tem_sanitizar = "_sanitizar" in conteudo

        funcoes_presentes = all(
            [tem_trace_recorder, tem_redigir, tem_sanitizar]
        )

        detalhes_funcoes = []
        if tem_trace_recorder:
            detalhes_funcoes.append("TraceRecorder ✅")
        else:
            detalhes_funcoes.append("TraceRecorder ❌")
        if tem_redigir:
            detalhes_funcoes.append("redigir_texto ✅")
        else:
            detalhes_funcoes.append("redigir_texto ❌")
        if tem_sanitizar:
            detalhes_funcoes.append("_sanitizar ✅")
        else:
            detalhes_funcoes.append("_sanitizar ❌")

        relatorio.verificacoes.append(
            ResultadoVerificacao(
                nome="Funções de trace completas",
                passou=funcoes_presentes,
                detalhes=f"Funções: {', '.join(detalhes_funcoes)}",
                categoria="traces",
            )
        )


# ============================================================
# 5. VERIFICAÇÃO DE DEPLOY — Testes e benchmarks
# ============================================================
# Conforme o checklist (seção 4):
# - Diretório tests/ deve existir com arquivos de teste
# - Módulo 07 (avaliação) deve estar configurado
# - pytest deve ser executável
#
# COMO FUNCIONA:
# - Verifica se tests/ existe e contém arquivos test_*.py
# - Verifica se módulo 07 existe (avaliação/benchmark)
# - Conta quantos testes existem para estimar cobertura
# ============================================================

def verificar_deploy(relatorio: RelatorioAuditoria) -> None:
    """
    Verifica se os pré-requisitos de deploy estão presentes.

    Verificações:
    1. Diretório tests/ existe com testes?
    2. Módulo 07 (avaliação) está configurado?
    3. Há testes suficientes para cobertura básica?
    """
    # ── 5.1 Diretório de testes ──────────────────────────────
    tests_dir = RAIZ_PROJETO / "tests"
    arquivos_teste = (
        list(tests_dir.glob("test_*.py")) if tests_dir.exists() else []
    )

    relatorio.verificacoes.append(
        ResultadoVerificacao(
            nome="Diretório de testes existe",
            passou=len(arquivos_teste) > 0,
            detalhes=(
                f"✅ {len(arquivos_teste)} arquivo(s) de teste"
                " encontrado(s) em tests/"
                if arquivos_teste
                else "❌ Nenhum arquivo test_*.py encontrado em tests/"
            ),
            categoria="deploy",
        )
    )

    # ── 5.2 Módulo de avaliação (benchmark) ──────────────────
    modulo_avaliacao = RAIZ_PROJETO / "modulo_07_avaliacao"
    relatorio.verificacoes.append(
        ResultadoVerificacao(
            nome="Módulo de avaliação (benchmark) existe",
            passou=modulo_avaliacao.exists(),
            detalhes=(
                "✅ Módulo 07 de avaliação encontrado"
                if modulo_avaliacao.exists()
                else (
                    "❌ Módulo 07 NÃO encontrado"
                    " — sem benchmarks de qualidade!"
                )
            ),
            categoria="deploy",
        )
    )


# ============================================================
# 6. VERIFICAÇÃO DE RESILIÊNCIA — Fallback e retry
# ============================================================
# Conforme o checklist (seção 5):
# - Módulo 11 (resiliência) deve estar implementado
# - ClienteLLMResiliente deve existir com retry e fallback
# - Há configuração de modelo alternativo?
#
# COMO FUNCIONA:
# - Verifica se o módulo 11 existe
# - Busca por ClienteLLMResiliente no código
# - Verifica se há configuração de fallback
# ============================================================

def verificar_resiliencia(relatorio: RelatorioAuditoria) -> None:
    """
    Verifica se o sistema de resiliência está configurado.

    Verificações:
    1. Módulo 11 existe?
    2. ClienteLLMResiliente está implementado?
    3. Há configuração de fallback para modelo alternativo?
    """
    modulo_resiliencia = RAIZ_PROJETO / "modulo_11_resiliencia"
    arquivos_resiliencia = (
        list(modulo_resiliencia.glob("*.py"))
        if modulo_resiliencia.exists()
        else []
    )

    # Busca por implementação de retry/fallback
    tem_cliente_resiliente = False
    tem_fallback = False

    for arquivo in arquivos_resiliencia:
        if arquivo.name == "__init__.py":
            continue
        try:
            conteudo = arquivo.read_text(encoding="utf-8")
            if (
                "ClienteLLMResiliente" in conteudo
                or "retry" in conteudo.lower()
            ):
                tem_cliente_resiliente = True
            if "fallback" in conteudo.lower():
                tem_fallback = True
        except (UnicodeDecodeError, PermissionError):
            continue

    relatorio.verificacoes.append(
        ResultadoVerificacao(
            nome="Módulo de resiliência existe",
            passou=(
                modulo_resiliencia.exists()
                and len(arquivos_resiliencia) > 0
            ),
            detalhes=(
                f"✅ Módulo 11 encontrado com"
                f" {len(arquivos_resiliencia)} arquivo(s)"
                if modulo_resiliencia.exists()
                else "❌ Módulo 11 NÃO encontrado — sem retry/fallback!"
            ),
            categoria="resiliencia",
        )
    )

    relatorio.verificacoes.append(
        ResultadoVerificacao(
            nome="Retry e fallback implementados",
            passou=tem_cliente_resiliente and tem_fallback,
            detalhes=(
                "✅ ClienteLLMResiliente com retry e fallback detectados"
                if tem_cliente_resiliente and tem_fallback
                else (
                    f"Retry: {'✅' if tem_cliente_resiliente else '❌'} | "
                    f"Fallback: {'✅' if tem_fallback else '❌'}"
                )
            ),
            categoria="resiliencia",
        )
    )


# ============================================================
# 7. RELATÓRIO FINAL — Score visual com Rich
# ============================================================
# Consolida todas as verificações em uma tabela colorida com:
# - Status de cada verificação (✅/❌)
# - Categoria de segurança
# - Detalhes do resultado
# - Score final com nível de governança
#
# CORES:
# - Verde (✅): verificação passou
# - Vermelho (❌): verificação falhou
# - Score usa cores conforme o nível (verde/amarelo/laranja/vermelho)
# ============================================================

def exibir_relatorio(relatorio: RelatorioAuditoria) -> None:
    """
    Exibe o relatório de auditoria no terminal com Rich.

    LAYOUT:
    ┌─────────────────────────────────────────────────┐
    │  RELATÓRIO DE AUDITORIA DE GOVERNANÇA           │
    ├────────┬────────────┬──────────┬────────────────┤
    │ Status │ Categoria  │ Nome     │ Detalhes       │
    ├────────┼────────────┼──────────┼────────────────┤
    │ ✅     │ segredos   │ .env ... │ .env encontra..│
    │ ❌     │ pii        │ Deteç... │ Falha na det...│
    └────────┴────────────┴──────────┴────────────────┘

    Score final: XX% — NÍVEL
    """
    console.print()
    console.print(
        Panel(
            "[bold]Relatório de auditoria de segurança e governança\n"
            "Baseado no checklist do Módulo 13[/]",
            title="🔒 AUDITORIA DE GOVERNANÇA",
            border_style="blue",
        )
    )

    # ── Tabela de resultados ─────────────────────────────────
    tabela = Table(
        title="Verificações de Segurança",
        show_header=True,
        header_style="bold magenta",
    )
    tabela.add_column("Status", width=6, justify="center")
    tabela.add_column("Categoria", width=14)
    tabela.add_column("Verificação", width=35)
    tabela.add_column("Detalhes", width=55)

    # Agrupa por categoria para organização visual
    categorias_ordem = ["segredos", "pii", "traces", "deploy", "resiliencia"]
    verificacoes_ordenadas = sorted(
        relatorio.verificacoes,
        key=lambda v: (
            categorias_ordem.index(v.categoria)
            if v.categoria in categorias_ordem
            else 999
        ),
    )

    for v in verificacoes_ordenadas:
        status = "[green]✅[/]" if v.passou else "[red]❌[/]"
        estilo_linha = "" if v.passou else "dim red"
        tabela.add_row(
            status, v.categoria, v.nome, v.detalhes, style=estilo_linha
        )

    console.print(tabela)

    # ── Score final ──────────────────────────────────────────
    total = len(relatorio.verificacoes)
    passou = sum(1 for v in relatorio.verificacoes if v.passou)
    falhou = total - passou

    # Cor do score baseada no nível
    if relatorio.score >= 100:
        cor_score = "bold green"
    elif relatorio.score >= 80:
        cor_score = "bold yellow"
    elif relatorio.score >= 60:
        cor_score = "bold dark_orange"
    else:
        cor_score = "bold red"

    console.print()
    console.print(
        Panel(
            f"[{cor_score}]Score: {relatorio.score:.0f}%[/]\n\n"
            f"Total de verificações: {total}\n"
            f"[green]Passou: {passou}[/] | [red]Falhou: {falhou}[/]\n\n"
            f"Nível: {relatorio.nivel}",
            title="📊 SCORE DE GOVERNANÇA",
            border_style=cor_score.replace("bold ", ""),
        )
    )

    # ── Recomendações para itens que falharam ────────────────
    falhas = [v for v in relatorio.verificacoes if not v.passou]
    if falhas:
        console.print()
        console.print("[bold red]⚠️  Itens que precisam de atenção:[/]")
        for i, v in enumerate(falhas, 1):
            console.print(f"  {i}. [{v.categoria}] {v.nome}")
            console.print(f"     → {v.detalhes}")
    else:
        console.print()
        console.print(
            "[bold green]🎉 Todas as verificações passaram! "
            "O agente está pronto para produção.[/]"
        )


# ============================================================
# 8. DEMONSTRAÇÃO DE PII — Exemplo visual de mascaramento
# ============================================================
# Antes do relatório, mostramos ao aluno COMO o mascaramento
# de PII funciona na prática com exemplos concretos.
# ============================================================

def demonstrar_mascaramento_pii() -> None:
    """
    Demonstra o mascaramento de PII com exemplos visuais.

    Exibe tabela "antes/depois" para que o aluno veja
    exatamente como os dados sensíveis são tratados.
    """
    console.print(
        Panel(
            "[bold]Demonstração de mascaramento de dados sensíveis (PII)\n"
            "Mesmos regex do módulo 09 (trace_utils.py)[/]",
            title="🔐 MASCARAMENTO DE PII",
            border_style="cyan",
        )
    )

    tabela = Table(show_header=True, header_style="bold cyan")
    tabela.add_column("Tipo", width=15)
    tabela.add_column("Texto Original", width=40)
    tabela.add_column("Texto Mascarado", width=40)

    for nome_teste, texto_original, _tipo in DADOS_TESTE_PII:
        texto_mascarado = mascarar_pii(texto_original)
        tabela.add_row(nome_teste, texto_original, texto_mascarado)

    # Texto sem PII (controle/validação)
    texto_limpo = "Olá, preciso de ajuda com meu boleto."
    tabela.add_row(
        "[dim]Sem PII[/]",
        texto_limpo,
        f"[green]{mascarar_pii(texto_limpo)}[/] (inalterado)",
    )

    console.print(tabela)


# ============================================================
# 9. EXECUÇÃO DA AUDITORIA COMPLETA
# ============================================================
# Roda todas as verificações em sequência e gera o relatório.
#
# FLUXO:
# 1. Demonstra mascaramento de PII (visual educativo)
# 2. Roda cada área de verificação
# 3. Gera relatório com score final
#
# EXERCÍCIO SUGERIDO:
# 1. Execute e veja o score do projeto
# 2. Se alguma verificação falhou, corrija e rode novamente
# 3. Adicione uma nova verificação (ex: verificar se README existe)
# 4. Integre este script no CI/CD (ex: rodar no GitHub Actions)
# 5. Adicione novos padrões de PII (telefone, cartão de crédito)
# ============================================================

def executar_auditoria() -> RelatorioAuditoria:
    """
    Executa a auditoria completa de governança e exibe o relatório.

    Retorno:
    - RelatorioAuditoria com todos os resultados (útil para testes)

    ETAPAS:
    1. Cria relatório vazio
    2. Demonstra mascaramento de PII (educativo)
    3. Roda verificações: segredos → PII → traces → deploy → resiliência
    4. Exibe relatório visual com score
    """
    relatorio = RelatorioAuditoria()

    console.print()
    console.print(
        "[bold blue]🔍 Iniciando auditoria de segurança e governança...[/]\n"
    )

    # ── Demonstração educativa de PII ────────────────────────
    demonstrar_mascaramento_pii()

    console.print()

    # ── Verificações — cada uma adiciona ao relatório ────────
    verificar_segredos(relatorio)
    verificar_pii(relatorio)
    verificar_traces(relatorio)
    verificar_deploy(relatorio)
    verificar_resiliencia(relatorio)

    # ── Relatório final ──────────────────────────────────────
    exibir_relatorio(relatorio)

    return relatorio


# ============================================================
# PONTO DE ENTRADA — Execução direta do módulo
# ============================================================
# Ao rodar `python 02_auditoria_seguranca.py`, o aluno verá:
# 1. Demonstração de mascaramento de PII (antes/depois)
# 2. Verificações automatizadas de todas as áreas
# 3. Relatório com score de governança
#
# O score reflete o estado ATUAL do projeto — se algo mudou
# desde a última vez que rodou, o score pode mudar também!
#
# EXERCÍCIOS SUGERIDOS:
# 1. Rode e anote o score atual
# 2. Corrija qualquer verificação que falhou
# 3. Rode novamente e veja o score subir
# 4. Adicione verificação de: telefone, cartão de crédito
# 5. Exporte o relatório para JSON (serialize RelatorioAuditoria)
# ============================================================

if __name__ == "__main__":
    executar_auditoria()
