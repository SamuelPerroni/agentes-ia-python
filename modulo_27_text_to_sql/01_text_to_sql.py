"""
============================================================
MÓDULO 27.1 - AGENTES COM BANCO DE DADOS (TEXT-TO-SQL)
============================================================
Neste módulo, aprendemos a construir um agente capaz de
converter perguntas em linguagem natural em queries SQL,
validá-las e executá-las com segurança.

CONCEITO CHAVE:
"Quais boletos venceram essa semana?" é uma pergunta simples
para um humano — mas exige uma JOIN entre tabelas, filtro
de data, e formatação específica no banco de dados.
Text-to-SQL elimina a necessidade de o usuário conhecer SQL.

POR QUE TEXT-TO-SQL EM APA?
- ERPs e CRMs geralmente têm banco relacional exposto
- Analistas não precisam mais escrever SQL manual
- O agente pode responder perguntas ad-hoc sem dashboard
- Auditoria: toda query gerada fica registrada

ARQUITETURA — Text-to-SQL SEGURO:

  ┌──────────────────────────────────────────────────────────┐
  │  Pergunta natural                                        │
  │       │                                                  │
  │       ▼                                                  │
  │  [ Schema Loader ] → injeta DDL no prompt                │
  │       │                                                  │
  │       ▼                                                  │
  │  [ LLM ] → gera SQL candidato                            │
  │       │                                                  │
  │       ▼                                                  │
  │  [ Validador SQL ] → bloqueia DROP/DELETE/INSERT         │
  │       │                                                  │
  │       ▼                                                  │
  │  [ Executor ] → SQLite (ou qualquer DBAPI2)              │
  │       │                                                  │
  │       ▼                                                  │
  │  [ Formatador ] → resposta em linguagem natural          │
  └──────────────────────────────────────────────────────────┘

SEGURANÇA — RISCOS DE SQL INJECTION VIA LLM:
- O LLM pode ser induzido a gerar DROP TABLE, UPDATE, etc.
- Validação estática de AST antes de executar
- Conexão com usuário read-only no banco de produção
- Limit forçado em toda SELECT (evita varredura total)

Tópicos cobertos:
1. Criação de banco SQLite em memória com dados de exemplo
2. Carregamento de schema (DDL) para o contexto do LLM
3. Geração de SQL via LLM (simulado) + extração da query
4. Validação de segurança: bloqueia DDL e DML perigosos
5. Execução com limite automático de linhas
6. Formatação da resposta em linguagem natural
============================================================
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# 1. BANCO DE DADOS EM MEMÓRIA
# ============================================================
# Usamos SQLite em memória para o demo — sem instalação.
# Em produção substitua a connection string pelo seu banco:
#   PostgreSQL: psycopg2.connect(...)
#   SQL Server: pyodbc.connect(...)
#   Oracle:     cx_Oracle.connect(...)
# A camada de validação e o agent permanecem iguais.
# ============================================================

SCHEMA_SQL = """
CREATE TABLE boletos (
    id          INTEGER PRIMARY KEY,
    sacado      TEXT NOT NULL,
    valor       REAL NOT NULL,
    vencimento  TEXT NOT NULL,   -- formato YYYY-MM-DD
    status      TEXT NOT NULL,   -- 'aberto', 'pago', 'vencido'
    banco       TEXT
);

CREATE TABLE pagamentos (
    id          INTEGER PRIMARY KEY,
    boleto_id   INTEGER REFERENCES boletos(id),
    data_pag    TEXT NOT NULL,
    valor_pago  REAL NOT NULL
);
"""

DADOS_INICIAIS = """
INSERT INTO boletos VALUES
  (1,'Empresa Alpha Ltda',  1500.00,'2026-03-25','vencido','Bradesco'),
  (2,'Beta Comércio S.A.',   890.00,'2026-04-15','aberto', 'Itaú'),
  (3,'Gama Serviços ME',    3200.00,'2026-04-05','vencido','Santander'),
  (4,'Delta Tech Ltda',     2100.00,'2026-04-20','aberto', 'Bradesco'),
  (5,'Epsilon Ind. S.A.',    450.00,'2026-03-10','pago',   'Itaú');

INSERT INTO pagamentos VALUES
  (1, 5, '2026-03-09', 450.00);
"""


def criar_banco() -> sqlite3.Connection:
    """Cria banco SQLite em memória com dados de exemplo."""
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA_SQL + DADOS_INICIAIS)
    return con


# ============================================================
# 2. SCHEMA LOADER
# ============================================================
# Injeta a DDL do banco no prompt do LLM para que ele
# conheça as tabelas, colunas e tipos antes de gerar SQL.
# ============================================================

def carregar_schema(con: sqlite3.Connection) -> str:
    """
    Extrai DDL das tabelas diretamente do sqlite_master.
    Em bancos externos, consulte information_schema.
    """
    cur = con.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type='table' ORDER BY name"
    )
    ddls = [row[0] for row in cur.fetchall() if row[0]]
    return "\n\n".join(ddls)


# ============================================================
# 3. VALIDADOR DE SQL
# ============================================================
# Antes de executar qualquer query gerada pelo LLM,
# validamos que ela não contém comandos perigosos.
# Usamos análise léxica simples (regex) — rápida e segura.
#
# Para validação mais robusta em produção, use sqlglot:
#   import sqlglot
#   ast = sqlglot.parse_one(sql)
#   # navegue a AST e rejeite nós ALTER/DROP/INSERT/UPDATE
# ============================================================

# Palavras-chave que nunca devem aparecer em queries somente-leitura
_PROIBIDOS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE"
    r"|REPLACE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


@dataclass
class ResultadoValidacao:
    """Resultado da validação de uma query SQL."""
    valida: bool
    motivo: str = ""


def validar_sql(sql: str) -> ResultadoValidacao:
    """
    Garante que a query é somente-leitura.
    Retorna ResultadoValidacao com motivo em caso de falha.
    """
    m = _PROIBIDOS.search(sql)
    if m:
        return ResultadoValidacao(
            False,
            f"Comando proibido detectado: '{m.group()}'. "
            "Apenas SELECT é permitido.",
        )
    if not re.search(r"\bSELECT\b", sql, re.IGNORECASE):
        return ResultadoValidacao(
            False, "Query não contém SELECT."
        )
    return ResultadoValidacao(True)


# ============================================================
# 4. GERADOR DE SQL (SIMULADO)
# ============================================================
# Em produção, chame o LLM passando o schema e a pergunta:
#
#   prompt = f"""
#   Schema do banco:
#   {schema}
#
#   Gere uma query SQLite para responder:
#   {pergunta}
#
#   Responda APENAS com o SQL, sem explicação.
#   Inclua LIMIT 100 em toda SELECT.
#   """
#   resposta = cliente.chat.completions.create(...)
#   sql = extrair_sql(resposta.choices[0].message.content)
#
# Aqui simulamos as respostas para o treinamento.
# ============================================================

_RESPOSTAS_SIMULADAS: dict[str, str] = {
    "boletos vencidos": (
        "SELECT id, sacado, valor, vencimento, banco "
        "FROM boletos WHERE status = 'vencido' "
        "ORDER BY vencimento LIMIT 100"
    ),
    "total em aberto": (
        "SELECT SUM(valor) AS total_aberto "
        "FROM boletos WHERE status = 'aberto'"
    ),
    "boletos por banco": (
        "SELECT banco, COUNT(*) AS quantidade, "
        "SUM(valor) AS total "
        "FROM boletos GROUP BY banco ORDER BY total DESC "
        "LIMIT 100"
    ),
    "boletos pagos": (
        "SELECT b.sacado, b.valor, p.data_pag, p.valor_pago "
        "FROM boletos b JOIN pagamentos p ON b.id = p.boleto_id "
        "LIMIT 100"
    ),
}


def simular_llm_sql(pergunta: str, _schema: str) -> str:
    """
    Simula a geração de SQL por um LLM.
    Retorna a query mais próxima da pergunta.
    """
    pergunta_lower = pergunta.lower()
    for chave, sql in _RESPOSTAS_SIMULADAS.items():
        if any(
            palavra in pergunta_lower
            for palavra in chave.split()
        ):
            return sql
    return (
        "SELECT id, sacado, valor, vencimento, status "
        "FROM boletos LIMIT 10"
    )


def extrair_sql(texto: str) -> str:
    """
    Extrai bloco SQL de uma resposta de LLM que pode
    conter markdown (```sql ... ```).
    """
    # Tenta extrair bloco de código SQL
    m = re.search(
        r"```(?:sql)?\s*(.*?)```", texto, re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1).strip()
    return texto.strip()


# ============================================================
# 5. AGENTE TEXT-TO-SQL
# ============================================================

class AgenteTextToSQL:
    """
    Agente que responde perguntas em linguagem natural
    consultando um banco de dados relacional.
    """

    def __init__(self, con: sqlite3.Connection) -> None:
        self._con = con
        self._schema = carregar_schema(con)

    def perguntar(self, pergunta: str) -> None:
        """Pipeline completo: pergunta → SQL → resultado."""
        console.rule(f"[cyan]Pergunta: {pergunta}")

        # 1. Gera SQL via LLM
        sql_bruto = simular_llm_sql(pergunta, self._schema)
        sql = extrair_sql(sql_bruto)
        console.print(
            Panel(sql, title="SQL Gerado", style="yellow")
        )

        # 2. Valida antes de executar
        val = validar_sql(sql)
        if not val.valida:
            console.print(
                f"[red]Query bloqueada: {val.motivo}[/]"
            )
            return

        # 3. Executa
        try:
            cur = self._con.execute(sql)
            colunas = [d[0] for d in cur.description]
            linhas = cur.fetchall()
        except sqlite3.Error as exc:
            console.print(f"[red]Erro SQL: {exc}[/]")
            return

        # 4. Exibe resultado
        if not linhas:
            console.print("[yellow]Nenhum resultado.[/]")
            return

        tabela = Table(
            title="Resultado",
            header_style="bold green",
            show_lines=True,
        )
        for col in colunas:
            tabela.add_column(col)
        for linha in linhas:
            tabela.add_row(
                *[str(v) if v is not None else "" for v in linha]
            )
        console.print(tabela)
        console.print(
            f"[dim]{len(linhas)} linha(s) retornada(s)[/]"
        )


# ============================================================
# 6. DEMO
# ============================================================

def demo_text_to_sql() -> None:
    """Demonstração do agente Text-to-SQL com banco em memória."""
    console.print(
        Panel(
            "[bold]Módulo 27 — Agentes com Banco de Dados "
            "(Text-to-SQL)[/]\n"
            "Perguntas em linguagem natural → SQL → resultado",
            style="bold blue",
        )
    )

    con = criar_banco()
    agente = AgenteTextToSQL(con)

    perguntas = [
        "Quais são os boletos vencidos?",
        "Qual o total em aberto?",
        "Como está a distribuição de boletos por banco?",
        "Mostre os boletos pagos com data de pagamento.",
    ]

    for pergunta in perguntas:
        agente.perguntar(pergunta)
        console.print()

    # Demonstra bloqueio de segurança
    console.rule("[red]Teste de Segurança — SQL Perigoso")
    sql_perigoso = "DROP TABLE boletos; SELECT 1"
    val = validar_sql(sql_perigoso)
    console.print(
        f"Query: [red]{sql_perigoso}[/]\n"
        f"Válida: {val.valida}\n"
        f"Motivo: {val.motivo}"
    )

    # Guia de referência
    console.print(
        Panel(
            "Para usar LLM real:\n\n"
            "1. Instale: pip install groq sqlglot\n"
            "2. Substitua simular_llm_sql() por chamada Groq\n"
            "3. Use sqlglot.parse_one(sql) para validação AST\n"
            "4. Crie usuário read-only no banco de produção\n"
            "5. Adicione LIMIT automático na query gerada",
            title="Próximos Passos para Produção",
            style="dim",
        )
    )

    con.close()


if __name__ == "__main__":
    demo_text_to_sql()
