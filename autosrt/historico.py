"""Histórico persistente de trabalhos, independente da memória do processo.

``JobQueue`` (ver :mod:`autosrt.jobs`) guarda os trabalhos só em memória --
reinicia o serviço e a lista inteira some, sem nada em disco que diga que
filme foi processado, quando, ou com quantas legendas. Este módulo grava
cada trabalho terminado num banco SQLite em :func:`autosrt.config.app_directory`
(o mesmo ``~/.config/autosrt`` que já guarda o ``config.json``) -- fora da
pasta do código, então sobrevive tanto a reinício do serviço quanto a
atualização do próprio autosrt (``git pull``, troca de branch).

Ligado a ``JobQueue`` via o parâmetro ``on_final``: este módulo não sabe
nada sobre filas ou threads, só grava o que recebe.
"""

import logging
import os
import sqlite3
import threading
from contextlib import closing

from . import config

logger = logging.getLogger(__name__)

NOME_ARQUIVO = "historico.db"

#: SQLite não é seguro para escrita concorrente sem coordenação externa --
#: isso substitui ela. Nunca disputado de verdade (um trabalho termina de
#: cada vez, max_workers raramente passa de 2), então o custo é zero.
_lock = threading.Lock()


def caminho_do_banco() -> str:
    return os.path.join(config.app_directory(), NOME_ARQUIVO)


def _conectar():
    caminho = caminho_do_banco()
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    conexao = sqlite3.connect(caminho, timeout=10)
    conexao.execute("""
        CREATE TABLE IF NOT EXISTS trabalhos (
            id TEXT PRIMARY KEY,
            quando TEXT NOT NULL,
            arquivo TEXT NOT NULL,
            estado TEXT NOT NULL,
            idioma TEXT,
            total INTEGER,
            traduzidas INTEGER,
            falhas INTEGER,
            erro TEXT,
            aviso TEXT
        )
    """)
    return conexao


def registrar(job) -> None:
    """Grava um trabalho terminado. Pensado para ``JobQueue(on_final=...)``.

    Nunca lança: um histórico que derruba o trabalho em si (ou a fila)
    seria pior do que nenhum histórico. Falha vira log, não exceção --
    ``JobQueue`` também protege essa chamada por fora, mas não custa nada
    este módulo se proteger sozinho também.
    """
    detalhes = job.detalhes or {}
    try:
        with _lock, closing(_conectar()) as conexao, conexao:
            conexao.execute(
                """INSERT OR REPLACE INTO trabalhos
                   (id, quando, arquivo, estado, idioma, total, traduzidas,
                    falhas, erro, aviso)
                   VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job.id, job.nome, job.estado, detalhes.get("idioma"),
                 detalhes.get("total"), detalhes.get("traduzidas"),
                 detalhes.get("falhas"), job.erro, detalhes.get("aviso")))
    except Exception:
        logger.exception(
            "não deu para gravar o histórico do trabalho %s", job.id)


def listar(limite=200) -> list:
    """Os trabalhos mais recentes primeiro, prontos para virar JSON."""
    with closing(_conectar()) as conexao:
        conexao.row_factory = sqlite3.Row
        linhas = conexao.execute(
            "SELECT * FROM trabalhos ORDER BY quando DESC, rowid DESC "
            "LIMIT ?", (limite,)).fetchall()
    return [dict(linha) for linha in linhas]
