"""Banco de vozes de atores, reaproveitado entre filmes.

Clonar e limpar (Demucs) a voz de um ator do zero a cada filme é desperdício
num acervo com dezenas de milhares de itens: o mesmo ator aparece em vários
filmes, e quanto mais material de referência limpo se acumula para ele, melhor
tende a ficar a clonagem — ao contrário de reprocessar o mesmo trabalho toda
vez. Este módulo guarda, por ator, os trechos de áudio já limpos (pós-Demucs)
extraídos de cada filme processado, para reuso.

A chave de um ator (``ator_id``) é o nome normalizado como o Jellyfin já
cataloga no elenco de cada filme (:mod:`autosrt` não inventa identidade de
pessoa — usa a que a biblioteca já tem). Casar com o Jellyfin fica para quem
chama este módulo; aqui dentro ``ator_id`` é só uma string.

Segue o mesmo padrão de :mod:`autosrt.historico`: SQLite em
:func:`autosrt.config.app_directory`, fora da pasta do código, sobrevive a
``git pull``/troca de branch e a reinício do serviço.
"""

import logging
import os
import sqlite3
import subprocess
import threading
from contextlib import closing
from datetime import datetime, timezone

from . import config

logger = logging.getLogger(__name__)

NOME_ARQUIVO = "vozes.db"

# Mesma razão do historico.py: nunca disputado de verdade, mas sem essa trava
# duas escritas quase simultâneas (dois filmes do mesmo ator terminando perto
# um do outro) podem corromper o SQLite.
_lock = threading.Lock()

# Duração máxima de referência combinada mandada pro XTTS. Mais do que isso
# não melhora a clonagem e só deixa a extração/concatenação mais lenta à toa.
DURACAO_MAXIMA_REFERENCIA_S = 90


def caminho_do_banco() -> str:
    return os.path.join(config.app_directory(), NOME_ARQUIVO)


def _conectar():
    caminho = caminho_do_banco()
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    conexao = sqlite3.connect(caminho, timeout=10)
    conexao.execute("""
        CREATE TABLE IF NOT EXISTS atores (
            id TEXT PRIMARY KEY,
            nome TEXT NOT NULL,
            criado_em TEXT NOT NULL
        )
    """)
    conexao.execute("""
        CREATE TABLE IF NOT EXISTS referencias_voz (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ator_id TEXT NOT NULL REFERENCES atores(id),
            caminho_audio TEXT NOT NULL,
            duracao_segundos REAL NOT NULL,
            filme_origem TEXT NOT NULL,
            locutor_diarizado TEXT,
            criado_em TEXT NOT NULL
        )
    """)
    conexao.execute("""
        CREATE INDEX IF NOT EXISTS idx_referencias_ator
        ON referencias_voz(ator_id)
    """)
    return conexao


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def registrar_ator(ator_id: str, nome: str) -> None:
    """Garante que o ator existe no banco. Idempotente."""
    with _lock, closing(_conectar()) as conexao:
        conexao.execute(
            "INSERT OR IGNORE INTO atores (id, nome, criado_em) VALUES (?, ?, ?)",
            (ator_id, nome, _agora()),
        )
        conexao.commit()


def adicionar_referencia(ator_id: str, caminho_audio: str, duracao_segundos: float,
                         filme_origem: str, locutor_diarizado: str = None) -> None:
    """Registra um novo trecho de voz limpo (pós-Demucs) para um ator.

    O arquivo em ``caminho_audio`` precisa continuar existindo depois -- este
    módulo só guarda o caminho, não copia o conteúdo. Cabe a quem chama
    manter esses arquivos num lugar estável (não numa pasta temporária de
    trabalho que pode ser limpa).
    """
    with _lock, closing(_conectar()) as conexao:
        conexao.execute("""
            INSERT INTO referencias_voz
                (ator_id, caminho_audio, duracao_segundos, filme_origem, locutor_diarizado, criado_em)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ator_id, caminho_audio, duracao_segundos, filme_origem, locutor_diarizado, _agora()))
        conexao.commit()


def referencias_de(ator_id: str) -> list:
    """Lista os trechos de voz guardados para um ator, mais recentes primeiro."""
    with closing(_conectar()) as conexao:
        conexao.row_factory = sqlite3.Row
        linhas = conexao.execute("""
            SELECT id, caminho_audio, duracao_segundos, filme_origem, locutor_diarizado, criado_em
            FROM referencias_voz WHERE ator_id = ? ORDER BY criado_em DESC
        """, (ator_id,)).fetchall()
        return [dict(linha) for linha in linhas]


def listar_atores() -> list:
    """Lista todos os atores com voz catalogada, com duração total acumulada."""
    with closing(_conectar()) as conexao:
        conexao.row_factory = sqlite3.Row
        linhas = conexao.execute("""
            SELECT a.id, a.nome, a.criado_em,
                   COUNT(r.id) AS total_referencias,
                   COALESCE(SUM(r.duracao_segundos), 0) AS duracao_total_segundos
            FROM atores a
            LEFT JOIN referencias_voz r ON r.ator_id = a.id
            GROUP BY a.id
            ORDER BY duracao_total_segundos DESC
        """).fetchall()
        return [dict(linha) for linha in linhas]


def montar_referencia_combinada(ator_id: str, destino: str,
                                max_segundos: float = DURACAO_MAXIMA_REFERENCIA_S) -> str:
    """Concatena os trechos de voz de um ator num arquivo só, pronto pro XTTS.

    Pega os trechos mais recentes primeiro até acumular ``max_segundos`` (não
    adianta mandar horas de referência pro XTTS -- ver :data:`DURACAO_MAXIMA_REFERENCIA_S`).
    Levanta :class:`ValueError` se o ator não tiver nenhuma referência.

    Returns:
        O caminho de ``destino``, para encadear a chamada.
    """
    referencias = referencias_de(ator_id)
    if not referencias:
        raise ValueError(f"Nenhuma referência de voz encontrada para o ator {ator_id!r}.")

    escolhidas = []
    acumulado = 0.0
    for ref in referencias:
        if acumulado >= max_segundos:
            break
        escolhidas.append(ref["caminho_audio"])
        acumulado += ref["duracao_segundos"]

    if len(escolhidas) == 1:
        subprocess.run(["cp", escolhidas[0], destino], check=True)
        return destino

    lista_concat = destino + ".concat.txt"
    with open(lista_concat, "w", encoding="utf-8") as f:
        for caminho in escolhidas:
            f.write(f"file '{os.path.abspath(caminho)}'\n")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lista_concat,
        destino, "-loglevel", "error",
    ], check=True)
    os.remove(lista_concat)
    return destino
