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


# --------------------------------------------------------------------------
# Locutores de cada filme dublado, e a ligação locutor -> ator
# --------------------------------------------------------------------------
#
# A dublagem identifica locutores (SPEAKER_00, SPEAKER_01...) sem saber quem
# são. Cada locutor de cada filme fica guardado aqui com um trecho da voz e o
# "vetor" da voz (embedding do XTTS). Na tela Atores a pessoa liga o locutor
# a um ator do elenco; daí em diante a voz entra no acervo do ator e, num
# próximo filme com o mesmo ator no elenco, o locutor parecido o bastante é
# reconhecido sozinho e usa a voz acumulada.

import hashlib as _hashlib
import json as _json
import math as _math
import re as _re
import shutil as _shutil
import unicodedata as _unicodedata

IGNORADO = "__ignorado__"
SIMILARIDADE_AUTOMATICA = 0.85


def _migrar(conexao):
    colunas = {linha[1] for linha in conexao.execute("PRAGMA table_info(atores)")}
    for nome, tipo in (("tmdb_id", "INTEGER"), ("foto", "TEXT"), ("bio", "TEXT"),
                       ("nascimento", "TEXT"), ("local_nascimento", "TEXT"),
                       ("embedding", "TEXT")):
        if nome not in colunas:
            conexao.execute(f"ALTER TABLE atores ADD COLUMN {nome} {tipo}")
    conexao.execute("""
        CREATE TABLE IF NOT EXISTS locutores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            caminho_filme TEXT NOT NULL,
            filme TEXT NOT NULL,
            locutor TEXT NOT NULL,
            caminho_audio TEXT NOT NULL,
            segundos REAL NOT NULL,
            falas INTEGER,
            segundos_fala REAL,
            embedding TEXT,
            ator_id TEXT,
            origem TEXT,
            similaridade REAL,
            elenco TEXT,
            criado_em TEXT NOT NULL,
            UNIQUE(caminho_filme, locutor)
        )
    """)
    conexao.commit()


def _conectar_v2():
    conexao = _conectar()
    _migrar(conexao)
    conexao.row_factory = sqlite3.Row
    return conexao


def id_de_ator(nome, tmdb_id=None):
    if tmdb_id:
        return f"tmdb-{int(tmdb_id)}"
    texto = _unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    return _re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-") or "ator"


def _cosseno(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    pa = _math.sqrt(sum(x * x for x in a))
    pb = _math.sqrt(sum(x * x for x in b))
    if not pa or not pb:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (pa * pb)


def _pasta_vozes():
    pasta = os.path.join(config.app_directory(), "vozes")
    os.makedirs(pasta, exist_ok=True)
    return pasta


def _elenco_do_filme(caminho_filme):
    """Elenco no TMDB (nome, personagem, foto, id). Lista vazia sem chave."""
    try:
        from . import tmdb
        return tmdb.elenco_do_arquivo(caminho_filme, config.get_tmdb_api_key()) or []
    except Exception:  # noqa: BLE001 - extra, nunca derruba
        logger.exception("elenco do TMDB falhou")
        return []


def registrar_locutores_do_filme(caminho_filme, referencias, embeddings):
    """Guarda a voz de cada locutor de um filme recém-dublado e tenta
    reconhecer, entre o elenco, atores cuja voz já está no acervo."""
    pasta = os.path.join(_pasta_vozes(),
                         _hashlib.sha1(caminho_filme.encode()).hexdigest()[:12])
    os.makedirs(pasta, exist_ok=True)
    elenco = _elenco_do_filme(caminho_filme)
    filme = os.path.splitext(os.path.basename(caminho_filme))[0]
    ids_elenco = {id_de_ator(p["nome"], p.get("tmdb_id")) for p in elenco}

    with _lock, closing(_conectar_v2()) as conexao:
        atores = [dict(a) for a in conexao.execute(
            "SELECT id, nome, embedding FROM atores WHERE embedding IS NOT NULL")]
        for nome_locutor, ref in referencias.items():
            destino = os.path.join(pasta, f"{nome_locutor}.wav")
            _shutil.copyfile(ref["arquivo"], destino)
            emb = embeddings.get(nome_locutor)
            existente = conexao.execute(
                "SELECT ator_id FROM locutores WHERE caminho_filme=? AND locutor=?",
                (caminho_filme, nome_locutor)).fetchone()
            ator_id = existente["ator_id"] if existente else None
            origem, similaridade = None, None
            if not ator_id and emb:
                candidatos = sorted(
                    ((_cosseno(emb, _json.loads(a["embedding"])), a) for a in atores
                     if a["id"] in ids_elenco), key=lambda x: -x[0])
                if candidatos and candidatos[0][0] >= SIMILARIDADE_AUTOMATICA:
                    similaridade, ator = candidatos[0]
                    ator_id, origem = ator["id"], "automatica"
            conexao.execute("""
                INSERT INTO locutores (caminho_filme, filme, locutor, caminho_audio, segundos,
                                       falas, segundos_fala, embedding, ator_id, origem,
                                       similaridade, elenco, criado_em)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(caminho_filme, locutor) DO UPDATE SET
                    caminho_audio=excluded.caminho_audio, segundos=excluded.segundos,
                    falas=excluded.falas, segundos_fala=excluded.segundos_fala,
                    embedding=excluded.embedding, elenco=excluded.elenco
            """, (caminho_filme, filme, nome_locutor, destino, ref["segundos"], ref.get("falas"),
                  ref.get("segundos_fala"), _json.dumps(emb) if emb else None, ator_id, origem,
                  similaridade, _json.dumps(elenco, ensure_ascii=False), _agora()))
            if origem == "automatica":
                conexao.execute("""INSERT INTO referencias_voz (ator_id, caminho_audio,
                    duracao_segundos, filme_origem, locutor_diarizado, criado_em)
                    VALUES (?,?,?,?,?,?)""",
                                (ator_id, destino, ref["segundos"], filme, nome_locutor, _agora()))
        conexao.commit()


def atores_dos_locutores(caminho_filme):
    """``{"SPEAKER_00": {"id": ..., "nome": ...}}`` dos locutores já ligados."""
    with closing(_conectar_v2()) as conexao:
        linhas = conexao.execute("""
            SELECT l.locutor, a.id, a.nome FROM locutores l JOIN atores a ON a.id = l.ator_id
            WHERE l.caminho_filme = ?""", (caminho_filme,)).fetchall()
        return {l["locutor"]: {"id": l["id"], "nome": l["nome"]} for l in linhas}


def listar_locutores(pendentes=True):
    with closing(_conectar_v2()) as conexao:
        filtro = "WHERE l.ator_id IS NULL" if pendentes else ""
        linhas = conexao.execute(f"""
            SELECT l.id, l.filme, l.caminho_filme, l.locutor, l.segundos, l.falas,
                   l.segundos_fala, l.ator_id, l.origem, l.similaridade, l.elenco,
                   a.nome AS ator
            FROM locutores l LEFT JOIN atores a ON a.id = l.ator_id {filtro}
            ORDER BY l.criado_em DESC, l.segundos_fala DESC""").fetchall()
    saida = []
    for linha in linhas:
        item = dict(linha)
        item["elenco"] = _json.loads(item["elenco"] or "[]")
        saida.append(item)
    return saida


def locutor(locutor_id):
    with closing(_conectar_v2()) as conexao:
        linha = conexao.execute("SELECT * FROM locutores WHERE id=?", (locutor_id,)).fetchone()
        return dict(linha) if linha else None


def salvar_ator(ator_id, nome, **extras):
    campos = {k: v for k, v in extras.items()
              if k in ("tmdb_id", "foto", "bio", "nascimento", "local_nascimento")
              and v is not None}
    with _lock, closing(_conectar_v2()) as conexao:
        conexao.execute("INSERT OR IGNORE INTO atores (id, nome, criado_em) VALUES (?, ?, ?)",
                        (ator_id, nome, _agora()))
        conexao.execute("UPDATE atores SET nome=? WHERE id=?", (nome, ator_id))
        for campo, valor in campos.items():
            conexao.execute(f"UPDATE atores SET {campo}=? WHERE id=?", (valor, ator_id))
        conexao.commit()


def _recalcular_embedding(conexao, ator_id):
    vetores = [_json.loads(l["embedding"]) for l in conexao.execute(
        "SELECT embedding FROM locutores WHERE ator_id=? AND embedding IS NOT NULL",
        (ator_id,))]
    media = [sum(c) / len(vetores) for c in zip(*vetores)] if vetores else None
    conexao.execute("UPDATE atores SET embedding=? WHERE id=?",
                    (_json.dumps(media) if media else None, ator_id))


def atribuir_locutor(locutor_id, ator_id, nome=None, origem="manual", **extras):
    """Liga um locutor a um ator (ou ``IGNORADO``). Desfaz a ligação anterior."""
    desatribuir_locutor(locutor_id)
    if ator_id == IGNORADO:
        with _lock, closing(_conectar_v2()) as conexao:
            conexao.execute("UPDATE locutores SET ator_id=?, origem=? WHERE id=?",
                            (IGNORADO, origem, locutor_id))
            conexao.commit()
        return
    salvar_ator(ator_id, nome or ator_id, **extras)
    with _lock, closing(_conectar_v2()) as conexao:
        loc = conexao.execute("SELECT * FROM locutores WHERE id=?", (locutor_id,)).fetchone()
        if not loc:
            raise ValueError("locutor não encontrado")
        conexao.execute("UPDATE locutores SET ator_id=?, origem=? WHERE id=?",
                        (ator_id, origem, locutor_id))
        conexao.execute("""INSERT INTO referencias_voz (ator_id, caminho_audio,
            duracao_segundos, filme_origem, locutor_diarizado, criado_em)
            VALUES (?,?,?,?,?,?)""", (ator_id, loc["caminho_audio"], loc["segundos"],
                                      loc["filme"], loc["locutor"], _agora()))
        _recalcular_embedding(conexao, ator_id)
        conexao.commit()


def desatribuir_locutor(locutor_id):
    with _lock, closing(_conectar_v2()) as conexao:
        loc = conexao.execute("SELECT * FROM locutores WHERE id=?", (locutor_id,)).fetchone()
        if not loc or not loc["ator_id"]:
            return
        ator_id = loc["ator_id"]
        conexao.execute("DELETE FROM referencias_voz WHERE ator_id=? AND caminho_audio=?",
                        (ator_id, loc["caminho_audio"]))
        conexao.execute("UPDATE locutores SET ator_id=NULL, origem=NULL WHERE id=?",
                        (locutor_id,))
        if ator_id != IGNORADO:
            _recalcular_embedding(conexao, ator_id)
            sobra = conexao.execute("SELECT COUNT(*) FROM referencias_voz WHERE ator_id=?",
                                    (ator_id,)).fetchone()[0]
            if not sobra:
                conexao.execute("DELETE FROM atores WHERE id=?", (ator_id,))
        conexao.commit()


def listar_atores_completo():
    """Atores com voz no acervo, com ficha e os filmes de onde veio a voz."""
    with closing(_conectar_v2()) as conexao:
        atores = [dict(a) for a in conexao.execute("""
            SELECT a.id, a.nome, a.tmdb_id, a.foto, a.bio, a.nascimento, a.local_nascimento,
                   COUNT(r.id) AS total_referencias,
                   COALESCE(SUM(r.duracao_segundos), 0) AS duracao_total_segundos
            FROM atores a LEFT JOIN referencias_voz r ON r.ator_id = a.id
            GROUP BY a.id ORDER BY duracao_total_segundos DESC""")]
        for ator in atores:
            ator["filmes"] = [dict(f) for f in conexao.execute("""
                SELECT filme_origem AS filme, SUM(duracao_segundos) AS segundos
                FROM referencias_voz WHERE ator_id=? GROUP BY filme_origem""", (ator["id"],))]
        return atores
