"""Reconhecimento de filme via TMDB (The Movie Database).

O IMDb não oferece API pública gratuita — o acesso aos dados dele exige
licenciamento comercial pago. O TMDB é o substituto de fato usado pela
maioria dos projetos abertos: é gratuito, referencia o próprio ID do IMDb e
cobre o mesmo catálogo.

O uso aqui é propositalmente modesto: a partir do nome do arquivo, tenta
achar o filme correspondente e devolve um resumo compacto (título, ano,
pôster, idioma original) para exibir na lista de arquivos do servidor —
principalmente para confirmar, antes de gastar meia hora de GPU
transcrevendo, que o arquivo é mesmo o filme que parece ser.

Sem chave configurada (:func:`autosrt.config.get_tmdb_api_key`), a busca
simplesmente não acontece: é um extra, nunca um requisito.
"""

import os
import re

import requests

from .language import language_name

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
SEARCH_TV_URL = "https://api.themoviedb.org/3/search/tv"
TV_DETAILS_URL = "https://api.themoviedb.org/3/tv/{id}"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

DEFAULT_TIMEOUT = 4  # segundos. Curto de propósito: é um extra na listagem
                      # de arquivos, não pode travar a página por causa dele.

# Ano plausível de lançamento: 19xx ou 20xx, isolado de outros dígitos. Evita
# confundir com resolução ("1080p", "2160p") ou tamanho ("700mb").
YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")

# Marcação de temporada/episódio nos nomes de release de série:
# "Nome.Da.Serie.S01E05.1080p...". O que vem antes é o título da série.
EPISODE_RE = re.compile(r"[Ss](\d{1,2})[Ee](\d{1,3})")


def guess_title_and_year(filename: str):
    """Chuta o título e o ano a partir do nome do arquivo.

    Nomes de release seguem um padrão razoavelmente previsível:
    ``Titulo.Do.Filme.2019.1080p.BluRay.x264-GRUPO.mkv``. O ano, quando
    presente, marca onde o título termina e as tags técnicas começam.

    >>> guess_title_and_year("Duna.Parte.Dois.2024.2160p.mkv")
    ('Duna Parte Dois', 2024)
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    cleaned = re.sub(r"[._]+", " ", stem)

    year = None
    corte = len(cleaned)
    for match in YEAR_RE.finditer(cleaned):
        year = int(match.group(1))
        corte = match.start()

    title = cleaned[:corte] if year else cleaned
    title = re.sub(r"[\s\(\[\-–—]+$", "", title).strip()
    title = re.sub(r"\s+", " ", title).strip()
    return title, year


def guess_series_title_and_episode(filename: str):
    """Chuta título da série, temporada e episódio a partir do nome do arquivo.

    >>> guess_series_title_and_episode("Nome.Da.Serie.S01E05.1080p.mkv")
    ('Nome Da Serie', 1, 5)

    Quando o nome não tem o padrão ``SxxExx``, devolve ``(None, None, None)``
    -- é sinal de que o arquivo não é (ou não parece ser) um episódio.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    cleaned = re.sub(r"[._]+", " ", stem)

    match = EPISODE_RE.search(cleaned)
    if not match:
        return None, None, None

    title = cleaned[:match.start()]
    title = re.sub(r"[\s\(\[\-–—]+$", "", title).strip()
    title = re.sub(r"\s+", " ", title).strip()
    return title, int(match.group(1)), int(match.group(2))


def _poster_url(poster_path):
    return f"{IMAGE_BASE}{poster_path}" if poster_path else None


def _release_year(release_date):
    if not release_date or len(release_date) < 4:
        return None
    try:
        return int(release_date[:4])
    except ValueError:
        return None


def search_movie(query, api_key, *, year=None, session=None,
                 timeout=DEFAULT_TIMEOUT):
    """Busca o filme mais provável no TMDB. Devolve o resultado bruto da API
    ou ``None``, sem nunca levantar exceção — é um extra, não pode derrubar
    a listagem de arquivos por falha de rede ou de configuração."""
    if not query or not api_key:
        return None

    params = {"api_key": api_key, "query": query, "language": "pt-BR",
             "include_adult": "false"}
    if year:
        params["year"] = year

    http = session or requests
    try:
        response = http.get(SEARCH_URL, params=params, timeout=timeout)
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None
    try:
        data = response.json()
    except ValueError:
        return None

    resultados = data.get("results") or []
    return resultados[0] if resultados else None


def search_tv(query, api_key, *, session=None, timeout=DEFAULT_TIMEOUT):
    """Como :func:`search_movie`, mas buscando séries (``search/tv``)."""
    if not query or not api_key:
        return None

    params = {"api_key": api_key, "query": query, "language": "pt-BR",
             "include_adult": "false"}

    http = session or requests
    try:
        response = http.get(SEARCH_TV_URL, params=params, timeout=timeout)
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None
    try:
        data = response.json()
    except ValueError:
        return None

    resultados = data.get("results") or []
    return resultados[0] if resultados else None


def tv_details(tv_id, api_key, *, session=None, timeout=DEFAULT_TIMEOUT):
    """Busca o total de temporadas/episódios de uma série pelo seu id no
    TMDB. A busca por nome (:func:`search_tv`) não traz esses números --
    só o detalhe da série (``tv/{id}``) traz."""
    http = session or requests
    try:
        response = http.get(TV_DETAILS_URL.format(id=tv_id),
                            params={"api_key": api_key, "language": "pt-BR"},
                            timeout=timeout)
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None


#: Cache dos detalhes de série (temporadas/episódios totais), por tv_id.
#: Separado do cache por arquivo: vários episódios da mesma série repetem o
#: mesmo tv_id, e esses números não mudam de um episódio pro outro.
_tv_details_cache = {}


def _tv_details_cached(tv_id, api_key, *, session=None, timeout=DEFAULT_TIMEOUT):
    if tv_id not in _tv_details_cache:
        _tv_details_cache[tv_id] = tv_details(tv_id, api_key, session=session,
                                              timeout=timeout)
    return _tv_details_cache[tv_id]


def _lookup_movie(path, api_key, *, session=None, timeout=DEFAULT_TIMEOUT):
    titulo_bruto, ano = guess_title_and_year(path)
    if not titulo_bruto:
        return None

    resultado = search_movie(titulo_bruto, api_key, year=ano, session=session,
                             timeout=timeout)
    if not resultado:
        return None

    idioma = resultado.get("original_language")
    return {
        "tipo": "filme",
        "titulo": resultado.get("title") or titulo_bruto,
        "ano": _release_year(resultado.get("release_date")),
        "poster": _poster_url(resultado.get("poster_path")),
        "sinopse": resultado.get("overview") or None,
        "idioma_original": idioma,
        "idioma_original_nome": language_name(idioma) if idioma else None,
    }


def _lookup_series(path, titulo_bruto, temporada, episodio, api_key, *,
                   session=None, timeout=DEFAULT_TIMEOUT):
    resultado = search_tv(titulo_bruto, api_key, session=session, timeout=timeout)
    if not resultado:
        return None

    detalhes = _tv_details_cached(resultado.get("id"), api_key, session=session,
                                  timeout=timeout) or {}
    idioma = resultado.get("original_language")
    return {
        "tipo": "serie",
        "titulo": resultado.get("name") or titulo_bruto,
        "poster": _poster_url(resultado.get("poster_path")),
        "sinopse": resultado.get("overview") or None,
        "idioma_original": idioma,
        "idioma_original_nome": language_name(idioma) if idioma else None,
        "temporada": temporada,
        "episodio": episodio,
        "temporadas_total": detalhes.get("number_of_seasons"),
        "episodios_total": detalhes.get("number_of_episodes"),
    }


def lookup_for_file(path, api_key, *, session=None, timeout=DEFAULT_TIMEOUT):
    """Reconhece o filme ou episódio de série de um arquivo de mídia pelo
    nome. Um nome com o padrão ``SxxExx`` (ex.: ``S01E05``) é tratado como
    episódio de série; senão, como filme.

    Returns:
        Para filme: ``{"tipo": "filme", "titulo", "ano", "poster",
        "sinopse", "idioma_original", "idioma_original_nome"}``.
        Para série: ``{"tipo": "serie", "titulo", "poster", "sinopse",
        "idioma_original", "idioma_original_nome", "temporada", "episodio",
        "temporadas_total", "episodios_total"}``.
        ``None`` quando não reconhece nada -- seja por falta de chave, nome
        irreconhecível, falha de rede, ou nenhum resultado no TMDB.
    """
    if not api_key:
        return None

    titulo_serie, temporada, episodio = guess_series_title_and_episode(path)
    if titulo_serie:
        return _lookup_series(path, titulo_serie, temporada, episodio, api_key,
                              session=session, timeout=timeout)

    return _lookup_movie(path, api_key, session=session, timeout=timeout)


#: Cache em memória, por nome de arquivo. O filme que um nome de arquivo
#: representa não muda durante a vida do processo, e cachear evita repetir a
#: mesma chamada de rede a cada vez que a página recarrega a listagem.
_cache = {}


def lookup_cached(path, api_key, *, session=None, timeout=DEFAULT_TIMEOUT):
    """Como :func:`lookup_for_file`, mas memorizado por nome de arquivo."""
    chave = os.path.basename(path)
    if chave not in _cache:
        _cache[chave] = lookup_for_file(path, api_key, session=session,
                                        timeout=timeout)
    return _cache[chave]


def limpar_cache() -> None:
    """Esvazia os caches. Usado pelos testes; útil também se o usuário
    renomear o arquivo para o nome certo depois de um reconhecimento errado."""
    _cache.clear()
    _tv_details_cache.clear()
