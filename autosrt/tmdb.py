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
IMAGE_BASE = "https://image.tmdb.org/t/p/w154"

DEFAULT_TIMEOUT = 4  # segundos. Curto de propósito: é um extra na listagem
                      # de arquivos, não pode travar a página por causa dele.

# Ano plausível de lançamento: 19xx ou 20xx, isolado de outros dígitos. Evita
# confundir com resolução ("1080p", "2160p") ou tamanho ("700mb").
YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


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


def lookup_for_file(path, api_key, *, session=None, timeout=DEFAULT_TIMEOUT):
    """Reconhece o filme de um arquivo de mídia pelo nome.

    Returns:
        ``{"titulo", "ano", "poster", "idioma_original"}`` quando encontra
        algo razoável, ou ``None`` — seja por falta de chave, nome
        irreconhecível, falha de rede, ou nenhum resultado no TMDB.
    """
    if not api_key:
        return None

    titulo_bruto, ano = guess_title_and_year(path)
    if not titulo_bruto:
        return None

    resultado = search_movie(titulo_bruto, api_key, year=ano, session=session,
                             timeout=timeout)
    if not resultado:
        return None

    idioma = resultado.get("original_language")
    return {
        "titulo": resultado.get("title") or titulo_bruto,
        "ano": _release_year(resultado.get("release_date")),
        "poster": _poster_url(resultado.get("poster_path")),
        "idioma_original": idioma,
        "idioma_original_nome": language_name(idioma) if idioma else None,
    }


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
    """Esvazia o cache. Usado pelos testes; útil também se o usuário renomear
    o arquivo para o nome certo depois de um reconhecimento errado."""
    _cache.clear()
