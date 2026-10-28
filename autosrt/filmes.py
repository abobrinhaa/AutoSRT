"""Ferramentas de catalogação do HD de filmes (fora da pasta de trabalho do
AutoSRT -- este é um acervo separado, montado em :func:`base_dir`).

Duas operações, pensadas para rodar pela fila de trabalhos da web
(:mod:`autosrt.jobs`) por serem lentas demais (minutos, por causa do
sistema de arquivos NTFS do HD) para caber numa requisição só:

``atualizar_filmes``
    Gera ``.nfo`` (metadados que o Jellyfin lê) para todo filme que ainda
    não tem um, buscando título/sinopse/ano no TMDB. Não toca em quem já
    tem ``.nfo`` -- ver ``atualizar_generos`` para esses.

``atualizar_generos``
    O gênero de cada filme é o nome da pasta onde ele está (a pasta é a
    taxonomia real do acervo, não o que o TMDB chama de gênero -- ver
    comentário em :func:`nfo_xml`). Varre os ``.nfo`` que já existem e
    corrige só a tag ``<genre>`` de quem estiver com o gênero errado ou
    faltando, sem tocar no resto do arquivo. Depois confere o gênero no
    próprio Jellyfin e corrige lá também (ver
    :func:`_corrigir_generos_jellyfin`) -- só o ``.nfo`` não basta.

``atualizar_legendas``
    Gera, ao lado de cada ``.srt`` do acervo, uma cópia ``.pt.ass`` com
    estilo fixo (cor e tamanho -- ver :data:`LEGENDA_COR`/:data:`LEGENDA_TAMANHO`).
    Existe porque o cliente Jellyfin do Fire Stick novo (Vega OS, app
    "Astra") ainda não tem tela de aparência de legenda, e a configuração de
    legenda do servidor só vale para quem lê essa configuração -- o Astra
    não lê. ``.ass`` carrega o estilo dentro do próprio arquivo, então
    funciona em qualquer cliente decente, sem depender de nada no servidor
    nem no app. O ``.srt`` original nunca é tocado nem removido -- os dois
    ficam como opções de legenda no Jellyfin.

Esta lógica é uma adaptação de ``batch_filmes.py``/``fix_genero_pendentes.py``
(``/root/jeannie-organizer``), que rodavam à mão no host. Reaproveitada aqui
porque o AutoSRT já tem fila de trabalhos com progresso/cancelamento, chave
TMDB e cache em :mod:`autosrt.config` -- resolvendo o problema de processo
zumbi que rodar esses scripts direto no host vinha causando.
"""

import json
import re
import time
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests

from . import config

#: Pastas que não são gênero de verdade, nunca varridas.
SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information", "metadata"}

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".m4v", ".wmv", ".mov", ".rmvb",
              ".divx", ".mpg", ".mpeg", ".ts", ".flv", ".webm", ".ogv"}

YEAR_TOLERANCE = 1
TITLE_RATIO_MIN = 0.55
#: Sem ano extraído do nome não há checagem cruzada por ano, então o ratio
#: de título precisa ser bem mais alto para não aceitar um match qualquer.
TITLE_RATIO_MIN_NO_YEAR = 0.90

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{id}"
TMDB_TIMEOUT = 10

CACHE_FILENAME = "tmdb_cache_filmes.json"
REVIEW_FILENAME = "filmes_review.json"


def base_dir() -> Path:
    """Onde está o acervo de filmes (ver :func:`autosrt.config.get_filmes_dir`)."""
    return Path(config.get_filmes_dir())


class SemChaveTMDB(Exception):
    """Sem chave TMDB configurada -- nenhuma das duas ferramentas funciona
    sem ela, já que ambas dependem de identificar o filme."""


class JellyfinNaoConfigurado(Exception):
    """Sem endereço/chave do Jellyfin configurados -- ver
    :func:`avisar_jellyfin`."""


class JellyfinFalhou(Exception):
    """O Jellyfin recusou ou não respondeu ao pedido de reler a pasta."""


def avisar_jellyfin() -> dict:
    """Pede para o Jellyfin reler a pasta de filmes agora, em vez de
    esperar ele notar sozinho (nem sempre nota -- é o motivo desta
    ferramenta existir: filme jogado no HD que não aparecia na lista).

    Só dispara o pedido; o próprio Jellyfin faz a releitura no fundo,
    fora desta chamada -- por isso devolve na hora, sem fila de trabalho.
    """
    url = config.get_jellyfin_url()
    chave = config.get_jellyfin_api_key()
    if not url or not chave:
        raise JellyfinNaoConfigurado(
            "Falta configurar o endereço e a chave do Jellyfin "
            "(jellyfin_url/jellyfin_api_key no config.json do servidor).")

    try:
        resposta = requests.post(f"{url.rstrip('/')}/Library/Refresh",
                                 params={"api_key": chave}, timeout=10)
    except requests.RequestException as exc:
        raise JellyfinFalhou(f"Não consegui falar com o Jellyfin: {exc}") from exc

    if resposta.status_code >= 300:
        raise JellyfinFalhou(f"O Jellyfin recusou o pedido (HTTP {resposta.status_code}).")
    return {"ok": True}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def _norm_text(s: str) -> str:
    return _strip_accents(s).lower().strip()


def title_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio()


#: Ruído de nome de release que não faz parte do título.
_NOISE_TOKENS = re.compile(
    r"\b("
    r"DUBLADO|LEGENDADO|DUAL[\s\-]?AUDIO|NACIONAL|"
    r"\d{3,4}p|4K|HDTV|WEB[\-\s]?RIP|WEBRIP|BLU[\-\s]?RAY|BLURAY|BDRIP|DVDRIP|DVD|TVR?RIP|"
    r"X264|X265|H264|H265|HEVC|AAC|AC3|REMUX|UPSCALE|"
    r"DOCUMENT[AÁ]RIO"
    r")\b", re.IGNORECASE)


def extract_title_year(name: str) -> Optional[tuple]:
    """Extrai ``(titulo, ano)`` do nome de um arquivo/pasta de filme.

    Cobre os padrões vistos no acervo: ``Titulo - 1944 DUBLADO.mkv``,
    ``Titulo (1996) - DVDRip.mkv``, ``Titulo.1994.1080p.BluRay.x264-GRUPO.mkv``.
    ``ano`` pode vir ``None`` quando não há um no nome; devolve ``None``
    (não uma tupla) quando nem título dá para extrair.
    """
    if re.fullmatch(r"(19|20)\d{2}", name.strip()):
        return None

    working = name
    if working.count(".") > working.count(" "):
        working = working.replace(".", " ")

    year = None
    title_part = working

    match = re.search(r"\((19|20)(\d{2})\)", working)
    if match:
        year = int(match.group(1) + match.group(2))
        title_part = working[:match.start()]
    else:
        candidatos = list(re.finditer(r"(?<!\d)(19|20)\d{2}(?!\d)", working))
        if candidatos:
            match = candidatos[-1]
            year = int(match.group(0))
            title_part = working[:match.start()]

    title_part = _NOISE_TOKENS.sub(" ", title_part)
    title_part = re.sub(r"[\-–_.]+$", " ", title_part)
    title_part = re.sub(r"\s+", " ", title_part).strip(" -–_,")

    if not title_part:
        return None
    return title_part, year


def nfo_xml(title, original_title, plot, genres, premiered, year, tmdb_id,
            tmdb_genres=None) -> str:
    def esc(s):
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # <genre> = nome da pasta (fonte da verdade da classificação do acervo
    # -- inclui categorias como CULT/EPICOS que o TMDB nem conhece). O
    # gênero que o TMDB retorna não vai em <genre>, vai em <tag>, só pra
    # não perder essa informação original do filme.
    genres_xml = "".join(f"  <genre>{esc(g)}</genre>\n" for g in genres)
    tags_xml = "".join(f"  <tag>{esc(g)}</tag>\n" for g in (tmdb_genres or []))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<movie>
  <title>{esc(title)}</title>
  <originaltitle>{esc(original_title)}</originaltitle>
  <plot>{esc(plot)}</plot>
{genres_xml}{tags_xml}  <premiered>{esc(premiered)}</premiered>
  <year>{esc(str(year) if year else "")}</year>
  <tmdbid>{tmdb_id}</tmdbid>
</movie>
"""


class MovieUnit:
    def __init__(self, kind, video_path: Path, nfo_path: Path, name_source: str,
                 genre_folder: str):
        self.kind = kind  # "file", "folder" ou "dvd"
        self.video_path = video_path
        self.nfo_path = nfo_path
        self.name_source = name_source
        self.genre_folder = genre_folder


def discover_units(diretorio: Path = None):
    """Varre o acervo e separa cada filme numa unidade (vídeo + onde o
    ``.nfo`` dele deveria ficar). Pastas com 0 ou vários vídeos (exceto
    DVD, identificado pelos ``.vob``) vão para a revisão em vez de
    adivinhar qual arquivo é o filme."""
    diretorio = diretorio or base_dir()
    units = []
    review = []
    if not diretorio.is_dir():
        return units, review

    for genre_dir in sorted(diretorio.iterdir()):
        if not genre_dir.is_dir() or genre_dir.name in SKIP_DIRS:
            continue
        for entry in sorted(genre_dir.iterdir()):
            if entry.is_file():
                if entry.suffix.lower() in VIDEO_EXTS:
                    units.append(MovieUnit("file", entry, entry.with_suffix(".nfo"),
                                           entry.stem, genre_dir.name))
                continue
            if entry.is_dir():
                videos = [f for f in entry.iterdir()
                         if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
                if len(videos) == 1:
                    units.append(MovieUnit("folder", videos[0], entry / "movie.nfo",
                                           entry.name, genre_dir.name))
                elif len(videos) == 0:
                    vobs = list(entry.rglob("*.[Vv][Oo][Bb]"))
                    if vobs:
                        units.append(MovieUnit("dvd", vobs[0], entry / "movie.nfo",
                                               entry.name, genre_dir.name))
                    else:
                        review.append({"path": str(entry),
                                      "motivo": "pasta sem vídeo direto (pode ter subpastas)"})
                else:
                    review.append({
                        "path": str(entry),
                        "motivo": f"pasta com {len(videos)} vídeos — não processado automaticamente",
                    })
    return units, review


class _TMDBClient:
    """Cliente TMDB minimalista com cache em disco, só o necessário para
    reconhecer filmes. Cache guardado junto do ``config.json`` -- sobrevive
    a troca de branch e fica fora do git."""

    def __init__(self, api_key: str, cache_path: Path):
        self.api_key = api_key
        self.cache_path = cache_path
        self._cache = self._load()

    def _load(self) -> dict:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False),
                                   encoding="utf-8")

    def search_movie(self, query, year=None):
        chave = f"search_{query}_{year}"
        if chave in self._cache:
            return self._cache[chave]
        params = {"api_key": self.api_key, "query": query, "language": "pt-BR"}
        if year:
            params["year"] = year
        try:
            resposta = requests.get(TMDB_SEARCH_URL, params=params, timeout=TMDB_TIMEOUT)
            resposta.raise_for_status()
            resultados = resposta.json().get("results", [])
        except requests.RequestException:
            return []
        self._cache[chave] = resultados
        self._save()
        return resultados

    def get_movie_details(self, movie_id, language="pt-BR"):
        chave = f"details_{movie_id}_{language}"
        if chave in self._cache:
            return self._cache[chave]
        try:
            resposta = requests.get(TMDB_MOVIE_URL.format(id=movie_id),
                                    params={"api_key": self.api_key, "language": language},
                                    timeout=TMDB_TIMEOUT)
            resposta.raise_for_status()
            dados = resposta.json()
        except requests.RequestException:
            return {}
        self._cache[chave] = dados
        self._save()
        return dados


def _tmdb_client() -> _TMDBClient:
    chave = config.get_tmdb_api_key()
    if not chave:
        raise SemChaveTMDB(
            "Falta configurar o acesso à base de dados de filmes (aba "
            "Tradução, nas Configurações) -- sem isso não dá para "
            "descobrir o nome, resumo e ano de cada filme.")
    return _TMDBClient(chave, Path(config.app_directory()) / CACHE_FILENAME)


def best_match(tmdb: _TMDBClient, title: str, year: Optional[int]):
    resultados = tmdb.search_movie(title, year=year)
    if not resultados and year:
        resultados = tmdb.search_movie(title, year=None)
    if not resultados:
        return None, 0.0, False

    pontuados = []
    for r in resultados:
        ratio = max(title_ratio(title, r.get("title", "")),
                   title_ratio(title, r.get("original_title", "")))
        r_year = None
        if r.get("release_date"):
            try:
                r_year = int(r["release_date"][:4])
            except ValueError:
                pass
        year_ok = year is None or (r_year is not None and abs(r_year - year) <= YEAR_TOLERANCE)
        pontuados.append((ratio, year_ok, r))

    com_ano_ok = [p for p in pontuados if p[1]]
    pool = com_ano_ok if com_ano_ok else pontuados
    pool.sort(key=lambda p: p[0], reverse=True)
    ratio, year_ok, candidato = pool[0]
    return candidato, ratio, year_ok


def atualizar_filmes(execute: bool, job=None, sleep_s: float = 0.1) -> dict:
    """Gera ``.nfo`` para todo filme do acervo que ainda não tem um.

    ``execute=False`` só simula: diz o que escreveria, sem gravar nada no
    HD -- é o padrão também em ``batch_filmes.py``, porque isso roda sobre
    um acervo de 13 TB e um engano custa caro para refazer.
    """
    def etapa(msg):
        if job is not None:
            job.etapa = msg

    def cancelado() -> bool:
        return job is not None and job.cancelar.is_set()

    etapa("Procurando filmes no acervo...")
    tmdb = _tmdb_client()
    units, review = discover_units()
    pendentes = [u for u in units if not u.nfo_path.exists()]

    relatorio = {"unidades_total": len(units), "ja_tinha_nfo": len(units) - len(pendentes),
                "processados": 0, "escritos": 0, "sem_extracao": 0, "sem_match": 0}

    for i, unit in enumerate(pendentes):
        if cancelado():
            break
        etapa(f"[{unit.genre_folder}] {unit.name_source}")
        if job is not None:
            job.progresso = int(i * 100 / len(pendentes)) if pendentes else 100

        extraido = extract_title_year(unit.name_source)
        if not extraido:
            # Sem título/ano extraído do nome: grava um .nfo mínimo mesmo
            # assim (só gênero = pasta + título bruto). O gênero de um
            # filme tem que estar certo em TODO filme do acervo, com
            # informação de sobra ou nenhuma -- não dá pra deixar o
            # Jellyfin cair pro gênero da internet só porque faltou dado
            # pra enriquecer com título/sinopse (ver autosrt-beta #ferramentas-filmes).
            relatorio["sem_extracao"] += 1
            review.append({"path": str(unit.video_path),
                          "motivo": "não consegui extrair título/ano do nome -- nfo mínimo gravado (só gênero + título bruto)"})
            nfo_content = nfo_xml(
                title=unit.name_source, original_title=unit.name_source, plot="",
                genres=[unit.genre_folder], premiered="", year=None, tmdb_id="")
            if execute:
                try:
                    unit.nfo_path.write_text(nfo_content, encoding="utf-8")
                    relatorio["escritos"] += 1
                except OSError as exc:
                    review.append({"path": str(unit.video_path), "motivo": f"erro ao escrever nfo mínimo: {exc}"})
            else:
                relatorio["escritos"] += 1
            continue
        title, year = extraido

        candidato, ratio, year_ok = best_match(tmdb, title, year)
        relatorio["processados"] += 1

        min_ratio = TITLE_RATIO_MIN if year is not None else TITLE_RATIO_MIN_NO_YEAR
        if not candidato or ratio < min_ratio or (year is not None and not year_ok):
            # Sem match confiável: mesma lógica -- grava o .nfo com o
            # gênero certo e o título que deu pra extrair do nome do
            # arquivo, sem sinopse/tmdbid. Fica no review pra revisão
            # manual depois, mas o gênero não fica esperando essa revisão.
            relatorio["sem_match"] += 1
            review.append({
                "path": str(unit.video_path),
                "titulo_extraido": title, "ano_extraido": year,
                "melhor_candidato": candidato.get("title") if candidato else None,
                "ratio": round(ratio, 2),
                "motivo": "sem match confiável no TMDB -- nfo mínimo gravado (título do nome do arquivo, sem sinopse/tmdbid)",
            })
            nfo_content = nfo_xml(
                title=title, original_title=title, plot="",
                genres=[unit.genre_folder], premiered=str(year) if year else "",
                year=year, tmdb_id="")
            if execute:
                try:
                    unit.nfo_path.write_text(nfo_content, encoding="utf-8")
                    relatorio["escritos"] += 1
                except OSError as exc:
                    review.append({"path": str(unit.video_path), "motivo": f"erro ao escrever nfo mínimo: {exc}"})
            else:
                relatorio["escritos"] += 1
            time.sleep(sleep_s)
            continue

        detalhes = tmdb.get_movie_details(candidato["id"], "pt-BR")
        if not detalhes.get("overview"):
            detalhes_en = tmdb.get_movie_details(candidato["id"], "en-US")
            detalhes = {**detalhes_en, **{k: v for k, v in detalhes.items() if v}}

        nfo_content = nfo_xml(
            title=detalhes.get("title") or candidato.get("title") or title,
            original_title=detalhes.get("original_title") or candidato.get("original_title") or title,
            plot=detalhes.get("overview", ""),
            genres=[unit.genre_folder],
            tmdb_genres=[g["name"] for g in detalhes.get("genres", [])],
            premiered=detalhes.get("release_date", ""),
            year=year or (detalhes.get("release_date", "")[:4] or None),
            tmdb_id=candidato["id"],
        )

        if execute:
            try:
                unit.nfo_path.write_text(nfo_content, encoding="utf-8")
                relatorio["escritos"] += 1
            except OSError as exc:
                review.append({"path": str(unit.video_path), "motivo": f"erro ao escrever nfo: {exc}"})
        else:
            relatorio["escritos"] += 1

        time.sleep(sleep_s)

    Path(config.app_directory(), REVIEW_FILENAME).write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    relatorio["revisao"] = len(review)
    relatorio["simulado"] = not execute
    return relatorio


_GENRE_RE = re.compile(r"<genre>(.*?)</genre>", re.IGNORECASE | re.DOTALL)
_MOVIE_TAG_RE = re.compile(r"<movie[^>]*>", re.IGNORECASE)


def _corrigir_genero(nfo_text: str, genero_certo: str):
    """Corrige a(s) tag(s) ``<genre>`` de um ``.nfo`` para ``genero_certo``.

    Devolve ``(texto_novo, mudou)``. Não toca em nada além da(s) tag(s)
    ``<genre>`` -- título, sinopse, ano, tmdbid etc. ficam exatamente como
    estavam, byte a byte, porque esses ``.nfo`` podem ter vindo de outra
    fonte além deste próprio gerador.
    """
    atuais = _GENRE_RE.findall(nfo_text)
    if atuais == [genero_certo]:
        return nfo_text, False

    sem_genero = _GENRE_RE.sub("", nfo_text)
    # Sobra de linha em branco onde a tag removida era a única coisa na
    # linha -- limpa para não acumular linhas vazias a cada correção.
    sem_genero = re.sub(r"\n[ \t]*\n", "\n", sem_genero)

    nova_tag = f"  <genre>{genero_certo}</genre>\n"
    match = _MOVIE_TAG_RE.search(sem_genero)
    if not match:
        return nfo_text, False  # não parece um .nfo de filme, não arrisca

    posicao = match.end()
    novo_texto = sem_genero[:posicao] + "\n" + nova_tag + sem_genero[posicao:].lstrip("\n")
    return novo_texto, True


def atualizar_generos(execute: bool, job=None) -> dict:
    """Corrige o ``<genre>`` dos ``.nfo`` que já existem no acervo, para o
    nome da pasta onde cada um está -- a regra atual (ver :func:`nfo_xml`),
    mas que só vale para ``.nfo`` novo. Quem já tinha ``.nfo`` de antes
    dessa regra (ou de outra fonte) não é tocado pela outra ferramenta.
    """
    def etapa(msg):
        if job is not None:
            job.etapa = msg

    def cancelado() -> bool:
        return job is not None and job.cancelar.is_set()

    diretorio = base_dir()
    etapa("Procurando .nfo no acervo...")
    nfos = []
    if diretorio.is_dir():
        for genre_dir in sorted(diretorio.iterdir()):
            if not genre_dir.is_dir() or genre_dir.name in SKIP_DIRS:
                continue
            for caminho in genre_dir.rglob("*.nfo"):
                nfos.append((caminho, genre_dir.name))

    relatorio = {"nfo_total": len(nfos), "corrigidos": 0, "ja_corretos": 0, "ignorados": 0}

    for i, (caminho, genero) in enumerate(nfos):
        if cancelado():
            break
        etapa(f"[{genero}] {caminho.name}")
        if job is not None:
            job.progresso = int(i * 100 / len(nfos)) if nfos else 100

        try:
            texto = caminho.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            relatorio["ignorados"] += 1
            continue

        novo_texto, mudou = _corrigir_genero(texto, genero)
        if not mudou:
            relatorio["ja_corretos"] += 1
            continue

        relatorio["corrigidos"] += 1
        if execute:
            caminho.write_text(novo_texto, encoding="utf-8")

    if not cancelado():
        try:
            relatorio.update(_corrigir_generos_jellyfin(execute, diretorio, etapa, cancelado))
        except (JellyfinNaoConfigurado, JellyfinFalhou) as exc:
            relatorio["jellyfin_erro"] = str(exc)

    relatorio["simulado"] = not execute
    return relatorio


def _pasta_genero(caminho: str, diretorio: Path) -> Optional[str]:
    """Nome da pasta de gênero de um filme, a partir do caminho que o
    Jellyfin guarda (``/mnt/filmes_storage/TERROR/x/x.mkv`` -> ``TERROR``).
    ``None`` se o filme não é do HD. Normaliza para NFC dos dois lados: o
    mesmo "Ç" pode vir em bytes diferentes e o nome parecer igual sem ser."""
    base = unicodedata.normalize("NFC", str(diretorio)).rstrip("/") + "/"
    caminho = unicodedata.normalize("NFC", caminho or "")
    if not caminho.startswith(base):
        return None
    pasta = caminho[len(base):].split("/")[0]
    return pasta if pasta and pasta not in SKIP_DIRS else None


def _corrigir_generos_jellyfin(execute: bool, diretorio: Path, etapa, cancelado) -> dict:
    """Deixa o gênero de cada filme *no Jellyfin* igual ao nome da pasta.

    Corrigir o ``.nfo`` não basta: o Jellyfin não troca um gênero que o
    filme já tem só porque o ``.nfo`` mudou (nem na releitura normal), não
    lê ``movie.nfo`` em pasta com dois vídeos, e ignora o ``.nfo`` de vez
    no filme com o campo Gênero travado -- e milhares estavam travados.
    Por isso grava direto pela API: tira a trava de Gênero (as outras
    travas ficam) e põe o nome da pasta. O resto do item volta como veio.
    """
    url = config.get_jellyfin_url()
    chave = config.get_jellyfin_api_key()
    if not url or not chave:
        raise JellyfinNaoConfigurado(
            "Falta configurar o endereço e a chave do Jellyfin "
            "(jellyfin_url/jellyfin_api_key no config.json do servidor).")
    url = url.rstrip("/")
    cabecalho = {"X-Emby-Token": chave}

    etapa("Conferindo os gêneros no Jellyfin...")
    try:
        usuarios = requests.get(f"{url}/Users", headers=cabecalho, timeout=30)
        usuarios.raise_for_status()
        admin = next((u["Id"] for u in usuarios.json()
                      if u.get("Policy", {}).get("IsAdministrator")), None)
        lista = requests.get(f"{url}/Items", headers=cabecalho, timeout=300, params={
            "Recursive": "true", "IncludeItemTypes": "Movie", "Fields": "Path,Genres"})
        lista.raise_for_status()
        itens = lista.json().get("Items", [])
    except (requests.RequestException, ValueError) as exc:
        raise JellyfinFalhou(f"Não consegui falar com o Jellyfin: {exc}") from exc
    if admin is None:
        raise JellyfinFalhou("A chave do Jellyfin não enxerga nenhum usuário administrador.")

    errados = []
    filmes_hd = 0
    for item in itens:
        pasta = _pasta_genero(item.get("Path"), diretorio)
        if pasta is None:
            continue
        filmes_hd += 1
        if item.get("Genres") != [pasta]:
            errados.append((item["Id"], pasta))

    relatorio = {"jellyfin_filmes": filmes_hd, "jellyfin_corrigidos": 0, "jellyfin_falhas": 0}
    if not execute:
        relatorio["jellyfin_corrigidos"] = len(errados)
        return relatorio

    for i, (item_id, pasta) in enumerate(errados):
        if cancelado():
            break
        etapa(f"Jellyfin: corrigindo gênero ({i + 1}/{len(errados)})")
        try:
            # Item completo (com LockedFields): é ele que volta no POST.
            atual = requests.get(f"{url}/Items/{item_id}", headers=cabecalho,
                                 params={"userId": admin}, timeout=60)
            atual.raise_for_status()
            item = atual.json()
            travas = [f for f in item.get("LockedFields", []) if f != "Genres"]
            # A listagem geral do Jellyfin às vezes vem atrasada; o item
            # sozinho é a verdade, então não regrava quem já está certo.
            if item.get("Genres") == [pasta] and travas == item.get("LockedFields", []):
                continue
            item["Genres"] = [pasta]
            item["LockedFields"] = travas
            item.pop("GenreItems", None)  # ainda traz os gêneros antigos
            resposta = requests.post(f"{url}/Items/{item_id}", headers=cabecalho,
                                     json=item, timeout=60)
            resposta.raise_for_status()
        except (requests.RequestException, ValueError):
            relatorio["jellyfin_falhas"] += 1
            continue
        relatorio["jellyfin_corrigidos"] += 1
    return relatorio


_PLOT_RE = re.compile(r"<plot>(.*?)</plot>", re.IGNORECASE | re.DOTALL)


def _nfo_sem_sinopse(caminho: Path) -> bool:
    """Diz se o ``.nfo`` não tem sinopse -- ``<plot>`` ausente, vazio, ou
    arquivo que não deu pra ler (nesse caso não mexe, então conta como
    "tem", pra não tentar de novo toda vez)."""
    try:
        texto = caminho.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    match = _PLOT_RE.search(texto)
    return not match or not match.group(1).strip()


def atualizar_sinopses(execute: bool, job=None, sleep_s: float = 0.1) -> dict:
    """Preenche sinopse, título e ano de filme que já tem ``.nfo`` mas
    ficou incompleto -- gerado sem TMDB (``atualizar_generos`` corrige só
    o gênero desses, nunca mexeu na sinopse) ou com um match que falhou na
    época e nunca foi tentado de novo.

    NUNCA mexe no gênero: o `.nfo` é reescrito inteiro, mas o ``<genre>``
    vem de novo de ``unit.genre_folder`` (a pasta onde o filme está hoje),
    nunca lido do arquivo antigo -- é a mesma fonte da verdade que
    :func:`atualizar_generos` usa, só que aplicada aqui de propósito pra
    um reescrever não arriscar perder o que o outro corrigiu.
    """
    def etapa(msg):
        if job is not None:
            job.etapa = msg

    def cancelado() -> bool:
        return job is not None and job.cancelar.is_set()

    etapa("Procurando filmes com ficha incompleta...")
    tmdb = _tmdb_client()
    units, _ = discover_units()
    pendentes = [u for u in units
                if u.nfo_path.exists() and _nfo_sem_sinopse(u.nfo_path)]

    relatorio = {"fichas_incompletas": len(pendentes), "processados": 0,
                "atualizados": 0, "sem_extracao": 0, "sem_match": 0}
    review = []

    for i, unit in enumerate(pendentes):
        if cancelado():
            break
        etapa(f"[{unit.genre_folder}] {unit.name_source}")
        if job is not None:
            job.progresso = int(i * 100 / len(pendentes)) if pendentes else 100

        extraido = extract_title_year(unit.name_source)
        if not extraido:
            relatorio["sem_extracao"] += 1
            continue
        title, year = extraido

        candidato, ratio, year_ok = best_match(tmdb, title, year)
        relatorio["processados"] += 1

        min_ratio = TITLE_RATIO_MIN if year is not None else TITLE_RATIO_MIN_NO_YEAR
        if not candidato or ratio < min_ratio or (year is not None and not year_ok):
            relatorio["sem_match"] += 1
            review.append({
                "path": str(unit.video_path),
                "titulo_extraido": title, "ano_extraido": year,
                "melhor_candidato": candidato.get("title") if candidato else None,
                "ratio": round(ratio, 2),
                "motivo": "sem match confiável no TMDB -- sinopse continua vazia",
            })
            time.sleep(sleep_s)
            continue

        detalhes = tmdb.get_movie_details(candidato["id"], "pt-BR")
        if not detalhes.get("overview"):
            detalhes_en = tmdb.get_movie_details(candidato["id"], "en-US")
            detalhes = {**detalhes_en, **{k: v for k, v in detalhes.items() if v}}

        nfo_content = nfo_xml(
            title=detalhes.get("title") or candidato.get("title") or title,
            original_title=detalhes.get("original_title") or candidato.get("original_title") or title,
            plot=detalhes.get("overview", ""),
            genres=[unit.genre_folder],
            tmdb_genres=[g["name"] for g in detalhes.get("genres", [])],
            premiered=detalhes.get("release_date", ""),
            year=year or (detalhes.get("release_date", "")[:4] or None),
            tmdb_id=candidato["id"],
        )

        if execute:
            try:
                unit.nfo_path.write_text(nfo_content, encoding="utf-8")
                relatorio["atualizados"] += 1
            except OSError as exc:
                review.append({"path": str(unit.video_path), "motivo": f"erro ao escrever nfo: {exc}"})
        else:
            relatorio["atualizados"] += 1

        time.sleep(sleep_s)

    if review:
        Path(config.app_directory(), "filmes_sinopses_review.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    relatorio["revisao"] = len(review)
    relatorio["simulado"] = not execute
    return relatorio


#: Onde fica o registro de quando o worker automático (ver
#: :func:`autosrt.web._worker_filmes_periodico`) rodou pela última vez.
#: Fora do git de propósito, igual o cache do TMDB -- é estado, não código.
ULTIMA_EXECUCAO_FILENAME = "filmes_worker_ultima_execucao.json"

#: O container roda em UTC (``timedatectl`` confirma), mas quem lê a hora
#: é gente no Brasil -- hora gravada já em horário local, pra bater com o
#: resto do homelab (ver TZ em homelab-info/app.py) sem depender de quem
#: olha o JSON cru saber converter fuso.
_TZ = ZoneInfo("America/Sao_Paulo")


def registrar_execucao_automatica(resultados: dict) -> None:
    """Grava quando o worker de 2 em 2 horas rodou pela última vez, e o
    resultado de cada ferramenta que ele chamou nessa rodada.

    Em arquivo, não em memória: o processo da web reinicia (deploy, queda
    de energia) e perderia isso se fosse só uma variável -- e é justamente
    esse "quando foi a última vez" que a tela precisa mostrar pro usuário
    confiar que o worker está rodando sozinho de verdade.
    """
    caminho = Path(config.app_directory()) / ULTIMA_EXECUCAO_FILENAME
    caminho.parent.mkdir(parents=True, exist_ok=True)
    dados = {
        "quando": datetime.now(_TZ).isoformat(timespec="seconds"),
        "resultados": resultados,
    }
    caminho.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")


def ultima_execucao_automatica() -> Optional[dict]:
    """Devolve o último registro de :func:`registrar_execucao_automatica`,
    ou ``None`` se o worker ainda não rodou nenhuma vez (processo acabou
    de subir, ou ainda não passaram as 2 horas da primeira rodada)."""
    caminho = Path(config.app_directory()) / ULTIMA_EXECUCAO_FILENAME
    if not caminho.exists():
        return None
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


#: Amarelo, tamanho grande -- o que o usuário pediu depois de testar no
#: Astra (Fire Stick / Vega OS). Mude aqui se quiser outro padrão; já
#: existente não é regerado, então um valor novo só vale para o que for
#: convertido dali em diante (ver :func:`atualizar_legendas`).
LEGENDA_COR = "FFFF00"
LEGENDA_TAMANHO = 64

_SRT_TEMPO_RE = re.compile(
    r"(\d\d:\d\d:\d\d,\d{3})\s*-->\s*(\d\d:\d\d:\d\d,\d{3})")
_SRT_TAG_RE = re.compile(r"</?[ibu]>", re.IGNORECASE)
#: Ordem de tentativa para abrir um .srt de origem desconhecida -- a
#: maioria do acervo é utf-8, mas tem gente antiga em cp1252/latin-1 (ver
#: comentário em :func:`_ler_srt`). latin-1 nunca falha (mapeia todo byte),
#: por isso fica por último, como rede de segurança.
_SRT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def _ler_srt(caminho: Path) -> Optional[str]:
    """Lê um ``.srt`` testando as codificações de :data:`_SRT_ENCODINGS`
    em ordem. ``None`` só se o arquivo não puder nem ser aberto (latin-1
    sempre decodifica algo, mesmo que errado)."""
    for codificacao in _SRT_ENCODINGS:
        try:
            return caminho.read_text(encoding=codificacao)
        except (UnicodeDecodeError, OSError):
            continue
    return None


def _tempo_srt_para_ass(tempo: str) -> str:
    """``00:00:04,770`` -> ``0:00:04.77`` (.ass usa centésimos, não
    milésimos, e não tem zero à esquerda na hora)."""
    h, m, resto = tempo.split(":")
    s, ms = resto.split(",")
    cs = round(int(ms) / 10)
    return f"{int(h)}:{m}:{s}.{cs:02d}"


def _cor_ass(rgb_hex: str) -> str:
    """``RRGGBB`` -> ``&H00BBGGRR``, a ordem (e o alfa invertido, 00 =
    opaco) que o formato .ass usa para cor."""
    r, g, b = rgb_hex[0:2], rgb_hex[2:4], rgb_hex[4:6]
    return f"&H00{b}{g}{r}".upper()


def _srt_para_ass(texto_srt: str, cor_hex: str, tamanho: int) -> tuple[str, int]:
    """Converte o texto de um ``.srt`` para ``.ass`` com estilo fixo.
    Devolve ``(texto_ass, quantidade_de_falas)`` -- 0 falas quer dizer que
    não achou nenhum bloco reconhecível (arquivo vazio ou formato
    inesperado), e quem chama trata isso como "não deu pra converter"."""
    falas = []
    for bloco in re.split(r"\r?\n\r?\n+", texto_srt.strip()):
        linhas = bloco.strip("\ufeff \r\n").splitlines()
        if len(linhas) < 2:
            continue
        # a 1a linha é o número sequencial do srt; às vezes falta
        i = 1 if linhas[0].strip().isdigit() else 0
        if i >= len(linhas):
            continue
        tempos = _SRT_TEMPO_RE.search(linhas[i])
        if not tempos:
            continue
        corpo = "\\N".join(linhas[i + 1:]).strip()
        corpo = _SRT_TAG_RE.sub("", corpo)
        if corpo:
            falas.append((_tempo_srt_para_ass(tempos.group(1)),
                          _tempo_srt_para_ass(tempos.group(2)), corpo))

    cabecalho = f"""[Script Info]
Title: Legenda estilizada (gerada por atualizar_legendas)
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{tamanho},{_cor_ass(cor_hex)},&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,2,30,30,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    linhas_evento = "".join(
        f"Dialogue: 0,{ini},{fim},Default,,0,0,0,,{corpo}\n"
        for ini, fim, corpo in falas)
    return cabecalho + linhas_evento, len(falas)


def atualizar_legendas(execute: bool, job=None,
                       cor: str = LEGENDA_COR, tamanho: int = LEGENDA_TAMANHO) -> dict:
    """Gera, ao lado de cada ``.srt`` do acervo, uma cópia ``.pt.ass`` com
    estilo fixo -- ver o comentário no topo do módulo. Pula quem já tem
    ``.pt.ass`` (não regera: se você já ajustou um filme na mão, ou já
    rodou essa ferramenta antes, ela não desfaz). No fim, se o Jellyfin
    estiver configurado, pede pra ele reler o acervo -- diferente de
    :func:`atualizar_generos`, aqui não precisa mexer item por item: o
    Jellyfin descobre o ``.pt.ass`` novo sozinho, como faz com qualquer
    legenda externa.
    """
    def etapa(msg):
        if job is not None:
            job.etapa = msg

    def cancelado() -> bool:
        return job is not None and job.cancelar.is_set()

    diretorio = base_dir()
    etapa("Procurando .srt no acervo...")
    arquivos = []
    if diretorio.is_dir():
        for genre_dir in sorted(diretorio.iterdir()):
            if not genre_dir.is_dir() or genre_dir.name in SKIP_DIRS:
                continue
            arquivos.extend(genre_dir.rglob("*.srt"))

    relatorio = {"srt_total": len(arquivos), "gerados": 0, "ja_existia": 0, "ignorados": 0}

    for i, caminho in enumerate(arquivos):
        if cancelado():
            break
        etapa(caminho.name)
        if job is not None:
            job.progresso = int(i * 100 / len(arquivos)) if arquivos else 100

        destino = caminho.parent / f"{caminho.stem}.pt.ass"
        if destino.is_file():
            relatorio["ja_existia"] += 1
            continue

        texto = _ler_srt(caminho)
        if texto is None:
            relatorio["ignorados"] += 1
            continue

        texto_ass, n_falas = _srt_para_ass(texto, cor, tamanho)
        if n_falas == 0:
            relatorio["ignorados"] += 1
            continue

        relatorio["gerados"] += 1
        if execute:
            destino.write_text(texto_ass, encoding="utf-8")

    if execute and relatorio["gerados"] and not cancelado():
        try:
            avisar_jellyfin()
            relatorio["jellyfin"] = {"ok": True}
        except (JellyfinNaoConfigurado, JellyfinFalhou) as exc:
            relatorio["jellyfin_erro"] = str(exc)

    relatorio["simulado"] = not execute
    return relatorio
