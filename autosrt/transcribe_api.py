"""Transcrição de áudio via API compatível com OpenAI (nuvem).

Alternativa ao Whisper local (:mod:`autosrt.transcribe`) para quem não tem
GPU. Usa o mesmo endpoint ``/audio/transcriptions`` que a OpenAI documenta,
sempre pedindo ``response_format=verbose_json`` — é o único jeito de a API
devolver ``segments`` com início e fim de cada trecho. Sem esses tempos não
haveria como montar uma legenda de verdade, só um bloco de texto corrido sem
nenhuma sincronia com o áudio; por isso um provedor que não devolve
segmentos é tratado como erro, não como um resultado pior.

Segue a mesma filosofia de "endpoint compatível com OpenAI" do resto do
projeto (ver :mod:`autosrt.llm`): trocar de provedor é só trocar a URL base.
"""

import os
import time

import requests

from .cue import Cue
from .transcribe import TranscriptionError

DEFAULT_TIMEOUT = 300  # segundos. Transcrição de um filme inteiro é lenta.
DEFAULT_MAX_RETRIES = 3

# Códigos que compensa repetir: limite de taxa e falhas temporárias do lado
# do servidor, no mesmo espírito do llm.py.
RETRIABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
}
DEFAULT_MODELS = {
    "openrouter": "openai/whisper-1",
    "openai": "whisper-1",
}


def _segments_to_cues(segments) -> list:
    """Converte os ``segments`` com timestamp da API em :class:`Cue`.

    Os tempos da API vêm em segundos (float); o AutoSRT trabalha em
    milissegundos internamente.
    """
    cues = []
    indice = 0
    for segmento in segments:
        texto = (segmento.get("text") or "").strip()
        if not texto:
            continue
        indice += 1
        inicio_ms = round(float(segmento.get("start", 0)) * 1000)
        fim_ms = round(float(segmento.get("end", 0)) * 1000)
        cues.append(Cue.from_source(index=indice, start=inicio_ms, end=fim_ms,
                                    source_text=texto))
    return cues


def transcribe_via_api(media_path, *, api_key, engine="openrouter",
                       base_url=None, model=None, language=None,
                       timeout=DEFAULT_TIMEOUT, max_retries=DEFAULT_MAX_RETRIES,
                       session=None) -> list:
    """Transcreve um arquivo de mídia pela API e devolve ``list[Cue]``.

    Args:
        media_path: vídeo ou áudio a transcrever.
        api_key: chave do provedor escolhido.
        engine: ``"openrouter"`` ou ``"openai"`` — decide a URL base e o
            nome do modelo padrão. Ignorado quando ``base_url`` é informado.
        base_url: URL base da API, para provedores customizados.
        model: nome do modelo. Sendo ``None``, usa o padrão do provedor.
        language: código do idioma falado, se souber.
        session: sessão de ``requests`` injetável pelos testes.

    Returns:
        Lista de :class:`~autosrt.cue.Cue`, com tempos reais vindos dos
        ``segments`` da API. Não há diarização por essa via — ``speaker``
        fica ``None`` em todas as legendas.

    Raises:
        TranscriptionError: arquivo ausente, chave ausente, falha de rede
            persistente, resposta sem ``segments``, ou nenhuma fala
            reconhecida.
    """
    if not os.path.isfile(media_path):
        raise TranscriptionError(f"Arquivo não encontrado: {media_path}")
    if not api_key:
        raise TranscriptionError("Chave de API não fornecida.")

    url_base = (base_url or BASE_URLS.get(engine) or BASE_URLS["openrouter"]).rstrip("/")
    url = f"{url_base}/audio/transcriptions"
    modelo = model or DEFAULT_MODELS.get(engine, "whisper-1")
    http = session or requests
    headers = {"Authorization": f"Bearer {api_key}"}

    resposta = None
    ultimo_erro = None
    for tentativa in range(max_retries):
        with open(media_path, "rb") as arquivo:
            files = {"file": (os.path.basename(media_path), arquivo)}
            data = {"model": modelo, "response_format": "verbose_json"}
            if language:
                data["language"] = language
            try:
                resposta = http.post(url, headers=headers, files=files,
                                     data=data, timeout=timeout)
            except requests.RequestException as exc:
                resposta = None
                ultimo_erro = f"falha de rede: {exc}"

        if resposta is not None:
            if resposta.status_code == 200:
                break
            detalhe = (resposta.text or "")[:300]
            ultimo_erro = f"HTTP {resposta.status_code}: {detalhe}"
            if resposta.status_code not in RETRIABLE_STATUS:
                raise TranscriptionError(
                    f"A API de transcrição recusou a requisição - {ultimo_erro}")
            resposta = None

        if tentativa < max_retries - 1:
            time.sleep(min(10, 2 ** tentativa))

    if resposta is None:
        raise TranscriptionError(
            f"A API de transcrição falhou após {max_retries} tentativas - {ultimo_erro}")

    try:
        corpo = resposta.json()
    except ValueError:
        raise TranscriptionError("A API de transcrição devolveu uma resposta que não é JSON.")

    segments = corpo.get("segments")
    if not segments:
        raise TranscriptionError(
            "A API não devolveu tempos por trecho (\"segments\"). Sem eles não "
            "é possível montar uma legenda sincronizada; confira se o modelo "
            "escolhido suporta response_format=verbose_json.")

    cues = _segments_to_cues(segments)
    if not cues:
        raise TranscriptionError("O áudio não teve fala reconhecida.")
    return cues
