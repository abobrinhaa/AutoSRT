"""Transcrição via API (OpenRouter/OpenAI) em vez de processamento local.

Vantagens:
- Não precisa de GPU local
- Processamento rápido (via API)
- Escalável (múltiplos arquivos)
- Usa mesma chave OpenRouter da tradução

Uso:
    from autosrt import config, transcribe_api

    # Configurar
    api_key = config.get_openrouter_api_key()

    # Transcrever via API
    texto = transcribe_api.transcribe(
        "audio.mp3",
        api_key=api_key,
        engine="openrouter"
    )
"""

import os
import requests


def transcribe_openrouter(audio_file, api_key, model="openai/whisper-1"):
    """Transcrever áudio via OpenRouter API.

    Args:
        audio_file: Caminho do arquivo de áudio (mp3, wav, m4a, etc)
        api_key: Chave da OpenRouter API
        model: Modelo a usar (padrão: openai/whisper-1)

    Returns:
        str: Texto transcrito

    Raises:
        FileNotFoundError: Se arquivo não existe
        ValueError: Se API retornar erro
        requests.RequestException: Erro de conexão
    """
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_file}")

    if not api_key:
        raise ValueError("Chave de API não fornecida")

    with open(audio_file, "rb") as f:
        files = {"file": (os.path.basename(audio_file), f)}
        headers = {"Authorization": f"Bearer {api_key}"}

        response = requests.post(
            "https://api.openrouter.ai/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data={"model": model},
            timeout=300  # 5 minutos
        )

    if response.status_code != 200:
        raise ValueError(f"Erro da API: {response.status_code} - {response.text}")

    result = response.json()
    if "text" not in result:
        raise ValueError(f"Resposta inesperada da API: {result}")

    return result["text"]


def transcribe_openai(audio_file, api_key):
    """Transcrever áudio via OpenAI API.

    Requer openai library: pip install openai

    Args:
        audio_file: Caminho do arquivo de áudio
        api_key: Chave da OpenAI API

    Returns:
        str: Texto transcrito
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI library não instalada. "
            "Instale com: pip install openai"
        )

    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_file}")

    if not api_key:
        raise ValueError("Chave de API não fornecida")

    client = OpenAI(api_key=api_key)

    with open(audio_file, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )

    return result.text


def transcribe(audio_file, engine="openrouter", api_key=None, **kwargs):
    """Transcrever com opção de engine.

    Args:
        audio_file: Arquivo de áudio
        engine: "openrouter" (padrão), "openai", ou "local"
        api_key: Chave de API (se usar OpenRouter/OpenAI)
        **kwargs: Opções adicionais para o engine

    Returns:
        str: Texto transcrito

    Examples:
        # Via OpenRouter (recomendado)
        texto = transcribe("podcast.mp3", engine="openrouter", api_key="sk-or-...")

        # Via OpenAI
        texto = transcribe("palestra.wav", engine="openai", api_key="sk-...")
    """
    if engine == "openrouter":
        return transcribe_openrouter(audio_file, api_key, **kwargs)
    elif engine == "openai":
        return transcribe_openai(audio_file, api_key)
    elif engine == "local":
        # Fallback para implementação local
        from . import transcribe as transcribe_local
        return transcribe_local.transcribe_file(audio_file, **kwargs)
    else:
        raise ValueError(f"Engine desconhecido: {engine}")
