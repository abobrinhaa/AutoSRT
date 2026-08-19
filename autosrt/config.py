"""Configuração local e chaves de API.

Restrição obrigatória do projeto: nenhuma chave pode ser embutida no código
nem versionada. A leitura acontece, nesta ordem, a partir de:

1. variável de ambiente (``TMDB_API_KEY``);
2. ``config.json`` ao lado do executável, que está no ``.gitignore``.
"""

import json
import os
import sys

CONFIG_FILENAME = "config.json"
TMDB_ENV_VAR = "TMDB_API_KEY"

EXAMPLE_CONFIG = {
    "openrouter_api_key": "sk-or-v1-...",
    "llm_base_url": "https://openrouter.ai/api/v1",
    "llm_model": "deepseek/deepseek-chat-v2.5",
    "faster_whisper_path": "/opt/faster-whisper-xxl/faster-whisper-xxl",
}


def app_directory() -> str:
    """Diretório do aplicativo, funcionando também empacotado com PyInstaller."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_path() -> str:
    return os.path.join(app_directory(), CONFIG_FILENAME)


def load_config() -> dict:
    """Lê o ``config.json``. Devolve dicionário vazio se não existir ou
    estiver corrompido — a ausência de configuração não é erro."""
    path = config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def get_setting(name: str, env_var: str = None, default=None):
    """Busca uma configuração no ambiente e depois no ``config.json``."""
    if env_var:
        value = os.environ.get(env_var)
        if value:
            return value
    value = load_config().get(name)
    return value if value else default


def get_tmdb_api_key():
    """Chave da API do TMDb, ou ``None`` se não estiver configurada."""
    return get_setting("tmdb_api_key", TMDB_ENV_VAR)


def get_openrouter_api_key():
    """Chave do OpenRouter, ou ``None`` se não estiver configurada."""
    return get_setting("openrouter_api_key", "OPENROUTER_API_KEY")


def get_whisper_path():
    """Caminho do executável do Faster-Whisper-XXL, se configurado."""
    return get_setting("faster_whisper_path", "FASTER_WHISPER_PATH")
