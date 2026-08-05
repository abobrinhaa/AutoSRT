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
