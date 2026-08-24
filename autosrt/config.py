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
    "llm_model": "deepseek/deepseek-chat",
    "faster_whisper_path": "/opt/faster-whisper-xxl/faster-whisper-xxl",
    # Opcional: reconhece o filme pelo nome do arquivo na lista do servidor.
    # Chave gratuita em https://www.themoviedb.org/settings/api
    "tmdb_api_key": "...",
}


def app_directory() -> str:
    """Onde fica o ``config.json``.

    A ordem existe porque o pacote instalado pode estar em ``site-packages``,
    que costuma ser somente leitura — gravar configuração ali falharia:

    1. ``AUTOSRT_CONFIG_DIR``, quando o administrador quer decidir;
    2. ao lado do executável, quando empacotado;
    3. ``~/.config/autosrt``, o lugar convencional em Linux.
    """
    escolhido = os.environ.get("AUTOSRT_CONFIG_DIR")
    if escolhido:
        return escolhido
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)

    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config")
    return os.path.join(base, "autosrt")


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


def save_config(data: dict) -> str:
    """Grava o ``config.json``, preservando o que já estava lá.

    Chaves com valor vazio ou ``None`` são removidas, o que permite limpar
    uma configuração sem editar o arquivo à mão.

    Returns:
        O caminho do arquivo gravado.
    """
    atual = load_config()
    for chave, valor in data.items():
        if valor in (None, ""):
            atual.pop(chave, None)
        else:
            atual[chave] = valor

    caminho = config_path()
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as handle:
        json.dump(atual, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    # A chave da API não é assunto de ninguém além do dono do servidor.
    try:
        os.chmod(caminho, 0o600)
    except OSError:
        pass
    return caminho


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


def get_openai_api_key():
    """Chave da OpenAI, ou ``None`` se não estiver configurada.

    Separada da chave do OpenRouter porque são provedores diferentes: usar a
    chave errada para o engine escolhido falharia com "não autorizado" sem
    nenhuma pista do motivo.
    """
    return get_setting("openai_api_key", "OPENAI_API_KEY")


def get_whisper_path():
    """Caminho do executável do Faster-Whisper-XXL, se configurado."""
    return get_setting("faster_whisper_path", "FASTER_WHISPER_PATH")
