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
    # Opcional: legendas por requisição ao motor LLM. Sem isso, decide
    # sozinho pelo endereço (bloco pequeno para servidor local, grande para
    # provedor na nuvem) -- só precisa disso quem quer sobrepor a detecção.
    "llm_block_size": "2",
    "faster_whisper_path": "/opt/faster-whisper-xxl/faster-whisper-xxl",
    # Opcional: modelo e precisão da transcrição, para toda a fila web.
    # Em branco usa turbo/auto, os padrões do transcribe.py.
    "whisper_model": "large-v3",
    "whisper_compute_type": "int8",
    # Opcional: reconhece o filme pelo nome do arquivo na lista do servidor.
    # Chave gratuita em https://www.themoviedb.org/settings/api
    "tmdb_api_key": "...",
    # Opcional: sensibilidade da VAD para toda transcrição feita pela fila
    # web (o CLI aceita os mesmos ajustes por linha de comando, que valem
    # só para aquela execução). Sem isso, usa o padrão do próprio Whisper.
    "vad_threshold": "0.2",
    "vad_min_silence_ms": "300",
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


def get_whisper_model():
    """Modelo do Whisper, ou ``None`` para o padrão do :mod:`~autosrt.transcribe`.

    Existe porque a fila web usava o padrão fixo e não tinha como trocar --
    o ajuste só existia no ``--modelo`` da linha de comando, justamente na
    interface que foi feita para quem não abre terminal.
    """
    return get_setting("whisper_model", "WHISPER_MODEL")


def get_whisper_compute_type():
    """Tipo de cálculo do Whisper (``auto``, ``int8``, ``float16``...).

    ``None`` mantém o ``auto``, que deixa o CTranslate2 escolher. Vale
    informar explicitamente quando a escolha automática não é a melhor para
    a placa: em GPUs Pascal (sem tensor cores), ``int8`` costuma ser mais
    rápido que ``float16`` e ocupa metade da memória, o que é o que permite
    rodar os modelos grandes numa placa de 5 GB.
    """
    return get_setting("whisper_compute_type", "WHISPER_COMPUTE_TYPE")


def get_llm_block_size():
    """Legendas por requisição ao motor LLM, ou ``None`` se não configurado.

    Sobrepõe a detecção automática por endereço em
    :func:`autosrt.llm_translate.translate_cues_llm` -- útil quando a
    heurística de "endereço local" (loopback ou IP privado) não bate com a
    realidade: um modelo pequeno servido num IP público, ou um modelo
    grande servido numa rede privada.
    """
    valor = get_setting("llm_block_size", "LLM_BLOCK_SIZE")
    try:
        return int(valor) if valor is not None else None
    except (TypeError, ValueError):
        return None


def get_normalize_audio():
    """Como tratar o volume do áudio antes de transcrever.

    Devolve ``"auto"`` (padrão), ``"sempre"`` ou ``"nunca"``. Valor
    desconhecido vira ``"auto"``, que é o comportamento seguro: mede o
    volume e só mexe no que está fraco demais para o Whisper reconhecer a
    fala. Veja :mod:`autosrt.audio`.
    """
    valor = (get_setting("normalize_audio", "AUTOSRT_NORMALIZE_AUDIO")
             or "").strip().lower()
    return valor if valor in ("auto", "sempre", "nunca") else "auto"


def get_whisper_language():
    """Idioma falado, ou ``None`` para o Whisper detectar sozinho.

    Vale fixar quando a transcrição falha logo no começo: o Whisper decide
    o idioma analisando os primeiros segundos do áudio, e um trecho inicial
    atípico (trilha, silêncio, vinheta) leva a uma detecção errada que
    estraga a transcrição até ele se estabilizar. Informando o idioma, essa
    etapa deixa de existir.
    """
    return get_setting("whisper_language", "WHISPER_LANGUAGE")


def get_vad_method():
    """Detector de fala usado pelo Whisper, ou ``None`` para o padrão.

    ``None`` mantém o :data:`autosrt.transcribe.DEFAULT_VAD`. Vale trocar
    quando a transcrição sai com buracos: cada método calibra o limiar de
    forma diferente, então mudar o método é tão candidato a resolver quanto
    mexer em :func:`get_vad_threshold`.
    """
    return get_setting("vad_method", "AUTOSRT_VAD_METHOD")


def get_vad_threshold():
    """Sensibilidade da VAD (0 a 1), ou ``None`` se não configurada.

    Quanto menor, mais sensível a fala baixa/sussurrada -- e mais risco de
    confundir ruído de fundo com fala. ``None`` deixa o Whisper no próprio
    padrão dele, sem mexer em nada.
    """
    valor = get_setting("vad_threshold", "AUTOSRT_VAD_THRESHOLD")
    try:
        return float(valor) if valor is not None else None
    except (TypeError, ValueError):
        return None


def get_vad_min_silence_ms():
    """Silêncio mínimo (ms) para a VAD considerar que uma fala terminou,
    ou ``None`` se não configurado."""
    valor = get_setting("vad_min_silence_ms", "AUTOSRT_VAD_MIN_SILENCE_MS")
    try:
        return int(valor) if valor is not None else None
    except (TypeError, ValueError):
        return None


def get_whisper_extra_args():
    """Argumentos extras repassados direto ao executável do Whisper local,
    para opções que este programa ainda não conhece por nome, ou ``None``
    se não configurado.

    Existia só na linha de comando (``--whisper-args``) -- é o escape hatch
    para uma opção do Faster-Whisper-XXL que não tem campo dedicado aqui,
    como ``--condition_on_previous_text False`` ou ``--no_speech_threshold``
    para alucinação que se repete pelo arquivo (uma alucinação vira contexto
    da seguinte quando ``condition_on_previous_text`` está ligado, que é o
    padrão do próprio Whisper).

    Devolve a string como veio, uma linha só, no formato de linha de
    comando (``--flag valor --outra-flag``). Quem chama é responsável por
    separar em lista (``shlex.split``); aqui não, porque um erro de aspas
    mal fechadas deve aparecer para quem gravou a configuração, não ser
    engolido em silêncio nesta função.
    """
    return get_setting("whisper_extra_args", "AUTOSRT_WHISPER_EXTRA_ARGS")


def get_diarize() -> bool:
    """Se a transcrição deve marcar quem fala cada linha. Padrão: sim.

    Existe porque a fila web diarizava sempre, sem como desligar -- o
    ``--sem-diarizacao`` só existia na linha de comando, justamente na
    interface que foi feita para quem não abre terminal.

    Desligar é o primeiro teste quando a legenda sai com buracos: a
    diarização é a maior diferença entre o que o AutoSRT pede ao Whisper e
    um comando digitado à mão, ela roda um segundo modelo (pyannote) por
    cima do áudio, e trecho que esse modelo não atribuir a ninguém é trecho
    que corre o risco de não virar legenda. A tradução perde só a dica de
    gênero por locutor; o texto continua saindo.
    """
    valor = get_setting("diarizar", "AUTOSRT_DIARIZE")
    if valor is None:
        return True
    return str(valor).strip().lower() not in ("nao", "não", "0", "false", "nunca")
