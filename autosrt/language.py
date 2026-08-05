"""Detecção do idioma de origem.

Correção do defeito 2. O fluxo antigo detectava o idioma a partir das dez
primeiras legendas, que num filme costumam ser cartela de título, aviso de
distribuidora ou créditos de abertura — muitas vezes em outro idioma, ou
curtas demais para o langdetect decidir. A amostra agora é distribuída ao
longo do arquivo inteiro e ignora legendas curtas.
"""

from langdetect import detect, DetectorFactory, LangDetectException
from langdetect.lang_detect_exception import ErrorCode

from .textfmt import TAG_RE, strip_invisible

# Deixa o langdetect determinístico: sem isso o mesmo texto pode devolver
# idiomas diferentes entre execuções.
DetectorFactory.seed = 0

# Legendas mais curtas que isto carregam pouca informação de idioma.
MIN_SAMPLE_CHARS = 15
# Quantas legendas coletar ao longo do arquivo.
SAMPLE_SIZE = 80

LANG_NAME = {
    "en": "Inglês", "es": "Espanhol", "pt": "Português", "cs": "Tcheco",
    "fr": "Francês", "de": "Alemão", "it": "Italiano", "nl": "Holandês",
    "ru": "Russo", "ja": "Japonês", "ar": "Árabe", "tr": "Turco",
    "pl": "Polonês", "sv": "Sueco", "fi": "Finlandês", "da": "Dinamarquês",
    "no": "Norueguês", "he": "Hebraico", "hi": "Hindi", "th": "Tailandês",
    "vi": "Vietnamita", "id": "Indonésio", "ms": "Malaio", "hu": "Húngaro",
    "ro": "Romeno", "bg": "Búlgaro", "uk": "Ucraniano", "hr": "Croata",
    "sr": "Sérvio", "sk": "Eslovaco", "sl": "Esloveno", "lt": "Lituano",
    "mt": "Maltês", "sq": "Albanês", "is": "Islandês", "ga": "Irlandês",
}


def language_name(code: str) -> str:
    """Nome do idioma para exibição, com o código como reserva."""
    if not code:
        return "desconhecido"
    return LANG_NAME.get(code, code.upper())


def build_sample(cues, sample_size: int = SAMPLE_SIZE) -> str:
    """Monta uma amostra de texto distribuída ao longo do arquivo."""
    usable = []
    for cue in cues:
        text = TAG_RE.sub("", strip_invisible(cue.source_text))
        text = " ".join(text.split())
        if len(text) >= MIN_SAMPLE_CHARS:
            usable.append(text)

    if not usable:
        # Nenhuma legenda longa o bastante: usa tudo que houver.
        usable = [" ".join(c.source_text.split()) for c in cues if c.source_text.strip()]

    if len(usable) <= sample_size:
        chosen = usable
    else:
        step = len(usable) / sample_size
        chosen = [usable[int(i * step)] for i in range(sample_size)]

    return " ".join(chosen)


def detect_language(cues) -> str:
    """Detecta o idioma das legendas.

    Devolve o código ISO 639-1. Lança ``LangDetectException`` se a amostra
    não permitir nenhuma conclusão.
    """
    sample = build_sample(cues)
    if not sample.strip():
        raise LangDetectException(
            ErrorCode.CantDetectError,
            "Arquivo sem texto suficiente para detectar o idioma")
    return detect(sample)
