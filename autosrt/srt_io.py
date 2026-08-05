"""Leitura e escrita de arquivos de legenda.

Diferença em relação ao fluxo antigo: o arquivo de entrada não é mais
reescrito para converter o encoding. A conversão acontece em memória, e o
arquivo original só é tocado no momento de salvar o resultado.
"""

import os

import chardet
import pysrt

from .cue import Cue

DEFAULT_ENCODING = "utf-8"
# Abaixo desta confiança o palpite do chardet não vale mais que o padrão.
MIN_DETECTION_CONFIDENCE = 0.60


def detect_encoding(path: str):
    """Descobre o encoding do arquivo, ou devolve ``None`` se não houver
    palpite confiável."""
    with open(path, "rb") as handle:
        raw = handle.read()

    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    result = chardet.detect(raw)
    if not result or not result.get("encoding"):
        return None
    if result.get("confidence", 0) < MIN_DETECTION_CONFIDENCE:
        return None
    return result["encoding"]


def read_text(path: str) -> str:
    """Lê o arquivo como texto, resolvendo o encoding sozinho."""
    encoding = detect_encoding(path)
    for candidate in (encoding, DEFAULT_ENCODING, "latin-1"):
        if not candidate:
            continue
        try:
            with open(path, "r", encoding=candidate) as handle:
                return handle.read()
        except (UnicodeDecodeError, LookupError):
            continue
    # latin-1 aceita qualquer byte, então só chegamos aqui se o arquivo sumiu.
    raise IOError(f"Não foi possível ler {path}")


def load_cues(path: str) -> list:
    """Carrega um arquivo .srt como lista de :class:`Cue`."""
    # Decodifica primeiro e só então entrega o texto ao pysrt: abrir o
    # arquivo pelo pysrt e deixar a decodificação falhar vaza o descritor.
    subs = pysrt.from_string(read_text(path))

    cues = []
    for position, sub in enumerate(subs, start=1):
        cues.append(Cue.from_source(
            index=position,
            start=sub.start.ordinal,
            end=sub.end.ordinal,
            source_text=sub.text))
    return cues


def save_cues(cues, path: str) -> None:
    """Grava as legendas em .srt UTF-8, usando o texto de trabalho."""
    subs = pysrt.SubRipFile()
    for position, cue in enumerate(cues, start=1):
        subs.append(pysrt.SubRipItem(
            index=position,
            start=pysrt.SubRipTime.from_ordinal(cue.start),
            end=pysrt.SubRipTime.from_ordinal(cue.end),
            text=cue.text))
    subs.save(path, encoding=DEFAULT_ENCODING)


def backup_path(path: str) -> str:
    """Caminho do backup correspondente a um arquivo de legenda."""
    directory = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(directory, f"{stem}_backup.srt")
