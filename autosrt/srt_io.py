"""Leitura e escrita de arquivos de legenda.

Diferença em relação ao fluxo antigo: o arquivo de entrada não é mais
reescrito para converter o encoding. A conversão acontece em memória, e o
arquivo original só é tocado no momento de salvar o resultado.
"""

import os

import chardet
import pysrt

from . import ssa
from .cue import Cue
from .errors import UnsupportedSubtitleError

DEFAULT_ENCODING = "utf-8"
SSA_EXTENSIONS = {".ssa", ".ass"}
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
    """Carrega um arquivo de legenda como lista de :class:`Cue`.

    Aceita ``.srt`` e também ``.ssa``/``.ass``, que são convertidos para o
    modelo interno. O formato é decidido pela extensão e, quando ela não
    corresponde ao conteúdo, pelo próprio conteúdo.

    Raises:
        UnsupportedSubtitleError: se nenhuma legenda puder ser extraída.
    """
    extension = os.path.splitext(path)[1].lower()
    encoding = detect_encoding(path) or DEFAULT_ENCODING

    if extension in SSA_EXTENSIONS:
        cues = ssa.load_cues(path, encoding=encoding)
        if cues:
            return cues
        raise UnsupportedSubtitleError(
            f"Nenhuma legenda encontrada em {os.path.basename(path)}.")

    # Decodifica primeiro e só então entrega o texto ao pysrt: abrir o
    # arquivo pelo pysrt e deixar a decodificação falhar vaza o descritor.
    text = read_text(path)
    cues = _cues_from_srt_text(text)
    if cues:
        return cues

    # Arquivo SSA com extensão .srt é comum o bastante para valer a tentativa.
    if ssa.looks_like_ssa(text):
        cues = ssa.load_cues(path, encoding=encoding)
        if cues:
            return cues

    raise UnsupportedSubtitleError(
        f"Nenhuma legenda encontrada em {os.path.basename(path)}. "
        "O arquivo está vazio ou não é uma legenda SRT/SSA válida.")


def _cues_from_srt_text(text: str) -> list:
    subs = pysrt.from_string(text)
    return [
        Cue.from_source(
            index=position,
            start=sub.start.ordinal,
            end=sub.end.ordinal,
            source_text=sub.text)
        for position, sub in enumerate(subs, start=1)
    ]


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


def srt_output_path(path: str) -> str:
    """Caminho .srt correspondente a um arquivo de legenda.

    Arquivos .srt devolvem o próprio caminho; SSA/ASS devolvem o irmão com
    extensão .srt, para que a conversão não sobrescreva o original.
    """
    stem, extension = os.path.splitext(path)
    if extension.lower() == ".srt":
        return path
    return stem + ".srt"


def convert_to_srt(input_path: str, output_path: str = None) -> str:
    """Converte uma legenda SSA/ASS para SRT, sem traduzir.

    Returns:
        O caminho do arquivo gravado.
    """
    if output_path is None:
        output_path = srt_output_path(input_path)
    if os.path.abspath(output_path) == os.path.abspath(input_path):
        raise UnsupportedSubtitleError(
            "A conversão precisa de um arquivo de saída diferente da entrada.")

    cues = load_cues(input_path)
    save_cues(cues, output_path)
    return output_path


def backup_path(path: str) -> str:
    """Caminho do backup correspondente a um arquivo de legenda."""
    directory = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(directory, f"{stem}_backup.srt")
