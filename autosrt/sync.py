"""Sincronização das legendas com o áudio.

O AutoSRT nunca mexia nos tempos das legendas — o único atributo que ele
escrevia era o texto. Este módulo é a funcionalidade nova.

São três níveis, do mais barato ao mais completo:

1. Deslocamento fixo, para quando a legenda inteira está adiantada ou
   atrasada pela mesma quantidade.
2. Ajuste linear de dois pontos, para quando o erro cresce ao longo do filme.
   É o caso da legenda vinda de um release com framerate diferente.
3. Sincronização automática via ``ffsubsync``, contra o áudio do vídeo ou
   contra uma legenda de referência já correta.
"""

import os
import re
import shutil
import subprocess

# Razões de framerate comuns entre releases de cinema e TV.
COMMON_FPS = (23.976, 24.0, 25.0, 29.97, 30.0)

TIMECODE_RE = re.compile(
    r"^(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[,.](?P<ms>\d{1,3}))?$")


class SyncError(Exception):
    """Falha ao sincronizar."""


def parse_timecode(value: str) -> int:
    """Converte um tempo digitado pelo usuário em milissegundos.

    Aceita ``00:01:23,456``, ``01:23.4``, ``1:23`` e também segundos puros
    como ``83.5``.

    Raises:
        SyncError: se o formato não for reconhecido.
    """
    text = (value or "").strip()
    if not text:
        raise SyncError("Informe um tempo.")

    match = TIMECODE_RE.match(text)
    if match:
        hours = int(match.group("h") or 0)
        minutes = int(match.group("m"))
        seconds = int(match.group("s"))
        fraction = match.group("ms") or "0"
        # "5" significa 500 ms, não 5 ms.
        milliseconds = int(fraction.ljust(3, "0")[:3])
        return ((hours * 3600 + minutes * 60 + seconds) * 1000) + milliseconds

    try:
        return round(float(text.replace(",", ".")) * 1000)
    except ValueError:
        raise SyncError(f"Tempo inválido: {value!r}. Use 00:01:23,456 ou segundos.")


def format_timecode(milliseconds: int) -> str:
    """Formata milissegundos como ``HH:MM:SS,mmm``."""
    milliseconds = max(0, int(milliseconds))
    hours, rest = divmod(milliseconds, 3600000)
    minutes, rest = divmod(rest, 60000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def shift_cues(cues, milliseconds: int) -> None:
    """Desloca todas as legendas pelo mesmo tempo, no lugar.

    Valores positivos atrasam a legenda, negativos adiantam. Tempos nunca
    ficam negativos.
    """
    for cue in cues:
        cue.start = max(0, cue.start + milliseconds)
        cue.end = max(0, cue.end + milliseconds)


def linear_sync(cues, source_a: int, target_a: int,
                source_b: int, target_b: int) -> tuple:
    """Ajusta os tempos por dois pontos de referência, no lugar.

    O usuário informa dois momentos em que sabe o tempo correto: tipicamente
    a primeira e a última fala do filme. ``source_*`` são os tempos como estão
    na legenda e ``target_*`` os tempos corretos, todos em milissegundos.

    Corrige simultaneamente o deslocamento e a deriva progressiva::

        escala = (target_b - target_a) / (source_b - source_a)
        t' = target_a + (t - source_a) * escala

    Returns:
        A tupla ``(escala, deslocamento_ms)`` que foi aplicada.

    Raises:
        SyncError: se os dois pontos de origem forem iguais, o que tornaria
            a escala indefinida.
    """
    if source_b == source_a:
        raise SyncError(
            "Os dois pontos de referência precisam ter tempos diferentes "
            "na legenda de origem.")

    scale = (target_b - target_a) / (source_b - source_a)
    offset = target_a - source_a * scale

    for cue in cues:
        cue.start = max(0, round(cue.start * scale + offset))
        cue.end = max(0, round(cue.end * scale + offset))

    return scale, offset


def fps_sync(cues, source_fps: float, target_fps: float) -> float:
    """Converte os tempos entre framerates, no lugar.

    ``source_fps`` é o framerate para o qual a legenda foi feita e
    ``target_fps`` o do vídeo em mãos. O caso clássico é uma legenda de
    release PAL (25 fps) sobre um vídeo de cinema (23,976 fps), que acumula
    4,27% de erro ao longo do filme.

    Returns:
        A escala aplicada.
    """
    if source_fps <= 0 or target_fps <= 0:
        raise SyncError("Os framerates precisam ser positivos.")

    scale = source_fps / target_fps
    for cue in cues:
        cue.start = max(0, round(cue.start * scale))
        cue.end = max(0, round(cue.end * scale))
    return scale


def ffsubsync_available() -> bool:
    """Diz se o ``ffsubsync`` está instalado e acessível."""
    return shutil.which("ffsubsync") is not None


def ffmpeg_available() -> bool:
    """Diz se o ``ffmpeg`` está acessível.

    Necessário apenas para sincronizar contra o áudio de um vídeo; o modo por
    legenda de referência dispensa.
    """
    return shutil.which("ffmpeg") is not None


def autosync(subtitle_path: str, reference_path: str, output_path: str,
             timeout: int = 900) -> str:
    """Sincroniza automaticamente uma legenda contra uma referência.

    A referência pode ser um arquivo de vídeo (o áudio é analisado por
    detecção de atividade de voz) ou outra legenda já sincronizada (muito
    mais rápido, porque dispensa extrair o áudio). O ``ffsubsync`` decide o
    modo pela extensão do arquivo.

    Args:
        subtitle_path: legenda fora de sincronia.
        reference_path: vídeo ou legenda de referência.
        output_path: onde gravar a legenda corrigida.
        timeout: limite em segundos para a execução.

    Returns:
        A saída de texto produzida pelo ``ffsubsync``.

    Raises:
        SyncError: se o ffsubsync não estiver instalado, se faltar o ffmpeg
            para uma referência de vídeo, ou se a execução falhar.
    """
    if not ffsubsync_available():
        raise SyncError(
            "O ffsubsync não está instalado. Instale com: pip install ffsubsync")

    if not os.path.exists(subtitle_path):
        raise SyncError(f"Legenda não encontrada: {subtitle_path}")
    if not os.path.exists(reference_path):
        raise SyncError(f"Referência não encontrada: {reference_path}")

    if not _is_subtitle_file(reference_path) and not ffmpeg_available():
        raise SyncError(
            "Sincronizar contra vídeo exige o ffmpeg instalado e no PATH. "
            "Use uma legenda de referência já sincronizada para dispensá-lo.")

    command = [
        "ffsubsync", reference_path,
        "-i", subtitle_path,
        "-o", output_path,
    ]

    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SyncError(f"O ffsubsync passou de {timeout} segundos e foi interrompido.")
    except OSError as exc:
        raise SyncError(f"Não foi possível executar o ffsubsync: {exc}")

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise SyncError(f"O ffsubsync falhou: {detail}")

    return completed.stdout


def _is_subtitle_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".srt", ".ass", ".ssa", ".vtt", ".sub"}
