"""Transcrição de áudio via Faster-Whisper-XXL (Purfview).

A integração é por subprocesso, no mesmo padrão usado para o ffsubsync. O
Faster-Whisper-XXL é um executável autocontido, com todas as bibliotecas
embutidas, disponível para Windows e para Linux — não é dependência Python e
não entra no ``requirements.txt``.

Duas coisas que a transcrição resolve de graça:

- **Sincronia.** Os tempos vêm do áudio, então a legenda nasce sincronizada.
  O ffsubsync continua útil apenas para legendas que já existiam.
- **Quem fala cada linha.** Com ``--diarize`` o Whisper marca o locutor de
  cada fala, que é justamente o elo que faltava para resolver o gênero.

Sobre ``compute_type``: ``int8_float16`` exige compute capability 7.0 ou
superior. Placas Pascal (série P, como a Quadro P2200) ficam abaixo disso,
e ``float16`` além de lento não cabe em 5 GB com o modelo turbo. Por isso o
padrão aqui é ``auto``, que deixa o CTranslate2 escolher o tipo mais rápido
que o hardware realmente suporta.
"""

import os
import re
import shutil
import subprocess

from .errors import UnsupportedSubtitleError

EXECUTABLE_NAMES = ("faster-whisper-xxl", "faster-whisper-xxl.exe",
                    "whisper-faster-xxl", "whisper-faster-xxl.exe")

DEFAULT_MODEL = "turbo"
DEFAULT_COMPUTE_TYPE = "auto"
DEFAULT_DIARIZE_MODEL = "pyannote_v3.1"
DEFAULT_VAD = "silero_v5"

# Modelos de diarização de uso pessoal e não comercial. Não entram como
# padrão para não impor uma restrição de licença sem o usuário saber.
RESTRICTED_DIARIZE_MODELS = {"reverb_v1", "reverb_v2"}

# "[SPEAKER_00]: fala", "SPEAKER_00: fala", "(SPEAKER 1) fala"
SPEAKER_RE = re.compile(
    r"^\s*[\[\(]?\s*(SPEAKER[ _]?\d+|SPK[ _]?\d+)\s*[\]\)]?\s*:?\s*",
    re.IGNORECASE)

PROGRESS_RE = re.compile(r"(\d{1,3})%")


class TranscriptionError(Exception):
    """Falha ao transcrever."""


def find_executable(explicit_path: str = None):
    """Localiza o executável do Faster-Whisper-XXL.

    Procura, nesta ordem: o caminho informado, a variável de ambiente
    ``FASTER_WHISPER_PATH`` e o ``PATH`` do sistema.
    """
    if explicit_path:
        return explicit_path if os.path.exists(explicit_path) else None

    from_env = os.environ.get("FASTER_WHISPER_PATH")
    if from_env and os.path.exists(from_env):
        return from_env

    for name in EXECUTABLE_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def whisper_available(explicit_path: str = None) -> bool:
    return find_executable(explicit_path) is not None


def build_command(media_path, output_dir, *, executable, model=DEFAULT_MODEL,
                  language=None, diarize=DEFAULT_DIARIZE_MODEL,
                  compute_type=DEFAULT_COMPUTE_TYPE, vad=DEFAULT_VAD,
                  max_speakers=None, extra_args=None) -> list:
    """Monta a linha de comando do Faster-Whisper-XXL.

    Separado da execução para poder ser verificado nos testes sem o
    executável presente.
    """
    command = [
        executable, media_path,
        "--model", model,
        "--output_dir", output_dir,
        "--output_format", "srt",
        "--compute_type", compute_type,
        "-pp",  # imprime progresso, que é lido para alimentar a interface
    ]

    if language:
        command += ["--language", language]
    if vad:
        command += ["--vad_method", vad]
    if diarize:
        command += ["--diarize", diarize]
        if max_speakers:
            command += ["--max_speakers", str(max_speakers)]
    if extra_args:
        command += list(extra_args)

    return command


def split_speaker(text: str):
    """Separa o rótulo de locutor do texto da fala.

    Devolve ``(locutor, texto)``. Sem rótulo, o locutor vem ``None``.

    >>> split_speaker("[SPEAKER_00]: Olá")
    ('SPEAKER_00', 'Olá')
    """
    lines = text.split("\n")
    match = SPEAKER_RE.match(lines[0]) if lines else None
    if not match:
        return None, text

    speaker = re.sub(r"[ ]+", "_", match.group(1).strip().upper())
    lines[0] = lines[0][match.end():]
    cleaned = "\n".join(lines).strip()
    return speaker, cleaned


def transcribe(media_path, *, output_dir=None, executable=None,
               model=DEFAULT_MODEL, language=None, diarize=DEFAULT_DIARIZE_MODEL,
               compute_type=DEFAULT_COMPUTE_TYPE, vad=DEFAULT_VAD,
               max_speakers=None, progress=None, cancel_event=None,
               timeout=None, runner=None) -> list:
    """Transcreve um arquivo de mídia e devolve a lista de :class:`Cue`.

    Args:
        media_path: vídeo ou áudio a transcrever.
        output_dir: onde o Whisper grava o .srt. Sendo ``None``, usa a pasta
            do próprio arquivo de mídia.
        model: nome do modelo. ``turbo`` é o ``large-v3-turbo``.
        language: código do idioma falado. ``None`` deixa o Whisper detectar.
        diarize: modelo de diarização, ou ``None`` para não diarizar.
        progress: chamada como ``progress(percentual)`` conforme o Whisper
            reporta o andamento.
        runner: injeção usada pelos testes, no lugar da execução real.

    Returns:
        Lista de legendas, com ``speaker`` preenchido quando houve diarização.

    Raises:
        TranscriptionError: executável ausente, arquivo inexistente, modelo
            de diarização inválido, ou falha na execução.
    """
    if not os.path.exists(media_path):
        raise TranscriptionError(f"Arquivo não encontrado: {media_path}")

    resolved = find_executable(executable)
    if not resolved:
        raise TranscriptionError(
            "O Faster-Whisper-XXL não foi encontrado. Baixe o executável em "
            "github.com/Purfview/whisper-standalone-win e deixe-o no PATH, "
            "ou aponte a variável FASTER_WHISPER_PATH para ele.")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(media_path))
    os.makedirs(output_dir, exist_ok=True)

    command = build_command(
        media_path, output_dir, executable=resolved, model=model,
        language=language, diarize=diarize, compute_type=compute_type,
        vad=vad, max_speakers=max_speakers)

    run = runner or _run_process
    run(command, progress=progress, cancel_event=cancel_event, timeout=timeout)

    return load_transcript(_expected_output(media_path, output_dir))


def _expected_output(media_path: str, output_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(media_path))[0]
    return os.path.join(output_dir, stem + ".srt")


def load_transcript(srt_path: str) -> list:
    """Lê o .srt produzido pelo Whisper, separando os rótulos de locutor."""
    from . import srt_io

    if not os.path.exists(srt_path):
        raise TranscriptionError(
            f"O Whisper terminou mas não gerou {os.path.basename(srt_path)}.")

    try:
        cues = srt_io.load_cues(srt_path)
    except UnsupportedSubtitleError as exc:
        raise TranscriptionError(str(exc))

    for cue in cues:
        speaker, text = split_speaker(cue.source_text)
        cue.speaker = speaker
        cue.source_text = text
        cue.text = text
    return cues


def _run_process(command, *, progress=None, cancel_event=None, timeout=None):
    """Executa o Whisper, repassando o progresso conforme ele é impresso."""
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, errors="replace")
    except OSError as exc:
        raise TranscriptionError(f"Não foi possível executar o Whisper: {exc}")

    output_tail = []
    try:
        for line in process.stdout:
            output_tail.append(line)
            del output_tail[:-40]
            if progress:
                match = PROGRESS_RE.search(line)
                if match:
                    progress(min(100, int(match.group(1))))
            if cancel_event is not None and cancel_event.is_set():
                process.terminate()
                raise TranscriptionError("Transcrição cancelada.")
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        raise TranscriptionError("O Whisper passou do tempo limite.")
    finally:
        if process.stdout:
            process.stdout.close()

    if process.returncode != 0:
        detail = "".join(output_tail).strip()[-800:]
        raise TranscriptionError(_explicar_falha(detail))


# O Whisper morre com um traceback de Python quando o arquivo não tem trilha
# de áudio: ele pede a faixa 0 ao demuxer e não vem nada. Gravação de tela
# feita sem microfone cai sempre aqui, e o traceback cru não diz o que fazer.
SEM_AUDIO = ("IndexError: tuple index out of range", "StreamContainer.get")


def _explicar_falha(detail: str) -> str:
    """Troca o traceback do Whisper por uma explicação, quando dá para saber."""
    if all(marca in detail for marca in SEM_AUDIO):
        return ("Esse arquivo não tem faixa de áudio, então não há fala para "
                "transcrever. Gravação de tela feita sem microfone fica assim. "
                "Confira o arquivo original ou mande o áudio separado.")
    return f"O Whisper falhou:\n{detail}"


def speakers_in(cues) -> list:
    """Lista os locutores encontrados, na ordem em que aparecem."""
    seen = []
    for cue in cues:
        if cue.speaker and cue.speaker not in seen:
            seen.append(cue.speaker)
    return seen
