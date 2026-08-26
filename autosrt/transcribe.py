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

import logging
import os
import re
import shlex
import shutil
import subprocess

from .errors import UnsupportedSubtitleError

logger = logging.getLogger(__name__)

EXECUTABLE_NAMES = ("faster-whisper-xxl", "faster-whisper-xxl.exe",
                    "whisper-faster-xxl", "whisper-faster-xxl.exe")

DEFAULT_MODEL = "turbo"
DEFAULT_COMPUTE_TYPE = "auto"
DEFAULT_DIARIZE_MODEL = "pyannote_v3.1"
# Detector de fala. Não é o padrão do próprio executável (que usa
# ``silero_v4_fw``): o v5 é mais recente, mas os dois calibram o limiar de
# forma diferente, e um valor de ``vad_threshold`` afinado para um não vale
# para o outro. Quando a transcrição sai com buracos, trocar o método é
# tão candidato quanto mexer no limiar -- por isso é configurável.
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
                  vad_threshold=None, vad_min_silence_ms=None,
                  max_speakers=None, extra_args=None) -> list:
    """Monta a linha de comando do Faster-Whisper-XXL.

    Separado da execução para poder ser verificado nos testes sem o
    executável presente.

    Args:
        vad_threshold: probabilidade mínima (0 a 1) para um trecho ser
            considerado fala. O padrão do próprio Whisper (não informado
            aqui) costuma ficar por volta de 0.5; baixar isso pega fala
            baixa/sussurrada que passaria batido, ao custo de arriscar
            confundir ruído de fundo com fala.
        vad_min_silence_ms: silêncio mínimo, em milissegundos, para a VAD
            considerar que uma fala terminou. Baixar isso evita que duas
            falas rápidas e coladas sejam tratadas como uma só (ou que a
            palavra final de uma fala seja cortada por engano).
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
    if vad_threshold is not None:
        command += ["--vad_threshold", str(vad_threshold)]
    if vad_min_silence_ms is not None:
        command += ["--vad_min_silence_duration_ms", str(vad_min_silence_ms)]
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
               vad_threshold=None, vad_min_silence_ms=None,
               max_speakers=None, progress=None, cancel_event=None,
               timeout=None, runner=None, extra_args=None) -> list:
    """Transcreve um arquivo de mídia e devolve a lista de :class:`Cue`.

    Args:
        media_path: vídeo ou áudio a transcrever.
        output_dir: onde o Whisper grava o .srt. Sendo ``None``, usa a pasta
            do próprio arquivo de mídia.
        model: nome do modelo. ``turbo`` é o ``large-v3-turbo``.
        language: código do idioma falado. ``None`` deixa o Whisper detectar.
        diarize: modelo de diarização, ou ``None`` para não diarizar.
        vad_threshold: veja :func:`build_command`. ``None`` (padrão) deixa o
            Whisper usar o próprio valor padrão, sem mexer em nada.
        vad_min_silence_ms: veja :func:`build_command`.
        progress: chamada como ``progress(percentual)`` conforme o Whisper
            reporta o andamento.
        runner: injeção usada pelos testes, no lugar da execução real.
        extra_args: argumentos extras repassados direto para o executável,
            para opções que este módulo ainda não conhece por nome.

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
        vad=vad, vad_threshold=vad_threshold,
        vad_min_silence_ms=vad_min_silence_ms, max_speakers=max_speakers,
        extra_args=extra_args)

    # O comando montado é a única forma de comparar o que o AutoSRT pede ao
    # Whisper com o que sai de um comando digitado à mão. Sem isso, quando o
    # resultado difere, não há como saber qual das opções acrescentadas aqui
    # é a responsável -- só sobra testar uma a uma, às cegas.
    logger.info("comando do Whisper: %s", shlex.join(command))

    run = runner or _run_process
    run(command, progress=progress, cancel_event=cancel_event, timeout=timeout)

    cues = load_transcript(_expected_output(media_path, output_dir))
    logger.info("transcrição de %s: %d legenda(s)",
                os.path.basename(media_path), len(cues))
    return cues


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


def _iter_lines(stream):
    """Percorre a saída quebrando em ``\\r`` além de ``\\n``.

    Barra de progresso reescreve sempre a mesma linha do terminal: ela
    termina cada atualização com ``\\r``, e só manda ``\\n`` quando acaba.
    Iterando o stream direto (que quebra só em ``\\n``), o laço fica parado
    até a transcrição inteira terminar -- o progresso é impresso, mas
    chega tarde demais para servir de progresso. Daí a barra travada em 0%
    e sem previsão de tempo: :meth:`autosrt.jobs.Job.segundos_restantes` só
    arrisca uma estimativa depois que o trabalho andou um pouco.
    """
    pedaco = []
    while True:
        char = stream.read(1)
        if not char:
            break
        if char in ("\r", "\n"):
            if pedaco:
                yield "".join(pedaco)
                pedaco = []
        else:
            pedaco.append(char)
    if pedaco:
        yield "".join(pedaco)


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
        for line in _iter_lines(process.stdout):
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
        # As linhas já vêm sem o terminador (_iter_lines o consome), então
        # o que junta de volta é o "\n" daqui -- sem ele o traceback do
        # Whisper chegaria numa linha só, ilegível.
        detail = "\n".join(output_tail).strip()[-800:]
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
