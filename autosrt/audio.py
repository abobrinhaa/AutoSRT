"""Preparo do áudio antes da transcrição.

Existe por causa de um caso concreto: um filme de 1974, digitalizado num AAC
de 65 kbps, com volume médio de -34,7 dB -- cerca de 10 dB abaixo do que
um filme com diálogo costuma ter. Nesse nível o Whisper parava de
reconhecer a fala e passava a alucinar "Thank you." em cima do ruído, que é
a muleta conhecida dele para áudio contínuo mas pouco inteligível. A
legenda saía com trechos inteiros vazios, sem nenhum erro em lugar nenhum:
o áudio existia, a transcrição rodava, e o resultado vinha errado.

Normalizar o volume antes de transcrever resolveu -- no mesmo trecho em que
só havia alucinação, apareceu a fala real. Daí este módulo.

O áudio normalizado sai como WAV 16 kHz mono, que é exatamente o formato
que o Whisper usa internamente: além de resolver o volume, poupa a
conversão que ele faria de qualquer jeito. Os tempos continuam valendo,
porque a duração não muda.
"""

import os
import re
import shutil
import subprocess
import threading
import time

#: Abaixo disto o áudio é considerado fraco demais para transcrever bem.
#: Filme com diálogo normal fica entre -25 e -20 dB; o caso que motivou
#: este módulo estava em -34,7 dB. A margem é de propósito: normalizar um
#: áudio que já estava bom não estraga nada, enquanto deixar passar um
#: áudio fraco custa uma legenda cheia de buracos.
LIMIAR_VOLUME_BAIXO = -28.0

#: Alvo do ``loudnorm``, na recomendação EBU R128 usada em difusão.
ALVO_LUFS = -16.0
ALVO_TRUE_PEAK = -1.5

MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


class AudioError(Exception):
    """Falha ao medir ou preparar o áudio."""


class AudioCancelled(AudioError):
    """O preparo do áudio foi cancelado pelo usuário."""


def _watch_cancel(process, cancel_event, cancelled):
    """Derruba o ffmpeg assim que ``cancel_event`` for acionado.

    ``subprocess.run`` bloqueia até o processo terminar sozinho, sem
    nenhuma forma de interromper -- num filme longo, cancelar durante a
    normalização do áudio ficava sem efeito nenhum até o ffmpeg acabar por
    conta própria. Esta thread observa o evento por fora e mata o processo
    assim que ele for acionado.
    """
    while process.poll() is None:
        if cancel_event.is_set():
            cancelled.set()
            process.terminate()
            return
        time.sleep(0.3)


def _run_ffmpeg(comando, *, timeout=None, cancel_event=None):
    """Roda o ffmpeg, cancelável a qualquer momento via ``cancel_event``."""
    try:
        processo = subprocess.Popen(
            comando, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, errors="replace")
    except OSError as exc:
        raise AudioError(f"Não foi possível executar o ffmpeg: {exc}")

    cancelled = threading.Event()
    watcher = None
    if cancel_event is not None:
        watcher = threading.Thread(
            target=_watch_cancel, args=(processo, cancel_event, cancelled),
            daemon=True)
        watcher.start()

    try:
        _, stderr = processo.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        processo.kill()
        processo.communicate()
        raise AudioError("O ffmpeg passou do tempo limite.")
    finally:
        if watcher is not None:
            watcher.join(timeout=1)

    if cancelled.is_set():
        raise AudioCancelled("Preparo do áudio cancelado.")

    return processo.returncode, stderr


def find_ffmpeg(executable=None):
    """Localiza o ffmpeg. Devolve ``None`` se não houver."""
    if executable and os.path.isfile(executable):
        return executable
    return shutil.which(os.environ.get("FFMPEG_PATH") or "ffmpeg")


def medir_volume_medio(media_path, *, ffmpeg=None, timeout=None,
                       cancel_event=None):
    """Mede o volume médio do áudio, em dB.

    Usa o filtro ``volumedetect``, que percorre o arquivo inteiro sem
    produzir saída. Num filme longo isso leva alguns segundos -- pouco
    perto do custo de transcrever, e é o que evita normalizar às cegas.

    Returns:
        O ``mean_volume`` em dB, ou ``None`` quando não dá para medir
        (sem ffmpeg, arquivo sem áudio, saída inesperada, ou cancelado).
        ``None`` é "não sei", e quem chama deve seguir sem normalizar em
        vez de tratar como erro: medir é uma otimização, não um
        pré-requisito.
    """
    binario = find_ffmpeg(ffmpeg)
    if not binario:
        return None

    comando = [binario, "-nostdin", "-i", media_path,
               "-af", "volumedetect", "-vn", "-sn", "-dn",
               "-f", "null", "-"]
    try:
        returncode, stderr = _run_ffmpeg(
            comando, timeout=timeout, cancel_event=cancel_event)
    except AudioError:
        return None

    # O volumedetect escreve no stderr, e o ffmpeg sai com código 0 mesmo
    # assim -- o que interessa é a linha, não o código de saída.
    achado = MEAN_VOLUME_RE.search(stderr or "")
    return float(achado.group(1)) if achado else None


def volume_baixo(media_path, *, limiar=LIMIAR_VOLUME_BAIXO, ffmpeg=None,
                 timeout=None, cancel_event=None) -> bool:
    """Diz se o áudio está fraco a ponto de atrapalhar a transcrição.

    Sem conseguir medir, responde ``False``: normalizar sem saber trocaria
    um problema conhecido por um palpite, e o custo de errar para mais é
    reprocessar o áudio de todo mundo à toa.
    """
    medido = medir_volume_medio(media_path, ffmpeg=ffmpeg, timeout=timeout,
                                cancel_event=cancel_event)
    return medido is not None and medido < limiar


def normalizar_para_wav(media_path, destino, *, ffmpeg=None, timeout=None,
                        alvo_lufs=ALVO_LUFS, true_peak=ALVO_TRUE_PEAK,
                        cancel_event=None) -> str:
    """Grava o áudio normalizado como WAV 16 kHz mono.

    16 kHz mono não é escolha arbitrária: é o formato que o Whisper usa
    internamente, então o arquivo já sai pronto e a conversão que ele faria
    deixa de acontecer duas vezes.

    Returns:
        O caminho gravado (o mesmo ``destino``).

    Raises:
        AudioError: sem ffmpeg, ou o ffmpeg falhou.
        AudioCancelled: cancelado pelo usuário via ``cancel_event``.
    """
    binario = find_ffmpeg(ffmpeg)
    if not binario:
        raise AudioError(
            "O ffmpeg não foi encontrado, e ele é necessário para normalizar "
            "o áudio. Instale-o e deixe-o no PATH, ou aponte FFMPEG_PATH "
            "para ele -- ou desligue a normalização na configuração.")

    pasta = os.path.dirname(os.path.abspath(destino))
    if pasta:
        os.makedirs(pasta, exist_ok=True)

    comando = [
        binario, "-nostdin", "-y", "-i", media_path,
        "-af", f"loudnorm=I={alvo_lufs}:TP={true_peak}:LRA=11",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        "-vn", "-sn", "-dn",
        destino,
    ]
    returncode, stderr = _run_ffmpeg(
        comando, timeout=timeout, cancel_event=cancel_event)

    if returncode != 0:
        detalhe = (stderr or "").strip()[-500:]
        raise AudioError(f"O ffmpeg falhou ao normalizar o áudio:\n{detalhe}")
    if not os.path.exists(destino):
        raise AudioError("O ffmpeg terminou mas não gerou o áudio normalizado.")
    return destino
