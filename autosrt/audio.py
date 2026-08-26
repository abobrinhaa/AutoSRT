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

#: "Duration: 01:52:33.44" na saída do ffmpeg, usado quando não há ffprobe.
DURACAO_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")

#: Quanto do áudio original o arquivo normalizado precisa cobrir para valer.
#: Não é 100% porque o último quadro de um container costuma ficar alguns
#: décimos aquém do que o cabeçalho anuncia, e reclamar disso seria ruído.
COBERTURA_MINIMA = 0.98

#: Diferença, em segundos, abaixo da qual nem vale comparar: um arquivo
#: curto não tem margem relativa que sirva de sinal.
FOLGA_SEGUNDOS = 5.0


class AudioError(Exception):
    """Falha ao medir ou preparar o áudio."""


def find_ffmpeg(executable=None):
    """Localiza o ffmpeg. Devolve ``None`` se não houver."""
    if executable and os.path.isfile(executable):
        return executable
    return shutil.which(os.environ.get("FFMPEG_PATH") or "ffmpeg")


def medir_volume_medio(media_path, *, ffmpeg=None, timeout=None):
    """Mede o volume médio do áudio, em dB.

    Usa o filtro ``volumedetect``, que percorre o arquivo inteiro sem
    produzir saída. Num filme longo isso leva alguns segundos -- pouco
    perto do custo de transcrever, e é o que evita normalizar às cegas.

    Returns:
        O ``mean_volume`` em dB, ou ``None`` quando não dá para medir
        (sem ffmpeg, arquivo sem áudio, saída inesperada). ``None`` é
        "não sei", e quem chama deve seguir sem normalizar em vez de
        tratar como erro: medir é uma otimização, não um pré-requisito.
    """
    binario = find_ffmpeg(ffmpeg)
    if not binario:
        return None

    comando = [binario, "-nostdin", "-i", media_path,
               "-af", "volumedetect", "-vn", "-sn", "-dn",
               "-f", "null", "-"]
    try:
        processo = subprocess.run(
            comando, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None

    # O volumedetect escreve no stderr, e o ffmpeg sai com código 0 mesmo
    # assim -- o que interessa é a linha, não o código de saída.
    achado = MEAN_VOLUME_RE.search(processo.stderr or "")
    return float(achado.group(1)) if achado else None


def find_ffprobe(executable=None):
    """Localiza o ffprobe. Devolve ``None`` se não houver."""
    if executable and os.path.isfile(executable):
        return executable
    return shutil.which(os.environ.get("FFPROBE_PATH") or "ffprobe")


def duracao_segundos(media_path, *, ffprobe=None, ffmpeg=None, timeout=None):
    """Duração do arquivo, em segundos, ou ``None`` se não der para saber.

    Tenta o ffprobe primeiro, que responde só o número. Sem ele, lê a linha
    ``Duration:`` que o ffmpeg imprime ao abrir o arquivo -- o ffprobe pode
    não estar instalado mesmo onde o ffmpeg está.
    """
    binario = find_ffprobe(ffprobe)
    if binario:
        comando = [binario, "-v", "error", "-show_entries",
                   "format=duration", "-of",
                   "default=noprint_wrappers=1:nokey=1", media_path]
        try:
            processo = subprocess.run(
                comando, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, errors="replace", timeout=timeout)
            return float((processo.stdout or "").strip())
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass

    binario = find_ffmpeg(ffmpeg)
    if not binario:
        return None
    try:
        processo = subprocess.run(
            [binario, "-nostdin", "-i", media_path], stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None

    achado = DURACAO_RE.search(processo.stderr or "")
    if not achado:
        return None
    horas, minutos, segundos = achado.groups()
    return int(horas) * 3600 + int(minutos) * 60 + float(segundos)


def formatar_tempo(segundos) -> str:
    """``6013`` vira ``1:40:13`` -- o formato em que o usuário vê a legenda."""
    segundos = int(segundos)
    return (f"{segundos // 3600}:{segundos % 3600 // 60:02d}:"
            f"{segundos % 60:02d}")


def volume_baixo(media_path, *, limiar=LIMIAR_VOLUME_BAIXO, ffmpeg=None,
                 timeout=None) -> bool:
    """Diz se o áudio está fraco a ponto de atrapalhar a transcrição.

    Sem conseguir medir, responde ``False``: normalizar sem saber trocaria
    um problema conhecido por um palpite, e o custo de errar para mais é
    reprocessar o áudio de todo mundo à toa.
    """
    medido = medir_volume_medio(media_path, ffmpeg=ffmpeg, timeout=timeout)
    return medido is not None and medido < limiar


def normalizar_para_wav(media_path, destino, *, ffmpeg=None, timeout=None,
                        alvo_lufs=ALVO_LUFS, true_peak=ALVO_TRUE_PEAK) -> str:
    """Grava o áudio normalizado como WAV 16 kHz mono.

    16 kHz mono não é escolha arbitrária: é o formato que o Whisper usa
    internamente, então o arquivo já sai pronto e a conversão que ele faria
    deixa de acontecer duas vezes.

    Returns:
        O caminho gravado (o mesmo ``destino``).

    Raises:
        AudioError: sem ffmpeg, ou o ffmpeg falhou.
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
    try:
        processo = subprocess.run(
            comando, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AudioError("O ffmpeg passou do tempo limite ao normalizar o áudio.")
    except OSError as exc:
        raise AudioError(f"Não foi possível executar o ffmpeg: {exc}")

    if processo.returncode != 0:
        detalhe = (processo.stderr or "").strip()[-500:]
        raise AudioError(f"O ffmpeg falhou ao normalizar o áudio:\n{detalhe}")
    if not os.path.exists(destino):
        raise AudioError("O ffmpeg terminou mas não gerou o áudio normalizado.")

    _conferir_cobertura(media_path, destino, timeout=timeout)
    return destino


def _conferir_cobertura(media_path, destino, *, timeout=None):
    """Recusa um áudio normalizado que ficou mais curto que o original.

    O ffmpeg encerra com código 0 mesmo quando para no meio de um arquivo
    com defeito: ele grava o que conseguiu ler e considera o trabalho
    feito. O WAV curto que sai daí é indistinguível de um bom -- existe, é
    válido, toca -- e o Whisper o transcreve inteiro sem reclamar. O
    resultado é uma legenda que simplesmente acaba no meio do filme, com o
    trabalho marcado como concluído e nenhum erro em lugar nenhum.

    Comparar as duas durações é o que transforma isso num aviso.
    """
    origem = duracao_segundos(media_path, timeout=timeout)
    saida = duracao_segundos(destino, timeout=timeout)
    if not origem or not saida:
        return
    if saida >= origem * COBERTURA_MINIMA or origem - saida <= FOLGA_SEGUNDOS:
        return

    raise AudioError(
        f"O ffmpeg parou de ler o áudio em {formatar_tempo(saida)}, mas o "
        f"arquivo tem {formatar_tempo(origem)} -- o áudio normalizado "
        "cobriria só o começo, e a legenda acabaria aí. Isso costuma ser "
        "defeito no arquivo de mídia a partir desse ponto; tente "
        "remuxá-lo (ffmpeg -i entrada.mkv -c copy saida.mkv) ou baixá-lo "
        "de novo.")
