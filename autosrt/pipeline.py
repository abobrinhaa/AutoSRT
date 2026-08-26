"""Orquestração do fluxo completo de tradução.

Mantém a lógica fora da interface, para que ela seja testável sem Tk.

A ordem aqui é a do plano: o texto de origem é preservado ao carregar, a
tradução escreve em ``cue.text``, e só então o arquivo é gravado. O passe de
correção de gênero (etapa 4) entra entre a tradução e a gravação, lendo os
dois textos.
"""

import contextlib
import logging
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass

from . import llm_translate, srt_io
from .language import detect_language, language_name
from .llm import LLMError
from .translate import DEFAULT_TARGET, TranslationCancelled, translate_cues

logger = logging.getLogger(__name__)

#: Abaixo disto, uma legenda que falhou por inteiro não é motivo para
#: interromper o trabalho -- um arquivo de teste com uma ou duas falas pode
#: legitimamente ter uma que o modelo não consiga encaixar no formato, e a
#: degradação graciosa (mantém o texto original só naquela linha) continua
#: sendo o comportamento certo. Acima disto, 100% de falha não é mais "uma
#: linha difícil": é sinal de que a chamada à API está quebrada por inteiro
#: (chave, modelo ou crédito), e o arquivo sairia todo no idioma original
#: sem nenhum aviso visível se isso não virasse erro.
MIN_CUES_PARA_FALHA_TOTAL_SER_ERRO = 3

#: Motor padrão. O modelo de linguagem traduz em blocos com contexto, que é
#: o que permite acertar gíria e concordância de gênero. O Google fica como
#: alternativa: traduz cada legenda isolada, sem contexto nenhum, mas não
#: depende de chave, de crédito nem de internet estável.
ENGINE_LLM = "llm"
ENGINE_GOOGLE = "google"
DEFAULT_ENGINE = ENGINE_LLM

# Fatia da barra que cabe à transcrição quando o trabalho também traduz. O
# Whisper lendo o áudio é de longe a parte lenta; a tradução leva o resto.
# Serve para a barra andar sempre para frente, e é o que faz a previsão de
# tempo bater: errar o peso não trava nada, só torce a estimativa.
PESO_TRANSCRICAO = 70


@dataclass
class PipelineResult:
    """Resumo de uma tradução concluída."""

    total: int
    translated: int
    failed: list
    detected_lang: str
    backup_path: str = None
    engine: str = DEFAULT_ENGINE

    @property
    def language_label(self) -> str:
        return language_name(self.detected_lang)

    @property
    def failure_count(self) -> int:
        return len(self.failed)


def translate_file(input_path, output_path=None, *, target=DEFAULT_TARGET,
                   engine=DEFAULT_ENGINE, llm_client=None, speaker_genders=None,
                   make_backup=True, progress=None, status=None,
                   cancel_event=None, translator_factory=None) -> PipelineResult:
    """Traduz um arquivo de legenda do começo ao fim.

    Args:
        input_path: arquivo .srt de entrada.
        output_path: destino. Sendo ``None``, sobrescreve a entrada.
        engine: ``"llm"`` traduz em blocos com contexto (acerta gíria e
            gênero); ``"google"`` traduz legenda por legenda, isolada.
        llm_client: cliente já pronto. Sendo ``None``, é montado a partir da
            configuração local.
        speaker_genders: ``{"SPEAKER_00": "homem", ...}`` vindo da diarização,
            usado apenas pelo motor ``llm``.
        make_backup: cria ``<nome>_backup.srt`` antes de sobrescrever.
        progress: chamada como ``progress(feitas, total)``, a partir das
            threads de trabalho.
        status: chamada com uma mensagem curta a cada mudança de fase.
        cancel_event: ``threading.Event`` para interromper.
        translator_factory: injeção usada pelos testes.

    Returns:
        :class:`PipelineResult`.

    Raises:
        TranslationCancelled: cancelado pelo usuário. **O arquivo não é
            gravado** — as legendas em memória estão parcialmente traduzidas.
    """
    # Entrada SSA/ASS grava num .srt irmão, para não sobrescrever o original
    # com um formato diferente do que ele tem.
    if output_path is None:
        output_path = srt_io.srt_output_path(input_path)

    def announce(message):
        if status:
            status(message)

    announce("Lendo arquivo...")
    cues = srt_io.load_cues(input_path)

    announce("Detectando idioma...")
    detected_lang = detect_language(cues)

    backup = None
    if make_backup and output_path == input_path:
        backup = srt_io.backup_path(input_path)
        try:
            shutil.copy(input_path, backup)
        except OSError:
            # Backup é conveniência, não pré-requisito: seguir sem ele é
            # melhor do que abortar uma tradução que o usuário já pediu.
            backup = None

    announce(f"Traduzindo de {language_name(detected_lang)}...")
    if engine == ENGINE_LLM:
        translated, failed = _translate_with_llm(
            cues, detected_lang, llm_client=llm_client,
            speaker_genders=speaker_genders, progress=progress,
            cancel_event=cancel_event)
    else:
        report = translate_cues(
            cues, detected_lang, target=target, progress=progress,
            cancel_event=cancel_event, translator_factory=translator_factory)
        translated, failed = report.translated, report.failed

    announce("Gravando...")
    srt_io.save_cues(cues, output_path)

    return PipelineResult(
        total=len(cues),
        translated=translated,
        failed=failed,
        detected_lang=detected_lang,
        backup_path=backup,
        engine=engine)


#: Como decidir se o áudio é normalizado antes de transcrever.
NORMALIZE_AUTO = "auto"
NORMALIZE_SEMPRE = "sempre"
NORMALIZE_NUNCA = "nunca"


@contextlib.contextmanager
def _audio_preparado(media_path, modo, *, announce=None):
    """Entrega o caminho a transcrever, normalizado quando fizer sentido.

    Devolve o próprio ``media_path`` quando não há o que fazer, ou um WAV
    normalizado numa pasta temporária que é apagada ao sair do contexto.

    ``auto`` (padrão) mede o volume e só normaliza o que está fraco: quem
    tem áudio bom não paga a conversão, e quem tem áudio ruim não precisa
    descobrir sozinho que precisava mexer nisso -- foi um caso que levou
    horas de investigação para achar, justamente porque não dá erro
    nenhum, só uma legenda com buracos.

    Falha ao normalizar não interrompe o trabalho: transcrever com o áudio
    original é pior que com ele normalizado, mas é muito melhor que não
    transcrever.
    """
    from . import audio as audio_module

    if modo == NORMALIZE_NUNCA or not modo:
        yield media_path
        return

    if modo == NORMALIZE_AUTO:
        if announce:
            announce("Conferindo o volume do áudio...")
        medido = audio_module.medir_volume_medio(media_path)
        if medido is None:
            # "Não sei" não pode passar por "está bom". Sem ffmpeg, a
            # medição falha para todo arquivo, e tratar isso em silêncio
            # desliga a normalização inteira sem ninguém perceber -- o
            # mesmo tipo de falha muda que esta função existe para evitar.
            logger.warning(
                "não deu para medir o volume de %s (ffmpeg ausente ou "
                "arquivo sem áudio); seguindo sem normalizar", media_path)
            if announce:
                announce("Não deu para medir o volume (ffmpeg ausente?); "
                         "seguindo sem normalizar.")
            yield media_path
            return
        if medido >= audio_module.LIMIAR_VOLUME_BAIXO:
            yield media_path
            return
        if announce:
            announce(f"Áudio fraco ({medido:.1f} dB): normalizando antes "
                     "de transcrever...")
    elif announce:
        announce("Normalizando o áudio...")

    pasta = tempfile.mkdtemp(prefix="autosrt-audio-")
    try:
        destino = os.path.join(
            pasta, os.path.splitext(os.path.basename(media_path))[0] + ".wav")
        try:
            yield audio_module.normalizar_para_wav(media_path, destino)
        except audio_module.AudioError as exc:
            logger.warning("normalização do áudio falhou (%s); "
                           "seguindo com o áudio original", exc)
            if announce:
                announce("Não deu para normalizar o áudio; seguindo assim mesmo.")
            yield media_path
    finally:
        shutil.rmtree(pasta, ignore_errors=True)


def _translate_with_llm(cues, detected_lang, *, llm_client, speaker_genders,
                        progress, cancel_event):
    from . import config

    client = llm_client or llm_translate.client_from_config()

    # Os blocos rodam em paralelo (várias threads), então "o último erro" é
    # só o mais recente a chegar, não necessariamente o primeiro -- basta
    # para dar um motivo concreto ao usuário em vez de nenhum.
    ultimo_erro_lock = threading.Lock()
    ultimo_erro = []

    def registrar_erro(exc):
        with ultimo_erro_lock:
            ultimo_erro[:] = [exc]

    falhas = llm_translate.translate_cues_llm(
        cues, language_name(detected_lang), client=client,
        block_size=config.get_llm_block_size(),
        speaker_genders=speaker_genders, progress=progress,
        cancel_event=cancel_event, on_error=registrar_erro)
    translated = len(cues) - len(falhas)

    if (translated == 0 and len(cues) >= MIN_CUES_PARA_FALHA_TOTAL_SER_ERRO):
        motivo = f" Motivo: {ultimo_erro[0]}" if ultimo_erro else ""
        raise LLMError(
            f"A tradução falhou para as {len(cues)} legendas do arquivo -- "
            "nenhuma foi traduzida. O arquivo ficaria todo no idioma "
            f"original sem nenhum aviso se isso não virasse erro agora.{motivo} "
            "Confira a chave de API, o modelo e o endereço configurados "
            "(painel de configuração, ou openrouter_api_key/llm_model/"
            "llm_base_url no config.json).")

    return translated, falhas


#: Extensões tratadas como mídia (transcrever antes de traduzir). Legendas
#: são reconhecidas por :data:`SUBTITLE_EXTENSIONS`; o que não for nem uma
#: coisa nem outra é recusado com mensagem clara.
MEDIA_EXTENSIONS = {
    ".mkv", ".mp4", ".avi", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".ts",
    ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma",
}
SUBTITLE_EXTENSIONS = {".srt", ".ssa", ".ass"}


#: Onde ficam as transcrições no idioma falado. É subpasta, e não um irmão
#: do vídeo, porque tocador de vídeo casa legenda pelo nome do arquivo: um
#: "filme.original.srt" ao lado do "filme.mkv" faria o VLC oferecer duas
#: faixas de legenda e obrigar a escolher entre elas.
ORIGINALS_DIRNAME = "originais"


def is_media(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in MEDIA_EXTENSIONS


def original_path_for(output_path: str) -> str:
    """Caminho da transcrição no idioma falado, correspondente à saída."""
    pasta = os.path.join(os.path.dirname(os.path.abspath(output_path)),
                         ORIGINALS_DIRNAME)
    nome = os.path.splitext(os.path.basename(output_path))[0] + ".srt"
    return os.path.join(pasta, nome)


def is_subtitle(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUBTITLE_EXTENSIONS


def process_media(media_path, output_path=None, *, engine=DEFAULT_ENGINE,
                  language=None, whisper_model=None, diarize=True,
                  whisper_path=None, llm_client=None, translate=True,
                  progress=None, status=None, cancel_event=None,
                  transcribe_runner=None, translator_factory=None,
                  keep_original=True, vad_method=None, vad_threshold=None,
                  vad_min_silence_ms=None, whisper_compute_type=None,
                  normalize_audio=NORMALIZE_AUTO,
                  condition_on_previous_text=None,
                  hallucination_silence_threshold=None,
                  transcribe_extra_args=None) -> PipelineResult:
    """Transcreve um arquivo de mídia e traduz o resultado.

    É o caminho completo: o áudio vira legenda já sincronizada e com o
    locutor de cada fala marcado, e essa legenda é traduzida em seguida.

    Args:
        media_path: vídeo ou áudio.
        output_path: destino do .srt traduzido. Sendo ``None``, grava ao lado
            da mídia com o mesmo nome.
        language: idioma falado, se você souber. ``None`` deixa o Whisper
            detectar.
        diarize: liga a marcação de locutor, que é o que permite ao tradutor
            acertar a concordância de gênero.
        translate: sendo ``False``, para depois de transcrever.
        keep_original: grava também o ``<nome>.original.srt`` no idioma
            falado, útil para conferir a transcrição.
        vad_method: detector de fala (``silero_v4_fw``, ``silero_v5``...).
            ``None`` usa o :data:`autosrt.transcribe.DEFAULT_VAD`.
        vad_threshold: veja :func:`autosrt.transcribe.build_command`. Sem
            efeito quando ``transcribe_runner`` é usado (motor via API não
            passa pela VAD local).
        vad_min_silence_ms: idem.
        whisper_compute_type: ``auto`` (padrão), ``int8``, ``float16``...
            ``None`` deixa o CTranslate2 escolher. Idem quanto ao
            ``transcribe_runner``.
        normalize_audio: ``"auto"`` (padrão) normaliza só quando o volume
            está fraco o bastante para atrapalhar o Whisper; ``"sempre"``
            normaliza sem medir; ``"nunca"`` desliga. Veja
            :mod:`autosrt.audio` para o caso que motivou isso.
        condition_on_previous_text: ``None`` (padrão) usa o padrão do
            :mod:`autosrt.transcribe` (desligado -- ver
            ``DEFAULT_CONDITION_ON_PREVIOUS_TEXT``), que evita que uma
            alucinação em trecho de silêncio/trilha sonora se realimente
            nos trechos seguintes. ``True`` liga de volta o comportamento
            padrão do próprio Whisper.
        hallucination_silence_threshold: segundos de silêncio que o Whisper
            pula quando desconfia de alucinação. ``None`` (padrão) não mexe
            nisso.
        transcribe_extra_args: argumentos extras repassados direto ao
            executável do Whisper local.

    Returns:
        :class:`PipelineResult`.
    """
    from . import transcribe as transcribe_module

    def announce(message):
        if status:
            status(message)

    if output_path is None:
        output_path = os.path.splitext(media_path)[0] + ".srt"

    announce("Transcrevendo o áudio... esta é a parte demorada.")

    # As duas fases dividem uma barra só. Antes cada uma contava de 0 a 100 na
    # sua vez, então a barra enchia, voltava para trás e enchia de novo -- e
    # ninguém conseguia ver quanto faltava para o fim de verdade.
    peso_audio = PESO_TRANSCRICAO if translate else 100

    def progresso_da_traducao(feitas, total):
        if progress and total:
            andado = (100 - peso_audio) * feitas / total
            progress(int(peso_audio + andado), 100)

    if transcribe_runner is not None:
        # Motor alternativo (ex.: transcrição via API na nuvem): substitui o
        # Whisper local por inteiro. A interface é a mesma -- devolve
        # list[Cue] -- mas é uma única chamada bloqueante, sem progresso
        # incremental nem diarização.
        cues = transcribe_runner(media_path, language=language,
                                 cancel_event=cancel_event)
        if progress:
            progress(peso_audio, 100)
    else:
        kwargs = {
            "language": language,
            "diarize": transcribe_module.DEFAULT_DIARIZE_MODEL if diarize else None,
            "executable": whisper_path,
            "cancel_event": cancel_event,
            "output_dir": os.path.dirname(os.path.abspath(output_path)) or ".",
            "vad_threshold": vad_threshold,
            "vad_min_silence_ms": vad_min_silence_ms,
            "extra_args": transcribe_extra_args,
        }
        # Só entra no kwargs quando informado -- sem isso, o padrão de
        # transcribe.transcribe() (desligado) vale sozinho, sem precisar
        # ser repetido aqui.
        if condition_on_previous_text is not None:
            kwargs["condition_on_previous_text"] = condition_on_previous_text
        if hallucination_silence_threshold is not None:
            kwargs["hallucination_silence_threshold"] = hallucination_silence_threshold
        if whisper_model:
            kwargs["model"] = whisper_model
        if whisper_compute_type:
            kwargs["compute_type"] = whisper_compute_type
        if vad_method:
            kwargs["vad"] = vad_method
        if progress:
            kwargs["progress"] = lambda pct: progress(pct * peso_audio // 100, 100)

        # Áudio fraco demais faz o Whisper trocar a fala por alucinação
        # ("Thank you." em cima do ruído) e devolver trechos inteiros
        # vazios, sem erro nenhum -- ver autosrt.audio. Normalizar antes
        # resolve, e o WAV vai para o Whisper no lugar da mídia.
        with _audio_preparado(media_path, normalize_audio,
                              announce=announce) as entrada:
            if entrada != media_path:
                # O Whisper nomeia o .srt pelo arquivo de entrada. Com o WAV
                # temporário no lugar da mídia, deixar o output_dir apontando
                # para a pasta do usuário largaria lá um .srt com o nome do
                # temporário; mandando para a mesma pasta descartável, ele
                # sai junto.
                kwargs["output_dir"] = os.path.dirname(entrada)
            cues = transcribe_module.transcribe(entrada, **kwargs)

    if not cues:
        raise ValueError("O Whisper não encontrou fala nenhuma no arquivo.")

    detected_lang = language or _safe_detect(cues)

    if keep_original:
        srt_io.save_cues(cues, original_path_for(output_path))

    if not translate:
        srt_io.save_cues(cues, output_path)
        return PipelineResult(total=len(cues), translated=0, failed=[],
                              detected_lang=detected_lang, engine="nenhum")

    speakers = transcribe_module.speakers_in(cues)
    if speakers:
        announce(f"Traduzindo... {len(speakers)} locutor(es) identificado(s).")
    else:
        announce("Traduzindo...")

    if engine == ENGINE_LLM:
        translated, failed = _translate_with_llm(
            cues, detected_lang, llm_client=llm_client, speaker_genders=None,
            progress=progresso_da_traducao, cancel_event=cancel_event)
    else:
        report = translate_cues(
            cues, detected_lang, progress=progresso_da_traducao,
            cancel_event=cancel_event,
            translator_factory=translator_factory)
        translated, failed = report.translated, report.failed

    announce("Gravando...")
    srt_io.save_cues(cues, output_path)

    return PipelineResult(total=len(cues), translated=translated, failed=failed,
                          detected_lang=detected_lang, engine=engine)


def _safe_detect(cues) -> str:
    """Detecta o idioma sem derrubar o processo se a amostra for pobre."""
    try:
        return detect_language(cues)
    except Exception:
        return ""


__all__ = ["PipelineResult", "translate_file", "process_media",
           "TranslationCancelled", "ENGINE_LLM", "ENGINE_GOOGLE",
           "is_media", "is_subtitle"]
