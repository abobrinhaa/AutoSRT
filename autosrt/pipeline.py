"""Orquestração do fluxo completo de tradução.

Mantém a lógica fora da interface, para que ela seja testável sem Tk.

A ordem aqui é a do plano: o texto de origem é preservado ao carregar, a
tradução escreve em ``cue.text``, e só então o arquivo é gravado. O passe de
correção de gênero (etapa 4) entra entre a tradução e a gravação, lendo os
dois textos.
"""

import shutil
from dataclasses import dataclass

from . import llm_translate, srt_io
from .language import detect_language, language_name
from .translate import DEFAULT_TARGET, TranslationCancelled, translate_cues

#: Motor padrão. O modelo de linguagem traduz em blocos com contexto, que é
#: o que permite acertar gíria e concordância de gênero. O Google fica como
#: alternativa: traduz cada legenda isolada, sem contexto nenhum, mas não
#: depende de chave, de crédito nem de internet estável.
ENGINE_LLM = "llm"
ENGINE_GOOGLE = "google"
DEFAULT_ENGINE = ENGINE_LLM


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


def _translate_with_llm(cues, detected_lang, *, llm_client, speaker_genders,
                        progress, cancel_event):
    client = llm_client or llm_translate.client_from_config()
    falhas = llm_translate.translate_cues_llm(
        cues, language_name(detected_lang), client=client,
        speaker_genders=speaker_genders, progress=progress,
        cancel_event=cancel_event)
    return len(cues) - len(falhas), falhas


__all__ = ["PipelineResult", "translate_file", "TranslationCancelled"]
