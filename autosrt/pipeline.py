"""Orquestração do fluxo completo de tradução.

Mantém a lógica fora da interface, para que ela seja testável sem Tk.

A ordem aqui é a do plano: o texto de origem é preservado ao carregar, a
tradução escreve em ``cue.text``, e só então o arquivo é gravado. O passe de
correção de gênero (etapa 4) entra entre a tradução e a gravação, lendo os
dois textos.
"""

import shutil
from dataclasses import dataclass

from . import srt_io
from .language import detect_language, language_name
from .translate import DEFAULT_TARGET, TranslationCancelled, translate_cues


@dataclass
class PipelineResult:
    """Resumo de uma tradução concluída."""

    total: int
    translated: int
    failed: list
    detected_lang: str
    backup_path: str = None

    @property
    def language_label(self) -> str:
        return language_name(self.detected_lang)

    @property
    def failure_count(self) -> int:
        return len(self.failed)


def translate_file(input_path, output_path=None, *, target=DEFAULT_TARGET,
                   make_backup=True, progress=None, status=None,
                   cancel_event=None, translator_factory=None) -> PipelineResult:
    """Traduz um arquivo de legenda do começo ao fim.

    Args:
        input_path: arquivo .srt de entrada.
        output_path: destino. Sendo ``None``, sobrescreve a entrada.
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
    report = translate_cues(
        cues, detected_lang, target=target, progress=progress,
        cancel_event=cancel_event, translator_factory=translator_factory)

    announce("Gravando...")
    srt_io.save_cues(cues, output_path)

    return PipelineResult(
        total=report.total,
        translated=report.translated,
        failed=report.failed,
        detected_lang=detected_lang,
        backup_path=backup)


__all__ = ["PipelineResult", "translate_file", "TranslationCancelled"]
