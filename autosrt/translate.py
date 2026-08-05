"""Tradução das legendas via Google Translate (``deep_translator``).

O motor de tradução continua sendo o Google, como antes. O que muda é o que
está em volta dele:

- O texto de origem é preservado em ``cue.source_text`` (etapa 2 do plano),
  e a formatação é devolvida ao lugar depois de traduzir.
- O número de threads é limitado e há intervalo mínimo entre requisições,
  para não ser bloqueado por limite de taxa (defeito 4).
- O retorno do tradutor é validado; falha não sobrescreve a legenda com lixo,
  e é reportada ao chamador (defeito 4).
- O contador de progresso é protegido por lock (defeito 1).
- O cancelamento interrompe o trabalho em voo em vez de apenas descartar o
  resultado no fim (defeito 3).
"""

import threading
import time
from dataclasses import dataclass, field

from .textfmt import prepare_for_translation, reassemble

# O endpoint gratuito do Google bloqueia rajadas. O fluxo antigo disparava
# todos os lotes de uma vez; estes limites trocam velocidade por não levar
# bloqueio no meio de um filme.
DEFAULT_MAX_WORKERS = 4
DEFAULT_CHUNK_SIZE = 25
DEFAULT_MAX_RETRIES = 3
DEFAULT_MIN_INTERVAL = 0.05  # segundos entre requisições, global
DEFAULT_TARGET = "pt"


class TranslationCancelled(Exception):
    """A tradução foi cancelada pelo usuário.

    O chamador não deve salvar o arquivo: as legendas em memória estão
    parcialmente traduzidas.
    """


@dataclass
class TranslationReport:
    """Resultado de uma tradução completa."""

    total: int = 0
    translated: int = 0
    failed: list = field(default_factory=list)

    @property
    def failure_count(self) -> int:
        return len(self.failed)


class _Throttle:
    """Garante um intervalo mínimo entre requisições, entre todas as threads."""

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._min_interval
        if delay:
            time.sleep(delay)


class _Progress:
    """Contador de progresso protegido por lock (defeito 1)."""

    def __init__(self, total: int, callback=None):
        self._total = total
        self._callback = callback
        self._lock = threading.Lock()
        self._done = 0

    def advance(self, amount: int = 1) -> None:
        with self._lock:
            self._done += amount
            done = self._done
        if self._callback:
            self._callback(done, self._total)


def default_translator_factory(source: str, target: str):
    """Cria um ``GoogleTranslator``.

    Importado aqui dentro para que os testes possam injetar um tradutor falso
    sem exigir a dependência nem acesso à rede.
    """
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source=source, target=target)


def _translate_segment(text, translator, throttle, cancel_event, max_retries):
    """Traduz um trecho, com nova tentativa em caso de falha.

    Devolve o texto traduzido, ou ``None`` se todas as tentativas falharem.
    """
    for attempt in range(max_retries):
        if cancel_event is not None and cancel_event.is_set():
            raise TranslationCancelled()

        throttle.wait()
        try:
            result = translator.translate(text)
        except Exception:
            result = None

        # O deep_translator pode devolver None ou string vazia sem levantar
        # exceção. O fluxo antigo gravava esse valor direto na legenda.
        if isinstance(result, str) and result.strip():
            return result

        if attempt < max_retries - 1:
            time.sleep(0.5 * (2 ** attempt))

    return None


def _translate_cue(cue, translator, throttle, cancel_event, max_retries) -> bool:
    """Traduz uma legenda no lugar. Devolve ``False`` se falhou."""
    prefix, segments, suffix, dialogue = prepare_for_translation(cue.source_text)

    translated_segments = []
    for segment in segments:
        if not segment:
            translated_segments.append(segment)
            continue
        translated = _translate_segment(
            segment, translator, throttle, cancel_event, max_retries)
        if translated is None:
            return False
        translated_segments.append(translated)

    cue.text = reassemble(prefix, translated_segments, suffix, dialogue)
    return True


def translate_cues(cues, source_lang, *, target=DEFAULT_TARGET,
                   translator_factory=None, progress=None, cancel_event=None,
                   max_workers=DEFAULT_MAX_WORKERS, chunk_size=DEFAULT_CHUNK_SIZE,
                   max_retries=DEFAULT_MAX_RETRIES,
                   min_interval=DEFAULT_MIN_INTERVAL) -> TranslationReport:
    """Traduz as legendas no lugar, escrevendo em ``cue.text``.

    Args:
        cues: lista de :class:`~autosrt.cue.Cue`. Modificada no lugar.
        source_lang: código do idioma de origem.
        target: código do idioma de destino.
        translator_factory: função ``(source, target) -> tradutor``. O tradutor
            precisa ter ``translate(text) -> str``. Usada pelos testes.
        progress: chamada como ``progress(feitas, total)``. **Invocada a
            partir das threads de trabalho** — a interface precisa marshalar
            para a thread principal.
        cancel_event: ``threading.Event``. Quando acionado, o trabalho em voo
            para e :class:`TranslationCancelled` é levantada.

    Returns:
        :class:`TranslationReport` com a contagem de sucesso e a lista de
        índices que falharam. Legendas que falharam mantêm o texto de origem.

    Raises:
        TranslationCancelled: se ``cancel_event`` for acionado. As legendas
            ficam parcialmente traduzidas e o arquivo não deve ser salvo.
    """
    from concurrent.futures import ThreadPoolExecutor

    if translator_factory is None:
        translator_factory = default_translator_factory

    report = TranslationReport(total=len(cues))
    if not cues:
        return report

    throttle = _Throttle(min_interval)
    counter = _Progress(len(cues), progress)
    results_lock = threading.Lock()

    chunks = [cues[i:i + chunk_size] for i in range(0, len(cues), chunk_size)]

    def process_chunk(chunk):
        # Um tradutor por thread: o GoogleTranslator não promete ser seguro
        # para uso simultâneo.
        translator = translator_factory(source_lang, target)
        for cue in chunk:
            if cancel_event is not None and cancel_event.is_set():
                raise TranslationCancelled()
            ok = _translate_cue(cue, translator, throttle, cancel_event, max_retries)
            with results_lock:
                if ok:
                    report.translated += 1
                else:
                    report.failed.append(cue.index)
            counter.advance()

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        cancelled = False
        error = None
        for future in futures:
            try:
                future.result()
            except TranslationCancelled:
                cancelled = True
                # Descarta o que ainda está na fila; o que já começou para
                # sozinho no próximo cue, ao ver o evento.
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:  # pragma: no cover - defensivo
                error = error or exc
    finally:
        executor.shutdown(wait=True)

    if cancelled or (cancel_event is not None and cancel_event.is_set()):
        raise TranslationCancelled()
    if error is not None:
        raise error

    report.failed.sort()
    return report
