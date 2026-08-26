"""Linha de comando do AutoSRT.

É o motor do servidor: a interface web chama exatamente as mesmas funções.

Uso típico::

    autosrt filme.mkv           # transcreve, traduz e grava filme.srt
    autosrt legenda.srt         # só traduz
    autosrt /filmes             # a pasta inteira, um de cada vez
"""

import argparse
import logging
import os
import shlex
import sys

from . import config, pipeline, srt_io, sync, transcribe, transcribe_api
from .errors import UnsupportedSubtitleError
from .language import language_name
from .llm import LLMError
from .sync import SyncError
from .transcribe import TranscriptionError
from .translate import TranslationCancelled

EXIT_OK = 0
EXIT_ERRO = 1
EXIT_NADA_A_FAZER = 2


def get_transcribe_runner(whisper_engine, reporter):
    """Motor de transcrição alternativo ao Whisper local, via API na nuvem.

    Substitui a transcrição inteira, não só o subprocesso: quem chama recebe
    de volta ``list[Cue]`` com tempos reais, no mesmo formato que o Whisper
    local devolve — só sem diarização, que a API não oferece.

    Args:
        whisper_engine: "openrouter", "openai", ou "local"
        reporter: Reporter para status/erros

    Returns:
        Função ``runner(media_path, *, language=None, **_) -> list[Cue]``,
        ou ``None`` quando o motor é "local" (Whisper local de sempre).
    """
    if whisper_engine == "local":
        return None

    # Cada provedor tem a própria chave: usar a chave errada falharia com
    # "não autorizado" sem dar nenhuma pista do motivo.
    if whisper_engine == "openai":
        api_key = config.get_openai_api_key()
        nome_config = "openai_api_key"
    else:
        api_key = config.get_openrouter_api_key()
        nome_config = "openrouter_api_key"

    if not api_key:
        reporter.error(
            f"--whisper-api {whisper_engine} requer chave de API. "
            f"Configure com: autosrt-config set {nome_config} sk-..."
        )
        raise ValueError("Chave de API não configurada")

    def transcriber_via_api(media_path, *, language=None, **_ignorado):
        # A transcrição por API é uma única requisição bloqueante: não há
        # como cancelar no meio, diferente do processo local que é sondado
        # em loop e pode ser terminado.
        reporter.status(f"Transcrevendo via {whisper_engine} API...")
        return transcribe_api.transcribe_via_api(
            media_path, api_key=api_key, engine=whisper_engine, language=language)

    return transcriber_via_api


def build_parser():
    parser = argparse.ArgumentParser(
        prog="autosrt",
        description="Transcreve, sincroniza e traduz legendas de filmes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""exemplos:
  autosrt filme.mkv                 transcreve o áudio e traduz
  autosrt legenda.srt               traduz uma legenda existente
  autosrt legenda.ssa               converte para .srt e traduz
  autosrt /filmes                   processa a pasta inteira
  autosrt legenda.srt --deslocar -2.5      adianta a legenda em 2,5s
  autosrt legenda.srt --sincronizar filme.mkv
  autosrt filme.mkv --so-transcrever       não traduz
""")

    parser.add_argument("entradas", nargs="+", metavar="ARQUIVO",
                        help="arquivos de mídia, legendas, ou pastas")
    parser.add_argument("-o", "--saida", metavar="DIR",
                        help="pasta de destino (padrão: ao lado da entrada)")

    grupo = parser.add_argument_group("tradução")
    grupo.add_argument("--motor", choices=[pipeline.ENGINE_LLM,
                                           pipeline.ENGINE_GOOGLE],
                       default=pipeline.ENGINE_LLM,
                       help="llm traduz em blocos com contexto e acerta gíria; "
                            "google traduz linha a linha, sem contexto "
                            "(padrão: llm)")
    grupo.add_argument("--so-traduzir", action="store_true",
                       help="não transcreve, mesmo recebendo um vídeo")
    grupo.add_argument("--so-converter", action="store_true",
                       help="só converte SSA/ASS para .srt, sem traduzir")

    grupo = parser.add_argument_group("transcrição")
    grupo.add_argument("--idioma", metavar="COD",
                       help="idioma falado (ex: en). Padrão: detectar")
    grupo.add_argument("--modelo", metavar="NOME",
                       default=transcribe.DEFAULT_MODEL,
                       help=f"modelo do Whisper (padrão: {transcribe.DEFAULT_MODEL})")
    grupo.add_argument("--normalizar-audio", dest="normalize_audio",
                       choices=["auto", "sempre", "nunca"], default="auto",
                       help="corrige áudio baixo antes de transcrever, que "
                            "senão faz o Whisper alucinar em vez de "
                            "reconhecer a fala (padrão: auto, mede e só "
                            "mexe no que está fraco)")
    grupo.add_argument("--vad-metodo", metavar="NOME", dest="vad_method",
                       help="detector de fala (silero_v4_fw, silero_v5, "
                            "pyannote_v3, webrtc...). Trocar o detector é "
                            "candidato a resolver legenda com buracos "
                            f"(padrão: {transcribe.DEFAULT_VAD})")
    grupo.add_argument("--compute-type", metavar="TIPO",
                       dest="whisper_compute_type",
                       help="precisão do cálculo na GPU (auto, int8, float16). "
                            "Em placas Pascal, int8 costuma ser mais rápido "
                            f"que float16 (padrão: {transcribe.DEFAULT_COMPUTE_TYPE})")
    grupo.add_argument("--whisper-api", metavar="ENGINE",
                       choices=["openrouter", "openai", "local"],
                       default="local",
                       help="usar Whisper via API em vez de GPU local. "
                            "Requer chave configurada (padrão: local)")
    grupo.add_argument("--sem-diarizacao", action="store_true",
                       help="não identifica quem fala. Mais rápido, mas o "
                            "tradutor perde a informação que corrige o gênero")
    grupo.add_argument("--so-transcrever", action="store_true",
                       help="transcreve e para, sem traduzir")
    grupo.add_argument("--vad-sensibilidade", type=float, metavar="0-1",
                       dest="vad_threshold",
                       help="baixar (ex: 0.2) pega fala baixa/sussurrada que "
                            "fica de fora com o padrão do Whisper, ao custo "
                            "de arriscar confundir ruído de fundo com fala. "
                            "Padrão: o do próprio Whisper, sem mexer em nada")
    grupo.add_argument("--vad-silencio-min", type=int, metavar="MS",
                       dest="vad_min_silence_ms",
                       help="silêncio mínimo (ms) para a VAD considerar que "
                            "uma fala terminou. Baixar evita cortar a "
                            "palavra final de uma fala rápida")
    grupo.add_argument("--whisper-args", metavar="\"ARGS\"",
                       dest="whisper_extra_args",
                       help="argumentos extras repassados direto ao "
                            "executável do Whisper local, para opções que "
                            "este programa ainda não conhece por nome. Uma "
                            "string só, como na linha de comando; ex: "
                            "--whisper-args \"--no_speech_threshold 0.3\"")

    grupo = parser.add_argument_group("sincronia (só para legendas)")
    grupo.add_argument("--deslocar", type=float, metavar="SEG",
                       help="desloca a legenda. Positivo atrasa, negativo adianta")
    grupo.add_argument("--sincronizar", metavar="REF",
                       help="sincroniza contra um vídeo ou legenda de referência")

    parser.add_argument("-q", "--quieto", action="store_true",
                        help="não mostra progresso")
    parser.add_argument("-v", "--verboso", action="store_true",
                        help="mostra o comando montado para o Whisper e "
                             "quantas legendas a transcrição rendeu, útil "
                             "para entender um resultado inesperado")
    return parser


class Reporter:
    """Imprime o andamento em português comum, na saída de erro."""

    def __init__(self, quiet=False, stream=None):
        self.quiet = quiet
        self.stream = stream or sys.stderr
        self._ultimo = None

    def status(self, message):
        if not self.quiet:
            print(f"  {message}", file=self.stream, flush=True)

    def progress(self, done, total):
        if self.quiet or not total:
            return
        pct = int(done * 100 / total)
        # Só reimprime a cada 5%, para não inundar o terminal.
        if self._ultimo is not None and pct - self._ultimo < 5 and pct < 100:
            return
        self._ultimo = pct
        print(f"    {pct}%", file=self.stream, flush=True)

    def info(self, message):
        print(message, file=self.stream, flush=True)

    def error(self, message):
        print(f"erro: {message}", file=sys.stderr, flush=True)


def expand_inputs(entradas) -> list:
    """Resolve pastas em arquivos, mantendo só o que dá para processar."""
    arquivos = []
    for entrada in entradas:
        if os.path.isdir(entrada):
            for nome in sorted(os.listdir(entrada)):
                caminho = os.path.join(entrada, nome)
                if os.path.isfile(caminho) and _processavel(caminho):
                    arquivos.append(caminho)
        else:
            arquivos.append(entrada)
    return arquivos


def _processavel(path: str) -> bool:
    if not (pipeline.is_media(path) or pipeline.is_subtitle(path)):
        return False
    # Não reprocessa o que o próprio AutoSRT gerou.
    nome = os.path.basename(path).lower()
    return not (nome.endswith("_backup.srt") or nome.endswith(".original.srt"))


def output_for(entrada, saida_dir, sufixo=".srt") -> str:
    base = os.path.splitext(os.path.basename(entrada))[0] + sufixo
    if saida_dir:
        os.makedirs(saida_dir, exist_ok=True)
        return os.path.join(saida_dir, base)
    return os.path.join(os.path.dirname(os.path.abspath(entrada)), base)


def process_one(entrada, args, reporter) -> bool:
    """Processa um arquivo. Devolve ``True`` se deu certo."""
    if not os.path.exists(entrada):
        reporter.error(f"não encontrei {entrada}")
        return False

    reporter.info(f"\n{os.path.basename(entrada)}")

    # Ajustes de tempo pedidos explicitamente encerram o comando: mexer nos
    # tempos e traduzir na mesma passada esconderia qual dos dois quebrou.
    if args.deslocar is not None or args.sincronizar:
        return _apply_sync(entrada, args, reporter)

    if args.so_converter:
        return _convert_only(entrada, args, reporter)

    saida = args.saida and output_for(entrada, args.saida) or None

    try:
        if pipeline.is_media(entrada) and not args.so_traduzir:
            transcribe_runner = None
            if args.whisper_api != "local":
                transcribe_runner = get_transcribe_runner(args.whisper_api, reporter)

            extra_args = shlex.split(args.whisper_extra_args) \
                if args.whisper_extra_args else None
            resultado = pipeline.process_media(
                entrada, saida, engine=args.motor, language=args.idioma,
                whisper_model=args.modelo, diarize=not args.sem_diarizacao,
                translate=not args.so_transcrever,
                transcribe_runner=transcribe_runner,
                normalize_audio=args.normalize_audio,
                vad_method=args.vad_method,
                vad_threshold=args.vad_threshold,
                vad_min_silence_ms=args.vad_min_silence_ms,
                whisper_compute_type=args.whisper_compute_type,
                transcribe_extra_args=extra_args,
                progress=reporter.progress, status=reporter.status)
        elif pipeline.is_media(entrada):
            reporter.error("--so-traduzir precisa de uma legenda, não de um vídeo")
            return False
        else:
            resultado = pipeline.translate_file(
                entrada, saida, engine=args.motor,
                progress=reporter.progress, status=reporter.status)
    except TranslationCancelled:
        reporter.error("cancelado")
        return False
    except (TranscriptionError, LLMError, UnsupportedSubtitleError,
            SyncError, ValueError, OSError) as exc:
        reporter.error(str(exc))
        return False

    _report_result(resultado, reporter)
    return True


def _report_result(resultado, reporter):
    idioma = language_name(resultado.detected_lang) if resultado.detected_lang else "?"
    reporter.info(f"  {resultado.total} legendas | idioma: {idioma}")
    if resultado.failure_count:
        reporter.info(
            f"  atenção: {resultado.failure_count} legenda(s) não traduzidas, "
            "mantidas no idioma original")


def _convert_only(entrada, args, reporter) -> bool:
    """Converte para .srt sem traduzir."""
    if pipeline.is_media(entrada):
        reporter.error("--so-converter vale para legendas, não para vídeo")
        return False

    destino = args.saida and output_for(entrada, args.saida) or \
        srt_io.srt_output_path(entrada)
    if os.path.abspath(destino) == os.path.abspath(entrada):
        reporter.error(f"{os.path.basename(entrada)} já é .srt, nada a converter")
        return False

    try:
        srt_io.convert_to_srt(entrada, destino)
    except (UnsupportedSubtitleError, OSError) as exc:
        reporter.error(str(exc))
        return False

    reporter.info(f"  convertido para {os.path.basename(destino)}")
    return True


def _apply_sync(entrada, args, reporter) -> bool:
    if pipeline.is_media(entrada):
        reporter.error("os ajustes de sincronia valem para legendas, não para vídeo")
        return False

    destino = args.saida and output_for(entrada, args.saida) or \
        srt_io.srt_output_path(entrada)

    try:
        if args.sincronizar:
            reporter.status("Sincronizando automaticamente...")
            sync.autosync(entrada, args.sincronizar, destino)
        else:
            cues = srt_io.load_cues(entrada)
            sync.shift_cues(cues, round(args.deslocar * 1000))
            srt_io.save_cues(cues, destino)
            reporter.status(f"Deslocado em {args.deslocar:+.3f}s")
    except (SyncError, UnsupportedSubtitleError, OSError) as exc:
        reporter.error(str(exc))
        return False

    reporter.info(f"  gravado em {os.path.basename(destino)}")
    return True


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    reporter = Reporter(quiet=args.quieto)

    # Só com -v, para não poluir a saída normal: o comando do Whisper é
    # longo e só interessa quando o resultado saiu diferente do esperado.
    logging.basicConfig(
        level=logging.INFO if args.verboso else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")

    arquivos = expand_inputs(args.entradas)
    if not arquivos:
        reporter.error("nenhum arquivo de mídia ou legenda encontrado")
        return EXIT_NADA_A_FAZER

    sucessos = 0
    for entrada in arquivos:
        if process_one(entrada, args, reporter):
            sucessos += 1

    total = len(arquivos)
    if total > 1:
        reporter.info(f"\n{sucessos} de {total} arquivo(s) processado(s).")
    return EXIT_OK if sucessos == total else EXIT_ERRO


if __name__ == "__main__":
    sys.exit(main())
