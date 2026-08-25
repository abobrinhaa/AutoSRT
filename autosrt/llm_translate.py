"""Tradução de legendas por modelo de linguagem, em blocos com contexto.

Diferença essencial em relação ao caminho do Google (:mod:`autosrt.translate`),
que traduz cada legenda isolada: aqui as legendas vão em blocos, acompanhadas
das falas vizinhas como contexto. É isso que permite:

- **Gíria e expressão idiomática.** "Break a leg" só é reconhecível como
  expressão quando se vê a cena; isolada, vira "quebre uma perna" e a frase
  resultante parece perfeitamente normal, sem nada que denuncie o erro. Esse
  é o motivo principal de usar um modelo aqui.
- **Concordância de gênero.** Com os rótulos de locutor vindos da diarização,
  o modelo sabe quem fala cada linha e conjuga certo, em vez de escolher no
  escuro entre "advogado" e "advogada".

Restrição inegociável do formato: a resposta precisa ter **exatamente** as
mesmas legendas da entrada, uma por uma. Se o modelo juntar ou dividir falas,
o alinhamento com os tempos quebra e a legenda inteira sai fora de sincronia.
Por isso a saída é delimitada, conferida, e há queda para tradução individual
quando a conferência falha.
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor

from .llm import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMClient, LLMError
from .translate import TranslationCancelled

# Quantas legendas por requisição, e quantas falas de vizinhança mandar junto
# apenas como contexto (não são traduzidas).
DEFAULT_BLOCK_SIZE = 20
DEFAULT_CONTEXT_LINES = 6

# Poucos workers de propósito: cada requisição já carrega um bloco inteiro de
# legendas, então o ganho é sobrepor a espera de rede entre blocos, não
# multiplicar o número de requisições por segundo contra o limite de taxa do
# provedor. Antes disso os blocos eram traduzidos um de cada vez -- um filme
# de centenas de blocos passava a maior parte do tempo apenas esperando a
# resposta de cada requisição, em sequência.
DEFAULT_MAX_WORKERS = 3

BLOCK_RE = re.compile(r"<(\d+)>(.*?)</\1>", re.DOTALL)

SYSTEM_PROMPT = """Você traduz legendas de filmes para português do Brasil.

Regras:
- Traduza o sentido, não as palavras. Gírias e expressões idiomáticas devem
  virar o equivalente natural em português, nunca tradução literal.
- Respeite o registro e a época do filme. Fala coloquial continua coloquial;
  fala formal continua formal.
- Mantenha o texto curto o suficiente para caber numa legenda.
- Use as falas vizinhas marcadas como CONTEXTO para entender a cena. Elas
  servem só para você se situar e NÃO devem ser traduzidas nem aparecer na
  resposta.
- Preste atenção em quem fala cada linha para acertar a concordância de
  gênero em português.
- Preserve as quebras de linha internas de cada legenda.
- Preserve as marcações de formatação exatamente como vieram, incluindo tags
  como <i></i> e o travessão que abre cada fala em legendas de diálogo.
- Não comente, não explique, não numere de novo. Devolva apenas os blocos.

Formato da resposta, obrigatório: para cada legenda a traduzir, devolva
exatamente um bloco <N>texto</N>, com o mesmo N que veio na entrada.
Devolva todos os blocos pedidos, nem mais nem menos. Nunca junte duas
legendas num bloco só, nunca divida uma legenda em dois blocos."""


class LLMTranslationError(Exception):
    """Falha ao traduzir com o modelo."""


def describe_speakers(speaker_genders) -> str:
    """Monta a linha de ficha dos locutores para o prompt."""
    if not speaker_genders:
        return ""
    partes = [f"{nome} é {genero}" for nome, genero in sorted(speaker_genders.items())]
    return "Quem é quem nesta cena: " + "; ".join(partes) + ".\n\n"


def build_prompt(block, context_before, context_after, source_lang,
                 speaker_genders=None) -> str:
    """Monta o pedido de tradução de um bloco."""
    linhas = []
    if source_lang:
        linhas.append(f"Idioma de origem: {source_lang}.\n")
    linhas.append(describe_speakers(speaker_genders))

    if context_before:
        linhas.append("CONTEXTO (falas anteriores, não traduza):")
        linhas.extend(_format_context(context_before))
        linhas.append("")

    linhas.append("LEGENDAS A TRADUZIR:")
    for numero, cue in block:
        marca = f" ({cue.speaker})" if cue.speaker else ""
        linhas.append(f"<{numero}>{marca and marca + ' '}{cue.source_text}</{numero}>")

    if context_after:
        linhas.append("")
        linhas.append("CONTEXTO (falas seguintes, não traduza):")
        linhas.extend(_format_context(context_after))

    return "\n".join(p for p in linhas if p is not None)


def _format_context(cues):
    saida = []
    for cue in cues:
        marca = f"({cue.speaker}) " if cue.speaker else ""
        saida.append(f"- {marca}{cue.source_text}".replace("\n", " "))
    return saida


def parse_response(text, expected_numbers) -> dict:
    """Extrai os blocos da resposta.

    Returns:
        Dicionário ``{numero: texto}`` apenas com os números esperados.
    """
    encontrados = {}
    for numero, conteudo in BLOCK_RE.findall(text or ""):
        indice = int(numero)
        if indice in expected_numbers:
            encontrados[indice] = _clean_block(conteudo)
    return encontrados


def _clean_block(text: str) -> str:
    # O modelo às vezes repete o rótulo de locutor; ele não é fala.
    text = re.sub(r"^\s*\((?:SPEAKER|SPK)[ _]?\d+\)\s*", "", text,
                  flags=re.IGNORECASE)
    return "\n".join(linha.strip() for linha in text.strip().split("\n")).strip()


class _ContadorProgresso:
    """Progresso agregado de threads concorrentes, protegido por lock."""

    def __init__(self, total, callback):
        self._total = total
        self._callback = callback
        self._lock = threading.Lock()
        self._feitas = 0

    def avancar(self) -> None:
        with self._lock:
            self._feitas += 1
            feitas = self._feitas
        if self._callback:
            self._callback(feitas, self._total)


def translate_cues_llm(cues, source_lang=None, *, client=None,
                       block_size=DEFAULT_BLOCK_SIZE,
                       context_lines=DEFAULT_CONTEXT_LINES,
                       speaker_genders=None, progress=None, cancel_event=None,
                       max_workers=DEFAULT_MAX_WORKERS, on_error=None):
    """Traduz as legendas no lugar, escrevendo em ``cue.text``.

    Os blocos são traduzidos em paralelo (``max_workers`` de cada vez, como
    o motor Google em :mod:`autosrt.translate`): cada um é uma requisição de
    rede independente, e blocos não compartilham nada entre si além do
    ``client``, que só é lido, nunca alterado, durante uma tradução.

    Args:
        cues: lista de :class:`~autosrt.cue.Cue`, modificada no lugar.
        source_lang: nome do idioma de origem, para orientar o modelo.
        client: :class:`~autosrt.llm.LLMClient` ou qualquer objeto com
            ``complete(system, user)``, seguro para ser chamado de várias
            threads ao mesmo tempo. Injetável pelos testes.
        speaker_genders: ``{"SPEAKER_00": "homem", ...}``, quando conhecido.
        progress: chamada como ``progress(feitas, total)``. **Invocada a
            partir das threads de trabalho** -- a interface precisa
            marshalar para a thread principal.
        cancel_event: ``threading.Event`` para interromper.
        max_workers: quantos blocos traduzir ao mesmo tempo.
        on_error: chamada com a :class:`~autosrt.llm.LLMError` toda vez que
            uma requisição falha (chave inválida, modelo inexistente, sem
            crédito, etc.). Sem isso o motivo real da falha é descartado, e
            quem chama só sabe que "algumas legendas falharam", sem saber
            por quê. **Invocada a partir das threads de trabalho** -- se
            precisar guardar o valor, proteja com lock. Chamada de novo a
            cada bloco/legenda que falhar, então quem só quer o último erro
            deve simplesmente sobrescrever o que guardou.

    Returns:
        Lista dos índices que não puderam ser traduzidos, em ordem
        crescente. Essas legendas mantêm o texto de origem.

    Raises:
        TranslationCancelled: se ``cancel_event`` for acionado. Blocos já em
            voo terminam; os que ainda não começaram são descartados.
    """
    if client is None:
        raise LLMTranslationError("Nenhum cliente de modelo foi configurado.")

    total = len(cues)
    if not total:
        return []

    blocos = []
    for inicio in range(0, total, block_size):
        fim = min(inicio + block_size, total)
        bloco = [(i + 1, cues[i]) for i in range(inicio, fim)]
        antes = cues[max(0, inicio - context_lines):inicio]
        depois = cues[fim:fim + context_lines]
        blocos.append((bloco, antes, depois))

    falhas = []
    falhas_lock = threading.Lock()
    contador = _ContadorProgresso(total, progress)

    def processar_bloco(bloco, antes, depois):
        if cancel_event is not None and cancel_event.is_set():
            raise TranslationCancelled()

        traduzidos = _translate_block(
            bloco, antes, depois, source_lang, speaker_genders, client,
            on_error=on_error)

        for numero, cue in bloco:
            texto = traduzidos.get(numero)
            if texto:
                cue.text = texto
            else:
                with falhas_lock:
                    falhas.append(cue.index)
            contador.avancar()

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futuros = [executor.submit(processar_bloco, *args) for args in blocos]
        cancelado = False
        erro = None
        for futuro in futuros:
            try:
                futuro.result()
            except TranslationCancelled:
                cancelado = True
                # Descarta o que ainda está na fila; o que já começou
                # termina sozinho no próximo bloco, ao ver o evento.
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:  # pragma: no cover - defensivo
                erro = erro or exc
    finally:
        executor.shutdown(wait=True)

    if cancelado or (cancel_event is not None and cancel_event.is_set()):
        raise TranslationCancelled()
    if erro is not None:
        raise erro

    falhas.sort()
    return falhas


def _translate_block(bloco, antes, depois, source_lang, speaker_genders, client,
                     on_error=None):
    """Traduz um bloco, com uma segunda tentativa e queda para individual."""
    esperados = {numero for numero, _ in bloco}
    prompt = build_prompt(bloco, antes, depois, source_lang, speaker_genders)

    try:
        resposta = client.complete(SYSTEM_PROMPT, prompt)
        traduzidos = parse_response(resposta, esperados)
    except LLMError as exc:
        traduzidos = {}
        if on_error:
            on_error(exc)

    if len(traduzidos) == len(esperados):
        return traduzidos

    # O modelo devolveu quantidade errada. Insistir no bloco inteiro costuma
    # repetir o mesmo desvio, então cada legenda que faltou vai sozinha - aí
    # não existe risco de desalinhar, ainda que se perca o contexto.
    faltando = [(n, c) for n, c in bloco if n not in traduzidos]
    for numero, cue in faltando:
        individual = _translate_single(cue, numero, antes, depois, source_lang,
                                       speaker_genders, client, on_error=on_error)
        if individual:
            traduzidos[numero] = individual
    return traduzidos


def _translate_single(cue, numero, antes, depois, source_lang, speaker_genders,
                      client, on_error=None):
    prompt = build_prompt([(numero, cue)], antes, depois, source_lang,
                          speaker_genders)
    try:
        resposta = client.complete(SYSTEM_PROMPT, prompt)
    except LLMError as exc:
        if on_error:
            on_error(exc)
        return None

    traduzidos = parse_response(resposta, {numero})
    if traduzidos:
        return traduzidos[numero]

    # Última chance: resposta sem os delimitadores, mas de uma linha só.
    limpo = _clean_block(re.sub(r"</?\d+>", "", resposta or ""))
    return limpo or None


def client_from_config(config_getter=None, **overrides):
    """Cria o cliente a partir da configuração local.

    A chave vem do ambiente ou do ``config.json``, nunca do código.

    Raises:
        LLMError: endereço customizado (ex.: servidor local) sem modelo
            configurado. ``DEFAULT_MODEL`` só é um padrão razoável para o
            OpenRouter -- aplicá-lo a qualquer outro endereço adivinharia um
            nome de modelo que quase certamente não existe lá, e o erro só
            apareceria depois, como se o modelo tivesse "sumido" sozinho.
    """
    from . import config

    getter = config_getter or config.get_setting
    api_key = overrides.pop("api_key", None) or getter(
        "openrouter_api_key", "OPENROUTER_API_KEY")
    base_url = overrides.pop("base_url", None) or getter(
        "llm_base_url", "LLM_BASE_URL") or DEFAULT_BASE_URL
    model = overrides.pop("model", None) or getter("llm_model", "LLM_MODEL")

    if not model:
        if base_url == DEFAULT_BASE_URL:
            model = DEFAULT_MODEL
        else:
            raise LLMError(
                f"Nenhum modelo configurado para {base_url}. "
                f'"{DEFAULT_MODEL}" só é um padrão razoável para o '
                "OpenRouter; para outro endereço, informe o modelo "
                "explicitamente (llm_model no config.json, ou no painel).")

    return LLMClient(api_key=api_key, base_url=base_url, model=model, **overrides)
