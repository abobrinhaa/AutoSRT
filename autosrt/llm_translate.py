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

from .llm import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMClient, LLMError

# Quantas legendas por requisição, e quantas falas de vizinhança mandar junto
# apenas como contexto (não são traduzidas).
DEFAULT_BLOCK_SIZE = 20
DEFAULT_CONTEXT_LINES = 6

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


def translate_cues_llm(cues, source_lang=None, *, client=None,
                       block_size=DEFAULT_BLOCK_SIZE,
                       context_lines=DEFAULT_CONTEXT_LINES,
                       speaker_genders=None, progress=None, cancel_event=None):
    """Traduz as legendas no lugar, escrevendo em ``cue.text``.

    Args:
        cues: lista de :class:`~autosrt.cue.Cue`, modificada no lugar.
        source_lang: nome do idioma de origem, para orientar o modelo.
        client: :class:`~autosrt.llm.LLMClient` ou qualquer objeto com
            ``complete(system, user)``. Injetável pelos testes.
        speaker_genders: ``{"SPEAKER_00": "homem", ...}``, quando conhecido.
        progress: chamada como ``progress(feitas, total)``.
        cancel_event: ``threading.Event`` para interromper.

    Returns:
        Lista dos índices que não puderam ser traduzidos. Essas legendas
        mantêm o texto de origem.
    """
    if client is None:
        raise LLMTranslationError("Nenhum cliente de modelo foi configurado.")

    total = len(cues)
    falhas = []
    feitas = 0

    for inicio in range(0, total, block_size):
        if cancel_event is not None and cancel_event.is_set():
            from .translate import TranslationCancelled
            raise TranslationCancelled()

        fim = min(inicio + block_size, total)
        bloco = [(i + 1, cues[i]) for i in range(inicio, fim)]
        antes = cues[max(0, inicio - context_lines):inicio]
        depois = cues[fim:fim + context_lines]

        traduzidos = _translate_block(
            bloco, antes, depois, source_lang, speaker_genders, client)

        for numero, cue in bloco:
            texto = traduzidos.get(numero)
            if texto:
                cue.text = texto
            else:
                falhas.append(cue.index)
            feitas += 1
            if progress:
                progress(feitas, total)

    return falhas


def _translate_block(bloco, antes, depois, source_lang, speaker_genders, client):
    """Traduz um bloco, com uma segunda tentativa e queda para individual."""
    esperados = {numero for numero, _ in bloco}
    prompt = build_prompt(bloco, antes, depois, source_lang, speaker_genders)

    try:
        resposta = client.complete(SYSTEM_PROMPT, prompt)
        traduzidos = parse_response(resposta, esperados)
    except LLMError:
        traduzidos = {}

    if len(traduzidos) == len(esperados):
        return traduzidos

    # O modelo devolveu quantidade errada. Insistir no bloco inteiro costuma
    # repetir o mesmo desvio, então cada legenda que faltou vai sozinha - aí
    # não existe risco de desalinhar, ainda que se perca o contexto.
    faltando = [(n, c) for n, c in bloco if n not in traduzidos]
    for numero, cue in faltando:
        individual = _translate_single(cue, numero, antes, depois, source_lang,
                                       speaker_genders, client)
        if individual:
            traduzidos[numero] = individual
    return traduzidos


def _translate_single(cue, numero, antes, depois, source_lang, speaker_genders,
                      client):
    prompt = build_prompt([(numero, cue)], antes, depois, source_lang,
                          speaker_genders)
    try:
        resposta = client.complete(SYSTEM_PROMPT, prompt)
    except LLMError:
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
    """
    from . import config

    getter = config_getter or config.get_setting
    api_key = overrides.pop("api_key", None) or getter(
        "openrouter_api_key", "OPENROUTER_API_KEY")
    base_url = overrides.pop("base_url", None) or getter(
        "llm_base_url", "LLM_BASE_URL") or DEFAULT_BASE_URL
    model = overrides.pop("model", None) or getter(
        "llm_model", "LLM_MODEL") or DEFAULT_MODEL

    return LLMClient(api_key=api_key, base_url=base_url, model=model, **overrides)
