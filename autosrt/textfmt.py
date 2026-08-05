"""Preservação de formatação ao redor da tradução.

O fluxo antigo destruía a formatação antes de traduzir: fundia as quebras de
linha num espaço e removia todas as tags, sem nunca restaurá-las. Isso apagava
a marcação de itálico e, pior para o passe de gênero, apagava a separação de
turnos de fala em legendas de diálogo.

Este módulo separa o que pode ser preservado do que precisa ser removido:

- Tags que envolvem a legenda inteira (``<i>...</i>``) são extraídas antes da
  tradução e recolocadas depois.
- Tags no meio da frase são removidas. Depois da tradução a ordem das palavras
  muda, e não há como remapear a posição delas de forma confiável.
- Diálogos (linhas iniciadas por travessão) têm cada turno traduzido
  separadamente, preservando quem fala o quê.
- Texto corrido é reagrupado em até duas linhas equilibradas, como manda a
  prática de legendagem.
"""

import re

# Tags de estilo/posicionamento (HTML do SRT e chaves do formato SSA).
TAG_RE = re.compile(r"<[^>]+>|\{[^}]*\}")
# Só tags HTML abrem envolvimento: as chaves do SSA não têm fechamento
# correspondente e são tratadas como ruído interno.
OPEN_TAGS_RE = re.compile(r"^((?:<[A-Za-z][^>]*>)+)")
TAG_NAME_RE = re.compile(r"<([A-Za-z][^\s/>]*)")

# Travessão de diálogo no início da linha.
DASH_RE = re.compile(r"^\s*[-–—]\s*")

# Caracteres invisíveis que atrapalham a tradução e a comparação de textos.
INVISIBLE_CHARS = ("﻿", "​", "‎", "‏")

EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE)

# Largura padrão de linha de legenda.
MAX_CHARS_PER_LINE = 42
MAX_LINES = 2


def strip_invisible(text: str) -> str:
    """Remove BOM, espaços de largura zero e marcas de direção."""
    for ch in INVISIBLE_CHARS:
        text = text.replace(ch, "")
    return text


def split_wrapping_tags(text: str):
    """Separa as tags que envolvem o texto inteiro.

    Devolve ``(prefixo, miolo, sufixo)``. Se as tags não envolverem tudo,
    prefixo e sufixo vêm vazios e o miolo é o texto original.

    >>> split_wrapping_tags("<i>Olá</i>")
    ('<i>', 'Olá', '</i>')
    """
    open_match = OPEN_TAGS_RE.match(text)
    if not open_match:
        return "", text, ""

    prefix = open_match.group(1)
    names = TAG_NAME_RE.findall(prefix)
    if not names:
        return "", text, ""

    # O fechamento tem que ser exatamente o espelho da abertura. Sem isso,
    # em "<i>Olá <b>mundo</b></i>" o </b> interno seria confundido com parte
    # do envolvimento.
    suffix = "".join(f"</{name}>" for name in reversed(names))
    if not text.endswith(suffix):
        return "", text, ""

    core = text[len(prefix):len(text) - len(suffix)]
    if not core:
        return "", text, ""
    return prefix, core, suffix


def is_dialogue(text: str) -> bool:
    """Diz se a legenda contém turnos de fala separados por travessão.

    Exige pelo menos duas linhas marcadas, para não confundir com uma única
    linha que apenas começa com hífen.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    return sum(1 for ln in lines if DASH_RE.match(ln)) >= 2


def split_dialogue_turns(text: str) -> list:
    """Separa os turnos de fala, já sem o travessão."""
    turns = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        turns.append(DASH_RE.sub("", line).strip())
    return turns


def prepare_for_translation(text: str):
    """Prepara uma legenda para ser traduzida.

    Devolve ``(prefixo, segmentos, sufixo, era_dialogo)``, onde ``segmentos``
    é a lista de trechos a mandar para o tradutor. Diálogos rendem um segmento
    por turno de fala; o resto rende um único segmento.
    """
    text = strip_invisible(text)
    prefix, core, suffix = split_wrapping_tags(text)

    dialogue = is_dialogue(core)
    if dialogue:
        segments = split_dialogue_turns(core)
    else:
        segments = [core.replace("\n", " ")]

    # Tags que sobraram estão no meio da frase e não sobrevivem à tradução.
    segments = [_clean_segment(seg) for seg in segments]
    return prefix, segments, suffix, dialogue


def _clean_segment(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def reassemble(prefix: str, segments, suffix: str, dialogue: bool) -> str:
    """Remonta a legenda traduzida, devolvendo a formatação ao lugar."""
    if dialogue:
        body = "\n".join(f"- {seg}" for seg in segments if seg)
    else:
        body = rewrap(" ".join(seg for seg in segments if seg))
    return f"{prefix}{body}{suffix}"


def rewrap(text: str, max_chars: int = MAX_CHARS_PER_LINE,
           max_lines: int = MAX_LINES) -> str:
    """Reagrupa o texto em até ``max_lines`` linhas equilibradas.

    A tradução para português costuma ficar mais longa que o original em
    inglês, então reagrupar evita a linha única gigante que o fluxo antigo
    produzia.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text

    words = text.split(" ")
    if len(words) < 2:
        return text

    # Ponto de corte mais próximo do meio, para as linhas ficarem parecidas.
    best_index, best_cost = 1, None
    for i in range(1, len(words)):
        first = " ".join(words[:i])
        second = " ".join(words[i:])
        cost = abs(len(first) - len(second))
        if len(first) > max_chars or len(second) > max_chars:
            # Penaliza estouro, mas não descarta: melhor estourar do que
            # devolver o texto sem nenhuma quebra.
            cost += 1000
        if best_cost is None or cost < best_cost:
            best_index, best_cost = i, cost

    first = " ".join(words[:best_index])
    second = " ".join(words[best_index:])

    if max_lines > 2 and len(second) > max_chars:
        return first + "\n" + rewrap(second, max_chars, max_lines - 1)
    return f"{first}\n{second}"
