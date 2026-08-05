"""Conversão de SSA/ASS para o modelo interno.

O formato SSA carrega coisas que o SRT não tem: estilos nomeados,
posicionamento na tela, rotação, karaokê, camadas. Nada disso sobrevive à
conversão — é achatado de propósito. O que dá para preservar é o essencial
para legenda: itálico, negrito, sublinhado e a quebra de linha.

Também é comum um arquivo SSA trazer vários eventos no mesmo intervalo de
tempo (dois personagens falando, ou um letreiro posicionado junto da fala).
Esses eventos são fundidos numa única legenda, porque o SRT não tem como
exibir dois blocos independentes ao mesmo tempo.

Limitação conhecida da fusão: os eventos fundidos viram linhas de uma mesma
legenda e, se não estiverem marcados com travessão, o passe de tradução os
trata como uma frase única. Para letreiros e legendas posicionadas — o caso
comum — isso não incomoda; para dois personagens falando ao mesmo tempo, a
tradução sai menos precisa do que se fossem separados.
"""

import os
import re

from .cue import Cue
from .errors import UnsupportedSubtitleError

# Identificadores de formato do pysubs2 por extensão. Passar o formato
# explicitamente evita depender da autodetecção, que exige um cabeçalho
# completo e falha em arquivos reais que trazem só a seção [Events].
FORMAT_BY_EXTENSION = {".ass": "ass", ".ssa": "ssa"}

# Bloco de override do SSA: {\i1}, {\pos(100,200)}, {\i1\b1}...
OVERRIDE_RE = re.compile(r"\{([^}]*)\}")
# Alternâncias de estilo que têm equivalente em SRT.
TOGGLE_RE = re.compile(r"\\([ibu])([01])")
TAG_RE = re.compile(r"<(/?)([ibu])>")


def convert_text(text: str) -> str:
    """Converte o texto de um evento SSA para o texto de uma legenda SRT."""
    text = OVERRIDE_RE.sub(_convert_override_block, text)

    # \N é quebra obrigatória; \n é quebra "leve" que o renderizador pode
    # ignorar; \h é espaço não separável.
    text = text.replace("\\N", "\n").replace("\\h", " ").replace("\\n", " ")

    # Barras invertidas soltas que sobraram não significam nada em SRT.
    text = text.replace("\\", "")

    text = "\n".join(line.strip() for line in text.split("\n"))
    return _balance_tags(text).strip()


def _convert_override_block(match) -> str:
    """Converte um bloco ``{...}``, mantendo só o que o SRT entende."""
    return "".join(
        f"<{tag}>" if state == "1" else f"</{tag}>"
        for tag, state in TOGGLE_RE.findall(match.group(1)))


def _balance_tags(text: str) -> str:
    """Fecha tags abertas e descarta fechamentos órfãos.

    Um evento SSA costuma abrir itálico sem fechar, porque o fim do evento já
    encerra o estilo. Em SRT isso vazaria para as legendas seguintes.
    """
    open_stack = []
    result = []
    position = 0

    for match in TAG_RE.finditer(text):
        result.append(text[position:match.start()])
        position = match.end()
        closing, name = match.group(1), match.group(2)

        if not closing:
            open_stack.append(name)
            result.append(f"<{name}>")
        elif name in open_stack:
            # Fecha o que estiver aberto acima dele, para não cruzar tags.
            while open_stack:
                current = open_stack.pop()
                result.append(f"</{current}>")
                if current == name:
                    break
        # Fechamento sem abertura correspondente é descartado.

    result.append(text[position:])
    while open_stack:
        result.append(f"</{open_stack.pop()}>")

    return "".join(result)


def _load_file(path: str, encoding: str):
    """Abre o arquivo com o pysubs2, tentando os formatos plausíveis.

    Raises:
        UnsupportedSubtitleError: se nenhuma tentativa funcionar.
    """
    import pysubs2

    extension = os.path.splitext(path)[1].lower()
    # Autodetecção primeiro; depois o formato sugerido pela extensão; por
    # último "ass", que é o parser mais tolerante dos dois.
    attempts = [None, FORMAT_BY_EXTENSION.get(extension), "ass"]

    last_error = None
    seen = set()
    for fmt in attempts:
        if fmt in seen:
            continue
        seen.add(fmt)
        try:
            return pysubs2.load(path, encoding=encoding, format_=fmt)
        except Exception as exc:
            last_error = exc

    raise UnsupportedSubtitleError(
        f"Não foi possível interpretar {os.path.basename(path)} como "
        f"SSA/ASS: {last_error}")


def load_cues(path: str, encoding: str = "utf-8") -> list:
    """Carrega um arquivo SSA/ASS como lista de :class:`Cue`.

    Eventos marcados como comentário são ignorados, assim como os que ficam
    vazios depois de remover a marcação. Eventos com o mesmo intervalo de
    tempo são fundidos numa legenda só.
    """
    subs = _load_file(path, encoding)

    grouped = {}
    order = []
    for event in subs:
        if event.is_comment or event.start >= event.end:
            continue
        text = convert_text(event.text)
        if not text:
            continue
        key = (event.start, event.end)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(text)

    order.sort()
    cues = []
    for position, key in enumerate(order, start=1):
        start, end = key
        cues.append(Cue.from_source(
            index=position, start=start, end=end,
            source_text="\n".join(grouped[key])))
    return cues


def looks_like_ssa(text: str) -> bool:
    """Reconhece conteúdo SSA/ASS, mesmo com extensão errada."""
    head = text[:4000].lower()
    return "[script info]" in head or "[events]" in head
