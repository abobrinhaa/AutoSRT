"""Limpeza do texto que vai para o arquivo de legenda.

Última barreira antes de gravar. O que chega aqui já passou pelo filtro de
alucinação (:mod:`autosrt.hallucination`) e pela tradução, e mesmo assim
aparecia lixo que o tocador mostra ao pé da letra, no meio da fala:

- **Quebra de linha escrita como texto.** O modelo de tradução às vezes
  devolve a quebra como ``<br>``, ``</br>``, ``\\n`` literal ou ``</n>`` --
  este último é o eco do ``<N>texto</N>`` que o prompt pede.
- **Tag que o formato não conhece.** O SRT só entende ``<i>``, ``<b>``,
  ``<u>`` e ``<font>``. O resto (``<12>``, ``<span>``, um ``</B>`` sem
  abertura) é sobra da numeração dos blocos ou de HTML, e sai escrito na
  tela.
- **Chinês (e japonês, coreano) no meio do português.** Modelos treinados
  com muito chinês (DeepSeek, Qwen) escorregam de idioma no meio da frase,
  e o Whisper tem clichê de encerramento em chinês também. Numa legenda
  que não é num desses idiomas, isso nunca é fala.

Legenda que fica vazia depois da limpeza era só lixo e sai do arquivo:
bloco sem texto quebra a leitura do .srt em alguns tocadores.
"""

import html
import logging
import re

from .textfmt import MAX_LINES, is_dialogue, rewrap, strip_invisible

logger = logging.getLogger(__name__)

#: Idiomas escritos com ideogramas ou silabários do leste asiático -- o
#: único caso em que esses caracteres são texto de verdade e ficam.
IDIOMAS_CJK = frozenset({"zh", "ja", "ko", "yue"})

# Ideogramas (unificados, extensões e compatibilidade), radicais, bopomofo,
# hiragana, katakana, hangul, a pontuação própria dessas escritas e as
# formas de largura cheia.
CJK_RE = re.compile(
    "["
    "ᄀ-ᇿ"
    "⺀-⿟"
    "　-〿"
    "぀-ヿ"
    "㄀-ㄯ"
    "㄰-㆏"
    "ㇰ-ㇿ"
    "㐀-䶿"
    "一-鿿"
    "가-힯"
    "豈-﫿"
    "＀-￯"
    "\U00020000-\U0003134f"
    "]")

# Pontuação de largura cheia que tem equivalente direto: vira o sinal comum
# em vez de sumir, para "Não！" sair "Não!", e não "Não".
_PONTUACAO_CJK = str.maketrans({
    "，": ",", "。": ".", "！": "!", "？": "?", "：": ":", "；": ";",
    "（": "(", "）": ")", "、": ",",
})

# Quebra de linha escrita como texto. O "\r\n" literal vem antes do "\n"
# na alternância, senão sobraria um "\r" solto.
_QUEBRA_RE = re.compile(r"\\r\\n|\\[nN]|<\s*/?\s*(?:br|n)\s*/?\s*>",
                        re.IGNORECASE)

# Só conta como tag o que abre com letra ou número colado no "<" -- "a < b
# > c" é texto. "Eu <3 você" também, por não fechar com ">".
_TAG_RE = re.compile(r"</?([A-Za-z][A-Za-z0-9]*|\d+)\b[^<>]*>")
_TAGS_DO_SRT = frozenset({"i", "b", "u", "font"})

# Chaves do SSA: só o posicionamento (``{\an8}``) tem efeito num .srt.
_CHAVES_RE = re.compile(r"\{[^{}]*\}")
_POSICIONAMENTO_RE = re.compile(r"\{\\an[1-9]\}")

_CONTROLE_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�]")
_CERCA_DE_CODIGO_RE = re.compile(r"^\s*```[\w-]*\s*$")
_ESPACOS_RE = re.compile(r"[ \t ]+")


def tem_cjk(texto: str) -> bool:
    """Diz se o texto tem algum caractere das escritas do leste asiático."""
    return bool(CJK_RE.search(texto or ""))


def predominio_cjk(texto: str) -> bool:
    """Diz se a maioria das letras do texto é ideograma/silabário CJK.

    Pontuação e espaço não contam: "字幕by索兰娅" tem cinco ideogramas
    contra duas letras latinas e é CJK; "Morei em 北京." não é.
    """
    letras = [c for c in texto or "" if c.isalpha()]
    if not letras:
        return False
    cjk = sum(1 for c in letras if CJK_RE.match(c))
    return cjk * 2 > len(letras)


def idioma_cjk(codigo: str) -> bool:
    """Diz se o código de idioma (``zh``, ``zh-CN``, ``ja``...) é CJK."""
    base = re.split(r"[-_]", (codigo or "").strip().lower())[0]
    return base in IDIOMAS_CJK


def _tags_permitidas(texto: str) -> str:
    """Mantém só as tags que o SRT entende, e só quando têm par."""
    abertas, fechadas = {}, {}
    for match in _TAG_RE.finditer(texto):
        nome = match.group(1).lower()
        if nome in _TAGS_DO_SRT:
            contagem = fechadas if match.group(0).startswith("</") else abertas
            contagem[nome] = contagem.get(nome, 0) + 1

    def trocar(match):
        nome = match.group(1).lower()
        if nome in _TAGS_DO_SRT and abertas.get(nome) == fechadas.get(nome):
            return match.group(0)
        # Espaço, e não nada: "Olá<span>mundo" não pode virar "Olámundo".
        return " "

    return _TAG_RE.sub(trocar, texto)


def _limpar_linha_cjk(linha: str) -> str:
    """Tira o CJK de uma linha em português (ou outra escrita latina).

    Linha dominada por ideogramas sai inteira: sobrar só o "by" de
    "字幕by索兰娅" seria trocar um lixo por outro.
    """
    if not tem_cjk(linha):
        return linha
    if predominio_cjk(linha):
        return ""
    limpa = CJK_RE.sub(" ", linha.translate(_PONTUACAO_CJK))
    # O que sobrou sem letra nem número é só pontuação órfã do trecho que
    # saiu ("你好！" -> "!"), não fala.
    if not any(c.isalnum() for c in limpa):
        return ""
    # E o espaço que o trecho deixou antes da pontuação também sai: "o que
    # 发生了." não pode virar "o que ." no fim da frase.
    return re.sub(r"\s+([,.!?;:])", r"\1", limpa)


def limpar_texto(texto: str, *, remover_cjk: bool = True) -> str:
    """Devolve o texto de uma legenda sem o lixo que o tocador mostraria.

    Args:
        texto: texto de uma legenda, com as quebras de linha dela.
        remover_cjk: tira ideogramas e silabários do leste asiático. Deve
            ser ``False`` só quando a legenda está num desses idiomas.

    Returns:
        O texto limpo, ou ``""`` se não sobrou nada que seja fala.

    >>> limpar_texto("Olá</br>mundo</n>")
    'Olá\\nmundo'
    """
    texto = html.unescape(strip_invisible(texto or ""))
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = _CONTROLE_RE.sub("", texto)
    texto = _QUEBRA_RE.sub("\n", texto)
    texto = _tags_permitidas(texto)
    texto = _CHAVES_RE.sub(
        lambda m: m.group(0) if _POSICIONAMENTO_RE.fullmatch(m.group(0)) else "",
        texto)

    linhas = []
    for linha in texto.split("\n"):
        if _CERCA_DE_CODIGO_RE.match(linha):
            continue
        if remover_cjk:
            linha = _limpar_linha_cjk(linha)
        linha = _ESPACOS_RE.sub(" ", linha).strip()
        if linha:
            linhas.append(linha)

    if len(linhas) > MAX_LINES and not is_dialogue("\n".join(linhas)):
        return rewrap(" ".join(linhas))
    return "\n".join(linhas)


def limpar_legendas(cues, *, remover_cjk: bool = True) -> list:
    """Limpa o texto de trabalho (``cue.text``) de cada legenda.

    ``source_text`` não é tocado: é o registro do que foi transcrito ou
    lido do arquivo, e é dele que a tradução parte.

    Returns:
        Nova lista, sem as legendas que eram só lixo, renumeradas a partir
        de 1. Se a limpeza esvaziaria o arquivo inteiro, devolve as
        legendas como vieram: arquivo vazio some sem aviso nenhum, e é pior
        do que um arquivo com lixo, que pelo menos se vê.
    """
    if not cues:
        return list(cues)

    limpos = [limpar_texto(cue.text, remover_cjk=remover_cjk) for cue in cues]
    if not any(limpos):
        logger.warning(
            "a limpeza esvaziaria todas as %d legendas do arquivo; mantendo "
            "o texto como veio", len(cues))
        return list(cues)

    mantidas = []
    for cue, limpo in zip(cues, limpos):
        if limpo:
            cue.text = limpo
            mantidas.append(cue)
        else:
            logger.info("legenda sem fala descartada na limpeza: %r",
                        cue.text.replace("\n", " "))

    if len(mantidas) != len(cues):
        for numero, cue in enumerate(mantidas, start=1):
            cue.index = numero
    return mantidas


__all__ = ["limpar_texto", "limpar_legendas", "tem_cjk", "predominio_cjk",
           "idioma_cjk", "IDIOMAS_CJK", "CJK_RE"]
