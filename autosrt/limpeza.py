"""Limpeza da legenda: o que atrapalha a leitura e não é fala.

Roda como um passe sobre as legendas já prontas -- transcritas ou vindas de
um arquivo existente -- e conserta o que nenhuma etapa anterior tem como
consertar sozinha:

- **Descrição de som para surdos (SDH).** "(TIROS)", "[música tensa]",
  "{RISOS}" descrevem o áudio para quem não ouve. Quem ouve não quer ler
  isso, e o tradutor ainda gasta requisição traduzindo barulho.
- **Tags de itálico quebradas.** ``<i>`` sem fechar contamina o resto do
  arquivo em muitos tocadores: tudo dali para frente aparece em itálico.
  Fechamento órfão e par vazio são o mesmo tipo de sujeira.
- **Espaço inválido.** Espaço antes de vírgula, espaço duplo, espaço
  inquebrável no meio da frase, sobra no fim da linha.
- **Quebra de linha desnecessária.** Uma frase curta partida em duas linhas
  é uma linha só que alguém quebrou sem motivo.
- **Linha longa demais.** O contrário: texto que passa da largura de leitura
  e precisa ser dividido em duas linhas equilibradas.

O passe é dividido em dois momentos, porque metade da limpeza depende do
resultado da tradução e a outra metade não:

- :func:`preparar_cues` roda antes de traduzir -- SDH, tag quebrada e
  espaço. Tirar o SDH aqui é o que evita gastar requisição traduzindo
  "(TIROS)", e é o que permite descobrir que uma legenda ficou vazia
  antes dela custar uma chamada ao tradutor.
- :func:`finalizar_cues` roda depois -- junta linha curta partida à toa,
  quebra linha que passou da largura. Antes da tradução não dá para
  decidir isso: o texto em português costuma sair mais comprido que o
  original em inglês.
"""

import logging
import re

from .textfmt import MAX_CHARS_PER_LINE, rewrap, strip_invisible

logger = logging.getLogger(__name__)

#: Descrição de som entre parênteses, colchetes ou chaves. Não exige que
#: ocupe a legenda inteira: "(SUSPIRA) Não posso" é comum, e o que se quer
#: tirar é só o pedaço descritivo.
SDH_RE = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")

#: Rótulo de locutor em maiúsculas no começo da linha ("JOHN:", "MULHER 2:"),
#: também convenção de SDH. Exige maiúsculas para não comer "Espera:" nem
#: um horário ("14:30").
ROTULO_SDH_RE = re.compile(r"^\s*[A-ZÀ-Þ][A-ZÀ-Þ0-9 .'#-]{1,24}:\s*")

#: Espaços que não são o espaço comum -- inquebrável, fino, ideográfico.
ESPACOS_ESTRANHOS = "     　"

#: Itálico é a única tag que legenda de filme usa de verdade. Negrito e
#: sublinhado aparecem, e são tratados do mesmo jeito.
TAGS_CONHECIDAS = ("i", "b", "u")

#: Até este tamanho, uma legenda de uma frase só não precisa de duas linhas.
#: É a largura de uma linha: se cabe numa, não há motivo para partir.
LIMITE_LINHA_UNICA = MAX_CHARS_PER_LINE

#: Travessão que abre turno de fala. Legenda de diálogo mantém as linhas
#: como estão -- juntá-las apagaria quem fala o quê.
TRAVESSAO_RE = re.compile(r"^\s*[-–—]\s*\S")

#: Mais de uma frase na legenda: aí a quebra de linha pode ser proposital.
FIM_DE_FRASE_RE = re.compile(r"[.!?…][\"'”’)\]]?\s")


def limpar_espacos(texto: str) -> str:
    """Normaliza os espaços, linha a linha.

    Não mexe nas quebras de linha: quem decide sobre elas é
    :func:`juntar_linhas_curtas` e :func:`quebrar_linhas_longas`.
    """
    for espaco in ESPACOS_ESTRANHOS:
        texto = texto.replace(espaco, " ")
    linhas = []
    for linha in texto.split("\n"):
        linha = re.sub(r"[ \t]+", " ", linha)
        # Espaço antes de pontuação é erro de digitação ou sobra de uma
        # remoção anterior (foi o SDH que saiu dali).
        linha = re.sub(r" +([,.;:!?…])", r"\1", linha)
        linha = re.sub(r"([¿¡\(\[]) +", r"\1", linha)
        linhas.append(linha.strip())
    return "\n".join(linhas)


def remover_sdh(texto: str) -> str:
    """Tira as descrições de som e os rótulos de locutor em maiúsculas.

    >>> remover_sdh("(TIROS) Corre!")
    'Corre!'
    """
    texto = SDH_RE.sub(" ", texto)
    linhas = [ROTULO_SDH_RE.sub("", linha) for linha in texto.split("\n")]
    return "\n".join(linhas)


def consertar_tags(texto: str) -> str:
    """Fecha, remove ou equilibra as tags de estilo da legenda.

    Uma abertura sem fechamento é fechada no fim do texto; um fechamento
    sem abertura é removido; um par que não envolve nada some. É o mínimo
    para que uma legenda quebrada não derrube a formatação das seguintes.
    """
    for nome in TAGS_CONHECIDAS:
        abre = len(re.findall(rf"<{nome}>", texto, re.IGNORECASE))
        fecha = len(re.findall(rf"</{nome}>", texto, re.IGNORECASE))

        if fecha > abre:
            # Remove os fechamentos sobrando, do fim para o começo: o último
            # é o mais provável de ser o intruso.
            for _ in range(fecha - abre):
                posicao = texto.lower().rfind(f"</{nome}>")
                if posicao < 0:
                    break
                texto = texto[:posicao] + texto[posicao + len(nome) + 3:]
        elif abre > fecha:
            texto += f"</{nome}>" * (abre - fecha)

        # Par vazio, inclusive com espaço no meio: não estiliza nada.
        texto = re.sub(rf"<{nome}>\s*</{nome}>", "", texto, flags=re.IGNORECASE)

    return texto


def _e_dialogo(texto: str) -> bool:
    linhas = [ln for ln in texto.split("\n") if ln.strip()]
    return sum(1 for ln in linhas if TRAVESSAO_RE.match(ln)) >= 2


def _sem_tags(texto: str) -> str:
    return re.sub(r"</?[A-Za-z][^>]*>", "", texto)


def juntar_linhas_curtas(texto: str) -> str:
    """Junta as linhas quando são uma frase curta partida à toa.

    Só junta o que cabe numa linha e forma uma frase só: duas frases em
    duas linhas é uma escolha de quem legendou, e diálogo com travessão
    marca quem fala -- os dois ficam como estão.
    """
    if "\n" not in texto or _e_dialogo(texto):
        return texto

    unido = re.sub(r"\s*\n\s*", " ", texto).strip()
    visivel = _sem_tags(unido)
    if len(visivel) > LIMITE_LINHA_UNICA:
        return texto
    if FIM_DE_FRASE_RE.search(visivel):
        return texto
    return unido


def quebrar_linhas_longas(texto: str, largura: int = MAX_CHARS_PER_LINE) -> str:
    """Divide em duas linhas equilibradas o que passou da largura de leitura.

    Diálogo fica intacto, e cada linha é medida sem as tags: ``<i>`` não
    ocupa espaço na tela.
    """
    if _e_dialogo(texto):
        return texto
    if all(len(_sem_tags(linha)) <= largura for linha in texto.split("\n")):
        return texto
    return rewrap(texto.replace("\n", " "), largura)


def preparar_texto(texto: str, *, remover_descricoes=True) -> str:
    """Limpa uma legenda antes de ela ir ao tradutor.

    Só o que não depende do resultado da tradução: descrição de som,
    espaço, tag quebrada. Juntar ou quebrar linha fica para depois --
    antes de traduzir ainda não se sabe o tamanho que o texto vai ter.

    Devolve ``""`` se não sobrar fala nenhuma (a legenda inteira era
    descrição de som).
    """
    texto = strip_invisible(texto)
    if remover_descricoes:
        texto = remover_sdh(texto)
    texto = limpar_espacos(texto)
    texto = consertar_tags(texto)
    if not _sem_tags(texto).strip():
        return ""
    return texto.strip()


def preparar_cues(cues, *, remover_descricoes=True) -> tuple:
    """Aplica :func:`preparar_texto` na lista inteira, antes da tradução.

    Descarta as legendas que eram só descrição de som -- não sobra nada
    para o tradutor traduzir, e a legenda não pode ficar em branco na
    tela. ``text`` acompanha ``source_text``: neste ponto do fluxo os
    dois ainda são o mesmo texto, a tradução ainda não rodou.

    Returns:
        ``(cues_preparados, removidas)`` -- a lista nova e quantas saíram.
    """
    preparados = []
    removidas = 0
    for cue in cues:
        origem = preparar_texto(cue.source_text or "",
                                remover_descricoes=remover_descricoes)
        if not origem:
            removidas += 1
            continue
        cue.source_text = origem
        cue.text = origem
        preparados.append(cue)

    if removidas:
        logger.info("limpeza: %d legenda(s) eram só descrição de som", removidas)
    return preparados, removidas


def finalizar_texto(texto: str) -> str:
    """Ajusta uma legenda depois da tradução, pronta para a tela.

    Conserta tag que a tradução possa ter quebrado, normaliza espaço de
    novo (o modelo é livre para devolver do jeito que quiser) e só então
    decide sobre as linhas: junta o que é uma frase curta partida à toa,
    quebra o que passou da largura de leitura.
    """
    texto = limpar_espacos(texto)
    texto = consertar_tags(texto)
    texto = juntar_linhas_curtas(texto)
    texto = quebrar_linhas_longas(texto)
    return texto.strip()


def finalizar_cues(cues) -> None:
    """Aplica :func:`finalizar_texto` em ``cue.text`` da lista inteira.

    Roda depois da tradução (ou no lugar dela, para quem só transcreve),
    logo antes de gravar o arquivo. Modifica no lugar; não remove
    legenda nenhuma -- a essa altura, uma legenda vazia seria falha da
    tradução, não sobra de limpeza, e é o passe de tradução que já trata
    isso mantendo o texto de origem.
    """
    for cue in cues:
        if cue.text:
            cue.text = finalizar_texto(cue.text)
