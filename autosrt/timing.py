"""Tempo de exibição de cada legenda.

O Whisper fecha a legenda no instante em que a fala termina. Para quem
ouve está certo; para quem lê, não: "Sim." dura 300 ms, e uma fala rápida
de duas linhas some antes de alguém terminar de ler. A tradução piora a
conta -- o português costuma sair mais longo que o inglês, no mesmo tempo
de tela.

O ajuste segue a prática de legendagem (a mesma dos guias de estilo de
streaming e do "corrigir tempo de exibição curto" do Subtitle Edit):

- **Duração mínima.** Nenhuma legenda fica menos que isto na tela.
- **Velocidade de leitura.** Em caracteres por segundo: legenda com mais
  texto fica mais tempo.

Com três limites, para não trocar um defeito por outro:

- **Nunca invade a legenda seguinte.** Estende só dentro do silêncio que
  já existe, deixando um respiro para a troca ser percebida.
- **Nunca encurta.** Legenda que já dura o bastante fica como está.
- **Tem teto.** Além dele a legenda fica "pendurada" depois de a cena já
  ter mudado.
"""

import re

#: Padrões, já ligados. 1,5 s dá para ler um "Sim." sem prender na tela
#: uma legenda que já passou; 15 caracteres por segundo fica um pouco
#: abaixo dos 17 que os guias de streaming usam para adulto em português,
#: porque a queixa que motivou isto foi justamente legenda sumindo rápido.
DURACAO_MINIMA_PADRAO = 1.5
CARACTERES_POR_SEGUNDO_PADRAO = 15

#: Até onde o ajuste estende uma legenda. Legenda que já vem mais longa
#: não é encurtada.
DURACAO_MAXIMA_MS = 7000

#: Respiro entre o fim de uma legenda e o começo da próxima (dois quadros a
#: 24 fps). Sem ele, duas legendas coladas parecem uma só mudando de texto.
INTERVALO_MINIMO_MS = 84

_TAGS_RE = re.compile(r"<[^>]+>|\{[^}]*\}")


def duracao_de_leitura_ms(texto: str, caracteres_por_segundo) -> int:
    """Tempo para ler ``texto`` na velocidade dada.

    Conta o que aparece na tela: tag não conta, quebra de linha vale um
    espaço. Velocidade zero (ou ausente) quer dizer "não medir".
    """
    if not caracteres_por_segundo or caracteres_por_segundo <= 0:
        return 0
    visivel = " ".join(_TAGS_RE.sub("", texto or "").split())
    return round(len(visivel) * 1000 / caracteres_por_segundo)


def ajustar_exibicao(cues, *, duracao_minima=None,
                     caracteres_por_segundo=None) -> int:
    """Estende o fim das legendas curtas demais para serem lidas.

    Mede o texto de trabalho (``cue.text``), que depois da tradução é o que
    vai para a tela. As legendas precisam estar em ordem de início, como
    saem da transcrição.

    Args:
        cues: lista de :class:`~autosrt.cue.Cue`, modificada no lugar.
        duracao_minima: segundos. ``None`` usa :data:`DURACAO_MINIMA_PADRAO`;
            ``0`` desliga este critério.
        caracteres_por_segundo: velocidade de leitura. ``None`` usa
            :data:`CARACTERES_POR_SEGUNDO_PADRAO`; ``0`` desliga este
            critério.

    Returns:
        Quantas legendas foram estendidas.
    """
    if duracao_minima is None:
        duracao_minima = DURACAO_MINIMA_PADRAO
    if caracteres_por_segundo is None:
        caracteres_por_segundo = CARACTERES_POR_SEGUNDO_PADRAO
    minima_ms = round(max(0, duracao_minima) * 1000)

    estendidas = 0
    for posicao, cue in enumerate(cues):
        desejada = max(minima_ms,
                       duracao_de_leitura_ms(cue.text, caracteres_por_segundo))
        desejada = min(desejada, DURACAO_MAXIMA_MS)
        if desejada <= cue.duration:
            continue

        fim = cue.start + desejada
        if posicao + 1 < len(cues):
            fim = min(fim, cues[posicao + 1].start - INTERVALO_MINIMO_MS)
        if fim > cue.end:
            cue.end = fim
            estendidas += 1
    return estendidas


__all__ = ["ajustar_exibicao", "duracao_de_leitura_ms",
           "DURACAO_MINIMA_PADRAO", "CARACTERES_POR_SEGUNDO_PADRAO",
           "DURACAO_MAXIMA_MS", "INTERVALO_MINIMO_MS"]
