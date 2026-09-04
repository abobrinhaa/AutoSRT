"""Remoção das frases que o Whisper inventa onde não há fala.

O modelo foi treinado, entre outras coisas, com legenda automática de
YouTube -- e junto veio o clichê de encerramento desses vídeos. Sobre
silêncio, trilha sonora ou ruído, sem fala nenhuma para transcrever, o
Whisper preenche o trecho com o que mais viu nessa posição: "Obrigado por
assistir", "Até a próxima", "Inscreva-se no canal", "Legendas pela
comunidade Amara.org".

As defesas que :mod:`autosrt.transcribe` já liga por padrão
(``condition_on_previous_text=False``, ``hallucination_silence_threshold``)
atacam o problema antes: uma quebra a alucinação em cadeia, a outra manda
pular o silêncio longo em vez de transcrevê-lo. Nenhuma das duas pega o
caso que sobra -- a frase curta grudada no fim da última fala real, dentro
do mesmo trecho de áudio -- e nenhuma das duas existe no motor via API
(:mod:`autosrt.transcribe_api`), que não passa por parâmetro nenhum do
Whisper local. Daí este filtro, que age depois, sobre a legenda pronta, e
por isso vale para os dois motores.

O risco de um filtro assim é apagar fala de verdade, então ele é dividido
em dois níveis:

- **Inequívocas.** "Obrigado por assistir", "Amara.org", "Subscribe to my
  channel" -- ninguém diz isso num filme. Saem onde estiverem.
- **Ambíguas.** "Obrigado", "Até a próxima", "Tchau" -- personagem diz o
  tempo todo. Só saem com sinal corroborante: estar na borda do arquivo,
  estar isolada por silêncio longo, ou repetir idêntica.

Em qualquer dos dois níveis, só sai a legenda que é **inteiramente** o
clichê: a mesma frase dentro de uma fala maior ("Obrigado por assistir ao
filme comigo ontem") é fala de verdade e fica.
"""

import logging
import re
import unicodedata
from collections import Counter

logger = logging.getLogger(__name__)

#: Silêncio, em milissegundos, a partir do qual uma legenda é considerada
#: sozinha no áudio. Três segundos é bem acima da pausa natural entre duas
#: falas de uma conversa, e é justamente o tipo de vão onde o Whisper
#: inventa texto.
LIMIAR_ISOLAMENTO_MS = 3000

#: Quantas legendas de cada ponta contam como "borda do arquivo". Duas, e
#: não uma, porque a alucinação de encerramento costuma vir depois de um
#: último "Tchau" também alucinado.
LEGENDAS_DE_BORDA = 2

# Frases que não são fala de filme em lugar nenhum: clichê de canal de
# vídeo e crédito de legendagem. Escritas já normalizadas (minúsculas, sem
# acento e sem pontuação) -- é assim que o texto da legenda chega para a
# comparação, e um teste garante que nenhuma constante fuja disso.
FRASES_INEQUIVOCAS = frozenset({
    "obrigado por assistir",
    "obrigada por assistir",
    "muito obrigado por assistir",
    "muito obrigada por assistir",
    "obrigado por assistirem",
    "obrigado por assistir ao video",
    "obrigado por assistir o video",
    "obrigado por assistirem ao video",
    "obrigado por assistir ate o final",
    "obrigado por ter assistido",
    "se inscreva no canal",
    "inscreva se no canal",
    "nao se esqueca de se inscrever",
    "nao se esqueca de curtir e se inscrever",
    "deixe seu like",
    "deixe seu like e se inscreva",
    "curta e se inscreva",
    "ate o proximo video",
    "nos vemos no proximo video",
    "thanks for watching",
    "thanks for watching this video",
    "thank you for watching",
    "thank you for watching this video",
    "please subscribe to my channel",
    "dont forget to subscribe",
    "like and subscribe",
    "see you in the next video",
    "gracias por ver el video",
    "gracias por ver este video",
    "suscribete al canal",
    "no olvides suscribirte",
})

# Famílias inteiras, onde listar cada variante seria interminável. Casam
# por busca no texto normalizado da legenda inteira, nunca dentro de uma
# fala maior -- quem chama já garantiu que a legenda é só isto.
PADROES_INEQUIVOCOS = (
    # O crédito da comunidade Amara é o caso mais conhecido de todos, e
    # aparece em dezenas de formulações diferentes em cada idioma.
    re.compile(r"\bamara org\b"),
    re.compile(r"^legenda(s|do|da)? (por|pela|pelo) "),
    re.compile(r"^subtitles? (by|created by|provided by)\b"),
    re.compile(r"^subtitulos (por|realizados)\b"),
    re.compile(r"\binscreva[m]? se\b.*\bcanal\b"),
    re.compile(r"\bsubscribe to (my|our|the) channel\b"),
)

# Fala legítima de filme que o Whisper também usa como clichê de silêncio.
# Não basta reconhecer a frase para remover: veja :func:`_sinais`.
FRASES_AMBIGUAS = frozenset({
    "obrigado",
    "obrigada",
    "muito obrigado",
    "muito obrigada",
    "obrigado a todos",
    "valeu",
    "ate a proxima",
    "ate a proxima vez",
    "ate mais",
    "ate logo",
    "ate breve",
    "tchau",
    "tchau tchau",
    "thank you",
    "thank you very much",
    "thanks",
    "bye",
    "bye bye",
    "goodbye",
    "see you",
    "see you next time",
    "gracias",
    "muchas gracias",
    "adios",
    "hasta la proxima",
    "hasta luego",
})

INEQUIVOCA = "inequivoca"
AMBIGUA = "ambigua"

_TAGS_RE = re.compile(r"<[^>]+>|\{[^}]*\}")
# Apostrofo sai antes do resto: virando espaço, "don't" quebraria em duas
# palavras e nenhuma frase da lista casaria.
_APOSTROFOS_RE = re.compile(r"['’´`]")
_NAO_ALFANUMERICO_RE = re.compile(r"[^0-9a-z ]+")
_ESPACOS_RE = re.compile(r"\s+")


def normalizar(texto: str) -> str:
    """Reduz a legenda à forma que as listas usam para comparar.

    Tira tags, acento, caixa e pontuação, para que "Até a próxima!!!",
    "ate a proxima" e "<i>Até a Próxima</i>" sejam a mesma coisa.

    >>> normalizar("Até a próxima!!!")
    'ate a proxima'
    """
    texto = _TAGS_RE.sub(" ", texto or "")
    texto = _APOSTROFOS_RE.sub("", texto)
    decomposto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in decomposto if not unicodedata.combining(c))
    texto = _NAO_ALFANUMERICO_RE.sub(" ", texto.casefold())
    return _ESPACOS_RE.sub(" ", texto).strip()


def _classificar_texto(normalizado: str, inequivocas) -> str:
    """Diz se um texto já normalizado é clichê, e de qual nível."""
    if not normalizado:
        return None
    if normalizado in inequivocas:
        return INEQUIVOCA
    if any(padrao.search(normalizado) for padrao in PADROES_INEQUIVOCOS):
        return INEQUIVOCA
    if normalizado in FRASES_AMBIGUAS:
        return AMBIGUA
    return None


def classificar(texto: str, inequivocas=FRASES_INEQUIVOCAS) -> str:
    """Classifica uma legenda inteira: ``INEQUIVOCA``, ``AMBIGUA`` ou ``None``.

    Uma legenda de duas linhas em que cada linha é um clichê ("Obrigado por
    assistir!" / "Até a próxima!") também conta: é o formato em que a
    alucinação de encerramento costuma sair.
    """
    classe = _classificar_texto(normalizar(texto), inequivocas)
    if classe or "\n" not in (texto or ""):
        return classe

    linhas = [linha for linha in texto.split("\n") if linha.strip()]
    if len(linhas) < 2:
        return None

    classes = [_classificar_texto(normalizar(linha), inequivocas)
               for linha in linhas]
    if not all(classes):
        return None
    return INEQUIVOCA if INEQUIVOCA in classes else AMBIGUA


def _sinais(cues, posicao, contagem) -> list:
    """Por que esta legenda é suspeita de não ser fala de verdade.

    Devolve os motivos encontrados, em texto, para o log dizer o que levou
    à remoção -- lista vazia quer dizer "nada além da frase em si", e nesse
    caso uma frase ambígua fica.
    """
    motivos = []
    cue = cues[posicao]

    if posicao < LEGENDAS_DE_BORDA or posicao >= len(cues) - LEGENDAS_DE_BORDA:
        motivos.append("na borda do arquivo")

    fim_anterior = cues[posicao - 1].end if posicao else 0
    if cue.start - fim_anterior >= LIMIAR_ISOLAMENTO_MS:
        motivos.append("silêncio longo antes")

    if (posicao + 1 < len(cues)
            and cues[posicao + 1].start - cue.end >= LIMIAR_ISOLAMENTO_MS):
        motivos.append("silêncio longo depois")

    repeticoes = contagem[normalizar(cue.source_text)]
    if repeticoes >= 2:
        motivos.append(f"repetida {repeticoes}x no arquivo")

    return motivos


def filtrar_alucinacoes(cues, *, extras=None) -> list:
    """Devolve as legendas sem as que o Whisper inventou.

    Args:
        cues: lista de :class:`~autosrt.cue.Cue` recém-transcritas. A
            lista recebida não é alterada; a devolvida é outra, com as
            mesmas legendas que sobraram -- essas, sim, renumeradas a
            partir de 1, porque o número da legenda é a posição dela no
            arquivo e quem sai deixa um buraco.
        extras: frases do usuário, tratadas como inequívocas. Cada acervo
            tem o seu clichê, e a lista embutida não cobre todos. Entradas
            vazias são ignoradas: uma linha em branco casaria com tudo.

    Returns:
        Nova lista de legendas. Se o filtro pegaria *todas* elas, devolve o
        arquivo inteiro como veio: legenda vazia é pior do que legenda com
        uma frase a mais, e não há como um arquivo ser só alucinação sem que
        o problema seja outro.
    """
    if not cues:
        return list(cues)

    inequivocas = set(FRASES_INEQUIVOCAS)
    for frase in extras or ():
        normalizada = normalizar(frase)
        if normalizada:
            inequivocas.add(normalizada)

    contagem = Counter(normalizar(cue.source_text) for cue in cues)

    mantidas = []
    for posicao, cue in enumerate(cues):
        classe = classificar(cue.source_text, inequivocas)
        if classe is None:
            mantidas.append(cue)
            continue

        if classe == INEQUIVOCA:
            motivo = "frase que não é fala de filme"
        else:
            sinais = _sinais(cues, posicao, contagem)
            if not sinais:
                # Frase ambígua cercada de diálogo é diálogo.
                mantidas.append(cue)
                continue
            motivo = ", ".join(sinais)

        logger.info("alucinação removida (%s): %r", motivo,
                    cue.source_text.replace("\n", " "))

    if not mantidas:
        logger.warning(
            "o filtro pegaria todas as %d legendas do arquivo; mantendo a "
            "transcrição como veio", len(cues))
        return list(cues)

    for numero, cue in enumerate(mantidas, start=1):
        cue.index = numero
    return mantidas


__all__ = ["filtrar_alucinacoes", "classificar", "normalizar",
           "FRASES_INEQUIVOCAS", "FRASES_AMBIGUAS", "INEQUIVOCA", "AMBIGUA"]
