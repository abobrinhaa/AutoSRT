"""Dublagem automática de um vídeo inteiro, com a voz clonada de cada locutor.

Junta num processo só o que foi validado à mão, etapa por etapa, nos pilotos
(ver ``PIPELINE.md`` na raiz do repositório):

1. separa a faixa de áudio no idioma original (o filme pode ter várias);
2. transcreve com marcação de locutor (o mesmo Whisper das legendas);
3. traduz com o mesmo tradutor das legendas (Ollama), juntando antes as
   frases que o Whisper partiu em duas -- traduzidas separadas, cada metade
   virava a frase inteira (bug achado no piloto do "Cannon");
4. separa voz e fundo com o Demucs, em pedaços de 10 minutos (o filme
   inteiro de uma vez estoura a RAM do CT104, OOM confirmado);
5. monta a voz de referência de cada locutor com os seus melhores 30 s (o
   XTTS só usa os primeiros 30 s da referência, ``max_ref_len``);
6. sintetiza cada fala com o XTTS-v2 (temperatura 0,65, sem ponto final no
   texto, sem divisão em frases -- parâmetros validados ouvindo);
7. acelera a fala até 1,6x quando ela não cabe no tempo original;
8. quando nem assim cabe, desacelera o vídeo daquele trecho (até 4x) em vez
   de cortar a fala -- o filme sai mais longo que o original, troca aceita;
9. monta a trilha amostra por amostra e o vídeo quadro a quadro, num
   encoder só: sem emendar arquivos de vídeo, que foi o que deixou
   timestamps quebrados e deu "erro fatal do player" no Jellyfin.

Roda no venv da dublagem (torch, TTS, Demucs), separado do serviço web, por
dois motivos: o serviço web não carrega nada disso, e a memória da placa de
vídeo volta inteira quando o processo termina.

A placa de vídeo é dividida com as legendas (portas 8000/8001), que têm
prioridade: antes de cada etapa que usa a placa, e de tempos em tempos no
meio da síntese, este processo confere as filas de legenda e só segue com
elas vazias -- se uma legenda começar no meio, ele descarrega o XTTS, espera
e continua depois. Legenda que já roda nunca disputa memória com a dublagem.

Conversa com quem chamou (a fila de dublagem de :mod:`autosrt.web`) por
stdout, uma linha JSON por evento::

    {"etapa": "Sintetizando falas... 120 de 1480", "progresso": 41}
    {"fim": {...relatório...}}
    {"erro": "mensagem"}

O resto (logs do Whisper, Demucs, ffmpeg) sai no stderr.

Cada etapa grava o que produziu na pasta de trabalho do arquivo e pula o que
já está lá: se o serviço cair no meio de um filme de horas, mandar dublar de
novo continua de onde parou.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction

from . import config

TAXA = 48000                 # Hz da trilha montada
TAXA_XTTS = 24000            # Hz que o XTTS-v2 devolve
BUFFER_BRECHA_S = 0.25       # "colchão" de fundo no começo de cada brecha (PIPELINE 7c)
REF_ALVO_S = 30.0            # o XTTS usa só os primeiros 30 s da referência
REF_MINIMA_S = 2.0           # abaixo disso a voz do locutor não serve de referência
RAZAO_SUSPEITA = 2.2         # fala sintetizada 2,2x mais longa que a original: repetir
RAZAO_ACEITA = 2.0
TENTATIVAS_SINTESE = 4
TEMPERATURA_REPETICAO = 0.6
MAX_CARACTERES_XTTS = 200    # o XTTS avisa (e corta) acima de ~203 caracteres em pt
DEMUCS_PEDACO_S = 600.0
DEMUCS_CONTEXTO_S = 2.0
CHECAR_FILAS_A_CADA = 15     # falas sintetizadas entre uma conferida e outra
TRADUZIR_EM_BLOCOS_DE = 60   # falas por rodada de tradução (confere as filas entre rodadas)

# Memória livre que cada etapa precisa na placa antes de começar (MB).
VRAM_WHISPER = 5500
VRAM_DEMUCS = 3500
VRAM_XTTS = 4000

PASTA_SAIDA_PADRAO = "/mnt/filmes_storage/DUBLAGEM IA"
RAIZ_TRABALHO_PADRAO = "/opt/autosrt-dub/trabalho"
SIMILARIDADE_AUTOMATICA = 0.85  # voz do locutor x voz salva do ator, para reaproveitar sozinho

#: Faixas de progresso (de, até) de cada etapa, somando 100.
FAIXAS = {
    "preparar": (0, 2),
    "transcrever": (2, 14),
    "traduzir": (14, 24),
    "separar": (24, 31),
    "referencias": (31, 32),
    "sintetizar": (32, 66),
    "ajustar": (66, 68),
    "audio": (68, 70),
    "video": (70, 96),
    "verificar": (96, 100),
}

# Códigos de idioma de 2 letras (Whisper/config) e os de 3 que o mkv usa.
_IDIOMAS_3 = {
    "en": {"en", "eng"}, "es": {"es", "spa"}, "fr": {"fr", "fra", "fre"},
    "it": {"it", "ita"}, "de": {"de", "deu", "ger"}, "ja": {"ja", "jpn"},
    "pt": {"pt", "por"}, "ru": {"ru", "rus"}, "zh": {"zh", "zho", "chi"},
    "ko": {"ko", "kor"}, "sv": {"sv", "swe"}, "nl": {"nl", "nld", "dut"},
}

_TERMINA_FRASE = re.compile(r"""[.!?…:;"')\]»]\s*$""")
_SO_MUSICA = re.compile(r"^[\s♪♫#*~-]*$")


class ErroDublagem(Exception):
    """Falha com mensagem pronta para mostrar na tela."""


# --------------------------------------------------------------------------
# Comunicação com quem chamou
# --------------------------------------------------------------------------

_protocolo = sys.stdout


def _isolar_stdout():
    """Separa o canal JSON do resto: tudo que bibliotecas e processos filhos
    escreverem em stdout (o TTS imprime bastante) vai pro stderr, e só
    :func:`emitir` escreve no stdout original."""
    global _protocolo
    _protocolo = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr


def emitir(**dados):
    _protocolo.write(json.dumps(dados, ensure_ascii=False) + "\n")
    _protocolo.flush()


def log(mensagem):
    print(f"[dublagem {time.strftime('%H:%M:%S')}] {mensagem}", file=sys.stderr, flush=True)


class Progresso:
    """Traduz "metade da síntese" em "49% do trabalho todo"."""

    def __init__(self):
        self._ultimo_envio = 0.0
        self._ultima_etapa = None

    def __call__(self, etapa, fracao, texto, forcar=False):
        de, ate = FAIXAS[etapa]
        pct = int(de + (ate - de) * max(0.0, min(1.0, fracao)))
        agora = time.time()
        if forcar or texto != self._ultima_etapa or agora - self._ultimo_envio >= 2:
            emitir(etapa=texto, progresso=pct, fase=etapa)
            self._ultimo_envio = agora
            self._ultima_etapa = texto


# --------------------------------------------------------------------------
# Configuração (config.json da instância, mesmo do resto do autosrt)
# --------------------------------------------------------------------------

def _cfg_num(nome, padrao):
    valor = config.get_setting(nome)
    if valor in (None, ""):
        return padrao
    try:
        return float(str(valor).replace(",", ".").rstrip("xX "))
    except ValueError:
        return padrao


def _cfg_sim(nome, padrao=True):
    valor = config.get_setting(nome)
    if valor in (None, ""):
        return padrao
    return str(valor).strip().lower() in ("1", "sim", "true", "on", "yes")


def opcoes():
    """As opções da dublagem lidas do config.json (aba Voz/Qualidade)."""
    frases = config.get_setting("dublagem_frases_nao_dublar") or ""
    return {
        "temperatura": _cfg_num("dublagem_temperatura", 0.65),
        "max_acelerar": max(1.0, _cfg_num("dublagem_max_acelerar", 1.6)),
        "max_desacelerar_video": max(1.0, _cfg_num("dublagem_max_desacelerar_video", 4.0)),
        "limpar_referencia": _cfg_sim("dublagem_limpar_referencia", True),
        "reaproveitar_voz": _cfg_sim("dublagem_reaproveitar_voz", True),
        "preservar_trilha": _cfg_sim("dublagem_preservar_trilha", True),
        "clonar_por_locutor": _cfg_sim("dublagem_clonar_por_locutor", True),
        "ajustar_ritmo": _cfg_sim("dublagem_ajustar_ritmo", True),
        "idioma_origem": (config.get_setting("dublagem_idioma_origem") or "").strip().lower(),
        "frases_nao_dublar": [f.strip().lower() for f in frases.splitlines() if f.strip()],
        "pasta_saida": (config.get_setting("dublagem_pasta_saida") or PASTA_SAIDA_PADRAO).strip(),
        "crf": int(_cfg_num("dublagem_crf", 18)),
        "preset": (config.get_setting("dublagem_preset") or "medium").strip(),
    }


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def _rodar(cmd, **kwargs):
    log("$ " + " ".join(str(c) for c in cmd))
    resultado = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if resultado.returncode != 0:
        detalhe = (resultado.stderr or resultado.stdout or "").strip().splitlines()[-5:]
        raise ErroDublagem(f"{os.path.basename(str(cmd[0]))} falhou: " + " | ".join(detalhe))
    return resultado


def _ffprobe_json(caminho):
    saida = _rodar(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                    "-of", "json", caminho]).stdout
    return json.loads(saida)


def _duracao(caminho):
    try:
        return float(_rodar(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", caminho]).stdout.strip())
    except (ErroDublagem, ValueError):
        return 0.0


def _gravar_json(caminho, dados):
    temporario = caminho + ".tmp"
    with open(temporario, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=1)
    os.replace(temporario, caminho)


def _ler_json(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def _slug(texto, limite=60):
    texto = re.sub(r"[^\w\- ]+", "", texto, flags=re.UNICODE).strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto[:limite].strip() or "video"


def pasta_de_trabalho(entrada, raiz=None, inicio=None, fim=None):
    """Pasta onde ficam os intermediários de um arquivo. Mesmo arquivo (mesmo
    caminho, tamanho e data) cai sempre na mesma pasta -- é isso que deixa
    um trabalho interrompido continuar de onde parou."""
    raiz = raiz or os.environ.get("AUTOSRT_DUB_TRABALHO", RAIZ_TRABALHO_PADRAO)
    estado = os.stat(entrada)
    chave = f"{os.path.realpath(entrada)}|{estado.st_size}|{int(estado.st_mtime)}|{inicio}|{fim}"
    resumo = hashlib.sha1(chave.encode("utf-8")).hexdigest()[:10]
    nome = _slug(os.path.splitext(os.path.basename(entrada))[0])
    return os.path.join(raiz, f"{nome}-{resumo}")


def caminho_de_saida(entrada, pasta_saida, inicio=None, fim=None):
    """``<pasta>/<nome> (dublado IA)/<nome> (dublado IA).mkv``. Uma pasta por
    filme porque o Jellyfin usa o nome da pasta como título."""
    base = os.path.splitext(os.path.basename(entrada))[0]
    if inicio is not None or fim is not None:
        base += f" [trecho {int(inicio or 0)}-{int(fim) if fim else 'fim'}s]"
    nome = f"{base} (dublado IA)"
    return os.path.join(pasta_saida, nome, nome + ".mkv")


# --------------------------------------------------------------------------
# Placa de vídeo dividida com as legendas
# --------------------------------------------------------------------------

def _filas_prioritarias():
    texto = os.environ.get(
        "AUTOSRT_DUB_FILAS_PRIORITARIAS",
        "http://127.0.0.1:8000,http://127.0.0.1:8001,http://127.0.0.1:8002")
    return [u.strip().rstrip("/") for u in texto.split(",") if u.strip()]


def filas_ocupadas():
    """Legendas rodando ou esperando nas filas que têm prioridade.
    Serviço fora do ar não usa a placa, então conta como livre."""
    import requests
    ocupadas = []
    for url in _filas_prioritarias():
        try:
            trabalhos = requests.get(url + "/api/trabalhos", timeout=5).json()
        except Exception:
            continue
        for trabalho in trabalhos:
            if trabalho.get("estado") in ("rodando", "pendente"):
                porta = url.rsplit(":", 1)[-1]
                ocupadas.append(f"{porta}: {trabalho.get('nome')}")
                break
    return ocupadas


def vram_livre_mb():
    try:
        saida = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return int(saida[0])
    except Exception:
        return None


def _descarregar_ollama():
    from .llm import unload_local_model
    unload_local_model(config.get_setting("llm_base_url", "LLM_BASE_URL") or "",
                       config.get_setting("llm_model", "LLM_MODEL") or "")


def aguardar_vez(progresso, etapa, fracao, precisa_mb=None):
    """Espera as legendas desocuparem a placa (e, se pedido, sobrar memória).

    Com as filas vazias e mesmo assim sem memória, quem segura a placa é o
    Ollama parado esperando o próprio ``keep_alive`` -- ninguém está usando,
    então dá pra descarregar."""
    aviso_anterior = None
    while True:
        ocupadas = filas_ocupadas()
        if not ocupadas:
            if precisa_mb is None:
                return
            livre = vram_livre_mb()
            if livre is None or livre >= precisa_mb:
                return
            _descarregar_ollama()
            time.sleep(5)
            livre = vram_livre_mb()
            if livre is None or livre >= precisa_mb:
                return
            aviso = (f"Esperando memória na placa de vídeo ({livre} MB livres, "
                     f"preciso de {precisa_mb} MB)")
        else:
            aviso = ("Esperando a placa de vídeo — legenda tem prioridade ("
                     + "; ".join(ocupadas) + ")")
        if aviso != aviso_anterior:
            log(aviso)
            progresso(etapa, fracao, aviso, forcar=True)
            aviso_anterior = aviso
        time.sleep(30)


# --------------------------------------------------------------------------
# 0. Reconhecer o arquivo
# --------------------------------------------------------------------------

def _fps(video):
    for chave in ("avg_frame_rate", "r_frame_rate"):
        valor = video.get(chave) or "0/0"
        try:
            num, den = (int(x) for x in valor.split("/"))
            if num > 0 and den > 0 and 5 <= num / den <= 121:
                return Fraction(num, den)
        except ValueError:
            continue
    raise ErroDublagem("Não consegui descobrir os quadros por segundo do vídeo.")


def _escolher_audio(audios, idioma):
    def codigo(s):
        return ((s.get("tags") or {}).get("language") or "").lower()

    aceitos = _IDIOMAS_3.get(idioma[:2], {idioma}) if idioma else set()
    for stream in audios:
        if codigo(stream) in aceitos:
            return stream
    # Dublando para português, a faixa em português nunca é a original.
    for stream in audios:
        if codigo(stream) not in _IDIOMAS_3["pt"]:
            return stream
    for stream in audios:
        if (stream.get("disposition") or {}).get("default"):
            return stream
    return audios[0]


def sondar(entrada, idioma, inicio=None, fim=None):
    info = _ffprobe_json(entrada)
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    if not video:
        raise ErroDublagem("Só dá para dublar vídeo: este arquivo não tem imagem.")
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if not audios:
        raise ErroDublagem("O arquivo não tem nenhuma faixa de áudio.")
    audio = _escolher_audio(audios, idioma)

    fps = _fps(video)
    duracao = float((info.get("format") or {}).get("duration") or 0)
    inicio_formato = float((info.get("format") or {}).get("start_time") or 0)
    inicio_video = float(video.get("start_time") or inicio_formato)
    deslocamento_video = max(0.0, inicio_video - inicio_formato)
    if duracao <= 0:
        raise ErroDublagem("Não consegui descobrir a duração do vídeo.")

    inicio_q = int(round((inicio or 0) * fps))
    fim_s = min(duracao - deslocamento_video, fim) if fim else duracao - deslocamento_video
    fim_q = int(math.floor(fim_s * fps))
    if fim_q - inicio_q < fps:
        raise ErroDublagem("Trecho curto demais para dublar.")

    largura = int(video.get("width") or 0)
    altura = int(video.get("height") or 0)
    if not largura or not altura:
        raise ErroDublagem("Não consegui ler o tamanho do vídeo.")
    sar = video.get("sample_aspect_ratio") or "1:1"
    if sar in ("0:1", "N/A"):
        sar = "1:1"

    return {
        "fps_num": fps.numerator, "fps_den": fps.denominator,
        "largura": largura - largura % 2, "altura": altura - altura % 2,
        "sar": sar,
        "indice_video": int(video["index"]),
        "indice_audio": int(audio["index"]),
        "idioma_audio": ((audio.get("tags") or {}).get("language") or "").lower(),
        "t0": deslocamento_video + inicio_q / fps,
        "quadros": fim_q - inicio_q,
        "duracao_original": duracao,
    }


# --------------------------------------------------------------------------
# 1. Áudio original
# --------------------------------------------------------------------------

def extrair_audio(entrada, info, trabalho, progresso):
    destino = os.path.join(trabalho, "original.wav")
    if os.path.exists(destino):
        return destino
    fps = Fraction(info["fps_num"], info["fps_den"])
    progresso("preparar", 0.1, "Separando a faixa de áudio original...", forcar=True)
    _rodar(["ffmpeg", "-y", "-v", "error", "-ss", f"{info['t0']:.6f}", "-i", entrada,
            "-t", f"{float(info['quadros'] / fps):.6f}",
            "-map", f"0:{info['indice_audio']}", "-ac", "2", "-ar", str(TAXA),
            "-c:a", "pcm_s16le", destino + ".parcial.wav"])
    os.replace(destino + ".parcial.wav", destino)
    return destino


# --------------------------------------------------------------------------
# 2. Transcrição com locutor
# --------------------------------------------------------------------------

def transcrever(original, idioma, trabalho, progresso):
    destino = os.path.join(trabalho, "transcricao.json")
    if os.path.exists(destino):
        return _ler_json(destino)

    from . import hallucination, pipeline, sanitize
    from . import transcribe as tr

    aguardar_vez(progresso, "transcrever", 0.0, VRAM_WHISPER)
    progresso("transcrever", 0.0, "Transcrevendo e identificando quem fala...", forcar=True)
    _descarregar_ollama()

    pasta = os.path.join(trabalho, "whisper")
    os.makedirs(pasta, exist_ok=True)
    kwargs = {
        "language": idioma or None,
        "diarize": tr.DEFAULT_DIARIZE_MODEL,
        "output_dir": pasta,
        "progress": lambda pct: progresso(
            "transcrever", pct / 100, f"Transcrevendo e identificando quem fala... {int(pct)}%"),
    }
    opcionais = {
        "model": config.get_whisper_model(),
        "compute_type": config.get_whisper_compute_type(),
        "vad": config.get_vad_method(),
        "vad_threshold": config.get_vad_threshold(),
        "vad_min_silence_ms": config.get_vad_min_silence_ms(),
        "condition_on_previous_text": config.get_condition_on_previous_text(),
        "hallucination_silence_threshold": config.get_hallucination_silence_threshold(),
        "standard": config.get_subtitle_standard(),
    }
    kwargs.update({k: v for k, v in opcionais.items() if v is not None})

    avisos = []
    with pipeline._audio_preparado(original, config.get_normalize_audio(),
                                   announce=lambda m: log(m),
                                   on_aviso=avisos.append) as entrada_whisper:
        if entrada_whisper != original:
            kwargs["output_dir"] = os.path.dirname(entrada_whisper)
        cues = tr.transcribe(entrada_whisper, **kwargs)

    if not cues:
        raise ErroDublagem("O Whisper não encontrou fala nenhuma no áudio.")
    if config.get_filtrar_alucinacoes():
        cues = hallucination.filtrar_alucinacoes(cues, extras=config.get_alucinacoes_extra())
    cues = sanitize.limpar_legendas(cues, remover_cjk=not sanitize.idioma_cjk(idioma or ""))

    dados = [{"index": c.index, "start": int(c.start), "end": int(c.end),
              "speaker": c.speaker, "source_text": c.source_text} for c in cues]
    _gravar_json(destino, dados)
    for aviso in avisos:
        log("aviso da transcrição: " + str(aviso))
    return dados


# --------------------------------------------------------------------------
# 3. Tradução
# --------------------------------------------------------------------------

def juntar_frases_partidas(falas, max_s=12.0, max_caracteres=180, max_brecha_ms=300):
    """Junta falas seguidas do mesmo locutor quando a primeira não termina a
    frase. Traduzidas separadas, cada metade virava a frase inteira na
    tradução (fala 9 do piloto, 6,8x mais longa que o espaço dela)."""
    saida = []
    for fala in falas:
        texto = " ".join(fala["source_text"].split())
        if saida:
            anterior = saida[-1]
            if (anterior["speaker"] == fala["speaker"]
                    and fala["start"] - anterior["end"] <= max_brecha_ms
                    and not _TERMINA_FRASE.search(anterior["source_text"])
                    and fala["end"] - anterior["start"] <= max_s * 1000
                    and len(anterior["source_text"]) + len(texto) + 1 <= max_caracteres):
                anterior["end"] = fala["end"]
                anterior["source_text"] = anterior["source_text"] + " " + texto
                continue
        saida.append(dict(fala, source_text=texto))
    for i, fala in enumerate(saida, 1):
        fala["index"] = i
    return saida


def _limpar_traducao(texto):
    texto = " ".join((texto or "").replace("\n", " ").split())
    return re.sub(r"^[-–—]\s*", "", texto)


def traduzir(transcricao, idioma, trabalho, progresso):
    destino = os.path.join(trabalho, "traducao.json")
    if os.path.exists(destino):
        return _ler_json(destino)

    from .cue import Cue
    from .pipeline import _translate_with_llm, _safe_detect

    falas = juntar_frases_partidas([dict(f) for f in transcricao])
    for fala in falas:
        if not fala.get("speaker"):
            fala["speaker"] = None
    cues = [Cue.from_source(f["index"], f["start"], f["end"], f["source_text"], f["speaker"])
            for f in falas]
    idioma = idioma or _safe_detect(cues) or "en"

    total = len(cues)
    for i in range(0, total, TRADUZIR_EM_BLOCOS_DE):
        aguardar_vez(progresso, "traduzir", i / total)
        progresso("traduzir", i / total, f"Traduzindo as falas... {i} de {total}")
        bloco = cues[i:i + TRADUZIR_EM_BLOCOS_DE]
        _, falhas = _translate_with_llm(bloco, idioma, llm_client=None, speaker_genders=None,
                                        progress=None, cancel_event=None)
        if falhas:
            # Segunda chance, uma por uma: fala que ficasse no idioma
            # original seria sintetizada em inglês com voz brasileira.
            for cue in bloco:
                if cue.text == cue.source_text:
                    try:
                        _translate_with_llm([cue], idioma, llm_client=None, speaker_genders=None,
                                            progress=None, cancel_event=None)
                    except Exception as exc:  # noqa: BLE001 - registra e segue
                        log(f"tradução falhou na fala {cue.index}: {exc}")

    if not filas_ocupadas():
        _descarregar_ollama()

    for fala, cue in zip(falas, cues):
        traduzido = cue.text != cue.source_text or not re.search(r"[A-Za-z]", cue.source_text)
        fala["text"] = _limpar_traducao(cue.text) if traduzido else ""
    _gravar_json(destino, {"idioma": idioma, "falas": falas})
    return _ler_json(destino)


# --------------------------------------------------------------------------
# 4. Separação de voz e fundo (Demucs)
# --------------------------------------------------------------------------

def separar_voz(original, info, trabalho, progresso):
    voz = os.path.join(trabalho, "voz.wav")
    fundo = os.path.join(trabalho, "fundo.wav")
    if os.path.exists(voz) and os.path.exists(fundo):
        return voz, fundo

    fps = Fraction(info["fps_num"], info["fps_den"])
    total = float(info["quadros"] / fps)
    pasta = os.path.join(trabalho, "separacao")
    os.makedirs(pasta, exist_ok=True)
    pedacos = max(1, math.ceil(total / DEMUCS_PEDACO_S))
    vozes, fundos = [], []

    for k in range(pedacos):
        voz_k = os.path.join(pasta, f"voz_{k:03d}.wav")
        fundo_k = os.path.join(pasta, f"fundo_{k:03d}.wav")
        vozes.append(voz_k)
        fundos.append(fundo_k)
        if os.path.exists(voz_k) and os.path.exists(fundo_k):
            continue
        aguardar_vez(progresso, "separar", k / pedacos, VRAM_DEMUCS)
        progresso("separar", k / pedacos,
                  f"Separando a voz da música e dos efeitos... parte {k + 1} de {pedacos}", forcar=True)

        a = k * DEMUCS_PEDACO_S
        b = min(total, a + DEMUCS_PEDACO_S)
        ca = max(0.0, a - DEMUCS_CONTEXTO_S)
        cb = min(total, b + DEMUCS_CONTEXTO_S)
        temp = os.path.join(pasta, "tmp")
        shutil.rmtree(temp, ignore_errors=True)
        os.makedirs(temp)
        pedaco = os.path.join(temp, "pedaco.wav")
        _rodar(["ffmpeg", "-y", "-v", "error", "-ss", f"{ca:.6f}", "-i", original,
                "-t", f"{cb - ca:.6f}", "-c:a", "pcm_s16le", pedaco])
        _rodar([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
                "-d", "cuda", "-o", temp, pedaco])
        saida_demucs = os.path.join(temp, "htdemucs", "pedaco")
        # Corta o contexto extra das pontas e já converte pra 48 kHz.
        for origem, destino in (("vocals.wav", voz_k), ("no_vocals.wav", fundo_k)):
            _rodar(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(saida_demucs, origem),
                    "-ss", f"{a - ca:.6f}", "-t", f"{b - a:.6f}",
                    "-ar", str(TAXA), "-ac", "2", "-c:a", "pcm_s16le", destino + ".parcial.wav"])
            os.replace(destino + ".parcial.wav", destino)
        shutil.rmtree(temp, ignore_errors=True)

    for lista, destino in ((vozes, voz), (fundos, fundo)):
        indice = os.path.join(pasta, os.path.basename(destino) + ".txt")
        with open(indice, "w", encoding="utf-8") as f:
            for caminho in lista:
                f.write(f"file '{caminho}'\n")
        _rodar(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", indice,
                "-c", "copy", destino + ".parcial.wav"])
        os.replace(destino + ".parcial.wav", destino)
    progresso("separar", 1.0, "Voz e fundo separados.", forcar=True)
    return voz, fundo


# --------------------------------------------------------------------------
# 5. Voz de referência de cada locutor
# --------------------------------------------------------------------------

def _atribuir_sem_locutor(falas):
    """Fala sem locutor marcado herda o da fala mais próxima no tempo."""
    com = [f for f in falas if f.get("speaker")]
    if not com:
        for f in falas:
            f["speaker"] = "SPEAKER_00"
        return
    for f in falas:
        if not f.get("speaker"):
            centro = (f["start"] + f["end"]) / 2
            vizinha = min(com, key=lambda g: abs((g["start"] + g["end"]) / 2 - centro))
            f["speaker"] = vizinha["speaker"]


def montar_referencias(falas, fonte, trabalho, progresso):
    """Os melhores 30 s de cada locutor: falas mais longas primeiro (entre
    1,5 e 15 s, sem música), da faixa já limpa pelo Demucs."""
    import numpy as np
    import soundfile as sf

    destino_indice = os.path.join(trabalho, "referencias.json")
    if os.path.exists(destino_indice):
        return _ler_json(destino_indice)

    pasta = os.path.join(trabalho, "referencias")
    os.makedirs(pasta, exist_ok=True)
    progresso("referencias", 0.0, "Montando a voz de referência de cada locutor...", forcar=True)

    por_locutor = {}
    for f in falas:
        por_locutor.setdefault(f["speaker"], []).append(f)

    indice = {}
    silencio = np.zeros(int(0.15 * TAXA), dtype=np.float32)
    with sf.SoundFile(fonte) as audio:
        for locutor, lista in sorted(por_locutor.items()):
            boas = [f for f in lista if 1500 <= f["end"] - f["start"] <= 15000
                    and not _SO_MUSICA.match(f["source_text"])]
            resto = [f for f in lista if f not in boas]
            candidatas = (sorted(boas, key=lambda f: f["start"] - f["end"])
                          + sorted(resto, key=lambda f: f["start"] - f["end"]))
            pedacos, total = [], 0.0
            for f in candidatas:
                if total >= REF_ALVO_S:
                    break
                ini = int(f["start"] / 1000 * TAXA)
                fim = int(f["end"] / 1000 * TAXA)
                if fim <= ini or ini >= audio.frames:
                    continue
                audio.seek(ini)
                trecho = audio.read(min(fim, audio.frames) - ini, dtype="float32", always_2d=True)
                pedacos.extend([trecho.mean(axis=1), silencio])
                total += len(trecho) / TAXA
            fala_total = sum(f["end"] - f["start"] for f in lista) / 1000
            if not pedacos:
                continue
            caminho = os.path.join(pasta, f"{locutor}.wav")
            sf.write(caminho, np.concatenate(pedacos), TAXA, subtype="PCM_16")
            indice[locutor] = {"arquivo": caminho, "segundos": round(total, 1),
                               "falas": len(lista), "segundos_fala": round(fala_total, 1)}
    _gravar_json(destino_indice, indice)
    return indice


# --------------------------------------------------------------------------
# 6. Síntese (XTTS-v2)
# --------------------------------------------------------------------------

def _texto_para_voz(texto):
    texto = re.sub(r"<[^>]+>", "", texto or "")
    texto = re.sub(r"[♪♫#]+", " ", texto)
    texto = re.sub(r"\[[^\]]*\]", " ", texto)
    texto = re.sub(r"^\s*[-–—]\s*", "", texto)
    texto = " ".join(texto.split())
    # Validado ouvindo: com ponto final o XTTS às vezes lê "ponto" em voz alta.
    return re.sub(r"[.]+$", "", texto).strip()


def _partes_do_texto(texto):
    """Texto longo demais pro XTTS vira partes, quebrando entre frases."""
    if len(texto) <= MAX_CARACTERES_XTTS:
        return [texto]
    frases = re.split(r"(?<=[.!?;:])\s+", texto)
    partes, atual = [], ""
    for frase in frases:
        if atual and len(atual) + len(frase) + 1 > MAX_CARACTERES_XTTS:
            partes.append(atual)
            atual = frase
        else:
            atual = (atual + " " + frase).strip()
    if atual:
        partes.append(atual)
    final = []
    for parte in partes:
        while len(parte) > MAX_CARACTERES_XTTS:
            corte = parte.rfind(",", 0, MAX_CARACTERES_XTTS)
            if corte < 40:
                corte = parte.rfind(" ", 0, MAX_CARACTERES_XTTS)
            final.append(parte[:corte].strip())
            parte = parte[corte + 1:].strip()
        final.append(parte)
    return [re.sub(r"[.]+$", "", p).strip() for p in final if p.strip()]


class Sintetizador:
    """XTTS-v2 carregado uma vez, com a "impressão" de cada voz calculada uma
    vez por locutor -- o ``tts_to_file`` dos pilotos recalculava a cada fala.
    Os parâmetros de geração são os mesmos do ``tts_to_file``: os do
    config do modelo, com a temperatura trocada."""

    def __init__(self):
        self.tts = None
        self.latentes = {}

    def carregar(self):
        os.environ["COQUI_TOS_AGREED"] = "1"
        import torch
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
        from TTS.api import TTS
        log("carregando XTTS-v2...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        self.latentes = {}

    def descarregar(self):
        import gc
        import torch
        self.tts = None
        self.latentes = {}
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def _modelo(self):
        return self.tts.synthesizer.tts_model

    @property
    def _config(self):
        return self.tts.synthesizer.tts_config

    def latente(self, chave, referencia):
        if chave not in self.latentes:
            cfg = self._config
            self.latentes[chave] = self._modelo.get_conditioning_latents(
                audio_path=[referencia], gpt_cond_len=cfg.gpt_cond_len,
                gpt_cond_chunk_len=cfg.gpt_cond_chunk_len,
                max_ref_length=cfg.max_ref_len, sound_norm_refs=cfg.sound_norm_refs)
        return self.latentes[chave]

    def embedding(self, chave, referencia):
        """Vetor da voz (512 números), usado para reconhecer o mesmo ator
        em outro filme."""
        _, emb = self.latente(chave, referencia)
        return [float(x) for x in emb.detach().cpu().flatten().tolist()]

    def falar(self, texto, chave, referencia, temperatura):
        import numpy as np
        gpt, emb = self.latente(chave, referencia)
        cfg = self._config
        pedacos = []
        for parte in _partes_do_texto(texto):
            saida = self._modelo.inference(
                parte, "pt", gpt, emb, temperature=temperatura,
                length_penalty=cfg.length_penalty, repetition_penalty=cfg.repetition_penalty,
                top_k=cfg.top_k, top_p=cfg.top_p)
            onda = saida["wav"]
            if hasattr(onda, "cpu"):
                onda = onda.cpu().numpy()
            pedacos.append(_aparar_silencio(np.asarray(onda, dtype=np.float32).squeeze()))
            pedacos.append(np.zeros(int(0.12 * TAXA_XTTS), dtype=np.float32))
        return np.concatenate(pedacos[:-1]) if pedacos else np.zeros(0, dtype=np.float32)


def _aparar_silencio(onda):
    """Tira o silêncio das pontas (o ``tts_to_file`` ainda somava 0,4 s de
    silêncio no fim de cada fala, o que forçava acelerar à toa), mantendo
    uma folga curta pra não comer o fim de palavra dita baixinho."""
    import librosa
    if onda.size == 0:
        return onda
    _, (ini, fim) = librosa.effects.trim(onda, top_db=40)
    ini = max(0, ini - int(0.03 * TAXA_XTTS))
    fim = min(len(onda), fim + int(0.08 * TAXA_XTTS))
    return onda[ini:fim]


def _vozes_do_banco(referencias, falas, entrada, opc):
    """Locutor já ligado a um ator (à mão, ou reconhecido pela voz entre o
    elenco do filme) usa a voz acumulada do ator -- mais material, voz
    melhor -- em vez da referência só deste filme."""
    if not opc["reaproveitar_voz"]:
        return {}
    try:
        from . import vozes
    except Exception:  # noqa: BLE001
        return {}
    escolhas = {}
    atribuidos = vozes.atores_dos_locutores(os.path.realpath(entrada))
    for locutor in referencias:
        ator = atribuidos.get(locutor)
        if not ator:
            continue
        try:
            destino = os.path.join(os.path.dirname(referencias[locutor]["arquivo"]),
                                   f"ator-{_slug(ator['id'])}.wav")
            vozes.montar_referencia_combinada(ator["id"], destino, max_segundos=REF_ALVO_S + 5)
            escolhas[locutor] = {"arquivo": destino, "ator": ator["nome"], "ator_id": ator["id"]}
        except ValueError:
            continue
    return escolhas


def sintetizar(falas, referencias, entrada, trabalho, info, opc, progresso, sint):
    import numpy as np
    import soundfile as sf

    pasta = os.path.join(trabalho, "sintese")
    os.makedirs(pasta, exist_ok=True)
    registro = os.path.join(trabalho, "sintese.jsonl")
    feitas = {}
    if os.path.exists(registro):
        with open(registro, encoding="utf-8") as f:
            for linha in f:
                dado = json.loads(linha)
                if os.path.exists(os.path.join(pasta, f"{dado['index']}.wav")):
                    feitas[dado["index"]] = dado

    fps = Fraction(info["fps_num"], info["fps_den"])
    principal = max(referencias, key=lambda k: referencias[k]["segundos_fala"]) if referencias else None
    banco = _vozes_do_banco(referencias, falas, entrada, opc)
    for locutor, escolha in banco.items():
        log(f"{locutor} usa a voz salva de {escolha['ator']}")

    def referencia_de(locutor):
        if not opc["clonar_por_locutor"] or locutor not in referencias:
            locutor = principal
        if locutor in banco:
            return f"banco:{banco[locutor]['ator_id']}", banco[locutor]["arquivo"]
        ref = referencias[locutor]
        if ref["segundos"] < REF_MINIMA_S and principal and locutor != principal:
            return referencia_de(principal)
        return locutor, ref["arquivo"]

    pendentes = [f for f in falas if f.get("dublar") and f["index"] not in feitas]
    total = len([f for f in falas if f.get("dublar")])
    if pendentes:
        aguardar_vez(progresso, "sintetizar", len(feitas) / max(total, 1), VRAM_XTTS)
        progresso("sintetizar", len(feitas) / max(total, 1), "Carregando o modelo de voz...", forcar=True)
        sint.carregar()

    with open(registro, "a", encoding="utf-8") as reg:
        for n, fala in enumerate(pendentes):
            if n and n % CHECAR_FILAS_A_CADA == 0 and filas_ocupadas():
                log("legenda entrou na fila: liberando a placa de vídeo")
                sint.descarregar()
                aguardar_vez(progresso, "sintetizar", len(feitas) / max(total, 1), VRAM_XTTS)
                sint.carregar()

            chave, ref = referencia_de(fala["speaker"])
            slot = (fala["q_fim"] - fala["q_ini"]) / fps
            texto = fala["texto_voz"]
            melhor, tentativas = None, 0
            temperatura = opc["temperatura"]
            while tentativas < TENTATIVAS_SINTESE:
                tentativas += 1
                onda = sint.falar(texto, chave, ref, temperatura)
                duracao = len(onda) / TAXA_XTTS
                if duracao >= 0.15 and (melhor is None or duracao < melhor[1]):
                    melhor = (onda, duracao)
                razao = duracao / max(float(slot), 0.3)
                if duracao >= 0.15 and razao <= RAZAO_SUSPEITA:
                    break
                if melhor is not None and melhor[1] / max(float(slot), 0.3) <= RAZAO_ACEITA:
                    break
                temperatura = TEMPERATURA_REPETICAO  # pilotos: repetir com 0,6
            if melhor is None:
                log(f"fala {fala['index']}: o XTTS não gerou áudio, fica o original")
                continue
            sf.write(os.path.join(pasta, f"{fala['index']}.wav"), melhor[0], TAXA_XTTS, subtype="PCM_16")
            dado = {"index": fala["index"], "duracao": round(melhor[1], 3), "tentativas": tentativas,
                    "voz": chave}
            reg.write(json.dumps(dado) + "\n")
            reg.flush()
            feitas[fala["index"]] = dado
            progresso("sintetizar", len(feitas) / max(total, 1),
                      f"Sintetizando as falas... {len(feitas)} de {total}")

    embeddings = {}
    if referencias and sint.tts is None and opc["reaproveitar_voz"]:
        aguardar_vez(progresso, "sintetizar", 1.0, VRAM_XTTS)
        sint.carregar()
    if sint.tts is not None:
        for locutor, ref in referencias.items():
            try:
                embeddings[locutor] = sint.embedding(locutor, ref["arquivo"])
            except Exception as exc:  # noqa: BLE001
                log(f"não consegui calcular a voz de {locutor}: {exc}")
        sint.descarregar()
    return feitas, banco, embeddings


# --------------------------------------------------------------------------
# 7. Ajuste de velocidade da fala
# --------------------------------------------------------------------------

def ajustar_velocidade(falas, feitas, trabalho, info, opc, progresso):
    pasta = os.path.join(trabalho, "ajustada")
    os.makedirs(pasta, exist_ok=True)
    fps = Fraction(info["fps_num"], info["fps_den"])
    finais = {}
    total = len(feitas)
    for n, fala in enumerate(f for f in falas if f["index"] in feitas):
        origem = os.path.join(trabalho, "sintese", f"{fala['index']}.wav")
        destino = os.path.join(pasta, f"{fala['index']}.wav")
        duracao = feitas[fala["index"]]["duracao"]
        slot = float((fala["q_fim"] - fala["q_ini"]) / fps)
        tempo = 1.0
        if opc["ajustar_ritmo"] and duracao > slot * 1.05:
            tempo = min(duracao / slot, opc["max_acelerar"])
        if not os.path.exists(destino):
            filtro = ["-af", f"atempo={tempo:.4f}"] if tempo > 1.0 else []
            _rodar(["ffmpeg", "-y", "-v", "error", "-i", origem, *filtro,
                    "-ar", str(TAXA), "-ac", "2", "-c:a", "pcm_s16le", destino + ".parcial.wav"])
            os.replace(destino + ".parcial.wav", destino)
        finais[fala["index"]] = {"arquivo": destino, "duracao": duracao / tempo, "tempo": tempo}
        if n % 25 == 0:
            progresso("ajustar", n / max(total, 1), f"Ajustando o ritmo das falas... {n} de {total}")
    return finais


# --------------------------------------------------------------------------
# 8. Linha do tempo e trilha de áudio
# --------------------------------------------------------------------------

def _amostra(quadros, fps):
    """Posição na trilha (em amostras) do começo do quadro ``quadros``.
    Arredonda a partir do total acumulado, nunca somando arredondamentos --
    é isso que impede o áudio de escorregar ao longo de um filme de horas."""
    num = 2 * quadros * TAXA * fps.denominator + fps.numerator
    return num // (2 * fps.numerator)


def montar_linha_do_tempo(falas, finais, info, opc):
    """Cada fala dublada ocupa um trecho do vídeo; o resto passa igual.

    Fala que cabe no tempo original: o vídeo segue no ritmo normal. Fala que
    não cabe nem acelerada: o vídeo daquele trecho fica mais lento até ela
    caber (até ``max_desacelerar_video``), e só o que passar disso é cortado."""
    fps = Fraction(info["fps_num"], info["fps_den"])
    segmentos = []
    cursor = 0
    for fala in falas:
        if fala["index"] not in finais:
            continue
        ini, fim = fala["q_ini"], fala["q_fim"]
        if ini > cursor:
            segmentos.append({"tipo": "brecha", "de": cursor, "ate": ini, "saida": ini - cursor})
        slot = fim - ini
        precisa = math.ceil(finais[fala["index"]]["duracao"] * fps)
        saida = slot if precisa <= slot else min(precisa, math.floor(slot * opc["max_desacelerar_video"]))
        segmentos.append({"tipo": "fala", "de": ini, "ate": fim, "saida": saida,
                          "index": fala["index"], "cortou": precisa > saida})
        cursor = fim
    if cursor < info["quadros"]:
        segmentos.append({"tipo": "brecha", "de": cursor, "ate": info["quadros"],
                          "saida": info["quadros"] - cursor})
    return segmentos


def montar_trilha(segmentos, finais, original, fundo, trabalho, opc, info, progresso):
    import numpy as np
    import soundfile as sf

    destino = os.path.join(trabalho, "trilha_dublada.wav")
    if os.path.exists(destino):
        return destino
    fps = Fraction(info["fps_num"], info["fps_den"])
    buffer = int(BUFFER_BRECHA_S * TAXA)
    rampa = int(0.005 * TAXA)

    def ler(arquivo, inicio, n):
        dados = np.zeros((n, 2), dtype=np.float32)
        if n <= 0 or inicio >= arquivo.frames:
            return dados
        arquivo.seek(max(0, inicio))
        lido = arquivo.read(min(n, arquivo.frames - inicio), dtype="float32", always_2d=True)
        if lido.shape[1] == 1:
            lido = np.repeat(lido, 2, axis=1)
        dados[:len(lido)] = lido[:, :2]
        return dados

    preservar = opc["preservar_trilha"]
    saida_q = 0
    with sf.SoundFile(original) as orig, sf.SoundFile(fundo) as bg, \
            sf.SoundFile(destino + ".parcial.wav", "w", samplerate=TAXA, channels=2,
                         subtype="PCM_16") as out:
        for n, seg in enumerate(segmentos):
            ini_out = _amostra(saida_q, fps)
            fim_out = _amostra(saida_q + seg["saida"], fps)
            quantas = fim_out - ini_out
            ini_src = _amostra(seg["de"], fps)
            if seg["tipo"] == "brecha":
                if preservar:
                    dados = ler(orig, ini_src, quantas)
                    k = min(quantas, buffer)
                    dados[:k] = ler(bg, ini_src, k)  # mascara o fim da palavra em inglês
                else:
                    dados = np.zeros((quantas, 2), dtype=np.float32)
            else:
                fundo_trecho = ler(bg, ini_src, quantas) if preservar else np.zeros((quantas, 2), np.float32)
                voz, _ = sf.read(finais[seg["index"]]["arquivo"], dtype="float32", always_2d=True)
                voz = voz[:quantas]
                if len(voz) > 2 * rampa:
                    curva = np.linspace(0, 1, rampa, dtype=np.float32)[:, None]
                    voz[:rampa] *= curva
                    voz[-rampa:] *= curva[::-1]
                dados = fundo_trecho.copy()
                m = len(voz)
                # Igual ao amix dos pilotos: enquanto a voz toca, metade de cada;
                # quando ela acaba, o fundo volta inteiro.
                dados[:m] = 0.5 * (fundo_trecho[:m] + voz[:, :2])
            out.write(np.clip(dados, -1.0, 1.0))
            saida_q += seg["saida"]
            if n % 50 == 0:
                progresso("audio", n / len(segmentos), "Montando a trilha dublada...")
    os.replace(destino + ".parcial.wav", destino)
    return destino


def _tempo_srt(segundos):
    ms = int(round(segundos * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def legenda_da_dublagem(segmentos, falas, finais, info, destino):
    """Legenda em português já no tempo do vídeo dublado (que mudou)."""
    fps = Fraction(info["fps_num"], info["fps_den"])
    por_indice = {f["index"]: f for f in falas}
    linhas, saida_q, n = [], 0, 0
    for seg in segmentos:
        if seg["tipo"] == "fala":
            ini = float(saida_q / fps)
            dur = min(finais[seg["index"]]["duracao"], float(seg["saida"] / fps))
            n += 1
            linhas.append(f"{n}\n{_tempo_srt(ini)} --> {_tempo_srt(ini + max(dur, 0.5))}\n"
                          f"{por_indice[seg['index']]['text']}\n")
        saida_q += seg["saida"]
    with open(destino, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))


# --------------------------------------------------------------------------
# 9. Vídeo, quadro a quadro, num encoder só
# --------------------------------------------------------------------------

def _trechos_de_video(segmentos):
    """Junta o que passa no ritmo normal num trecho só (um decodificador
    por trecho); cada fala com vídeo mais lento vira um trecho próprio."""
    trechos = []
    for seg in segmentos:
        normal = seg["saida"] == seg["ate"] - seg["de"]
        if normal and trechos and trechos[-1]["normal"] and trechos[-1]["ate"] == seg["de"]:
            trechos[-1]["ate"] = seg["ate"]
            trechos[-1]["saida"] += seg["saida"]
        else:
            trechos.append({"de": seg["de"], "ate": seg["ate"], "saida": seg["saida"], "normal": normal})
    return trechos


def montar_video(entrada, trilha, segmentos, info, opc, destino, legenda, progresso):
    fps = Fraction(info["fps_num"], info["fps_den"])
    largura, altura = info["largura"], info["altura"]
    tamanho_quadro = largura * altura * 3 // 2
    total = sum(s["saida"] for s in segmentos)
    trechos = _trechos_de_video(segmentos)

    os.makedirs(os.path.dirname(destino), exist_ok=True)
    parcial = destino + ".parcial"
    codificador = [
        "nice", "-n", "19", "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{largura}x{altura}",
        "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "-",
        "-i", trilha,
    ]
    if legenda:
        codificador += ["-i", legenda]
    codificador += ["-map", "0:v:0", "-map", "1:a:0"]
    if legenda:
        codificador += ["-map", "2:s:0", "-c:s", "srt", "-metadata:s:s:0", "language=por",
                        "-metadata:s:s:0", "title=Português (falas da dublagem)"]
    codificador += [
        "-vf", f"setsar={info['sar'].replace(':', '/')}",
        "-c:v", "libx264", "-preset", opc["preset"], "-crf", str(opc["crf"]),
        "-tune", "film", "-pix_fmt", "yuv420p",
        "-af", "loudnorm=I=-18:TP=-1.5:LRA=11", "-ar", str(TAXA),
        "-c:a", "aac", "-b:a", "192k",
        "-metadata:s:a:0", "language=por", "-metadata:s:a:0", "title=Português (dublagem IA)",
        "-map_chapters", "-1", "-f", "matroska", parcial,
    ]
    log("$ " + " ".join(codificador))
    enc = subprocess.Popen(codificador, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    erros_enc = []

    def drenar():
        for linha in enc.stderr:
            erros_enc.append(linha.decode("utf-8", "replace").strip())
    import threading
    threading.Thread(target=drenar, daemon=True).start()

    escritos = 0
    escala = f"scale={largura}:{altura}," if True else ""
    try:
        for trecho in trechos:
            entrada_q = trecho["ate"] - trecho["de"]
            fator = trecho["saida"] / entrada_q
            pts = "PTS-STARTPTS" if trecho["normal"] else f"(PTS-STARTPTS)*{fator:.9f}"
            inicio = max(0.0, info["t0"] + (trecho["de"] - 0.5) / float(fps))
            filtro = (f"setpts={pts},fps={fps.numerator}/{fps.denominator},"
                      f"tpad=stop_mode=clone:stop=-1,{escala}format=yuv420p")
            dec = subprocess.Popen(
                ["nice", "-n", "19", "ffmpeg", "-v", "error", "-ss", f"{inicio:.6f}", "-i", entrada,
                 "-map", f"0:{info['indice_video']}", "-an", "-sn", "-dn", "-vf", filtro,
                 "-frames:v", str(trecho["saida"]), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ultimo = None
            for _ in range(trecho["saida"]):
                quadro = dec.stdout.read(tamanho_quadro)
                if len(quadro) < tamanho_quadro:
                    if ultimo is None:
                        raise ErroDublagem("O vídeo terminou antes do esperado num trecho "
                                           f"(quadro {trecho['de']}).")
                    quadro = ultimo
                    log(f"trecho {trecho['de']}: faltou quadro, repetindo o anterior")
                enc.stdin.write(quadro)
                ultimo = quadro
                escritos += 1
                if escritos % 240 == 0:
                    progresso("video", escritos / total,
                              f"Montando o vídeo... {int(100 * escritos / total)}%")
            dec.stdout.close()
            dec.stderr.read()
            dec.wait()
        enc.stdin.close()
    except BrokenPipeError:
        pass
    enc.wait()
    if enc.returncode != 0:
        raise ErroDublagem("O encoder de vídeo falhou: " + " | ".join(erros_enc[-5:]))
    return parcial, total


def verificar(parcial, destino, quadros, info, progresso, completa=True):
    fps = Fraction(info["fps_num"], info["fps_den"])
    esperado = float(quadros / fps)
    progresso("verificar", 0.0, "Conferindo o arquivo final...", forcar=True)
    duracao = _duracao(parcial)
    if abs(duracao - esperado) > 1.0:
        raise ErroDublagem(f"O arquivo final tem {duracao:.1f}s, mas deveria ter {esperado:.1f}s.")
    if completa:
        # Pega erro de decodificação que não aparece só lendo os metadados.
        _rodar(["nice", "-n", "19", "ffmpeg", "-v", "error", "-xerror", "-i", parcial,
                "-f", "null", "-"])
    os.replace(parcial, destino)
    return duracao


# --------------------------------------------------------------------------
# Orquestração
# --------------------------------------------------------------------------

def preparar_falas(traducao, info, opc):
    """Leva os tempos pra grade de quadros do vídeo e marca o que dublar."""
    fps = Fraction(info["fps_num"], info["fps_den"])
    falas = traducao["falas"]
    _atribuir_sem_locutor(falas)
    cursor = 0
    for fala in falas:
        ini = max(cursor, int(round(fala["start"] / 1000 * fps)))
        fim = min(info["quadros"], max(ini + 1, int(round(fala["end"] / 1000 * fps))))
        fala["q_ini"], fala["q_fim"] = ini, fim
        texto = _texto_para_voz(fala.get("text") or "")
        fala["texto_voz"] = texto
        origem = fala["source_text"].lower()
        pular = (not texto or not re.search(r"[A-Za-zÀ-ÿ]", texto) or fim <= ini
                 or "♪" in fala["source_text"] or _SO_MUSICA.match(fala["source_text"])
                 or any(frase in origem for frase in opc["frases_nao_dublar"]))
        fala["dublar"] = not pular
        if fala["dublar"]:
            cursor = fim
    return falas


def _limpar_intermediarios(trabalho):
    for nome in ("original.wav", "voz.wav", "fundo.wav", "trilha_dublada.wav"):
        try:
            os.remove(os.path.join(trabalho, nome))
        except FileNotFoundError:
            pass
    for pasta in ("separacao", "ajustada", "whisper"):
        shutil.rmtree(os.path.join(trabalho, pasta), ignore_errors=True)


def _registrar_vozes(entrada, referencias, embeddings, falas, trabalho):
    """Guarda a voz de cada locutor no banco, pra ser ligada a um ator na
    tela Atores e reaproveitada nos próximos filmes."""
    try:
        from . import vozes
        vozes.registrar_locutores_do_filme(os.path.realpath(entrada), referencias, embeddings)
    except Exception as exc:  # noqa: BLE001 - o filme dublado não depende disto
        log(f"não consegui guardar as vozes no banco: {exc}")


def dublar(entrada, *, inicio=None, fim=None, raiz_trabalho=None, saida=None,
           verificacao_completa=True):
    progresso = Progresso()
    opc = opcoes()
    if not os.path.isfile(entrada):
        raise ErroDublagem("O arquivo não está mais na pasta.")
    idioma = opc["idioma_origem"] or (config.get_whisper_language() or "").lower()

    trabalho = pasta_de_trabalho(entrada, raiz_trabalho, inicio, fim)
    os.makedirs(trabalho, exist_ok=True)
    destino = saida or caminho_de_saida(entrada, opc["pasta_saida"], inicio, fim)
    log(f"pasta de trabalho: {trabalho}")
    log(f"saída: {destino}")
    emitir(trabalho=trabalho, saida=destino)

    caminho_info = os.path.join(trabalho, "info.json")
    if os.path.exists(caminho_info):
        info = _ler_json(caminho_info)
    else:
        progresso("preparar", 0.0, "Lendo o arquivo...", forcar=True)
        info = sondar(entrada, idioma, inicio, fim)
        _gravar_json(caminho_info, info)
    if not idioma and info.get("idioma_audio"):
        idioma = info["idioma_audio"][:2]

    original = extrair_audio(entrada, info, trabalho, progresso)
    transcricao = transcrever(original, idioma, trabalho, progresso)
    traducao = traduzir(transcricao, idioma, trabalho, progresso)
    falas = preparar_falas(traducao, info, opc)
    voz, fundo = separar_voz(original, info, trabalho, progresso)
    referencias = montar_referencias(falas, voz if opc["limpar_referencia"] else original,
                                     trabalho, progresso)
    if not referencias:
        raise ErroDublagem("Não sobrou nenhuma fala para dublar.")

    sint = Sintetizador()
    feitas, banco, embeddings = sintetizar(falas, referencias, entrada, trabalho, info, opc,
                                           progresso, sint)
    _registrar_vozes(entrada, referencias, embeddings, falas, trabalho)
    finais = ajustar_velocidade(falas, feitas, trabalho, info, opc, progresso)
    segmentos = montar_linha_do_tempo(falas, finais, info, opc)
    _gravar_json(os.path.join(trabalho, "linha_do_tempo.json"), segmentos)
    trilha = montar_trilha(segmentos, finais, original, fundo, trabalho, opc, info, progresso)
    legenda = os.path.join(trabalho, "falas_dubladas.srt")
    legenda_da_dublagem(segmentos, falas, finais, info, legenda)

    parcial, quadros = montar_video(entrada, trilha, segmentos, info, opc, destino, legenda, progresso)
    duracao = verificar(parcial, destino, quadros, info, progresso, completa=verificacao_completa)
    shutil.copyfile(legenda, os.path.splitext(destino)[0] + ".pt.srt")

    fps = Fraction(info["fps_num"], info["fps_den"])
    falas_fala = [s for s in segmentos if s["tipo"] == "fala"]
    relatorio = {
        "saida": destino,
        "falas": len(falas),
        "dubladas": len(finais),
        "aceleradas": sum(1 for f in finais.values() if f["tempo"] > 1.0),
        "video_mais_lento": sum(1 for s in falas_fala if s["saida"] > s["ate"] - s["de"]),
        "cortadas": sum(1 for s in falas_fala if s["cortou"]),
        "repetidas": sum(1 for f in feitas.values() if f["tentativas"] > 1),
        "locutores": len(referencias),
        "vozes_do_banco": {k: v["ator"] for k, v in banco.items()},
        "duracao_original_s": round(float(info["quadros"] / fps), 1),
        "duracao_final_s": round(duracao, 1),
        "trabalho": trabalho,
    }
    _gravar_json(os.path.join(trabalho, "relatorio.json"), relatorio)
    _limpar_intermediarios(trabalho)
    progresso("verificar", 1.0, "Pronto!", forcar=True)
    emitir(fim=relatorio)
    return relatorio


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dubla um vídeo para português com IA.")
    parser.add_argument("entrada")
    parser.add_argument("--inicio", type=float, help="segundo inicial (teste com trecho)")
    parser.add_argument("--fim", type=float, help="segundo final (teste com trecho)")
    parser.add_argument("--trabalho", help="pasta raiz dos intermediários")
    parser.add_argument("--saida", help="arquivo .mkv de saída")
    parser.add_argument("--verificacao-rapida", action="store_true",
                        help="não decodifica o arquivo final inteiro para conferir")
    args = parser.parse_args(argv)

    _isolar_stdout()
    try:
        dublar(args.entrada, inicio=args.inicio, fim=args.fim, raiz_trabalho=args.trabalho,
               saida=args.saida, verificacao_completa=not args.verificacao_rapida)
    except ErroDublagem as exc:
        emitir(erro=str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001 - qualquer falha vira mensagem na tela
        import traceback
        traceback.print_exc()
        emitir(erro=f"{exc.__class__.__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
