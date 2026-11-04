"""Rotas da dublagem e da tela nova (barra lateral, Atores, Configurações).

Só entra no ar com ``AUTOSRT_DUBLAGEM=1`` (instância da porta 8002): as
instâncias de produção (8000) e beta (8001) continuam exatamente como
estavam, mesmo que este código chegue nelas por um merge.

A dublagem em si roda em outro processo (:mod:`autosrt.dublagem`, no venv
com torch/TTS/Demucs). Aqui fica só a fila: um trabalho por vez, que sobe o
processo, lê o progresso que ele imprime e o mata se alguém cancelar.

As ferramentas do HD (atualizar fichas, gêneros, sinopses, legendas) não
rodam aqui: são repassadas para a produção (``AUTOSRT_FILMES_PROXY``,
porta 8000), que já é dona delas -- duas instâncias mexendo nos mesmos
``.nfo`` ao mesmo tempo poderiam gravar uma por cima da outra.
"""

import collections
import json
import logging
import os
import signal
import subprocess
import threading
import time

from flask import Response, jsonify, request, send_file

from . import config, historico, pipeline, tmdb, vozes
from .jobs import FINAIS, JobQueue

logger = logging.getLogger(__name__)

PYTHON_DUBLAGEM_PADRAO = "/opt/autosrt-dub/venv/bin/python"
RAIZ_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO_PENDENTES = "dublagens_pendentes.json"

#: Opções da dublagem que a tela grava (abas Voz e Qualidade) e seus padrões.
OPCOES_DUBLAGEM = {
    "dublagem_motor": "local",
    "dublagem_modelo": "referencia",
    "dublagem_temperatura": "0.65",
    "dublagem_limpar_referencia": "sim",
    "dublagem_reaproveitar_voz": "sim",
    "dublagem_preservar_trilha": "sim",
    "dublagem_idioma_origem": "",
    "dublagem_clonar_por_locutor": "sim",
    "dublagem_ajustar_ritmo": "sim",
    "dublagem_max_acelerar": "1.6",
    "dublagem_max_desacelerar_video": "4.0",
    "dublagem_frases_nao_dublar": "",
    "dublagem_pasta_saida": "/mnt/filmes_storage/DUBLAGEM IA",
    "dublagem_crf": "18",
    "dublagem_preset": "medium",
    "dublagem_auto": "false",
}


def _opcao(nome):
    valor = config.get_setting(nome)
    return OPCOES_DUBLAGEM[nome] if valor in (None, "") else str(valor)


def dublagem_automatica() -> bool:
    return _opcao("dublagem_auto").lower() in ("1", "true", "sim", "on")


# --------------------------------------------------------------------------
# Fila de dublagem
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Índice do acervo
# --------------------------------------------------------------------------
#
# A pasta desta instância é o acervo inteiro (14 TB, ~85 mil itens num HD
# NTFS). A listagem do autosrt varre tudo e ainda consulta o TMDB item a
# item -- o que é instantâneo na pasta de uploads da produção, aqui nunca
# termina. Então o acervo é varrido uma vez, em segundo plano, e a tela lê
# desse índice; o TMDB só é consultado para os poucos itens que aparecem na
# tela.

INTERVALO_INDICE_S = 30 * 60
ITENS_POR_PAGINA = 60

#: Acervo em HD, só de leitura: é de onde se escolhe filme para legendar ou
#: dublar. O arquivo enviado pelo navegador nunca vai parar aqui -- ele fica
#: na área temporária dentro do container (``media_dir``, igual à produção),
#: e só sai de lá quando alguém decide o que fazer com ele.
ACERVO_PADRAO = "/mnt/filmes_storage"


class IndiceDoAcervo:
    def __init__(self, raiz):
        self.raiz = raiz
        self._lock = threading.Lock()
        self._itens = []
        self.quando = None
        self.varrendo = False

    @property
    def pronto(self):
        return self.quando is not None

    def atualizar(self):
        with self._lock:
            if self.varrendo:
                return
            self.varrendo = True
        comeco = time.time()
        itens = []
        try:
            for raiz, pastas, nomes in os.walk(self.raiz):
                pastas[:] = [p for p in pastas if p not in
                             ("$RECYCLE.BIN", "System Volume Information", "metadata",
                              "dump", pipeline.ORIGINALS_DIRNAME)]
                for nome in nomes:
                    caminho = os.path.join(raiz, nome)
                    if not pipeline.is_media(caminho):
                        continue
                    try:
                        tamanho = os.path.getsize(caminho)
                    except OSError:
                        continue
                    relativo = os.path.relpath(caminho, self.raiz).replace(os.sep, "/")
                    itens.append({"nome": relativo, "pasta": os.path.dirname(relativo) or ".",
                                  "bytes": tamanho, "busca": relativo.lower()})
        except Exception:  # noqa: BLE001
            logger.exception("varredura do acervo")
        finally:
            with self._lock:
                if itens:
                    itens.sort(key=lambda i: i["nome"])
                    self._itens = itens
                    self.quando = time.time()
                self.varrendo = False
        logger.info("acervo indexado: %d vídeos em %.0fs", len(itens), time.time() - comeco)

    def em_segundo_plano(self):
        def laco():
            while True:
                self.atualizar()
                time.sleep(INTERVALO_INDICE_S)
        threading.Thread(target=laco, daemon=True).start()

    def adicionar(self, relativo, tamanho):
        """Põe um arquivo recém-chegado no índice na hora, sem esperar a
        próxima varredura do HD inteiro."""
        with self._lock:
            self._itens = [i for i in self._itens if i["nome"] != relativo]
            self._itens.append({"nome": relativo,
                                "pasta": os.path.dirname(relativo) or ".",
                                "bytes": tamanho, "busca": relativo.lower()})
            self._itens.sort(key=lambda i: i["nome"])
            if self.quando is None:
                self.quando = time.time()

    def remover(self, relativo):
        with self._lock:
            self._itens = [i for i in self._itens if i["nome"] != relativo]

    def listar(self, pasta="", busca="", pagina=0, limite=ITENS_POR_PAGINA):
        with self._lock:
            itens = self._itens
        pasta = (pasta or "").strip("/")
        if busca:
            alvo = busca.lower()
            achados = [i for i in itens if alvo in i["busca"]]
            subpastas = []
        else:
            prefixo = pasta + "/" if pasta else ""
            nivel = [i for i in itens if i["nome"].startswith(prefixo)]
            achados, vistas = [], {}
            for item in nivel:
                resto = item["nome"][len(prefixo):]
                if "/" in resto:
                    nome_pasta = resto.split("/", 1)[0]
                    dados = vistas.setdefault(nome_pasta, {"nome": nome_pasta, "itens": 0, "bytes": 0})
                    dados["itens"] += 1
                    dados["bytes"] += item["bytes"]
                else:
                    achados.append(item)
            subpastas = sorted(vistas.values(), key=lambda p: p["nome"])
        total = len(achados)
        return subpastas, achados[pagina * limite:(pagina + 1) * limite], total


def _tamanho_legivel(bytes_):
    for unidade in ("B", "KB", "MB", "GB"):
        if bytes_ < 1024 or unidade == "GB":
            return f"{bytes_:.0f} {unidade}" if unidade in ("B", "KB") else f"{bytes_:.1f} {unidade}"
        bytes_ /= 1024
    return f"{bytes_:.1f} GB"


def _caminho_pendentes():
    return os.path.join(config.app_directory(), ARQUIVO_PENDENTES)


def _pendentes():
    try:
        with open(_caminho_pendentes(), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return []


_lock_pendentes = threading.Lock()


def _gravar_pendentes(lista):
    os.makedirs(config.app_directory(), exist_ok=True)
    with open(_caminho_pendentes() + ".tmp", "w", encoding="utf-8") as f:
        json.dump(lista, f, ensure_ascii=False)
    os.replace(_caminho_pendentes() + ".tmp", _caminho_pendentes())


def _lembrar(caminho, extras):
    with _lock_pendentes:
        lista = [p for p in _pendentes() if p["caminho"] != caminho]
        lista.append({"caminho": caminho, **extras})
        _gravar_pendentes(lista)


def _esquecer(caminho):
    with _lock_pendentes:
        _gravar_pendentes([p for p in _pendentes() if p["caminho"] != caminho])


def _executar_dublagem(job):
    """Operário da fila: roda :mod:`autosrt.dublagem` noutro processo."""
    python = os.environ.get("AUTOSRT_DUB_PYTHON", PYTHON_DUBLAGEM_PADRAO)
    comando = [python, "-m", "autosrt.dublagem", job.entrada]
    for opcao in ("inicio", "fim"):
        if job.detalhes.get(opcao) is not None:
            comando += [f"--{opcao}", str(job.detalhes[opcao])]

    pasta_logs = os.path.join(config.app_directory(), "logs")
    os.makedirs(pasta_logs, exist_ok=True)
    caminho_log = os.path.join(pasta_logs, f"dublagem-{job.id}.log")
    job.detalhes["log"] = caminho_log
    job.etapa = "Começando..."

    with open(caminho_log, "a", encoding="utf-8") as log:
        processo = subprocess.Popen(comando, cwd=RAIZ_REPO, stdout=subprocess.PIPE,
                                    stderr=log, text=True, bufsize=1,
                                    start_new_session=True)

        def vigiar_cancelamento():
            while processo.poll() is None:
                if job.cancelar.wait(1):
                    try:
                        os.killpg(processo.pid, signal.SIGTERM)
                        time.sleep(10)
                        if processo.poll() is None:
                            os.killpg(processo.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return
        threading.Thread(target=vigiar_cancelamento, daemon=True).start()

        erro = None
        for linha in processo.stdout:
            linha = linha.strip()
            if not linha.startswith("{"):
                continue
            try:
                evento = json.loads(linha)
            except ValueError:
                continue
            if "etapa" in evento:
                job.etapa = evento["etapa"]
            if "progresso" in evento:
                job.progresso = int(evento["progresso"])
            if "fase" in evento:
                job.detalhes["fase"] = evento["fase"]
            if "saida" in evento:
                job.detalhes["saida"] = evento["saida"]
            if "trabalho" in evento:
                job.detalhes["trabalho"] = evento["trabalho"]
            if "erro" in evento:
                erro = evento["erro"]
            if "fim" in evento:
                job.detalhes["relatorio"] = evento["fim"]
                job.resultado = evento["fim"].get("saida")
        processo.wait()

    if job.cancelar.is_set():
        raise RuntimeError("Cancelado.")
    if processo.returncode != 0:
        raise RuntimeError(erro or f"A dublagem parou com o código {processo.returncode} "
                                   f"(detalhes em {caminho_log}).")
    _esquecer(job.entrada)


def _ao_terminar(job):
    # Terminou de qualquer jeito (pronto, erro ou cancelado): não volta
    # sozinho no próximo início do serviço -- erro repetiria em loop.
    _esquecer(job.entrada)
    try:
        historico.registrar(job)
    except Exception:  # noqa: BLE001
        logger.exception("histórico da dublagem")


# --------------------------------------------------------------------------
# Registro das rotas
# --------------------------------------------------------------------------

def registrar(app, fila_legendas, media_dir, auto_enfileirar_legenda):
    fila = JobQueue(_executar_dublagem, max_workers=1, on_final=_ao_terminar)
    app.config["FILA_DUBLAGEM"] = fila
    # Duas pastas, de propósito:
    #  - media_dir: área temporária DENTRO do container, onde cai o que é
    #    enviado pelo navegador (mesma regra da produção -- o arquivo fica
    #    guardado ali esperando você escolher o que fazer com ele);
    #  - acervo_dir: o HD de 14 TB, só de leitura, de onde se escolhe filme
    #    que já está no acervo. Nada é gravado nele por esta tela.
    acervo_dir = os.path.abspath(os.environ.get("AUTOSRT_ACERVO_DIR", ACERVO_PADRAO))
    app.config["ACERVO_DIR"] = acervo_dir
    indice = IndiceDoAcervo(acervo_dir)
    app.config["INDICE_ACERVO"] = indice
    if os.environ.get("AUTOSRT_INDICE") != "0":
        indice.em_segundo_plano()

    def enfileirar(caminho, **extras):
        if fila.em_uso(caminho):
            return None, "Esse arquivo já está na fila de dublagem."
        _lembrar(caminho, extras)
        job = fila.enviar(os.path.basename(caminho), caminho, **extras)
        return job, None

    # Dublagem que estava na fila quando o serviço parou volta sozinha e
    # continua de onde parou (cada etapa já pronta é pulada).
    for pendente in _pendentes():
        if os.path.isfile(pendente["caminho"]):
            extras = {k: v for k, v in pendente.items() if k != "caminho"}
            fila.enviar(os.path.basename(pendente["caminho"]), pendente["caminho"], **extras)

    def caminho_de(nome, fonte="envio", exigir_video=True):
        """Resolve um nome vindo da tela, na área temporária ou no acervo."""
        raiz = acervo_dir if fonte == "acervo" else media_dir
        if not nome:
            return None, ("Arquivo inválido.", 400)
        caminho = os.path.abspath(os.path.join(raiz, nome))
        if not caminho.startswith(os.path.abspath(raiz) + os.sep):
            return None, ("Arquivo inválido.", 400)
        if not os.path.isfile(caminho):
            return None, ("Esse arquivo não está mais lá.", 404)
        if exigir_video and (not pipeline.is_media(caminho)
                             or os.path.splitext(caminho)[1].lower() in {
                                 ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg",
                                 ".opus", ".wma"}):
            return None, ("Só dá para dublar vídeo.", 400)
        return caminho, None

    @app.post("/api/dublar")
    def dublar():
        """Põe filmes na fila de dublagem, venham da área temporária
        (``fonte: envio``) ou do acervo em HD (``fonte: acervo``)."""
        dados = request.json or {}
        fonte = dados.get("fonte") or "envio"
        nomes = dados.get("arquivos") or [dados.get("arquivo")]
        enfileirados, recusados = [], []
        for nome in nomes:
            caminho, problema = caminho_de(nome, fonte)
            if problema:
                recusados.append({"arquivo": nome, "erro": problema[0]})
                continue
            extras = {k: dados[k] for k in ("inicio", "fim") if dados.get(k) is not None}
            job, erro = enfileirar(caminho, **extras)
            if erro:
                recusados.append({"arquivo": nome, "erro": erro})
            else:
                enfileirados.append(job.para_json())
        status = 202 if enfileirados else 400
        resposta = {"enfileirados": enfileirados, "recusados": recusados}
        if not enfileirados and recusados:
            resposta["erro"] = recusados[0]["erro"]
        return jsonify(resposta), status

    @app.post("/api/legendar-acervo")
    def legendar_acervo():
        """Gera legenda para um filme que está no acervo em HD.

        A rota ``/api/processar`` da produção só aceita arquivo da área
        temporária; esta faz o mesmo para o acervo, usando a mesma fila de
        legenda, então um trabalho só roda por vez como sempre."""
        dados = request.json or {}
        nomes = dados.get("arquivos") or [dados.get("arquivo")]
        enfileirados, recusados = [], []
        for nome in nomes:
            caminho, problema = caminho_de(nome, "acervo", exigir_video=False)
            if problema:
                recusados.append({"arquivo": nome, "erro": problema[0]})
                continue
            if fila_legendas.em_uso(caminho):
                recusados.append({"arquivo": nome, "erro": "Já está na fila de legenda."})
                continue
            fila_legendas.max_workers = config.get_max_workers()
            job = fila_legendas.enviar(os.path.basename(nome), caminho,
                                       acao=dados.get("acao") or "completo", diarizar=True)
            enfileirados.append(job.para_json())
        status = 202 if enfileirados else 400
        resposta = {"enfileirados": enfileirados, "recusados": recusados}
        if not enfileirados and recusados:
            resposta["erro"] = recusados[0]["erro"]
        return jsonify(resposta), status

    @app.get("/api/dublagens")
    def dublagens():
        return jsonify([j.para_json() for j in fila.listar(limite=30)])

    @app.post("/api/dublagem/<job_id>/cancelar")
    def cancelar_dublagem(job_id):
        job = fila.buscar(job_id)
        if job:
            _esquecer(job.entrada)
        return jsonify({"cancelado": fila.cancelar(job_id)})

    @app.delete("/api/dublagem/<job_id>")
    def dispensar_dublagem(job_id):
        return jsonify({"ok": fila.remover(job_id)})

    @app.get("/api/dublagem/<job_id>/log")
    def log_da_dublagem(job_id):
        job = fila.buscar(job_id)
        caminho = job and job.detalhes.get("log")
        if not caminho or not os.path.exists(caminho):
            return jsonify({"erro": "Sem registro."}), 404
        with open(caminho, encoding="utf-8", errors="replace") as f:
            linhas = collections.deque(f, maxlen=300)
        return Response("".join(linhas), mimetype="text/plain; charset=utf-8")

    @app.get("/api/acervo")
    def acervo():
        """Navega ou busca no acervo, a partir do índice em memória.

        O TMDB (pôster, sinopse) só é consultado para os itens desta página
        -- item a item no acervo inteiro nunca terminaria."""
        from .dublagem import caminho_de_saida
        pasta = request.args.get("pasta", "")
        busca = request.args.get("q", "").strip()
        try:
            pagina = max(0, int(request.args.get("pagina", 0)))
        except ValueError:
            pagina = 0
        subpastas, itens, total = indice.listar(pasta, busca, pagina)

        chave_tmdb = config.get_tmdb_api_key()
        saida_dublagem = _opcao("dublagem_pasta_saida")

        # Em paralelo: são até 60 consultas de rede por página e, em
        # sequência, uma página demorava meio minuto na primeira vez (depois
        # o cache por nome de arquivo resolve). O resultado vai para o mesmo
        # cache que lookup_cached lê em seguida.
        fichas = {}
        if chave_tmdb:
            from concurrent.futures import ThreadPoolExecutor
            caminhos = [os.path.join(acervo_dir, i["nome"]) for i in itens]
            with ThreadPoolExecutor(max_workers=8) as executor:
                for caminho, ficha in zip(caminhos, executor.map(
                        lambda c: tmdb.lookup_cached(c, chave_tmdb), caminhos)):
                    fichas[caminho] = ficha

        resposta = []
        for item in itens:
            caminho = os.path.join(acervo_dir, item["nome"])
            legenda = None
            for ext in (".srt", ".pt.srt", ".ass"):
                irma = os.path.splitext(caminho)[0] + ext
                if os.path.exists(irma):
                    legenda = os.path.relpath(irma, acervo_dir).replace(os.sep, "/")
                    break
            resposta.append({
                "nome": item["nome"],
                "pasta": item["pasta"],
                "tamanho": _tamanho_legivel(item["bytes"]),
                "tipo": "video",
                "tem_legenda": bool(legenda),
                "dublado": os.path.exists(caminho_de_saida(caminho, saida_dublagem)),
                "na_fila": fila.em_uso(caminho) or fila_legendas.em_uso(caminho),
                "filme": fichas.get(caminho),
            })
        return jsonify({
            "pasta": pasta,
            "pastas": subpastas,
            "itens": resposta,
            "total": total,
            "pagina": pagina,
            "por_pagina": ITENS_POR_PAGINA,
            "indice_pronto": indice.pronto,
            "indexando": indice.varrendo,
            "indexado_em": indice.quando,
        })

    @app.post("/api/acervo/reindexar")
    def reindexar():
        threading.Thread(target=indice.atualizar, daemon=True).start()
        return jsonify({"ok": True})

    # ---------------- configuração da dublagem ----------------

    @app.get("/api/config/dublagem")
    def ler_config_dublagem():
        return jsonify({nome: _opcao(nome) for nome in OPCOES_DUBLAGEM})

    @app.post("/api/config/dublagem")
    def gravar_config_dublagem():
        dados = request.json or {}
        novos = {}
        for nome, valor in dados.items():
            if nome not in OPCOES_DUBLAGEM:
                return jsonify({"erro": f"Opção desconhecida: {nome}"}), 400
            valor = str(valor if valor is not None else "").strip()
            if nome in ("dublagem_temperatura", "dublagem_max_acelerar",
                        "dublagem_max_desacelerar_video", "dublagem_crf") and valor:
                try:
                    numero = float(valor.replace(",", ".").rstrip("xX "))
                except ValueError:
                    return jsonify({"erro": f"Valor inválido em {nome}: {valor}"}), 400
                limites = {"dublagem_temperatura": (0.1, 1.0), "dublagem_max_acelerar": (1.0, 2.0),
                           "dublagem_max_desacelerar_video": (1.0, 8.0), "dublagem_crf": (12, 30)}
                baixo, alto = limites[nome]
                if not baixo <= numero <= alto:
                    return jsonify({"erro": f"{nome} precisa ficar entre {baixo} e {alto}."}), 400
                valor = str(numero)
            if nome == "dublagem_motor" and valor not in ("local",):
                return jsonify({"erro": "Por enquanto só o XTTS local está disponível."}), 400
            novos[nome] = valor
        if not novos:
            return jsonify({"erro": "Nada para gravar."}), 400
        atual = config.load_config()
        atual.update(novos)
        config.save_config(atual)
        return jsonify({"ok": True})

    # ---------------- atores e locutores ----------------

    @app.get("/api/atores")
    def atores():
        return jsonify(vozes.listar_atores_completo())

    @app.get("/api/atores/<ator_id>/previa")
    def previa_do_ator(ator_id):
        pasta = os.path.join(config.app_directory(), "vozes", "previas")
        os.makedirs(pasta, exist_ok=True)
        destino = os.path.join(pasta, f"{ator_id}.wav")
        try:
            vozes.montar_referencia_combinada(ator_id, destino, max_segundos=30)
        except ValueError:
            return jsonify({"erro": "Esse ator não tem voz guardada."}), 404
        return send_file(destino, mimetype="audio/wav")

    @app.get("/api/atores/<ator_id>/teste_voz")
    def teste_voz_do_ator(ator_id):
        """Sintetiza a frase de teste em português com a voz clonada do ator,
        em vez de só reproduzir o trecho de referência original (que pode
        estar no idioma original do filme, não em português)."""
        pasta_ref = os.path.join(config.app_directory(), "vozes", "previas")
        pasta_teste = os.path.join(config.app_directory(), "vozes", "testes")
        os.makedirs(pasta_ref, exist_ok=True)
        os.makedirs(pasta_teste, exist_ok=True)
        referencia = os.path.join(pasta_ref, f"{ator_id}.wav")
        destino = os.path.join(pasta_teste, f"{ator_id}.wav")
        try:
            vozes.montar_referencia_combinada(ator_id, referencia, max_segundos=30)
        except ValueError:
            return jsonify({"erro": "Esse ator não tem voz guardada."}), 404

        python = os.environ.get("AUTOSRT_DUB_PYTHON", PYTHON_DUBLAGEM_PADRAO)
        comando = [python, "-m", "autosrt.dublagem",
                  "--teste-voz-referencia", referencia, "--saida", destino]
        try:
            resultado = subprocess.run(comando, cwd=RAIZ_REPO, capture_output=True,
                                       text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return jsonify({"erro": "Demorou demais pra sintetizar o teste. Tenta de novo."}), 504
        if resultado.returncode != 0 or not os.path.exists(destino):
            erro = (resultado.stdout or "").strip().splitlines()
            mensagem = erro[-1] if erro else "Não consegui sintetizar o teste de voz."
            try:
                mensagem = json.loads(mensagem).get("erro", mensagem)
            except (ValueError, AttributeError):
                pass
            return jsonify({"erro": mensagem}), 502
        return send_file(destino, mimetype="audio/wav")

    @app.get("/api/locutores")
    def locutores():
        pendentes = request.args.get("todos") != "1"
        return jsonify(vozes.listar_locutores(pendentes=pendentes))

    @app.get("/api/locutor/<int:locutor_id>/audio")
    def audio_do_locutor(locutor_id):
        loc = vozes.locutor(locutor_id)
        if not loc or not os.path.exists(loc["caminho_audio"]):
            return jsonify({"erro": "Áudio não encontrado."}), 404
        return send_file(loc["caminho_audio"], mimetype="audio/wav")

    @app.post("/api/locutor/<int:locutor_id>/atribuir")
    def atribuir(locutor_id):
        dados = request.json or {}
        if dados.get("ignorar"):
            vozes.atribuir_locutor(locutor_id, vozes.IGNORADO)
            return jsonify({"ok": True})
        chave = config.get_tmdb_api_key()
        ficha = None
        if dados.get("tmdb_id"):
            ficha = tmdb.pessoa(dados["tmdb_id"], chave)
        elif dados.get("nome"):
            ficha = tmdb.buscar_pessoa(dados["nome"].strip(), chave)
        nome = (ficha or {}).get("nome") or (dados.get("nome") or "").strip()
        if not nome:
            return jsonify({"erro": "Diga quem é o ator."}), 400
        ator_id = vozes.id_de_ator(nome, (ficha or {}).get("tmdb_id"))
        extras = {k: (ficha or {}).get(k) for k in
                  ("tmdb_id", "foto", "bio", "nascimento", "local_nascimento")}
        try:
            vozes.atribuir_locutor(locutor_id, ator_id, nome, **extras)
        except ValueError as exc:
            return jsonify({"erro": str(exc)}), 404
        return jsonify({"ok": True, "ator_id": ator_id, "nome": nome})

    @app.post("/api/locutor/<int:locutor_id>/desatribuir")
    def desatribuir(locutor_id):
        vozes.desatribuir_locutor(locutor_id)
        return jsonify({"ok": True})

    # ---------------- ferramentas do HD (repassadas para a produção) ----------------

    proxy = os.environ.get("AUTOSRT_FILMES_PROXY", "").rstrip("/")
    if proxy:
        import requests

        @app.route("/api/filmes/<path:resto>", methods=["GET", "POST"])
        def filmes_na_producao(resto):
            try:
                resposta = requests.request(
                    request.method, f"{proxy}/api/filmes/{resto}",
                    json=request.get_json(silent=True) if request.method == "POST" else None,
                    params=request.args, timeout=30)
            except requests.RequestException as exc:
                return jsonify({"erro": f"A produção (porta 8000) não respondeu: {exc}"}), 502
            return Response(resposta.content, status=resposta.status_code,
                            mimetype=resposta.headers.get("Content-Type", "application/json"))

    # ---------------- automação ----------------

    def auto_dublar(guardados):
        if not dublagem_automatica():
            return []
        enfileirados = []
        for nome in guardados:
            caminho, problema = caminho_de(nome)
            if problema:
                continue
            job, erro = enfileirar(caminho)
            if job:
                enfileirados.append(job.para_json())
        return enfileirados

    app.config["AUTO_DUBLAR"] = auto_dublar

    @app.get("/api/automacao")
    def automacao():
        return jsonify({
            "legenda_automatica": config.get_auto_processar(),
            "dublagem_automatica": dublagem_automatica(),
            "fila_legendas": sum(1 for j in fila_legendas.listar(50) if j.estado not in FINAIS),
            "fila_dublagem": sum(1 for j in fila.listar(50) if j.estado not in FINAIS),
        })

    @app.post("/api/automacao")
    def gravar_automacao():
        dados = request.json or {}
        atual = config.load_config()
        if "dublagem_automatica" in dados:
            atual["dublagem_auto"] = "true" if dados["dublagem_automatica"] else "false"
        if "legenda_automatica" in dados:
            atual["auto_processar"] = "true" if dados["legenda_automatica"] else "false"
        config.save_config(atual)
        return jsonify({"ok": True})

    return fila
