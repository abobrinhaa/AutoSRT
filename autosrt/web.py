"""Interface web do AutoSRT.

Pensada para uso em rede local por quem não abre terminal: escolher o
arquivo, apertar um botão, baixar a legenda.

Duas formas de entrada, pelo motivo prático de tamanho. Filme tem vários
gigabytes e enviar isso pelo navegador é lento e pesado, então vídeo é
escolhido de uma pasta do servidor — o arquivo chega lá por cópia de rede.
Legenda tem alguns quilobytes e pode ser enviada direto pela página.
"""

import io
import logging
import os
import tempfile

from flask import (Flask, jsonify, request, send_file)

from . import config, llm, pipeline, srt_io, sync, tmdb, transcribe
from .jobs import JobQueue
from .transcribe import TranscriptionError

DEFAULT_MEDIA_DIR = "midia"
DEFAULT_PORT = 8000

#: Filtro do seletor de arquivos do sistema. Precisa refletir exatamente o
#: que o servidor aceita — um teste compara as duas listas, porque um filtro
#: desatualizado esconde arquivo válido sem dar nenhuma pista ao usuário.
EXTENSOES_ACEITAS = sorted(pipeline.MEDIA_EXTENSIONS | pipeline.SUBTITLE_EXTENSIONS)
# Em rede local, enviar um filme pelo navegador é questão de um ou dois
# minutos - irrelevante perto da meia hora que a transcrição leva depois.
# O limite existe só para não encher o disco por acidente.
DEFAULT_MAX_UPLOAD_GB = 8

# /api/transcribe é síncrono de propósito (ver docstring da rota): quem
# chama espera a legenda na mesma requisição, sem passar pela fila. Isso
# prende uma worker thread do Flask até o Whisper terminar; o limite existe
# para essa requisição eventualmente desistir em vez de travar para sempre
# se o processo do Whisper pendurar.
DEFAULT_TRANSCRIBE_API_TIMEOUT = 1800  # 30 minutos


def create_app(media_dir=None, engine=pipeline.ENGINE_LLM, max_upload_gb=None,
               transcribe_api_timeout=None):
    app = Flask(__name__)

    if max_upload_gb is None:
        max_upload_gb = float(os.environ.get("AUTOSRT_MAX_UPLOAD_GB",
                                             DEFAULT_MAX_UPLOAD_GB))
    app.config["MAX_CONTENT_LENGTH"] = int(max_upload_gb * 1024 ** 3)
    app.config["MAX_UPLOAD_GB"] = max_upload_gb

    if transcribe_api_timeout is None:
        transcribe_api_timeout = float(
            os.environ.get("AUTOSRT_TRANSCRIBE_API_TIMEOUT",
                           DEFAULT_TRANSCRIBE_API_TIMEOUT))
    app.config["TRANSCRIBE_API_TIMEOUT"] = transcribe_api_timeout

    media_dir = os.path.abspath(
        media_dir or os.environ.get("AUTOSRT_MEDIA_DIR", DEFAULT_MEDIA_DIR))
    os.makedirs(media_dir, exist_ok=True)
    app.config["MEDIA_DIR"] = media_dir

    fila = JobQueue(lambda job: _executar(job, engine))
    app.config["FILA"] = fila

    _register_routes(app, fila, media_dir, engine)
    return app


#: Cada operação existe sozinha. Nem sempre se quer a corrente inteira:
#: às vezes é só traduzir uma legenda que já se tem, às vezes só transcrever
#: para conferir o que o Whisper entendeu, às vezes só acertar o tempo.
ACOES_MIDIA = [
    ("completo", "Transcrever e traduzir"),
    ("transcrever", "Só transcrever"),
]
ACOES_LEGENDA = [
    ("traduzir", "Traduzir"),
    ("deslocar", "Ajustar o tempo..."),
]
ACAO_CONVERTER = ("converter", "Só converter para .srt")
ACAO_LEGENDA_EXISTENTE = ("traduzir_existente", "Usar a legenda existente")

#: Extensões de legenda procuradas ao lado de um vídeo, em ordem de
#: preferência: SRT primeiro, porque dispensa conversão.
EXTENSOES_IRMAS = (".srt", ".ssa", ".ass")


def legenda_irma(caminho_video):
    """Legenda de mesmo nome, na mesma pasta do vídeo, se houver.

    Existindo, traduzi-la custa um minuto contra a meia hora de GPU que a
    transcrição levaria para produzir o que já está ali.
    """
    raiz = os.path.splitext(caminho_video)[0]
    for extensao in EXTENSOES_IRMAS:
        candidato = raiz + extensao
        if os.path.isfile(candidato):
            return candidato
    return None


def filme_irma(caminho_legenda):
    """Vídeo de mesmo nome, na mesma pasta da legenda, se houver.

    Se existir, faz sentido oferecer "deslocar" para sincronizar a legenda
    com o filme. Sem o filme irmão, deslocar é improdutivo.
    """
    raiz = os.path.splitext(caminho_legenda)[0]
    for extensao in pipeline.MEDIA_EXTENSIONS:
        candidato = raiz + extensao
        if os.path.isfile(candidato):
            return candidato
    return None


def acoes_para(caminho) -> list:
    """Operações aplicáveis a um arquivo, na ordem em que fazem sentido."""
    if pipeline.is_media(caminho):
        acoes = list(ACOES_MIDIA)
        # Legenda pronta ao lado vira a primeira opção, e portanto o padrão:
        # transcrever de novo o que já existe é o desperdício mais caro que
        # este programa consegue cometer.
        if legenda_irma(caminho):
            acoes.insert(0, ACAO_LEGENDA_EXISTENTE)
        return [{"id": i, "rotulo": r} for i, r in acoes]

    acoes = [("traduzir", "Traduzir")]
    if os.path.splitext(caminho)[1].lower() in {".ssa", ".ass"}:
        acoes.append(("converter", "Só converter para .srt"))
    # Ajustar o tempo é só somar segundos, e nisso o vídeo não entra. Mas
    # descobrir quantos segundos são se faz assistindo: sem o filme na pasta
    # a opção só rende legenda torta de outro jeito.
    if filme_irma(caminho):
        acoes.append(("deslocar", "Ajustar o tempo..."))
    return [{"id": i, "rotulo": r} for i, r in acoes]


def _executar(job, engine):
    """Roda um trabalho. Chamado pelo operário da fila."""
    def status(mensagem):
        job.etapa = mensagem

    def progresso(feitas, total):
        if total:
            job.progresso = min(100, int(feitas * 100 / total))

    acao = job.detalhes.get("acao") or "completo"

    if acao == "deslocar":
        return _deslocar(job, status)
    if acao == "converter":
        return _converter(job, status)
    if acao == "traduzir_existente":
        return _traduzir_existente(job, engine, status, progresso)

    if pipeline.is_media(job.entrada):
        resultado = pipeline.process_media(
            job.entrada, engine=engine, status=status, progress=progresso,
            cancel_event=job.cancelar, translate=(acao != "transcrever"),
            language=config.get_whisper_language(),
            diarize=config.get_diarize(),
            normalize_audio=config.get_normalize_audio(),
            vad_method=config.get_vad_method(),
            vad_threshold=config.get_vad_threshold(),
            vad_min_silence_ms=config.get_vad_min_silence_ms(),
            whisper_model=config.get_whisper_model(),
            whisper_compute_type=config.get_whisper_compute_type())
        job.resultado = os.path.splitext(job.entrada)[0] + ".srt"
    else:
        saida = srt_io.srt_output_path(job.entrada)
        resultado = pipeline.translate_file(
            job.entrada, saida, engine=engine, status=status,
            progress=progresso, cancel_event=job.cancelar)
        job.resultado = saida

    job.detalhes.update({
        "total": resultado.total,
        "traduzidas": resultado.translated,
        "falhas": resultado.failure_count,
        "idioma": resultado.language_label,
    })


def _traduzir_existente(job, engine, status, progresso):
    """Traduz a legenda que já acompanha o vídeo, sem transcrever nada."""
    legenda = legenda_irma(job.entrada)
    if not legenda:
        raise ValueError(
            "A legenda que estava ao lado do vídeo não está mais lá. "
            "Escolha 'Transcrever e traduzir'.")

    status(f"Usando {os.path.basename(legenda)}, sem transcrever.")
    saida = srt_io.srt_output_path(legenda)
    resultado = pipeline.translate_file(
        legenda, saida, engine=engine, status=status, progress=progresso,
        cancel_event=job.cancelar)

    job.resultado = saida
    job.detalhes.update({
        "total": resultado.total,
        "traduzidas": resultado.translated,
        "falhas": resultado.failure_count,
        "idioma": resultado.language_label,
    })


def _deslocar(job, status):
    """Move todos os tempos da legenda, sem traduzir nada."""
    segundos = float(job.detalhes.get("segundos") or 0)
    status(f"Deslocando em {segundos:+.3f}s...")

    cues = srt_io.load_cues(job.entrada)
    sync.shift_cues(cues, round(segundos * 1000))
    saida = srt_io.srt_output_path(job.entrada)
    srt_io.save_cues(cues, saida)

    job.resultado = saida
    job.detalhes.update({"total": len(cues)})


def _converter(job, status):
    """Converte SSA/ASS para SRT, sem traduzir nada."""
    status("Convertendo para .srt...")
    saida = srt_io.convert_to_srt(job.entrada)
    job.resultado = saida
    job.detalhes.update({"total": len(srt_io.load_cues(saida))})


def _dentro_da_pasta(caminho, pasta) -> bool:
    """Barra caminhos que escapam da pasta de trabalho."""
    caminho = os.path.abspath(caminho)
    pasta = os.path.abspath(pasta)
    return caminho == pasta or caminho.startswith(pasta + os.sep)


def _listar_arquivos(media_dir) -> list:
    # Buscada uma vez por listagem, não por arquivo: é rápida (variável de
    # ambiente ou um `.json` local) e evita reabrir o config.json centenas
    # de vezes numa pasta com muitos filmes.
    chave_tmdb = config.get_tmdb_api_key()

    itens = []
    for raiz, pastas, nomes in os.walk(media_dir):
        # As transcrições no idioma falado não são material de entrada.
        pastas[:] = [p for p in pastas if p != pipeline.ORIGINALS_DIRNAME]
        for nome in sorted(nomes):
            caminho = os.path.join(raiz, nome)
            if not (pipeline.is_media(caminho) or pipeline.is_subtitle(caminho)):
                continue
            minusculo = nome.lower()
            if minusculo.endswith("_backup.srt") or minusculo.endswith(".original.srt"):
                continue
            relativo = os.path.relpath(caminho, media_dir)
            eh_midia = pipeline.is_media(caminho)
            irma = legenda_irma(caminho) if eh_midia else None
            itens.append({
                "nome": relativo,
                "pasta": os.path.dirname(relativo) or ".",
                "tamanho": _tamanho_legivel(os.path.getsize(caminho)),
                "tipo": "video" if eh_midia else "legenda",
                "tem_legenda": bool(irma),
                "acoes": acoes_para(caminho),
                # Reconhecimento por nome de arquivo é palpite, não certeza:
                # None sem chave configurada, ou quando o TMDB não achou
                # nada parecido o bastante para arriscar um selo errado.
                "filme": tmdb.lookup_cached(caminho, chave_tmdb)
                        if eh_midia and chave_tmdb else None,
            })
    return sorted(itens, key=lambda i: i["nome"])


def _tamanho_legivel(bytes_) -> str:
    for unidade in ("B", "KB", "MB", "GB"):
        if bytes_ < 1024 or unidade == "GB":
            return f"{bytes_:.0f} {unidade}" if unidade in ("B", "KB") \
                else f"{bytes_:.1f} {unidade}"
        bytes_ /= 1024
    return f"{bytes_:.1f} GB"


def _register_routes(app, fila, media_dir, engine):

    @app.get("/")
    def index():
        # O filtro do seletor é montado a partir do que o servidor aceita,
        # em vez de escrito à mão na página, para não sair de sincronia.
        return PAGINA.replace("{{ACCEPT}}", ",".join(EXTENSOES_ACEITAS))

    @app.get("/api/arquivos")
    def arquivos():
        return jsonify(_listar_arquivos(media_dir))

    @app.get("/api/legenda/<path:nome>")
    def baixar_legenda(nome):
        """Baixa uma legenda que já está na pasta, sem reprocessar nada.

        Só legendas: o resto da pasta é filme, que ninguém quer puxar pela
        página e cujo envio prenderia a fila por minutos.
        """
        caminho = os.path.join(media_dir, nome)
        if not _dentro_da_pasta(caminho, media_dir):
            return jsonify({"erro": "Arquivo inválido."}), 400
        if not os.path.isfile(caminho):
            return jsonify({"erro": "Esse arquivo não está mais na pasta."}), 404
        if not pipeline.is_subtitle(caminho):
            return jsonify({"erro": "Só dá para baixar legendas."}), 400
        return send_file(caminho, as_attachment=True,
                         download_name=os.path.basename(caminho))

    def _enfileirar(pedido):
        """Valida um pedido e o coloca na fila. Devolve ``(job, erro, status)``."""
        nome = (pedido or {}).get("arquivo", "")
        caminho = os.path.join(media_dir, nome)
        if not nome or not _dentro_da_pasta(caminho, media_dir):
            return None, "Arquivo inválido.", 400
        if not os.path.isfile(caminho):
            return None, "Esse arquivo não está mais na pasta.", 404

        acao = pedido.get("acao") or "completo"
        validas = {a["id"] for a in acoes_para(caminho)} | {"completo"}
        if acao not in validas:
            return None, f"A ação '{acao}' não vale para este arquivo.", 400

        job = fila.enviar(os.path.basename(nome), caminho, acao=acao,
                          segundos=pedido.get("segundos"))
        return job, None, 202

    @app.post("/api/processar")
    def processar():
        job, erro, status = _enfileirar(request.json)
        if erro:
            return jsonify({"erro": erro}), status
        return jsonify(job.para_json()), 202

    @app.post("/api/processar-lote")
    def processar_lote():
        pedidos = (request.json or {}).get("itens") or []
        if not pedidos:
            return jsonify({"erro": "Nenhum arquivo selecionado."}), 400

        enfileirados, recusados = [], []
        for pedido in pedidos:
            job, erro, _ = _enfileirar(pedido)
            if job:
                enfileirados.append(job.para_json())
            else:
                recusados.append({"arquivo": pedido.get("arquivo"), "erro": erro})

        return jsonify({"enfileirados": enfileirados,
                        "recusados": recusados}), 202

    @app.post("/api/enviar")
    def enviar():
        """Recebe um ou mais arquivos e os guarda, sem processar.

        Guardar e processar são passos separados de propósito. Quem envia o
        filme junto com a legenda espera que os dois sejam considerados em
        conjunto; enfileirar cada arquivo assim que chega faria a legenda ser
        traduzida sozinha antes de o vídeo terminar de subir, desperdiçando
        justamente o pareamento.
        """
        arquivos = request.files.getlist("arquivo")
        arquivos = [a for a in arquivos if a and a.filename]
        if not arquivos:
            return jsonify({"erro": "Nenhum arquivo recebido."}), 400

        guardados, recusados = [], []
        for arquivo in arquivos:
            nome = os.path.basename(arquivo.filename)
            if not (pipeline.is_subtitle(nome) or pipeline.is_media(nome)):
                recusados.append({
                    "arquivo": nome,
                    "erro": f"Não sei o que fazer com {nome}. Envie um vídeo, "
                            "um áudio, ou uma legenda (.srt, .ssa, .ass).",
                })
                continue
            arquivo.save(os.path.join(media_dir, nome))
            guardados.append(nome)

        if not guardados:
            return jsonify({"erro": recusados[0]["erro"],
                            "recusados": recusados}), 400

        return jsonify({"guardados": guardados, "recusados": recusados}), 201

    @app.get("/api/config")
    def ler_config():
        """Estado da configuração. As chaves nunca voltam para o navegador."""
        chave = config.get_openrouter_api_key()
        base_url = config.get_setting("llm_base_url", "LLM_BASE_URL") or llm.DEFAULT_BASE_URL
        modelo_configurado = config.get_setting("llm_model", "LLM_MODEL")
        # DEFAULT_MODEL só é um padrão razoável para o próprio OpenRouter;
        # mostrá-lo como se fosse "o modelo configurado" para um endereço
        # diferente (ex.: servidor local) é enganoso -- ninguém digitou
        # isso, e um servidor local quase certamente não tem esse modelo.
        modelo = modelo_configurado or (
            llm.DEFAULT_MODEL if base_url == llm.DEFAULT_BASE_URL else "")
        return jsonify({
            "tem_chave": bool(chave),
            "chave_do_ambiente": bool(os.environ.get("OPENROUTER_API_KEY")),
            "modelo": modelo,
            "base_url": base_url,
            "tem_chave_tmdb": bool(config.get_tmdb_api_key()),
            # Endereços padrão dos dois modos, para o botão de alternar na
            # página não precisar adivinhar nem duplicar essas constantes.
            "openrouter_base_url": llm.DEFAULT_BASE_URL,
            "local_base_url": llm.LOCAL_BASE_URL,
            # None quando não configurado -- "não mexi nisso, decide
            # sozinho pelo endereço" (ver llm_translate.translate_cues_llm).
            "llm_block_size": config.get_llm_block_size(),
            # Vazio quando não configurado -- é "não mexi nisso, usa o
            # padrão do transcribe.py" (turbo / auto).
            "whisper_model": config.get_whisper_model(),
            "whisper_compute_type": config.get_whisper_compute_type(),
            "whisper_language": config.get_whisper_language(),
            "diarizar": config.get_diarize(),
            "normalize_audio": config.get_normalize_audio(),
            "vad_method": config.get_vad_method(),
            "vad_method_padrao": transcribe.DEFAULT_VAD,
            # Vazio quando não configurado -- não é "0", é "não mexi nisso,
            # deixa o Whisper no próprio padrão dele".
            "vad_threshold": config.get_vad_threshold(),
            "vad_min_silence_ms": config.get_vad_min_silence_ms(),
        })

    @app.post("/api/config")
    def gravar_config():
        dados = request.json or {}
        novos = {}

        if "chave" in dados:
            novos["openrouter_api_key"] = (dados.get("chave") or "").strip()
        if "chave_tmdb" in dados:
            novos["tmdb_api_key"] = (dados.get("chave_tmdb") or "").strip()
        for campo, chave_config in (("modelo", "llm_model"),
                                    ("base_url", "llm_base_url")):
            if campo in dados:
                novos[chave_config] = (dados.get(campo) or "").strip()

        if "llm_block_size" in dados:
            valor = str(dados.get("llm_block_size") or "").strip()
            if valor:
                try:
                    if int(valor) < 1:
                        raise ValueError
                except ValueError:
                    return jsonify({
                        "erro": "Legendas por requisição precisa ser um "
                                "número inteiro positivo (ex: 2)."}), 400
            novos["llm_block_size"] = valor

        # Texto livre de propósito: a lista de modelos e de compute types de
        # quem transcreve muda com a versão do Faster-Whisper, e uma lista
        # fixa aqui viraria mentira na próxima atualização dele.
        if "diarizar" in dados:
            # Checkbox manda booleano; o config.json guarda texto, que é o
            # que o get_setting sabe ler de volta (e do ambiente também).
            novos["diarizar"] = "sim" if dados.get("diarizar") else "nao"

        if "normalize_audio" in dados:
            valor = str(dados.get("normalize_audio") or "").strip().lower()
            if valor and valor not in ("auto", "sempre", "nunca"):
                return jsonify({
                    "erro": "Normalização do áudio precisa ser auto, sempre "
                            "ou nunca."}), 400
            novos["normalize_audio"] = valor

        for campo in ("whisper_model", "whisper_compute_type", "vad_method",
                      "whisper_language"):
            if campo in dados:
                novos[campo] = str(dados.get(campo) or "").strip()

        if "vad_threshold" in dados:
            valor = str(dados.get("vad_threshold") or "").strip()
            if valor:
                try:
                    float(valor)
                except ValueError:
                    return jsonify({
                        "erro": "Sensibilidade da VAD precisa ser um número "
                                "entre 0 e 1 (ex: 0.2)."}), 400
            novos["vad_threshold"] = valor

        if "vad_min_silence_ms" in dados:
            valor = str(dados.get("vad_min_silence_ms") or "").strip()
            if valor:
                try:
                    int(valor)
                except ValueError:
                    return jsonify({
                        "erro": "Silêncio mínimo precisa ser um número "
                                "inteiro de milissegundos (ex: 300)."}), 400
            novos["vad_min_silence_ms"] = valor

        if not novos:
            return jsonify({"erro": "Nada para gravar."}), 400

        try:
            config.save_config(novos)
        except OSError as exc:
            return jsonify({"erro": f"Não consegui gravar a configuração: {exc}"}), 500

        # A chave nova muda o que o TMDB reconheceria; o cache antigo (vazio,
        # de quando não havia chave) não vale mais.
        if "tmdb_api_key" in novos:
            tmdb.limpar_cache()

        return jsonify({"ok": True, "aviso": _aviso_de_ambiente(novos)})

    def _aviso_de_ambiente(novos):
        """Variável de ambiente vence o arquivo; avisar evita confusão."""
        if novos.get("openrouter_api_key") and os.environ.get("OPENROUTER_API_KEY"):
            return ("Salvo, mas a variável de ambiente OPENROUTER_API_KEY tem "
                    "prioridade e continuará sendo usada. Remova-a para valer "
                    "a chave gravada aqui.")
        return None

    @app.get("/api/modelos")
    def modelos():
        """Lista os modelos do endereço informado (ou do configurado).

        A chave nunca sai do servidor: quem chama manda no máximo o
        ``base_url`` a testar (útil antes mesmo de salvar a configuração,
        para conferir se o endereço digitado responde); a chave usada na
        requisição é sempre a que já está guardada aqui.
        """
        base_url = (request.args.get("base_url") or "").strip() or \
            config.get_setting("llm_base_url", "LLM_BASE_URL") or llm.DEFAULT_BASE_URL
        chave = config.get_openrouter_api_key()
        try:
            lista = llm.list_models(base_url, api_key=chave)
        except llm.LLMError as exc:
            return jsonify({"erro": str(exc)}), 502
        return jsonify({"modelos": lista})

    @app.get("/api/trabalhos")
    def trabalhos():
        return jsonify([j.para_json() for j in fila.listar()])

    @app.delete("/api/trabalho/<job_id>")
    def dispensar(job_id):
        """Tira um trabalho terminado da lista, sem tocar nos arquivos."""
        if not fila.remover(job_id):
            return jsonify({"erro": "Esse trabalho não terminou, ou já saiu "
                                    "da lista."}), 400
        return jsonify({"ok": True})

    @app.post("/api/trabalhos/limpar")
    def limpar_trabalhos():
        """Varre a lista dos que já terminaram. O que está rodando fica."""
        return jsonify({"removidos": fila.limpar_terminados()})

    @app.delete("/api/arquivo/<path:nome>")
    def excluir(nome):
        """Apaga um filme ou legenda da pasta de trabalho."""
        caminho = os.path.join(media_dir, nome)
        if not _dentro_da_pasta(caminho, media_dir):
            return jsonify({"erro": "Arquivo inválido."}), 400
        if not os.path.isfile(caminho):
            return jsonify({"erro": "Esse arquivo não está mais na pasta."}), 404
        if not (pipeline.is_media(caminho) or pipeline.is_subtitle(caminho)):
            return jsonify({"erro": "Só dá para apagar filme ou legenda."}), 400
        # Apagar debaixo de um trabalho em andamento deixaria o Whisper lendo
        # um arquivo que sumiu, e o erro sairia lá na frente, sem explicação.
        if fila.em_uso(caminho):
            return jsonify({"erro": "Esse arquivo está sendo processado "
                                    "agora. Espere terminar."}), 409

        try:
            os.remove(caminho)
        except OSError as exc:
            return jsonify({"erro": f"Não consegui apagar: {exc}"}), 500
        return jsonify({"ok": True, "apagado": nome})

    @app.get("/api/trabalho/<job_id>")
    def trabalho(job_id):
        job = fila.buscar(job_id)
        if not job:
            return jsonify({"erro": "Trabalho não encontrado."}), 404
        return jsonify(job.para_json())

    @app.post("/api/trabalho/<job_id>/cancelar")
    def cancelar(job_id):
        return jsonify({"cancelado": fila.cancelar(job_id)})

    @app.get("/api/baixar/<job_id>")
    def baixar(job_id):
        job = fila.buscar(job_id)
        if not job or not job.resultado or not os.path.exists(job.resultado):
            return jsonify({"erro": "Resultado indisponível."}), 404
        return send_file(job.resultado, as_attachment=True,
                         download_name=os.path.basename(job.resultado))

    @app.post("/api/transcribe")
    def transcrever():
        """Transcreve um arquivo enviado e devolve a legenda na hora.

        Existe para ferramentas de fora -- o Subtitle Edit, por exemplo --
        mandarem o arquivo e receberem o texto na mesma requisicao, sem
        passar pela fila nem deixar nada guardado no servidor.

        Usa o mesmo Faster-Whisper-XXL que a interface usa. Nao ha um segundo
        motor: o executavel e apontado por FASTER_WHISPER_PATH ou pela
        configuracao, e o modelo, a diarizacao e a leitura do .srt sao os
        mesmos do resto do programa.

            curl -X POST -F "file=@audio.mp3" http://localhost:8000/api/transcribe
            curl -X POST -F "file=@video.mp4" \
                 "http://localhost:8000/api/transcribe?format=srt" -o saida.srt

        Aceita ``idioma`` (padrao: detectar) e ``diarizar=0`` para nao
        identificar quem fala, que e mais rapido.

        A requisicao fica presa ate o Whisper terminar -- e o proprio
        proposito da rota, receber tudo numa tacada so. Existe um limite
        (``AUTOSRT_TRANSCRIBE_API_TIMEOUT``, 1800s por padrao) so para a
        requisicao desistir e devolver erro em vez de travar para sempre se
        o processo do Whisper pendurar.
        """
        arquivo = request.files.get("file")
        if not arquivo or not arquivo.filename:
            return jsonify({"erro": "Nenhum arquivo enviado."}), 400

        nome = os.path.basename(arquivo.filename)
        if not pipeline.is_media(nome):
            return jsonify({
                "erro": f"Nao sei o que fazer com {nome}. Envie um video ou "
                        "um audio (mp3, wav, m4a, mkv, mp4, e afins)."
            }), 400

        # Tudo dentro de uma pasta temporaria: o arquivo enviado, o .srt que o
        # Whisper escreve ao lado dele e a copia que sai na resposta. Nada
        # sobra na pasta de trabalho, que e de quem usa a pagina.
        with tempfile.TemporaryDirectory(prefix="autosrt-api-") as pasta:
            entrada = os.path.join(pasta, nome)
            arquivo.save(entrada)

            diarizar = request.args.get("diarizar", "1") != "0"
            try:
                cues = transcribe.transcribe(
                    entrada, output_dir=pasta,
                    language=request.args.get("idioma") or None,
                    diarize=transcribe.DEFAULT_DIARIZE_MODEL if diarizar else None,
                    timeout=app.config["TRANSCRIBE_API_TIMEOUT"])
            except TranscriptionError as exc:
                return jsonify({"erro": str(exc)}), 500

            if not cues:
                return jsonify({
                    "erro": "O Whisper nao encontrou fala nenhuma no arquivo."
                }), 422

            # Serializa pelo srt_io, o mesmo que grava as legendas da fila,
            # para a saida da API ser identica a do resto do programa.
            saida = os.path.join(pasta, "resposta.srt")
            srt_io.save_cues(cues, saida)
            with open(saida, encoding="utf-8") as legenda:
                texto = legenda.read()

        if request.args.get("format") == "srt":
            return send_file(
                io.BytesIO(texto.encode("utf-8")),
                mimetype="application/x-subrip", as_attachment=True,
                download_name=os.path.splitext(nome)[0] + ".srt")

        return jsonify({
            "sucesso": True,
            "arquivo": nome,
            # Amostra curta demais não derruba a resposta: vem vazio.
            "idioma": pipeline._safe_detect(cues),
            "legendas": texto,
            "linhas": len(cues),
        })

    @app.errorhandler(413)
    def grande_demais(_):
        limite = app.config["MAX_UPLOAD_GB"]
        return jsonify({
            "erro": f"Arquivo maior que o limite de {limite:g} GB. Copie-o "
                    "direto para a pasta do servidor, que ele aparece na lista."
        }), 413


PAGINA = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoSRT</title>
<style>
  :root {
    color-scheme: light dark;
    --bg-grad: linear-gradient(180deg, #f8f9fb 0%, #eef0f4 100%);
    --surface: #ffffff;
    --surface-2: #eef0f4;
    --surface-hover: #e6e9ef;
    --border: #dfe2e8;
    --text: #16161c;
    --text-muted: #666a75;
    --accent: #3b82f6;
    --accent-hover: #2f6fd8;
    --accent-soft: #e6efff;
    --success: #16a34a;
    --success-soft: #e5f7ec;
    --danger: #dc2626;
    --danger-soft: #fdebea;
    --warning: #b45309;
    --shadow-sm: 0 1px 2px rgba(20, 22, 30, .05);
    --shadow-md: 0 10px 30px rgba(20, 22, 30, .10);
    --radius: 14px;
    --radius-sm: 9px;
  }
  /* Sem preferência salva, segue o tema do sistema -- e sem JS nenhum
     rodado ainda, para a primeira pintura já vir no tom certo. */
  @media (prefers-color-scheme: dark) {
    :root:not([data-tema="claro"]) {
      --bg-grad: linear-gradient(180deg, #1b1b1f 0%, #17171b 100%);
      --surface: #212129; --surface-2: #26262f; --surface-hover: #2c2c36;
      --border: #313139; --text: #ecedf1; --text-muted: #9a9aa2;
      --accent-hover: #5c9bfc; --accent-soft: #1e2c47;
      --success: #4ade80; --success-soft: #16311f;
      --danger: #f87171; --danger-soft: #3a1c1c; --warning: #fbbf24;
      --shadow-sm: 0 1px 2px rgba(0, 0, 0, .35);
      --shadow-md: 0 10px 30px rgba(0, 0, 0, .45);
    }
  }
  /* Escolha manual (botão no topo) sempre vence a preferência do sistema. */
  :root[data-tema="escuro"] {
    --bg-grad: linear-gradient(180deg, #1b1b1f 0%, #17171b 100%);
    --surface: #212129; --surface-2: #26262f; --surface-hover: #2c2c36;
    --border: #313139; --text: #ecedf1; --text-muted: #9a9aa2;
    --accent-hover: #5c9bfc; --accent-soft: #1e2c47;
    --success: #4ade80; --success-soft: #16311f;
    --danger: #f87171; --danger-soft: #3a1c1c; --warning: #fbbf24;
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, .35);
    --shadow-md: 0 10px 30px rgba(0, 0, 0, .45);
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
         background: var(--bg-grad); color: var(--text);
         transition: background-color .2s ease, color .2s ease; }
  main { max-width: 820px; margin: 0 auto; padding: 24px 16px 64px; }
  .cabecalho { display: flex; align-items: flex-start; justify-content: space-between;
               gap: 16px; flex-wrap: wrap; }
  h1 { font-size: 26px; margin: 8px 0 4px; letter-spacing: -.01em; }
  p.sub { color: var(--text-muted); margin: 0 0 28px; }
  h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .07em;
       color: var(--text-muted); margin: 32px 0 12px; }
  .alternar-tema { flex-shrink: 0; width: 40px; height: 40px; padding: 0; margin-top: 4px;
                    border-radius: 999px; background: var(--surface);
                    border: 1px solid var(--border); color: var(--text);
                    display: flex; align-items: center; justify-content: center;
                    box-shadow: var(--shadow-sm); transition: transform .15s ease, background .15s ease; }
  .alternar-tema:hover { background: var(--surface-hover); transform: translateY(-1px); }
  .alternar-tema svg { width: 19px; height: 19px; }
  .alternar-tema .icone-lua { display: none; }
  :root[data-tema="escuro"] .alternar-tema .icone-sol { display: none; }
  :root[data-tema="escuro"] .alternar-tema .icone-lua { display: block; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-tema="claro"]) .alternar-tema .icone-sol { display: none; }
    :root:not([data-tema="claro"]) .alternar-tema .icone-lua { display: block; }
  }
  .drop { border: 2px dashed var(--border); border-radius: var(--radius); padding: 30px 16px;
          text-align: center; color: var(--text-muted); cursor: pointer; background: var(--surface);
          transition: border-color .15s, background .15s, transform .15s; }
  .drop:hover, .drop.ativo { border-color: var(--accent); background: var(--accent-soft); }
  .drop.ativo { transform: scale(1.008); }
  .drop svg { width: 28px; height: 28px; color: var(--text-muted); }
  .drop #dica { margin: 8px 0 14px; font-size: 14px; }
  .tag.legenda-pronta { color: var(--success); }
  .lista { border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden;
           background: var(--surface); box-shadow: var(--shadow-sm); }
  .item { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding: 12px 14px;
          border-bottom: 1px solid var(--border); transition: background .15s ease; }
  .item:hover { background: var(--surface-2); }
  .item:last-child { border-bottom: 0; }
  /* min-width impede que o nome seja espremido até sumir quando a linha
     acumula selos e um seletor longo; os selos não encolhem, o nome sim. */
  .item .nome { flex: 1 1 auto; min-width: 9ch; overflow: hidden;
                text-overflow: ellipsis; white-space: nowrap; }
  .item .tag { flex-shrink: 0; }
  /* Marca o que acabou de subir, para o usuário achar onde clicar. */
  .item.novo { background: var(--accent-soft); box-shadow: inset 3px 0 0 var(--accent); }
  .item .acao { flex-shrink: 1; max-width: 220px; }
  /* Selo de reconhecimento do TMDB: pôster pequeno + título (ano). É um
     palpite pelo nome do arquivo, por isso fica discreto, não em destaque. */
  .filme { display: flex; align-items: center; gap: 6px; flex-shrink: 0;
           max-width: 200px; }
  .filme .poster { width: 24px; height: 34px; object-fit: cover;
                   border-radius: 3px; flex-shrink: 0; background: var(--surface-2); }
  .filme .titulo { font-size: 12px; color: var(--text-muted); overflow: hidden;
                   text-overflow: ellipsis; white-space: nowrap; }
  #arquivo { display: none; }
  #escolher { display: inline-block; padding: 9px 18px; background: var(--accent);
              color: #fff; border-radius: 999px; cursor: pointer; font-weight: 600;
              border: none; transition: background .15s ease, transform .15s ease; }
  #escolher:hover { background: var(--accent-hover); transform: translateY(-1px); }
  .pasta { padding: 8px 14px; background: var(--surface-2); font-size: 13px;
           color: var(--text-muted); border-bottom: 1px solid var(--border); }
  .barra-acoes { display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
                 margin-bottom: 10px; padding: 10px 14px; border-radius: var(--radius-sm);
                 background: var(--surface-2); }
  /* Mesmo motivo do details#config label[hidden] acima: display:flex é de
     autor e vence o display:none do hidden vindo do navegador -- sem isto,
     a pasta vazia mostrava a barra de ações do mesmo jeito. */
  .barra-acoes[hidden] { display: none; }
  .barra-acoes .conta { flex: 1; color: var(--text-muted); font-size: 14px; }
  select { font: inherit; background: var(--surface); color: var(--text);
           border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 8px; }
  input[type=checkbox] { width: 17px; height: 17px; accent-color: var(--accent);
                         cursor: pointer; }
  label.tudo { display: flex; align-items: center; gap: 8px; cursor: pointer;
               color: var(--text-muted); font-size: 14px; }
  .tag { font-size: 12px; color: var(--text-muted); background: var(--surface-2);
         padding: 2px 8px; border-radius: 999px; }
  button { font: inherit; border: 0; border-radius: var(--radius-sm); padding: 8px 16px;
           background: var(--accent); color: #fff; cursor: pointer;
           transition: background .15s ease, transform .1s ease; }
  button:hover:enabled { background: var(--accent-hover); transform: translateY(-1px); }
  button.fantasma { background: var(--surface-2); color: var(--text); }
  button.fantasma:hover:enabled { background: var(--surface-hover); }
  button:disabled { opacity: .5; cursor: default; transform: none; }
  button:focus-visible, input:focus-visible, select:focus-visible, a:focus-visible,
  summary:focus-visible, [tabindex]:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 2px; }
  .barra { height: 6px; background: var(--surface-2); border-radius: 999px;
           overflow: hidden; margin-top: 8px; }
  .barra > div { height: 100%; background: var(--accent); width: 0;
                 transition: width .3s; }
  .trabalho { border: 1px solid var(--border); border-radius: var(--radius);
              padding: 14px; margin-bottom: 10px; background: var(--surface);
              box-shadow: var(--shadow-sm); }
  .trabalho .topo { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .etapa { color: var(--text-muted); font-size: 14px; margin-top: 6px; }
  .erro { color: var(--danger); }
  .ok { color: var(--success); }
  .vazio { color: var(--text-muted); padding: 20px; text-align: center; }
  a.baixar { color: var(--accent); text-decoration: none; font-weight: 600; }
  a.baixar:hover { text-decoration: underline; }
  .item a.baixar { flex-shrink: 0; }
  .apagar { flex-shrink: 0; background: none; border: 1px solid var(--danger-soft);
            color: var(--danger); }
  .apagar:hover:enabled { background: var(--danger-soft); }
  .dispensar { font-size: 18px; line-height: 1; padding: 2px 8px; }
  h2 #limpar { float: right; text-transform: none; letter-spacing: 0;
               font-size: 13px; }
  details#config, details#config-transcricao {
    margin-top: 20px; border: 1px solid var(--border); border-radius: var(--radius);
    background: var(--surface); box-shadow: var(--shadow-sm); }
  details#config summary, details#config-transcricao summary {
    padding: 14px; cursor: pointer; color: var(--text-muted); font-size: 14px;
    display: flex; align-items: center; gap: 8px; }
  details#config summary::marker, details#config-transcricao summary::marker {
    color: var(--accent); }
  details#config .painel, details#config-transcricao .painel {
    padding: 0 14px 14px; display: grid; gap: 12px; }
  details#config label, details#config-transcricao label {
    display: grid; gap: 6px; font-size: 14px; color: var(--text-muted); }
  /* Sem isto, o atributo "hidden" perde para a regra acima: display:grid é
     de autor e sempre vence o display:none da folha de estilo do navegador,
     não importa a especificidade -- então esconder via JS não escondia
     nada de verdade. */
  details#config label[hidden] { display: none; }
  details#config input, details#config-transcricao input {
    font: inherit; background: var(--surface-2); color: var(--text);
    border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 10px; }
  details#config-transcricao p.dica-config { margin: 0; }
  .rodape { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .dica-config { flex: 1; font-size: 13px; color: var(--text-muted); }
  .selo { font-size: 12px; padding: 2px 8px; border-radius: 999px;
          background: var(--surface-2); }
  .selo.ok { color: var(--success); }
  .selo.falta { color: var(--warning); }
  .modo-provedor { display: flex; gap: 8px; flex-wrap: wrap; }
  .modo-provedor button { flex: 1; min-width: 140px; background: var(--surface-2);
                          color: var(--text); border: 1px solid var(--border); }
  .modo-provedor button.ativo { background: var(--accent-soft); border-color: var(--accent);
                                color: var(--text); }
  .modo-provedor .tag { margin-left: 6px; }
  /* Cor sozinha não é garantia de dar para notar; o texto "selecionado"
     deixa explícito qual dos dois modos está valendo agora. */
  .modo-provedor .marca-selecionado { display: none; color: var(--success);
                                      font-size: 12px; }
  .modo-provedor button.ativo .marca-selecionado { display: inline; }
  #mensagem-salvar.sucesso { color: var(--success); }
  .linha-modelo { display: flex; gap: 8px; flex-wrap: wrap; }
  .linha-caixa { flex-direction: row; align-items: center; gap: 8px; }
  .linha-caixa input { width: auto; }
  .linha-modelo input { flex: 1; min-width: 160px; }
  .lista-modelos { max-height: 220px; overflow-y: auto; border: 1px solid var(--border);
                   border-radius: var(--radius-sm); }
  .lista-modelos button { display: block; width: 100%; text-align: left;
                          background: none; border: 0; border-radius: 0;
                          border-bottom: 1px solid var(--border); color: var(--text);
                          padding: 8px 10px; font-size: 13px; }
  .lista-modelos button:last-child { border-bottom: 0; }
  .lista-modelos button:hover { background: var(--surface-2); }
  .lista-modelos .preco { display: block; color: var(--text-muted); font-size: 12px;
                          margin-top: 2px; }
  /* Tooltips: ficam nas abas de configuração (summary) e em campos que
     merecem uma explicação curta -- o "?" ao lado do rótulo. */
  .rotulo-com-ajuda { display: inline-flex; align-items: center; gap: 6px; }
  .ajuda { display: inline-flex; align-items: center; justify-content: center;
           width: 16px; height: 16px; border-radius: 999px; background: var(--surface-2);
           color: var(--text-muted); font-size: 11px; font-weight: 700; cursor: help;
           border: 1px solid var(--border); }
  [data-tip] { position: relative; }
  [data-tip]::after {
    content: attr(data-tip); position: absolute; bottom: calc(100% + 8px); left: 50%;
    transform: translateX(-50%) translateY(4px); background: var(--text); color: var(--surface);
    padding: 7px 10px; border-radius: 8px; font-size: 12px; line-height: 1.4;
    width: max-content; max-width: min(260px, 80vw); box-shadow: var(--shadow-md);
    opacity: 0; pointer-events: none; transition: opacity .15s ease, transform .15s ease;
    z-index: 30; white-space: normal; text-align: left; font-weight: 400; }
  [data-tip]:hover::after, [data-tip]:focus-visible::after {
    opacity: 1; transform: translateX(-50%) translateY(0); }
  summary[data-tip]::after { left: 0; transform: translateX(0) translateY(4px); }
  summary[data-tip]:hover::after, summary[data-tip]:focus-visible::after {
    transform: translateX(0) translateY(0); }
  @media (max-width: 560px) {
    .item .acao { max-width: none; flex: 1 1 100%; }
    .barra-acoes { flex-direction: column; align-items: stretch; }
    .trabalho .topo .acao { flex: 1 1 100%; display: flex; gap: 8px; }
    [data-tip]::after, summary[data-tip]::after { max-width: min(220px, 78vw); }
  }
</style>
</head>
<body>
<main>
  <div class="cabecalho">
    <div>
      <h1>AutoSRT</h1>
      <p class="sub">Transcreve o áudio do filme e traduz a legenda para português.</p>
    </div>
    <button type="button" id="alternar-tema" class="alternar-tema" aria-pressed="false"
            title="Alternar entre tema claro e escuro" aria-label="Alternar entre tema claro e escuro">
      <svg class="icone-sol" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle>
        <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path></svg>
      <svg class="icone-lua" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M20.4 14.7A8.5 8.5 0 1 1 9.3 3.6a7 7 0 0 0 11.1 11.1Z"></path></svg>
    </button>
  </div>

  <div class="drop" id="drop">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
         stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M12 16V4M12 4l-4 4M12 4l4 4"></path>
      <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"></path>
    </svg>
    <strong>Arraste o filme e a legenda aqui</strong>
    <div id="dica">pode mandar os dois juntos &mdash; v&iacute;deo, &aacute;udio ou legenda</div>
    <label for="arquivo" id="escolher">Escolher arquivos...</label>
    <div class="barra" id="barra-envio" hidden><div></div></div>
    <input type="file" id="arquivo" accept="{{ACCEPT}}" multiple>
  </div>

  <h2>Arquivos no servidor</h2>
  <div class="barra-acoes" id="barra-acoes" hidden>
    <label class="tudo"><input type="checkbox" id="tudo"> Selecionar todos</label>
    <span class="conta" id="conta"></span>
    <button id="lote" disabled>Processar selecionados</button>
    <button id="apagar-lote" class="apagar" disabled>Apagar selecionados</button>
  </div>
  <div class="lista" id="lista"><div class="vazio">Carregando...</div></div>

  <h2>Trabalhos <button id="limpar" class="fantasma" hidden>Limpar terminados</button></h2>
  <div id="trabalhos"><div class="vazio">Nada ainda.</div></div>

  <details id="config">
    <summary data-tip="Endere&ccedil;o, chave e modelo do servi&ccedil;o usado para traduzir as legendas para portugu&ecirc;s.">Configura&ccedil;&atilde;o da tradu&ccedil;&atilde;o <span id="estado-chave"></span></summary>
    <div class="painel">
      <label><span class="rotulo-com-ajuda"><span id="rotulo-endereco">Endere&ccedil;o da API</span><span class="ajuda" tabindex="0" data-tip="Para onde mandar o texto a traduzir: o endere&ccedil;o da API do OpenRouter, ou de um servidor compat&iacute;vel na sua rede, como o Ollama.">?</span></span>
        <input type="text" id="base_url" placeholder="https://openrouter.ai/api/v1">
      </label>
      <label id="campo-chave"><span class="rotulo-com-ajuda">Chave do OpenRouter<span class="ajuda" tabindex="0" data-tip="Sua chave de API do OpenRouter. Fica guardada s&oacute; no servidor e nunca volta para esta p&aacute;gina.">?</span></span>
        <input type="password" id="chave" placeholder="sk-or-v1-..." autocomplete="off">
      </label>
      <label><span class="rotulo-com-ajuda">Modelo<span class="ajuda" tabindex="0" data-tip="Qual modelo de linguagem usar na tradu&ccedil;&atilde;o. Clique em 'Buscar modelos' para ver os dispon&iacute;veis nesse endere&ccedil;o.">?</span></span>
        <div class="linha-modelo">
          <input type="text" id="modelo" placeholder="deepseek/deepseek-chat" data-exemplo-openrouter="deepseek/deepseek-chat" data-exemplo-local="llama3.1">
          <button type="button" id="buscar-modelos" class="fantasma">Buscar modelos</button>
        </div>
      </label>
      <span class="dica-config" id="estado-modelos"></span>
      <div id="lista-modelos" class="lista-modelos" hidden></div>
      <label><span class="rotulo-com-ajuda">Legendas por requisi&ccedil;&atilde;o<span class="ajuda" tabindex="0" data-tip="Quantas legendas mandar de uma vez pro modelo traduzir. Deixe em branco para decidir sozinho pelo endere&ccedil;o (bloco pequeno pra servidor local, grande pra provedor na nuvem). Sobrescreva se o seu servidor local n&atilde;o for detectado como local (ex.: container acessando o host por IP da rede), ou se quiser um bloco menor mesmo na nuvem.">?</span></span>
        <input type="text" id="block_size" placeholder="ex: 2 -- em branco decide sozinho pelo endere&ccedil;o">
      </label>
      <label><span class="rotulo-com-ajuda">Chave do TMDB<span class="ajuda" tabindex="0" data-tip="Opcional. Com uma chave do TMDB, o AutoSRT reconhece o filme pelo nome do arquivo e mostra o p&ocirc;ster na lista.">?</span></span> <span id="estado-chave-tmdb"></span>
        <input type="password" id="chave_tmdb" placeholder="opcional &mdash; reconhece o filme pelo nome do arquivo" autocomplete="off">
      </label>
      <div class="modo-provedor">
        <button type="button" id="modo-openrouter">OpenRouter<span class="tag">pago</span><span class="marca-selecionado"> &#10003; selecionado</span></button>
        <button type="button" id="modo-local">Local<span class="tag">gr&aacute;tis</span><span class="marca-selecionado"> &#10003; selecionado</span></button>
      </div>
      <div class="rodape">
        <span class="dica-config" id="mensagem-salvar">As chaves ficam no config.json do servidor e nunca voltam para esta p&aacute;gina.</span>
        <button id="salvar">Salvar</button>
      </div>
    </div>
  </details>

  <details id="config-transcricao">
    <summary data-tip="Ajustes finos da detec&ccedil;&atilde;o de fala (VAD) usada pelo Whisper ao transcrever.">Configura&ccedil;&atilde;o da transcri&ccedil;&atilde;o <span id="estado-vad" class="selo"></span></summary>
    <div class="painel">
      <p class="dica-config">Vale para todo arquivo processado pela fila -- n&atilde;o &eacute; por v&iacute;deo. Deixe em branco para usar o padr&atilde;o do pr&oacute;prio Whisper, sem mexer em nada.</p>
      <label><span class="rotulo-com-ajuda">Normalizar o &aacute;udio<span class="ajuda" tabindex="0" data-tip="&Aacute;udio muito baixo faz o Whisper trocar a fala por alucina&ccedil;&atilde;o (o cl&aacute;ssico 'Thank you.' em cima do ru&iacute;do) e deixar trechos inteiros sem legenda, sem dar erro nenhum. Em 'auto' o volume &eacute; medido e s&oacute; o &aacute;udio fraco &eacute; corrigido. Precisa do ffmpeg instalado.">?</span></span>
        <select id="normalize_audio">
          <option value="auto">Auto -- s&oacute; quando o &aacute;udio estiver fraco (recomendado)</option>
          <option value="sempre">Sempre -- normaliza todo arquivo</option>
          <option value="nunca">Nunca -- usa o &aacute;udio como veio</option>
        </select>
      </label>
      <label><span class="rotulo-com-ajuda">Idioma falado<span class="ajuda" tabindex="0" data-tip="C&oacute;digo do idioma do &aacute;udio (en, es, fr...). Em branco, o Whisper detecta sozinho analisando os primeiros segundos -- e um come&ccedil;o at&iacute;pico (trilha, sil&ecirc;ncio, vinheta) pode levar a uma detec&ccedil;&atilde;o errada que estraga justamente o in&iacute;cio da transcri&ccedil;&atilde;o. Informar o idioma elimina esse risco.">?</span></span>
        <input type="text" id="whisper_language" placeholder="em branco = detectar sozinho. Ex: en, es, fr">
      </label>
      <label class="linha-caixa"><input type="checkbox" id="diarizar" checked>
        <span class="rotulo-com-ajuda">Identificar quem fala (diariza&ccedil;&atilde;o)<span class="ajuda" tabindex="0" data-tip="Marca o locutor de cada fala, o que d&aacute; ao tradutor a dica de g&ecirc;nero ('ele disse' x 'ela disse'). Custa um segundo modelo rodando por cima do &aacute;udio: se a legenda est&aacute; saindo com trechos faltando, desligar isto &eacute; o primeiro teste -- trecho que esse modelo n&atilde;o atribui a ningu&eacute;m corre o risco de n&atilde;o virar legenda. O texto continua saindo; some s&oacute; a dica de g&ecirc;nero.">?</span></span>
      </label>
      <label><span class="rotulo-com-ajuda">Modelo do Whisper<span class="ajuda" tabindex="0" data-tip="Qual modelo transcreve o &aacute;udio. Em branco usa o 'turbo' (r&aacute;pido e leve). O 'large-v3' &eacute; mais preciso, por&eacute;m mais pesado e mais lento -- em placa de 5 GB ele s&oacute; cabe com folga em int8.">?</span></span>
        <input type="text" id="whisper_model" placeholder="em branco = turbo (padr&atilde;o). Ex: large-v3, medium, small">
      </label>
      <label><span class="rotulo-com-ajuda">Tipo de c&aacute;lculo<span class="ajuda" tabindex="0" data-tip="Precis&atilde;o num&eacute;rica usada na GPU. Em branco ('auto') o pr&oacute;prio CTranslate2 escolhe. Em placas Pascal (s&eacute;rie P, sem tensor cores) o 'int8' costuma ser mais r&aacute;pido que 'float16' e ocupa metade da mem&oacute;ria.">?</span></span>
        <input type="text" id="whisper_compute_type" placeholder="em branco = auto. Ex: int8, float16, float32">
      </label>
      <label><span class="rotulo-com-ajuda">Detector de fala (VAD)<span class="ajuda" tabindex="0" data-tip="Qual detector decide onde h&aacute; fala. Cada um calibra a sensibilidade de um jeito, ent&atilde;o trocar o detector &eacute; t&atilde;o candidato a resolver legenda com buracos quanto mexer na sensibilidade. O padr&atilde;o do pr&oacute;prio execut&aacute;vel &eacute; silero_v4_fw.">?</span></span>
        <input type="text" id="vad_method" placeholder="em branco = padr&atilde;o. Ex: silero_v5, pyannote_v3, webrtc">
      </label>
      <label><span class="rotulo-com-ajuda">Sensibilidade da VAD (0 a 1)<span class="ajuda" tabindex="0" data-tip="O quanto o Whisper precisa 'ouvir' pra considerar que h&aacute; fala. Um valor menor pega fala mais baixa, mas tamb&eacute;m mais ru&iacute;do.">?</span></span>
        <input type="text" id="vad_threshold" placeholder="ex: 0.2 -- menor pega fala mais baixa">
      </label>
      <label><span class="rotulo-com-ajuda">Sil&ecirc;ncio m&iacute;nimo (ms)<span class="ajuda" tabindex="0" data-tip="Tempo m&iacute;nimo de sil&ecirc;ncio, em milissegundos, para separar duas falas. Evita cortar a &uacute;ltima palavra de falas r&aacute;pidas.">?</span></span>
        <input type="text" id="vad_min_silence_ms" placeholder="ex: 300 -- evita cortar a &uacute;ltima palavra de falas r&aacute;pidas">
      </label>
      <div class="rodape">
        <span class="dica-config" id="mensagem-salvar-vad">Mudan&ccedil;as valem a partir do pr&oacute;ximo trabalho enviado &agrave; fila.</span>
        <button id="salvar-vad">Salvar</button>
      </div>
    </div>
  </details>
</main>

<script>
const $ = (id) => document.getElementById(id);

// Tema claro/escuro: começa pela preferência do sistema (a folha de estilo
// já cobre isso sozinha, sem esperar o JS) e a partir daí obedece só a
// escolha manual, guardada para persistir entre visitas.
const CHAVE_TEMA = 'autosrt-tema';

function temaPreferido() {
  const salvo = localStorage.getItem(CHAVE_TEMA);
  if (salvo === 'claro' || salvo === 'escuro') return salvo;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'escuro' : 'claro';
}

function aplicarTema(tema) {
  document.documentElement.setAttribute('data-tema', tema);
  $('alternar-tema').setAttribute('aria-pressed', String(tema === 'escuro'));
}

aplicarTema(temaPreferido());
$('alternar-tema').onclick = () => {
  const novo = document.documentElement.getAttribute('data-tema') === 'escuro' ? 'claro' : 'escuro';
  localStorage.setItem(CHAVE_TEMA, novo);
  aplicarTema(novo);
};

async function carregarArquivos() {
  const r = await fetch('/api/arquivos');
  const itens = await r.json();
  const lista = $('lista');
  $('barra-acoes').hidden = !itens.length;

  if (!itens.length) {
    lista.innerHTML = '<div class="vazio">Nenhum arquivo na pasta do servidor.</div>';
    return;
  }

  lista.innerHTML = '';
  let pastaAtual = null;
  for (const item of itens) {
    if (item.pasta !== pastaAtual) {
      pastaAtual = item.pasta;
      if (pastaAtual !== '.') {
        const cabecalho = document.createElement('div');
        cabecalho.className = 'pasta';
        cabecalho.textContent = pastaAtual;
        lista.appendChild(cabecalho);
      }
    }
    lista.appendChild(linhaArquivo(item));
  }
  contar();
}

// Nome vira caminho de URL: escapa cada pedaço, mas mantém as barras, senão
// legenda dentro de subpasta não é encontrada.
const caminhoUrl = (nome) => nome.split('/').map(encodeURIComponent).join('/');

function linhaArquivo(item) {
  const div = document.createElement('div');
  div.className = 'item';
  div.dataset.arquivo = item.nome;
  div.innerHTML = `<input type="checkbox" class="marca">
    <span class="nome"></span>
    <span class="tag"></span><span class="tag"></span>
    <select class="acao"></select>
    <button>Processar</button>`;

  const nome = div.querySelector('.nome');
  nome.textContent = item.pasta === '.' ? item.nome : item.nome.split('/').pop();
  nome.title = item.nome;  // nome longo é truncado na linha; aqui vem inteiro
  div.querySelectorAll('.tag')[0].textContent = item.tipo;
  div.querySelectorAll('.tag')[1].textContent = item.tamanho;

  if (item.tem_legenda) {
    const selo = document.createElement('span');
    selo.className = 'tag legenda-pronta';
    selo.textContent = 'tem legenda';
    // Junto dos outros selos, não colado ao nome: assim o nome mantém a
    // largura que sobra em vez de disputar espaço no meio da linha.
    div.querySelectorAll('.tag')[1].after(selo);
  }

  // Reconhecimento do TMDB pelo nome do arquivo: é palpite, então some
  // silenciosamente quando o servidor não achou nada em vez de mostrar um
  // selo vazio ou errado.
  if (item.filme) {
    const bloco = document.createElement('span');
    bloco.className = 'filme';
    if (item.filme.poster) {
      const img = document.createElement('img');
      img.className = 'poster';
      img.src = item.filme.poster;
      img.alt = '';
      img.loading = 'lazy';
      bloco.appendChild(img);
    }
    const titulo = document.createElement('span');
    titulo.className = 'titulo';
    titulo.textContent = item.filme.ano
      ? `${item.filme.titulo} (${item.filme.ano})` : item.filme.titulo;
    titulo.title = 'Reconhecido pelo nome do arquivo, via TMDB';
    bloco.appendChild(titulo);
    div.querySelector('.nome').after(bloco);
  }

  // Legenda que já está no servidor se baixa direto, sem passar pela fila:
  // quem só quer o arquivo de ontem não precisa reprocessar o filme.
  if (item.tipo === 'legenda') {
    const baixar = document.createElement('a');
    baixar.className = 'baixar';
    baixar.href = '/api/legenda/' + caminhoUrl(item.nome);
    baixar.textContent = 'Baixar';
    div.querySelector('button').before(baixar);
  }

  const apagar = document.createElement('button');
  apagar.className = 'apagar';
  apagar.textContent = 'Apagar';
  apagar.title = 'Apaga este arquivo do servidor';
  apagar.onclick = async () => {
    if (!confirm(`Apagar ${item.nome} do servidor?\\n\\nNão dá para desfazer.`)) return;
    apagar.disabled = true;
    const r = await fetch('/api/arquivo/' + caminhoUrl(item.nome), {method: 'DELETE'});
    if (!r.ok) {
      apagar.disabled = false;
      alert((await r.json()).erro || 'Não consegui apagar.');
      return;
    }
    carregarArquivos();
  };
  div.appendChild(apagar);

  const seletor = div.querySelector('.acao');
  for (const acao of item.acoes) {
    const opcao = document.createElement('option');
    opcao.value = acao.id;
    opcao.textContent = acao.rotulo;
    seletor.appendChild(opcao);
  }

  if (recemChegados.includes(item.nome)) div.classList.add('novo');

  div.querySelector('.marca').onchange = contar;
  div.querySelector('button').onclick = () => processarUm(div);
  return div;
}

function pedidoDe(div) {
  const acao = div.querySelector('.acao').value;
  const pedido = {arquivo: div.dataset.arquivo, acao: acao};
  if (acao === 'deslocar') {
    const resposta = prompt(
      'Quantos segundos deslocar?\\n' +
      'Positivo atrasa a legenda, negativo adianta. Exemplo: -2.5');
    if (resposta === null) return null;
    const segundos = parseFloat(resposta.replace(',', '.'));
    if (isNaN(segundos)) { alert('Valor inválido: ' + resposta); return null; }
    pedido.segundos = segundos;
  }
  return pedido;
}

async function processarUm(div) {
  const pedido = pedidoDe(div);
  if (!pedido) return;

  const botao = div.querySelector('button');
  botao.disabled = true;
  botao.textContent = 'Enviado';
  const r = await fetch('/api/processar', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(pedido)
  });
  const dados = await r.json();
  if (!r.ok) {
    alert(dados.erro);
    botao.disabled = false;
    botao.textContent = 'Processar';
  }
  atualizar();
}

function selecionados() {
  return Array.from(document.querySelectorAll('.item'))
    .filter((div) => div.querySelector('.marca').checked);
}

function contar() {
  const marcados = selecionados();
  $('conta').textContent = marcados.length
    ? `${marcados.length} selecionado(s)` : '';
  $('lote').disabled = !marcados.length;
  $('apagar-lote').disabled = !marcados.length;
  const todos = document.querySelectorAll('.item').length;
  $('tudo').checked = todos > 0 && marcados.length === todos;
}

$('tudo').onchange = (e) => {
  document.querySelectorAll('.marca').forEach((c) => { c.checked = e.target.checked; });
  contar();
};

$('lote').onclick = async () => {
  const pedidos = [];
  for (const div of selecionados()) {
    const pedido = pedidoDe(div);
    if (pedido) pedidos.push(pedido);
  }
  if (!pedidos.length) return;

  $('lote').disabled = true;
  const r = await fetch('/api/processar-lote', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({itens: pedidos})
  });
  const dados = await r.json();
  if (dados.recusados && dados.recusados.length) {
    alert(dados.recusados.map((x) => `${x.arquivo}: ${x.erro}`).join('\\n'));
  }
  document.querySelectorAll('.marca').forEach((c) => { c.checked = false; });
  contar();
  atualizar();
};

$('apagar-lote').onclick = async () => {
  const alvos = selecionados().map((div) => div.dataset.arquivo).filter(Boolean);
  if (!alvos.length) return;

  // Confirmação lista os nomes: "apagar 12 arquivos" sem dizer quais é o
  // tipo de clique que se lamenta depois. Acima de 10 a lista viraria um
  // paredão, então corta e diz quantos ficaram de fora.
  const amostra = alvos.slice(0, 10).join('\\n');
  const resto = alvos.length > 10 ? `\\n...e mais ${alvos.length - 10}` : '';
  if (!confirm(`Apagar ${alvos.length} arquivo(s) do servidor?\\n\\n`
      + amostra + resto + '\\n\\nNão dá para desfazer.')) return;

  $('apagar-lote').disabled = true;
  // Um DELETE por arquivo, reusando a rota que já recusa o que está sendo
  // processado agora -- apagar debaixo de um trabalho em andamento deixaria
  // o Whisper lendo um arquivo que sumiu.
  const falhas = [];
  for (const nome of alvos) {
    const r = await fetch('/api/arquivo/' + caminhoUrl(nome), {method: 'DELETE'});
    if (!r.ok) {
      const dados = await r.json().catch(() => ({}));
      falhas.push(`${nome}: ${dados.erro || 'não consegui apagar'}`);
    }
  }

  if (falhas.length) alert(falhas.join('\\n'));
  contar();
  carregarArquivos();
};

$('drop').onclick = (e) => {
  // O label abre o seletor sozinho, e o clique nele sobe até aqui: sem esta
  // saída a caixa de arquivos seria pedida duas vezes no mesmo clique.
  if (e.target.closest('#escolher, #arquivo')) return;
  $('arquivo').click();
};
$('arquivo').onchange = (e) => { if (e.target.files.length) enviar(e.target.files); };
$('drop').ondragover = (e) => { e.preventDefault(); $('drop').classList.add('ativo'); };
$('drop').ondragleave = () => $('drop').classList.remove('ativo');
$('drop').ondrop = (e) => {
  e.preventDefault();
  $('drop').classList.remove('ativo');
  if (e.dataTransfer.files.length) enviar(e.dataTransfer.files);
};

const DICA_PADRAO = 'pode mandar os dois juntos — vídeo, áudio ou legenda';
let recemChegados = [];

function enviar(listaDeArquivos) {
  // XHR e nao fetch porque so ele reporta o andamento do envio. Filme leva
  // um ou dois minutos na rede local, e pagina parada sem sinal nenhum
  // parece travada.
  const arquivos = Array.from(listaDeArquivos);
  const dados = new FormData();
  arquivos.forEach((a) => dados.append('arquivo', a));

  const rotulo = arquivos.length === 1
    ? arquivos[0].name : `${arquivos.length} arquivos`;

  const barra = $('barra-envio');
  const preenchimento = barra.querySelector('div');
  const dica = $('dica');
  barra.hidden = false;
  preenchimento.style.width = '0%';

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/enviar');

  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round(e.loaded * 100 / e.total);
    preenchimento.style.width = pct + '%';
    dica.textContent = `Enviando ${rotulo}... ${pct}%`;
  };

  xhr.onload = () => {
    barra.hidden = true;
    dica.textContent = DICA_PADRAO;
    let resposta = {};
    try { resposta = JSON.parse(xhr.responseText); } catch (e) {}

    if (xhr.status >= 400) {
      alert(resposta.erro || 'Falha no envio.');
    } else if (resposta.recusados && resposta.recusados.length) {
      alert(resposta.recusados.map((x) => x.erro).join('\\n'));
    }

    // Os arquivos ficam guardados, não processados: quem manda o filme com a
    // legenda precisa que os dois estejam no lugar antes de decidir a ação.
    recemChegados = resposta.guardados || [];
    carregarArquivos();
  };

  xhr.onerror = () => {
    barra.hidden = true;
    dica.textContent = DICA_PADRAO;
    alert('Falha no envio. Verifique a conexão com o servidor.');
  };

  xhr.send(dados);
}

// Previsão em texto curto. Vem vazia enquanto o servidor não tem base para
// estimar, para a etapa nunca mostrar um tempo inventado.
function faltam(segundos) {
  if (segundos === null || segundos === undefined) return '';
  if (segundos < 60) return ' · falta menos de 1 min';
  const minutos = Math.round(segundos / 60);
  if (minutos < 60) return ` · faltam ~${minutos} min`;
  const horas = Math.floor(minutos / 60);
  const resto = minutos % 60;
  return ` · faltam ~${horas}h${resto ? String(resto).padStart(2, '0') : ''}`;
}

function cartao(job) {
  const div = document.createElement('div');
  div.className = 'trabalho';
  div.innerHTML = `<div class="topo">
      <strong class="nome"></strong><span style="flex:1"></span>
      <span class="acao"></span>
    </div>
    <div class="etapa"></div>
    <div class="barra"><div></div></div>`;
  div.querySelector('.nome').textContent = job.nome;

  const etapa = div.querySelector('.etapa');
  if (job.estado === 'erro') {
    etapa.className = 'etapa erro';
    etapa.textContent = job.erro || 'Falhou.';
  } else if (job.estado === 'concluido') {
    const d = job.detalhes || {};
    etapa.className = 'etapa ok';
    etapa.textContent = `${d.total || 0} legendas` +
      (d.idioma ? ` · ${d.idioma}` : '') +
      (d.falhas ? ` · ${d.falhas} não traduzida(s)` : '');
  } else {
    etapa.textContent = job.etapa + faltam(job.restante_seg);
  }

  div.querySelector('.barra > div').style.width =
    (job.estado === 'concluido' ? 100 : job.progresso) + '%';

  const acao = div.querySelector('.acao');
  if (job.baixavel) {
    const a = document.createElement('a');
    a.className = 'baixar';
    a.href = '/api/baixar/' + job.id;
    a.textContent = 'Baixar legenda';
    acao.appendChild(a);
  }

  // Trabalho terminado (deu certo, falhou ou foi cancelado) sai da lista no
  // ×. Sem isso um erro fica na tela até o servidor reiniciar.
  if (job.pronto) {
    const dispensar = document.createElement('button');
    dispensar.className = 'fantasma dispensar';
    dispensar.textContent = '×';
    dispensar.title = 'Tirar da lista';
    dispensar.onclick = async () => {
      dispensar.disabled = true;
      await fetch('/api/trabalho/' + job.id, {method: 'DELETE'});
      atualizar();
    };
    acao.appendChild(dispensar);
  } else if (!job.pronto) {
    const b = document.createElement('button');
    b.className = 'fantasma';
    b.textContent = 'Cancelar';
    b.onclick = async () => {
      await fetch('/api/trabalho/' + job.id + '/cancelar', {method: 'POST'});
      atualizar();
    };
    acao.appendChild(b);
  }
  return div;
}

async function atualizar() {
  const r = await fetch('/api/trabalhos');
  const jobs = await r.json();
  const alvo = $('trabalhos');
  $('limpar').hidden = !jobs.some((j) => j.pronto);
  if (!jobs.length) {
    alvo.innerHTML = '<div class="vazio">Nada ainda.</div>';
    return;
  }
  alvo.innerHTML = '';
  jobs.forEach((job) => alvo.appendChild(cartao(job)));
}

$('limpar').onclick = async () => {
  await fetch('/api/trabalhos/limpar', {method: 'POST'});
  atualizar();
};

// Guardado depois de cada /api/config: os botões de modo e o salvar
// precisam saber se já existe chave configurada, sem poder ler o valor dela
// (a chave nunca volta para a página).
let configAtual = {};

function marcarModoAtivo() {
  const url = $('base_url').value.trim();
  const local = url === configAtual.local_base_url;
  $('modo-openrouter').classList.toggle('ativo', url === configAtual.openrouter_base_url);
  $('modo-local').classList.toggle('ativo', local);

  // "API" não descreve bem um servidor rodando na própria máquina -- o
  // rótulo troca de vocabulário no modo local, não só de valor.
  $('rotulo-endereco').textContent = local ? 'Endereço do servidor' : 'Endereço da API';

  // Servidor local normalmente não pede chave nenhuma -- o campo some
  // inteiro em vez de só trocar de rótulo, que ainda deixava parecer que
  // era preciso preencher alguma coisa ali.
  $('campo-chave').hidden = local;

  const modeloInput = $('modelo');
  modeloInput.placeholder = local
    ? modeloInput.dataset.exemploLocal : modeloInput.dataset.exemploOpenrouter;

  // O selo do topo é sobre a configuração já salva (a chave nunca volta
  // pra cá pra saber se o que está no campo bate); em modo local, não ter
  // chave não é uma falta -- é o normal, e não devia soar como aviso.
  const selo = $('estado-chave');
  if (local) {
    selo.className = 'selo ok';
    selo.textContent = configAtual.tem_chave
      ? 'chave configurada (opcional aqui)' : 'chave opcional (modo local)';
  } else {
    selo.className = 'selo ' + (configAtual.tem_chave ? 'ok' : 'falta');
    selo.textContent = configAtual.tem_chave
      ? (configAtual.chave_do_ambiente ? 'chave do ambiente' : 'chave configurada')
      : 'sem chave';
  }
}

// Slug "fornecedor/modelo" (com barra) só existe no catálogo do OpenRouter;
// servidor local usa nomes como "llama3.1", sem barra. Ao trocar de modo,
// um valor que claramente pertence ao outro lado é limpo, para não salvar
// por engano um modelo que não existe no servidor escolhido.
function limparModeloSeIncompativel(local) {
  const modeloInput = $('modelo');
  const temBarra = modeloInput.value.includes('/');
  if ((local && temBarra) || (!local && modeloInput.value && !temBarra)) {
    modeloInput.value = '';
  }
}

async function carregarConfig() {
  const r = await fetch('/api/config');
  const c = await r.json();
  configAtual = c;
  $('modelo').value = c.modelo || '';
  $('base_url').value = c.base_url || '';
  // null (não configurado) vira campo vazio -- decide sozinho pelo endereço.
  $('block_size').value = c.llm_block_size ?? '';

  const seloTmdb = $('estado-chave-tmdb');
  seloTmdb.className = 'selo ' + (c.tem_chave_tmdb ? 'ok' : 'falta');
  seloTmdb.textContent = c.tem_chave_tmdb ? 'ativo' : 'sem chave';

  // null (não configurado) vira campo vazio, não "null" escrito na tela.
  $('normalize_audio').value = c.normalize_audio || 'auto';
  $('whisper_language').value = c.whisper_language ?? '';
  $('diarizar').checked = c.diarizar !== false;
  $('whisper_model').value = c.whisper_model ?? '';
  $('whisper_compute_type').value = c.whisper_compute_type ?? '';
  $('vad_method').value = c.vad_method ?? '';
  $('vad_method').placeholder =
    'em branco = ' + c.vad_method_padrao + '. Ex: silero_v4_fw, pyannote_v3, webrtc';
  $('vad_threshold').value = c.vad_threshold ?? '';
  $('vad_min_silence_ms').value = c.vad_min_silence_ms ?? '';
  const seloVad = $('estado-vad');
  const vadAtiva = c.vad_threshold !== null || c.vad_min_silence_ms !== null
    || !!c.whisper_model || !!c.whisper_compute_type || !!c.vad_method
    || !!c.whisper_language || c.diarizar === false;
  seloVad.className = 'selo ' + (vadAtiva ? 'ok' : '');
  seloVad.textContent = vadAtiva ? 'ajustada' : 'padrão do Whisper';

  marcarModoAtivo();

  // Sem chave, a tradução não funciona: abre o painel para não deixar o
  // usuário descobrir isso só quando o primeiro trabalho falhar. Não vale
  // para o modo local, que não costuma precisar de chave nenhuma.
  if (!c.tem_chave && $('base_url').value !== c.local_base_url) {
    $('config').open = true;
  }
}

// Só troca o endereço; o usuário continua podendo digitar por cima, para um
// servidor OpenAI-compatível diferente de Ollama.
$('modo-openrouter').onclick = () => {
  $('base_url').value = configAtual.openrouter_base_url;
  limparModeloSeIncompativel(false);
  marcarModoAtivo();
};
$('modo-local').onclick = () => {
  $('base_url').value = configAtual.local_base_url;
  limparModeloSeIncompativel(true);
  marcarModoAtivo();
};
$('base_url').addEventListener('input', marcarModoAtivo);

$('buscar-modelos').onclick = async () => {
  const botao = $('buscar-modelos');
  const estado = $('estado-modelos');
  const lista = $('lista-modelos');
  botao.disabled = true;
  estado.textContent = 'Buscando...';
  lista.hidden = true;
  lista.innerHTML = '';

  const r = await fetch('/api/modelos?base_url=' + encodeURIComponent($('base_url').value));
  const dados = await r.json();
  botao.disabled = false;

  if (!r.ok) {
    estado.textContent = dados.erro || 'Não consegui buscar os modelos.';
    return;
  }
  if (!dados.modelos.length) {
    estado.textContent = 'Nenhum modelo encontrado nesse endereço.';
    return;
  }

  estado.textContent = `${dados.modelos.length} modelo(s) encontrado(s). Clique para escolher.`;
  for (const modelo of dados.modelos) {
    const item = document.createElement('button');
    item.type = 'button';
    item.innerHTML = `${modelo.id}` +
      (modelo.preco ? `<span class="preco">${modelo.preco}</span>` : '');
    item.onclick = () => {
      $('modelo').value = modelo.id;
      lista.hidden = true;
    };
    lista.appendChild(item);
  }
  lista.hidden = false;
};

$('salvar').onclick = async () => {
  const baseUrl = $('base_url').value.trim();
  const modelo = $('modelo').value.trim();

  // O modelo padrão embutido só existe no catálogo do OpenRouter; salvar
  // vazio em qualquer outro endereço fazia a tela voltar a mostrar aquele
  // nome como se fosse o configurado, sem ninguém ter digitado isso -- daí
  // a impressão de que o modelo local "sumia" e virava um nome do OpenRouter.
  if (!modelo && baseUrl !== configAtual.openrouter_base_url) {
    alert('Informe um modelo antes de salvar. Sem isso a tradução não vai '
        + 'funcionar nesse endereço -- o "deepseek/deepseek-chat" padrão só '
        + 'existe no catálogo do OpenRouter.');
    $('modelo').focus();
    return;
  }

  const corpo = {modelo: modelo, base_url: baseUrl,
                 llm_block_size: $('block_size').value.trim()};
  if ($('chave').value) {
    corpo.chave = $('chave').value;
  } else if (!configAtual.tem_chave && $('base_url').value === configAtual.local_base_url) {
    // Servidor local (Ollama e afins) normalmente ignora a chave, mas o
    // cliente exige algum valor não vazio -- sem isso o usuário bateria
    // num erro de "falta a chave" só por escolher o modo grátis.
    corpo.chave = 'local';
  }
  if ($('chave_tmdb').value) corpo.chave_tmdb = $('chave_tmdb').value;

  $('salvar').disabled = true;
  const r = await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(corpo)
  });
  const dados = await r.json();
  $('salvar').disabled = false;

  if (!r.ok) { alert(dados.erro); return; }
  if (dados.aviso) alert(dados.aviso);
  $('chave').value = '';
  $('chave_tmdb').value = '';
  carregarConfig();
  mostrarSucesso('mensagem-salvar');
};

$('salvar-vad').onclick = async () => {
  const corpo = {
    normalize_audio: $('normalize_audio').value,
    whisper_language: $('whisper_language').value.trim(),
    diarizar: $('diarizar').checked,
    whisper_model: $('whisper_model').value.trim(),
    whisper_compute_type: $('whisper_compute_type').value.trim(),
    vad_method: $('vad_method').value.trim(),
    vad_threshold: $('vad_threshold').value.trim(),
    vad_min_silence_ms: $('vad_min_silence_ms').value.trim(),
  };

  $('salvar-vad').disabled = true;
  const r = await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(corpo)
  });
  const dados = await r.json();
  $('salvar-vad').disabled = false;

  if (!r.ok) { alert(dados.erro); return; }
  carregarConfig();
  mostrarSucesso('mensagem-salvar-vad');
};

// Salvar sem aviso nenhum parecia não ter feito nada -- essa é a única
// confirmação de que a gravação realmente aconteceu. A mensagem original de
// cada rodapé é lida uma vez e guardada no próprio elemento, pra função
// servir os dois painéis sem precisar de uma variável por painel.
function mostrarSucesso(idMensagem) {
  const mensagem = $(idMensagem);
  if (mensagem.dataset.textoPadrao === undefined) {
    mensagem.dataset.textoPadrao = mensagem.textContent;
  }
  mensagem.textContent = 'Configuração salva.';
  mensagem.classList.add('sucesso');
  clearTimeout(Number(mensagem.dataset.timeoutId) || undefined);
  mensagem.dataset.timeoutId = setTimeout(() => {
    mensagem.textContent = mensagem.dataset.textoPadrao;
    mensagem.classList.remove('sucesso');
  }, 4000);
}

carregarConfig();
carregarArquivos();
atualizar();
setInterval(atualizar, 2000);
</script>
</body>
</html>
"""


def main(argv=None):
    """Sobe o servidor web."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="autosrt-web", description="Interface web do AutoSRT.")
    parser.add_argument("--pasta", help="pasta com os filmes e legendas")
    parser.add_argument("--porta", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0",
                        help="0.0.0.0 deixa acessível na rede local")
    parser.add_argument("--motor", choices=[pipeline.ENGINE_LLM,
                                            pipeline.ENGINE_GOOGLE],
                        default=pipeline.ENGINE_LLM)
    parser.add_argument("--verboso", action="store_true",
                        help="mostra o comando montado para o Whisper e "
                             "quantas legendas cada transcrição rendeu, "
                             "útil para entender um resultado inesperado")
    args = parser.parse_args(argv)

    # Sem isso os logger.info do pacote não saem em lugar nenhum, e um
    # resultado estranho vira adivinhação: não dá para comparar o comando
    # que o AutoSRT monta com um digitado à mão se ele nunca é mostrado.
    logging.basicConfig(
        level=logging.DEBUG if args.verboso else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = create_app(media_dir=args.pasta, engine=args.motor)
    print(f"AutoSRT em http://{args.host}:{args.porta}")
    print(f"Pasta de trabalho: {app.config['MEDIA_DIR']}")
    app.run(host=args.host, port=args.porta, threaded=True)


if __name__ == "__main__":
    main()
