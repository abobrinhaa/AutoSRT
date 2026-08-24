"""Interface web do AutoSRT.

Pensada para uso em rede local por quem não abre terminal: escolher o
arquivo, apertar um botão, baixar a legenda.

Duas formas de entrada, pelo motivo prático de tamanho. Filme tem vários
gigabytes e enviar isso pelo navegador é lento e pesado, então vídeo é
escolhido de uma pasta do servidor — o arquivo chega lá por cópia de rede.
Legenda tem alguns quilobytes e pode ser enviada direto pela página.
"""

import io
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
            cancel_event=job.cancelar, translate=(acao != "transcrever"))
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
        return jsonify({
            "tem_chave": bool(chave),
            "chave_do_ambiente": bool(os.environ.get("OPENROUTER_API_KEY")),
            "modelo": config.get_setting("llm_model", "LLM_MODEL") or llm.DEFAULT_MODEL,
            "base_url": config.get_setting("llm_base_url", "LLM_BASE_URL")
                        or llm.DEFAULT_BASE_URL,
            "tem_chave_tmdb": bool(config.get_tmdb_api_key()),
            # Endereços padrão dos dois modos, para o botão de alternar na
            # página não precisar adivinhar nem duplicar essas constantes.
            "openrouter_base_url": llm.DEFAULT_BASE_URL,
            "local_base_url": llm.LOCAL_BASE_URL,
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
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 16px/1.5 system-ui, -apple-system, sans-serif;
         background: #1b1b1f; color: #e8e8ea; }
  main { max-width: 780px; margin: 0 auto; padding: 24px 16px 64px; }
  h1 { font-size: 24px; margin: 8px 0 4px; }
  p.sub { color: #9a9aa2; margin: 0 0 28px; }
  h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .06em;
       color: #9a9aa2; margin: 32px 0 12px; }
  .drop { border: 2px dashed #3a3a42; border-radius: 12px; padding: 28px 16px;
          text-align: center; color: #9a9aa2; cursor: pointer;
          transition: border-color .15s, background .15s; }
  .drop:hover, .drop.ativo { border-color: #3b82f6; background: #21212a; }
  .drop #dica { margin: 4px 0 14px; font-size: 14px; }
  .tag.legenda-pronta { color: #4ade80; }
  .lista { border: 1px solid #2e2e36; border-radius: 12px; overflow: hidden; }
  .item { display: flex; align-items: center; gap: 12px; padding: 12px 14px;
          border-bottom: 1px solid #2e2e36; }
  .item:last-child { border-bottom: 0; }
  /* min-width impede que o nome seja espremido até sumir quando a linha
     acumula selos e um seletor longo; os selos não encolhem, o nome sim. */
  .item .nome { flex: 1 1 auto; min-width: 9ch; overflow: hidden;
                text-overflow: ellipsis; white-space: nowrap; }
  .item .tag { flex-shrink: 0; }
  /* Marca o que acabou de subir, para o usuário achar onde clicar. */
  .item.novo { background: #1e2536; box-shadow: inset 3px 0 0 #3b82f6; }
  .item .acao { flex-shrink: 1; max-width: 220px; }
  /* Selo de reconhecimento do TMDB: pôster pequeno + título (ano). É um
     palpite pelo nome do arquivo, por isso fica discreto, não em destaque. */
  .filme { display: flex; align-items: center; gap: 6px; flex-shrink: 0;
           max-width: 200px; }
  .filme .poster { width: 24px; height: 34px; object-fit: cover;
                   border-radius: 3px; flex-shrink: 0; background: #2a2a33; }
  .filme .titulo { font-size: 12px; color: #9a9aa2; overflow: hidden;
                   text-overflow: ellipsis; white-space: nowrap; }
  #arquivo { display: none; }
  #escolher { display: inline-block; padding: 8px 16px; background: #3b82f6;
              color: white; border-radius: 6px; cursor: pointer; font-weight: 500;
              border: none; }
  .pasta { padding: 8px 14px; background: #23232b; font-size: 13px;
           color: #9a9aa2; border-bottom: 1px solid #2e2e36; }
  .barra-acoes { display: flex; align-items: center; gap: 12px;
                 margin-bottom: 10px; padding: 10px 14px; border-radius: 10px;
                 background: #23232b; }
  .barra-acoes .conta { flex: 1; color: #9a9aa2; font-size: 14px; }
  select { font: inherit; background: #2a2a33; color: #e8e8ea;
           border: 1px solid #3a3a42; border-radius: 8px; padding: 6px 8px; }
  input[type=checkbox] { width: 17px; height: 17px; accent-color: #3b82f6;
                         cursor: pointer; }
  label.tudo { display: flex; align-items: center; gap: 8px; cursor: pointer;
               color: #9a9aa2; font-size: 14px; }
  .tag { font-size: 12px; color: #9a9aa2; background: #2a2a33;
         padding: 2px 8px; border-radius: 999px; }
  button { font: inherit; border: 0; border-radius: 8px; padding: 8px 16px;
           background: #3b82f6; color: #fff; cursor: pointer; }
  button:hover { background: #2f6fd8; }
  button.fantasma { background: #2a2a33; color: #d0d0d6; }
  button:disabled { opacity: .5; cursor: default; }
  .barra { height: 6px; background: #2a2a33; border-radius: 999px;
           overflow: hidden; margin-top: 8px; }
  .barra > div { height: 100%; background: #3b82f6; width: 0;
                 transition: width .3s; }
  .trabalho { border: 1px solid #2e2e36; border-radius: 12px;
              padding: 14px; margin-bottom: 10px; }
  .trabalho .topo { display: flex; align-items: center; gap: 12px; }
  .etapa { color: #9a9aa2; font-size: 14px; margin-top: 6px; }
  .erro { color: #f87171; }
  .ok { color: #4ade80; }
  .vazio { color: #9a9aa2; padding: 20px; text-align: center; }
  a.baixar { color: #3b82f6; text-decoration: none; font-weight: 500; }
  .item a.baixar { flex-shrink: 0; }
  .apagar { flex-shrink: 0; background: none; border: 1px solid #4a2b2b;
            color: #f87171; }
  .apagar:hover:enabled { background: #2a1c1c; border-color: #7f1d1d; }
  .dispensar { font-size: 18px; line-height: 1; padding: 2px 8px; }
  h2 #limpar { float: right; text-transform: none; letter-spacing: 0;
               font-size: 13px; }
  details#config { margin-top: 36px; border: 1px solid #2e2e36;
                   border-radius: 12px; }
  details#config summary { padding: 14px; cursor: pointer; color: #9a9aa2;
                           font-size: 14px; }
  details#config .painel { padding: 0 14px 14px; display: grid; gap: 12px; }
  details#config label { display: grid; gap: 6px; font-size: 14px;
                         color: #9a9aa2; }
  details#config input { font: inherit; background: #2a2a33; color: #e8e8ea;
                         border: 1px solid #3a3a42; border-radius: 8px;
                         padding: 8px 10px; }
  .rodape { display: flex; align-items: center; gap: 12px; }
  .dica-config { flex: 1; font-size: 13px; color: #6f6f78; }
  .selo { font-size: 12px; padding: 2px 8px; border-radius: 999px;
          background: #2a2a33; }
  .selo.ok { color: #4ade80; }
  .selo.falta { color: #fbbf24; }
  .modo-provedor { display: flex; gap: 8px; }
  .modo-provedor button { flex: 1; background: #2a2a33; color: #d0d0d6;
                          border: 1px solid #3a3a42; }
  .modo-provedor button.ativo { background: #21304f; border-color: #3b82f6;
                                color: #e8e8ea; }
  .modo-provedor .tag { margin-left: 6px; }
  /* Cor sozinha não é garantia de dar para notar; o texto "selecionado"
     deixa explícito qual dos dois modos está valendo agora. */
  .modo-provedor .marca-selecionado { display: none; color: #4ade80;
                                      font-size: 12px; }
  .modo-provedor button.ativo .marca-selecionado { display: inline; }
  #mensagem-salvar.sucesso { color: #4ade80; }
  .linha-modelo { display: flex; gap: 8px; }
  .linha-modelo input { flex: 1; }
  .lista-modelos { max-height: 220px; overflow-y: auto; border: 1px solid #3a3a42;
                   border-radius: 8px; }
  .lista-modelos button { display: block; width: 100%; text-align: left;
                          background: none; border: 0; border-radius: 0;
                          border-bottom: 1px solid #2e2e36; color: #e8e8ea;
                          padding: 8px 10px; font-size: 13px; }
  .lista-modelos button:last-child { border-bottom: 0; }
  .lista-modelos button:hover { background: #23232b; }
  .lista-modelos .preco { display: block; color: #9a9aa2; font-size: 12px;
                          margin-top: 2px; }
</style>
</head>
<body>
<main>
  <h1>AutoSRT</h1>
  <p class="sub">Transcreve o áudio do filme e traduz a legenda para português.</p>

  <div class="drop" id="drop">
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
  </div>
  <div class="lista" id="lista"><div class="vazio">Carregando...</div></div>

  <h2>Trabalhos <button id="limpar" class="fantasma" hidden>Limpar terminados</button></h2>
  <div id="trabalhos"><div class="vazio">Nada ainda.</div></div>

  <details id="config">
    <summary>Configura&ccedil;&atilde;o da tradu&ccedil;&atilde;o <span id="estado-chave"></span></summary>
    <div class="painel">
      <label><span id="rotulo-endereco">Endere&ccedil;o da API</span>
        <input type="text" id="base_url" placeholder="https://openrouter.ai/api/v1">
      </label>
      <label><span id="rotulo-chave">Chave do OpenRouter</span>
        <input type="password" id="chave" placeholder="sk-or-v1-..." autocomplete="off"
              data-exemplo-openrouter="sk-or-v1-..."
              data-exemplo-local="normalmente dispens&aacute;vel aqui">
      </label>
      <label>Modelo
        <div class="linha-modelo">
          <input type="text" id="modelo" placeholder="deepseek/deepseek-chat" data-exemplo-openrouter="deepseek/deepseek-chat" data-exemplo-local="llama3.1">
          <button type="button" id="buscar-modelos" class="fantasma">Buscar modelos</button>
        </div>
      </label>
      <span class="dica-config" id="estado-modelos"></span>
      <div id="lista-modelos" class="lista-modelos" hidden></div>
      <label>Chave do TMDB <span id="estado-chave-tmdb"></span>
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
</main>

<script>
const $ = (id) => document.getElementById(id);

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

  // O campo é o mesmo nos dois modos (é sempre "openrouter_api_key" na
  // configuração), mas o rótulo e o exemplo "OpenRouter" ficam errados
  // depois de trocar para Local -- só o texto ao redor muda, não o campo.
  $('rotulo-chave').textContent = local
    ? 'Chave (opcional em modo local)' : 'Chave do OpenRouter';
  const chaveInput = $('chave');
  chaveInput.placeholder = local
    ? chaveInput.dataset.exemploLocal : chaveInput.dataset.exemploOpenrouter;

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

  const seloTmdb = $('estado-chave-tmdb');
  seloTmdb.className = 'selo ' + (c.tem_chave_tmdb ? 'ok' : 'falta');
  seloTmdb.textContent = c.tem_chave_tmdb ? 'ativo' : 'sem chave';

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
  const corpo = {modelo: $('modelo').value, base_url: $('base_url').value};
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
  mostrarSucessoDoSalvar();
};

// Salvar sem aviso nenhum parecia não ter feito nada -- essa é a única
// confirmação de que a gravação realmente aconteceu.
const DICA_SALVAR_PADRAO = $('mensagem-salvar').textContent;
let limparMensagemSalvar = null;

function mostrarSucessoDoSalvar() {
  const mensagem = $('mensagem-salvar');
  mensagem.textContent = 'Configuração salva.';
  mensagem.classList.add('sucesso');
  if (limparMensagemSalvar) clearTimeout(limparMensagemSalvar);
  limparMensagemSalvar = setTimeout(() => {
    mensagem.textContent = DICA_SALVAR_PADRAO;
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
    args = parser.parse_args(argv)

    app = create_app(media_dir=args.pasta, engine=args.motor)
    print(f"AutoSRT em http://{args.host}:{args.porta}")
    print(f"Pasta de trabalho: {app.config['MEDIA_DIR']}")
    app.run(host=args.host, port=args.porta, threaded=True)


if __name__ == "__main__":
    main()
