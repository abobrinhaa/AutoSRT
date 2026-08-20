"""Interface web do AutoSRT.

Pensada para uso em rede local por quem não abre terminal: escolher o
arquivo, apertar um botão, baixar a legenda.

Duas formas de entrada, pelo motivo prático de tamanho. Filme tem vários
gigabytes e enviar isso pelo navegador é lento e pesado, então vídeo é
escolhido de uma pasta do servidor — o arquivo chega lá por cópia de rede.
Legenda tem alguns quilobytes e pode ser enviada direto pela página.
"""

import os

from flask import (Flask, jsonify, request, send_file)

from . import config, llm, pipeline, srt_io, sync
from .jobs import JobQueue

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


def create_app(media_dir=None, engine=pipeline.ENGINE_LLM, max_upload_gb=None):
    app = Flask(__name__)

    if max_upload_gb is None:
        max_upload_gb = float(os.environ.get("AUTOSRT_MAX_UPLOAD_GB",
                                             DEFAULT_MAX_UPLOAD_GB))
    app.config["MAX_CONTENT_LENGTH"] = int(max_upload_gb * 1024 ** 3)
    app.config["MAX_UPLOAD_GB"] = max_upload_gb

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

    acoes = list(ACOES_LEGENDA)
    if os.path.splitext(caminho)[1].lower() in {".ssa", ".ass"}:
        acoes.insert(1, ACAO_CONVERTER)
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
            irma = legenda_irma(caminho) if pipeline.is_media(caminho) else None
            itens.append({
                "nome": relativo,
                "pasta": os.path.dirname(relativo) or ".",
                "tamanho": _tamanho_legivel(os.path.getsize(caminho)),
                "tipo": "video" if pipeline.is_media(caminho) else "legenda",
                "tem_legenda": bool(irma),
                "acoes": acoes_para(caminho),
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
        """Estado da configuração. A chave nunca volta para o navegador."""
        chave = config.get_openrouter_api_key()
        return jsonify({
            "tem_chave": bool(chave),
            "chave_do_ambiente": bool(os.environ.get("OPENROUTER_API_KEY")),
            "modelo": config.get_setting("llm_model", "LLM_MODEL") or llm.DEFAULT_MODEL,
            "base_url": config.get_setting("llm_base_url", "LLM_BASE_URL")
                        or llm.DEFAULT_BASE_URL,
        })

    @app.post("/api/config")
    def gravar_config():
        dados = request.json or {}
        novos = {}

        if "chave" in dados:
            novos["openrouter_api_key"] = (dados.get("chave") or "").strip()
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

        return jsonify({"ok": True, "aviso": _aviso_de_ambiente(novos)})

    def _aviso_de_ambiente(novos):
        """Variável de ambiente vence o arquivo; avisar evita confusão."""
        if novos.get("openrouter_api_key") and os.environ.get("OPENROUTER_API_KEY"):
            return ("Salvo, mas a variável de ambiente OPENROUTER_API_KEY tem "
                    "prioridade e continuará sendo usada. Remova-a para valer "
                    "a chave gravada aqui.")
        return None

    @app.get("/api/trabalhos")
    def trabalhos():
        return jsonify([j.para_json() for j in fila.listar()])

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
</style>
</head>
<body>
<main>
  <h1>AutoSRT</h1>
  <p class="sub">Transcreve o áudio do filme e traduz a legenda para português.</p>

  <div class="drop" id="drop">
    <strong>Arraste o filme e a legenda aqui</strong>
    <div id="dica">pode mandar os dois juntos &mdash; v&iacute;deo, &aacute;udio ou legenda</div>
    <button id="escolher" type="button">Escolher arquivos...</button>
    <div class="barra" id="barra-envio" hidden><div></div></div>
    <input type="file" id="arquivo" accept="{{ACCEPT}}" multiple hidden>
  </div>

  <h2>Arquivos no servidor</h2>
  <div class="barra-acoes" id="barra-acoes" hidden>
    <label class="tudo"><input type="checkbox" id="tudo"> Selecionar todos</label>
    <span class="conta" id="conta"></span>
    <button id="lote" disabled>Processar selecionados</button>
  </div>
  <div class="lista" id="lista"><div class="vazio">Carregando...</div></div>

  <h2>Trabalhos</h2>
  <div id="trabalhos"><div class="vazio">Nada ainda.</div></div>

  <details id="config">
    <summary>Configura&ccedil;&atilde;o da tradu&ccedil;&atilde;o <span id="estado-chave"></span></summary>
    <div class="painel">
      <label>Chave do OpenRouter
        <input type="password" id="chave" placeholder="sk-or-v1-..." autocomplete="off">
      </label>
      <label>Modelo
        <input type="text" id="modelo" placeholder="deepseek/deepseek-chat-v2.5">
      </label>
      <label>Endere&ccedil;o da API
        <input type="text" id="base_url" placeholder="https://openrouter.ai/api/v1">
      </label>
      <div class="rodape">
        <span class="dica-config">A chave fica no config.json do servidor e nunca volta para esta p&aacute;gina.</span>
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

const abrirSeletor = () => $('arquivo').click();
$('escolher').onclick = (e) => { e.stopPropagation(); abrirSeletor(); };
$('drop').onclick = abrirSeletor;
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
      alert(resposta.recusados.map((x) => x.erro).join('\n'));
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
      (d.idioma ? ` \\u00b7 ${d.idioma}` : '') +
      (d.falhas ? ` \\u00b7 ${d.falhas} não traduzida(s)` : '');
  } else {
    etapa.textContent = job.etapa;
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
  if (!jobs.length) {
    alvo.innerHTML = '<div class="vazio">Nada ainda.</div>';
    return;
  }
  alvo.innerHTML = '';
  jobs.forEach((job) => alvo.appendChild(cartao(job)));
}

async function carregarConfig() {
  const r = await fetch('/api/config');
  const c = await r.json();
  $('modelo').value = c.modelo || '';
  $('base_url').value = c.base_url || '';

  const selo = $('estado-chave');
  selo.className = 'selo ' + (c.tem_chave ? 'ok' : 'falta');
  selo.textContent = c.tem_chave
    ? (c.chave_do_ambiente ? 'chave do ambiente' : 'chave configurada')
    : 'sem chave';

  // Sem chave, a tradução não funciona: abre o painel para não deixar o
  // usuário descobrir isso só quando o primeiro trabalho falhar.
  if (!c.tem_chave) $('config').open = true;
}

$('salvar').onclick = async () => {
  const corpo = {modelo: $('modelo').value, base_url: $('base_url').value};
  if ($('chave').value) corpo.chave = $('chave').value;

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
  carregarConfig();
};

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
