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

from . import pipeline, srt_io
from .jobs import JobQueue

DEFAULT_MEDIA_DIR = "midia"
DEFAULT_PORT = 8000
# Legendas são pequenas; o limite existe para barrar envio acidental de vídeo.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def create_app(media_dir=None, engine=pipeline.ENGINE_LLM):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    media_dir = os.path.abspath(
        media_dir or os.environ.get("AUTOSRT_MEDIA_DIR", DEFAULT_MEDIA_DIR))
    os.makedirs(media_dir, exist_ok=True)
    app.config["MEDIA_DIR"] = media_dir

    fila = JobQueue(lambda job: _executar(job, engine))
    app.config["FILA"] = fila

    _register_routes(app, fila, media_dir, engine)
    return app


def _executar(job, engine):
    """Roda um trabalho. Chamado pelo operário da fila."""
    def status(mensagem):
        job.etapa = mensagem

    def progresso(feitas, total):
        if total:
            job.progresso = min(100, int(feitas * 100 / total))

    if pipeline.is_media(job.entrada):
        resultado = pipeline.process_media(
            job.entrada, engine=engine, status=status, progress=progresso,
            cancel_event=job.cancelar)
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


def _dentro_da_pasta(caminho, pasta) -> bool:
    """Barra caminhos que escapam da pasta de trabalho."""
    caminho = os.path.abspath(caminho)
    pasta = os.path.abspath(pasta)
    return caminho == pasta or caminho.startswith(pasta + os.sep)


def _listar_arquivos(media_dir) -> list:
    itens = []
    for raiz, _, nomes in os.walk(media_dir):
        for nome in sorted(nomes):
            caminho = os.path.join(raiz, nome)
            if not (pipeline.is_media(caminho) or pipeline.is_subtitle(caminho)):
                continue
            minusculo = nome.lower()
            if minusculo.endswith("_backup.srt") or minusculo.endswith(".original.srt"):
                continue
            itens.append({
                "nome": os.path.relpath(caminho, media_dir),
                "tamanho": _tamanho_legivel(os.path.getsize(caminho)),
                "tipo": "video" if pipeline.is_media(caminho) else "legenda",
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
        return PAGINA

    @app.get("/api/arquivos")
    def arquivos():
        return jsonify(_listar_arquivos(media_dir))

    @app.post("/api/processar")
    def processar():
        nome = (request.json or {}).get("arquivo", "")
        caminho = os.path.join(media_dir, nome)
        if not nome or not _dentro_da_pasta(caminho, media_dir):
            return jsonify({"erro": "Arquivo inválido."}), 400
        if not os.path.isfile(caminho):
            return jsonify({"erro": "Esse arquivo não está mais na pasta."}), 404

        job = fila.enviar(os.path.basename(nome), caminho)
        return jsonify(job.para_json()), 202

    @app.post("/api/enviar")
    def enviar():
        arquivo = request.files.get("arquivo")
        if not arquivo or not arquivo.filename:
            return jsonify({"erro": "Nenhum arquivo recebido."}), 400

        nome = os.path.basename(arquivo.filename)
        if not pipeline.is_subtitle(nome):
            return jsonify({
                "erro": "Aqui só entram legendas (.srt, .ssa, .ass). "
                        "Vídeo é grande demais para enviar pelo navegador: "
                        "copie para a pasta do servidor e ele aparece na lista."
            }), 400

        destino = os.path.join(media_dir, nome)
        arquivo.save(destino)
        job = fila.enviar(nome, destino)
        return jsonify(job.para_json()), 202

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
        return jsonify({
            "erro": "Arquivo grande demais. Pela página só dá para enviar "
                    "legendas; copie o vídeo para a pasta do servidor."
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
  .lista { border: 1px solid #2e2e36; border-radius: 12px; overflow: hidden; }
  .item { display: flex; align-items: center; gap: 12px; padding: 12px 14px;
          border-bottom: 1px solid #2e2e36; }
  .item:last-child { border-bottom: 0; }
  .item .nome { flex: 1; min-width: 0; overflow: hidden;
                text-overflow: ellipsis; white-space: nowrap; }
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
</style>
</head>
<body>
<main>
  <h1>AutoSRT</h1>
  <p class="sub">Transcreve o áudio do filme e traduz a legenda para português.</p>

  <div class="drop" id="drop">
    <strong>Arraste uma legenda aqui</strong><br>
    .srt, .ssa ou .ass &mdash; ou clique para escolher
    <input type="file" id="arquivo" accept=".srt,.ssa,.ass" hidden>
  </div>

  <h2>Arquivos no servidor</h2>
  <div class="lista" id="lista"><div class="vazio">Carregando...</div></div>

  <h2>Trabalhos</h2>
  <div id="trabalhos"><div class="vazio">Nada ainda.</div></div>
</main>

<script>
const $ = (id) => document.getElementById(id);

async function carregarArquivos() {
  const r = await fetch('/api/arquivos');
  const itens = await r.json();
  const lista = $('lista');
  if (!itens.length) {
    lista.innerHTML = '<div class="vazio">Nenhum arquivo na pasta do servidor.</div>';
    return;
  }
  lista.innerHTML = '';
  for (const item of itens) {
    const div = document.createElement('div');
    div.className = 'item';
    div.innerHTML = `<span class="nome"></span>
      <span class="tag"></span><span class="tag"></span>
      <button>Processar</button>`;
    div.querySelector('.nome').textContent = item.nome;
    div.querySelectorAll('.tag')[0].textContent = item.tipo;
    div.querySelectorAll('.tag')[1].textContent = item.tamanho;
    div.querySelector('button').onclick = () => processar(item.nome, div);
    lista.appendChild(div);
  }
}

async function processar(nome, div) {
  const botao = div.querySelector('button');
  botao.disabled = true;
  botao.textContent = 'Enviado';
  const r = await fetch('/api/processar', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({arquivo: nome})
  });
  const dados = await r.json();
  if (!r.ok) { alert(dados.erro); botao.disabled = false; botao.textContent = 'Processar'; }
  atualizar();
}

$('drop').onclick = () => $('arquivo').click();
$('arquivo').onchange = (e) => { if (e.target.files[0]) enviar(e.target.files[0]); };
$('drop').ondragover = (e) => { e.preventDefault(); $('drop').classList.add('ativo'); };
$('drop').ondragleave = () => $('drop').classList.remove('ativo');
$('drop').ondrop = (e) => {
  e.preventDefault();
  $('drop').classList.remove('ativo');
  if (e.dataTransfer.files[0]) enviar(e.dataTransfer.files[0]);
};

async function enviar(arquivo) {
  const dados = new FormData();
  dados.append('arquivo', arquivo);
  const r = await fetch('/api/enviar', {method: 'POST', body: dados});
  const resposta = await r.json().catch(() => ({erro: 'Falha no envio.'}));
  if (!r.ok) alert(resposta.erro);
  carregarArquivos();
  atualizar();
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
