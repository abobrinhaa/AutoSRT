import io
import logging
import os
import re
import shutil
import tempfile
import time
import unittest
from unittest import mock

from autosrt import jobs, pipeline, web

# Os testes de erro provocam falhas de propósito; o traceback delas no log
# só poluiria a saída da suíte.
logging.getLogger("autosrt.jobs").setLevel(logging.CRITICAL)

SRT = """1
00:00:01,000 --> 00:00:03,000
This is an English sentence for the test.
"""


class EchoLLM:
    def complete(self, system, user):
        from autosrt import llm_translate
        return "\n".join(f"<{n}>PT:{c.strip()}</{n}>"
                         for n, c in llm_translate.BLOCK_RE.findall(user))


class BaseWeb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.app = web.create_app(media_dir=self.tmp)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def escrever(self, nome, conteudo=SRT):
        caminho = os.path.join(self.tmp, nome)
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as handle:
            handle.write(conteudo)
        return caminho

    def tocar(self, nome):
        caminho = os.path.join(self.tmp, nome)
        with open(caminho, "wb") as handle:
            handle.write(b"\0")
        return caminho

    def esperar(self, job_id, limite=5):
        fim = time.time() + limite
        while time.time() < fim:
            dados = self.client.get(f"/api/trabalho/{job_id}").get_json()
            if dados["pronto"]:
                return dados
            time.sleep(0.05)
        self.fail("o trabalho não terminou a tempo")


class TestPagina(BaseWeb):
    def test_pagina_abre(self):
        resposta = self.client.get("/")
        self.assertEqual(resposta.status_code, 200)
        self.assertIn("AutoSRT", resposta.get_data(as_text=True))

    def test_pagina_nao_puxa_recurso_externo(self):
        # Sem internet na rede local, um CDN deixaria a página quebrada. O que
        # importa é o que a página CARREGA - endereço em placeholder de campo
        # é só texto e não busca nada.
        html = self.client.get("/").get_data(as_text=True)
        for padrao in ("<script src=", "<link ", "@import", "url(http",
                       'src="http', "src='http"):
            self.assertNotIn(padrao, html, f"a página carrega algo externo: {padrao}")

    def test_filtro_do_seletor_reflete_o_que_o_servidor_aceita(self):
        # Filtro desatualizado esconde arquivo válido no explorador sem dar
        # nenhuma pista de que ele existe.
        html = self.client.get("/").get_data(as_text=True)
        aceitas = re.search(r'id="arquivo" accept="([^"]+)"', html).group(1)
        do_seletor = set(aceitas.split(","))
        do_servidor = pipeline.MEDIA_EXTENSIONS | pipeline.SUBTITLE_EXTENSIONS
        self.assertEqual(do_seletor, do_servidor)

    def test_seletor_aceita_mais_de_um_arquivo(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("multiple", html)

    def test_nenhum_marcador_de_template_sobra_na_pagina(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertNotIn("{{", html)

    def test_requisicoes_do_script_sao_todas_relativas(self):
        html = self.client.get("/").get_data(as_text=True)
        for chamada in re.findall(r"fetch\(\s*['\"]([^'\"]+)", html):
            self.assertTrue(chamada.startswith("/"),
                            f"fetch para endereço não relativo: {chamada}")


class TestListagem(BaseWeb):
    def test_lista_videos_e_legendas(self):
        self.escrever("a.srt")
        self.tocar("b.mkv")
        itens = self.client.get("/api/arquivos").get_json()
        self.assertEqual({i["nome"] for i in itens}, {"a.srt", "b.mkv"})

    def test_classifica_o_tipo(self):
        self.escrever("a.srt")
        self.tocar("b.mkv")
        tipos = {i["nome"]: i["tipo"] for i in self.client.get("/api/arquivos").get_json()}
        self.assertEqual(tipos, {"a.srt": "legenda", "b.mkv": "video"})

    def test_esconde_backup_e_original(self):
        self.escrever("filme.srt")
        self.escrever("filme_backup.srt")
        self.escrever("filme.original.srt")
        itens = self.client.get("/api/arquivos").get_json()
        self.assertEqual([i["nome"] for i in itens], ["filme.srt"])

    def test_ignora_arquivo_de_outro_tipo(self):
        self.tocar("nota.txt")
        self.assertEqual(self.client.get("/api/arquivos").get_json(), [])

    def test_encontra_em_subpasta(self):
        self.escrever(os.path.join("classicos", "a.srt"))
        itens = self.client.get("/api/arquivos").get_json()
        self.assertEqual(itens[0]["nome"], os.path.join("classicos", "a.srt"))

    def test_esconde_a_pasta_de_originais(self):
        self.escrever("filme.srt")
        self.escrever(os.path.join(pipeline.ORIGINALS_DIRNAME, "filme.srt"))
        itens = self.client.get("/api/arquivos").get_json()
        self.assertEqual([i["nome"] for i in itens], ["filme.srt"])


class TestReconhecimentoTMDB(BaseWeb):
    """O selo de filme na listagem é um extra: nunca deve travar nem exigir
    rede quando não há chave configurada, e some sozinho quando o TMDB não
    reconhece nada."""

    def setUp(self):
        super().setUp()
        from autosrt import config, tmdb
        self.tmdb = tmdb
        self._chave_original = config.get_tmdb_api_key
        self._lookup_original = tmdb.lookup_cached
        tmdb.limpar_cache()

    def tearDown(self):
        from autosrt import config
        config.get_tmdb_api_key = self._chave_original
        self.tmdb.lookup_cached = self._lookup_original
        self.tmdb.limpar_cache()

    def test_sem_chave_nao_tem_campo_filme(self):
        from autosrt import config
        config.get_tmdb_api_key = lambda: None
        self.tocar("Duna.2021.mkv")

        itens = self.client.get("/api/arquivos").get_json()
        self.assertIsNone(itens[0]["filme"])

    def test_com_chave_e_reconhecimento_traz_o_filme(self):
        from autosrt import config
        config.get_tmdb_api_key = lambda: "chave-teste"
        self.tmdb.lookup_cached = lambda *a, **kw: {
            "titulo": "Duna", "ano": 2021,
            "poster": "https://image.tmdb.org/t/p/w154/x.jpg",
            "idioma_original": "en", "idioma_original_nome": "Inglês",
        }
        self.tocar("Duna.2021.1080p.mkv")

        itens = self.client.get("/api/arquivos").get_json()
        self.assertEqual(itens[0]["filme"]["titulo"], "Duna")
        self.assertEqual(itens[0]["filme"]["ano"], 2021)

    def test_legenda_nunca_recebe_campo_filme(self):
        from autosrt import config
        config.get_tmdb_api_key = lambda: "chave-teste"
        self.tmdb.lookup_cached = lambda *a, **kw: {"titulo": "Duna", "ano": 2021,
                                                     "poster": None,
                                                     "idioma_original": "en",
                                                     "idioma_original_nome": "Inglês"}
        self.escrever("legenda.srt")

        itens = self.client.get("/api/arquivos").get_json()
        self.assertIsNone(itens[0]["filme"])

    def test_sem_correspondencia_o_selo_nao_aparece(self):
        from autosrt import config
        config.get_tmdb_api_key = lambda: "chave-teste"
        self.tmdb.lookup_cached = lambda *a, **kw: None
        self.tocar("video_qualquer.mkv")

        itens = self.client.get("/api/arquivos").get_json()
        self.assertIsNone(itens[0]["filme"])


class TestSeguranca(BaseWeb):
    def test_recusa_caminho_para_fora_da_pasta(self):
        resposta = self.client.post("/api/processar",
                                    json={"arquivo": "../../etc/passwd"})
        self.assertEqual(resposta.status_code, 400)

    def test_recusa_caminho_absoluto(self):
        resposta = self.client.post("/api/processar",
                                    json={"arquivo": "/etc/passwd"})
        self.assertIn(resposta.status_code, (400, 404))

    def test_recusa_nome_vazio(self):
        self.assertEqual(
            self.client.post("/api/processar", json={"arquivo": ""}).status_code,
            400)

    def test_arquivo_inexistente(self):
        resposta = self.client.post("/api/processar", json={"arquivo": "nada.srt"})
        self.assertEqual(resposta.status_code, 404)


class TestEnvio(BaseWeb):
    def enviar(self, nome, conteudo=SRT):
        return self.client.post(
            "/api/enviar",
            data={"arquivo": (io.BytesIO(conteudo.encode("utf-8")), nome)},
            content_type="multipart/form-data")

    def enviar_varios(self, *pares):
        return self.client.post(
            "/api/enviar",
            data={"arquivo": [(io.BytesIO(c.encode("utf-8")), n)
                              for n, c in pares]},
            content_type="multipart/form-data")

    def test_aceita_legenda(self):
        resposta = self.enviar("filme.srt")
        self.assertEqual(resposta.status_code, 201)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "filme.srt")))

    def test_aceita_video(self):
        # Em rede local, enviar filme pelo navegador leva um ou dois minutos,
        # irrelevante perto do tempo de transcrição.
        resposta = self.enviar("filme.mkv", "conteudo")
        self.assertEqual(resposta.status_code, 201)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "filme.mkv")))

    def test_aceita_filme_e_legenda_de_uma_vez(self):
        resposta = self.enviar_varios(("filme.mkv", "video"), ("filme.srt", SRT))
        self.assertEqual(resposta.status_code, 201)
        self.assertEqual(sorted(resposta.get_json()["guardados"]),
                         ["filme.mkv", "filme.srt"])
        for nome in ("filme.mkv", "filme.srt"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp, nome)))

    def test_o_par_enviado_junto_e_reconhecido(self):
        self.enviar_varios(("filme.mkv", "video"), ("filme.srt", SRT))
        itens = self.client.get("/api/arquivos").get_json()
        video = next(i for i in itens if i["nome"] == "filme.mkv")
        self.assertTrue(video["tem_legenda"])
        self.assertEqual(video["acoes"][0]["id"], "traduzir_existente")

    def test_enviar_nao_processa_nada(self):
        # Enfileirar no envio traduziria a legenda sozinha antes de o vídeo
        # chegar, desperdiçando justamente o pareamento.
        self.enviar_varios(("filme.mkv", "video"), ("filme.srt", SRT))
        self.assertEqual(self.client.get("/api/trabalhos").get_json(), [])

    def test_arquivo_invalido_no_meio_nao_impede_os_validos(self):
        resposta = self.enviar_varios(("filme.srt", SRT), ("nota.txt", "x"))
        self.assertEqual(resposta.status_code, 201)
        dados = resposta.get_json()
        self.assertEqual(dados["guardados"], ["filme.srt"])
        self.assertEqual(len(dados["recusados"]), 1)

    def test_recusa_tipo_desconhecido(self):
        resposta = self.enviar("planilha.xlsx", "conteudo")
        self.assertEqual(resposta.status_code, 400)
        self.assertIn("legenda", resposta.get_json()["erro"])

    def test_limite_de_tamanho_explica_a_alternativa(self):
        app = web.create_app(media_dir=self.tmp, max_upload_gb=0.000001)
        app.config["TESTING"] = True
        cliente = app.test_client()
        resposta = cliente.post(
            "/api/enviar",
            data={"arquivo": (io.BytesIO(b"x" * 5000), "grande.srt")},
            content_type="multipart/form-data")
        self.assertEqual(resposta.status_code, 413)
        self.assertIn("pasta do servidor", resposta.get_json()["erro"])

    def test_recusa_sem_arquivo(self):
        resposta = self.client.post("/api/enviar", data={},
                                    content_type="multipart/form-data")
        self.assertEqual(resposta.status_code, 400)

    def test_nome_com_caminho_e_reduzido(self):
        self.enviar("../../fora.srt")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "fora.srt")))


class TestProcessamento(BaseWeb):
    def setUp(self):
        super().setUp()
        self._original = pipeline._translate_with_llm
        pipeline._translate_with_llm = self._fake_translate

    def tearDown(self):
        pipeline._translate_with_llm = self._original

    @staticmethod
    def _fake_translate(cues, lang, **kwargs):
        for cue in cues:
            cue.text = "PT:" + cue.source_text
        return len(cues), []

    def test_processa_e_disponibiliza_download(self):
        self.escrever("filme.srt")
        job = self.client.post("/api/processar",
                               json={"arquivo": "filme.srt"}).get_json()
        final = self.esperar(job["id"])

        self.assertEqual(final["estado"], jobs.CONCLUIDO)
        self.assertTrue(final["baixavel"])
        self.assertEqual(final["detalhes"]["total"], 1)

        baixado = self.client.get(f"/api/baixar/{job['id']}")
        try:
            self.assertEqual(baixado.status_code, 200)
            self.assertIn("PT:", baixado.get_data(as_text=True))
        finally:
            baixado.close()

    def test_trabalho_aparece_na_lista(self):
        self.escrever("filme.srt")
        self.client.post("/api/processar", json={"arquivo": "filme.srt"})
        trabalhos = self.client.get("/api/trabalhos").get_json()
        self.assertEqual(len(trabalhos), 1)

    def test_erro_vira_mensagem_e_nao_derruba(self):
        self.escrever("vazio.srt", "")
        job = self.client.post("/api/processar",
                               json={"arquivo": "vazio.srt"}).get_json()
        final = self.esperar(job["id"])
        self.assertEqual(final["estado"], jobs.ERRO)
        self.assertTrue(final["erro"])
        # O servidor continua de pé.
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_download_de_trabalho_inexistente(self):
        self.assertEqual(self.client.get("/api/baixar/naoexiste").status_code, 404)


class TestAcoesDisponiveis(unittest.TestCase):
    def ids(self, caminho):
        return [a["id"] for a in web.acoes_para(caminho)]

    def test_video_pode_so_transcrever(self):
        self.assertEqual(self.ids("filme.mkv"), ["completo", "transcrever"])

    def test_legenda_nao_oferece_transcrever(self):
        self.assertNotIn("transcrever", self.ids("filme.srt"))

    def test_legenda_sozinha_so_traduz(self):
        # Sem o filme irmão na pasta, ajustar o tempo não é oferecido.
        self.assertEqual(self.ids("filme.srt"), ["traduzir"])

    def test_legenda_com_filme_tambem_ajusta_o_tempo(self):
        pasta = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, pasta, ignore_errors=True)
        for nome in ("filme.mkv", "filme.srt"):
            open(os.path.join(pasta, nome), "wb").close()

        ids = self.ids(os.path.join(pasta, "filme.srt"))
        self.assertIn("traduzir", ids)
        self.assertIn("deslocar", ids)

    def test_converter_so_aparece_para_ssa(self):
        self.assertIn("converter", self.ids("filme.ssa"))
        self.assertIn("converter", self.ids("filme.ass"))
        self.assertNotIn("converter", self.ids("filme.srt"))


class TestLegendaJaExistente(unittest.TestCase):
    """Vídeo com legenda ao lado não deveria ser transcrito por engano.

    Transcrever leva dezenas de minutos de GPU para produzir o que já está
    no disco; traduzir a legenda existente leva pouco mais de um minuto.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.video = os.path.join(self.tmp, "filme.mkv")
        with open(self.video, "wb") as handle:
            handle.write(b"\0")

    def criar_legenda(self, extensao=".srt"):
        caminho = os.path.join(self.tmp, "filme" + extensao)
        with open(caminho, "w", encoding="utf-8") as handle:
            handle.write(SRT)
        return caminho

    def test_sem_legenda_ao_lado(self):
        self.assertIsNone(web.legenda_irma(self.video))

    def test_encontra_srt_de_mesmo_nome(self):
        esperado = self.criar_legenda(".srt")
        self.assertEqual(web.legenda_irma(self.video), esperado)

    def test_encontra_ssa(self):
        esperado = self.criar_legenda(".ssa")
        self.assertEqual(web.legenda_irma(self.video), esperado)

    def test_srt_tem_preferencia_sobre_ssa(self):
        self.criar_legenda(".ssa")
        srt = self.criar_legenda(".srt")
        self.assertEqual(web.legenda_irma(self.video), srt)

    def test_nome_diferente_nao_conta(self):
        outro = os.path.join(self.tmp, "outro-filme.srt")
        with open(outro, "w", encoding="utf-8") as handle:
            handle.write(SRT)
        self.assertIsNone(web.legenda_irma(self.video))

    def test_acao_de_aproveitar_aparece_primeiro(self):
        self.criar_legenda()
        ids = [a["id"] for a in web.acoes_para(self.video)]
        self.assertEqual(ids[0], "traduzir_existente")

    def test_sem_legenda_a_acao_nao_aparece(self):
        ids = [a["id"] for a in web.acoes_para(self.video)]
        self.assertNotIn("traduzir_existente", ids)

    def test_transcrever_continua_disponivel(self):
        self.criar_legenda()
        ids = [a["id"] for a in web.acoes_para(self.video)]
        self.assertIn("completo", ids)


class TestAcoesIndividuais(BaseWeb):
    def setUp(self):
        super().setUp()
        self._original = pipeline._translate_with_llm
        pipeline._translate_with_llm = lambda cues, lang, **kw: (len(cues), [])

    def tearDown(self):
        pipeline._translate_with_llm = self._original

    def test_deslocar_move_os_tempos_sem_traduzir(self):
        self.escrever("filme.srt")
        self.tocar("filme.mkv")  # o filme irmão é o que libera o deslocar
        job = self.client.post("/api/processar", json={
            "arquivo": "filme.srt", "acao": "deslocar", "segundos": 2.5,
        }).get_json()
        self.esperar(job["id"])

        cues = pipeline.srt_io.load_cues(os.path.join(self.tmp, "filme.srt"))
        self.assertEqual(cues[0].start, 3500)
        # O texto continua em inglês: deslocar não traduz.
        self.assertIn("English", cues[0].source_text)

    def test_deslocar_legenda_sem_filme_e_recusado(self):
        # A regra não é só esconder a opção do menu: o pedido feito na mão,
        # ou vindo de uma página velha aberta antes, também é barrado.
        self.escrever("sozinha.srt")
        resposta = self.client.post("/api/processar", json={
            "arquivo": "sozinha.srt", "acao": "deslocar", "segundos": 2.5,
        })
        self.assertEqual(resposta.status_code, 400)
        self.assertIn("deslocar", resposta.get_json()["erro"])

        # E a legenda fica como estava.
        cues = pipeline.srt_io.load_cues(os.path.join(self.tmp, "sozinha.srt"))
        self.assertEqual(cues[0].start, 1000)

    def test_converter_gera_srt_sem_traduzir(self):
        ssa = ("[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
               "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
               "MarginV, Effect, Text\n"
               "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello there\n")
        self.escrever("filme.ssa", ssa)
        job = self.client.post("/api/processar", json={
            "arquivo": "filme.ssa", "acao": "converter"}).get_json()
        final = self.esperar(job["id"])

        self.assertEqual(final["estado"], jobs.CONCLUIDO)
        destino = os.path.join(self.tmp, "filme.srt")
        self.assertTrue(os.path.exists(destino))
        with open(destino, encoding="utf-8") as handle:
            self.assertIn("Hello there", handle.read())

    def test_acao_invalida_para_o_tipo_e_recusada(self):
        self.escrever("filme.srt")
        resposta = self.client.post("/api/processar", json={
            "arquivo": "filme.srt", "acao": "transcrever"})
        self.assertEqual(resposta.status_code, 400)
        self.assertIn("não vale", resposta.get_json()["erro"])

    def test_acao_inexistente_e_recusada(self):
        self.escrever("filme.srt")
        resposta = self.client.post("/api/processar", json={
            "arquivo": "filme.srt", "acao": "explodir"})
        self.assertEqual(resposta.status_code, 400)

    def test_aproveita_a_legenda_em_vez_de_transcrever(self):
        self.escrever("filme.srt")
        self.tocar("filme.mkv")

        itens = self.client.get("/api/arquivos").get_json()
        video = next(i for i in itens if i["nome"] == "filme.mkv")
        self.assertTrue(video["tem_legenda"])
        self.assertEqual(video["acoes"][0]["id"], "traduzir_existente")

        job = self.client.post("/api/processar", json={
            "arquivo": "filme.mkv", "acao": "traduzir_existente"}).get_json()
        final = self.esperar(job["id"])
        self.assertEqual(final["estado"], jobs.CONCLUIDO)
        self.assertEqual(final["detalhes"]["total"], 1)

    def test_aproveitar_e_recusado_sem_legenda_ao_lado(self):
        self.tocar("filme.mkv")
        resposta = self.client.post("/api/processar", json={
            "arquivo": "filme.mkv", "acao": "traduzir_existente"})
        self.assertEqual(resposta.status_code, 400)


class TestLote(BaseWeb):
    def setUp(self):
        super().setUp()
        self._original = pipeline._translate_with_llm
        pipeline._translate_with_llm = lambda cues, lang, **kw: (len(cues), [])

    def tearDown(self):
        pipeline._translate_with_llm = self._original

    def test_enfileira_varios_de_uma_vez(self):
        for nome in ("a.srt", "b.srt", "c.srt"):
            self.escrever(nome)
        resposta = self.client.post("/api/processar-lote", json={"itens": [
            {"arquivo": "a.srt", "acao": "traduzir"},
            {"arquivo": "b.srt", "acao": "traduzir"},
            {"arquivo": "c.srt", "acao": "traduzir"},
        ]})
        self.assertEqual(resposta.status_code, 202)
        self.assertEqual(len(resposta.get_json()["enfileirados"]), 3)

    def test_pasta_inteira_com_acoes_diferentes(self):
        self.escrever("legenda.srt")
        self.escrever("outra.srt")
        self.tocar("outra.mkv")  # o filme irmão é o que libera o deslocar
        resposta = self.client.post("/api/processar-lote", json={"itens": [
            {"arquivo": "legenda.srt", "acao": "traduzir"},
            {"arquivo": "outra.srt", "acao": "deslocar", "segundos": 1},
        ]})
        jobs_criados = resposta.get_json()["enfileirados"]
        self.assertEqual(len(jobs_criados), 2)
        for job in jobs_criados:
            self.esperar(job["id"])
        # A deslocada não foi traduzida.
        cues = pipeline.srt_io.load_cues(os.path.join(self.tmp, "outra.srt"))
        self.assertEqual(cues[0].start, 2000)

    def test_arquivo_ruim_no_meio_nao_impede_os_outros(self):
        self.escrever("bom.srt")
        resposta = self.client.post("/api/processar-lote", json={"itens": [
            {"arquivo": "bom.srt", "acao": "traduzir"},
            {"arquivo": "sumiu.srt", "acao": "traduzir"},
        ]})
        dados = resposta.get_json()
        self.assertEqual(len(dados["enfileirados"]), 1)
        self.assertEqual(len(dados["recusados"]), 1)

    def test_lote_vazio_e_recusado(self):
        self.assertEqual(
            self.client.post("/api/processar-lote", json={"itens": []}).status_code,
            400)

    def test_lote_recusa_caminho_para_fora(self):
        resposta = self.client.post("/api/processar-lote", json={"itens": [
            {"arquivo": "../../etc/passwd", "acao": "traduzir"}]})
        self.assertEqual(len(resposta.get_json()["recusados"]), 1)


class TestConfiguracao(BaseWeb):
    def setUp(self):
        super().setUp()
        from autosrt import config
        self.config = config
        self._app_dir = config.app_directory
        config.app_directory = lambda: self.tmp
        self._env = os.environ.pop("OPENROUTER_API_KEY", None)

    def tearDown(self):
        self.config.app_directory = self._app_dir
        if self._env is not None:
            os.environ["OPENROUTER_API_KEY"] = self._env

    def test_sem_chave_configurada(self):
        dados = self.client.get("/api/config").get_json()
        self.assertFalse(dados["tem_chave"])

    def test_grava_a_chave(self):
        resposta = self.client.post("/api/config", json={"chave": "sk-or-v1-teste"})
        self.assertEqual(resposta.status_code, 200)
        self.assertTrue(self.client.get("/api/config").get_json()["tem_chave"])

    def test_a_chave_nunca_volta_para_o_navegador(self):
        self.client.post("/api/config", json={"chave": "sk-or-v1-segredo"})
        corpo = self.client.get("/api/config").get_data(as_text=True)
        self.assertNotIn("sk-or-v1-segredo", corpo)

    def test_grava_modelo_e_endereco(self):
        self.client.post("/api/config", json={
            "modelo": "deepseek/deepseek-chat-v2.5",
            "base_url": "http://localhost:11434/v1"})
        dados = self.client.get("/api/config").get_json()
        self.assertEqual(dados["modelo"], "deepseek/deepseek-chat-v2.5")
        self.assertEqual(dados["base_url"], "http://localhost:11434/v1")

    def test_arquivo_gravado_e_so_do_dono(self):
        self.client.post("/api/config", json={"chave": "sk-or-v1-teste"})
        caminho = os.path.join(self.tmp, "config.json")
        self.assertEqual(os.stat(caminho).st_mode & 0o777, 0o600)

    def test_valor_vazio_limpa_a_configuracao(self):
        self.client.post("/api/config", json={"chave": "sk-or-v1-teste"})
        self.client.post("/api/config", json={"chave": ""})
        self.assertFalse(self.client.get("/api/config").get_json()["tem_chave"])

    def test_pedido_sem_nada_e_recusado(self):
        self.assertEqual(
            self.client.post("/api/config", json={}).status_code, 400)

    def test_avisa_quando_o_ambiente_tem_prioridade(self):
        os.environ["OPENROUTER_API_KEY"] = "do-ambiente"
        try:
            resposta = self.client.post("/api/config", json={"chave": "do-arquivo"})
            self.assertIn("prioridade", resposta.get_json()["aviso"])
        finally:
            os.environ.pop("OPENROUTER_API_KEY", None)

    def test_sem_chave_tmdb_configurada(self):
        dados = self.client.get("/api/config").get_json()
        self.assertFalse(dados["tem_chave_tmdb"])

    def test_grava_a_chave_tmdb(self):
        resposta = self.client.post("/api/config", json={"chave_tmdb": "abc123"})
        self.assertEqual(resposta.status_code, 200)
        self.assertTrue(self.client.get("/api/config").get_json()["tem_chave_tmdb"])

    def test_chave_tmdb_nunca_volta_para_o_navegador(self):
        self.client.post("/api/config", json={"chave_tmdb": "segredo-tmdb"})
        corpo = self.client.get("/api/config").get_data(as_text=True)
        self.assertNotIn("segredo-tmdb", corpo)

    def test_expoe_os_enderecos_padrao_dos_dois_modos(self):
        from autosrt import llm
        dados = self.client.get("/api/config").get_json()
        self.assertEqual(dados["openrouter_base_url"], llm.DEFAULT_BASE_URL)
        self.assertEqual(dados["local_base_url"], llm.LOCAL_BASE_URL)


class TestModelos(BaseWeb):
    """/api/modelos alimenta o botão "Buscar modelos" do painel: nunca
    expõe a chave ao navegador, e delega a chamada de rede a llm.list_models
    (já testado à parte)."""

    def test_lista_modelos_do_endereco_informado(self):
        from autosrt import web as web_module
        with mock.patch.object(web_module.llm, "list_models") as fake:
            fake.return_value = [{"id": "modelo-a", "preco": "grátis"}]
            resposta = self.client.get(
                "/api/modelos?base_url=http://localhost:11434/v1")

        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.get_json()["modelos"],
                         [{"id": "modelo-a", "preco": "grátis"}])
        self.assertEqual(fake.call_args.args[0], "http://localhost:11434/v1")

    def test_sem_base_url_usa_o_configurado(self):
        from autosrt import config, web as web_module
        self._app_dir = config.app_directory
        config.app_directory = lambda: self.tmp
        self.addCleanup(setattr, config, "app_directory", self._app_dir)
        self.client.post("/api/config", json={"base_url": "http://meu-servidor/v1"})

        with mock.patch.object(web_module.llm, "list_models") as fake:
            fake.return_value = []
            self.client.get("/api/modelos")

        self.assertEqual(fake.call_args.args[0], "http://meu-servidor/v1")

    def test_falha_vira_502_com_mensagem(self):
        from autosrt import web as web_module
        from autosrt.llm import LLMError
        with mock.patch.object(web_module.llm, "list_models") as fake:
            fake.side_effect = LLMError("servidor local fora do ar")
            resposta = self.client.get("/api/modelos")

        self.assertEqual(resposta.status_code, 502)
        self.assertIn("fora do ar", resposta.get_json()["erro"])

    def test_nao_expoe_a_chave_ao_navegador(self):
        from autosrt import config, web as web_module
        self._app_dir = config.app_directory
        config.app_directory = lambda: self.tmp
        self.addCleanup(setattr, config, "app_directory", self._app_dir)
        self.client.post("/api/config", json={"chave": "sk-or-v1-segredo"})

        with mock.patch.object(web_module.llm, "list_models") as fake:
            fake.return_value = []
            resposta = self.client.get("/api/modelos")

        self.assertNotIn("sk-or-v1-segredo", resposta.get_data(as_text=True))
        self.assertEqual(fake.call_args.kwargs["api_key"], "sk-or-v1-segredo")


class TestFila(unittest.TestCase):
    def test_executa_um_de_cada_vez(self):
        simultaneos = []
        pico = []

        def worker(job):
            simultaneos.append(1)
            pico.append(len(simultaneos))
            time.sleep(0.05)
            simultaneos.pop()

        fila = jobs.JobQueue(worker)
        enviados = [fila.enviar(f"j{i}", f"/tmp/{i}") for i in range(5)]

        fim = time.time() + 5
        while time.time() < fim and any(
                j.estado not in jobs.FINAIS for j in enviados):
            time.sleep(0.02)

        self.assertTrue(all(j.estado == jobs.CONCLUIDO for j in enviados))
        # A GPU nao comporta dois trabalhos ao mesmo tempo.
        self.assertEqual(max(pico), 1)

    def test_falha_no_worker_vira_estado_de_erro(self):
        def worker(job):
            raise RuntimeError("quebrou")

        fila = jobs.JobQueue(worker)
        job = fila.enviar("j", "/tmp/j")
        fim = time.time() + 5
        while time.time() < fim and job.estado not in jobs.FINAIS:
            time.sleep(0.02)
        self.assertEqual(job.estado, jobs.ERRO)
        self.assertIn("quebrou", job.erro)

    def test_cancelar_antes_de_comecar(self):
        def worker(job):
            time.sleep(0.2)

        fila = jobs.JobQueue(worker)
        primeiro = fila.enviar("a", "/tmp/a")
        segundo = fila.enviar("b", "/tmp/b")
        fila.cancelar(segundo.id)

        fim = time.time() + 5
        while time.time() < fim and segundo.estado not in jobs.FINAIS:
            time.sleep(0.02)
        self.assertEqual(segundo.estado, jobs.CANCELADO)
        self.assertEqual(primeiro.estado, jobs.CONCLUIDO)

    def test_fila_volta_a_funcionar_depois_de_esvaziar(self):
        fila = jobs.JobQueue(lambda job: None)
        for _ in range(2):
            job = fila.enviar("j", "/tmp/j")
            fim = time.time() + 5
            while time.time() < fim and job.estado not in jobs.FINAIS:
                time.sleep(0.02)
            self.assertEqual(job.estado, jobs.CONCLUIDO)


if __name__ == "__main__":
    unittest.main()
