import io
import logging
import os
import re
import tempfile
import time
import unittest

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

    def test_aceita_legenda(self):
        resposta = self.enviar("filme.srt")
        self.assertEqual(resposta.status_code, 202)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "filme.srt")))

    def test_aceita_video(self):
        # Em rede local, enviar filme pelo navegador leva um ou dois minutos,
        # irrelevante perto do tempo de transcrição.
        resposta = self.enviar("filme.mkv", "conteudo")
        self.assertEqual(resposta.status_code, 202)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "filme.mkv")))

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

    def test_legenda_pode_traduzir_e_ajustar_tempo(self):
        ids = self.ids("filme.srt")
        self.assertIn("traduzir", ids)
        self.assertIn("deslocar", ids)

    def test_converter_so_aparece_para_ssa(self):
        self.assertIn("converter", self.ids("filme.ssa"))
        self.assertIn("converter", self.ids("filme.ass"))
        self.assertNotIn("converter", self.ids("filme.srt"))


class TestAcoesIndividuais(BaseWeb):
    def setUp(self):
        super().setUp()
        self._original = pipeline._translate_with_llm
        pipeline._translate_with_llm = lambda cues, lang, **kw: (len(cues), [])

    def tearDown(self):
        pipeline._translate_with_llm = self._original

    def test_deslocar_move_os_tempos_sem_traduzir(self):
        self.escrever("filme.srt")
        job = self.client.post("/api/processar", json={
            "arquivo": "filme.srt", "acao": "deslocar", "segundos": 2.5,
        }).get_json()
        self.esperar(job["id"])

        cues = pipeline.srt_io.load_cues(os.path.join(self.tmp, "filme.srt"))
        self.assertEqual(cues[0].start, 3500)
        # O texto continua em inglês: deslocar não traduz.
        self.assertIn("English", cues[0].source_text)

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
