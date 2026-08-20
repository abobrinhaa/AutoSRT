import io
import logging
import os
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
        # Sem internet na rede local, um CDN deixaria a página quebrada.
        html = self.client.get("/").get_data(as_text=True)
        self.assertNotIn("http://", html.replace("http://{args.host}", ""))
        self.assertNotIn("https://", html)


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

    def test_recusa_video_explicando(self):
        resposta = self.enviar("filme.mkv", "conteudo")
        self.assertEqual(resposta.status_code, 400)
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
