"""Tirar trabalho terminado da lista e apagar arquivo já usado."""

import os
import shutil
import tempfile
import threading
import unittest

from autosrt import jobs, pipeline, web


SRT = """1
00:00:01,000 --> 00:00:02,000
Hello there.
"""


class BaseComPasta(unittest.TestCase):
    """App de verdade, com o tradutor trocado por um dublê instantâneo.

    Sem isso cada trabalho tentaria falar com a API do modelo, e o teste
    ficaria dependendo de rede e de chave configurada.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        original = pipeline._translate_with_llm
        self.addCleanup(setattr, pipeline, "_translate_with_llm", original)
        pipeline._translate_with_llm = lambda cues, lang, **kw: (len(cues), [])

        app = web.create_app(media_dir=self.tmp)
        app.config["TESTING"] = True
        self.app = app
        self.fila = app.config["FILA"]
        self.client = app.test_client()

    def escrever(self, nome, conteudo=SRT):
        caminho = os.path.join(self.tmp, nome)
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)
        return caminho

    def processar_ate_o_fim(self, nome):
        """Enfileira e espera terminar, para o trabalho ficar em estado final."""
        caminho = self.escrever(nome)
        job = self.fila.enviar(nome, caminho, acao="traduzir")
        for _ in range(500):
            if job.estado in jobs.FINAIS:
                return job
            threading.Event().wait(0.01)
        self.fail(f"o trabalho de {nome} não terminou a tempo")


class TestExcluirArquivo(BaseComPasta):
    def test_apaga_legenda(self):
        caminho = self.escrever("legenda.srt")
        resposta = self.client.delete("/api/arquivo/legenda.srt")
        self.assertEqual(resposta.status_code, 200)
        self.assertFalse(os.path.exists(caminho))

    def test_apaga_filme(self):
        caminho = self.escrever("filme.mkv", "nao importa")
        resposta = self.client.delete("/api/arquivo/filme.mkv")
        self.assertEqual(resposta.status_code, 200)
        self.assertFalse(os.path.exists(caminho))

    def test_apaga_dentro_de_subpasta(self):
        caminho = self.escrever(os.path.join("serie", "ep1.srt"))
        resposta = self.client.delete("/api/arquivo/serie/ep1.srt")
        self.assertEqual(resposta.status_code, 200)
        self.assertFalse(os.path.exists(caminho))

    def test_apagar_a_legenda_nao_leva_o_filme_junto(self):
        filme = self.escrever("filme.mkv", "nao importa")
        self.escrever("filme.srt")
        self.client.delete("/api/arquivo/filme.srt")
        self.assertTrue(os.path.exists(filme))

    def test_arquivo_inexistente_da_404(self):
        self.assertEqual(
            self.client.delete("/api/arquivo/sumiu.srt").status_code, 404)

    def test_recusa_arquivo_que_nao_e_filme_nem_legenda(self):
        caminho = self.escrever("anotacao.txt", "oi")
        resposta = self.client.delete("/api/arquivo/anotacao.txt")
        self.assertEqual(resposta.status_code, 400)
        self.assertTrue(os.path.exists(caminho))

    def test_recusa_caminho_que_escapa_da_pasta(self):
        fora = os.path.join(os.path.dirname(self.tmp), "nao_apague.srt")
        with open(fora, "w", encoding="utf-8") as arquivo:
            arquivo.write(SRT)
        self.addCleanup(os.remove, fora)

        for tentativa in ("../nao_apague.srt", "..%2fnao_apague.srt",
                          "serie/../../nao_apague.srt"):
            with self.subTest(tentativa=tentativa):
                self.client.delete(f"/api/arquivo/{tentativa}")
                self.assertTrue(os.path.exists(fora),
                                f"{tentativa} apagou arquivo fora da pasta")

    def test_arquivo_ja_processado_pode_ser_apagado(self):
        job = self.processar_ate_o_fim("pronta.srt")
        self.assertEqual(job.estado, jobs.CONCLUIDO)
        self.assertEqual(
            self.client.delete("/api/arquivo/pronta.srt").status_code, 200)

    def test_nao_apaga_arquivo_em_processamento(self):
        # Apagar no meio deixaria o trabalho lendo um arquivo que sumiu, e o
        # erro sairia lá na frente, sem explicação. Prendo a tradução para
        # pegar o trabalho justamente enquanto ele roda.
        segurando = threading.Event()
        soltar = threading.Event()
        self.addCleanup(soltar.set)

        def travar(cues, lang, **kw):
            segurando.set()
            soltar.wait(10)
            return len(cues), []

        pipeline._translate_with_llm = travar

        caminho = self.escrever("ocupado.srt")
        self.fila.enviar("ocupado.srt", caminho, acao="traduzir")
        self.assertTrue(segurando.wait(10), "o trabalho não começou")

        resposta = self.client.delete("/api/arquivo/ocupado.srt")
        self.assertEqual(resposta.status_code, 409)
        self.assertIn("processado", resposta.get_json()["erro"])
        self.assertTrue(os.path.exists(caminho))


class TestLimparTrabalhos(BaseComPasta):
    def test_dispensa_trabalho_terminado(self):
        job = self.processar_ate_o_fim("a.srt")
        resposta = self.client.delete(f"/api/trabalho/{job.id}")
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(self.client.get("/api/trabalhos").get_json(), [])

    def test_dispensar_nao_apaga_o_arquivo(self):
        job = self.processar_ate_o_fim("b.srt")
        self.client.delete(f"/api/trabalho/{job.id}")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "b.srt")))

    def test_dispensar_trabalho_inexistente_da_400(self):
        self.assertEqual(
            self.client.delete("/api/trabalho/naoexiste").status_code, 400)

    def test_limpar_leva_os_terminados(self):
        for nome in ("c.srt", "d.srt", "e.srt"):
            self.processar_ate_o_fim(nome)

        resposta = self.client.post("/api/trabalhos/limpar")
        self.assertEqual(resposta.get_json()["removidos"], 3)
        self.assertEqual(self.client.get("/api/trabalhos").get_json(), [])

    def test_limpar_com_a_lista_vazia_nao_reclama(self):
        resposta = self.client.post("/api/trabalhos/limpar")
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.get_json()["removidos"], 0)


class TestFilaDireto(unittest.TestCase):
    """Os estados que não dá para arranjar pela HTTP, direto na fila."""

    def setUp(self):
        self.soltar = threading.Event()
        self.rodando = threading.Event()
        self.addCleanup(self.soltar.set)

        def worker(job):
            self.rodando.set()
            self.soltar.wait(10)

        self.fila = jobs.JobQueue(worker)

    def test_nao_remove_trabalho_que_ainda_roda(self):
        job = self.fila.enviar("x.srt", "/tmp/x.srt")
        self.assertTrue(self.rodando.wait(10))
        self.assertFalse(self.fila.remover(job.id))
        self.assertEqual(len(self.fila.listar()), 1)

    def test_limpar_deixa_o_que_ainda_roda(self):
        rodando = self.fila.enviar("x.srt", "/tmp/x.srt")
        self.assertTrue(self.rodando.wait(10))

        terminado = self.fila.enviar("y.srt", "/tmp/y.srt")
        terminado.estado = jobs.ERRO  # nem chegou ao operário, preso atrás

        self.assertEqual(self.fila.limpar_terminados(), 1)
        self.assertEqual([j.id for j in self.fila.listar()], [rodando.id])

    def test_em_uso_so_vale_enquanto_nao_termina(self):
        job = self.fila.enviar("x.srt", "/tmp/x.srt")
        self.assertTrue(self.rodando.wait(10))
        self.assertTrue(self.fila.em_uso("/tmp/x.srt"))
        self.assertFalse(self.fila.em_uso("/tmp/outro.srt"))

        job.estado = jobs.CONCLUIDO
        self.assertFalse(self.fila.em_uso("/tmp/x.srt"))


if __name__ == "__main__":
    unittest.main()
