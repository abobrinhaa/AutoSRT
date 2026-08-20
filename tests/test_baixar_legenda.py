"""Testes do download de legendas que já estão na pasta do servidor."""

import os
import shutil
import tempfile
import unittest

from autosrt import web


LEGENDA = """1
00:00:01,000 --> 00:00:02,000
Oi.
"""


class TestBaixarLegenda(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.escrever("filme.srt", LEGENDA)
        app = web.create_app(media_dir=self.tmp)
        app.config["TESTING"] = True
        self.client = app.test_client()

    def escrever(self, nome, conteudo):
        caminho = os.path.join(self.tmp, nome)
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)
        return caminho

    def test_baixa_a_legenda_como_anexo(self):
        resposta = self.client.get("/api/legenda/filme.srt")
        self.assertEqual(resposta.status_code, 200)
        self.assertIn("Oi.", resposta.get_data(as_text=True))
        self.assertIn("filme.srt", resposta.headers["Content-Disposition"])
        self.assertIn("attachment", resposta.headers["Content-Disposition"])

    def test_baixa_legenda_em_subpasta(self):
        self.escrever(os.path.join("serie", "ep1.srt"), LEGENDA)
        resposta = self.client.get("/api/legenda/serie/ep1.srt")
        self.assertEqual(resposta.status_code, 200)
        self.assertIn("ep1.srt", resposta.headers["Content-Disposition"])

    def test_legenda_inexistente_da_404(self):
        resposta = self.client.get("/api/legenda/sumiu.srt")
        self.assertEqual(resposta.status_code, 404)

    def test_recusa_arquivo_que_nao_e_legenda(self):
        self.escrever("filme.mp4", "nao e legenda")
        resposta = self.client.get("/api/legenda/filme.mp4")
        self.assertEqual(resposta.status_code, 400)

    def test_recusa_caminho_que_escapa_da_pasta(self):
        # O alvo existe e é uma legenda de verdade: só o caminho é que sai
        # da pasta de trabalho, e é isso que precisa ser barrado.
        fora = os.path.join(os.path.dirname(self.tmp), "fora.srt")
        with open(fora, "w", encoding="utf-8") as arquivo:
            arquivo.write(LEGENDA)
        self.addCleanup(os.remove, fora)

        for tentativa in ("../fora.srt", "..%2ffora.srt", "serie/../../fora.srt"):
            with self.subTest(tentativa=tentativa):
                resposta = self.client.get(f"/api/legenda/{tentativa}")
                self.assertNotEqual(resposta.status_code, 200)
                self.assertNotIn("Oi.", resposta.get_data(as_text=True))

    def test_lista_marca_o_tipo_usado_pelo_botao(self):
        # O link de baixar aparece por item.tipo === 'legenda'; se o rótulo
        # mudar sem a página acompanhar, o botão some sem ninguém notar.
        itens = self.client.get("/api/arquivos").get_json()
        tipos = {i["nome"]: i["tipo"] for i in itens}
        self.assertEqual(tipos["filme.srt"], "legenda")


if __name__ == "__main__":
    unittest.main()
