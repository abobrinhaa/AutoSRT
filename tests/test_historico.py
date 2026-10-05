"""Histórico persistente de trabalhos (autosrt.historico).

Cada teste redireciona config.app_directory para uma pasta descartável --
sem isso, gravaria no ~/.config/autosrt/historico.db de verdade (mesmo
cuidado que test_web.py já tem para o config.json).
"""

import os
import tempfile
import unittest

from autosrt import config, historico


class _Job:
    """Só o que historico.registrar olha de um autosrt.jobs.Job de verdade."""

    def __init__(self, id, nome, estado, erro=None, detalhes=None):
        self.id = id
        self.nome = nome
        self.estado = estado
        self.erro = erro
        self.detalhes = detalhes or {}


class BaseHistorico(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._app_dir = config.app_directory
        config.app_directory = lambda: self.tmp
        self.addCleanup(setattr, config, "app_directory", self._app_dir)


class TestRegistrarEListar(BaseHistorico):
    def test_roundtrip_basico(self):
        job = _Job("abc123", "Filme.mp4", "concluido", detalhes={
            "idioma": "Inglês", "total": 217, "traduzidas": 203,
            "falhas": 14, "aviso": None,
        })
        historico.registrar(job)

        linhas = historico.listar()
        self.assertEqual(len(linhas), 1)
        self.assertEqual(linhas[0]["id"], "abc123")
        self.assertEqual(linhas[0]["arquivo"], "Filme.mp4")
        self.assertEqual(linhas[0]["estado"], "concluido")
        self.assertEqual(linhas[0]["total"], 217)
        self.assertEqual(linhas[0]["falhas"], 14)

    def test_grava_o_aviso_quando_tem(self):
        job = _Job("x", "Hijack.mp4", "concluido", detalhes={
            "aviso": "A legenda cobre só até 42:17, mas o vídeo tem 90min",
        })
        historico.registrar(job)
        self.assertIn("42:17", historico.listar()[0]["aviso"])

    def test_job_com_erro_grava_o_motivo(self):
        job = _Job("e1", "Quebrado.mp4", "erro",
                   erro="O Whisper não encontrou fala nenhuma no arquivo.")
        historico.registrar(job)
        linha = historico.listar()[0]
        self.assertEqual(linha["estado"], "erro")
        self.assertIn("não encontrou fala", linha["erro"])
        # Nenhum detalhe de tradução para um job que nem chegou lá --
        # colunas numéricas ficam None, não zero (zero seria "zero
        # legendas", uma afirmação; None é "não chegou a essa etapa").
        self.assertIsNone(linha["total"])

    def test_detalhes_vazio_nao_quebra(self):
        job = _Job("sem-detalhe", "Cancelado.mp4", "cancelado")
        historico.registrar(job)
        self.assertEqual(historico.listar()[0]["estado"], "cancelado")

    def test_mais_recente_primeiro(self):
        for i in range(3):
            historico.registrar(_Job(f"id{i}", f"filme{i}.mp4", "concluido"))

        nomes = [linha["arquivo"] for linha in historico.listar()]
        self.assertEqual(nomes, ["filme2.mp4", "filme1.mp4", "filme0.mp4"])

    def test_limite_corta_a_lista(self):
        for i in range(5):
            historico.registrar(_Job(f"id{i}", f"filme{i}.mp4", "concluido"))
        self.assertEqual(len(historico.listar(limite=2)), 2)

    def test_lista_vazia_sem_nenhum_registro(self):
        self.assertEqual(historico.listar(), [])

    def test_reprocessar_o_mesmo_id_substitui_a_linha(self):
        # Mesmo job id reenviado (ex.: usuário mandou o mesmo filme de
        # novo e por acaso bateu o uuid -- improvável, mas o contrato do
        # INSERT OR REPLACE é esse) não deve duplicar linha no histórico.
        historico.registrar(_Job("mesmo-id", "v1.mp4", "erro", erro="falhou"))
        historico.registrar(_Job("mesmo-id", "v1.mp4", "concluido",
                                 detalhes={"total": 10}))
        linhas = historico.listar()
        self.assertEqual(len(linhas), 1)
        self.assertEqual(linhas[0]["estado"], "concluido")


class TestCaminhoDoBanco(BaseHistorico):
    def test_fica_dentro_do_app_directory(self):
        self.assertEqual(
            os.path.dirname(historico.caminho_do_banco()), self.tmp)

    def test_cria_a_pasta_se_nao_existir(self):
        config.app_directory = lambda: os.path.join(self.tmp, "sub", "dir")
        historico.registrar(_Job("x", "a.mp4", "concluido"))
        self.assertTrue(os.path.exists(historico.caminho_do_banco()))


if __name__ == "__main__":
    unittest.main()
