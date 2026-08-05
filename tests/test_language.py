import unittest

from autosrt.cue import Cue
from autosrt.language import build_sample, detect_language, language_name


def make_cues(*texts):
    return [Cue.from_source(i, i * 1000, i * 1000 + 900, t)
            for i, t in enumerate(texts, start=1)]


class TestNomeDeIdioma(unittest.TestCase):
    def test_codigo_conhecido(self):
        self.assertEqual(language_name("en"), "Inglês")

    def test_codigo_desconhecido_vira_maiuscula(self):
        self.assertEqual(language_name("xx"), "XX")

    def test_codigo_vazio(self):
        self.assertEqual(language_name(""), "desconhecido")


class TestAmostragem(unittest.TestCase):
    def test_ignora_legendas_curtas(self):
        cues = make_cues("Oi", "-", "Uma frase suficientemente longa aqui")
        amostra = build_sample(cues)
        self.assertIn("suficientemente", amostra)
        self.assertNotIn("Oi", amostra)

    def test_amostra_percorre_o_arquivo_inteiro(self):
        # Créditos em francês no começo, filme em inglês no resto: a amostra
        # precisa alcançar o final do arquivo.
        cues = make_cues(
            *["Une production française de cinéma"] * 10,
            *["This is an English sentence spoken here"] * 200)
        amostra = build_sample(cues)
        self.assertIn("English", amostra)

    def test_usa_legendas_curtas_quando_nao_ha_outras(self):
        cues = make_cues("Oi", "Tchau")
        self.assertTrue(build_sample(cues).strip())


class TestDeteccao(unittest.TestCase):
    def test_detecta_ingles(self):
        cues = make_cues(
            *["This is an English sentence that should be detected"] * 20)
        self.assertEqual(detect_language(cues), "en")

    def test_detecta_portugues(self):
        cues = make_cues(
            *["Esta é uma frase em português que deve ser detectada"] * 20)
        self.assertEqual(detect_language(cues), "pt")

    def test_creditos_iniciais_nao_dominam_a_deteccao(self):
        # O fluxo antigo olhava só as 10 primeiras legendas e erraria aqui.
        cues = make_cues(
            *["Una producción española de cine clásico"] * 10,
            *["This is an English sentence that should be detected"] * 150)
        self.assertEqual(detect_language(cues), "en")


if __name__ == "__main__":
    unittest.main()
