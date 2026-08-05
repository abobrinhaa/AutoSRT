import unittest

from autosrt import textfmt


class TestWrappingTags(unittest.TestCase):
    def test_extrai_tag_que_envolve_tudo(self):
        self.assertEqual(
            textfmt.split_wrapping_tags("<i>Olá mundo</i>"),
            ("<i>", "Olá mundo", "</i>"))

    def test_tag_no_meio_nao_conta_como_envolvimento(self):
        prefix, core, suffix = textfmt.split_wrapping_tags("Olá <b>mundo</b> inteiro")
        self.assertEqual(prefix, "")
        self.assertEqual(suffix, "")
        self.assertEqual(core, "Olá <b>mundo</b> inteiro")

    def test_texto_sem_tag(self):
        self.assertEqual(
            textfmt.split_wrapping_tags("Olá"), ("", "Olá", ""))


class TestDialogo(unittest.TestCase):
    def test_reconhece_dialogo_com_dois_travessoes(self):
        self.assertTrue(textfmt.is_dialogue("- Você viu?\n- Vi."))

    def test_linha_unica_com_hifen_nao_e_dialogo(self):
        self.assertFalse(textfmt.is_dialogue("- Só uma fala"))

    def test_separa_turnos_removendo_travessao(self):
        self.assertEqual(
            textfmt.split_dialogue_turns("- Você viu?\n- Vi."),
            ["Você viu?", "Vi."])


class TestPreparacao(unittest.TestCase):
    def test_dialogo_vira_um_segmento_por_turno(self):
        prefix, segments, suffix, dialogue = textfmt.prepare_for_translation(
            "- Você viu?\n- Vi.")
        self.assertTrue(dialogue)
        self.assertEqual(segments, ["Você viu?", "Vi."])

    def test_texto_corrido_vira_segmento_unico_sem_quebra(self):
        _, segments, _, dialogue = textfmt.prepare_for_translation(
            "Uma frase longa\nquebrada em duas linhas")
        self.assertFalse(dialogue)
        self.assertEqual(segments, ["Uma frase longa quebrada em duas linhas"])

    def test_remove_tags_internas_e_preserva_as_externas(self):
        prefix, segments, suffix, _ = textfmt.prepare_for_translation(
            "<i>Olá <b>mundo</b></i>")
        self.assertEqual(prefix, "<i>")
        self.assertEqual(suffix, "</i>")
        self.assertEqual(segments, ["Olá mundo"])

    def test_remove_caracteres_invisiveis(self):
        _, segments, _, _ = textfmt.prepare_for_translation("﻿Olá​")
        self.assertEqual(segments, ["Olá"])


class TestRemontagem(unittest.TestCase):
    def test_dialogo_volta_com_travessao_e_quebra(self):
        resultado = textfmt.reassemble("", ["Você viu?", "Vi."], "", True)
        self.assertEqual(resultado, "- Você viu?\n- Vi.")

    def test_tags_externas_voltam_ao_lugar(self):
        resultado = textfmt.reassemble("<i>", ["Olá mundo"], "</i>", False)
        self.assertEqual(resultado, "<i>Olá mundo</i>")

    def test_ciclo_completo_preserva_formatacao(self):
        original = "<i>- Você viu isso?\n- Vi, sim.</i>"
        prefix, segments, suffix, dialogue = textfmt.prepare_for_translation(original)
        # Simula o tradutor devolvendo os mesmos textos.
        resultado = textfmt.reassemble(prefix, segments, suffix, dialogue)
        self.assertEqual(resultado, original)


class TestRewrap(unittest.TestCase):
    def test_texto_curto_fica_em_uma_linha(self):
        self.assertEqual(textfmt.rewrap("Olá mundo"), "Olá mundo")

    def test_texto_longo_e_quebrado_em_duas_linhas(self):
        texto = ("Esta é uma frase bastante longa que "
                 "certamente ultrapassa o limite de linha")
        resultado = textfmt.rewrap(texto)
        linhas = resultado.split("\n")
        self.assertEqual(len(linhas), 2)
        # Nenhuma palavra pode se perder na quebra.
        self.assertEqual(" ".join(linhas), texto)

    def test_quebra_fica_equilibrada(self):
        texto = "palavra " * 12
        linhas = textfmt.rewrap(texto.strip()).split("\n")
        self.assertLessEqual(abs(len(linhas[0]) - len(linhas[1])), 10)


if __name__ == "__main__":
    unittest.main()
