import os
import tempfile
import unittest

from autosrt import srt_io, ssa

HEADER = """[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize
Style: Default,Arial,20

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ssa(*dialogue_lines):
    return HEADER + "\n".join(dialogue_lines) + "\n"


class TestConversaoDeTexto(unittest.TestCase):
    def test_italico_vira_tag_html(self):
        self.assertEqual(
            ssa.convert_text(r"{\i1}Olá{\i0}"), "<i>Olá</i>")

    def test_negrito_e_sublinhado(self):
        self.assertEqual(ssa.convert_text(r"{\b1}A{\b0}"), "<b>A</b>")
        self.assertEqual(ssa.convert_text(r"{\u1}A{\u0}"), "<u>A</u>")

    def test_bloco_combinado(self):
        self.assertEqual(ssa.convert_text(r"{\i1\b1}Olá"), "<i><b>Olá</b></i>")

    def test_posicionamento_e_descartado(self):
        self.assertEqual(
            ssa.convert_text(r"{\pos(100,200)}Olá"), "Olá")

    def test_karaoke_e_descartado(self):
        self.assertEqual(ssa.convert_text(r"{\k50}Olá"), "Olá")

    def test_quebra_obrigatoria_vira_nova_linha(self):
        self.assertEqual(ssa.convert_text(r"- Viu?\N- Vi."), "- Viu?\n- Vi.")

    def test_quebra_leve_vira_espaco(self):
        self.assertEqual(ssa.convert_text(r"Olá\nmundo"), "Olá mundo")

    def test_espaco_nao_separavel(self):
        self.assertEqual(ssa.convert_text(r"Olá\hmundo"), "Olá mundo")

    def test_tag_aberta_e_fechada_no_fim(self):
        # Sem isso o itálico vazaria para as legendas seguintes.
        self.assertEqual(ssa.convert_text(r"{\i1}Olá"), "<i>Olá</i>")

    def test_fechamento_orfao_e_descartado(self):
        self.assertEqual(ssa.convert_text(r"Olá{\i0}"), "Olá")


class TestLeituraDeArquivo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def write(self, content, name="teste.ssa"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def test_le_tempos_e_texto(self):
        path = self.write(build_ssa(
            r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello there."))
        cues = ssa.load_cues(path)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].start, 1000)
        self.assertEqual(cues[0].end, 3000)
        self.assertEqual(cues[0].source_text, "Hello there.")

    def test_ignora_comentarios(self):
        path = self.write(build_ssa(
            r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Vale",
            r"Comment: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,Nao vale"))
        cues = ssa.load_cues(path)
        self.assertEqual([c.source_text for c in cues], ["Vale"])

    def test_funde_eventos_no_mesmo_intervalo(self):
        path = self.write(build_ssa(
            r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Primeira",
            r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Segunda"))
        cues = ssa.load_cues(path)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].source_text, "Primeira\nSegunda")

    def test_descarta_evento_que_fica_vazio(self):
        path = self.write(build_ssa(
            r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\pos(1,2)}",
            r"Dialogue: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,Vale"))
        cues = ssa.load_cues(path)
        self.assertEqual([c.source_text for c in cues], ["Vale"])

    def test_le_arquivo_sem_secao_de_estilos(self):
        # Arquivo real costuma vir sem [V4+ Styles]; a autodetecção do
        # pysubs2 falha nesse caso e precisamos do formato explícito.
        magro = ("[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
                 "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
                 "MarginV, Effect, Text\n"
                 "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Olá\n")
        cues = ssa.load_cues(self.write(magro))
        self.assertEqual([c.source_text for c in cues], ["Olá"])

    def test_arquivo_sem_eventos_devolve_lista_vazia(self):
        # O parser do pysubs2 é tolerante e aceita conteúdo sem eventos sem
        # reclamar. Quem transforma isso em erro é o srt_io.load_cues.
        path = self.write("nada disso e legenda", "quebrado.ssa")
        self.assertEqual(ssa.load_cues(path), [])

    def test_srt_io_transforma_arquivo_sem_eventos_em_erro(self):
        path = self.write("nada disso e legenda", "quebrado.ssa")
        with self.assertRaises(srt_io.UnsupportedSubtitleError):
            srt_io.load_cues(path)

    def test_ordena_por_tempo(self):
        path = self.write(build_ssa(
            r"Dialogue: 0,0:00:09.00,0:00:10.00,Default,,0,0,0,,Depois",
            r"Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Antes"))
        cues = ssa.load_cues(path)
        self.assertEqual([c.source_text for c in cues], ["Antes", "Depois"])


class TestIntegracaoComSrtIo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def write(self, content, name):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def test_load_cues_aceita_ssa(self):
        path = self.write(build_ssa(
            r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\i1}Olá{\i0}"),
            "f.ssa")
        cues = srt_io.load_cues(path)
        self.assertEqual(cues[0].source_text, "<i>Olá</i>")

    def test_load_cues_aceita_ass(self):
        path = self.write(build_ssa(
            r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Olá"), "f.ass")
        self.assertEqual(len(srt_io.load_cues(path)), 1)

    def test_ssa_com_extensao_srt_ainda_e_lido(self):
        path = self.write(build_ssa(
            r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Olá"),
            "enganado.srt")
        cues = srt_io.load_cues(path)
        self.assertEqual(cues[0].source_text, "Olá")

    def test_arquivo_invalido_levanta_erro_claro(self):
        path = self.write("isto nao e legenda nenhuma", "lixo.srt")
        with self.assertRaises(srt_io.UnsupportedSubtitleError):
            srt_io.load_cues(path)

    def test_converte_ssa_para_srt(self):
        path = self.write(build_ssa(
            r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,- Viu?\N- Vi."),
            "f.ssa")
        destino = srt_io.convert_to_srt(path)
        self.assertTrue(destino.endswith(".srt"))
        self.assertTrue(os.path.exists(destino))
        cues = srt_io.load_cues(destino)
        self.assertEqual(cues[0].source_text, "- Viu?\n- Vi.")

    def test_conversao_nao_sobrescreve_a_origem(self):
        path = self.write(build_ssa(
            r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Olá"), "f.ssa")
        with open(path, encoding="utf-8") as handle:
            antes = handle.read()
        srt_io.convert_to_srt(path)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), antes)

    def test_caminho_de_saida_srt(self):
        self.assertTrue(srt_io.srt_output_path("/a/b.ssa").endswith("b.srt"))
        self.assertEqual(srt_io.srt_output_path("/a/b.srt"), "/a/b.srt")


if __name__ == "__main__":
    unittest.main()
