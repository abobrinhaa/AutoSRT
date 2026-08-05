import os
import tempfile
import unittest

from autosrt import srt_io
from autosrt.cue import Cue

SAMPLE = """1
00:00:01,000 --> 00:00:03,000
Hello there.

2
00:00:04,500 --> 00:00:06,000
<i>How are you?</i>

3
00:00:07,000 --> 00:00:09,000
- Fine.
- Good.
"""


class TestLeitura(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def write(self, content, encoding="utf-8", name="teste.srt"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding=encoding) as handle:
            handle.write(content)
        return path

    def test_le_tempos_e_textos(self):
        cues = srt_io.load_cues(self.write(SAMPLE))
        self.assertEqual(len(cues), 3)
        self.assertEqual(cues[0].start, 1000)
        self.assertEqual(cues[0].end, 3000)
        self.assertEqual(cues[0].source_text, "Hello there.")

    def test_texto_de_trabalho_comeca_igual_ao_original(self):
        cues = srt_io.load_cues(self.write(SAMPLE))
        for cue in cues:
            self.assertEqual(cue.text, cue.source_text)

    def test_preserva_quebra_de_linha_do_dialogo(self):
        cues = srt_io.load_cues(self.write(SAMPLE))
        self.assertIn("\n", cues[2].source_text)

    def test_le_arquivo_latin1(self):
        conteudo = ("1\n00:00:01,000 --> 00:00:02,000\n"
                    "Ação e coração\n")
        cues = srt_io.load_cues(self.write(conteudo, encoding="latin-1"))
        self.assertEqual(len(cues), 1)
        self.assertIn("Ação", cues[0].source_text)

    def test_nao_reescreve_o_arquivo_de_entrada(self):
        path = self.write(SAMPLE, encoding="latin-1")
        with open(path, "rb") as handle:
            antes = handle.read()
        srt_io.load_cues(path)
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), antes)


class TestEscrita(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_ida_e_volta_preserva_conteudo(self):
        origem = os.path.join(self.tmp, "in.srt")
        with open(origem, "w", encoding="utf-8") as handle:
            handle.write(SAMPLE)

        cues = srt_io.load_cues(origem)
        destino = os.path.join(self.tmp, "out.srt")
        srt_io.save_cues(cues, destino)

        relidos = srt_io.load_cues(destino)
        self.assertEqual(len(relidos), len(cues))
        for antes, depois in zip(cues, relidos):
            self.assertEqual(antes.start, depois.start)
            self.assertEqual(antes.end, depois.end)
            self.assertEqual(antes.source_text, depois.source_text)

    def test_grava_o_texto_de_trabalho_nao_o_original(self):
        cues = [Cue(index=1, start=0, end=1000,
                    source_text="Hello", text="Olá")]
        destino = os.path.join(self.tmp, "out.srt")
        srt_io.save_cues(cues, destino)
        with open(destino, encoding="utf-8") as handle:
            conteudo = handle.read()
        self.assertIn("Olá", conteudo)
        self.assertNotIn("Hello", conteudo)

    def test_renumera_as_legendas(self):
        cues = [Cue(index=7, start=0, end=1000, source_text="a", text="a"),
                Cue(index=9, start=2000, end=3000, source_text="b", text="b")]
        destino = os.path.join(self.tmp, "out.srt")
        srt_io.save_cues(cues, destino)
        relidos = srt_io.load_cues(destino)
        self.assertEqual([c.index for c in relidos], [1, 2])


class TestBackupPath(unittest.TestCase):
    def test_nome_do_backup(self):
        resultado = srt_io.backup_path(os.path.join("pasta", "filme.srt"))
        self.assertEqual(resultado, os.path.join("pasta", "filme_backup.srt"))


if __name__ == "__main__":
    unittest.main()
