import os
import tempfile
import threading
import unittest

from autosrt import srt_io
from autosrt.pipeline import ENGINE_GOOGLE, ENGINE_LLM, translate_file
from autosrt.translate import TranslationCancelled

SAMPLE = """1
00:00:01,000 --> 00:00:03,000
This is an English sentence for the test.

2
00:00:04,000 --> 00:00:06,000
<i>Another English line goes here.</i>

3
00:00:07,000 --> 00:00:09,000
- Did you see that?
- I certainly did see it.
"""


class FakeTranslator:
    def __init__(self, source=None, target=None):
        pass

    def translate(self, text):
        return "[pt] " + text


def fake_factory(source, target):
    return FakeTranslator(source, target)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.origem = os.path.join(self.tmp, "filme.srt")
        with open(self.origem, "w", encoding="utf-8") as handle:
            handle.write(SAMPLE)

    def test_traduz_e_grava(self):
        destino = os.path.join(self.tmp, "saida.srt")
        resultado = translate_file(self.origem, destino,
                                   engine=ENGINE_GOOGLE, translator_factory=fake_factory)

        self.assertEqual(resultado.total, 3)
        self.assertEqual(resultado.translated, 3)
        self.assertEqual(resultado.failure_count, 0)
        self.assertEqual(resultado.detected_lang, "en")

        cues = srt_io.load_cues(destino)
        self.assertTrue(all(c.source_text.startswith("[pt]") or "[pt]" in c.source_text
                            for c in cues))

    def test_tempos_ficam_intactos(self):
        destino = os.path.join(self.tmp, "saida.srt")
        antes = srt_io.load_cues(self.origem)
        translate_file(self.origem, destino, engine=ENGINE_GOOGLE, translator_factory=fake_factory)
        depois = srt_io.load_cues(destino)
        self.assertEqual([(c.start, c.end) for c in antes],
                         [(c.start, c.end) for c in depois])

    def test_formatacao_sobrevive_ao_fluxo_completo(self):
        destino = os.path.join(self.tmp, "saida.srt")
        translate_file(self.origem, destino, engine=ENGINE_GOOGLE, translator_factory=fake_factory)
        cues = srt_io.load_cues(destino)

        self.assertTrue(cues[1].source_text.startswith("<i>"))
        self.assertTrue(cues[1].source_text.endswith("</i>"))
        # O diálogo continua com dois turnos.
        self.assertEqual(len(cues[2].source_text.split("\n")), 2)

    def test_cria_backup_ao_sobrescrever(self):
        translate_file(self.origem, engine=ENGINE_GOOGLE, translator_factory=fake_factory)
        backup = srt_io.backup_path(self.origem)
        self.assertTrue(os.path.exists(backup))
        with open(backup, encoding="utf-8") as handle:
            self.assertIn("This is an English sentence", handle.read())

    def test_nao_cria_backup_ao_gravar_em_outro_arquivo(self):
        destino = os.path.join(self.tmp, "saida.srt")
        resultado = translate_file(self.origem, destino,
                                   engine=ENGINE_GOOGLE, translator_factory=fake_factory)
        self.assertIsNone(resultado.backup_path)

    def test_cancelamento_nao_grava_o_arquivo(self):
        destino = os.path.join(self.tmp, "saida.srt")
        cancel = threading.Event()
        cancel.set()

        with self.assertRaises(TranslationCancelled):
            translate_file(self.origem, destino, cancel_event=cancel,
                           engine=ENGINE_GOOGLE, translator_factory=fake_factory)

        self.assertFalse(os.path.exists(destino),
                         "o arquivo foi gravado apesar do cancelamento")

    def test_arquivo_vazio_e_rejeitado(self):
        vazio = os.path.join(self.tmp, "vazio.srt")
        open(vazio, "w", encoding="utf-8").close()
        with self.assertRaises(ValueError):
            translate_file(vazio, engine=ENGINE_GOOGLE, translator_factory=fake_factory)

    def test_relata_o_progresso(self):
        destino = os.path.join(self.tmp, "saida.srt")
        vistos = []
        translate_file(self.origem, destino, engine=ENGINE_GOOGLE, translator_factory=fake_factory,
                       progress=lambda done, total: vistos.append((done, total)))
        self.assertEqual(len(vistos), 3)
        self.assertEqual(vistos[-1], (3, 3))


class EchoLLM:
    """Cliente de modelo falso: devolve os blocos pedidos, traduzidos."""

    def complete(self, system, user):
        from autosrt import llm_translate
        return "\n".join(
            f"<{n}>PT:{c.strip()}</{n}>"
            for n, c in llm_translate.BLOCK_RE.findall(user))


class TestPipelineComModelo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.origem = os.path.join(self.tmp, "filme.srt")
        with open(self.origem, "w", encoding="utf-8") as handle:
            handle.write(SAMPLE)

    def test_traduz_pelo_modelo(self):
        destino = os.path.join(self.tmp, "saida.srt")
        resultado = translate_file(self.origem, destino, engine=ENGINE_LLM,
                                   llm_client=EchoLLM())
        self.assertEqual(resultado.engine, ENGINE_LLM)
        self.assertEqual(resultado.translated, 3)
        self.assertEqual(resultado.failure_count, 0)

    def test_o_modelo_e_o_padrao(self):
        # A tradução por modelo passou a ser o caminho principal; o Google
        # ficou como alternativa.
        destino = os.path.join(self.tmp, "saida.srt")
        resultado = translate_file(self.origem, destino, llm_client=EchoLLM())
        self.assertEqual(resultado.engine, ENGINE_LLM)

    def test_tempos_intactos_com_o_modelo(self):
        destino = os.path.join(self.tmp, "saida.srt")
        antes = srt_io.load_cues(self.origem)
        translate_file(self.origem, destino, engine=ENGINE_LLM,
                       llm_client=EchoLLM())
        depois = srt_io.load_cues(destino)
        self.assertEqual([(c.start, c.end) for c in antes],
                         [(c.start, c.end) for c in depois])

    def test_numero_de_legendas_nao_muda(self):
        destino = os.path.join(self.tmp, "saida.srt")
        translate_file(self.origem, destino, engine=ENGINE_LLM,
                       llm_client=EchoLLM())
        self.assertEqual(len(srt_io.load_cues(destino)), 3)


if __name__ == "__main__":
    unittest.main()
