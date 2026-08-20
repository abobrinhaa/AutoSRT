import io
import os
import tempfile
import unittest
from unittest import mock

from autosrt import cli, pipeline, srt_io

SRT = """1
00:00:01,000 --> 00:00:03,000
This is an English sentence for the test.

2
00:00:04,000 --> 00:00:06,000
Another English line goes here now.
"""

SSA = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,This is an English sentence.
"""


class EchoLLM:
    def complete(self, system, user):
        from autosrt import llm_translate
        return "\n".join(f"<{n}>PT:{c.strip()}</{n}>"
                         for n, c in llm_translate.BLOCK_RE.findall(user))


class BaseCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.err = io.StringIO()

    def write(self, nome, conteudo=SRT):
        caminho = os.path.join(self.tmp, nome)
        with open(caminho, "w", encoding="utf-8") as handle:
            handle.write(conteudo)
        return caminho

    def touch(self, nome):
        caminho = os.path.join(self.tmp, nome)
        with open(caminho, "wb") as handle:
            handle.write(b"\0")
        return caminho

    def run_cli(self, argv):
        with mock.patch.object(cli.sys, "stderr", self.err):
            return cli.main(argv)


class TestParser(unittest.TestCase):
    def test_motor_padrao_e_o_modelo(self):
        args = cli.build_parser().parse_args(["x.srt"])
        self.assertEqual(args.motor, pipeline.ENGINE_LLM)

    def test_aceita_varias_entradas(self):
        args = cli.build_parser().parse_args(["a.srt", "b.srt"])
        self.assertEqual(args.entradas, ["a.srt", "b.srt"])

    def test_deslocar_e_numero(self):
        args = cli.build_parser().parse_args(["x.srt", "--deslocar", "-2.5"])
        self.assertEqual(args.deslocar, -2.5)

    def test_diarizacao_ligada_por_padrao(self):
        self.assertFalse(cli.build_parser().parse_args(["x.srt"]).sem_diarizacao)


class TestExpandInputs(BaseCLI):
    def test_pasta_vira_lista_de_arquivos(self):
        self.write("a.srt")
        self.write("b.srt")
        self.touch("nota.txt")
        arquivos = cli.expand_inputs([self.tmp])
        self.assertEqual([os.path.basename(a) for a in arquivos],
                         ["a.srt", "b.srt"])

    def test_ignora_backup_e_original(self):
        self.write("filme.srt")
        self.write("filme_backup.srt")
        self.write("filme.original.srt")
        arquivos = cli.expand_inputs([self.tmp])
        self.assertEqual([os.path.basename(a) for a in arquivos], ["filme.srt"])

    def test_reconhece_video_e_audio(self):
        self.touch("filme.mkv")
        self.touch("audio.mp3")
        arquivos = cli.expand_inputs([self.tmp])
        self.assertEqual(len(arquivos), 2)

    def test_arquivo_avulso_passa_direto(self):
        self.assertEqual(cli.expand_inputs(["/qualquer/x.srt"]), ["/qualquer/x.srt"])


class TestTraducao(BaseCLI):
    def test_traduz_uma_legenda(self):
        entrada = self.write("filme.srt")
        with mock.patch.object(pipeline, "_translate_with_llm") as fake:
            fake.side_effect = lambda cues, lang, **kw: (
                [setattr(c, "text", "PT:" + c.source_text) for c in cues],
                (len(cues), []))[1]
            código = self.run_cli([entrada])
        self.assertEqual(código, cli.EXIT_OK)
        with open(entrada, encoding="utf-8") as handle:
            self.assertIn("PT:", handle.read())

    def test_motor_google(self):
        entrada = self.write("filme.srt")
        with mock.patch.object(pipeline, "translate_cues") as fake:
            from autosrt.translate import TranslationReport
            fake.return_value = TranslationReport(total=2, translated=2)
            código = self.run_cli([entrada, "--motor", "google"])
        self.assertEqual(código, cli.EXIT_OK)
        self.assertTrue(fake.called)

    def test_converte_ssa_e_grava_srt(self):
        entrada = self.write("filme.ssa", SSA)
        with mock.patch.object(pipeline, "_translate_with_llm",
                               return_value=(1, [])):
            código = self.run_cli([entrada])
        self.assertEqual(código, cli.EXIT_OK)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "filme.srt")))

    def test_pasta_de_saida(self):
        entrada = self.write("filme.srt")
        destino = os.path.join(self.tmp, "saida")
        with mock.patch.object(pipeline, "_translate_with_llm",
                               return_value=(2, [])):
            self.run_cli([entrada, "-o", destino])
        self.assertTrue(os.path.exists(os.path.join(destino, "filme.srt")))


class TestSoConverter(BaseCLI):
    def test_converte_sem_traduzir(self):
        entrada = self.write("filme.ssa", SSA)
        código = self.run_cli([entrada, "--so-converter"])
        self.assertEqual(código, cli.EXIT_OK)
        destino = os.path.join(self.tmp, "filme.srt")
        self.assertTrue(os.path.exists(destino))
        with open(destino, encoding="utf-8") as handle:
            # Continua em inglês: converter não traduz.
            self.assertIn("This is an English sentence", handle.read())

    def test_srt_nao_tem_o_que_converter(self):
        entrada = self.write("filme.srt")
        self.assertEqual(self.run_cli([entrada, "--so-converter"]), cli.EXIT_ERRO)
        self.assertIn("já é .srt", self.err.getvalue())

    def test_video_e_recusado(self):
        entrada = self.touch("filme.mkv")
        self.assertEqual(self.run_cli([entrada, "--so-converter"]), cli.EXIT_ERRO)


class TestSincronia(BaseCLI):
    def test_deslocamento(self):
        entrada = self.write("filme.srt")
        código = self.run_cli([entrada, "--deslocar", "2"])
        self.assertEqual(código, cli.EXIT_OK)
        cues = srt_io.load_cues(entrada)
        self.assertEqual(cues[0].start, 3000)

    def test_deslocamento_negativo(self):
        entrada = self.write("filme.srt")
        self.run_cli([entrada, "--deslocar", "-1"])
        self.assertEqual(srt_io.load_cues(entrada)[0].start, 0)

    def test_sincronia_em_video_e_recusada(self):
        entrada = self.touch("filme.mkv")
        código = self.run_cli([entrada, "--deslocar", "2"])
        self.assertEqual(código, cli.EXIT_ERRO)
        self.assertIn("legendas", self.err.getvalue())


class TestErros(BaseCLI):
    def test_arquivo_inexistente(self):
        código = self.run_cli([os.path.join(self.tmp, "nada.srt")])
        self.assertEqual(código, cli.EXIT_ERRO)
        self.assertIn("não encontrei", self.err.getvalue())

    def test_pasta_vazia(self):
        código = self.run_cli([self.tmp])
        self.assertEqual(código, cli.EXIT_NADA_A_FAZER)

    def test_so_traduzir_com_video_e_recusado(self):
        entrada = self.touch("filme.mkv")
        código = self.run_cli([entrada, "--so-traduzir"])
        self.assertEqual(código, cli.EXIT_ERRO)

    def test_falha_num_arquivo_nao_impede_os_outros(self):
        bom = self.write("bom.srt")
        with mock.patch.object(pipeline, "_translate_with_llm",
                               return_value=(2, [])):
            código = self.run_cli([bom, os.path.join(self.tmp, "nada.srt")])
        self.assertEqual(código, cli.EXIT_ERRO)
        self.assertIn("1 de 2", self.err.getvalue())


class TestTranscricao(BaseCLI):
    def test_video_dispara_transcricao(self):
        entrada = self.touch("filme.mkv")
        with mock.patch.object(pipeline, "process_media") as fake:
            fake.return_value = pipeline.PipelineResult(
                total=3, translated=3, failed=[], detected_lang="en")
            código = self.run_cli([entrada])
        self.assertEqual(código, cli.EXIT_OK)
        self.assertTrue(fake.called)
        self.assertTrue(fake.call_args.kwargs["diarize"])

    def test_sem_diarizacao_desliga(self):
        entrada = self.touch("filme.mkv")
        with mock.patch.object(pipeline, "process_media") as fake:
            fake.return_value = pipeline.PipelineResult(
                total=1, translated=1, failed=[], detected_lang="en")
            self.run_cli([entrada, "--sem-diarizacao"])
        self.assertFalse(fake.call_args.kwargs["diarize"])

    def test_so_transcrever_nao_traduz(self):
        entrada = self.touch("filme.mkv")
        with mock.patch.object(pipeline, "process_media") as fake:
            fake.return_value = pipeline.PipelineResult(
                total=1, translated=0, failed=[], detected_lang="en")
            self.run_cli([entrada, "--so-transcrever"])
        self.assertFalse(fake.call_args.kwargs["translate"])


class TestReporter(unittest.TestCase):
    def test_quieto_nao_imprime(self):
        stream = io.StringIO()
        reporter = cli.Reporter(quiet=True, stream=stream)
        reporter.status("oi")
        reporter.progress(1, 2)
        self.assertEqual(stream.getvalue(), "")

    def test_progresso_nao_inunda(self):
        stream = io.StringIO()
        reporter = cli.Reporter(stream=stream)
        for i in range(1, 101):
            reporter.progress(i, 100)
        # Uma linha a cada 5%, não cem linhas.
        self.assertLess(len(stream.getvalue().splitlines()), 25)


if __name__ == "__main__":
    unittest.main()
