import os
import tempfile
import unittest

from autosrt import transcribe
from autosrt.transcribe import TranscriptionError

SRT_COM_LOCUTOR = """1
00:00:01,000 --> 00:00:03,000
[SPEAKER_00]: He was a lawyer for years.

2
00:00:04,000 --> 00:00:06,000
[SPEAKER_01]: Did you speak to him?

3
00:00:07,000 --> 00:00:09,000
[SPEAKER_00]: I did, yes.
"""

SRT_SEM_LOCUTOR = """1
00:00:01,000 --> 00:00:03,000
He was a lawyer for years.
"""


class TestSplitSpeaker(unittest.TestCase):
    def test_colchetes_com_dois_pontos(self):
        self.assertEqual(
            transcribe.split_speaker("[SPEAKER_00]: Olá"),
            ("SPEAKER_00", "Olá"))

    def test_sem_colchetes(self):
        self.assertEqual(
            transcribe.split_speaker("SPEAKER_01: Olá"),
            ("SPEAKER_01", "Olá"))

    def test_parenteses_e_espaco(self):
        self.assertEqual(
            transcribe.split_speaker("(SPEAKER 2) Olá"),
            ("SPEAKER_2", "Olá"))

    def test_texto_sem_rotulo_fica_intacto(self):
        self.assertEqual(
            transcribe.split_speaker("Olá mundo"), (None, "Olá mundo"))

    def test_so_a_primeira_linha_carrega_o_rotulo(self):
        speaker, texto = transcribe.split_speaker("[SPEAKER_00]: Olá\nmundo")
        self.assertEqual(speaker, "SPEAKER_00")
        self.assertEqual(texto, "Olá\nmundo")

    def test_nao_confunde_dialogo_com_locutor(self):
        self.assertEqual(
            transcribe.split_speaker("- Você viu?"), (None, "- Você viu?"))


class TestBuildCommand(unittest.TestCase):
    def build(self, **kwargs):
        return transcribe.build_command(
            "filme.mkv", "/saida", executable="fw", **kwargs)

    def test_inclui_modelo_e_saida(self):
        command = self.build()
        self.assertIn("--model", command)
        self.assertIn(transcribe.DEFAULT_MODEL, command)
        self.assertIn("/saida", command)

    def test_formato_de_saida_e_srt(self):
        command = self.build()
        posicao = command.index("--output_format")
        self.assertEqual(command[posicao + 1], "srt")

    def test_diarizacao_ligada_por_padrao(self):
        command = self.build()
        self.assertIn("--diarize", command)
        self.assertIn(transcribe.DEFAULT_DIARIZE_MODEL, command)

    def test_diarizacao_pode_ser_desligada(self):
        self.assertNotIn("--diarize", self.build(diarize=None))

    def test_compute_type_padrao_e_auto(self):
        # int8_float16 exige compute capability 7.0; "auto" protege placas
        # Pascal como a Quadro P2200.
        command = self.build()
        posicao = command.index("--compute_type")
        self.assertEqual(command[posicao + 1], "auto")

    def test_idioma_so_entra_quando_informado(self):
        self.assertNotIn("--language", self.build())
        self.assertIn("--language", self.build(language="en"))

    def test_max_speakers_so_com_diarizacao(self):
        self.assertIn("--max_speakers", self.build(max_speakers=4))
        self.assertNotIn("--max_speakers",
                         self.build(diarize=None, max_speakers=4))

    def test_modelo_restrito_nao_e_padrao(self):
        # reverb_* é de uso pessoal e nao comercial.
        self.assertNotIn(transcribe.DEFAULT_DIARIZE_MODEL,
                         transcribe.RESTRICTED_DIARIZE_MODELS)

    def test_sensibilidade_da_vad_fica_de_fora_por_padrao(self):
        # Sem pedir explicitamente, o comportamento não muda: quem nunca
        # ouviu falar de VAD não é afetado por isso.
        command = self.build()
        self.assertNotIn("--vad_threshold", command)
        self.assertNotIn("--vad_min_silence_duration_ms", command)

    def test_vad_threshold_entra_quando_informado(self):
        command = self.build(vad_threshold=0.2)
        posicao = command.index("--vad_threshold")
        self.assertEqual(command[posicao + 1], "0.2")

    def test_vad_threshold_zero_ainda_entra(self):
        # 0 é um valor válido (o mais sensível possível) -- "if vad_threshold"
        # trataria isso como falso e sumiria com a flag por engano.
        command = self.build(vad_threshold=0)
        self.assertIn("--vad_threshold", command)

    def test_vad_min_silence_entra_quando_informado(self):
        command = self.build(vad_min_silence_ms=200)
        posicao = command.index("--vad_min_silence_duration_ms")
        self.assertEqual(command[posicao + 1], "200")

    def test_extra_args_vao_no_final(self):
        command = self.build(extra_args=["--no_speech_threshold", "0.3"])
        self.assertEqual(command[-2:], ["--no_speech_threshold", "0.3"])


class TestLoadTranscript(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def write(self, content, name="filme.srt"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def test_extrai_locutor_de_cada_legenda(self):
        cues = transcribe.load_transcript(self.write(SRT_COM_LOCUTOR))
        self.assertEqual([c.speaker for c in cues],
                         ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])

    def test_texto_fica_sem_o_rotulo(self):
        cues = transcribe.load_transcript(self.write(SRT_COM_LOCUTOR))
        self.assertEqual(cues[0].source_text, "He was a lawyer for years.")
        self.assertEqual(cues[0].text, cues[0].source_text)

    def test_sem_diarizacao_o_locutor_fica_nulo(self):
        cues = transcribe.load_transcript(self.write(SRT_SEM_LOCUTOR))
        self.assertIsNone(cues[0].speaker)
        self.assertEqual(cues[0].source_text, "He was a lawyer for years.")

    def test_arquivo_ausente_levanta_erro(self):
        with self.assertRaises(TranscriptionError):
            transcribe.load_transcript(os.path.join(self.tmp, "nao-existe.srt"))


class TestSpeakersIn(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        path = os.path.join(self.tmp, "filme.srt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(SRT_COM_LOCUTOR)
        self.cues = transcribe.load_transcript(path)

    def test_lista_sem_repetir_e_na_ordem(self):
        self.assertEqual(transcribe.speakers_in(self.cues),
                         ["SPEAKER_00", "SPEAKER_01"])


class TestTranscribe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.media = os.path.join(self.tmp, "filme.mkv")
        with open(self.media, "wb") as handle:
            handle.write(b"fake")
        # Executável de mentira: o runner injetado é que faz o trabalho, mas
        # a validação de produção exige que o arquivo exista.
        self.fake_exe = os.path.join(self.tmp, "faster-whisper-xxl")
        with open(self.fake_exe, "w") as handle:
            handle.write("")

    def fake_runner(self, saida=SRT_COM_LOCUTOR):
        """Runner que grava o .srt como o Whisper faria, sem executar nada."""
        def runner(command, **kwargs):
            output_dir = command[command.index("--output_dir") + 1]
            destino = os.path.join(output_dir, "filme.srt")
            with open(destino, "w", encoding="utf-8") as handle:
                handle.write(saida)
        return runner

    def test_fluxo_completo_com_runner_falso(self):
        cues = transcribe.transcribe(
            self.media, output_dir=self.tmp, executable=self.fake_exe,
            runner=self.fake_runner())
        self.assertEqual(len(cues), 3)
        self.assertEqual(cues[0].speaker, "SPEAKER_00")

    def test_arquivo_de_midia_ausente(self):
        with self.assertRaises(TranscriptionError):
            transcribe.transcribe(os.path.join(self.tmp, "nada.mkv"),
                                  executable=self.fake_exe, runner=self.fake_runner())

    def test_executavel_ausente_da_mensagem_util(self):
        with self.assertRaises(TranscriptionError) as contexto:
            transcribe.transcribe(self.media, executable="/nao/existe/fw")
        mensagem = str(contexto.exception)
        self.assertIn("Purfview", mensagem)
        self.assertIn("FASTER_WHISPER_PATH", mensagem)

    def test_sensibilidade_da_vad_chega_ate_o_comando(self):
        comandos = []

        def runner(command, **kwargs):
            comandos.append(command)
            output_dir = command[command.index("--output_dir") + 1]
            with open(os.path.join(output_dir, "filme.srt"), "w",
                     encoding="utf-8") as handle:
                handle.write(SRT_COM_LOCUTOR)

        transcribe.transcribe(
            self.media, output_dir=self.tmp, executable=self.fake_exe,
            runner=runner, vad_threshold=0.15, vad_min_silence_ms=300,
            extra_args=["--no_speech_threshold", "0.3"])

        command = comandos[0]
        self.assertEqual(command[command.index("--vad_threshold") + 1], "0.15")
        self.assertEqual(
            command[command.index("--vad_min_silence_duration_ms") + 1], "300")
        self.assertIn("--no_speech_threshold", command)

    def test_whisper_que_nao_gera_saida_levanta_erro(self):
        def runner_vazio(command, **kwargs):
            pass

        with self.assertRaises(TranscriptionError):
            transcribe.transcribe(self.media, output_dir=self.tmp,
                                  executable=self.fake_exe, runner=runner_vazio)


class TestFindExecutable(unittest.TestCase):
    def test_variavel_de_ambiente(self):
        tmp = tempfile.mkdtemp()
        fake = os.path.join(tmp, "faster-whisper-xxl")
        with open(fake, "w") as handle:
            handle.write("")
        anterior = os.environ.get("FASTER_WHISPER_PATH")
        os.environ["FASTER_WHISPER_PATH"] = fake
        try:
            self.assertEqual(transcribe.find_executable(), fake)
            self.assertTrue(transcribe.whisper_available())
        finally:
            if anterior is None:
                del os.environ["FASTER_WHISPER_PATH"]
            else:
                os.environ["FASTER_WHISPER_PATH"] = anterior

    def test_caminho_inexistente_devolve_nulo(self):
        self.assertIsNone(transcribe.find_executable("/nao/existe/fw"))


if __name__ == "__main__":
    unittest.main()
