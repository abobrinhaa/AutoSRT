import os
import subprocess
import unittest
from unittest import mock

from autosrt import audio


SAIDA_VOLUMEDETECT = """\
[Parsed_volumedetect_0 @ 0x55] n_samples: 26460000
[Parsed_volumedetect_0 @ 0x55] mean_volume: -34.7 dB
[Parsed_volumedetect_0 @ 0x55] max_volume: -18.8 dB
"""


def resultado(returncode=0, stderr=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout="", stderr=stderr)


class TestMedirVolumeMedio(unittest.TestCase):
    def test_le_o_mean_volume_da_saida(self):
        with mock.patch.object(audio, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(subprocess, "run",
                               return_value=resultado(stderr=SAIDA_VOLUMEDETECT)):
            self.assertEqual(audio.medir_volume_medio("filme.mp4"), -34.7)

    def test_volume_positivo_ou_zero_tambem_e_lido(self):
        with mock.patch.object(audio, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(subprocess, "run", return_value=resultado(
                 stderr="mean_volume: 0.0 dB")):
            self.assertEqual(audio.medir_volume_medio("filme.mp4"), 0.0)

    def test_sem_ffmpeg_devolve_none(self):
        with mock.patch.object(audio, "find_ffmpeg", return_value=None):
            self.assertIsNone(audio.medir_volume_medio("filme.mp4"))

    def test_saida_sem_a_linha_devolve_none(self):
        with mock.patch.object(audio, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(subprocess, "run",
                               return_value=resultado(stderr="nada aqui")):
            self.assertIsNone(audio.medir_volume_medio("filme.mp4"))

    def test_ffmpeg_que_nao_executa_devolve_none(self):
        with mock.patch.object(audio, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(subprocess, "run", side_effect=OSError("boom")):
            self.assertIsNone(audio.medir_volume_medio("filme.mp4"))


class TestVolumeBaixo(unittest.TestCase):
    """O caso real: -34,7 dB fazia o Whisper alucinar em vez de transcrever."""

    def com_volume(self, valor):
        return mock.patch.object(audio, "medir_volume_medio", return_value=valor)

    def test_audio_do_caso_real_e_considerado_baixo(self):
        with self.com_volume(-34.7):
            self.assertTrue(audio.volume_baixo("filme.mp4"))

    def test_audio_normal_de_filme_nao_e_baixo(self):
        with self.com_volume(-22.0):
            self.assertFalse(audio.volume_baixo("filme.mp4"))

    def test_exatamente_no_limiar_nao_conta_como_baixo(self):
        with self.com_volume(audio.LIMIAR_VOLUME_BAIXO):
            self.assertFalse(audio.volume_baixo("filme.mp4"))

    def test_sem_conseguir_medir_nao_normaliza(self):
        # "Não sei" não pode virar "normaliza": mexeria no áudio de todo
        # mundo por um palpite.
        with self.com_volume(None):
            self.assertFalse(audio.volume_baixo("filme.mp4"))


class TestNormalizarParaWav(unittest.TestCase):
    def test_sem_ffmpeg_levanta_erro_explicativo(self):
        with mock.patch.object(audio, "find_ffmpeg", return_value=None):
            with self.assertRaises(audio.AudioError) as ctx:
                audio.normalizar_para_wav("filme.mp4", "/tmp/saida.wav")
        self.assertIn("ffmpeg", str(ctx.exception))

    def test_monta_o_comando_com_loudnorm_e_16k_mono(self):
        capturado = {}

        def falso_run(comando, **kwargs):
            capturado["comando"] = comando
            return resultado()

        with mock.patch.object(audio, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(subprocess, "run", side_effect=falso_run), \
             mock.patch.object(os.path, "exists", return_value=True):
            audio.normalizar_para_wav("filme.mp4", "/tmp/saida.wav")

        comando = capturado["comando"]
        filtro = comando[comando.index("-af") + 1]
        self.assertIn("loudnorm", filtro)
        # 16 kHz mono é o formato que o próprio Whisper usa internamente.
        self.assertEqual(comando[comando.index("-ar") + 1], "16000")
        self.assertEqual(comando[comando.index("-ac") + 1], "1")

    def test_ffmpeg_que_falha_levanta_erro_com_o_motivo(self):
        with mock.patch.object(audio, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(subprocess, "run", return_value=resultado(
                 returncode=1, stderr="Invalid data found")):
            with self.assertRaises(audio.AudioError) as ctx:
                audio.normalizar_para_wav("filme.mp4", "/tmp/saida.wav")
        self.assertIn("Invalid data", str(ctx.exception))

    def test_ffmpeg_que_nao_gera_arquivo_levanta_erro(self):
        with mock.patch.object(audio, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(subprocess, "run", return_value=resultado()), \
             mock.patch.object(os.path, "exists", return_value=False):
            with self.assertRaises(audio.AudioError):
                audio.normalizar_para_wav("filme.mp4", "/tmp/saida.wav")


if __name__ == "__main__":
    unittest.main()
