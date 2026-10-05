import os
import shutil
import subprocess
import unittest
from unittest import mock

from autosrt import audio


SAIDA_VOLUMEDETECT = """\
[Parsed_volumedetect_0 @ 0x55] n_samples: 26460000
[Parsed_volumedetect_0 @ 0x55] mean_volume: -34.7 dB
[Parsed_volumedetect_0 @ 0x55] max_volume: -18.8 dB
"""


def resultado(returncode=0, stderr="", stdout=""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr)


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


class TestDetectarCorrupcao(unittest.TestCase):
    """Reconhecer áudio corrompido no stderr do ffmpeg, com o minuto
    aproximado -- caso real: "Silhouette 1990.mp4" (05/out/2026) parava de
    transcrever aos 19min porque o stream de áudio quebrava ali, sem
    nenhum erro visível antes disso."""

    def test_sem_padrao_conhecido_devolve_none(self):
        self.assertIsNone(audio.detectar_corrupcao("tudo normal por aqui"))

    def test_stderr_vazio_devolve_none(self):
        self.assertIsNone(audio.detectar_corrupcao(""))
        self.assertIsNone(audio.detectar_corrupcao(None))

    def test_acha_o_padrao_e_o_minuto_mais_recente_antes_dele(self):
        stderr = (
            "size=N/A time=00:19:32.10 bitrate=N/A speed=120x\n"
            "[aac] Invalid data found when processing input\n"
        )
        aviso = audio.detectar_corrupcao(stderr)
        self.assertIsNotNone(aviso)
        self.assertIn("19:32", aviso)

    def test_sem_linha_de_progresso_antes_nao_inventa_minuto(self):
        aviso = audio.detectar_corrupcao(
            "[aac] Invalid data found when processing input")
        self.assertIsNotNone(aviso)
        self.assertNotIn(" por volta de", aviso)

    def test_so_conta_o_time_antes_do_erro_nao_depois(self):
        # Um "time=" depois do erro é da passada seguinte (o ffmpeg segue
        # tentando decodificar); usar ele inventaria um minuto que vem
        # DEPOIS de onde o problema realmente apareceu.
        stderr = (
            "size=N/A time=00:05:00.00 bitrate=N/A\n"
            "[aac] invalid band type\n"
            "size=N/A time=01:40:00.00 bitrate=N/A\n"
        )
        aviso = audio.detectar_corrupcao(stderr)
        self.assertIn("5:00", aviso)
        self.assertNotIn("1:40:00", aviso)

    def test_reconhece_cada_padrao_conhecido(self):
        exemplos = [
            "Invalid data found when processing input",
            "Pulse data corrupt or invalid.",
            "invalid band type",
            "skip_data_stream_element: Input buffer exhausted",
        ]
        for linha in exemplos:
            with self.subTest(linha=linha):
                self.assertIsNotNone(audio.detectar_corrupcao(linha))


class TestMedirVolumeMedioAvisaCorrupcao(unittest.TestCase):
    """``on_aviso`` aproveita a mesma passada do volumedetect -- não é uma
    segunda chamada ao ffmpeg, só o stderr que já ia ser descartado."""

    def test_chama_on_aviso_quando_acha_corrupcao(self):
        avisos = []
        with mock.patch.object(audio, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(audio, "_run_ffmpeg", return_value=(
                 0, "Invalid data found when processing input")):
            audio.medir_volume_medio("filme.mp4", on_aviso=avisos.append)
        self.assertEqual(len(avisos), 1)

    def test_nao_chama_on_aviso_quando_esta_tudo_bem(self):
        avisos = []
        with mock.patch.object(audio, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(audio, "_run_ffmpeg",
                               return_value=(0, SAIDA_VOLUMEDETECT)):
            audio.medir_volume_medio("filme.mp4", on_aviso=avisos.append)
        self.assertEqual(avisos, [])

    def test_sem_on_aviso_nao_quebra(self):
        with mock.patch.object(audio, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(audio, "_run_ffmpeg", return_value=(
                 0, "Invalid data found when processing input")):
            # Sem on_aviso, só precisa não lançar -- a medição continua
            # funcionando normalmente pra quem não quer saber de avisos.
            self.assertIsNone(audio.medir_volume_medio("filme.mp4"))


class TestDuracaoDoVideo(unittest.TestCase):
    def test_le_a_duracao_da_saida_do_ffprobe(self):
        with mock.patch.object(shutil, "which", return_value="ffprobe"), \
             mock.patch.object(subprocess, "run", return_value=resultado(
                 stdout="5306.131000\n")):
            self.assertEqual(audio.duracao_do_video("filme.mp4"), 5306.131)

    def test_sem_ffprobe_devolve_none(self):
        with mock.patch.object(shutil, "which", return_value=None):
            self.assertIsNone(audio.duracao_do_video("filme.mp4"))

    def test_saida_nao_numerica_devolve_none(self):
        with mock.patch.object(shutil, "which", return_value="ffprobe"), \
             mock.patch.object(subprocess, "run",
                               return_value=resultado(stdout="N/A\n")):
            self.assertIsNone(audio.duracao_do_video("filme.mp4"))

    def test_timeout_devolve_none(self):
        with mock.patch.object(shutil, "which", return_value="ffprobe"), \
             mock.patch.object(
                 subprocess, "run",
                 side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=5)):
            self.assertIsNone(audio.duracao_do_video("filme.mp4"))


if __name__ == "__main__":
    unittest.main()
