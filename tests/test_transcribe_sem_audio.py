"""O Whisper morre feio quando o arquivo não tem trilha de áudio."""

import unittest

from autosrt import transcribe
from autosrt.transcribe import TranscriptionError


# Recortado de uma falha real: gravação de tela .mov feita sem microfone.
TRACEBACK_SEM_AUDIO = """\
Traceback (most recent call last):
  File "__main__.py", line 2324, in <module>
  File "__main__.py", line 1990, in cli
  File "__main__.py", line 448, in get_diarization
  File "faster_whisper/diarization.py", line 73, in run_model
  File "faster_whisper/audio.py", line 67, in decode_audio
  File "faster_whisper/audio.py", line 123, in _resample_frames
  File "faster_whisper/audio.py", line 110, in _group_frames
  File "faster_whisper/audio.py", line 100, in _ignore_invalid_frames
  File "av/container/input.pyx", line 211, in decode
  File "av/container/input.pyx", line 142, in demux
  File "av/container/streams.pyx", line 117, in av.container.streams.StreamContainer.get
IndexError: tuple index out of range
[PYI-28058:ERROR] Failed to execute script '__main__' due to unhandled exception!
"""


class TestExplicarFalha(unittest.TestCase):
    def test_falta_de_audio_vira_explicacao(self):
        mensagem = transcribe._explicar_falha(TRACEBACK_SEM_AUDIO)
        self.assertIn("não tem faixa de áudio", mensagem)
        self.assertIn("gravação de tela", mensagem.lower())
        # O traceback não ajuda quem só quer a legenda pronta.
        self.assertNotIn("Traceback", mensagem)
        self.assertNotIn("IndexError", mensagem)

    def test_outra_falha_mantem_o_traceback(self):
        # Sem certeza da causa, esconder a saída do Whisper só atrapalha
        # quem for investigar.
        bruto = "RuntimeError: CUDA out of memory"
        mensagem = transcribe._explicar_falha(bruto)
        self.assertIn(bruto, mensagem)
        self.assertIn("O Whisper falhou", mensagem)

    def test_indexerror_de_outra_origem_mantem_o_traceback(self):
        # A mesma exceção em outro ponto não é falta de áudio: sem a marca do
        # demuxer, a mensagem amigável seria um palpite errado.
        bruto = 'File "outra_coisa.py", line 9\nIndexError: tuple index out of range'
        mensagem = transcribe._explicar_falha(bruto)
        self.assertIn("O Whisper falhou", mensagem)
        self.assertIn("IndexError", mensagem)


class TestRunProcess(unittest.TestCase):
    """A explicação precisa chegar até quem chamou, não só existir."""

    def test_processo_que_falha_por_falta_de_audio(self):
        import sys

        # Um processo de verdade: cospe o traceback e sai com código 1.
        codigo = f"import sys; sys.stdout.write({TRACEBACK_SEM_AUDIO!r}); sys.exit(1)"
        with self.assertRaises(TranscriptionError) as erro:
            transcribe._run_process([sys.executable, "-c", codigo])

        self.assertIn("não tem faixa de áudio", str(erro.exception))


if __name__ == "__main__":
    unittest.main()
