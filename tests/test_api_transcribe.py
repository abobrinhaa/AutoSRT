"""Testes para endpoint /api/transcribe (Whisper API)."""

import io
import json
import os
import tempfile
import unittest
from unittest import mock

from autosrt import web


class TestTranscribeAPI(unittest.TestCase):
    """Testa o endpoint /api/transcribe."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.app = web.create_app(media_dir=self.tmp)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_transcribe_sem_arquivo(self):
        """Rejeita requisição sem arquivo."""
        resposta = self.client.post("/api/transcribe", data={})
        self.assertEqual(resposta.status_code, 400)
        dados = resposta.get_json()
        self.assertIn("erro", dados)
        self.assertIn("arquivo", dados["erro"].lower())

    def test_transcribe_arquivo_invalido(self):
        """Rejeita arquivo que não é áudio/vídeo."""
        resposta = self.client.post(
            "/api/transcribe",
            data={"file": (io.BytesIO(b"conteudo"), "arquivo.txt")}
        )
        self.assertEqual(resposta.status_code, 400)
        dados = resposta.get_json()
        self.assertIn("erro", dados)
        self.assertIn("arquivo", dados["erro"].lower())

    def test_transcribe_arquivo_vazio(self):
        """Rejeita arquivo vazio."""
        resposta = self.client.post(
            "/api/transcribe",
            data={"file": (io.BytesIO(b""), "")}
        )
        self.assertEqual(resposta.status_code, 400)

    @mock.patch("autosrt.web.WhisperModel")
    def test_transcribe_sucesso_json(self, mock_whisper):
        """Transcrição bem-sucedida retorna JSON."""
        # Mock do modelo Whisper
        mock_model = mock.Mock()
        mock_whisper.return_value = mock_model

        mock_segment = mock.Mock()
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = "Olá, como vai?"

        mock_info = mock.Mock()
        mock_info.language = "pt"

        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        # Fazer requisição
        resposta = self.client.post(
            "/api/transcribe",
            data={"file": (io.BytesIO(b"fake audio"), "audio.mp3")}
        )

        self.assertEqual(resposta.status_code, 200)
        dados = resposta.get_json()
        self.assertTrue(dados["sucesso"])
        self.assertEqual(dados["idioma"], "pt")
        self.assertIn("legendas", dados)
        self.assertIn("Olá, como vai?", dados["legendas"])

    @mock.patch("autosrt.web.WhisperModel")
    def test_transcribe_sucesso_srt(self, mock_whisper):
        """Transcrição retorna arquivo .srt quando solicitado."""
        # Mock do modelo
        mock_model = mock.Mock()
        mock_whisper.return_value = mock_model

        mock_segment = mock.Mock()
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = "Teste de transcrição"

        mock_info = mock.Mock()
        mock_info.language = "pt"

        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        # Solicitar como .srt
        resposta = self.client.post(
            "/api/transcribe?format=srt",
            data={"file": (io.BytesIO(b"fake audio"), "audio.mp3")}
        )

        self.assertEqual(resposta.status_code, 200)
        self.assertIn(b".srt", resposta.headers["Content-Disposition"])
        self.assertIn(b"Teste de transc", resposta.data)

    @mock.patch("autosrt.web.WhisperModel")
    def test_transcribe_multiplos_segmentos(self, mock_whisper):
        """Transcrição com múltiplos segmentos."""
        mock_model = mock.Mock()
        mock_whisper.return_value = mock_model

        seg1 = mock.Mock()
        seg1.start = 0.0
        seg1.end = 2.0
        seg1.text = "Primeira linha"

        seg2 = mock.Mock()
        seg2.start = 3.0
        seg2.end = 5.5
        seg2.text = "Segunda linha"

        mock_info = mock.Mock()
        mock_info.language = "pt"

        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

        resposta = self.client.post(
            "/api/transcribe",
            data={"file": (io.BytesIO(b"fake audio"), "audio.mp3")}
        )

        dados = resposta.get_json()
        self.assertEqual(dados["linhas"], 2)
        self.assertIn("Primeira linha", dados["legendas"])
        self.assertIn("Segunda linha", dados["legendas"])

    @mock.patch("autosrt.web.WhisperModel")
    def test_transcribe_formato_srt(self, mock_whisper):
        """Verifica formato SRT correto."""
        mock_model = mock.Mock()
        mock_whisper.return_value = mock_model

        segment = mock.Mock()
        segment.start = 1.5  # 1,5 segundos
        segment.end = 5.25  # 5,25 segundos
        segment.text = "Teste"

        mock_info = mock.Mock()
        mock_info.language = "pt"

        mock_model.transcribe.return_value = ([segment], mock_info)

        resposta = self.client.post(
            "/api/transcribe",
            data={"file": (io.BytesIO(b"fake audio"), "audio.mp3")}
        )

        dados = resposta.get_json()
        legendas = dados["legendas"]

        # Verificar formato SRT
        self.assertIn("1\n", legendas)  # Número da legenda
        self.assertIn("00:00:01,500 --> 00:00:05,250", legendas)  # Timecode correto
        self.assertIn("Teste", legendas)  # Texto

    def test_transcribe_arquivo_mp3(self):
        """Aceita arquivo .mp3."""
        resposta = self.client.post(
            "/api/transcribe",
            data={"file": (io.BytesIO(b"fake mp3"), "audio.mp3")}
        )
        # Vai falhar porque Whisper não está mockado, mas pelo menos aceita o tipo
        self.assertNotEqual(resposta.status_code, 400)

    def test_transcribe_arquivo_wav(self):
        """Aceita arquivo .wav."""
        resposta = self.client.post(
            "/api/transcribe",
            data={"file": (io.BytesIO(b"fake wav"), "audio.wav")}
        )
        self.assertNotEqual(resposta.status_code, 400)

    def test_transcribe_arquivo_video(self):
        """Aceita arquivo .mp4 (vídeo)."""
        resposta = self.client.post(
            "/api/transcribe",
            data={"file": (io.BytesIO(b"fake mp4"), "video.mp4")}
        )
        self.assertNotEqual(resposta.status_code, 400)


if __name__ == "__main__":
    unittest.main()
