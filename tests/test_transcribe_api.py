"""Testes para transcrição via API (OpenRouter/OpenAI)."""

import os
import tempfile
import unittest
from unittest import mock

from autosrt import transcribe_api


class TestTranscribeAPI(unittest.TestCase):
    """Testa transcrição via API."""

    def test_transcribe_openrouter_requer_arquivo_valido(self):
        """Transcrição rejeita arquivo inexistente."""
        with self.assertRaises(FileNotFoundError):
            transcribe_api.transcribe_openrouter(
                "/nao/existe/arquivo.mp3",
                api_key="sk-or-teste"
            )

    def test_transcribe_openrouter_requer_api_key(self):
        """Transcrição rejeita chave vazia."""
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            with self.assertRaises(ValueError):
                transcribe_api.transcribe_openrouter(f.name, api_key="")

    @mock.patch("autosrt.transcribe_api.requests.post")
    def test_transcribe_openrouter_sucesso(self, mock_post):
        """Transcrição bem-sucedida via OpenRouter."""
        # Mock da resposta
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Olá, como vai?"}
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(b"fake audio data")
            f.flush()

            resultado = transcribe_api.transcribe_openrouter(
                f.name,
                api_key="sk-or-teste"
            )

        self.assertEqual(resultado, "Olá, como vai?")
        mock_post.assert_called_once()

    @mock.patch("autosrt.transcribe_api.requests.post")
    def test_transcribe_openrouter_erro_api(self, mock_post):
        """Transcrição falha se API retorna erro."""
        mock_response = mock.Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(b"fake audio")
            f.flush()

            with self.assertRaises(ValueError) as ctx:
                transcribe_api.transcribe_openrouter(
                    f.name,
                    api_key="sk-or-invalida"
                )

            self.assertIn("401", str(ctx.exception))

    @mock.patch("autosrt.transcribe_api.requests.post")
    def test_transcribe_openrouter_resposta_invalida(self, mock_post):
        """Transcrição falha se resposta não tem 'text'."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "algo deu errado"}
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(b"fake audio")
            f.flush()

            with self.assertRaises(ValueError):
                transcribe_api.transcribe_openrouter(f.name, api_key="sk-or-teste")

    def test_transcribe_openai_requer_arquivo_valido(self):
        """Transcrição OpenAI rejeita arquivo inexistente."""
        with self.assertRaises(FileNotFoundError):
            transcribe_api.transcribe_openai(
                "/nao/existe/arquivo.mp3",
                api_key="sk-teste"
            )

    def test_transcribe_openai_requer_api_key(self):
        """Transcrição OpenAI rejeita chave vazia."""
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            with self.assertRaises(ValueError):
                transcribe_api.transcribe_openai(f.name, api_key="")

    @mock.patch("autosrt.transcribe_api.OpenAI")
    def test_transcribe_openai_sucesso(self, mock_openai_class):
        """Transcrição bem-sucedida via OpenAI."""
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client

        mock_transcription = mock.Mock()
        mock_transcription.text = "Hello world"
        mock_client.audio.transcriptions.create.return_value = mock_transcription

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(b"fake audio")
            f.flush()

            resultado = transcribe_api.transcribe_openai(
                f.name,
                api_key="sk-teste"
            )

        self.assertEqual(resultado, "Hello world")

    @mock.patch("autosrt.transcribe_api.transcribe_openrouter")
    def test_transcribe_wrapper_openrouter(self, mock_api):
        """Wrapper transcribe() roteia para openrouter."""
        mock_api.return_value = "Texto transcrito"

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(b"fake audio")
            f.flush()

            resultado = transcribe_api.transcribe(
                f.name,
                engine="openrouter",
                api_key="sk-or-teste"
            )

        self.assertEqual(resultado, "Texto transcrito")
        mock_api.assert_called_once()

    @mock.patch("autosrt.transcribe_api.transcribe_openai")
    def test_transcribe_wrapper_openai(self, mock_api):
        """Wrapper transcribe() roteia para openai."""
        mock_api.return_value = "Texto transcrito"

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(b"fake audio")
            f.flush()

            resultado = transcribe_api.transcribe(
                f.name,
                engine="openai",
                api_key="sk-teste"
            )

        self.assertEqual(resultado, "Texto transcrito")
        mock_api.assert_called_once()

    def test_transcribe_engine_desconhecido(self):
        """Wrapper transcribe() rejeita engine inválido."""
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(b"fake audio")
            f.flush()

            with self.assertRaises(ValueError) as ctx:
                transcribe_api.transcribe(
                    f.name,
                    engine="engine_invalido"
                )

            self.assertIn("desconhecido", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
