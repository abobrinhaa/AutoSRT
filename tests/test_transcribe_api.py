"""Testes para transcrição via API (OpenRouter/OpenAI)."""

import os
import tempfile
import unittest
from unittest import mock

from autosrt import transcribe_api
from autosrt.transcribe import TranscriptionError

VERBOSE_JSON_OK = {
    "segments": [
        {"start": 0.0, "end": 2.5, "text": " Hello there."},
        {"start": 2.5, "end": 5.0, "text": " How are you?"},
    ]
}


def resposta(status_code=200, corpo=None, texto=""):
    mock_resp = mock.Mock(status_code=status_code, text=texto)
    mock_resp.json.return_value = corpo or {}
    return mock_resp


class TestTranscribeViaApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        self.tmp.write(b"fake audio data")
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_requer_arquivo_valido(self):
        with self.assertRaises(TranscriptionError):
            transcribe_api.transcribe_via_api(
                "/nao/existe/arquivo.mp3", api_key="sk-or-teste")

    def test_requer_api_key(self):
        with self.assertRaises(TranscriptionError):
            transcribe_api.transcribe_via_api(self.tmp.name, api_key=None)

    def test_sucesso_monta_cues_com_tempos_reais(self):
        sessao = mock.Mock()
        sessao.post.return_value = resposta(200, VERBOSE_JSON_OK)

        cues = transcribe_api.transcribe_via_api(
            self.tmp.name, api_key="sk-or-teste", session=sessao)

        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].start, 0)
        self.assertEqual(cues[0].end, 2500)
        self.assertEqual(cues[0].text, "Hello there.")
        self.assertEqual(cues[1].start, 2500)
        self.assertEqual(cues[1].end, 5000)

    def test_pede_verbose_json(self):
        sessao = mock.Mock()
        sessao.post.return_value = resposta(200, VERBOSE_JSON_OK)

        transcribe_api.transcribe_via_api(
            self.tmp.name, api_key="sk-or-teste", session=sessao)

        _, kwargs = sessao.post.call_args
        self.assertEqual(kwargs["data"]["response_format"], "verbose_json")

    def test_usa_a_url_base_do_engine_por_padrao(self):
        sessao = mock.Mock()
        sessao.post.return_value = resposta(200, VERBOSE_JSON_OK)

        transcribe_api.transcribe_via_api(
            self.tmp.name, api_key="sk-teste", engine="openai", session=sessao)

        url = sessao.post.call_args[0][0]
        self.assertEqual(url, "https://api.openai.com/v1/audio/transcriptions")

    def test_base_url_customizada_tem_prioridade(self):
        sessao = mock.Mock()
        sessao.post.return_value = resposta(200, VERBOSE_JSON_OK)

        transcribe_api.transcribe_via_api(
            self.tmp.name, api_key="sk-teste", base_url="http://localhost:9000/v1",
            session=sessao)

        url = sessao.post.call_args[0][0]
        self.assertEqual(url, "http://localhost:9000/v1/audio/transcriptions")

    def test_erro_de_autenticacao_nao_repete(self):
        sessao = mock.Mock()
        sessao.post.return_value = resposta(401, texto="Unauthorized")

        with self.assertRaises(TranscriptionError):
            transcribe_api.transcribe_via_api(
                self.tmp.name, api_key="sk-invalida", session=sessao)

        sessao.post.assert_called_once()

    def test_limite_de_taxa_repete_e_depois_desiste(self):
        sessao = mock.Mock()
        sessao.post.return_value = resposta(429, texto="rate limited")

        with self.assertRaises(TranscriptionError):
            transcribe_api.transcribe_via_api(
                self.tmp.name, api_key="sk-teste", session=sessao,
                max_retries=3)

        self.assertEqual(sessao.post.call_count, 3)

    def test_repete_apos_falha_transitoria_e_depois_funciona(self):
        sessao = mock.Mock()
        sessao.post.side_effect = [
            resposta(503, texto="indisponivel"),
            resposta(200, VERBOSE_JSON_OK),
        ]

        cues = transcribe_api.transcribe_via_api(
            self.tmp.name, api_key="sk-teste", session=sessao, max_retries=3)

        self.assertEqual(len(cues), 2)
        self.assertEqual(sessao.post.call_count, 2)

    def test_falha_de_rede_nao_derruba_o_processo(self):
        import requests

        sessao = mock.Mock()
        sessao.post.side_effect = requests.ConnectionError("sem rede")

        with self.assertRaises(TranscriptionError):
            transcribe_api.transcribe_via_api(
                self.tmp.name, api_key="sk-teste", session=sessao, max_retries=2)

    def test_sem_segments_e_erro_claro(self):
        sessao = mock.Mock()
        sessao.post.return_value = resposta(200, {"text": "sem segmentacao"})

        with self.assertRaises(TranscriptionError) as ctx:
            transcribe_api.transcribe_via_api(
                self.tmp.name, api_key="sk-teste", session=sessao)
        self.assertIn("segments", str(ctx.exception))

    def test_segmentos_vazios_de_texto_sao_ignorados(self):
        sessao = mock.Mock()
        sessao.post.return_value = resposta(200, {"segments": [
            {"start": 0, "end": 1, "text": "   "},
            {"start": 1, "end": 2, "text": "Ola"},
        ]})

        cues = transcribe_api.transcribe_via_api(
            self.tmp.name, api_key="sk-teste", session=sessao)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].index, 1)

    def test_nenhuma_fala_reconhecida_e_erro(self):
        sessao = mock.Mock()
        sessao.post.return_value = resposta(200, {"segments": []})

        with self.assertRaises(TranscriptionError):
            transcribe_api.transcribe_via_api(
                self.tmp.name, api_key="sk-teste", session=sessao)

    def test_resposta_nao_json_e_erro_claro(self):
        sessao = mock.Mock()
        resp = mock.Mock(status_code=200, text="")
        resp.json.side_effect = ValueError("not json")
        sessao.post.return_value = resp

        with self.assertRaises(TranscriptionError):
            transcribe_api.transcribe_via_api(
                self.tmp.name, api_key="sk-teste", session=sessao)


if __name__ == "__main__":
    unittest.main()
