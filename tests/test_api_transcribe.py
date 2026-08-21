"""Testes do endpoint /api/transcribe.

O endpoint usa o mesmo Faster-Whisper-XXL do resto do programa, que é um
executável chamado por subprocesso. Os testes trocam
``autosrt.transcribe.transcribe`` por um dublê: rodar o executável de
verdade exigiria GPU, o binário instalado e minutos por arquivo.
"""

import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

from autosrt import srt_io, web
from autosrt.cue import Cue
from autosrt.transcribe import TranscriptionError


def cue(indice, inicio_ms, fim_ms, texto):
    """Uma legenda igual à que o Whisper devolve, já sem rótulo de locutor."""
    return Cue(index=indice, start=inicio_ms, end=fim_ms,
               source_text=texto, text=texto)


class BaseAPI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        app = web.create_app(media_dir=self.tmp)
        app.config["TESTING"] = True
        self.client = app.test_client()

    def enviar(self, nome="audio.mp3", query="", dados=b"audio de mentira"):
        return self.client.post(
            f"/api/transcribe{query}",
            data={"file": (io.BytesIO(dados), nome)})

    def dublar(self, cues=None, erro=None):
        """Troca o Whisper por um dublê e devolve o mock, para inspeção."""
        def falso(media_path, **kwargs):
            if erro:
                raise erro
            # O executável de verdade lê o arquivo enviado: se ele não
            # estiver no disco neste ponto, o endpoint salvou errado.
            if not os.path.isfile(media_path):
                raise AssertionError(f"{media_path} não chegou ao disco")
            return cues if cues is not None else []

        patch = mock.patch("autosrt.transcribe.transcribe",
                           side_effect=falso, autospec=True)
        self.addCleanup(patch.stop)
        return patch.start()


class TestValidacao(BaseAPI):
    def test_sem_arquivo(self):
        resposta = self.client.post("/api/transcribe", data={})
        self.assertEqual(resposta.status_code, 400)
        self.assertIn("erro", resposta.get_json())

    def test_arquivo_sem_nome(self):
        self.assertEqual(self.enviar(nome="").status_code, 400)

    def test_recusa_o_que_nao_e_midia(self):
        resposta = self.enviar(nome="anotacao.txt")
        self.assertEqual(resposta.status_code, 400)
        self.assertIn("anotacao.txt", resposta.get_json()["erro"])

    def test_recusa_legenda(self):
        # Legenda não se transcreve: já é texto.
        self.assertEqual(self.enviar(nome="legenda.srt").status_code, 400)

    def test_aceita_audio_e_video(self):
        self.dublar([cue(1, 0, 1000, "oi")])
        for nome in ("audio.mp3", "audio.wav", "audio.m4a",
                     "video.mp4", "video.mkv"):
            with self.subTest(nome=nome):
                self.assertEqual(self.enviar(nome=nome).status_code, 200)


class TestRespostaJSON(BaseAPI):
    def test_transcricao_devolve_srt_no_json(self):
        self.dublar([cue(1, 0, 2500, "Bom dia, senhor.")])
        dados = self.enviar().get_json()

        self.assertTrue(dados["sucesso"])
        self.assertEqual(dados["arquivo"], "audio.mp3")
        self.assertEqual(dados["linhas"], 1)
        self.assertIn("Bom dia, senhor.", dados["legendas"])
        self.assertIn("00:00:00,000 --> 00:00:02,500", dados["legendas"])

    def test_varias_legendas_saem_numeradas_e_em_ordem(self):
        self.dublar([cue(1, 0, 2000, "Primeira"),
                     cue(2, 3000, 5500, "Segunda"),
                     cue(3, 6000, 7000, "Terceira")])
        dados = self.enviar().get_json()

        self.assertEqual(dados["linhas"], 3)
        corpo = dados["legendas"]
        self.assertLess(corpo.index("Primeira"), corpo.index("Segunda"))
        self.assertLess(corpo.index("Segunda"), corpo.index("Terceira"))
        self.assertIn("00:00:03,000 --> 00:00:05,500", corpo)

    def test_o_srt_devolvido_e_relido_sem_perda(self):
        # A prova de que a saída é .srt de verdade, e não texto parecido:
        # o próprio leitor do programa a aceita de volta.
        original = [cue(1, 1500, 4250, "Uma linha"),
                    cue(2, 9000, 12000, "Outra linha")]
        self.dublar(original)
        corpo = self.enviar().get_json()["legendas"]

        caminho = os.path.join(self.tmp, "volta.srt")
        with open(caminho, "w", encoding="utf-8") as arquivo:
            arquivo.write(corpo)

        relidas = srt_io.load_cues(caminho)
        self.assertEqual([c.start for c in relidas], [1500, 9000])
        self.assertEqual([c.end for c in relidas], [4250, 12000])
        self.assertEqual([c.text for c in relidas], ["Uma linha", "Outra linha"])

    def test_idioma_detectado_vem_na_resposta(self):
        self.dublar([cue(1, 0, 3000,
                         "This is an English sentence, long enough to detect.")])
        self.assertEqual(self.enviar().get_json()["idioma"], "en")

    def test_amostra_curta_nao_derruba_a_resposta(self):
        # Detectar idioma de "ah" é impossível; a transcrição continua valendo.
        self.dublar([cue(1, 0, 500, "ah")])
        resposta = self.enviar()
        self.assertEqual(resposta.status_code, 200)
        self.assertIn("idioma", resposta.get_json())


class TestRespostaSRT(BaseAPI):
    def test_format_srt_devolve_arquivo(self):
        self.dublar([cue(1, 0, 2500, "Teste de transcrição")])
        resposta = self.enviar(nome="filme.mp4", query="?format=srt")

        self.assertEqual(resposta.status_code, 200)
        disposicao = resposta.headers["Content-Disposition"]
        self.assertIn("attachment", disposicao)
        self.assertIn("filme.srt", disposicao)
        self.assertIn("Teste de transcrição",
                      resposta.get_data(as_text=True))

    def test_arquivo_sai_em_utf8(self):
        # Subtitle Edit lê UTF-8; acento quebrado aqui vira legenda suja lá.
        self.dublar([cue(1, 0, 2000, "Coração, ação e não")])
        resposta = self.enviar(query="?format=srt")
        self.assertIn("Coração, ação e não", resposta.data.decode("utf-8"))


class TestOpcoes(BaseAPI):
    def test_idioma_e_repassado_ao_whisper(self):
        falso = self.dublar([cue(1, 0, 1000, "hello")])
        self.enviar(query="?idioma=en")
        self.assertEqual(falso.call_args.kwargs["language"], "en")

    def test_sem_idioma_deixa_o_whisper_detectar(self):
        falso = self.dublar([cue(1, 0, 1000, "hello")])
        self.enviar()
        self.assertIsNone(falso.call_args.kwargs["language"])

    def test_diarizacao_ligada_por_padrao(self):
        falso = self.dublar([cue(1, 0, 1000, "oi")])
        self.enviar()
        self.assertIsNotNone(falso.call_args.kwargs["diarize"])

    def test_diarizar_zero_desliga(self):
        falso = self.dublar([cue(1, 0, 1000, "oi")])
        self.enviar(query="?diarizar=0")
        self.assertIsNone(falso.call_args.kwargs["diarize"])


class TestFalhas(BaseAPI):
    def test_erro_do_whisper_vira_mensagem(self):
        self.dublar(erro=TranscriptionError("O Faster-Whisper-XXL não foi "
                                            "encontrado."))
        resposta = self.enviar()
        self.assertEqual(resposta.status_code, 500)
        self.assertIn("não foi encontrado", resposta.get_json()["erro"])

    def test_arquivo_sem_fala_avisa_em_vez_de_devolver_vazio(self):
        self.dublar([])
        resposta = self.enviar()
        self.assertEqual(resposta.status_code, 422)
        self.assertIn("fala", resposta.get_json()["erro"])


class TestLimpeza(BaseAPI):
    def test_nao_deixa_nada_na_pasta_de_trabalho(self):
        # A pasta é de quem usa a página: transcrição de fora não polui.
        self.dublar([cue(1, 0, 1000, "oi")])
        self.enviar(nome="passageiro.mp4")
        self.assertEqual(os.listdir(self.tmp), [])

    def test_apaga_o_temporario_mesmo_quando_falha(self):
        self.dublar(erro=TranscriptionError("quebrou"))
        antes = set(os.listdir(tempfile.gettempdir()))
        self.enviar()
        sobraram = [n for n in set(os.listdir(tempfile.gettempdir())) - antes
                    if n.startswith("autosrt-api-")]
        self.assertEqual(sobraram, [])


class TestCaminhoRealAteOSubprocesso(BaseAPI):
    """Sem dublar o transcribe: só o subprocesso do executável.

    Assim o teste passa pelo build_command, pelo lugar onde o .srt é
    procurado e pelo load_transcript de verdade. É onde moram os enganos que
    dublar o módulo inteiro esconde -- pasta de saída errada, nome de arquivo
    que não bate, rótulo de locutor deixado no texto.
    """

    def dublar_o_executavel(self, conteudo_srt):
        """Faz o "Whisper" gravar um .srt onde o de verdade gravaria."""
        gravados = {}

        def run(command, **kwargs):
            # Grava onde o comando montado mandou gravar. É isso que prova que
            # o lado que pede e o lado que procura o .srt combinam.
            destino = command[command.index("--output_dir") + 1]
            entrada = command[1]
            alvo = os.path.join(
                destino,
                os.path.splitext(os.path.basename(entrada))[0] + ".srt")
            with open(alvo, "w", encoding="utf-8") as arquivo:
                arquivo.write(conteudo_srt)
            gravados["alvo"] = alvo

        for alvo, novo in (("find_executable", lambda *a, **k: "/bin/true"),
                           ("_run_process", run)):
            patch = mock.patch(f"autosrt.transcribe.{alvo}", novo)
            self.addCleanup(patch.stop)
            patch.start()
        return gravados

    def test_le_o_srt_que_o_executavel_gravou(self):
        self.dublar_o_executavel(
            "1\n00:00:01,000 --> 00:00:03,500\nBom dia.\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nBoa noite.\n")

        dados = self.enviar(nome="filme.mp4").get_json()
        self.assertEqual(dados["linhas"], 2)
        self.assertIn("Bom dia.", dados["legendas"])
        self.assertIn("00:00:04,000 --> 00:00:06,000", dados["legendas"])

    def test_rotulo_de_locutor_nao_vaza_para_a_legenda(self):
        # Com diarização o executável escreve "SPEAKER_00: fala". Quem chama a
        # API quer a fala, não o rótulo.
        self.dublar_o_executavel(
            "1\n00:00:01,000 --> 00:00:02,000\nSPEAKER_00: Bom dia.\n")

        legendas = self.enviar().get_json()["legendas"]
        self.assertIn("Bom dia.", legendas)
        self.assertNotIn("SPEAKER_00", legendas)

    def test_executavel_que_nao_gera_nada_vira_erro_explicado(self):
        gravados = self.dublar_o_executavel("")
        gravados.clear()

        def nao_grava(command, **kwargs):
            return None

        patch = mock.patch("autosrt.transcribe._run_process", nao_grava)
        self.addCleanup(patch.stop)
        patch.start()

        resposta = self.enviar()
        self.assertEqual(resposta.status_code, 500)
        self.assertIn("não gerou", resposta.get_json()["erro"])


if __name__ == "__main__":
    unittest.main()
