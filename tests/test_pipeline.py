import os
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from autosrt import audio, llm_translate, srt_io
from autosrt.cue import Cue
from autosrt.llm import LLMError
from autosrt.pipeline import ENGINE_GOOGLE, ENGINE_LLM, process_media, translate_file
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


class TestLimpezaNoFluxoDeTraducao(unittest.TestCase):
    """A limpeza (SDH, tag quebrada, linha) roda dentro de translate_file,
    então quem chama não precisa lembrar de fazer isso à parte."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def escrever(self, conteudo):
        origem = os.path.join(self.tmp, "filme.srt")
        with open(origem, "w", encoding="utf-8") as handle:
            handle.write(conteudo)
        return origem

    def test_legenda_so_de_sdh_nao_sobra_no_arquivo_final(self):
        origem = self.escrever(
            "1\n00:00:01,000 --> 00:00:03,000\n(TIROS)\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nCorre agora!\n")
        destino = os.path.join(self.tmp, "saida.srt")

        resultado = translate_file(origem, destino, engine=ENGINE_LLM,
                                   llm_client=EchoLLM())

        self.assertEqual(resultado.total, 1)
        cues = srt_io.load_cues(destino)
        self.assertEqual(len(cues), 1)
        self.assertNotIn("TIROS", cues[0].text)

    def test_tag_de_italico_quebrada_sai_fechada(self):
        origem = self.escrever(
            "1\n00:00:01,000 --> 00:00:03,000\n<i>Fala em itálico sem fechar\n")
        destino = os.path.join(self.tmp, "saida.srt")

        translate_file(origem, destino, engine=ENGINE_LLM, llm_client=EchoLLM())

        texto = srt_io.load_cues(destino)[0].text
        self.assertEqual(texto.count("<i>"), texto.count("</i>"))

    def test_frase_curta_partida_em_duas_linhas_sai_junta(self):
        origem = self.escrever(
            "1\n00:00:01,000 --> 00:00:03,000\nNão\nvá embora.\n")
        destino = os.path.join(self.tmp, "saida.srt")

        translate_file(origem, destino, engine=ENGINE_LLM, llm_client=EchoLLM())

        self.assertNotIn("\n", srt_io.load_cues(destino)[0].text)


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


class TestMotorDeTranscricaoAlternativo(unittest.TestCase):
    """process_media aceita um motor substituto ao Whisper local (ex.: via
    API na nuvem). Regressão de um bug em que esse motor era passado para o
    parâmetro errado e a transcrição real nunca era usada -- só um .srt
    local, que nunca existia, era procurado em seguida."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.midia = os.path.join(self.tmp, "filme.mkv")
        open(self.midia, "wb").close()
        self.chamado_com = None

    def fake_runner(self, media_path, **kwargs):
        self.chamado_com = (media_path, kwargs)
        return [
            Cue.from_source(index=1, start=0, end=2000, source_text="Hello there."),
            Cue.from_source(index=2, start=2000, end=4000, source_text="How are you?"),
        ]

    def test_usa_o_motor_alternativo_em_vez_do_local(self):
        destino = os.path.join(self.tmp, "saida.srt")
        resultado = process_media(self.midia, destino, translate=False,
                                  transcribe_runner=self.fake_runner)

        self.assertEqual(resultado.total, 2)
        cues = srt_io.load_cues(destino)
        self.assertEqual(cues[0].source_text, "Hello there.")
        self.assertEqual(cues[0].start, 0)
        self.assertEqual(cues[1].start, 2000)
        self.assertEqual(self.chamado_com[0], self.midia)

    def test_reporta_o_fim_da_fase_mesmo_sem_granularidade(self):
        # Uma chamada de API só não tem "por cento concluído" no meio do
        # caminho como o processo local reporta; o que importa é a barra
        # não ficar parada nem voltar para trás quando a fase termina.
        destino = os.path.join(self.tmp, "saida.srt")
        vistos = []
        process_media(self.midia, destino, translate=False,
                      transcribe_runner=self.fake_runner,
                      progress=lambda done, total: vistos.append((done, total)))
        self.assertIn((100, 100), vistos)


class TestAvisoDeCobertura(unittest.TestCase):
    """Transcrição que acaba no meio do filme termina sem erro nenhum: o
    trabalho é dado como concluído e a legenda parece boa até o ponto onde
    simplesmente para. Comparar o fim da última legenda com a duração do
    arquivo é o que transforma isso num aviso em vez de uma descoberta
    feita assistindo."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.midia = os.path.join(self.tmp, "filme.mkv")
        open(self.midia, "wb").close()
        self.destino = os.path.join(self.tmp, "saida.srt")

    def runner(self, media_path, **kwargs):
        # Última legenda termina em 19:50 (1190 s).
        return [Cue.from_source(index=1, start=1_180_000, end=1_190_000,
                                source_text="Last line before the file dies.")]

    def processar(self, duracao):
        with mock.patch.object(audio, "duracao_segundos", return_value=duracao):
            return process_media(self.midia, self.destino, translate=False,
                                 transcribe_runner=self.runner)

    def test_avisa_quando_a_legenda_acaba_no_meio_do_filme(self):
        resultado = self.processar(6753.0)
        self.assertIsNotNone(resultado.aviso)
        self.assertIn("0:19:50", resultado.aviso)
        self.assertIn("1:52:33", resultado.aviso)

    def test_a_legenda_continua_sendo_gravada(self):
        # O aviso não é erro: o que foi transcrito vale e tem de sair.
        self.processar(6753.0)
        self.assertEqual(len(srt_io.load_cues(self.destino)), 1)

    def test_arquivo_coberto_ate_o_fim_nao_recebe_aviso(self):
        self.assertIsNone(self.processar(1200.0).aviso)

    def test_creditos_e_silencio_no_fim_nao_viram_aviso(self):
        # Filme que acaba em música não é transcrição truncada; o corte é
        # generoso de propósito.
        self.assertIsNone(self.processar(1400.0).aviso)

    def test_sem_conseguir_medir_a_duracao_nao_inventa_aviso(self):
        self.assertIsNone(self.processar(None).aviso)


class QuebradoLLM:
    """Cliente que nunca consegue traduzir nada, para simular chave/modelo/
    crédito quebrados na API."""

    def complete(self, system, user):
        raise LLMError("sem créditos")


class TestFalhaTotalDeTraducaoLLM(unittest.TestCase):
    """Antes, uma tradução que falhava para 100% das falas terminava como
    trabalho "concluído" -- o arquivo saía todo no idioma original sem
    nenhum aviso visível na interface nem no terminal."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def escrever(self, nome, conteudo):
        caminho = os.path.join(self.tmp, nome)
        with open(caminho, "w", encoding="utf-8") as handle:
            handle.write(conteudo)
        return caminho

    def test_falha_em_arquivo_normal_vira_erro_visivel(self):
        origem = self.escrever("filme.srt", SAMPLE)  # 3 falas
        destino = os.path.join(self.tmp, "saida.srt")

        with self.assertRaises(LLMError) as ctx:
            translate_file(origem, destino, engine=ENGINE_LLM,
                           llm_client=QuebradoLLM())
        self.assertIn("chave", str(ctx.exception).lower())

    def test_erro_final_inclui_o_motivo_real_da_api(self):
        # Regressão: "confira chave/modelo/endereço" sem dizer qual dos três
        # obrigava a trocar um de cada vez até acertar. Modelo descontinuado
        # no provedor (404 "No endpoints found") é exatamente o caso real
        # que motivou isso -- parecia problema de chave, não era.
        class ModeloMortoLLM:
            def complete(self, system, user):
                raise LLMError(
                    "A API recusou a requisição - HTTP 404: No endpoints "
                    "found for deepseek/deepseek-chat-v2.5.")

        origem = self.escrever("filme.srt", SAMPLE)
        destino = os.path.join(self.tmp, "saida.srt")

        with self.assertRaises(LLMError) as ctx:
            translate_file(origem, destino, engine=ENGINE_LLM,
                           llm_client=ModeloMortoLLM())
        mensagem = str(ctx.exception)
        self.assertIn("404", mensagem)
        self.assertIn("No endpoints found", mensagem)
        # Não fica um arquivo "pronto" com tudo em inglês, sem aviso nenhum.
        self.assertFalse(os.path.exists(destino))

    def test_arquivo_bem_pequeno_continua_com_degradacao_graciosa(self):
        # Uma legenda de teste com pouquíssimas falas pode legitimamente ter
        # uma que o modelo não encaixe no formato -- isso não é sinal de que
        # a API inteira está quebrada, e não deveria virar erro fatal.
        origem = self.escrever("curto.srt",
                               "1\n00:00:01,000 --> 00:00:03,000\nHi.\n")
        destino = os.path.join(self.tmp, "curto_pt.srt")

        resultado = translate_file(origem, destino, engine=ENGINE_LLM,
                                   llm_client=QuebradoLLM())
        self.assertEqual(resultado.failure_count, 1)
        self.assertTrue(os.path.exists(destino))

    def test_falha_parcial_continua_com_degradacao_graciosa(self):
        class MetadeQuebradoLLM:
            def complete(self, system, user):
                numeros = {n for n, _ in llm_translate.BLOCK_RE.findall(user)}
                if "3" in numeros:
                    raise LLMError("essa parte falha sempre")
                return "\n".join(f"<{n}>PT:{c.strip()}</{n}>"
                                for n, c in llm_translate.BLOCK_RE.findall(user))

        origem = self.escrever("filme.srt", SAMPLE)
        destino = os.path.join(self.tmp, "saida.srt")

        resultado = translate_file(origem, destino, engine=ENGINE_LLM,
                                   llm_client=MetadeQuebradoLLM())
        self.assertGreater(resultado.translated, 0)
        self.assertTrue(os.path.exists(destino))


class TestSensibilidadeDaVAD(unittest.TestCase):
    """process_media repassa os parâmetros de VAD para o Whisper local, sem
    afetar quem nunca pediu nada (padrão continua None -- sem mexer em
    nada)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.midia = os.path.join(self.tmp, "filme.mkv")
        open(self.midia, "wb").close()

    def test_parametros_chegam_ao_whisper_local(self):
        destino = os.path.join(self.tmp, "saida.srt")
        with mock.patch("autosrt.transcribe.transcribe") as fake:
            fake.return_value = [
                Cue.from_source(index=1, start=0, end=1000, source_text="Oi."),
            ]
            process_media(self.midia, destino, translate=False,
                          vad_threshold=0.2, vad_min_silence_ms=300,
                          transcribe_extra_args=["--no_speech_threshold", "0.3"])

        self.assertEqual(fake.call_args.kwargs["vad_threshold"], 0.2)
        self.assertEqual(fake.call_args.kwargs["vad_min_silence_ms"], 300)
        self.assertEqual(fake.call_args.kwargs["extra_args"],
                         ["--no_speech_threshold", "0.3"])

    def test_sem_pedir_nada_fica_none(self):
        destino = os.path.join(self.tmp, "saida.srt")
        with mock.patch("autosrt.transcribe.transcribe") as fake:
            fake.return_value = [
                Cue.from_source(index=1, start=0, end=1000, source_text="Oi."),
            ]
            process_media(self.midia, destino, translate=False)

        self.assertIsNone(fake.call_args.kwargs["vad_threshold"])
        self.assertIsNone(fake.call_args.kwargs["vad_min_silence_ms"])
        self.assertIsNone(fake.call_args.kwargs["extra_args"])


class TestPastaDeOriginais(unittest.TestCase):
    """A transcrição no idioma falado não pode ficar ao lado do vídeo.

    Tocador de vídeo casa legenda pelo nome do arquivo; um irmão
    "filme.original.srt" faria o VLC oferecer duas faixas de legenda.
    """

    def test_vai_para_subpasta(self):
        from autosrt.pipeline import ORIGINALS_DIRNAME, original_path_for
        caminho = original_path_for(os.path.join("/filmes", "casablanca.srt"))
        self.assertEqual(os.path.dirname(caminho),
                         os.path.join("/filmes", ORIGINALS_DIRNAME))
        self.assertEqual(os.path.basename(caminho), "casablanca.srt")

    def test_nao_fica_ao_lado_do_video(self):
        from autosrt.pipeline import original_path_for
        caminho = original_path_for("/filmes/casablanca.srt")
        self.assertNotEqual(os.path.dirname(caminho), "/filmes")

if __name__ == "__main__":
    unittest.main()
