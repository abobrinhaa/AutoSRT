import contextlib
import os
import tempfile
import threading
import unittest
from unittest import mock

from autosrt import llm_translate, pipeline, srt_io, transcribe
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


class TestEtapaDuranteATranscricao(unittest.TestCase):
    """A etapa mostrada tem que ser a que esta acontecendo.

    O preparo do audio fala por ultimo -- medir o volume, normalizar, ou
    explicar que nao deu. Como a transcricao era anunciada antes dele e
    nada a repetia, a tela passava a fase mais longa do trabalho exibindo
    uma etapa ja terminada, com a barra parada em zero. Travado e
    exatamente com o que isso se parece.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.midia = os.path.join(self.tmp, "filme.mkv")
        open(self.midia, "wb").close()
        self.etapas = []
        self.etapa_no_momento = None

    def espiar(self, *args, **kwargs):
        """Guarda a etapa que estava na tela no instante da transcricao."""
        self.etapa_no_momento = self.etapas[-1] if self.etapas else None
        return [Cue.from_source(index=1, start=0, end=2000,
                                source_text="Hello there.")]

    @contextlib.contextmanager
    def preparo_falante(self, caminho, modo, announce=None):
        """Preparo de audio que fala logo antes de entregar o arquivo.

        E o caso relatado: "Audio fraco (-28,8 dB): normalizando antes de
        transcrever...", a ultima mensagem antes da espera longa.
        """
        if announce:
            announce("Áudio fraco (-28.8 dB): normalizando antes de transcrever...")
        yield caminho

    def rodar(self, **extras):
        process_media(self.midia, os.path.join(self.tmp, "saida.srt"),
                      translate=False, status=self.etapas.append, **extras)

    def test_whisper_local_anuncia_depois_do_preparo_do_audio(self):
        # O ramo onde o defeito apareceu: Whisper local, audio normalizado.
        with mock.patch.object(pipeline, "_audio_preparado",
                               self.preparo_falante), \
             mock.patch.object(transcribe, "transcribe", self.espiar):
            self.rodar()
        self.assertEqual(self.etapa_no_momento, pipeline.ETAPA_TRANSCREVENDO)

    def test_motor_alternativo_tambem_anuncia(self):
        with mock.patch.object(pipeline, "_audio_preparado", self.preparo_falante):
            self.rodar(transcribe_runner=lambda caminho, **kw: self.espiar())
        self.assertEqual(self.etapa_no_momento, pipeline.ETAPA_TRANSCREVENDO)

    def test_sem_normalizacao_a_etapa_continua_certa(self):
        with mock.patch.object(transcribe, "transcribe", self.espiar):
            self.rodar(normalize_audio=pipeline.NORMALIZE_NUNCA)
        self.assertEqual(self.etapa_no_momento, pipeline.ETAPA_TRANSCREVENDO)

    def test_a_mensagem_do_preparo_nao_e_a_ultima(self):
        # A forma generica do bug: qualquer coisa que o preparo do audio
        # resolva dizer nao pode sobrar na tela durante o Whisper.
        with mock.patch.object(pipeline, "_audio_preparado",
                               self.preparo_falante), \
             mock.patch.object(transcribe, "transcribe", self.espiar):
            self.rodar()
        self.assertNotIn("normalizando", self.etapa_no_momento)


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


class TestAntiAlucinacao(unittest.TestCase):
    """process_media repassa condition_on_previous_text e
    hallucination_silence_threshold para o Whisper local -- e sem pedir
    nada, o padrão de transcribe.transcribe() (desligado) fica valendo
    sozinho, sem process_media repetir esse padrão."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.midia = os.path.join(self.tmp, "filme.mkv")
        open(self.midia, "wb").close()

    def _transcrever(self, **kwargs):
        destino = os.path.join(self.tmp, "saida.srt")
        with mock.patch("autosrt.transcribe.transcribe") as fake:
            fake.return_value = [
                Cue.from_source(index=1, start=0, end=1000, source_text="Oi."),
            ]
            process_media(self.midia, destino, translate=False, **kwargs)
        return fake.call_args.kwargs

    def test_sem_pedir_nada_nao_sobrescreve_o_padrao(self):
        kwargs = self._transcrever()
        self.assertNotIn("condition_on_previous_text", kwargs)
        self.assertNotIn("hallucination_silence_threshold", kwargs)

    def test_condition_on_previous_text_chega_ao_whisper_local(self):
        kwargs = self._transcrever(condition_on_previous_text=True)
        self.assertTrue(kwargs["condition_on_previous_text"])

    def test_hallucination_silence_threshold_chega_ao_whisper_local(self):
        kwargs = self._transcrever(hallucination_silence_threshold=2.0)
        self.assertEqual(kwargs["hallucination_silence_threshold"], 2.0)


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

class TestFiltroDeAlucinacao(unittest.TestCase):
    """A frase inventada pelo Whisper não pode chegar ao arquivo final.

    O filtro roda logo depois de transcrever, antes de gravar a pasta de
    originais e antes de traduzir -- assim a alucinação não é traduzida
    (gastando requisição) nem some só do arquivo traduzido, deixando o
    original sujo.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.midia = os.path.join(self.tmp, "filme.mkv")
        open(self.midia, "wb").close()
        self.destino = os.path.join(self.tmp, "saida.srt")

    def _transcritas(self):
        return [
            Cue.from_source(index=1, start=1000, end=3000,
                            source_text="Você viu o que aconteceu ontem?"),
            Cue.from_source(index=2, start=3000, end=5000,
                            source_text="Vi, e não acreditei."),
            Cue.from_source(index=3, start=5000, end=7000,
                            source_text="Ninguém acreditou."),
            Cue.from_source(index=4, start=7000, end=9000,
                            source_text="O delegado ainda está lá?"),
            Cue.from_source(index=5, start=30000, end=33000,
                            source_text="Obrigado por assistir!"),
        ]

    def _rodar(self, **kwargs):
        with mock.patch("autosrt.transcribe.transcribe") as fake:
            fake.return_value = self._transcritas()
            resultado = process_media(self.midia, self.destino,
                                      translate=False, **kwargs)
        return resultado

    def test_sai_do_arquivo_final(self):
        resultado = self._rodar()
        textos = [c.source_text for c in srt_io.load_cues(self.destino)]
        self.assertNotIn("Obrigado por assistir!", textos)
        self.assertEqual(len(textos), 4)
        self.assertEqual(resultado.total, 4)

    def test_sai_tambem_da_pasta_de_originais(self):
        self._rodar()
        original = srt_io.load_cues(pipeline.original_path_for(self.destino))
        self.assertEqual(len(original), 4)

    def test_pode_ser_desligado(self):
        self._rodar(filter_hallucinations=False)
        self.assertEqual(len(srt_io.load_cues(self.destino)), 5)

    def test_vale_para_o_motor_via_api(self):
        """O motor na nuvem não passa por nenhum parâmetro do Whisper local.

        Sem este filtro, ele não tem defesa nenhuma contra alucinação.
        """
        cues = self._transcritas()

        def runner(caminho, language=None, cancel_event=None):
            return cues

        process_media(self.midia, self.destino, translate=False,
                      transcribe_runner=runner)
        self.assertEqual(len(srt_io.load_cues(self.destino)), 4)

    def test_frases_extras_do_usuario_chegam_ao_filtro(self):
        with mock.patch("autosrt.transcribe.transcribe") as fake:
            fake.return_value = self._transcritas()[:4] + [
                Cue.from_source(index=5, start=30000, end=33000,
                                source_text="Legendas: João da Silva")]
            process_media(self.midia, self.destino, translate=False,
                          extra_hallucinations=["Legendas: João da Silva"])
        self.assertEqual(len(srt_io.load_cues(self.destino)), 4)

    def test_transcricao_que_so_tem_clichê_nao_vira_arquivo_vazio(self):
        with mock.patch("autosrt.transcribe.transcribe") as fake:
            fake.return_value = [
                Cue.from_source(index=1, start=1000, end=3000,
                                source_text="Obrigado por assistir!"),
            ]
            process_media(self.midia, self.destino, translate=False)
        self.assertEqual(len(srt_io.load_cues(self.destino)), 1)


if __name__ == "__main__":
    unittest.main()
