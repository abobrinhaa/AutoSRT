import threading
import unittest

from autosrt.cue import Cue
from autosrt.translate import TranslationCancelled, translate_cues


class FakeTranslator:
    """Tradutor falso: devolve o texto com um prefixo, sem tocar na rede."""

    def __init__(self, source=None, target=None, prefix="[pt] "):
        self.prefix = prefix
        self.calls = []

    def translate(self, text):
        self.calls.append(text)
        return self.prefix + text


class FailingTranslator:
    """Falha sempre, para exercitar o caminho de erro."""

    def __init__(self, source=None, target=None, exception=True):
        self.exception = exception

    def translate(self, text):
        if self.exception:
            raise RuntimeError("sem rede")
        return ""


def make_cues(*texts):
    return [Cue.from_source(i, i * 1000, i * 1000 + 900, t)
            for i, t in enumerate(texts, start=1)]


def factory_for(translator_class, **kwargs):
    def factory(source, target):
        return translator_class(source, target, **kwargs)
    return factory


class TestTraducaoBasica(unittest.TestCase):
    def test_traduz_todas_as_legendas(self):
        cues = make_cues("Hello", "World")
        report = translate_cues(
            cues, "en", translator_factory=factory_for(FakeTranslator),
            min_interval=0)
        self.assertEqual(report.translated, 2)
        self.assertEqual(report.failure_count, 0)
        self.assertEqual([c.text for c in cues], ["[pt] Hello", "[pt] World"])

    def test_texto_de_origem_e_preservado(self):
        cues = make_cues("Hello")
        translate_cues(cues, "en", translator_factory=factory_for(FakeTranslator),
                       min_interval=0)
        self.assertEqual(cues[0].source_text, "Hello")
        self.assertNotEqual(cues[0].text, cues[0].source_text)

    def test_lista_vazia_nao_quebra(self):
        report = translate_cues([], "en",
                                translator_factory=factory_for(FakeTranslator))
        self.assertEqual(report.total, 0)


class TestPreservacaoDeFormatacao(unittest.TestCase):
    def test_italico_sobrevive_a_traducao(self):
        cues = make_cues("<i>Hello</i>")
        translate_cues(cues, "en", translator_factory=factory_for(FakeTranslator),
                       min_interval=0)
        self.assertTrue(cues[0].text.startswith("<i>"))
        self.assertTrue(cues[0].text.endswith("</i>"))

    def test_dialogo_mantem_os_dois_turnos(self):
        cues = make_cues("- Did you see?\n- I did.")
        translate_cues(cues, "en", translator_factory=factory_for(FakeTranslator),
                       min_interval=0)
        linhas = cues[0].text.split("\n")
        self.assertEqual(len(linhas), 2)
        self.assertTrue(all(l.startswith("- ") for l in linhas))

    def test_cada_turno_de_dialogo_e_traduzido_separadamente(self):
        translators = []

        def factory(source, target):
            t = FakeTranslator(source, target)
            translators.append(t)
            return t

        cues = make_cues("- Did you see?\n- I did.")
        translate_cues(cues, "en", translator_factory=factory, min_interval=0)
        chamadas = [c for t in translators for c in t.calls]
        self.assertEqual(chamadas, ["Did you see?", "I did."])


class TestFalhasDoTradutor(unittest.TestCase):
    def test_excecao_mantem_o_texto_original(self):
        cues = make_cues("Hello")
        report = translate_cues(
            cues, "en", translator_factory=factory_for(FailingTranslator),
            min_interval=0, max_retries=2)
        self.assertEqual(report.translated, 0)
        self.assertEqual(report.failed, [1])
        # A legenda não pode ficar vazia nem com lixo.
        self.assertEqual(cues[0].text, "Hello")

    def test_retorno_vazio_conta_como_falha(self):
        cues = make_cues("Hello")
        report = translate_cues(
            cues, "en",
            translator_factory=factory_for(FailingTranslator, exception=False),
            min_interval=0, max_retries=2)
        self.assertEqual(report.failed, [1])
        self.assertEqual(cues[0].text, "Hello")

    def test_tenta_novamente_antes_de_desistir(self):
        tentativas = []

        class FlakyTranslator:
            def __init__(self, source, target):
                pass

            def translate(self, text):
                tentativas.append(text)
                if len(tentativas) < 3:
                    raise RuntimeError("instável")
                return "[pt] " + text

        cues = make_cues("Hello")
        report = translate_cues(cues, "en",
                                translator_factory=lambda s, t: FlakyTranslator(s, t),
                                min_interval=0, max_retries=3)
        self.assertEqual(report.translated, 1)
        self.assertEqual(len(tentativas), 3)


class TestProgresso(unittest.TestCase):
    def test_contagem_final_bate_com_o_total(self):
        # Muitas legendas com várias threads: expõe corrida no contador.
        cues = make_cues(*[f"line {i}" for i in range(200)])
        vistos = []
        lock = threading.Lock()

        def progress(done, total):
            with lock:
                vistos.append(done)

        report = translate_cues(
            cues, "en", translator_factory=factory_for(FakeTranslator),
            progress=progress, min_interval=0, max_workers=8, chunk_size=5)

        self.assertEqual(report.translated, 200)
        self.assertEqual(len(vistos), 200)
        self.assertEqual(max(vistos), 200)
        # Sem perda de incremento: todos os valores de 1 a 200 aparecem.
        self.assertEqual(sorted(vistos), list(range(1, 201)))


class TestCancelamento(unittest.TestCase):
    def test_cancelar_levanta_excecao(self):
        cues = make_cues(*[f"line {i}" for i in range(100)])
        cancel = threading.Event()
        cancel.set()

        with self.assertRaises(TranslationCancelled):
            translate_cues(cues, "en",
                           translator_factory=factory_for(FakeTranslator),
                           cancel_event=cancel, min_interval=0)

    def test_cancelar_no_meio_interrompe_o_trabalho(self):
        total = 400
        cues = make_cues(*[f"line {i}" for i in range(total)])
        cancel = threading.Event()
        traduzidas = []
        lock = threading.Lock()

        class CountingTranslator:
            def __init__(self, source, target):
                pass

            def translate(self, text):
                with lock:
                    traduzidas.append(text)
                    quantas = len(traduzidas)
                if quantas >= 50:
                    cancel.set()
                return "[pt] " + text

        with self.assertRaises(TranslationCancelled):
            translate_cues(cues, "en",
                           translator_factory=lambda s, t: CountingTranslator(s, t),
                           cancel_event=cancel, min_interval=0,
                           max_workers=4, chunk_size=10)

        # O ponto do defeito 3: o trabalho para logo, em vez de seguir até o fim.
        self.assertLess(len(traduzidas), total,
                        "o trabalho continuou depois do cancelamento")


if __name__ == "__main__":
    unittest.main()
