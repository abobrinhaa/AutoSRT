import threading
import unittest

from autosrt import llm_translate
from autosrt.cue import Cue
from autosrt.llm import LLMError
from autosrt.llm_translate import (LLMTranslationError, build_prompt,
                                   describe_speakers, parse_response,
                                   translate_cues_llm)
from autosrt.translate import TranslationCancelled


def make_cues(*textos, speakers=None):
    cues = []
    for i, texto in enumerate(textos, start=1):
        speaker = speakers[i - 1] if speakers else None
        cues.append(Cue.from_source(i, i * 1000, i * 1000 + 900, texto,
                                    speaker=speaker))
    return cues


class EchoClient:
    """Devolve os blocos pedidos, prefixando o texto."""

    def __init__(self, prefixo="PT:"):
        self.prefixo = prefixo
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        blocos = []
        for numero, conteudo in llm_translate.BLOCK_RE.findall(user):
            blocos.append(f"<{numero}>{self.prefixo}{conteudo.strip()}</{numero}>")
        return "\n".join(blocos)


class TestDescribeSpeakers(unittest.TestCase):
    def test_vazio_nao_gera_texto(self):
        self.assertEqual(describe_speakers(None), "")
        self.assertEqual(describe_speakers({}), "")

    def test_lista_os_locutores(self):
        texto = describe_speakers({"SPEAKER_00": "homem", "SPEAKER_01": "mulher"})
        self.assertIn("SPEAKER_00 é homem", texto)
        self.assertIn("SPEAKER_01 é mulher", texto)


class TestBuildPrompt(unittest.TestCase):
    def test_inclui_as_legendas_numeradas(self):
        cues = make_cues("Hello", "World")
        prompt = build_prompt([(1, cues[0]), (2, cues[1])], [], [], "inglês")
        self.assertIn("<1>", prompt)
        self.assertIn("<2>", prompt)
        self.assertIn("Hello", prompt)

    def test_contexto_aparece_marcado_e_fora_dos_blocos(self):
        cues = make_cues("Antes", "Alvo", "Depois")
        prompt = build_prompt([(2, cues[1])], [cues[0]], [cues[2]], "inglês")
        self.assertIn("CONTEXTO", prompt)
        self.assertIn("Antes", prompt)
        self.assertIn("Depois", prompt)
        # O contexto nao pode estar em bloco numerado, senao vira traducao.
        numeros = {int(n) for n, _ in llm_translate.BLOCK_RE.findall(prompt)}
        self.assertEqual(numeros, {2})

    def test_locutor_entra_no_bloco(self):
        cues = make_cues("I'm a lawyer.", speakers=["SPEAKER_00"])
        prompt = build_prompt([(1, cues[0])], [], [], "inglês")
        self.assertIn("SPEAKER_00", prompt)

    def test_genero_dos_locutores_entra_no_prompt(self):
        cues = make_cues("I'm a lawyer.", speakers=["SPEAKER_00"])
        prompt = build_prompt([(1, cues[0])], [], [], "inglês",
                              {"SPEAKER_00": "homem"})
        self.assertIn("SPEAKER_00 é homem", prompt)


class TestParseResponse(unittest.TestCase):
    def test_extrai_blocos(self):
        self.assertEqual(
            parse_response("<1>Olá</1>\n<2>Mundo</2>", {1, 2}),
            {1: "Olá", 2: "Mundo"})

    def test_ignora_numeros_nao_pedidos(self):
        self.assertEqual(parse_response("<1>Olá</1><9>Lixo</9>", {1}), {1: "Olá"})

    def test_tolera_conversa_em_volta(self):
        texto = "Claro! Aqui está:\n<1>Olá</1>\nEspero ter ajudado."
        self.assertEqual(parse_response(texto, {1}), {1: "Olá"})

    def test_preserva_quebra_interna(self):
        self.assertEqual(
            parse_response("<1>- Viu?\n- Vi.</1>", {1}), {1: "- Viu?\n- Vi."})

    def test_remove_rotulo_de_locutor_repetido(self):
        self.assertEqual(
            parse_response("<1>(SPEAKER_00) Olá</1>", {1}), {1: "Olá"})

    def test_resposta_vazia(self):
        self.assertEqual(parse_response("", {1}), {})


class TestTraducao(unittest.TestCase):
    def test_traduz_todas(self):
        cues = make_cues("Hello", "World", "Again")
        falhas = translate_cues_llm(cues, "inglês", client=EchoClient(),
                                    block_size=2)
        self.assertEqual(falhas, [])
        self.assertTrue(all(c.text.startswith("PT:") for c in cues))

    def test_texto_de_origem_preservado(self):
        cues = make_cues("Hello")
        translate_cues_llm(cues, "inglês", client=EchoClient())
        self.assertEqual(cues[0].source_text, "Hello")

    def test_sem_cliente_e_erro(self):
        with self.assertRaises(LLMTranslationError):
            translate_cues_llm(make_cues("Hello"), "inglês")

    def test_relata_progresso(self):
        cues = make_cues("a", "b", "c")
        vistos = []
        translate_cues_llm(cues, "inglês", client=EchoClient(), block_size=2,
                           progress=lambda d, t: vistos.append((d, t)))
        self.assertEqual(vistos[-1], (3, 3))

    def test_blocos_respeitam_o_tamanho(self):
        cliente = EchoClient()
        translate_cues_llm(make_cues(*[f"linha {i}" for i in range(10)]),
                           "inglês", client=cliente, block_size=4)
        # 10 legendas em blocos de 4 => 3 requisicoes.
        self.assertEqual(len(cliente.prompts), 3)

    def test_cancelamento(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(TranslationCancelled):
            translate_cues_llm(make_cues("a", "b"), "inglês",
                               client=EchoClient(), cancel_event=cancel)


class TestAlinhamento(unittest.TestCase):
    """A contagem de legendas nao pode mudar, senao os tempos desalinham."""

    def test_bloco_incompleto_cai_para_traducao_individual(self):
        class MetadeClient:
            def __init__(self):
                self.chamadas = 0

            def complete(self, system, user):
                self.chamadas += 1
                blocos = llm_translate.BLOCK_RE.findall(user)
                if len(blocos) > 1:
                    # Devolve so o primeiro: simula o modelo juntando falas.
                    numero, conteudo = blocos[0]
                    return f"<{numero}>PT:{conteudo.strip()}</{numero}>"
                numero, conteudo = blocos[0]
                return f"<{numero}>PT:{conteudo.strip()}</{numero}>"

        cliente = MetadeClient()
        cues = make_cues("um", "dois", "tres")
        falhas = translate_cues_llm(cues, "inglês", client=cliente, block_size=3)

        self.assertEqual(falhas, [])
        self.assertEqual(len(cues), 3)
        self.assertTrue(all(c.text.startswith("PT:") for c in cues))
        # Uma tentativa em bloco mais uma individual por legenda que faltou.
        self.assertEqual(cliente.chamadas, 3)

    def test_modelo_que_falha_mantem_texto_original(self):
        class QuebradoClient:
            def complete(self, system, user):
                raise LLMError("sem creditos")

        cues = make_cues("Hello", "World")
        falhas = translate_cues_llm(cues, "inglês", client=QuebradoClient(),
                                    block_size=2)
        self.assertEqual(falhas, [1, 2])
        self.assertEqual([c.text for c in cues], ["Hello", "World"])

    def test_resposta_sem_delimitador_ainda_e_aproveitada(self):
        class SemTagClient:
            def __init__(self):
                self.chamadas = 0

            def complete(self, system, user):
                self.chamadas += 1
                return "Olá mundo" if self.chamadas > 1 else "resposta inutil"

        cues = make_cues("Hello")
        falhas = translate_cues_llm(cues, "inglês", client=SemTagClient())
        self.assertEqual(falhas, [])
        self.assertEqual(cues[0].text, "Olá mundo")

    def test_numero_de_legendas_nunca_muda(self):
        class CaoticoClient:
            def complete(self, system, user):
                return "<1>a</1><1>b</1><99>c</99>"

        cues = make_cues("um", "dois")
        translate_cues_llm(cues, "inglês", client=CaoticoClient(), block_size=2)
        self.assertEqual(len(cues), 2)


if __name__ == "__main__":
    unittest.main()
