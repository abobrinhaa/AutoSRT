import re
import threading
import time
import unittest

from autosrt import llm_translate
from autosrt.cue import Cue
from autosrt.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMError
from autosrt.llm_translate import (LLMTranslationError, build_prompt,
                                   client_from_config, describe_speakers,
                                   parse_response, translate_cues_llm)
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


class TestPromptDeNomeProprio(unittest.TestCase):
    """Regressão: o modelo traduziu o sobrenome "Cannon" como "canhão" --
    o system prompt precisa dizer explicitamente para não traduzir nome
    próprio, com esse caso como exemplo."""

    def test_regra_de_nome_proprio_esta_no_prompt(self):
        self.assertIn("Cannon", llm_translate.SYSTEM_PROMPT)
        self.assertIn("canhão", llm_translate.SYSTEM_PROMPT)

    def test_pede_para_nao_traduzir_nomes(self):
        self.assertIn("não se traduz", llm_translate.SYSTEM_PROMPT.lower())


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

    def test_remove_rotulo_em_formato_diferente_do_prompt(self):
        # Regressão: build_prompt manda a dica como "(SPEAKER_00)", mas o
        # modelo é livre para ecoar em outro formato -- nesse caso relatado,
        # colchetes com dois-pontos, que a limpeza não reconhecia porque só
        # sabia o formato exato de parênteses.
        self.assertEqual(
            parse_response(
                "<1539>[SPEAKER_15]: Pense em mim quando estiver fumando.</1539>",
                {1539}),
            {1539: "Pense em mim quando estiver fumando."})

    def test_remove_rotulo_sem_parenteses_nem_colchetes(self):
        self.assertEqual(
            parse_response("<1>SPEAKER_00: Olá</1>", {1}), {1: "Olá"})

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

    def test_rotulo_de_locutor_nao_vaza_para_a_legenda_final(self):
        # Regressão relatada: o modelo ecoou a dica de locutor no formato
        # "[SPEAKER_15]: fala" (colchetes+dois-pontos) em vez do formato
        # "(SPEAKER_15)" que build_prompt manda, e o rótulo sobrava na
        # legenda final porque só o formato de parênteses era limpo.
        class ClienteQueEcoaColchetes:
            def complete(self, system, user):
                blocos = []
                for numero, conteudo in llm_translate.BLOCK_RE.findall(user):
                    speaker = "SPEAKER_15" if numero == "1" else "SPEAKER_09"
                    # conteudo já vem com "(SPEAKER_XX) " na frente -- é a
                    # dica que build_prompt manda; troca pelo formato que o
                    # modelo "ecoou" em vez de manter os dois.
                    texto = re.sub(r"^\(SPEAKER_\d+\)\s*", "", conteudo.strip())
                    blocos.append(f"<{numero}>[{speaker}]: PT:{texto}</{numero}>")
                return "\n".join(blocos)

        cues = make_cues("Think of me when you're smoking.", "Oh, I will.",
                         speakers=["SPEAKER_15", "SPEAKER_09"])
        falhas = translate_cues_llm(cues, "inglês",
                                    client=ClienteQueEcoaColchetes())

        self.assertEqual(falhas, [])
        for cue in cues:
            self.assertNotIn("SPEAKER", cue.text)
        self.assertEqual(cues[0].text,
                         "PT:Think of me when you're smoking.")

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

    def test_on_error_recebe_o_motivo_real_da_falha(self):
        class QuebradoClient:
            def complete(self, system, user):
                raise LLMError("HTTP 404: No endpoints found for o/modelo-morto")

        vistos = []
        cues = make_cues("Hello", "World")
        translate_cues_llm(cues, "inglês", client=QuebradoClient(),
                           block_size=2, on_error=vistos.append)

        self.assertTrue(vistos)
        self.assertIn("404", str(vistos[-1]))
        self.assertIn("modelo-morto", str(vistos[-1]))

    def test_sem_on_error_nao_quebra_nada(self):
        # O parâmetro é opcional -- quem não passa nada continua funcionando
        # como antes.
        class QuebradoClient:
            def complete(self, system, user):
                raise LLMError("sem creditos")

        cues = make_cues("Hello")
        falhas = translate_cues_llm(cues, "inglês", client=QuebradoClient())
        self.assertEqual(falhas, [1])

    def test_on_error_tambem_dispara_na_tentativa_individual(self):
        # Bloco com mais de uma legenda: a primeira falha na resposta em
        # bloco (formato errado) e cai para tentativa individual, que também
        # falha -- o motivo relatado deve ser o da tentativa individual.
        class MeioQuebradoClient:
            def __init__(self):
                self.chamadas = 0

            def complete(self, system, user):
                self.chamadas += 1
                if self.chamadas == 1:
                    return "resposta sem os delimitadores esperados"
                raise LLMError("HTTP 429: limite de taxa excedido")

        vistos = []
        cues = make_cues("Hello", "World")
        translate_cues_llm(cues, "inglês", client=MeioQuebradoClient(),
                           block_size=2, on_error=vistos.append)

        self.assertTrue(any("429" in str(e) for e in vistos))

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


class TestVerificacaoDeMudancaDeIdioma(unittest.TestCase):
    """Regressão: bater a contagem de blocos não prova que a tradução
    aconteceu -- um modelo travado que devolve a entrada intacta passava
    despercebido, porque a única checagem era "veio um bloco pra cada
    número pedido"."""

    def test_eco_da_entrada_conta_como_falha(self):
        class EcoClient:
            def complete(self, system, user):
                blocos = []
                for numero, conteudo in llm_translate.BLOCK_RE.findall(user):
                    blocos.append(f"<{numero}>{conteudo}</{numero}>")
                return "\n".join(blocos)

        cues = make_cues("Hello", "World")
        falhas = translate_cues_llm(cues, "inglês", client=EcoClient(),
                                    block_size=2)

        self.assertEqual(falhas, [1, 2])
        self.assertEqual([c.text for c in cues], ["Hello", "World"])

    def test_eco_so_em_uma_legenda_cai_para_individual(self):
        # A primeira legenda do bloco vem intacta (falha), a segunda vem
        # traduzida de verdade -- só a primeira deve cair pra tentativa
        # individual, e como continua intacta lá também, falha de vez.
        class MeioEcoClient:
            def complete(self, system, user):
                blocos = []
                for numero, conteudo in llm_translate.BLOCK_RE.findall(user):
                    conteudo = conteudo.strip()
                    texto = conteudo if conteudo == "Hello" else f"PT:{conteudo}"
                    blocos.append(f"<{numero}>{texto}</{numero}>")
                return "\n".join(blocos)

        cues = make_cues("Hello", "World")
        falhas = translate_cues_llm(cues, "inglês", client=MeioEcoClient(),
                                    block_size=2)

        self.assertEqual(falhas, [1])
        self.assertEqual(cues[0].text, "Hello")
        self.assertEqual(cues[1].text, "PT:World")

    def test_diferenca_de_maiusculas_ou_espaco_nao_conta_como_traducao(self):
        class QuaseEcoClient:
            def complete(self, system, user):
                blocos = []
                for numero, conteudo in llm_translate.BLOCK_RE.findall(user):
                    texto = f"  {conteudo.strip().upper()}  "
                    blocos.append(f"<{numero}>{texto}</{numero}>")
                return "\n".join(blocos)

        cues = make_cues("Hello")
        falhas = translate_cues_llm(cues, "inglês", client=QuaseEcoClient())
        self.assertEqual(falhas, [1])


class TestTamanhoDeBlocoAutomatico(unittest.TestCase):
    """Motor local (Ollama etc.) ganha bloco bem menor sozinho, sem que quem
    chama precise saber disso -- é o caso do pipeline, que não passa
    block_size explicitamente."""

    def test_local_usa_bloco_pequeno_por_padrao(self):
        class ClienteLocal(EchoClient):
            base_url = "http://localhost:11434/v1"

        cliente = ClienteLocal()
        cues = make_cues(*[f"linha {i}" for i in range(4)])
        translate_cues_llm(cues, "inglês", client=cliente)

        # DEFAULT_LOCAL_BLOCK_SIZE == 2: 4 legendas viram 2 requisições.
        self.assertEqual(len(cliente.prompts), 2)

    def test_provedor_normal_usa_bloco_grande_por_padrao(self):
        class ClienteNuvem(EchoClient):
            base_url = "https://openrouter.ai/api/v1"

        cliente = ClienteNuvem()
        cues = make_cues(*[f"linha {i}" for i in range(4)])
        translate_cues_llm(cues, "inglês", client=cliente)

        # DEFAULT_BLOCK_SIZE == 20: as 4 legendas cabem numa requisição só.
        self.assertEqual(len(cliente.prompts), 1)

    def test_block_size_explicito_tem_prioridade_sobre_deteccao(self):
        class ClienteLocal(EchoClient):
            base_url = "http://localhost:11434/v1"

        cliente = ClienteLocal()
        cues = make_cues(*[f"linha {i}" for i in range(4)])
        translate_cues_llm(cues, "inglês", client=cliente, block_size=4)

        self.assertEqual(len(cliente.prompts), 1)


class TestParalelismo(unittest.TestCase):
    """Regressão de desempenho: os blocos precisam rodar ao mesmo tempo, não
    um de cada vez -- era isso que fazia o motor padrão (LLM) ser mais lento
    que o alternativo (Google), que já é paralelo."""

    def test_blocos_rodam_ao_mesmo_tempo(self):
        simultaneos = []
        pico = []
        lock = threading.Lock()

        class SlowClient:
            def complete(self, system, user):
                with lock:
                    simultaneos.append(1)
                    pico.append(len(simultaneos))
                time.sleep(0.05)
                blocos = [f"<{n}>PT:{c.strip()}</{n}>"
                         for n, c in llm_translate.BLOCK_RE.findall(user)]
                with lock:
                    simultaneos.pop()
                return "\n".join(blocos)

        cues = make_cues(*[f"linha {i}" for i in range(9)])
        translate_cues_llm(cues, "inglês", client=SlowClient(), block_size=3,
                           max_workers=3)

        self.assertGreater(max(pico), 1)

    def test_nenhum_incremento_de_progresso_se_perde(self):
        cues = make_cues(*[f"linha {i}" for i in range(60)])
        vistos = []
        lock = threading.Lock()

        def progress(done, total):
            with lock:
                vistos.append(done)

        falhas = translate_cues_llm(cues, "inglês", client=EchoClient(),
                                    block_size=3, max_workers=6,
                                    progress=progress)

        self.assertEqual(falhas, [])
        self.assertEqual(len(vistos), 60)
        # Sem perda de incremento: todos os valores de 1 a 60 aparecem.
        self.assertEqual(sorted(vistos), list(range(1, 61)))

    def test_cancelar_no_meio_interrompe_blocos_pendentes(self):
        cancel = threading.Event()
        chamadas = []
        lock = threading.Lock()

        class CancelaNoMeioClient:
            def complete(self, system, user):
                with lock:
                    chamadas.append(1)
                    quantas = len(chamadas)
                if quantas >= 2:
                    cancel.set()
                time.sleep(0.02)
                blocos = [f"<{n}>PT:{c.strip()}</{n}>"
                         for n, c in llm_translate.BLOCK_RE.findall(user)]
                return "\n".join(blocos)

        cues = make_cues(*[f"linha {i}" for i in range(30)])
        with self.assertRaises(TranslationCancelled):
            translate_cues_llm(cues, "inglês", client=CancelaNoMeioClient(),
                               block_size=2, max_workers=2, cancel_event=cancel)

        # Nem todos os blocos rodaram: o cancelamento cortou o resto.
        self.assertLess(len(chamadas), 15)


class TestClientFromConfig(unittest.TestCase):
    """Regressão: um endereço customizado (servidor local, por exemplo) sem
    modelo configurado caía de volta no DEFAULT_MODEL do OpenRouter --
    silenciosamente errado, porque esse nome quase certamente não existe
    num servidor local. O sintoma relatado foi o campo de modelo "sumir" e
    virar o modelo do OpenRouter sozinho."""

    def getter_de(self, valores):
        return lambda nome, env=None: valores.get(nome)

    def test_sem_base_url_customizado_usa_o_padrao_do_openrouter(self):
        cliente = client_from_config(config_getter=self.getter_de({}),
                                     api_key="chave")
        self.assertEqual(cliente.model, DEFAULT_MODEL)
        self.assertEqual(cliente.base_url, DEFAULT_BASE_URL)

    def test_base_url_do_openrouter_explicita_tambem_usa_o_padrao(self):
        cliente = client_from_config(config_getter=self.getter_de({
            "llm_base_url": DEFAULT_BASE_URL}), api_key="chave")
        self.assertEqual(cliente.model, DEFAULT_MODEL)

    def test_base_url_customizado_sem_modelo_e_erro_claro(self):
        with self.assertRaises(LLMError) as ctx:
            client_from_config(config_getter=self.getter_de({
                "llm_base_url": "http://localhost:11434/v1"}), api_key="chave")
        mensagem = str(ctx.exception)
        self.assertIn("localhost:11434", mensagem)
        self.assertIn("modelo", mensagem.lower())

    def test_base_url_customizado_com_modelo_funciona(self):
        cliente = client_from_config(config_getter=self.getter_de({
            "llm_base_url": "http://localhost:11434/v1",
            "llm_model": "llama3.1"}), api_key="chave")
        self.assertEqual(cliente.model, "llama3.1")
        self.assertEqual(cliente.base_url, "http://localhost:11434/v1")


if __name__ == "__main__":
    unittest.main()
