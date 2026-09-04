import json
import os
import tempfile
import unittest

from autosrt import config


class TestLocalDoArquivo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._env = {k: os.environ.get(k) for k in
                     ("AUTOSRT_CONFIG_DIR", "XDG_CONFIG_HOME")}
        for chave in self._env:
            os.environ.pop(chave, None)

    def tearDown(self):
        for chave, valor in self._env.items():
            if valor is None:
                os.environ.pop(chave, None)
            else:
                os.environ[chave] = valor

    def test_variavel_dedicada_tem_prioridade(self):
        os.environ["AUTOSRT_CONFIG_DIR"] = self.tmp
        self.assertEqual(config.app_directory(), self.tmp)

    def test_cai_no_diretorio_de_configuracao_do_usuario(self):
        os.environ["XDG_CONFIG_HOME"] = self.tmp
        self.assertEqual(config.app_directory(),
                         os.path.join(self.tmp, "autosrt"))

    def test_nao_grava_dentro_do_pacote(self):
        # site-packages costuma ser somente leitura.
        pacote = os.path.dirname(os.path.abspath(config.__file__))
        self.assertNotIn(pacote, config.config_path())


class TestLeituraEscrita(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._anterior = os.environ.get("AUTOSRT_CONFIG_DIR")
        os.environ["AUTOSRT_CONFIG_DIR"] = os.path.join(self.tmp, "sub", "dir")

    def tearDown(self):
        if self._anterior is None:
            os.environ.pop("AUTOSRT_CONFIG_DIR", None)
        else:
            os.environ["AUTOSRT_CONFIG_DIR"] = self._anterior

    def test_cria_a_pasta_se_faltar(self):
        config.save_config({"llm_model": "x"})
        self.assertTrue(os.path.exists(config.config_path()))

    def test_ida_e_volta(self):
        config.save_config({"openrouter_api_key": "sk-teste"})
        self.assertEqual(config.get_openrouter_api_key(), "sk-teste")

    def test_preserva_o_que_ja_estava(self):
        config.save_config({"llm_model": "modelo-a"})
        config.save_config({"openrouter_api_key": "sk-teste"})
        dados = config.load_config()
        self.assertEqual(dados["llm_model"], "modelo-a")
        self.assertEqual(dados["openrouter_api_key"], "sk-teste")

    def test_valor_vazio_remove(self):
        config.save_config({"llm_model": "modelo-a"})
        config.save_config({"llm_model": ""})
        self.assertNotIn("llm_model", config.load_config())

    def test_arquivo_corrompido_nao_derruba(self):
        os.makedirs(config.app_directory(), exist_ok=True)
        with open(config.config_path(), "w", encoding="utf-8") as handle:
            handle.write("{isso nao e json")
        self.assertEqual(config.load_config(), {})

    def test_ambiente_vence_o_arquivo(self):
        config.save_config({"openrouter_api_key": "do-arquivo"})
        os.environ["OPENROUTER_API_KEY"] = "do-ambiente"
        try:
            self.assertEqual(config.get_openrouter_api_key(), "do-ambiente")
        finally:
            os.environ.pop("OPENROUTER_API_KEY", None)

    def test_json_gravado_e_legivel(self):
        config.save_config({"llm_model": "x"})
        with open(config.config_path(), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["llm_model"], "x")


class TestSensibilidadeDaVAD(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._anterior = os.environ.get("AUTOSRT_CONFIG_DIR")
        os.environ["AUTOSRT_CONFIG_DIR"] = self.tmp
        self._env = {k: os.environ.pop(k, None) for k in
                     ("AUTOSRT_VAD_THRESHOLD", "AUTOSRT_VAD_MIN_SILENCE_MS")}

    def tearDown(self):
        if self._anterior is None:
            os.environ.pop("AUTOSRT_CONFIG_DIR", None)
        else:
            os.environ["AUTOSRT_CONFIG_DIR"] = self._anterior
        for chave, valor in self._env.items():
            if valor is not None:
                os.environ[chave] = valor
            else:
                os.environ.pop(chave, None)

    def test_sem_configuracao_e_none(self):
        self.assertIsNone(config.get_vad_threshold())
        self.assertIsNone(config.get_vad_min_silence_ms())

    def test_le_do_arquivo(self):
        config.save_config({"vad_threshold": "0.2", "vad_min_silence_ms": "300"})
        self.assertEqual(config.get_vad_threshold(), 0.2)
        self.assertEqual(config.get_vad_min_silence_ms(), 300)

    def test_ambiente_tem_prioridade(self):
        config.save_config({"vad_threshold": "0.2"})
        os.environ["AUTOSRT_VAD_THRESHOLD"] = "0.05"
        self.assertEqual(config.get_vad_threshold(), 0.05)

    def test_valor_invalido_nao_derruba(self):
        config.save_config({"vad_threshold": "nao-e-numero"})
        self.assertIsNone(config.get_vad_threshold())


class TestAntiAlucinacao(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._anterior = os.environ.get("AUTOSRT_CONFIG_DIR")
        os.environ["AUTOSRT_CONFIG_DIR"] = self.tmp
        self._env = {k: os.environ.pop(k, None) for k in
                     ("AUTOSRT_CONDITION_ON_PREVIOUS_TEXT",
                      "AUTOSRT_HALLUCINATION_SILENCE_THRESHOLD")}

    def tearDown(self):
        if self._anterior is None:
            os.environ.pop("AUTOSRT_CONFIG_DIR", None)
        else:
            os.environ["AUTOSRT_CONFIG_DIR"] = self._anterior
        for chave, valor in self._env.items():
            if valor is not None:
                os.environ[chave] = valor
            else:
                os.environ.pop(chave, None)

    def test_sem_configuracao_e_none(self):
        # None, não False: quem chama não deve repassar nada, deixando o
        # padrão (desligado) de transcribe.py valer sozinho.
        self.assertIsNone(config.get_condition_on_previous_text())
        self.assertIsNone(config.get_hallucination_silence_threshold())

    def test_le_do_arquivo(self):
        config.save_config({"condition_on_previous_text": "true",
                            "hallucination_silence_threshold": "2"})
        self.assertTrue(config.get_condition_on_previous_text())
        self.assertEqual(config.get_hallucination_silence_threshold(), 2.0)

    def test_false_explicito_e_diferente_de_nao_configurado(self):
        config.save_config({"condition_on_previous_text": "false"})
        self.assertFalse(config.get_condition_on_previous_text())

    def test_ambiente_tem_prioridade(self):
        config.save_config({"hallucination_silence_threshold": "2"})
        os.environ["AUTOSRT_HALLUCINATION_SILENCE_THRESHOLD"] = "4"
        self.assertEqual(config.get_hallucination_silence_threshold(), 4.0)

    def test_valor_invalido_nao_derruba(self):
        config.save_config({"hallucination_silence_threshold": "nao-e-numero"})
        self.assertIsNone(config.get_hallucination_silence_threshold())


class TestProcessarAoEnviar(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._anterior = os.environ.get("AUTOSRT_CONFIG_DIR")
        os.environ["AUTOSRT_CONFIG_DIR"] = self.tmp
        self._env = os.environ.pop("AUTOSRT_AUTO_PROCESSAR", None)

    def tearDown(self):
        if self._anterior is None:
            os.environ.pop("AUTOSRT_CONFIG_DIR", None)
        else:
            os.environ["AUTOSRT_CONFIG_DIR"] = self._anterior
        # Repor só quando havia valor esconderia o que os testes daqui
        # escrevem: a variável vazaria ligada para os módulos seguintes.
        if self._env is None:
            os.environ.pop("AUTOSRT_AUTO_PROCESSAR", None)
        else:
            os.environ["AUTOSRT_AUTO_PROCESSAR"] = self._env

    def test_sem_configuracao_e_desligado(self):
        # Uma transcrição custa meia hora de GPU: ligar por omissão gastaria
        # a placa de quem só quis guardar um arquivo na pasta.
        self.assertFalse(config.get_auto_processar())

    def test_le_do_arquivo(self):
        config.save_config({"auto_processar": "true"})
        self.assertTrue(config.get_auto_processar())

    def test_false_desliga(self):
        config.save_config({"auto_processar": "false"})
        self.assertFalse(config.get_auto_processar())

    def test_ambiente_tem_prioridade(self):
        config.save_config({"auto_processar": "false"})
        os.environ["AUTOSRT_AUTO_PROCESSAR"] = "true"
        self.assertTrue(config.get_auto_processar())

    def test_valor_invalido_nao_liga(self):
        # Na dúvida, não gasta GPU.
        config.save_config({"auto_processar": "talvez"})
        self.assertFalse(config.get_auto_processar())


class TestModeloDoWhisper(unittest.TestCase):
    """A fila web usava o padrão fixo do transcribe.py e não tinha como
    trocar -- o ajuste só existia no --modelo do CLI, justamente na
    interface pensada para quem não abre terminal."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._anterior = os.environ.get("AUTOSRT_CONFIG_DIR")
        os.environ["AUTOSRT_CONFIG_DIR"] = self.tmp
        self._env = {k: os.environ.pop(k, None) for k in
                     ("WHISPER_MODEL", "WHISPER_COMPUTE_TYPE")}

    def tearDown(self):
        if self._anterior is None:
            os.environ.pop("AUTOSRT_CONFIG_DIR", None)
        else:
            os.environ["AUTOSRT_CONFIG_DIR"] = self._anterior
        for chave, valor in self._env.items():
            if valor is not None:
                os.environ[chave] = valor
            else:
                os.environ.pop(chave, None)

    def test_sem_configuracao_e_none(self):
        self.assertIsNone(config.get_whisper_model())
        self.assertIsNone(config.get_whisper_compute_type())

    def test_le_do_arquivo(self):
        config.save_config({"whisper_model": "large-v3",
                            "whisper_compute_type": "int8"})
        self.assertEqual(config.get_whisper_model(), "large-v3")
        self.assertEqual(config.get_whisper_compute_type(), "int8")

    def test_ambiente_tem_prioridade(self):
        config.save_config({"whisper_model": "large-v3"})
        os.environ["WHISPER_MODEL"] = "medium"
        self.assertEqual(config.get_whisper_model(), "medium")


class TestMetodoDaVAD(unittest.TestCase):
    """O detector de fala estava fixo em silero_v5 no código -- nem o padrão
    do próprio executável (silero_v4_fw) era alcançável. Trocar o método é
    candidato a resolver transcrição com buracos, tanto quanto o limiar."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._anterior = os.environ.get("AUTOSRT_CONFIG_DIR")
        os.environ["AUTOSRT_CONFIG_DIR"] = self.tmp
        self._env = os.environ.pop("AUTOSRT_VAD_METHOD", None)

    def tearDown(self):
        if self._anterior is None:
            os.environ.pop("AUTOSRT_CONFIG_DIR", None)
        else:
            os.environ["AUTOSRT_CONFIG_DIR"] = self._anterior
        if self._env is not None:
            os.environ["AUTOSRT_VAD_METHOD"] = self._env
        else:
            os.environ.pop("AUTOSRT_VAD_METHOD", None)

    def test_sem_configuracao_e_none(self):
        self.assertIsNone(config.get_vad_method())

    def test_le_do_arquivo(self):
        config.save_config({"vad_method": "silero_v4_fw"})
        self.assertEqual(config.get_vad_method(), "silero_v4_fw")

    def test_ambiente_tem_prioridade(self):
        config.save_config({"vad_method": "silero_v4_fw"})
        os.environ["AUTOSRT_VAD_METHOD"] = "webrtc"
        self.assertEqual(config.get_vad_method(), "webrtc")


class TestTamanhoDeBlocoDoLLM(unittest.TestCase):
    """llm_block_size sobrepõe a detecção automática por endereço em
    translate_cues_llm -- necessário para quem serve um modelo local por um
    IP que a heurística não reconhece, ou quer forçar outro valor."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._anterior = os.environ.get("AUTOSRT_CONFIG_DIR")
        os.environ["AUTOSRT_CONFIG_DIR"] = self.tmp
        self._env_llm = os.environ.pop("LLM_BLOCK_SIZE", None)

    def tearDown(self):
        if self._anterior is None:
            os.environ.pop("AUTOSRT_CONFIG_DIR", None)
        else:
            os.environ["AUTOSRT_CONFIG_DIR"] = self._anterior
        if self._env_llm is not None:
            os.environ["LLM_BLOCK_SIZE"] = self._env_llm
        else:
            os.environ.pop("LLM_BLOCK_SIZE", None)

    def test_sem_configuracao_e_none(self):
        self.assertIsNone(config.get_llm_block_size())

    def test_le_do_arquivo(self):
        config.save_config({"llm_block_size": "2"})
        self.assertEqual(config.get_llm_block_size(), 2)

    def test_ambiente_tem_prioridade(self):
        config.save_config({"llm_block_size": "2"})
        os.environ["LLM_BLOCK_SIZE"] = "5"
        self.assertEqual(config.get_llm_block_size(), 5)

    def test_valor_invalido_nao_derruba(self):
        config.save_config({"llm_block_size": "nao-e-numero"})
        self.assertIsNone(config.get_llm_block_size())


class TestFiltroDeAlucinacao(unittest.TestCase):
    """Ligado por padrão: é correção de defeito, não preferência.

    A frase inventada pelo Whisper ("Obrigado por assistir" sobre o
    silêncio final) chega ao arquivo sem nenhum aviso, e quem não sabe que
    ela existe não vai procurar um botão para desligá-la.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._anterior = os.environ.get("AUTOSRT_CONFIG_DIR")
        os.environ["AUTOSRT_CONFIG_DIR"] = self.tmp
        self._env = {k: os.environ.pop(k, None) for k in
                     ("AUTOSRT_FILTRAR_ALUCINACOES", "AUTOSRT_ALUCINACOES_EXTRA")}

    def tearDown(self):
        if self._anterior is None:
            os.environ.pop("AUTOSRT_CONFIG_DIR", None)
        else:
            os.environ["AUTOSRT_CONFIG_DIR"] = self._anterior
        for chave, valor in self._env.items():
            if valor is not None:
                os.environ[chave] = valor
            else:
                os.environ.pop(chave, None)

    def test_ligado_por_padrao(self):
        self.assertTrue(config.get_filtrar_alucinacoes())

    def test_false_desliga(self):
        config.save_config({"filtrar_alucinacoes": "false"})
        self.assertFalse(config.get_filtrar_alucinacoes())

    def test_valor_desconhecido_nao_desliga(self):
        # Desligar por engano de digitação seria voltar ao defeito em
        # silêncio -- o padrão seguro é continuar filtrando.
        config.save_config({"filtrar_alucinacoes": "talvez"})
        self.assertTrue(config.get_filtrar_alucinacoes())

    def test_ambiente_tem_prioridade(self):
        config.save_config({"filtrar_alucinacoes": "true"})
        os.environ["AUTOSRT_FILTRAR_ALUCINACOES"] = "false"
        self.assertFalse(config.get_filtrar_alucinacoes())

    def test_sem_frases_extras_e_lista_vazia(self):
        self.assertEqual(config.get_alucinacoes_extra(), [])

    def test_frases_extras_uma_por_linha(self):
        config.save_config(
            {"alucinacoes_extra": "Legendas: João\nAssista o próximo"})
        self.assertEqual(config.get_alucinacoes_extra(),
                         ["Legendas: João", "Assista o próximo"])

    def test_frases_extras_tambem_aceitam_lista(self):
        # Quem edita o config.json à mão escreve JSON, não texto com \n.
        config.save_config({"alucinacoes_extra": ["Legendas: João"]})
        self.assertEqual(config.get_alucinacoes_extra(), ["Legendas: João"])

    def test_linha_em_branco_nao_vira_frase(self):
        config.save_config({"alucinacoes_extra": "Legendas: João\n\n   \n"})
        self.assertEqual(config.get_alucinacoes_extra(), ["Legendas: João"])


if __name__ == "__main__":
    unittest.main()
