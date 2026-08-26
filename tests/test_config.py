import json
import os
import shutil
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


if __name__ == "__main__":
    unittest.main()


class TestDiarizacao(unittest.TestCase):
    """A diarização é ligada por padrão e desligável pela configuração."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._app_dir = config.app_directory
        config.app_directory = lambda: self.tmp
        self.addCleanup(setattr, config, "app_directory", self._app_dir)
        self._env = os.environ.pop("AUTOSRT_DIARIZE", None)
        self.addCleanup(self._restaurar_env)

    def _restaurar_env(self):
        os.environ.pop("AUTOSRT_DIARIZE", None)
        if self._env is not None:
            os.environ["AUTOSRT_DIARIZE"] = self._env

    def test_padrao_e_ligada(self):
        self.assertTrue(config.get_diarize())

    def test_desligada_pelo_arquivo(self):
        config.save_config({"diarizar": "nao"})
        self.assertFalse(config.get_diarize())

    def test_religada_pelo_arquivo(self):
        config.save_config({"diarizar": "nao"})
        config.save_config({"diarizar": "sim"})
        self.assertTrue(config.get_diarize())

    def test_desligada_pelo_ambiente(self):
        os.environ["AUTOSRT_DIARIZE"] = "nao"
        self.assertFalse(config.get_diarize())


class TestArgumentosExtrasDoWhisper(unittest.TestCase):
    """Escape hatch para opção do Faster-Whisper-XXL sem campo dedicado
    (ex.: --condition_on_previous_text para alucinação que se propaga pelo
    arquivo) -- existia só no --whisper-args do CLI."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._app_dir = config.app_directory
        config.app_directory = lambda: self.tmp
        self.addCleanup(setattr, config, "app_directory", self._app_dir)
        self._env = os.environ.pop("AUTOSRT_WHISPER_EXTRA_ARGS", None)
        self.addCleanup(self._restaurar_env)

    def _restaurar_env(self):
        os.environ.pop("AUTOSRT_WHISPER_EXTRA_ARGS", None)
        if self._env is not None:
            os.environ["AUTOSRT_WHISPER_EXTRA_ARGS"] = self._env

    def test_sem_configuracao_e_none(self):
        self.assertIsNone(config.get_whisper_extra_args())

    def test_le_do_arquivo(self):
        config.save_config({
            "whisper_extra_args": "--condition_on_previous_text False"})
        self.assertEqual(config.get_whisper_extra_args(),
                         "--condition_on_previous_text False")

    def test_ambiente_tem_prioridade(self):
        config.save_config({"whisper_extra_args": "--no_speech_threshold 0.6"})
        os.environ["AUTOSRT_WHISPER_EXTRA_ARGS"] = "--vad_filter False"
        self.assertEqual(config.get_whisper_extra_args(), "--vad_filter False")
