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


if __name__ == "__main__":
    unittest.main()
