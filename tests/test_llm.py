import unittest

import requests

from autosrt.llm import (DEFAULT_BASE_URL, LLMClient, LLMError,
                         is_local_base_url, list_models)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("sem json")
        return self._payload


def ok_payload(content="Olá"):
    return {"choices": [{"message": {"content": content}}]}


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        resultado = self.responses.pop(0) if self.responses else FakeResponse(
            200, ok_payload())
        if isinstance(resultado, Exception):
            raise resultado
        return resultado

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers})
        resultado = self.responses.pop(0) if self.responses else FakeResponse(
            200, {"data": []})
        if isinstance(resultado, Exception):
            raise resultado
        return resultado


class TestConstrucao(unittest.TestCase):
    def test_sem_chave_e_erro_com_instrucao(self):
        with self.assertRaises(LLMError) as contexto:
            LLMClient(api_key=None)
        self.assertIn("OPENROUTER_API_KEY", str(contexto.exception))

    def test_base_url_padrao_e_openrouter(self):
        cliente = LLMClient(api_key="k", session=FakeSession())
        self.assertEqual(cliente.base_url, DEFAULT_BASE_URL)

    def test_barra_final_e_removida(self):
        cliente = LLMClient(api_key="k", base_url="https://x.com/v1/",
                            session=FakeSession())
        self.assertEqual(cliente.base_url, "https://x.com/v1")


class TestRequisicao(unittest.TestCase):
    def build(self, *responses, **kwargs):
        session = FakeSession(*responses)
        cliente = LLMClient(api_key="chave", session=session, **kwargs)
        return cliente, session

    def test_envia_para_chat_completions(self):
        cliente, session = self.build()
        cliente.complete("sistema", "usuario")
        self.assertTrue(session.calls[0]["url"].endswith("/chat/completions"))

    def test_manda_a_chave_no_cabecalho(self):
        cliente, session = self.build()
        cliente.complete("sistema", "usuario")
        self.assertEqual(session.calls[0]["headers"]["Authorization"],
                         "Bearer chave")

    def test_manda_as_duas_mensagens(self):
        cliente, session = self.build()
        cliente.complete("sistema", "usuario")
        mensagens = session.calls[0]["json"]["messages"]
        self.assertEqual([m["role"] for m in mensagens], ["system", "user"])
        self.assertEqual(mensagens[1]["content"], "usuario")

    def test_temperatura_baixa_por_padrao(self):
        cliente, session = self.build()
        cliente.complete("s", "u")
        self.assertLessEqual(session.calls[0]["json"]["temperature"], 0.3)

    def test_devolve_o_conteudo(self):
        cliente, _ = self.build(FakeResponse(200, ok_payload("traduzido")))
        self.assertEqual(cliente.complete("s", "u"), "traduzido")


class TestErros(unittest.TestCase):
    def build(self, *responses, **kwargs):
        session = FakeSession(*responses)
        return LLMClient(api_key="k", session=session, **kwargs), session

    def test_401_nao_repete(self):
        cliente, session = self.build(
            FakeResponse(401, text="chave invalida"),
            FakeResponse(200, ok_payload()))
        with self.assertRaises(LLMError):
            cliente.complete("s", "u")
        self.assertEqual(len(session.calls), 1)

    def test_429_repete_e_sucede(self):
        cliente, session = self.build(
            FakeResponse(429, text="devagar"),
            FakeResponse(200, ok_payload("pronto")),
            max_retries=3)
        self.assertEqual(cliente.complete("s", "u"), "pronto")
        self.assertEqual(len(session.calls), 2)

    def test_falha_de_rede_repete(self):
        cliente, session = self.build(
            requests.RequestException("caiu"),
            FakeResponse(200, ok_payload("pronto")),
            max_retries=3)
        self.assertEqual(cliente.complete("s", "u"), "pronto")

    def test_desiste_apos_o_limite(self):
        cliente, session = self.build(
            FakeResponse(503), FakeResponse(503), max_retries=2)
        with self.assertRaises(LLMError):
            cliente.complete("s", "u")
        self.assertEqual(len(session.calls), 2)

    def test_erro_no_corpo_com_status_200(self):
        cliente, _ = self.build(
            FakeResponse(200, {"error": {"message": "modelo inexistente"}}))
        with self.assertRaises(LLMError) as contexto:
            cliente.complete("s", "u")
        self.assertIn("modelo inexistente", str(contexto.exception))

    def test_resposta_sem_json(self):
        cliente, _ = self.build(FakeResponse(200, None, text="<html>"))
        with self.assertRaises(LLMError):
            cliente.complete("s", "u")

    def test_formato_inesperado(self):
        cliente, _ = self.build(FakeResponse(200, {"resultado": "?"}))
        with self.assertRaises(LLMError):
            cliente.complete("s", "u")

    def test_conteudo_vazio(self):
        cliente, _ = self.build(FakeResponse(200, ok_payload("   ")))
        with self.assertRaises(LLMError):
            cliente.complete("s", "u")


class TestListModels(unittest.TestCase):
    def test_monta_id_e_preco_do_openrouter(self):
        session = FakeSession(FakeResponse(200, {"data": [
            {"id": "deepseek/deepseek-chat-v3.1",
             "pricing": {"prompt": "0.00000014", "completion": "0.00000028"}},
        ]}))
        modelos = list_models(session=session)
        self.assertEqual(modelos, [{
            "id": "deepseek/deepseek-chat-v3.1",
            "preco": "$0.14/1M entrada · $0.28/1M saída",
        }])

    def test_sem_pricing_e_none(self):
        session = FakeSession(FakeResponse(200, {"data": [{"id": "llama3.1"}]}))
        modelos = list_models(session=session)
        self.assertIsNone(modelos[0]["preco"])

    def test_pricing_zerado_e_gratis(self):
        session = FakeSession(FakeResponse(200, {"data": [
            {"id": "modelo-gratis", "pricing": {"prompt": "0", "completion": "0"}},
        ]}))
        modelos = list_models(session=session)
        self.assertEqual(modelos[0]["preco"], "grátis")

    def test_ordena_por_id(self):
        session = FakeSession(FakeResponse(200, {"data": [
            {"id": "zeta"}, {"id": "alfa"},
        ]}))
        modelos = list_models(session=session)
        self.assertEqual([m["id"] for m in modelos], ["alfa", "zeta"])

    def test_ignora_itens_sem_id(self):
        session = FakeSession(FakeResponse(200, {"data": [
            {"pricing": {}}, {"id": "valido"},
        ]}))
        modelos = list_models(session=session)
        self.assertEqual([m["id"] for m in modelos], ["valido"])

    def test_manda_a_chave_quando_ha_uma(self):
        session = FakeSession(FakeResponse(200, {"data": []}))
        list_models(api_key="sk-teste", session=session)
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Bearer sk-teste")

    def test_sem_chave_nao_manda_cabecalho_de_autorizacao(self):
        session = FakeSession(FakeResponse(200, {"data": []}))
        list_models(session=session)
        self.assertEqual(session.calls[0]["headers"], {})

    def test_erro_http_vira_llmerror(self):
        session = FakeSession(FakeResponse(401, text="sem permissao"))
        with self.assertRaises(LLMError):
            list_models(session=session)

    def test_falha_de_rede_vira_llmerror(self):
        session = FakeSession(requests.ConnectionError("sem rede"))
        with self.assertRaises(LLMError):
            list_models(session=session)

    def test_formato_inesperado_vira_llmerror(self):
        session = FakeSession(FakeResponse(200, {"nao": "e uma lista"}))
        with self.assertRaises(LLMError):
            list_models(session=session)


class TestIsLocalBaseUrl(unittest.TestCase):
    def test_localhost_e_local(self):
        self.assertTrue(is_local_base_url("http://localhost:11434/v1"))

    def test_127_0_0_1_e_local(self):
        self.assertTrue(is_local_base_url("http://127.0.0.1:1234/v1"))

    def test_ipv6_loopback_e_local(self):
        self.assertTrue(is_local_base_url("http://[::1]:11434/v1"))

    def test_porta_diferente_da_padrao_continua_local(self):
        self.assertTrue(is_local_base_url("http://localhost:8080/v1"))

    def test_openrouter_nao_e_local(self):
        self.assertFalse(is_local_base_url(DEFAULT_BASE_URL))

    def test_ip_de_rede_local_nao_e_loopback(self):
        # Outra máquina na rede local não é "esta máquina" -- só loopback
        # conta, porque é o que garante que o modelo por trás é pequeno o
        # bastante para precisar do bloco de tradução menor.
        self.assertFalse(is_local_base_url("http://192.168.1.5:11434/v1"))

    def test_vazio_nao_e_local(self):
        self.assertFalse(is_local_base_url(""))


if __name__ == "__main__":
    unittest.main()
