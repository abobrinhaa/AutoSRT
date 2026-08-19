"""Cliente para APIs de modelo de linguagem compatíveis com OpenAI.

O padrão OpenAI virou o denominador comum: OpenRouter, DeepSeek direto,
Groq, Together e até o Ollama local expõem o mesmo endpoint
``/chat/completions``. Por isso o cliente aqui não é amarrado a nenhum
provedor — trocar de serviço é mudar ``base_url`` e ``model`` na
configuração, sem tocar em código.

O padrão é o OpenRouter, que dá acesso a vários modelos com uma chave só.
"""

import json
import time

import requests

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Trocável sem mexer no código, pela variável LLM_MODEL ou pelo config.json.
# Se o provedor aposentar este identificador, a API responde com erro de
# modelo inexistente e basta apontar outro.
DEFAULT_MODEL = "deepseek/deepseek-chat-v2.5"
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_RETRIES = 4

# Códigos que compensa repetir: limite de taxa e falhas temporárias do lado
# do servidor. Os demais (chave inválida, crédito acabado) não melhoram com
# nova tentativa.
RETRIABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """Falha ao falar com a API do modelo."""


class LLMClient:
    """Cliente mínimo de chat completions.

    Args:
        api_key: chave da API. Lida da configuração, nunca embutida no código.
        base_url: endereço base da API compatível com OpenAI.
        model: identificador do modelo, no formato do provedor.
        app_name: nome enviado no cabeçalho de atribuição do OpenRouter.
    """

    def __init__(self, api_key, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL,
                 timeout=DEFAULT_TIMEOUT, max_retries=DEFAULT_MAX_RETRIES,
                 app_name="AutoSRT", session=None):
        if not api_key:
            raise LLMError(
                "Falta a chave da API. Defina OPENROUTER_API_KEY no ambiente "
                "ou 'openrouter_api_key' no config.json.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.app_name = app_name
        self.session = session or requests.Session()

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # O OpenRouter usa estes dois para atribuição; provedores que não
            # os conhecem simplesmente ignoram.
            "HTTP-Referer": "https://github.com/abobrinhaa/AutoSRT",
            "X-Title": self.app_name,
        }

    def complete(self, system_prompt, user_prompt, *, temperature=0.2,
                 max_tokens=None) -> str:
        """Envia uma conversa de um turno e devolve o texto da resposta.

        A temperatura é baixa de propósito: tradução de legenda precisa ser
        previsível, não criativa.

        Raises:
            LLMError: falha de rede, resposta inválida, ou erro do provedor
                que não melhora com nova tentativa.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        url = f"{self.base_url}/chat/completions"
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url, headers=self._headers(), json=payload,
                    timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = f"falha de rede: {exc}"
            else:
                if response.status_code == 200:
                    return self._extract_text(response)
                detail = (response.text or "")[:300]
                last_error = f"HTTP {response.status_code}: {detail}"
                if response.status_code not in RETRIABLE_STATUS:
                    raise LLMError(f"A API recusou a requisição - {last_error}")

            if attempt < self.max_retries - 1:
                time.sleep(min(30, 2 ** attempt))

        raise LLMError(f"A API falhou após {self.max_retries} tentativas - {last_error}")

    @staticmethod
    def _extract_text(response) -> str:
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError):
            raise LLMError("A API devolveu uma resposta que não é JSON.")

        # Alguns provedores devolvem 200 com um erro no corpo.
        if isinstance(data, dict) and data.get("error"):
            raise LLMError(f"A API devolveu erro: {data['error']}")

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Resposta em formato inesperado: {str(data)[:300]}")

        if not isinstance(content, str) or not content.strip():
            raise LLMError("A API devolveu uma resposta vazia.")
        return content
