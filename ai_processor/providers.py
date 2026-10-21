"""Minimal provider boundary; no fallback and no credentials in diagnostics."""
from typing import Protocol
from .openai_client import OpenAIClient


class LLMProvider(Protocol):
    def generate_response(self, messages: list[dict[str, str]]) -> str: ...


class ProviderError(RuntimeError):
    pass


class GigaChatProvider:
    def __init__(self, settings, sdk_factory=None):
        settings.validate_provider()
        if sdk_factory is None:
            try:
                from gigachat import GigaChat
            except ImportError:
                raise ProviderError("Не установлен официальный SDK gigachat; см. requirements-gigachat.txt") from None
            sdk_factory = GigaChat
        self.model = settings.gigachat_model
        try:
            self.client = sdk_factory(
                credentials=settings.gigachat_credentials, scope=settings.gigachat_scope,
                model=self.model, verify_ssl_certs=True,
                ca_bundle_file=settings.gigachat_ca_bundle_file or None,
                base_url="https://gigachat.devices.sberbank.ru/api/v1",
                auth_url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                timeout=60.0, max_retries=0)
        except Exception:
            raise ProviderError("GigaChat: ошибка настройки SDK или сертификатов TLS.") from None

    def generate_response(self, messages):
        try:
            # Official SDK's supported root chat compatibility API.
            response = self.client.chat({"model": self.model, "messages": messages,
                                         "temperature": 0.0, "max_tokens": 3000})
            if response.choices[0].finish_reason != "stop":
                raise ValueError("Incomplete response")
            text = response.choices[0].message.content
            if not isinstance(text, str) or not text:
                raise ValueError("Empty response")
            return text
        except Exception:
            raise ProviderError("GigaChat: не удалось получить ответ. Проверьте авторизацию, доступ и сертификаты TLS.") from None

    def close(self):
        self.client.close()


def create_provider(settings, *, openai_sdk=None, gigachat_factory=None):
    settings.validate_provider()
    if settings.llm_provider == "openai":
        try:
            return OpenAIClient(settings.openai_api_key, settings.openai_model,
                                temperature=0.0, max_tokens=3000, client=openai_sdk)
        except Exception:
            raise ProviderError("OpenAI: ошибка настройки клиента.") from None
    if settings.llm_provider == "gigachat":
        return GigaChatProvider(settings, gigachat_factory)
    raise ProviderError("Неизвестный LLM_PROVIDER; автоматическое переключение запрещено")
