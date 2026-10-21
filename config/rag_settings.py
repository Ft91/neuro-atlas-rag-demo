"""Standalone atlas settings: no Telegram dependency, no implicit .env discovery."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from dotenv import dotenv_values


@dataclass(frozen=True)
class RAGSettings:
    llm_provider: str = "openai"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = field(default="", repr=False)
    gigachat_model: str = "GigaChat"
    gigachat_credentials: str = field(default="", repr=False)
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_ca_bundle_file: str = ""
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    search_top_k: int = 5
    hybrid_search: bool = True
    cache_enabled: bool = True

    @classmethod
    def from_env(cls, env=None, env_file=None):
        values = dict(dotenv_values(env_file, interpolate=False)) if env_file else {}
        values.update(os.environ if env is None else env)
        names = {name: values.get(name.upper()) or default.default
                 for name, default in cls.__dataclass_fields__.items()}
        try:
            names["search_top_k"] = int(names["search_top_k"])
            flag = str(names["hybrid_search"]).lower()
            if names["search_top_k"] < 1 or flag not in ("true", "false"):
                raise ValueError()
            names["hybrid_search"] = flag == "true"
            cache_flag = str(names["cache_enabled"]).lower()
            if cache_flag not in ("true", "false"):
                raise ValueError()
            names["cache_enabled"] = cache_flag == "true"
        except (ValueError, TypeError):
            raise ValueError("Неверные SEARCH_TOP_K/HYBRID_SEARCH/CACHE_ENABLED") from None
        settings = cls(**names)
        if settings.llm_provider not in ("openai", "gigachat"):
            raise ValueError("LLM_PROVIDER должен быть openai или gigachat")
        if (settings.embedding_provider, settings.embedding_model) != ("openai", "text-embedding-3-small"):
            raise ValueError("Текущий индекс требует openai / text-embedding-3-small; пересчёт запрещён")
        return settings

    def validate_provider(self):
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("Не задан OPENAI_API_KEY в локальном .env")
        if self.llm_provider == "gigachat":
            if not self.gigachat_credentials:
                raise ValueError("Не задан GIGACHAT_CREDENTIALS в локальном .env")
            if self.gigachat_ca_bundle_file and not Path(self.gigachat_ca_bundle_file).is_file():
                raise ValueError("Файл GIGACHAT_CA_BUNDLE_FILE недоступен")
