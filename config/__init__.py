"""Configuration exports; importing standalone RAG settings does not load bot .env."""
__all__ = ["Settings"]


def __getattr__(name):
    if name == "Settings":
        from .settings import Settings
        return Settings
    raise AttributeError(name)
