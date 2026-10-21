"""
Конфигурация логирования.
"""

import logging
import sys


def setup_logging(level: str = "INFO"):
    """
    Настраивает систему логирования.
    
    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
    """
    # Получаем уровень
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Формат логов
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Настраиваем root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Отключаем избыточное логирование от внешних библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    logging.info(f"Логирование настроено: уровень={level}")



class SecretFilter(logging.Filter):
    def __init__(self, secrets):
        super().__init__()
        from urllib.parse import quote
        self.secrets = sorted({v for s in secrets if s for v in (s, quote(s, safe=""))}, key=len, reverse=True)

    def filter(self, record):
        text = record.getMessage()
        for value in self.secrets:
            text = text.replace(value, "[REDACTED]")
        record.msg, record.args = text, ()
        record.exc_info = record.exc_text = record.stack_info = None
        return True


def protect_secrets(secrets):
    guard = SecretFilter(secrets)
    for handler in logging.getLogger().handlers:
        handler.addFilter(guard)
