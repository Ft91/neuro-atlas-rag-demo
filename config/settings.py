"""
Настройки приложения.
Централизованное управление конфигурацией.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()


@dataclass
class Settings:
    """Настройки приложения."""
    
    # Telegram
    telegram_token: str
    
    # AI Provider (выбор: "openai" или "gigachat")
    ai_provider: str = "gigachat"
    
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 1000
    
    # GigaChat
    gigachat_authorization_key: Optional[str] = None
    gigachat_model: str = "GigaChat"
    gigachat_temperature: float = 0.7
    gigachat_max_tokens: int = 1000
    gigachat_verify_ssl: bool = False  # False для разработки из-за самоподписанного сертификата
    
    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "documents"
    
    # RAG параметры
    rag_n_results: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 100
    
    # Session управление
    session_timeout: int = 3600  # 1 час в секундах
    max_context_messages: int = 10
    
    @classmethod
    def from_env(cls) -> "Settings":
        """
        Создает настройки из переменных окружения.
        
        Returns:
            Экземпляр Settings
            
        Raises:
            ValueError: Если обязательные переменные не установлены
        """
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        ai_provider = os.getenv("AI_PROVIDER", "openai").lower()
        
        if not telegram_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN не установлен. "
                "Установите переменную окружения или создайте .env файл."
            )
        
        # Проверяем наличие ключа для выбранного провайдера
        if ai_provider == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY не установлен. "
                    "Установите переменную окружения или создайте .env файл."
                )
            gigachat_key = None
        elif ai_provider == "gigachat":
            gigachat_key = os.getenv("GIGACHAT_AUTHORIZATION_KEY")
            if not gigachat_key:
                raise ValueError(
                    "GIGACHAT_AUTHORIZATION_KEY не установлен. "
                    "Установите переменную окружения или создайте .env файл."
                )
            openai_api_key = os.getenv("OPENAI_API_KEY")  # Все равно нужен для embeddings
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY все еще нужен для генерации embeddings. "
                    "Установите переменную окружения или создайте .env файл."
                )
        else:
            raise ValueError(
                f"Неподдерживаемый AI_PROVIDER: {ai_provider}. "
                "Используйте 'openai' или 'gigachat'."
            )
        
        return cls(
            telegram_token=telegram_token,
            ai_provider=ai_provider,
            openai_api_key=openai_api_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", 
                "text-embedding-3-small"
            ),
            gigachat_authorization_key=gigachat_key,
            gigachat_model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
            gigachat_temperature=float(os.getenv("GIGACHAT_TEMPERATURE", "0.7")),
            gigachat_max_tokens=int(os.getenv("GIGACHAT_MAX_TOKENS", "1000")),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "documents"),
        )
    
    def validate(self) -> bool:
        """
        Проверяет валидность настроек.
        
        Returns:
            True если все настройки валидны
        """
        # Проверяем токен Telegram
        if not self.telegram_token:
            return False
        
        # Проверяем ключи для выбранного провайдера
        if self.ai_provider == "openai" and not self.openai_api_key:
            return False
        
        if self.ai_provider == "gigachat":
            if not self.gigachat_authorization_key:
                return False
            # OpenAI ключ все еще нужен для embeddings
            if not self.openai_api_key:
                return False
        
        # Проверяем числовые значения
        if self.rag_n_results <= 0 or self.chunk_size <= 0:
            return False
        
        if self.chunk_overlap >= self.chunk_size:
            return False
        
        return True

