"""
Клиент для работы с OpenAI API.
Инкапсулирует логику общения с OpenAI.
"""

import openai
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Клиент для OpenAI API."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        *, client=None
    ):
        """
        Инициализирует OpenAI клиент.
        
        Args:
            api_key: API ключ OpenAI
            model: Модель для использования
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
        """
        self.client = client if client is not None else openai.OpenAI(
            api_key=api_key, base_url="https://api.openai.com/v1", max_retries=0, timeout=60.0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"OpenAI клиент инициализирован: модель={model}")
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Генерирует ответ с помощью GPT.
        
        Args:
            messages: Список сообщений для GPT
            temperature: Температура (опционально)
            max_tokens: Макс токены (опционально)
            
        Returns:
            Сгенерированный ответ
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Ответ сгенерирован: {len(answer)} символов")
            
            return answer
        
        except Exception as e:
            logger.error("Ошибка OpenAI; подробности скрыты")
            raise RuntimeError("OpenAI: не удалось получить ответ. Проверьте доступ и настройки API.") from None
    
    def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Генерирует ответ с потоковой передачей.
        
        Args:
            messages: Список сообщений для GPT
            temperature: Температура (опционально)
            max_tokens: Макс токены (опционально)
            
        Yields:
            Части ответа
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error("Ошибка OpenAI streaming; подробности скрыты")
            raise RuntimeError("OpenAI: не удалось получить поток ответа.") from None

