"""
Клиент для работы с GigaChat API.
Инкапсулирует логику общения с GigaChat.
"""

import requests
import logging
import time
import uuid
from typing import List, Dict, Optional, Union
import urllib3

from .config import GigaChatConfig

logger = logging.getLogger(__name__)

# Отключаем предупреждения о небезопасных запросах (для самоподписанных сертификатов)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GigaChatClient:
    """Клиент для GigaChat API."""
    
    OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    API_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
    
    def __init__(
        self,
        authorization_key: str = None,
        model: str = "GigaChat",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        scope: str = "GIGACHAT_API_PERS",
        config: GigaChatConfig = None
    ):
        """
        Инициализирует GigaChat клиент.
        
        Args:
            authorization_key: Authorization key в формате Basic <key>
            model: Модель для использования (GigaChat, GigaChat-Pro, GigaChat-Plus)
            temperature: Температура генерации (0.0 - 2.0)
            max_tokens: Максимальное количество токенов
            scope: Область доступа API
            config: Объект конфигурации (опционально, переопределяет другие параметры)
        """
        # Если передана конфигурация, используем её
        if config:
            self.authorization_key = config.authorization_key
            self.model = config.model
            self.temperature = config.temperature
            self.max_tokens = config.max_tokens
            self.scope = config.scope
            self.oauth_url = config.oauth_url
            self.api_base_url = config.api_base_url
            self.verify_ssl = config.verify_ssl
            self.timeout = config.timeout
        else:
            # Иначе используем переданные параметры
            if not authorization_key:
                raise ValueError("authorization_key или config должны быть указаны")
            self.authorization_key = authorization_key
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.scope = scope
            self.oauth_url = self.OAUTH_URL
            self.api_base_url = self.API_BASE_URL
            self.verify_ssl = True
            self.timeout = 30
        
        # Access token будет получен при первом запросе
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0
        
        logger.info(f"GigaChat клиент инициализирован: модель={self.model}")
    
    def _get_access_token(self) -> str:
        """
        Получает access token для авторизации запросов.
        Кеширует токен и обновляет его при необходимости.
        
        Returns:
            Access token
        """
        # Проверяем, есть ли валидный токен
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token
        
        # Получаем новый токен
        try:
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json',
                'RqUID': str(uuid.uuid4()),
                'Authorization': f'Basic {self.authorization_key}'
            }
            
            payload = {
                'scope': self.scope
            }
            
            response = requests.post(
                self.oauth_url,
                headers=headers,
                data=payload,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            token_data = response.json()
            self._access_token = token_data['access_token']
            
            # Устанавливаем время истечения токена (обычно 30 минут)
            # Вычитаем 5 минут для запаса
            expires_in = token_data.get('expires_at', 1800) - 300
            self._token_expires_at = time.time() + expires_in
            
            logger.info("Access token успешно получен")
            return self._access_token
            
        except Exception as e:
            logger.error(f"Ошибка получения access token: {e}")
            raise
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Генерирует ответ с помощью GigaChat.
        
        Args:
            messages: Список сообщений для GigaChat (формат: [{"role": "user/assistant/system", "content": "text"}])
            temperature: Температура (опционально)
            max_tokens: Макс токены (опционально)
            
        Returns:
            Сгенерированный ответ
        """
        try:
            access_token = self._get_access_token()
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {access_token}'
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            }
            
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=payload,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content']
            
            logger.info(f"Ответ сгенерирован: {len(answer)} символов")
            return answer
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            raise
    
    def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Генерирует ответ с потоковой передачей.
        
        Args:
            messages: Список сообщений для GigaChat
            temperature: Температура (опционально)
            max_tokens: Макс токены (опционально)
            
        Yields:
            Части ответа
        """
        try:
            access_token = self._get_access_token()
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {access_token}'
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                "stream": True
            }
            
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=payload,
                verify=self.verify_ssl,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data_text = line_text[6:]  # Убираем 'data: '
                        if data_text == '[DONE]':
                            break
                        
                        try:
                            import json
                            chunk_data = json.loads(data_text)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            logger.error(f"Ошибка streaming генерации: {e}")
            raise
    
    def get_models(self) -> List[str]:
        """
        Получает список доступных моделей.
        
        Returns:
            Список названий моделей
        """
        try:
            access_token = self._get_access_token()
            
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {access_token}'
            }
            
            response = requests.get(
                f"{self.api_base_url}/models",
                headers=headers,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            models_data = response.json()
            models = [model['id'] for model in models_data.get('data', [])]
            
            logger.info(f"Получены модели: {models}")
            return models
            
        except Exception as e:
            logger.error(f"Ошибка получения списка моделей: {e}")
            raise

