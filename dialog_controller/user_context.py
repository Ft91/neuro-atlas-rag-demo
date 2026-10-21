"""
Контекст пользователя.
Хранит информацию о текущем диалоге и состоянии пользователя.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class UserContext:
    """Контекст диалога пользователя."""
    
    def __init__(self, user_id: str, max_messages=20, max_message_chars=2000):
        """
        Инициализирует контекст пользователя.
        
        Args:
            user_id: ID пользователя
        """
        self.user_id = str(user_id)
        self.max_messages = max_messages
        self.max_message_chars = max_message_chars
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # История сообщений (для контекста ИИ)
        self.conversation_history: List[Dict[str, str]] = []
        
        # Метаданные пользователя
        self.metadata: Dict[str, Any] = {}
        
        # Состояние диалога
        self.state: str = "active"  # active, waiting, etc.
        
        # Счетчик сообщений в текущей сессии
        self.message_count: int = 0
        
        logger.debug(f"Создан UserContext для {user_id}")
    
    def add_message(self, role: str, content: str):
        """
        Добавляет сообщение в историю диалога.
        
        Args:
            role: Роль отправителя ('user' или 'assistant')
            content: Содержимое сообщения
        """
        self.conversation_history.append({
            "role": role,
            "content": content[:self.max_message_chars],
            "timestamp": datetime.now().isoformat()
        })
        
        self.conversation_history = self.conversation_history[-self.max_messages:]
        self.message_count += 1
        self.update_last_activity()
        
        logger.debug(
            f"Добавлено сообщение в контекст {self.user_id}: "
            f"role={role}, length={len(content)}"
        )
    
    def get_conversation_history(
        self,
        max_messages: int = None,
        include_timestamps: bool = False
    ) -> List[Dict[str, str]]:
        """
        Получает историю диалога.
        
        Args:
            max_messages: Максимальное количество последних сообщений
            include_timestamps: Включать ли timestamps в результат
            
        Returns:
            История сообщений
        """
        history = self.conversation_history
        
        # Ограничиваем количество сообщений
        if max_messages:
            history = history[-max_messages:]
        
        # Убираем timestamps если не нужны
        if not include_timestamps:
            history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in history
            ]
        
        return history
    
    def clear_conversation_history(self):
        """Очищает историю диалога."""
        self.conversation_history = []
        self.metadata.clear()
        self.message_count = 0
        logger.info(f"История диалога очищена для {self.user_id}")
    
    def update_last_activity(self):
        """Обновляет время последней активности."""
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_seconds: int) -> bool:
        """
        Проверяет, истекла ли сессия.
        
        Args:
            timeout_seconds: Таймаут в секундах
            
        Returns:
            True если сессия истекла
        """
        timeout_delta = timedelta(seconds=timeout_seconds)
        return datetime.now() - self.last_activity > timeout_delta
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Получает метаданные.
        
        Args:
            key: Ключ метаданных
            default: Значение по умолчанию
            
        Returns:
            Значение метаданных
        """
        return self.metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any):
        """
        Устанавливает метаданные.
        
        Args:
            key: Ключ метаданных
            value: Значение
        """
        self.metadata[key] = value
        logger.debug("Метаданные диалога обновлены")
    
    def get_session_duration(self) -> timedelta:
        """
        Получает продолжительность сессии.
        
        Returns:
            Продолжительность сессии
        """
        return datetime.now() - self.created_at
    
    def get_idle_time(self) -> timedelta:
        """
        Получает время простоя с последней активности.
        
        Returns:
            Время простоя
        """
        return datetime.now() - self.last_activity
    
    def to_dict(self) -> Dict:
        """
        Конвертирует контекст в словарь.
        
        Returns:
            Словарь с данными контекста
        """
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "conversation_length": len(self.conversation_history),
            "state": self.state,
            "metadata": self.metadata
        }

