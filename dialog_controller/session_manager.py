"""
Менеджер сессий пользователей.
Управляет состоянием диалогов и контекстом пользователей.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

from .user_context import UserContext

logger = logging.getLogger(__name__)


class SessionManager:
    """Менеджер сессий пользователей."""
    
    def __init__(self, session_timeout: int = 3600, max_messages=20, max_sessions=256):
        """
        Инициализирует менеджер сессий.
        
        Args:
            session_timeout: Таймаут сессии в секундах (по умолчанию 1 час)
        """
        self.sessions: Dict[str, UserContext] = {}
        self.session_timeout = session_timeout
        if min(session_timeout, max_messages, max_sessions) < 1:
            raise ValueError("Session limits must be positive")
        self.max_messages, self.max_sessions = max_messages, max_sessions
        
        logger.info(f"SessionManager инициализирован (timeout={session_timeout}s)")
    
    def get_or_create_session(self, user_id: str) -> UserContext:
        """
        Получает существующую сессию или создает новую.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Контекст пользователя
        """
        user_id = str(user_id)
        
        # Проверяем существующую сессию
        if user_id in self.sessions:
            session = self.sessions[user_id]
            
            # Проверяем, не истекла ли сессия
            if not session.is_expired(self.session_timeout):
                session.update_last_activity()
                return session
            else:
                logger.info(f"Сессия пользователя {user_id} истекла, создаем новую")
                # Сессия истекла, создаем новую
                del self.sessions[user_id]
        
        # Создаем новую сессию
        self.cleanup_expired_sessions()
        if len(self.sessions) >= self.max_sessions:
            oldest = min(self.sessions, key=lambda key: self.sessions[key].last_activity)
            del self.sessions[oldest]
        session = UserContext(user_id, max_messages=self.max_messages)
        self.sessions[user_id] = session
        
        logger.info(f"Создана новая сессия для пользователя {user_id}")
        
        return session
    
    def get_session(self, user_id: str) -> Optional[UserContext]:
        """
        Получает существующую сессию без создания новой.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Контекст пользователя или None
        """
        user_id = str(user_id)
        session = self.sessions.get(user_id)
        if session and session.is_expired(self.session_timeout):
            del self.sessions[user_id]
            return None
        return session
    
    def delete_session(self, user_id: str):
        """
        Удаляет сессию пользователя.
        
        Args:
            user_id: ID пользователя
        """
        user_id = str(user_id)
        if user_id in self.sessions:
            del self.sessions[user_id]
            logger.info(f"Сессия пользователя {user_id} удалена")
    
    def cleanup_expired_sessions(self):
        """Удаляет истекшие сессии."""
        expired = []
        
        for user_id, session in self.sessions.items():
            if session.is_expired(self.session_timeout):
                expired.append(user_id)
        
        for user_id in expired:
            del self.sessions[user_id]
            logger.info(f"Удалена истекшая сессия: {user_id}")
        
        if expired:
            logger.info(f"Очищено {len(expired)} истекших сессий")
    
    def get_active_session_count(self) -> int:
        """
        Получает количество активных сессий.
        
        Returns:
            Количество активных сессий
        """
        # Сначала очищаем истекшие
        self.cleanup_expired_sessions()
        return len(self.sessions)
    
    def get_all_user_ids(self) -> list:
        """
        Получает список всех активных пользователей.
        
        Returns:
            Список ID пользователей
        """
        self.cleanup_expired_sessions()
        return list(self.sessions.keys())

