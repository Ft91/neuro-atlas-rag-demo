"""
База данных пользователей.
Хранит персонализированную информацию о пользователях.
"""

import json
import os
from typing import Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class UserDatabase:
    """Простая файловая БД для хранения данных пользователей."""
    
    def __init__(self, storage_path: str = "./user_data.json"):
        """
        Инициализирует БД пользователей.
        
        Args:
            storage_path: Путь к файлу хранения
        """
        self.storage_path = storage_path
        self.users = self._load_users()
        logger.info(f"UserDB инициализирована: {storage_path}")
    
    def _load_users(self) -> Dict:
        """Загружает данные пользователей из файла."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки user_db: {e}")
                return {}
        return {}
    
    def _save_users(self):
        """Сохраняет данные пользователей в файл."""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.users, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения user_db: {e}")
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """
        Получает данные пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Данные пользователя или None
        """
        return self.users.get(str(user_id))
    
    def create_or_update_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Создает или обновляет пользователя.
        
        Args:
            user_id: ID пользователя
            name: Имя пользователя
            metadata: Дополнительные метаданные
        """
        user_id = str(user_id)
        
        if user_id not in self.users:
            # Создаем нового пользователя
            self.users[user_id] = {
                "id": user_id,
                "name": name,
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "message_count": 0,
                "preferences": {},
                "metadata": metadata or {}
            }
            logger.info(f"Создан пользователь {user_id}")
        else:
            # Обновляем существующего
            if name:
                self.users[user_id]["name"] = name
            self.users[user_id]["last_active"] = datetime.now().isoformat()
            if metadata:
                self.users[user_id]["metadata"].update(metadata)
        
        self._save_users()
    
    def increment_message_count(self, user_id: str):
        """
        Увеличивает счетчик сообщений пользователя.
        
        Args:
            user_id: ID пользователя
        """
        user_id = str(user_id)
        if user_id in self.users:
            self.users[user_id]["message_count"] += 1
            self.users[user_id]["last_active"] = datetime.now().isoformat()
            self._save_users()
    
    def set_preference(self, user_id: str, key: str, value: Any):
        """
        Устанавливает пользовательскую настройку.
        
        Args:
            user_id: ID пользователя
            key: Ключ настройки
            value: Значение
        """
        user_id = str(user_id)
        if user_id in self.users:
            self.users[user_id]["preferences"][key] = value
            self._save_users()
    
    def get_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Получает пользовательскую настройку.
        
        Args:
            user_id: ID пользователя
            key: Ключ настройки
            default: Значение по умолчанию
            
        Returns:
            Значение настройки или default
        """
        user_id = str(user_id)
        user = self.users.get(user_id)
        if user:
            return user.get("preferences", {}).get(key, default)
        return default
    
    def get_all_users(self) -> Dict:
        """
        Получает всех пользователей.
        
        Returns:
            Словарь всех пользователей
        """
        return self.users
    
    def get_user_count(self) -> int:
        """
        Получает количество пользователей.
        
        Returns:
            Количество пользователей
        """
        return len(self.users)

