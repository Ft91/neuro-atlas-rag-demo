"""
Agent - Модульная RAG система с Telegram интерфейсом.

Архитектура:
- interface: Telegram bot и взаимодействие
- dialog_controller: Управление сессиями пользователей
- memory_manager: Формирование промптов и работа с контекстом
- ai_processor: Обработка запросов через OpenAI
- storage: Векторная БД и хранилище пользователей
"""

__version__ = "1.0.0"

