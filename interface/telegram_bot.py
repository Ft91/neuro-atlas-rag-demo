"""
Telegram бот.
Основной интерфейс взаимодействия с пользователями.
"""

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
import logging

from .handlers import BotHandlers

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram бот с RAG функциональностью."""
    
    def __init__(self, token, session_manager, pipeline_factory, records,
                 allowed_user_ids=(), allowed_chat_ids=()):
        self.token = token
        self.handlers = BotHandlers(session_manager, pipeline_factory, records,
                                    allowed_user_ids, allowed_chat_ids)
        # Per-dialog order is enforced by handlers; unrelated updates run concurrently.
        self.application = Application.builder().token(token).concurrent_updates(16).build()
        self._register_handlers()
        self.application.add_handler(MessageHandler(filters.ALL, self.handlers.unsupported_message))
        self.application.add_error_handler(self.handlers.error_handler)

    def _register_handlers(self):
        """Регистрирует обработчики команд и сообщений."""
        # Команды
        self.application.add_handler(
            CommandHandler("start", self.handlers.start_command)
        )
        self.application.add_handler(
            CommandHandler("help", self.handlers.help_command)
        )
        self.application.add_handler(
            CommandHandler("stats", self.handlers.stats_command)
        )
        self.application.add_handler(
            CommandHandler("clear", self.handlers.clear_command)
        )
        
        # Текстовые сообщения
        self.application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handlers.handle_message
            )
        )
        
        logger.info("Обработчики зарегистрированы")
    
    def run(self):
        """Запускает бота."""
        logger.info("Запуск Telegram бота...")
        logger.info("Бот готов к работе! Нажмите Ctrl+C для остановки.")
        
        # Запускаем бота
        from utils.telegram_lock import TelegramTokenLock
        with TelegramTokenLock(self.token):
            self.application.run_polling(allowed_updates=["message"], bootstrap_retries=0, drop_pending_updates=True)

