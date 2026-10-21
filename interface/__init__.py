"""Модуль интерфейса (Telegram bot)."""

from .telegram_bot import TelegramBot
from .handlers import BotHandlers

__all__ = ["TelegramBot", "BotHandlers"]

