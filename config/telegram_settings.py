"""Telegram access is closed unless user IDs are explicitly allowed."""
from dataclasses import dataclass, field
import os
from dotenv import dotenv_values


@dataclass(frozen=True)
class TelegramSettings:
    token: str = field(default="", repr=False)
    allowed_user_ids: frozenset = frozenset()
    allowed_chat_ids: frozenset = frozenset()
    history_messages: int = 20
    session_timeout: int = 3600
    max_sessions: int = 256

    @classmethod
    def from_env(cls, env=None, env_file=None):
        values = dict(dotenv_values(env_file, interpolate=False)) if env_file else {}
        values.update(os.environ if env is None else env)
        def ids(name, positive):
            try:
                result = frozenset(int(x.strip()) for x in (values.get(name) or "").split(",") if x.strip())
                if any(x <= 0 if positive else x == 0 for x in result):
                    raise ValueError()
                return result
            except ValueError:
                raise ValueError("Неверный список разрешённых Telegram ID") from None
        try:
            history = int(values.get("DIALOG_HISTORY_MESSAGES", "20"))
            timeout = int(values.get("DIALOG_SESSION_TIMEOUT", "3600"))
            sessions = int(values.get("DIALOG_MAX_SESSIONS", "256"))
            if not (2 <= history <= 100 and 60 <= timeout <= 86400 and 1 <= sessions <= 4096):
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError("Неверные ограничения памяти диалога") from None
        return cls(values.get("TELEGRAM_BOT_TOKEN") or "", ids("TELEGRAM_ALLOWED_USER_IDS", True),
                   ids("TELEGRAM_ALLOWED_CHAT_IDS", False), history, timeout, sessions)

    def validate_start(self):
        if not self.token:
            raise ValueError("Не задан TELEGRAM_BOT_TOKEN")
        if not self.allowed_user_ids:
            raise ValueError("TELEGRAM_ALLOWED_USER_IDS пуст: доступ закрыт")
