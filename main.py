"""Atlas Telegram entry point; network startup requires two explicit flags."""
import argparse
import logging
from pathlib import Path
from config.rag_settings import RAGSettings
from config.telegram_settings import TelegramSettings
from dialog_controller import SessionManager
from interface import TelegramBot
from rag.indexing import load_sample, SearchIndex
from rag.runtime import build_pipeline
from utils.logging_config import setup_logging, protect_secrets

ROOT = Path(__file__).resolve().parent


def initialize_components(telegram_settings, rag_settings, *, pipeline_factory=None):
    telegram_settings.validate_start()
    records, _, _ = load_sample(ROOT / "data/atlas.sqlite3", ROOT / "data/sample_selection.json")
    documents, indexed, _ = SearchIndex(ROOT / "data/atlas_search.sqlite3").read()
    if len(documents) != 49 or set(indexed) != {r["record_id"] for r in records}:
        raise ValueError("Нужен индекс 49 документов из 20 статей атласа")
    manager = SessionManager(telegram_settings.session_timeout, telegram_settings.history_messages,
                             telegram_settings.max_sessions)
    if pipeline_factory is None:
        rag_settings.validate_provider()
        if not rag_settings.openai_api_key:
            raise ValueError("Для новых вопросов нужен OPENAI_API_KEY эмбеддингов")
        pipeline_factory = lambda: build_pipeline(ROOT, rag_settings, allow_paid=True)
    return TelegramBot(telegram_settings.token, manager, pipeline_factory,
                       {r["record_id"]: r for r in records}, telegram_settings.allowed_user_ids,
                       telegram_settings.allowed_chat_ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-telegram", action="store_true")
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args(argv)
    if not (args.run_telegram and args.allow_paid):
        print("Запуск отключён. После разрешения используйте --run-telegram --allow-paid.")
        return 0
    setup_logging()
    try:
        tg = TelegramSettings.from_env(env_file=ROOT / ".env")
        rag = RAGSettings.from_env(env_file=ROOT / ".env")
        protect_secrets([tg.token, rag.openai_api_key, rag.gigachat_credentials])
        bot = initialize_components(tg, rag)
        bot.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        # Do not print raw initialization/SDK exceptions or tracebacks.
        logging.error("Запуск остановлен. Проверьте список доступа, ключи, TLS, индекс и отсутствие другого бота.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
