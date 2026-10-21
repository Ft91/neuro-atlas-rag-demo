"""Existing Telegram handlers connected exclusively to the atlas pipeline."""
import asyncio
from contextlib import asynccontextmanager
import logging
from telegram.error import Conflict
from dialog_controller.followups import dialog_key, resolve_followup
from rag.focus import evidence_refs
from .atlas_format import render_result, split_telegram

logger = logging.getLogger(__name__)


class BotHandlers:
    def __init__(self, session_manager, pipeline_factory, records, allowed_user_ids=(), allowed_chat_ids=()):
        if len(records) != 20:
            raise ValueError("Telegram requires the selected 20 atlas records")
        self.session_manager, self.pipeline_factory, self.records = session_manager, pipeline_factory, records
        self.allowed_user_ids, self.allowed_chat_ids = frozenset(allowed_user_ids), frozenset(allowed_chat_ids)
        self._locks = {}
        self._workers = asyncio.Semaphore(4)

    async def _send(self, message, text):
        try:
            for chunk in split_telegram(text):
                await message.reply_text(chunk, parse_mode=None, disable_web_page_preview=True)
            return True
        except Exception:
            logger.warning("Telegram: ответ не доставлен; подробности скрыты")
            return False

    async def _authorize(self, update):
        message, user, chat = update.message, update.effective_user, update.effective_chat
        if not message or not user or not chat:
            return None
        allowed = (user.id in self.allowed_user_ids and not user.is_bot
                   and not getattr(message, "sender_chat", None)
                   and (chat.type == "private" or chat.id in self.allowed_chat_ids))
        if not allowed:
            await self._send(message, "Доступ закрыт.")
            return None
        return dialog_key(user.id, chat.id, getattr(message, "message_thread_id", None))

    @asynccontextmanager
    async def _dialog(self, key):
        entry = self._locks.setdefault(key, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                self._locks.pop(key, None)

    async def start_command(self, update, context):
        key = await self._authorize(update)
        if key is None:
            return
        await self._send(update.message, "Я отвечаю по 20 статьям атласа. Например: «Большие глаза».\n/start — начало; /help — примеры; /stats — состояние; /clear — очистить текущий диалог.")

    async def help_command(self, update, context):
        if await self._authorize(update) is None:
            return
        await self._send(update.message, "Примеры: Большие глаза; Физическая и психическая выносливость; Мотивация: мотивация через образ.\nНесколько вопросов разделяйте точкой с запятой. После ответа с [1], [2] можно спросить: «А что означает второй признак?».\nУточнение относится к последнему ответу в этом диалоге. /clear очищает его историю. Вопрос — до 4000 символов.")

    async def stats_command(self, update, context):
        key = await self._authorize(update)
        if key is None:
            return
        async with self._dialog(key):
            session = self.session_manager.get_session(key)
            count = len(session.conversation_history) if session else 0
            stats = session.get_metadata("cache_stats", {"hit": 0, "miss": 0, "bypass": 0}) if session else {"hit": 0, "miss": 0, "bypass": 0}
            await self._send(update.message, f"Атлас: 20 статей, 49 поисковых документов.\nСохранено сообщений текущего диалога: {count}. Кэш текущей сессии: hit={stats['hit']}, miss={stats['miss']}, bypass={stats['bypass']}.")

    async def clear_command(self, update, context):
        key = await self._authorize(update)
        if key is None:
            return
        async with self._dialog(key):
            self.session_manager.delete_session(key)
            await self._send(update.message, "История текущего диалога очищена.")

    def _answer(self, question, scope="standalone", history_dependent=False, focus=None):
        # Each worker owns its provider client, including GigaChat auth state.
        pipeline = self.pipeline_factory()
        pipeline.cache_scope = scope
        pipeline.history_dependent = history_dependent
        pipeline.focus_refs = focus
        try:
            return pipeline.answer(question)
        finally:
            provider = pipeline.provider
            try:
                if hasattr(provider, "close"):
                    provider.close()
                elif hasattr(provider, "client"):
                    provider.client.close()
            except Exception:
                logger.warning("Не удалось закрыть клиент провайдера")

    async def handle_message(self, update, context):
        key = await self._authorize(update)
        if key is None:
            return
        question = update.message.text or ""
        if not question.strip() or len(question) > 4000:
            await self._send(update.message, "Отправьте непустой вопрос до 4000 символов. Длинный вопрос разделите на несколько сообщений.")
            return
        async with self._dialog(key):
            session = self.session_manager.get_or_create_session(key)
            resolved, clarification, focus = resolve_followup(question, session, self.records)
            if clarification:
                await self._send(update.message, clarification)
                return
            fragments = []
            try:
                async with self._workers:
                    result = await asyncio.to_thread(self._answer, resolved, key, resolved != question or focus is not None, focus)
                stats = session.get_metadata("cache_stats", {"hit": 0, "miss": 0, "bypass": 0})
                event = result.get("cache", "bypass")
                if event in stats:
                    stats[event] += 1
                session.set_metadata("cache_stats", stats)
                text, refs = render_result(result)
                if not set(refs) <= self.records.keys():
                    raise ValueError("Unexpected source")
                fragments = evidence_refs(result)
            except Exception:
                logger.warning("Ошибка RAG; подробности скрыты")
                text, refs = "Не удалось обработать вопрос. Проверьте настройки API или повторите позже.", []
            if await self._send(update.message, text):
                session.add_message("user", question)
                session.conversation_history[-1]["resolved_question"] = resolved[:session.max_message_chars]
                session.add_message("assistant", text)
                session.conversation_history[-1]["source_ids"] = list(refs)
                session.conversation_history[-1]["source_refs"] = [dict(ref) for ref in fragments]
                session.set_metadata("last_evidence", fragments)
                session.set_metadata("last_sources", refs)
                session.set_metadata("last_resolved_question", resolved)
            else:
                # Some chunks may have arrived: avoid resolving against an older answer.
                session.metadata.pop("last_sources", None)
                session.metadata.pop("last_evidence", None)

    async def unsupported_message(self, update, context):
        if await self._authorize(update) is not None:
            await self._send(update.message, "Отправьте текстовый вопрос или используйте /help.")

    async def error_handler(self, update, context):
        # Never log raw exceptions, Update objects, HTTP URLs or credentials.
        if isinstance(context.error, Conflict):
            logger.error("Telegram: конфликт другого экземпляра. Polling остановлен.")
            context.application.stop_running()
        else:
            logger.warning("Telegram: ошибка обновления; подробности скрыты")
