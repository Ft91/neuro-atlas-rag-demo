"""Telegram updates, delivery, LLM and polling are mocked; actual FTS/canonical RAG."""
import asyncio
import contextlib
from datetime import datetime, timedelta
from hashlib import sha256
import io
import json
import logging
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from config.rag_settings import RAGSettings
from config.telegram_settings import TelegramSettings
from dialog_controller import SessionManager
from dialog_controller.followups import dialog_key
from interface.handlers import BotHandlers
from interface.atlas_format import render_result, split_telegram
from interface.telegram_bot import TelegramBot
from rag.evidence import EvidencePolicy
from rag.indexing import SearchIndex
from rag.pipeline import RAGPipeline
from rag.retrieval import HybridRetriever
from tools.ask_atlas import DemoProvider
from utils.logging_config import SecretFilter
from utils.telegram_lock import TelegramTokenLock

ROOT = Path(__file__).resolve().parents[1]
EYES = "atlas_4e62d32ded3acf57d7f17ca9"
JAW = "atlas_5093dfc5b8aac70a0d956dbc"


def update(text, user=11, chat=101, thread=None, kind="private"):
    message = NS(text=text, message_thread_id=thread, sender_chat=None, reply_text=AsyncMock())
    return NS(message=message, effective_user=NS(id=user, is_bot=False), effective_chat=NS(id=chat, type=kind))


def sent(u):
    return "".join(call.args[0] for call in u.message.reply_text.call_args_list)


class OfflinePipeline(RAGPipeline):
    def answer(self, question):
        return super().answer(question, mode="fts")


class TelegramIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.guard = patch("socket.socket.connect", side_effect=AssertionError("Network forbidden"))
        self.guard.start()
        self.records = EvidencePolicy(ROOT).records
        self.manager = SessionManager(max_messages=6)
        self.queries, self.payloads, self.results = [], [], []
        owner = self
        class Provider(DemoProvider):
            def generate_response(self, messages):
                owner.payloads.append(json.loads(messages[-1]["content"]))
                return super().generate_response(messages)
        def factory():
            p = OfflinePipeline(ROOT, HybridRetriever(SearchIndex(ROOT / "data/atlas_search.sqlite3")), Provider())
            original = p.answer
            def answer(q):
                owner.queries.append(q)
                result = original(q)
                owner.results.append(result)
                return result
            p.answer = answer
            return p
        self.factory = factory
        self.handlers = BotHandlers(self.manager, factory, self.records, {11, 12}, {-100})

    async def asyncTearDown(self):
        self.guard.stop()

    async def ask(self, query, **kwargs):
        u = update(query, **kwargs)
        await self.handlers.handle_message(u, NS())
        return u

    async def test_01_direct_reverse_and_motivation(self):
        for query, rid in [("Большие глаза", EYES), ("Физическая и психическая выносливость", JAW),
                           ("Мотивация: мотивация через образ", EYES)]:
            with self.subTest(query=query):
                u = await self.ask(query)
                self.assertIn(self.records[rid]["title"], sent(u))
                self.assertIn(rid.removeprefix("atlas_")[:8], sent(u))
                self.assertEqual(self.results[-1]["sources"][0]["record_id"], rid)
                self.assertNotIn("atlas_", sent(u))

    async def test_02_compound_then_second_sign_before_search(self):
        first = await self.ask("Большие глаза; Физическая и психическая выносливость")
        session = self.manager.get_session(dialog_key(11, 101))
        self.assertEqual(session.get_metadata("last_sources"), [EYES, JAW])
        self.assertIn("[1] Большие глаза", sent(first))
        self.assertIn("[2] Выраженные желваки", sent(first))
        second = await self.ask("А что означает второй признак?")
        self.assertEqual(self.queries[-1], "Выраженные желваки")
        self.assertIn("Потенциальные усилия.", sent(second))
        self.assertEqual(self.results[-1]["sources"][0]["record_id"], JAW)

    async def test_03_followup_motivation_and_ambiguity(self):
        await self.ask("Большие глаза")
        u = await self.ask("А как его мотивировать?")
        self.assertEqual(self.queries[-1], "Мотивация: Большие глаза")
        self.assertIn("фотографии результата", sent(u))
        await self.ask("Большие глаза; Физическая и психическая выносливость")
        calls = len(self.queries)
        u = await self.ask("А как его мотивировать?")
        self.assertIn("Уточните", sent(u))
        self.assertEqual(len(self.queries), calls)

    async def test_04_no_information_even_with_history(self):
        await self.ask("Большие глаза")
        session = self.manager.get_session(dialog_key(11,101))
        session.add_message("assistant", "SECRET_FALSE_FACT: группа крови определяется глазами")
        for query in ("Какая группа крови у человека с большими глазами?", "Расписание авиарейсов Владивосток"):
            u = await self.ask(query)
            self.assertEqual(sent(u), "В атласе не нашёл подтверждения")
            self.assertEqual(self.results[-1]["sources"], [])
        self.assertNotIn("SECRET_FALSE_FACT", json.dumps(self.payloads))

    async def test_05_clear_only_current_dialog(self):
        await self.ask("Большие глаза")
        await self.ask("Физическая и психическая выносливость", chat=102)
        await self.ask("Большие глаза", user=12)
        await self.handlers.clear_command(update("/clear"), NS())
        self.assertIsNone(self.manager.get_session(dialog_key(11,101)))
        self.assertIsNotNone(self.manager.get_session(dialog_key(11,102)))
        self.assertIsNotNone(self.manager.get_session(dialog_key(12,101)))
        before = len(self.queries)
        u = await self.ask("А что означает второй признак?")
        self.assertIn("Уточните", sent(u))
        self.assertEqual(len(self.queries), before)

    async def test_06_user_chat_and_topic_isolation(self):
        await self.ask("Большие глаза; Физическая и психическая выносливость", chat=-100, thread=1, kind="supergroup")
        for kwargs in ({"user":12, "chat":-100, "thread":1}, {"chat":-100,"thread":2}, {"chat":102}):
            if kwargs.get("chat") == -100:
                kwargs["kind"] = "supergroup"
            u = await self.ask("А что означает второй признак?", **kwargs)
            self.assertIn("Уточните", sent(u))
        u = await self.ask("А что означает второй признак?", chat=-100, thread=1, kind="supergroup")
        self.assertIn("Выраженные желваки", sent(u))

    async def test_07_unauthorized_all_handlers_and_group_closed(self):
        for name in ("start_command", "help_command", "stats_command", "clear_command", "handle_message", "unsupported_message"):
            u = update("Большие глаза", user=999)
            await getattr(self.handlers, name)(u, NS())
            self.assertEqual(sent(u), "Доступ закрыт.")
        u = await self.ask("Большие глаза", chat=-999, kind="supergroup")
        self.assertEqual(sent(u), "Доступ закрыт.")
        self.assertEqual(self.queries, [])
        self.assertEqual(self.manager.sessions, {})
        closed = BotHandlers(self.manager, self.factory, self.records)
        u = update("Большие глаза")
        await closed.handle_message(u, NS())
        self.assertEqual(sent(u), "Доступ закрыт.")

    async def test_08_memory_limits_and_expiry(self):
        for _ in range(5):
            await self.ask("Большие глаза")
        session = self.manager.get_session(dialog_key(11,101))
        self.assertEqual(len(session.conversation_history), 6)
        session.add_message("user", "x" * 10000)
        self.assertEqual(len(session.conversation_history[-1]["content"]), 2000)
        session.last_activity = datetime.now() - timedelta(hours=2)
        self.assertIsNone(self.manager.get_session(dialog_key(11,101)))
        manager = SessionManager(max_sessions=2)
        for n in range(4): manager.get_or_create_session(str(n))
        self.assertEqual(len(manager.sessions), 2)

    async def test_09_long_input_and_unicode_output(self):
        u = await self.ask("я" * 4001)
        self.assertIn("4000", sent(u))
        self.assertEqual(self.queries, [])
        text = ("😀 текст\n" * 1500)
        chunks = split_telegram(text)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(c.encode("utf-16-le")) // 2 <= 3500 for c in chunks))
        message = update("").message
        self.assertTrue(await self.handlers._send(message, text))
        self.assertEqual("".join(c.args[0] for c in message.reply_text.call_args_list), text)
        self.assertTrue(all(c.kwargs["parse_mode"] is None for c in message.reply_text.call_args_list))

    async def test_10_provider_error_and_delivery_failure(self):
        self.handlers.pipeline_factory = Mock(side_effect=RuntimeError("SECRET_SENTINEL"))
        with self.assertLogs("interface.handlers", level="WARNING") as logs:
            u = await self.ask("Большие глаза")
        self.assertNotIn("SECRET_SENTINEL", sent(u) + str(logs.output))
        self.assertIn("Не удалось", sent(u))
        self.handlers.pipeline_factory = self.factory
        u = update("Большие глаза", chat=999)
        u.message.reply_text.side_effect = RuntimeError("SECRET_SENTINEL")
        await self.handlers.handle_message(u, NS())
        session = self.manager.get_session(dialog_key(11,999))
        self.assertEqual(session.conversation_history, [])
        self.assertEqual(session.get_metadata("last_sources", []), [])
        await self.ask("Большие глаза", chat=999)
        self.assertTrue(session.get_metadata("last_sources"))
        await self.handlers.handle_message(u, NS())
        self.assertEqual(session.get_metadata("last_sources", []), [])

    async def test_11_slow_request_does_not_block_other_dialog(self):
        entered, release = threading.Event(), threading.Event()
        original = self.factory
        def factory():
            p = original()
            answer = p.answer
            def slow(q):
                if q == "Большие глаза":
                    entered.set()
                    if not release.wait(5): raise RuntimeError("Timed out in test")
                return answer(q)
            p.answer = slow
            return p
        self.handlers.pipeline_factory = factory
        task = asyncio.create_task(self.ask("Большие глаза"))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            other = await asyncio.wait_for(self.ask("Физическая и психическая выносливость", user=12), 3)
            self.assertIn("Выраженные желваки", sent(other))
            self.assertFalse(task.done())
            clear = asyncio.create_task(self.handlers.clear_command(update("/clear"), NS()))
            await asyncio.sleep(0)
            self.assertFalse(clear.done())
        finally:
            release.set()
        await task
        await clear
        self.assertIsNone(self.manager.get_session(dialog_key(11,101)))
        self.assertEqual(self.handlers._locks, {})

    async def test_12_commands_and_conflict_shutdown(self):
        for name in ("start_command", "help_command", "stats_command"):
            u = update("/command")
            await getattr(self.handlers, name)(u, NS())
            self.assertTrue(sent(u))
            self.assertNotIn("корпоратив", sent(u))
        from telegram.error import Conflict
        app = Mock()
        with self.assertLogs("interface.handlers", level="ERROR") as logs:
            await self.handlers.error_handler(None, NS(error=Conflict("SECRET_SENTINEL"), application=app))
        app.stop_running.assert_called_once()
        self.assertNotIn("SECRET_SENTINEL", str(logs.output))

    async def test_13_source_quotes_retained_and_full_ids_in_memory(self):
        u = await self.ask("Большие глаза; Физическая и психическая выносливость")
        r = self.results[-1]
        for p in r["parts"]:
            for c in p["evidence"]:
                self.assertIn(c["quote"], self.records[c["record_id"]]["original_text"])
                self.assertIn(c["quote"], sent(u))
        refs = self.manager.get_session(dialog_key(11,101)).get_metadata("last_sources")
        self.assertEqual(refs, [s["record_id"] for s in r["sources"]])
        session = self.manager.get_session(dialog_key(11,101))
        self.assertEqual(session.conversation_history[-1]["source_ids"], refs)
        await self.ask("Большие глаза")
        self.assertEqual(session.conversation_history[-3]["source_ids"], [EYES,JAW])

    async def test_14_write_reproducible_dialog_examples(self):
        dialogs = []
        for q in ("Большие глаза; Физическая и психическая выносливость", "А что означает второй признак?", "/clear", "А что означает второй признак?", "Большие глаза", "А как его мотивировать?", "Какая группа крови у человека с большими глазами?"):
            u = update(q)
            await (self.handlers.clear_command(u, NS()) if q == "/clear" else self.handlers.handle_message(u, NS()))
            session = self.manager.get_session(dialog_key(11,101))
            dialogs.append({"user":q,"bot":sent(u),"record_ids":session.get_metadata("last_sources", []) if session else []})
        (ROOT / "reports/stage5_dialog_examples.json").write_text(json.dumps({"mode":"offline mock Telegram + mock LLM + real FTS/canonical sources", "dialog":dialogs},ensure_ascii=False,indent=2),encoding="utf-8")


class TelegramStartupTests(unittest.TestCase):
    def test_01_access_settings_closed_and_validated(self):
        self.assertFalse(TelegramSettings.from_env(env={}).allowed_user_ids)
        settings = TelegramSettings.from_env(env={"TELEGRAM_ALLOWED_USER_IDS":"11,12", "TELEGRAM_ALLOWED_CHAT_IDS":"-100", "TELEGRAM_BOT_TOKEN":"SECRET_SENTINEL"})
        self.assertEqual(settings.allowed_user_ids, {11,12})
        self.assertNotIn("SECRET_SENTINEL", repr(settings))
        for values in ({"TELEGRAM_ALLOWED_USER_IDS":"x"},{"TELEGRAM_ALLOWED_USER_IDS":"-1"},{"DIALOG_HISTORY_MESSAGES":"0"}):
            with self.assertRaises(ValueError): TelegramSettings.from_env(env=values)
        with self.assertRaises(ValueError): TelegramSettings(token="dummy").validate_start()

    def test_02_single_instance_lock_releases_and_never_saves_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            token = "dummy-secret-token"
            with TelegramTokenLock(token,tmp):
                with self.assertRaises(RuntimeError):
                    with TelegramTokenLock(token,tmp): pass
            with TelegramTokenLock(token,tmp): pass
            for p in Path(tmp).iterdir():
                self.assertNotIn(token,p.name)
                self.assertNotIn(token,p.read_text())

    def test_03_secret_redaction_including_exception(self):
        record = logging.LogRecord("sdk",logging.ERROR,"",1,"URL bot%s and %s",("SECRET_TOKEN","SECRET_KEY"),None)
        record.exc_text = "SECRET_KEY"
        SecretFilter(["SECRET_TOKEN","SECRET_KEY"]).filter(record)
        self.assertNotIn("SECRET",record.getMessage())
        self.assertIsNone(record.exc_text)

    def test_04_application_commands_and_polling_are_mocked(self):
        builder, app = Mock(), Mock()
        builder.token.return_value = builder
        builder.concurrent_updates.return_value = builder
        builder.build.return_value = app
        with patch("interface.telegram_bot.Application.builder",return_value=builder):
            bot = TelegramBot("dummy-token",SessionManager(),Mock(),EvidencePolicy(ROOT).records,{11})
        self.assertEqual(builder.concurrent_updates.call_args.args,(16,))
        names = set()
        for call in app.add_handler.call_args_list:
            handler = call.args[0]
            names.update(getattr(handler,"commands",()))
        self.assertTrue({"start","help","stats","clear"} <= names)
        app.add_error_handler.assert_called_once()
        with patch("utils.telegram_lock.TelegramTokenLock") as lock:
            bot.run()
        lock.assert_called_once_with("dummy-token")
        app.run_polling.assert_called_once_with(allowed_updates=["message"],bootstrap_retries=0,drop_pending_updates=True)

    def test_05_main_no_flags_never_polls_or_loads_secrets(self):
        import main
        for args in ([],["--run-telegram"],["--allow-paid"]):
            with patch.object(main.TelegramSettings,"from_env") as settings, patch.object(main,"initialize_components") as init, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(args),0)
                settings.assert_not_called()
                init.assert_not_called()

    def test_06_main_keeps_provider_choice_and_atlas_factory(self):
        import main
        self.assertEqual(main.ROOT,ROOT)
        for name in ("openai","gigachat"):
            settings = RAGSettings(llm_provider=name, openai_api_key="dummy", gigachat_credentials="dummy")
            with patch.object(main,"TelegramBot") as bot, patch.object(main,"build_pipeline") as build:
                main.initialize_components(TelegramSettings(token="dummy",allowed_user_ids=frozenset({11})),settings)
                factory = bot.call_args.args[2]
                factory()
                self.assertEqual(build.call_args.args, (ROOT,settings))
                self.assertTrue(build.call_args.kwargs["allow_paid"])
                self.assertEqual(len(bot.call_args.args[3]),20)
