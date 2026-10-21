"""Real SQLite/FTS, mock LLM. No embeddings or network required."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch
from rag.cache import SQLiteCache, cache_key
from rag.pipeline import RAGPipeline
from rag.indexing import SearchIndex
from rag.retrieval import HybridRetriever
from config.rag_settings import RAGSettings
from tools.ask_atlas import DemoProvider

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "Большие глаза"


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = SQLiteCache(Path(self.temp.name) / "cache.sqlite3")
        self.provider = Mock(wraps=DemoProvider())
        self.retriever = HybridRetriever(SearchIndex(ROOT / "data/atlas_search.sqlite3"))
        self.retriever.search = Mock(wraps=self.retriever.search)
        self.pipeline = RAGPipeline(ROOT, self.retriever, self.provider, cache=self.cache,
                                    cache_identity={"provider": "openai", "model": "gpt-4o-mini"})

    def ask(self, question=QUESTION):
        return self.pipeline.answer(question, mode="fts")

    def test_first_and_repeat_skip_search_embeddings_llm(self):
        first = self.ask()
        self.assertEqual(first["cache"], "miss")
        self.assertEqual(first["status"], "answered")
        self.provider.generate_response.side_effect = AssertionError("LLM must not run")
        self.retriever.search.side_effect = AssertionError("Search/embeddings must not run")
        repeat = self.ask()
        self.assertEqual(repeat["cache"], "hit")
        self.assertEqual(repeat["generation_calls"], 0)
        for name in ("answer", "parts", "sources"):
            self.assertEqual(first[name], repeat[name])
        self.assertEqual(self.cache.stats(), {"hits": 1, "misses": 1, "entries": 1})
        self.assertEqual(self.provider.generate_response.call_count, 1)
        self.assertEqual(self.retriever.search.call_count, 1)
        (ROOT / "reports/stage6_cache_example.json").write_text(json.dumps(
            {"question": QUESTION, "first": first, "repeat": repeat, "stats": self.cache.stats()},
            ensure_ascii=False, indent=2), encoding="utf-8")

    def test_model_and_provider_invalidation(self):
        self.ask()
        for key, value in (("model", "other-model"), ("provider", "gigachat")):
            self.pipeline.cache_identity[key] = value
            self.assertEqual(self.ask()["cache"], "miss")
        self.assertEqual(self.provider.generate_response.call_count, 3)

    def test_source_and_index_invalidation(self):
        self.ask()
        # In-memory snapshots only: canonical SQLite and DOCX are untouched.
        rid = next(iter(self.pipeline.policy.records))
        self.pipeline.policy.records[rid]["source_revision"] = "new-source-version"
        self.assertEqual(self.ask()["cache"], "miss")
        self.retriever.revision = "new-index-version"
        self.assertEqual(self.ask()["cache"], "miss")

    def test_settings_policy_prompt_invalidation(self):
        self.ask()
        from rag.retrieval import SearchSettings
        self.retriever.settings = SearchSettings(6, True)
        self.assertEqual(self.ask()["cache"], "miss")
        self.pipeline.policy.policy["version"] = "new-policy"
        self.assertEqual(self.ask()["cache"], "miss")
        with patch("rag.pipeline.SYSTEM", "new system prompt"):
            self.assertEqual(self.ask()["cache"], "miss")
        before = self.pipeline._cache_identity("fts", None)
        self.assertNotEqual(before, self.pipeline._cache_identity("hybrid", None))
        self.assertNotEqual(before, self.pipeline._cache_identity("fts", "section"))

    def test_history_bypasses_even_existing_entry(self):
        self.ask()
        self.pipeline.history_dependent = True
        self.assertEqual(self.ask()["cache"], "bypass")
        self.assertEqual(self.provider.generate_response.call_count, 2)
        self.assertEqual(self.cache.stats()["hits"], 0)

    def test_dialog_scopes_never_share(self):
        for scope in ("101:0:11", "101:0:12", "102:0:11", "101:1:11"):
            self.pipeline.cache_scope = scope
            self.assertEqual(self.ask()["cache"], "miss")
            self.assertEqual(self.ask()["cache"], "hit")
            self.assertEqual(self.cache.stats(scope)["hits"], 1)
        self.assertEqual(self.cache.stats()["entries"], 4)

    def test_corrupt_quote_rejected(self):
        self.ask()
        with self.cache.connect() as db:
            key, payload = db.execute("SELECT key,payload FROM answers").fetchone()
            value = json.loads(payload)
            value["parts"][0]["evidence"][0]["quote"] = "Invented characteristic"
            db.execute("UPDATE answers SET payload=? WHERE key=?", (json.dumps(value), key))
        self.assertEqual(self.ask()["cache"], "miss")
        self.assertEqual(self.provider.generate_response.call_count, 2)

    def test_answer_rerendered_from_verified_quotes(self):
        first = self.ask()
        with self.cache.connect() as db:
            key, payload = db.execute("SELECT key,payload FROM answers").fetchone()
            value = json.loads(payload); value["answer"] = "Invented claim"
            db.execute("UPDATE answers SET payload=? WHERE key=?", (json.dumps(value), key))
        self.assertEqual(self.ask()["answer"], first["answer"])

    def test_errors_unknown_and_unverified_not_cached(self):
        self.provider.generate_response.side_effect = RuntimeError("secret")
        self.assertEqual(self.ask()["status"], "provider_error")
        self.provider.generate_response.side_effect = None
        self.provider.generate_response.return_value = '{"parts": [{"part_id":0,"evidence":[{"quote":"fake"}]}]}'
        self.assertEqual(self.ask()["status"], "no_information")
        self.assertEqual(self.ask("Расписание авиарейсов Владивосток")["status"], "no_information")
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_clear_and_inflight_generation(self):
        self.ask()
        _, epoch = self.cache.get("pending", "standalone", lambda v: v)
        self.cache.clear("standalone")
        self.assertFalse(self.cache.put("pending", "standalone", {"answer":"late"}, epoch))
        self.assertEqual(self.cache.stats(), {"hits":0,"misses":0,"entries":0})
        self.assertEqual(self.ask()["cache"], "miss")
        self.cache.clear()
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_parallel_connections(self):
        def run(i):
            local = SQLiteCache(self.cache.path)
            key = cache_key(str(i), {}, "scope")
            value, epoch = local.get(key, "scope", lambda v:v)
            self.assertIsNone(value)
            local.put(key, "scope", {"value":i}, epoch)
            self.assertEqual(local.get(key, "scope", lambda v:v)[0], {"value":i})
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run, range(24)))
        self.assertEqual(self.cache.stats(), {"hits":24,"misses":24,"entries":24})
        with self.cache.connect() as db:
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_normalization_preserves_compound_boundaries(self):
        self.assertEqual(cache_key(" x ", {}, "s"), cache_key("x", {}, "s"))
        self.assertNotEqual(cache_key("x\ny", {}, "s"), cache_key("x y", {}, "s"))

    def test_config_and_disabled_cache(self):
        self.assertTrue(RAGSettings.from_env({}).cache_enabled)
        self.assertFalse(RAGSettings.from_env({"CACHE_ENABLED":"false"}).cache_enabled)
        with self.assertRaises(ValueError):
            RAGSettings.from_env({"CACHE_ENABLED":"maybe"})
        self.pipeline.cache = None
        self.assertEqual(self.ask()["cache"], "bypass")
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_filesystem_source_change_blocks_stale_instance(self):
        self.ask()
        changed = dict(self.pipeline._source_snapshot)
        changed["atlas_neurotypology_v15.docx"] = "changed"
        with patch.object(self.pipeline, "_source_fingerprint", return_value=changed):
            result = self.ask()
        self.assertEqual(result["status"], "source_changed")
        self.assertEqual(self.provider.generate_response.call_count, 1)



class TelegramCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_access_memory_followup_and_private_stats(self):
        from dialog_controller import SessionManager
        from dialog_controller.followups import dialog_key
        from interface.handlers import BotHandlers
        from test_telegram_atlas import update, sent, OfflinePipeline
        from types import SimpleNamespace as NS
        with tempfile.TemporaryDirectory() as temp:
            cache = SQLiteCache(Path(temp) / "cache.sqlite3")
            provider = Mock(wraps=DemoProvider())
            pipelines = []
            def factory():
                p = OfflinePipeline(ROOT, HybridRetriever(SearchIndex(ROOT / "data/atlas_search.sqlite3")),
                    provider, cache=cache, cache_identity={"provider":"mock","model":"test"})
                pipelines.append(p)
                return p
            manager = SessionManager()
            from rag.evidence import EvidencePolicy
            handlers = BotHandlers(manager, factory, EvidencePolicy(ROOT).records, {11,12}, {-100})
            async def ask(text, **kw):
                u = update(text, **kw)
                await handlers.handle_message(u, NS())
                return u
            with patch.object(cache, "get", wraps=cache.get) as get:
                denied = await ask(QUESTION, user=99)
                self.assertIn("Доступ закрыт", sent(denied))
                get.assert_not_called()
                await ask(QUESTION)
                await ask(QUESTION)
                self.assertEqual(cache.stats(dialog_key(11,101))["hits"], 1)
                calls = get.call_count
                await ask("А какие рекомендации по мотивации?")
                self.assertEqual(get.call_count, calls)
                self.assertTrue(pipelines[-1].history_dependent)
                self.assertIn("Мотивация:", manager.get_session(dialog_key(11,101)).get_metadata("last_resolved_question"))
                await ask(QUESTION, user=12)
                await ask(QUESTION, chat=102)
                self.assertEqual(cache.stats(dialog_key(12,101))["hits"], 0)
                self.assertEqual(cache.stats(dialog_key(11,102))["hits"], 0)
                stats = update("/stats")
                await handlers.stats_command(stats, NS())
                self.assertIn("hit=1, miss=1, bypass=1", sent(stats))
                await handlers.clear_command(update("/clear"), NS())
                self.assertIsNone(manager.get_session(dialog_key(11,101)))
                before = get.call_count
                clarification = await ask("А какие рекомендации по мотивации?")
                self.assertIn("Уточните", sent(clarification))
                self.assertEqual(get.call_count, before)
