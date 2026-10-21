"""Fragment-level memory regression: real canonical SQLite/FTS and mock clients."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from dialog_controller import SessionManager
from dialog_controller.followups import dialog_key
from interface.handlers import BotHandlers
from rag.cache import SQLiteCache
from rag.evidence import EvidencePolicy
from rag.focus import evidence_refs
from rag.indexing import SearchIndex
from rag.pipeline import RAGPipeline
from rag.retrieval import HybridRetriever, NO_INFORMATION
from test_telegram_atlas import update, sent

ROOT = Path(__file__).resolve().parents[1]
EYES = "atlas_4e62d32ded3acf57d7f17ca9"
UNIT = "body/620"


class AllEvidence:
    def generate_response(self, messages):
        data = json.loads(messages[-1]["content"])
        if data.get("task") == "explain_fragment":
            card = data["canonical_evidence"]
            return json.dumps({"status":"explained", **card, "explanation":data["reviewed_variants"][0]}, ensure_ascii=False)
        return json.dumps({"parts":[{"part_id":p["part_id"], "evidence":p["allowed_evidence"]}
                                    for p in data["parts"]]}, ensure_ascii=False)


class OfflinePipeline(RAGPipeline):
    def answer(self, question):
        return super().answer(question, mode="fts")


class FragmentFollowupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.policy = EvidencePolicy(ROOT)
        self.manager = SessionManager()
        self.cache = SQLiteCache(Path(self.temp.name) / "cache.sqlite3")
        self.provider = Mock(wraps=AllEvidence())
        self.results, self.pipelines = [], []
        def factory():
            p = OfflinePipeline(ROOT, HybridRetriever(SearchIndex(ROOT / "data/atlas_search.sqlite3")),
                                self.provider, cache=self.cache, cache_identity={"provider":"mock","model":"all-evidence"})
            p.retriever.search = Mock(wraps=p.retriever.search)
            original = p.answer
            def answer(q):
                result = original(q)
                self.results.append(result)
                return result
            p.answer = answer
            self.pipelines.append(p)
            return p
        self.handlers = BotHandlers(self.manager, factory, self.policy.records, {11,12})

    async def ask(self, question, **kw):
        u = update(question, **kw)
        await self.handlers.handle_message(u, NS())
        return u

    def session(self):
        return self.manager.get_session(dialog_key(11,101))

    async def test_regression_photos_then_meaning_stays_on_paragraph(self):
        first = await self.ask("Фотографии результата")
        self.assertIn("Мотивация через образ.", sent(first))
        self.assertEqual([r["unit_id"] for r in self.session().get_metadata("last_evidence")], [UNIT])
        second = await self.ask("А что это означает?")
        result = self.results[-1]
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["cache"], "bypass")
        self.assertEqual(result["generation_calls"], 1)
        self.pipelines[-1].retriever.search.assert_not_called()
        self.assertEqual(self.provider.generate_response.call_count, 2)
        card = result["parts"][0]["evidence"][0]
        self.assertEqual(card, self.policy.card(EYES, UNIT))
        self.assertEqual(len(result["parts"][0]["evidence"]), 1)
        self.assertIn("Идея в том, что наглядное представление", sent(second))
        self.assertIn("фотография результата или мысленно представленный успех", sent(second))
        self.assertNotIn(card["quote"], sent(second))
        self.assertEqual([s["record_id"] for s in result["sources"]], [EYES])
        # The model fixture returns ALL allowed cards, so the former article-wide
        # rewrite would expose every other paragraph and fail this assertion.
        others = [c for c in self.policy.requests("Большие глаза")[0]["cards"] if c["unit_id"] != UNIT]
        self.assertGreater(len(others), 1)
        for c in others:
            self.assertNotIn(c["quote"], sent(second))
        third = await self.ask("Объясни подробнее")
        self.assertEqual(sent(third), sent(second))
        (ROOT / "reports/fragment_followup_example.json").write_text(json.dumps(
            {"mode":"offline mock Telegram + mock LLM + real canonical/FTS", "dialog":[
                {"user":"Фотографии результата","bot":sent(first)},
                {"user":"А что это означает?","bot":sent(second)}],
             "focused_result":result, "full_article_card_count":len(others)+1},ensure_ascii=False,indent=2),encoding="utf-8")

    async def test_whole_article_requires_explicit_request(self):
        await self.ask("Фотографии результата")
        await self.ask("Покажи всю статью")
        result = self.results[-1]
        self.assertEqual(result["parts"][0]["evidence"], self.policy.requests("Большие глаза")[0]["cards"])
        self.assertGreater(len(result["parts"][0]["evidence"]), 1)
        self.assertFalse(result.get("focused", False))
        await self.handlers.clear_command(update("/clear"), NS())
        await self.ask('Покажи всю статью «Большие глаза»')
        self.assertEqual(self.results[-1]["status"], "answered")

    async def test_multiple_fragments_require_choice_and_support_ordinal(self):
        await self.ask("Большие глаза")
        refs = self.session().get_metadata("last_evidence")
        before = len(self.results)
        ambiguous = await self.ask("А что это означает?")
        self.assertIn("Уточните, какой фрагмент", sent(ambiguous))
        self.assertEqual(len(self.results), before)
        await self.ask("Объясни второй фрагмент")
        self.assertEqual(self.results[-1]["parts"][0]["evidence"],
                         [self.policy.card(refs[1]["record_id"], refs[1]["unit_id"])])

    async def test_cache_hit_rehydrates_focus_and_followup_never_reads_cache(self):
        await self.ask("Фотографии результата")
        await self.ask("Фотографии результата")
        self.assertEqual(self.results[-1]["cache"], "hit")
        with patch.object(self.cache, "get", side_effect=AssertionError("Focus must bypass cache")), \
             patch.object(self.cache, "put", side_effect=AssertionError("Focus must bypass cache")):
            await self.ask("А что это означает?")
        self.assertEqual(self.results[-1]["parts"][0]["evidence"][0]["unit_id"], UNIT)
        self.assertEqual(self.results[-1]["cache"], "bypass")

    async def test_clear_isolation_and_no_fallback_for_legacy_memory(self):
        await self.ask("Фотографии результата")
        for kw in ({"user":12}, {"chat":102}):
            u = await self.ask("А что это означает?", **kw)
            self.assertIn("Уточните", sent(u))
        self.session().metadata.pop("last_evidence")
        u = await self.ask("А что это означает?")
        self.assertIn("Уточните", sent(u))
        await self.ask("Фотографии результата")
        await self.handlers.clear_command(update("/clear"), NS())
        u = await self.ask("А что это означает?")
        self.assertIn("Уточните", sent(u))

    async def test_tampered_or_stale_focus_never_becomes_evidence(self):
        for field, value in (("quote_sha256", "bad"), ("source_revision", "old"), ("unit_id", "body/999999")):
            await self.ask("Фотографии результата")
            self.session().get_metadata("last_evidence")[0][field] = value
            u = await self.ask("А что это означает?")
            self.assertEqual(sent(u), NO_INFORMATION)
            self.assertEqual(self.results[-1]["validation_error"], "invalid_focus")
            self.assertEqual(self.session().get_metadata("last_evidence"), [])

    async def test_failed_delivery_and_negative_answer_clear_focus(self):
        await self.ask("Фотографии результата")
        failed = update("Большие глаза")
        failed.message.reply_text.side_effect = RuntimeError("delivery failed")
        await self.handlers.handle_message(failed, NS())
        self.assertIsNone(self.session().get_metadata("last_evidence"))
        await self.ask("Фотографии результата")
        await self.ask("Какая группа крови у человека с большими глазами?")
        self.assertEqual(self.session().get_metadata("last_evidence"), [])
        self.assertIn("Уточните", sent(await self.ask("А что это означает?")))

    async def test_prose_cannot_override_focus_and_extra_claim_not_accepted(self):
        await self.ask("Фотографии результата")
        self.session().add_message("assistant", "SECRET_FALSE_FACT")
        self.session().set_metadata("last_resolved_question", "invented")
        u = await self.ask("А что это означает?")
        self.assertNotIn("SECRET_FALSE_FACT", sent(u))
        p = self.pipelines[-1]
        result = p.answer("А это означает определённую группу крови?")
        self.assertEqual(result["status"], "no_information")
        self.assertEqual(result["validation_error"], "invalid_focus")
