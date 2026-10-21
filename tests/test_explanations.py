"""Natural reviewed paraphrases, strict evidence, no real model or Telegram calls."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from rag import explanation
from rag.evidence import normalize
import test_fragment_followups as focus_tests
from test_fragment_followups import EYES, UNIT, ROOT
from test_telegram_atlas import sent, update


class ExplanationTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = focus_tests.FragmentFollowupTests.asyncSetUp
    ask = focus_tests.FragmentFollowupTests.ask
    session = focus_tests.FragmentFollowupTests.session

    async def seed(self):
        await self.ask("Фотографии результата")
        self.provider.reset_mock()

    def reply(self, **changes):
        card = self.policy.card(EYES, UNIT)
        variants = explanation.reviewed_variants(ROOT, self.policy.records[EYES], card)
        data = {"status":"explained", **card, "explanation":variants[0]}
        data.update(changes)
        self.provider.generate_response.side_effect = lambda messages: json.dumps(data, ensure_ascii=False)

    async def test_three_aliases_get_natural_explanation_from_one_provider_call(self):
        await self.seed()
        for question in ("А что это означает?", "Объясни подробнее", "Что имеется в виду?"):
            before = self.provider.generate_response.call_count
            u = await self.ask(question)
            self.assertEqual(self.provider.generate_response.call_count, before+1)
            result = self.results[-1]
            self.assertEqual(result["generation_calls"], 1)
            self.assertEqual(result["cache"], "bypass")
            self.assertIn("фотография результата или мысленно представленный успех", sent(u))
            card = self.policy.card(EYES, UNIT)
            self.assertNotIn(card["quote"], sent(u))
            payload = json.loads(self.provider.generate_response.call_args.args[0][-1]["content"])
            self.assertEqual(payload["canonical_evidence"], card)
            self.assertNotIn("canonical_records", payload)
            self.assertNotIn("conversation_history", payload)
            self.pipelines[-1].retriever.search.assert_not_called()

    async def test_original_and_quote_are_exact_and_skip_provider_and_cache(self):
        await self.seed()
        await self.ask("Что имеется в виду?")
        for q in ("Покажи оригинал", "Процитируй"):
            with patch.object(self.provider, "generate_response", side_effect=AssertionError("No LLM for quotes")), \
                 patch.object(self.cache, "get", side_effect=AssertionError("No cache for quotes")), \
                 patch.object(self.cache, "put", side_effect=AssertionError("No cache for quotes")):
                u = await self.ask(q)
            self.assertIn(self.policy.card(EYES,UNIT)["quote"], sent(u))
            self.assertEqual(self.results[-1]["generation_calls"], 0)
            self.assertNotIn("explanation", self.results[-1]["parts"][0])

    async def test_real_quote_does_not_validate_invented_causes_or_traits(self):
        await self.seed()
        for text in ("Фотографии результата гарантируют богатство через выброс дофамина.",
                     "Большие глаза означают лидерство и высокий интеллект.",
                     "Представление успеха никогда не вызывает эмоционального подъёма."):
            self.reply(explanation=text)
            u = await self.ask("А что это означает?")
            self.assertEqual(self.results[-1]["status"], "explanation_unavailable")
            self.assertIn("Не могу дать подтверждённое пояснение", sent(u))
            self.assertNotIn(text, sent(u))
            self.assertNotIn(text, json.dumps(self.session().conversation_history, ensure_ascii=False))
            self.assertEqual(self.session().get_metadata("last_evidence")[0]["unit_id"], UNIT)
        original = await self.ask("Покажи оригинал")
        self.assertIn(self.policy.card(EYES,UNIT)["quote"], sent(original))

    async def test_wrong_ids_quotes_and_extra_model_fields_are_rejected(self):
        await self.seed()
        for bad in ({"record_id":"wrong"}, {"unit_id":"body/615"}, {"quote":"fabricated"},
                    {"extra_claim":"SECRET_FALSE_FACT"}):
            self.reply(**bad)
            u = await self.ask("Что имеется в виду?")
            self.assertEqual(self.results[-1]["validation_error"], "unconfirmed_explanation")
            self.assertNotIn("SECRET_FALSE_FACT", sent(u))

    async def test_copy_of_original_is_not_presented_as_explanation(self):
        await self.seed()
        self.reply(explanation=self.policy.card(EYES,UNIT)["quote"])
        u = await self.ask("Объясни подробнее")
        self.assertIn(explanation.UNCONFIRMED, sent(u))
        self.assertEqual(self.results[-1]["status"], "explanation_unavailable")

    async def test_model_abstention_preserves_source_for_quote(self):
        await self.seed()
        self.reply(status="insufficient", explanation="")
        u = await self.ask("Объясни подробнее")
        self.assertIn(explanation.UNCONFIRMED, sent(u))
        self.assertEqual(self.results[-1]["validation_error"], "explanation_abstained")
        self.assertEqual(self.results[-1]["sources"][0]["record_id"], EYES)
        self.assertIn(self.policy.card(EYES,UNIT)["quote"], sent(await self.ask("Процитируй")))

    async def test_compound_explanation_and_quote_share_one_generation(self):
        await self.seed()
        u = await self.ask("Объясни подробнее; Покажи оригинал")
        result = self.results[-1]
        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(result["parts"]), 2)
        self.assertIn("explanation", result["parts"][0])
        self.assertNotIn("explanation", result["parts"][1])
        self.assertEqual(result["generation_calls"], 1)
        self.assertEqual(self.provider.generate_response.call_count, 1)
        self.assertIn(self.policy.card(EYES,UNIT)["quote"], sent(u))
        self.assertEqual(len(self.session().get_metadata("last_evidence")), 1)

    async def test_compound_invalid_explanation_keeps_exact_original(self):
        await self.seed()
        self.reply(explanation="Invented explanation")
        u = await self.ask("Что имеется в виду? Процитируй")
        self.assertEqual(self.results[-1]["status"], "partial")
        self.assertIn(explanation.UNCONFIRMED, sent(u))
        self.assertIn(self.policy.card(EYES,UNIT)["quote"], sent(u))
        self.assertNotIn("Invented explanation", sent(u))

    async def test_provider_error_no_retry_no_secret_and_no_cache(self):
        await self.seed()
        stats = self.cache.stats()
        self.provider.generate_response.side_effect = RuntimeError("SECRET_PROVIDER_KEY")
        u = await self.ask("Объясни подробнее")
        self.assertEqual(self.provider.generate_response.call_count, 1)
        self.assertEqual(self.results[-1]["status"], "provider_error")
        self.assertNotIn("SECRET_PROVIDER_KEY", sent(u))
        self.assertEqual(self.cache.stats(), stats)

    async def test_stale_paraphrase_policy_stops_before_provider(self):
        await self.seed()
        with patch("rag.explanation.reviewed_variants", side_effect=ValueError("Stale")):
            u = await self.ask("Объясни подробнее")
        self.provider.generate_response.assert_not_called()
        self.assertIn(explanation.UNCONFIRMED, sent(u))
        self.assertEqual(self.results[-1]["validation_error"], "invalid_explanation_policy")
        u = await self.ask("Покажи оригинал")
        self.assertIn(self.policy.card(EYES,UNIT)["quote"], sent(u))

    async def test_all_92_reviewed_paragraphs_match_current_20_records(self):
        cards = {c["unit_id"]:c for r in self.policy.routes.values() for c in r["cards"]}
        data = json.loads((ROOT/"data/explanation_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(len(cards),92)
        self.assertEqual(len(data["entries"]),92)
        self.assertEqual({r["unit_id"] for r in data["entries"]},set(cards))
        self.assertEqual(len({r["record_id"] for r in data["entries"]}),20)
        for card in cards.values():
            variants = explanation.reviewed_variants(ROOT,self.policy.records[card["record_id"]],card)
            for text in variants:
                self.assertNotEqual(normalize(text),normalize(card["quote"]))
                raw = json.dumps({"status":"explained",**card,"explanation":text},ensure_ascii=False)
                self.assertEqual(explanation.validate_response(raw,card,variants),text)

    async def test_quote_clear_and_other_users_cannot_reuse_focus(self):
        await self.seed()
        before = self.provider.generate_response.call_count
        for kw in ({"user":12},{"chat":102}):
            self.assertIn("Уточните",sent(await self.ask("Процитируй",**kw)))
        await self.handlers.clear_command(update("/clear"),NS())
        self.assertIn("Уточните",sent(await self.ask("Что имеется в виду?")))
        self.assertEqual(self.provider.generate_response.call_count,before)
