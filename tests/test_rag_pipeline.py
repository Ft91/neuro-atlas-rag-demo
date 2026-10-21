"""Offline source-grounding tests, including hostile mock LLM responses."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import Mock, patch

from ai_processor.openai_client import OpenAIClient
from ai_processor.providers import create_provider, ProviderError
from config.rag_settings import RAGSettings
from rag.evidence import EvidencePolicy
from rag.indexing import SearchIndex
from rag.pipeline import RAGPipeline
from rag.retrieval import HybridRetriever, NO_INFORMATION
from rag.runtime import build_pipeline
from tools.ask_atlas import DemoProvider

ROOT = Path(__file__).resolve().parents[1]
EYES = "atlas_4e62d32ded3acf57d7f17ca9"
JAW = "atlas_5093dfc5b8aac70a0d956dbc"
NEGATIVES = ["Расписание авиарейсов Владивосток", "Какая группа крови у человека с большими глазами?"]


class RecordingProvider(DemoProvider):
    def __init__(self, mutate=None):
        self.calls = 0
        self.mutate = mutate
        self.payload = None

    def generate_response(self, messages):
        self.calls += 1
        self.payload = json.loads(messages[-1]["content"])
        data = json.loads(super().generate_response(messages))
        if self.mutate:
            self.mutate(data)
        return json.dumps(data, ensure_ascii=False)


def fake_openai():
    sdk = Mock()
    def complete(**kwargs):
        content = DemoProvider().generate_response(kwargs["messages"])
        return NS(choices=[NS(message=NS(content=content), finish_reason="stop")])
    sdk.chat.completions.create.side_effect = complete
    return sdk


def fake_gigachat():
    sdk = Mock()
    def chat(payload):
        content = DemoProvider().generate_response(payload["messages"])
        return NS(choices=[NS(message=NS(content=content), finish_reason="stop")])
    sdk.chat.side_effect = chat
    return sdk


class RAGPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = patch("socket.socket.connect", side_effect=AssertionError("Network forbidden"))
        cls.guard.start()
        cls.index = SearchIndex(ROOT / "data/atlas_search.sqlite3")
        cls.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.copy = Path(cls.temp.name)
        (cls.copy / "data").mkdir()
        for name in ("atlas.sqlite3", "atlas_search.sqlite3", "sample_selection.json", "evidence_policy.json",
                     "search_relations.json", "retrieval_cases_10.json", "query_vectors_c80364e84e45bacb.json"):
            shutil.copy2(ROOT / "data" / name, cls.copy / "data" / name)
        shutil.copytree(ROOT / "data/chroma_openai", cls.copy / "data/chroma_openai")
        cls.real = build_pipeline(cls.copy, RAGSettings(), provider=RecordingProvider())

    @classmethod
    def tearDownClass(cls):
        cls.guard.stop()
        cls.temp.cleanup()

    def pipeline(self, provider=None):
        return RAGPipeline(ROOT, HybridRetriever(self.index), provider or RecordingProvider())

    def test_01_forward_original_characteristic(self):
        p = self.pipeline()
        r = p.answer("Большие глаза", mode="fts")
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["sources"][0]["record_id"], EYES)
        self.assertIn("Образное восприятие.", r["answer"])
        self.assertEqual(p.provider.calls, 1)

    def test_02_reverse_original_relation(self):
        r = self.pipeline().answer("Физическая и психическая выносливость", mode="fts")
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["sources"][0]["record_id"], JAW)
        self.assertIn("Выраженные желваки", r["answer"])
        self.assertIn("Длительная и равномерная вовлечённость", r["answer"])

    def test_03_motivation_and_interaction(self):
        for query, uid in [("Как мотивировать человека с характеристикой «мотивация через образ»?", "body/620"),
                           ("Как взаимодействовать с человеком с характеристикой «ценит ясность аргументов»?", "body/1133")]:
            with self.subTest(query=query):
                r = self.pipeline().answer(query, mode="fts")
                self.assertEqual(r["status"], "answered")
                self.assertEqual(r["parts"][0]["evidence"][0]["unit_id"], uid)

    def test_04_both_negatives_even_with_real_semantic_neighbours(self):
        for query in NEGATIVES:
            with self.subTest(query=query):
                self.real.provider = RecordingProvider()
                r = self.real.answer(query, mode="hybrid")
                self.assertTrue(r["retrieved_record_ids"])
                self.assertEqual(r["answer"], NO_INFORMATION)
                self.assertEqual(r["sources"], [])
                self.assertEqual(self.real.provider.calls, 0)

    def test_05_partial_composite_preserves_unknown_part(self):
        p = self.pipeline()
        r = p.answer("Большие глаза и какая группа крови у человека с большими глазами?", mode="fts")
        self.assertEqual(r["status"], "partial")
        self.assertEqual(len(r["parts"]), 2)
        self.assertEqual(r["parts"][1]["status"], "unrecognized_request")
        self.assertIn(NO_INFORMATION, r["answer"])
        self.assertEqual(p.provider.calls, 1)

    def test_06_source_fields_are_canonical(self):
        p = self.pipeline()
        r = p.answer("Большие глаза", mode="fts")
        for src in r["sources"]:
            canonical = p.policy.records[src["record_id"]]
            for key, value in src.items():
                self.assertEqual(value, canonical[key])
        self.assertEqual(p.provider.payload["canonical_records"][0]["source_document"], "atlas_neurotypology_v15.docx")
        self.assertTrue(all("original_text" in x for x in p.provider.payload["canonical_records"]))

    def test_07_fabricated_or_shortened_quote_rejected(self):
        for edit in (lambda c: c.update(quote=c["quote"] + " Новое утверждение."),
                     lambda c: c.update(quote=c["quote"].split(".")[0]),
                     lambda c: c.update(record_id=JAW),
                     lambda c: c.update(unit_id="body/1096")):
            with self.subTest(edit=edit):
                p = self.pipeline(RecordingProvider(lambda d: edit(d["parts"][0]["evidence"][0])))
                r = p.answer("Большие глаза", mode="fts")
                self.assertEqual(r["status"], "no_information")
                self.assertEqual(r["validation_error"], "invalid_evidence_response")
                self.assertEqual(r["sources"], [])

    def test_08_real_quote_cannot_support_different_claim(self):
        def steal(d):
            d["parts"][1]["evidence"] = d["parts"][0]["evidence"]
        r = self.pipeline(RecordingProvider(steal)).answer("Большие глаза; Какая группа крови у человека с большими глазами?", mode="fts")
        self.assertEqual(r["status"], "no_information")
        self.assertEqual(r["validation_error"], "invalid_evidence_response")

    def test_09_free_claims_and_duplicate_parts_rejected(self):
        for mutate in (lambda d: d.update(answer="Вымышленное свойство"),
                       lambda d: d["parts"].append(deepcopy(d["parts"][0]))):
            r = self.pipeline(RecordingProvider(mutate)).answer("Большие глаза", mode="fts")
            self.assertEqual(r["status"], "no_information")
            self.assertNotIn("Вымышленное", r["answer"])

    def test_10_no_inferred_practical_advice(self):
        p = self.pipeline()
        r = p.answer("Мотивация: физическая и психическая выносливость", mode="fts")
        self.assertEqual(r["answer"], NO_INFORMATION)
        self.assertEqual(p.provider.calls, 0)

    def test_11_negation_and_extra_subject_not_discarded(self):
        for q in ("Большие глаза не связаны с воображением", "Признаки: не выносливость", "Большие глаза и группа крови"):
            with self.subTest(query=q):
                self.assertEqual(self.pipeline().answer(q, mode="fts")["answer"], NO_INFORMATION)

    def test_12_all_twenty_titles_have_source_grounded_forward_path(self):
        p = self.pipeline()
        for r in p.policy.records.values():
            with self.subTest(record=r["title"]):
                answer = p.answer(r["title"], mode="fts")
                self.assertEqual(answer["status"], "answered")
                self.assertEqual(answer["sources"][0]["record_id"], r["record_id"])

    def test_13_section_filter_does_not_bypass_retrieval(self):
        p = self.pipeline()
        r = p.answer("Большие глаза", mode="fts", section="13. Желваки")
        self.assertEqual(r["answer"], NO_INFORMATION)
        self.assertEqual(p.provider.calls, 0)

    def test_14_invalid_json_and_model_abstention(self):
        provider = Mock()
        provider.generate_response.return_value = "not JSON"
        self.assertEqual(self.pipeline(provider).answer("Большие глаза", mode="fts")["validation_error"], "invalid_evidence_response")
        r = self.pipeline(RecordingProvider(lambda d: d["parts"][0].update(evidence=[]))).answer("Большие глаза", mode="fts")
        self.assertEqual(r["parts"][0]["status"], "model_abstained")

    def test_15_provider_and_search_errors_sanitized(self):
        provider = Mock()
        provider.generate_response.side_effect = RuntimeError("SECRET_SENTINEL")
        p = self.pipeline(provider)
        r = p.answer("Большие глаза", mode="fts")
        self.assertEqual(r["status"], "provider_error")
        self.assertNotIn("SECRET_SENTINEL", json.dumps(r))
        with patch.object(p.retriever, "search", side_effect=RuntimeError("SECRET_SENTINEL")):
            r = p.answer("Большие глаза")
        self.assertEqual(r["status"], "search_error")
        self.assertNotIn("SECRET_SENTINEL", json.dumps(r))

    def test_16_provider_switch_preserves_real_vectors_and_retrieval(self):
        original = self.real.retriever.vector_db.collection.get(include=["embeddings", "metadatas", "documents"])
        configurations = [RAGSettings(openai_api_key="dummy"), RAGSettings(llm_provider="gigachat", gigachat_credentials="dummy")]
        results = []
        for settings in configurations:
            sdk = fake_openai() if settings.llm_provider == "openai" else fake_gigachat()
            factory = Mock(return_value=sdk)
            provider = create_provider(settings, openai_sdk=sdk if settings.llm_provider == "openai" else None,
                                       gigachat_factory=factory)
            pipeline = build_pipeline(self.copy, settings, provider=provider)
            r = pipeline.answer("Большие глаза", mode="hybrid")
            self.assertEqual(r["status"], "answered")
            results.append(r)
            if settings.llm_provider == "gigachat":
                self.assertTrue(factory.call_args.kwargs["verify_ssl_certs"])
                self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        self.assertEqual(results[0]["answer"], results[1]["answer"])
        self.assertEqual(results[0]["retrieved_record_ids"], results[1]["retrieved_record_ids"])
        after = self.real.retriever.vector_db.collection.get(include=["embeddings", "metadatas", "documents"])
        self.assertEqual(original["ids"], after["ids"])
        self.assertEqual(original["documents"], after["documents"])
        self.assertEqual(original["metadatas"], after["metadatas"])
        self.assertTrue((original["embeddings"] == after["embeddings"]).all())

    def test_17_stored_query_vectors_work_in_all_modes(self):
        for mode in ("fts", "vector", "hybrid"):
            for query, rid in (("Большие глаза", EYES), ("Физическая и психическая выносливость", JAW)):
                with self.subTest(mode=mode, query=query):
                    self.real.provider = RecordingProvider()
                    r = self.real.answer(query, mode=mode)
                    self.assertEqual(r["status"], "answered")
                    self.assertEqual(r["sources"][0]["record_id"], rid)

    def test_18_canonical_snapshot_conflict_rejected(self):
        retriever = HybridRetriever(self.index)
        retriever.records = deepcopy(retriever.records)
        retriever.records[EYES]["original_text"] += " forged"
        with self.assertRaises(ValueError):
            RAGPipeline(ROOT, retriever, RecordingProvider())


    def test_19_reviewed_paraphrases_keep_full_source_qualifiers(self):
        cases = [("Кому помогает мысленно увидеть будущий успех?", EYES),
                 ("У кого эмоции легко прочитать по лицу?", "atlas_5913955ab51898e87d98f218"),
                 ("Какой признак связан с потребностью в точных объяснениях?", "atlas_5400907cce460d25a23af2a4"),
                 ("Кого сильнее мотивирует крупная награда сразу?", "atlas_990a457b7f6545224f97d6c8")]
        self.real.provider = RecordingProvider()
        for query, rid in cases:
            with self.subTest(query=query):
                result = self.real.answer(query, mode="hybrid")
                self.assertEqual(result["status"], "answered")
                self.assertEqual(result["sources"][0]["record_id"], rid)
        self.assertIn("может мотивировать", result["answer"])
        strict = self.real.answer("Какой признак связан с умением долго работать без перерывов?", mode="hybrid")
        self.assertEqual(strict["answer"], NO_INFORMATION)


class ProviderTests(unittest.TestCase):
    def test_01_missing_keys_and_sdk(self):
        with self.assertRaisesRegex(ValueError, "GIGACHAT_CREDENTIALS"):
            create_provider(RAGSettings(llm_provider="gigachat"))
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            create_provider(RAGSettings())
        with patch.dict("sys.modules", {"gigachat": None}):
            with self.assertRaisesRegex(ProviderError, "SDK gigachat"):
                create_provider(RAGSettings(llm_provider="gigachat", gigachat_credentials="dummy"))

    def test_02_gigachat_tls_scope_ca_and_error_no_fallback(self):
        with tempfile.NamedTemporaryFile(suffix=".pem") as ca:
            settings = RAGSettings(llm_provider="gigachat", gigachat_credentials="SECRET_SENTINEL",
                                   gigachat_scope="GIGACHAT_API_CORP", gigachat_ca_bundle_file=ca.name)
            sdk = fake_gigachat()
            factory = Mock(return_value=sdk)
            with patch.dict("os.environ", {"GIGACHAT_VERIFY_SSL_CERTS": "false"}):
                provider = create_provider(settings, gigachat_factory=factory)
            self.assertTrue(factory.call_args.kwargs["verify_ssl_certs"])
            self.assertEqual(factory.call_args.kwargs["scope"], "GIGACHAT_API_CORP")
            self.assertEqual(factory.call_args.kwargs["ca_bundle_file"], ca.name)
            sdk.chat.side_effect = RuntimeError("SECRET_SENTINEL")
            with self.assertRaises(ProviderError) as exc:
                provider.generate_response([])
            self.assertNotIn("SECRET_SENTINEL", str(exc.exception))
            self.assertEqual(sdk.chat.call_count, 1)

    def test_03_openai_reused_client_zero_temperature_and_sanitized_error(self):
        sdk = fake_openai()
        provider = create_provider(RAGSettings(openai_api_key="dummy"), openai_sdk=sdk)
        self.assertIsInstance(provider, OpenAIClient)
        provider.generate_response([{"role": "user", "content": '{"parts": []}'}], temperature=0.0)
        self.assertEqual(sdk.chat.completions.create.call_args.kwargs["temperature"], 0.0)
        sdk.chat.completions.create.side_effect = RuntimeError("SECRET_SENTINEL")
        with self.assertLogs("ai_processor.openai_client", level="ERROR") as logs:
            with self.assertRaises(RuntimeError) as exc:
                provider.generate_response([])
        self.assertNotIn("SECRET_SENTINEL", str(exc.exception) + str(logs.output))
        with patch("ai_processor.openai_client.openai.OpenAI", return_value=sdk) as ctor:
            OpenAIClient("dummy")
            self.assertEqual(ctor.call_args.kwargs["max_retries"], 0)

    def test_04_environment_configuration_and_no_secret_repr(self):
        settings = RAGSettings.from_env(env={"LLM_PROVIDER": "gigachat", "GIGACHAT_CREDENTIALS": "SECRET_SENTINEL"})
        self.assertEqual(settings.llm_provider, "gigachat")
        self.assertEqual(settings.embedding_model, "text-embedding-3-small")
        search = RAGSettings.from_env(env={"SEARCH_TOP_K": "3", "HYBRID_SEARCH": "false"})
        self.assertEqual((search.search_top_k, search.hybrid_search), (3, False))
        self.assertNotIn("SECRET_SENTINEL", repr(settings))
        for env in ({"LLM_PROVIDER": "invalid"}, {"EMBEDDING_PROVIDER": "gigachat"}, {"EMBEDDING_MODEL": "mock"}):
            with self.assertRaises(ValueError):
                RAGSettings.from_env(env=env)

    def test_05_no_default_paid_execution(self):
        with self.assertRaisesRegex(ValueError, "разрешения"):
            build_pipeline(ROOT, RAGSettings())

    def test_06_env_file_separate_keys_and_legacy_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("LLM_PROVIDER=gigachat\nGIGACHAT_CREDENTIALS=dummy-giga\nOPENAI_API_KEY=dummy-openai\n", encoding="utf-8")
            settings = RAGSettings.from_env(env={}, env_file=path)
            self.assertEqual(settings.gigachat_credentials, "dummy-giga")
            self.assertEqual(settings.openai_api_key, "dummy-openai")
            self.assertEqual(settings.llm_provider, "gigachat")
        # Compatibility export is lazy: no bot module import during RAG settings load.
        import config
        with patch("dotenv.load_dotenv") as load:
            self.assertEqual(config.Settings.__name__, "Settings")

    def test_07_missing_ca_and_truncated_gigachat_response(self):
        with self.assertRaisesRegex(ValueError, "CA_BUNDLE"):
            create_provider(RAGSettings(llm_provider="gigachat", gigachat_credentials="dummy",
                                       gigachat_ca_bundle_file="missing-test-bundle.pem"), gigachat_factory=Mock())
        sdk = Mock()
        sdk.chat.return_value = NS(choices=[NS(message=NS(content="{}"), finish_reason="length")])
        p = create_provider(RAGSettings(llm_provider="gigachat", gigachat_credentials="dummy"), gigachat_factory=Mock(return_value=sdk))
        with self.assertRaises(ProviderError):
            p.generate_response([])
