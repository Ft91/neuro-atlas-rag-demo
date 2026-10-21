"""Real FTS5 + real Chroma tested offline with explicitly controlled vectors.

These fixtures test plumbing, NOT semantic quality of any production model.
Expected IDs were checked against the source BEFORE running retrieval.
"""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag.indexing import SearchIndex, build_documents, index_vectors, load_sample
from rag.retrieval import HybridRetriever, SearchSettings, reciprocal_rank_fusion
from storage.vector_db import VectorDatabase

ROOT = Path(__file__).resolve().parents[1]
GOLD = {
    "Большие глаза": "atlas_4e62d32ded3acf57d7f17ca9",
    "Выраженные желваки": "atlas_5093dfc5b8aac70a0d956dbc",
    "Густые брови": "atlas_5913955ab51898e87d98f218",
    "Глубокий желобок": "atlas_5400907cce460d25a23af2a4",
    "Большой рот": "atlas_990a457b7f6545224f97d6c8",
    "Прямой лоб": "atlas_1388c92819572df3e289019e",
}
QUERIES = {
    "Кому помогает представить картину будущего успеха?": 0,
    "фотографии результата": 0,
    "физическая и психическая выносливость": 1,
    "Быстро и ярко показывает симпатию": 2,
    "Ценит ясность аргументов": 3,
    "большой результат сразу": 4,
    "длительно придерживаться выбранного направления": 5,
}


class FixtureEmbeddings:
    """Explicit deterministic test vectors; never used as an OpenAI fallback."""
    def __init__(self):
        self.calls = 0
        self.texts = 0

    def __call__(self, texts):
        self.calls += 1
        self.texts += len(texts)
        vectors = []
        for text in texts:
            axis = QUERIES.get(text, 7)
            is_document = text.startswith(("Раздел:", "Поисковая перестановка;"))
            if is_document:
                axis = 6
            for number, sign in enumerate(GOLD):
                if text == sign or ("Признак: " + sign + "\n") in text or (
                        text.startswith("Поисковая перестановка;") and
                        (("\n" + sign + "\n") in text or text.endswith("\n" + sign))):
                    axis = number
                    break
            vector = [0.0] * 8
            vector[axis] = 1.0
            vectors.append(vector)
        return vectors


class HybridSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonical = ROOT / "data/atlas.sqlite3"
        cls.canonical_hash = sha256(cls.canonical.read_bytes()).hexdigest()
        cls.records, cls.units, cls.manifest = load_sample(cls.canonical, ROOT / "data/sample_selection.json")
        cls.relations = json.loads((ROOT / "data/search_relations.json").read_text(encoding="utf-8"))
        cls.documents, cls.coverage, cls.revision = build_documents(cls.records, cls.units, cls.relations)
        cls.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.index = SearchIndex(Path(cls.temp.name) / "search.sqlite3")
        cls.index.build(cls.documents, cls.records, cls.revision)
        cls.embed = FixtureEmbeddings()
        cls.vector = VectorDatabase(str(Path(cls.temp.name) / "chroma"), "atlas_fixture",
                                    embedding_function=cls.embed, embedding_model="fixture-only-v1",
                                    embedding_dimensions=8)
        # Any accidental use of the paid embedding endpoint fails this suite.
        cls.api_guard = patch.object(cls.vector.openai, "embeddings", create=True)
        cls.api = cls.api_guard.start()
        cls.api.create.side_effect = AssertionError("Paid embeddings are forbidden in offline tests")
        cls.first_upserts = index_vectors(cls.vector, cls.documents, cls.revision)
        cls.retriever = HybridRetriever(cls.index, cls.vector, vector_max_distance=0.2)

    @classmethod
    def tearDownClass(cls):
        cls.api_guard.stop()
        cls.temp.cleanup()

    def test_01_exact_external_sign(self):
        response = self.retriever.search("Большие глаза", mode="fts", direction="forward")
        self.assertEqual(response["results"][0]["record_id"], GOLD["Большие глаза"])

    def test_02_semantic_paraphrase_with_controlled_embeddings(self):
        response = self.retriever.search("Кому помогает представить картину будущего успеха?", mode="vector")
        self.assertEqual([r["record_id"] for r in response["results"]], [GOLD["Большие глаза"]])

    def test_03_reverse_source_relation(self):
        response = self.retriever.search("физическая и психическая выносливость", mode="fts", direction="reverse")
        self.assertEqual(response["results"][0]["record_id"], GOLD["Выраженные желваки"])
        self.assertIn("физическая и психическая выносливость", response["results"][0]["evidence"][0]["original_text"])

    def test_04_hybrid_rrf_has_both_channels(self):
        response = self.retriever.search("фотографии результата", mode="hybrid")
        first = response["results"][0]
        self.assertEqual(first["record_id"], GOLD["Большие глаза"])
        self.assertEqual(first["channel_ranks"], {"fts": 1, "vector": 1})
        self.assertAlmostEqual(first["score"], 2 / 61)

    def test_05_canonical_deduplication_and_top_k(self):
        for mode in ("fts", "vector", "hybrid"):
            result = self.retriever.search("фотографии результата", mode=mode, top_k=2)["results"]
            ids = [r["record_id"] for r in result]
            self.assertEqual(len(set(ids)), len(ids))
            self.assertLessEqual(len(ids), 2)
        same_record = [d for d in self.documents if d["record_id"] == GOLD["Большие глаза"]]
        merged = reciprocal_rank_fusion({"fts": same_record, "vector": same_record[::-1]})
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0][1], 2 / 61)

    def test_06_sources_and_sections(self):
        response = self.retriever.search("Большие глаза", mode="hybrid", section="8. Глаза")
        self.assertTrue(response["results"])
        for result in response["results"]:
            self.assertEqual(result["record_id"], GOLD["Большие глаза"])
            self.assertIn("8. Глаза", result["source"]["section_path"])
            self.assertEqual(result["source"]["source_revision"], self.manifest["source_revision"])
            self.assertEqual(result["source"]["source_document"], "atlas_neurotypology_v15.docx")
            self.assertEqual(result["source"]["source_position"]["start"], 613)
            self.assertEqual(result["source"]["record_id"], result["record_id"])
        self.assertEqual(self.retriever.search("фотографии", mode="fts", section="11. Нос")["results"], [])

    def test_07_reindex_no_duplicates_or_extra_embeddings(self):
        before_calls = self.embed.calls
        self.index.build(self.documents, self.records, self.revision)
        self.assertEqual(index_vectors(self.vector, self.documents, self.revision), 0)
        self.assertEqual(self.embed.calls, before_calls)
        self.assertEqual(len(self.index.read()[0]), len(self.documents))
        self.assertEqual(self.vector.collection.count(), len(self.documents))

    def test_08_unknown_query(self):
        for mode in ("fts", "vector", "hybrid"):
            response = self.retriever.search("Расписание авиарейсов Владивосток", mode=mode)
            self.assertEqual(response["results"], [])
            self.assertEqual(response["message"], "В атласе не нашёл подтверждения")

    def test_09_full_twenty_record_coverage_and_verbatim_fragments(self):
        self.assertEqual(len(self.coverage), 20)
        self.assertEqual({d["record_id"] for d in self.documents}, {r["record_id"] for r in self.records})
        for record in self.records:
            parts = [d for d in self.documents if d["record_id"] == record["record_id"] and d["kind"] == "general"]
            self.assertEqual("\n".join(p["original_text"] for p in parts), record["original_text"])
            self.assertEqual([uid for p in parts for uid in p["source_position"]["unit_ids"]],
                             record["source_position"]["unit_ids"])
            for p in parts:
                self.assertIn(record["title"], p["search_text"])
        self.assertEqual(sha256(self.canonical.read_bytes()).hexdigest(), self.canonical_hash)
        self.assertEqual(set(self.vector.collection.get()["ids"]), {d["document_id"] for d in self.documents})

    def test_10_bad_relation_is_rejected(self):
        bad = deepcopy(self.relations)
        bad["relations"][0]["evidence"] += " Invented claim."
        with self.assertRaisesRegex(ValueError, "verbatim"):
            build_documents(self.records, self.units, bad)

    def test_11_vector_identity_guard(self):
        original = self.vector.embedding_model
        try:
            self.vector.embedding_model = "different-model"
            with self.assertRaises(ValueError):
                index_vectors(self.vector, self.documents, self.revision)
            with self.assertRaises(ValueError):
                self.retriever.search("Большие глаза", mode="vector")
        finally:
            self.vector.embedding_model = original

    def test_12_explicit_threshold_and_backend_errors(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            HybridRetriever(self.index, self.vector).search("Большие глаза", mode="vector")
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            HybridRetriever(self.index).search("Большие глаза", mode="hybrid")
        with self.assertRaises(ValueError):
            self.vector._validate_embeddings([[1.0]], 1)

    def test_13_index_revision_rejects_overwrite(self):
        with self.assertRaisesRegex(ValueError, "revision differs"):
            self.index.build(self.documents, self.records, "different-revision")
        self.assertEqual(self.index.read()[2], self.revision)

    def test_14_settings_and_multiple_alternatives(self):
        with patch.dict(os.environ, {"SEARCH_TOP_K": "2", "HYBRID_SEARCH": "true"}):
            self.assertEqual(SearchSettings.from_env(), SearchSettings(2, True))
        with patch.dict(os.environ, {"SEARCH_TOP_K": "0"}):
            with self.assertRaises(ValueError):
                SearchSettings.from_env()
        alternatives = self.retriever.search("правила", mode="fts")["results"]
        self.assertGreater(len(alternatives), 1)
        self.assertTrue(all(r["source"] and r["evidence"] for r in alternatives))

    def test_15_all_six_reviewed_reverse_relations(self):
        for query, axis in QUERIES.items():
            if query.startswith("Кому"):
                continue
            expected = list(GOLD.values())[axis]
            results = self.retriever.search(query, mode="fts", direction="reverse")["results"]
            self.assertTrue(results, query)
            self.assertEqual(results[0]["record_id"], expected, query)

    def test_16_save_reviewable_examples(self):
        examples = [
            self.retriever.search("Большие глаза", mode="fts", direction="forward"),
            self.retriever.search("физическая и психическая выносливость", mode="fts", direction="reverse"),
            self.retriever.search("фотографии результата", mode="hybrid"),
            self.retriever.search("Кому помогает представить картину будущего успеха?", mode="vector"),
            self.retriever.search("Расписание авиарейсов Владивосток", mode="hybrid"),
        ]
        queries = ["Большие глаза", "физическая и психическая выносливость", "фотографии результата",
                   "Кому помогает представить картину будущего успеха?", "Расписание авиарейсов Владивосток"]
        for query, example in zip(queries, examples):
            example["query"] = query
        report = {
            "vector_engine": "real local ChromaDB", "embedding_model": "fixture-only-v1",
            "embedding_dimensions": 8, "test_only_distance_threshold": 0.2,
            "real_semantic_quality_evaluated": False, "production_vector_index_created": False,
            "canonical_records": 20, "vector_documents": self.vector.collection.count(),
            "fts_documents": len(self.index.read()[0]), "index_revision": self.revision,
            "paid_embedding_calls": self.api.create.call_count,
            "examples": examples,
        }
        self.assertEqual(report["paid_embedding_calls"], 0)
        (ROOT / "reports/search_examples_mock.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
