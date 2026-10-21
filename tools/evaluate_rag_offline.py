"""Reproduce stage 4 mock-generation examples; no network or production DB writes."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
from config.rag_settings import RAGSettings
from rag.runtime import build_pipeline
from tools.ask_atlas import DemoProvider

ROOT = Path(__file__).resolve().parents[1]


def evaluate():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        copy = Path(tmp)
        (copy / "data").mkdir()
        for name in ("atlas.sqlite3", "atlas_search.sqlite3", "sample_selection.json", "evidence_policy.json",
                     "search_relations.json", "retrieval_cases_10.json", "query_vectors_c80364e84e45bacb.json"):
            shutil.copy2(ROOT / "data" / name, copy / "data" / name)
        shutil.copytree(ROOT / "data/chroma_openai", copy / "data/chroma_openai")
        pipeline = build_pipeline(copy, RAGSettings(), provider=DemoProvider())
        cases = json.loads((copy / "data/retrieval_cases_10.json").read_text(encoding="utf-8"))
        evaluated = []
        for c in cases:
            result = pipeline.answer(c["query"], mode="hybrid")
            evaluated.append({**c, "result": result})
        examples = []
        for kind, query in (("forward", "Большие глаза"),
                            ("reverse", "Физическая и психическая выносливость"),
                            ("motivation", "Как мотивировать человека с характеристикой «мотивация через образ»?"),
                            ("interaction", "Как взаимодействовать с человеком с характеристикой «ценит ясность аргументов»?"),
                            ("partial", "Большие глаза; Какая группа крови у человека с большими глазами?")):
            examples.append({"kind": kind, "query": query, "result": pipeline.answer(query, mode="fts")})
        positives = [r for r in evaluated if r["expected_record_ids"]]
        negatives = [r for r in evaluated if not r["expected_record_ids"]]
        report = {"llm_mode": "deterministic_mock", "network_blocked": True, "paid_api_calls": 0,
                  "retrieval": "real saved OpenAI document/query vectors on a temporary Chroma copy + actual FTS5/RRF",
                  "positive_cases": len(positives),
                  "positive_cases_answered_with_expected_source": sum(
                      bool(set(r["expected_record_ids"]) & {s["record_id"] for s in r["result"]["sources"]}) for r in positives),
                  "positive_cases_abstained": sum(r["result"]["status"] == "no_information" for r in positives),
                  "negative_cases": len(negatives),
                  "negative_cases_correctly_abstained": sum(r["result"]["status"] == "no_information" for r in negatives),
                  "examples": examples, "stage3_cases": evaluated}
    before = json.loads((ROOT / "reports/stage4_before.json").read_text(encoding="utf-8"))
    changed = [p for p,h in before.items() if sha256((ROOT / p).read_bytes()).hexdigest() != h]
    manifest = json.loads((ROOT / "reports/copy_manifest.json").read_text(encoding="utf-8"))
    source = Path(manifest["source_root"])
    original_changes = [r["path"] for r in manifest["copied_files"] if sha256((source / r["path"]).read_bytes()).hexdigest() != r["sha256"]]
    report["protected_files_checked"] = len(before)
    report["changed_protected_files"] = changed
    report["original_project_changed_files"] = original_changes
    report["original_project_files_checked"] = len(manifest["copied_files"])
    if changed or original_changes:
        raise ValueError("Protected artifacts changed")
    (ROOT / "reports/stage4_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden for offline evaluation")):
        report = evaluate()
    print(json.dumps({k:v for k,v in report.items() if k not in ("examples", "stage3_cases")}, ensure_ascii=False, indent=2))
