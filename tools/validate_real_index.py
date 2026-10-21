"""Validate persisted stage 3 artifacts locally; network access is blocked."""
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from rag.indexing import encoded, index_vectors
from tools.real_embeddings import ROOT, MODEL, DIMENSIONS, load_inputs, new_vector, compare, save


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def validate(root=ROOT):
    plan = json.loads((root / "reports/stage3_preflight.json").read_text(encoding="utf-8"))
    protected = {
        "data/atlas_search.sqlite3": "search_sqlite_sha256",
        "data/atlas.sqlite3": "canonical_sha256",
        "data/atlas_neurotypology_v15.docx": "docx_sha256",
        "data/retrieval_cases_10.json": "cases_sha256",
    }
    def check_sources():
        hashes = {name: digest(root / name) for name in protected}
        assert all(hashes[name] == plan[key] for name, key in protected.items()), "Protected source changed"
        return hashes

    hashes = check_sources()
    ledger_path = root / "reports/stage3_api_ledger.json"
    ledger_hash = digest(ledger_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger["attempts"]) == 2
    assert [a["inputs"] for a in ledger["attempts"]] == [49, 10]
    assert all(a["status"] == "success" for a in ledger["attempts"])
    index, docs, records, revision, cases = load_inputs(root)
    vector = new_vector(root, revision)
    stored = vector.collection.get(include=["embeddings", "documents", "metadatas"])
    assert len(stored["ids"]) == 49 == len(set(stored["ids"]))
    vector._validate_embeddings(stored["embeddings"], 49)
    by_id = {d["document_id"]: d for d in docs}
    assert set(stored["ids"]) == set(by_id)
    for i, did in enumerate(stored["ids"]):
        d = by_id[did]
        assert stored["documents"][i] == d["search_text"]
        expected = {"record_id": d["record_id"], "kind": d["kind"], "index_revision": revision,
                    "source_document": d["source_document"], "source_revision": d["source_revision"],
                    "section_path": encoded(d["section_path"]), "source_position": encoded(d["source_position"])}
        assert stored["metadatas"][i] == expected
    assert {m["record_id"] for m in stored["metadatas"]} == set(records)

    def forbidden_embeddings(texts):
        raise AssertionError("Reindex requested embeddings for unchanged documents")
    vector.embedding_function = forbidden_embeddings
    changed = index_vectors(vector, docs, revision)
    assert changed == 0 and vector.collection.count() == 49
    query_path = root / ("data/query_vectors_" + plan["cases_sha256"][:16] + ".json")
    cached = json.loads(query_path.read_text(encoding="utf-8"))
    assert cached["model"] == MODEL and cached["dimensions"] == DIMENSIONS
    assert cached["cases_sha256"] == plan["cases_sha256"]
    vector._validate_embeddings(cached["vectors"], len(cases))
    result = compare(index, vector, cases, cached["vectors"])
    previous = json.loads((root / "reports/stage3_real_evaluation.json").read_text(encoding="utf-8"))
    assert result["summary"] == previous["summary"]
    assert result["questions"] == previous["questions"]
    assert digest(ledger_path) == ledger_hash
    assert check_sources() == hashes
    manifest = json.loads((root / "reports/copy_manifest.json").read_text(encoding="utf-8"))
    source_root = Path(manifest["source_root"])
    assert all(digest(source_root / f["path"]) == f["sha256"] for f in manifest["copied_files"]), "Original project changed"
    cost = Decimal(ledger["reported_tokens"]) * Decimal(str(plan["price_usd_per_million_tokens"])) / Decimal(1000000)
    report = {"status": "passed", "network_blocked": True, "new_api_requests": 0,
              "total_api_requests": len(ledger["attempts"]), "reported_tokens": ledger["reported_tokens"],
              "usage_based_cost_usd": str(cost), "billing_invoice_verified": False,
              "canonical_records_covered": len(records), "indexed_documents": len(stored["ids"]),
              "query_vectors": len(cached["vectors"]), "dimensions": DIMENSIONS,
              "model": MODEL, "collection": vector.collection_name, "index_revision": revision,
              "duplicate_document_ids": 0, "reindex_changed_documents": changed,
              "search_text_and_source_metadata_verified": True, "evaluation_reproduced": True,
              "api_ledger_unchanged": True, "protected_sha256": hashes,
              "original_project_manifest_files_unchanged": len(manifest["copied_files"]),
              "summary": result["summary"]}
    save(root / "reports/stage3_validation.json", report)
    return report


if __name__ == "__main__":
    with patch("socket.socket.connect", side_effect=RuntimeError("Network disabled for local validation")):
        print(json.dumps(validate(), ensure_ascii=False, indent=2))
