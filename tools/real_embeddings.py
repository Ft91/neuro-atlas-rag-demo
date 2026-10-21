"""Stage 3: default is local preparation only. Paid execution needs --execute-paid."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import argparse

from rag.indexing import SearchIndex, index_vectors
from rag.retrieval import HybridRetriever, reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[1]
MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
PRICE_PER_MILLION = 0.02
MAX_REQUESTS = 2


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temp.replace(path)


def load_inputs(root=ROOT):
    index = SearchIndex(root / "data/atlas_search.sqlite3")
    docs, records, revision = index.read()
    cases = json.loads((root / "data/retrieval_cases_10.json").read_text(encoding="utf-8"))
    if len(docs) != 49 or len(records) != 20 or len(cases) != 10:
        raise ValueError("Only the frozen 49 documents / 20 records / 10 questions are authorized")
    if len({c["query"] for c in cases}) != 10:
        raise ValueError("Duplicate evaluation queries")
    for case in cases:
        if not set(case["expected_record_ids"]) <= records.keys():
            raise ValueError("Unknown evaluation ground truth")
    return index, docs, records, revision, cases


def new_vector(root, revision):
    from storage.vector_db import VectorDatabase
    vector = VectorDatabase(
        str(root / "data/chroma_openai"), "atlas_openai_" + revision[:16],
        embedding_model=MODEL, embedding_dimensions=DIMENSIONS)
    identity = {"index_revision": revision, "embedding_model": MODEL,
                "embedding_dimensions": DIMENSIONS, "hnsw:space": "cosine"}
    collection = vector.get_or_create_collection(metadata=identity)
    if any((collection.metadata or {}).get(k) != v for k, v in identity.items()):
        raise ValueError("Collection identity mismatch: no mock/real mixing")
    return vector


def prepare(root=ROOT):
    _, docs, records, revision, cases = load_inputs(root)
    vector = new_vector(root, revision)
    existing = set(vector.collection.get()["ids"])
    if existing - {d["document_id"] for d in docs}:
        raise ValueError("Unexpected vector documents")
    pending = [d for d in docs if d["document_id"] not in existing]
    batches = {"documents": [d["search_text"] for d in pending],
               "questions": [c["query"] for c in cases]}
    # Byte count is a conservative cl100k byte-BPE token budget, NOT exact tokenization.
    sizes = {kind: {"inputs": len(texts), "characters": sum(map(len, texts)),
                    "utf8_byte_budget": sum(len(t.encode("utf-8")) for t in texts),
                    "largest_input_utf8_bytes": max((len(t.encode("utf-8")) for t in texts), default=0)}
             for kind, texts in batches.items()}
    if any(s["largest_input_utf8_bytes"] > 8192 or s["utf8_byte_budget"] > 300000 for s in sizes.values()):
        raise ValueError("Conservative input budget exceeds embedding limits")
    budget = sum(s["utf8_byte_budget"] for s in sizes.values())
    plan = {"status": "awaiting_paid_api_permission", "model": MODEL, "dimensions": DIMENSIONS,
            "index_revision": revision, "collection_name": vector.collection_name,
            "persist_directory": str(root / "data/chroma_openai"),
            "existing_vectors": len(existing), "batches": sizes,
            "max_http_requests": MAX_REQUESTS, "sdk_max_retries": 0,
            "new_embeddings_first_run": len(pending)+len(cases),
            "token_count_exact": None, "conservative_token_budget_utf8_bytes": budget,
            "price_usd_per_million_tokens": PRICE_PER_MILLION,
            "conservative_cost_usd": budget * PRICE_PER_MILLION / 1000000,
            "key_present_in_process": bool(os.getenv("OPENAI_API_KEY")),
            "questions_reused_for_all_modes": True,
            "search_sqlite_sha256": sha256((root/"data/atlas_search.sqlite3").read_bytes()).hexdigest(),
            "canonical_sha256": sha256((root/"data/atlas.sqlite3").read_bytes()).hexdigest(),
            "docx_sha256": sha256((root/"data/atlas_neurotypology_v15.docx").read_bytes()).hexdigest(),
            "cases_sha256": sha256((root/"data/retrieval_cases_10.json").read_bytes()).hexdigest(),
            "actual_paid_api_requests": 0}
    save(root / "reports/stage3_preflight.json", plan)
    return plan


def paid_batch(client, texts, ledger_path):
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"attempts": [], "reported_tokens": 0}
    if len(ledger["attempts"]) >= MAX_REQUESTS:
        raise RuntimeError("Approved request budget exhausted; do not retry without renewed permission")
    event = {"inputs": len(texts), "status": "attempt_started"}
    ledger["attempts"].append(event)
    save(ledger_path, ledger)  # Account for failed/interrupted requests as attempts.
    try:
        response = client.embeddings.create(model=MODEL, input=texts, dimensions=DIMENSIONS, encoding_format="float")
        data = sorted(response.data, key=lambda row: row.index)
        if [row.index for row in data] != list(range(len(texts))):
            raise ValueError("Embedding response count/order mismatch")
        vectors = [row.embedding for row in data]
        if any(len(v) != DIMENSIONS or not all(math.isfinite(x) for x in v) for v in vectors):
            raise ValueError("Embedding dimensions or values invalid")
        event.update(status="success", reported_tokens=response.usage.total_tokens)
        ledger["reported_tokens"] += response.usage.total_tokens
        save(ledger_path, ledger)
        return vectors
    except Exception as exc:
        event.update(status="failed", error_type=type(exc).__name__)
        save(ledger_path, ledger)
        # Never put raw SDK errors, headers or credentials in logs.
        raise RuntimeError("Embedding request failed; see sanitized ledger. No automatic retry.") from None


def compare(index, vector, cases, query_vectors):
    docs, records, revision = index.read()
    by_id = {d["document_id"]: d for d in docs}
    lexical = HybridRetriever(index)
    evaluations, distance_observations = [], []
    for case, embedding in zip(cases, query_vectors):
        response = vector.collection.query(query_embeddings=[embedding], n_results=len(docs),
                                           where={"index_revision": revision})
        semantic = [dict(by_id[i], distance=d) for i, d in zip(response["ids"][0], response["distances"][0])]
        fts = lexical.lexical(case["query"])
        modes = {}
        for name, channels in (("vector", {"vector": semantic}), ("fts", {"fts": fts}),
                               ("hybrid", {"vector": semantic, "fts": fts})):
            ranked = reciprocal_rank_fusion(channels)[:5]
            ids = [item[0] for item in ranked]
            expected = case["expected_record_ids"]
            modes[name] = {
                "hit_at_5": bool(set(ids) & set(expected)) if expected else None,
                "reciprocal_rank": next((1/(i+1) for i, rid in enumerate(ids) if rid in expected), 0) if expected else None,
                "false_positive_on_negative": bool(ids) if not expected else None,
                "top5": [{"record_id": rid, "title": records[rid]["title"], "rrf": score,
                          "source_document": records[rid]["source_document"],
                          "section_path": records[rid]["section_path"], "channel_ranks": ranks}
                         for rid, score, _, ranks in ranked]}
        expected_distances = [d["distance"] for d in semantic if d["record_id"] in case["expected_record_ids"]]
        distance_observations.append({"case_id": case["case_id"], "negative": not case["expected_record_ids"],
                                      "nearest_distance": semantic[0]["distance"],
                                      "nearest_expected_distance": min(expected_distances) if expected_distances else None})
        evaluations.append(dict(case, modes=modes))
    summary = {}
    for mode in ("vector", "fts", "hybrid"):
        positives = [r["modes"][mode] for r in evaluations if r["expected_record_ids"]]
        negatives = [r["modes"][mode] for r in evaluations if not r["expected_record_ids"]]
        summary[mode] = {"hit_at_5": sum(r["hit_at_5"] for r in positives)/len(positives),
                         "mrr_at_5": sum(r["reciprocal_rank"] for r in positives)/len(positives),
                         "negative_false_positive_rate": sum(r["false_positive_on_negative"] for r in negatives)/len(negatives)}
    return {"baseline": "unfiltered cosine candidates; no production relevance threshold applied",
            "threshold_changed": False, "summary": summary, "questions": evaluations,
            "distance_observations": distance_observations,
            "limitations": ["Ten questions are a pilot, not an independent calibration/holdout dataset.",
                           "Unfiltered vector search returns neighbours even for unsupported questions.",
                           "Threshold changes require inspecting measured positive/negative distances; none is selected automatically."]}


def execute(root=ROOT):
    # Called ONLY after explicit human approval and --execute-paid.
    index, docs, records, revision, cases = load_inputs(root)
    plan = json.loads((root/"reports/stage3_preflight.json").read_text(encoding="utf-8"))
    checks = [(root/"data/atlas_search.sqlite3", "search_sqlite_sha256"),
              (root/"data/atlas.sqlite3", "canonical_sha256"),
              (root/"data/atlas_neurotypology_v15.docx", "docx_sha256"),
              (root/"data/retrieval_cases_10.json", "cases_sha256")]
    if plan["index_revision"] != revision or any(sha256(p.read_bytes()).hexdigest()!=plan[k] for p,k in checks):
        raise ValueError("Approved input plan changed")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY missing in process environment; no request made")
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://api.openai.com/v1", max_retries=0, timeout=60.0)
    ledger_path = root/"reports/stage3_api_ledger.json"
    vector = new_vector(root, revision)
    vector.embedding_function = lambda texts: paid_batch(client, texts, ledger_path)
    created = index_vectors(vector, docs, revision)
    query_path = root/("data/query_vectors_" + plan["cases_sha256"][:16] + ".json")
    if query_path.exists():
        cached = json.loads(query_path.read_text(encoding="utf-8"))
        if cached["model"] != MODEL or cached["dimensions"] != DIMENSIONS or cached["cases_sha256"] != plan["cases_sha256"]:
            raise ValueError("Query vector identity mismatch")
        query_vectors = cached["vectors"]
    else:
        query_vectors = paid_batch(client, [c["query"] for c in cases], ledger_path)
        save(query_path, {"model": MODEL, "dimensions": DIMENSIONS,
                          "cases_sha256": plan["cases_sha256"], "vectors": query_vectors})
    vector._validate_embeddings(query_vectors, len(cases))
    result = compare(index, vector, cases, query_vectors)
    result.update(new_document_embeddings=created, indexed_documents=vector.collection.count(),
                  api_ledger=json.loads(ledger_path.read_text(encoding="utf-8")))
    result["api_requests_attempted"] = len(result["api_ledger"]["attempts"])
    result["embeddings_created_successfully"] = sum(a["inputs"] for a in result["api_ledger"]["attempts"] if a["status"] == "success")
    if any(sha256(p.read_bytes()).hexdigest()!=plan[k] for p,k in checks):
        raise ValueError("Protected input changed during evaluation")
    save(root/"reports/stage3_real_evaluation.json", result)
    client.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-paid", action="store_true",
                        help="Only after explicit human approval of stage3_preflight.json")
    args = parser.parse_args()
    result = execute() if args.execute_paid else prepare()
    print(json.dumps(result if not args.execute_paid else result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
