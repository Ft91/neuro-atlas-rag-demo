"""Build only the 20-record offline FTS5 index; no API clients are instantiated."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from rag.indexing import SearchIndex, build_documents, load_sample
from rag.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]


def build_sample_index(root=ROOT):
    canonical = root / "data/atlas.sqlite3"
    before = sha256(canonical.read_bytes()).hexdigest()
    records, units, manifest = load_sample(canonical, root / "data/sample_selection.json")
    relations = json.loads((root / "data/search_relations.json").read_text(encoding="utf-8"))
    if relations["source_revision"] != manifest["source_revision"]:
        raise ValueError("Relation source revision mismatch")
    documents, coverage, revision = build_documents(records, units, relations)
    index = SearchIndex(root / "data/atlas_search.sqlite3")
    index.build(documents, records, revision)
    after = sha256(canonical.read_bytes()).hexdigest()
    if before != after:
        raise ValueError("Canonical SQLite changed")
    report = {"index_version": "atlas-search-v1", "index_revision": revision,
              "source_revision": manifest["source_revision"], "canonical_records": len(records),
              "search_documents": len(documents), "document_types": dict(Counter(d["kind"] for d in documents)),
              "full_coverage": all(v["complete"] for v in coverage.values()), "coverage": coverage,
              "canonical_sha256_before": before, "canonical_sha256_after": after,
              "canonical_unchanged": before == after, "indexed_other_atlas_records": 0,
              "fts_status": "ready", "production_vector_status": "not_built",
              "api_calls": 0, "target_fragment_chars": 900,
              "oversize_general_fragments": [
                  {"document_id": d["document_id"], "characters": len(d["original_text"])}
                  for d in documents if d["kind"] == "general" and len(d["original_text"]) > 900],
              "limitations": ["Whole paragraphs are indivisible; titles/external-sign anchors repeat in search text.",
                             "Six source-reviewed relations have forward/reverse permutations; no inferred relations.",
                             "FTS5 unicode61 does not perform Russian stemming.",
                             "Real semantic relevance and rejection thresholds require OpenAI embeddings and calibration."]}
    (root / "reports/indexing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return index, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query")
    parser.add_argument("--direction", choices=["general", "forward", "reverse"], default="general")
    parser.add_argument("--section")
    args = parser.parse_args()
    index, report = build_sample_index()
    value = HybridRetriever(index).search(args.query, mode="fts", direction=args.direction, section=args.section) if args.query else {
        k: report[k] for k in ("canonical_records", "search_documents", "document_types", "full_coverage", "canonical_unchanged")}
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
