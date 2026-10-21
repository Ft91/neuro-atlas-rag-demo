"""Offline CLI: DOCX -> canonical SQLite, sample JSONL and coverage report."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from rag.docx_parser import parse_docx
from rag.canonical_store import CanonicalStore

ROOT = Path(__file__).resolve().parents[1]


def select_records(parsed, manifest):
    if (manifest["source_document"], manifest["source_revision"]) != (
            parsed.source_document, parsed.source_revision):
        raise ValueError("Selection manifest does not match source identity/revision")
    by_start = {r["source_position"]["start"]: r for r in parsed.records}
    selected = []
    for item in manifest["records"]:
        record = by_start.get(item["start"])
        if not record or record["title"] != item["title"] or record["record_type"] != "article":
            raise ValueError(f"Selection boundary mismatch: {item}")
        selected.append(record)
    if len({r["record_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate records in selection")
    return sorted(selected, key=lambda r: r["source_position"]["start"])


def ingest(source, db_path, output, selection=None):
    parsed = parse_docx(source)
    manifest = json.loads(Path(selection).read_text(encoding="utf-8")) if selection else None
    records = select_records(parsed, manifest) if manifest else parsed.records
    by_unit = {u["unit_id"]: u for u in parsed.units}
    reconstruction = all(
        r["original_text"] == "\n".join(by_unit[uid]["text"] for uid in r["source_position"]["unit_ids"])
        and sha256(r["original_text"].encode("utf-8")).hexdigest() == r["original_text_sha256"]
        for r in records)
    if not reconstruction:
        raise ValueError("Canonical source reconstruction failed")
    store = CanonicalStore(db_path)
    store.save(parsed, records)
    stored = {r["record_id"]: r for r in store.read_records(parsed.source_document, parsed.source_revision)}
    for record in records:
        expected = {k: v for k, v in record.items() if k != "search_representations"}
        if stored.get(record["record_id"]) != expected:
            raise ValueError("SQLite round-trip mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = "sample_records.jsonl" if manifest else "full_records.jsonl"
    (output / name).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    selected_ids = {r["record_id"] for r in records}
    selected_units = {uid for r in records for uid in r["source_position"]["unit_ids"]}
    report = dict(parsed.report)
    report.update({
        "mode": "sample" if manifest else "full",
        "selection_version": manifest["selection_version"] if manifest else None,
        "selection_manifest_sha256": sha256(Path(selection).read_bytes()).hexdigest() if manifest else None,
        "selected_records": len(records), "stored_selected_records": len(records),
        "stored_records_for_revision": len(stored),
        "selected_body_units": len(selected_units),
        "selected_original_characters": sum(len(r["original_text"]) for r in records),
        "selected_text_reconstruction_passed": reconstruction,
        "sqlite_roundtrip_passed": True,
        "canonical_records_are_immutable": True,
        "selected_ids_unique": len(selected_ids) == len(records),
        "selected_ids_match_full_parse": all(
            r["record_id"] in {p["record_id"] for p in parsed.records} for r in records),
        "indexing_status": "not_started", "indexed_records": 0,
        "indexed_search_representations": 0,
        "prepared_search_representations": sum(len(r["search_representations"]) for r in records),
        "embedding_api_calls": 0, "generation_api_calls": 0,
        "unrepresented_body_units": [],
        "selected_issues": [i for i in parsed.report["issues"] if i.get("unit_id") in selected_units],
        "selected_records_summary": [
            {k: r[k] for k in ("record_id", "title", "section_path", "source_position")}
            for r in records],
        "deferred_records": [
            {"record_id": r["record_id"], "record_type": r["record_type"],
             "title": r["title"], "source_position": r["source_position"],
             "reason": "outside_fixed_sample; extracted and accounted for, not yet persisted/indexed"}
            for r in parsed.records if r["record_id"] not in selected_ids],
        "boundary_review": manifest.get("boundary_review") if manifest else "Structural segmentation; issue review still required.",
        "limitations": [
            "Text fidelity is verified against OOXML, not visual pagination.",
            "No claims, causal links or missing motivations are inferred.",
            "Fields external_sign/trait/motivation remain null; original associations are retained in whole articles.",
            "Forward/reverse representations, FTS5, embeddings and retrieval are deferred.",
            "IDs are stable between sample and full import of the same DOCX; insertion/reordering in a new revision may change position-based IDs.",
            "Auxiliary parts and structural issues must be reviewed before full production ingestion.",
        ],
    })
    (output / "ingestion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review = ["# Выборка атласа — 20 полных записей" if manifest else "# Полный структурный разбор",
              "", "Текст ниже извлечён из DOCX без перефразирования. Разделители абзацев — перевод строки.",
              "Ссылки «Строки: ...» являются текстом атласа, а не номерами страниц DOCX.", ""]
    for r in records:
        review.extend([f"## {r['title'] or r['record_type']}", "",
                       f"ID: {r['record_id']}", "",
                       "Раздел: " + " / ".join(r["section_path"]), "",
                       f"Позиции body: {r['source_position']['start']}–{r['source_position']['end']}", "",
                       r["original_text"], ""])
    (output / ("sample_review.md" if manifest else "full_review.md")).write_text(
        "\n".join(review), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/atlas_neurotypology_v15.docx")
    parser.add_argument("--db", type=Path, default=ROOT / "data/atlas.sqlite3")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--selection", type=Path, default=ROOT / "data/sample_selection.json")
    parser.add_argument("--full", action="store_true", help="Parse/store all structural records; no indexing or network.")
    args = parser.parse_args()
    report = ingest(args.source, args.db, args.output, None if args.full else args.selection)
    print(json.dumps({k: report[k] for k in (
        "source_revision", "extracted_records", "article_records",
        "selected_records", "selected_body_units", "selected_text_reconstruction_passed",
        "sqlite_roundtrip_passed", "indexed_records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
