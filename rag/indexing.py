"""Offline search documents + immutable-revision FTS5 index for the fixed sample."""
from contextlib import contextmanager, closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from rag.canonical_store import encoded
from rag.docx_parser import digest

INDEX_VERSION = "atlas-search-v1"


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def load_sample(canonical_path, selection_path):
    """Read only persisted selected records, never parse/import the full DOCX."""
    manifest = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    with closing(readonly(canonical_path)) as db:
        rows = db.execute("SELECT payload FROM canonical_records WHERE source_document=? AND source_revision=?",
                          (manifest["source_document"], manifest["source_revision"])).fetchall()
        all_records = [json.loads(row[0]) for row in rows]
        starts = {r["source_position"]["start"]: r for r in all_records}
        records = []
        for expected in manifest["records"]:
            record = starts.get(expected["start"])
            if record is None or record["title"] != expected["title"]:
                raise ValueError("Canonical sample does not match selection manifest")
            expected_id = "atlas_" + digest(record["source_document"] + "\0" +
                record["source_position"]["unit_ids"][0] + "\0" + record["record_type"])[:24]
            if record["record_id"] != expected_id or digest(record["original_text"]) != record["original_text_sha256"]:
                raise ValueError("Canonical record integrity failure")
            records.append(record)
        if len(records) != 20 or len({r["record_id"] for r in records}) != 20:
            raise ValueError("Stage 2 is restricted to exactly 20 unique selected articles")
        needed = {uid for r in records for uid in r["source_position"]["unit_ids"]}
        units = {}
        for uid in sorted(needed):
            row = db.execute("SELECT payload FROM source_units WHERE source_document=? AND source_revision=? AND unit_id=?",
                             (manifest["source_document"], manifest["source_revision"], uid)).fetchone()
            if not row:
                raise ValueError("Missing canonical source unit")
            units[uid] = json.loads(row[0])
    for record in records:
        if "\n".join(units[u]["text"] for u in record["source_position"]["unit_ids"]) != record["original_text"]:
            raise ValueError("Source units do not reconstruct the canonical record")
    return records, units, manifest


def build_documents(records, units, relations, target_chars=900):
    """Group whole source paragraphs; long paragraphs are never arbitrarily cut."""
    if target_chars < 1:
        raise ValueError("target_chars must be positive")
    documents, coverage = [], {}
    by_record = {r["record_id"]: r for r in records}

    def add(record, kind, unit_ids, original, search_text, suffix):
        document_id = "search_" + digest(encoded([
            INDEX_VERSION, record["source_revision"], record["record_id"], kind, suffix]))[:32]
        documents.append({
            "document_id": document_id, "record_id": record["record_id"], "kind": kind,
            "title": record["title"], "original_text": original, "search_text": search_text,
            "source_document": record["source_document"], "source_revision": record["source_revision"],
            "section_path": record["section_path"],
            "source_position": {"unit_ids": unit_ids, "start": units[unit_ids[0]]["source_position"],
                                "end": units[unit_ids[-1]]["source_position"]},
        })

    for record in records:
        groups, current, size = [], [], 0
        for uid in record["source_position"]["unit_ids"]:
            text = units[uid]["text"]
            if current and size + 1 + len(text) > target_chars:
                groups.append(current)
                current, size = [], 0
            current.append(uid)
            size += len(text) + (1 if len(current) > 1 else 0)
        if current:
            groups.append(current)
        # Exact title and explicitly labelled external sign remain available in every chunk.
        anchors = [record["title"]]
        anchors += [units[u]["text"] for u in record["source_position"]["unit_ids"]
                    if units[u]["text"].startswith("Внешний признак.")]
        prefix = "Раздел: " + " / ".join(record["section_path"]) + "\nПризнак: " + "\n".join(anchors)
        for group in groups:
            original = "\n".join(units[u]["text"] for u in group)
            add(record, "general", group, original, prefix + "\nФрагмент источника:\n" + original,
                group)
        reconstructed = "\n".join("\n".join(units[u]["text"] for u in g) for g in groups)
        coverage[record["record_id"]] = {
            "unit_ids": [u for group in groups for u in group],
            "complete": reconstructed == record["original_text"],
            "general_fragments": len(groups),
        }
    seen = set()
    for relation in relations["relations"]:
        record = by_record.get(relation["record_id"])
        uid = relation["unit_id"]
        if not record or record["title"] != relation["sign"] or uid not in record["source_position"]["unit_ids"]:
            raise ValueError("Relation does not belong to selected canonical record")
        if units[uid]["text"] != relation["evidence"]:
            raise ValueError("Relation evidence is not a verbatim source unit")
        key = (record["record_id"], uid)
        if key in seen:
            raise ValueError("Duplicate relation")
        seen.add(key)
        for kind in ("forward", "reverse"):
            # Reverse is a retrieval permutation, never an assertion of logical equivalence.
            pair = [relation["sign"], relation["evidence"]]
            if kind == "reverse":
                pair.reverse()
            search_text = "Поисковая перестановка; утверждение атласа, не новая причинная связь.\n" + "\n".join(pair)
            add(record, kind, [uid], relation["evidence"], search_text, uid)
    if not all(c["complete"] for c in coverage.values()):
        raise ValueError("Fragment coverage failed")
    revision = digest(encoded({"version": INDEX_VERSION, "target_chars": target_chars,
                               "records": [(r["record_id"], r["original_text_sha256"]) for r in records],
                               "documents": documents}))
    for document in documents:
        document["index_revision"] = revision
    return documents, coverage, revision


class SearchIndex:
    """Dedicated search SQLite, separate from the immutable canonical database."""
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def connect(self, read_only=False):
        db = readonly(self.path) if read_only else sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def build(self, documents, records, revision):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            try:
                db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(title, text, tokenize='unicode61')")
            except sqlite3.OperationalError as exc:
                raise RuntimeError("FTS5 unavailable; use a separately implemented simple lexical fallback, no server required") from exc
            db.executescript("""
                CREATE TABLE IF NOT EXISTS index_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS search_documents(
                    row_id INTEGER PRIMARY KEY, document_id TEXT UNIQUE NOT NULL,
                    record_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS record_sources(record_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)
            previous = db.execute("SELECT value FROM index_metadata WHERE key='index_revision'").fetchone()
            if previous and previous[0] != revision:
                raise ValueError("Index revision differs. Use a new index path; existing data is never deleted.")
            expected_docs = {d["document_id"] for d in documents}
            existing_docs = {row[0] for row in db.execute("SELECT document_id FROM search_documents")}
            if existing_docs - expected_docs:
                raise ValueError("Unexpected extra search documents")
            for record in records:
                payload = encoded(record)
                existing = db.execute("SELECT payload FROM record_sources WHERE record_id=?", (record["record_id"],)).fetchone()
                if existing and existing[0] != payload:
                    raise ValueError("Source snapshot conflict")
                db.execute("INSERT OR IGNORE INTO record_sources VALUES (?,?)", (record["record_id"], payload))
            for document in documents:
                existing = db.execute("SELECT row_id,payload FROM search_documents WHERE document_id=?",
                                      (document["document_id"],)).fetchone()
                if existing:
                    if existing[1] != encoded(document):
                        raise ValueError("Search document conflict")
                    continue
                row_id = db.execute("INSERT INTO search_documents(document_id,record_id,kind,payload) VALUES (?,?,?,?)",
                                    (document["document_id"], document["record_id"], document["kind"], encoded(document))).lastrowid
                db.execute("INSERT INTO search_fts(rowid,title,text) VALUES (?,?,?)",
                           (row_id, document["title"], document["search_text"]))
            db.execute("INSERT OR IGNORE INTO index_metadata VALUES ('index_revision',?)", (revision,))
            # Check actual stored content, not only requested counts.
            rows = db.execute("SELECT d.payload,f.title,f.text FROM search_documents d LEFT JOIN search_fts f ON f.rowid=d.row_id").fetchall()
            if len(rows) != len(documents) or db.execute("SELECT COUNT(*) FROM search_fts").fetchone()[0] != len(documents):
                raise ValueError("FTS coverage mismatch")
            for payload, title, text in rows:
                d = json.loads(payload)
                if title != d["title"] or text != d["search_text"]:
                    raise ValueError("FTS content mismatch")
            if db.execute("SELECT COUNT(*) FROM record_sources").fetchone()[0] != len(records):
                raise ValueError("Canonical source coverage mismatch")

    def read(self):
        with self.connect(read_only=True) as db:
            documents = [json.loads(r[0]) for r in db.execute("SELECT payload FROM search_documents ORDER BY row_id")]
            records = {r[0]: json.loads(r[1]) for r in db.execute("SELECT record_id,payload FROM record_sources")}
            revision = db.execute("SELECT value FROM index_metadata WHERE key='index_revision'").fetchone()[0]
        return documents, records, revision


def index_vectors(vector_db, documents, revision):
    """Reuse storage.VectorDatabase; never silently replace a different index/model."""
    identity = {"index_revision": revision, "embedding_model": vector_db.embedding_model,
                "embedding_dimensions": vector_db.embedding_dimensions, "hnsw:space": "cosine"}
    collection = vector_db.get_or_create_collection(metadata=identity)
    for key, value in identity.items():
        if (collection.metadata or {}).get(key) != value:
            raise ValueError(f"Vector index identity mismatch: {key}")
    count = vector_db.upsert_documents(
        [d["search_text"] for d in documents],
        [{"record_id": d["record_id"], "kind": d["kind"], "index_revision": revision,
          "source_document": d["source_document"], "source_revision": d["source_revision"],
          "section_path": encoded(d["section_path"]),
          "source_position": encoded(d["source_position"])} for d in documents],
        [d["document_id"] for d in documents])
    stored = set(collection.get()["ids"])
    if stored != {d["document_id"] for d in documents}:
        raise ValueError("Vector coverage mismatch")
    return count
