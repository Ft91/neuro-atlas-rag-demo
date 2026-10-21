"""Append-only canonical SQLite storage, separate from search representations."""
import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class CanonicalStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sources(
                    source_document TEXT NOT NULL, source_revision TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    PRIMARY KEY(source_document, source_revision));
                CREATE TABLE IF NOT EXISTS source_units(
                    source_document TEXT NOT NULL, source_revision TEXT NOT NULL,
                    unit_id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(source_document, source_revision, unit_id),
                    FOREIGN KEY(source_document,source_revision)
                    REFERENCES sources(source_document,source_revision));
                CREATE TABLE IF NOT EXISTS canonical_records(
                    source_document TEXT NOT NULL, source_revision TEXT NOT NULL,
                    record_id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(source_document,source_revision,record_id),
                    FOREIGN KEY(source_document,source_revision)
                    REFERENCES sources(source_document,source_revision));
                CREATE TABLE IF NOT EXISTS search_representations(
                    source_document TEXT NOT NULL, source_revision TEXT NOT NULL,
                    record_id TEXT NOT NULL, kind TEXT NOT NULL, version TEXT NOT NULL,
                    text TEXT NOT NULL,
                    PRIMARY KEY(source_document,source_revision,record_id,kind,version),
                    FOREIGN KEY(source_document,source_revision,record_id)
                    REFERENCES canonical_records(source_document,source_revision,record_id));
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _immutable(db, table, key_columns, key_values, value_columns, values):
        where = " AND ".join(c + "=?" for c in key_columns)
        row = db.execute(f"SELECT {','.join(value_columns)} FROM {table} WHERE {where}",
                         key_values).fetchone()
        if row is not None:
            if tuple(row) != tuple(values):
                raise ValueError(f"Immutable {table} conflict: {key_values}")
            return
        columns = key_columns + value_columns
        db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                   key_values + values)

    def save(self, parsed, records):
        source = [parsed.source_document, parsed.source_revision]
        known = {r["record_id"]: r for r in parsed.records}
        if any(r != known.get(r["record_id"]) for r in records):
            raise ValueError("Only exact parser records may be stored")
        needed = {uid for r in records for uid in r["source_position"]["unit_ids"] + r["context_unit_ids"]}
        with self.connect() as db:
            self._immutable(db, "sources", ["source_document", "source_revision"], source,
                            ["parser_version"], [parsed.report["parser_version"]])
            for unit in parsed.units:
                if unit["unit_id"] in needed:
                    self._immutable(db, "source_units", ["source_document", "source_revision", "unit_id"],
                                    source + [unit["unit_id"]], ["payload"], [encoded(unit)])
            for record in records:
                canonical = {k: v for k, v in record.items() if k != "search_representations"}
                self._immutable(db, "canonical_records", ["source_document", "source_revision", "record_id"],
                                source + [record["record_id"]], ["payload"], [encoded(canonical)])
                for representation in record["search_representations"]:
                    self._immutable(db, "search_representations",
                                    ["source_document", "source_revision", "record_id", "kind", "version"],
                                    source + [record["record_id"], representation["kind"], representation["version"]],
                                    ["text"], [representation["text"]])

    def read_records(self, source_document, revision):
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM canonical_records WHERE source_document=? AND source_revision=?",
                (source_document, revision)).fetchall()
        return sorted((json.loads(row[0]) for row in rows), key=lambda r: r["source_position"]["start"])
