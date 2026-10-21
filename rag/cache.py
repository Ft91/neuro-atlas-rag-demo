"""Private, versioned SQLite answer cache. No network and no shared connections."""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import unicodedata


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def cache_key(question, identity, scope):
    # Preserve punctuation and newlines: they delimit compound questions.
    return digest({"question": unicodedata.normalize("NFC", question).strip(),
                   "identity": identity, "scope": scope})


class SQLiteCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS answers (
                    key TEXT PRIMARY KEY, scope TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS counters (
                    scope TEXT PRIMARY KEY, hits INTEGER NOT NULL, misses INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS cache_state (id INTEGER PRIMARY KEY, epoch INTEGER NOT NULL);
                INSERT OR IGNORE INTO cache_state VALUES (1, 0);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, key, scope, validate):
        scope = digest(scope)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            epoch = db.execute("SELECT epoch FROM cache_state WHERE id=1").fetchone()[0]
            row = db.execute("SELECT payload FROM answers WHERE key=? AND scope=?", (key, scope)).fetchone()
            value = None
            if row:
                try:
                    candidate = json.loads(row[0])
                    value = validate(candidate)
                except (ValueError, TypeError, KeyError, AttributeError):
                    pass
                if value is None:
                    db.execute("DELETE FROM answers WHERE key=?", (key,))
            hit = int(value is not None)
            db.execute("INSERT INTO counters VALUES (?, ?, ?) ON CONFLICT(scope) DO UPDATE SET hits=hits+excluded.hits, misses=misses+excluded.misses", (scope, hit, 1-hit))
            return value, epoch

    def put(self, key, scope, value, epoch):
        payload = json.dumps(value, ensure_ascii=False)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT epoch FROM cache_state WHERE id=1").fetchone()[0] != epoch:
                return False  # A clear during generation must not resurrect old answers.
            db.execute("INSERT INTO answers VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload", (key, digest(scope), payload))
            return True

    def stats(self, scope=None):
        with self.connect() as db:
            if scope is None:
                row = db.execute("SELECT coalesce(sum(hits),0), coalesce(sum(misses),0) FROM counters").fetchone()
                count = db.execute("SELECT count(*) FROM answers").fetchone()[0]
            else:
                row = db.execute("SELECT hits, misses FROM counters WHERE scope=?", (digest(scope),)).fetchone() or (0,0)
                count = db.execute("SELECT count(*) FROM answers WHERE scope=?", (digest(scope),)).fetchone()[0]
            return {"hits": row[0], "misses": row[1], "entries": count}

    def clear(self, scope=None):
        # Transactional DELETE, never remove a live SQLite/WAL file.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE cache_state SET epoch=epoch+1 WHERE id=1")
            for table in ("answers", "counters"):
                if scope is None:
                    db.execute("DELETE FROM " + table)
                else:
                    db.execute("DELETE FROM " + table + " WHERE scope=?", (digest(scope),))
