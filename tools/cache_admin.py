"""Local answer-cache statistics and transactional cleanup. Never starts RAG."""
import argparse
import json
from pathlib import Path
from rag.cache import SQLiteCache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stats", "clear"))
    parser.add_argument("--scope", help="Exact chat:thread:user dialog key; omitted means all scopes")
    args = parser.parse_args(argv)
    path = Path(__file__).resolve().parents[1] / "data/rag_cache.sqlite3"
    if not path.exists():
        print(json.dumps({"hits": 0, "misses": 0, "entries": 0}))
        return
    cache = SQLiteCache(path)
    if args.action == "clear":
        cache.clear(args.scope)
    print(json.dumps(cache.stats(args.scope)))


if __name__ == "__main__":
    main()
