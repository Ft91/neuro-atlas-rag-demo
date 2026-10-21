"""Independent FTS5/vector retrieval with canonical-record Reciprocal Rank Fusion."""
from dataclasses import dataclass
import json
import os
import re

NO_INFORMATION = "В атласе не нашёл подтверждения"
STOP_WORDS = set("а и или в во на по о об от для с со к у как какой какая какие что это ли из при через".split())


@dataclass(frozen=True)
class SearchSettings:
    top_k: int = 5
    hybrid_search: bool = True

    @classmethod
    def from_env(cls):
        top_k = int(os.getenv("SEARCH_TOP_K", "5"))
        flag = os.getenv("HYBRID_SEARCH", "true").lower()
        if top_k <= 0 or flag not in ("true", "false"):
            raise ValueError("Invalid SEARCH_TOP_K/HYBRID_SEARCH")
        return cls(top_k, flag == "true")


def reciprocal_rank_fusion(rankings, constant=60):
    """Deduplicate each channel BEFORE assigning ranks; never sum BM25/distances."""
    scores, matches, ranks = {}, {}, {}
    for channel, documents in rankings.items():
        seen = set()
        for document in documents:
            rid = document["record_id"]
            matches.setdefault(rid, {})[document["document_id"]] = document
            if rid in seen:
                continue
            seen.add(rid)
            rank = len(seen)
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (constant + rank)
            ranks.setdefault(rid, {})[channel] = rank
    return [(rid, scores[rid], list(matches[rid].values()), ranks[rid])
            for rid in sorted(scores, key=lambda rid: (-scores[rid], rid))]


class HybridRetriever:
    def __init__(self, index, vector_db=None, settings=None, vector_max_distance=None):
        self.index = index
        self.vector_db = vector_db
        self.settings = settings or SearchSettings.from_env()
        # Default search still requires calibration. candidate_only=True deliberately
        # returns unverified neighbours for the independent canonical evidence gate.
        self.vector_max_distance = vector_max_distance
        self.documents, self.records, self.revision = index.read()
        self.by_id = {d["document_id"]: d for d in self.documents}

    def _eligible(self, section, direction):
        if direction not in ("general", "forward", "reverse"):
            raise ValueError("direction must be general, forward or reverse")
        return [d for d in self.documents
                if (direction == "general" or d["kind"] == direction)
                and (section is None or section in d["section_path"]
                     or section == " / ".join(d["section_path"]))]

    def lexical(self, query, section=None, direction="general"):
        eligible = {d["document_id"] for d in self._eligible(section, direction)}
        terms = list(dict.fromkeys(t for t in re.findall(r"\w+", query.lower(), flags=re.UNICODE)
                                   if t not in STOP_WORDS))[:32]
        if not terms or not eligible:
            return []
        expression = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        with self.index.connect(read_only=True) as db:
            rows = db.execute("""
                SELECT d.document_id,bm25(search_fts,5.0,1.0) AS score
                FROM search_fts JOIN search_documents d ON d.row_id=search_fts.rowid
                WHERE search_fts MATCH ? ORDER BY score,d.document_id
            """, (expression,)).fetchall()
        return [dict(self.by_id[doc_id], bm25=score) for doc_id, score in rows if doc_id in eligible]

    def semantic(self, query, section=None, direction="general", *, candidate_only=False):
        if self.vector_db is None:
            raise RuntimeError("Vector backend not configured; explicitly use mode='fts' for offline search")
        if self.vector_max_distance is None and not candidate_only:
            raise ValueError("Set a calibrated cosine distance threshold; mock threshold is not valid for OpenAI")
        collection = self.vector_db.collection or self.vector_db.get_or_create_collection()
        identity = collection.metadata or {}
        expected = {"index_revision": self.revision, "embedding_model": self.vector_db.embedding_model,
                    "embedding_dimensions": self.vector_db.embedding_dimensions, "hnsw:space": "cosine"}
        if any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError("Vector collection does not match search index/model")
        eligible = self._eligible(section, direction)
        if not eligible:
            return []
        record_ids = sorted({d["record_id"] for d in eligible})
        clauses = [{"index_revision": self.revision}, {"record_id": {"$in": record_ids}}]
        if direction != "general":
            clauses.append({"kind": direction})
        response = self.vector_db.search(query, n_results=len(eligible), where={"$and": clauses})
        matches = []
        for doc_id, distance in zip(response["ids"][0], response["distances"][0]):
            if doc_id not in self.by_id:
                raise ValueError("Unknown vector source ID")
            if candidate_only or distance <= self.vector_max_distance:
                matches.append(dict(self.by_id[doc_id], distance=distance))
        return matches

    def search(self, query, mode=None, section=None, direction="general", top_k=None, *, candidate_only=False):
        mode = mode or ("hybrid" if self.settings.hybrid_search else "vector")
        count = self.settings.top_k if top_k is None else top_k
        if count < 1 or mode not in ("fts", "vector", "hybrid"):
            raise ValueError("Invalid search mode/top_k")
        rankings = {}
        if mode in ("fts", "hybrid"):
            rankings["fts"] = self.lexical(query, section, direction)
        if mode in ("vector", "hybrid"):
            rankings["vector"] = self.semantic(query, section, direction, candidate_only=candidate_only)
        merged = reciprocal_rank_fusion(rankings)
        results = []
        for rid, score, evidence, channel_ranks in merged[:count]:
            record = self.records[rid]
            results.append({
                "record_id": rid, "title": record["title"], "score": score, "channel_ranks": channel_ranks,
                "source": {k: record[k] for k in ("source_document", "source_revision",
                                                  "section_path", "source_position", "record_id")},
                "original_text": record["original_text"],
                "evidence": [{"document_id": d["document_id"], "kind": d["kind"],
                              "original_text": d["original_text"], "source_position": d["source_position"]}
                             for d in evidence],
            })
        return {"status": ("candidates" if candidate_only else "matches") if results else "no_information",
                "unverified_candidates": candidate_only,
                "message": "" if results else NO_INFORMATION, "mode": mode,
                "index_revision": self.revision, "results": results}
