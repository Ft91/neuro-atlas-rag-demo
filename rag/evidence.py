"""Conservative request-to-source mapping, independent of LLM judgments."""
import json
import re
from pathlib import Path
from rag.indexing import load_sample


def normalize(text):
    return " ".join(text.casefold().replace("ё", "е").split()).strip(" .?!«»\"")


def split_question(question):
    # Do not split 'физическая и психическая выносливость'. Unsupported
    # conjunctions remain one unsupported request rather than silently dropping text.
    pieces = re.split(r"[;?\n]+|,?\s+(?:и|а)\s+(?=(?:какой|какая|какие|каково|как|что|где|когда)\b)", question, flags=re.I)
    return [p.strip() for p in pieces if p.strip(" .?!")]


class EvidencePolicy:
    def __init__(self, root):
        root = Path(root)
        records, self.units, manifest = load_sample(root / "data/atlas.sqlite3", root / "data/sample_selection.json")
        self.records = {r["record_id"]: r for r in records}
        self.policy = json.loads((root / "data/evidence_policy.json").read_text(encoding="utf-8"))
        reviewed = json.loads((root / "data/search_relations.json").read_text(encoding="utf-8"))
        if self.policy["source_revision"] != manifest["source_revision"] or reviewed["source_revision"] != manifest["source_revision"]:
            raise ValueError("Evidence source revision mismatch")
        self.routes = {}
        for rid, r in self.records.items():
            body = [u for u in r["source_position"]["unit_ids"]
                    if self.units[u]["heading_level"] is None
                    and not self.units[u]["text"].startswith(("Строки:", "Внешний признак."))]
            title = r["title"]
            for text in (title, "Характеристики: " + title,
                         'Какие характеристики связаны с признаком «' + title + '»',
                         'Что в атласе сказано о признаке «' + title + '»'):
                self.add(text, "forward", rid, body)
        for entry in self.policy["relations"]:
            rid, uid = entry["record_id"], entry["unit_id"]
            r = self.records[rid]
            match = next((x for x in reviewed["relations"] if x["record_id"] == rid and x["unit_id"] == uid), None)
            if match is None or match["sign"] != r["title"] or match["evidence"] != self.units[uid]["text"]:
                raise ValueError("Unreviewed or changed reverse relation")
            self.card(rid, uid)
            for alias in entry["aliases"]:
                if normalize(alias) not in normalize(self.units[uid]["text"]):
                    raise ValueError("Alias must be a literal source phrase")
                for text in (alias, "Признаки: " + alias,
                             'Какой внешний признак связан с характеристикой «' + alias + '»'):
                    self.add(text, "reverse", rid, [uid])
            for item in entry.get("reviewed_paraphrases", []):
                cards = [self.card(rid, u) for u in item["unit_ids"]]
                if not any(item["source_phrase"] in c["quote"] for c in cards):
                    raise ValueError("Reviewed paraphrase source changed")
                self.add("Признаки: " + item["phrase"], "reverse", rid, item["unit_ids"])
                for template in item["templates"]:
                    self.add(template.format(phrase=item["phrase"]), "reverse", rid, item["unit_ids"])
            for kind, ids in entry["practical"].items():
                if kind not in ("motivation", "interaction") or not ids:
                    raise ValueError("Invalid practical evidence")
                label = "Мотивация" if kind == "motivation" else "Взаимодействие"
                verb = "мотивировать" if kind == "motivation" else "взаимодействовать с"
                for target in [r["title"]] + entry["aliases"]:
                    self.add(label + ": " + target, kind, rid, ids)
                for alias in entry["aliases"]:
                    if kind == "motivation":
                        self.add('Как ' + verb + ' человека с характеристикой «' + alias + '»', kind, rid, ids)
                    if kind == "interaction":
                        self.add('Как взаимодействовать с человеком с характеристикой «' + alias + '»', kind, rid, ids)

    def card(self, rid, uid):
        r = self.records[rid]
        if uid not in r["source_position"]["unit_ids"]:
            raise ValueError("Evidence belongs to another record")
        quote = self.units[uid]["text"]
        if not quote or quote not in r["original_text"]:
            raise ValueError("Invalid canonical evidence")
        return {"record_id": rid, "unit_id": uid, "quote": quote}

    def add(self, text, direction, rid, ids):
        route = self.routes.setdefault(normalize(text), {"direction": direction, "cards": []})
        if route["direction"] != direction:
            raise ValueError("Ambiguous request direction")
        for uid in ids:
            card = self.card(rid, uid)
            if card not in route["cards"]:
                route["cards"].append(card)

    def requests(self, question):
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("Введите непустой вопрос длиной до 4000 символов")
        parts = split_question(question)
        if not parts or len(parts) > 8:
            raise ValueError("Разделите вопрос максимум на 8 частей")
        return [{"part_id": i, "question": text,
                 **self.routes.get(normalize(text), {"direction": "unsupported", "cards": []})}
                for i, text in enumerate(parts)]
