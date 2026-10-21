"""References to delivered canonical paragraphs, never historical prose as evidence."""
from hashlib import sha256
import re
from rag.evidence import normalize, split_question

GENERIC = {"а что это означает", "что это означает", "что это значит", "а что это значит",
           "объясни это", "объясни подробнее", "расскажи подробнее", "подробнее",
           "что имеется в виду", "а что имеется в виду"}
QUOTES = {"покажи оригинал", "процитируй", "покажи исходный текст", "покажи оригинал абзаца"}
ORDINALS = {"первый": 1, "второй": 2, "третий": 3, "четвертый": 4, "пятый": 5}


def fragment_intents(question):
    parts = split_question(question)
    if not parts or len(parts) > 8:
        return None
    result = []
    for part in parts:
        text = normalize(part)
        if text in GENERIC or text in QUOTES:
            mode, number = ("quote" if text in QUOTES else "explain"), 0
        else:
            match = re.fullmatch(r"(?:а )?(что означает|объясни|расскажи про|подробнее про|процитируй|покажи оригинал) (первый|второй|третий|четвертый|пятый|[1-9][0-9]?)(?:-й)? (?:фрагмент|абзац|пункт)", text)
            if not match:
                return None
            word = match.group(2)
            number = ORDINALS[word] if word in ORDINALS else int(word)
            mode = "quote" if match.group(1) in ("процитируй", "покажи оригинал") else "explain"
        result.append({"question": part, "mode": mode, "number": number})
    # Different explicitly selected fragments require separate resolution.
    if len({p["number"] for p in result if p["number"]}) > 1:
        return None
    return result


def fragment_reference(question):
    intents = fragment_intents(question)
    return max(p["number"] for p in intents) if intents else None


def evidence_refs(result):
    sources = {r["record_id"]: r for r in result["sources"]}
    refs = []
    for part in result["parts"]:
        if part["status"] not in ("supported", "source_only"):
            continue
        for card in part["evidence"]:
            ref = {"record_id": card["record_id"], "unit_id": card["unit_id"],
                   "source_revision": sources[card["record_id"]]["source_revision"],
                   "quote_sha256": sha256(card["quote"].encode("utf-8")).hexdigest()}
            if ref not in refs:
                refs.append(ref)
    return refs if len(refs) <= 64 else []


def canonical_focus(policy, refs):
    if not isinstance(refs, list) or len(refs) != 1:
        raise ValueError("Ambiguous fragment")
    ref = refs[0]
    if not isinstance(ref, dict) or set(ref) != {"record_id", "unit_id", "source_revision", "quote_sha256"}:
        raise ValueError("Invalid fragment reference")
    record = policy.records[ref["record_id"]]
    if ref["source_revision"] != record["source_revision"]:
        raise ValueError("Stale fragment")
    card = policy.card(ref["record_id"], ref["unit_id"])
    if sha256(card["quote"].encode("utf-8")).hexdigest() != ref["quote_sha256"]:
        raise ValueError("Changed fragment")
    return card

