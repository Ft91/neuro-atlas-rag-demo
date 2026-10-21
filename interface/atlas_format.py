"""Plain-text Telegram rendering with numbered sources and lossless UTF-16 chunks."""
from rag.retrieval import NO_INFORMATION


def render_result(result):
    if result["status"] in ("search_error", "provider_error"):
        return result["answer"], []
    sources = {s["record_id"]: s for s in result["sources"]}
    ids = list(sources)
    short = {rid: rid.removeprefix("atlas_")[:8] for rid in ids}
    for rid in ids:
        if list(short.values()).count(short[rid]) > 1:
            short = {r: r.removeprefix("atlas_") for r in ids}
            break
    lines = []
    for part in result["parts"]:
        if len(result["parts"]) > 1:
            lines.append("Часть " + str(part["part_id"] + 1) + ": " + part["question"])
        if not part["evidence"]:
            lines.append(NO_INFORMATION)
        for card in part["evidence"]:
            rid = card["record_id"]
            body = part.get("explanation", part.get("explanation_refusal", card["quote"])) if result.get("focused") else card["quote"]
            lines.append(f"[{ids.index(rid) + 1}] {sources[rid]['title']}\n{body}")
    if ids:
        lines.append("Источники:")
        lines.extend(f"[{i}] {' / '.join(sources[rid]['section_path'])} · id:{short[rid]}" for i,rid in enumerate(ids, 1))
    return "\n\n".join(lines) or NO_INFORMATION, ids


def split_telegram(text, limit=3500):
    if limit < 2:
        raise ValueError("Chunk size too small")
    # Conservative limit in UTF-16 code units, preserving every original character.
    chunks, current, size = [], [], 0
    for char in text:
        width = 2 if ord(char) > 0xFFFF else 1
        if size + width > limit:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(char)
        size += width
    if current:
        chunks.append("".join(current))
    return chunks
