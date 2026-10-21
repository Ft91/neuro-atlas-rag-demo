"""Resolve references to verified source IDs only; never send history as evidence."""
import re
from rag.evidence import normalize
from rag.focus import fragment_reference

CLARIFY = "Уточните название признака или сначала задайте вопрос об атласе."
ORDINALS = {"первый": 1, "второй": 2, "третий": 3, "четвертый": 4, "пятый": 5}


def dialog_key(user_id, chat_id, thread_id=None):
    return f"{int(chat_id)}:{int(thread_id or 0)}:{int(user_id)}"


def resolve_followup(question, session, records):
    text = normalize(question)
    refs = session.get_metadata("last_sources", [])
    # Ignore any stored prose/title: resolve IDs against the CURRENT canonical store.
    titles = [records[rid]["title"] for rid in refs if rid in records]
    number = fragment_reference(question)
    if number is not None:
        fragments = session.get_metadata("last_evidence", [])
        if not isinstance(fragments, list) or not fragments:
            return None, "Уточните фрагмент: в предыдущем ответе нет сохранённой ссылки на абзац.", None
        if number == 0:
            if len(fragments) != 1:
                return None, "Уточните, какой фрагмент объяснить: например, «Объясни второй фрагмент». Можно явно попросить всю статью.", None
            number = 1
        if number > len(fragments):
            return None, "Уточните номер фрагмента предыдущего ответа.", None
        focus = fragments[number - 1]
        if not isinstance(focus, dict) or focus.get("record_id") not in refs:
            return None, CLARIFY, None
        return question, None, [dict(focus)]
    if text in ("покажи всю статью", "расскажи обо всей статье", "расскажи про всю статью",
                "вся статья", "покажи все характеристики", "все характеристики статьи"):
        return (titles[0], None, None) if len(titles) == 1 else (None, CLARIFY, None)
    for record in records.values():
        title = normalize(record["title"])
        if text in ("покажи всю статью «" + title, "расскажи про всю статью «" + title,
                    "покажи всю статью " + title, "вся статья: " + title):
            return record["title"], None, None
    match = re.fullmatch(r"(?:а )?(?:что означает|расскажи про|подробнее про) (первый|второй|третий|четвертый|пятый|[1-9][0-9]?)(?:-й)? признак", text)
    if match:
        word = match.group(1)
        n = ORDINALS[word] if word in ORDINALS else int(word)
        return (titles[n - 1], None, None) if n <= len(titles) else (None, CLARIFY, None)
    if text in ("а как его мотивировать", "как его мотивировать", "а что с мотивацией", "а как мотивировать", "а какие рекомендации по мотивации", "какие рекомендации по мотивации"):
        return ("Мотивация: " + titles[0], None, None) if len(titles) == 1 else (None, CLARIFY, None)
    return question, None, None
