"""Lossless text extraction and conservative structural segmentation of DOCX.

No SDK, dotenv, network, or model calls. Positions are 1-based body child
indices, not page numbers. IDs are independent of sample size and revision.
"""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
PARSER_VERSION = "atlas-structural-v1"


def tag(name):
    return f"{{{W}}}{name}"


def digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


def visible_text(element):
    """Preserve text-node whitespace, explicit tabs, breaks and hyphens."""
    pieces = []
    for node in element.iter():
        if node.tag == tag("t"):
            pieces.append(node.text or "")
        elif node.tag == tag("tab"):
            pieces.append("\t")
        elif node.tag in (tag("br"), tag("cr")):
            pieces.append("\n")
        elif node.tag == tag("noBreakHyphen"):
            pieces.append("\u2011")
        elif node.tag == tag("softHyphen"):
            pieces.append("\u00ad")
    return "".join(pieces)


def _style_levels(package):
    if "word/styles.xml" not in package.namelist():
        return {}
    root = ET.fromstring(package.read("word/styles.xml"))
    styles = {s.get(tag("styleId")): s for s in root.findall("w:style", NS)}

    def resolve(style_id, seen):
        if style_id in seen or style_id not in styles:
            return None
        seen = seen | {style_id}
        s = styles[style_id]
        outline = s.find("w:pPr/w:outlineLvl", NS)
        if outline is not None:
            value = int(outline.get(tag("val")))
            return value + 1 if value < 9 else None
        name = s.find("w:name", NS)
        match = re.fullmatch(r"heading\s*(\d+)", name.get(tag("val"), ""), re.I) if name is not None else None
        if match:
            return int(match[1])
        parent = s.find("w:basedOn", NS)
        return resolve(parent.get(tag("val")), seen) if parent is not None else None

    return {name: resolve(name, set()) for name in styles}


def _paragraph(p, levels):
    style_node = p.find("w:pPr/w:pStyle", NS)
    style = style_node.get(tag("val"), "") if style_node is not None else ""
    outline = p.find("w:pPr/w:outlineLvl", NS)
    level = levels.get(style)
    if level is None:
        match = re.fullmatch(r"Heading(\d+)", style, re.I)
        level = int(match[1]) if match else None
    if outline is not None:
        number = int(outline.get(tag("val")))
        level = number + 1 if number < 9 else None
    if style.lower() == "atlaspart":
        level = 0
    return {"text": visible_text(p), "style": style, "heading_level": level}


def _table(table):
    """Keep full rows/cells, paragraph order, merge metadata and nested tables."""
    rows = []
    for row_index, tr in enumerate(table.findall("w:tr", NS), 1):
        cells = []
        for cell_index, tc in enumerate(tr.findall("w:tc", NS), 1):
            blocks = []
            for child in tc:
                if child.tag == tag("p"):
                    blocks.append({"kind": "paragraph", "text": visible_text(child)})
                elif child.tag == tag("tbl"):
                    blocks.append({"kind": "table", **_table(child)})
                elif child.tag != tag("tcPr"):
                    blocks.append({"kind": "unresolved", "text": visible_text(child),
                                   "xml": ET.tostring(child, encoding="unicode")})
            span = tc.find("w:tcPr/w:gridSpan", NS)
            merge = tc.find("w:tcPr/w:vMerge", NS)
            cells.append({
                "cell": cell_index,
                "grid_span": int(span.get(tag("val"))) if span is not None else 1,
                "vertical_merge": merge.get(tag("val"), "continue") if merge is not None else None,
                "blocks": blocks, "text": "\n".join(b["text"] for b in blocks),
            })
        rows.append({"row": row_index, "cells": cells})
    return {"rows": rows, "text": "\n".join(
        "\t".join(c["text"] for c in r["cells"]) for r in rows)}


@dataclass
class ParsedDocument:
    source_document: str
    source_revision: str
    units: list
    records: list
    report: dict


def parse_docx(path, source_document=None):
    path = Path(path)
    source_document = source_document or path.name
    revision = sha256(path.read_bytes()).hexdigest()
    issues, units, auxiliary = [], [], []
    with ZipFile(path) as package:
        root = ET.fromstring(package.read("word/document.xml"))
        body = root.find("w:body", NS)
        if body is None:
            raise ValueError("DOCX has no document body")
        levels = _style_levels(package)
        counts = {name: len(body.findall(".//w:" + name, NS))
                  for name in ("p", "tbl", "tr", "tc")}
        counts = dict(zip(("paragraphs", "tables", "rows", "cells"), counts.values()))
        for position, node in enumerate(body, 1):
            if node.tag == tag("sectPr"):
                continue
            unit_id = f"body/{position}"
            if node.tag == tag("p"):
                data = {"kind": "paragraph", **_paragraph(node, levels)}
            elif node.tag == tag("tbl"):
                data = {"kind": "table", "style": "", "heading_level": None, **_table(node)}
                issues.append({"code": "table_kept_whole", "unit_id": unit_id,
                               "reason": "Row relations are not inferred; entire table is retained."})
            else:
                data = {"kind": "unresolved", "style": "", "heading_level": None,
                        "text": visible_text(node)}
                issues.append({"code": "unsupported_body_block", "unit_id": unit_id,
                               "reason": node.tag, "action": "retained with raw XML"})
            raw = ET.tostring(node, encoding="unicode")
            data.update(unit_id=unit_id, source_position=position,
                        raw_xml=raw, text_nodes=[n.text or "" for n in node.iter(tag("t"))])
            units.append(data)
            for special in ("drawing", "pict", "ins", "del", "sym", "footnoteReference",
                            "endnoteReference", "numPr", "fldSimple", "instrText"):
                if node.find(".//w:" + special, NS) is not None:
                    issues.append({"code": "requires_review", "unit_id": unit_id,
                                   "reason": special,
                                   "action": "raw XML retained; no inferred rendering"})
        for name in package.namelist():
            if re.fullmatch(r"word/(header\d*|footer\d*|footnotes|endnotes|comments)\.xml", name):
                node = ET.fromstring(package.read(name))
                auxiliary.append({"part": name, "text": visible_text(node),
                                  "reason": "Outside main document body; retained in source DOCX, requires separate review."})
        original_nodes = [n.text or "" for n in body.iter(tag("t"))]

    records, current, sections, section_units = [], [], {}, {}
    current_path, current_context, current_type = [], [], "front_matter"

    def finish():
        if not current:
            return
        text = "\n".join(u["text"] for u in current)
        record_type = current_type
        start = current[0]["unit_id"]
        record_id = "atlas_" + digest(source_document + "\0" + start + "\0" + record_type)[:24]
        records.append({
            "record_id": record_id, "original_text": text,
            "original_text_sha256": digest(text),
            "source_document": source_document, "source_revision": revision,
            "section_path": list(current_path),
            "source_position": {"start": current[0]["source_position"],
                                "end": current[-1]["source_position"],
                                "unit_ids": [u["unit_id"] for u in current]},
            "context_unit_ids": list(current_context), "record_type": record_type,
            "title": current[0]["text"] if current[0]["heading_level"] is not None else "",
            "external_sign": None, "trait": None, "motivation": None,
            "search_representations": [{
                "kind": "general", "version": "verbatim-normalized-v1",
                "text": " ".join(text.split())}],
        })
        current.clear()

    for unit_index, unit in enumerate(units):
        level = unit["heading_level"]
        if level is not None and level <= 2:
            finish()
            next_heading = next((u["heading_level"] for u in units[unit_index + 1:] if u["heading_level"] is not None), None)
            if level == 0 and 1 in sections and next_heading != 1:
                issues.append({"code": "part_boundary_review", "unit_id": unit["unit_id"],
                               "reason": "Part boundary encountered after a chapter; hierarchy follows XML, without relocating content."})
            for key in list(sections):
                if key >= level:
                    del sections[key]
                    del section_units[key]
            sections[level] = unit["text"]
            section_units[level] = unit["unit_id"]
            current_path = [sections[k] for k in sorted(sections)]
            current_context = [section_units[k] for k in sorted(section_units) if k < level]
            current_type = "article" if level == 2 else "structural_heading"
            current.append(unit)
            if level == 2 and 1 not in sections:
                issues.append({"code": "missing_chapter_heading", "unit_id": unit["unit_id"],
                               "reason": "H2 has no current H1; no chapter is inferred."})
            if level < 2:
                finish()
        else:
            if not current:
                current_path = [sections[k] for k in sorted(sections)]
                current_context = [section_units[k] for k in sorted(section_units)]
                current_type = "section_preamble" if sections else "front_matter"
            current.append(unit)
            if level is not None:
                issues.append({"code": "nested_heading_kept", "unit_id": unit["unit_id"],
                               "parent_unit_id": current[0]["unit_id"],
                               "reason": "Nested heading and all following content retained inside parent article; semantic relation not inferred."})
    finish()
    assigned = [uid for r in records for uid in r["source_position"]["unit_ids"]]
    text_nodes = [t for u in units for t in u["text_nodes"]]
    covered = assigned == [u["unit_id"] for u in units] and text_nodes == original_nodes
    if not covered:
        raise ValueError("Source coverage invariant failed")
    occurrences = Counter(r["original_text_sha256"] for r in records if r["original_text"].strip())
    report = {
        "parser_version": PARSER_VERSION, "source_document": source_document,
        "source_revision": revision, "source_counts": counts,
        "body_units": len(units), "nonempty_body_units": sum(bool(u["text"].strip()) for u in units),
        "extracted_records": len(records),
        "article_records": sum(r["record_type"] == "article" for r in records),
        "all_body_units_covered_once": covered, "source_text_nodes_preserved": text_nodes == original_nodes,
        "duplicate_text_groups": sum(n > 1 for n in occurrences.values()),
        "duplicate_text_records": sum(n - 1 for n in occurrences.values()),
        "issues": issues, "auxiliary_parts": auxiliary,
        "boundary_policy": "Whole H2 article through next H2/H1/Atlaspart. Nested headings retained. Structural headings and preambles stored separately.",
        "id_policy": "sha256(source_document, first body position, record_type); independent of selection/revision. Insertions before a record may change its ID in a new DOCX revision.",
        "text_policy": "Exact OOXML text nodes with tabs/breaks; paragraph separators LF and table cell separators TAB are explicit serialization, not original characters.",
    }
    return ParsedDocument(source_document, revision, units, records, report)
