import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from rag.docx_parser import W, parse_docx
from rag.canonical_store import CanonicalStore
from tools.ingest_atlas import ROOT, ingest, select_records


def paragraph(text, style=None):
    p = ET.Element(f"{{{W}}}p")
    if style:
        props = ET.SubElement(p, f"{{{W}}}pPr")
        ET.SubElement(props, f"{{{W}}}pStyle", {f"{{{W}}}val": style})
    run = ET.SubElement(p, f"{{{W}}}r")
    ET.SubElement(run, f"{{{W}}}t").text = text
    return p


def fixture(path, blocks, auxiliary=None):
    root = ET.Element(f"{{{W}}}document")
    body = ET.SubElement(root, f"{{{W}}}body")
    body.extend(blocks)
    with ZipFile(path, "w") as z:
        z.writestr("word/document.xml", ET.tostring(root, encoding="utf-8"))
        for name, xml in (auxiliary or {}).items():
            z.writestr(name, xml)
    return path


class ParserTests(unittest.TestCase):
    def test_exact_whitespace_short_paragraphs_and_nested_headings(self):
        with tempfile.TemporaryDirectory() as d:
            blocks = [paragraph("Глава", "Heading1"), paragraph("Признак", "Heading2"),
                      paragraph("  А  "), paragraph(""), paragraph("Дополнение", "Heading3"),
                      paragraph("Б"), paragraph("Следующий", "Heading2"), paragraph("В")]
            result = parse_docx(fixture(Path(d) / "test.docx", blocks))
            article = next(r for r in result.records if r["title"] == "Признак")
            self.assertEqual(article["original_text"], "Признак\n  А  \n\nДополнение\nБ")
            self.assertEqual(article["section_path"], ["Глава", "Признак"])
            self.assertIsNone(article["trait"])
            self.assertTrue(result.report["all_body_units_covered_once"])
            self.assertTrue(any(i["code"] == "nested_heading_kept" for i in result.report["issues"]))

    def test_table_cells_merges_nested_tables_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            table = ET.Element(f"{{{W}}}tbl")
            row = ET.SubElement(table, f"{{{W}}}tr")
            cell = ET.SubElement(row, f"{{{W}}}tc")
            props = ET.SubElement(cell, f"{{{W}}}tcPr")
            ET.SubElement(props, f"{{{W}}}gridSpan", {f"{{{W}}}val": "2"})
            cell.extend([paragraph("Признак"), paragraph("Коротко")])
            cell2 = ET.SubElement(row, f"{{{W}}}tc")
            cell2.append(paragraph("Характеристика"))
            nested = ET.SubElement(cell2, f"{{{W}}}tbl")
            nested_cell = ET.SubElement(ET.SubElement(nested, f"{{{W}}}tr"), f"{{{W}}}tc")
            nested_cell.append(paragraph("Рекомендация"))
            result = parse_docx(fixture(Path(d) / "table.docx", [
                paragraph("Глава", "Heading1"), paragraph("Статья", "Heading2"),
                paragraph("До"), table, paragraph("После")]))
            unit = next(u for u in result.units if u["kind"] == "table")
            self.assertEqual(unit["text"], "Признак\nКоротко\tХарактеристика\nРекомендация")
            self.assertEqual(unit["rows"][0]["cells"][0]["grid_span"], 2)
            self.assertEqual(result.report["source_counts"]["tables"], 2)
            article = result.records[-1]
            self.assertEqual(article["original_text"], "Статья\nДо\n" + unit["text"] + "\nПосле")
            self.assertTrue(result.report["source_text_nodes_preserved"])

    def test_tabs_breaks_hyperlinks_and_unicode(self):
        with tempfile.TemporaryDirectory() as d:
            p = paragraph("Ёж — ")
            run = p.find(f"{{{W}}}r")
            ET.SubElement(run, f"{{{W}}}tab")
            ET.SubElement(run, f"{{{W}}}t").text = "А"
            ET.SubElement(run, f"{{{W}}}br")
            link = ET.SubElement(p, f"{{{W}}}hyperlink")
            link.append(paragraph("Б").find(f"{{{W}}}r"))
            result = parse_docx(fixture(Path(d) / "unicode.docx", [p]))
            self.assertEqual(result.records[0]["original_text"], "Ёж — \tА\nБ")

    def test_ambiguous_part_does_not_invent_chapter(self):
        with tempfile.TemporaryDirectory() as d:
            result = parse_docx(fixture(Path(d) / "parts.docx", [
                paragraph("Глава 1", "Heading1"), paragraph("Часть II", "Atlaspart"),
                paragraph("Статья", "Heading2"), paragraph("Текст")]))
            self.assertEqual(result.records[-1]["section_path"], ["Часть II", "Статья"])
            self.assertTrue(any(i["code"] == "missing_chapter_heading" for i in result.report["issues"]))

    def test_unrecognized_block_and_auxiliary_are_reported(self):
        with tempfile.TemporaryDirectory() as d:
            wrapper = ET.Element(f"{{{W}}}sdt")
            wrapper.append(paragraph("Не терять"))
            result = parse_docx(fixture(Path(d) / "other.docx", [wrapper], {
                "word/footer1.xml": f'<w:ftr xmlns:w="{W}"><w:p><w:r><w:t>1</w:t></w:r></w:p></w:ftr>'}))
            self.assertEqual(result.records[0]["original_text"], "Не терять")
            self.assertTrue(any(i["code"] == "unsupported_body_block" for i in result.report["issues"]))
            self.assertEqual(result.report["auxiliary_parts"][0]["text"], "1")


class AtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROOT / "data/atlas_neurotypology_v15.docx"
        cls.parsed = parse_docx(cls.source)
        cls.manifest = json.loads((ROOT / "data/sample_selection.json").read_text(encoding="utf-8"))
        cls.sample = select_records(cls.parsed, cls.manifest)

    def test_twenty_complete_articles_and_section_diversity(self):
        self.assertEqual(len(self.sample), 20)
        self.assertEqual(len({r["section_path"][0] for r in self.sample}), 6)
        self.assertEqual(len({r["section_path"][1] for r in self.sample}), 20)
        self.assertEqual(self.parsed.report["source_counts"]["paragraphs"], 2228)
        for r in self.sample:
            self.assertEqual(r["source_document"], self.source.name)
            self.assertEqual(r["source_revision"], self.manifest["source_revision"])
            self.assertEqual(r["section_path"][-1], r["title"])
            self.assertTrue(r["original_text"].splitlines()[-1].startswith("Строки:"))
        eyes = next(r for r in self.sample if r["title"] == "Большие глаза")
        self.assertIn("Мотивация через образ.", eyes["original_text"])
        forehead = next(r for r in self.sample if r["title"] == "Прямой лоб")
        self.assertIn("Прямой лоб - дополнительные проявления", forehead["original_text"])

    def test_independent_xml_text_comparison(self):
        with ZipFile(self.source) as z:
            body = ET.fromstring(z.read("word/document.xml")).find(f"{{{W}}}body")
        for record in self.sample:
            start, end = record["source_position"]["start"], record["source_position"]["end"]
            expected = "\n".join("".join(n.text or "" for n in p.iter(f"{{{W}}}t"))
                                 for p in list(body)[start - 1:end])
            self.assertEqual(expected, record["original_text"])
        all_assigned = [uid for r in self.parsed.records for uid in r["source_position"]["unit_ids"]]
        self.assertEqual(all_assigned, [u["unit_id"] for u in self.parsed.units])

    def test_ids_repeat_and_match_full_import(self):
        again = parse_docx(self.source)
        self.assertEqual([r["record_id"] for r in again.records],
                         [r["record_id"] for r in self.parsed.records])
        with tempfile.TemporaryDirectory() as d:
            store = CanonicalStore(Path(d) / "test.sqlite3")
            store.save(self.parsed, self.sample)
            before = store.read_records(self.parsed.source_document, self.parsed.source_revision)
            store.save(self.parsed, self.sample)
            self.assertEqual(before, store.read_records(self.parsed.source_document, self.parsed.source_revision))
            store.save(self.parsed, self.parsed.records)
            all_records = {r["record_id"]: r for r in store.read_records(
                self.parsed.source_document, self.parsed.source_revision)}
            self.assertEqual(len(all_records), len(self.parsed.records))
            for record in before:
                self.assertEqual(record, all_records[record["record_id"]])

    def test_revision_guard_and_immutable_storage(self):
        bad = dict(self.manifest, source_revision="wrong")
        with self.assertRaises(ValueError):
            select_records(self.parsed, bad)
        with tempfile.TemporaryDirectory() as d:
            store = CanonicalStore(Path(d) / "test.sqlite3")
            store.save(self.parsed, self.sample)
            with store.connect() as db:
                db.execute("UPDATE canonical_records SET payload='{}' WHERE record_id=?",
                           (self.sample[0]["record_id"],))
            with self.assertRaises(ValueError):
                store.save(self.parsed, self.sample)

    def test_cli_report_and_search_are_separate(self):
        with tempfile.TemporaryDirectory() as d:
            report = ingest(self.source, Path(d) / "test.sqlite3", Path(d) / "out",
                            ROOT / "data/sample_selection.json")
            self.assertEqual(report["selected_records"], 20)
            self.assertEqual(report["indexed_records"], 0)
            self.assertEqual(len(report["deferred_records"]), len(self.parsed.records) - 20)
            store = CanonicalStore(Path(d) / "test.sqlite3")
            with store.connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM search_representations").fetchone()[0], 20)
                payload = json.loads(db.execute("SELECT payload FROM canonical_records LIMIT 1").fetchone()[0])
                self.assertNotIn("search_representations", payload)


if __name__ == "__main__":
    unittest.main()
