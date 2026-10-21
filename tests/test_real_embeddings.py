import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest
from tools.real_embeddings import paid_batch, DIMENSIONS, load_inputs

class RealEmbeddingPreparationTests(unittest.TestCase):
    def test_fixed_49_documents_ten_questions(self):
        _,docs,records,_,cases=load_inputs()
        self.assertEqual((len(docs),len(records),len(cases)),(49,20,10))
        self.assertEqual(sum(not c["expected_record_ids"] for c in cases),2)

    def test_paid_budget_order_and_usage_with_fake_client(self):
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            def create(**kwargs):
                calls.append(kwargs)
                return S(data=[S(index=i,embedding=[float(i)]*DIMENSIONS) for i in reversed(range(len(kwargs["input"])))],
                         usage=S(total_tokens=5))
            client=S(embeddings=S(create=create))
            ledger=Path(d)/"ledger.json"
            self.assertEqual(paid_batch(client,["a","b"],ledger)[0][0],0)
            paid_batch(client,["q"],ledger)
            with self.assertRaises(RuntimeError): paid_batch(client,["third"],ledger)
            self.assertEqual(len(calls),2)
            self.assertEqual(json.loads(ledger.read_text())["reported_tokens"],10)

    def test_bad_dimension_counts_attempt_and_sanitizes_error(self):
        with tempfile.TemporaryDirectory() as d:
            client=S(embeddings=S(create=lambda **kw:S(data=[S(index=0,embedding=[1.0])],usage=S(total_tokens=1))))
            ledger=Path(d)/"ledger.json"
            with self.assertRaises(RuntimeError): paid_batch(client,["a"],ledger)
            data=json.loads(ledger.read_text())
            self.assertEqual(len(data["attempts"]),1)
            self.assertEqual(data["attempts"][0]["error_type"],"ValueError")
