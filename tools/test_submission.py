"""Run 14 existing data-independent tests without secrets, atlas or external networking."""
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0, str(ROOT / "tests"))
    original = socket.socket.connect
    def blocked_connect(sock, address):
        caller = sys._getframe(1)
        if caller.f_globals.get("__name__") == "socket" and caller.f_code.co_name == "socketpair":
            return original(sock, address)
        raise AssertionError("External network forbidden during submission tests")
    names = ["test_docx_parser.ParserTests", "test_real_embeddings.RealEmbeddingPreparationTests.test_paid_budget_order_and_usage_with_fake_client",
             "test_real_embeddings.RealEmbeddingPreparationTests.test_bad_dimension_counts_attempt_and_sanitizes_error",
             "test_rag_pipeline.ProviderTests"]
    with patch("socket.socket.connect", blocked_connect):
        suite = unittest.defaultTestLoader.loadTestsFromNames(names)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "atlas_required": False, "external_network_blocked": True,
              "paid_api_calls": 0, "real_telegram_polling": False}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/submission_tests.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return not result.wasSuccessful()


if __name__ == "__main__":
    raise SystemExit(main())
