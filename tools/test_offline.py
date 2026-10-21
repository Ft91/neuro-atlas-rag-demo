"""Run all tests with external sockets blocked, allowing Windows asyncio socketpair only."""
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = socket.socket.connect
    def blocked_connect(sock, address):
        caller = sys._getframe(1)
        # Windows implements socket.socketpair using a loopback TCP connection.
        # Permit that stdlib implementation only, never general loopback proxies/HTTP.
        if caller.f_globals.get("__name__") == "socket" and caller.f_code.co_name == "socketpair":
            return original(sock, address)
        raise AssertionError("External network forbidden during offline tests")
    with patch("socket.socket.connect", blocked_connect):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / "tests")))
    report = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "external_network_blocked": True,
              "windows_asyncio_socketpair_allowed": True, "telegram_polling_real": False, "paid_api_calls": 0}
    (ROOT / "reports/stage6_tests.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return not result.wasSuccessful()


if __name__ == "__main__":
    raise SystemExit(main())
