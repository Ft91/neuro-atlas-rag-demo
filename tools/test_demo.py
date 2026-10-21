"""Run all existing tests applicable to the ready-data demo; excludes full-DOCX import tests."""
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]

def flatten(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite): yield from flatten(item)
        else: yield item

def main():
    original=socket.socket.connect
    def blocked(sock,address):
        caller=sys._getframe(1)
        if caller.f_globals.get('__name__')=='socket' and caller.f_code.co_name=='socketpair':
            return original(sock,address)
        raise AssertionError('External network forbidden in demo tests')
    with patch('socket.socket.connect',blocked):
        tests=list(flatten(unittest.defaultTestLoader.discover(str(ROOT/'tests'))))
        excluded=[t.id() for t in tests if t.id().startswith('test_docx_parser.AtlasTests.')]
        assert len(excluded)==5,'Unexpected DOCX test inventory'
        suite=unittest.TestSuite(t for t in tests if t.id() not in excluded)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'skipped':len(result.skipped),'excluded_tests':excluded,
            'exclusion_reason':'These five tests parse the full original DOCX, intentionally not distributed.',
            'external_network_blocked':True,'paid_api_calls':0,'real_telegram_polling':False}
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/demo_tests.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return not result.wasSuccessful()
if __name__=='__main__':raise SystemExit(main())
