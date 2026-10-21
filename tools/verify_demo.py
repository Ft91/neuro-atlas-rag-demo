"""Verify the ready 20-article demo offline. Operates on a temporary data copy only."""
from pathlib import Path
import json
import hashlib
import socket
import sqlite3
import sys
import tempfile
import shutil
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class MockProvider:
    def __init__(self): self.calls = 0
    def generate_response(self, messages):
        self.calls += 1
        data = json.loads(messages[-1]['content'])
        return json.dumps({'parts':[{'part_id':p['part_id'],'evidence':p['allowed_evidence'][:1]}
                                   for p in data['parts']]}, ensure_ascii=False)

def verify():
    original_connect = socket.socket.connect
    def blocked(sock, address):
        caller = sys._getframe(1)
        if caller.f_globals.get('__name__') == 'socket' and caller.f_code.co_name == 'socketpair':
            return original_connect(sock,address)
        raise AssertionError('Network prohibited in demo verification')
    assert not list(ROOT.rglob('*.docx')), 'Full DOCX must not be shipped'
    assert not (ROOT/'.env').exists(), 'Verify clean unpacked delivery before adding credentials'
    expected = json.loads((ROOT/'DATA_MANIFEST.json').read_text(encoding='utf-8'))
    for rel,h in expected['sha256'].items():
        assert hashlib.sha256((ROOT/'data'/rel).read_bytes()).hexdigest() == h, rel
    with patch('socket.socket.connect',blocked), tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        from config.rag_settings import RAGSettings
        from rag.indexing import load_sample,SearchIndex,encoded
        from rag.runtime import build_pipeline
        copy=Path(tmp);shutil.copytree(ROOT/'data',copy/'data')
        records,units,manifest=load_sample(copy/'data/atlas.sqlite3',copy/'data/sample_selection.json')
        docs,indexed,revision=SearchIndex(copy/'data/atlas_search.sqlite3').read()
        assert len(records)==20 and len(docs)==49 and set(indexed)=={r['record_id'] for r in records}
        with sqlite3.connect((copy/'data/atlas.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
            assert db.execute('SELECT count(*) FROM canonical_records').fetchone()[0]==20
            required={u for r in records for u in r['source_position']['unit_ids']+r['context_unit_ids']}
            rows={u:json.loads(p) for u,p in db.execute('SELECT unit_id,payload FROM source_units')}
            assert set(rows)==required
            body={u for r in records for u in r['source_position']['unit_ids']}
            assert all(rows[u]['heading_level'] is not None for u in required-body)
            assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        provider=MockProvider()
        # Exercise the normal runtime/cache path, but replace the provider factory;
        # no real SDK call is possible and all external connections are blocked.
        with patch('rag.runtime.create_provider',return_value=provider):
            pipeline=build_pipeline(copy,RAGSettings(),allow_paid=True)
        vector=pipeline.retriever.vector_db
        stored=vector.collection.get(include=['embeddings','documents','metadatas'])
        assert len(stored['ids'])==49 and set(stored['ids'])=={d['document_id'] for d in docs}
        vector._validate_embeddings(stored['embeddings'],49)
        by_id={d['document_id']:d for d in docs}
        for i,did in enumerate(stored['ids']):
            d=by_id[did]
            assert stored['documents'][i]==d['search_text']
            assert stored['metadatas'][i]=={'record_id':d['record_id'],'kind':d['kind'],'index_revision':revision,
                'source_document':d['source_document'],'source_revision':d['source_revision'],
                'section_path':encoded(d['section_path']),'source_position':encoded(d['source_position'])}
            assert d['source_revision']==manifest['source_revision']
        query_cases=[('Большие глаза','atlas_4e62d32ded3acf57d7f17ca9'),
                     ('Физическая и психическая выносливость','atlas_5093dfc5b8aac70a0d956dbc'),
                     ('Фотографии результата','atlas_4e62d32ded3acf57d7f17ca9')]
        results=[]
        for query,rid in query_cases:
            for mode in ('fts','vector','hybrid'):
                response=pipeline.answer(query,mode=mode)
                assert response['status']=='answered',(query,mode,response['status'])
                assert rid in {s['record_id'] for s in response['sources']},(query,mode)
                for part in response['parts']:
                    for card in part['evidence']:
                        source=pipeline.policy.records[card['record_id']]
                        assert card['quote'] in source['original_text']
                        assert pipeline.policy.units[card['unit_id']]['text']==card['quote']
                results.append({'question':query,'mode':mode,'record_id':rid,'canonical_quotes_verified':True})
        calls=provider.calls
        with patch.object(pipeline.retriever,'search',side_effect=AssertionError('Cache hit must bypass retrieval')):
            cached=pipeline.answer('Большие глаза',mode='hybrid')
        assert provider.calls==calls and cached['status']=='answered'
        assert pipeline._source_snapshot['atlas_neurotypology_v15.docx'] is None
        assert vector.collection.count()==49
        report={'records':20,'source_units':len(rows),'documents':49,'vectors':49,'dimensions':1536,
                'collection':vector.collection.name,'source_revision':manifest['source_revision'],'index_revision':revision,
                'data_manifest_verified':True,'original_text_and_metadata_preserved':True,'docx_required':False,
                'search_checks':results,'cache_without_docx':'passed','api_calls':0,'embedding_calls':0,
                'new_search_documents':0,'reindexing':False,'network_blocked':True,'test_provider':'mock',
                'runtime_storage_opened_on_temporary_copy':True}
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/demo_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    verify()
