"""运行：python -m unittest model_management.test_batch_delete。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from model_management import api
from model_management.store import ModelStore


class BatchDeleteTest(unittest.TestCase):
    def test_delete_all_includes_more_than_one_page(self):
        app = FastAPI()
        app.include_router(api.router)
        with patch.object(api, 'store') as store:
            ids = [f'm{index}' for index in range(25)]
            store.list_ids.return_value = ids
            store.delete.side_effect = lambda model_id: {'id': model_id, 'is_deleted': True}
            result = TestClient(app).post('/model-registry/delete-all').json()
            self.assertEqual([item['id'] for item in result['deleted']], ids)
            self.assertEqual(store.delete.call_count, 25)

    def test_soft_delete_partial_failure_and_validation(self):
        app = FastAPI()
        app.include_router(api.router)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ModelStore(root / 'registry.sqlite3')
            with patch.object(api, 'store', store), patch.object(store, '_refresh_cache'):
                for model_id in ('a', 'b', 'c'):
                    path = root / model_id
                    path.mkdir()
                    (path / 'weights').write_text('keep')
                    job = SimpleNamespace(id=model_id, model='rf', purpose='news',
                                          finished_at='2026-11-20', source='train', test_source='test')
                    store.register(job=job, model_path=path, params={}, evaluation={}, label_mapping={'0': 'test'})
                client = TestClient(app)
                response = client.post('/model-registry/batch-delete', json={'ids': ['a', 'missing', 'b', 'a']})
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertEqual([item['id'] for item in result['deleted']], ['a', 'b'])
                self.assertEqual([item['id'] for item in result['failed']], ['missing'])
                self.assertEqual([item['id'] for item in store.list()['items']], ['c'])
                for model_id in ('a', 'b'):
                    self.assertEqual((root / f'_del_{model_id}' / 'weights').read_text(), 'keep')
                with patch.object(store, 'delete', side_effect=OSError('directory locked')):
                    result = client.post('/model-registry/batch-delete', json={'ids': ['c']}).json()
                    self.assertEqual(result['deleted'], [])
                    self.assertEqual(result['failed'][0]['id'], 'c')
                for ids in ([], [' '], ['x'] * 101):
                    self.assertEqual(client.post('/model-registry/batch-delete', json={'ids': ids}).status_code, 422)
                result = client.post('/model-registry/delete-all').json()
                self.assertEqual([item['id'] for item in result['deleted']], ['c'])
                self.assertEqual(result['failed'], [])
                self.assertEqual(store.list_ids(), [])
                self.assertEqual((root / '_del_c' / 'weights').read_text(), 'keep')
                self.assertEqual(client.post('/model-registry/delete-all').json(), {'deleted': [], 'failed': []})


if __name__ == '__main__':
    unittest.main()
