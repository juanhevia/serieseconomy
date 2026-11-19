"""Integración de la aplicación HTML con pesos fijos e histórico temporal."""
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from oilprice.web import Application, handler_for


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / 'history.csv'
        cls.app = Application(history_path=cls.path)
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(cls.app))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def request(self, path, body=None, headers=None):
        headers = {'X-Brent-Client': '1', **(headers or {})}
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body).encode()
        req = Request(self.url + path, data=body, headers=headers)
        try:
            response = urlopen(req, timeout=90)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.read()

    def test_html_assets_and_local_access(self):
        for path, marker in [('/', b'forecast-form'), ('/app.js', b'/api/forecast'), ('/styles.css', b'.sidebar')]:
            status, body = self.request(path)
            self.assertEqual(status, 200)
            self.assertIn(marker, body)
        self.assertEqual(self.request('/../app.py')[0], 404)
        self.assertEqual(self.request('/api/state', headers={'Host': 'foreign.example'})[0], 403)
        self.assertEqual(self.request('/api/save', {}, {'Origin': 'https://foreign.example'})[0], 403)

    def test_predictions_and_evaluation(self):
        for horizon, architecture in [(5, 'RNN'), (63, 'LSTM'), (252, 'RNN')]:
            status, body = self.request('/api/forecast', {'horizon': horizon, 'origin': '2026-11-06'})
            self.assertEqual(status, 200, body)
            forecast = json.loads(body)
            self.assertEqual(forecast['architecture'], architecture)
            self.assertLess(forecast['last_observation_date'], forecast['origin_date'])
            status, body = self.request(f'/api/evaluation/{horizon}')
            self.assertEqual(status, 200)
            evaluation = json.loads(body)
            self.assertEqual(evaluation['selected'], architecture)
            self.assertEqual(len(evaluation['metrics']), 4)
            self.assertGreater(len(evaluation['series']), 700)
        for payload in [[], {'horizon': True}, {'horizon': 7}, {'horizon': 5, 'origin': '2099-01-01'}]:
            self.assertEqual(self.request('/api/forecast', payload)[0], 400)

    def test_upload_conflict_save_and_restore(self):
        content = b'Date,brent\n2026-11-06,110\n'
        status, body = self.request('/api/upload', content)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['added'], 1)
        self.assertEqual(json.loads(self.request('/api/upload', content)[1])['added'], 0)
        self.assertEqual(self.request('/api/upload', b'Date,brent\n2026-11-06,999\n')[0], 400)
        self.assertEqual(self.app.history.loc['2026-11-06'], 110)
        self.assertEqual(self.request('/api/save', {})[0], 200)
        restored = Application(history_path=self.path)
        self.assertEqual(restored.history.loc['2026-11-06'], 110)
        self.assertFalse(restored.changed)
        self.assertIn(b'2026-11-06,110', self.request('/api/history.csv')[1])


if __name__ == '__main__':
    unittest.main()
