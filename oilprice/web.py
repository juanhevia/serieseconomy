"""Servidor local para la interfaz HTML. Sin dependencias web ni entrenamiento."""
import argparse
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import urlparse
import pandas as pd
from oilprice.inference import (forecast, history_csv, latest_origin, load_bundle,
                                merge_history, read_update, save_history, today)

ROOT = Path(__file__).resolve().parents[1]

class Application:
    def __init__(self, root=ROOT, history_path=None):
        self.root = Path(root)
        self.folder = self.root / 'artifacts/brent_v1'
        self.bundle, self.history = load_bundle(self.folder, self.root)
        self.history_path = Path(history_path) if history_path else self.root / 'data/inference/history.csv'
        if self.history_path.exists():
            self.history, _ = merge_history(self.history, read_update(self.history_path.read_bytes()))
        self.changed = False
        self.lock = threading.RLock()
        self.metrics = pd.read_csv(self.folder / 'metrics_test.csv')
        self.predictions = pd.read_csv(self.folder / 'predictions_test.csv')

    def state(self):
        observed = self.history.dropna()
        lag = self.bundle['models']['5']['metadata']['preparation_config']['publication_lag_bdays']
        return {'today': str(today().date()), 'origin': str(latest_origin(observed, lag).date()),
                'last_date': str(observed.index[-1].date()), 'observations': len(observed), 'changed': self.changed,
                'history': [[str(d.date()), float(p)] for d, p in observed.items()],
                'models': {h: {'architecture': m['architecture'], 'coverage': m['range']['test_coverage']} for h, m in self.bundle['models'].items()}}

    def predict(self, data):
        if type(data.get('horizon')) is not int or data['horizon'] not in (5, 63, 252):
            raise ValueError('Elige un horizonte de 5, 63 o 252 días.')
        value = data.get('origin')
        if not isinstance(value, str) or len(value) != 10:
            raise ValueError('Elige una fecha de predicción válida.')
        date.fromisoformat(value)
        return forecast(self.history, value, data['horizon'], self.bundle, self.folder)

    def update(self, content):
        merged, count = merge_history(self.history, read_update(content))
        self.history = merged
        self.changed = self.changed or count > 0
        return {'added': count, 'state': self.state()}

    def save(self):
        save_history(self.history, self.history_path)
        self.changed = False
        return {'saved': True}

    def evaluation(self, horizon):
        if horizon not in (5, 63, 252):
            raise ValueError('Horizonte no válido.')
        selected = self.bundle['models'][str(horizon)]['architecture']
        frame = self.predictions[(self.predictions.horizon_bdays == horizon) & (self.predictions.architecture == selected)]
        return {'selected': selected, 'metrics': self.metrics[self.metrics.horizon_bdays == horizon].to_dict('records'),
                'series': frame[['target_date', 'actual_usd', 'predicted_usd']].values.tolist()}


def handler_for(application):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, content, mime='application/json; charset=utf-8'):
            if not isinstance(content, bytes):
                content = json.dumps(content, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(content)

        def allowed_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

        def do_GET(self):
            if not self.allowed_host():
                self.reply(403, {'error': 'Acceso local únicamente.'})
                return
            path = urlparse(self.path).path
            try:
                with application.lock:
                    if path == '/api/state':
                        self.reply(200, application.state())
                    elif path.startswith('/api/evaluation/'):
                        self.reply(200, application.evaluation(int(path.rsplit('/', 1)[-1])))
                    elif path == '/api/history.csv':
                        self.reply(200, history_csv(application.history), 'text/csv; charset=utf-8')
                    elif path == '/api/metrics.csv':
                        self.reply(200, (application.folder / 'metrics_test.csv').read_bytes(), 'text/csv; charset=utf-8')
                    elif path in ('/', '/index.html', '/styles.css', '/app.js', '/favicon.svg'):
                        name = 'index.html' if path == '/' else path[1:]
                        mime = mimetypes.guess_type(name)[0] or 'application/octet-stream'
                        self.reply(200, (application.root / 'web' / name).read_bytes(), mime + '; charset=utf-8')
                    else:
                        self.reply(404, {'error': 'Página no encontrada.'})
            except (ValueError, OSError) as exc:
                self.reply(400, {'error': str(exc)})

        def do_POST(self):
            origin = self.headers.get('Origin')
            allowed = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
            if not self.allowed_host() or self.headers.get('X-Brent-Client') != '1' or (origin and origin not in allowed):
                self.reply(403, {'error': 'Petición no autorizada desde otra página.'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 <= size <= 5_000_000:
                    raise ValueError('El archivo supera el límite de 5 MB.')
                content = self.rfile.read(size)
                path = urlparse(self.path).path
                with application.lock:
                    if path == '/api/forecast':
                        payload = json.loads(content)
                        if not isinstance(payload, dict):
                            raise ValueError('Petición de predicción no válida.')
                        result = application.predict(payload)
                    elif path == '/api/upload':
                        result = application.update(content)
                    elif path == '/api/save':
                        result = application.save()
                    else:
                        self.reply(404, {'error': 'Acción no encontrada.'})
                        return
                self.reply(200, result)
            except (ValueError, OSError, RuntimeError) as exc:
                self.reply(400, {'error': str(exc)})

        def log_message(self, format, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8502)
    parser.add_argument('--history', type=Path, help='Ruta alternativa del histórico local')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(Application(history_path=args.history)))
    print(f'Aplicación HTML lista en http://127.0.0.1:{server.server_port} · Ctrl+C para detener', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
