"""Run with python steering_dashboard/server.py [--demo]."""
import argparse
import json
import math
import threading
import uuid
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from engine import DEFAULT_DIRECTION, DemoEngine, SteeringEngine

STATIC = Path(__file__).parent / 'static'
COLLECTED_DATA = Path(__file__).resolve().parents[1] / 'results/experimental_steering_dataset'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.score_steering_data import classify_response, score_dataset


def validate(payload):
    prompt = payload.get('prompt')
    alphas = payload.get('alphas', [0, .1, .2, .25, .5])
    tokens = payload.get('max_tokens', 256)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
        raise ValueError('Enter a prompt between 1 and 16,000 characters.')
    if not isinstance(alphas, list) or not 1 <= len(alphas) <= 12:
        raise ValueError('Choose between 1 and 12 alpha values.')
    if any(isinstance(a, bool) or not isinstance(a, (float, int)) or not math.isfinite(a) or not -2 <= a <= 2 for a in alphas):
        raise ValueError('Alpha values must be finite numbers between -2 and 2.')
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 16 <= tokens <= 1024:
        raise ValueError('Token limit must be an integer between 16 and 1024.')
    # Always include an unsteered control using exactly the same prompt.
    return dict(prompt=prompt.strip(), alphas=list(dict.fromkeys([0.0, *alphas])), max_tokens=tokens)


class App:
    def __init__(self, engine, demo=False, model_id='google/gemma-2-2b-it', dataset_dir=COLLECTED_DATA):
        self.engine, self.demo, self.model_id = engine, demo, model_id
        self.dataset_dir = Path(dataset_dir)
        self.jobs = {}
        self.lock = threading.Lock()
        self.busy = False

    def start(self, config):
        with self.lock:
            if self.busy:
                raise RuntimeError('A sweep is already running. Wait for it to finish.')
            self.busy = True
            job = dict(id=uuid.uuid4().hex, status='loading', created_at=datetime.now(timezone.utc).isoformat(),
                       model=self.model_id, demo=self.demo, config=config, results=[], error=None)
            self.jobs[job['id']] = job
            # Keep bounded in-memory history. Exports are user-controlled.
            if len(self.jobs) > 30:
                del self.jobs[next(iter(self.jobs))]
            snapshot = json.loads(json.dumps(job))
        threading.Thread(target=self.run, args=(job,), daemon=True).start()
        return snapshot

    def run(self, job):
        try:
            self.engine.load()
            with self.lock:
                job['status'] = 'running'
            for alpha in job['config']['alphas']:
                result = self.engine.generate(job['config']['prompt'], alpha, job['config']['max_tokens'])
                with self.lock:
                    job['results'].append(dict(alpha=alpha, **result))
            with self.lock:
                job['status'] = 'complete'
        except Exception as exc:
            with self.lock:
                job['status'], job['error'] = 'failed', str(exc)
        finally:
            with self.lock:
                self.busy = False


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, mime='application/json'):
            body = json.dumps(data, allow_nan=False).encode() if mime == 'application/json' else data
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ('/api/collected-results', '/api/jailbreak-comparison'):
                try:
                    dataset = json.loads((app.dataset_dir / 'dashboard.json').read_text())
                    if path == '/api/jailbreak-comparison':
                        return self.reply(200, score_dataset(dataset))
                    for row in dataset['results']:
                        row['refusal_evaluation'] = classify_response(row.get('response'))
                    report_path = app.dataset_dir / 'collection_report.json'
                    dataset['collection_report'] = json.loads(report_path.read_text()) if report_path.exists() else None
                    return self.reply(200, dataset)
                except FileNotFoundError:
                    return self.reply(404, {'error': 'No collected dataset found. Run scripts/collect_steering_data.py --run first.'})
                except (ValueError, OSError, KeyError, TypeError):
                    return self.reply(503, {'error': 'The saved dataset could not be read. Try again after collection finishes.'})
            if path == '/api/status':
                with app.lock:
                    return self.reply(200, dict(demo=app.demo, model=app.model_id, busy=app.busy,
                                               latest_job_id=next(reversed(app.jobs), None)))
            if path.startswith('/api/runs/'):
                with app.lock:
                    job = app.jobs.get(path.rsplit('/', 1)[-1])
                    return self.reply(200 if job else 404, job or {'error': 'Run not found.'})
            files = {'/': ('index.html', 'text/html; charset=utf-8'),
                     '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                     '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if path not in files:
                return self.reply(404, {'error': 'Not found.'})
            name, mime = files[path]
            self.reply(200, (STATIC / name).read_bytes(), mime)

        def do_POST(self):
            # Only same-origin browser requests; no permissive CORS or cross-site form submissions.
            if self.headers.get('Origin') and self.headers['Origin'] != 'http://' + self.headers.get('Host', ''):
                return self.reply(403, {'error': 'Cross-origin requests are disabled.'})
            if self.path != '/api/runs':
                return self.reply(404, {'error': 'Not found.'})
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 100000:
                    raise ValueError('Invalid request size.')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Expected a JSON object.')
                config = validate(payload)
                self.reply(202, app.start(config))
            except (ValueError, TypeError) as exc:
                self.reply(400, {'error': str(exc)})
            except RuntimeError as exc:
                self.reply(409, {'error': str(exc)})
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model', default='google/gemma-2-2b-it')
    parser.add_argument('--direction-dir', type=Path, default=DEFAULT_DIRECTION)
    args = parser.parse_args()
    app = App(DemoEngine() if args.demo else SteeringEngine(args.model, args.direction_dir), args.demo, args.model)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(app))
    print(f'Steering Lab: http://127.0.0.1:{args.port} ({"DEMO" if args.demo else "live model"})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
