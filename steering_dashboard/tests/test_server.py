import json
import sys
import threading
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import App, handler_for, validate


class FakeEngine:
    def load(self):
        pass

    def generate(self, prompt, alpha, max_tokens):
        return dict(response=f'{prompt}:{alpha}', tokens=3, seconds=0.01, truncated=False)


class Tests(unittest.TestCase):
    def test_collected_results_without_model_loading(self):
        class NoInference:
            def load(self):
                raise AssertionError('Viewing saved results must not load a model.')

        with tempfile.TemporaryDirectory() as directory:
            app = App(NoInference(), dataset_dir=directory)
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(app))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            url = f'http://127.0.0.1:{server.server_port}/api/collected-results'
            try:
                with self.assertRaises(HTTPError) as error:
                    urlopen(url)
                self.assertEqual(error.exception.code, 404)
                data = {'complete': True, 'results': [{'alpha': 0, 'response': '<script>example</script>'}]}
                path = Path(directory) / 'dashboard.json'
                path.write_text(json.dumps(data))
                (Path(directory) / 'collection_report.json').write_text('{"invocation_seconds": 493.2}')
                with urlopen(url) as response:
                    loaded = json.load(response)
                self.assertEqual(loaded['results'][0]['response'], data['results'][0]['response'])
                self.assertEqual(loaded['results'][0]['refusal_evaluation']['label'], 'non_refusal')
                self.assertEqual(loaded['collection_report']['invocation_seconds'], 493.2)
                self.assertFalse(app.busy)
                self.assertFalse(app.jobs)
                path.write_text(json.dumps(dict(config=dict(model='fixture', max_tokens=160), results=[
                    dict(prompt_id='a', prompt='same', alpha=a, response=r)
                    for a, r in [(0, 'Sure'), (.1, 'I cannot help'), (.2, 'Sure')]])))
                with urlopen(url.replace('collected-results', 'jailbreak-comparison')) as response:
                    comparison = json.load(response)
                self.assertEqual([r['estimated_success_rate'] for r in comparison['by_alpha']], [1, 0, 1])
                self.assertFalse(app.busy)
                path.write_text('invalid')
                with self.assertRaises(HTTPError) as error:
                    urlopen(url)
                self.assertEqual(error.exception.code, 503)
            finally:
                server.shutdown()
                server.server_close()

    def test_validation(self):
        self.assertEqual(validate({'prompt': ' hello ', 'alphas': [.2, .2]})['alphas'], [0, .2])
        for payload in [{'prompt': ''}, {'prompt': 'x', 'alphas': [float('nan')]},
                        {'prompt': 'x', 'alphas': [True]}, {'prompt': 'x', 'max_tokens': 0},
                        {'prompt': 'x', 'alphas': [3]}, {'prompt': 'x', 'alphas': []}]:
            with self.assertRaises(ValueError):
                validate(payload)

    def test_partial_failure_releases_worker(self):
        class Failing(FakeEngine):
            def generate(self, prompt, alpha, max_tokens):
                if alpha:
                    raise RuntimeError('generation failed')
                return super().generate(prompt, alpha, max_tokens)
        app = App(Failing())
        job = dict(config=validate({'prompt': 'hello', 'alphas': [.1]}), results=[])
        app.busy = True
        app.run(job)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(len(job['results']), 1)
        self.assertFalse(app.busy)

    def test_exclusive_sweep(self):
        app = App(FakeEngine())
        app.busy = True
        with self.assertRaises(RuntimeError):
            app.start(validate({'prompt': 'hello'}))

    def test_http_sweep(self):
        app = App(FakeEngine(), demo=True)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(app))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base) as response:
                self.assertIn(b'Response spectrum', response.read())
            req = Request(base + '/api/runs', data=json.dumps({'prompt': '<script>hello</script>', 'alphas': [.1, .2]}).encode(), headers={'Content-Type': 'application/json'})
            with urlopen(req) as response:
                self.assertEqual(response.status, 202)
                job = json.load(response)
            for _ in range(100):
                with urlopen(base + '/api/runs/' + job['id']) as response:
                    job = json.load(response)
                if job['status'] == 'complete':
                    break
                time.sleep(.01)
            self.assertEqual(job['status'], 'complete')
            self.assertEqual([r['alpha'] for r in job['results']], [0, .1, .2])
            bad = Request(base + '/api/runs', data=b'{}', headers={'Origin': 'https://other.example'})
            with self.assertRaises(HTTPError) as error:
                urlopen(bad)
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
