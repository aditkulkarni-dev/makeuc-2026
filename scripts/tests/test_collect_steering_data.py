import importlib.util
import json
import tempfile
import sys
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

spec = importlib.util.spec_from_file_location('collector', Path(__file__).resolve().parents[1] / 'collect_steering_data.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def extract(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            path.write_text(json.dumps(data))
            return collector.extract_prompts(path)

    def test_result_rows_deduplicate_original_prompts(self):
        rows = [{'original_prompt': 'same', 'message_id': 1, 'attack_prompt': 'ignore this', 'coefficient': a} for a in [0, .1, .5]]
        prompts = self.extract(rows)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]['prompt'], 'same')
        self.assertEqual(prompts[0]['source_ids'], [1])
        self.assertEqual(prompts[0]['prompt_id'], self.extract(['same'])[0]['prompt_id'])

    def test_normalized_dataset_round_trip(self):
        prompts = self.extract(['one', 'two'])
        self.assertEqual(self.extract({'prompts': prompts}), prompts)

    def test_missing_original_is_rejected(self):
        with self.assertRaises(ValueError):
            self.extract([{'attack_prompt': 'not an original prompt'}])

    def test_working_pair_preserves_exact_prompt_and_reference(self):
        prompt = 'question  suffix with trailing space '
        rows = [dict(row_index=147, message_id=0, suffix_id='suffix-id',
                     prompt_with_suffix=prompt, verified_success=True,
                     baseline_response='old response', published_score=.8)]
        extracted = self.extract(rows)
        self.assertEqual(extracted[0]['prompt'], prompt)
        self.assertEqual(extracted[0]['source_ids'], [147])
        self.assertTrue(extracted[0]['source_records'][0]['verified_success'])
        self.assertEqual(self.extract({'prompts': extracted}), extracted)

    def test_jsonl_baseline_keeps_suffix_variants_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'baseline.jsonl'
            path.write_text('\n'.join(json.dumps(dict(row_index=i, message_id=0, suffix_id=i,
                                                     prompt=f'question suffix {i}')) for i in range(2)))
            extracted = collector.extract_prompts(path)
        self.assertEqual(len(extracted), 2)
        self.assertEqual([p['source_ids'] for p in extracted], [[0], [1]])

    def test_summary_does_not_invent_success_labels(self):
        records = [dict(alpha=0, seconds=2, tokens=10, truncated=True)]
        summary = collector.summarize(records, {'alphas': [0, .1]}, [{}], False)
        self.assertIsNone(summary['by_alpha'][0]['jailbreak_success_rate'])
        self.assertIsNone(summary['by_alpha'][1]['mean_tokens'])
        self.assertEqual(summary['by_alpha'][0]['truncated_count'], 1)
        self.assertFalse(summary['complete'])

    def test_collect_and_resume_without_duplicate_generations(self):
        class FakeEngine:
            calls = 0

            def __init__(self, *args):
                pass

            def load(self):
                pass

            def generate(self, prompt, alpha, max_tokens):
                FakeEngine.calls += 1
                return dict(response='fixture', tokens=10, seconds=.01, truncated=False)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.json'
            source.write_text(json.dumps(['prompt one', 'prompt two']))
            (root / 'direction.pt').write_bytes(b'fixture direction')
            (root / 'direction_metadata.json').write_text('{"layer": 1}')
            args = ['collector', '--input', str(source), '--output', str(root / 'out'),
                    '--direction-dir', str(root), '--alphas', '0', '.1', '--run']
            with patch.object(sys, 'argv', args), patch.object(collector, 'SteeringEngine', FakeEngine), redirect_stdout(StringIO()):
                collector.main()
                initial_calls = FakeEngine.calls
                collector.main()
            records = (root / 'out/records.jsonl').read_text().splitlines()
            self.assertEqual(len(records), 4)
            self.assertEqual(FakeEngine.calls, initial_calls)
            self.assertTrue(json.loads((root / 'out/dashboard.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main()
