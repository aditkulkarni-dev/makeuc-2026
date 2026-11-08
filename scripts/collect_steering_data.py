"""Extract a fixed prompt set, estimate runtime, and collect paired alpha sweeps.

Default: extract and benchmark only. Add --run to collect responses.
Source prompts are preserved exactly; no new suffix is constructed or optimized.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTAL_DATA = ROOT / 'scripts/data/experimental-5b9aa51'
sys.path.insert(0, str(ROOT / 'steering_dashboard'))
from engine import DEFAULT_DIRECTION, SteeringEngine


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def extract_prompts(path):
    """Accept result arrays, dashboard exports, or a {prompts: [...]} dataset."""
    raw = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
           if path.suffix == '.jsonl' else json.loads(path.read_text()))
    rows = raw.get('prompts', raw.get('results', [])) if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError('Input must contain a nonempty prompt/result array.')
    prompts = {}
    for row in rows:
        text = row if isinstance(row, str) else row.get('prompt_with_suffix', row.get('original_prompt', row.get('prompt')))
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Each row needs prompt_with_suffix, original_prompt, or prompt (or must be a string).')
        key = digest(text)[:16]
        if key not in prompts:
            prompts[key] = dict(prompt_id=key, prompt=text, source_ids=[])
        if isinstance(row, dict):
            metadata = {k: row[k] for k in ('row_index', 'message_id', 'suffix_id', 'published_score',
                        'published_category', 'verified_success', 'baseline_response') if k in row}
            references = row.get('source_records', [metadata] if metadata else [])
            if references:
                existing = prompts[key].setdefault('source_records', [])
                for reference in references:
                    if reference not in existing:
                        existing.append(reference)
            source_ids = row.get('source_ids')
            if source_ids is None:
                source_id = row.get('row_index', row.get('message_id', row.get('prompt_id', row.get('test_id'))))
                source_ids = [] if source_id is None else [source_id]
            if not isinstance(source_ids, list):
                raise ValueError('source_ids must be a list.')
            for source_id in source_ids:
                if source_id not in prompts[key]['source_ids']:
                    prompts[key]['source_ids'].append(source_id)
    return list(prompts.values())


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def summarize(records, config, prompts, complete):
    summaries = []
    for alpha in config['alphas']:
        group = [r for r in records if r['alpha'] == alpha]
        summaries.append(dict(alpha=alpha, responses=len(group),
            mean_generation_seconds=statistics.mean(r['seconds'] for r in group) if group else None,
            mean_tokens=statistics.mean(r['tokens'] for r in group) if group else None,
            truncated_count=sum(r['truncated'] for r in group),
            jailbreak_success_rate=None))
    return dict(schema_version=1, complete=complete, config=config, prompt_count=len(prompts),
                response_count=len(records), by_alpha=summaries, results=records,
                evaluation=dict(status='not_connected', jailbreak_success_rate=None,
                                note='Responses and truncation are not success labels. External published scores are not reused.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=EXPERIMENTAL_DATA / 'results/working_pairs.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/experimental_steering_dataset')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0, .1, .2, .25, .5])
    parser.add_argument('--max-tokens', type=int, default=160)
    parser.add_argument('--minutes', type=float, default=30)
    parser.add_argument('--limit', type=int, help='Use first N unique prompts, in source order.')
    parser.add_argument('--model', default='google/gemma-2-2b-it')
    parser.add_argument('--direction-dir', type=Path, default=DEFAULT_DIRECTION)
    parser.add_argument('--extract-only', action='store_true', help='No model loading or inference.')
    parser.add_argument('--run', action='store_true', help='Collect after printing the benchmark estimate.')
    args = parser.parse_args()
    if (not args.alphas or any(not math.isfinite(a) or abs(a) > 2 for a in args.alphas)
            or not 16 <= args.max_tokens <= 1024 or not math.isfinite(args.minutes) or args.minutes <= 0
            or (args.limit is not None and args.limit < 1)):
        parser.error('Use finite alphas within ±2, 16–1024 tokens, and positive minutes/limit.')
    started = time.perf_counter()
    prompts = extract_prompts(args.input.resolve())
    if args.limit:
        prompts = prompts[:args.limit]
    alphas = list(dict.fromkeys([0.0, *args.alphas]))
    if args.extract_only:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / 'prompts.json', {'prompts': prompts})
        print(f'Extracted {len(prompts)} unique source prompts without loading a model or direction.', flush=True)
        print(f'Prompt dataset: {args.output / "prompts.json"}', flush=True)
        return
    config = dict(model=args.model, alphas=alphas, max_tokens=args.max_tokens,
                  direction_sha256=hashlib.sha256((args.direction_dir / 'direction.pt').read_bytes()).hexdigest(),
                  direction_metadata=json.loads((args.direction_dir / 'direction_metadata.json').read_text()),
                  decoding='greedy', intervention='layer_input + alpha * direction',
                  prompt_ids=[p['prompt_id'] for p in prompts])
    config_id = digest(config)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    records_path = args.output / 'records.jsonl'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['config_id'] != config_id:
        raise ValueError('Output belongs to a different configuration. Choose a new --output directory.')
    manifest = dict(config_id=config_id, config=config,
                    source=dict(path=str(args.input.resolve()), sha256=hashlib.sha256(args.input.read_bytes()).hexdigest()),
                    created_at=datetime.now(timezone.utc).isoformat())
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    atomic_json(args.output / 'prompts.json', {'prompts': prompts})
    records = []
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            row = json.loads(line)
            if row['config_id'] != config_id:
                raise ValueError('Checkpoint configuration mismatch.')
            records.append(row)
    done = {(r['prompt_id'], r['alpha']) for r in records}
    remaining = sum((p['prompt_id'], a) not in done for p in prompts for a in alphas)
    print(f'Extracted {len(prompts)} unique source prompts; {len(alphas)} alphas; {remaining} responses remaining.', flush=True)
    if remaining == 0:
        atomic_json(args.output / 'dashboard.json', summarize(records, config, prompts, True))
        print('All requested pairs are already saved; no model inference needed.', flush=True)
        return
    engine = SteeringEngine(args.model, args.direction_dir.resolve())
    load_started = time.perf_counter()
    engine.load()
    load_seconds = time.perf_counter() - load_started
    # Harmless fixed workload for throughput only; benchmark responses are never dataset rows.
    calibration_prompt = 'Explain how rainbows form in detail, including refraction, reflection, and dispersion.'
    engine.generate(calibration_prompt, 0, 16)  # warm up kernels
    samples = [engine.generate(calibration_prompt, alpha, 64) for alpha in (0, .1, .5)]
    seconds_per_token = max(r['seconds'] / max(1, r['tokens']) for r in samples)
    # Full token budget + 50% margin. Real prompts/early stopping can vary substantially.
    per_response = seconds_per_token * args.max_tokens * 1.5 + .5
    estimated_seconds = remaining * per_response
    plan = dict(remaining_responses=remaining, model_load_seconds=round(load_seconds, 2),
                benchmark_seconds_per_token=seconds_per_token, benchmark_samples=[{k:v for k,v in r.items() if k != 'response'} for r in samples],
                estimated_remaining_minutes=round(estimated_seconds / 60, 2),
                estimated_total_minutes=round((time.perf_counter() - started + estimated_seconds) / 60, 2),
                budget_minutes=args.minutes, caveat='Approximation from a benign 64-token workload with 50% margin; not a hard deadline.')
    atomic_json(args.output / 'runtime_estimate.json', plan)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        print('Estimate only. Add --run to collect, or --extract-only to skip inference.', flush=True)
        return
    deadline = started + args.minutes * 60
    try:
        with records_path.open('a') as output:
            for prompt in prompts:
                missing = [a for a in alphas if (prompt['prompt_id'], a) not in done]
                # Start a prompt only if its remaining alpha group should fit. Keeps comparisons paired.
                if missing and time.perf_counter() + len(missing) * per_response > deadline:
                    print('Budget reached: not starting another prompt group. Rerun to resume.', flush=True)
                    break
                for alpha in missing:
                    result = engine.generate(prompt['prompt'], alpha, args.max_tokens)
                    row = dict(config_id=config_id, prompt_id=prompt['prompt_id'], prompt=prompt['prompt'],
                               source_ids=prompt['source_ids'], source_records=prompt.get('source_records', []),
                               alpha=alpha, manual_rating=None, **result)
                    output.write(json.dumps(row, allow_nan=False) + '\n')
                    output.flush()
                    records.append(row)
                    done.add((prompt['prompt_id'], alpha))
                    atomic_json(args.output / 'dashboard.json', summarize(records, config, prompts, False))
                    print(f'{len(records)}/{len(prompts) * len(alphas)} saved · alpha={alpha:g} · {result["seconds"]:.2f}s', flush=True)
    finally:
        complete = all((p['prompt_id'], a) in done for p in prompts for a in alphas)
        atomic_json(args.output / 'dashboard.json', summarize(records, config, prompts, complete))
        atomic_json(args.output / 'collection_report.json', dict(
            invocation_seconds=round(time.perf_counter() - started, 2),
            total_saved_generation_seconds=round(sum(r['seconds'] for r in records), 2),
            complete=complete, saved_responses=len(records), expected_responses=len(prompts) * len(alphas)))
        print(f'Dataset {"complete" if complete else "partial (resumable)"}: {args.output / "dashboard.json"}', flush=True)


if __name__ == '__main__':
    main()
