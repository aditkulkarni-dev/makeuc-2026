# Collecting a steering comparison dataset

## Source data

The updated `experimental` branch at commit
`5b9aa51e5dfae56a19e954cd5e791e6fd397959c` contains the baseline evaluator, manual
review script, 200 baseline response rows, 30 saved manual reviews, and 27 exported
working pairs. A pinned copy lives in `scripts/data/experimental-5b9aa51/`.
`SOURCE.json` records upstream locations. The remote scripts were read, not executed.

Source: https://github.com/aditkulkarni-dev/makeuc-2026/tree/5b9aa51e5dfae56a19e954cd5e791e6fd397959c/gemma-gcg

All 200 baseline rows have **one message ID and 200 distinct combined prompt/suffix
strings**. The approved subset has 27 variants of that same base question. This
is a comparison across suffix variants of one question, not 27 different questions.

The friend's workflow generates a Gemma baseline with greedy decoding and a
160-token cap, filters refusal prefixes and published scores below 0.3, then asks
for human review. `working_pairs.json` exports the rows marked `y` after filtering.
Published scores come from the upstream dataset; they do not score new generations.

The collector now defaults to all 27 working pairs. It preserves each combined
prompt exactly, including whitespace, without constructing or optimizing suffixes.
It retains row index, message/suffix IDs, source review, published score, and source
baseline response as reference metadata. Every selected combined prompt is rerun
at each alpha, including alpha 0, under a single controlled inference configuration.

## Run

From the repository root:

```bash
# Extract the approved prompt set without loading or calling the model.
.venv/bin/python scripts/collect_steering_data.py --extract-only

# Benchmark and estimate runtime without collecting experiment responses.
.venv/bin/python scripts/collect_steering_data.py

# Collect all 27 variants at 0, 0.1, 0.2, 0.25, and 0.5 (135 responses).
.venv/bin/python scripts/collect_steering_data.py --run
```

Default generation limit: 160 new tokens, matching the friend's cap. The collector
reuses `steering_dashboard/engine.py`: greedy decoding, the local direction and its
metadata-selected layer, and the same prompt at every alpha. Alpha 0 is always
included and has no hook. GPU inference in this engine uses float16, while the
friend's baseline evaluator used bfloat16. Therefore the regenerated alpha-0 output
is the control for steering comparisons; source baseline responses are references,
not interchangeable controls. Do not compare the old 256-token pilot as if it were
the same experiment.

The local benchmark estimated **15.9 minutes** for 135 responses at 160 tokens,
including a 50% margin. Prompt length, GPU load, and early stopping affect actual
timing. `runtime_estimate.json` stores calibration measurements.

The completed initial run saved all 135 responses in **8.22 minutes wall time**
(**8.04 minutes generation time**).
84 responses reached the 160-token cap and are flagged as truncated.

`--minutes 30` limits when another prompt group can start based on the estimate.
It is not a hard termination deadline. An in-progress generation finishes so it
can be saved. Rerunning the same command resumes saved prompt/alpha pairs; completed
datasets skip model loading entirely. Use a new output directory if settings change.

To extract every baseline variant, including unreviewed and unsuccessful ones:

```bash
.venv/bin/python scripts/collect_steering_data.py \
  --input scripts/data/experimental-5b9aa51/results/baseline_results.jsonl \
  --output results/all_baseline_variants --extract-only
```

Collecting all 200 variants with five alphas means 1,000 responses and is unlikely
to fit the 30-minute target. Use `--limit` to select a bounded source-order subset
or allow resumable runs. The default 27-pair subset fits the target but was selected
for earlier successes. Its eventual success rate would be **conditional on that
selection**, not an overall jailbreak success rate across all prompts or attacks.

Accepted inputs: JSON or JSONL rows containing `prompt_with_suffix`, `original_prompt`,
or `prompt`; arrays of strings; or an object with a `prompts` or `results` array.
`prompts.json` can itself be passed back as `--input`. Controls include
`--alphas 0 0.1 0.2`, `--max-tokens 256`, `--limit 5`, `--model`, and `--direction-dir`.
A model needs a direction generated for that exact model.

## Outputs

Default directory: `results/experimental_steering_dataset/`.

- `manifest.json`: configuration ID, input hash, model, direction hash and layer.
- `prompts.json`: exact source prompts with stable content-based IDs and references.
- `runtime_estimate.json`: benchmark timings and estimated runtime.
- `collection_report.json`: completion, response count, invocation and generation time.
- `records.jsonl`: flushed per-response checkpoints for resume.
- `dashboard.json`: response records and summary metrics grouped by alpha.

Each response includes `prompt_id`, `prompt`, `source_ids`, `source_records`, `alpha`,
`response`, `tokens`, `seconds`, `truncated`, and `manual_rating: null`. Source reviews
apply only to the earlier baseline; they are not copied into new response ratings.
Summary metrics are response count, average token count, average generation time,
and truncation count. `complete` identifies whether all requested pairs were saved.
Success-rate fields remain null until the **new responses** are evaluated. Neither
refusal-prefix matching nor response length alone establishes jailbreak success.
The website now reads these files through `/api/collected-results`. Click
**View collected results** to browse the saved experiment without new inference.

The earlier pilot remains separately in `results/steering_dataset/`: 10 original
prompts from the old local GCG results, five alphas, and 256-token outputs. It is
preserved for reference and is not mixed into the friend's experimental dataset.

## Tests

```bash
.venv/bin/python -m unittest discover -s scripts/tests -v
```

## Compute the success-rate comparison

```bash
.venv/bin/python scripts/score_steering_data.py
```

This scores the already-generated responses, saves `jailbreak_comparison.json`,
and requires no new model inference. It copies the friend's exact refusal-prefix
filter and counts non-refusals as **estimated** successes on a shared paired set
at alpha 0, 0.1, and 0.2. Current counts: 27/27, 15/27, and 7/27. The main dashboard
shows these as a three-bar chart. Human verification and the upstream published
score threshold from the friend's full review workflow are not simulated or reused.
See the scorer's output for the exact rule, prefixes, exclusions, and per-response
classifications. Use `--input` to score a newly collected dataset.
