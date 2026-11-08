# Steering Lab

A local dashboard for the activation steering implemented in `scripts/steering.py`.
Everything for this tool lives in this folder; the original script is unchanged.

From the repository root, run:

```bash
.venv/bin/python steering_dashboard/server.py
```

Open **http://127.0.0.1:8000**. Enter a prompt and run a sweep with the button or
**Ctrl+Enter**. Set the two alpha slider handles and choose 2–12 evenly spaced
points, including both endpoints. Defaults are −0.5 to 0.5 with 5 points. For
example, −0.5 to 0.5 with 3 points yields `-0.5, 0, 0.5`. The server always adds
an alpha 0 control if the selected points do not include it; the preview indicates
this extra generation. Either endpoint beyond ±0.5 turns the bar red with a brief
shake on crossing the threshold (respects reduced-motion preferences).

The model dropdown previews Gemma 2, Llama 3.1, and Qwen 2.5. It is UI-only;
inference and exported model metadata still use the active server model.
The server loads the model once, when the first sweep starts. Completed responses
appear as the sweep progresses. Refreshing the page reconnects to the latest run.

To try the interface without model access, GPU memory, or additional dependencies:

```bash
python3 steering_dashboard/server.py --demo
```

Demo responses are explicitly labeled placeholders and have no evaluation value.
Use another port with `--port 8001` if needed. The server binds to localhost.

## Model setup

The existing repository virtual environment already has PyTorch and Transformers.
For a new environment, install `steering_dashboard/requirements.txt`, choosing a
PyTorch build suitable for your hardware. Gemma requires access to
`google/gemma-2-2b-it` on Hugging Face and a locally authenticated account (`hf auth login`).
First inference may download model weights if they are not cached. A GPU is
recommended; CPU inference uses float32 and is considerably slower.

The default direction directory is resolved relative to the repository, regardless
of the shell's current directory:

```
refusal_direction_gen/refusal_direction/pipeline/runs/gemma-2-2b-it/
  direction.pt
  direction_metadata.json
```

It must contain the direction and its matching layer metadata. A root-level
`direction.pt` alone is insufficient. Override with `--direction-dir /absolute/path`.
`--model` can select a compatible model, but the direction must have been produced
for that model; a matching hidden size alone does not prove compatibility.

## Experiment behavior

Each response uses the same prompt, chat template, greedy decoding, and token
limit. Nonzero alpha installs a pre-hook at the metadata-selected transformer
layer and adds `alpha * direction` to every incoming token activation, exactly
as in the script. Alpha 0 installs no hook. Hooks are removed in `finally`, and
only one sweep runs at a time, avoiding model-state contamination. Unlike the
script, no attack prompt or suffix is hardcoded or automatically appended.

Cards show elapsed generation time, generated-token count, and whether the token
limit was reached. Timing excludes model loading and prompt tokenization. Token
counts can include a terminating special token. Smaller output limits can make a
response incomplete; they are not quality scores.

Rate cards as useful/coherent, refusal, poor/incoherent, or mixed/unsure. Borders
reflect your assessment. Ratings survive refresh within the same browser tab.
The ±0.5 quality warning reflects your observations, not an automatic
quality verdict. A refusal label alone does not measure jailbreak success or response quality.

Export JSON includes the prompt, model, alpha settings, responses, timing, manual
ratings, demo flag, and an `evaluation` object reserved for the future evaluator:

```json
{"jailbreak_success_rate": null, "status": "not_connected"}
```

Your friend's statistics integration can replace this placeholder and the pending
dashboard tile after defining success labels, evaluation criteria, and the dataset
denominator. No fabricated success statistics are displayed now.

Runs live in server memory (last 30) and are lost when it stops. Download JSON to
keep an experiment. Prompts are not automatically written to disk. The interface
uses optional Google Fonts with system-font fallbacks; inference is local after
the model and tokenizer have been cached.

## Checks

```bash
python3 -m unittest discover -s steering_dashboard/tests -v
node --check steering_dashboard/static/app.js
```

Tests cover validation, baseline inclusion, job serialization, partial failure,
and the HTTP sweep contract with a fake engine, without downloading model weights.
