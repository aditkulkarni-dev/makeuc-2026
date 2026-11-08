# Steering Lab — MakeUC 2026

A local experiment dashboard for studying whether activation steering can strengthen an open language model's refusal behavior against adversarial prompts. Built for MakeUC 2026 using **Gemma 2 2B Instruct** (`google/gemma-2-2b-it`).

## Problem

Language models can refuse a harmful request but respond to the same request when an adversarial suffix is appended. This project investigates whether reinforcing a model's internal refusal direction can reduce that behavior, and how different steering strengths affect the generated responses.

## Solution

Steering Lab adds a scaled refusal vector to the activations entering a selected transformer layer during inference:

```text
steered_activation = activation + alpha * refusal_direction
```

The dashboard compares the same prompt across steering strengths, always including an unmodified control (`alpha = 0`). It supports side-by-side responses, manual ratings, generation timing, token counts, and JSON exports. Collection scripts run reproducible comparisons on saved prompt/suffix variants; a results viewer and refusal-prefix scorer help inspect those experiments.

This implementation builds on the refusal-direction research and code credited under [Research attribution](#research-attribution).

## Install and run

### Quick demo

Requirements: **Python 3.10 or newer** and Git. Run these commands in a terminal:

```bash
git clone https://github.com/aditkulkarni-dev/makeuc-2026.git
cd makeuc-2026
python3 steering_dashboard/server.py --demo
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Demo mode needs no extra Python packages, model access, or GPU. Interactive responses are labeled placeholders; the collected-results viewer can still display the real outputs saved in the repository.

The frontend is plain HTML, CSS, and JavaScript, served by Python. **No frontend build step or Node.js installation is required to run it.** Stop the server with `Ctrl+C`.

### Live model inference

From the repository root, create an environment and install the dashboard dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r steering_dashboard/requirements.txt
```

The requirements include PyTorch, Transformers, and Accelerate. Use a PyTorch build appropriate for your hardware. A CUDA-capable GPU is recommended; CPU inference uses float32 and is considerably slower.

Request access to [Gemma 2 2B Instruct on Hugging Face](https://huggingface.co/google/gemma-2-2b-it), accept its model terms, then authenticate:

```bash
hf auth login
```

Live inference also requires **both** of these model-specific artifacts:

```text
refusal_direction_gen/refusal_direction/pipeline/runs/gemma-2-2b-it/
├── direction.pt
└── direction_metadata.json
```

The metadata identifies the steering layer. The `pipeline/runs/` directory is ignored by Git, so a fresh clone does not include these files. The root-level `direction.pt` alone is insufficient. Obtain a matching direction and metadata for `google/gemma-2-2b-it`, or generate them using the bundled research pipeline. See its [README](refusal_direction_gen/refusal_direction/README.md) for the upstream workflow; it has older, separate dependencies and model-specific configuration, so use a separate environment for that pipeline.

Once the artifacts are in place, start the dashboard:

```bash
python steering_dashboard/server.py
```

The model loads on the first sweep and may download weights if they are not cached. Open [http://127.0.0.1:8000](http://127.0.0.1:8000), enter a prompt, select an alpha range, and run the comparison.

To use artifacts stored elsewhere or change the port:

```bash
python steering_dashboard/server.py \
  --direction-dir /absolute/path/to/gemma-2-2b-it \
  --port 8001
```

The server binds to localhost. The dashboard's model dropdown is a UI preview; the active model is selected with the server's `--model` argument and requires a direction produced for that exact model.

## Collect and evaluate experiments

With the live-inference environment active and matching artifacts available, run from the repository root:

```bash
# Collect a new dataset without overwriting the bundled experiment.
python scripts/collect_steering_data.py --run --output results/my_steering_dataset

# Score the new responses without additional model inference.
python scripts/score_steering_data.py --input results/my_steering_dataset/dashboard.json
```

The default collection uses 27 saved suffix variants of one base question at alpha values `0`, `0.1`, `0.2`, `0.25`, and `0.5`, with a 160-token generation limit. It saves responses, timings, configuration metadata, and resumable checkpoints. See [collection instructions](scripts/STEERING_DATA.md) for input formats and options.

The dashboard reads the bundled experiment from `results/experimental_steering_dataset/`. Click **View collected results** to inspect it. To rescore that saved experiment without loading a model:

```bash
python scripts/score_steering_data.py
```

The chart reports a **non-refusal estimate** based on response prefixes. It does not establish harmful compliance or response quality. The 27 variants were selected for prior baseline success and represent one base question, so their scores do not measure general jailbreak resistance. Inspect the actual responses and truncation flags alongside the chart.

## Repository layout

| Path | Purpose |
| --- | --- |
| `steering_dashboard/` | Local web interface, HTTP server, and inference engine |
| `scripts/steering.py` | Standalone steering experiment |
| `scripts/collect_steering_data.py` | Paired response collection and checkpoints |
| `scripts/score_steering_data.py` | Refusal-prefix scoring of saved responses |
| `scripts/data/` | Pinned source prompts and review metadata |
| `results/` | Saved experiments and evaluation reports |
| `refusal_direction_gen/refusal_direction/` | Bundled research code for extracting and evaluating refusal directions |

For dashboard behavior and additional options, see the [dashboard README](steering_dashboard/README.md).

## Checks

These checks use fake inference or saved data and do not download model weights:

```bash
python3 -m unittest discover -s steering_dashboard/tests -v
python3 -m unittest discover -s scripts/tests -v
```

If Node.js is installed, check the frontend JavaScript syntax with:

```bash
node --check steering_dashboard/static/app.js
```

## Research attribution

The refusal-direction methodology and bundled code in `refusal_direction_gen/refusal_direction/` originate from [Andy Arditi and collaborators' research repository](https://github.com/andyrdt/refusal_direction). This project's dashboard, collection workflow, and steering experiments build on that work. The bundled research code retains its [Apache License 2.0](refusal_direction_gen/refusal_direction/LICENSE).

**Reference:** Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). *Refusal in language models is mediated by a single direction*. arXiv:2406.11717. [https://doi.org/10.48550/arXiv.2406.11717](https://doi.org/10.48550/arXiv.2406.11717)

```bibtex
@article{arditi2024refusal,
  title   = {Refusal in Language Models Is Mediated by a Single Direction},
  author  = {Arditi, Andy and Obeso, Oscar and Syed, Aaquib and
             Paleka, Daniel and Panickssery, Nina and Gurnee, Wes and
             Nanda, Neel},
  journal = {arXiv preprint arXiv:2406.11717},
  year    = {2024},
  doi     = {10.48550/arXiv.2406.11717},
  url     = {https://arxiv.org/abs/2406.11717}
}
```
