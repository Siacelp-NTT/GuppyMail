# Contributing

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common Commands

```bash
python scripts/download_enron.py
python src/preprocess.py
python scripts/generate_summaries.py --max-per-batch 5
python scripts/build_quality_training_data.py
python scripts/evaluate.py
python app.py
```

Phase 5 report artifacts:

```bash
python scripts/aggregate_human_eval.py
python scripts/generate_artifacts.py
python scripts/plot_metrics.py
python scripts/generate_codebase_guide.py
```

## Code Style

- Use Python 3 and 4-space indentation.
- Prefer `pathlib.Path`, JSON helpers, and explicit CLI arguments.
- Keep reusable logic in `src/` and runnable workflows in `scripts/`.
- Add docstrings to new modules, classes, and functions.
- Keep comments focused on non-obvious data, API, or model behavior.

## Data and Secrets

- Do not commit `credentials.json`, `token.json`, `.env`, API keys, raw datasets, checkpoints, or private email content.
- Large generated data belongs under `data/`, `models/`, `checkpoints/`, or `evaluation/artifacts/`.
- Commit small report summaries, scripts, and documentation when useful.

## Validation

Before submitting changes, run the smallest useful checks for the touched area. Examples:

```bash
python3 -m py_compile app.py inference.py src/*.py scripts/*.py
pytest
python scripts/generate_artifacts.py
```
