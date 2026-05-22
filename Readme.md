# GuppyEmail - Email Summarizer Prototype

GuppyEmail is a student-scale email summarization project. It prepares an email dataset, trains a tiny decoder-only transformer from scratch, evaluates the model, and wraps it in a Gradio app with priority classification and action item extraction.

The project is intentionally small and explainable. It is not a production assistant, but it demonstrates the full path from raw email data to a working summarization demo and final report artifacts.

## Current Highlights

- Small transformer model, about 8.7M parameters.
- Enron email cleaning and quality-filtered training data.
- API-generated reference summaries for supervised learning.
- Gradio app for email summarization.
- Optional Gmail API path.
- Rule-based priority classification: urgent, important, normal, low.
- Rule-based extraction for tasks, deadlines, requests, and meetings.
- Phase 5 report artifacts, charts, human-evaluation survey, and visual codebase guide.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the local app after model artifacts are available:

```bash
python app.py
```

Open:

```text
http://localhost:7860
```

## Data Pipeline

```bash
python scripts/download_enron.py
python src/preprocess.py
python scripts/generate_summaries.py --workers 3
python scripts/build_quality_training_data.py
python scripts/train_tokenizer.py
```

`scripts/generate_summaries.py` requires `OPENAI_API_KEY` or `--api-key`. Avoid running full API jobs unless you intend to regenerate labels.

## Training

Training is designed for Google Colab:

1. Open `notebooks/train_email_guppylm.ipynb`.
2. Upload the prepared training data.
3. Train the model.
4. Download `best_model.pt`, `config.json`, tokenizer, and evaluation outputs.
5. Place local runtime artifacts under `checkpoints/`, `models/`, or `data/training_quality/` as expected by the app.

Model configuration used by the current evaluation:

| Field | Value |
|---|---:|
| Parameters | about 8.7M |
| Layers | 6 |
| Hidden dimension | 384 |
| Attention heads | 6 |
| FFN hidden | 768 |
| Vocabulary | 4,096 BPE tokens |
| Max sequence length | 512 |

## Evaluation

Run automated evaluation:

```bash
python scripts/evaluate.py
```

Current saved metrics in `evaluation/eval_results.json`:

| Metric | Value |
|---|---:|
| Test perplexity | about 13.0 |
| ROUGE-1 | about 0.372 |
| ROUGE-2 | about 0.169 |
| ROUGE-L | about 0.322 |

The random-weight baseline in `evaluation/baseline_results.json` has perplexity around 4547, so the trained model is substantially better than the untrained baseline.

## Phase 5 Artifacts

Generate final report artifacts:

```bash
python scripts/aggregate_human_eval.py
python scripts/generate_human_eval_packet.py
python scripts/generate_artifacts.py
python scripts/plot_metrics.py
python scripts/generate_codebase_guide.py
```

## Project Structure

```text
email-summarizer/
├── app.py                         # Gradio app
├── inference.py                   # Runtime model inference wrapper
├── config.py, model.py             # Compatibility exports
├── src/
│   ├── preprocess.py               # Email cleaning
│   ├── pipeline.py                 # End-to-end app pipeline
│   ├── guppyemail_model.py         # Transformer model
│   ├── guppyemail_data.py          # ChatML data helpers
│   ├── classifier.py               # Priority classification
│   ├── action_extractor.py         # Action item extraction
│   ├── gmail_client.py             # Gmail API wrapper
│   └── data_dashboard.py           # Data dashboard
├── scripts/
│   ├── download_enron.py
│   ├── generate_summaries.py
│   ├── build_quality_training_data.py
│   ├── train_tokenizer.py
│   ├── evaluate.py
│   ├── export_model.py
│   ├── generate_artifacts.py
│   ├── plot_metrics.py
│   ├── aggregate_human_eval.py
│   └── generate_codebase_guide.py
├── data/                           # Generated datasets, ignored by git
├── evaluation/                     # Evaluation JSON and Phase 5 artifacts
├── report/                         # Report, presentation, screenshots, guide
├── notebooks/                      # Colab training notebooks
├── tests/                          # Pytest tests
└── tasks/                          # Project task specs, ignored by git
```

## Testing and Validation

Run the available tests:

```bash
pytest
```

Run syntax checks:

```bash
python3 -m py_compile app.py inference.py src/*.py scripts/*.py
```

For app-level validation, run:

```bash
python app.py
```

## Limitations

- GuppyEmail is a tiny model and can repeat phrases or miss details.
- Enron email data does not fully represent modern personal email.
- API-generated labels can contain bias or imperfect summaries.
- Priority classification and action extraction are rule-based, so they are transparent but brittle.
- Human evaluation requires completed independent survey responses before it can be considered final.
- Gmail integration must be used carefully because it can expose private data.

## Security

Do not commit secrets or private data. The repository ignores common sensitive artifacts such as:

- `credentials.json`
- `token.json`
- `.env`
- raw data under `data/raw/`
- generated model checkpoints
- exported model binaries

Keep API keys in environment variables.

## References

- GuppyLM: https://github.com/arman-bd/guppylm
- GuppyLM model card: https://huggingface.co/arman-bd/guppylm-9M
- Enron email dataset source used by the project: https://huggingface.co/datasets/jacquelinehe/enron-emails
- Summarized email dataset: https://huggingface.co/datasets/Siacelp/eron-mails-summarized