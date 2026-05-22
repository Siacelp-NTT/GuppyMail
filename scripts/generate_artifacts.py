"""Generate Phase 5 mentor-report artifacts from current project outputs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path("evaluation/artifacts")


def load_json(path: str | Path) -> Any:
    """Load JSON from a path, returning None when missing or malformed."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def count_jsonl(path: str | Path) -> int:
    """Count JSONL rows without loading the whole file."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def file_size(path: str | Path) -> str:
    """Return a compact human-readable file size."""
    path = Path(path)
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


def metric(data: dict[str, Any] | None, name: str) -> float | None:
    """Read an evaluation metric from supported result shapes."""
    if not data:
        return None
    if name == "perplexity":
        return data.get("perplexity") or data.get("evaluation", {}).get("test_perplexity")
    if name in ("rouge1", "rouge2", "rougeL"):
        return data.get("rouge", {}).get(name)
    return data.get(name)


def fmt(value: Any, digits: int = 3) -> str:
    """Format optional numeric values for Markdown tables."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text and ensure the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def latest_training_metrics() -> dict[str, Any]:
    """Return normalized training metrics from artifacts or history."""
    artifact_metrics = load_json(ARTIFACT_DIR / "training_metrics.json")
    history = load_json("evaluation/training_history.json")
    if isinstance(artifact_metrics, dict) and artifact_metrics.get("steps"):
        steps = artifact_metrics["steps"]
    elif isinstance(history, list):
        steps = history
    else:
        steps = []
    if not steps:
        return {"steps": [], "best_eval_loss": None, "final_train_loss": None, "total_time_seconds": None}
    eval_steps = [row for row in steps if row.get("eval_loss") is not None]
    best = min(eval_steps, key=lambda row: row["eval_loss"]) if eval_steps else {}
    final = steps[-1]
    artifact_seconds = (
        artifact_metrics.get("total_time_seconds")
        if isinstance(artifact_metrics, dict)
        else None
    )
    return {
        "steps": steps,
        "best_eval_loss": best.get("eval_loss"),
        "best_step": best.get("step"),
        "final_train_loss": final.get("train_loss"),
        "final_eval_loss": final.get("eval_loss"),
        "final_eval_perplexity": final.get("eval_perplexity"),
        "total_time_seconds": final.get("elapsed_sec") or artifact_seconds,
        "num_steps": final.get("step"),
    }


def generate_training_metrics_artifact() -> None:
    """Write the expected Phase 5 training metrics artifact when history exists."""
    output_path = ARTIFACT_DIR / "training_metrics.json"
    existing = load_json(output_path)
    if isinstance(existing, dict) and existing.get("steps"):
        return

    history = load_json("evaluation/training_history.json")
    if not isinstance(history, list) or not history:
        return

    eval_steps = [row for row in history if row.get("eval_loss") is not None]
    best = min(eval_steps, key=lambda row: row["eval_loss"]) if eval_steps else {}
    final = history[-1]
    payload = {
        "source": "evaluation/training_history.json",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "model_params": {"params": 8_700_000},
            "train_params": {
                "max_steps": final.get("step"),
                "note": "Normalized from saved training history.",
            },
        },
        "steps": history,
        "best_eval_loss": best.get("eval_loss"),
        "best_step": best.get("step"),
        "final_train_loss": final.get("train_loss"),
        "final_eval_loss": final.get("eval_loss"),
        "final_eval_perplexity": final.get("eval_perplexity"),
        "total_time_seconds": final.get("elapsed_sec"),
        "num_steps": final.get("step"),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def generate_readme() -> None:
    """Write an index for generated mentor artifacts."""
    lines = [
        "# Phase 5 Artifact Index",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Artifact | Purpose |",
        "|---|---|",
        "| dataset_summary.md | Data pipeline counts, filtering, and tokenizer statistics |",
        "| model_card.md | Model architecture, size, training, and evaluation facts |",
        "| comparison_table.md | Baseline vs trained model comparison |",
        "| samples_gallery.md | Example email, reference summary, and model output pairs |",
        "| metrics_summary.json | Machine-readable headline metrics |",
        "| loss_curve.png | Training/evaluation loss chart |",
        "| rouge_chart.png | ROUGE score chart |",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python scripts/generate_artifacts.py",
        "python scripts/plot_metrics.py",
        "```",
    ]
    write_text(ARTIFACT_DIR / "README.md", "\n".join(lines))


def generate_dataset_summary() -> None:
    """Write report-ready data pipeline statistics."""
    profile = load_json("report/data_profile_metrics.json") or {}
    quality = load_json("data/training_quality/quality_metrics.json") or {}
    tokenizer = load_json("data/training_quality/tokenizer_metrics.json") or {}
    counts = profile.get("counts", {})
    rows = [
        ("Raw Enron sample", "data/raw/enron_sample.jsonl", counts.get("raw") or count_jsonl("data/raw/enron_sample.jsonl")),
        ("Cleaned emails", "data/processed/cleaned_emails.jsonl", counts.get("cleaned") or count_jsonl("data/processed/cleaned_emails.jsonl")),
        ("Clean summaries", "data/summaries/en_summaries.clean.jsonl", counts.get("clean_summaries") or count_jsonl("data/summaries/en_summaries.clean.jsonl")),
        ("Quality filtered", "data/training_quality/filtered_source.jsonl", counts.get("quality_filtered") or count_jsonl("data/training_quality/filtered_source.jsonl")),
        ("Train split", "data/training_quality/train.jsonl", counts.get("train") or count_jsonl("data/training_quality/train.jsonl")),
        ("Validation split", "data/training_quality/val.jsonl", counts.get("val") or count_jsonl("data/training_quality/val.jsonl")),
        ("Test split", "data/training_quality/test.jsonl", counts.get("test") or count_jsonl("data/training_quality/test.jsonl")),
    ]
    lines = [
        "# Dataset Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Pipeline Counts",
        "",
        "| Stage | File | Count |",
        "|---|---|---:|",
    ]
    lines += [f"| {name} | `{path}` | {count:,} |" for name, path, count in rows]
    lines += [
        "",
        "## Filtering Notes",
        "",
        f"- Coverage after cleaning and summary generation: {profile.get('coverage_pct', 'N/A')}%",
        f"- Duplicates removed during cleaning: {profile.get('duplicates_removed_during_cleaning', 'N/A')}",
        f"- Accepted quality rows: {quality.get('accepted_rows', 'N/A')}",
        f"- Rejected quality rows: {quality.get('rejected_rows', 'N/A')}",
        f"- No-summary rows: {quality.get('no_summary_rows', 'N/A')}",
        "",
        "## Length and Tokenization",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for label, section, key in [
        ("Cleaned email median chars", "cleaned_length", "median"),
        ("Cleaned email mean chars", "cleaned_length", "mean"),
        ("Summary median chars", "summary_length", "median"),
        ("Summary mean chars", "summary_length", "mean"),
        ("Summary median words", "summary_words", "median"),
        ("Compression ratio mean", "compression_ratio", "mean"),
    ]:
        lines.append(f"| {label} | {profile.get(section, {}).get(key, 'N/A')} |")
    tok = profile.get("tokenization", {}) or tokenizer
    lines += [
        f"| Vocab size | {tok.get('vocab_size', quality.get('vocab_size', 'N/A'))} |",
        f"| Max sequence length | {tok.get('max_seq_len', quality.get('max_seq_len', 'N/A'))} |",
        f"| Truncated rows | {tok.get('truncated_rows', quality.get('truncated_rows', 'N/A'))} |",
    ]
    write_text(ARTIFACT_DIR / "dataset_summary.md", "\n".join(lines))


def generate_model_card() -> None:
    """Write a compact model card for the trained model."""
    eval_data = load_json("evaluation/eval_results.json")
    train = latest_training_metrics()
    config = eval_data.get("config", {}) if isinstance(eval_data, dict) else {}
    lines = [
        "# Model Card: GuppyEmail",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Intended Use",
        "",
        "GuppyEmail is a small decoder-only transformer trained from scratch to produce short email summaries for a student prototype and demo UI.",
        "",
        "## Architecture",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Parameters | ~8.7M |",
        f"| Layers | {config.get('n_layers', 6)} |",
        f"| Hidden dimension | {config.get('d_model', 384)} |",
        f"| Attention heads | {config.get('n_heads', 6)} |",
        f"| FFN hidden | {config.get('ffn_hidden', 768)} |",
        f"| Vocabulary | {config.get('vocab_size', 4096)} |",
        f"| Max sequence length | {config.get('max_seq_len', 512)} |",
        f"| Checkpoint size | {file_size('checkpoints/best_model.pt')} |",
        "",
        "## Training",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Logged final step | {fmt(train.get('num_steps'), 0)} |",
        f"| Best eval loss | {fmt(train.get('best_eval_loss'))} |",
        f"| Best step | {fmt(train.get('best_step'), 0)} |",
        f"| Final train loss | {fmt(train.get('final_train_loss'))} |",
        f"| Final eval loss | {fmt(train.get('final_eval_loss'))} |",
        f"| Training time | {fmt(train.get('total_time_seconds'), 1)} sec |",
        "",
        "## Evaluation",
        "",
        "| Metric | Value | Target | Status |",
        "|---|---:|---:|---|",
    ]
    targets = [("Perplexity", "perplexity", 40, "<"), ("ROUGE-1", "rouge1", 0.15, ">="), ("ROUGE-2", "rouge2", 0.05, ">="), ("ROUGE-L", "rougeL", 0.10, ">=")]
    for label, key, target, direction in targets:
        value = metric(eval_data, key)
        passed = value is not None and ((value < target) if direction == "<" else (value >= target))
        lines.append(f"| {label} | {fmt(value)} | {direction} {target} | {'pass' if passed else 'review'} |")
    lines += [
        "",
        "## Limitations",
        "",
        "- The model is intentionally small and can repeat phrases or miss details.",
        "- It was trained on summarized Enron-style email data, not private Gmail data.",
        "- Priority and action extraction are deterministic rule-based helpers, not learned classifiers.",
    ]
    write_text(ARTIFACT_DIR / "model_card.md", "\n".join(lines))


def generate_comparison_table() -> None:
    """Write baseline-vs-trained quantitative and qualitative comparison."""
    baseline = load_json("evaluation/baseline_results.json")
    trained = load_json("evaluation/eval_results.json")
    human = load_json("evaluation/human_eval_results.json")
    rows = [
        ("Perplexity", metric(baseline, "perplexity"), metric(trained, "perplexity"), "Lower is better"),
        ("ROUGE-1", metric(baseline, "rouge1"), metric(trained, "rouge1"), "Higher is better"),
        ("ROUGE-2", metric(baseline, "rouge2"), metric(trained, "rouge2"), "Higher is better"),
        ("ROUGE-L", metric(baseline, "rougeL"), metric(trained, "rougeL"), "Higher is better"),
    ]
    lines = [
        "# Baseline vs Trained GuppyEmail",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Metric | Baseline | Trained | Notes |",
        "|---|---:|---:|---|",
    ]
    lines += [f"| {name} | {fmt(base)} | {fmt(trained_value)} | {note} |" for name, base, trained_value, note in rows]
    lines += [
        "",
        "## Qualitative Difference",
        "",
        "| Area | Baseline | Trained model |",
        "|---|---|---|",
        "| Readability | Random token sequences | Short, mostly readable summaries |",
        "| Relevance | Not meaningfully tied to the email | Often captures people, dates, requests, or status |",
        "| Repetition | High | Still present but reduced |",
        "| Demo usefulness | No | Useful as a small-model prototype with limitations |",
    ]
    if isinstance(human, dict):
        lines += [
            "",
            "## Human Evaluation Status",
            "",
            f"- Status: {human.get('status', 'unknown')}",
            f"- Completed responses: {human.get('completed_response_count', 0)}",
            f"- Overall average: {human.get('overall_avg', 'N/A')}",
        ]
    write_text(ARTIFACT_DIR / "comparison_table.md", "\n".join(lines))


def generate_samples_gallery() -> None:
    """Write a side-by-side sample output gallery."""
    eval_data = load_json("evaluation/eval_results.json") or {}
    baseline = load_json("evaluation/baseline_results.json") or {}
    samples = eval_data.get("samples", [])
    baseline_samples = baseline.get("samples", [])
    lines = [
        "# Sample Outputs Gallery",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Side-by-side examples from the evaluation output.",
    ]
    if not samples:
        lines.append("\nNo samples found in `evaluation/eval_results.json`.")
    for index, sample in enumerate(samples[:8], start=1):
        base = baseline_samples[index - 1].get("output", "N/A") if index - 1 < len(baseline_samples) else "N/A"
        email = " ".join(sample.get("email", "").split())[:500]
        lines += [
            "",
            f"## Example {index}",
            "",
            f"**Email excerpt:** {email}",
            "",
            "| Source | Text |",
            "|---|---|",
            f"| Reference | {sample.get('reference', 'N/A')} |",
            f"| Trained GuppyEmail | {sample.get('generated', sample.get('raw_generated', 'N/A'))} |",
            f"| Baseline | {base[:260]} |",
            "",
            f"ROUGE-L for example: {fmt(sample.get('rougeL'))}",
        ]
    write_text(ARTIFACT_DIR / "samples_gallery.md", "\n".join(lines))


def generate_metrics_summary() -> None:
    """Write machine-readable summary metrics for reuse in reports or UI."""
    eval_data = load_json("evaluation/eval_results.json")
    baseline = load_json("evaluation/baseline_results.json")
    profile = load_json("report/data_profile_metrics.json") or {}
    human = load_json("evaluation/human_eval_results.json")
    train = latest_training_metrics()
    summary = {
        "project": "GuppyEmail",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "dataset": profile.get("counts", {}),
        "training": {key: value for key, value in train.items() if key != "steps"},
        "evaluation": {
            "perplexity": metric(eval_data, "perplexity"),
            "rouge1": metric(eval_data, "rouge1"),
            "rouge2": metric(eval_data, "rouge2"),
            "rougeL": metric(eval_data, "rougeL"),
        },
        "baseline": {
            "perplexity": metric(baseline, "perplexity"),
            "rouge1": metric(baseline, "rouge1"),
            "rouge2": metric(baseline, "rouge2"),
            "rougeL": metric(baseline, "rougeL"),
        },
        "human_evaluation": human,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT_DIR / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def main() -> None:
    """Generate all Markdown and JSON mentor artifacts."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    generate_training_metrics_artifact()
    generate_readme()
    generate_dataset_summary()
    generate_model_card()
    generate_comparison_table()
    generate_samples_gallery()
    generate_metrics_summary()
    for path in sorted(ARTIFACT_DIR.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
