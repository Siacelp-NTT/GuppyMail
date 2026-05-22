"""Generate Phase 5 report charts from training and evaluation outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARTIFACT_DIR = Path("evaluation/artifacts")


def load_json(path: Path) -> Any:
    """Load a JSON file or return None when it is missing."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def training_steps(data: Any) -> list[dict[str, Any]]:
    """Return a normalized list of training step dictionaries."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("steps", [])
    return []


def plot_loss_curve(metrics_path: Path, output_path: Path) -> bool:
    """Plot train/evaluation loss curves from training metrics."""
    data = load_json(metrics_path)
    steps_data = training_steps(data)
    if not steps_data:
        print(f"No training steps found in {metrics_path}; skipping loss curve.")
        return False

    steps = [row["step"] for row in steps_data]
    train_loss = [row["train_loss"] for row in steps_data]
    eval_loss = [row["eval_loss"] for row in steps_data if row.get("eval_loss") is not None]
    eval_steps = [row["step"] for row in steps_data if row.get("eval_loss") is not None]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, train_loss, label="Training loss", color="#2563eb", linewidth=1.6)
    if eval_loss:
        ax.plot(eval_steps, eval_loss, label="Evaluation loss", color="#dc2626", linewidth=2.1)
        best_index = min(range(len(eval_loss)), key=eval_loss.__getitem__)
        ax.scatter([eval_steps[best_index]], [eval_loss[best_index]], color="#111827", zorder=4)
        ax.annotate(
            f"Best eval: {eval_loss[best_index]:.3f}",
            xy=(eval_steps[best_index], eval_loss[best_index]),
            xytext=(12, 16),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#374151"},
            fontsize=9,
        )

    final = steps_data[-1]
    stats = [
        f"Final train: {final.get('train_loss', 0):.3f}",
        f"Final eval: {final.get('eval_loss', 0):.3f}",
        f"Final perplexity: {final.get('eval_perplexity', 0):.1f}",
        f"Steps logged: {len(steps_data)}",
    ]
    ax.text(
        0.98,
        0.95,
        "\n".join(stats),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f9fafb", "edgecolor": "#d1d5db"},
    )
    ax.set_title("GuppyEmail Training Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")
    return True


def eval_metric(data: dict[str, Any], key: str) -> Any:
    """Read an evaluation metric from either flat or nested result JSON."""
    if key == "perplexity":
        return data.get("perplexity") or data.get("evaluation", {}).get("test_perplexity")
    return data.get(key)


def plot_rouge_bar(eval_path: Path, output_path: Path) -> bool:
    """Plot ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    data = load_json(eval_path)
    if not isinstance(data, dict) or not data.get("rouge"):
        print(f"No ROUGE data found in {eval_path}; skipping ROUGE chart.")
        return False

    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    keys = ["rouge1", "rouge2", "rougeL"]
    values = [float(data["rouge"].get(key, 0.0)) for key in keys]
    colors = ["#2563eb", "#16a34a", "#dc2626"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.axhline(0.10, color="#6b7280", linestyle="--", linewidth=1, alpha=0.65)
    ax.set_title("GuppyEmail ROUGE Scores", fontsize=14, fontweight="bold")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, max(values + [0.10]) * 1.3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")
    return True


def main() -> None:
    """Generate all available chart artifacts."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = ARTIFACT_DIR / "training_metrics.json"
    if not metrics_path.exists():
        metrics_path = Path("evaluation/training_history.json")
    plot_loss_curve(metrics_path, ARTIFACT_DIR / "loss_curve.png")
    plot_rouge_bar(Path("evaluation/eval_results.json"), ARTIFACT_DIR / "rouge_chart.png")


if __name__ == "__main__":
    main()
