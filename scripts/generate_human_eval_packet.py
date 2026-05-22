"""Generate ready-to-rate human evaluation materials from model samples."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


EVAL_PATH = Path("evaluation/eval_results.json")
PACKET_PATH = Path("evaluation/human_eval_packet.md")
FORM_PATH = Path("evaluation/human_eval_form.html")


def load_samples(limit: int = 5) -> list[dict[str, Any]]:
    """Load evaluation samples for human scoring."""
    with EVAL_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("samples", [])[:limit]


def compact(value: str, limit: int = 900) -> str:
    """Collapse whitespace and truncate long text for rater readability."""
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def generate_packet(samples: list[dict[str, Any]]) -> None:
    """Write a Markdown packet with real examples and blank score tables."""
    lines = [
        "# GuppyEmail Human Evaluation Packet",
        "",
        "Score each generated summary from 1 to 5.",
        "",
        "| Score | Meaning |",
        "|---:|---|",
        "| 1 | Poor |",
        "| 2 | Weak |",
        "| 3 | Acceptable |",
        "| 4 | Good |",
        "| 5 | Excellent |",
        "",
        "Dimensions: fluency, relevance, conciseness, usefulness.",
    ]
    for index, sample in enumerate(samples, start=1):
        lines += [
            "",
            f"## Example {index}",
            "",
            f"**Original email excerpt:** {compact(sample.get('email', ''))}",
            "",
            f"**Reference summary:** {compact(sample.get('reference', ''), 500)}",
            "",
            f"**GuppyEmail summary:** {compact(sample.get('generated', sample.get('raw_generated', '')), 500)}",
            "",
            "| Fluency | Relevance | Conciseness | Usefulness | Notes |",
            "|---:|---:|---:|---:|---|",
            "|  |  |  |  |  |",
        ]
    lines += [
        "",
        "After rating, transfer scores to `evaluation/human_eval_responses.json` and run:",
        "",
        "```bash",
        "python scripts/aggregate_human_eval.py",
        "```",
    ]
    PACKET_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def option_tags(name: str) -> str:
    """Return select options for a 1-5 rating field."""
    return (
        f'<select name="{name}" required>'
        '<option value="">Score</option>'
        + "".join(f'<option value="{score}">{score}</option>' for score in range(1, 6))
        + "</select>"
    )


def generate_form(samples: list[dict[str, Any]]) -> None:
    """Write a static local HTML form that exports survey JSON."""
    cards = []
    for index, sample in enumerate(samples, start=1):
        cards.append(
            f"""
            <section class="card" data-example="{index}">
              <h2>Example {index}</h2>
              <h3>Original email excerpt</h3>
              <p>{html.escape(compact(sample.get('email', '')))}</p>
              <h3>Reference summary</h3>
              <p>{html.escape(compact(sample.get('reference', ''), 500))}</p>
              <h3>GuppyEmail summary</h3>
              <p>{html.escape(compact(sample.get('generated', sample.get('raw_generated', '')), 500))}</p>
              <div class="ratings">
                <label>Fluency {option_tags(f'fluency-{index}')}</label>
                <label>Relevance {option_tags(f'relevance-{index}')}</label>
                <label>Conciseness {option_tags(f'conciseness-{index}')}</label>
                <label>Usefulness {option_tags(f'usefulness-{index}')}</label>
              </div>
              <label>Notes <textarea name="notes-{index}" rows="2"></textarea></label>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GuppyEmail Human Evaluation Form</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #f6f7f9; color: #172033; }}
    header {{ background: #172033; color: white; padding: 24px; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 20px; }}
    .card {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 16px; margin: 0 0 16px; }}
    h1, h2, h3 {{ margin-top: 0; }}
    p {{ line-height: 1.55; }}
    .ratings {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 12px 0; }}
    label {{ display: grid; gap: 6px; font-weight: 600; }}
    select, textarea, input {{ width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px; font: inherit; }}
    button {{ border: 0; background: #2557a7; color: white; border-radius: 6px; padding: 10px 14px; font-weight: 700; cursor: pointer; }}
    pre {{ white-space: pre-wrap; background: #111827; color: #e5e7eb; padding: 14px; border-radius: 8px; }}
    @media (max-width: 760px) {{ .ratings {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>GuppyEmail Human Evaluation Form</h1>
    <p>Rate each generated summary from 1 to 5 for fluency, relevance, conciseness, and usefulness.</p>
  </header>
  <main>
    <label>Rater ID <input id="rater" value="rater-1"></label>
    {''.join(cards)}
    <button id="export">Export JSON</button>
    <pre id="output">Fill the form, then export JSON.</pre>
  </main>
  <script>
    document.getElementById('export').addEventListener('click', () => {{
      const rater = document.getElementById('rater').value || 'anonymous';
      const rows = [];
      document.querySelectorAll('.card').forEach((card) => {{
        const id = card.dataset.example;
        rows.push({{
          rater,
          example_id: Number(id),
          fluency: Number(card.querySelector(`[name="fluency-${{id}}"]`).value),
          relevance: Number(card.querySelector(`[name="relevance-${{id}}"]`).value),
          conciseness: Number(card.querySelector(`[name="conciseness-${{id}}"]`).value),
          usefulness: Number(card.querySelector(`[name="usefulness-${{id}}"]`).value),
          notes: card.querySelector(`[name="notes-${{id}}"]`).value
        }});
      }});
      document.getElementById('output').textContent = JSON.stringify(rows, null, 2);
    }});
  </script>
</body>
</html>
"""
    FORM_PATH.write_text(html_text, encoding="utf-8")


def main() -> None:
    """Generate Markdown and HTML human evaluation materials."""
    samples = load_samples()
    generate_packet(samples)
    generate_form(samples)
    print(f"Wrote {PACKET_PATH}")
    print(f"Wrote {FORM_PATH}")


if __name__ == "__main__":
    main()
