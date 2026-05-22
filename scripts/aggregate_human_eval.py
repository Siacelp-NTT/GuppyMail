"""Aggregate human evaluation survey responses for Phase 5."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


SCORE_FIELDS = ("fluency", "relevance", "conciseness", "usefulness")


def load_json(path: Path) -> Any:
    """Load JSON from disk, returning None when the file is absent."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalize_responses(raw: Any) -> list[dict[str, Any]]:
    """Normalize supported response JSON shapes into a flat response list."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("responses", "ratings", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Survey responses must be a list or a dict with a responses list.")


def numeric_score(value: Any) -> int | None:
    """Return a valid 1-5 score or None for invalid/missing values."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def aggregate_survey(responses_path: Path, output_path: Path) -> dict[str, Any]:
    """Parse survey responses, compute averages, and write result JSON."""
    responses = normalize_responses(load_json(responses_path))
    completed: list[dict[str, Any]] = []
    invalid_counts: Counter[str] = Counter()

    for response in responses:
        scores = {field: numeric_score(response.get(field)) for field in SCORE_FIELDS}
        if all(scores[field] is not None for field in SCORE_FIELDS):
            completed.append({**response, **scores})
        else:
            for field, value in scores.items():
                if value is None:
                    invalid_counts[field] += 1

    result: dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "responses_path": str(responses_path),
        "response_count": len(responses),
        "completed_response_count": len(completed),
        "status": "complete" if completed else "pending_no_completed_surveys",
        "score_scale": "1=poor, 3=acceptable, 5=excellent",
        "invalid_counts": dict(invalid_counts),
    }

    if completed:
        for field in SCORE_FIELDS:
            values = [int(response[field]) for response in completed]
            result[f"{field}_avg"] = round(mean(values), 2)
            result[f"{field}_min"] = min(values)
            result[f"{field}_max"] = max(values)
        result["overall_avg"] = round(
            mean(float(result[f"{field}_avg"]) for field in SCORE_FIELDS),
            2,
        )
    else:
        for field in SCORE_FIELDS:
            result[f"{field}_avg"] = None
        result["overall_avg"] = None
        result["note"] = (
            "No completed independent human ratings were found. "
            "Fill evaluation/human_eval_responses.json and rerun this script."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("evaluation/human_eval_responses.json"),
        help="JSON file containing completed survey responses.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/human_eval_results.json"),
        help="Destination for aggregate human-evaluation results.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the human evaluation aggregation CLI."""
    args = parse_args()
    result = aggregate_survey(args.responses, args.output)
    print(f"Wrote {args.output}")
    print(f"Status: {result['status']} ({result['completed_response_count']} completed)")


if __name__ == "__main__":
    main()
