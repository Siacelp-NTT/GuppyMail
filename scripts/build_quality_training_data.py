"""Build a higher-quality guppyemail retraining split.

This script starts from generated summary data, removes low-signal examples from
the summarizer training set, and saves rejected generic/no-summary rows for
separate fallback handling.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.train_tokenizer import (
    build_rows,
    full_chatml,
    record_email,
    record_summary,
    split_rows,
    train_tokenizer,
    write_jsonl,
)


DATA_DIR = BASE_DIR / "data"
SUM_DIR = DATA_DIR / "summaries"
DEFAULT_INPUT = SUM_DIR / "en_summaries.clean.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "training_quality"

GENERIC_SUMMARY_RE = re.compile(
    r"^\s*(email received[,;:]?\s*)?nothing specific to report\.?\s*$",
    re.IGNORECASE,
)

TRAINABLE_SUMMARY_TYPES = {"", "summary"}
TRAINABLE_QUALITIES = {"", "good"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def word_count(text: str) -> int:
    return len((text or "").split())


def rejection_reasons(row: dict, args: argparse.Namespace) -> list[str]:
    email = record_email(row)
    summary = record_summary(row)
    email_words = word_count(email)
    summary_words = word_count(summary)
    reasons = []
    summary_type = str(row.get("summary_type", "")).strip().lower()
    summary_quality = str(row.get("summary_quality", row.get("quality", ""))).strip().lower()

    if summary_type not in TRAINABLE_SUMMARY_TYPES:
        reasons.append(f"summary_type_{summary_type}")
    if summary_quality not in TRAINABLE_QUALITIES:
        reasons.append(f"summary_quality_{summary_quality}")

    if not email or not summary:
        reasons.append("missing_text")
        return reasons

    if GENERIC_SUMMARY_RE.match(summary):
        reasons.append("generic_summary")
    summary_lower = summary.lower().strip()
    if summary_lower.startswith(("{", "[")) or "\"summary_type\"" in summary_lower or '"quality"' in summary_lower:
        reasons.append("json_fragment_summary")
    if email_words < args.min_email_words:
        reasons.append("email_too_short")
    if email_words > args.max_email_words:
        reasons.append("email_too_long")
    if summary_words < args.min_summary_words:
        reasons.append("summary_too_short")
    if summary_words > args.max_summary_words:
        reasons.append("summary_too_long")
    if email_words and summary_words / email_words > args.max_summary_email_ratio:
        reasons.append("weak_compression")
    if email.upper().count("[IMAGE]") >= args.max_image_markers:
        reasons.append("image_heavy")
    if re.search(r"\bunsubscribe\b", email, re.IGNORECASE) and email_words < args.marketing_word_limit:
        reasons.append("short_marketing")

    return reasons


def filter_rows(rows: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict], Counter]:
    accepted = []
    rejected = []
    reason_counts = Counter()

    for row in rows:
        reasons = rejection_reasons(row, args)
        for reason in reasons:
            reason_counts[reason] += 1

        email = record_email(row)
        summary = record_summary(row)
        annotated = {
            **row,
            "email_words": word_count(email),
            "summary_words": word_count(summary),
            "summary_email_ratio": round(word_count(summary) / max(1, word_count(email)), 4),
        }

        if reasons:
            annotated["reject_reasons"] = reasons
            rejected.append(annotated)
        else:
            annotated["quality_label"] = "summary_train"
            accepted.append(annotated)

    return accepted, rejected, reason_counts


def is_fallback_row(row: dict) -> bool:
    summary_type = str(row.get("summary_type", "")).strip().lower()
    summary_quality = str(row.get("summary_quality", row.get("quality", ""))).strip().lower()
    reasons = set(row.get("reject_reasons", []))

    if summary_type in {"no_summary_needed", "noise"}:
        return True
    if "generic_summary" in reasons:
        return True
    if "missing_text" in reasons and summary_quality in {"weak", "noise"}:
        return True
    return False


def write_report(output_dir: Path, metrics: dict, reason_counts: Counter) -> None:
    lines = [
        "# guppyemail Quality Training Data Report",
        "",
        f"- Source rows: {metrics['source_rows']:,}",
        f"- Accepted summarization rows: {metrics['accepted_rows']:,}",
        f"- Rejected rows: {metrics['rejected_rows']:,}",
        f"- Generic/no-summary rows: {metrics['generic_summary_rows']:,}",
        f"- Fallback/no-summary rows: {metrics['no_summary_rows']:,}",
        f"- Train rows: {metrics['train_rows']:,}",
        f"- Eval rows: {metrics['eval_rows']:,}",
        f"- Test rows: {metrics['test_rows']:,}",
        f"- Truncated rows after token fitting: {metrics['truncated_rows']:,}",
        "",
        "## Rejection Reasons",
        "",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"- `{reason}`: {count:,}")
    lines.extend(
        [
            "",
            "## Intended Use",
            "",
            "- Use this split to retrain the summarizer.",
            "- Use rejected generic rows for a separate no-summary/fallback rule, not as dominant summarizer targets.",
            "- Keep evaluation honest by reporting both raw model output and post-processed fallback output.",
            "",
        ]
    )
    (output_dir / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_training_zip(output_dir: Path, zip_output: str | None) -> str | None:
    if not zip_output:
        return None
    zip_path = Path(zip_output)
    if not zip_path.is_absolute():
        zip_path = BASE_DIR / zip_path
    files = [
        "train.jsonl",
        "eval.jsonl",
        "val.jsonl",
        "test.jsonl",
        "tokenizer.json",
        "tokenizer_metrics.json",
        "quality_metrics.json",
        "quality_report.md",
    ]
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        for name in files:
            path = output_dir / name
            if path.exists():
                archive.write(path, f"data/training/{name}")
    return str(zip_path)


def prepare_quality_training_data(args: argparse.Namespace) -> dict:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = load_jsonl(input_path)
    accepted, rejected, reason_counts = filter_rows(source_rows, args)
    if not accepted:
        raise ValueError("No rows survived quality filtering. Relax filter thresholds.")

    filtered_source = output_dir / "filtered_source.jsonl"
    rejected_source = output_dir / "rejected_source.jsonl"
    generic_source = output_dir / "no_summary_source.jsonl"

    write_jsonl(filtered_source, accepted)
    write_jsonl(rejected_source, rejected)
    fallback_rows = [row for row in rejected if is_fallback_row(row)]
    write_jsonl(generic_source, fallback_rows)

    tokenizer_input = [
        full_chatml(record_email(row), record_summary(row))
        for row in accepted
        if record_email(row) and record_summary(row)
    ]
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer = train_tokenizer(tokenizer_input, tokenizer_path, args.vocab_size)

    formatted, skipped, truncated = build_rows(accepted, tokenizer, args.max_seq_len)
    train, eval_rows, test = split_rows(formatted, args.val_pct, args.test_pct, args.seed)

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)
    write_jsonl(output_dir / "val.jsonl", eval_rows)
    write_jsonl(output_dir / "test.jsonl", test)

    metrics = {
        "source": str(input_path),
        "output_dir": str(output_dir),
        "source_rows": len(source_rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "generic_summary_rows": reason_counts.get("generic_summary", 0),
        "no_summary_rows": len(fallback_rows),
        "skipped_rows_after_filter": skipped,
        "truncated_rows": truncated,
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "test_rows": len(test),
        "vocab_size": tokenizer.get_vocab_size(),
        "max_seq_len": int(args.max_seq_len),
        "filters": {
            "min_email_words": args.min_email_words,
            "max_email_words": args.max_email_words,
            "min_summary_words": args.min_summary_words,
            "max_summary_words": args.max_summary_words,
            "max_summary_email_ratio": args.max_summary_email_ratio,
            "max_image_markers": args.max_image_markers,
            "marketing_word_limit": args.marketing_word_limit,
        },
        "rejection_reasons": dict(reason_counts),
        "artifacts": {
            "filtered_source": str(filtered_source),
            "rejected_source": str(rejected_source),
            "no_summary_source": str(generic_source),
            "tokenizer": str(tokenizer_path),
        },
    }

    (output_dir / "quality_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "tokenizer_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(output_dir, metrics, reason_counts)
    zip_path = write_training_zip(output_dir, args.zip_output)
    if zip_path:
        metrics["artifacts"]["training_zip"] = zip_path
        (output_dir / "quality_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (output_dir / "tokenizer_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quality-filtered guppyemail training data.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--val-pct", type=float, default=10)
    parser.add_argument("--test-pct", type=float, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-email-words", type=int, default=20)
    parser.add_argument("--max-email-words", type=int, default=1200)
    parser.add_argument("--min-summary-words", type=int, default=6)
    parser.add_argument("--max-summary-words", type=int, default=80)
    parser.add_argument("--max-summary-email-ratio", type=float, default=0.8)
    parser.add_argument("--max-image-markers", type=int, default=3)
    parser.add_argument("--marketing-word-limit", type=int, default=120)
    parser.add_argument("--zip-output", default="training_quality.zip", help="Colab-ready zip path. Use empty string to disable.")
    return parser.parse_args()


def main() -> None:
    metrics = prepare_quality_training_data(parse_args())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
