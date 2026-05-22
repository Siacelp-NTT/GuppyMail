"""
Generate report-ready analytics for the email summarization dataset.

Outputs:
    data/charts/report_*.png
    report/data_profile.md
    report/data_profile_metrics.json
    report/dataset_statistics.csv

Usage:
    python scripts/analyze_data.py
"""

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "enron_sample.jsonl"
CLEAN_PATH = DATA_DIR / "processed" / "cleaned_emails.jsonl"
CLEAN_PROFILE_PATH = DATA_DIR / "processed" / "cleaned_emails.profile.json"
SUM_PATH = DATA_DIR / "summaries" / "en_summaries.jsonl"
CLEAN_SUM_PATH = DATA_DIR / "summaries" / "en_summaries.clean.jsonl"
QUALITY_DIR = DATA_DIR / "training_quality"
QUALITY_FILTERED_PATH = QUALITY_DIR / "filtered_source.jsonl"
QUALITY_NO_SUMMARY_PATH = QUALITY_DIR / "no_summary_source.jsonl"
QUALITY_METRICS_PATH = QUALITY_DIR / "quality_metrics.json"
TRAIN_DIR = QUALITY_DIR if (QUALITY_DIR / "train.jsonl").exists() else DATA_DIR / "training"
TRAIN_PATH = TRAIN_DIR / "train.jsonl"
VAL_PATH = TRAIN_DIR / "val.jsonl"
TEST_PATH = TRAIN_DIR / "test.jsonl"
TOKENIZER_PATH = TRAIN_DIR / "tokenizer.json"
TOKENIZER_METRICS_PATH = TRAIN_DIR / "tokenizer_metrics.json"
CHARTS_DIR = DATA_DIR / "charts"
REPORT_DIR = BASE_DIR / "report"

STOPWORDS = {
    "the", "and", "for", "you", "that", "this", "with", "from", "have", "are",
    "not", "your", "will", "can", "all", "our", "has", "was", "but", "they",
    "his", "her", "she", "him", "about", "there", "their", "would", "could",
    "please", "thanks", "thank", "email", "message", "sent", "subject", "to",
    "of", "in", "on", "at", "is", "it", "be", "as", "or", "by", "if", "we",
}


def load_jsonl(path):
    """Load jsonl."""
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_json(path):
    """Load json."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def text_of(record):
    """Handle text of."""
    return (
        record.get("cleaned_body")
        or record.get("email")
        or record.get("text")
        or record.get("message")
        or ""
    )


def summary_of(record):
    """Handle summary of."""
    return record.get("summary", "")


def token_count_of(record):
    """Handle token count of."""
    value = record.get("token_count")
    if isinstance(value, int):
        return value
    return 0


def norm_text(text):
    """Handle norm text."""
    return re.sub(r"\s+", " ", text).strip().lower()


def hash_text(text):
    """Handle hash text."""
    return hashlib.md5(norm_text(text).encode("utf-8")).hexdigest()


def percentile(values, q):
    """Handle percentile."""
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(math.ceil(q * len(values))) - 1))
    return values[idx]


def stats(values):
    """Handle stats."""
    if not values:
        return {"count": 0, "min": 0, "median": 0, "mean": 0, "p95": 0, "max": 0}
    values = sorted(values)
    return {
        "count": len(values),
        "min": values[0],
        "median": values[len(values) // 2],
        "mean": round(sum(values) / len(values), 2),
        "p95": percentile(values, 0.95),
        "max": values[-1],
    }


def summary_flags(summary, max_chars=300):
    """Handle summary flags."""
    text = (summary or "").strip()
    lower = text.lower()
    flags = []
    if len(text) < 10:
        flags.append("too_short")
    if len(text) > max_chars:
        flags.append("too_long")
    if lower.startswith(("error:", "empty:")):
        flags.append("generation_error")
    if lower.startswith(("{", "[")) or "\"summary_type\"" in lower or '"quality"' in lower:
        flags.append("json_fragment")
    leakage_terms = [
        "under 50 words", "final:", "let's craft", "the summary should",
        "to be thorough", "need to keep it", "check:", "potential:",
        "we need to produce", "we need to summarize", "the email seems",
    ]
    if any(term in lower for term in leakage_terms):
        flags.append("reasoning_leak")
    words = lower.replace("\n", " ").split()
    if len(words) >= 12:
        repeats = sum(1 for a, b in zip(words, words[1:]) if a == b)
        counts = Counter(words)
        if repeats >= 3 or max(counts.values(), default=0) >= 8:
            flags.append("repetition")
    return flags


def save_fig(fig, name):
    """Save fig."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def style_axes(ax):
    """Handle style axes."""
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def chart_pipeline(counts):
    """Create the pipeline chart."""
    labels = ["Raw", "Cleaned unique", "Summarized", "Quality filtered", "Train+Val+Test"]
    values = [
        counts["raw"],
        counts["cleaned_unique"],
        counts["summaries"],
        counts["quality_filtered"],
        counts["train"] + counts["val"] + counts["test"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#334155", "#0f766e", "#2563eb", "#16a34a", "#ca8a04"]
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Dataset Pipeline Counts")
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax)
    return save_fig(fig, "report_pipeline_counts.png")


def chart_lengths(raw, cleaned, clean_summaries):
    """Create the lengths chart."""
    raw_lens = [len(text_of(r)) for r in raw]
    clean_lens = [len(text_of(r)) for r in cleaned]
    sum_lens = [len(summary_of(r)) for r in clean_summaries]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, values, title, color in [
        (axes[0], raw_lens, "Raw Email Lengths", "#64748b"),
        (axes[1], clean_lens, "Cleaned Email Lengths", "#0f766e"),
        (axes[2], sum_lens, "Summary Lengths", "#2563eb"),
    ]:
        ax.hist(values, bins=50, color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Characters")
        ax.set_ylabel("Count")
        style_axes(ax)
    return save_fig(fig, "report_length_distributions.png")


def chart_cleaning_retention(cleaned):
    """Create the cleaning retention chart."""
    ratios = []
    original = []
    cleaned_len = []
    for r in cleaned:
        o = r.get("original_length") or len(r.get("original_text", ""))
        c = r.get("cleaned_length") or len(text_of(r))
        if o:
            ratios.append(min(c / o, 1.5))
            original.append(o)
            cleaned_len.append(c)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(ratios, bins=40, color="#0f766e", alpha=0.85)
    axes[0].set_title("Cleaning Retention Ratio")
    axes[0].set_xlabel("cleaned_length / original_length")
    axes[0].set_ylabel("Emails")
    style_axes(axes[0])
    axes[1].scatter(original[:8000], cleaned_len[:8000], s=6, alpha=0.25, color="#2563eb")
    axes[1].set_title("Original vs Cleaned Length")
    axes[1].set_xlabel("Original characters")
    axes[1].set_ylabel("Cleaned characters")
    style_axes(axes[1])
    return save_fig(fig, "report_cleaning_retention.png")


def chart_summary_relationship(clean_summaries):
    """Create the summary relationship chart."""
    x = [len(text_of(r)) for r in clean_summaries]
    y = [len(summary_of(r)) for r in clean_summaries]
    ratios = [len(summary_of(r)) / max(len(text_of(r)), 1) for r in clean_summaries]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(x[:10000], y[:10000], s=6, alpha=0.25, color="#7c3aed")
    axes[0].set_title("Input Length vs Summary Length")
    axes[0].set_xlabel("Cleaned email characters")
    axes[0].set_ylabel("Summary characters")
    style_axes(axes[0])
    axes[1].hist(ratios, bins=50, color="#ca8a04", alpha=0.85)
    axes[1].set_title("Compression Ratio")
    axes[1].set_xlabel("summary_length / cleaned_length")
    axes[1].set_ylabel("Records")
    style_axes(axes[1])
    return save_fig(fig, "report_summary_compression.png")


def chart_quality(summaries):
    """Create the quality chart."""
    flag_counter = Counter()
    for r in summaries:
        for flag in summary_flags(summary_of(r)):
            flag_counter[flag] += 1
    labels = list(flag_counter.keys()) or ["no_flags"]
    values = [flag_counter[k] for k in labels] or [0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="#dc2626")
    ax.set_title("Summary Quality Flags")
    ax.set_ylabel("Flagged rows")
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax)
    return save_fig(fig, "report_quality_flags.png")


def chart_summary_label_quality(summaries, quality_metrics):
    """Create the summary label quality chart."""
    type_counts = Counter(str(r.get("summary_type") or "missing").strip().lower() for r in summaries)
    quality_counts = Counter(str(r.get("summary_quality") or r.get("quality") or "missing").strip().lower() for r in summaries)
    training_counts = {
        "accepted": int(quality_metrics.get("accepted_rows") or 0),
        "rejected": int(quality_metrics.get("rejected_rows") or 0),
        "fallback": int(quality_metrics.get("no_summary_rows") or 0),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [
        (axes[0], type_counts, "Summary Type Labels", "#2563eb"),
        (axes[1], quality_counts, "Summary Quality Labels", "#16a34a"),
        (axes[2], training_counts, "Quality Builder Outcome", "#ca8a04"),
    ]

    for ax, counts, title, color in panels:
        labels = list(counts.keys()) or ["none"]
        values = [counts[label] for label in labels] or [0]
        bars = ax.bar(labels, values, color=color, alpha=0.9)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(title)
        ax.set_ylabel("Rows")
        ax.tick_params(axis="x", rotation=25)
        style_axes(ax)

    return save_fig(fig, "report_summary_label_quality.png")


def chart_splits(train, val, test):
    """Create the splits chart."""
    splits = {"train": train, "val": val, "test": test}
    labels = list(splits)
    counts = [len(splits[k]) for k in labels]
    med_text = [stats([len(text_of(r)) for r in splits[k]])["median"] for k in labels]
    has_token_counts = any(token_count_of(r) for rows in splits.values() for r in rows)
    if has_token_counts:
        med_second = [stats([token_count_of(r) for r in splits[k] if token_count_of(r)])["median"] for k in labels]
        second_label = "tokens"
    else:
        med_second = [stats([len(summary_of(r)) for r in splits[k]])["median"] for k in labels]
        second_label = "summary"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, counts, color=["#16a34a", "#2563eb", "#ca8a04"])
    axes[0].set_title("Training Split Sizes")
    axes[0].set_ylabel("Rows")
    style_axes(axes[0])
    x = range(len(labels))
    axes[1].bar([i - 0.18 for i in x], med_text, width=0.36, label="text chars", color="#0f766e")
    axes[1].bar([i + 0.18 for i in x], med_second, width=0.36, label=second_label, color="#7c3aed")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_title("Median Size by Split")
    axes[1].set_ylabel("Characters / tokens")
    axes[1].legend()
    style_axes(axes[1])
    return save_fig(fig, "report_split_balance.png")


def chart_duplicates(raw, cleaned, summaries):
    """Create the duplicates chart."""
    datasets = {
        "raw": raw,
        "cleaned": cleaned,
        "summaries": summaries,
    }
    exact = []
    prefix = []
    labels = []
    for name, rows in datasets.items():
        labels.append(name)
        texts = [text_of(r) for r in rows]
        exact_hashes = [hash_text(t) for t in texts]
        prefix_hashes = [hash_text(norm_text(t)[:200]) for t in texts]
        exact.append(len(exact_hashes) - len(set(exact_hashes)))
        prefix.append(len(prefix_hashes) - len(set(prefix_hashes)))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    ax.bar([i - 0.18 for i in x], exact, width=0.36, label="exact normalized", color="#2563eb")
    ax.bar([i + 0.18 for i in x], prefix, width=0.36, label="first-200 near duplicate", color="#f97316")
    ax.set_xticks(list(x), labels)
    ax.set_title("Duplicate and Near-Duplicate Diagnostics")
    ax.set_ylabel("Rows beyond first occurrence")
    ax.legend()
    style_axes(ax)
    return save_fig(fig, "report_duplicate_diagnostics.png")


def chart_top_terms(clean_summaries):
    """Create the top terms chart."""
    counter = Counter()
    for r in clean_summaries:
        words = re.findall(r"[a-zA-Z]{4,}", summary_of(r).lower())
        counter.update(w for w in words if w not in STOPWORDS)
    common = counter.most_common(20)
    labels = [w for w, _ in common][::-1]
    values = [c for _, c in common][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, values, color="#0f766e")
    ax.set_title("Most Common Summary Terms")
    ax.set_xlabel("Frequency")
    style_axes(ax)
    return save_fig(fig, "report_top_summary_terms.png")


def chart_tokenizer_lengths(train, val, test, tokenization):
    """Create the tokenizer lengths chart."""
    splits = {"train": train, "eval": val, "test": test}
    split_lengths = {name: [token_count_of(r) for r in rows if token_count_of(r)] for name, rows in splits.items()}
    if not any(split_lengths.values()):
        return save_fig(plt.figure(figsize=(7, 4)), "report_tokenizer_lengths.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"train": "#16a34a", "eval": "#2563eb", "test": "#ca8a04"}
    for split, lengths in split_lengths.items():
        if lengths:
            axes[0].hist(lengths, bins=40, alpha=0.45, label=split, color=colors[split])
    max_seq_len = tokenization.get("max_seq_len", 0)
    if max_seq_len:
        axes[0].axvline(max_seq_len, color="#dc2626", linestyle="--", linewidth=1.5, label=f"max={max_seq_len}")
    axes[0].set_title("Tokenized Sample Lengths")
    axes[0].set_xlabel("Tokens per ChatML sample")
    axes[0].set_ylabel("Rows")
    axes[0].legend()
    style_axes(axes[0])

    labels = list(splits)
    rates = []
    for split, rows in splits.items():
        truncated = sum(1 for r in rows if r.get("truncated"))
        rates.append((truncated / len(rows) * 100) if rows else 0)
    axes[1].bar(labels, rates, color=[colors[label] for label in labels])
    axes[1].set_title("Truncation Rate by Split")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Rows truncated (%)")
    style_axes(axes[1])
    return save_fig(fig, "report_tokenizer_lengths.png")


def write_statistics_csv(metrics):
    """Write statistics csv."""
    path = REPORT_DIR / "dataset_statistics.csv"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    writer.writerow([f"{key}.{sub_key}", sub_value])
            else:
                writer.writerow([key, value])
    return str(path)


def write_markdown(metrics, charts):
    """Write markdown."""
    path = REPORT_DIR / "data_profile.md"
    label_quality = metrics.get("summary_label_quality", {})
    type_counts = label_quality.get("summary_type_counts", {})
    quality_counts = label_quality.get("summary_quality_counts", {})
    builder = label_quality.get("quality_builder", {})
    lines = [
        "# Data Profile",
        "",
        "## Dataset Counts",
        "",
        "| Stage | Rows |",
        "|---|---:|",
        f"| Raw sampled emails | {metrics['counts']['raw']:,} |",
        f"| Cleaned emails | {metrics['counts']['cleaned']:,} |",
        f"| Canonical unique cleaned emails | {metrics['counts']['cleaned_unique']:,} |",
        f"| Generated summaries | {metrics['counts']['summaries']:,} |",
        f"| Clean summary rows | {metrics['counts']['clean_summaries']:,} |",
        f"| Quality filtered summary pairs | {metrics['counts']['quality_filtered']:,} |",
        f"| No-summary fallback rows | {metrics['counts']['no_summary']:,} |",
        f"| Train / validation / test | {metrics['counts']['train']:,} / {metrics['counts']['val']:,} / {metrics['counts']['test']:,} |",
        "",
        "## Key Statistics",
        "",
        "| Measure | Median | Mean | P95 | Max |",
        "|---|---:|---:|---:|---:|",
        f"| Cleaned email length | {metrics['cleaned_length']['median']:,} | {metrics['cleaned_length']['mean']:,} | {metrics['cleaned_length']['p95']:,} | {metrics['cleaned_length']['max']:,} |",
        f"| Summary length | {metrics['summary_length']['median']:,} | {metrics['summary_length']['mean']:,} | {metrics['summary_length']['p95']:,} | {metrics['summary_length']['max']:,} |",
        f"| Summary words | {metrics['summary_words']['median']:,} | {metrics['summary_words']['mean']:,} | {metrics['summary_words']['p95']:,} | {metrics['summary_words']['max']:,} |",
        "",
        "## Quality Notes",
        "",
        f"- Clean summary coverage: {metrics['coverage_pct']}% of canonical unique cleaned emails.",
        f"- Duplicates removed during cleaning: {metrics['duplicates_removed_during_cleaning']:,}.",
        f"- Exact duplicate rows still present in cleaned file: {metrics['cleaned_exact_duplicates']:,}.",
        f"- Flagged generated summaries: {metrics['flagged_summaries']:,}.",
        f"- Cleaned first-200 near-duplicate diagnostic rows: {metrics['cleaned_prefix_duplicates']:,}.",
        f"- Median compression ratio: {metrics['compression_ratio']['median']}.",
        f"- Tokenizer vocabulary size: {metrics['tokenization']['vocab_size']:,}.",
        f"- Tokenized max sequence length: {metrics['tokenization']['max_seq_len']:,}.",
        f"- Tokenized rows truncated: {metrics['tokenization']['truncated_rows']:,} ({metrics['tokenization']['truncated_pct']}%).",
        f"- Training artifact source: `{metrics['training_artifact_dir']}`.",
        "",
        "## Summary Label Quality",
        "",
        f"- Summary labels: good/trainable `{type_counts.get('summary', 0):,}`, no-summary `{type_counts.get('no_summary_needed', 0):,}`, noise `{type_counts.get('noise', 0):,}`, malformed `{type_counts.get('malformed', 0):,}`.",
        f"- Quality labels: good `{quality_counts.get('good', 0):,}`, weak `{quality_counts.get('weak', 0):,}`, noise `{quality_counts.get('noise', 0):,}`.",
        f"- Quality builder outcome: accepted `{builder.get('accepted_rows', 0):,}`, rejected `{builder.get('rejected_rows', 0):,}`, fallback/no-summary `{builder.get('no_summary_rows', 0):,}`.",
        "",
        "## Generated Figures",
        "",
    ]
    for chart in charts:
        rel = Path(chart).relative_to(BASE_DIR)
        lines.append(f"- `{rel}`")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def build_metrics(raw, cleaned, summaries, clean_summaries, quality_filtered, no_summary, train, val, test, clean_profile, quality_metrics):
    """Build metrics."""
    summary_lengths = [len(summary_of(r)) for r in quality_filtered]
    summary_words = [len(summary_of(r).split()) for r in quality_filtered]
    cleaned_lengths = [len(text_of(r)) for r in cleaned]
    compression = [len(summary_of(r)) / max(len(text_of(r)), 1) for r in quality_filtered]
    cleaned_exact_hashes = [hash_text(text_of(r)) for r in cleaned]
    cleaned_prefix_hashes = [hash_text(norm_text(text_of(r))[:200]) for r in cleaned]
    cleaned_unique_count = len(set(cleaned_exact_hashes))
    cleaned_exact_duplicates = len(cleaned_exact_hashes) - cleaned_unique_count
    flagged = sum(1 for r in summaries if summary_flags(summary_of(r)))
    coverage = (len(quality_filtered) / cleaned_unique_count * 100) if cleaned_unique_count else 0
    duplicates_removed = int(clean_profile.get("duplicate_rows_removed") or 0)
    tokenizer_metrics = {}
    if TOKENIZER_METRICS_PATH.exists():
        try:
            tokenizer_metrics = json.loads(TOKENIZER_METRICS_PATH.read_text())
        except json.JSONDecodeError:
            tokenizer_metrics = {}
    token_lengths = [token_count_of(r) for r in train + val + test if token_count_of(r)]
    split_token_lengths = {
        "train": [token_count_of(r) for r in train if token_count_of(r)],
        "val": [token_count_of(r) for r in val if token_count_of(r)],
        "test": [token_count_of(r) for r in test if token_count_of(r)],
    }
    truncated_rows = sum(1 for r in train + val + test if r.get("truncated"))
    total_tokenized = len(train) + len(val) + len(test)
    summary_type_counts = Counter(str(r.get("summary_type") or "missing").strip().lower() for r in summaries)
    summary_quality_counts = Counter(str(r.get("summary_quality") or r.get("quality") or "missing").strip().lower() for r in summaries)
    return {
        "counts": {
            "raw": len(raw),
            "cleaned": len(cleaned),
            "cleaned_unique": cleaned_unique_count,
            "cleaned_exact_duplicates": cleaned_exact_duplicates,
            "summaries": len(summaries),
            "clean_summaries": len(clean_summaries),
            "quality_filtered": len(quality_filtered),
            "no_summary": len(no_summary),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "training_artifact_dir": str(TRAIN_DIR.relative_to(BASE_DIR)),
        "cleaned_length": stats(cleaned_lengths),
        "summary_length": stats(summary_lengths),
        "summary_words": stats(summary_words),
        "compression_ratio": stats([round(v, 4) for v in compression]),
        "coverage_pct": round(coverage, 2),
        "duplicates_removed_during_cleaning": duplicates_removed,
        "cleaned_exact_duplicates": cleaned_exact_duplicates,
        "flagged_summaries": flagged,
        "cleaned_prefix_duplicates": len(cleaned_prefix_hashes) - len(set(cleaned_prefix_hashes)),
        "summary_label_quality": {
            "summary_type_counts": dict(summary_type_counts),
            "summary_quality_counts": dict(summary_quality_counts),
            "quality_builder": {
                "accepted_rows": int(quality_metrics.get("accepted_rows") or len(quality_filtered)),
                "rejected_rows": int(quality_metrics.get("rejected_rows") or 0),
                "no_summary_rows": int(quality_metrics.get("no_summary_rows") or len(no_summary)),
                "generic_summary_rows": int(quality_metrics.get("generic_summary_rows") or 0),
            },
        },
        "clean_profile": {
            "raw_rows": int(clean_profile.get("raw_rows") or 0),
            "skipped_rows": int(clean_profile.get("skipped_rows") or 0),
            "dedupe_enabled": bool(clean_profile.get("dedupe_enabled", False)),
            "dedupe_hash": clean_profile.get("dedupe_hash", ""),
        },
        "tokenization": {
            "vocab_size": int(tokenizer_metrics.get("vocab_size") or 0),
            "max_seq_len": int(tokenizer_metrics.get("max_seq_len") or 0),
            "rows": total_tokenized,
            "truncated_rows": truncated_rows,
            "truncated_pct": round((truncated_rows / total_tokenized * 100) if total_tokenized else 0, 2),
            "token_length": stats(token_lengths),
            "train_token_length": stats(split_token_lengths["train"]),
            "val_token_length": stats(split_token_lengths["val"]),
            "test_token_length": stats(split_token_lengths["test"]),
        },
    }


def generate_report():
    """Generate report."""
    raw = load_jsonl(RAW_PATH)
    cleaned = load_jsonl(CLEAN_PATH)
    clean_profile = load_json(CLEAN_PROFILE_PATH)
    summaries = load_jsonl(SUM_PATH)
    clean_summaries = load_jsonl(CLEAN_SUM_PATH)
    quality_filtered = load_jsonl(QUALITY_FILTERED_PATH) or clean_summaries
    no_summary = load_jsonl(QUALITY_NO_SUMMARY_PATH)
    quality_metrics = load_json(QUALITY_METRICS_PATH)
    train = load_jsonl(TRAIN_PATH)
    val = load_jsonl(VAL_PATH)
    test = load_jsonl(TEST_PATH)

    metrics = build_metrics(raw, cleaned, summaries, clean_summaries, quality_filtered, no_summary, train, val, test, clean_profile, quality_metrics)
    charts = [
        chart_pipeline(metrics["counts"]),
        chart_lengths(raw, cleaned, quality_filtered),
        chart_cleaning_retention(cleaned),
        chart_summary_relationship(quality_filtered),
        chart_summary_label_quality(summaries, quality_metrics),
        chart_quality(summaries),
        chart_splits(train, val, test),
        chart_duplicates(raw, cleaned, summaries),
        chart_top_terms(quality_filtered),
        chart_tokenizer_lengths(train, val, test, metrics["tokenization"]),
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "data_profile_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    csv_path = write_statistics_csv(metrics)
    md_path = write_markdown(metrics, charts)

    return {
        "metrics": metrics,
        "charts": charts,
        "markdown": md_path,
        "json": str(metrics_path),
        "csv": csv_path,
    }


def main():
    """Run the command-line entry point."""
    result = generate_report()
    print("Generated analytics report")
    print(f"Markdown: {result['markdown']}")
    print(f"Metrics:  {result['json']}")
    print(f"CSV:      {result['csv']}")
    for chart in result["charts"]:
        print(f"Chart:    {chart}")


if __name__ == "__main__":
    main()
