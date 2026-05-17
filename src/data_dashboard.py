"""
GuppyLM Data Dashboard — Control panel for the email data pipeline.

Tabs: Pipeline · Data · Quality · Charts · Summarization · System

Usage:
    python src/data_dashboard.py
"""

import json
import hashlib
import os
import psutil
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import subprocess
import sys

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import gradio as gr

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
SUM_DIR = DATA_DIR / "summaries"
TRAIN_DIR = DATA_DIR / "training"
CHARTS_DIR = DATA_DIR / "charts"
STATE_FILE = DATA_DIR / "pipeline_state.json"
SETTINGS_FILE = DATA_DIR / "settings.json"

for d in [DATA_DIR, RAW_DIR, PROC_DIR, SUM_DIR, TRAIN_DIR, CHARTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── State ───────────────────────────────────────────────────────────────────

STAGES = ["download", "clean", "summarize", "format", "train"]

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"stages": {}, "started": None}

def save_state(state):
    state["updated"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def init_state():
    state = load_state()
    for s in STAGES:
        if s not in state["stages"]:
            state["stages"][s] = {"status": "pending", "count": 0, "time": None}
    if not state.get("started"):
        state["started"] = datetime.now().isoformat()
    save_state(state)
    return state

# ─── Settings ────────────────────────────────────────────────────────────────

def load_settings():
    defaults = {"show_summarization": False}
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            defaults.update(json.load(f))
    return defaults

def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)

# ─── Data helpers ────────────────────────────────────────────────────────────

def count_jsonl(path):
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)

def load_jsonl(path, limit=None):
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out

def email_hash(text):
    return hashlib.md5(text.strip().encode()).hexdigest()

def deduplicate_records(records, existing_hashes):
    unique = []
    skipped = 0
    for r in records:
        text = r.get("text", r.get("cleaned_body", ""))
        h = email_hash(text)
        if h not in existing_hashes:
            existing_hashes.add(h)
            unique.append(r)
        else:
            skipped += 1
    return unique, skipped

def get_existing_hashes(path):
    hashes = set()
    if not path.exists():
        return hashes
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                text = r.get("text", r.get("cleaned_body", ""))
                hashes.add(email_hash(text))
            except (json.JSONDecodeError, KeyError):
                continue
    return hashes

def get_counts():
    return {
        "raw": count_jsonl(RAW_DIR / "enron_sample.jsonl"),
        "cleaned": count_jsonl(PROC_DIR / "cleaned_emails.jsonl"),
        "summarized": count_jsonl(SUM_DIR / "en_summaries.jsonl"),
        "training": count_jsonl(TRAIN_DIR / "train.jsonl") if (TRAIN_DIR / "train.jsonl").exists() else 0,
    }

def pipeline_count_markdowns():
    counts = get_counts()
    return (
        f"**{counts['raw']:,}**",
        f"**{counts['cleaned']:,}**",
        f"**{counts['summarized']:,}**",
        f"**{counts['training']:,}**",
    )

def get_raw_stats():
    p = RAW_DIR / "enron_stats.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def record_text(record):
    return record.get("cleaned_body") or record.get("text") or record.get("message") or ""

def record_hash(record):
    return email_hash(record_text(record)[:200])

def summary_flags(summary, max_chars=300):
    text = (summary or "").strip()
    lower = text.lower()
    flags = []

    if len(text) < 10:
        flags.append("too_short")
    if len(text) > int(max_chars):
        flags.append("too_long")
    if lower.startswith(("error:", "empty:")):
        flags.append("generation_error")

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
        if repeats >= 3:
            flags.append("repetition")
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        if max(counts.values(), default=0) >= 8:
            flags.append("repetition")

    return flags

def quality_snapshot(max_summary_chars=300):
    raw = load_jsonl(RAW_DIR / "enron_sample.jsonl")
    cleaned = load_jsonl(PROC_DIR / "cleaned_emails.jsonl")
    summaries = load_jsonl(SUM_DIR / "en_summaries.jsonl")

    raw_hashes = [record_hash(r) for r in raw]
    clean_hashes = [record_hash(r) for r in cleaned]
    summary_hashes = [record_hash(r) for r in summaries]
    clean_unique = set(clean_hashes)
    summary_unique = set(summary_hashes)

    bad_rows = []
    for i, r in enumerate(summaries):
        flags = summary_flags(r.get("summary", ""), max_chars=max_summary_chars)
        if flags:
            bad_rows.append({
                "index": i,
                "flags": ",".join(sorted(set(flags))),
                "subject": r.get("subject", "")[:80],
                "summary_len": len(r.get("summary", "")),
                "summary": r.get("summary", "")[:180].replace("\n", " "),
            })

    rows = [
        ["Raw rows", f"{len(raw):,}"],
        ["Cleaned rows", f"{len(cleaned):,}"],
        ["Summary rows", f"{len(summaries):,}"],
        ["Raw duplicate-ish rows", f"{len(raw_hashes) - len(set(raw_hashes)):,}"],
        ["Cleaned duplicate-ish rows", f"{len(clean_hashes) - len(clean_unique):,}"],
        ["Summary duplicate-ish rows", f"{len(summary_hashes) - len(summary_unique):,}"],
        ["Unique cleaned covered by summaries", f"{len(clean_unique & summary_unique):,}/{len(clean_unique):,}"],
        ["Missing unique cleaned summaries", f"{len(clean_unique - summary_unique):,}"],
        ["Flagged summary rows", f"{len(bad_rows):,}"],
    ]
    return rows, bad_rows, {
        "raw": raw,
        "cleaned": cleaned,
        "summaries": summaries,
        "clean_unique": clean_unique,
        "summary_unique": summary_unique,
    }

def run_quality_scan(max_summary_chars=300):
    rows, bad_rows, _ = quality_snapshot(max_summary_chars)
    table = [[r["index"], r["flags"], r["summary_len"], r["subject"], r["summary"]] for r in bad_rows[:100]]
    note = (
        f"Scanned current JSONL files. Showing first {min(len(table), 100):,} "
        f"of {len(bad_rows):,} flagged summaries."
    )
    return rows, table, note

def export_bad_summaries(max_summary_chars=300):
    _, bad_rows, snapshot = quality_snapshot(max_summary_chars)
    bad_indexes = {r["index"] for r in bad_rows}
    records = [r for i, r in enumerate(snapshot["summaries"]) if i in bad_indexes]
    out = SUM_DIR / "bad_summaries.jsonl"
    write_jsonl(out, records)
    return f"Wrote {len(records):,} flagged rows to {out.relative_to(BASE_DIR)}"

def build_clean_summaries(max_summary_chars=300):
    summaries = load_jsonl(SUM_DIR / "en_summaries.jsonl")
    seen = set()
    clean = []
    skipped_bad = 0
    skipped_dupe = 0

    for r in summaries:
        h = record_hash(r)
        if h in seen:
            skipped_dupe += 1
            continue
        seen.add(h)

        if summary_flags(r.get("summary", ""), max_chars=max_summary_chars):
            skipped_bad += 1
            continue
        clean.append(r)

    out = SUM_DIR / "en_summaries.clean.jsonl"
    write_jsonl(out, clean)
    return (
        f"Wrote {len(clean):,} clean summaries to {out.relative_to(BASE_DIR)}. "
        f"Skipped {skipped_dupe:,} duplicates and {skipped_bad:,} flagged summaries."
    )

def export_missing_cleaned():
    cleaned = load_jsonl(PROC_DIR / "cleaned_emails.jsonl")
    summaries = load_jsonl(SUM_DIR / "en_summaries.jsonl")
    summary_hashes = {record_hash(r) for r in summaries}
    seen = set()
    missing = []

    for r in cleaned:
        h = record_hash(r)
        if h in seen:
            continue
        seen.add(h)
        if h not in summary_hashes:
            missing.append(r)

    out = PROC_DIR / "missing_summaries.jsonl"
    write_jsonl(out, missing)
    return f"Wrote {len(missing):,} unique cleaned rows missing summaries to {out.relative_to(BASE_DIR)}."

def build_unique_cleaned():
    cleaned = load_jsonl(PROC_DIR / "cleaned_emails.jsonl")
    seen = set()
    unique = []

    for r in cleaned:
        h = record_hash(r)
        if h in seen:
            continue
        seen.add(h)
        unique.append(r)

    out = PROC_DIR / "cleaned_emails.unique.jsonl"
    write_jsonl(out, unique)
    return f"Wrote {len(unique):,} unique cleaned rows to {out.relative_to(BASE_DIR)}."

def generate_training_files(source_name="clean summaries", val_pct=10, test_pct=5, seed=42, max_summary_chars=300):
    source = SUM_DIR / "en_summaries.clean.jsonl"
    if source_name == "all summaries" or not source.exists():
        source = SUM_DIR / "en_summaries.jsonl"

    records = load_jsonl(source)
    formatted = []
    seen = set()
    for r in records:
        h = record_hash(r)
        if h in seen:
            continue
        seen.add(h)

        email = r.get("cleaned_body", "").strip()
        summary = r.get("summary", "").strip()
        if not email or not summary or summary_flags(summary, max_chars=max_summary_chars):
            continue
        formatted.append({
            "email": email,
            "summary": summary,
            "subject": r.get("subject", ""),
            "source_hash": h,
        })

    rng = random.Random(int(seed))
    rng.shuffle(formatted)

    total = len(formatted)
    test_n = int(total * float(test_pct) / 100)
    val_n = int(total * float(val_pct) / 100)
    test = formatted[:test_n]
    val = formatted[test_n:test_n + val_n]
    train = formatted[test_n + val_n:]

    write_jsonl(TRAIN_DIR / "train.jsonl", train)
    write_jsonl(TRAIN_DIR / "val.jsonl", val)
    write_jsonl(TRAIN_DIR / "test.jsonl", test)

    state = init_state()
    state["stages"]["format"] = {"status": "done", "count": total, "time": datetime.now().isoformat()}
    save_state(state)

    return (
        f"Generated training splits from {source.relative_to(BASE_DIR)}: "
        f"train={len(train):,}, val={len(val):,}, test={len(test):,}."
    )

def sync_pipeline_state():
    counts = get_counts()
    state = init_state()
    state["stages"]["download"] = {
        "status": "done" if counts["raw"] else "pending",
        "count": counts["raw"],
        "time": datetime.now().isoformat(),
    }
    state["stages"]["clean"] = {
        "status": "done" if counts["cleaned"] else "pending",
        "count": counts["cleaned"],
        "time": datetime.now().isoformat(),
    }
    state["stages"]["summarize"] = {
        "status": "done" if counts["summarized"] and counts["summarized"] >= counts["cleaned"] else ("partial" if counts["summarized"] else "pending"),
        "count": counts["summarized"],
        "time": datetime.now().isoformat(),
    }
    state["stages"]["format"] = {
        "status": "done" if counts["training"] else "pending",
        "count": counts["training"],
        "time": datetime.now().isoformat(),
    }
    save_state(state)
    return f"Synced state from files: raw={counts['raw']:,}, cleaned={counts['cleaned']:,}, summaries={counts['summarized']:,}, training={counts['training']:,}."

# ─── Hardware ────────────────────────────────────────────────────────────────

def get_hw():
    cpu = psutil.cpu_percent(interval=0.3)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    out = {
        "cpu": cpu,
        "ram_gb": round(ram.used / 1e9, 1),
        "ram_total_gb": round(ram.total / 1e9, 1),
        "ram_pct": ram.percent,
        "disk_gb": round(disk.used / 1e9, 1),
        "disk_total_gb": round(disk.total / 1e9, 1),
        "disk_pct": disk.percent,
    }
    if HAS_TORCH and torch.cuda.is_available():
        try:
            out["gpu"] = torch.cuda.utilization(0)
            out["gpu_mem"] = round(torch.cuda.memory_allocated(0) / 1e9, 1)
            out["gpu_total"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        except Exception:
            out["gpu"] = 0
            out["gpu_mem"] = 0
            out["gpu_total"] = 0
    else:
        out["gpu"] = None
    return out

# ─── Charts ──────────────────────────────────────────────────────────────────

STYLE = {
    "bg": "#16161e",
    "fg": "#c9cdd8",
    "accent": "#00d4aa",
    "accent2": "#7c3aed",
    "grid": "#2a2a3e",
}

def apply_style(ax):
    ax.set_facecolor(STYLE["bg"])
    for spine in ax.spines.values():
        spine.set_color(STYLE["grid"])
    ax.tick_params(colors=STYLE["fg"], labelsize=9)
    ax.xaxis.label.set_color(STYLE["fg"])
    ax.yaxis.label.set_color(STYLE["fg"])
    ax.title.set_color(STYLE["fg"])
    ax.title.set_fontsize(11)
    ax.grid(True, alpha=0.1, color=STYLE["grid"])

def chart_length_dist():
    records = load_jsonl(PROC_DIR / "cleaned_emails.jsonl", limit=5000)
    key = "cleaned_body"
    if not records:
        records = load_jsonl(RAW_DIR / "enron_sample.jsonl", limit=5000)
        key = "text"
    if not records:
        return None
    lengths = [len(r.get(key, "")) for r in records]
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=STYLE["bg"])
    ax.hist(lengths, bins=40, color=STYLE["accent"], edgecolor=STYLE["bg"], alpha=0.75, linewidth=0.5)
    ax.set_xlabel("Length (chars)")
    ax.set_ylabel("Count")
    ax.set_title("Length Distribution")
    apply_style(ax)
    fig.tight_layout()
    p = str(CHARTS_DIR / "length_dist.png")
    fig.savefig(p, dpi=120, facecolor=STYLE["bg"])
    plt.close(fig)
    return p

def chart_funnel():
    c = get_counts()
    labels = ["Raw", "Cleaned", "Summarized", "Training"]
    vals = [c["raw"], c["cleaned"], c["summarized"], c["training"]]
    if sum(vals) == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 2.5), facecolor=STYLE["bg"])
    colors = [STYLE["accent"], STYLE["accent2"], "#e07a5f", "#81b29a"]
    bars = ax.barh(labels, vals, color=colors, height=0.45, edgecolor=STYLE["bg"])
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_width() + max(vals) * 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{v:,}", va="center", color=STYLE["fg"], fontsize=9, fontweight="500")
    ax.set_title("Pipeline")
    apply_style(ax)
    ax.set_xticks([])
    fig.tight_layout()
    p = str(CHARTS_DIR / "funnel.png")
    fig.savefig(p, dpi=120, facecolor=STYLE["bg"])
    plt.close(fig)
    return p

def chart_summary_len():
    records = load_jsonl(SUM_DIR / "en_summaries.jsonl", limit=5000)
    if not records:
        return None
    lengths = [len(r.get("summary", "")) for r in records]
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=STYLE["bg"])
    ax.hist(lengths, bins=30, color=STYLE["accent2"], edgecolor=STYLE["bg"], alpha=0.75, linewidth=0.5)
    ax.set_xlabel("Summary Length (chars)")
    ax.set_ylabel("Count")
    ax.set_title("Summary Lengths")
    apply_style(ax)
    fig.tight_layout()
    p = str(CHARTS_DIR / "summary_len.png")
    fig.savefig(p, dpi=120, facecolor=STYLE["bg"])
    plt.close(fig)
    return p

def render_charts():
    return chart_length_dist(), chart_funnel(), chart_summary_len()

REPORT_CHART_FILES = [
    CHARTS_DIR / "report_pipeline_counts.png",
    CHARTS_DIR / "report_length_distributions.png",
    CHARTS_DIR / "report_cleaning_retention.png",
    CHARTS_DIR / "report_summary_compression.png",
    CHARTS_DIR / "report_quality_flags.png",
    CHARTS_DIR / "report_split_balance.png",
    CHARTS_DIR / "report_duplicate_diagnostics.png",
    CHARTS_DIR / "report_top_summary_terms.png",
]

def report_chart_paths():
    return [str(p) if p.exists() else None for p in REPORT_CHART_FILES]

def generate_full_analytics():
    script = BASE_DIR / "scripts" / "analyze_data.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired:
        return ("Analytics generation timed out after 240 seconds.", *report_chart_paths())
    if result.returncode != 0:
        return (f"Analytics generation failed:\n{result.stderr.strip()}", *report_chart_paths())
    return (result.stdout.strip(), *report_chart_paths())

# ─── Pipeline actions ────────────────────────────────────────────────────────

def run_download(n_emails):
    n = int(n_emails)
    state = init_state()
    state["stages"]["download"] = {"status": "running", "count": 0, "time": datetime.now().isoformat()}
    save_state(state)

    script = BASE_DIR / "scripts" / "download_enron.py"
    if not script.exists():
        state["stages"]["download"] = {"status": "error", "count": 0, "time": "Script not found"}
        save_state(state)
        return "Script not found: scripts/download_enron.py"

    env = os.environ.copy()
    env["SAMPLE_SIZE"] = str(n)

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=600,
            env=env
        )
        count = get_counts()["raw"]
        state["stages"]["download"] = {"status": "done", "count": count, "time": datetime.now().isoformat()}
        save_state(state)
        out = result.stdout.strip()
        return out if result.returncode == 0 else f"Error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        state["stages"]["download"] = {"status": "timeout", "count": 0, "time": "Timed out"}
        save_state(state)
        return "Download timed out (10 min limit)"
    except Exception as e:
        state["stages"]["download"] = {"status": "error", "count": 0, "time": str(e)}
        save_state(state)
        return f"Error: {e}"

def run_clean():
    state = init_state()
    state["stages"]["clean"] = {"status": "running", "count": 0, "time": datetime.now().isoformat()}
    save_state(state)

    script = BASE_DIR / "src" / "preprocess.py"
    if not script.exists():
        state["stages"]["clean"] = {"status": "error", "count": 0, "time": "Script not found"}
        save_state(state)
        return "Script not found: src/preprocess.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=600
        )
        count = get_counts()["cleaned"]
        state["stages"]["clean"] = {"status": "done", "count": count, "time": datetime.now().isoformat()}
        save_state(state)
        out = result.stdout.strip()
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        derived = "\n".join([build_unique_cleaned(), export_missing_cleaned(), sync_pipeline_state()])
        return f"{out}\n\nDerived files refreshed:\n{derived}"
    except subprocess.TimeoutExpired:
        state["stages"]["clean"] = {"status": "timeout", "count": 0, "time": "Timed out"}
        save_state(state)
        return "Clean timed out (10 min limit)"
    except Exception as e:
        state["stages"]["clean"] = {"status": "error", "count": 0, "time": str(e)}
        save_state(state)
        return f"Error: {e}"

# ─── Tab: Pipeline ───────────────────────────────────────────────────────────

def tab_pipeline():
    counts = get_counts()

    with gr.Tab("Pipeline"):
        with gr.Row():
            with gr.Column():
                raw_count = gr.Markdown(f"**{counts['raw']:,}**")
                gr.Markdown("Raw", elem_classes=["muted"])
            with gr.Column():
                cleaned_count = gr.Markdown(f"**{counts['cleaned']:,}**")
                gr.Markdown("Cleaned", elem_classes=["muted"])
            with gr.Column():
                summarized_count = gr.Markdown(f"**{counts['summarized']:,}**")
                gr.Markdown("Summarized", elem_classes=["muted"])
            with gr.Column():
                training_count = gr.Markdown(f"**{counts['training']:,}**")
                gr.Markdown("Training", elem_classes=["muted"])

        gr.HTML("<hr style='border:none;border-top:1px solid #2a2a3e;margin:16px 0'>")

        refresh_counts_btn = gr.Button("Refresh Counts", variant="primary")

        with gr.Row():
            with gr.Column():
                sample_size = gr.Number(label="Sample size", value=max(counts["raw"], 25000), precision=0)
                download_btn = gr.Button("Download", variant="primary")
                download_out = gr.Textbox(label="", lines=3, interactive=False)

            with gr.Column():
                clean_btn = gr.Button("Run Clean", variant="primary")
                clean_out = gr.Textbox(label="", lines=3, interactive=False)

        def run_clean_and_refresh():
            out = run_clean()
            return (out, *pipeline_count_markdowns())

        download_btn.click(run_download, inputs=[sample_size], outputs=[download_out])
        clean_btn.click(
            run_clean_and_refresh,
            outputs=[clean_out, raw_count, cleaned_count, summarized_count, training_count],
        )
        refresh_counts_btn.click(
            pipeline_count_markdowns,
            outputs=[raw_count, cleaned_count, summarized_count, training_count],
        )

# ─── Tab: Data ───────────────────────────────────────────────────────────────

DATASET_FILES = {
    "raw": RAW_DIR / "enron_sample.jsonl",
    "cleaned": PROC_DIR / "cleaned_emails.jsonl",
    "unique cleaned": PROC_DIR / "cleaned_emails.unique.jsonl",
    "missing summary inputs": PROC_DIR / "missing_summaries.jsonl",
    "summaries": SUM_DIR / "en_summaries.jsonl",
    "clean summaries": SUM_DIR / "en_summaries.clean.jsonl",
    "bad summaries": SUM_DIR / "bad_summaries.jsonl",
    "train": TRAIN_DIR / "train.jsonl",
    "validation": TRAIN_DIR / "val.jsonl",
    "test": TRAIN_DIR / "test.jsonl",
}

def dataset_body(record):
    return (
        record.get("cleaned_body")
        or record.get("email")
        or record.get("text")
        or record.get("message")
        or record.get("original_text")
        or ""
    )

def dataset_summary(record):
    return record.get("summary", "")

def dataset_subject(record):
    return record.get("subject", "")

def dataset_flags(record):
    summary = dataset_summary(record)
    if not summary:
        return ""
    return ",".join(summary_flags(summary))

def load_dataset_records(name):
    return load_jsonl(DATASET_FILES.get(name, RAW_DIR / "enron_sample.jsonl"))

def filter_dataset_records(records, search_text="", quality_filter="all"):
    if search_text:
        query = search_text.lower()
        records = [
            r for r in records
            if query in dataset_body(r).lower()
            or query in dataset_summary(r).lower()
            or query in dataset_subject(r).lower()
        ]

    if quality_filter == "flagged summaries":
        records = [r for r in records if dataset_flags(r)]
    elif quality_filter == "clean summaries":
        records = [r for r in records if dataset_summary(r) and not dataset_flags(r)]
    elif quality_filter == "has summary":
        records = [r for r in records if dataset_summary(r)]
    elif quality_filter == "no summary":
        records = [r for r in records if not dataset_summary(r)]

    return records

def data_browser_page(name, search_text="", quality_filter="all", page_num=1, page_size=25):
    records = filter_dataset_records(load_dataset_records(name), search_text, quality_filter)
    page_size = max(5, int(page_size or 25))
    page_num = max(1, int(page_num or 1))
    max_page = max(1, (len(records) + page_size - 1) // page_size)
    page_num = min(page_num, max_page)
    start = (page_num - 1) * page_size
    chunk = records[start:start + page_size]

    rows = []
    for offset, r in enumerate(chunk):
        body = dataset_body(r)
        summary = dataset_summary(r)
        rows.append([
            start + offset,
            dataset_flags(r),
            dataset_subject(r)[:80],
            len(body),
            len(summary),
            body[:180].replace("\n", " "),
            summary[:180].replace("\n", " "),
        ])

    info = f"{len(records):,} matching rows · page {page_num}/{max_page} · source `{DATASET_FILES[name].relative_to(BASE_DIR)}`"
    return rows, info

def data_record_detail(name, row_index=0, search_text="", quality_filter="all"):
    records = filter_dataset_records(load_dataset_records(name), search_text, quality_filter)
    if not records:
        return "", "", "{}"

    idx = max(0, min(int(row_index or 0), len(records) - 1))
    r = records[idx]
    body = dataset_body(r)
    summary = dataset_summary(r)
    meta = {
        k: v for k, v in r.items()
        if k not in {"text", "message", "cleaned_body", "email", "summary", "original_text"}
    }
    meta["_filtered_row"] = idx
    meta["_email_chars"] = len(body)
    meta["_summary_chars"] = len(summary)
    meta["_quality_flags"] = dataset_flags(r)
    return body, summary, json.dumps(meta, indent=2, ensure_ascii=False)

def tab_data():
    with gr.Tab("Data"):
        initial_dataset = "clean summaries" if DATASET_FILES["clean summaries"].exists() else "summaries"
        initial_rows, initial_info = data_browser_page(initial_dataset)
        initial_email, initial_summary, initial_meta = data_record_detail(initial_dataset, 0)

        with gr.Row():
            dataset_sel = gr.Dropdown(
                choices=list(DATASET_FILES.keys()),
                value=initial_dataset,
                label="Dataset",
            )
            search = gr.Textbox(label="Search", placeholder="Search subject, email text, or summary...")
            quality_filter = gr.Dropdown(
                choices=["all", "has summary", "no summary", "flagged summaries", "clean summaries"],
                value="all",
                label="Filter",
            )

        browser = gr.Dataframe(
            headers=["Row", "Flags", "Subject", "Email Chars", "Summary Chars", "Email Preview", "Summary Preview"],
            value=initial_rows,
            wrap=True,
            interactive=False,
            label="Rows",
        )

        with gr.Row():
            page = gr.Number(label="Page", value=1, precision=0)
            page_size = gr.Slider(minimum=10, maximum=100, value=25, step=5, label="Rows per page")
            refresh_btn = gr.Button("Refresh", variant="primary")
            page_info = gr.Markdown(initial_info)

        gr.Markdown("Select a row number from the table, then load the full text below.")
        with gr.Row():
            selected_row = gr.Number(label="Filtered row index", value=0, precision=0)
            detail_btn = gr.Button("Load Full Row")

        with gr.Row():
            email_detail = gr.Textbox(label="Email / Input", value=initial_email, lines=14, interactive=False)
            summary_detail = gr.Textbox(label="Summary / Target", value=initial_summary, lines=14, interactive=False)

        metadata_detail = gr.Code(label="Metadata", value=initial_meta, language="json", lines=10, interactive=False)

        def load_page(name, search_text, filter_name, page_num, size):
            return data_browser_page(name, search_text, filter_name, page_num, size)

        def reset_and_load(name, search_text, filter_name, size):
            rows, info = data_browser_page(name, search_text, filter_name, 1, size)
            return rows, info, 1

        dataset_sel.change(reset_and_load, inputs=[dataset_sel, search, quality_filter, page_size], outputs=[browser, page_info, page])
        quality_filter.change(reset_and_load, inputs=[dataset_sel, search, quality_filter, page_size], outputs=[browser, page_info, page])
        search.submit(reset_and_load, inputs=[dataset_sel, search, quality_filter, page_size], outputs=[browser, page_info, page])
        refresh_btn.click(load_page, inputs=[dataset_sel, search, quality_filter, page, page_size], outputs=[browser, page_info])
        page.change(load_page, inputs=[dataset_sel, search, quality_filter, page, page_size], outputs=[browser, page_info])
        page_size.change(reset_and_load, inputs=[dataset_sel, search, quality_filter, page_size], outputs=[browser, page_info, page])
        detail_btn.click(data_record_detail, inputs=[dataset_sel, selected_row, search, quality_filter], outputs=[email_detail, summary_detail, metadata_detail])


# ─── Tab: Quality ────────────────────────────────────────────────────────────

def tab_quality():
    with gr.Tab("Quality"):
        gr.Markdown(
            "Scan generated data, identify duplicate-ish rows and bad summaries, "
            "then prepare clean training splits without overwriting the raw source files."
        )

        with gr.Row():
            max_summary_chars = gr.Slider(
                minimum=80,
                maximum=600,
                value=300,
                step=20,
                label="Max acceptable summary chars",
            )
            scan_btn = gr.Button("Scan Data Quality", variant="primary")
            sync_btn = gr.Button("Sync Pipeline State")

        metrics = gr.Dataframe(
            headers=["Metric", "Value"],
            label="Current Data Health",
            interactive=False,
        )
        flagged = gr.Dataframe(
            headers=["Index", "Flags", "Summary Len", "Subject", "Summary Preview"],
            label="Flagged Summary Rows",
            wrap=True,
            interactive=False,
        )
        quality_note = gr.Markdown("")

        with gr.Row():
            export_bad_btn = gr.Button("Export Bad Summaries")
            clean_btn = gr.Button("Build Clean Summaries", variant="primary")
            unique_clean_btn = gr.Button("Export Unique Cleaned")
            missing_btn = gr.Button("Export Missing Summary Inputs")
            action_out = gr.Textbox(label="Action Output", lines=3, interactive=False)

        gr.HTML("<hr style='border:none;border-top:1px solid #2a2a3e;margin:16px 0'>")

        with gr.Row():
            source_sel = gr.Dropdown(
                choices=["clean summaries", "all summaries"],
                value="clean summaries",
                label="Training source",
            )
            val_pct = gr.Slider(minimum=1, maximum=25, value=10, step=1, label="Validation %")
            test_pct = gr.Slider(minimum=0, maximum=20, value=5, step=1, label="Test %")
            seed = gr.Number(label="Shuffle seed", value=42, precision=0)

        train_btn = gr.Button("Generate Train/Val/Test Files", variant="primary")
        train_out = gr.Textbox(label="Training Output", lines=2, interactive=False)

        scan_btn.click(run_quality_scan, inputs=[max_summary_chars], outputs=[metrics, flagged, quality_note])
        export_bad_btn.click(export_bad_summaries, inputs=[max_summary_chars], outputs=[action_out])
        clean_btn.click(build_clean_summaries, inputs=[max_summary_chars], outputs=[action_out])
        unique_clean_btn.click(build_unique_cleaned, outputs=[action_out])
        missing_btn.click(export_missing_cleaned, outputs=[action_out])
        train_btn.click(generate_training_files, inputs=[source_sel, val_pct, test_pct, seed, max_summary_chars], outputs=[train_out])
        sync_btn.click(sync_pipeline_state, outputs=[action_out])

# ─── Tab: Charts ─────────────────────────────────────────────────────────────

def tab_charts():
    with gr.Tab("Charts"):
        gr.Markdown("Quick dashboard charts plus report-ready analytics artifacts for the paper/document.")

        with gr.Row():
            c1 = gr.Image(label="Length Distribution", show_label=False)
            c2 = gr.Image(label="Pipeline", show_label=False)

        c3 = gr.Image(label="Summary Lengths", show_label=False)

        refresh_btn = gr.Button("Refresh", variant="primary")
        refresh_btn.click(render_charts, outputs=[c1, c2, c3])

        gr.HTML("<hr style='border:none;border-top:1px solid #2a2a3e;margin:16px 0'>")
        analytics_btn = gr.Button("Generate Full Report Analytics", variant="primary")
        analytics_out = gr.Textbox(label="Analytics Output", lines=8, interactive=False)

        report_paths = report_chart_paths()
        with gr.Row():
            r1 = gr.Image(label="Pipeline Counts", value=report_paths[0])
            r2 = gr.Image(label="Length Distributions", value=report_paths[1])
        with gr.Row():
            r3 = gr.Image(label="Cleaning Retention", value=report_paths[2])
            r4 = gr.Image(label="Summary Compression", value=report_paths[3])
        with gr.Row():
            r5 = gr.Image(label="Quality Flags", value=report_paths[4])
            r6 = gr.Image(label="Split Balance", value=report_paths[5])
        with gr.Row():
            r7 = gr.Image(label="Duplicate Diagnostics", value=report_paths[6])
            r8 = gr.Image(label="Top Summary Terms", value=report_paths[7])

        analytics_btn.click(
            generate_full_analytics,
            outputs=[analytics_out, r1, r2, r3, r4, r5, r6, r7, r8],
        )

        auto_timer = gr.Timer(30)
        auto_timer.tick(render_charts, outputs=[c1, c2, c3])

        render_charts()

# ─── Tab: Summarization ─────────────────────────────────────────────────────

PID_FILE = DATA_DIR / ".summarize_pid"

def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def _load_pid():
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            pass
    return None

def _save_pid(pid):
    PID_FILE.write_text(str(pid))

def _clear_pid():
    if PID_FILE.exists():
        PID_FILE.unlink()

def tail_log(n=80):
    log_path = DATA_DIR / "summaries" / "summarize.log"
    if not log_path.exists():
        return ""
    with open(log_path) as f:
        lines = f.readlines()
    return "".join(lines[-n:]) if lines else ""

_last_count = 0
_last_count_ts = datetime.now()

def _detect_stall(current_count):
    global _last_count, _last_count_ts
    if current_count != _last_count:
        _last_count = current_count
        _last_count_ts = datetime.now()
        return None
    elapsed = (datetime.now() - _last_count_ts).total_seconds()
    if elapsed > 120:
        return f"  \n⚠ **Stuck?** — {current_count:,} summaries unchanged for {int(elapsed)}s"
    return f""

_process = None

def tab_summarization():
    global _process

    with gr.Tab("Summarization"):
        with gr.Row():
            with gr.Column():
                model_input = gr.Dropdown(
                    choices=["deepseek-v4-flash", "gpt-4o-mini"],
                    value="deepseek-v4-flash",
                    label="Model"
                )
                api_key_input = gr.Textbox(label="API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
                base_url_input = gr.Textbox(label="Base URL", value=os.environ.get("OPENAI_BASE_URL", ""))

            with gr.Column():
                workers_slider = gr.Slider(minimum=1, maximum=25, value=3, step=1, label="Concurrent Workers")
                max_tokens = gr.Slider(minimum=50, maximum=500, value=120, step=10, label="Max Tokens")
                cooldown_every = gr.Slider(minimum=50, maximum=1000, value=100, step=50, label="Pause every N reqs")
                cooldown_secs = gr.Slider(minimum=5, maximum=120, value=15, step=5, label="Pause duration (s)")
                max_per_batch = gr.Slider(minimum=500, maximum=3000, value=1500, step=500, label="Restart after N requests")
                input_source = gr.Dropdown(
                    choices=["cleaned_emails.jsonl", "missing_summaries.jsonl", "cleaned_emails.unique.jsonl"],
                    value="cleaned_emails.jsonl",
                    label="Input file",
                )

        with gr.Row():
            start_btn = gr.Button("Start Summarization", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")

        status_text = gr.Markdown("Status: **Idle**")
        cost_estimate = gr.Markdown("")
        log_area = gr.Code(label="Live Log", language=None, lines=10, interactive=False)
        log_visible = gr.Checkbox(label="Show Log", value=True)

        def update_cost(model, max_tok):
            prices = {
                "gpt-4o-mini": {"input": 0.15, "output": 0.60},
                "deepseek-v4-flash": {"input": 0.07, "output": 0.28},
            }
            p = prices.get(model, prices["gpt-4o-mini"])
            cleaned = count_jsonl(PROC_DIR / "cleaned_emails.jsonl")
            summarized = count_jsonl(SUM_DIR / "en_summaries.jsonl")
            remaining = cleaned - summarized
            input_cost = (remaining * 500 / 1e6) * p["input"]
            output_cost = (remaining * max_tok / 1e6) * p["output"]
            total = input_cost + output_cost
            return f"Emails: {cleaned:,} cleaned  ·  {summarized:,} done  ·  {remaining:,} remaining\nEst. cost: **${total:.2f}** (input ${input_cost:.2f} + output ${output_cost:.2f})"

        model_input.change(update_cost, inputs=[model_input, max_tokens], outputs=[cost_estimate])
        max_tokens.change(update_cost, inputs=[model_input, max_tokens], outputs=[cost_estimate])

        def start_summarization(model, api_key, base_url, max_tok, workers, cooldown_n, cooldown_s, batch_limit, input_file):
            global _process

            pid = _load_pid()
            log_path = str(DATA_DIR / "summaries" / "summarize.log")
            if pid and _pid_alive(pid):
                return "Status: **Already running (PID {})**".format(pid), tail_log()

            script = BASE_DIR / "scripts" / "generate_summaries.py"
            if not script.exists():
                return "Status: **Error: Script not found**", ""
            input_path = PROC_DIR / input_file
            if not input_path.exists():
                return f"Status: **Error: input not found: {input_path.relative_to(BASE_DIR)}**", tail_log()

            env = os.environ.copy()
            if api_key:
                env["OPENAI_API_KEY"] = api_key
            if base_url:
                env["OPENAI_BASE_URL"] = base_url

            cmd = [
                sys.executable, str(script),
                "--model", model,
                "--workers", str(int(workers)),
                "--max-tokens", str(int(max_tok)),
                "--input", str(input_path),
                "--cooldown-every", str(int(cooldown_n)),
                "--cooldown-secs", str(int(cooldown_s)),
                "--max-per-batch", str(int(batch_limit)),
                "--log", log_path,
            ]

            try:
                # Clear old log
                Path(log_path).write_text("")
                err_path = str(DATA_DIR / "summaries" / "summarize.err.log")
                err_file = open(err_path, "a", buffering=1)
                _process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=err_file,
                    text=True,
                )
                _save_pid(_process.pid)
                return "Status: **Running (PID {})**".format(_process.pid), tail_log()
            except Exception as e:
                _process = None
                return "Status: **Error: {}**".format(e), tail_log()

        def stop_summarization():
            global _process

            pid = _load_pid()
            killed = False

            if _process and _process.poll() is None:
                _process.terminate()
                _process = None
                killed = True

            if pid and _pid_alive(pid):
                try:
                    os.kill(pid, 15)
                except OSError:
                    pass
                killed = True

            _clear_pid()
            return "Status: **{}**".format("Stopped" if killed else "No process running"), tail_log()

        def poll_status():
            global _process, _last_count

            summarized = count_jsonl(SUM_DIR / "en_summaries.jsonl")
            cleaned = count_jsonl(PROC_DIR / "cleaned_emails.jsonl")
            stall = _detect_stall(summarized)

            pid = _load_pid()

            if pid and _pid_alive(pid):
                pct = round(summarized / cleaned * 100, 1) if cleaned > 0 else 0
                return "Status: **Running (PID {})** — {:,}/{:,} ({:.1f}%){}".format(pid, summarized, cleaned, pct, stall), tail_log()

            if pid and not _pid_alive(pid):
                _clear_pid()
                _process = None
                return "Status: **Finished** — {:,} summaries".format(summarized), tail_log()

            if _process:
                if _process.poll() is None:
                    pct = round(summarized / cleaned * 100, 1) if cleaned > 0 else 0
                    return "Status: **Running (PID {})** — {:,}/{:,} ({:.1f}%){}".format(_process.pid, summarized, cleaned, pct, stall), tail_log()
                else:
                    _process = None
                    return "Status: **Finished** — {:,} summaries".format(summarized), tail_log()

            return "Status: **Idle**", tail_log()

        start_btn.click(start_summarization, inputs=[model_input, api_key_input, base_url_input, max_tokens, workers_slider, cooldown_every, cooldown_secs, max_per_batch, input_source], outputs=[status_text, log_area])
        stop_btn.click(stop_summarization, outputs=[status_text, log_area])

        status_timer = gr.Timer(3)
        status_timer.tick(poll_status, outputs=[status_text, log_area])

        update_cost("deepseek-v4-flash", 120)

# ─── Tab: System ─────────────────────────────────────────────────────────────

def tab_system():
    with gr.Tab("System"):
        with gr.Row():
            cpu = gr.Number(label="CPU %", value=0, interactive=False)
            ram = gr.Number(label="RAM GB", value=0, interactive=False)
            gpu = gr.Number(label="GPU %", value=0, interactive=False)
            disk = gr.Number(label="Disk %", value=0, interactive=False)

        hw_details = gr.Markdown("")

        state = init_state()
        timing = gr.Dataframe(
            headers=["Stage", "Status", "Count", "Time"],
            value=[[s, v.get("status", "pending"), v.get("count", 0), v.get("time", "—")] for s, v in state.get("stages", {}).items()],
            interactive=False
        )

        refresh_btn = gr.Button("Refresh", variant="primary")

        def refresh():
            hw = get_hw()
            state = init_state()
            t = [[s, v.get("status", "pending"), v.get("count", 0), v.get("time", "—")] for s, v in state.get("stages", {}).items()]
            detail = f"RAM {hw['ram_gb']}/{hw['ram_total_gb']} GB  ·  Disk {hw['disk_gb']}/{hw['disk_total_gb']} GB"
            if hw["gpu"] is not None:
                detail += f"  ·  GPU {hw['gpu']}%  ·  VRAM {hw.get('gpu_mem', 0)}/{hw.get('gpu_total', 0)} GB"
            detail += f"\nStarted: {state.get('started', '—')}  ·  Updated: {state.get('updated', '—')}"
            outs = [hw["cpu"], hw["ram_gb"], hw.get("gpu", 0) or 0, hw["disk_pct"], detail, t]
            return outs

        refresh_btn.click(refresh, outputs=[cpu, ram, gpu, disk, hw_details, timing])

        sys_timer = gr.Timer(5)
        sys_timer.tick(refresh, outputs=[cpu, ram, gpu, disk, hw_details, timing])

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    init_state()

    css = """
    .gradio-container {
        max-width: 1200px !important;
        padding: 32px 40px !important;
        background: #0f1117 !important;
    }
    body { background: #0f1117 !important; }
    #header { font-size: 22px !important; font-weight: 600 !important; margin-bottom: 8px !important; }
    .tabs {
        margin-top: 8px !important;
        border-bottom: 1px solid #1f2230 !important;
    }
    .tab-nav button {
        background: transparent !important;
        border: none !important;
        color: #8b8fa3 !important;
        font-size: 14px !important;
        padding: 10px 24px !important;
        border-radius: 0 !important;
        transition: color 0.2s !important;
    }
    .tab-nav button.selected {
        color: #00d4aa !important;
        border-bottom: 2px solid #00d4aa !important;
        font-weight: 600 !important;
    }
    button.primary {
        background: #00d4aa !important;
        color: #0f1117 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
        transition: background 0.2s !important;
    }
    button.primary:hover { background: #00b894 !important; }
    .block { background: transparent !important; border: none !important; }
    input, .dropdown input, .number input {
        background: #161822 !important;
        border: 1px solid #1f2230 !important;
        border-radius: 8px !important;
        color: #c9cdd8 !important;
        padding: 8px 12px !important;
    }
    .dataframe table {
        background: #161822 !important;
        border-radius: 8px !important;
        border: 1px solid #1f2230 !important;
    }
    .dataframe td, .dataframe th {
        background: transparent !important;
        color: #c9cdd8 !important;
        border-color: #1f2230 !important;
    }
    .muted {
        color: #6b7280 !important;
        font-size: 11px !important;
        margin-top: -6px !important;
        font-weight: 400 !important;
    }
    label { color: #8b8fa3 !important; font-size: 12px !important; }
    h2, h3 { color: #e4e6ed !important; }
    p { color: #8b8fa3 !important; }
    hr { border-color: #1f2230 !important; }
    .slider input { background: #161822 !important; }
    """

    with gr.Blocks(title="GuppyLM") as app:
        gr.Markdown("GuppyLM · Data Pipeline", elem_id="header")

        tab_pipeline()
        tab_data()
        tab_quality()
        tab_charts()
        tab_summarization()
        tab_system()

    app.launch(server_name="0.0.0.0", server_port=7860, css=css)

if __name__ == "__main__":
    main()
