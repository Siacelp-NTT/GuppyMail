"""
GuppyLM Data Dashboard — Control panel for the email data pipeline.

Tabs: Pipeline · Data · Charts · System
Summarization tab is toggleable via Settings.

Usage:
    python src/data_dashboard.py
"""

import json
import hashlib
import os
import psutil
import pandas as pd
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

def get_raw_stats():
    p = RAW_DIR / "enron_stats.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

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
        return out if result.returncode == 0 else f"Error: {result.stderr.strip()}"
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
            for label, key in [("Raw", "raw"), ("Cleaned", "cleaned"), ("Summarized", "summarized"), ("Training", "training")]:
                with gr.Column():
                    v = gr.Markdown(f"**{counts[key]:,}**")
                    gr.Markdown(label, elem_classes=["muted"])

        gr.HTML("<hr style='border:none;border-top:1px solid #2a2a3e;margin:16px 0'>")

        with gr.Row():
            with gr.Column():
                sample_size = gr.Number(label="Sample size", value=25000, precision=0)
                download_btn = gr.Button("Download", variant="primary")
                download_out = gr.Textbox(label="", lines=3, interactive=False)

            with gr.Column():
                clean_btn = gr.Button("Run Clean", variant="primary")
                clean_out = gr.Textbox(label="", lines=3, interactive=False)

        download_btn.click(run_download, inputs=[sample_size], outputs=[download_out])
        clean_btn.click(run_clean, outputs=[clean_out])

# ─── Tab: Data ───────────────────────────────────────────────────────────────

def tab_data():
    with gr.Tab("Data"):
        with gr.Row():
            dataset_sel = gr.Dropdown(choices=["raw", "cleaned", "summarized"], value="raw", label="")
            search = gr.Textbox(label="", placeholder="Search...")

        browser = gr.Dataframe(headers=["#", "Subject", "Len", "Preview"], wrap=True, interactive=False)

        with gr.Row():
            page = gr.Number(label="Page", value=1, precision=0)
            page_info = gr.Markdown("")

        def load_data(name, search_text="", page_num=1):
            files = {
                "raw": RAW_DIR / "enron_sample.jsonl",
                "cleaned": PROC_DIR / "cleaned_emails.jsonl",
                "summarized": SUM_DIR / "en_summaries.jsonl",
            }
            keys = {"raw": "text", "cleaned": "cleaned_body", "summarized": "summary"}
            records = load_jsonl(files[name])
            key = keys[name]

            if search_text:
                s = search_text.lower()
                records = [r for r in records if s in r.get(key, "").lower() or s in r.get("subject", "").lower()]

            page_size = 25
            page_num = max(1, int(page_num))
            start = (page_num - 1) * page_size
            chunk = records[start:start + page_size]
            max_page = max(1, len(records) // page_size + 1)

            rows = []
            for i, r in enumerate(chunk):
                rows.append([start + i, r.get("subject", "")[:40], len(r.get(key, "")), r.get(key, "")[:80]])

            info = f"{len(records):,} records  ·  page {page_num}/{max_page}"
            return rows, info

        dataset_sel.change(load_data, inputs=[dataset_sel, search, page], outputs=[browser, page_info])
        search.submit(load_data, inputs=[dataset_sel, search, page], outputs=[browser, page_info])
        page.change(load_data, inputs=[dataset_sel, search, page], outputs=[browser, page_info])

        load_data("raw")

# ─── Tab: Charts ─────────────────────────────────────────────────────────────

def tab_charts():
    with gr.Tab("Charts"):
        with gr.Row():
            c1 = gr.Image(label="Length Distribution", show_label=False)
            c2 = gr.Image(label="Pipeline", show_label=False)

        c3 = gr.Image(label="Summary Lengths", show_label=False)

        refresh_btn = gr.Button("Refresh", variant="primary")
        refresh_btn.click(render_charts, outputs=[c1, c2, c3])

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

def tail_log(n=15):
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
                workers_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Concurrent Workers")
                max_tokens = gr.Slider(minimum=100, maximum=2000, value=800, step=100, label="Max Tokens")
                cooldown_every = gr.Slider(minimum=50, maximum=500, value=100, step=50, label="Pause every N reqs")
                cooldown_secs = gr.Slider(minimum=5, maximum=60, value=15, step=5, label="Pause duration (s)")

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

        def start_summarization(model, api_key, base_url, max_tok, workers, cooldown_n, cooldown_s):
            global _process

            pid = _load_pid()
            log_path = str(DATA_DIR / "summaries" / "summarize.log")
            if pid and _pid_alive(pid):
                return "Status: **Already running (PID {})**".format(pid), tail_log()

            script = BASE_DIR / "scripts" / "generate_summaries.py"
            if not script.exists():
                return "Status: **Error: Script not found**", ""

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
                "--cooldown-every", str(int(cooldown_n)),
                "--cooldown-secs", str(int(cooldown_s)),
                "--log", log_path,
            ]

            try:
                # Clear old log
                Path(log_path).write_text("")
                _process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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

        start_btn.click(start_summarization, inputs=[model_input, api_key_input, base_url_input, max_tokens, workers_slider, cooldown_every, cooldown_secs], outputs=[status_text, log_area])
        stop_btn.click(stop_summarization, outputs=[status_text, log_area])

        status_timer = gr.Timer(3)
        status_timer.tick(poll_status, outputs=[status_text, log_area])

        update_cost("deepseek-v4-flash", 800)

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
        tab_charts()
        tab_summarization()
        tab_system()

    app.launch(server_name="0.0.0.0", server_port=7860, css=css)

if __name__ == "__main__":
    main()
