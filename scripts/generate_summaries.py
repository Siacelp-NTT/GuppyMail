"""
Task 1.3 — Generate Quirky Email Summaries via LLM API

Generates short, conversational, quirky summaries for cleaned emails.
Supports concurrent processing (2-3 emails at once via ThreadPoolExecutor).

Usage:
    export OPENAI_API_KEY="your-key"
    python scripts/generate_summaries.py
    python scripts/generate_summaries.py --workers 3

Cost: ~$2.50 for 25K emails (with concurrency, ~40-60 min)
"""

import os
import json
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, wait as fut_wait, FIRST_COMPLETED
from tqdm import tqdm

import requests


def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_duration(seconds):
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def short_error(error):
    return " ".join(str(error).split())[:180]


SYSTEM_PROMPT = """You are a tiny, friendly email assistant. Your summaries are:
- VERY short (1-3 sentences max, under 50 words)
- Conversational and casual (not formal/corporate)
- Simple vocabulary (avoid jargon)
- Focus on: what happened, what needs to be done, any deadlines
- Slightly quirky but still informative
- Use lowercase for a casual feel

Example style:
"hey, so the meeting got moved to feb 5th at 2pm. bring your reports. boss wants everyone there by friday."

NOT this style:
"The quarterly review meeting has been rescheduled from February 10th to February 5th at 14:00. All department heads are requested to submit their financial reports..."
"""


def summarize_one(api_key, base_url, email_text, subject, model, max_tokens):
    if len(email_text) > 3000:
        email_text = email_text[:3000] + "... [truncated]"
    full_text = f"Subject: {subject}\n\n{email_text}" if subject else email_text
    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Connection": "close"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this email in your casual style:\n\n{full_text}"}
        ],
        "temperature": 0.5,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, json=body, headers=headers, timeout=(30, 120))
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    return content or "email received, nothing specific to report"


def summarize_batch(emails, model, api_key, base_url, max_tokens, workers, checkpoint_every=50, save_callback=None, echo_fn=print, cooldown_every=100, cooldown_secs=15, max_per_batch=0):
    results = [None] * len(emails)
    errors = []
    completed = 0
    started = time.time()
    last_hb = time.time()
    last_progress = time.time()
    last_progress_count = 0
    warned_stall = False
    next_cooldown = cooldown_every

    key = api_key or os.environ.get("OPENAI_API_KEY")
    url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def job(idx):
        email = emails[idx]
        cleaned = email.get("cleaned_body", "")
        subject = email.get("subject", "")
        for attempt in range(3):
            try:
                summary = summarize_one(key, url, cleaned, subject, model, max_tokens)
                return idx, summary, email
            except requests.exceptions.Timeout as e:
                echo_fn(f"[{now_ts()}] timeout idx={idx} attempt={attempt+1}/3 subject={subject[:60]!r} error={short_error(e)}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))  # 5s, 10s backoff
                else:
                    raise RuntimeError(f"timeout after 3 attempts: {e}")
            except Exception as e:
                echo_fn(f"[{now_ts()}] error idx={idx} attempt={attempt+1}/3 subject={subject[:60]!r} error={short_error(e)}")
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise RuntimeError(f"retry #{attempt+1} failed: {e}")
        return idx, None, email

    executor = ThreadPoolExecutor(max_workers=workers)
    early_exit = False
    try:
        futures = {executor.submit(job, i): i for i in range(len(emails))}
        pending = set(futures.keys())

        with tqdm(total=len(emails), desc="Summarizing") as pbar:
            while pending:
                done, pending = fut_wait(
                    pending, timeout=10, return_when=FIRST_COMPLETED
                )

                for future in done:
                    idx = futures[future]
                    try:
                        idx, summary, email = future.result()
                    except Exception as e:
                        errors.append({"index": idx, "error": short_error(e)})
                        email = emails[idx]
                        email["summary"] = "error: could not summarize"
                        email["summary_model"] = model
                    else:
                        if summary:
                            email["summary"] = summary
                            email["summary_model"] = model
                        else:
                            email["summary"] = "empty: retries exhausted"
                            email["summary_model"] = model
                            errors.append({"index": idx, "error": "empty after 3 retries"})
                    results[idx] = email
                    completed += 1
                    pbar.update(1)

                if save_callback and completed and completed % checkpoint_every == 0:
                    save_callback([r for r in results if r is not None])

                if completed >= next_cooldown:
                    echo_fn(f"[{now_ts()}] cooldown {cooldown_secs}s after {completed:,}/{len(emails):,} requests")
                    time.sleep(cooldown_secs)
                    next_cooldown = completed + cooldown_every
                    last_progress = time.time()

                now = time.time()

                if completed > last_progress_count:
                    last_progress_count = completed
                    last_progress = now
                    warned_stall = False

                if now - last_hb >= 30:
                    err_sample = ""
                    if errors:
                        last_err = errors[-1]["error"][:80]
                        err_sample = f" (last: {last_err})"
                    rate = completed / max(now - started, 1)
                    eta = (len(emails) - completed) / rate if rate > 0 else 0
                    echo_fn(
                        f"[{now_ts()}] progress {completed:,}/{len(emails):,} "
                        f"({completed / len(emails) * 100:.1f}%) errors={len(errors):,} "
                        f"pending={len(pending):,} rate={rate:.2f}/s eta={fmt_duration(eta)}{err_sample}"
                    )
                    last_hb = now

                stall_secs = now - last_progress
                if completed < len(emails) and stall_secs > 60 and not warned_stall:
                    echo_fn(f"[{now_ts()}] stalled no new summaries in {int(stall_secs)}s pending={len(pending):,} workers={workers}")
                    warned_stall = True

                if max_per_batch > 0 and completed >= max_per_batch:
                    echo_fn(f"[{now_ts()}] batch_limit reached completed={completed:,}; saving and restarting")
                    if save_callback:
                        save_callback([r for r in results if r is not None])
                    early_exit = True
                    break

                if completed < len(emails) and stall_secs > 180:
                    echo_fn(f"[{now_ts()}] timeout stalled_for={int(stall_secs)}s completed={completed:,}; saving and restarting")
                    if save_callback:
                        save_callback([r for r in results if r is not None])
                    early_exit = True
                    break

            if early_exit:
                executor.shutdown(wait=False, cancel_futures=True)
    finally:
        if not early_exit:
            executor.shutdown(wait=True)
    return [r for r in results if r is not None], errors


def load_emails(path):
    emails = []
    with open(path) as f:
        for line in f:
            try:
                emails.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return emails


def save_results(results, path):
    tmp = path + ".tmp"
    bak = path + ".bak"
    with open(tmp, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    if os.path.exists(path):
        try:
            os.replace(path, bak)
        except OSError:
            pass
    os.replace(tmp, path)


def main():
    os.chdir(os.path.dirname(__file__) + "/.." if "__file__" in dir() else ".")

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent API workers (default: 3)")
    parser.add_argument("--input", default="data/processed/cleaned_emails.jsonl")
    parser.add_argument("--output", default="data/summaries/en_summaries.jsonl")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--log", default="", help="Write output to this log file")
    parser.add_argument("--cooldown-every", type=int, default=100, help="Pause after this many requests (default: 100)")
    parser.add_argument("--cooldown-secs", type=int, default=15, help="Seconds to pause (default: 15)")
    parser.add_argument("--max-per-batch", type=int, default=1500, help="Max requests per session, then exit for clean restart (default: 1500)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log = open(args.log, "a", buffering=1) if args.log else None

    if not args.api_key and not os.environ.get("OPENAI_API_KEY"):
        def echo(msg):
            print(msg)
            if log:
                log.write(msg + "\n")
        echo("ERROR: OPENAI_API_KEY not set!")
        echo("  export OPENAI_API_KEY='your-key'")
        echo("  or use --api-key <key>")
        return

    def echo(msg):
        print(msg, flush=True)
        if log:
            log.write(msg + "\n")
            log.flush()

    run_started = time.time()
    emails = load_emails(input_path)

    existing = load_emails(output_path)
    start_idx = len(existing)
    echo("=" * 72)
    echo(f"[{now_ts()}] summary generation started")
    echo(f"input={input_path}")
    echo(f"output={output_path}")
    echo(f"model={args.model} workers={args.workers} max_tokens={args.max_tokens}")
    echo(f"cooldown_every={args.cooldown_every} cooldown_secs={args.cooldown_secs} max_per_batch={args.max_per_batch}")
    echo(f"input_rows={len(emails):,} existing_output_rows={len(existing):,}")
    if start_idx > 0:
        echo(f"[{now_ts()}] loaded existing summaries; scanning input for missing hashes")

    # dedup: skip cleaned_bodies already summarized (shifted after re-download)
    import hashlib
    seen_hashes = set()
    for r in existing:
        body = r.get("cleaned_body", "")
        seen_hashes.add(hashlib.md5(body[:200].encode()).hexdigest())

    deduped = []
    skipped = 0
    for e in emails:
        body = e.get("cleaned_body", "")
        h = hashlib.md5(body[:200].encode()).hexdigest()
        if h in seen_hashes:
            skipped += 1
        else:
            deduped.append(e)
    if skipped:
        echo(f"[{now_ts()}] skipped_already_summarized={skipped:,}")
    emails = deduped

    echo(f"[{now_ts()}] emails_to_summarize={len(emails):,}")
    echo(f"[{now_ts()}] estimated_time={fmt_duration(len(emails) * 3.5 / max(args.workers, 1))}")
    echo("-" * 72)

    checkpoint_count = [0]  # mutable counter

    def checkpoint(new_results):
        checkpoint_count[0] += 1
        all_r = existing + new_results
        save_results(all_r, output_path)
        echo(f"[{now_ts()}] checkpoint={checkpoint_count[0]} batch_saved={len(new_results):,} output_rows={len(all_r):,}")

    total_errors = []
    remaining = emails
    batch_n = 0

    while remaining:
        batch_n += 1
        existing = load_emails(output_path)
        echo(f"[{now_ts()}] batch_start batch={batch_n} remaining={len(remaining):,} existing_output_rows={len(existing):,}")

        results, errors = summarize_batch(
            remaining, args.model, args.api_key, args.base_url, args.max_tokens, args.workers,
            checkpoint_every=50, save_callback=checkpoint, echo_fn=echo,
            cooldown_every=args.cooldown_every, cooldown_secs=args.cooldown_secs,
            max_per_batch=args.max_per_batch
        )
        total_errors.extend(errors)
        save_results(existing + results, output_path)
        echo(f"[{now_ts()}] batch_saved_final batch={batch_n} saved={len(results):,} output_rows={len(existing) + len(results):,} errors_this_batch={len(errors):,}")

        if len(results) < len(remaining):
            echo(f"[{now_ts()}] batch_incomplete completed={len(results):,}/{len(remaining):,}; restarting fresh session")
            processed_bodies = set()
            for r in results:
                processed_bodies.add(r.get("cleaned_body", "")[:200])
            remaining = [e for e in remaining if e.get("cleaned_body", "")[:200] not in processed_bodies]
            echo(f"[{now_ts()}] auto_restart remaining={len(remaining):,} sleep=30s")
            time.sleep(30)
        else:
            remaining = []

    all_results = load_emails(output_path)
    echo("-" * 72)
    echo(f"[{now_ts()}] done output_rows={len(all_results):,} errors={len(total_errors):,} elapsed={fmt_duration(time.time() - run_started)}")
    echo(f"saved_to={output_path}")

    if all_results:
        echo("=" * 60)
        echo("EXAMPLE SUMMARIES")
        echo("=" * 60)
        for r in all_results[-5:]:
            body = r.get("cleaned_body", "")[:150]
            summary = r.get("summary", "")
            echo(f"\nEmail: {body}...")
            echo(f"Summary: {summary}")

    if log:
        log.close()


if __name__ == "__main__":
    main()
