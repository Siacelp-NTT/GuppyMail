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
from concurrent.futures import ThreadPoolExecutor, wait as fut_wait, FIRST_COMPLETED
from tqdm import tqdm

from openai import OpenAI


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


def summarize_one(email_text, subject, model, api_key, base_url, max_tokens):
    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", None),
        max_retries=0,
    )
    try:
        if len(email_text) > 3000:
            email_text = email_text[:3000] + "... [truncated]"
        full_text = f"Subject: {subject}\n\n{email_text}" if subject else email_text
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize this email in your casual style:\n\n{full_text}"}
            ],
            temperature=0.5,
            max_tokens=max_tokens,
            timeout=45,
        )
        return response.choices[0].message.content.strip() or "email received, nothing specific to report"
    finally:
        client.close()


def summarize_batch(emails, model, api_key, base_url, max_tokens, workers, checkpoint_every=50, save_callback=None, echo_fn=print, cooldown_every=100, cooldown_secs=15):
    results = [None] * len(emails)
    errors = []
    completed = 0
    last_hb = time.time()
    last_progress = time.time()
    last_progress_count = 0
    warned_stall = False
    next_cooldown = cooldown_every

    def job(idx):
        email = emails[idx]
        cleaned = email.get("cleaned_body", "")
        subject = email.get("subject", "")
        for attempt in range(3):
            try:
                summary = summarize_one(cleaned, subject, model, api_key, base_url, max_tokens)
                return idx, summary, email
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise RuntimeError(f"retry #{attempt+1} failed: {e}")
        return idx, None, email

    with ThreadPoolExecutor(max_workers=workers) as executor:
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
                        errors.append({"index": idx, "error": str(e)[:120]})
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

                # cooldown: pause to avoid rate limits
                if completed >= next_cooldown:
                    echo_fn(f"  [cooldown] {cooldown_secs}s pause after {completed} requests to avoid rate limits")
                    time.sleep(cooldown_secs)
                    next_cooldown = completed + cooldown_every
                    last_progress = time.time()  # reset stall clock during cooldown

                now = time.time()

                # progress tracking
                if completed > last_progress_count:
                    last_progress_count = completed
                    last_progress = now
                    warned_stall = False

                # heartbeat every 30s
                if now - last_hb >= 30:
                    err_sample = ""
                    if errors:
                        last_err = errors[-1]["error"][:80]
                        err_sample = f" (last: {last_err})"
                    echo_fn(f"  [heartbeat] {completed}/{len(emails)} done, {len(errors)} errors{err_sample}, {len(pending)} pending")
                    last_hb = now

                # stall warning: no progress for >60s while workers are still pending
                stall_secs = now - last_progress
                if completed < len(emails) and stall_secs > 60 and not warned_stall:
                    echo_fn(f"  [stalled] no new summaries in {int(stall_secs)}s — {len(pending)} pending, {workers} workers")
                    warned_stall = True

                # auto-stop if completely stuck for >3 min — save and exit
                if completed < len(emails) and stall_secs > 180:
                    echo_fn(f"  [timeout] stalled for {int(stall_secs)}s — saving {completed} summaries and exiting")
                    if save_callback:
                        save_callback([r for r in results if r is not None])
                    break

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
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--log", default="", help="Write output to this log file")
    parser.add_argument("--cooldown-every", type=int, default=100, help="Pause after this many requests (default: 100)")
    parser.add_argument("--cooldown-secs", type=int, default=15, help="Seconds to pause (default: 15)")
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

    emails = load_emails(input_path)

    existing = load_emails(output_path)
    start_idx = len(existing)
    if start_idx > 0:
        echo(f"Resuming from email {start_idx} (existing: {len(existing)} summaries)")
    emails = emails[start_idx:]

    echo(f"Emails to summarize: {len(emails)}")
    echo(f"Model: {args.model}")
    echo(f"Workers: {args.workers} (concurrent)")
    echo(f"Cooldown: {args.cooldown_secs}s pause every {args.cooldown_every} requests")
    echo(f"Checkpoint: every 50 completions")
    echo(f"Estimated time: ~{len(emails) * 3.5 / args.workers / 60:.0f} min")
    echo("")

    checkpoint_count = [0]  # mutable counter

    def checkpoint(new_results):
        checkpoint_count[0] += 1
        all_r = existing + new_results
        save_results(all_r, output_path)
        echo(f"  [checkpoint {checkpoint_count[0]}] saved {len(new_results)} summaries ({len(all_r):,} total)")

    results, errors = summarize_batch(
        emails, args.model, args.api_key, args.base_url, args.max_tokens, args.workers,
        checkpoint_every=50, save_callback=checkpoint, echo_fn=echo,
        cooldown_every=args.cooldown_every, cooldown_secs=args.cooldown_secs
    )

    all_results = existing + results
    save_results(all_results, output_path)

    echo(f"\nDone! Summaries: {len(all_results)}, Errors: {len(errors)}")
    echo(f"Saved to: {output_path}")

    if all_results:
        echo("=" * 60)
        echo("EXAMPLE SUMMARIES")
        echo("=" * 60)
        for r in existing[-3:] + results[:3]:
            body = r.get("cleaned_body", "")[:150]
            summary = r.get("summary", "")
            echo(f"\nEmail: {body}...")
            echo(f"Summary: {summary}")

    if log:
        log.close()


if __name__ == "__main__":
    main()
