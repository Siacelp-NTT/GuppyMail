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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI


def create_client(api_key=None, base_url=None):
    return OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", None),
    )


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
    client = create_client(api_key, base_url)
    if len(email_text) > 3000:
        email_text = email_text[:3000] + "... [truncated]"
    full_text = f"Subject: {subject}\n\n{email_text}" if subject else email_text
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize this email in your casual style:\n\n{full_text}"}
            ],
            temperature=0.5,
            max_tokens=max_tokens,
            timeout=60,
        )
        return response.choices[0].message.content.strip() or "email received, nothing specific to report"
    except Exception:
        raise


def summarize_batch(emails, model, api_key, base_url, max_tokens, workers, checkpoint_every=50, save_callback=None):
    results = [None] * len(emails)
    errors = []
    completed = 0

    def job(idx):
        email = emails[idx]
        cleaned = email.get("cleaned_body", "")
        subject = email.get("subject", "")
        summary = summarize_one(cleaned, subject, model, api_key, base_url, max_tokens)
        return idx, summary, email

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(job, i): i for i in range(len(emails))}

        with tqdm(total=len(emails), desc="Summarizing") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    idx, summary, email = future.result()
                    email["summary"] = summary
                    email["summary_model"] = model
                    results[idx] = email
                except Exception as e:
                    errors.append({"index": idx, "error": str(e)})
                completed += 1
                pbar.update(1)

                if save_callback and completed % checkpoint_every == 0:
                    save_callback([r for r in results if r is not None])

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
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


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
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not args.api_key and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        print("  export OPENAI_API_KEY='your-key'")
        print("  or use --api-key <key>")
        return

    emails = load_emails(input_path)

    existing = load_emails(output_path)
    start_idx = len(existing)
    if start_idx > 0:
        print(f"Resuming from email {start_idx} (existing: {len(existing)} summaries)")
    emails = emails[start_idx:]

    print(f"Emails to summarize: {len(emails)}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers} (concurrent)")
    print(f"Checkpoint: every 50 completions")
    print(f"Estimated time: ~{len(emails) * 1.5 / args.workers / 60:.0f} min")
    print()

    checkpoint_count = [0]  # mutable counter

    def checkpoint(new_results):
        checkpoint_count[0] += 1
        all_r = existing + new_results
        save_results(all_r, output_path)
        print(f"  [checkpoint {checkpoint_count[0]}] saved {len(new_results)} summaries ({len(all_r):,} total)")

    results, errors = summarize_batch(
        emails, args.model, args.api_key, args.base_url, args.max_tokens, args.workers,
        checkpoint_every=50, save_callback=checkpoint
    )

    all_results = existing + results
    save_results(all_results, output_path)

    print(f"\nDone! Summaries: {len(all_results)}, Errors: {len(errors)}")
    print(f"Saved to: {output_path}")

    if all_results:
        print("\n" + "=" * 60)
        print("EXAMPLE SUMMARIES")
        print("=" * 60)
        for r in existing[-3:] + results[:3]:
            body = r.get("cleaned_body", "")[:150]
            summary = r.get("summary", "")
            print(f"\nEmail: {body}...")
            print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
