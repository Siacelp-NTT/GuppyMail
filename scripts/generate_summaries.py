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

import json
import hashlib
import os
import re
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, wait as fut_wait, FIRST_COMPLETED
from pathlib import Path
from tqdm import tqdm

import requests


def now_ts():
    """Handle now ts."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_duration(seconds):
    """Handle fmt duration."""
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def short_error(error):
    """Handle short error."""
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

STRUCTURED_SYSTEM_PROMPT = """You create training labels for guppyemail, a tiny email summarization model.

Return ONLY valid JSON with this schema:
{"summary_type":"summary|no_summary_needed|noise","quality":"good|weak|noise","summary":"short casual lowercase summary","reason":"brief reason"}

Rules:
- If the email has concrete information, summarize it in 1-2 short lowercase sentences.
- Mention dates, deadlines, meetings, requests, decisions, and action items when present.
- Do not use vague fallback text like "email received, nothing specific to report" as a summary.
- If the email is too short, empty, only an attachment notice, only quoted history, or has no useful content, set summary_type to "no_summary_needed", quality to "weak", and summary to "".
- If the email is mostly markup, image placeholders, unsubscribe text, or corrupted content, set summary_type to "noise", quality to "noise", and summary to "".
- Keep useful summaries under 50 words and reason under 12 words.
- Return minified one-line JSON only. No markdown, code fences, comments, or extra text.
"""

GENERIC_SUMMARY_RE = re.compile(
    r"^\s*(email received[,;:]?\s*)?nothing specific to report\.?\s*$",
    re.IGNORECASE,
)

DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
DASHSCOPE_MULTIMODAL_MODELS = {
    "qwen3.5-plus",
    "qwen3.5-plus-2026-02-15",
}


def normalize_hash(text):
    """Handle normalize hash."""
    normalized = re.sub(r"\s+", " ", (text or "")).strip().lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def resolve_provider(provider, model):
    """Handle resolve provider."""
    provider = (provider or "auto").strip().lower()
    if provider != "auto":
        return provider
    return "dashscope" if (model or "").startswith("qwen") else "openai"


def dashscope_endpoint(base_url, model):
    """Handle dashscope endpoint."""
    base = (base_url or DASHSCOPE_DEFAULT_BASE_URL).rstrip("/")
    if dashscope_uses_multimodal(model):
        return base + "/services/aigc/multimodal-generation/generation"
    return base + "/services/aigc/text-generation/generation"


def dashscope_uses_multimodal(model):
    """Handle dashscope uses multimodal."""
    model_name = (model or "").lower()
    return model_name in DASHSCOPE_MULTIMODAL_MODELS or "-vl" in model_name


def dashscope_content(model, text):
    """Handle dashscope content."""
    if dashscope_uses_multimodal(model):
        return [{"text": text}]
    return text


def extract_dashscope_content(data):
    """Handle extract dashscope content."""
    output = data.get("output") or {}
    choices = output.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
    return str(output.get("text", "") or "")


def parse_structured_summary(content):
    """Handle parse structured summary."""
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    if not raw.startswith("{") and "{" in raw and "}" in raw:
        raw = raw[raw.find("{") : raw.rfind("}") + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return malformed_summary(content, f"malformed_json: {short_error(exc)}")

    if not isinstance(data, dict):
        return malformed_summary(content, "structured output was not a json object")

    summary = " ".join(str(data.get("summary", "")).strip().split())
    summary_type = str(data.get("summary_type", "")).strip().lower()
    quality = str(data.get("quality", "")).strip().lower()
    reason = str(data.get("reason", "")).strip()

    if summary_type not in {"summary", "no_summary_needed", "noise"}:
        return malformed_summary(content, f"invalid summary_type: {summary_type or 'missing'}")
    if quality not in {"good", "weak", "noise"}:
        return malformed_summary(content, f"invalid quality: {quality or 'missing'}")

    if GENERIC_SUMMARY_RE.match(summary):
        summary = ""
        summary_type = "no_summary_needed"
        quality = "weak"
        reason = reason or "generic fallback removed"
    elif summary_type == "summary" and not summary:
        return malformed_summary(content, "summary label had empty summary")
    elif summary_type in {"no_summary_needed", "noise"} and summary:
        return malformed_summary(content, f"{summary_type} label had non-empty summary")

    return {
        "summary": summary,
        "summary_type": summary_type,
        "summary_quality": quality,
        "summary_reason": reason,
        "summary_raw_response": content,
    }


def malformed_summary(content, reason):
    """Handle malformed summary."""
    return {
        "summary": "",
        "summary_type": "malformed",
        "summary_quality": "noise",
        "summary_reason": reason,
        "summary_raw_response": content,
    }


def summarize_one(api_key, base_url, email_text, subject, model, max_tokens, structured=True, temperature=0.0, provider="openai"):
    """Handle summarize one."""
    if len(email_text) > 3000:
        email_text = email_text[:3000] + "... [truncated]"
    full_text = f"Subject: {subject}\n\n{email_text}" if subject else email_text
    user_prompt = f"Analyze this email for guppyemail training:\n\n{full_text}" if structured else f"Summarize this email in your casual style:\n\n{full_text}"
    system_prompt = STRUCTURED_SYSTEM_PROMPT if structured else SYSTEM_PROMPT

    if provider == "dashscope":
        url = dashscope_endpoint(base_url, model)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Connection": "close"}
        body = {
            "model": model,
            "input": {
                "messages": [
                    {"role": "system", "content": dashscope_content(model, system_prompt)},
                    {"role": "user", "content": dashscope_content(model, user_prompt)},
                ]
            },
            "parameters": {
                "temperature": temperature,
                "result_format": "message",
            },
        }
        if max_tokens and max_tokens > 0:
            body["parameters"]["max_tokens"] = max_tokens
        if structured:
            body["parameters"]["response_format"] = {"type": "json_object"}
        resp = requests.post(url, json=body, headers=headers, timeout=(30, 120))
        resp.raise_for_status()
        content = extract_dashscope_content(resp.json()).strip()
        if structured:
            return parse_structured_summary(content)
        return {
            "summary": content or "",
            "summary_type": "summary" if content and not GENERIC_SUMMARY_RE.match(content) else "no_summary_needed",
            "summary_quality": "good" if content and not GENERIC_SUMMARY_RE.match(content) else "weak",
            "summary_reason": "dashscope plain summary",
            "summary_raw_response": content,
        }

    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Connection": "close"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if structured:
        body["response_format"] = {"type": "json_object"}
    resp = requests.post(url, json=body, headers=headers, timeout=(30, 120))
    if structured and resp.status_code in {400, 422} and "response_format" in body:
        body.pop("response_format", None)
        resp = requests.post(url, json=body, headers=headers, timeout=(30, 120))
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    if structured:
        return parse_structured_summary(content)
    return {
        "summary": content or "",
        "summary_type": "summary" if content and not GENERIC_SUMMARY_RE.match(content) else "no_summary_needed",
        "summary_quality": "good" if content and not GENERIC_SUMMARY_RE.match(content) else "weak",
        "summary_reason": "legacy plain summary",
        "summary_raw_response": content,
    }


def summarize_batch(emails, model, api_key, base_url, max_tokens, workers, checkpoint_every=50, save_callback=None, echo_fn=print, cooldown_every=100, cooldown_secs=15, max_per_batch=0, structured=True, temperature=0.0, provider="openai"):
    """Handle summarize batch."""
    results = [None] * len(emails)
    errors = []
    completed = 0
    started = time.time()
    last_hb = time.time()
    last_progress = time.time()
    last_progress_count = 0
    warned_stall = False
    next_cooldown = cooldown_every

    if provider == "dashscope":
        key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        url = base_url or os.environ.get("DASHSCOPE_BASE_URL", DASHSCOPE_DEFAULT_BASE_URL)
    else:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def job(idx):
        """Handle job."""
        email = emails[idx]
        cleaned = email.get("cleaned_body", "")
        subject = email.get("subject", "")
        for attempt in range(3):
            try:
                summary_data = summarize_one(key, url, cleaned, subject, model, max_tokens, structured=structured, temperature=temperature, provider=provider)
                return idx, summary_data, email
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
                        idx, summary_data, email = future.result()
                    except Exception as e:
                        errors.append({"index": idx, "error": short_error(e)})
                        email = emails[idx]
                        email["summary"] = "error: could not summarize"
                        email["summary_type"] = "error"
                        email["summary_quality"] = "noise"
                        email["summary_reason"] = short_error(e)
                        email["summary_model"] = model
                    else:
                        summary_data = summary_data or {}
                        email["summary"] = summary_data.get("summary", "")
                        email["summary_type"] = summary_data.get("summary_type", "summary" if email["summary"] else "no_summary_needed")
                        email["summary_quality"] = summary_data.get("summary_quality", "good" if email["summary"] else "weak")
                        email["summary_reason"] = summary_data.get("summary_reason", "")
                        if "summary_raw_response" in summary_data:
                            email["summary_raw_response"] = summary_data["summary_raw_response"]
                        email["summary_model"] = model
                        if email["summary_type"] == "malformed":
                            errors.append({"index": idx, "error": email.get("summary_reason", "malformed structured output")})
                        if not email["summary"] and email["summary_type"] == "summary":
                            email["summary_type"] = "no_summary_needed"
                            email["summary_quality"] = "weak"
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
    """Load emails."""
    emails = []
    if not os.path.exists(path):
        return emails
    with open(path) as f:
        for line in f:
            try:
                emails.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return emails


def save_results(results, path):
    """Save results."""
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
    save_malformed_results(results, path)


def save_malformed_results(results, path):
    """Save malformed results."""
    malformed = [r for r in results if r.get("summary_type") == "malformed"]
    malformed_path = str(Path(path).with_suffix(".malformed.jsonl"))
    if not malformed:
        if os.path.exists(malformed_path):
            os.remove(malformed_path)
        return

    tmp = malformed_path + ".tmp"
    with open(tmp, "w") as f:
        for r in malformed:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, malformed_path)


def archive_summary_outputs(output_path):
    """Move existing summary outputs aside so a clean redo cannot append to them."""
    output = Path(output_path)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = output.parent / "archive"
    archived = []

    candidates = [output]
    if output.name == "en_summaries.jsonl":
        candidates.append(output.with_name("en_summaries.clean.jsonl"))
        candidates.append(output.with_name("en_summaries.malformed.jsonl"))

    for path in candidates:
        if not path.exists():
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_path = archive_dir / f"{path.stem}.{timestamp}{path.suffix}"
        counter = 1
        while archived_path.exists():
            archived_path = archive_dir / f"{path.stem}.{timestamp}.{counter}{path.suffix}"
            counter += 1
        os.replace(path, archived_path)
        archived.append((path, archived_path))

    return archived


def main():
    """Run the command-line entry point."""
    os.chdir(os.path.dirname(__file__) + "/.." if "__file__" in dir() else ".")

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent API workers (default: 3)")
    parser.add_argument("--input", default="data/processed/cleaned_emails.jsonl")
    parser.add_argument("--output", default="data/summaries/en_summaries.jsonl")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--provider", choices=["auto", "openai", "dashscope"], default=os.environ.get("LLM_PROVIDER", "auto"))
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--legacy-summary", action="store_true", help="Disable structured JSON labels and request plain summaries only.")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--log", default="", help="Write output to this log file")
    parser.add_argument("--cooldown-every", type=int, default=100, help="Pause after this many requests (default: 100)")
    parser.add_argument("--cooldown-secs", type=int, default=15, help="Seconds to pause (default: 15)")
    parser.add_argument("--max-per-batch", type=int, default=1500, help="Max requests per session, then exit for clean restart (default: 1500)")
    parser.add_argument("--reset-output", action="store_true", help="Archive existing output files before starting a clean redo.")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log = open(args.log, "a", buffering=1) if args.log else None

    provider = resolve_provider(args.provider, args.model)
    has_key = bool(args.api_key)
    if provider == "dashscope":
        has_key = has_key or bool(os.environ.get("DASHSCOPE_API_KEY"))
    else:
        has_key = has_key or bool(os.environ.get("OPENAI_API_KEY"))

    if not has_key:
        def echo(msg):
            """Handle echo."""
            print(msg)
            if log:
                log.write(msg + "\n")
        key_name = "DASHSCOPE_API_KEY" if provider == "dashscope" else "OPENAI_API_KEY"
        echo(f"ERROR: {key_name} not set!")
        echo(f"  export {key_name}='your-key'")
        echo("  or use --api-key <key>")
        return

    def echo(msg):
        """Handle echo."""
        print(msg, flush=True)
        if log:
            log.write(msg + "\n")
            log.flush()

    run_started = time.time()
    emails = load_emails(input_path)

    if args.reset_output:
        archived = archive_summary_outputs(output_path)
    else:
        archived = []

    existing = load_emails(output_path)
    start_idx = len(existing)
    echo("=" * 72)
    echo(f"[{now_ts()}] summary generation started")
    echo(f"input={input_path}")
    echo(f"output={output_path}")
    structured = not args.legacy_summary
    echo(f"provider={provider} model={args.model} workers={args.workers} max_tokens={args.max_tokens} temperature={args.temperature} structured={structured}")
    echo(f"cooldown_every={args.cooldown_every} cooldown_secs={args.cooldown_secs} max_per_batch={args.max_per_batch}")
    if archived:
        for old_path, archived_path in archived:
            echo(f"archived={old_path} -> {archived_path}")
    echo(f"input_rows={len(emails):,} existing_output_rows={len(existing):,}")
    if start_idx > 0:
        echo(f"[{now_ts()}] loaded existing summaries; scanning input for missing hashes")

    # dedup: skip cleaned_bodies already summarized (shifted after re-download)
    seen_hashes = set()
    for r in existing:
        body = r.get("cleaned_body", "")
        if body:
            seen_hashes.add(normalize_hash(body))

    deduped = []
    skipped = 0
    for e in emails:
        body = e.get("cleaned_body", "")
        h = normalize_hash(body)
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
        """Handle checkpoint."""
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
            max_per_batch=args.max_per_batch, structured=structured, temperature=args.temperature,
            provider=provider,
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
            summary_type = r.get("summary_type", "")
            quality = r.get("summary_quality", "")
            echo(f"\nEmail: {body}...")
            echo(f"Summary [{summary_type}/{quality}]: {summary}")

    if log:
        log.close()


if __name__ == "__main__":
    main()
