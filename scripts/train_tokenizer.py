"""
Prepare the guppyemail tokenizer and ChatML training splits for email summarization.

Outputs:
    data/training/tokenizer.json
    data/training/train.jsonl
    data/training/eval.jsonl
    data/training/val.jsonl
    data/training/test.jsonl

Usage:
    python scripts/train_tokenizer.py
"""

import argparse
import json
import random
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SUM_DIR = DATA_DIR / "summaries"
TRAIN_DIR = DATA_DIR / "training"

SPECIAL_TOKENS = ["<pad>", "<|im_start|>", "<|im_end|>"]


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_email(row):
    return (
        row.get("cleaned_body")
        or row.get("email")
        or row.get("text")
        or row.get("message")
        or ""
    ).strip()


def record_summary(row):
    return (row.get("summary") or "").strip()


def full_chatml(email, summary):
    return (
        f"<|im_start|>user\n{email}<|im_end|>\n"
        f"<|im_start|>assistant\n{summary}<|im_end|>"
    )


def train_tokenizer(texts, output_path, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(str(output_path))
    return tokenizer


def fit_chatml(tokenizer, email, summary, max_seq_len):
    text = full_chatml(email, summary)
    ids = tokenizer.encode(text).ids
    if len(ids) <= max_seq_len:
        return text, len(ids), False

    prefix = "<|im_start|>user\n"
    middle = "<|im_end|>\n<|im_start|>assistant\n"
    suffix = "<|im_end|>"
    fixed_ids = tokenizer.encode(prefix + middle + summary + suffix).ids
    available_email_tokens = max(0, int(max_seq_len) - len(fixed_ids))

    email_ids = tokenizer.encode(email).ids[:available_email_tokens]
    short_email = tokenizer.decode(email_ids, skip_special_tokens=False).strip()
    fitted = full_chatml(short_email, summary)
    fitted_ids = tokenizer.encode(fitted).ids

    if len(fitted_ids) > max_seq_len:
        fitted = tokenizer.decode(fitted_ids[: int(max_seq_len)], skip_special_tokens=False)
        fitted_ids = tokenizer.encode(fitted).ids

    return fitted, len(fitted_ids), True


def build_rows(source_rows, tokenizer, max_seq_len):
    formatted = []
    skipped = 0
    truncated = 0

    for row in source_rows:
        email = record_email(row)
        summary = record_summary(row)
        if not email or not summary:
            skipped += 1
            continue

        text, token_count, was_truncated = fit_chatml(tokenizer, email, summary, max_seq_len)
        truncated += int(was_truncated)
        formatted.append(
            {
                "text": text,
                "category": "email_summary",
                "subject": row.get("subject", ""),
                "source_hash": row.get("source_hash", ""),
                "token_count": token_count,
                "truncated": was_truncated,
            }
        )

    return formatted, skipped, truncated


def split_rows(rows, val_pct, test_pct, seed):
    rng = random.Random(int(seed))
    rows = list(rows)
    rng.shuffle(rows)

    total = len(rows)
    test_n = int(total * float(test_pct) / 100)
    val_n = int(total * float(val_pct) / 100)

    test = rows[:test_n]
    eval_rows = rows[test_n : test_n + val_n]
    train = rows[test_n + val_n :]
    return train, eval_rows, test


def prepare_training_data(
    input_path,
    output_dir,
    vocab_size=4096,
    max_seq_len=128,
    val_pct=10,
    test_pct=5,
    seed=42,
):
    rows = load_jsonl(input_path)
    if not rows:
        raise FileNotFoundError(f"No source rows found at {input_path}")

    full_texts = [full_chatml(record_email(r), record_summary(r)) for r in rows if record_email(r) and record_summary(r)]
    if not full_texts:
        raise ValueError("No usable email/summary pairs found in source file.")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer = train_tokenizer(full_texts, tokenizer_path, vocab_size)

    formatted, skipped, truncated = build_rows(rows, tokenizer, int(max_seq_len))
    train, eval_rows, test = split_rows(formatted, val_pct, test_pct, seed)

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)
    write_jsonl(output_dir / "val.jsonl", eval_rows)
    write_jsonl(output_dir / "test.jsonl", test)

    metrics = {
        "source": str(input_path),
        "source_rows": len(rows),
        "usable_rows": len(formatted),
        "skipped_rows": skipped,
        "truncated_rows": truncated,
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "test_rows": len(test),
        "vocab_size": tokenizer.get_vocab_size(),
        "max_seq_len": int(max_seq_len),
        "tokenizer": str(tokenizer_path),
    }
    (output_dir / "tokenizer_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train guppyemail BPE tokenizer and ChatML splits.")
    parser.add_argument("--input", default=str(SUM_DIR / "en_summaries.clean.jsonl"))
    parser.add_argument("--output-dir", default=str(TRAIN_DIR))
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--val-pct", type=float, default=10)
    parser.add_argument("--test-pct", type=float, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = prepare_training_data(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
