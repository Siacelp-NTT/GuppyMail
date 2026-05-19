"""Evaluate guppyemail on held-out email summarization data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.guppyemail_config import GuppyEmailConfig
from src.guppyemail_data import EmailSummaryDataset, collate_batch, load_chatml_rows
from src.guppyemail_model import GuppyEmailLM
from src.guppyemail_postprocess import fallback_to_email


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def load_config_from_checkpoint(ckpt: dict, checkpoint_path: Path) -> GuppyEmailConfig:
    cfg = to_dict(ckpt.get("model_config")) or to_dict(ckpt.get("config"))
    if not cfg:
        config_path = checkpoint_path.parent / "config.json"
        if config_path.exists():
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            cfg = raw.get("model", raw)
    return GuppyEmailConfig.from_dict(cfg)


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[GuppyEmailLM, GuppyEmailConfig, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt).__name__}")

    config = load_config_from_checkpoint(ckpt, checkpoint_path)
    model = GuppyEmailLM(config).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, ckpt


@torch.no_grad()
def compute_perplexity(
    model: GuppyEmailLM,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, int]:
    total_loss = 0.0
    total_tokens = 0
    model.eval()

    for batch_index, (x, y) in enumerate(tqdm(loader, desc="perplexity")):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        supervised_tokens = int((y != -100).sum().item())
        if supervised_tokens:
            total_loss += float(loss.item()) * supervised_tokens
            total_tokens += supervised_tokens

    if total_tokens == 0:
        return float("inf"), float("inf"), 0

    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(min(mean_loss, 20)), total_tokens


def build_prompt(email_text: str) -> str:
    return f"<|im_start|>user\n{email_text}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def generate_summary(
    model: GuppyEmailLM,
    tokenizer: Tokenizer,
    email_text: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    prompt_ids = tokenizer.encode(build_prompt(email_text)).ids
    prompt_ids = prompt_ids[-model.config.max_seq_len :]
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    output = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    generated_ids = output[0].tolist()[len(prompt_ids) :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    text = text.split("<|im_end|>", 1)[0]
    text = text.replace("<|im_start|>", "").replace("assistant", "")
    return " ".join(text.strip().split())


def compute_rouge(
    model: GuppyEmailLM,
    tokenizer: Tokenizer,
    rows: list[dict],
    device: torch.device,
    max_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    use_generic_fallback: bool,
    fallback_max_chars: int,
) -> dict:
    sample_rows = rows if max_samples == 0 else rows[:max_samples]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    samples = []
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for row in tqdm(sample_rows, desc="rouge"):
        raw_generated = generate_summary(
            model,
            tokenizer,
            row["email"],
            device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        generated, used_fallback = (
            fallback_to_email(row["email"], raw_generated, fallback_max_chars)
            if use_generic_fallback
            else (raw_generated, False)
        )
        scores = scorer.score(row["reference"], generated)
        result = {
            "email": row["email"][:500],
            "reference": row["reference"],
            "raw_generated": raw_generated,
            "generated": generated,
            "used_generic_fallback": used_fallback,
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
        for metric in totals:
            totals[metric] += result[metric]
        samples.append(result)

    count = len(samples)
    if count == 0:
        return {"sample_count": 0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "samples": []}

    return {
        "sample_count": count,
        "rouge1": totals["rouge1"] / count,
        "rouge2": totals["rouge2"] / count,
        "rougeL": totals["rougeL"] / count,
        "generic_fallback_count": sum(1 for sample in samples if sample["used_generic_fallback"]),
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate guppyemail on the test split.")
    parser.add_argument("--checkpoint", default=str(BASE_DIR / "checkpoints" / "best_model.pt"))
    parser.add_argument("--tokenizer", default=str(BASE_DIR / "data" / "training" / "tokenizer.json"))
    parser.add_argument("--data", default=str(BASE_DIR / "data" / "training" / "test.jsonl"))
    parser.add_argument("--output", default=str(BASE_DIR / "evaluation" / "eval_results.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--loss-batches", type=int, default=0, help="0 means all batches.")
    parser.add_argument("--rouge-samples", type=int, default=200, help="0 means all rows.")
    parser.add_argument("--sample-outputs", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1, help="1 gives deterministic greedy output.")
    parser.add_argument(
        "--generic-fallback",
        choices=["none", "email"],
        default="email",
        help="Replace generic no-summary outputs with the original email for user-facing results.",
    )
    parser.add_argument(
        "--fallback-max-chars",
        type=int,
        default=0,
        help="0 means keep the full email when generic fallback is used.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    data_path = Path(args.data)
    output_path = Path(args.output)

    device = select_device(args.device)
    print(f"device: {device}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model, config, ckpt = load_model(checkpoint_path, device)
    print(f"model: guppyemail")
    print(f"parameters: {model.parameter_count():,}")
    print(f"checkpoint step: {ckpt.get('step', 'unknown')}")
    print(f"checkpoint eval_loss: {ckpt.get('eval_loss', 'unknown')}")

    dataset = EmailSummaryDataset(data_path, tokenizer, config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, config.pad_id),
        pin_memory=device.type == "cuda",
    )
    rows = load_chatml_rows(data_path)
    print(f"test rows: {len(rows):,}; loss rows loaded: {len(dataset):,}; skipped: {dataset.skipped:,}")

    max_batches = None if args.loss_batches == 0 else args.loss_batches
    test_loss, test_perplexity, supervised_tokens = compute_perplexity(
        model, loader, device, max_batches=max_batches
    )
    print(f"test_loss: {test_loss:.4f}")
    print(f"test_perplexity: {test_perplexity:.2f}")

    rouge = compute_rouge(
        model,
        tokenizer,
        rows,
        device,
        max_samples=args.rouge_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_generic_fallback=args.generic_fallback == "email",
        fallback_max_chars=None if args.fallback_max_chars == 0 else args.fallback_max_chars,
    )
    print(f"ROUGE-1: {rouge['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge['rougeL']:.4f}")
    print(f"generic fallbacks: {rouge['generic_fallback_count']}")

    results = {
        "model": "guppyemail",
        "split": str(data_path.relative_to(BASE_DIR) if data_path.is_relative_to(BASE_DIR) else data_path),
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(BASE_DIR) if checkpoint_path.is_relative_to(BASE_DIR) else checkpoint_path),
            "step": ckpt.get("step"),
            "eval_loss": ckpt.get("eval_loss"),
            "eval_perplexity": ckpt.get("eval_perplexity"),
        },
        "config": vars(config),
        "evaluation": {
            "test_loss": test_loss,
            "test_perplexity": test_perplexity,
            "supervised_tokens": supervised_tokens,
            "loss_batches": args.loss_batches,
            "rouge_sample_count": rouge["sample_count"],
            "rouge_generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "generic_fallback": args.generic_fallback,
                "fallback_max_chars": args.fallback_max_chars,
                "seed": args.seed,
            },
            "generic_fallback_count": rouge["generic_fallback_count"],
        },
        "rouge": {
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
        },
        "samples": rouge["samples"][: args.sample_outputs],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
