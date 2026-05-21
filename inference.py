"""Inference helper for exported guppyemail models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

try:
    from src.guppyemail_config import GuppyEmailConfig
    from src.guppyemail_model import GuppyEmailLM
    from src.guppyemail_postprocess import fallback_to_email
except ModuleNotFoundError:
    from guppyemail_config import GuppyEmailConfig
    from guppyemail_model import GuppyEmailLM

    def fallback_to_email(
        email_text: str,
        generated_summary: str,
        max_email_chars: int | None = None,
    ) -> tuple[str, bool]:
        return generated_summary.strip(), False


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(config_path: str | Path) -> GuppyEmailConfig:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return GuppyEmailConfig.from_dict(raw.get("model", raw))


def build_prompt(email_text: str) -> str:
    return f"<|im_start|>user\n{email_text}<|im_end|>\n<|im_start|>assistant\n"


class GuppyEmailInference:
    """Small wrapper around the trained guppyemail checkpoint."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        config_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.config_path = Path(config_path) if config_path else self.model_path.with_name("config.json")
        self.device = select_device(device)

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.config = load_config(self.config_path)
        self.model = GuppyEmailLM(self.config).to(self.device)

        payload = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        load_result = self.model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"lm_head.weight"}
        missing = set(load_result.missing_keys)
        unexpected = set(load_result.unexpected_keys)
        if missing - allowed_missing or unexpected:
            raise RuntimeError(
                f"Invalid model weights: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        self.model.eval()

    @torch.no_grad()
    def generate_summary(
        self,
        email_text: str,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 1,
        use_generic_fallback: bool = True,
        fallback_max_chars: int | None = None,
    ) -> str:
        prompt_ids = self.tokenizer.encode(build_prompt(email_text)).ids
        prompt_ids = prompt_ids[-self.config.max_seq_len :]
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        output = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        generated_ids = output[0].tolist()[len(prompt_ids) :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        text = text.split("<|im_end|>", 1)[0]
        text = text.replace("<|im_start|>", "").replace("assistant", "")
        text = " ".join(text.strip().split())
        if use_generic_fallback:
            text, _ = fallback_to_email(email_text, text, fallback_max_chars)
        return text

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 80,
        top_k: int = 1,
    ) -> dict[str, Any]:
        user_messages = [message.get("content", "") for message in messages if message.get("role") == "user"]
        prompt = user_messages[-1] if user_messages else ""
        started = time.time()
        summary = self.generate_summary(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return {
            "id": f"guppyemail-{int(started * 1000)}",
            "object": "chat.completion",
            "created": int(started),
            "model": "guppyemail",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": summary},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run guppyemail inference.")
    parser.add_argument("email", nargs="?", default=None)
    parser.add_argument("--model", default="models/minimal/model.pt")
    parser.add_argument("--tokenizer", default="models/minimal/tokenizer.json")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    email = args.email or sys.stdin.read().strip()
    if not email:
        raise SystemExit("Provide email text as an argument or on stdin.")

    engine = GuppyEmailInference(args.model, args.tokenizer, args.config, args.device)
    print(
        engine.generate_summary(
            email,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    main()
