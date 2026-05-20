"""Export guppyemail for deployment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.guppyemail_config import GuppyEmailConfig
from src.guppyemail_model import GuppyEmailLM


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(payload).__name__}")
    return payload


def load_config(ckpt: dict[str, Any], checkpoint_path: Path) -> GuppyEmailConfig:
    cfg = ckpt.get("model_config") or ckpt.get("config")
    if not cfg:
        config_path = checkpoint_path.with_name("config.json")
        if config_path.exists() and config_path.stat().st_size:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            cfg = raw.get("model", raw)
    return GuppyEmailConfig.from_dict(cfg)


def state_dict_from_checkpoint(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a state dict.")
    return state_dict


def convert_state_dict(
    state_dict: dict[str, torch.Tensor],
    dtype: str,
) -> dict[str, torch.Tensor]:
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if key != "lm_head.weight"
    }
    if dtype == "float32":
        return {key: value.detach().cpu() for key, value in state_dict.items()}
    if dtype != "float16":
        raise ValueError(f"Unsupported dtype: {dtype}")
    return {
        key: value.detach().cpu().half() if torch.is_floating_point(value) else value.detach().cpu()
        for key, value in state_dict.items()
    }


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1_000_000


def dir_size_mb(path: Path) -> float:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / 1_000_000


def write_config(config: GuppyEmailConfig, output_path: Path) -> None:
    output_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def write_manifest(
    output_dir: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    config: GuppyEmailConfig,
    ckpt: dict[str, Any],
    dtype: str,
    model_file: Path,
) -> None:
    model = GuppyEmailLM(config)
    manifest = {
        "model": "guppyemail",
        "format": "pytorch_state_dict",
        "weights": model_file.name,
        "dtype": dtype,
        "dropped_tied_weights": ["lm_head.weight"],
        "parameters": model.parameter_count(),
        "source_checkpoint": str(checkpoint_path.relative_to(BASE_DIR)),
        "source_tokenizer": str(tokenizer_path.relative_to(BASE_DIR)),
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_eval_loss": ckpt.get("eval_loss"),
        "checkpoint_eval_perplexity": ckpt.get("eval_perplexity"),
        "max_seq_len": config.max_seq_len,
        "vocab_size": config.vocab_size,
        "model_size_mb": round(file_size_mb(model_file), 2),
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def export_pytorch(
    checkpoint_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    dtype: str,
) -> Path:
    ckpt = load_checkpoint(checkpoint_path)
    config = load_config(ckpt, checkpoint_path)
    state_dict = convert_state_dict(state_dict_from_checkpoint(ckpt), dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "pytorch_model.bin"
    torch.save(state_dict, model_file)
    write_config(config, output_dir / "config.json")
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
    write_manifest(output_dir, checkpoint_path, tokenizer_path, config, ckpt, dtype, model_file)

    print(f"PyTorch export: {file_size_mb(model_file):.1f} MB -> {output_dir.relative_to(BASE_DIR)}/")
    return model_file


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    dtype: str,
    require_onnx: bool,
) -> bool:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        message = "ONNX export skipped: install `onnx` or run with an environment that provides it."
        if require_onnx:
            raise RuntimeError(message) from exc
        print(message)
        return False

    ckpt = load_checkpoint(checkpoint_path)
    config = load_config(ckpt, checkpoint_path)
    model = GuppyEmailLM(config)
    model.load_state_dict(state_dict_from_checkpoint(ckpt))
    model.eval()
    if dtype == "float16":
        model.half()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randint(0, config.vocab_size, (1, min(32, config.max_seq_len)), dtype=torch.long)
    try:
        torch.onnx.export(
            model,
            (dummy,),
            output_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq", 2: "vocab"},
            },
            opset_version=14,
            dynamo=False,
        )
    except ModuleNotFoundError as exc:
        message = (
            f"ONNX export skipped: missing `{exc.name}`. "
            "Install it or rerun with `--skip-onnx` to export only PyTorch/minimal artifacts."
        )
        if require_onnx:
            raise RuntimeError(message) from exc
        print(message)
        return False
    print(f"ONNX export: {file_size_mb(output_path):.1f} MB -> {output_path.relative_to(BASE_DIR)}")
    return True


def copy_minimal_sources(output_dir: Path) -> None:
    shutil.copy2(BASE_DIR / "inference.py", output_dir / "inference.py")
    src_dir = output_dir / "src"
    src_dir.mkdir(exist_ok=True)
    for name in [
        "__init__.py",
        "guppyemail_config.py",
        "guppyemail_model.py",
        "guppyemail_postprocess.py",
    ]:
        shutil.copy2(BASE_DIR / "src" / name, src_dir / name)


def write_runner(output_dir: Path) -> None:
    runner = '''#!/usr/bin/env python3
"""Interactive runner for exported guppyemail."""

from inference import GuppyEmailInference


def main():
    engine = GuppyEmailInference("model.pt", "tokenizer.json", "config.json", device="cpu")
    while True:
        prompt = input("\\nEmail> ").strip()
        if prompt.lower() in {"quit", "exit"}:
            break
        result = engine.chat_completion([{"role": "user", "content": prompt}])
        print(f"guppyemail> {result['choices'][0]['message']['content']}")


if __name__ == "__main__":
    main()
'''
    path = output_dir / "run.py"
    path.write_text(runner, encoding="utf-8")
    path.chmod(0o755)


def export_minimal(
    checkpoint_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    dtype: str,
) -> Path:
    ckpt = load_checkpoint(checkpoint_path)
    config = load_config(ckpt, checkpoint_path)
    state_dict = convert_state_dict(state_dict_from_checkpoint(ckpt), dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "model.pt"
    torch.save(state_dict, model_file)
    write_config(config, output_dir / "config.json")
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
    copy_minimal_sources(output_dir)
    write_runner(output_dir)
    write_manifest(output_dir, checkpoint_path, tokenizer_path, config, ckpt, dtype, model_file)

    print(f"Minimal export: {dir_size_mb(output_dir):.1f} MB -> {output_dir.relative_to(BASE_DIR)}/")
    return model_file


def verify_export(model_path: Path, tokenizer_path: Path, config_path: Path) -> str:
    from inference import GuppyEmailInference

    engine = GuppyEmailInference(model_path, tokenizer_path, config_path, device="cpu")
    return engine.generate_summary(
        "The budget review meeting has moved to Friday at 2pm. Please bring final numbers.",
        max_new_tokens=32,
        temperature=0.8,
        top_k=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export guppyemail deployment artifacts.")
    parser.add_argument("--checkpoint", default=str(BASE_DIR / "checkpoints" / "best_model.pt"))
    parser.add_argument("--tokenizer", default=str(BASE_DIR / "data" / "training_quality" / "tokenizer.json"))
    parser.add_argument("--pytorch-dir", default=str(BASE_DIR / "models" / "pytorch"))
    parser.add_argument("--minimal-dir", default=str(BASE_DIR / "models" / "minimal"))
    parser.add_argument("--onnx-path", default=str(BASE_DIR / "models" / "guppyemail.onnx"))
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--require-onnx", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    pytorch_dir = Path(args.pytorch_dir)
    minimal_dir = Path(args.minimal_dir)
    onnx_path = Path(args.onnx_path)

    print("Exporting guppyemail for deployment...")
    print(f"checkpoint={checkpoint_path.relative_to(BASE_DIR) if checkpoint_path.is_relative_to(BASE_DIR) else checkpoint_path}")
    print(f"tokenizer={tokenizer_path.relative_to(BASE_DIR) if tokenizer_path.is_relative_to(BASE_DIR) else tokenizer_path}")
    print(f"dtype={args.dtype}")

    export_pytorch(checkpoint_path, tokenizer_path, pytorch_dir, args.dtype)
    export_minimal(checkpoint_path, tokenizer_path, minimal_dir, args.dtype)
    if not args.skip_onnx:
        export_onnx(checkpoint_path, onnx_path, args.dtype, args.require_onnx)

    if not args.no_verify:
        summary = verify_export(minimal_dir / "model.pt", minimal_dir / "tokenizer.json", minimal_dir / "config.json")
        print(f"Verification summary: {summary}")

    print("Export complete.")


if __name__ == "__main__":
    main()
