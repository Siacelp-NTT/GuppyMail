"""Configuration objects for guppyemail."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class GuppyEmailConfig:
    vocab_size: int = 4096
    max_seq_len: int = 512
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_hidden: int = 768
    dropout: float = 0.1
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    @classmethod
    def from_dict(cls, values: dict | None) -> "GuppyEmailConfig":
        if not values:
            return cls()
        valid = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in valid})


@dataclass
class TrainConfig:
    batch_size: int = 16
    grad_accum_steps: int = 2
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 10000
    eval_interval: int = 250
    eval_batches: int = 80
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 2

    @classmethod
    def from_dict(cls, values: dict | None) -> "TrainConfig":
        if not values:
            return cls()
        valid = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in valid})
