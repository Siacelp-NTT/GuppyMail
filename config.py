"""Compatibility exports for guppyemail configuration."""

from src.guppyemail_config import GuppyEmailConfig, TrainConfig

GuppyConfig = GuppyEmailConfig

__all__ = ["GuppyEmailConfig", "GuppyConfig", "TrainConfig"]
