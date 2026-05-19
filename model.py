"""Compatibility exports for the guppyemail model."""

from src.guppyemail_model import GuppyEmailLM

GuppyLM = GuppyEmailLM

__all__ = ["GuppyEmailLM", "GuppyLM"]
