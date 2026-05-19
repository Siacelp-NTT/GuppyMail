"""Post-processing helpers for guppyemail outputs."""

from __future__ import annotations

import re


GENERIC_NO_SUMMARY_PATTERNS = [
    re.compile(r"^\s*email received[,;:]?\s*nothing specific to report\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*nothing specific to report\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*no specific action(?:s)?(?: needed| required)?\.?\s*$", re.IGNORECASE),
]


def is_generic_no_summary(text: str) -> bool:
    """Return True when the model emitted a generic non-summary fallback."""
    normalized = " ".join((text or "").strip().split())
    return any(pattern.match(normalized) for pattern in GENERIC_NO_SUMMARY_PATTERNS)


def fallback_to_email(
    email_text: str,
    generated_summary: str,
    max_email_chars: int | None = None,
) -> tuple[str, bool]:
    """
    Replace generic no-summary outputs with the original email text.

    This is intended for user-facing output. Pure model evaluation can disable it.
    """
    if not is_generic_no_summary(generated_summary):
        return generated_summary.strip(), False

    email = " ".join((email_text or "").strip().split())
    if max_email_chars is not None and len(email) > max_email_chars:
        email = email[:max_email_chars].rstrip() + "..."
    return email, True
