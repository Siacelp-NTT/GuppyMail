"""Lightweight email processing pipeline helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from src.action_extractor import ActionExtractor, ActionItems
from src.classifier import Priority, PriorityAnalysis, PriorityClassifier


SummaryFn = Callable[[str], str]


class EmailPipelineResult(TypedDict):
    """Output shape shared by the UI and future Gmail pipeline."""

    summary: str
    priority: Priority
    priority_analysis: PriorityAnalysis
    action_items: ActionItems


def fallback_summary(email_text: str) -> str:
    """Small placeholder until full model inference is wired in Task 4."""
    normalized = " ".join((email_text or "").split())
    if not normalized:
        return "No email text provided."
    return normalized[:240].rstrip() + ("..." if len(normalized) > 240 else "")


class EmailProcessingPipeline:
    """Combine a summarizer callable with rule-based priority classification."""

    def __init__(
        self,
        summarizer: SummaryFn | None = None,
        classifier: PriorityClassifier | None = None,
        action_extractor: ActionExtractor | None = None,
    ) -> None:
        self.summarizer = summarizer or fallback_summary
        self.classifier = classifier or PriorityClassifier()
        self.action_extractor = action_extractor or ActionExtractor()

    def summarize_email(self, email_text: str, subject: str = "") -> EmailPipelineResult:
        summary = self.summarizer(email_text)
        priority_analysis = self.classifier.get_priority_score(email_text, subject)
        action_items = self.action_extractor.extract(email_text)
        return {
            "summary": summary,
            "priority": priority_analysis["priority"],
            "priority_analysis": priority_analysis,
            "action_items": action_items,
        }


_default_pipeline = EmailProcessingPipeline()


def summarize_email(email_text: str, subject: str = "") -> EmailPipelineResult:
    """Summarize and classify one email with the default lightweight pipeline."""
    return _default_pipeline.summarize_email(email_text, subject)
