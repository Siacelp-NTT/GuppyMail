"""End-to-end email summarization pipeline helpers."""

from __future__ import annotations

from collections.abc import Callable
from collections import Counter
from pathlib import Path
from typing import TypedDict

from src.action_extractor import ActionExtractor, ActionItems
from src.classifier import Priority, PriorityAnalysis, PriorityClassifier
from src.preprocess import preprocess_email


SummaryFn = Callable[[str], str]
BASE_DIR = Path(__file__).resolve().parent.parent


class EmailPipelineResult(TypedDict):
    """Output shape shared by the UI and future Gmail pipeline."""

    email: str
    subject: str
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
    """Run preprocess -> summarize -> priority classification -> action extraction."""

    def __init__(
        self,
        summarizer: SummaryFn | None = None,
        classifier: PriorityClassifier | None = None,
        action_extractor: ActionExtractor | None = None,
    ) -> None:
        """Initialize the instance."""
        self.summarizer = summarizer or fallback_summary
        self.classifier = classifier or PriorityClassifier()
        self.action_extractor = action_extractor or ActionExtractor()
        self.history: list[EmailPipelineResult] = []

    def preprocess(self, email_text: str, subject: str = "") -> tuple[str, str]:
        """Clean email text and preserve the best available subject."""
        raw_text = email_text or ""
        parts = preprocess_email(raw_text)
        cleaned = " ".join((parts.get("body") or raw_text).split())
        effective_subject = subject.strip() or parts.get("subject", "").strip()
        if not cleaned:
            cleaned = " ".join(raw_text.split())
        return cleaned, effective_subject

    def summarize_email(self, email_text: str, subject: str = "") -> EmailPipelineResult:
        """Handle summarize email."""
        cleaned, effective_subject = self.preprocess(email_text, subject)
        summary = self.summarizer(cleaned)
        priority_analysis = self.classifier.get_priority_score(cleaned, effective_subject)
        action_items = self.action_extractor.extract(cleaned)
        result: EmailPipelineResult = {
            "email": cleaned,
            "subject": effective_subject,
            "summary": summary,
            "priority": priority_analysis["priority"],
            "priority_analysis": priority_analysis,
            "action_items": action_items,
        }
        self.add_to_history(result)
        return result

    def add_to_history(self, result: EmailPipelineResult) -> None:
        """Store recent pipeline results for dashboard stats."""
        self.history.append(result)
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def clear_history(self) -> None:
        """Clear session dashboard history."""
        self.history.clear()

    def get_dashboard_stats(self) -> dict:
        """Return session statistics for the app dashboard."""
        priority_counts = Counter(result.get("priority", "normal") for result in self.history)
        action_counts = Counter()
        for result in self.history:
            for key, values in result.get("action_items", {}).items():
                action_counts[key] += len(values)
        return {
            "total": len(self.history),
            "priority_breakdown": dict(priority_counts),
            "action_items": dict(action_counts),
            "recent": self.history[-10:],
        }

    def summarize_message(self, message) -> EmailPipelineResult:
        """Summarize a GmailMessage-like object with body and subject attributes."""
        return self.summarize_email(
            getattr(message, "body", ""),
            getattr(message, "subject", ""),
        )

    def summarize_batch(self, emails: list[str]) -> list[EmailPipelineResult]:
        """Summarize multiple raw email strings."""
        return [self.summarize_email(email) for email in emails]


_default_pipeline = EmailProcessingPipeline()


def summarize_email(email_text: str, subject: str = "") -> EmailPipelineResult:
    """Summarize and classify one email with the default lightweight pipeline."""
    return _default_pipeline.summarize_email(email_text, subject)


def create_pipeline(
    model_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> EmailProcessingPipeline | None:
    """Create a model-backed pipeline when exported artifacts exist."""
    minimal_model = BASE_DIR / "models" / "minimal" / "model.pt"
    minimal_tokenizer = BASE_DIR / "models" / "minimal" / "tokenizer.json"
    minimal_config = BASE_DIR / "models" / "minimal" / "config.json"
    checkpoint_model = BASE_DIR / "checkpoints" / "best_model.pt"
    quality_tokenizer = BASE_DIR / "data" / "training_quality" / "tokenizer.json"
    checkpoint_config = BASE_DIR / "checkpoints" / "config.json"

    if model_path and tokenizer_path:
        selected_model = Path(model_path)
        selected_tokenizer = Path(tokenizer_path)
        selected_config = Path(config_path) if config_path else selected_model.with_name("config.json")
    elif minimal_model.exists() and minimal_tokenizer.exists() and minimal_config.exists():
        selected_model = minimal_model
        selected_tokenizer = minimal_tokenizer
        selected_config = minimal_config
    elif checkpoint_model.exists() and quality_tokenizer.exists() and checkpoint_config.exists():
        selected_model = checkpoint_model
        selected_tokenizer = quality_tokenizer
        selected_config = checkpoint_config
    else:
        return None

    from inference import GuppyEmailInference

    engine = GuppyEmailInference(selected_model, selected_tokenizer, selected_config, device="auto")
    return EmailProcessingPipeline(summarizer=engine.generate_summary)
