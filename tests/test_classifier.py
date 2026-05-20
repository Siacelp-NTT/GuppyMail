from __future__ import annotations

import pytest

from src.classifier import PriorityClassifier
from src.pipeline import EmailProcessingPipeline, summarize_email


@pytest.fixture
def classifier() -> PriorityClassifier:
    return PriorityClassifier()


@pytest.mark.parametrize(
    ("subject", "body"),
    [
        ("URGENT: deadline today", "Please send the report ASAP."),
        ("Action required", "Legal needs the signed contract immediately."),
        ("Critical alert", "Respond now, the access token expires in 24 hours."),
    ],
)
def test_classifies_urgent_keywords(classifier: PriorityClassifier, subject: str, body: str) -> None:
    assert classifier.classify(body, subject) == "urgent"


@pytest.mark.parametrize(
    ("subject", "body"),
    [
        ("Payment Reminder", "Your invoice is due soon."),
        ("Contract Review", "Please review the contract before Friday."),
        ("Meeting", "Confirm your attendance for the required meeting."),
    ],
)
def test_classifies_important_keywords(
    classifier: PriorityClassifier, subject: str, body: str
) -> None:
    assert classifier.classify(body, subject) == "important"


@pytest.mark.parametrize(
    ("subject", "body"),
    [
        ("Newsletter", "This week's news update includes a sale and discount code."),
        ("FYI", "For your information only. No action needed."),
        ("Automated notice", "This noreply message is an automatic receipt."),
    ],
)
def test_classifies_low_priority_keywords(
    classifier: PriorityClassifier, subject: str, body: str
) -> None:
    assert classifier.classify(body, subject) == "low"


def test_defaults_to_normal_without_patterns(classifier: PriorityClassifier) -> None:
    body = "The team had lunch together and shared notes from last week."
    assert classifier.classify(body, "Quick note") == "normal"


def test_higher_priority_wins_over_low_priority(classifier: PriorityClassifier) -> None:
    body = "Automated alert: critical issue detected. Action required immediately."
    assert classifier.classify(body, "FYI") == "urgent"


def test_priority_score_includes_matches_and_reason(classifier: PriorityClassifier) -> None:
    result = classifier.get_priority_score("Please approve the proposal before Friday.")
    assert result["priority"] == "important"
    assert result["matches"]
    assert "business" in result["reason"].lower()


def test_pipeline_returns_summary_and_priority() -> None:
    pipeline = EmailProcessingPipeline(summarizer=lambda text: "meeting moved to friday at 2pm.")
    result = pipeline.summarize_email(
        "The meeting moved to Friday at 2pm. Please confirm your attendance.",
        "Meeting update",
    )
    assert result["summary"] == "meeting moved to friday at 2pm."
    assert result["priority"] == "important"
    assert result["priority_analysis"]["priority"] == "important"
    assert result["action_items"]


def test_default_pipeline_function() -> None:
    result = summarize_email("URGENT: production issue needs response now.")
    assert result["priority"] == "urgent"
    assert result["summary"]
    assert "action_items" in result
