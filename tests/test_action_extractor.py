from __future__ import annotations

import pytest

from src.action_extractor import ActionExtractor
from src.pipeline import EmailProcessingPipeline, summarize_email


@pytest.fixture
def extractor() -> ActionExtractor:
    return ActionExtractor()


def test_extracts_explicit_task_deadline_and_request(extractor: ActionExtractor) -> None:
    email = (
        "Please send me the quarterly report by Friday. "
        "Also, call me at 555-1234 to discuss the budget."
    )
    result = extractor.extract(email)
    assert "tasks" in result
    assert any("send me the quarterly report" in task for task in result["tasks"])
    assert result["deadlines"] == ["Friday"]
    assert any("555-1234" in request for request in result["requests"])


def test_extracts_meeting_and_review_task(extractor: ActionExtractor) -> None:
    email = "The meeting is scheduled for Monday at 2pm. Please review the proposal before then."
    result = extractor.extract(email)
    assert any("review the proposal" in task for task in result["tasks"])
    assert any("Monday at 2pm" in meeting for meeting in result["meetings"])


@pytest.mark.parametrize(
    "email",
    [
        "",
        "Just wanted to keep you updated on the project status. No action needed.",
        "FYI only, the office lunch menu has been posted.",
    ],
)
def test_returns_empty_dict_when_no_actions(extractor: ActionExtractor, email: str) -> None:
    assert extractor.extract(email) == {}
    assert extractor.has_action_items(email) is False


def test_extracts_labeled_todo_and_numeric_deadline(extractor: ActionExtractor) -> None:
    email = "TODO: update the client deck. Deadline: 12/15/2026."
    result = extractor.extract(email)
    assert result["tasks"] == ["update the client deck"]
    assert result["deadlines"] == ["12/15/2026"]


def test_deduplicates_repeated_items(extractor: ActionExtractor) -> None:
    email = "Please send the report. Please send the report."
    result = extractor.extract(email)
    assert result["tasks"] == ["send the report"]


def test_pipeline_includes_action_items() -> None:
    pipeline = EmailProcessingPipeline(summarizer=lambda text: "report needed by friday.")
    result = pipeline.summarize_email("Please send the report by Friday.", "Report request")
    assert result["summary"] == "report needed by friday."
    assert result["priority"] == "important"
    assert result["action_items"]["tasks"] == ["send the report by Friday"]
    assert result["action_items"]["deadlines"] == ["Friday"]


def test_default_pipeline_includes_action_items() -> None:
    result = summarize_email("Action: bring your reports. Meeting moved to Feb 5 at 2pm.")
    assert result["action_items"]["tasks"] == ["bring your reports"]
    assert any("Feb 5 at 2pm" in meeting for meeting in result["action_items"]["meetings"])
