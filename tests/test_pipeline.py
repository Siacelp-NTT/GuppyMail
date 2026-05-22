from __future__ import annotations

from dataclasses import dataclass

from src.pipeline import EmailProcessingPipeline, summarize_email


@dataclass
class Message:
    subject: str
    body: str


def test_pipeline_preprocesses_before_summarizing() -> None:
    seen = {}

    def summarizer(text: str) -> str:
        seen["text"] = text
        return "current message summary"

    pipeline = EmailProcessingPipeline(summarizer=summarizer)
    result = pipeline.summarize_email(
        "Please send the report by Friday.\n\n-----Original Message-----\nOld thread text.",
        "Report request",
    )

    assert seen["text"] == "Please send the report by Friday."
    assert result["email"] == "Please send the report by Friday."
    assert result["subject"] == "Report request"
    assert result["summary"] == "current message summary"
    assert result["priority"] == "important"
    assert result["action_items"]["deadlines"] == ["Friday"]


def test_pipeline_can_use_subject_from_headers() -> None:
    pipeline = EmailProcessingPipeline(summarizer=lambda text: "header summary")
    result = pipeline.summarize_email(
        "From: a@example.com\nSubject: URGENT Deadline today\n\nPlease submit the proposal ASAP."
    )
    assert result["subject"] == "URGENT Deadline today"
    assert result["priority"] == "urgent"


def test_pipeline_summarizes_message_object() -> None:
    pipeline = EmailProcessingPipeline(summarizer=lambda text: "message summary")
    message = Message(
        subject="Meeting",
        body="The meeting is scheduled for Monday at 2pm. Please confirm attendance.",
    )
    result = pipeline.summarize_message(message)
    assert result["summary"] == "message summary"
    assert result["subject"] == "Meeting"
    assert result["action_items"]["meetings"] == ["Monday at 2pm"]


def test_pipeline_summarizes_batches() -> None:
    pipeline = EmailProcessingPipeline(summarizer=lambda text: text[:8])
    results = pipeline.summarize_batch(["First email body.", "Second email body."])
    assert [result["summary"] for result in results] == ["First em", "Second e"]
    assert pipeline.get_dashboard_stats()["total"] == 2


def test_pipeline_dashboard_stats_and_clear_history() -> None:
    pipeline = EmailProcessingPipeline(summarizer=lambda text: "summary")
    pipeline.summarize_email("URGENT: submit the proposal ASAP.", "Deadline today")
    pipeline.summarize_email("No action needed. Newsletter update.", "FYI")

    stats = pipeline.get_dashboard_stats()
    assert stats["total"] == 2
    assert stats["priority_breakdown"]["urgent"] == 1
    assert stats["priority_breakdown"]["low"] == 1
    assert stats["action_items"]["tasks"] == 1
    assert len(stats["recent"]) == 2

    pipeline.clear_history()
    assert pipeline.get_dashboard_stats()["total"] == 0


def test_default_pipeline_keeps_existing_fallback_behavior() -> None:
    result = summarize_email("Action: bring your reports. Meeting moved to Feb 5 at 2pm.")
    assert result["summary"]
    assert result["action_items"]["tasks"] == ["bring your reports"]
