"""Regex-based action item extraction for emails."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypedDict


class ActionItems(TypedDict, total=False):
    """Structured action items found in an email."""

    tasks: list[str]
    deadlines: list[str]
    requests: list[str]
    meetings: list[str]


@dataclass(frozen=True)
class ExtractorPattern:
    """Named compiled regex pattern used by the extractor."""

    name: str
    regex: re.Pattern[str]

    @classmethod
    def compile(cls, name: str, pattern: str) -> "ExtractorPattern":
        return cls(name=name, regex=re.compile(pattern, re.IGNORECASE | re.MULTILINE))


class ActionExtractor:
    """Extract explicit tasks, deadlines, requests, and meetings from email text."""

    def __init__(self) -> None:
        item_text = r"[^.\n?!;]+(?:\.(?!\s|$)[^.\n?!;]+)*"
        self.no_action_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r"\bno\s+action\s+(?:needed|required)\b",
                r"\bfor\s+your\s+information\s+only\b",
                r"\bfyi\s+only\b",
            ]
        ]
        self.task_patterns = [
            ExtractorPattern.compile(
                "polite_task",
                rf"\b(?:please|can you|could you|would you|need you to|you need to|we need to|must|should)\s+({item_text})",
            ),
            ExtractorPattern.compile(
                "imperative_task",
                rf"\b((?:send|submit|complete|finish|review|check|update|prepare|write|bring|provide|confirm|approve|sign|call|email|contact)\b{item_text})",
            ),
            ExtractorPattern.compile("labeled_task", rf"\b(?:action|todo|task)\s*:\s*({item_text})"),
        ]
        self.deadline_patterns = [
            ExtractorPattern.compile(
                "due_date_words",
                r"\b(?:due|deadline(?: is)?|needed)\s+(?:by|on|at|before)?\s*((?:today|tomorrow|tonight|eod|end of day(?:\s+today)?|end of week)|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b",
            ),
            ExtractorPattern.compile(
                "by_before_date",
                r"\b(?:by|before)\s+((?:end of day(?:\s+today)?|end of week|today|tomorrow|tonight|eod)|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
            ),
            ExtractorPattern.compile(
                "numeric_deadline",
                r"\b(?:due|deadline)\s*:?\s*(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
            ),
        ]
        self.request_patterns = [
            ExtractorPattern.compile(
                "polite_request",
                rf"\b(?:please|could you|would you|can you)\s+({item_text})",
            ),
            ExtractorPattern.compile(
                "let_me_know",
                rf"\b(?:let me know|let us know|notify me)\s+({item_text})",
            ),
            ExtractorPattern.compile(
                "contact_request",
                rf"\b((?:call|email|reach|contact)\s+(?:me|us|[a-z][\w.-]+@[\w.-]+\.\w+){item_text})",
            ),
        ]
        self.meeting_patterns = [
            ExtractorPattern.compile(
                "scheduled_meeting",
                rf"\b(?:meeting|call|conference)\s+(?:is\s+)?(?:scheduled\s+)?(?:for|on|at|moved to)\s+({item_text}?(?:\b\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm)\b|$))",
            ),
            ExtractorPattern.compile(
                "schedule_meeting",
                rf"\bschedule(?:d)?\s+(?:a\s+)?(?:meeting|call|conference)\s+(?:for|on|at)\s+({item_text})",
            ),
        ]

    def extract(self, email_text: str) -> ActionItems:
        """
        Extract action items from email text.

        Returns an empty dict when no action items are found.
        """
        text = self._normalize_text(email_text)
        if not text:
            return {}

        results: dict[str, list[str]] = {
            "tasks": self._extract_group(text, self.task_patterns, limit=5),
            "deadlines": self._extract_group(text, self.deadline_patterns, limit=3),
            "requests": self._extract_group(text, self.request_patterns, limit=3),
            "meetings": self._extract_group(text, self.meeting_patterns, limit=3),
        }

        if self._is_no_action_notice(text) and not any(
            results[key] for key in ("tasks", "deadlines", "requests", "meetings")
        ):
            return {}

        return {key: value for key, value in results.items() if value}

    def has_action_items(self, email_text: str) -> bool:
        """Return True when explicit action items are found."""
        return bool(self.extract(email_text))

    def _extract_group(
        self,
        text: str,
        patterns: list[ExtractorPattern],
        limit: int,
    ) -> list[str]:
        values: list[str] = []
        for pattern in patterns:
            for match in pattern.regex.finditer(text):
                value = self._match_text(match)
                value = self._clean_item(value)
                if self._is_valid_item(value):
                    values.append(value)
        return self._dedupe(values)[:limit]

    @staticmethod
    def _match_text(match: re.Match[str]) -> str:
        groups = [group for group in match.groups() if group]
        if groups:
            return " ".join(groups)
        return match.group(0)

    @staticmethod
    def _normalize_text(email_text: str) -> str:
        return re.sub(r"[ \t]+", " ", (email_text or "").replace("\r", "\n")).strip()

    @staticmethod
    def _clean_item(value: str) -> str:
        value = " ".join((value or "").split())
        value = re.sub(r"^(?:to|that|if|about)\s+", "", value, flags=re.IGNORECASE)
        return value.strip(" ,:-")

    def _is_valid_item(self, value: str) -> bool:
        if not value or len(value) < 3 or len(value) > 160:
            return False
        if any(pattern.search(value) for pattern in self.no_action_patterns):
            return False
        return True

    def _is_no_action_notice(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.no_action_patterns)

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for value in values:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(value)
        return deduped


def main() -> None:
    """Run a small CLI smoke test."""
    extractor = ActionExtractor()
    examples = [
        "Please send me the quarterly report by Friday. Also, call me at 555-1234 to discuss the budget.",
        "The meeting is scheduled for Monday at 2pm. Please review the proposal before then.",
        "Just wanted to keep you updated on the project status. No action needed.",
        "URGENT: Submit the invoice by end of day today. Contact john@company.com if you have questions.",
    ]

    for email in examples:
        print(f"Email: {email[:80]}...")
        results = extractor.extract(email)
        if results:
            print(f"  Tasks: {results.get('tasks', [])}")
            print(f"  Deadlines: {results.get('deadlines', [])}")
            print(f"  Requests: {results.get('requests', [])}")
            print(f"  Meetings: {results.get('meetings', [])}")
        else:
            print("  No action items found")
        print()


if __name__ == "__main__":
    main()
