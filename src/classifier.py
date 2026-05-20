"""Rule-based email priority classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, TypedDict


Priority = Literal["urgent", "important", "normal", "low"]


class PriorityAnalysis(TypedDict):
    """Detailed classifier output for dashboards and debugging."""

    priority: Priority
    matches: list[str]
    reason: str


@dataclass(frozen=True)
class PatternGroup:
    """Compiled regex group for one priority label."""

    priority: Priority
    reason: str
    patterns: tuple[re.Pattern[str], ...]

    @classmethod
    def compile(cls, priority: Priority, reason: str, patterns: list[str]) -> "PatternGroup":
        return cls(
            priority=priority,
            reason=reason,
            patterns=tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns),
        )

    def matches(self, text: str) -> list[str]:
        return [pattern.pattern for pattern in self.patterns if pattern.search(text)]


class PriorityClassifier:
    """Classify email priority using deterministic keyword and phrase rules."""

    def __init__(self) -> None:
        self.urgent = PatternGroup.compile(
            "urgent",
            "Contains urgent timing, escalation, or action-required language.",
            [
                r"\burgent\b",
                r"\basap\b",
                r"\bimmediate(?:ly)?\b",
                r"\bcritical\b",
                r"\bemergency\b",
                r"\bpriority\s*(?:1|one|high)\b",
                r"\bhigh\s*priority\b",
                r"\baction\s*required\b",
                r"\brespond\s+(?:now|immediately|asap)\b",
                r"\bneed(?:ed|s)?\s+(?:now|immediately|asap)\b",
                r"\bdeadline\b.{0,80}\b(?:today|tonight|eod|end of day)\b",
                r"\b(?:today|tonight|eod|end of day)\b.{0,80}\bdeadline\b",
                r"\btomorrow\b.{0,80}\bdeadline\b",
                r"\bdue\b.{0,80}\b(?:today|tonight|eod|end of day|tomorrow)\b",
                r"\bwithin\s+(?:24|twenty[-\s]?four)\s*hours?\b",
                r"\bexpires?\b",
                r"\bdo\s*not\s*delete\b",
            ],
        )
        self.important = PatternGroup.compile(
            "important",
            "Contains business, deadline, approval, or follow-up language.",
            [
                r"\bimportant\b",
                r"\bdeadline\b",
                r"\bdue\s*(?:date|soon|monday|tuesday|wednesday|thursday|friday|next week)\b",
                r"\breview\b.{0,80}\b(?:by|before|needed|required)\b",
                r"\b(?:approval|approve|approved)\b",
                r"\bmeeting\b.{0,80}\b(?:required|mandatory|confirm|attendance)\b",
                r"\bconfirm\b.{0,80}\b(?:attendance|availability|receipt|by|before)\b",
                r"\bfollow\s*up\b",
                r"\bpayment\b",
                r"\binvoice\b",
                r"\bcontract\b",
                r"\bproposal\b",
                r"\breport\b",
                r"\bsigned?\b.{0,80}\b(?:contract|agreement|document)\b",
                r"\blegal\b.{0,80}\b(?:needs?|review|approval)\b",
            ],
        )
        self.low = PatternGroup.compile(
            "low",
            "Looks informational, promotional, automated, or no-action.",
            [
                r"\bnewsletter\b",
                r"\bpromo(?:tion|tional)?\b",
                r"\bsale\b",
                r"\bdiscount\b",
                r"\bspecial\s+offer\b",
                r"\bunsubscribe\b",
                r"\bautomated?\b",
                r"\bautomatic\b",
                r"\bno[-\s]?reply\b",
                r"\bnoreply\b",
                r"\bnews\b.{0,80}\bupdate\b",
                r"\bno\s*action\s*(?:needed|required)?\b",
                r"\bfor\s*your\s*information\b",
                r"\bfyi\b",
            ],
        )
        self.priority_order = (self.urgent, self.important, self.low)

    def classify(self, email_text: str, subject: str = "") -> Priority:
        """Return the highest-priority label matched by the email text."""
        return self.get_priority_score(email_text, subject)["priority"]

    def get_priority_score(self, email_text: str, subject: str = "") -> PriorityAnalysis:
        """Return the label plus matched rules and a human-readable reason."""
        combined = self._normalize(email_text, subject)

        for group in self.priority_order:
            matches = group.matches(combined)
            if matches:
                return {
                    "priority": group.priority,
                    "matches": matches,
                    "reason": group.reason,
                }

        return {
            "priority": "normal",
            "matches": [],
            "reason": "No specific priority patterns detected.",
        }

    @staticmethod
    def _normalize(email_text: str, subject: str = "") -> str:
        return " ".join(f"{subject or ''} {email_text or ''}".split()).lower()


def main() -> None:
    """Run a small CLI smoke test."""
    classifier = PriorityClassifier()
    examples = [
        ("URGENT: Deadline today", "Please send the report ASAP. It is due by 5pm today."),
        ("Project Update", "The meeting moved to Friday at 2pm."),
        ("Newsletter", "Check out our latest products and special discounts. Unsubscribe here."),
        ("Payment Reminder", "Your invoice #12345 is due soon."),
        ("FYI", "No action needed. The system will be down for maintenance."),
    ]

    for subject, body in examples:
        result = classifier.get_priority_score(body, subject)
        print(f"Subject: {subject}")
        print(f"  Priority: {result['priority']}")
        print(f"  Reason: {result['reason']}")
        if result["matches"]:
            print(f"  Matches: {', '.join(result['matches'])}")
        print()


if __name__ == "__main__":
    main()
