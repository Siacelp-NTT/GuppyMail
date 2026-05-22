"""Gmail API client for fetching read-only email data."""

from __future__ import annotations

import argparse
import base64
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CREDENTIALS_PATH = BASE_DIR / "credentials.json"
DEFAULT_TOKEN_PATH = BASE_DIR / "token.json"


class GmailClientError(RuntimeError):
    """Raised when Gmail cannot be configured or reached."""


@dataclass(frozen=True)
class GmailMessage:
    """Plain email shape used by the app."""

    id: str
    thread_id: str
    subject: str
    sender: str
    date: str
    snippet: str
    body: str

    def preview(self, max_body_chars: int = 180) -> str:
        """Handle preview."""
        body = " ".join(self.body.split())
        if len(body) > max_body_chars:
            body = body[:max_body_chars].rstrip() + "..."
        return f"{self.subject or '(no subject)'}\nFrom: {self.sender}\n{body}"


def decode_body_data(data: str) -> str:
    """Decode Gmail's URL-safe base64 body payload."""
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", errors="replace")


def html_to_text(value: str) -> str:
    """Convert simple HTML email content to readable text."""
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"(?i)</p\s*>", "\n", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return clean_text(value)


def clean_text(value: str) -> str:
    """Handle clean text."""
    value = value.replace("\r", "\n").replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def display_path(path: Path) -> str:
    """Handle display path."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def header_value(headers: list[dict[str, str]], name: str) -> str:
    """Handle header value."""
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def iter_parts(payload: dict[str, Any]):
    """Yield MIME payload nodes depth-first, including nested multipart parts."""
    yield payload
    for part in payload.get("parts", []) or []:
        yield from iter_parts(part)


def extract_body(payload: dict[str, Any]) -> str:
    """Extract the best available text body from a Gmail message payload."""
    plain_candidates: list[str] = []
    html_candidates: list[str] = []

    for part in iter_parts(payload):
        body_data = part.get("body", {}).get("data", "")
        if not body_data:
            continue
        mime_type = part.get("mimeType", "")
        decoded = decode_body_data(body_data)
        if mime_type == "text/plain":
            plain_candidates.append(clean_text(decoded))
        elif mime_type == "text/html":
            html_candidates.append(html_to_text(decoded))
        elif not mime_type.startswith("multipart/"):
            plain_candidates.append(clean_text(decoded))

    for candidate in plain_candidates + html_candidates:
        if candidate:
            return candidate
    return ""


class GmailClient:
    """Small Gmail API wrapper for OAuth and message retrieval."""

    def __init__(
        self,
        credentials_path: str | Path = DEFAULT_CREDENTIALS_PATH,
        token_path: str | Path = DEFAULT_TOKEN_PATH,
    ) -> None:
        """Initialize the instance."""
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.service = None
        self.creds: Credentials | None = None

    def has_credentials_file(self) -> bool:
        """Return whether credentials file is available."""
        return self.credentials_path.exists()

    def has_token_file(self) -> bool:
        """Return whether token file is available."""
        return self.token_path.exists()

    def authenticate(
        self,
        run_oauth: bool = True,
        local_server_port: int = 8080,
        open_browser: bool = False,
    ) -> Credentials:
        """Load, refresh, or create Gmail OAuth credentials."""
        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        if not creds or not creds.valid:
            if not run_oauth:
                raise GmailClientError(
                    "Gmail is not authenticated. Run `python src/gmail_client.py --auth` first."
                )
            if not self.credentials_path.exists():
                raise GmailClientError(
                    f"Missing OAuth credentials file: {self.credentials_path.relative_to(BASE_DIR)}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
            creds = flow.run_local_server(
                host="localhost",
                port=local_server_port,
                open_browser=open_browser,
                prompt="consent",
            )
            self.token_path.write_text(creds.to_json(), encoding="utf-8")

        self.creds = creds
        return creds

    def connect(self, run_oauth: bool = False):
        """Build and cache the Gmail service."""
        if self.service is None:
            creds = self.authenticate(run_oauth=run_oauth)
            self.service = build("gmail", "v1", credentials=creds)
        return self.service

    def status(self) -> dict[str, Any]:
        """Return local Gmail setup state without network access."""
        return {
            "credentials_path": display_path(self.credentials_path),
            "token_path": display_path(self.token_path),
            "has_credentials": self.has_credentials_file(),
            "has_token": self.has_token_file(),
        }

    def list_message_ids(self, max_results: int = 10, query: str = "") -> list[dict[str, str]]:
        """Handle list message ids."""
        service = self.connect(run_oauth=False)
        try:
            response = (
                service.users()
                .messages()
                .list(userId="me", maxResults=max_results, q=query or None)
                .execute()
            )
        except HttpError as exc:
            raise GmailClientError(f"Gmail list failed: {exc}") from exc
        return response.get("messages", []) or []

    def get_message(self, message_id: str, body_max_chars: int = 6000) -> GmailMessage:
        """Return message."""
        service = self.connect(run_oauth=False)
        try:
            message = (
                service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
        except HttpError as exc:
            raise GmailClientError(f"Gmail message fetch failed: {exc}") from exc

        payload = message.get("payload", {})
        headers = payload.get("headers", []) or []
        body = extract_body(payload)
        if body_max_chars and len(body) > body_max_chars:
            body = body[:body_max_chars].rstrip() + "..."

        return GmailMessage(
            id=message_id,
            thread_id=message.get("threadId", ""),
            subject=header_value(headers, "Subject"),
            sender=header_value(headers, "From"),
            date=header_value(headers, "Date"),
            snippet=message.get("snippet", ""),
            body=body,
        )

    def fetch_recent(
        self,
        max_results: int = 10,
        query: str = "",
        body_max_chars: int = 6000,
    ) -> list[GmailMessage]:
        """Handle fetch recent."""
        messages = self.list_message_ids(max_results=max_results, query=query)
        return [
            self.get_message(message["id"], body_max_chars=body_max_chars)
            for message in messages
            if message.get("id")
        ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Authenticate and fetch Gmail messages.")
    parser.add_argument("--auth", action="store_true", help="Run OAuth and save token.json.")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--query", default="")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()
    client = GmailClient()

    if args.auth:
        client.authenticate(run_oauth=True, local_server_port=args.port, open_browser=args.open_browser)
        print(f"Authenticated. Token saved to {DEFAULT_TOKEN_PATH.relative_to(BASE_DIR)}")

    emails = client.fetch_recent(max_results=args.max_results, query=args.query)
    print(f"Fetched {len(emails)} emails")
    for index, email in enumerate(emails, 1):
        print(f"\n--- Email {index} ---")
        print(f"From: {email.sender}")
        print(f"Subject: {email.subject}")
        print(f"Date: {email.date}")
        print(email.body[:500])


if __name__ == "__main__":
    main()
