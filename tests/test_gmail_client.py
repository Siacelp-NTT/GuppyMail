from __future__ import annotations

import base64

from src.gmail_client import GmailClient, decode_body_data, extract_body, header_value, html_to_text


def encoded(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("utf-8").rstrip("=")


def test_decode_body_data_handles_unpadded_gmail_base64() -> None:
    assert decode_body_data(encoded("hello gmail")) == "hello gmail"


def test_html_to_text_strips_tags_and_entities() -> None:
    assert html_to_text("<p>Hello&nbsp;<b>team</b></p><script>x</script>") == "Hello team"


def test_extract_body_prefers_nested_plain_text() -> None:
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "parts": [
                    {"mimeType": "text/html", "body": {"data": encoded("<p>HTML body</p>")}},
                    {"mimeType": "text/plain", "body": {"data": encoded("Plain body")}},
                ],
            }
        ],
    }
    assert extract_body(payload) == "Plain body"


def test_extract_body_falls_back_to_html() -> None:
    payload = {"mimeType": "text/html", "body": {"data": encoded("<p>Meeting at 2pm</p>")}}
    assert extract_body(payload) == "Meeting at 2pm"


def test_header_value_is_case_insensitive() -> None:
    headers = [{"name": "subject", "value": "Budget"}, {"name": "From", "value": "a@example.com"}]
    assert header_value(headers, "Subject") == "Budget"
    assert header_value(headers, "from") == "a@example.com"


def test_status_does_not_require_network(tmp_path) -> None:
    client = GmailClient(credentials_path=tmp_path / "credentials.json", token_path=tmp_path / "token.json")
    assert client.status()["has_credentials"] is False
    assert client.status()["has_token"] is False
