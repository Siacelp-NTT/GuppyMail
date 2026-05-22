"""Gradio app for guppyemail inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr

from inference import GuppyEmailInference
from src.gmail_client import GmailClient, GmailClientError, GmailMessage
from src.pipeline import EmailProcessingPipeline


BASE_DIR = Path(__file__).resolve().parent

MINIMAL_MODEL = BASE_DIR / "models" / "minimal" / "model.pt"
MINIMAL_TOKENIZER = BASE_DIR / "models" / "minimal" / "tokenizer.json"
MINIMAL_CONFIG = BASE_DIR / "models" / "minimal" / "config.json"
CHECKPOINT_MODEL = BASE_DIR / "checkpoints" / "best_model.pt"
CHECKPOINT_CONFIG = BASE_DIR / "checkpoints" / "config.json"
QUALITY_TOKENIZER = BASE_DIR / "data" / "training_quality" / "tokenizer.json"


_pipeline: EmailProcessingPipeline | None = None
_model_info: dict[str, Any] = {}
_gmail_client: GmailClient | None = None
_gmail_messages: list[GmailMessage] = []

APP_CSS = """
body { background: #f5f7fb; }
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 24px 28px 32px !important;
}
#app-title h1 {
    font-size: 26px !important;
    line-height: 1.2 !important;
    margin-bottom: 2px !important;
    letter-spacing: 0 !important;
}
#app-title p {
    margin-top: 0 !important;
    color: #5d667a !important;
}
.panel {
    border: 1px solid #d9e0ea !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}
.primary button, button.primary {
    border-radius: 8px !important;
    font-weight: 600 !important;
}
textarea, input {
    border-radius: 8px !important;
}
#model-status textarea {
    font-size: 12px !important;
    color: #5d667a !important;
}
"""

PRIORITY_ORDER = ("urgent", "important", "normal", "low")


def available_artifacts() -> tuple[Path, Path, Path] | None:
    """Return the best available model/tokenizer/config set."""
    if MINIMAL_MODEL.exists() and MINIMAL_TOKENIZER.exists() and MINIMAL_CONFIG.exists():
        return MINIMAL_MODEL, MINIMAL_TOKENIZER, MINIMAL_CONFIG
    if CHECKPOINT_MODEL.exists() and QUALITY_TOKENIZER.exists() and CHECKPOINT_CONFIG.exists():
        return CHECKPOINT_MODEL, QUALITY_TOKENIZER, CHECKPOINT_CONFIG
    return None


def get_pipeline() -> EmailProcessingPipeline | None:
    """Lazy-load the model-backed processing pipeline."""
    global _pipeline, _model_info
    if _pipeline is not None:
        return _pipeline

    artifacts = available_artifacts()
    if artifacts is None:
        _model_info = {
            "status": "Demo mode",
            "detail": "Model artifacts were not found.",
        }
        return None

    model_path, tokenizer_path, config_path = artifacts
    engine = GuppyEmailInference(model_path, tokenizer_path, config_path, device="auto")
    _pipeline = EmailProcessingPipeline(summarizer=engine.generate_summary)
    _model_info = {
        "status": "Ready",
        "device": str(engine.device),
        "parameters": f"{engine.model.parameter_count():,}",
        "model": str(model_path.relative_to(BASE_DIR)),
        "tokenizer": str(tokenizer_path.relative_to(BASE_DIR)),
        "max_seq_len": engine.config.max_seq_len,
    }
    return _pipeline


def format_priority(priority: str) -> str:
    """Format priority for display."""
    labels = {
        "urgent": "Urgent",
        "important": "Important",
        "normal": "Normal",
        "low": "Low",
    }
    return labels.get(priority, priority.title())


def format_actions(action_items: dict[str, list[str]]) -> str:
    """Format actions for display."""
    if not action_items:
        return "No action items detected."

    lines: list[str] = []
    labels = {
        "tasks": "Tasks",
        "deadlines": "Deadlines",
        "meetings": "Meetings",
        "requests": "Requests",
    }
    for key in ("tasks", "deadlines", "meetings", "requests"):
        values = action_items.get(key) or []
        if values:
            lines.append(f"{labels[key]}: " + "; ".join(values[:4]))
    return "\n".join(lines) if lines else "No action items detected."


def format_model_info() -> str:
    """Format model info for display."""
    if not _model_info:
        artifacts = available_artifacts()
        if artifacts is None:
            return "Demo mode: model artifacts were not found."
        model_path, tokenizer_path, _ = artifacts
        return (
            "Ready to load on first summary | "
            f"{model_path.relative_to(BASE_DIR)} | "
            f"{tokenizer_path.relative_to(BASE_DIR)}"
        )
    if _model_info.get("status") != "Ready":
        return f"{_model_info.get('status', 'Unavailable')}: {_model_info.get('detail', '')}"
    return (
        f"Ready on {_model_info['device']} | "
        f"{_model_info['parameters']} parameters | "
        f"max sequence {_model_info['max_seq_len']} | "
        f"{_model_info['model']}"
    )


def priority_breakdown_text(priority_counts: dict[str, int]) -> str:
    """Handle priority breakdown text."""
    if not priority_counts:
        return "No summarized emails yet."
    total = sum(priority_counts.values())
    lines = []
    for priority in PRIORITY_ORDER:
        count = priority_counts.get(priority, 0)
        pct = (count / total * 100) if total else 0
        lines.append(f"{format_priority(priority)}: {count} ({pct:.0f}%)")
    return "\n".join(lines)


def action_summary_text(action_counts: dict[str, int]) -> str:
    """Handle action summary text."""
    if not action_counts:
        return "No action items detected yet."
    labels = {
        "tasks": "Tasks",
        "deadlines": "Deadlines",
        "meetings": "Meetings",
        "requests": "Requests",
    }
    return "\n".join(
        f"{labels.get(key, key.title())}: {action_counts.get(key, 0)}"
        for key in ("tasks", "deadlines", "meetings", "requests")
    )


def dashboard_stats_text(stats: dict) -> str:
    """Handle dashboard stats text."""
    return (
        f"Total summarized: {stats['total']}\n"
        f"Priority labels: {stats['priority_breakdown'] or {}}\n"
        f"Action items: {stats['action_items'] or {}}"
    )


def dashboard_recent_text(recent: list[dict]) -> str:
    """Handle dashboard recent text."""
    if not recent:
        return "No recent summaries yet."
    lines: list[str] = []
    for result in reversed(recent):
        subject = result.get("subject") or "(no subject)"
        priority = format_priority(result.get("priority", "normal"))
        summary = result.get("summary", "")
        actions = result.get("action_items", {})
        action_total = sum(len(values) for values in actions.values())
        lines.append(f"{subject}\nPriority: {priority} | Action items: {action_total}\n{summary}")
    return "\n\n".join(lines)


def dashboard_outputs() -> tuple[str, str, str, str]:
    """Handle dashboard outputs."""
    pipeline = _pipeline
    stats = pipeline.get_dashboard_stats() if pipeline else {
        "total": 0,
        "priority_breakdown": {},
        "action_items": {},
        "recent": [],
    }
    return (
        dashboard_stats_text(stats),
        priority_breakdown_text(stats["priority_breakdown"]),
        dashboard_recent_text(stats["recent"]),
        action_summary_text(stats["action_items"]),
    )


def clear_dashboard() -> tuple[str, str, str, str]:
    """Handle clear dashboard."""
    if _pipeline:
        _pipeline.clear_history()
    return dashboard_outputs()


def get_gmail_client() -> GmailClient:
    """Return gmail client."""
    global _gmail_client
    if _gmail_client is None:
        _gmail_client = GmailClient()
    return _gmail_client


def gmail_setup_message() -> str:
    """Handle gmail setup message."""
    status = get_gmail_client().status()
    if not status["has_credentials"]:
        return (
            "Gmail credentials are missing. Save your OAuth desktop client as "
            "`credentials.json` in the project root."
        )
    if not status["has_token"]:
        return (
            "Gmail is not authenticated yet. In WSL/SSH, run this in the project root:\n\n"
            "`python src/gmail_client.py --auth --port 8080`\n\n"
            "If your browser is not on the same machine, tunnel the OAuth callback first:\n\n"
            "`ssh -L 8080:localhost:8080 user@windows-server`"
        )
    return "Gmail token is available. Fetch recent messages when ready."


def gmail_status() -> str:
    """Handle gmail status."""
    status = get_gmail_client().status()
    return (
        f"credentials.json: {'found' if status['has_credentials'] else 'missing'}\n"
        f"token.json: {'found' if status['has_token'] else 'missing'}\n\n"
        f"{gmail_setup_message()}"
    )


def fetch_gmail_messages(max_results: int, query: str):
    """Fetch Gmail messages and return dropdown choices plus preview fields."""
    global _gmail_messages
    try:
        count = max(1, min(int(max_results or 10), 25))
        _gmail_messages = get_gmail_client().fetch_recent(max_results=count, query=(query or "").strip())
    except GmailClientError as exc:
        _gmail_messages = []
        return gr.update(choices=[], value=None), "", f"{gmail_status()}\n\nFetch failed: {exc}"
    except Exception as exc:  # Defensive: Google client errors are verbose and user-facing.
        _gmail_messages = []
        return gr.update(choices=[], value=None), "", f"{gmail_status()}\n\nFetch failed: {type(exc).__name__}: {exc}"

    choices = [
        f"{index + 1}. {message.subject or '(no subject)'} - {message.sender}"
        for index, message in enumerate(_gmail_messages)
    ]
    if not _gmail_messages:
        return gr.update(choices=[], value=None), "", "No Gmail messages matched that query."

    first = _gmail_messages[0]
    return gr.update(choices=choices, value=choices[0]), first.preview(), f"Fetched {len(_gmail_messages)} Gmail messages."


def select_gmail_message(selection: str) -> tuple[str, str, str]:
    """Handle select gmail message."""
    if not selection or not _gmail_messages:
        return "", "", ""
    try:
        index = int(selection.split(".", 1)[0]) - 1
    except ValueError:
        return "", "", ""
    if index < 0 or index >= len(_gmail_messages):
        return "", "", ""
    message = _gmail_messages[index]
    return message.subject, message.body, message.preview()


def summarize_selected_gmail(selection: str) -> tuple[str, str, str, str]:
    """Handle summarize selected gmail."""
    subject, body, _ = select_gmail_message(selection)
    return summarize_email(body, subject)


def summarize_email(email_text: str, subject: str = "") -> tuple[str, str, str, str]:
    """Run model summary plus rule-based priority and action extraction."""
    email_text = (email_text or "").strip()
    subject = (subject or "").strip()

    if not email_text:
        return (
            "Paste an email to generate a summary.",
            "Normal",
            "No action items detected.",
            format_model_info(),
        )

    pipeline = get_pipeline()
    if pipeline is None:
        return (
            "Demo mode: model artifacts were not found.",
            "Normal",
            "No action items detected.",
            format_model_info(),
        )

    result = pipeline.summarize_email(email_text, subject)
    return (
        result["summary"],
        format_priority(result["priority"]),
        format_actions(result["action_items"]),
        format_model_info(),
    )


def clear_outputs() -> tuple[str, str, str, str, str, str]:
    """Handle clear outputs."""
    return "", "", "", "Normal", "No action items detected.", format_model_info()


def build_app() -> gr.Blocks:
    """Build app."""
    with gr.Blocks(title="guppyemail") as ui:
        gr.Markdown(
            "# guppyemail\nSmall local email summarizer",
            elem_id="app-title",
        )

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("## Email")
                subject_input = gr.Textbox(
                    label="Subject",
                    placeholder="Budget review moved to Friday",
                    lines=1,
                    max_lines=1,
                )
                email_input = gr.Textbox(
                    label="Email",
                    placeholder="Paste the email body here.",
                    lines=12,
                    max_lines=18,
                )
                with gr.Row():
                    summarize_btn = gr.Button("Summarize", variant="primary")
                    clear_btn = gr.Button("Clear")

                gr.Markdown("## Gmail")
                gmail_status_box = gr.Textbox(
                    label="Gmail Status",
                    value=gmail_status,
                    lines=5,
                    interactive=False,
                )
                with gr.Row():
                    gmail_query = gr.Textbox(
                        label="Gmail Search",
                        placeholder="newer_than:7d -category:promotions",
                        scale=4,
                    )
                    gmail_limit = gr.Number(
                        label="Max",
                        value=10,
                        minimum=1,
                        maximum=25,
                        step=1,
                        scale=1,
                    )
                fetch_gmail_btn = gr.Button("Fetch Gmail", variant="primary")
                gmail_select = gr.Dropdown(
                    label="Messages",
                    choices=[],
                    value=None,
                    interactive=True,
                )
                gmail_preview = gr.Textbox(label="Preview", lines=6, interactive=False)
                use_gmail_btn = gr.Button("Summarize Selected")

            with gr.Column(scale=4, elem_classes=["panel"]):
                summary_output = gr.Textbox(label="Summary", lines=5, interactive=False)
                priority_output = gr.Textbox(label="Priority", lines=1, interactive=False)
                actions_output = gr.Textbox(label="Action Items", lines=6, interactive=False)
                model_output = gr.Textbox(
                    label="Model",
                    value=format_model_info,
                    lines=2,
                    interactive=False,
                    elem_id="model-status",
                )
                gr.Markdown("## Dashboard")
                stats_output = gr.Textbox(label="Statistics", lines=4, interactive=False)
                priority_display = gr.Textbox(
                    label="Priority Breakdown",
                    lines=5,
                    interactive=False,
                )
                action_display = gr.Textbox(
                    label="Action Item Summary",
                    lines=5,
                    interactive=False,
                )
                recent_display = gr.Textbox(
                    label="Recent Summaries",
                    lines=8,
                    interactive=False,
                )
                with gr.Row():
                    refresh_dashboard_btn = gr.Button("Refresh Dashboard", variant="primary")
                    clear_dashboard_btn = gr.Button("Clear Dashboard")

        summarize_btn.click(
            summarize_email,
            inputs=[email_input, subject_input],
            outputs=[summary_output, priority_output, actions_output, model_output],
        ).then(
            dashboard_outputs,
            outputs=[stats_output, priority_display, recent_display, action_display],
        )
        email_input.submit(
            summarize_email,
            inputs=[email_input, subject_input],
            outputs=[summary_output, priority_output, actions_output, model_output],
        ).then(
            dashboard_outputs,
            outputs=[stats_output, priority_display, recent_display, action_display],
        )
        clear_btn.click(
            clear_outputs,
            outputs=[subject_input, email_input, summary_output, priority_output, actions_output, model_output],
        )
        fetch_gmail_btn.click(
            fetch_gmail_messages,
            inputs=[gmail_limit, gmail_query],
            outputs=[gmail_select, gmail_preview, gmail_status_box],
        )
        gmail_select.change(
            select_gmail_message,
            inputs=[gmail_select],
            outputs=[subject_input, email_input, gmail_preview],
        )
        use_gmail_btn.click(
            summarize_selected_gmail,
            inputs=[gmail_select],
            outputs=[summary_output, priority_output, actions_output, model_output],
        ).then(
            dashboard_outputs,
            outputs=[stats_output, priority_display, recent_display, action_display],
        )
        refresh_dashboard_btn.click(
            dashboard_outputs,
            outputs=[stats_output, priority_display, recent_display, action_display],
        )
        clear_dashboard_btn.click(
            clear_dashboard,
            outputs=[stats_output, priority_display, recent_display, action_display],
        )

    return ui


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the guppyemail Gradio app.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


app = build_app()


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()
    print(f"Starting guppyemail app on http://localhost:{args.port}")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=APP_CSS,
        theme=gr.themes.Base(),
    )


if __name__ == "__main__":
    main()
