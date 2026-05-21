"""Gradio app for guppyemail inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr

from inference import GuppyEmailInference
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
    labels = {
        "urgent": "Urgent",
        "important": "Important",
        "normal": "Normal",
        "low": "Low",
    }
    return labels.get(priority, priority.title())


def format_actions(action_items: dict[str, list[str]]) -> str:
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
    if not _model_info:
        get_pipeline()
    if _model_info.get("status") != "Ready":
        return f"{_model_info.get('status', 'Unavailable')}: {_model_info.get('detail', '')}"
    return (
        f"Ready on {_model_info['device']} | "
        f"{_model_info['parameters']} parameters | "
        f"max sequence {_model_info['max_seq_len']} | "
        f"{_model_info['model']}"
    )


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
    return "", "", "", "Normal", "No action items detected.", format_model_info()


def build_app() -> gr.Blocks:
    with gr.Blocks(title="guppyemail") as ui:
        gr.Markdown(
            "# guppyemail\nSmall local email summarizer",
            elem_id="app-title",
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes=["panel"]):
                subject_input = gr.Textbox(
                    label="Subject",
                    placeholder="Budget review moved to Friday",
                    lines=1,
                    max_lines=1,
                )
                email_input = gr.Textbox(
                    label="Email",
                    placeholder="Paste the email body here.",
                    lines=14,
                    max_lines=22,
                )
                with gr.Row():
                    summarize_btn = gr.Button("Summarize", variant="primary")
                    clear_btn = gr.Button("Clear")

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

        gr.Examples(
            examples=[
                [
                    "Budget review",
                    "The Q3 budget meeting has moved to Friday at 2pm. Please bring final numbers and confirm attendance by Thursday.",
                ],
                [
                    "URGENT: Client deadline today",
                    "The client deadline is today at 5pm. We need the final proposal submitted immediately. Contact john@company.com if you have questions.",
                ],
                [
                    "FYI office update",
                    "No action needed. The office maintenance window is scheduled for Saturday morning.",
                ],
            ],
            inputs=[subject_input, email_input],
            label="Examples",
        )

        summarize_btn.click(
            summarize_email,
            inputs=[email_input, subject_input],
            outputs=[summary_output, priority_output, actions_output, model_output],
        )
        email_input.submit(
            summarize_email,
            inputs=[email_input, subject_input],
            outputs=[summary_output, priority_output, actions_output, model_output],
        )
        clear_btn.click(
            clear_outputs,
            outputs=[subject_input, email_input, summary_output, priority_output, actions_output, model_output],
        )

    return ui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the guppyemail Gradio app.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


app = build_app()


def main() -> None:
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
