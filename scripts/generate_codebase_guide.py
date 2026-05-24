"""Generate a visual HTML guide to the email summarizer codebase."""

from __future__ import annotations

import ast
import html
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return a repo root path that remains writable under WSL path casing."""
    root = Path(__file__).resolve().parent.parent
    if str(root).startswith("/mnt/"):
        lowered = Path(str(root).lower())
        if lowered.exists():
            return lowered
    return root


ROOT = project_root()
OUTPUT = ROOT / "report" / "codebase_visual_guide.html"
PYTHON_PATHS = [
    ROOT / "app.py",
    ROOT / "config.py",
    ROOT / "inference.py",
    ROOT / "model.py",
    *sorted((ROOT / "src").glob("*.py")),
    *sorted((ROOT / "scripts").glob("*.py")),
]


@dataclass
class Symbol:
    """One class or function discovered in a Python module."""

    name: str
    kind: str
    line: int
    doc: str


@dataclass
class ModuleInfo:
    """Static facts collected for one Python source file."""

    path: Path
    title: str
    doc: str
    imports: list[str]
    symbols: list[Symbol]
    lines: int
    size_kb: float


def rel(path: Path) -> str:
    """Return a repository-relative path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


def read_text(path: Path) -> str:
    """Read UTF-8 text with replacement for imperfect files."""
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> Any:
    """Load JSON if present, otherwise return None."""
    if not path.exists():
        return None
    try:
        return json.loads(read_text(path))
    except json.JSONDecodeError:
        return None


def direct_symbols(tree: ast.Module) -> list[Symbol]:
    """Collect top-level and class-level symbols without nested closures."""
    items: list[Symbol] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            items.append(Symbol(node.name, "class", node.lineno, ast.get_docstring(node) or "No docstring yet."))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    items.append(
                        Symbol(
                            f"{node.name}.{child.name}",
                            "method",
                            child.lineno,
                            ast.get_docstring(child) or "No docstring yet.",
                        )
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append(Symbol(node.name, "function", node.lineno, ast.get_docstring(node) or "No docstring yet."))
    return items


def import_names(tree: ast.Module) -> list[str]:
    """Collect imported module names for a dependency overview."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return sorted(names)


def module_info(path: Path) -> ModuleInfo | None:
    """Parse one Python file into module metadata."""
    if not path.exists() or path.name == "__init__.py":
        return None
    text = read_text(path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    return ModuleInfo(
        path=path,
        title=rel(path),
        doc=ast.get_docstring(tree) or "No module docstring yet.",
        imports=import_names(tree),
        symbols=direct_symbols(tree),
        lines=text.count("\n") + 1,
        size_kb=path.stat().st_size / 1024,
    )


def collect_modules() -> list[ModuleInfo]:
    """Collect metadata for all first-party Python modules."""
    modules = [module_info(path) for path in PYTHON_PATHS]
    return [module for module in modules if module is not None]


def collect_tasks() -> list[dict[str, str]]:
    """Read task markdown files and extract title/goal snippets."""
    tasks: list[dict[str, str]] = []
    for path in sorted((ROOT / "tasks").glob("*.md")):
        text = read_text(path)
        lines = text.splitlines()
        title = next((line.lstrip("# ").strip() for line in lines if line.startswith("#")), path.stem)
        goal = ""
        for index, line in enumerate(lines):
            if line.strip().lower() == "## goal":
                goal = next((candidate.strip() for candidate in lines[index + 1 :] if candidate.strip()), "")
                break
        tasks.append({"file": rel(path), "title": title, "goal": goal})
    return tasks


def count_jsonl(path: Path) -> int:
    """Count rows in a JSONL file."""
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def data_inventory() -> list[dict[str, str]]:
    """Return core data and evaluation artifacts for the guide."""
    paths = [
        "data/raw/enron_sample.jsonl",
        "data/processed/cleaned_emails.jsonl",
        "data/summaries/en_summaries.clean.jsonl",
        "data/training_quality/train.jsonl",
        "data/training_quality/val.jsonl",
        "data/training_quality/test.jsonl",
        "evaluation/baseline_results.json",
        "evaluation/eval_results.json",
        "evaluation/human_eval_results.json",
        "evaluation/artifacts/metrics_summary.json",
        "report/final_report.md",
    ]
    rows: list[dict[str, str]] = []
    for raw_path in paths:
        path = ROOT / raw_path
        rows.append(
            {
                "path": raw_path,
                "exists": "yes" if path.exists() else "no",
                "size": f"{path.stat().st_size / 1024:.1f} KB" if path.exists() else "-",
                "rows": f"{count_jsonl(path):,}" if path.suffix == ".jsonl" and path.exists() else "-",
            }
        )
    return rows


def architecture_steps() -> list[dict[str, str]]:
    """Describe the end-to-end system flow in plain language."""
    return [
        {"name": "Download", "file": "scripts/download_enron.py", "detail": "Fetches Enron emails and writes raw JSONL/parquet files."},
        {"name": "Clean", "file": "src/preprocess.py", "detail": "Removes HTML, signatures, quoted threads, disclaimers, and duplicates."},
        {"name": "Summarize", "file": "scripts/generate_summaries.py", "detail": "Calls an LLM API to create supervised reference summaries."},
        {"name": "Filter and Tokenize", "file": "scripts/build_quality_training_data.py, scripts/train_tokenizer.py", "detail": "Keeps quality rows, splits train/val/test, and trains the BPE tokenizer."},
        {"name": "Train", "file": "notebooks/train_email_guppylm.ipynb", "detail": "Trains a small decoder-only transformer from scratch in Colab."},
        {"name": "Evaluate", "file": "scripts/evaluate.py", "detail": "Computes perplexity, ROUGE, and example generations."},
        {"name": "Serve", "file": "app.py, inference.py, src/pipeline.py", "detail": "Runs Gradio UI, model inference, priority classification, and action extraction."},
        {"name": "Report", "file": "scripts/generate_artifacts.py, scripts/plot_metrics.py", "detail": "Builds Phase 5 charts, summaries, and final report inputs."},
    ]


def relationship_rows(modules: list[ModuleInfo]) -> list[tuple[str, str]]:
    """Build first-party dependency rows from imports."""
    first_party = {"src", "scripts", "app", "model", "config", "inference"}
    rows: list[tuple[str, str]] = []
    for module in modules:
        for imported in module.imports:
            root = imported.split(".")[0]
            if root in first_party:
                rows.append((module.title, imported))
    return rows


def esc(value: Any) -> str:
    """HTML-escape display values."""
    return html.escape(str(value), quote=True)


def badge(text: str, class_name: str = "") -> str:
    """Return a styled badge span."""
    return f'<span class="badge {class_name}">{esc(text)}</span>'


def render_html(modules: list[ModuleInfo], tasks: list[dict[str, str]]) -> str:
    """Render the complete static HTML guide."""
    eval_data = load_json(ROOT / "evaluation" / "eval_results.json") or {}
    metrics = load_json(ROOT / "evaluation" / "artifacts" / "metrics_summary.json") or {}
    counts = metrics.get("dataset", {}) if isinstance(metrics, dict) else {}
    symbols = [symbol for module in modules for symbol in module.symbols]
    relations = relationship_rows(modules)
    grouped: dict[str, list[ModuleInfo]] = defaultdict(list)
    for module in modules:
        group = module.title.split("/", 1)[0] if "/" in module.title else "root"
        grouped[group].append(module)

    cards = []
    for module in modules:
        symbol_preview = ", ".join(symbol.name for symbol in module.symbols[:6])
        cards.append(
            f"""
            <article class="module-card" data-search="{esc((module.title + ' ' + module.doc + ' ' + symbol_preview).lower())}">
              <div class="module-head">
                <h3>{esc(module.title)}</h3>
                <span>{module.lines} lines</span>
              </div>
              <p>{esc(module.doc)}</p>
              <div class="chips">
                {badge(f"{len(module.symbols)} symbols")}
                {badge(f"{len(module.imports)} imports")}
                {badge(f"{module.size_kb:.1f} KB")}
              </div>
              <details>
                <summary>Functions and classes</summary>
                <ul>
                  {''.join(f'<li><strong>{esc(s.name)}</strong> <em>{esc(s.kind)}</em> - line {s.line}<br><span>{esc(s.doc)}</span></li>' for s in module.symbols)}
                </ul>
              </details>
            </article>
            """
        )

    task_cards = []
    for task in tasks:
        task_cards.append(
            f"""
            <article class="task-card" data-search="{esc((task['file'] + ' ' + task['title'] + ' ' + task['goal']).lower())}">
              <div class="module-head"><h3>{esc(task['title'])}</h3><span>{esc(task['file'])}</span></div>
              <p>{esc(task['goal'] or 'No goal section found.')}</p>
            </article>
            """
        )

    symbol_rows = []
    for module in modules:
        for symbol in module.symbols:
            symbol_rows.append(
                f"<tr data-search=\"{esc((module.title + ' ' + symbol.name + ' ' + symbol.doc).lower())}\"><td>{esc(symbol.name)}</td><td>{esc(symbol.kind)}</td><td>{esc(module.title)}:{symbol.line}</td><td>{esc(symbol.doc)}</td></tr>"
            )

    relation_items = "".join(f"<li><strong>{esc(source)}</strong> imports <code>{esc(target)}</code></li>" for source, target in relations[:160])
    inventory_rows = "".join(
        f"<tr><td><code>{esc(row['path'])}</code></td><td>{esc(row['exists'])}</td><td>{esc(row['size'])}</td><td>{esc(row['rows'])}</td></tr>"
        for row in data_inventory()
    )
    flow = "".join(
        f"""
        <div class="flow-step">
          <div class="flow-title">{index}. {esc(step['name'])}</div>
          <code>{esc(step['file'])}</code>
          <p>{esc(step['detail'])}</p>
        </div>
        """
        for index, step in enumerate(architecture_steps(), start=1)
    )
    groups = "".join(
        f"<li><strong>{esc(name)}</strong>: {', '.join(esc(module.title) for module in mods)}</li>"
        for name, mods in sorted(grouped.items())
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GuppyEmail Codebase Visual Guide</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #18212f;
      --muted: #64748b;
      --line: #d8dee8;
      --blue: #2557a7;
      --green: #1f7a4d;
      --red: #a72f2f;
      --yellow: #9a6a05;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
    header {{ background: #172033; color: white; padding: 28px 32px; }}
    header h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: 0; }}
    header p {{ margin: 0; color: #cbd5e1; max-width: 920px; line-height: 1.55; }}
    nav {{ position: sticky; top: 0; z-index: 5; display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 24px; background: rgba(246,247,249,0.96); border-bottom: 1px solid var(--line); backdrop-filter: blur(8px); }}
    nav a {{ text-decoration: none; color: var(--ink); border: 1px solid var(--line); padding: 8px 10px; border-radius: 6px; background: white; font-size: 14px; }}
    main {{ padding: 24px; max-width: 1420px; margin: 0 auto; }}
    section {{ margin: 0 0 28px; }}
    h2 {{ font-size: 22px; margin: 0 0 14px; }}
    h3 {{ margin: 0; font-size: 16px; }}
    p {{ line-height: 1.55; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid var(--line); }}
    th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 10px; vertical-align: top; }}
    th {{ background: #eef2f7; font-size: 13px; text-transform: uppercase; color: #475569; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .stat {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    .stat strong {{ display: block; font-size: 26px; margin-bottom: 4px; }}
    .stat span {{ color: var(--muted); }}
    .flow {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .flow-step, .module-card, .task-card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .flow-title {{ font-weight: 700; margin-bottom: 8px; color: var(--blue); }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .module-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; margin-bottom: 8px; }}
    .module-head span {{ color: var(--muted); font-size: 12px; white-space: nowrap; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }}
    .badge {{ display: inline-block; border: 1px solid var(--line); color: #334155; padding: 3px 7px; border-radius: 999px; font-size: 12px; background: #f8fafc; }}
    details {{ margin-top: 12px; }}
    summary {{ cursor: pointer; color: var(--blue); font-weight: 600; }}
    li {{ margin: 8px 0; }}
    .searchbar {{ display: flex; gap: 10px; margin-bottom: 14px; }}
    .searchbar input {{ width: 100%; padding: 10px 12px; border: 1px solid var(--line); border-radius: 6px; font-size: 15px; }}
    .two-col {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 12px; }}
    .callout {{ border-left: 4px solid var(--blue); background: white; padding: 12px 14px; border-radius: 0 8px 8px 0; }}
    .legend {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    footer {{ padding: 24px 32px; color: var(--muted); border-top: 1px solid var(--line); }}
    @media (max-width: 1000px) {{ .stats, .flow, .grid, .two-col, .legend {{ grid-template-columns: 1fr; }} main {{ padding: 16px; }} header {{ padding: 22px 18px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>GuppyEmail Codebase Visual Guide</h1>
    <p>A visual, no-code-first map of the project: what each file does, how the data moves, how modules depend on each other, where tasks fit, and what metrics prove the current model state.</p>
  </header>
  <nav>
    <a href="#overview">Overview</a>
    <a href="#flow">Pipeline Flow</a>
    <a href="#modules">Modules</a>
    <a href="#symbols">Function Catalog</a>
    <a href="#relationships">Relationships</a>
    <a href="#tasks">Tasks</a>
    <a href="#data">Data and Metrics</a>
    <a href="#reading">How to Read This Repo</a>
  </nav>
  <main>
    <section id="overview">
      <div class="stats">
        <div class="stat"><strong>{len(modules)}</strong><span>Python modules mapped</span></div>
        <div class="stat"><strong>{len(symbols)}</strong><span>Functions/classes/methods cataloged</span></div>
        <div class="stat"><strong>{len(tasks)}</strong><span>Task documents indexed</span></div>
        <div class="stat"><strong>{fmt_metric(eval_data)}</strong><span>Trained model perplexity</span></div>
      </div>
    </section>

    <section id="flow">
      <h2>Pipeline Flow</h2>
      <div class="flow">{flow}</div>
    </section>

    <section id="modules">
      <h2>Module Map</h2>
      <div class="searchbar"><input id="moduleSearch" placeholder="Search files, docstrings, or function names"></div>
      <div class="grid" id="moduleGrid">{''.join(cards)}</div>
    </section>

    <section id="symbols">
      <h2>Function and Class Catalog</h2>
      <div class="searchbar"><input id="symbolSearch" placeholder="Search every function, method, class, file, or docstring"></div>
      <table>
        <thead><tr><th>Name</th><th>Kind</th><th>Location</th><th>Purpose</th></tr></thead>
        <tbody id="symbolRows">{''.join(symbol_rows)}</tbody>
      </table>
    </section>

    <section id="relationships">
      <h2>Code Relationships</h2>
      <div class="two-col">
        <div class="panel">
          <h3>First-party imports</h3>
          <ul>{relation_items or '<li>No first-party imports detected.</li>'}</ul>
        </div>
        <div class="panel">
          <h3>Directory responsibilities</h3>
          <ul>{groups}</ul>
        </div>
      </div>
    </section>

    <section id="tasks">
      <h2>Task Roadmap</h2>
      <div class="searchbar"><input id="taskSearch" placeholder="Search task title, goal, or file"></div>
      <div class="grid" id="taskGrid">{''.join(task_cards)}</div>
    </section>

    <section id="data">
      <h2>Data, Models, and Metrics</h2>
      <div class="legend">
        <div class="callout">
          <strong>Headline model result</strong>
          <p>Evaluation JSON reports perplexity {esc(metric_text(eval_data, 'perplexity'))}, ROUGE-1 {esc(metric_text(eval_data, 'rouge1'))}, ROUGE-2 {esc(metric_text(eval_data, 'rouge2'))}, and ROUGE-L {esc(metric_text(eval_data, 'rougeL'))}.</p>
        </div>
        <div class="callout">
          <strong>Dataset scale</strong>
          <p>Current metrics summary reports {esc(counts.get('raw', 'N/A'))} raw rows, {esc(counts.get('cleaned', 'N/A'))} cleaned rows, and {esc(counts.get('train', 'N/A'))} train rows.</p>
        </div>
      </div>
      <table>
        <thead><tr><th>Artifact</th><th>Exists</th><th>Size</th><th>Rows</th></tr></thead>
        <tbody>{inventory_rows}</tbody>
      </table>
    </section>

    <section id="reading">
      <h2>How to Understand the Repo Fast</h2>
      <div class="grid">
        <article class="panel"><h3>Start with the product</h3><p>Open <code>app.py</code> to see the user-facing Gradio app and the workflows it exposes.</p></article>
        <article class="panel"><h3>Then inspect the pipeline</h3><p><code>src/pipeline.py</code> connects cleaning, model summary generation, priority classification, and action extraction.</p></article>
        <article class="panel"><h3>Then inspect the model</h3><p><code>src/guppyemail_model.py</code>, <code>src/guppyemail_data.py</code>, and <code>inference.py</code> explain the transformer, training data shape, and runtime generation.</p></article>
        <article class="panel"><h3>Use reports for proof</h3><p><code>evaluation/artifacts/</code> and <code>report/final_report.md</code> summarize data counts, model metrics, limitations, and next steps.</p></article>
        <article class="panel"><h3>Use tasks for project history</h3><p>The <code>tasks/</code> files show the intended progression from raw data through UI, Gmail integration, evaluation, and submission artifacts.</p></article>
        <article class="panel"><h3>Use search</h3><p>Use the search boxes on this page to jump from a concept like "priority", "tokenizer", "Gmail", "ROUGE", or "summary" to the files and functions that implement it.</p></article>
      </div>
    </section>
  </main>
  <footer>Generated {datetime.now().isoformat(timespec='seconds')} from the current workspace.</footer>
  <script>
    function filterCards(inputId, containerId, selector) {{
      const input = document.getElementById(inputId);
      const container = document.getElementById(containerId);
      input.addEventListener('input', () => {{
        const query = input.value.toLowerCase().trim();
        container.querySelectorAll(selector).forEach((item) => {{
          item.style.display = item.dataset.search.includes(query) ? '' : 'none';
        }});
      }});
    }}
    filterCards('moduleSearch', 'moduleGrid', '.module-card');
    filterCards('taskSearch', 'taskGrid', '.task-card');
    document.getElementById('symbolSearch').addEventListener('input', (event) => {{
      const query = event.target.value.toLowerCase().trim();
      document.querySelectorAll('#symbolRows tr').forEach((row) => {{
        row.style.display = row.dataset.search.includes(query) ? '' : 'none';
      }});
    }});
  </script>
</body>
</html>
"""


def metric_text(eval_data: dict[str, Any], name: str) -> str:
    """Format a metric from evaluation JSON."""
    value = None
    if name == "perplexity":
        value = eval_data.get("perplexity") or eval_data.get("evaluation", {}).get("test_perplexity")
    elif eval_data.get("rouge"):
        value = eval_data["rouge"].get(name)
    return "N/A" if value is None else f"{float(value):.3f}"


def fmt_metric(eval_data: dict[str, Any]) -> str:
    """Return the perplexity display for the overview card."""
    return metric_text(eval_data, "perplexity")


def main() -> None:
    """Generate the static codebase guide HTML."""
    modules = collect_modules()
    tasks = collect_tasks()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_html(modules, tasks), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
