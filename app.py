from __future__ import annotations

import cgi
import html
import io
import json
import re
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from e2r2_pipeline import (
    build_case_base,
    build_e2r2_prompt,
    case_vectors_dataset,
    dataframe_to_csv_bytes,
    holdout_dataset,
    holdout_vectors_dataset,
    local_e2r2_reasoning,
    parse_data_dictionary,
    read_uploaded_table,
    retrieve_similar_cases,
    summarize_peer_alignment,
    train_e2r2_baseline,
)


HOST = "127.0.0.1"
DEFAULT_PORT = 8765
RATIONALE_SIMPLIFIER_MODEL = "gpt-5-nano"

OPENAI_LLM_MODELS = [
    ("gpt-5.4-mini", "GPT-5.4 mini - low cost, strong general reasoning"),
    ("gpt-5-mini", "GPT-5 mini - low cost, strong reasoning"),
    ("gpt-5-nano", "GPT-5 nano - lowest cost, light reasoning"),
    ("o3-mini", "o3-mini - moderate cost, dedicated reasoning"),
    ("o4-mini", "o4-mini - moderate cost, fast o-series reasoning"),
    ("gpt-4.1-mini", "GPT-4.1 mini - low cost, fastest baseline"),
]

STATE: Dict[str, Any] = {
    "dataset": None,
    "dictionary_df": None,
    "dictionary": {},
    "prediction_goal": "",
    "training": None,
    "case_base": None,
    "shap_table": None,
    "prediction": None,
    "error": None,
}

PROGRESS_LOCK = threading.Lock()
PROGRESS: Dict[str, Any] = {
    "active": False,
    "stage": "Idle",
    "percent": 0,
}


def set_progress(stage: str, percent: float, active: bool = True) -> None:
    with PROGRESS_LOCK:
        PROGRESS.update(
            {
                "active": active,
                "stage": stage,
                "percent": max(0, min(100, round(float(percent), 1))),
            }
        )


def progress_snapshot() -> Dict[str, Any]:
    with PROGRESS_LOCK:
        return dict(PROGRESS)


def load_table_from_upload(field: Any) -> Optional[pd.DataFrame]:
    if field is None or not getattr(field, "filename", ""):
        return None
    data = field.file.read()
    file_obj = io.BytesIO(data)
    file_obj.name = field.filename
    return read_uploaded_table(file_obj)


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def metric_card(label: str, value: Any) -> str:
    if isinstance(value, float):
        value_text = f"{value:.3f}"
    else:
        value_text = str(value)
    return f"<div class='metric'><span>{esc(label)}</span><strong>{esc(value_text)}</strong></div>"


def table_html(df: Optional[pd.DataFrame], max_rows: int = 25, drop_vectors: bool = True) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No rows yet.</p>"
    display = df.copy()
    if drop_vectors:
        display = display[[c for c in display.columns if not str(c).startswith("vector_")]]
    display = display.head(max_rows)
    return display.to_html(index=False, classes="data-table", border=0, escape=True)


def option_tags(values, selected=None) -> str:
    tags = []
    for value in values:
        is_selected = " selected" if str(value) == str(selected) else ""
        tags.append(f"<option value='{esc(value)}'{is_selected}>{esc(value)}</option>")
    return "\n".join(tags)


def model_option_tags(selected: str = "gpt-4.1-mini") -> str:
    tags = []
    for value, label in OPENAI_LLM_MODELS:
        is_selected = " selected" if value == selected else ""
        tags.append(f"<option value='{esc(value)}'{is_selected}>{esc(label)} ({esc(value)})</option>")
    return "\n".join(tags)


def css() -> str:
    return """
    <style>
      :root {
        --ink: #1f2933;
        --muted: #607080;
        --line: #d9e2ec;
        --panel: #f7f9fb;
        --accent: #0f766e;
        --accent-dark: #115e59;
        --warn: #9a3412;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Segoe UI", Arial, sans-serif;
        color: var(--ink);
        background: #ffffff;
      }
      header {
        padding: 28px 40px 18px;
        border-bottom: 1px solid var(--line);
        background: #fbfcfd;
      }
      h1 { margin: 0 0 8px; font-size: 28px; font-weight: 650; letter-spacing: 0; }
      h2 { margin: 0 0 16px; font-size: 19px; font-weight: 650; letter-spacing: 0; }
      h3 { margin: 18px 0 10px; font-size: 16px; letter-spacing: 0; }
      main { padding: 24px 40px 48px; max-width: 1280px; }
      section {
        border-bottom: 1px solid var(--line);
        padding: 22px 0;
      }
      label { display: block; font-size: 13px; font-weight: 600; color: #344454; margin: 0 0 6px; }
      input, select, textarea {
        width: 100%;
        border: 1px solid #b7c4d1;
        border-radius: 6px;
        padding: 9px 10px;
        font: inherit;
        background: #fff;
      }
      input[type=file] { padding: 7px; }
      textarea { min-height: 130px; resize: vertical; }
      button, .button {
        border: 0;
        border-radius: 6px;
        padding: 10px 14px;
        background: var(--accent);
        color: #fff;
        font-weight: 650;
        text-decoration: none;
        cursor: pointer;
        display: inline-block;
      }
      button:hover, .button:hover { background: var(--accent-dark); }
      .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; align-items: end; }
      .span-2 { grid-column: span 2; }
      .span-3 { grid-column: span 3; }
      .span-4 { grid-column: span 4; }
      .span-6 { grid-column: span 6; }
      .span-8 { grid-column: span 8; }
      .span-12 { grid-column: span 12; }
      .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; }
      .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; }
      .metric span { display: block; color: var(--muted); font-size: 12px; }
      .metric strong { display: block; margin-top: 5px; font-size: 20px; }
      .muted { color: var(--muted); }
      .error { border-left: 4px solid #b91c1c; padding: 12px 14px; background: #fff5f5; color: #7f1d1d; }
      .warn { border-left: 4px solid var(--warn); padding: 12px 14px; background: #fff7ed; color: #7c2d12; }
      .data-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 6px; }
      .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
      .data-table th, .data-table td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
      .data-table th { background: var(--panel); position: sticky; top: 0; z-index: 1; }
      .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
      .progress-overlay {
        position: fixed;
        inset: 0;
        display: none;
        align-items: center;
        justify-content: center;
        padding: 24px;
        background: rgba(255, 255, 255, 0.86);
        z-index: 20;
      }
      .progress-overlay.visible { display: flex; }
      .progress-panel {
        width: min(560px, 100%);
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fff;
        padding: 22px;
        box-shadow: 0 18px 60px rgba(31, 41, 51, 0.18);
      }
      .progress-bar {
        height: 12px;
        border-radius: 999px;
        background: #e6edf3;
        overflow: hidden;
        margin: 14px 0 8px;
      }
      .progress-fill {
        height: 100%;
        width: 0%;
        background: var(--accent);
        transition: width 240ms ease;
      }
      .stage-list {
        margin: 14px 0 0;
        padding: 0;
        list-style: none;
        color: var(--muted);
        font-size: 13px;
      }
      .stage-list li { padding: 4px 0; }
      .stage-list li.current { color: var(--ink); font-weight: 650; }
      pre {
        white-space: pre-wrap;
        background: #f6f8fa;
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 14px;
        max-height: 420px;
        overflow: auto;
      }
      @media (max-width: 820px) {
        header, main { padding-left: 18px; padding-right: 18px; }
        .grid { grid-template-columns: 1fr; }
        .span-2, .span-3, .span-4, .span-6, .span-8, .span-12 { grid-column: span 1; }
      }
    </style>
    """


def page(title: str, body: str) -> bytes:
    status = ""
    if STATE.get("error"):
        status = f"<div class='error'><strong>Something needs attention.</strong><br>{esc(STATE['error'])}</div>"
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  {css()}
</head>
<body>
  <header>
    <h1>E2R2 Data Analysis Pipeline</h1>
    <div class="muted">Explain, Embed, Retrieve, and Reason for tabular binary prediction tasks.</div>
  </header>
  <main>
    {status}
    {body}
  </main>
  {progress_overlay()}
</body>
</html>"""
    return html_doc.encode("utf-8")


def progress_overlay() -> str:
    return """
  <div id="progress-overlay" class="progress-overlay" aria-live="polite" aria-hidden="true">
    <div class="progress-panel">
      <h2>E2R2 is running</h2>
      <div id="progress-stage" class="muted">Starting the pipeline...</div>
      <div class="progress-bar"><div id="progress-fill" class="progress-fill"></div></div>
      <div id="progress-percent" class="muted">0%</div>
      <ul class="stage-list">
        <li data-stage="Preparing">Preparing data</li>
        <li data-stage="Training">Training baseline models</li>
        <li data-stage="Selecting">Selecting correct high-confidence cases</li>
        <li data-stage="SHAP">Running SHAP</li>
        <li data-stage="Aggregating">Aggregating SHAP predictors</li>
        <li data-stage="Computing">Computing top predictors</li>
        <li data-stage="Embedding">Embedding case base</li>
        <li data-stage="Complete">Finishing up</li>
      </ul>
    </div>
  </div>
  <script>
    const trainForm = document.getElementById("train-form");
    const overlay = document.getElementById("progress-overlay");
    const stageText = document.getElementById("progress-stage");
    const progressFill = document.getElementById("progress-fill");
    const progressPercent = document.getElementById("progress-percent");
    const stageItems = Array.from(document.querySelectorAll(".stage-list li"));
    let progressTimer = null;

    function showProgress() {
      overlay.classList.add("visible");
      overlay.setAttribute("aria-hidden", "false");
      updateProgress({ stage: "Starting the pipeline...", percent: 2 });
    }

    function updateProgress(data) {
      const percent = Math.max(0, Math.min(100, Number(data.percent || 0)));
      const stage = data.stage || "Working...";
      stageText.textContent = stage;
      progressFill.style.width = `${percent}%`;
      progressPercent.textContent = `${Math.round(percent)}%`;
      stageItems.forEach((item) => {
        const token = item.dataset.stage;
        item.classList.toggle("current", stage.includes(token));
      });
    }

    async function pollProgress() {
      try {
        const response = await fetch("/progress", { cache: "no-store" });
        if (response.ok) {
          updateProgress(await response.json());
        }
      } catch (error) {
      }
    }

    if (trainForm) {
      trainForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        showProgress();
        progressTimer = setInterval(pollProgress, 700);
        try {
          const response = await fetch(trainForm.action, {
            method: "POST",
            body: new FormData(trainForm),
          });
          await pollProgress();
          window.location.href = "/";
        } catch (error) {
          stageText.textContent = "The run stopped before finishing. Refreshing the app...";
          window.location.href = "/";
        } finally {
          if (progressTimer) {
            clearInterval(progressTimer);
          }
        }
      });
    }
  </script>
    """


def upload_section() -> str:
    return """
    <section>
      <h2>Source Files</h2>
      <form method="post" action="/upload" enctype="multipart/form-data">
        <div class="grid">
          <div class="span-6">
            <label for="dataset">Raw dataset</label>
            <input id="dataset" name="dataset" type="file" accept=".csv,.xlsx,.xls" required>
          </div>
          <div class="span-6">
            <label for="dictionary">Data dictionary</label>
            <input id="dictionary" name="dictionary" type="file" accept=".csv,.xlsx,.xls">
          </div>
          <div class="span-12">
            <label for="prediction_goal">Prediction goal</label>
            <textarea id="prediction_goal" name="prediction_goal" placeholder="Example: Predict whether a freshman student will not return for the second fall semester."></textarea>
          </div>
          <div class="span-12">
            <button type="submit">Load files</button>
          </div>
        </div>
      </form>
    </section>
    """


def configuration_section() -> str:
    df = STATE.get("dataset")
    if df is None:
        return upload_section()
    columns = list(df.columns)
    target = STATE.get("target_column") or columns[-1]
    values = sorted(df[target].dropna().astype(str).unique().tolist()) if target in df else []
    positive = STATE.get("positive_label") or (values[-1] if values else "")
    ignored = STATE.get("ignored_columns") or []

    ignored_boxes = []
    for column in columns:
        checked = " checked" if column in ignored else ""
        ignored_boxes.append(
            f"<label><input style='width:auto' type='checkbox' name='ignored_columns' value='{esc(column)}'{checked}> {esc(column)}</label>"
        )

    return f"""
    <section>
      <h2>Prediction Setup</h2>
      <form id="train-form" method="post" action="/train">
        <div class="grid">
          <div class="span-4">
            <label for="target_column">Outcome column</label>
            <select id="target_column" name="target_column">{option_tags(columns, target)}</select>
          </div>
          <div class="span-4">
            <label for="positive_label">Positive outcome</label>
            <input id="positive_label" name="positive_label" value="{esc(positive)}">
          </div>
          <div class="span-2">
            <label for="test_size">Holdout share</label>
            <input id="test_size" name="test_size" type="number" min="0.1" max="0.5" step="0.05" value="0.25">
          </div>
          <div class="span-2">
            <label for="confidence_threshold">Case confidence</label>
            <input id="confidence_threshold" name="confidence_threshold" type="number" min="0.5" max="0.99" step="0.01" value="0.90">
          </div>
          <div class="span-2">
            <label for="minimum_cases">Minimum cases</label>
            <input id="minimum_cases" name="minimum_cases" type="number" min="5" max="500" step="5" value="25">
          </div>
          <div class="span-2">
            <label for="top_predictors">Top predictors</label>
            <input id="top_predictors" name="top_predictors" type="number" min="3" max="20" value="8">
          </div>
          <div class="span-4">
            <label for="attribution_method">Attribution method</label>
            <select id="attribution_method" name="attribution_method">
              <option value="actual_shap" selected>Actual SHAP in probability units</option>
              <option value="fast_probability_perturbation">Fast approximation</option>
            </select>
          </div>
          <div class="span-2">
            <label for="shap_background_size">SHAP background cases</label>
            <input id="shap_background_size" name="shap_background_size" type="number" min="25" max="1000" step="25" value="100">
          </div>
          <div class="span-2">
            <label for="shap_kernel_nsamples">Kernel SHAP samples</label>
            <input id="shap_kernel_nsamples" name="shap_kernel_nsamples" type="number" min="100" max="5000" step="100" value="100">
          </div>
          <div class="span-8">
            <label>Ignore columns</label>
            <div class="grid">{''.join(f"<div class='span-3'>{box}</div>" for box in ignored_boxes)}</div>
          </div>
          <div class="span-12">
            <button type="submit">Run E2R2 pipeline</button>
          </div>
        </div>
      </form>
      <h3>Dataset Preview</h3>
      <div class="data-wrap">{table_html(df, max_rows=8)}</div>
    </section>
    """


def training_section() -> str:
    training = STATE.get("training")
    case_base = STATE.get("case_base")
    shap_table = STATE.get("shap_table")
    if training is None:
        return ""

    metrics = training.best_run.metrics
    metric_html = "".join(
        metric_card(label, metrics.get(key))
        for label, key in [
            ("Accuracy", "accuracy"),
            ("Recall", "recall"),
            ("Specificity", "specificity"),
            ("Precision", "precision"),
            ("F1", "f1"),
            ("AUC", "auc"),
        ]
    )
    model_rows = pd.DataFrame(
        [
            {
                "model": run.name,
                **{k: round(v, 4) for k, v in run.metrics.items()},
                "selection_score": round(run.score, 4),
            }
            for run in training.all_runs
        ]
    ).sort_values("selection_score", ascending=False)

    warning = ""
    if metrics.get("auc", 0) < 0.7 and metrics.get("balanced_accuracy", 0) < 0.7:
        warning = """
        <div class="warn">
          The selected baseline did not reach the usual 0.70 AUC or balanced-accuracy checkpoint.
          The case base was still prepared, but the E2R2 outputs should be treated as exploratory.
        </div>
        """

    return f"""
    <section>
      <h2>Baseline Model</h2>
      {warning}
      <p><strong>Selected model:</strong> {esc(training.best_run.name)}</p>
      <p class="muted">Retrieval uses {esc(training.similarity_weighting)}. All encoded predictor features receive equal weight when the cosine similarity score is calculated.</p>
      <div class="metrics">{metric_html}</div>
      <h3>Model Candidates</h3>
      <div class="data-wrap">{table_html(model_rows, max_rows=10)}</div>
    </section>
    <section>
      <h2>High-Confidence Case Base</h2>
      <p class="muted">{len(case_base) if case_base is not None else 0} correctly predicted, high-confidence precedent cases prepared from the holdout set.</p>
      <div class="actions">
        <a class="button" href="/download?name=case_base">Download case base</a>
        <a class="button" href="/download?name=shap_table">Download SHAP table</a>
        <a class="button" href="/download?name=holdout">Download holdout dataset</a>
        <a class="button" href="/download?name=case_vectors">Download case vectors</a>
        <a class="button" href="/download?name=holdout_vectors">Download holdout vectors</a>
      </div>
      <h3>Cases</h3>
      <div class="data-wrap">{table_html(case_base, max_rows=12)}</div>
      <h3>Top Predictors and SHAP Scores</h3>
      <div class="data-wrap">{table_html(shap_table, max_rows=20)}</div>
    </section>
    """


def prediction_section() -> str:
    training = STATE.get("training")
    case_base = STATE.get("case_base")
    if training is None or case_base is None:
        return ""
    row_options = [
        f"<option value='{esc(idx)}'>Holdout row {esc(idx)}</option>" for idx in training.X_test.index.tolist()
    ]
    result = STATE.get("prediction")
    result_html = ""
    if result:
        api_html = ""
        if result.get("llm_response"):
            api_html = f"<h3>LLM Response</h3><pre>{esc(result['llm_response'])}</pre>"
        elif result.get("llm_error"):
            api_html = f"<div class='warn'>{esc(result['llm_error'])}</div>"
        detail_html = ""
        if result.get("detailed_rationale"):
            detail_html = (
                "<details><summary>Detailed rationale</summary>"
                f"<p>{esc(result.get('detailed_rationale'))}</p>"
                "</details>"
            )
        simplifier_note = ""
        if result.get("simplified_with_model"):
            simplifier_note = (
                f"<p class='muted'>Displayed explanation simplified for general audiences with "
                f"{esc(result.get('simplified_with_model'))}.</p>"
            )
        elif result.get("simplification_error"):
            simplifier_note = f"<div class='warn'>{esc(result.get('simplification_error'))}</div>"
        result_html = f"""
        <h3>Prediction</h3>
        <div class="metrics">
          {metric_card("Predicted outcome", result.get("predicted_outcome"))}
          {metric_card("Confidence level", result.get("confidence_level"))}
        </div>
        <h3>Reasoning</h3>
        <p>{esc(result.get("rationale"))}</p>
        {simplifier_note}
        {detail_html}
        <h3>Retrieved Cases</h3>
        <div class="data-wrap">{table_html(result.get("retrieved"), max_rows=8)}</div>
        <h3>Alignment</h3>
        <div class="data-wrap">{table_html(result.get("alignment"), max_rows=10)}</div>
        <h3>Prepared LLM Prompt</h3>
        <pre>{esc(result.get("prompt"))}</pre>
        {api_html}
        """

    return f"""
    <section>
      <h2>Holdout Case</h2>
      <form method="post" action="/predict" enctype="multipart/form-data">
        <div class="grid">
          <div class="span-4">
            <label for="holdout_index">Existing holdout row</label>
            <select id="holdout_index" name="holdout_index">{''.join(row_options)}</select>
          </div>
          <div class="span-4">
            <label for="holdout_file">New holdout file</label>
            <input id="holdout_file" name="holdout_file" type="file" accept=".csv,.xlsx,.xls">
          </div>
          <div class="span-2">
            <label for="neighbors">Retrieved cases</label>
            <input id="neighbors" name="neighbors" type="number" min="1" max="12" value="5">
          </div>
          <div class="span-4">
            <label for="api_key">OpenAI API key</label>
            <input id="api_key" name="api_key" type="password" autocomplete="off">
          </div>
          <div class="span-3">
            <label for="model">LLM model</label>
            <select id="model" name="model">{model_option_tags()}</select>
          </div>
          <div class="span-12">
            <button type="submit">Predict holdout case</button>
          </div>
        </div>
      </form>
      {result_html}
    </section>
    """


def home() -> bytes:
    body = upload_section()
    if STATE.get("dataset") is not None:
        body = configuration_section() + training_section() + prediction_section()
    return page("E2R2 Data Analysis Pipeline", body)


def reset_error() -> None:
    STATE["error"] = None


def parse_post(handler: BaseHTTPRequestHandler) -> cgi.FieldStorage:
    return cgi.FieldStorage(
        fp=handler.rfile,
        headers=handler.headers,
        environ={
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": handler.headers.get("Content-Type", ""),
        },
    )


def form_get(form: cgi.FieldStorage, key: str, default: Any = "") -> Any:
    if key not in form:
        return default
    value = form[key]
    if isinstance(value, list):
        return [item.value for item in value]
    return value.value


def response_token_limit(model: str) -> int:
    if "pro" in model:
        return 6000
    if model in {"gpt-5-mini", "gpt-5-nano"}:
        return 3200
    if model.startswith("gpt-5") or model.startswith("o"):
        return 2400
    return 1200


def call_openai_responses(api_key: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": response_token_limit(model),
    }
    if not model.startswith("gpt-5") and not model.startswith("o"):
        payload["temperature"] = 0.2
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if "output_text" in payload:
        return payload["output_text"]
    chunks = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(content.get("text", ""))
    return "\n".join(chunks).strip() or json.dumps(payload, indent=2)


def parse_llm_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def simplify_final_rationale(
    api_key: str,
    prediction_goal: str,
    predicted_outcome: Any,
    confidence_level: Any,
    detailed_rationale: str,
) -> str:
    simplify_prompt = f"""You are rewriting a technical decision rationale for a general audience.

Prediction goal:
{prediction_goal or "Predict the outcome for this case."}

Predicted outcome:
{predicted_outcome}

Confidence level:
{confidence_level}

Detailed rationale:
{detailed_rationale}

Rewrite this as one short, plain-language paragraph for a non-expert user.
Requirements:
- Keep the same conclusion and overall meaning.
- Use everyday language.
- Briefly explain the practical implications.
- If the confidence is low or moderate, make the uncertainty understandable without sounding alarmist.
- Do not mention SHAP, precedent retrieval, cosine similarity, Stage 1, Stage 2, baseline verification, or other technical framework terms.
- Do not invent new facts, advice, or guarantees.
- Return plain text only.
"""
    return call_openai_responses(api_key, RATIONALE_SIMPLIFIER_MODEL, simplify_prompt).strip()


class E2R2Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/download":
            self.handle_download(parse_qs(parsed.query))
            return
        if parsed.path == "/progress":
            self.handle_progress()
            return
        self.respond(200, home())

    def do_POST(self) -> None:
        try:
            reset_error()
            parsed = urlparse(self.path)
            if parsed.path == "/upload":
                self.handle_upload()
            elif parsed.path == "/train":
                self.handle_train()
            elif parsed.path == "/predict":
                self.handle_predict()
            else:
                self.respond(404, page("Not found", "<section><h2>Not found</h2></section>"))
        except Exception as exc:
            set_progress("Stopped before finishing", 100, active=False)
            STATE["error"] = f"{exc}\n\n{traceback.format_exc(limit=3)}"
            self.respond(500, home())

    def handle_upload(self) -> None:
        form = parse_post(self)
        dataset = load_table_from_upload(form["dataset"]) if "dataset" in form else None
        dictionary_df = load_table_from_upload(form["dictionary"]) if "dictionary" in form else None
        if dataset is None or dataset.empty:
            raise ValueError("The dataset file did not contain readable rows.")
        STATE.update(
            {
                "dataset": dataset,
                "dictionary_df": dictionary_df,
                "dictionary": parse_data_dictionary(dictionary_df),
                "prediction_goal": form_get(form, "prediction_goal", ""),
                "training": None,
                "case_base": None,
                "shap_table": None,
                "prediction": None,
            }
        )
        self.redirect("/")

    def handle_train(self) -> None:
        df = STATE.get("dataset")
        if df is None:
            raise ValueError("Load a dataset first.")
        set_progress("Preparing pipeline settings", 3)
        form = parse_post(self)
        target_column = form_get(form, "target_column")
        positive_label = form_get(form, "positive_label")
        ignored = form_get(form, "ignored_columns", [])
        if isinstance(ignored, str):
            ignored = [ignored]
        test_size = float(form_get(form, "test_size", 0.25))
        confidence_threshold = float(form_get(form, "confidence_threshold", 0.9))
        minimum_cases = int(float(form_get(form, "minimum_cases", 25)))
        top_predictors = int(float(form_get(form, "top_predictors", 8)))
        attribution_method = form_get(form, "attribution_method", "actual_shap")
        shap_background_size = int(float(form_get(form, "shap_background_size", 100)))
        shap_kernel_nsamples = int(float(form_get(form, "shap_kernel_nsamples", 100)))

        training = train_e2r2_baseline(
            df,
            target_column=target_column,
            positive_label=positive_label,
            ignored_columns=ignored,
            test_size=test_size,
            progress_callback=set_progress,
        )
        case_base, shap_table = build_case_base(
            training,
            data_dictionary=STATE.get("dictionary"),
            confidence_threshold=confidence_threshold,
            minimum_cases=minimum_cases,
            top_predictors=top_predictors,
            attribution_method=attribution_method,
            shap_background_size=shap_background_size,
            shap_kernel_nsamples=shap_kernel_nsamples,
            progress_callback=set_progress,
        )
        set_progress("Complete: results are ready", 100, active=False)
        STATE.update(
            {
                "target_column": target_column,
                "positive_label": positive_label,
                "ignored_columns": ignored,
                "training": training,
                "case_base": case_base,
                "shap_table": shap_table,
                "attribution_method": attribution_method,
                "shap_background_size": shap_background_size,
                "shap_kernel_nsamples": shap_kernel_nsamples,
                "prediction": None,
            }
        )
        self.redirect("/")

    def handle_progress(self) -> None:
        payload = json.dumps(progress_snapshot()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def handle_predict(self) -> None:
        training = STATE.get("training")
        case_base = STATE.get("case_base")
        shap_table = STATE.get("shap_table")
        if training is None or case_base is None or shap_table is None:
            raise ValueError("Run the E2R2 pipeline first.")
        form = parse_post(self)
        uploaded_holdout = load_table_from_upload(form["holdout_file"]) if "holdout_file" in form else None
        if uploaded_holdout is not None and not uploaded_holdout.empty:
            holdout_row = uploaded_holdout.drop(columns=[training.target_column], errors="ignore").iloc[0]
            holdout_row = holdout_row.reindex(training.feature_columns)
        else:
            raw_index = form_get(form, "holdout_index")
            index_value = int(raw_index) if str(raw_index).isdigit() else raw_index
            holdout_row = training.X_test.loc[index_value]

        neighbors = int(float(form_get(form, "neighbors", 5)))
        retrieved = retrieve_similar_cases(training, case_base, holdout_row, k=neighbors)
        alignment = summarize_peer_alignment(holdout_row, retrieved, shap_table, training)
        prompt = build_e2r2_prompt(
            training,
            holdout_row,
            retrieved,
            shap_table,
            alignment,
            data_dictionary=STATE.get("dictionary"),
            prediction_goal=STATE.get("prediction_goal", ""),
        )
        fallback_predicted_outcome, fallback_confidence_level, fallback_rationale = local_e2r2_reasoning(
            training,
            holdout_row,
            retrieved,
            alignment,
        )
        api_key = form_get(form, "api_key", "")
        model = form_get(form, "model", "gpt-4.1-mini")
        llm_response = ""
        llm_error = ""
        simplification_error = ""
        simplified_with_model = ""
        detailed_rationale = ""
        predicted_outcome = fallback_predicted_outcome
        confidence_level = fallback_confidence_level
        rationale = fallback_rationale
        if api_key:
            try:
                llm_response = call_openai_responses(api_key, model, prompt)
                parsed = parse_llm_json(llm_response)
                llm_predicted_outcome = parsed.get("final_predicted_outcome", parsed.get("predicted_outcome", ""))
                llm_confidence_level = parsed.get("final_confidence_level", parsed.get("confidence_level", ""))
                detailed_rationale = parsed.get("final_rationale", parsed.get("rationale", ""))
                if llm_predicted_outcome and llm_confidence_level and detailed_rationale:
                    predicted_outcome = llm_predicted_outcome
                    confidence_level = llm_confidence_level
                    rationale = detailed_rationale
                    try:
                        simplified = simplify_final_rationale(
                            api_key=api_key,
                            prediction_goal=STATE.get("prediction_goal", ""),
                            predicted_outcome=predicted_outcome,
                            confidence_level=confidence_level,
                            detailed_rationale=detailed_rationale,
                        )
                        if simplified:
                            rationale = simplified
                            simplified_with_model = RATIONALE_SIMPLIFIER_MODEL
                    except Exception as exc:
                        simplification_error = (
                            f"The prediction was generated successfully, but the plain-language "
                            f"rewrite step failed: {exc}"
                        )
                else:
                    llm_error = (
                        "The selected LLM returned a response, but the app could not extract the "
                        "expected prediction fields. The local E2R2 result is shown instead."
                    )
            except Exception as exc:
                llm_error = f"The local E2R2 result was produced, but the LLM call failed: {exc}"
        STATE["prediction"] = {
            "predicted_outcome": predicted_outcome,
            "confidence_level": confidence_level,
            "rationale": rationale,
            "detailed_rationale": detailed_rationale,
            "retrieved": retrieved,
            "alignment": alignment,
            "prompt": prompt,
            "llm_response": llm_response,
            "llm_error": llm_error,
            "simplification_error": simplification_error,
            "simplified_with_model": simplified_with_model,
        }
        self.redirect("/")

    def handle_download(self, query: Dict[str, Any]) -> None:
        name = query.get("name", [""])[0]
        if name == "case_base":
            df = STATE.get("case_base")
            filename = "e2r2_case_base.csv"
        elif name == "shap_table":
            df = STATE.get("shap_table")
            filename = "e2r2_shap_scores.csv"
        elif name == "holdout":
            training = STATE.get("training")
            df = holdout_dataset(training) if training is not None else None
            filename = "e2r2_holdout_dataset.csv"
        elif name == "case_vectors":
            case_base = STATE.get("case_base")
            df = case_vectors_dataset(case_base) if case_base is not None else None
            filename = "e2r2_case_vectors.csv"
        elif name == "holdout_vectors":
            training = STATE.get("training")
            df = holdout_vectors_dataset(training) if training is not None else None
            filename = "e2r2_holdout_vectors.csv"
        else:
            self.respond(404, page("Not found", "<section><h2>Download not found</h2></section>"))
            return
        if df is None:
            self.respond(404, page("Not ready", "<section><h2>Run the pipeline first.</h2></section>"))
            return
        if name in {"case_vectors", "holdout_vectors"}:
            download_df = df
        else:
            download_df = df[[c for c in df.columns if not str(c).startswith("vector_")]]
        payload = dataframe_to_csv_bytes(download_df)
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def respond(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


def run() -> None:
    server = None
    port = DEFAULT_PORT
    for candidate in range(DEFAULT_PORT, DEFAULT_PORT + 30):
        try:
            server = ThreadingHTTPServer((HOST, candidate), E2R2Handler)
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError("No open local port was found for the E2R2 app.")
    url = f"http://{HOST}:{port}"
    print(f"E2R2 app running at {url}")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    run()
