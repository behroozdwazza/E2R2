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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests

from e2r2_pipeline import dataframe_to_csv_bytes, infer_llm_role, read_uploaded_table


HOST = "127.0.0.1"
DEFAULT_PORT = 8775

OPENAI_LLM_MODELS = [
    ("gpt-5.1", "GPT-5.1"),
    ("gpt-5.4-mini", "GPT-5.4 mini"),
    ("gpt-5", "GPT-5"),
    ("gpt-5-mini", "GPT-5 mini"),
    ("gpt-5-nano", "GPT-5 nano"),
    ("gpt-5-pro", "GPT-5 pro"),
    ("gpt-4.1", "GPT-4.1"),
    ("gpt-4.1-mini", "GPT-4.1 mini"),
    ("gpt-4.1-nano", "GPT-4.1 nano"),
]

BASELINE_COLUMNS = {
    "baseline_predicted_outcome",
    "baseline_positive_probability",
    "baseline_confidence",
    "baseline_correct",
}

STATE: Dict[str, Any] = {
    "case_base": None,
    "shap_scores": None,
    "holdout": None,
    "target_column": "",
    "feature_columns": [],
    "case_vectors": None,
    "holdout_vectors": None,
    "case_row_positions": None,
    "retrieval_mode": "reconstructed exported features",
    "api_key": "",
    "model": "gpt-5.1",
    "prediction_goal": "",
    "include_baseline_signal": True,
    "individual_result": None,
    "batch_results": None,
    "error": None,
}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def load_upload(field: Any) -> Optional[pd.DataFrame]:
    if field is None or not getattr(field, "filename", ""):
        return None
    data = field.file.read()
    file_obj = io.BytesIO(data)
    file_obj.name = field.filename
    return read_uploaded_table(file_obj)


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


def model_option_tags(selected: str) -> str:
    return "\n".join(
        f"<option value='{esc(value)}'{' selected' if value == selected else ''}>{esc(label)} ({esc(value)})</option>"
        for value, label in OPENAI_LLM_MODELS
    )


def table_html(df: Optional[pd.DataFrame], max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No rows yet.</p>"
    return df.head(max_rows).to_html(index=False, classes="data-table", border=0, escape=True)


def infer_target_column(holdout: pd.DataFrame) -> str:
    columns = list(holdout.columns)
    if "baseline_predicted_outcome" in columns:
        idx = columns.index("baseline_predicted_outcome")
        if idx > 0:
            return columns[idx - 1]
    candidates = [c for c in columns if c.upper().endswith("STATUS") or "OUTCOME" in c.upper()]
    return candidates[-1] if candidates else ""


def feature_columns_for_retrieval(holdout: pd.DataFrame, target_column: str) -> List[str]:
    excluded = set(BASELINE_COLUMNS) | {"row_index", target_column}
    return [c for c in holdout.columns if c not in excluded]


def build_retrieval_vectors(
    holdout: pd.DataFrame,
    case_base: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    features = holdout[feature_columns].copy()
    numeric_cols = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    categorical_cols = [c for c in features.columns if c not in numeric_cols]

    numeric = features[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=features.index)
    for col in numeric.columns:
        numeric[col] = numeric[col].fillna(numeric[col].median())
        std = numeric[col].std()
        numeric[col] = 0 if not std or pd.isna(std) else (numeric[col] - numeric[col].mean()) / std

    categorical = pd.get_dummies(
        features[categorical_cols].fillna("__missing__").astype(str),
        dummy_na=False,
    ) if categorical_cols else pd.DataFrame(index=features.index)

    matrix = pd.concat([numeric, categorical], axis=1).to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    holdout_vectors = matrix / norms

    row_to_position = {row_index: pos for pos, row_index in enumerate(holdout["row_index"].tolist())}
    case_positions = [
        row_to_position[row_index]
        for row_index in case_base["row_index"].tolist()
        if row_index in row_to_position
    ]
    if not case_positions:
        raise ValueError("None of the case-base row_index values were found in the holdout dataset.")
    return holdout_vectors, holdout_vectors[case_positions], case_positions


def vector_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith("vector_")]


def load_exported_vectors(
    holdout: pd.DataFrame,
    case_base: pd.DataFrame,
    holdout_vector_df: Optional[pd.DataFrame],
    case_vector_df: Optional[pd.DataFrame],
) -> Optional[Tuple[np.ndarray, np.ndarray, List[int]]]:
    if holdout_vector_df is None or case_vector_df is None:
        return None
    h_cols = vector_columns(holdout_vector_df)
    c_cols = vector_columns(case_vector_df)
    if not h_cols or not c_cols:
        raise ValueError("Vector files must contain vector_ columns.")
    if h_cols != c_cols:
        raise ValueError("Holdout vector and case vector files have different vector columns.")

    holdout_joined = holdout[["row_index"]].merge(
        holdout_vector_df[["row_index"] + h_cols],
        on="row_index",
        how="left",
    )
    if holdout_joined[h_cols].isna().any().any():
        raise ValueError("Some holdout rows were missing from e2r2_holdout_vectors.csv.")

    case_joined = case_base[["case_id", "row_index"]].merge(
        case_vector_df[["case_id", "row_index"] + c_cols],
        on=["case_id", "row_index"],
        how="left",
    )
    if case_joined[c_cols].isna().any().any():
        raise ValueError("Some case-base rows were missing from e2r2_case_vectors.csv.")

    row_to_position = {row_index: pos for pos, row_index in enumerate(holdout["row_index"].tolist())}
    case_positions = [
        row_to_position[row_index]
        for row_index in case_base["row_index"].tolist()
        if row_index in row_to_position
    ]
    if len(case_positions) != len(case_base):
        raise ValueError("Some case-base row_index values were not found in the holdout dataset.")
    return (
        holdout_joined[h_cols].to_numpy(dtype=float),
        case_joined[c_cols].to_numpy(dtype=float),
        case_positions,
    )


def prepare_loaded_data(
    case_base: pd.DataFrame,
    shap_scores: pd.DataFrame,
    holdout: pd.DataFrame,
    case_vector_df: Optional[pd.DataFrame] = None,
    holdout_vector_df: Optional[pd.DataFrame] = None,
) -> None:
    required_case = {"case_id", "row_index", "predicted_outcome", "confidence", "top_predictors"}
    required_shap = {"case_id", "feature", "value", "shap_score", "abs_shap_score"}
    required_holdout = {"row_index", "baseline_predicted_outcome", "baseline_confidence"}
    missing = []
    if not required_case.issubset(case_base.columns):
        missing.append(f"case base missing: {sorted(required_case - set(case_base.columns))}")
    if not required_shap.issubset(shap_scores.columns):
        missing.append(f"SHAP scores missing: {sorted(required_shap - set(shap_scores.columns))}")
    if not required_holdout.issubset(holdout.columns):
        missing.append(f"holdout missing: {sorted(required_holdout - set(holdout.columns))}")
    if missing:
        raise ValueError("; ".join(missing))

    target_column = infer_target_column(holdout)
    feature_columns = feature_columns_for_retrieval(holdout, target_column)
    exported_vectors = load_exported_vectors(holdout, case_base, holdout_vector_df, case_vector_df)
    if exported_vectors is not None:
        holdout_vectors, case_vectors, case_positions = exported_vectors
        retrieval_mode = "exported training-pipeline vectors"
    else:
        holdout_vectors, case_vectors, case_positions = build_retrieval_vectors(
            holdout,
            case_base,
            feature_columns,
        )
        retrieval_mode = "reconstructed exported features"
    STATE.update(
        {
            "case_base": case_base,
            "shap_scores": shap_scores,
            "holdout": holdout,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "holdout_vectors": holdout_vectors,
            "case_vectors": case_vectors,
            "case_row_positions": case_positions,
            "retrieval_mode": retrieval_mode,
            "individual_result": None,
            "batch_results": None,
        }
    )


def retrieve_cases(row_index: Any, k: int) -> pd.DataFrame:
    case_base = STATE["case_base"]
    holdout = STATE["holdout"]
    holdout_vectors = STATE["holdout_vectors"]
    case_vectors = STATE["case_vectors"]
    case_positions = STATE["case_row_positions"]
    row_to_position = {value: pos for pos, value in enumerate(holdout["row_index"].tolist())}
    if row_index not in row_to_position:
        raise ValueError(f"Holdout row_index {row_index} was not found.")
    vector = holdout_vectors[row_to_position[row_index]]
    similarities = case_vectors @ vector
    rows = case_base.copy()
    rows["similarity"] = similarities
    rows = rows[rows["row_index"] != row_index]
    return rows.sort_values("similarity", ascending=False).head(k).reset_index(drop=True)


def feature_definition(feature: str) -> str:
    shap_scores = STATE["shap_scores"]
    if "definition" not in shap_scores.columns:
        return ""
    matches = shap_scores.loc[shap_scores["feature"].astype(str) == str(feature), "definition"].dropna()
    return "" if matches.empty else str(matches.iloc[0])


def shap_weighted_alignment(row: pd.Series, retrieved: pd.DataFrame) -> str:
    shap_scores = STATE["shap_scores"]
    rows = []
    retrieved_case_ids = set(retrieved["case_id"].tolist())
    relevant = shap_scores[shap_scores["case_id"].isin(retrieved_case_ids)].copy()
    if relevant.empty:
        return "- No SHAP alignment rows were available."
    for feature, group in relevant.groupby("feature"):
        mean_shap = group["shap_score"].mean()
        mean_abs = group["abs_shap_score"].mean()
        positive_count = int((group["shap_score"] > 0).sum())
        negative_count = int((group["shap_score"] < 0).sum())
        peer_values = group["value"].dropna().astype(str).value_counts().head(3)
        holdout_value = row.get(feature, "")
        matching_count = int((group["value"].astype(str) == str(holdout_value)).sum())
        if mean_shap > 0:
            direction = "pushes toward the case outcome"
        elif mean_shap < 0:
            direction = "pushes away from the case outcome"
        else:
            direction = "has near-neutral direction"
        rows.append(
            {
                "feature": feature,
                "line": (
                    f"- {feature}: holdout={holdout_value}; peer values="
                    f"{', '.join([f'{value} ({count})' for value, count in peer_values.items()])}; "
                    f"mean SHAP={mean_shap:.4f}, mean |SHAP|={mean_abs:.4f}; "
                    f"{direction}; sign counts +/{positive_count}, -/{negative_count}; "
                    f"exact value matches among retrieved cases={matching_count}"
                ),
                "mean_abs": mean_abs,
            }
        )
    rows = sorted(rows, key=lambda item: item["mean_abs"], reverse=True)[:10]
    return "\n".join(item["line"] for item in rows)


def build_prompt(row: pd.Series, retrieved: pd.DataFrame, neighbors: int) -> str:
    shap_scores = STATE["shap_scores"]
    target_column = STATE["target_column"]
    prediction_goal = STATE["prediction_goal"]
    role = infer_llm_role(prediction_goal, target_column)

    excluded = set(BASELINE_COLUMNS) | {"row_index", target_column}
    feature_lines = []
    for feature, value in row.items():
        if feature in excluded:
            continue
        definition = feature_definition(feature)
        suffix = f" ({definition})" if definition else ""
        feature_lines.append(f"- {feature}{suffix}: {value}")

    peer_blocks = []
    for _, peer in retrieved.iterrows():
        peer_shap = shap_scores[shap_scores["case_id"] == peer["case_id"]].sort_values("rank").head(8)
        units = (
            peer_shap["shap_output_units"].dropna().iloc[0]
            if "shap_output_units" in peer_shap.columns and not peer_shap["shap_output_units"].dropna().empty
            else "probability"
        )
        shap_lines = [
            f"  - {item.feature}: value={item.value}, SHAP contribution={item.shap_score:.4f} ({units})"
            for item in peer_shap.itertuples()
        ]
        peer_blocks.append(
            "\n".join(
                [
                    f"Case {peer.case_id}: outcome={peer.predicted_outcome}, "
                    f"baseline confidence={peer.confidence:.3f}, retrieval similarity={peer.similarity:.3f}",
                    *shap_lines,
                ]
            )
        )

    weighted_alignment = shap_weighted_alignment(row, retrieved)
    include_baseline = bool(STATE.get("include_baseline_signal", True))
    baseline_probability = row.get("baseline_positive_probability", "")
    baseline_confidence = row.get("baseline_confidence", "")
    baseline_outcome = row.get("baseline_predicted_outcome", "")
    baseline_block = (
        "\n".join(
            [
                "Baseline model signal for this holdout case:",
                f"- baseline_predicted_outcome: {baseline_outcome}",
                f"- baseline_positive_probability: {baseline_probability}",
                f"- baseline_confidence: {baseline_confidence}",
            ]
        )
        if include_baseline
        else "Baseline model signal for this holdout case: intentionally omitted for this experiment."
    )

    return f"""{role}

You are using the E2R2 framework: Explain, Embed, Retrieve, and Reason.
E2R2 combines a baseline predictive model, SHAP feature attributions, similarity-based precedent retrieval, and professional reasoning. The retrieved precedents come from a case base of correctly predicted, high-confidence holdout cases. Your job is to synthesize this evidence for the new holdout case.

Important rules:
- Do not use majority vote alone.
- Compare the holdout case with retrieved precedent cases on SHAP-important predictors.
- Interpret SHAP sign and magnitude explicitly. Larger absolute SHAP values carry more explanatory weight than smaller values.
- A positive SHAP contribution means that feature pushed the precedent case toward its listed outcome; a negative SHAP contribution means it pushed away from that outcome.
- When deciding, prioritize features with large absolute SHAP values and evaluate whether the holdout case matches or diverges from the precedent values on those high-weight features.
- Treat retrieval similarity, SHAP-weighted alignment, and feature alignment as complementary evidence.
- If a baseline model signal is provided, treat it as one additional signal. If it is omitted, do not infer it.
- Do not mention the true outcome; it is not provided for reasoning.
- Return valid JSON only, with no Markdown.

Prediction goal:
{prediction_goal or f"Predict {target_column}."}

{baseline_block}

Holdout case profile:
{chr(10).join(feature_lines)}

Retrieved E2R2 precedent cases, top {neighbors}:
{chr(10).join(peer_blocks)}

SHAP-weighted alignment summary across retrieved cases:
{weighted_alignment}

Return this JSON object:
{{
  "predicted_outcome": "one of the domain labels",
  "confidence_level": "low | moderate | high",
  "rationale": "one professional paragraph that explicitly uses SHAP sign, SHAP magnitude, precedent similarity, and baseline confidence"
}}
"""


def call_openai(api_key: str, model: str, prompt: str) -> str:
    payload: Dict[str, Any] = {"model": model, "input": prompt}
    if not model.startswith("gpt-5"):
        payload["temperature"] = 0.2
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("output_text"):
        return data["output_text"]
    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(content.get("text", ""))
    return "\n".join(chunks).strip()


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


def run_prediction_for_row(row_index: Any, neighbors: int) -> Dict[str, Any]:
    api_key = STATE["api_key"]
    if not api_key:
        raise ValueError("Enter and save your OpenAI API key first.")
    holdout = STATE["holdout"]
    row = holdout.loc[holdout["row_index"] == row_index].iloc[0]
    retrieved = retrieve_cases(row_index, neighbors)
    prompt = build_prompt(row, retrieved, neighbors)
    raw = call_openai(api_key, STATE["model"], prompt)
    parsed = parse_llm_json(raw)
    target_column = STATE["target_column"]
    return {
        "row_index": row_index,
        "actual_outcome": row.get(target_column, ""),
        "baseline_predicted_outcome": row.get("baseline_predicted_outcome", ""),
        "baseline_confidence": row.get("baseline_confidence", ""),
        "baseline_signal_in_prompt": bool(STATE.get("include_baseline_signal", True)),
        "llm_predicted_outcome": parsed.get("predicted_outcome", ""),
        "llm_confidence_level": parsed.get("confidence_level", ""),
        "llm_rationale": parsed.get("rationale", ""),
        "llm_raw_response": raw,
        "prompt": prompt,
        "retrieved_case_ids": ", ".join(retrieved["case_id"].astype(str).tolist()),
    }


def css() -> str:
    return """
    <style>
      :root { --ink:#1f2933; --muted:#607080; --line:#d9e2ec; --panel:#f7f9fb; --accent:#0f766e; --accent-dark:#115e59; }
      * { box-sizing: border-box; }
      body { margin:0; font-family:"Segoe UI", Arial, sans-serif; color:var(--ink); background:#fff; }
      header { padding:28px 40px 18px; border-bottom:1px solid var(--line); background:#fbfcfd; }
      main { padding:24px 40px 48px; max-width:1280px; }
      h1 { margin:0 0 8px; font-size:28px; font-weight:650; letter-spacing:0; }
      h2 { margin:0 0 16px; font-size:19px; font-weight:650; letter-spacing:0; }
      h3 { margin:18px 0 10px; font-size:16px; letter-spacing:0; }
      section { border-bottom:1px solid var(--line); padding:22px 0; }
      label { display:block; font-size:13px; font-weight:600; color:#344454; margin:0 0 6px; }
      input, select, textarea { width:100%; border:1px solid #b7c4d1; border-radius:6px; padding:9px 10px; font:inherit; background:#fff; }
      textarea { min-height:90px; resize:vertical; }
      button, .button { border:0; border-radius:6px; padding:10px 14px; background:var(--accent); color:#fff; font-weight:650; text-decoration:none; cursor:pointer; display:inline-block; }
      button:hover, .button:hover { background:var(--accent-dark); }
      .grid { display:grid; grid-template-columns:repeat(12, 1fr); gap:16px; align-items:end; }
      .span-2 { grid-column:span 2; } .span-3 { grid-column:span 3; } .span-4 { grid-column:span 4; } .span-5 { grid-column:span 5; } .span-6 { grid-column:span 6; } .span-8 { grid-column:span 8; } .span-12 { grid-column:span 12; }
      .muted { color:var(--muted); }
      .error { border-left:4px solid #b91c1c; padding:12px 14px; background:#fff5f5; color:#7f1d1d; }
      .metric { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:10px 12px; }
      .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; }
      .data-wrap { overflow:auto; border:1px solid var(--line); border-radius:6px; }
      .data-table { width:100%; border-collapse:collapse; font-size:13px; }
      .data-table th, .data-table td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
      .data-table th { background:var(--panel); position:sticky; top:0; z-index:1; }
      pre { white-space:pre-wrap; background:#f6f8fa; border:1px solid var(--line); border-radius:6px; padding:14px; max-height:420px; overflow:auto; }
      .actions { display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
      @media (max-width:820px) { header, main { padding-left:18px; padding-right:18px; } .grid { grid-template-columns:1fr; } .span-2,.span-3,.span-4,.span-5,.span-6,.span-8,.span-12 { grid-column:span 1; } }
    </style>
    """


def page(body: str) -> bytes:
    status = ""
    if STATE.get("error"):
        status = f"<div class='error'><strong>Something needs attention.</strong><br>{esc(STATE['error'])}</div>"
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>E2R2 LLM Experiment Lab</title>
  {css()}
</head>
<body>
  <header>
    <h1>E2R2 LLM Experiment Lab</h1>
    <div class="muted">Upload exported E2R2 files and run LLM predictions without retraining.</div>
  </header>
  <main>{status}{body}</main>
</body>
</html>"""
    return html_doc.encode("utf-8")


def upload_section() -> str:
    loaded = STATE["holdout"] is not None
    metrics = ""
    if loaded:
        metrics = f"""
        <div class="metrics">
          <div class="metric"><strong>{len(STATE['case_base'])}</strong><br><span class="muted">case-base records</span></div>
          <div class="metric"><strong>{len(STATE['shap_scores'])}</strong><br><span class="muted">SHAP rows</span></div>
          <div class="metric"><strong>{len(STATE['holdout'])}</strong><br><span class="muted">holdout rows</span></div>
          <div class="metric"><strong>{esc(STATE['target_column'])}</strong><br><span class="muted">actual outcome column</span></div>
          <div class="metric"><strong>{esc(STATE['retrieval_mode'])}</strong><br><span class="muted">retrieval mode</span></div>
        </div>
        """
    return f"""
    <section>
      <h2>Exported E2R2 Files</h2>
      <form method="post" action="/upload" enctype="multipart/form-data">
        <div class="grid">
          <div class="span-4"><label>Case base CSV</label><input name="case_base" type="file" accept=".csv,.xlsx,.xls" required></div>
          <div class="span-4"><label>SHAP scores CSV</label><input name="shap_scores" type="file" accept=".csv,.xlsx,.xls" required></div>
          <div class="span-4"><label>Holdout dataset CSV</label><input name="holdout" type="file" accept=".csv,.xlsx,.xls" required></div>
          <div class="span-6"><label>Case vectors CSV</label><input name="case_vectors" type="file" accept=".csv,.xlsx,.xls"></div>
          <div class="span-6"><label>Holdout vectors CSV</label><input name="holdout_vectors" type="file" accept=".csv,.xlsx,.xls"></div>
          <div class="span-12"><button type="submit">Load experiment files</button></div>
        </div>
      </form>
      {metrics}
    </section>
    """


def settings_section() -> str:
    if STATE["holdout"] is None:
        return ""
    saved = "Saved" if STATE.get("api_key") else "Not saved"
    baseline_checked = " checked" if STATE.get("include_baseline_signal", True) else ""
    return f"""
    <section>
      <h2>LLM Settings</h2>
      <form method="post" action="/settings">
        <div class="grid">
          <div class="span-5"><label>OpenAI API key</label><input name="api_key" type="password" autocomplete="off" placeholder="Paste once; stored in this local app session"></div>
          <div class="span-3"><label>API key status</label><input value="{saved}" disabled></div>
          <div class="span-4"><label>LLM model</label><select name="model">{model_option_tags(STATE['model'])}</select></div>
          <div class="span-12">
            <label><input style="width:auto" name="include_baseline_signal" type="checkbox" value="yes"{baseline_checked}> Include baseline model signal in prompt</label>
          </div>
          <div class="span-12"><label>Prediction goal</label><textarea name="prediction_goal">{esc(STATE.get('prediction_goal', ''))}</textarea></div>
          <div class="span-12"><button type="submit">Save settings</button></div>
        </div>
      </form>
    </section>
    """


def individual_section() -> str:
    if STATE["holdout"] is None:
        return ""
    holdout = STATE["holdout"]
    options = "\n".join(
        f"<option value='{esc(value)}'>Row {esc(value)}</option>"
        for value in holdout["row_index"].head(1000).tolist()
    )
    result = STATE.get("individual_result")
    result_html = ""
    if result:
        compact = pd.DataFrame([{k: v for k, v in result.items() if k not in {"prompt", "llm_raw_response"}}])
        result_html = f"""
        <h3>Latest Individual Result</h3>
        <div class="data-wrap">{table_html(compact, max_rows=1)}</div>
        <h3>Prompt</h3><pre>{esc(result.get('prompt'))}</pre>
        <h3>Raw LLM Response</h3><pre>{esc(result.get('llm_raw_response'))}</pre>
        """
    return f"""
    <section>
      <h2>Individual Prediction</h2>
      <form method="post" action="/predict-one">
        <div class="grid">
          <div class="span-4"><label>Holdout row</label><select name="row_index">{options}</select></div>
          <div class="span-2"><label>Retrieved cases</label><input name="neighbors" type="number" min="1" max="12" value="5"></div>
          <div class="span-12"><button type="submit">Generate LLM prediction</button></div>
        </div>
      </form>
      {result_html}
    </section>
    """


def batch_section() -> str:
    if STATE["holdout"] is None:
        return ""
    results = STATE.get("batch_results")
    result_html = ""
    if results is not None and not results.empty:
        result_html = f"""
        <h3>Batch Results</h3>
        <div class="actions"><a class="button" href="/download?name=batch">Download batch predictions</a></div>
        <div class="data-wrap">{table_html(results.drop(columns=['prompt', 'llm_raw_response'], errors='ignore'), max_rows=25)}</div>
        """
    return f"""
    <section>
      <h2>Batch Prediction</h2>
      <form method="post" action="/predict-batch">
        <div class="grid">
          <div class="span-3"><label>Start after first rows</label><input name="offset" type="number" min="0" value="0"></div>
          <div class="span-3"><label>Number of rows</label><input name="limit" type="number" min="1" value="25"></div>
          <div class="span-2"><label>Retrieved cases</label><input name="neighbors" type="number" min="1" max="12" value="5"></div>
          <div class="span-12"><button type="submit">Run batch</button></div>
        </div>
      </form>
      <p class="muted">For the entire holdout batch, set Number of rows to the full holdout count. Large runs can take time and use API credits.</p>
      {result_html}
    </section>
    """


def home() -> bytes:
    body = upload_section() + settings_section() + individual_section() + batch_section()
    return page(body)


class LLMExperimentHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/download":
            self.handle_download(parse_qs(parsed.query))
            return
        self.respond(200, home())

    def do_POST(self) -> None:
        try:
            STATE["error"] = None
            parsed = urlparse(self.path)
            if parsed.path == "/upload":
                self.handle_upload()
            elif parsed.path == "/settings":
                self.handle_settings()
            elif parsed.path == "/predict-one":
                self.handle_predict_one()
            elif parsed.path == "/predict-batch":
                self.handle_predict_batch()
            else:
                self.respond(404, page("<section><h2>Not found</h2></section>"))
        except Exception as exc:
            STATE["error"] = f"{exc}\n\n{traceback.format_exc(limit=3)}"
            self.respond(500, home())

    def handle_upload(self) -> None:
        form = parse_post(self)
        case_base = load_upload(form["case_base"]) if "case_base" in form else None
        shap_scores = load_upload(form["shap_scores"]) if "shap_scores" in form else None
        holdout = load_upload(form["holdout"]) if "holdout" in form else None
        case_vectors = load_upload(form["case_vectors"]) if "case_vectors" in form else None
        holdout_vectors = load_upload(form["holdout_vectors"]) if "holdout_vectors" in form else None
        if case_base is None or shap_scores is None or holdout is None:
            raise ValueError("Upload all three exported E2R2 files.")
        prepare_loaded_data(case_base, shap_scores, holdout, case_vectors, holdout_vectors)
        self.redirect("/")

    def handle_settings(self) -> None:
        form = parse_post(self)
        api_key = form_get(form, "api_key", "")
        if api_key:
            STATE["api_key"] = api_key
        STATE["model"] = form_get(form, "model", STATE["model"])
        STATE["prediction_goal"] = form_get(form, "prediction_goal", "")
        STATE["include_baseline_signal"] = "include_baseline_signal" in form
        self.redirect("/")

    def handle_predict_one(self) -> None:
        form = parse_post(self)
        row_index = form_get(form, "row_index")
        row_index = int(row_index) if str(row_index).isdigit() else row_index
        neighbors = int(float(form_get(form, "neighbors", 5)))
        STATE["individual_result"] = run_prediction_for_row(row_index, neighbors)
        self.redirect("/")

    def handle_predict_batch(self) -> None:
        form = parse_post(self)
        offset = int(float(form_get(form, "offset", 0)))
        limit = int(float(form_get(form, "limit", 25)))
        neighbors = int(float(form_get(form, "neighbors", 5)))
        holdout = STATE["holdout"]
        row_indexes = holdout["row_index"].iloc[offset : offset + limit].tolist()
        results = []
        for row_index in row_indexes:
            results.append(run_prediction_for_row(row_index, neighbors))
        STATE["batch_results"] = pd.DataFrame(results)
        self.redirect("/")

    def handle_download(self, query: Dict[str, Any]) -> None:
        name = query.get("name", [""])[0]
        if name != "batch" or STATE.get("batch_results") is None:
            self.respond(404, page("<section><h2>No batch results to download.</h2></section>"))
            return
        payload = dataframe_to_csv_bytes(STATE["batch_results"])
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="e2r2_llm_batch_predictions.csv"')
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
            server = ThreadingHTTPServer((HOST, candidate), LLMExperimentHandler)
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError("No open local port was found for the LLM experiment app.")
    url = f"http://{HOST}:{port}"
    print(f"E2R2 LLM Experiment Lab running at {url}")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    run()
