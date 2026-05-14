from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from e2r2_pipeline import (
    apply_baseline_verification,
    baseline_verification_thresholds_from_holdout,
    baseline_verification_thresholds_from_training,
    build_case_base,
    build_e2r2_prompt,
    case_vectors_dataset,
    dataframe_to_csv_bytes,
    first_present_value,
    holdout_dataset,
    holdout_vectors_dataset,
    infer_llm_role,
    parse_data_dictionary,
    read_uploaded_table,
    retrieve_similar_cases,
    summarize_peer_alignment,
    train_e2r2_baseline,
    value_is_present,
)


OPENAI_LLM_MODELS = [
    ("gpt-5.4-mini", "GPT-5.4 mini - low cost, strong general reasoning"),
    ("gpt-5-mini", "GPT-5 mini - low cost, strong reasoning"),
    ("gpt-5-nano", "GPT-5 nano - lowest cost, light reasoning"),
    ("o3-mini", "o3-mini - moderate cost, dedicated reasoning"),
    ("o4-mini", "o4-mini - moderate cost, fast o-series reasoning"),
    ("gpt-4.1-mini", "GPT-4.1 mini - low cost, fastest baseline"),
]

RATIONALE_SIMPLIFIER_MODEL = "gpt-5-nano"

BASELINE_COLUMNS = {
    "baseline_predicted_outcome",
    "baseline_positive_probability",
    "baseline_confidence",
    "baseline_correct",
}


st.set_page_config(page_title="E2R2", layout="wide")


def uploaded_table(uploaded_file: Any) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    return read_uploaded_table(uploaded_file)


def visible_case_base(case_base: pd.DataFrame) -> pd.DataFrame:
    return case_base[[c for c in case_base.columns if not str(c).startswith("vector_")]]


def visible_prediction_results(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "row_index",
        "actual_outcome",
        "llm_predicted_outcome",
        "llm_confidence_level",
        "llm_rationale",
    ]
    available = [column for column in columns if column in df.columns]
    return df[available].copy() if available else df.copy()


def response_token_limit(model: str) -> int:
    if "pro" in model:
        return 6000
    if model in {"gpt-5-mini", "gpt-5-nano"}:
        return 3200
    if model.startswith("gpt-5") or model.startswith("o"):
        return 2400
    return 1200


def call_openai(api_key: str, model: str, prompt: str) -> str:
    if not api_key:
        raise ValueError("No OpenAI API key was provided.")
    payload: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": response_token_limit(model),
    }
    if not model.startswith("gpt-5") and not model.startswith("o"):
        payload["temperature"] = 0.2
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=90,
        )
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(
            "The OpenAI request timed out. Try a faster model such as gpt-4.1-mini or gpt-5-mini, "
            "reduce the number of retrieved cases, or rerun the request."
        ) from exc
    if not response.ok:
        detail = response.text
        try:
            detail = response.json().get("error", {}).get("message", response.text)
        except Exception:
            pass
        raise RuntimeError(f"OpenAI API request failed ({response.status_code}): {detail}")
    data = response.json()
    if data.get("output_text"):
        return data["output_text"]
    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(content.get("text", ""))
    text = "\n".join(chunks).strip()
    if text:
        return text

    status = data.get("status", "unknown")
    incomplete = data.get("incomplete_details") or {}
    error = data.get("error") or {}
    reason = incomplete.get("reason") or error.get("message") or "No visible text was returned."
    raise RuntimeError(
        f"The model returned no visible LLM output. Status: {status}. Reason: {reason}. "
        f"For pro reasoning models, try again, use fewer retrieved cases, or use gpt-4.1-mini / gpt-5-mini for interactive runs."
    )


def secret_api_key() -> str:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"])
        if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            return str(st.secrets["openai"]["api_key"])
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY", "")


def resolved_api_key(input_key: str) -> str:
    return input_key.strip() or secret_api_key()


def parse_json_response(text: str) -> Dict[str, Any]:
    try:
        return normalize_llm_fields(json.loads(text))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return parse_labeled_response(text)
        try:
            return normalize_llm_fields(json.loads(match.group(0)))
        except json.JSONDecodeError:
            return parse_labeled_response(text)


def normalized_field_name(key: Any) -> str:
    text = str(key).strip().lower()
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def clean_llm_scalar(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = value.strip().rstrip(",")
        cleaned = cleaned.strip().strip('"').strip("'").strip()
        return cleaned
    return value


def normalize_llm_fields(value: Any, prefix: str = "") -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    alias_map = {
        "predicted_outcome": "predicted_outcome",
        "prediction": "predicted_outcome",
        "predicted_class": "predicted_outcome",
        "outcome": "predicted_outcome",
        "verdict": "predicted_outcome",
        "class_label": "predicted_outcome",
        "final_predicted_outcome": "final_predicted_outcome",
        "final_prediction": "final_predicted_outcome",
        "final_predicted_class": "final_predicted_outcome",
        "final_outcome": "final_predicted_outcome",
        "final_verdict": "final_predicted_outcome",
        "llm_predicted_outcome": "final_predicted_outcome",
        "confidence_level": "confidence_level",
        "confidence": "confidence_level",
        "predicted_confidence": "confidence_level",
        "final_confidence_level": "final_confidence_level",
        "final_confidence": "final_confidence_level",
        "llm_confidence_level": "final_confidence_level",
        "rationale": "rationale",
        "reasoning": "rationale",
        "explanation": "rationale",
        "final_rationale": "final_rationale",
        "final_reasoning": "final_rationale",
        "final_explanation": "final_rationale",
        "llm_rationale": "final_rationale",
        "stage1_predicted_outcome": "stage1_predicted_outcome",
        "stage1_verdict": "stage1_predicted_outcome",
        "stage1_confidence_level": "stage1_confidence_level",
        "stage1_confidence": "stage1_confidence_level",
        "stage1_rationale": "stage1_rationale",
        "stage1_reasoning": "stage1_rationale",
        "agreement": "agreement",
        "baseline_p_positive": "baseline_p_positive",
        "verification_rule_applied": "verification_rule_applied",
        "verification_rule": "verification_rule_applied",
    }
    parsed: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = normalized_field_name(raw_key)
        full_key = f"{prefix}_{key}" if prefix else key
        target = alias_map.get(full_key, alias_map.get(key))
        if target and not isinstance(raw_value, (dict, list)):
            parsed[target] = clean_llm_scalar(raw_value)
        if isinstance(raw_value, dict):
            nested = normalize_llm_fields(raw_value, key)
            for nested_key, nested_value in nested.items():
                parsed.setdefault(nested_key, nested_value)
            nested_without_prefix = normalize_llm_fields(raw_value)
            for nested_key, nested_value in nested_without_prefix.items():
                parsed.setdefault(nested_key, nested_value)
    return parsed


def parse_labeled_response(text: str) -> Dict[str, Any]:
    alias_map = {
        "predicted_outcome": "predicted_outcome",
        "predicted outcome": "predicted_outcome",
        "prediction": "predicted_outcome",
        "predicted class": "predicted_outcome",
        "outcome": "predicted_outcome",
        "verdict": "predicted_outcome",
        "confidence_level": "confidence_level",
        "confidence level": "confidence_level",
        "confidence": "confidence_level",
        "rationale": "rationale",
        "reasoning": "rationale",
        "explanation": "rationale",
        "stage1_predicted_outcome": "stage1_predicted_outcome",
        "stage1 predicted outcome": "stage1_predicted_outcome",
        "stage1 verdict": "stage1_predicted_outcome",
        "stage1_confidence_level": "stage1_confidence_level",
        "stage1 confidence level": "stage1_confidence_level",
        "stage1 confidence": "stage1_confidence_level",
        "stage1_rationale": "stage1_rationale",
        "stage1 rationale": "stage1_rationale",
        "stage1 reasoning": "stage1_rationale",
        "agreement": "agreement",
        "baseline_p_positive": "baseline_p_positive",
        "baseline p positive": "baseline_p_positive",
        "verification_rule_applied": "verification_rule_applied",
        "verification rule applied": "verification_rule_applied",
        "verification rule": "verification_rule_applied",
        "final_predicted_outcome": "final_predicted_outcome",
        "final predicted outcome": "final_predicted_outcome",
        "final prediction": "final_predicted_outcome",
        "final predicted class": "final_predicted_outcome",
        "final outcome": "final_predicted_outcome",
        "final verdict": "final_predicted_outcome",
        "final_confidence_level": "final_confidence_level",
        "final confidence level": "final_confidence_level",
        "final confidence": "final_confidence_level",
        "final_rationale": "final_rationale",
        "final rationale": "final_rationale",
        "final reasoning": "final_rationale",
        "final explanation": "final_rationale",
    }
    parsed: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_key and current_key in parsed:
                parsed[current_key] = f"{parsed[current_key]}\n".rstrip()
            continue
        line = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line)
        line = line.replace("**", "").strip()
        match = re.match(r'^["\']?([A-Za-z0-9_ \-]+?)["\']?\s*[:=-]\s*(.*)$', line)
        if match:
            raw_key = match.group(1).strip().lower().replace("-", " ")
            key = alias_map.get(raw_key)
            if key:
                current_key = key
                parsed[key] = clean_llm_scalar(match.group(2))
                continue
        if current_key:
            previous = str(parsed.get(current_key, "")).rstrip()
            parsed[current_key] = f"{previous} {line}".strip() if previous else line
    if "agreement" in parsed:
        value = str(parsed["agreement"]).strip().lower()
        if value in {"true", "false"}:
            parsed["agreement"] = value == "true"
    if "baseline_p_positive" in parsed:
        try:
            parsed["baseline_p_positive"] = float(str(parsed["baseline_p_positive"]).strip())
        except ValueError:
            pass
    return parsed


def extracted_prediction_fields(parsed: Dict[str, Any]) -> Tuple[Any, Any, str]:
    predicted = first_present_value(
        parsed.get("final_predicted_outcome"),
        parsed.get("predicted_outcome"),
        parsed.get("stage1_predicted_outcome"),
    )
    confidence = first_present_value(
        parsed.get("final_confidence_level"),
        parsed.get("confidence_level"),
        parsed.get("stage1_confidence_level"),
    )
    rationale = first_present_value(
        parsed.get("final_rationale"),
        parsed.get("rationale"),
        parsed.get("stage1_rationale"),
    )
    return clean_llm_scalar(predicted), clean_llm_scalar(confidence), str(clean_llm_scalar(rationale) or "")


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
    return call_openai(api_key, RATIONALE_SIMPLIFIER_MODEL, simplify_prompt).strip()


def model_select(key: str, default: str = "gpt-4.1-mini") -> str:
    ids = [value for value, _ in OPENAI_LLM_MODELS]
    index = ids.index(default) if default in ids else 0
    selected = st.selectbox(
        "LLM model",
        OPENAI_LLM_MODELS,
        index=index,
        key=key,
        format_func=lambda item: f"{item[1]} ({item[0]})",
    )
    return selected[0]


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


def original_feature_for_encoded_column(
    encoded_column: str,
    feature_columns: List[str],
    categorical_columns: List[str],
) -> str:
    name = str(encoded_column)
    if name in feature_columns:
        return name
    for column in sorted(categorical_columns, key=len, reverse=True):
        if name == column or name.startswith(f"{column}_"):
            return column
    return name


def shap_similarity_weights(
    shap_scores: pd.DataFrame,
    encoded_columns: List[str],
    feature_columns: List[str],
    categorical_columns: List[str],
) -> Tuple[np.ndarray, str]:
    if shap_scores is None or "feature" not in shap_scores.columns or "abs_shap_score" not in shap_scores.columns:
        return np.ones(len(encoded_columns), dtype=float), "reconstructed unweighted features"
    importance_frame = shap_scores[["feature", "abs_shap_score"]].copy()
    importance_frame["feature"] = importance_frame["feature"].astype(str)
    importance_frame["abs_shap_score"] = pd.to_numeric(importance_frame["abs_shap_score"], errors="coerce")
    importance = (
        importance_frame
        .groupby("feature")["abs_shap_score"]
        .mean()
        .to_dict()
    )
    raw = np.asarray(
        [
            float(importance.get(original_feature_for_encoded_column(col, feature_columns, categorical_columns), 0.0))
            for col in encoded_columns
        ],
        dtype=float,
    )
    raw = np.nan_to_num(np.abs(raw), nan=0.0, posinf=0.0, neginf=0.0)
    if float(raw.sum()) <= 0:
        return np.ones(len(encoded_columns), dtype=float), "reconstructed unweighted features"
    weights = np.sqrt(raw / max(float(raw.mean()), 1e-12))
    weights = np.clip(weights, 0.25, 4.0)
    return weights.astype(float), "SHAP-weighted reconstructed features"


def build_retrieval_vectors(
    holdout: pd.DataFrame,
    case_base: pd.DataFrame,
    feature_columns: List[str],
    shap_scores: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, str]:
    features = holdout[feature_columns].copy()
    numeric_cols = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    categorical_cols = [c for c in features.columns if c not in numeric_cols]

    numeric = features[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=features.index)
    for col in numeric.columns:
        numeric[col] = numeric[col].fillna(numeric[col].median())
        std = numeric[col].std()
        numeric[col] = 0 if not std or pd.isna(std) else (numeric[col] - numeric[col].mean()) / std
    categorical = (
        pd.get_dummies(features[categorical_cols].fillna("__missing__").astype(str), dummy_na=False)
        if categorical_cols
        else pd.DataFrame(index=features.index)
    )
    encoded = pd.concat([numeric, categorical], axis=1)
    weights, weighting_label = shap_similarity_weights(
        shap_scores,
        [str(c) for c in encoded.columns.tolist()],
        feature_columns,
        categorical_cols,
    )
    matrix = encoded.to_numpy(dtype=float)
    if len(weights) == matrix.shape[1]:
        matrix = matrix * weights
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    holdout_vectors = matrix / norms
    row_to_position = {row_index: pos for pos, row_index in enumerate(holdout["row_index"].tolist())}
    case_positions = [row_to_position[idx] for idx in case_base["row_index"].tolist()]
    return holdout_vectors, holdout_vectors[case_positions], weighting_label


def vector_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith("vector_")]


def exact_vectors(
    holdout: pd.DataFrame,
    case_base: pd.DataFrame,
    holdout_vectors_df: Optional[pd.DataFrame],
    case_vectors_df: Optional[pd.DataFrame],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if holdout_vectors_df is None or case_vectors_df is None:
        return None
    h_cols = vector_columns(holdout_vectors_df)
    c_cols = vector_columns(case_vectors_df)
    if not h_cols or h_cols != c_cols:
        raise ValueError("Vector files must contain matching vector_ columns.")
    holdout_joined = holdout[["row_index"]].merge(
        holdout_vectors_df[["row_index"] + h_cols], on="row_index", how="left"
    )
    case_joined = case_base[["case_id", "row_index"]].merge(
        case_vectors_df[["case_id", "row_index"] + c_cols], on=["case_id", "row_index"], how="left"
    )
    if holdout_joined[h_cols].isna().any().any() or case_joined[c_cols].isna().any().any():
        raise ValueError("The vector files do not match the uploaded case base and holdout rows.")
    return holdout_joined[h_cols].to_numpy(dtype=float), case_joined[c_cols].to_numpy(dtype=float)


def lab_retrieve(
    row_index: Any,
    case_base: pd.DataFrame,
    holdout: pd.DataFrame,
    holdout_vectors: np.ndarray,
    case_vectors: np.ndarray,
    k: int,
) -> pd.DataFrame:
    row_to_position = {value: pos for pos, value in enumerate(holdout["row_index"].tolist())}
    vector = holdout_vectors[row_to_position[row_index]]
    retrieved = case_base.copy()
    retrieved["similarity"] = case_vectors @ vector
    retrieved = retrieved[retrieved["row_index"] != row_index]
    return retrieved.sort_values("similarity", ascending=False).head(k).reset_index(drop=True)


def lab_feature_definition(shap_scores: pd.DataFrame, feature: str) -> str:
    if "definition" not in shap_scores.columns:
        return ""
    matches = shap_scores.loc[shap_scores["feature"].astype(str) == str(feature), "definition"].dropna()
    return "" if matches.empty else str(matches.iloc[0])


def normalize_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.lower()


def infer_positive_outcome_label(values: List[Any], prediction_goal: str) -> str:
    labels = []
    seen = set()
    for value in values:
        raw = str(value)
        normalized = normalize_label(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        labels.append((raw, normalized))
    if not labels:
        return ""
    for raw, normalized in labels:
        if normalized == "1":
            return raw
    numeric = []
    for raw, normalized in labels:
        try:
            numeric.append((raw, float(normalized)))
        except ValueError:
            pass
    if len(numeric) == len(labels) and numeric:
        return max(numeric, key=lambda item: item[1])[0]
    risk_terms = ["attrition", "churn", "default", "fraud", "readmission", "dropout", "risk", "positive", "yes", "true"]
    goal_text = normalize_label(prediction_goal)
    for raw, normalized in labels:
        if any(term in normalized for term in risk_terms):
            return raw
        if normalized and normalized in goal_text:
            return raw
    safe_terms = ["retained", "stay", "persist", "negative", "no", "false", "safe", "continue"]
    safe_norms = {normalized for _, normalized in labels if any(term in normalized for term in safe_terms)}
    if len(labels) == 2 and len(safe_norms) == 1:
        for raw, normalized in labels:
            if normalized not in safe_norms:
                return raw
    return labels[-1][0]


def positive_class_specification(holdout: pd.DataFrame, target_column: str, prediction_goal: str) -> Tuple[str, str]:
    values: List[Any] = []
    if target_column and target_column in holdout.columns:
        values = holdout[target_column].dropna().astype(str).unique().tolist()
    elif "baseline_predicted_outcome" in holdout.columns:
        values = holdout["baseline_predicted_outcome"].dropna().astype(str).unique().tolist()
    label = infer_positive_outcome_label(values, prediction_goal or target_column) if values else ""
    description = prediction_goal or (f"cases with outcome label {label}" if label else "the positive/risk outcome")
    return str(label), description


def lab_shap_alignment(row: pd.Series, retrieved: pd.DataFrame, shap_scores: pd.DataFrame) -> str:
    relevant = shap_scores[shap_scores["case_id"].isin(retrieved["case_id"])].copy()
    rows = []
    for feature, group in relevant.groupby("feature"):
        mean_shap = group["shap_score"].mean()
        mean_abs = group["abs_shap_score"].mean()
        positive_count = int((group["shap_score"] > 0).sum())
        negative_count = int((group["shap_score"] < 0).sum())
        peer_values = group["value"].dropna().astype(str).value_counts().head(3)
        holdout_value = row.get(feature, "")
        matching_count = int((group["value"].astype(str) == str(holdout_value)).sum())
        rows.append(
            {
                "mean_abs": mean_abs,
                "line": (
                    f"- {feature}: holdout={holdout_value}; peer values="
                    f"{', '.join([f'{value} ({count})' for value, count in peer_values.items()])}; "
                    f"mean SHAP={mean_shap:.4f}, mean |SHAP|={mean_abs:.4f}; "
                    f"sign counts +/{positive_count}, -/{negative_count}; exact matches={matching_count}"
                ),
            }
        )
    rows = sorted(rows, key=lambda item: item["mean_abs"], reverse=True)[:10]
    return "\n".join(item["line"] for item in rows) or "- No SHAP alignment rows were available."


def lab_prompt(
    row: pd.Series,
    holdout: pd.DataFrame,
    retrieved: pd.DataFrame,
    shap_scores: pd.DataFrame,
    target_column: str,
    prediction_goal: str,
    neighbors: int,
) -> str:
    role = infer_llm_role(prediction_goal, target_column)
    positive_class_label, positive_class_description = positive_class_specification(
        holdout,
        target_column,
        prediction_goal,
    )
    excluded = set(BASELINE_COLUMNS) | {"row_index", target_column}
    relevant = shap_scores[shap_scores["case_id"].isin(retrieved["case_id"])].copy()
    if relevant.empty:
        prompt_features = [c for c in row.index if c not in excluded][:20]
    else:
        prompt_features = (
            relevant.groupby("feature", as_index=False)["abs_shap_score"]
            .mean()
            .sort_values("abs_shap_score", ascending=False)
            .head(20)["feature"]
            .tolist()
        )
    feature_lines = []
    for feature in prompt_features:
        if feature in excluded or feature not in row.index:
            continue
        value = row.get(feature)
        definition = lab_feature_definition(shap_scores, feature)
        suffix = f" ({definition})" if definition else ""
        feature_lines.append(f"- {feature}{suffix}: {value}")

    peer_blocks = []
    for _, peer in retrieved.iterrows():
        peer_shap = shap_scores[shap_scores["case_id"] == peer["case_id"]].sort_values("rank").head(5)
        units = (
            peer_shap["shap_output_units"].dropna().iloc[0]
            if "shap_output_units" in peer_shap.columns and not peer_shap["shap_output_units"].dropna().empty
            else "probability"
        )
        shap_lines = [
            f"  - {item.feature}: value={item.value}, SHAP contribution to positive-class probability={item.shap_score:.4f} ({units})"
            for item in peer_shap.itertuples()
        ]
        peer_blocks.append(
            "\n".join(
                [
                    f"Case {peer.case_id}: outcome={peer.predicted_outcome}, "
                    f"retrieval similarity={peer.similarity:.3f}",
                    *shap_lines,
                ]
            )
        )

    outcome_counts = retrieved["predicted_outcome"].astype(str).value_counts()
    n_pos = (
        int((retrieved["predicted_outcome"].astype(str) == str(positive_class_label)).sum())
        if positive_class_label
        else 0
    )
    n_total = int(len(retrieved))
    pos_share = (100.0 * n_pos / n_total) if n_total else 0.0
    top_similarity = float(retrieved["similarity"].max()) if not retrieved.empty else 0.0
    neighbor_count_lines = [
        f"- {outcome}: {count} retrieved neighbors"
        for outcome, count in outcome_counts.items()
    ]
    precedent_counts_block = "\n".join(
        [
            "Precedent outcome counts:",
            f"- N={n_total}",
            f"- n_pos={n_pos}",
            f"- pos_share={pos_share:.1f}%",
            *neighbor_count_lines,
            f"- top retrieval similarity={top_similarity:.3f}",
        ]
    )
    baseline_predicted_outcome = row.get("baseline_predicted_outcome", "")
    baseline_confidence = row.get("baseline_confidence", "")
    thresholds = baseline_verification_thresholds_from_holdout(
        holdout, target_column, positive_class_label
    )
    positive_override_threshold = thresholds["positive_override_threshold"]
    negative_override_threshold = thresholds["negative_override_threshold"]

    return f"""{role}

You are using the E2R2 framework: Explain, Embed, Retrieve, and Reason.
E2R2 combines SHAP feature attributions, similarity-based precedent retrieval, and professional reasoning. The retrieved precedents come from a case base of correctly predicted, high-confidence holdout cases. Your job is to decide the outcome using precedent-based evidence, then verify that decision against an independent baseline machine-learning model's prediction.

Prediction goal:
{prediction_goal or f"Predict {target_column}."}

Positive class specification (CRITICAL — read this before any reasoning):
The positive class label for this task is provided explicitly below as positive_class_label. Use this label exactly as given. Do not infer the positive class from the retrieved precedent outcomes or from any other signal in the prompt. In particular:
- The positive class is NOT "whichever label the retrieved precedents happen to share." High-confidence retrieval can return all-one-class neighbors for either class, and that does not change which label is positive.
- A precedent outcome value equal to positive_class_label means that precedent is a positive-class case. A precedent outcome value not equal to positive_class_label means that precedent is a negative-class case.
- "pos_share" in the decision rule is the share of retrieved precedents whose outcome equals positive_class_label, regardless of which label that happens to be (0, 1, "churn", "default", etc.).
- All SHAP values are explained against probability of positive_class_label, as stated in the SHAP interpretation section below.

positive_class_label: {positive_class_label}
positive_class_description: {positive_class_description}

Reasoning workflow:
You will produce the final verdict in two internal stages. Complete Stage 1 fully before considering Stage 2. Do not allow knowledge of the baseline model's prediction (shown in the Stage 2 inputs) to influence your Stage 1 reasoning.

=================================================================
STAGE 1 — PRECEDENT-BASED REASONING
=================================================================

Reasoning guidance:
Use your best professional judgment to predict the outcome from the evidence shown below: the holdout case profile, the retrieved precedent cases, their outcomes, their similarity scores, and their top SHAP predictors.

Detailed comparison task:
- Identify the most important predictors that recur across the retrieved precedents' top 5 SHAP predictors.
- Start from the recurring top-predictor patterns, then compare the holdout case against the precedents predictor by predictor using the exact values shown when helpful.
- Explain whether the holdout case matches the precedent pattern, differs in a meaningful way, or shows mixed evidence on those predictors.
- Highlight critical differences on the strongest predictors, especially when the holdout case shows a protective pattern where positive precedents showed risk, or vice versa.
- If several predictors reflect the same underlying issue, treat them as one cluster rather than counting them as separate concerns.
- Use that comparison, together with precedent outcomes and similarity scores, to reach the Stage 1 verdict — following the decision rule below.

Decision rule (apply before writing the Stage 1 rationale):
1. Let N be the total number of retrieved precedents and let n_pos be the count of those precedents whose outcome equals positive_class_label (as specified at the top of the prompt — do not re-infer this from the precedent outcomes). Compute pos_share = n_pos / N, expressed as a percentage. If all retrieved precedents share an outcome that is not equal to positive_class_label, then n_pos = 0 and pos_share = 0%; this is a strong negative signal, not a positive one.
2. If pos_share >= 60%, the default Stage 1 verdict is positive. You may only override this default to a negative verdict if ALL of the following are true:
   (a) The holdout case shows a clearly protective pattern on the strongest recurring SHAP predictor (the one with the largest mean |SHAP|) — meaning the holdout value crosses into a clearly healthy range on that variable, not merely a relatively-better range than the precedent values.
   (b) The holdout also shows a clearly protective pattern on at least one other top-3 recurring predictor.
   (c) The holdout does not exhibit any late-stage disengagement or acute-risk signal of the kind described in the structural-risk checklist below.
   If you override the default, state the override explicitly in the Stage 1 rationale and name each condition you relied on.
3. If pos_share <= 20%, the default Stage 1 verdict is negative. You may only override to a positive verdict if the holdout shows a clear, high-leverage leading indicator of the positive outcome (e.g., a behavior, status, or value that is widely understood in the domain as a near-certain precursor of the outcome — see the structural-risk checklist below for how to recognize one).
4. If pos_share is between 20% and 60% (exclusive of both bounds), treat as borderline and resolve using the structural-risk checklist below.

Borderline structural-risk checklist (apply only when pos_share is in the borderline band between 20% and 60% exclusive, OR when pos_share is between 60% and 80% inclusive, OR when the top retrieval similarity across the retrieved neighbors is below 0.78):

A "structural-risk marker" is a predictor value in the holdout case that, in domain knowledge or in the recurring SHAP pattern of the case base, is a strong leading indicator of the positive outcome on its own. Identify markers using the following heuristics in order:
- Any predictor that appears in the top 5 SHAP list of at least 80% of the retrieved precedents, where the holdout's value matches the precedent cluster's risk-direction value (e.g., the same categorical level, or a numeric value in the same risk-side range).
- Any predictor in the holdout case that, by standard domain understanding, indicates an actively unfolding adverse event, withdrawal of support, missed obligation, or break in a normally-continuous process. These are typically the highest-leverage markers even when not present in the retrieved SHAP lists.
- Any predictor whose value sits past a well-understood absolute risk threshold for the domain (e.g., a performance or health metric below a level commonly treated as concerning), regardless of how the holdout compares to the precedent values.

Count how many such markers are present in the holdout. If 2 or more markers are present, lean toward the positive verdict. If 0 or 1 markers are present, lean toward the negative verdict. Name explicitly in the Stage 1 rationale which markers triggered, which heuristic identified each, and which did not trigger.

SHAP interpretation:
- All SHAP values are explained against the probability of positive_class_label (as specified at the top of the prompt), not against each precedent's own outcome.
- A positive SHAP contribution increases the modeled probability of positive_class_label; a negative SHAP contribution decreases it.
- Larger absolute SHAP values indicate stronger model contributions.

Cautions when comparing holdout values to precedent values:
- A holdout value that is better than the precedent cluster but still in an absolute risk zone is NOT protective. For example, if the precedents have very extreme risk-side values on a key predictor and the holdout has a moderately-less-extreme but still risk-side value, treat this as a milder version of the same risk pattern, not as a protective signal.
- Only call a difference "protective" when the holdout value crosses into a clearly healthy or low-risk range on that variable by standard domain understanding, not merely when it is relatively better than the neighbors. State the healthy range you are using when you invoke protection.
- Late-stage or behavioral signals (e.g., a discontinuation, a missed step in a normally-continuous process, a withdrawal of support, an acute event) typically outweigh upstream profile differences. If such a signal is present in the holdout, it should generally not be overridden by a more favorable value on an upstream predictor.

Stage 1 confidence calibration (apply after determining the Stage 1 verdict):
- High: pos_share >= 80% AND top retrieval similarity >= 0.85 AND the holdout matches the precedent cluster on the strongest recurring SHAP predictor's absolute range (not just its direction). OR, for a negative verdict: pos_share = 0% AND no structural-risk-checklist markers are present AND top similarity >= 0.80.
- Moderate: precedents and structural signals point the same direction, but one of the high-confidence conditions above is missing (e.g., similarity is lower, or the holdout's value on the strongest predictor sits in a softer-but-still-aligned range).
- Low: precedent signal and structural-risk checklist disagree, OR top similarity < 0.75, OR pos_share is in the borderline band (between 20% and 60% exclusive) with mixed feature evidence, OR the decision rule above required an explicit override.

At the end of Stage 1, internally fix three values before reading any Stage 2 input:
- stage1_verdict (the positive or negative class label)
- stage1_confidence (low | moderate | high)
- stage1_rationale (a detailed paragraph as described in the JSON schema below)

=================================================================
STAGE 2 — BASELINE-MODEL VERIFICATION
=================================================================

Now consult the baseline machine-learning model's prediction shown in the Stage 2 inputs. The baseline model was trained independently of the precedent retrieval and SHAP attribution pipeline, so its prediction is a largely independent signal. When both signals agree, the Stage 1 verdict is reliable. When they disagree, the resolution depends on how confident the baseline is.

Critical clarifications before applying the rules:
- baseline_predicted_outcome is a HARD CLASS LABEL (e.g., 0 or 1, "positive" or "negative"). It is not a probability. Use it only as a label.
- baseline_confidence is a number between 0.50 and 1.00 that expresses how confident the baseline is in the class it predicted. A baseline_confidence of 0.5163 still means the baseline predicts its stated class — just weakly. It does NOT mean the baseline is undecided between classes.
- "Agreement" is a strict equality check between two labels. It does not depend on any probability or confidence value. baseline_confidence near 0.50 does NOT make agreement false.

Step 1 — Determine agreement.
Compare two values:
- stage1_verdict (the class label you fixed at the end of Stage 1; e.g., 1)
- baseline_predicted_outcome (the class label provided in the Stage 2 input; e.g., 1)

If these two labels are identical, then agreement = TRUE. If they are different, agreement = FALSE. Do not consult baseline_confidence or any other value when determining agreement.

Worked examples:
- stage1_verdict = 1, baseline_predicted_outcome = 1, baseline_confidence = 0.52 → agreement = TRUE (both predict the same class; the low confidence does not change this).
- stage1_verdict = 1, baseline_predicted_outcome = 1, baseline_confidence = 0.99 → agreement = TRUE.
- stage1_verdict = 1, baseline_predicted_outcome = 0, baseline_confidence = 0.55 → agreement = FALSE.
- stage1_verdict = 0, baseline_predicted_outcome = 1, baseline_confidence = 0.80 → agreement = FALSE.

Step 2 — Compute baseline_p_positive (for output transparency only; do NOT use this in the verification rules).
- If baseline_predicted_outcome equals positive_class_label, baseline_p_positive = baseline_confidence.
- If baseline_predicted_outcome does not equal positive_class_label, baseline_p_positive = 1 - baseline_confidence.

Step 3 — Apply the verification rules in order. Stop at the first rule that fires.

Rule A — Agreement. If agreement is TRUE (as determined in Step 1), keep the Stage 1 verdict and confidence unchanged. final_verdict = stage1_verdict. final_confidence = stage1_confidence. Do not consider baseline_confidence in this rule — agreement alone fires Rule A regardless of how confident or unconfident the baseline is.

Rule B — Dataset-calibrated baseline override. If agreement is FALSE and baseline_predicted_outcome equals positive_class_label, override to the positive class only when baseline_confidence is greater than the average baseline confidence among true positive cases in the holdout set minus 0.10. If agreement is FALSE and baseline_predicted_outcome does not equal positive_class_label, override to the negative class only when baseline_confidence is greater than the average baseline confidence among true negative cases in the holdout set. final_confidence = moderate under override because two independent signals disagreed.

Rule C — Dataset-calibrated keep Stage 1. If agreement is FALSE and baseline_confidence does not exceed the relevant true-positive or true-negative threshold, keep the Stage 1 verdict. Cap final_confidence at moderate: downgrade high to moderate, and leave moderate or low unchanged.

Do not invoke any rule other than the three above. Do not override based on the Stage 1 rationale text or the Stage 1 confidence — only the agreement check, baseline_predicted_outcome, baseline_confidence, and the dataset-calibrated threshold for the baseline's predicted class drive the verification.

Final rationale composition:
- If Rule A fired, the final rationale is the Stage 1 rationale, with a one-sentence note that the baseline model agreed (state baseline_confidence). The note should affirm agreement even if baseline_confidence is low; a low-confidence agreement is still agreement.
- If Rule B fired, the final rationale should keep the Stage 1 evidence summary but append a sentence explaining that the baseline model disagreed, exceeded the dataset-calibrated threshold for its predicted class, and that the verification step adopted the baseline verdict.
- If Rule C fired, the final rationale is the Stage 1 rationale, with a one-sentence note that the baseline model disagreed but did not exceed the dataset-calibrated threshold for its predicted class, so it was not strong enough to override.

=================================================================
INPUTS
=================================================================

Holdout case profile:
{chr(10).join(feature_lines)}

Retrieved E2R2 precedent cases:
{chr(10).join(peer_blocks)}

Recurring top predictors across retrieved precedents:
{lab_recurring_top_predictors(row, retrieved, shap_scores, per_case_top_k=5, top_n=5)}

SHAP-weighted alignment summary across retrieved cases:
{lab_shap_alignment(row, retrieved, shap_scores)}

{precedent_counts_block}

Baseline model output:
- baseline_predicted_outcome: {baseline_predicted_outcome}
- baseline_confidence: {float(baseline_confidence):.4f}

Dataset-calibrated baseline override thresholds:
- positive_override_threshold_true_positive_mean_minus_0_10: {positive_override_threshold:.4f}
- negative_override_threshold_true_negative_mean: {negative_override_threshold:.4f}

=================================================================
OUTPUT
=================================================================

Return this JSON object:
{{
  "stage1_predicted_outcome": "one of the domain labels",
  "stage1_confidence_level": "low | moderate | high",
  "stage1_rationale": "one detailed professional paragraph that (a) compares the holdout case with the retrieved precedents on the recurring top SHAP predictors, (b) states N, n_pos, and pos_share, and names which branch of the Stage 1 decision rule applied, (c) lists which structural-risk-checklist markers were checked and which triggered when applicable, and (d) explains how that combined evidence led to the Stage 1 verdict",
  "agreement": true | false,
  "baseline_p_positive": <numeric value between 0 and 1>,
  "verification_rule_applied": "A | B | C",
  "final_predicted_outcome": "one of the domain labels",
  "final_confidence_level": "low | moderate | high",
  "final_rationale": "the Stage 1 rationale plus the appropriate verification note as described in the Final rationale composition section"
}}

Base the rationale on the evidence shown in the prompt. The true outcome is not provided for reasoning.
"""


def lab_recurring_top_predictors(
    row: pd.Series,
    retrieved: pd.DataFrame,
    shap_scores: pd.DataFrame,
    per_case_top_k: int = 5,
    top_n: int = 5,
) -> str:
    relevant = shap_scores[shap_scores["case_id"].isin(retrieved["case_id"])].copy()
    if "rank" in relevant.columns:
        relevant = relevant[relevant["rank"] <= per_case_top_k].copy()
    if relevant.empty:
        return "- No recurring top-predictor rows were available."
    total_cases = max(int(retrieved["case_id"].nunique()), 1)
    rows = []
    for feature, group in relevant.groupby("feature"):
        holdout_value = row.get(feature, "")
        peer_values = group["value"].dropna().astype(str).value_counts().head(3)
        mean_abs = float(group["abs_shap_score"].mean())
        mean_shap = float(group["shap_score"].mean())
        case_count = int(group["case_id"].nunique())
        if mean_shap > 0:
            direction = "usually increases positive-class probability"
        elif mean_shap < 0:
            direction = "usually decreases positive-class probability"
        else:
            direction = "has mixed overall direction"
        definition = lab_feature_definition(shap_scores, feature)
        suffix = f" ({definition})" if definition else ""
        rows.append(
            {
                "case_count": case_count,
                "mean_abs": mean_abs,
                "line": (
                    f"- {feature}{suffix}: appears in {case_count}/{total_cases} retrieved cases' top {per_case_top_k} predictors; "
                    f"holdout={holdout_value}; precedent values={', '.join([f'{value} ({count})' for value, count in peer_values.items()])}; "
                    f"mean |SHAP|={mean_abs:.4f}; {direction}"
                ),
            }
        )
    rows = sorted(rows, key=lambda item: (item["case_count"], item["mean_abs"]), reverse=True)[:top_n]
    return "\n".join(item["line"] for item in rows)


def run_lab_prediction(
    row_index: Any,
    lab: Dict[str, Any],
    api_key: str,
    model: str,
    prediction_goal: str,
    neighbors: int,
) -> Dict[str, Any]:
    holdout = lab["holdout"]
    row = holdout.loc[holdout["row_index"] == row_index].iloc[0]
    retrieved = lab_retrieve(
        row_index, lab["case_base"], holdout, lab["holdout_vectors"], lab["case_vectors"], neighbors
    )
    prompt = lab_prompt(
        row,
        holdout,
        retrieved,
        lab["shap_scores"],
        lab["target_column"],
        prediction_goal,
        neighbors,
    )
    raw = call_openai(api_key, model, prompt)
    parsed = parse_json_response(raw)
    positive_label, _ = positive_class_specification(holdout, lab["target_column"], prediction_goal)
    thresholds = baseline_verification_thresholds_from_holdout(holdout, lab["target_column"], positive_label)
    parsed = apply_baseline_verification(
        parsed,
        baseline_predicted_outcome=row.get("baseline_predicted_outcome", ""),
        baseline_confidence=row.get("baseline_confidence", ""),
        positive_class_label=positive_label,
        **thresholds,
    )
    final_predicted, final_confidence, final_rationale = extracted_prediction_fields(parsed)
    return {
        "row_index": row_index,
        "actual_outcome": row.get(lab["target_column"], ""),
        "baseline_predicted_outcome": row.get("baseline_predicted_outcome", ""),
        "baseline_confidence": row.get("baseline_confidence", ""),
        "baseline_signal_in_prompt": True,
        "stage1_predicted_outcome": parsed.get("stage1_predicted_outcome", ""),
        "stage1_confidence_level": parsed.get("stage1_confidence_level", ""),
        "stage1_rationale": parsed.get("stage1_rationale", ""),
        "agreement": parsed.get("agreement", ""),
        "baseline_p_positive": parsed.get("baseline_p_positive", ""),
        "baseline_positive_override_threshold": parsed.get("baseline_positive_override_threshold", ""),
        "baseline_negative_override_threshold": parsed.get("baseline_negative_override_threshold", ""),
        "baseline_override_threshold_used": parsed.get("baseline_override_threshold_used", ""),
        "verification_rule_applied": parsed.get("verification_rule_applied", ""),
        "llm_predicted_outcome": final_predicted,
        "llm_confidence_level": final_confidence,
        "llm_rationale": final_rationale,
        "retrieval_mode": lab["retrieval_mode"],
        "retrieved_case_ids": ", ".join(retrieved["case_id"].astype(str).tolist()),
        "prompt": prompt,
        "llm_raw_response": raw,
    }


def main_pipeline_tab() -> None:
    st.header("Full E2R2 Pipeline")
    st.caption("Train the baseline model, build the SHAP-informed case base, and export experiment files.")

    dataset_file = st.file_uploader("Raw dataset", type=["csv", "xlsx", "xls"], key="main_dataset")
    dictionary_file = st.file_uploader("Data dictionary", type=["csv", "xlsx", "xls"], key="main_dictionary")
    prediction_goal = st.text_area(
        "Prediction goal",
        value=st.session_state.get("main_prediction_goal", ""),
        key="main_prediction_goal",
    )

    if dataset_file is None:
        st.info("Upload a raw dataset to begin.")
        return

    df = uploaded_table(dataset_file)
    dictionary_df = uploaded_table(dictionary_file)
    data_dictionary = parse_data_dictionary(dictionary_df)
    st.dataframe(df.head(10), width="stretch")

    columns = list(df.columns)
    target_column = st.selectbox("Outcome column", columns, index=len(columns) - 1)
    values = sorted(df[target_column].dropna().astype(str).unique().tolist())
    positive_label = st.selectbox("Positive outcome", values, index=max(len(values) - 1, 0))
    ignored = st.multiselect("Ignore columns", [c for c in columns if c != target_column])

    c1, c2, c3, c4 = st.columns(4)
    test_size = c1.number_input("Holdout share", 0.1, 0.5, 0.25, 0.05)
    confidence_threshold = c2.number_input("Case confidence", 0.5, 0.99, 0.9, 0.01)
    minimum_cases = c3.number_input("Minimum cases", 5, 5000, 25, 5)
    top_predictors = c4.number_input("Top predictors", 3, 20, 8, 1)

    c5, c6, c7 = st.columns(3)
    attribution_method = c5.selectbox(
        "Attribution method",
        ["actual_shap", "fast_probability_perturbation"],
        format_func=lambda x: "Actual SHAP in probability units" if x == "actual_shap" else "Fast approximation",
    )
    shap_background_size = c6.number_input("SHAP background cases", 25, 1000, 100, 25)
    shap_kernel_nsamples = c7.number_input("Kernel SHAP samples", 100, 5000, 100, 100)

    if st.button("Run E2R2 pipeline", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        def cb(stage: str, percent: float, active: bool = True) -> None:
            status.write(stage)
            progress.progress(min(100, max(0, int(percent))))

        training = train_e2r2_baseline(
            df,
            target_column=target_column,
            positive_label=positive_label,
            ignored_columns=ignored,
            test_size=test_size,
            progress_callback=cb,
        )
        case_base, shap_table = build_case_base(
            training,
            data_dictionary=data_dictionary,
            confidence_threshold=confidence_threshold,
            minimum_cases=int(minimum_cases),
            top_predictors=int(top_predictors),
            attribution_method=attribution_method,
            shap_background_size=int(shap_background_size),
            shap_kernel_nsamples=int(shap_kernel_nsamples),
            progress_callback=cb,
        )
        cb("Complete: results are ready", 100)
        st.session_state["main_results"] = {
            "training": training,
            "case_base": case_base,
            "shap_table": shap_table,
            "prediction_goal": prediction_goal,
            "data_dictionary": data_dictionary,
        }

    results = st.session_state.get("main_results")
    if not results:
        return

    training = results["training"]
    case_base = results["case_base"]
    shap_table = results["shap_table"]
    st.subheader("Baseline Model")
    st.write(f"Selected model: **{training.best_run.name}**")
    retrieval_label = getattr(training, "similarity_weighting", "equal-weight cosine retrieval")
    st.caption(
        f"Retrieval uses {retrieval_label}. All encoded predictor features receive equal "
        "weight when the cosine similarity score is calculated."
    )
    st.dataframe(pd.DataFrame([training.best_run.metrics]), width="stretch")

    st.subheader("Downloads")
    downloads = [
        ("Case base", "e2r2_case_base.csv", visible_case_base(case_base)),
        ("SHAP scores", "e2r2_shap_scores.csv", shap_table),
        ("Holdout dataset", "e2r2_holdout_dataset.csv", holdout_dataset(training)),
        ("Case vectors", "e2r2_case_vectors.csv", case_vectors_dataset(case_base)),
        ("Holdout vectors", "e2r2_holdout_vectors.csv", holdout_vectors_dataset(training)),
    ]
    cols = st.columns(5)
    for col, (label, filename, table) in zip(cols, downloads):
        col.download_button(label, dataframe_to_csv_bytes(table), filename, "text/csv")

    st.subheader("Holdout Prediction")
    row_index = st.selectbox("Holdout row", training.X_test.index.tolist(), key="main_holdout_row")
    neighbors = st.number_input("Retrieved cases", 1, 12, 5, key="main_neighbors")
    api_key = st.text_input("OpenAI API key", type="password", key="main_api_key")
    model = model_select("main_model", default="gpt-4.1-mini")
    if "pro" in model:
        st.info("Pro models can take several minutes and may be unreliable for interactive Streamlit runs. For quick experiments, use gpt-4.1-mini, gpt-5-mini, or gpt-5.4-mini.")
    if st.button("Predict selected holdout case"):
        holdout_row = training.X_test.loc[row_index]
        retrieved = retrieve_similar_cases(training, case_base, holdout_row, k=int(neighbors))
        alignment = summarize_peer_alignment(holdout_row, retrieved, shap_table, training)
        prompt = build_e2r2_prompt(
            training,
            holdout_row,
            retrieved,
            shap_table,
            alignment,
            data_dictionary=results["data_dictionary"],
            prediction_goal=results["prediction_goal"],
        )
        st.text_area("Prepared prompt", prompt, height=320)
        key = resolved_api_key(api_key)
        if not key:
            st.warning("Enter an OpenAI API key, or add OPENAI_API_KEY to Streamlit secrets, to generate the LLM response.")
        else:
            try:
                with st.spinner("Generating LLM response..."):
                    response_text = call_openai(key, model, prompt)
                parsed = parse_json_response(response_text)
                thresholds = baseline_verification_thresholds_from_training(training)
                baseline_positive_probability = float(
                    training.best_run.pipeline.predict_proba(pd.DataFrame([holdout_row]))[0, 1]
                )
                baseline_predicted_outcome = (
                    training.positive_label if baseline_positive_probability >= 0.5 else training.negative_label
                )
                parsed = apply_baseline_verification(
                    parsed,
                    baseline_predicted_outcome=baseline_predicted_outcome,
                    baseline_confidence=max(baseline_positive_probability, 1 - baseline_positive_probability),
                    positive_class_label=training.positive_label,
                    **thresholds,
                )
                final_predicted, final_confidence, detailed_rationale = extracted_prediction_fields(parsed)
                if value_is_present(final_predicted) and value_is_present(final_confidence) and value_is_present(detailed_rationale):
                    simplified_rationale = detailed_rationale
                    simplify_note = ""
                    try:
                        with st.spinner("Simplifying the explanation for general audiences..."):
                            simplified = simplify_final_rationale(
                                api_key=key,
                                prediction_goal=results["prediction_goal"],
                                predicted_outcome=final_predicted,
                                confidence_level=final_confidence,
                                detailed_rationale=detailed_rationale,
                            )
                        if simplified:
                            simplified_rationale = simplified
                            simplify_note = f"Simplified explanation generated with {RATIONALE_SIMPLIFIER_MODEL}."
                    except Exception as exc:
                        simplify_note = f"The prediction succeeded, but the plain-language rewrite step failed: {exc}"

                    st.subheader("Prediction")
                    c1, c2 = st.columns(2)
                    c1.metric("Predicted outcome", str(final_predicted))
                    c2.metric("Confidence level", str(final_confidence))
                    st.subheader("Reasoning")
                    st.write(simplified_rationale)
                    if simplify_note:
                        if simplify_note.startswith("The prediction succeeded"):
                            st.warning(simplify_note)
                        else:
                            st.caption(simplify_note)
                    with st.expander("Detailed rationale"):
                        st.write(detailed_rationale)
                    st.text_area("Raw LLM response", response_text, height=220)
                else:
                    st.warning("The selected LLM returned a response, but the app could not extract the expected prediction fields.")
                    st.text_area("Raw LLM response", response_text, height=220)
            except Exception as exc:
                st.error(str(exc))


def llm_lab_tab() -> None:
    st.header("LLM Experiment Lab")
    st.caption("Load exported E2R2 files and run individual or batch LLM predictions.")

    case_file = st.file_uploader("Case base CSV", type=["csv", "xlsx", "xls"], key="lab_case")
    shap_file = st.file_uploader("SHAP scores CSV", type=["csv", "xlsx", "xls"], key="lab_shap")
    holdout_file = st.file_uploader("Holdout dataset CSV", type=["csv", "xlsx", "xls"], key="lab_holdout")
    case_vector_file = st.file_uploader("Case vectors CSV", type=["csv", "xlsx", "xls"], key="lab_case_vec")
    holdout_vector_file = st.file_uploader("Holdout vectors CSV", type=["csv", "xlsx", "xls"], key="lab_holdout_vec")

    if st.button("Load experiment files"):
        if not case_file or not shap_file or not holdout_file:
            st.error("Upload the case base, SHAP scores, and holdout dataset.")
            return
        case_base = uploaded_table(case_file)
        shap_scores = uploaded_table(shap_file)
        holdout = uploaded_table(holdout_file)
        case_vectors_df = uploaded_table(case_vector_file)
        holdout_vectors_df = uploaded_table(holdout_vector_file)
        target_column = infer_target_column(holdout)
        feature_columns = feature_columns_for_retrieval(holdout, target_column)
        exact = exact_vectors(holdout, case_base, holdout_vectors_df, case_vectors_df)
        if exact is not None:
            holdout_vectors, case_vectors = exact
            retrieval_mode = "exported training-pipeline vectors"
        else:
            holdout_vectors, case_vectors, retrieval_mode = build_retrieval_vectors(
                holdout, case_base, feature_columns, shap_scores
            )
        st.session_state["lab"] = {
            "case_base": case_base,
            "shap_scores": shap_scores,
            "holdout": holdout,
            "target_column": target_column,
            "holdout_vectors": holdout_vectors,
            "case_vectors": case_vectors,
            "retrieval_mode": retrieval_mode,
        }

    lab = st.session_state.get("lab")
    if not lab:
        return

    st.success(f"Loaded. Retrieval mode: {lab['retrieval_mode']}")
    st.write(
        f"{len(lab['case_base'])} case-base records, {len(lab['shap_scores'])} SHAP rows, "
        f"{len(lab['holdout'])} holdout rows."
    )

    api_key = st.text_input("OpenAI API key", type="password", key="lab_api_key")
    model = model_select("lab_model", default="gpt-4.1-mini")
    if "pro" in model:
        st.info("Pro models can take several minutes and may be unreliable for interactive Streamlit runs. For quick experiments, use gpt-4.1-mini, gpt-5-mini, or gpt-5.4-mini.")
    prediction_goal = st.text_area("Prediction goal", key="lab_prediction_goal")
    neighbors = st.number_input("Retrieved cases", 1, 12, 5, key="lab_neighbors")

    mode = st.radio("Run mode", ["Individual", "Batch"], horizontal=True)
    if mode == "Individual":
        row_index = st.selectbox("Holdout row", lab["holdout"]["row_index"].tolist())
        if st.button("Generate LLM prediction"):
            key = resolved_api_key(api_key)
            if not key:
                st.warning("Enter an OpenAI API key, or add OPENAI_API_KEY to Streamlit secrets, to generate predictions.")
            else:
                try:
                    with st.spinner("Generating LLM prediction..."):
                        result = run_lab_prediction(
                            row_index, lab, key, model, prediction_goal, int(neighbors)
                        )
                    st.dataframe(
                        visible_prediction_results(pd.DataFrame([result])),
                        width="stretch",
                    )
                    st.text_area("Prompt", result["prompt"], height=320)
                    st.text_area("Raw LLM response", result["llm_raw_response"], height=220)
                except Exception as exc:
                    st.error(str(exc))
    else:
        c1, c2 = st.columns(2)
        offset = c1.number_input("Start after first rows", 0, len(lab["holdout"]) - 1, 0)
        limit = c2.number_input("Number of rows", 1, len(lab["holdout"]), 25)
        if st.button("Run batch"):
            key = resolved_api_key(api_key)
            if not key:
                st.warning("Enter an OpenAI API key, or add OPENAI_API_KEY to Streamlit secrets, to run a batch.")
            else:
                rows = lab["holdout"]["row_index"].iloc[int(offset) : int(offset) + int(limit)].tolist()
                results = []
                progress = st.progress(0)
                status = st.empty()
                try:
                    for i, row in enumerate(rows, start=1):
                        status.write(f"Processing row {row}. Completed {i - 1} of {len(rows)} predictions.")
                        results.append(
                            run_lab_prediction(row, lab, key, model, prediction_goal, int(neighbors))
                        )
                        progress.progress(int(i / max(len(rows), 1) * 100))
                        status.write(f"Completed {i} of {len(rows)} predictions.")
                    result_df = pd.DataFrame(results)
                    st.session_state["lab_batch"] = result_df
                    status.success(f"Batch complete: {len(result_df)} predictions generated.")
                    st.dataframe(visible_prediction_results(result_df), width="stretch")
                except Exception as exc:
                    st.error(str(exc))
        if "lab_batch" in st.session_state:
            st.download_button(
                "Download batch predictions",
                dataframe_to_csv_bytes(st.session_state["lab_batch"]),
                "e2r2_llm_batch_predictions.csv",
                "text/csv",
            )


st.title("E2R2 Decision Support")
tab1, tab2 = st.tabs(["Full Pipeline", "LLM Experiment Lab"])
with tab1:
    main_pipeline_tab()
with tab2:
    llm_lab_tab()
