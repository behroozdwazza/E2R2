from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from e2r2_pipeline import (
    build_case_base,
    build_e2r2_prompt,
    case_vectors_dataset,
    dataframe_to_csv_bytes,
    holdout_dataset,
    holdout_vectors_dataset,
    infer_llm_role,
    parse_data_dictionary,
    read_uploaded_table,
    retrieve_similar_cases,
    summarize_peer_alignment,
    train_e2r2_baseline,
)


OPENAI_LLM_MODELS = [
    "gpt-5.1",
    "gpt-5.4-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]

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


def call_openai(api_key: str, model: str, prompt: str) -> str:
    payload: Dict[str, Any] = {"model": model, "input": prompt}
    if not model.startswith("gpt-5"):
        payload["temperature"] = 0.2
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
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


def parse_json_response(text: str) -> Dict[str, Any]:
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


def model_select(key: str) -> str:
    return st.selectbox("LLM model", OPENAI_LLM_MODELS, index=0, key=key)


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
) -> Tuple[np.ndarray, np.ndarray]:
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
    matrix = pd.concat([numeric, categorical], axis=1).to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    holdout_vectors = matrix / norms
    row_to_position = {row_index: pos for pos, row_index in enumerate(holdout["row_index"].tolist())}
    case_positions = [row_to_position[idx] for idx in case_base["row_index"].tolist()]
    return holdout_vectors, holdout_vectors[case_positions]


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
    retrieved: pd.DataFrame,
    shap_scores: pd.DataFrame,
    target_column: str,
    prediction_goal: str,
    include_baseline: bool,
    neighbors: int,
) -> str:
    role = infer_llm_role(prediction_goal, target_column)
    excluded = set(BASELINE_COLUMNS) | {"row_index", target_column}
    feature_lines = []
    for feature, value in row.items():
        if feature in excluded:
            continue
        definition = lab_feature_definition(shap_scores, feature)
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

    baseline_block = (
        "\n".join(
            [
                "Baseline model signal for this holdout case:",
                f"- baseline_predicted_outcome: {row.get('baseline_predicted_outcome', '')}",
                f"- baseline_positive_probability: {row.get('baseline_positive_probability', '')}",
                f"- baseline_confidence: {row.get('baseline_confidence', '')}",
            ]
        )
        if include_baseline
        else "Baseline model signal for this holdout case: intentionally omitted for this experiment."
    )

    return f"""{role}

You are using the E2R2 framework: Explain, Embed, Retrieve, and Reason. E2R2 combines a baseline predictive model, SHAP feature attributions, similarity-based precedent retrieval, and professional reasoning.

Important rules:
- Do not use majority vote alone.
- Interpret SHAP sign and magnitude explicitly; larger absolute SHAP values carry more weight.
- Compare the holdout case with retrieved precedents on high-magnitude SHAP features.
- If a baseline signal is provided, treat it as one additional signal. If omitted, do not infer it.
- Return valid JSON only, with no Markdown.

Prediction goal:
{prediction_goal or f"Predict {target_column}."}

{baseline_block}

Holdout case profile:
{chr(10).join(feature_lines)}

Retrieved E2R2 precedent cases, top {neighbors}:
{chr(10).join(peer_blocks)}

SHAP-weighted alignment summary across retrieved cases:
{lab_shap_alignment(row, retrieved, shap_scores)}

Return this JSON object:
{{
  "predicted_outcome": "one of the domain labels",
  "confidence_level": "low | moderate | high",
  "rationale": "one professional paragraph grounded in E2R2 evidence"
}}
"""


def run_lab_prediction(
    row_index: Any,
    lab: Dict[str, Any],
    api_key: str,
    model: str,
    prediction_goal: str,
    include_baseline: bool,
    neighbors: int,
) -> Dict[str, Any]:
    holdout = lab["holdout"]
    row = holdout.loc[holdout["row_index"] == row_index].iloc[0]
    retrieved = lab_retrieve(
        row_index, lab["case_base"], holdout, lab["holdout_vectors"], lab["case_vectors"], neighbors
    )
    prompt = lab_prompt(
        row,
        retrieved,
        lab["shap_scores"],
        lab["target_column"],
        prediction_goal,
        include_baseline,
        neighbors,
    )
    raw = call_openai(api_key, model, prompt)
    parsed = parse_json_response(raw)
    return {
        "row_index": row_index,
        "actual_outcome": row.get(lab["target_column"], ""),
        "baseline_predicted_outcome": row.get("baseline_predicted_outcome", ""),
        "baseline_confidence": row.get("baseline_confidence", ""),
        "baseline_signal_in_prompt": include_baseline,
        "llm_predicted_outcome": parsed.get("predicted_outcome", ""),
        "llm_confidence_level": parsed.get("confidence_level", ""),
        "llm_rationale": parsed.get("rationale", ""),
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
    st.dataframe(df.head(10), use_container_width=True)

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
    st.dataframe(pd.DataFrame([training.best_run.metrics]), use_container_width=True)

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
    model = model_select("main_model")
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
        if api_key:
            st.text_area("LLM response", call_openai(api_key, model, prompt), height=220)


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
            holdout_vectors, case_vectors = build_retrieval_vectors(holdout, case_base, feature_columns)
            retrieval_mode = "reconstructed exported features"
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
    model = model_select("lab_model")
    prediction_goal = st.text_area("Prediction goal", key="lab_prediction_goal")
    include_baseline = st.checkbox("Include baseline model signal in prompt", value=True)
    neighbors = st.number_input("Retrieved cases", 1, 12, 5, key="lab_neighbors")

    mode = st.radio("Run mode", ["Individual", "Batch"], horizontal=True)
    if mode == "Individual":
        row_index = st.selectbox("Holdout row", lab["holdout"]["row_index"].tolist())
        if st.button("Generate LLM prediction"):
            result = run_lab_prediction(
                row_index, lab, api_key, model, prediction_goal, include_baseline, int(neighbors)
            )
            st.dataframe(pd.DataFrame([{k: v for k, v in result.items() if k not in {"prompt", "llm_raw_response"}}]))
            st.text_area("Prompt", result["prompt"], height=320)
            st.text_area("Raw LLM response", result["llm_raw_response"], height=220)
    else:
        c1, c2 = st.columns(2)
        offset = c1.number_input("Start after first rows", 0, len(lab["holdout"]) - 1, 0)
        limit = c2.number_input("Number of rows", 1, len(lab["holdout"]), 25)
        if st.button("Run batch"):
            rows = lab["holdout"]["row_index"].iloc[int(offset) : int(offset) + int(limit)].tolist()
            results = [
                run_lab_prediction(row, lab, api_key, model, prediction_goal, include_baseline, int(neighbors))
                for row in rows
            ]
            result_df = pd.DataFrame(results)
            st.session_state["lab_batch"] = result_df
            st.dataframe(result_df.drop(columns=["prompt", "llm_raw_response"], errors="ignore"), use_container_width=True)
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
