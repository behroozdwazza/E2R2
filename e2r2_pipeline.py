from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


@dataclass
class ModelRun:
    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    score: float


@dataclass
class TrainingResult:
    best_run: ModelRun
    all_runs: List[ModelRun]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_predictions: pd.DataFrame
    feature_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    positive_label: Any
    negative_label: Any
    target_column: str


def read_uploaded_table(uploaded_file: Any) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Upload a CSV or Excel file.")


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def parse_data_dictionary(dictionary_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    if dictionary_df is None or dictionary_df.empty:
        return {}

    columns = {normalize_column_name(c): c for c in dictionary_df.columns}
    name_candidates = [
        "variable",
        "feature",
        "field",
        "column",
        "column_name",
        "name",
        "attribute",
    ]
    definition_candidates = [
        "definition",
        "description",
        "meaning",
        "label",
        "notes",
        "values",
    ]

    name_col = next((columns[c] for c in name_candidates if c in columns), dictionary_df.columns[0])
    definition_cols = [columns[c] for c in definition_candidates if c in columns and columns[c] != name_col]
    if not definition_cols:
        definition_cols = [c for c in dictionary_df.columns if c != name_col][:2]

    mapping: Dict[str, str] = {}
    for _, row in dictionary_df.iterrows():
        raw_name = row.get(name_col)
        if pd.isna(raw_name):
            continue
        pieces = []
        for col in definition_cols:
            value = row.get(col)
            if pd.notna(value) and str(value).strip():
                pieces.append(f"{col}: {value}")
        mapping[str(raw_name)] = "; ".join(pieces)
    return mapping


def coerce_binary_target(y: pd.Series, positive_label: Any) -> pd.Series:
    return (y.astype(str) == str(positive_label)).astype(int)


def split_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_columns = [
        c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique(dropna=True) > 2
    ]
    categorical_columns = [c for c in X.columns if c not in numeric_columns]
    return numeric_columns, categorical_columns


def make_preprocessor(numeric_columns: Sequence[str], categorical_columns: Sequence[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_models() -> Dict[str, Any]:
    return {
        "Logistic regression": LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "Extra trees": ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "Gradient boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def compute_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (proba >= threshold).astype(int)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, proba)
    except ValueError:
        metrics["auc"] = np.nan
    return metrics


def train_e2r2_baseline(
    df: pd.DataFrame,
    target_column: str,
    positive_label: Any,
    ignored_columns: Optional[Sequence[str]] = None,
    test_size: float = 0.25,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> TrainingResult:
    if progress_callback:
        progress_callback("Preparing data and checking the outcome column", 8)
    ignored_columns = list(ignored_columns or [])
    model_df = df.drop(columns=[c for c in ignored_columns if c in df.columns]).copy()
    model_df = model_df.dropna(subset=[target_column])

    y_raw = model_df[target_column]
    y = coerce_binary_target(y_raw, positive_label)
    if y.nunique() != 2:
        raise ValueError(
            "The selected positive outcome did not produce a binary target. "
            "Check the outcome column and positive outcome value."
        )
    feature_columns = [c for c in model_df.columns if c != target_column]
    X = model_df[feature_columns]

    negative_values = [v for v in y_raw.dropna().unique().tolist() if str(v) != str(positive_label)]
    negative_label = negative_values[0] if negative_values else "not " + str(positive_label)

    numeric_columns, categorical_columns = split_feature_types(X)
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    runs: List[ModelRun] = []
    models = candidate_models()
    for position, (name, estimator) in enumerate(models.items(), start=1):
        if progress_callback:
            percent = 18 + ((position - 1) / max(len(models), 1)) * 42
            progress_callback(f"Training baseline model: {name}", percent)
        preprocessor = make_preprocessor(numeric_columns, categorical_columns)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba)
        score = np.nan_to_num(metrics.get("auc", np.nan), nan=metrics["balanced_accuracy"])
        score = 0.65 * score + 0.35 * metrics["balanced_accuracy"]
        runs.append(ModelRun(name=name, pipeline=pipeline, metrics=metrics, score=score))

    if progress_callback:
        progress_callback("Selecting the best baseline model", 62)
    best_run = sorted(runs, key=lambda run: run.score, reverse=True)[0]
    best_proba = best_run.pipeline.predict_proba(X_test)[:, 1]
    test_predictions = pd.DataFrame(
        {
            "row_index": X_test.index,
            "actual": y_test.values,
            "positive_probability": best_proba,
            "predicted": (best_proba >= 0.5).astype(int),
            "confidence": np.maximum(best_proba, 1 - best_proba),
        },
        index=X_test.index,
    )

    return TrainingResult(
        best_run=best_run,
        all_runs=runs,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_predictions=test_predictions,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        positive_label=positive_label,
        negative_label=negative_label,
        target_column=target_column,
    )


def baseline_values(X_train: pd.DataFrame, numeric_columns: Sequence[str]) -> Dict[str, Any]:
    baselines: Dict[str, Any] = {}
    for col in X_train.columns:
        series = X_train[col].dropna()
        if col in numeric_columns:
            baselines[col] = float(series.median()) if not series.empty else 0
        else:
            baselines[col] = series.mode().iloc[0] if not series.mode().empty else ""
    return baselines


def local_feature_attributions(
    pipeline: Pipeline,
    row: pd.Series,
    baselines: Dict[str, Any],
    predicted_class: int,
    max_features: int = 8,
) -> pd.DataFrame:
    row_df = pd.DataFrame([row])
    original_probability = pipeline.predict_proba(row_df)[0, predicted_class]
    records = []
    for feature, baseline in baselines.items():
        perturbed = row.copy()
        perturbed[feature] = baseline
        changed_probability = pipeline.predict_proba(pd.DataFrame([perturbed]))[0, predicted_class]
        records.append(
            {
                "feature": feature,
                "value": row.get(feature),
                "baseline_value": baseline,
                "shap_score": float(original_probability - changed_probability),
                "abs_shap_score": float(abs(original_probability - changed_probability)),
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("abs_shap_score", ascending=False)
        .head(max_features)
        .reset_index(drop=True)
    )


def transformed_feature_to_original(
    transformed_feature: str,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> str:
    name = str(transformed_feature).replace("numeric__", "").replace("categorical__", "")
    if name in feature_columns:
        return name
    for column in sorted(categorical_columns, key=len, reverse=True):
        if name == column or name.startswith(f"{column}_"):
            return column
    return name


def class_shap_values(shap_values: Any, predicted_class: int) -> np.ndarray:
    values = np.asarray(shap_values)
    if isinstance(shap_values, list):
        return np.asarray(shap_values[predicted_class])
    if values.ndim == 3:
        if values.shape[2] > predicted_class:
            return values[:, :, predicted_class]
        if values.shape[0] > predicted_class:
            return values[predicted_class, :, :]
    return values


def actual_shap_attributions(
    training: TrainingResult,
    selected: pd.DataFrame,
    baselines: Dict[str, Any],
    top_predictors: int = 8,
    background_sample_size: int = 100,
    kernel_nsamples: int = 100,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[Any, pd.DataFrame]:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "True SHAP mode requires the shap package. Install it once with: "
            "C:\\Users\\davazdab\\.conda\\envs\\knime_python\\python.exe -m pip install shap"
        ) from exc

    if progress_callback:
        progress_callback("Running true SHAP setup", 72)

    pipeline = training.best_run.pipeline
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    selected_X = training.X_test.loc[selected.index]
    background_X = training.X_train.sample(
        n=min(background_sample_size, len(training.X_train)),
        random_state=RANDOM_STATE,
    )
    background_vectors = np.asarray(preprocessor.transform(background_X), dtype=float)
    selected_vectors = np.asarray(preprocessor.transform(selected_X), dtype=float)
    transformed_feature_names = list(preprocessor.get_feature_names_out())
    original_names = [
        transformed_feature_to_original(name, training.feature_columns, training.categorical_columns)
        for name in transformed_feature_names
    ]

    if progress_callback:
        progress_callback(f"Running probability-space SHAP for {len(selected)} case-base records", 76)

    def predict_from_vectors(vectors: np.ndarray) -> np.ndarray:
        return model.predict_proba(np.asarray(vectors, dtype=float))

    if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier)):
        try:
            explainer = shap.TreeExplainer(
                model,
                data=background_vectors,
                feature_perturbation="interventional",
                model_output="probability",
            )
            all_values = explainer.shap_values(selected_vectors, check_additivity=False)
        except Exception:
            if progress_callback:
                progress_callback("Tree SHAP probability mode was unavailable; using Kernel SHAP", 76)
            explainer = shap.KernelExplainer(predict_from_vectors, background_vectors)
            all_values = explainer.shap_values(selected_vectors, nsamples=kernel_nsamples)
    else:
        # LinearExplainer and the default TreeExplainer explain raw model scores for many
        # classifiers. KernelExplainer against predict_proba keeps scores in probability units.
        explainer = shap.KernelExplainer(predict_from_vectors, background_vectors)
        all_values = explainer.shap_values(selected_vectors, nsamples=kernel_nsamples)

    predicted_classes = selected["predicted"].astype(int).tolist()
    rows_by_index: Dict[Any, pd.DataFrame] = {}
    for position, (idx, pred_row) in enumerate(selected.iterrows()):
        if progress_callback:
            percent = 78 + ((position + 1) / max(len(selected), 1)) * 14
            progress_callback(f"Aggregating SHAP predictors for case {position + 1} of {len(selected)}", percent)
        values_for_class = class_shap_values(all_values, predicted_classes[position])
        if values_for_class.ndim == 1:
            row_values = values_for_class
        else:
            row_values = values_for_class[position]
        aggregated: Dict[str, Dict[str, float]] = {}
        for transformed_name, original_name, shap_score in zip(
            transformed_feature_names,
            original_names,
            row_values,
        ):
            bucket = aggregated.setdefault(original_name, {"shap_score": 0.0, "abs_shap_score": 0.0})
            bucket["shap_score"] += float(shap_score)
            bucket["abs_shap_score"] += abs(float(shap_score))
        raw_row = training.X_test.loc[idx]
        records = [
            {
                "feature": feature,
                "value": raw_row.get(feature),
                "baseline_value": baselines.get(feature),
                "shap_score": scores["shap_score"],
                "abs_shap_score": scores["abs_shap_score"],
                "shap_background_size": len(background_X),
                "shap_kernel_nsamples": kernel_nsamples,
            }
            for feature, scores in aggregated.items()
        ]
        rows_by_index[idx] = (
            pd.DataFrame(records)
            .sort_values("abs_shap_score", ascending=False)
            .head(top_predictors)
            .reset_index(drop=True)
        )
    return rows_by_index


def high_confidence_subset(
    training: TrainingResult,
    confidence_threshold: float,
    minimum_cases: int,
) -> pd.DataFrame:
    predictions = training.test_predictions.copy()
    predictions["correct"] = predictions["predicted"] == predictions["actual"]
    predictions = predictions[predictions["correct"]].sort_values("confidence", ascending=False)
    selected = predictions[predictions["confidence"] >= confidence_threshold]
    if len(selected) < minimum_cases:
        selected = predictions.head(min(minimum_cases, len(predictions)))
    if selected.empty:
        raise ValueError(
            "No correctly predicted holdout cases were available for the E2R2 case base. "
            "Try a larger holdout share, review the target setup, or improve the baseline model."
        )
    return selected


def holdout_dataset(training: TrainingResult) -> pd.DataFrame:
    holdout = training.X_test.copy()
    predictions = training.test_predictions.reindex(holdout.index)
    holdout.insert(0, "row_index", predictions["row_index"])
    holdout[training.target_column] = predictions["actual"].map(
        {1: training.positive_label, 0: training.negative_label}
    )
    holdout["baseline_predicted_outcome"] = predictions["predicted"].map(
        {1: training.positive_label, 0: training.negative_label}
    )
    holdout["baseline_positive_probability"] = predictions["positive_probability"]
    holdout["baseline_confidence"] = predictions["confidence"]
    holdout["baseline_correct"] = predictions["predicted"] == predictions["actual"]
    return holdout.reset_index(drop=True)


def holdout_vectors_dataset(training: TrainingResult) -> pd.DataFrame:
    vectors = transformed_vectors(training, training.X_test)
    vector_columns = [f"vector_{i}" for i in range(vectors.shape[1])]
    df = pd.DataFrame(vectors, columns=vector_columns)
    df.insert(0, "row_index", training.X_test.index.tolist())
    return df


def case_vectors_dataset(case_base: pd.DataFrame) -> pd.DataFrame:
    vector_columns = [c for c in case_base.columns if str(c).startswith("vector_")]
    columns = ["case_id", "row_index"] + vector_columns
    return case_base[columns].copy()


def transformed_vectors(training: TrainingResult, X: pd.DataFrame) -> np.ndarray:
    vectors = training.best_run.pipeline.named_steps["preprocess"].transform(X)
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def build_case_base(
    training: TrainingResult,
    data_dictionary: Optional[Dict[str, str]] = None,
    confidence_threshold: float = 0.9,
    minimum_cases: int = 25,
    top_predictors: int = 8,
    attribution_method: str = "actual_shap",
    shap_background_size: int = 100,
    shap_kernel_nsamples: int = 100,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dictionary = data_dictionary or {}
    if progress_callback:
        progress_callback("Selecting correctly predicted high-confidence cases", 68)
    selected = high_confidence_subset(training, confidence_threshold, minimum_cases)
    if progress_callback:
        progress_callback(f"Preparing attribution baselines for {len(selected)} case-base records", 72)
    baselines = baseline_values(training.X_train, training.numeric_columns)
    shap_attributions: Dict[Any, pd.DataFrame] = {}
    if attribution_method == "actual_shap":
        shap_attributions = actual_shap_attributions(
            training,
            selected,
            baselines,
            top_predictors=top_predictors,
            background_sample_size=shap_background_size,
            kernel_nsamples=shap_kernel_nsamples,
            progress_callback=progress_callback,
        )
    rows = []
    shap_rows = []
    for case_id, (idx, pred_row) in enumerate(selected.iterrows(), start=1):
        if progress_callback:
            if attribution_method == "actual_shap":
                percent = 92 + (case_id / max(len(selected), 1)) * 2
                progress_callback(f"Building SHAP case-base row {case_id} of {len(selected)}", percent)
            else:
                percent = 72 + (case_id / max(len(selected), 1)) * 20
                progress_callback(f"Computing top predictors for case {case_id} of {len(selected)}", percent)
        raw_row = training.X_test.loc[idx]
        predicted_class = int(pred_row["predicted"])
        if attribution_method == "actual_shap":
            attributions = shap_attributions[idx]
            attribution_label = "actual_shap_probability"
        else:
            attributions = local_feature_attributions(
                training.best_run.pipeline,
                raw_row,
                baselines,
                predicted_class=predicted_class,
                max_features=top_predictors,
            )
            attribution_label = "fast_probability_perturbation"
        for rank, attr in attributions.iterrows():
            shap_rows.append(
                {
                    "case_id": case_id,
                    "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                    "rank": rank + 1,
                    "feature": attr["feature"],
                    "definition": data_dictionary.get(str(attr["feature"]), ""),
                    "value": attr["value"],
                    "baseline_value": attr["baseline_value"],
                    "shap_score": attr["shap_score"],
                    "abs_shap_score": attr["abs_shap_score"],
                    "shap_output_units": "probability" if attribution_method == "actual_shap" else "probability_delta",
                    "shap_background_size": attr.get("shap_background_size", ""),
                    "shap_kernel_nsamples": attr.get("shap_kernel_nsamples", ""),
                    "attribution_method": attribution_label,
                }
            )
        outcome = training.positive_label if predicted_class == 1 else training.negative_label
        rows.append(
            {
                "case_id": case_id,
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "predicted_class": predicted_class,
                "predicted_outcome": outcome,
                "actual_class": int(pred_row["actual"]),
                "correct": bool(pred_row["correct"]),
                "positive_probability": float(pred_row["positive_probability"]),
                "confidence": float(pred_row["confidence"]),
                "top_predictors": ", ".join(attributions["feature"].astype(str).tolist()),
            }
        )
    case_base = pd.DataFrame(rows)
    shap_table = pd.DataFrame(shap_rows)
    if progress_callback:
        progress_callback("Embedding case-base records for retrieval", 94)
    case_vectors = transformed_vectors(training, training.X_test.loc[selected.index])
    vector_columns = [f"vector_{i}" for i in range(case_vectors.shape[1])]
    vectors_df = pd.DataFrame(case_vectors, columns=vector_columns)
    case_base = pd.concat([case_base.reset_index(drop=True), vectors_df], axis=1)
    return case_base, shap_table


def retrieve_similar_cases(
    training: TrainingResult,
    case_base: pd.DataFrame,
    holdout_row: pd.Series,
    k: int = 5,
) -> pd.DataFrame:
    vector_columns = [c for c in case_base.columns if c.startswith("vector_")]
    if not vector_columns:
        return pd.DataFrame()
    holdout_vector = transformed_vectors(training, pd.DataFrame([holdout_row]))[0]
    case_vectors = case_base[vector_columns].to_numpy(dtype=float)
    similarities = case_vectors @ holdout_vector
    retrieved = case_base.drop(columns=vector_columns).copy()
    retrieved["similarity"] = similarities
    return retrieved.sort_values("similarity", ascending=False).head(k).reset_index(drop=True)


def summarize_peer_alignment(
    holdout_row: pd.Series,
    retrieved: pd.DataFrame,
    shap_table: pd.DataFrame,
    training: TrainingResult,
) -> pd.DataFrame:
    records = []
    top_features = (
        shap_table[shap_table["case_id"].isin(retrieved["case_id"])]
        .groupby("feature", as_index=False)["abs_shap_score"]
        .mean()
        .sort_values("abs_shap_score", ascending=False)
        .head(8)["feature"]
        .tolist()
    )
    for feature in top_features:
        peer_rows = training.X_test.loc[
            training.X_test.index.isin(
                retrieved["row_index"].apply(lambda value: int(value) if str(value).isdigit() else value)
            )
        ]
        peer_values = peer_rows[feature].dropna() if feature in peer_rows else pd.Series(dtype=object)
        if peer_values.empty:
            peer_summary = "missing among retrieved cases"
            alignment = "not enough peer information"
        elif pd.api.types.is_numeric_dtype(peer_values):
            peer_summary = f"{peer_values.min():.3g} to {peer_values.max():.3g}"
            value = holdout_row.get(feature)
            if pd.isna(value):
                alignment = "missing in holdout"
            elif peer_values.min() <= float(value) <= peer_values.max():
                alignment = "within retrieved peer range"
            else:
                alignment = "outside retrieved peer range"
        else:
            common = peer_values.astype(str).value_counts().head(3)
            peer_summary = ", ".join([f"{idx} ({count})" for idx, count in common.items()])
            holdout_value = str(holdout_row.get(feature))
            alignment = "matches common peer value" if holdout_value in common.index else "differs from common peer value"
        records.append(
            {
                "feature": feature,
                "holdout_value": holdout_row.get(feature),
                "retrieved_peer_pattern": peer_summary,
                "alignment": alignment,
            }
        )
    return pd.DataFrame(records)


def infer_llm_role(prediction_goal: str, target_column: str) -> str:
    text = f"{prediction_goal} {target_column}".lower()
    student_terms = ["student", "attrition", "retention", "enrollment", "freshman", "college", "academic"]
    if any(term in text for term in student_terms):
        return (
            "You are an experienced academic advisor and student-success analyst who specializes "
            "in identifying students at risk of attrition and explaining risk factors in practical, "
            "support-oriented language."
        )
    health_terms = ["patient", "clinical", "hospital", "medical", "triage", "diagnosis"]
    if any(term in text for term in health_terms):
        return (
            "You are a careful clinical decision-support analyst. You explain predictive evidence "
            "in cautious, non-diagnostic language suitable for review by qualified professionals."
        )
    finance_terms = ["fraud", "credit", "loan", "default", "customer", "churn", "account"]
    if any(term in text for term in finance_terms):
        return (
            "You are a senior business decision-support analyst who specializes in translating "
            "predictive model evidence into clear, action-oriented risk assessments."
        )
    return (
        "You are a senior decision-support analyst who specializes in explaining predictive model "
        "outputs with case-based evidence and feature-level attribution."
    )


def shap_weighted_alignment_summary(
    holdout_row: pd.Series,
    retrieved: pd.DataFrame,
    shap_table: pd.DataFrame,
) -> str:
    relevant = shap_table[shap_table["case_id"].isin(retrieved["case_id"])].copy()
    if relevant.empty:
        return "- No SHAP alignment rows were available."
    rows = []
    for feature, group in relevant.groupby("feature"):
        mean_shap = group["shap_score"].mean()
        mean_abs = group["abs_shap_score"].mean()
        positive_count = int((group["shap_score"] > 0).sum())
        negative_count = int((group["shap_score"] < 0).sum())
        peer_values = group["value"].dropna().astype(str).value_counts().head(3)
        holdout_value = holdout_row.get(feature, "")
        matching_count = int((group["value"].astype(str) == str(holdout_value)).sum())
        if mean_shap > 0:
            direction = "pushes toward the precedent outcome"
        elif mean_shap < 0:
            direction = "pushes away from the precedent outcome"
        else:
            direction = "has near-neutral direction"
        rows.append(
            {
                "mean_abs": mean_abs,
                "line": (
                    f"- {feature}: holdout={holdout_value}; peer values="
                    f"{', '.join([f'{value} ({count})' for value, count in peer_values.items()])}; "
                    f"mean SHAP={mean_shap:.4f}, mean |SHAP|={mean_abs:.4f}; "
                    f"{direction}; sign counts +/{positive_count}, -/{negative_count}; "
                    f"exact value matches among retrieved cases={matching_count}"
                ),
            }
        )
    rows = sorted(rows, key=lambda item: item["mean_abs"], reverse=True)[:10]
    return "\n".join(item["line"] for item in rows)


def build_e2r2_prompt(
    training: TrainingResult,
    holdout_row: pd.Series,
    retrieved: pd.DataFrame,
    shap_table: pd.DataFrame,
    alignment: pd.DataFrame,
    data_dictionary: Optional[Dict[str, str]] = None,
    prediction_goal: str = "",
) -> str:
    data_dictionary = data_dictionary or {}
    role = infer_llm_role(prediction_goal, training.target_column)
    positive_probability = training.best_run.pipeline.predict_proba(pd.DataFrame([holdout_row]))[0, 1]
    baseline_predicted_class = int(positive_probability >= 0.5)
    baseline_outcome = training.positive_label if baseline_predicted_class == 1 else training.negative_label
    baseline_confidence = max(positive_probability, 1 - positive_probability)
    holdout_features = []
    for feature, value in holdout_row.items():
        definition = data_dictionary.get(str(feature), "")
        suffix = f" ({definition})" if definition else ""
        holdout_features.append(f"- {feature}{suffix}: {value}")

    peer_blocks = []
    for _, peer in retrieved.iterrows():
        peer_shap = shap_table[shap_table["case_id"] == peer["case_id"]].head(6)
        units = (
            peer_shap["shap_output_units"].dropna().iloc[0]
            if "shap_output_units" in peer_shap.columns and not peer_shap["shap_output_units"].dropna().empty
            else "probability"
        )
        shap_lines = [
            f"  - {row.feature}: value={row.value}, SHAP contribution={row.shap_score:.4f} ({units})"
            for row in peer_shap.itertuples()
        ]
        peer_blocks.append(
            "\n".join(
                [
                    f"Case {peer.case_id}: outcome={peer.predicted_outcome}, "
                    f"confidence={peer.confidence:.3f}, similarity={peer.similarity:.3f}",
                    *shap_lines,
                ]
            )
        )

    weighted_alignment = shap_weighted_alignment_summary(holdout_row, retrieved, shap_table)
    alignment_lines = [
        f"- {row.feature}: holdout={row.holdout_value}; peers={row.retrieved_peer_pattern}; {row.alignment}"
        for row in alignment.itertuples()
    ]
    labels = f"{training.positive_label} vs. {training.negative_label}"
    return f"""{role}

You are using the E2R2 framework: Explain, Embed, Retrieve, and Reason.
E2R2 combines four kinds of evidence:
1. Explain: a baseline supervised model predicts the holdout case and provides feature-level SHAP contributions.
2. Embed: cases are represented as normalized feature vectors after preprocessing.
3. Retrieve: the system retrieves the most similar precedent cases from a case base made only of correctly predicted, high-confidence holdout cases.
4. Reason: you compare the holdout case against those precedents on the most important SHAP-informed predictors, then give a transparent prediction and rationale.

Important reasoning rules:
- Do not simply majority-vote the retrieved cases.
- Interpret SHAP sign and magnitude explicitly. Larger absolute SHAP values carry more explanatory weight than smaller values.
- A positive SHAP contribution means that feature pushed the precedent case toward its listed outcome; a negative SHAP contribution means it pushed away from that outcome.
- When deciding, prioritize features with large absolute SHAP values and evaluate whether the holdout case matches or diverges from the precedent values on those high-weight features.
- Treat similarity, baseline confidence, SHAP-weighted alignment, and feature alignment as complementary evidence.
- Use the data dictionary definitions when they clarify a feature.
- Be explicit about both risk-increasing and risk-reducing signals.
- If the evidence is mixed, say so and choose a moderate confidence level.
- Do not invent feature values or facts not shown in the prompt.

Prediction goal:
{prediction_goal or f"Predict {training.target_column}. Labels: {labels}."}

Baseline model signal:
- Selected baseline model: {training.best_run.name}
- Baseline predicted outcome: {baseline_outcome}
- Positive-outcome probability for {training.positive_label}: {positive_probability:.4f}
- Baseline confidence in predicted class: {baseline_confidence:.4f}
- Candidate labels: {labels}

Your task:
Predict the outcome for the holdout case using the E2R2 evidence below. Produce a professional decision-support response with only:
1. predicted_outcome
2. confidence_level: low, moderate, or high
3. rationale: one polished paragraph that explicitly uses SHAP sign, SHAP magnitude, precedent similarity, and baseline confidence

Holdout case:
{chr(10).join(holdout_features)}

Retrieved precedent cases from the E2R2 case base:
{chr(10).join(peer_blocks)}

SHAP-informed alignment summary:
{chr(10).join(alignment_lines)}

SHAP-weighted alignment summary across retrieved cases:
{weighted_alignment}
"""


def local_e2r2_reasoning(
    training: TrainingResult,
    holdout_row: pd.Series,
    retrieved: pd.DataFrame,
    alignment: pd.DataFrame,
) -> Tuple[str, str, str]:
    proba = training.best_run.pipeline.predict_proba(pd.DataFrame([holdout_row]))[0, 1]
    predicted_class = int(proba >= 0.5)
    predicted_outcome = training.positive_label if predicted_class == 1 else training.negative_label
    confidence = max(proba, 1 - proba)
    confidence_label = "high" if confidence >= 0.8 else "moderate" if confidence >= 0.6 else "low"

    peer_counts = retrieved["predicted_class"].value_counts().to_dict()
    peer_phrase = (
        f"{peer_counts.get(1, 0)} retrieved cases supported {training.positive_label} and "
        f"{peer_counts.get(0, 0)} supported {training.negative_label}"
    )
    strongest = alignment.head(4)
    positive_bits = strongest[strongest["alignment"].str.contains("within|matches", case=False, na=False)]
    caution_bits = strongest[~strongest.index.isin(positive_bits.index)]

    support_text = "; ".join(
        f"{row.feature} {row.alignment}" for row in positive_bits.itertuples()
    )
    caution_text = "; ".join(
        f"{row.feature} {row.alignment}" for row in caution_bits.itertuples()
    )
    rationale_parts = [
        f"The baseline model assigns this case a {proba:.1%} probability for {training.positive_label}, leading to {predicted_outcome}.",
        f"Among the five most similar high-confidence precedents, {peer_phrase}.",
    ]
    if support_text:
        rationale_parts.append(f"The strongest alignment signals are: {support_text}.")
    if caution_text:
        rationale_parts.append(f"The main divergences or missing signals are: {caution_text}.")
    rationale_parts.append(
        "This judgment follows the E2R2 pattern by combining model confidence, precedent similarity, and feature-level attribution evidence rather than using peer majority alone."
    )
    return str(predicted_outcome), confidence_label, " ".join(rationale_parts)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
