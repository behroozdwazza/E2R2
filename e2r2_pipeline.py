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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


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
    similarity_weighting: str = "unweighted cosine"
    similarity_weight_min: float = 1.0
    similarity_weight_max: float = 1.0
    similarity_feature_weights: Optional[np.ndarray] = None


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


def coerce_categorical_values(X: Any) -> pd.DataFrame:
    frame = pd.DataFrame(X).copy()
    return frame.apply(lambda col: col.map(lambda value: np.nan if pd.isna(value) else str(value)))


def make_preprocessor(numeric_columns: Sequence[str], categorical_columns: Sequence[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "stringify",
                FunctionTransformer(
                    coerce_categorical_values,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
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


def default_similarity_weights(pipeline: Pipeline) -> Tuple[np.ndarray, str, float, float]:
    """Return equal weights so cosine similarity uses all encoded predictors uniformly."""
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        n_features = len(preprocessor.get_feature_names_out())
    except Exception:
        n_features = 0
    if n_features <= 0:
        return np.ones(0, dtype=float), "unweighted cosine", 1.0, 1.0
    weights = np.ones(n_features, dtype=float)
    return weights, "unweighted cosine", 1.0, 1.0


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
    similarity_weights, similarity_weighting, similarity_min, similarity_max = default_similarity_weights(
        best_run.pipeline
    )
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
        similarity_weighting=similarity_weighting,
        similarity_weight_min=similarity_min,
        similarity_weight_max=similarity_max,
        similarity_feature_weights=similarity_weights,
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
    max_features: int = 8,
) -> pd.DataFrame:
    row_df = pd.DataFrame([row])
    original_probability = pipeline.predict_proba(row_df)[0, 1]
    records = []
    for feature, baseline in baselines.items():
        perturbed = row.copy()
        perturbed[feature] = baseline
        changed_probability = pipeline.predict_proba(pd.DataFrame([perturbed]))[0, 1]
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


def class_shap_values(shap_values: Any, class_index: int) -> np.ndarray:
    values = np.asarray(shap_values)
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])
    if values.ndim == 3:
        if values.shape[2] > class_index:
            return values[:, :, class_index]
        if values.shape[0] > class_index:
            return values[class_index, :, :]
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

    rows_by_index: Dict[Any, pd.DataFrame] = {}
    for position, (idx, pred_row) in enumerate(selected.iterrows()):
        if progress_callback:
            percent = 78 + ((position + 1) / max(len(selected), 1)) * 14
            progress_callback(f"Aggregating SHAP predictors for case {position + 1} of {len(selected)}", percent)
        values_for_class = class_shap_values(all_values, 1)
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
    weights = training.similarity_feature_weights
    if weights is not None and len(weights) == vectors.shape[1]:
        vectors = vectors * weights
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
            attribution_label = "actual_shap_positive_class_probability"
        else:
            attributions = local_feature_attributions(
                training.best_run.pipeline,
                raw_row,
                baselines,
                max_features=top_predictors,
            )
            attribution_label = "fast_positive_class_probability_perturbation"
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
                    "shap_output_units": (
                        "positive_class_probability"
                        if attribution_method == "actual_shap"
                        else "positive_class_probability_delta"
                    ),
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
        progress_callback(f"Embedding case-base records for {training.similarity_weighting} retrieval", 94)
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
        .head(5)["feature"]
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
    return (
        "You are an experienced domain analyst working in the field implied by the prediction goal below. "
        "You specialize in identifying cases at risk of the outcome of concern and explaining risk factors "
        "in practical, support-oriented language appropriate to that domain."
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
            direction = "increases positive-class probability"
        elif mean_shap < 0:
            direction = "decreases positive-class probability"
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
    rows = sorted(rows, key=lambda item: item["mean_abs"], reverse=True)[:5]
    return "\n".join(item["line"] for item in rows)


def recurring_top_predictor_summary(
    holdout_row: pd.Series,
    retrieved: pd.DataFrame,
    shap_table: pd.DataFrame,
    data_dictionary: Optional[Dict[str, str]] = None,
    top_n: int = 5,
    per_case_top_k: int = 5,
) -> str:
    data_dictionary = data_dictionary or {}
    relevant = shap_table[shap_table["case_id"].isin(retrieved["case_id"])].copy()
    if "rank" in relevant.columns:
        relevant = relevant[relevant["rank"] <= per_case_top_k].copy()
    if relevant.empty:
        return "- No recurring top-predictor rows were available."
    rows = []
    total_cases = max(int(retrieved["case_id"].nunique()), 1)
    for feature, group in relevant.groupby("feature"):
        holdout_value = holdout_row.get(feature, "")
        peer_values = group["value"].dropna().astype(str).value_counts().head(3)
        case_count = int(group["case_id"].nunique())
        mean_abs = float(group["abs_shap_score"].mean())
        mean_shap = float(group["shap_score"].mean())
        if mean_shap > 0:
            direction = "usually increases positive-class probability"
        elif mean_shap < 0:
            direction = "usually decreases positive-class probability"
        else:
            direction = "has mixed overall direction"
        definition = data_dictionary.get(str(feature), "")
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
    holdout_features = []
    for feature, value in holdout_row.items():
        definition = data_dictionary.get(str(feature), "")
        suffix = f" ({definition})" if definition else ""
        holdout_features.append(f"- {feature}{suffix}: {value}")

    peer_blocks = []
    for _, peer in retrieved.iterrows():
        peer_shap = shap_table[shap_table["case_id"] == peer["case_id"]].head(5)
        units = (
            peer_shap["shap_output_units"].dropna().iloc[0]
            if "shap_output_units" in peer_shap.columns and not peer_shap["shap_output_units"].dropna().empty
            else "probability"
        )
        shap_lines = [
            f"  - {row.feature}: value={row.value}, SHAP contribution to positive-class probability={row.shap_score:.4f} ({units})"
            for row in peer_shap.itertuples()
        ]
        peer_blocks.append(
            "\n".join(
                [
                    f"Case {peer.case_id}: outcome={peer.predicted_outcome}, "
                    f"similarity={peer.similarity:.3f}",
                    *shap_lines,
                ]
            )
        )

    weighted_alignment = shap_weighted_alignment_summary(holdout_row, retrieved, shap_table)
    recurring_predictors = recurring_top_predictor_summary(
        holdout_row,
        retrieved,
        shap_table,
        data_dictionary=data_dictionary,
        top_n=5,
        per_case_top_k=5,
    )
    labels = f"{training.positive_label} vs. {training.negative_label}"
    if "predicted_class" in retrieved.columns:
        positive_neighbor_count = int((retrieved["predicted_class"] == 1).sum())
        negative_neighbor_count = int((retrieved["predicted_class"] == 0).sum())
    else:
        positive_neighbor_count = int((retrieved["predicted_outcome"].astype(str) == str(training.positive_label)).sum())
        negative_neighbor_count = int((retrieved["predicted_outcome"].astype(str) == str(training.negative_label)).sum())
    n_total = int(len(retrieved))
    top_similarity = float(retrieved["similarity"].max()) if not retrieved.empty else 0.0
    baseline_positive_probability = float(training.best_run.pipeline.predict_proba(pd.DataFrame([holdout_row]))[0, 1])
    baseline_predicted_class = int(baseline_positive_probability >= 0.5)
    baseline_predicted_outcome = training.positive_label if baseline_predicted_class == 1 else training.negative_label
    baseline_confidence = max(baseline_positive_probability, 1 - baseline_positive_probability)
    positive_class_description = prediction_goal or f"cases with outcome label {training.positive_label}"
    return f"""You are an experienced domain analyst working in the field implied by the prediction goal below. You specialize in identifying cases at risk of the outcome of concern and explaining risk factors in practical, support-oriented language appropriate to that domain.
You are using the E2R2 framework: Explain, Embed, Retrieve, and Reason.
E2R2 combines SHAP feature attributions, similarity-based precedent retrieval, and professional reasoning. The retrieved precedents come from a case base of correctly predicted, high-confidence holdout cases. Your job is to decide the outcome using precedent-based evidence, then verify that decision against an independent baseline machine-learning model's prediction.

Prediction goal:
{prediction_goal or f"Predict {training.target_column}. Labels: {labels}."}

Positive class specification (CRITICAL — read this before any reasoning):
The positive class label for this task is provided explicitly below as positive_class_label. Use this label exactly as given. Do not infer the positive class from the retrieved precedent outcomes or from any other signal in the prompt. In particular:
- The positive class is NOT "whichever label the retrieved precedents happen to share." High-confidence retrieval can return all-one-class neighbors for either class, and that does not change which label is positive.
- A precedent outcome value equal to positive_class_label means that precedent is a positive-class case. A precedent outcome value not equal to positive_class_label means that precedent is a negative-class case.
- "pos_share" in the decision rule is the share of retrieved precedents whose outcome equals positive_class_label, regardless of which label that happens to be (0, 1, "churn", "default", etc.).
- All SHAP values are explained against probability of positive_class_label, as stated in the SHAP interpretation section below.

positive_class_label: {training.positive_label}
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
- Count how many such markers are present in the holdout. If 2 or more markers are present, lean toward the positive verdict. If 0 or 1 markers are present, lean toward the negative verdict. Name explicitly in the Stage 1 rationale which markers triggered, which heuristic identified each, and which did not trigger.

SHAP interpretation:
- All SHAP values are explained against the probability of positive_class_label (as specified at the top of the prompt), not against each precedent's own outcome.
- A positive SHAP contribution increases the modeled probability of positive_class_label; a negative SHAP contribution decreases it.
- Larger absolute SHAP values indicate stronger model contributions.

Cautions when comparing holdout values to precedent values:
- A holdout value that is better than the precedent cluster but still in an absolute risk zone is NOT protective. If the precedents have very extreme risk-side values on a key predictor and the holdout has a moderately-less-extreme but still risk-side value, treat this as a milder version of the same risk pattern, not as a protective signal.
- Only call a difference protective when the holdout value crosses into a clearly healthy or low-risk range on that variable by standard domain understanding, not merely when it is relatively better than the neighbors. State the healthy range you are using when you invoke protection.
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
Rule B — Confident baseline override. If agreement is FALSE AND baseline_confidence is at least 0.70, override to match the baseline. final_verdict = baseline_predicted_outcome. final_confidence = moderate (never high under override, since two independent signals disagreed). This rule applies symmetrically — a confident baseline whose predicted label is not positive_class_label overrides a Stage 1 positive verdict toward the negative class, and a confident baseline whose predicted label equals positive_class_label overrides a Stage 1 negative verdict toward the positive class.
Rule C — Weak baseline, keep Stage 1. If agreement is FALSE AND baseline_confidence is below 0.60, keep the Stage 1 verdict. final_verdict = stage1_verdict. final_confidence = stage1_confidence, but capped at moderate (downgrade high to moderate; leave moderate and low unchanged).
Rule D — Borderline disagreement. If agreement is FALSE AND baseline_confidence is between 0.60 and 0.70 (inclusive on the lower bound, exclusive on the upper), keep the Stage 1 verdict but downgrade final_confidence to low. Note explicitly in the final rationale that the baseline model disagreed at borderline confidence.

Do not invoke any rule other than the four above. Do not override based on the Stage 1 rationale text or the Stage 1 confidence — only the agreement check and the baseline_confidence value drive the verification.

Final rationale composition:
- If Rule A fired, the final rationale is the Stage 1 rationale, with a one-sentence note that the baseline model agreed (state baseline_confidence). The note should affirm agreement even if baseline_confidence is low; a low-confidence agreement is still agreement.
- If Rule B fired, the final rationale should keep the Stage 1 evidence summary but append a sentence explaining that the baseline model disagreed at confident strength (state baseline_confidence) and that the verification step adopted the baseline verdict.
- If Rule C fired, the final rationale is the Stage 1 rationale, with a one-sentence note that the baseline model disagreed at low confidence (state baseline_confidence) and was not strong enough to override.
- If Rule D fired, the final rationale is the Stage 1 rationale, with a one-sentence note that the baseline model disagreed at borderline confidence (state baseline_confidence), which was not sufficient to override but reduced confidence in the final verdict.

=================================================================
INPUTS
=================================================================

Holdout case profile:
{chr(10).join(holdout_features)}

Retrieved E2R2 precedent cases:
{chr(10).join(peer_blocks)}

Recurring top predictors across retrieved precedents:
{recurring_predictors}

SHAP-weighted alignment summary across retrieved cases:
{weighted_alignment}

Precedent outcome counts:
- N={n_total}
- n_pos={positive_neighbor_count}
- pos_share={(100.0 * positive_neighbor_count / max(n_total, 1)):.1f}%
- {positive_neighbor_count} positive ({training.positive_label}); {negative_neighbor_count} negative ({training.negative_label})
- top retrieval similarity={top_similarity:.3f}

Baseline model output:
- baseline_predicted_outcome: {baseline_predicted_outcome}
- baseline_confidence: {baseline_confidence:.4f}

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
  "verification_rule_applied": "A | B | C | D",
  "final_predicted_outcome": "one of the domain labels",
  "final_confidence_level": "low | moderate | high",
  "final_rationale": "the Stage 1 rationale plus the appropriate verification note as described in the Final rationale composition section"
}}

Base the rationale on the evidence shown in the prompt. The true outcome is not provided for reasoning.
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
