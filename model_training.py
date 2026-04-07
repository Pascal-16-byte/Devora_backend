"""
Reusable training pipeline for the Devora productivity classifier.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from .data_generation import generate_dataset
except ImportError:
    from data_generation import generate_dataset

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "feature_importance.png"

FEATURES = [
    "coding_hours",
    "learning_time",
    "commits_per_day",
    "lines_of_code",
    "bugs_fixed",
    "sleep_hours",
    "distractions",
    "deep_work_ratio",
    "distraction_intensity",
    "task_completion_rate",
    "meetings_per_day",
    "break_time",
    "focus_score",
    "avg_focus_last_7_days",
    "avg_distraction_last_7_days",
    "personal_baseline_score",
]
TARGET = "productivity_level"
CLASSES = ["Low", "Medium", "High"]
USER_DATA_REDUCED_SYNTHETIC_THRESHOLD = 500
USER_DATA_ONLY_THRESHOLD = 2000
BASELINE_BY_CLASS = {
    "Low": 35.0,
    "Medium": 60.0,
    "High": 85.0,
}


def load_base_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    if dataset_path.exists():
        return pd.read_csv(dataset_path)
    return generate_dataset(save_path=str(dataset_path))


def _numeric_series(df: pd.DataFrame, column_name: str, default: float = float("nan")) -> pd.Series:
    if column_name in df.columns:
        return pd.to_numeric(df[column_name], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _derive_personalization_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    focus_series = _numeric_series(enriched, "focus_score")
    distraction_series = _numeric_series(enriched, "distractions")
    score_series = _numeric_series(enriched, "productivity_score")
    coding_series = _numeric_series(enriched, "coding_hours", default=0.0).fillna(0.0)
    task_series = _numeric_series(enriched, "task_completion_rate", default=0.0).fillna(0.0)
    meetings_series = _numeric_series(enriched, "meetings_per_day", default=0.0).fillna(0.0)

    if TARGET in enriched.columns:
        target_source = enriched[TARGET]
    elif "predicted_label" in enriched.columns:
        target_source = enriched["predicted_label"]
    elif "productivity_level" in enriched.columns:
        target_source = enriched["productivity_level"]
    else:
        target_source = pd.Series([""] * len(enriched), index=enriched.index, dtype="string")

    baseline_from_label = (
        pd.Series(target_source, index=enriched.index, dtype="string").map(BASELINE_BY_CLASS)
    )
    baseline_series = score_series.fillna(pd.to_numeric(baseline_from_label, errors="coerce"))

    derived_learning = (
        _numeric_series(enriched, "learning_time")
        .fillna((coding_series * 0.18).clip(lower=0.0, upper=4.0))
    )
    derived_deep_work = (
        _numeric_series(enriched, "deep_work_ratio")
        .fillna(
            (
                (coding_series / 8.0) * 0.45
                + task_series * 0.4
                + (focus_series.fillna(50.0) / 100.0) * 0.25
                - (meetings_series / 12.0) * 0.1
            ).clip(lower=0.0, upper=1.0)
        )
    )
    derived_distraction_intensity = (
        _numeric_series(enriched, "distraction_intensity")
        .fillna((distraction_series.fillna(0.0) * 1.1).clip(lower=0.0, upper=30.0))
    )

    derived_features = {
        "learning_time": derived_learning,
        "deep_work_ratio": derived_deep_work,
        "distraction_intensity": derived_distraction_intensity,
        "avg_focus_last_7_days": focus_series,
        "avg_distraction_last_7_days": distraction_series,
        "personal_baseline_score": baseline_series,
    }

    for feature_name, derived_series in derived_features.items():
        if feature_name in enriched.columns:
            existing_series = pd.to_numeric(enriched[feature_name], errors="coerce")
            enriched[feature_name] = existing_series.fillna(derived_series)
        else:
            enriched[feature_name] = derived_series

    return enriched


def sanitize_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sanitized = _derive_personalization_features(df)

    for feature in FEATURES:
        sanitized[feature] = _numeric_series(sanitized, feature)

    target_series = None
    for candidate in (TARGET, "predicted_label", "productivity_level"):
        if candidate in sanitized.columns:
            target_series = sanitized[candidate]
            break

    if target_series is None:
        raise KeyError(
            f"Training dataframe must include one of: {TARGET}, predicted_label, productivity_level."
        )

    sanitized[TARGET] = target_series.astype("string").str.strip()
    sanitized[TARGET] = sanitized[TARGET].where(sanitized[TARGET].isin(CLASSES))

    sanitized = sanitized.dropna(subset=[TARGET]).copy()

    for feature in FEATURES:
        median_value = sanitized[feature].median()
        fill_value = 0.0 if pd.isna(median_value) else float(median_value)
        sanitized[feature] = sanitized[feature].fillna(fill_value)

    return sanitized[FEATURES + [TARGET]]


def _apply_dataframe_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    if df.empty or weight <= 0:
        return df.iloc[0:0].copy()

    whole_copies = int(weight)
    fractional = max(0.0, weight - whole_copies)
    frames: list[pd.DataFrame] = []

    if whole_copies > 0:
        frames.extend(df.copy() for _ in range(whole_copies))
    if fractional > 0:
        sample_count = max(1, int(round(len(df) * fractional)))
        frames.append(df.sample(n=min(sample_count, len(df)), random_state=42, replace=False).copy())

    if not frames:
        frames.append(df.copy())

    return pd.concat(frames, ignore_index=True)


def compute_user_weight(user_samples: int) -> float:
    if user_samples < 100:
        return 1.0
    if user_samples < 500:
        return 2.5
    if user_samples < 1000:
        return 4.0
    return 6.0


def merge_training_sources(
    base_df: pd.DataFrame,
    user_df: pd.DataFrame | None = None,
    user_weight: float | None = None,
    reduced_synthetic_threshold: int = USER_DATA_REDUCED_SYNTHETIC_THRESHOLD,
    user_only_threshold: int = USER_DATA_ONLY_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    synthetic_df = sanitize_training_dataframe(base_df)
    user_clean_df = (
        sanitize_training_dataframe(user_df)
        if user_df is not None and not user_df.empty
        else pd.DataFrame(columns=[*FEATURES, TARGET])
    )

    raw_synthetic_samples = len(synthetic_df)
    raw_user_samples = len(user_clean_df)
    synthetic_strategy = "full"
    effective_weight = float(user_weight) if user_weight is not None else compute_user_weight(raw_user_samples)

    if raw_user_samples > user_only_threshold:
        merged_df = user_clean_df.reset_index(drop=True)
        synthetic_df = synthetic_df.iloc[0:0].copy()
        effective_user_df = user_clean_df.reset_index(drop=True)
        synthetic_strategy = "user_only"
    else:
        if raw_user_samples > reduced_synthetic_threshold and not synthetic_df.empty:
            synthetic_df = synthetic_df.sample(
                n=max(1, len(synthetic_df) // 2),
                random_state=42,
                replace=False,
            ).reset_index(drop=True)
            synthetic_strategy = "reduced_50pct"

        effective_user_df = _apply_dataframe_weight(user_clean_df, effective_weight)
        frames = [synthetic_df]
        if not effective_user_df.empty:
            frames.append(effective_user_df)
        merged_df = pd.concat(frames, ignore_index=True)

    synthetic_samples = len(synthetic_df)
    user_samples = len(effective_user_df)
    total_samples = synthetic_samples + user_samples
    metadata = {
        "synthetic_samples": synthetic_samples,
        "user_samples": user_samples,
        "user_ratio": round(user_samples / total_samples, 4) if total_samples else 0.0,
        "raw_synthetic_samples": raw_synthetic_samples,
        "raw_user_samples": raw_user_samples,
        "effective_user_weight": effective_weight,
        "synthetic_strategy": synthetic_strategy,
    }
    return merged_df, metadata


def preprocess(df: pd.DataFrame):
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)
    y = label_encoder.transform(df[TARGET])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[FEATURES].values)
    return x_scaled, y, scaler, label_encoder


def train_model(x_train, y_train) -> RandomForestClassifier:
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    classifier.fit(x_train, y_train)
    return classifier


def evaluate_model(classifier, x_test, y_test, label_encoder: LabelEncoder):
    labels = list(range(len(label_encoder.classes_)))
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n[OK] Test Accuracy: {accuracy * 100:.2f}%\n")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    class_counts = pd.Series(y_test).value_counts()
    cv_folds = min(5, len(y_test), int(class_counts.min())) if not class_counts.empty else 0
    if cv_folds >= 2:
        cv_scores = cross_val_score(classifier, x_test, y_test, cv=cv_folds)
        print(
            f"[OK] {cv_folds}-Fold CV Accuracy: {cv_scores.mean() * 100:.2f}% +/- {cv_scores.std() * 100:.2f}%"
        )
    else:
        print("[OK] Cross-validation skipped: insufficient class coverage in evaluation split.")

    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    feature_importances = classifier.feature_importances_

    _save_confusion_matrix(matrix, label_encoder)
    _save_feature_importance_plot(feature_importances)

    return accuracy, matrix, feature_importances


def _save_confusion_matrix(matrix, label_encoder: LabelEncoder) -> None:
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - Productivity Classifier")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()


def _save_feature_importance_plot(feature_importances) -> None:
    feature_frame = pd.DataFrame({"feature": FEATURES, "importance": feature_importances})
    feature_frame = feature_frame.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feature_frame["feature"], feature_frame["importance"], color="#0ea5e9")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close()


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict[str, Any] | None:
    if not model_path.exists():
        return None

    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


def _next_model_version(previous_bundle: dict[str, Any] | None) -> str:
    if not previous_bundle:
        return "v1"

    previous_version = str(previous_bundle.get("version", "v1"))
    try:
        version_number = int(previous_version.lstrip("v"))
    except ValueError:
        version_number = 1
    return f"v{version_number + 1}"


def build_model_bundle(
    classifier,
    scaler,
    label_encoder: LabelEncoder,
    accuracy: float,
    matrix,
    feature_importances,
    total_training_samples: int,
    dataset_composition: dict[str, Any] | None = None,
    previous_bundle: dict[str, Any] | None = None,
    latest_user_timestamp: str | None = None,
) -> dict[str, Any]:
    previous_accuracy = None
    if previous_bundle:
        previous_accuracy = previous_bundle.get("accuracy")

    return {
        "model": classifier,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "features": FEATURES,
        "classes": CLASSES,
        "accuracy": float(accuracy),
        "previous_accuracy": (
            float(previous_accuracy) if previous_accuracy is not None else None
        ),
        "confusion_matrix": matrix.tolist(),
        "feature_importances": dict(zip(FEATURES, feature_importances.tolist())),
        "version": _next_model_version(previous_bundle),
        "trained_on": int(total_training_samples),
        "total_training_samples": int(total_training_samples),
        "training_samples": int(total_training_samples),
        "model_name": "RandomForestClassifier (n_estimators=200)",
        "last_training_data_timestamp": latest_user_timestamp,
        "dataset_composition": dataset_composition or {},
    }


def save_model_bundle(bundle: dict[str, Any], model_path: Path = MODEL_PATH) -> None:
    with open(model_path, "wb") as model_file:
        pickle.dump(bundle, model_file)


def run_training_pipeline(
    training_df: pd.DataFrame,
    dataset_composition: dict[str, Any] | None = None,
    previous_bundle: dict[str, Any] | None = None,
    model_path: Path = MODEL_PATH,
    latest_user_timestamp: str | None = None,
) -> dict[str, Any]:
    sanitized = sanitize_training_dataframe(training_df)
    x_scaled, y, scaler, label_encoder = preprocess(sanitized)

    stratify_labels = y if len(set(y)) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels,
    )

    print(f"[OK] Split: {len(x_train)} train / {len(x_test)} test samples")

    classifier = train_model(x_train, y_train)
    accuracy, matrix, feature_importances = evaluate_model(
        classifier,
        x_test,
        y_test,
        label_encoder,
    )

    bundle = build_model_bundle(
        classifier=classifier,
        scaler=scaler,
        label_encoder=label_encoder,
        accuracy=accuracy,
        matrix=matrix,
        feature_importances=feature_importances,
        total_training_samples=len(sanitized),
        dataset_composition=dataset_composition,
        previous_bundle=previous_bundle,
        latest_user_timestamp=latest_user_timestamp,
    )
    save_model_bundle(bundle, model_path=model_path)
    return bundle


def run_pipeline() -> float:
    print("=" * 55)
    print("  Devora ML Pipeline")
    print("=" * 55)

    base_df = load_base_dataset()
    bundle = run_training_pipeline(base_df, previous_bundle=load_model_bundle())
    print("\n[OK] Pipeline complete!")
    return float(bundle["accuracy"])


if __name__ == "__main__":
    run_pipeline()
