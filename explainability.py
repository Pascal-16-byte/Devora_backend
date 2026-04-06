import os
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "coding_hours": "coding hours",
    "learning_time": "learning time",
    "commits_per_day": "commits per day",
    "lines_of_code": "lines of code",
    "bugs_fixed": "bugs fixed",
    "sleep_hours": "sleep hours",
    "distractions": "distractions",
    "deep_work_ratio": "deep work ratio",
    "distraction_intensity": "distraction intensity",
    "task_completion_rate": "task completion rate",
    "meetings_per_day": "meetings per day",
    "break_time": "break time",
    "focus_score": "focus score",
    "avg_focus_last_7_days": "7 day average focus",
    "avg_distraction_last_7_days": "7 day average distractions",
    "personal_baseline_score": "personal baseline score",
}

FEATURE_DIRECTIONS = {
    "coding_hours": "high",
    "learning_time": "high",
    "commits_per_day": "high",
    "lines_of_code": "high",
    "bugs_fixed": "high",
    "sleep_hours": "high",
    "distractions": "low",
    "deep_work_ratio": "high",
    "distraction_intensity": "low",
    "task_completion_rate": "high",
    "meetings_per_day": "low",
    "break_time": "low",
    "focus_score": "high",
    "avg_focus_last_7_days": "high",
    "avg_distraction_last_7_days": "low",
    "personal_baseline_score": "high",
}


def _load_shap():
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Add `shap` to backend requirements and install dependencies."
        ) from exc
    return shap


@lru_cache(maxsize=1)
def load_explainer(model_path=None):
    shap = _load_shap()
    resolved_model_path = model_path or os.path.join(os.path.dirname(__file__), "model.pkl")

    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            f"model.pkl not found at '{resolved_model_path}'. Run `python model_training.py` first."
        )

    with open(resolved_model_path, "rb") as model_file:
        bundle = pickle.load(model_file)

    explainer = shap.TreeExplainer(bundle["model"])
    return {
        **bundle,
        "explainer": explainer,
    }


def build_explainer_bundle(bundle):
    shap = _load_shap()
    return {
        **bundle,
        "explainer": shap.TreeExplainer(bundle["model"]),
    }


def get_shap_values(input_df, model_path=None, bundle=None):
    if bundle is not None and "explainer" in bundle:
        explainability_bundle = bundle
    elif bundle is not None:
        explainability_bundle = build_explainer_bundle(bundle)
    else:
        explainability_bundle = load_explainer(model_path)
    explainer = explainability_bundle["explainer"]
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        return np.stack([np.asarray(class_values) for class_values in shap_values], axis=0)

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        return np.transpose(shap_values, (2, 0, 1))

    if shap_values.ndim == 2:
        return shap_values[np.newaxis, ...]

    raise ValueError(f"Unsupported SHAP output shape: {shap_values.shape}")


def normalize_contributions(class_shap_values, feature_names):
    absolute_values = np.abs(class_shap_values)
    total = float(absolute_values.sum())

    if total == 0:
        return {feature: 0.0 for feature in feature_names}

    return {
        feature: round(float(value / total), 4)
        for feature, value in zip(feature_names, absolute_values)
    }


def normalize_signed_contributions(class_shap_values, feature_names):
    absolute_values = np.abs(class_shap_values)
    total = float(absolute_values.sum())

    if total == 0:
        return {feature: 0.0 for feature in feature_names}

    return {
        feature: round(float(value / total), 4)
        for feature, value in zip(feature_names, class_shap_values)
    }


def _describe_feature(feature_name):
    direction = FEATURE_DIRECTIONS.get(feature_name)
    label = FEATURE_LABELS.get(feature_name, feature_name.replace("_", " "))
    return f"{direction} {label}" if direction else label


def _format_feature_list(feature_names):
    described = [_describe_feature(feature_name) for feature_name in feature_names]

    if not described:
        return ""
    if len(described) == 1:
        return described[0]
    if len(described) == 2:
        return f"{described[0]} and {described[1]}"
    return f"{', '.join(described[:-1])}, and {described[-1]}"


def _format_feature_value(feature_value):
    if isinstance(feature_value, (int, float, np.floating)):
        numeric_value = float(feature_value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return f"{numeric_value:.2f}".rstrip("0").rstrip(".")
    return str(feature_value)


def _format_ranked_impacts(ranked_items, input_row):
    formatted = []

    for feature_name, contribution in ranked_items:
        raw_value = _format_feature_value(input_row[feature_name])
        formatted.append(
            f"{FEATURE_LABELS.get(feature_name, feature_name.replace('_', ' '))} "
            f"({raw_value}, {contribution:+.1%})"
        )

    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"


def generate_explanation(shap_values, input_features, predicted_class="Medium", top_k=3):
    if isinstance(input_features, pd.DataFrame):
        feature_names = list(input_features.columns)
        input_row = input_features.iloc[0].to_dict()
    else:
        feature_names = list(input_features.keys())
        input_row = dict(input_features)

    class_shap_values = np.asarray(shap_values, dtype=float).reshape(-1)
    contributions = dict(zip(feature_names, class_shap_values))

    signed_contributions = normalize_signed_contributions(class_shap_values, feature_names)
    positive_impacts = [
        (feature_name, contribution)
        for feature_name, contribution in sorted(
            signed_contributions.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if contribution > 0
    ][:top_k]

    negative_impacts = [
        (feature_name, contribution)
        for feature_name, contribution in sorted(
            signed_contributions.items(),
            key=lambda item: item[1],
        )
        if contribution < 0
    ][:top_k]

    level_text = str(predicted_class).lower()
    explanation_parts = []

    if positive_impacts:
        explanation_parts.append(
            f"Your productivity is {level_text}. The strongest upward SHAP pushes came from "
            f"{_format_ranked_impacts(positive_impacts, input_row)}."
        )
    else:
        explanation_parts.append(
            f"Your productivity is {level_text} based on a balanced mix of your current work signals."
        )

    if negative_impacts:
        explanation_parts.append(
            f"The main downward pulls came from {_format_ranked_impacts(negative_impacts, input_row)}."
        )

    explanation = " ".join(explanation_parts[:2])
    return {
        "explanation": explanation,
        "top_positive_features": [feature_name for feature_name, _ in positive_impacts],
        "top_negative_features": [feature_name for feature_name, _ in negative_impacts],
        "feature_importance": normalize_contributions(class_shap_values, feature_names),
        "feature_contributions": signed_contributions,
    }
