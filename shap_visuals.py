from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .explainability import build_explainer_bundle, get_shap_values, load_explainer
    from .model_training import sanitize_training_dataframe
except ImportError:
    from explainability import build_explainer_bundle, get_shap_values, load_explainer
    from model_training import sanitize_training_dataframe

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATASET_PATH = BASE_DIR / "dataset.csv"


def _ensure_static_dir():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    return STATIC_DIR


def _resolve_base_value(explainer, pred_idx):
    base_values = np.asarray(explainer.expected_value)
    if base_values.ndim == 0:
        return float(base_values)
    return float(base_values[pred_idx])


def _load_dataset(features):
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"dataset.csv not found at '{DATASET_PATH}'. Run the training pipeline first."
        )
    dataset = pd.read_csv(DATASET_PATH)
    sanitized = sanitize_training_dataframe(dataset)
    return sanitized[features]


def _create_static_url(filename):
    return f"/static/{filename}"


def generate_local_plot(input_df, predicted_class_index, model_path=None, prediction_id=None, bundle=None):
    bundle = build_explainer_bundle(bundle) if bundle is not None else load_explainer(model_path)
    scaler = bundle["scaler"]
    features = bundle["features"]
    explainer = bundle["explainer"]

    static_dir = _ensure_static_dir()
    ordered_input_df = input_df[features].copy()
    scaled_input_df = pd.DataFrame(
        scaler.transform(ordered_input_df.values),
        columns=features,
    )

    shap_values = get_shap_values(scaled_input_df, model_path=model_path, bundle=bundle)
    class_shap_values = shap_values[predicted_class_index][0]
    base_value = _resolve_base_value(explainer, predicted_class_index)

    import shap

    local_explanation = shap.Explanation(
        values=class_shap_values,
        base_values=base_value,
        data=ordered_input_df.iloc[0].values,
        feature_names=features,
    )

    filename = f"shap_local_{prediction_id or uuid4().hex}.png"
    output_path = static_dir / filename

    plt.close("all")
    shap.plots.waterfall(local_explanation, max_display=len(features), show=False)
    figure = plt.gcf()
    figure.set_size_inches(10, 6)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close("all")

    return _create_static_url(filename)


@lru_cache(maxsize=1)
def generate_global_plot(model_path=None):
    bundle = load_explainer(model_path)
    scaler = bundle["scaler"]
    features = bundle["features"]

    static_dir = _ensure_static_dir()
    feature_df = _load_dataset(features)
    scaled_feature_df = pd.DataFrame(
        scaler.transform(feature_df.values),
        columns=features,
    )

    shap_values = get_shap_values(scaled_feature_df, model_path)
    global_values = np.mean(np.abs(shap_values), axis=(0, 1))

    import shap

    global_explanation = shap.Explanation(
        values=global_values,
        feature_names=features,
    )

    filename = f"shap_global_{uuid4().hex}.png"
    output_path = static_dir / filename

    plt.close("all")
    shap.plots.bar(global_explanation, max_display=len(features), show=False)
    figure = plt.gcf()
    figure.set_size_inches(10, 6)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close("all")

    return _create_static_url(filename)
