"""
Retraining entrypoint for continuous-learning Devora models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .database import get_new_data_count, get_training_data
    from .model_training import (
        DATASET_PATH,
        MODEL_PATH,
        load_base_dataset,
        load_model_bundle,
        merge_training_sources,
        run_training_pipeline,
    )
except ImportError:
    from database import get_new_data_count, get_training_data
    from model_training import (
        DATASET_PATH,
        MODEL_PATH,
        load_base_dataset,
        load_model_bundle,
        merge_training_sources,
        run_training_pipeline,
    )

MIN_NEW_SAMPLES_FOR_RETRAIN = 100


def _format_improvement(old_accuracy: float | None, new_accuracy: float | None) -> str:
    if old_accuracy is None or new_accuracy is None:
        return "N/A"
    delta = (new_accuracy - old_accuracy) * 100
    return f"{delta:+.2f}%"


def retrain_model(
    min_new_samples: int = MIN_NEW_SAMPLES_FOR_RETRAIN,
    force: bool = False,
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
    user_weight: float | None = None,
) -> dict[str, Any]:
    existing_bundle = load_model_bundle(model_path)
    previous_accuracy = (
        float(existing_bundle["accuracy"]) if existing_bundle and existing_bundle.get("accuracy") is not None else None
    )
    last_training_data_timestamp = (
        existing_bundle.get("last_training_data_timestamp") if existing_bundle else None
    )

    new_data_count = get_new_data_count(last_training_data_timestamp)
    user_df = get_training_data()

    if not force and new_data_count < min_new_samples:
        base_sample_count = len(load_base_dataset(dataset_path))
        trained_on = (
            int(existing_bundle.get("trained_on") or existing_bundle.get("total_training_samples") or 0)
            if existing_bundle
            else base_sample_count
        )
        if trained_on == 0:
            trained_on = base_sample_count
        return {
            "status": "skipped",
            "reason": f"Need at least {min_new_samples} new samples before retraining.",
            "old_accuracy": previous_accuracy,
            "new_accuracy": previous_accuracy,
            "improvement": "+0.00%" if previous_accuracy is not None else "N/A",
            "version": existing_bundle.get("version", "v1") if existing_bundle else "v1",
            "new_samples": new_data_count,
            "total_training_samples": trained_on,
            "dataset_composition": existing_bundle.get("dataset_composition", {}) if existing_bundle else {},
        }

    base_df = load_base_dataset(dataset_path)
    merged_df, dataset_composition = merge_training_sources(
        base_df=base_df,
        user_df=user_df,
        user_weight=user_weight,
    )
    latest_user_timestamp = (
        user_df["timestamp"].max() if not user_df.empty and "timestamp" in user_df.columns else None
    )
    updated_bundle = run_training_pipeline(
        training_df=merged_df,
        dataset_composition=dataset_composition,
        previous_bundle=existing_bundle,
        model_path=model_path,
        latest_user_timestamp=latest_user_timestamp,
    )

    new_accuracy = float(updated_bundle["accuracy"])
    return {
        "status": "retrained",
        "old_accuracy": previous_accuracy,
        "new_accuracy": new_accuracy,
        "improvement": _format_improvement(previous_accuracy, new_accuracy),
        "version": updated_bundle["version"],
        "new_samples": new_data_count,
        "total_training_samples": int(updated_bundle["trained_on"]),
        "dataset_composition": updated_bundle.get("dataset_composition", dataset_composition),
    }


if __name__ == "__main__":
    result = retrain_model(force=True)
    print(result)
