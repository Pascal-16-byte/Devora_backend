"""
User-specific model selection and lightweight retraining helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any

from sklearn.ensemble import RandomForestClassifier

try:
    from .database import get_user_training_data
    from .model_training import (
        CLASSES,
        FEATURES,
        TARGET,
        preprocess,
        sanitize_training_dataframe,
    )
except ImportError:
    from database import get_user_training_data
    from model_training import (
        CLASSES,
        FEATURES,
        TARGET,
        preprocess,
        sanitize_training_dataframe,
    )

USER_MODEL_MIN_SAMPLES = 300
USER_MODEL_REFRESH_SAMPLES = 25
USER_MODEL_NAME = "PersonalRandomForestClassifier (n_estimators=80)"

user_models: dict[str, dict[str, Any]] = {}
_user_model_lock = Lock()


def _train_user_classifier(x_train, y_train) -> RandomForestClassifier:
    classifier = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    classifier.fit(x_train, y_train)
    return classifier


def _build_user_bundle(user_id: str, user_df) -> dict[str, Any] | None:
    sanitized = sanitize_training_dataframe(user_df)
    if len(sanitized) < USER_MODEL_MIN_SAMPLES:
        return None
    if sanitized[TARGET].nunique() < 2:
        return None

    x_scaled, y, scaler, label_encoder = preprocess(sanitized)
    classifier = _train_user_classifier(x_scaled, y)
    latest_timestamp = (
        user_df["timestamp"].max() if "timestamp" in user_df.columns and not user_df.empty else None
    )

    return {
        "model": classifier,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "features": FEATURES,
        "classes": CLASSES,
        "accuracy": None,
        "previous_accuracy": None,
        "confusion_matrix": [],
        "feature_importances": dict(zip(FEATURES, classifier.feature_importances_.tolist())),
        "version": f"user-{user_id}",
        "trained_on": int(len(sanitized)),
        "total_training_samples": int(len(sanitized)),
        "training_samples": int(len(sanitized)),
        "model_name": USER_MODEL_NAME,
        "last_training_data_timestamp": latest_timestamp,
        "dataset_composition": {
            "raw_user_samples": int(len(user_df)),
            "user_samples": int(len(sanitized)),
            "synthetic_samples": 0,
            "user_ratio": 1.0,
            "synthetic_strategy": "user_only",
            "effective_user_weight": 1.0,
        },
        "user_id": user_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def get_personalized_model(user_id: str | None, global_bundle: dict[str, Any]) -> dict[str, Any]:
    if not user_id:
        return {
            "bundle": global_bundle,
            "personalization": {
                "model_used": "global",
                "data_points_used": int(global_bundle.get("trained_on", 0)),
                "user_data_points": 0,
                "cache_hit": False,
                "reason": "missing_user_id",
            },
        }

    user_df = get_user_training_data(user_id)
    raw_user_samples = len(user_df)
    if raw_user_samples < USER_MODEL_MIN_SAMPLES:
        return {
            "bundle": global_bundle,
            "personalization": {
                "model_used": "global",
                "data_points_used": int(global_bundle.get("trained_on", 0)),
                "user_data_points": raw_user_samples,
                "cache_hit": False,
                "reason": "insufficient_user_samples",
            },
        }

    with _user_model_lock:
        cached_entry = user_models.get(user_id)
        if cached_entry:
            cached_sample_count = int(cached_entry.get("sample_count", 0))
            if raw_user_samples - cached_sample_count < USER_MODEL_REFRESH_SAMPLES:
                return {
                    "bundle": cached_entry["bundle"],
                    "personalization": {
                        "model_used": "user",
                        "data_points_used": cached_sample_count,
                        "user_data_points": raw_user_samples,
                        "cache_hit": True,
                        "reason": "cached_user_model",
                    },
                }

    bundle = _build_user_bundle(user_id, user_df)
    if bundle is None:
        return {
            "bundle": global_bundle,
            "personalization": {
                "model_used": "global",
                "data_points_used": int(global_bundle.get("trained_on", 0)),
                "user_data_points": raw_user_samples,
                "cache_hit": False,
                "reason": "insufficient_label_variance",
            },
        }

    cache_entry = {
        "bundle": bundle,
        "sample_count": int(bundle["trained_on"]),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with _user_model_lock:
        user_models[user_id] = cache_entry

    return {
        "bundle": bundle,
        "personalization": {
            "model_used": "user",
            "data_points_used": int(bundle["trained_on"]),
            "user_data_points": raw_user_samples,
            "cache_hit": False,
            "reason": "trained_user_model",
        },
    }


def update_user_model(user_id: str | None) -> dict[str, Any]:
    if not user_id:
        return {"status": "skipped", "reason": "missing_user_id"}

    user_df = get_user_training_data(user_id)
    raw_user_samples = len(user_df)
    if raw_user_samples < USER_MODEL_MIN_SAMPLES:
        return {
            "status": "skipped",
            "reason": "insufficient_user_samples",
            "user_data_points": raw_user_samples,
        }

    with _user_model_lock:
        cached_entry = user_models.get(user_id)
        if cached_entry:
            cached_sample_count = int(cached_entry.get("sample_count", 0))
            if raw_user_samples - cached_sample_count < USER_MODEL_REFRESH_SAMPLES:
                return {
                    "status": "cached",
                    "reason": "refresh_threshold_not_reached",
                    "user_data_points": raw_user_samples,
                }

    bundle = _build_user_bundle(user_id, user_df)
    if bundle is None:
        return {
            "status": "skipped",
            "reason": "insufficient_label_variance",
            "user_data_points": raw_user_samples,
        }

    with _user_model_lock:
        user_models[user_id] = {
            "bundle": bundle,
            "sample_count": int(bundle["trained_on"]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    return {
        "status": "updated",
        "reason": "user_model_refreshed",
        "user_data_points": raw_user_samples,
        "trained_on": int(bundle["trained_on"]),
    }


def clear_user_model_cache(user_id: str | None = None) -> None:
    with _user_model_lock:
        if user_id:
            user_models.pop(user_id, None)
        else:
            user_models.clear()
