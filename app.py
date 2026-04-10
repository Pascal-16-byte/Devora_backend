"""
FastAPI backend for Devora with continuous learning and realtime telemetry.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from .coaching_engine import generate_coaching_feedback
    from .database import (
        get_latest_activity_log,
        get_new_data_count,
        get_recent_activity_logs,
        get_recent_prediction_events,
        get_total_prediction_count,
        get_user_personalization_context,
        init_db,
        save_activity_log,
        save_prediction,
        save_prediction_event,
    )
    from .explainability import generate_explanation, get_shap_values, load_explainer
    from .insights_engine import generate_comparative_insights
    from .pattern_engine import generate_behavioral_patterns
    from .performance_cache import build_cache_key, create_cache_manager
    from .personalization import clear_user_model_cache, get_personalized_model, update_user_model
    from .proactive_engine import generate_proactive_alerts
    from .model_training import DATASET_PATH, FEATURES as TRAINING_FEATURES, load_model_bundle
    from .retrain_model import MIN_NEW_SAMPLES_FOR_RETRAIN, retrain_model
    from .shap_visuals import STATIC_DIR, generate_global_plot, generate_local_plot
    from .simulation_engine import run_what_if_analysis
    from .temporal_engine import generate_temporal_insights
except ImportError:
    from coaching_engine import generate_coaching_feedback
    from database import (
        get_latest_activity_log,
        get_new_data_count,
        get_recent_activity_logs,
        get_recent_prediction_events,
        get_total_prediction_count,
        get_user_personalization_context,
        init_db,
        save_activity_log,
        save_prediction,
        save_prediction_event,
    )
    from explainability import generate_explanation, get_shap_values, load_explainer
    from insights_engine import generate_comparative_insights
    from pattern_engine import generate_behavioral_patterns
    from performance_cache import build_cache_key, create_cache_manager
    from personalization import clear_user_model_cache, get_personalized_model, update_user_model
    from proactive_engine import generate_proactive_alerts
    from model_training import DATASET_PATH, FEATURES as TRAINING_FEATURES, load_model_bundle
    from retrain_model import MIN_NEW_SAMPLES_FOR_RETRAIN, retrain_model
    from shap_visuals import STATIC_DIR, generate_global_plot, generate_local_plot
    from simulation_engine import run_what_if_analysis
    from temporal_engine import generate_temporal_insights

MODEL_NAME = "RandomForestClassifier (n_estimators=200)"
AUTO_RETRAIN_THRESHOLD = MIN_NEW_SAMPLES_FOR_RETRAIN
MODEL_PATH = DATASET_PATH.with_name("model.pkl")
LOGGER = logging.getLogger("devora.api")
CACHE = create_cache_manager()

PREDICTION_CORE_CACHE_PREFIX = "prediction-core:"
MODEL_STATS_CACHE_PREFIX = "model-stats:"
PERSONALIZATION_CONTEXT_CACHE_PREFIX = "personalization-context:"
PERSONAL_INSIGHTS_CACHE_PREFIX = "personal-insights:"
RECENT_EVENTS_CACHE_PREFIX = "recent-events:"
REALTIME_OVERVIEW_CACHE_PREFIX = "realtime-overview:"
LATEST_ACTIVITY_CACHE_PREFIX = "latest-activity:"
RECENT_ACTIVITY_CACHE_PREFIX = "recent-activity:"

PREDICTION_CORE_CACHE_TTL_SECONDS = 300
PERSONALIZATION_CONTEXT_CACHE_TTL_SECONDS = 120
MODEL_STATS_CACHE_TTL_SECONDS = 300
PERSONAL_INSIGHTS_CACHE_TTL_SECONDS = 12
REALTIME_OVERVIEW_CACHE_TTL_SECONDS = 3
LATEST_ACTIVITY_CACHE_TTL_SECONDS = 3
RECENT_ACTIVITY_CACHE_TTL_SECONDS = 3
RECENT_EVENTS_CACHE_TTL_SECONDS = 4

STATIC_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_allowed_origins() -> list[str]:
    configured = os.getenv("DEVORA_ALLOWED_ORIGINS", "").strip()
    if configured:
        origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
        if origins:
            return origins

    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]


ALLOWED_ORIGINS = _resolve_allowed_origins()
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    try:
        yield
    finally:
        _cleanup_inactive_trackers()
        for tracker_id in list(active_trackers.keys()):
            _stop_tracker_process(tracker_id)


app = FastAPI(
    title="Devora API",
    description="AI-powered programmer productivity analyzer with continuous learning",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=800)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_retrain_lock = Lock()
TRACKER_SCRIPT_PATH = Path(__file__).resolve().parent / "tracker.py"
_TRACKER_SUPPRESS_OUTPUT = subprocess.DEVNULL
active_trackers: dict[str, dict[str, Any]] = {}


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections = [conn for conn in self._connections if conn is not websocket]

    async def broadcast(self, message: dict[str, Any]) -> None:
        async with self._lock:
            connections = list(self._connections)

        stale: list[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                stale.append(connection)

        if stale:
            async with self._lock:
                self._connections = [conn for conn in self._connections if conn not in stale]


ws_manager = ConnectionManager()


@app.middleware("http")
async def add_request_timing(request, call_next):
    started_at = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    LOGGER.info(
        "request path=%s method=%s status=%s duration_ms=%.2f",
        request.url.path,
        request.method,
        response.status_code,
        elapsed_ms,
    )
    return response


class ProgrammerMetrics(BaseModel):
    coding_hours: float = Field(..., ge=0, le=24, description="Daily coding hours")
    learning_time: float = Field(0, ge=0, le=24, description="Daily learning hours")
    commits_per_day: float = Field(..., ge=0, le=50, description="Git commits per day")
    lines_of_code: float = Field(..., ge=0, le=5000, description="Lines of code written")
    bugs_fixed: float = Field(..., ge=0, le=50, description="Bugs fixed")
    sleep_hours: float = Field(..., ge=0, le=12, description="Sleep hours last night")
    distractions: float = Field(..., ge=0, le=30, description="Number of distractions")
    deep_work_ratio: float = Field(0, ge=0, le=1, description="Ratio of time spent in deep work")
    distraction_intensity: float = Field(0, ge=0, le=30, description="Context-weighted distraction intensity")
    task_completion_rate: float = Field(
        ...,
        ge=0,
        le=1,
        description="Task completion rate (0-1)",
    )
    meetings_per_day: float = Field(..., ge=0, le=20, description="Meetings per day")
    break_time: float = Field(..., ge=0, le=480, description="Total break time (minutes)")
    focus_score: float = Field(..., ge=0, le=100, description="Self-rated focus score")


class PredictRequest(ProgrammerMetrics):
    user_id: str | None = None


class SimulationRequest(PredictRequest):
    pass


class RealtimePredictRequest(PredictRequest):
    tracker_id: str | None = None
    source: str = Field(default="tracker")


class BatchPredictItem(PredictRequest):
    pass


class ActivitySnapshot(BaseModel):
    tracker_id: str | None = None
    app_name: str
    app_display_name: str | None = None
    category: str
    active_window: str
    window_title: str
    context: dict[str, Any] = Field(default_factory=dict)
    cpu_percent: float = 0
    memory_mb: float = 0
    idle_seconds: float = 0
    is_idle: bool = False
    session_seconds: float = 0
    coding_seconds: float = 0
    distraction_seconds: float = 0
    communication_seconds: float = 0
    break_seconds: float = 0
    app_usage: dict[str, Any] = Field(default_factory=dict)
    process_metrics: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: str | None = None


class PredictionResponse(BaseModel):
    prediction_id: str
    productivity_level: str
    productivity_score: float
    probabilities: dict[str, float]
    feature_importance: dict[str, float]
    feature_contributions: dict[str, float]
    explanation: str
    shap_local_plot_url: str
    advice: str
    personal_insight: dict[str, Any] = Field(default_factory=dict)
    temporal_insight: dict[str, Any] = Field(default_factory=dict)
    pattern_insight: dict[str, Any] = Field(default_factory=dict)
    proactive: dict[str, Any] = Field(default_factory=dict)
    coaching: dict[str, Any] = Field(default_factory=dict)
    personalization: dict[str, Any] = Field(default_factory=dict)


class SimulationScenarioResponse(BaseModel):
    name: str
    score: float
    level: str
    improvement: float
    explanation: str


class SimulationRecommendationResponse(BaseModel):
    name: str
    improvement: float
    explanation: str


class SimulationResponse(BaseModel):
    scenarios: list[SimulationScenarioResponse]
    best_scenario: str
    best_score: float
    current_score: float
    current_level: str
    improvement: float
    recommendations: list[SimulationRecommendationResponse] = Field(default_factory=list)


class ModelStatsResponse(BaseModel):
    accuracy: float
    accuracy_pct: str
    previous_accuracy: float | None = None
    previous_accuracy_pct: str | None = None
    improvement: str
    confusion_matrix: list[list[int]]
    class_labels: list[str]
    feature_importances: dict[str, float]
    model_name: str
    training_samples: int
    total_training_samples: int
    version: str
    dataset_composition: dict[str, Any] = Field(default_factory=dict)


class ShapGlobalResponse(BaseModel):
    shap_global_plot_url: str


class RetrainResponse(BaseModel):
    status: str
    old_accuracy: float | None = None
    new_accuracy: float | None = None
    improvement: str
    version: str
    total_training_samples: int
    new_samples: int
    dataset_composition: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class TrackingSessionResponse(BaseModel):
    tracker_id: str
    started_at: str
    status: str


class TrackingSessionStatusResponse(BaseModel):
    tracker_id: str | None = None
    started_at: str | None = None
    status: str


class RealtimeOverviewResponse(BaseModel):
    latest_prediction: dict[str, Any] | None = None
    predictions: list[dict[str, Any]]
    latest_activity: dict[str, Any] | None = None
    activity: list[dict[str, Any]]


def _cleanup_inactive_trackers() -> None:
    stale_tracker_ids = [
        tracker_id
        for tracker_id, tracker_meta in active_trackers.items()
        if tracker_meta["process"].poll() is not None
    ]
    for tracker_id in stale_tracker_ids:
        active_trackers.pop(tracker_id, None)


def _get_active_tracker() -> tuple[str, dict[str, Any]] | tuple[None, None]:
    _cleanup_inactive_trackers()
    for tracker_id, tracker_meta in active_trackers.items():
        return tracker_id, tracker_meta
    return None, None


def _launch_tracker_process(tracker_id: str) -> subprocess.Popen[Any]:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    # The tracker is a long-lived child process, so we detach it from the request
    # lifecycle and silence stdio to avoid blocking the API worker on pipe buffers.
    return subprocess.Popen(
        [
            sys.executable,
            str(TRACKER_SCRIPT_PATH),
            "--tracker_id",
            tracker_id,
        ],
        cwd=str(TRACKER_SCRIPT_PATH.parent),
        stdout=_TRACKER_SUPPRESS_OUTPUT,
        stderr=_TRACKER_SUPPRESS_OUTPUT,
        creationflags=creation_flags,
    )


def _stop_tracker_process(tracker_id: str) -> bool:
    tracker_meta = active_trackers.get(tracker_id)
    if not tracker_meta:
        return False

    process: subprocess.Popen[Any] = tracker_meta["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    active_trackers.pop(tracker_id, None)
    return True


class PersonalInsightsResponse(BaseModel):
    personal_insight: dict[str, Any] = Field(default_factory=dict)
    temporal_insight: dict[str, Any] = Field(default_factory=dict)
    pattern_insight: dict[str, Any] = Field(default_factory=dict)
    proactive: dict[str, Any] = Field(default_factory=dict)
    coaching: dict[str, Any] = Field(default_factory=dict)
    latest_prediction_id: str | None = None
    events_analyzed: int = 0


def _invalidate_runtime_caches(user_id: str | None = None) -> None:
    CACHE.invalidate_prefix(REALTIME_OVERVIEW_CACHE_PREFIX)
    CACHE.invalidate_prefix(LATEST_ACTIVITY_CACHE_PREFIX)
    CACHE.invalidate_prefix(RECENT_ACTIVITY_CACHE_PREFIX)
    CACHE.invalidate_prefix(RECENT_EVENTS_CACHE_PREFIX)
    CACHE.invalidate_prefix(PERSONAL_INSIGHTS_CACHE_PREFIX)
    if user_id:
        CACHE.invalidate_prefix(PERSONALIZATION_CONTEXT_CACHE_PREFIX)


def _default_personalization_context(metrics: ProgrammerMetrics) -> dict[str, Any]:
    return {
        "avg_focus_last_7_days": round(float(metrics.focus_score), 2),
        "avg_distraction_last_7_days": round(float(metrics.distractions), 2),
        "personal_baseline_score": 50.0,
        "samples_used": 0,
    }


def _compact_prediction_event(event: dict[str, Any]) -> dict[str, Any]:
    features = event.get("features") or {}
    return {
        "id": event.get("id"),
        "timestamp": event.get("timestamp"),
        "source": event.get("source"),
        "tracker_id": event.get("tracker_id"),
        "productivity_level": event.get("productivity_level"),
        "productivity_score": event.get("productivity_score"),
        "features": {
            "focus_score": features.get("focus_score"),
            "meetings_per_day": features.get("meetings_per_day"),
        },
    }


def _compact_activity_log(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": entry.get("id"),
        "timestamp": entry.get("timestamp"),
        "tracker_id": entry.get("tracker_id"),
        "app_name": entry.get("app_name"),
        "app_display_name": entry.get("app_display_name"),
        "category": entry.get("category"),
        "window_title": entry.get("window_title"),
        "context": entry.get("context") or {},
        "idle_seconds": entry.get("idle_seconds"),
        "is_idle": entry.get("is_idle"),
        "app_usage": entry.get("app_usage") or {},
    }


def _get_cached_personalization_context(
    user_id: str | None,
    metrics: ProgrammerMetrics,
) -> dict[str, Any]:
    if not user_id:
        return _default_personalization_context(metrics)

    cache_key = build_cache_key(
        PERSONALIZATION_CONTEXT_CACHE_PREFIX.rstrip(":"),
        {"user_id": user_id},
    )
    return CACHE.get_or_set(
        cache_key,
        PERSONALIZATION_CONTEXT_CACHE_TTL_SECONDS,
        lambda: get_user_personalization_context(user_id),
    )


def _get_cached_latest_activity_log() -> dict[str, Any] | None:
    cache_key = f"{LATEST_ACTIVITY_CACHE_PREFIX}singleton"
    return CACHE.get_or_set(
        cache_key,
        LATEST_ACTIVITY_CACHE_TTL_SECONDS,
        get_latest_activity_log,
    )


def _get_cached_recent_activity_logs(limit: int) -> list[dict[str, Any]]:
    cache_key = build_cache_key(
        RECENT_ACTIVITY_CACHE_PREFIX.rstrip(":"),
        {"limit": limit},
    )
    return CACHE.get_or_set(
        cache_key,
        RECENT_ACTIVITY_CACHE_TTL_SECONDS,
        lambda: get_recent_activity_logs(limit=limit),
    )


def _get_cached_recent_prediction_events(
    limit: int,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    cache_key = build_cache_key(
        RECENT_EVENTS_CACHE_PREFIX.rstrip(":"),
        {"limit": limit, "user_id": user_id},
    )
    return CACHE.get_or_set(
        cache_key,
        RECENT_EVENTS_CACHE_TTL_SECONDS,
        lambda: get_recent_prediction_events(limit=limit, user_id=user_id) if user_id else get_recent_prediction_events(limit=limit),
    )


def compute_productivity_score(metrics: ProgrammerMetrics, probability_map: dict[str, float]) -> float:
    """
    Blended score: weighted feature heuristic (60%) + model confidence (40%).
    Always returns a value in [0, 100].
    """
    m = metrics
    effective_distractions = max(float(m.distractions), float(m.distraction_intensity))
    feature_score = (
        0.18 * min(m.coding_hours / 10, 1.0)
        + 0.06 * min(m.learning_time / 4, 1.0)
        + 0.12 * min(m.commits_per_day / 8, 1.0)
        + 0.10 * min(m.lines_of_code / 400, 1.0)
        + 0.10 * min(m.bugs_fixed / 5, 1.0)
        + 0.10 * min(m.sleep_hours / 8, 1.0)
        + 0.08 * max(0, 1 - effective_distractions / 10)
        + 0.08 * min(m.deep_work_ratio, 1.0)
        + 0.10 * m.task_completion_rate
        + 0.07 * max(0, 1 - m.meetings_per_day / 6)
        + 0.05 * max(0, 1 - m.break_time / 120)
        + 0.06 * (m.focus_score / 100)
    ) * 100

    class_scores = {"Low": 20.0, "Medium": 55.0, "High": 90.0}
    proba_score = sum(
        float(probability_map.get(class_name, 0.0)) * score
        for class_name, score in class_scores.items()
    )

    blended = 0.6 * feature_score + 0.4 * proba_score
    return round(float(np.clip(blended, 0, 100)), 1)


def get_advice(level: str) -> str:
    advice_map = {
        "High": (
            "Outstanding productivity! Maintain your deep work sessions, "
            "consistent sleep, and low-distraction environment. "
            "Consider mentoring others or tackling stretch goals."
        ),
        "Medium": (
            "Solid performance with room to improve. Try reducing meetings, "
            "increasing focus blocks, and ensuring 7-8 hours of sleep. "
            "Small habit improvements can push you to High."
        ),
        "Low": (
            "Productivity signals are below optimal. Focus on sleep quality, "
            "reducing distractions, and using time-blocking techniques. "
            "Even 1-2 extra focused hours can make a big difference."
        ),
    }
    return advice_map.get(level, "Keep going!")


def clear_model_caches() -> None:
    _load_cached_model_bundle.cache_clear()
    load_explainer.cache_clear()
    generate_global_plot.cache_clear()
    clear_user_model_cache()
    CACHE.invalidate_prefix(PREDICTION_CORE_CACHE_PREFIX)
    CACHE.invalidate_prefix(MODEL_STATS_CACHE_PREFIX)
    CACHE.invalidate_prefix(PERSONALIZATION_CONTEXT_CACHE_PREFIX)
    CACHE.invalidate_prefix(PERSONAL_INSIGHTS_CACHE_PREFIX)


def format_accuracy(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value * 100:.2f}%"


def format_improvement(previous_accuracy: float | None, current_accuracy: float | None) -> str:
    if previous_accuracy is None or current_accuracy is None:
        return "N/A"
    return f"{(current_accuracy - previous_accuracy) * 100:+.2f}%"


def _build_enriched_feature_payload(
    metrics: ProgrammerMetrics,
    user_id: str | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    input_payload = metrics.model_dump()
    personalization_context = _get_cached_personalization_context(user_id, metrics)
    input_payload.update(
        {
            "avg_focus_last_7_days": float(personalization_context["avg_focus_last_7_days"]),
            "avg_distraction_last_7_days": float(
                personalization_context["avg_distraction_last_7_days"]
            ),
            "personal_baseline_score": float(personalization_context["personal_baseline_score"]),
        }
    )
    return input_payload, personalization_context


def _align_probabilities(
    raw_probabilities: np.ndarray,
    model_classes: np.ndarray,
    label_encoder,
) -> tuple[np.ndarray, dict[str, float]]:
    probability_map = {str(class_name): 0.0 for class_name in label_encoder.classes_}
    for encoded_class, probability in zip(model_classes, raw_probabilities):
        class_name = label_encoder.inverse_transform([int(encoded_class)])[0]
        probability_map[str(class_name)] = float(probability)
    aligned = np.array([probability_map[str(class_name)] for class_name in label_encoder.classes_])
    return aligned, probability_map


def _predict_with_bundle(bundle: dict[str, Any], input_df: pd.DataFrame) -> dict[str, Any]:
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    features = bundle["features"]
    model = bundle["model"]

    x_scaled = scaler.transform(input_df[features].values)
    predicted_class = int(model.predict(x_scaled)[0])
    raw_probabilities = model.predict_proba(x_scaled)[0]
    aligned_probabilities, probability_map = _align_probabilities(
        raw_probabilities=raw_probabilities,
        model_classes=np.asarray(model.classes_),
        label_encoder=label_encoder,
    )
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "raw_probabilities": raw_probabilities,
        "aligned_probabilities": aligned_probabilities,
        "probability_map": probability_map,
        "scaled_df": pd.DataFrame(x_scaled, columns=features),
        "class_index": list(model.classes_).index(predicted_class),
    }


@lru_cache(maxsize=1)
def _load_cached_model_bundle() -> dict[str, Any]:
    bundle = load_model_bundle(MODEL_PATH)
    if bundle is None:
        raise FileNotFoundError(
            f"model.pkl not found at '{MODEL_PATH}'. Run `python model_training.py` first."
        )
    return bundle


def get_bundle():
    bundle = dict(_load_cached_model_bundle())
    if "version" not in bundle:
        bundle["version"] = "v1"
    if "model_name" not in bundle:
        bundle["model_name"] = MODEL_NAME
    total_samples = int(
        bundle.get("trained_on")
        or bundle.get("total_training_samples")
        or bundle.get("training_samples")
        or _infer_dataset_rows()
    )
    bundle["trained_on"] = total_samples
    bundle["total_training_samples"] = total_samples
    bundle["training_samples"] = total_samples
    bundle.setdefault("previous_accuracy", None)
    bundle.setdefault("dataset_composition", {})
    return bundle


def _infer_dataset_rows() -> int:
    if DATASET_PATH.exists():
        return len(pd.read_csv(DATASET_PATH))
    return 0


def _run_retraining(force: bool = False) -> dict[str, Any]:
    if not _retrain_lock.acquire(blocking=False):
        bundle = get_bundle()
        return {
            "status": "busy",
            "old_accuracy": bundle.get("previous_accuracy"),
            "new_accuracy": bundle.get("accuracy"),
            "improvement": format_improvement(bundle.get("previous_accuracy"), bundle.get("accuracy")),
            "version": bundle.get("version", "v1"),
            "total_training_samples": int(bundle.get("trained_on", 0)),
            "new_samples": 0,
            "dataset_composition": bundle.get("dataset_composition", {}),
            "reason": "Retraining is already in progress.",
        }

    try:
        result = retrain_model(force=force)
        if result["status"] == "retrained":
            clear_model_caches()
        return result
    except Exception as exc:
        LOGGER.exception("Retraining failed")
        bundle = get_bundle()
        return {
            "status": "failed",
            "old_accuracy": bundle.get("previous_accuracy"),
            "new_accuracy": bundle.get("accuracy"),
            "improvement": format_improvement(bundle.get("previous_accuracy"), bundle.get("accuracy")),
            "version": bundle.get("version", "v1"),
            "total_training_samples": int(bundle.get("trained_on", 0)),
            "new_samples": 0,
            "dataset_composition": bundle.get("dataset_composition", {}),
            "reason": str(exc),
        }
    finally:
        _retrain_lock.release()


def _schedule_auto_retrain(background_tasks: BackgroundTasks) -> None:
    bundle = get_bundle()
    last_timestamp = bundle.get("last_training_data_timestamp")
    new_data_count = get_new_data_count(last_timestamp)
    if new_data_count >= AUTO_RETRAIN_THRESHOLD and not _retrain_lock.locked():
        background_tasks.add_task(_run_retraining, False)


def _schedule_user_model_refresh(background_tasks: BackgroundTasks, user_id: str | None) -> None:
    if user_id:
        background_tasks.add_task(update_user_model, user_id)


def _resolve_prediction_context(
    metrics: ProgrammerMetrics,
    user_id: str | None = None,
) -> dict[str, Any]:
    global_bundle = get_bundle()
    label_encoder = global_bundle["label_encoder"]
    input_payload, personalization_context = _build_enriched_feature_payload(metrics, user_id)
    input_df = pd.DataFrame([input_payload], columns=TRAINING_FEATURES)

    global_prediction = _predict_with_bundle(global_bundle, input_df)
    selected_model = get_personalized_model(user_id, global_bundle)
    selected_bundle = selected_model["bundle"]
    personalization_metadata = dict(selected_model["personalization"])

    final_probabilities = global_prediction["aligned_probabilities"]
    final_probability_map = dict(global_prediction["probability_map"])
    level = str(global_prediction["predicted_label"])
    explanation_bundle = global_bundle
    explanation_prediction = global_prediction

    if personalization_metadata.get("model_used") == "user":
        user_prediction = _predict_with_bundle(selected_bundle, input_df)
        final_probabilities = (
            0.7 * user_prediction["aligned_probabilities"]
            + 0.3 * global_prediction["aligned_probabilities"]
        )
        final_probability_map = {
            str(class_name): round(float(probability), 6)
            for class_name, probability in zip(label_encoder.classes_, final_probabilities)
        }
        level = str(label_encoder.classes_[int(np.argmax(final_probabilities))])

    explanation_encoded_class = int(label_encoder.transform([level])[0])
    if (
        personalization_metadata.get("model_used") == "user"
        and explanation_encoded_class in set(selected_bundle["model"].classes_)
    ):
        explanation_bundle = selected_bundle
        explanation_prediction = {
            **_predict_with_bundle(selected_bundle, input_df),
            "class_index": list(selected_bundle["model"].classes_).index(explanation_encoded_class),
        }
    else:
        explanation_bundle = global_bundle
        explanation_prediction = {
            **global_prediction,
            "class_index": list(global_bundle["model"].classes_).index(explanation_encoded_class),
        }

    score = compute_productivity_score(metrics, final_probability_map)
    return {
        "global_bundle": global_bundle,
        "input_payload": input_payload,
        "personalization_context": personalization_context,
        "input_df": input_df,
        "global_prediction": global_prediction,
        "selected_bundle": selected_bundle,
        "personalization_metadata": personalization_metadata,
        "final_probabilities": final_probabilities,
        "final_probability_map": final_probability_map,
        "level": level,
        "explanation_bundle": explanation_bundle,
        "explanation_prediction": explanation_prediction,
        "score": score,
    }


def _build_prediction_core(
    metrics: ProgrammerMetrics,
    user_id: str | None = None,
) -> dict[str, Any]:
    prediction_context = _resolve_prediction_context(metrics, user_id=user_id)
    input_payload = prediction_context["input_payload"]
    personalization_context = prediction_context["personalization_context"]
    explanation_bundle = prediction_context["explanation_bundle"]
    explanation_prediction = prediction_context["explanation_prediction"]
    level = prediction_context["level"]
    score = prediction_context["score"]
    global_prediction = prediction_context["global_prediction"]
    final_probabilities = prediction_context["final_probabilities"]
    final_probability_map = prediction_context["final_probability_map"]
    personalization_metadata = dict(prediction_context["personalization_metadata"])
    global_bundle = prediction_context["global_bundle"]
    selected_bundle = prediction_context["selected_bundle"]

    cache_key = build_cache_key(
        PREDICTION_CORE_CACHE_PREFIX.rstrip(":"),
        {
            "user_id": user_id,
            "input_payload": input_payload,
            "global_version": global_bundle.get("version"),
            "global_samples": global_bundle.get("trained_on"),
            "selected_version": selected_bundle.get("version"),
            "selected_samples": selected_bundle.get("trained_on"),
            "model_used": personalization_metadata.get("model_used"),
        },
    )
    cached_core = CACHE.get(cache_key)
    if cached_core is not None:
        cached_personalization = dict(cached_core.get("personalization") or {})
        cached_personalization["prediction_cache_hit"] = True
        return {
            **cached_core,
            "personalization": cached_personalization,
        }

    base_probability_confidence = max(global_prediction["aligned_probabilities"])
    final_probability_confidence = max(final_probabilities)
    personalization_metadata.update(
        {
            "confidence_boost": round(
                max(0.0, float(final_probability_confidence - base_probability_confidence)) * 100,
                2,
            ),
            "history_points_used": int(personalization_context.get("samples_used", 0)),
            "blend_ratio": (
                {"user": 0.7, "global": 0.3}
                if personalization_metadata.get("model_used") == "user"
                else {"user": 0.0, "global": 1.0}
            ),
            "prediction_cache_hit": False,
        }
    )

    shap_values = get_shap_values(
        explanation_prediction["scaled_df"],
        model_path=MODEL_PATH if explanation_bundle is global_bundle else None,
        bundle=explanation_bundle if explanation_bundle is not global_bundle else None,
    )
    shap_values_single = shap_values[explanation_prediction["class_index"]][0]
    explanation_result = generate_explanation(
        shap_values_single,
        prediction_context["input_df"][explanation_bundle["features"]],
        predicted_class=level,
    )
    shap_local_plot_url = generate_local_plot(
        input_df=prediction_context["input_df"],
        predicted_class_index=explanation_prediction["class_index"],
        model_path=MODEL_PATH if explanation_bundle is global_bundle else None,
        prediction_id=cache_key.split(":")[-1][:24],
        bundle=explanation_bundle if explanation_bundle is not global_bundle else None,
    )

    prediction_core = {
        "input_payload": input_payload,
        "productivity_level": level,
        "productivity_score": score,
        "probabilities": {
            str(class_name): round(float(probability) * 100, 1)
            for class_name, probability in final_probability_map.items()
        },
        "feature_importance": explanation_result["feature_importance"],
        "feature_contributions": explanation_result["feature_contributions"],
        "explanation": explanation_result["explanation"],
        "shap_local_plot_url": shap_local_plot_url,
        "advice": get_advice(level),
        "personalization": personalization_metadata,
    }
    CACHE.set(cache_key, prediction_core, PREDICTION_CORE_CACHE_TTL_SECONDS)
    return prediction_core


def _build_prediction_response(
    metrics: ProgrammerMetrics,
    user_id: str | None = None,
    tracker_id: str | None = None,
    source: str = "manual",
) -> PredictionResponse:
    prediction_core = _build_prediction_core(metrics, user_id=user_id)
    input_payload = dict(prediction_core["input_payload"])
    prediction_id = uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()

    save_prediction(
        {
            **input_payload,
            "id": prediction_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "predicted_label": prediction_core["productivity_level"],
            "productivity_score": prediction_core["productivity_score"],
        }
    )

    latest_activity = _get_cached_latest_activity_log()
    event_payload = {
        "id": prediction_id,
        "timestamp": timestamp,
        "source": source,
        "user_id": user_id,
        "tracker_id": tracker_id,
        "productivity_level": prediction_core["productivity_level"],
        "productivity_score": prediction_core["productivity_score"],
        "explanation": prediction_core["explanation"],
        "advice": prediction_core["advice"],
        "shap_local_plot_url": prediction_core["shap_local_plot_url"],
        "probabilities": prediction_core["probabilities"],
        "features": input_payload,
        "feature_importance": prediction_core["feature_importance"],
        "feature_contributions": prediction_core["feature_contributions"],
        "personalization": prediction_core["personalization"],
        "latest_activity": latest_activity,
        "app_usage": (latest_activity or {}).get("app_usage") or {},
    }
    historical_events = _get_cached_recent_prediction_events(limit=60, user_id=user_id)

    personal_insight = generate_comparative_insights(event_payload, historical_events)
    timeline_events = [*historical_events, {**event_payload, "personal_insight": personal_insight}][-60:]
    temporal_insight = generate_temporal_insights(timeline_events)
    pattern_insight = generate_behavioral_patterns(timeline_events)
    proactive = generate_proactive_alerts(
        {
            "features": input_payload,
            "app_usage": event_payload["app_usage"],
            "temporal_insight": temporal_insight,
            "timestamp": timestamp,
        },
        timeline_events[-10:],
        pattern_insight,
    )

    enriched_event = {
        **event_payload,
        "personal_insight": personal_insight,
        "temporal_insight": temporal_insight,
        "pattern_insight": pattern_insight,
        "proactive": proactive,
    }
    coaching = generate_coaching_feedback(enriched_event, historical_events)
    enriched_event["coaching"] = coaching

    save_prediction_event(enriched_event)
    _invalidate_runtime_caches(user_id)

    return PredictionResponse(
        prediction_id=prediction_id,
        productivity_level=prediction_core["productivity_level"],
        productivity_score=prediction_core["productivity_score"],
        probabilities=enriched_event["probabilities"],
        feature_importance=prediction_core["feature_importance"],
        feature_contributions=prediction_core["feature_contributions"],
        explanation=prediction_core["explanation"],
        shap_local_plot_url=prediction_core["shap_local_plot_url"],
        advice=prediction_core["advice"],
        personal_insight=personal_insight,
        temporal_insight=temporal_insight,
        pattern_insight=pattern_insight,
        proactive=proactive,
        coaching=coaching,
        personalization=prediction_core["personalization"],
    )


def _predict_summary(metrics: ProgrammerMetrics, user_id: str | None = None) -> dict[str, Any]:
    prediction_core = _build_prediction_core(metrics, user_id=user_id)
    return {
        "productivity_level": prediction_core["productivity_level"],
        "productivity_score": prediction_core["productivity_score"],
        "probabilities": prediction_core["probabilities"],
    }


async def _broadcast_latest_event() -> None:
    events = _get_cached_recent_prediction_events(limit=1)
    latest_event = events[-1] if events else None
    if latest_event:
        await ws_manager.broadcast({"type": "prediction_update", "data": latest_event})


def _build_realtime_overview(limit: int) -> dict[str, Any]:
    predictions = _get_cached_recent_prediction_events(limit=limit)
    activity = _get_cached_recent_activity_logs(limit=limit)
    latest_prediction = predictions[-1] if predictions else None
    latest_activity = activity[-1] if activity else _get_cached_latest_activity_log()
    return {
        "latest_prediction": latest_prediction,
        "predictions": [_compact_prediction_event(item) for item in predictions],
        "latest_activity": latest_activity,
        "activity": [_compact_activity_log(item) for item in activity],
    }


def _build_personal_insights_payload(limit: int, user_id: str | None) -> dict[str, Any]:
    events = _get_cached_recent_prediction_events(limit=limit, user_id=user_id)
    if not events:
        return {
            "personal_insight": {},
            "temporal_insight": {},
            "pattern_insight": {},
            "proactive": {},
            "coaching": {},
            "latest_prediction_id": None,
            "events_analyzed": 0,
        }

    latest_event = events[-1]
    personal_insight = latest_event.get("personal_insight") or generate_comparative_insights(
        latest_event,
        events[:-1],
    )
    timeline_events = events[-60:]
    temporal_insight = latest_event.get("temporal_insight") or generate_temporal_insights(timeline_events)
    pattern_insight = latest_event.get("pattern_insight") or generate_behavioral_patterns(timeline_events)
    proactive = latest_event.get("proactive") or generate_proactive_alerts(
        {
            "features": latest_event.get("features") or {},
            "app_usage": latest_event.get("app_usage") or {},
            "temporal_insight": temporal_insight,
            "timestamp": latest_event.get("timestamp"),
        },
        timeline_events[-10:],
        pattern_insight,
    )
    coaching = latest_event.get("coaching") or generate_coaching_feedback(
        {
            **latest_event,
            "personal_insight": personal_insight,
            "temporal_insight": temporal_insight,
            "pattern_insight": pattern_insight,
            "proactive": proactive,
        },
        events[:-1],
    )
    return {
        "personal_insight": personal_insight,
        "temporal_insight": temporal_insight,
        "pattern_insight": pattern_insight,
        "proactive": proactive,
        "coaching": coaching,
        "latest_prediction_id": latest_event.get("id"),
        "events_analyzed": len(events),
    }


def _build_model_stats_payload() -> dict[str, Any]:
    bundle = get_bundle()
    accuracy = float(bundle["accuracy"])
    previous_accuracy = bundle.get("previous_accuracy")
    total_training_samples = int(bundle.get("trained_on", 0))
    return {
        "accuracy": round(accuracy, 4),
        "accuracy_pct": format_accuracy(accuracy) or "N/A",
        "previous_accuracy": (
            round(float(previous_accuracy), 4) if previous_accuracy is not None else None
        ),
        "previous_accuracy_pct": format_accuracy(previous_accuracy),
        "improvement": format_improvement(previous_accuracy, accuracy),
        "confusion_matrix": bundle["confusion_matrix"],
        "class_labels": bundle["classes"],
        "feature_importances": bundle["feature_importances"],
        "model_name": bundle.get("model_name", MODEL_NAME),
        "training_samples": total_training_samples,
        "total_training_samples": total_training_samples,
        "version": bundle.get("version", "v1"),
        "dataset_composition": bundle.get("dataset_composition", {}),
    }


@app.get("/", tags=["health"])
def root():
    return {
        "status": "ok",
        "app": "Devora API v3.0",
        "continuous_learning": True,
        "realtime_tracking": True,
        "total_predictions_logged": get_total_prediction_count(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(payload: PredictRequest, background_tasks: BackgroundTasks):
    try:
        metrics = ProgrammerMetrics(
            **payload.model_dump(include=set(ProgrammerMetrics.model_fields.keys()))
        )
        response = await asyncio.to_thread(
            _build_prediction_response,
            metrics,
            payload.user_id,
            None,
            "manual",
        )
        _schedule_auto_retrain(background_tasks)
        _schedule_user_model_refresh(background_tasks, payload.user_id)
        await _broadcast_latest_event()
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/simulate", response_model=SimulationResponse, tags=["prediction"])
async def simulate(payload: SimulationRequest):
    try:
        metrics = ProgrammerMetrics(
            **payload.model_dump(include=set(ProgrammerMetrics.model_fields.keys()))
        )

        simulation_result = await asyncio.to_thread(
            run_what_if_analysis,
            metrics.model_dump(),
            lambda scenario_input: _predict_summary(
                ProgrammerMetrics(**scenario_input),
                user_id=payload.user_id,
            ),
        )
        return SimulationResponse(**simulation_result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/batch-predict", response_model=list[PredictionResponse], tags=["prediction"])
async def batch_predict(payloads: list[BatchPredictItem], background_tasks: BackgroundTasks):
    if not payloads:
        return []
    if len(payloads) > 25:
        raise HTTPException(status_code=400, detail="Batch size exceeds the maximum of 25 requests.")

    try:
        metrics_payloads = [
            (
                ProgrammerMetrics(
                    **payload.model_dump(include=set(ProgrammerMetrics.model_fields.keys()))
                ),
                payload.user_id,
            )
            for payload in payloads
        ]
        responses = await asyncio.gather(
            *[
                asyncio.to_thread(
                    _build_prediction_response,
                    metrics,
                    user_id,
                    None,
                    "batch",
                )
                for metrics, user_id in metrics_payloads
            ]
        )
        for _, user_id in metrics_payloads:
            _schedule_user_model_refresh(background_tasks, user_id)
        _schedule_auto_retrain(background_tasks)
        await _broadcast_latest_event()
        return responses
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _handle_realtime_predict(
    payload: RealtimePredictRequest,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    try:
        metrics = ProgrammerMetrics(**payload.model_dump(include=set(ProgrammerMetrics.model_fields.keys())))
        response = await asyncio.to_thread(
            _build_prediction_response,
            metrics,
            payload.user_id,
            payload.tracker_id,
            payload.source,
        )
        _schedule_auto_retrain(background_tasks)
        _schedule_user_model_refresh(background_tasks, payload.user_id)
        await _broadcast_latest_event()
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/realtime/predict", response_model=PredictionResponse, tags=["prediction"])
async def realtime_predict(payload: RealtimePredictRequest, background_tasks: BackgroundTasks):
    return await _handle_realtime_predict(payload, background_tasks)


@app.post("/predict/realtime", response_model=PredictionResponse, tags=["prediction"])
async def predict_realtime_alias(payload: RealtimePredictRequest, background_tasks: BackgroundTasks):
    return await _handle_realtime_predict(payload, background_tasks)


async def _handle_activity_log(snapshot: ActivitySnapshot):
    try:
        payload = snapshot.model_dump()
        payload["timestamp"] = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
        record_id = save_activity_log(payload)
        _invalidate_runtime_caches()
        latest = _get_cached_latest_activity_log()
        await ws_manager.broadcast({"type": "activity_update", "data": latest})
        return {"status": "ok", "id": record_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/activity/logs", tags=["realtime"])
async def ingest_activity_log(snapshot: ActivitySnapshot):
    return await _handle_activity_log(snapshot)


@app.post("/activity", tags=["realtime"])
async def ingest_activity_log_alias(snapshot: ActivitySnapshot):
    return await _handle_activity_log(snapshot)


@app.post("/tracking/session", response_model=TrackingSessionResponse, tags=["realtime"])
def create_tracking_session():
    tracker_id, tracker_meta = _get_active_tracker()
    if tracker_id and tracker_meta:
        return TrackingSessionResponse(
            tracker_id=tracker_id,
            started_at=tracker_meta["started_at"],
            status="tracking",
        )

    tracker_id = uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        process = _launch_tracker_process(tracker_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start tracker: {exc}") from exc

    active_trackers[tracker_id] = {
        "process": process,
        "started_at": started_at,
    }

    return TrackingSessionResponse(
        tracker_id=tracker_id,
        started_at=started_at,
        status="tracking",
    )


@app.get("/tracking/session", response_model=TrackingSessionStatusResponse, tags=["realtime"])
def get_tracking_session():
    tracker_id, tracker_meta = _get_active_tracker()
    if tracker_id and tracker_meta:
        return TrackingSessionStatusResponse(
            tracker_id=tracker_id,
            started_at=tracker_meta["started_at"],
            status="tracking",
        )

    return TrackingSessionStatusResponse(status="stopped")


@app.post("/tracking/stop/{tracker_id}", tags=["realtime"])
def stop_tracking_session(tracker_id: str):
    _cleanup_inactive_trackers()

    if tracker_id not in active_trackers:
        raise HTTPException(status_code=404, detail="Tracker process not found")

    _stop_tracker_process(tracker_id)
    return {"tracker_id": tracker_id, "status": "stopped"}


@app.get("/realtime/overview", response_model=RealtimeOverviewResponse, tags=["realtime"])
def realtime_overview(limit: int = 30):
    cache_key = build_cache_key(
        REALTIME_OVERVIEW_CACHE_PREFIX.rstrip(":"),
        {"limit": limit},
    )
    overview = CACHE.get_or_set(
        cache_key,
        REALTIME_OVERVIEW_CACHE_TTL_SECONDS,
        lambda: _build_realtime_overview(limit),
    )
    return RealtimeOverviewResponse(
        latest_prediction=overview["latest_prediction"],
        predictions=overview["predictions"],
        latest_activity=overview["latest_activity"],
        activity=overview["activity"],
    )


@app.get("/insights/personal", response_model=PersonalInsightsResponse, tags=["insights"])
def personal_insights(limit: int = 60, user_id: str | None = None):
    cache_key = build_cache_key(
        PERSONAL_INSIGHTS_CACHE_PREFIX.rstrip(":"),
        {"limit": limit, "user_id": user_id},
    )
    payload = CACHE.get_or_set(
        cache_key,
        PERSONAL_INSIGHTS_CACHE_TTL_SECONDS,
        lambda: _build_personal_insights_payload(limit, user_id),
    )
    return PersonalInsightsResponse(**payload)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        overview = realtime_overview(limit=20)
        await websocket.send_json({"type": "bootstrap", "data": overview.model_dump()})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)


@app.post("/retrain", response_model=RetrainResponse, tags=["model"])
def retrain():
    try:
        result = _run_retraining(force=False)
        return RetrainResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/model-stats", response_model=ModelStatsResponse, tags=["model"])
def model_stats():
    cache_key = f"{MODEL_STATS_CACHE_PREFIX}current"
    payload = CACHE.get_or_set(
        cache_key,
        MODEL_STATS_CACHE_TTL_SECONDS,
        _build_model_stats_payload,
    )
    return ModelStatsResponse(**payload)


@app.get("/shap-global", response_model=ShapGlobalResponse, tags=["explainability"])
def shap_global():
    try:
        return ShapGlobalResponse(
            shap_global_plot_url=generate_global_plot(MODEL_PATH),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
