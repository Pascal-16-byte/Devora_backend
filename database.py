"""
SQLite persistence for Devora predictions and realtime telemetry.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "devinsight.db"
TRAINING_TABLE = "user_data"
ACTIVITY_TABLE = "activity_logs"
PREDICTION_EVENTS_TABLE = "prediction_events"

FEATURE_COLUMNS = [
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


@contextmanager
def get_connection():
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TRAINING_TABLE} (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                coding_hours REAL,
                learning_time REAL,
                commits_per_day REAL,
                lines_of_code REAL,
                bugs_fixed REAL,
                sleep_hours REAL,
                distractions REAL,
                deep_work_ratio REAL,
                distraction_intensity REAL,
                task_completion_rate REAL,
                meetings_per_day REAL,
                break_time REAL,
                focus_score REAL,
                avg_focus_last_7_days REAL,
                avg_distraction_last_7_days REAL,
                personal_baseline_score REAL,
                predicted_label TEXT NOT NULL,
                productivity_score REAL NOT NULL
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {ACTIVITY_TABLE} (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                tracker_id TEXT,
                app_name TEXT,
                app_display_name TEXT,
                category TEXT,
                active_window TEXT,
                window_title TEXT,
                context_json TEXT,
                cpu_percent REAL,
                memory_mb REAL,
                idle_seconds REAL,
                is_idle INTEGER NOT NULL DEFAULT 0,
                session_seconds REAL,
                coding_seconds REAL,
                distraction_seconds REAL,
                communication_seconds REAL,
                break_seconds REAL,
                app_usage_json TEXT NOT NULL,
                process_metrics_json TEXT NOT NULL,
                raw_payload_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PREDICTION_EVENTS_TABLE} (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                user_id TEXT,
                tracker_id TEXT,
                productivity_level TEXT NOT NULL,
                productivity_score REAL NOT NULL,
                explanation TEXT,
                advice TEXT,
                shap_local_plot_url TEXT,
                probabilities_json TEXT NOT NULL,
                features_json TEXT NOT NULL,
                feature_importance_json TEXT NOT NULL,
                feature_contributions_json TEXT NOT NULL,
                personal_insight_json TEXT,
                temporal_insight_json TEXT,
                pattern_insight_json TEXT,
                proactive_json TEXT,
                coaching_json TEXT,
                personalization_json TEXT,
                latest_activity_json TEXT,
                app_usage_json TEXT
            )
            """
        )
        training_columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({TRAINING_TABLE})").fetchall()
        }
        if "user_id" not in training_columns:
            connection.execute(f"ALTER TABLE {TRAINING_TABLE} ADD COLUMN user_id TEXT")
        for feature_name in FEATURE_COLUMNS:
            if feature_name not in training_columns:
                connection.execute(f"ALTER TABLE {TRAINING_TABLE} ADD COLUMN {feature_name} REAL")

        activity_columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({ACTIVITY_TABLE})").fetchall()
        }
        if "context_json" not in activity_columns:
            connection.execute(f"ALTER TABLE {ACTIVITY_TABLE} ADD COLUMN context_json TEXT")

        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({PREDICTION_EVENTS_TABLE})").fetchall()
        }
        if "user_id" not in columns:
            connection.execute(
                f"ALTER TABLE {PREDICTION_EVENTS_TABLE} ADD COLUMN user_id TEXT"
            )
        if "personal_insight_json" not in columns:
            connection.execute(
                f"ALTER TABLE {PREDICTION_EVENTS_TABLE} ADD COLUMN personal_insight_json TEXT"
            )
        if "coaching_json" not in columns:
            connection.execute(
                f"ALTER TABLE {PREDICTION_EVENTS_TABLE} ADD COLUMN coaching_json TEXT"
            )
        if "temporal_insight_json" not in columns:
            connection.execute(
                f"ALTER TABLE {PREDICTION_EVENTS_TABLE} ADD COLUMN temporal_insight_json TEXT"
            )
        if "pattern_insight_json" not in columns:
            connection.execute(
                f"ALTER TABLE {PREDICTION_EVENTS_TABLE} ADD COLUMN pattern_insight_json TEXT"
            )
        if "proactive_json" not in columns:
            connection.execute(
                f"ALTER TABLE {PREDICTION_EVENTS_TABLE} ADD COLUMN proactive_json TEXT"
            )
        if "personalization_json" not in columns:
            connection.execute(
                f"ALTER TABLE {PREDICTION_EVENTS_TABLE} ADD COLUMN personalization_json TEXT"
            )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TRAINING_TABLE}_timestamp ON {TRAINING_TABLE}(timestamp)"
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TRAINING_TABLE}_user_id ON {TRAINING_TABLE}(user_id)"
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{ACTIVITY_TABLE}_timestamp ON {ACTIVITY_TABLE}(timestamp)"
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{ACTIVITY_TABLE}_tracker_timestamp ON {ACTIVITY_TABLE}(tracker_id, timestamp)"
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{PREDICTION_EVENTS_TABLE}_timestamp ON {PREDICTION_EVENTS_TABLE}(timestamp)"
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{PREDICTION_EVENTS_TABLE}_user_id ON {PREDICTION_EVENTS_TABLE}(user_id)"
        )
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{PREDICTION_EVENTS_TABLE}_tracker_timestamp ON {PREDICTION_EVENTS_TABLE}(tracker_id, timestamp)"
        )
        connection.commit()


def _json_dump(value: Any) -> str:
    return json.dumps({} if value is None else value, ensure_ascii=True)


def save_prediction(data: dict[str, Any]) -> str:
    record = {
        "id": data.get("id") or str(uuid4()),
        "timestamp": data["timestamp"],
        "user_id": data.get("user_id"),
        "coding_hours": data.get("coding_hours"),
        "learning_time": data.get("learning_time"),
        "commits_per_day": data.get("commits_per_day"),
        "lines_of_code": data.get("lines_of_code"),
        "bugs_fixed": data.get("bugs_fixed"),
        "sleep_hours": data.get("sleep_hours"),
        "distractions": data.get("distractions"),
        "deep_work_ratio": data.get("deep_work_ratio"),
        "distraction_intensity": data.get("distraction_intensity"),
        "task_completion_rate": data.get("task_completion_rate"),
        "meetings_per_day": data.get("meetings_per_day"),
        "break_time": data.get("break_time"),
        "focus_score": data.get("focus_score"),
        "avg_focus_last_7_days": data.get("avg_focus_last_7_days"),
        "avg_distraction_last_7_days": data.get("avg_distraction_last_7_days"),
        "personal_baseline_score": data.get("personal_baseline_score"),
        "predicted_label": data["predicted_label"],
        "productivity_score": data["productivity_score"],
    }

    with get_connection() as connection:
        connection.execute(
            f"""
            INSERT INTO {TRAINING_TABLE} (
                id, timestamp, user_id, coding_hours, learning_time, commits_per_day,
                lines_of_code, bugs_fixed, sleep_hours, distractions, deep_work_ratio,
                distraction_intensity, task_completion_rate, meetings_per_day, break_time,
                focus_score, avg_focus_last_7_days, avg_distraction_last_7_days,
                personal_baseline_score, predicted_label, productivity_score
            ) VALUES (
                :id, :timestamp, :user_id, :coding_hours, :learning_time, :commits_per_day,
                :lines_of_code, :bugs_fixed, :sleep_hours, :distractions, :deep_work_ratio,
                :distraction_intensity, :task_completion_rate, :meetings_per_day, :break_time,
                :focus_score, :avg_focus_last_7_days, :avg_distraction_last_7_days,
                :personal_baseline_score, :predicted_label, :productivity_score
            )
            """,
            record,
        )
        connection.commit()

    return record["id"]


def save_activity_log(data: dict[str, Any]) -> str:
    record = {
        "id": data.get("id") or str(uuid4()),
        "timestamp": data["timestamp"],
        "tracker_id": data.get("tracker_id"),
        "app_name": data.get("app_name"),
        "app_display_name": data.get("app_display_name"),
        "category": data.get("category"),
        "active_window": data.get("active_window"),
        "window_title": data.get("window_title"),
        "context_json": _json_dump(data.get("context")),
        "cpu_percent": data.get("cpu_percent"),
        "memory_mb": data.get("memory_mb"),
        "idle_seconds": data.get("idle_seconds"),
        "is_idle": 1 if data.get("is_idle") else 0,
        "session_seconds": data.get("session_seconds"),
        "coding_seconds": data.get("coding_seconds"),
        "distraction_seconds": data.get("distraction_seconds"),
        "communication_seconds": data.get("communication_seconds"),
        "break_seconds": data.get("break_seconds"),
        "app_usage_json": _json_dump(data.get("app_usage")),
        "process_metrics_json": _json_dump(data.get("process_metrics")),
        "raw_payload_json": _json_dump(data),
    }

    with get_connection() as connection:
        connection.execute(
            f"""
            INSERT INTO {ACTIVITY_TABLE} (
                id, timestamp, tracker_id, app_name, app_display_name, category,
                active_window, window_title, context_json, cpu_percent, memory_mb,
                idle_seconds, is_idle, session_seconds, coding_seconds, distraction_seconds,
                communication_seconds, break_seconds, app_usage_json, process_metrics_json,
                raw_payload_json
            ) VALUES (
                :id, :timestamp, :tracker_id, :app_name, :app_display_name, :category,
                :active_window, :window_title, :context_json, :cpu_percent, :memory_mb,
                :idle_seconds, :is_idle, :session_seconds, :coding_seconds, :distraction_seconds,
                :communication_seconds, :break_seconds, :app_usage_json, :process_metrics_json,
                :raw_payload_json
            )
            """,
            record,
        )
        connection.commit()

    return record["id"]


def save_prediction_event(data: dict[str, Any]) -> str:
    record = {
        "id": data.get("id") or str(uuid4()),
        "timestamp": data["timestamp"],
        "source": data.get("source", "manual"),
        "user_id": data.get("user_id"),
        "tracker_id": data.get("tracker_id"),
        "productivity_level": data["productivity_level"],
        "productivity_score": data["productivity_score"],
        "explanation": data.get("explanation"),
        "advice": data.get("advice"),
        "shap_local_plot_url": data.get("shap_local_plot_url"),
        "probabilities_json": _json_dump(data.get("probabilities")),
        "features_json": _json_dump(data.get("features")),
        "feature_importance_json": _json_dump(data.get("feature_importance")),
        "feature_contributions_json": _json_dump(data.get("feature_contributions")),
        "personal_insight_json": _json_dump(data.get("personal_insight")),
        "temporal_insight_json": _json_dump(data.get("temporal_insight")),
        "pattern_insight_json": _json_dump(data.get("pattern_insight")),
        "proactive_json": _json_dump(data.get("proactive")),
        "coaching_json": _json_dump(data.get("coaching")),
        "personalization_json": _json_dump(data.get("personalization")),
        "latest_activity_json": _json_dump(data.get("latest_activity")),
        "app_usage_json": _json_dump(data.get("app_usage")),
    }

    with get_connection() as connection:
        connection.execute(
            f"""
            INSERT INTO {PREDICTION_EVENTS_TABLE} (
                id, timestamp, source, user_id, tracker_id, productivity_level, productivity_score,
                explanation, advice, shap_local_plot_url, probabilities_json, features_json,
                feature_importance_json, feature_contributions_json, personal_insight_json,
                temporal_insight_json, pattern_insight_json, proactive_json,
                coaching_json, personalization_json,
                latest_activity_json, app_usage_json
            ) VALUES (
                :id, :timestamp, :source, :user_id, :tracker_id, :productivity_level, :productivity_score,
                :explanation, :advice, :shap_local_plot_url, :probabilities_json, :features_json,
                :feature_importance_json, :feature_contributions_json, :personal_insight_json,
                :temporal_insight_json, :pattern_insight_json, :proactive_json,
                :coaching_json, :personalization_json,
                :latest_activity_json, :app_usage_json
            )
            """,
            record,
        )
        connection.commit()

    return record["id"]


def _get_training_data_query(user_id: str | None = None) -> tuple[str, tuple[Any, ...]]:
    base_query = f"""
        SELECT
            user_id,
            {", ".join(FEATURE_COLUMNS)},
            predicted_label,
            productivity_score,
            timestamp
        FROM {TRAINING_TABLE}
    """
    if user_id:
        return (
            base_query + " WHERE user_id = ? ORDER BY timestamp ASC",
            (user_id,),
        )
    return (base_query + " ORDER BY timestamp ASC", ())


def get_training_data() -> pd.DataFrame:
    with get_connection() as connection:
        query, params = _get_training_data_query()
        frame = pd.read_sql_query(query, connection, params=params)

    if frame.empty:
        return pd.DataFrame(
            columns=["user_id", *FEATURE_COLUMNS, "predicted_label", "productivity_score", "timestamp"]
        )

    return frame


def get_user_training_data(user_id: str) -> pd.DataFrame:
    if not user_id:
        return pd.DataFrame(
            columns=["user_id", *FEATURE_COLUMNS, "predicted_label", "productivity_score", "timestamp"]
        )

    with get_connection() as connection:
        query, params = _get_training_data_query(user_id)
        frame = pd.read_sql_query(query, connection, params=params)

    if frame.empty:
        return pd.DataFrame(
            columns=["user_id", *FEATURE_COLUMNS, "predicted_label", "productivity_score", "timestamp"]
        )

    return frame


def get_user_personalization_context(user_id: str, lookback_days: int = 7) -> dict[str, float | int]:
    user_frame = get_user_training_data(user_id)
    if user_frame.empty:
        return {
            "avg_focus_last_7_days": 50.0,
            "avg_distraction_last_7_days": 0.0,
            "personal_baseline_score": 50.0,
            "samples_used": 0,
        }

    frame = user_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    recent_frame = frame.loc[frame["timestamp"] >= cutoff].copy()
    if recent_frame.empty:
        recent_frame = frame

    focus_value = pd.to_numeric(recent_frame["focus_score"], errors="coerce").mean()
    distraction_value = pd.to_numeric(recent_frame["distractions"], errors="coerce").mean()
    baseline_value = pd.to_numeric(frame["productivity_score"], errors="coerce").mean()

    return {
        "avg_focus_last_7_days": round(float(focus_value if pd.notna(focus_value) else 50.0), 2),
        "avg_distraction_last_7_days": round(
            float(distraction_value if pd.notna(distraction_value) else 0.0),
            2,
        ),
        "personal_baseline_score": round(float(baseline_value if pd.notna(baseline_value) else 50.0), 2),
        "samples_used": int(len(frame)),
    }


def _loads_maybe(value: str | None) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def get_latest_activity_log() -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            f"""
            SELECT *
            FROM {ACTIVITY_TABLE}
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ).fetchone()

    if not row:
        return None

    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "tracker_id": row["tracker_id"],
        "app_name": row["app_name"],
        "app_display_name": row["app_display_name"],
        "category": row["category"],
        "active_window": row["active_window"],
        "window_title": row["window_title"],
        "context": _loads_maybe(row["context_json"]),
        "cpu_percent": row["cpu_percent"],
        "memory_mb": row["memory_mb"],
        "idle_seconds": row["idle_seconds"],
        "is_idle": bool(row["is_idle"]),
        "session_seconds": row["session_seconds"],
        "coding_seconds": row["coding_seconds"],
        "distraction_seconds": row["distraction_seconds"],
        "communication_seconds": row["communication_seconds"],
        "break_seconds": row["break_seconds"],
        "app_usage": _loads_maybe(row["app_usage_json"]),
        "process_metrics": _loads_maybe(row["process_metrics_json"]),
    }


def get_recent_prediction_events(limit: int = 30, user_id: str | None = None) -> list[dict[str, Any]]:
    with get_connection() as connection:
        if user_id:
            rows = connection.execute(
                f"""
                SELECT *
                FROM {PREDICTION_EVENTS_TABLE}
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                f"""
                SELECT *
                FROM {PREDICTION_EVENTS_TABLE}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    events: list[dict[str, Any]] = []
    for row in reversed(rows):
        events.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "source": row["source"],
                "user_id": row["user_id"],
                "tracker_id": row["tracker_id"],
                "productivity_level": row["productivity_level"],
                "productivity_score": row["productivity_score"],
                "explanation": row["explanation"],
                "advice": row["advice"],
                "shap_local_plot_url": row["shap_local_plot_url"],
                "probabilities": _loads_maybe(row["probabilities_json"]),
                "features": _loads_maybe(row["features_json"]),
                "feature_importance": _loads_maybe(row["feature_importance_json"]),
                "feature_contributions": _loads_maybe(row["feature_contributions_json"]),
                "personal_insight": _loads_maybe(row["personal_insight_json"]),
                "temporal_insight": _loads_maybe(row["temporal_insight_json"]),
                "pattern_insight": _loads_maybe(row["pattern_insight_json"]),
                "proactive": _loads_maybe(row["proactive_json"]),
                "coaching": _loads_maybe(row["coaching_json"]),
                "personalization": _loads_maybe(row["personalization_json"]),
                "latest_activity": _loads_maybe(row["latest_activity_json"]),
                "app_usage": _loads_maybe(row["app_usage_json"]),
            }
        )
    return events


def get_recent_activity_logs(limit: int = 30) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            f"""
            SELECT *
            FROM {ACTIVITY_TABLE}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    entries: list[dict[str, Any]] = []
    for row in reversed(rows):
        entries.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "tracker_id": row["tracker_id"],
                "app_name": row["app_name"],
                "app_display_name": row["app_display_name"],
                "category": row["category"],
                "active_window": row["active_window"],
                "window_title": row["window_title"],
                "context": _loads_maybe(row["context_json"]),
                "cpu_percent": row["cpu_percent"],
                "memory_mb": row["memory_mb"],
                "idle_seconds": row["idle_seconds"],
                "is_idle": bool(row["is_idle"]),
                "session_seconds": row["session_seconds"],
                "coding_seconds": row["coding_seconds"],
                "distraction_seconds": row["distraction_seconds"],
                "communication_seconds": row["communication_seconds"],
                "break_seconds": row["break_seconds"],
                "app_usage": _loads_maybe(row["app_usage_json"]),
                "process_metrics": _loads_maybe(row["process_metrics_json"]),
            }
        )
    return entries


def get_new_data_count(since_timestamp: str | None = None) -> int:
    with get_connection() as connection:
        if since_timestamp:
            row = connection.execute(
                f"SELECT COUNT(*) AS total FROM {TRAINING_TABLE} WHERE timestamp > ?",
                (since_timestamp,),
            ).fetchone()
        else:
            row = connection.execute(
                f"SELECT COUNT(*) AS total FROM {TRAINING_TABLE}"
            ).fetchone()

    return int(row["total"]) if row else 0


def get_total_prediction_count() -> int:
    return get_new_data_count()
