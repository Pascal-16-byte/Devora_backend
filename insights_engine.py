from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

FEATURE_LABELS = {
    "coding_hours": "coding hours",
    "learning_time": "learning time",
    "commits_per_day": "commits",
    "lines_of_code": "lines of code",
    "bugs_fixed": "bugs fixed",
    "sleep_hours": "sleep",
    "distractions": "distractions",
    "deep_work_ratio": "deep work ratio",
    "distraction_intensity": "distraction intensity",
    "task_completion_rate": "task completion",
    "meetings_per_day": "meetings",
    "break_time": "break time",
    "focus_score": "focus score",
}

def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_number(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _avg(items: list[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def _pct_change(current: float, baseline: float) -> float:
    if abs(baseline) < 1e-6:
        return 0.0 if abs(current) < 1e-6 else 100.0
    return ((current - baseline) / abs(baseline)) * 100.0


def _collect_feature_average(events: list[dict[str, Any]], feature: str) -> float:
    return _avg([_safe_number((event.get("features") or {}).get(feature)) for event in events])


def _build_baseline(events: list[dict[str, Any]]) -> dict[str, float]:
    feature_keys = list(FEATURE_LABELS.keys())
    return {
        "avg_productivity_score": _avg([_safe_number(event.get("productivity_score")) for event in events]),
        **{f"avg_{feature}": _collect_feature_average(events, feature) for feature in feature_keys},
    }


def _format_feature_delta(feature: str, current_value: float, previous_value: float) -> str:
    diff = current_value - previous_value
    label = FEATURE_LABELS.get(feature, feature.replace("_", " "))

    if feature in {"meetings_per_day", "bugs_fixed", "commits_per_day"}:
        if abs(diff) < 0.2:
            return f"{label.capitalize()} were steady"
        direction = "increased" if diff > 0 else "decreased"
        return f"{label.capitalize()} {direction} by {abs(diff):.1f}"

    if feature == "task_completion_rate":
        return f"Task completion changed by {abs(diff) * 100:.0f}%"

    return f"{label.capitalize()} changed by {abs(diff):.0f}%"


def _build_summary(score_change: float, current_score: float, yesterday_score: float) -> str:
    if abs(score_change) < 2:
        return f"You are holding steady versus yesterday at {current_score:.0f}/100"
    if score_change > 0:
        return f"You are {abs(score_change):.0f}% more productive than yesterday"
    return f"You are {abs(score_change):.0f}% less focused than yesterday"


def _build_coach_message(
    score_change: float,
    drivers: list[str],
    feature_today: dict[str, float],
    baseline: dict[str, float],
    momentum: bool,
) -> str:
    if momentum:
        return "Momentum is building. Protect the routines that are improving focus and keep your next block interruption-free."

    distractions = feature_today.get("distractions", 0.0)
    avg_distractions = baseline.get("avg_distractions", 0.0)
    meetings = feature_today.get("meetings_per_day", 0.0)
    avg_meetings = baseline.get("avg_meetings_per_day", 0.0)
    coding_hours = feature_today.get("coding_hours", 0.0)
    focus_score = feature_today.get("focus_score", 0.0)

    if distractions > avg_distractions * 1.3 and distractions > 0:
        return (
            f"You had {abs(_pct_change(distractions, avg_distractions)):.0f}% more distractions than your usual pattern. "
            "That is the clearest reason your score slipped, so try a 90-minute uninterrupted deep work block."
        )
    if meetings > avg_meetings and focus_score < baseline.get("avg_focus_score", 0.0):
        return "Meetings are crowding out deep work. Batch follow-ups and reserve one protected focus window after meetings."
    if coding_hours >= baseline.get("avg_coding_hours", 0.0) and score_change < 0:
        return "You are putting in time, but the workday looks fragmented. Reduce context switching before adding more hours."
    if drivers:
        return f"{drivers[0]}. Tighten the next session around one priority and reduce switching costs."
    return "Your pattern is stable. Keep the next block focused on one meaningful task."


def generate_comparative_insights(
    current_event: dict[str, Any],
    past_events: list[dict[str, Any]],
) -> dict[str, Any]:
    current_timestamp = _parse_timestamp(current_event.get("timestamp"))
    all_events = sorted(
        [*past_events, current_event],
        key=lambda event: _parse_timestamp(event.get("timestamp")),
    )
    lookback_start = current_timestamp - timedelta(hours=48)
    relevant_events = [
        event for event in all_events if _parse_timestamp(event.get("timestamp")) >= lookback_start
    ]

    today_start = current_timestamp - timedelta(hours=24)
    yesterday_start = current_timestamp - timedelta(hours=48)

    today_events = [
        event for event in relevant_events if _parse_timestamp(event.get("timestamp")) >= today_start
    ]
    yesterday_events = [
        event
        for event in relevant_events
        if yesterday_start <= _parse_timestamp(event.get("timestamp")) < today_start
    ]

    historical_events = sorted(past_events, key=lambda event: _parse_timestamp(event.get("timestamp")))
    baseline_events = historical_events[-20:] if historical_events else []
    baseline = _build_baseline(baseline_events or today_events or [current_event])

    avg_score_today = _avg([_safe_number(event.get("productivity_score")) for event in today_events])
    avg_score_yesterday = _avg([_safe_number(event.get("productivity_score")) for event in yesterday_events])
    if not yesterday_events:
        avg_score_yesterday = baseline.get("avg_productivity_score", avg_score_today)

    score_change = round(avg_score_today - avg_score_yesterday, 1)
    feature_today = {
        feature: _collect_feature_average(today_events, feature) for feature in FEATURE_LABELS
    }
    feature_yesterday = {
        feature: _collect_feature_average(yesterday_events, feature) for feature in FEATURE_LABELS
    }
    if not yesterday_events:
        for feature in FEATURE_LABELS:
            feature_yesterday[feature] = baseline.get(f"avg_{feature}", feature_today[feature])

    drivers: list[str] = []
    alerts: list[str] = []
    feature_deltas: dict[str, dict[str, float]] = {}

    for feature in FEATURE_LABELS:
        today_value = feature_today[feature]
        yesterday_value = feature_yesterday[feature]
        delta = today_value - yesterday_value
        delta_pct = _pct_change(today_value, yesterday_value)
        feature_deltas[feature] = {
            "today": round(today_value, 2),
            "yesterday": round(yesterday_value, 2),
            "delta": round(delta, 2),
            "delta_pct": round(delta_pct, 1),
        }

    current_features = current_event.get("features") or {}
    current_distractions = _safe_number(current_features.get("distractions"))
    current_meetings = _safe_number(current_features.get("meetings_per_day"))
    current_focus = _safe_number(current_features.get("focus_score"))
    current_coding_hours = _safe_number(current_features.get("coding_hours"))

    if (
        feature_deltas["distractions"]["delta"] > 0
        and score_change < 0
    ):
        drivers.append(f"Distractions increased by {abs(feature_deltas['distractions']['delta_pct']):.0f}%")

    if feature_deltas["meetings_per_day"]["delta"] > 0:
        drivers.append(
            f"Meetings increased by {abs(feature_deltas['meetings_per_day']['delta']):.1f}"
        )

    if feature_deltas["focus_score"]["delta"] < 0:
        drivers.append(f"Focus score fell by {abs(feature_deltas['focus_score']['delta']):.0f} points")

    if baseline.get("avg_distractions", 0.0) > 0 and current_distractions > baseline["avg_distractions"] * 1.3:
        alerts.append("You are getting interrupted more than usual")
        drivers.append(
            f"Distractions are {abs(_pct_change(current_distractions, baseline['avg_distractions'])):.0f}% above your norm"
        )

    if current_meetings > baseline.get("avg_meetings_per_day", 0.0) and current_focus < baseline.get("avg_focus_score", 0.0):
        alerts.append("Meetings are hurting your deep work")

    if current_coding_hours >= baseline.get("avg_coding_hours", 0.0) and _safe_number(current_event.get("productivity_score")) < baseline.get("avg_productivity_score", 0.0):
        alerts.append("You are busy, but not productive (likely context switching)")

    last_three = all_events[-3:]
    momentum = (
        len(last_three) == 3
        and last_three[0].get("productivity_score") is not None
        and _safe_number(last_three[0].get("productivity_score"))
        < _safe_number(last_three[1].get("productivity_score"))
        < _safe_number(last_three[2].get("productivity_score"))
    )
    if momentum:
        alerts.append("Momentum is building - keep this pattern")

    ranked_drivers = sorted(
        feature_deltas.items(),
        key=lambda item: abs(item[1]["delta_pct"]),
        reverse=True,
    )
    for feature, delta_info in ranked_drivers:
        if len(drivers) >= 3:
            break
        if abs(delta_info["delta"]) < 0.1:
            continue
        description = _format_feature_delta(feature, delta_info["today"], delta_info["yesterday"])
        if description not in drivers:
            drivers.append(description)

    summary = _build_summary(score_change, avg_score_today, avg_score_yesterday)
    coach_message = _build_coach_message(score_change, drivers, feature_today, baseline, momentum)

    return {
        "score_change": round(score_change, 1),
        "avg_score_today": round(avg_score_today, 1),
        "avg_score_yesterday": round(avg_score_yesterday, 1),
        "summary": summary,
        "drivers": drivers[:3],
        "coach_message": coach_message,
        "alerts": alerts[:4],
        "baseline": {
            "avg_focus_score": round(baseline.get("avg_focus_score", 0.0), 1),
            "avg_distractions": round(baseline.get("avg_distractions", 0.0), 1),
            "avg_meetings": round(baseline.get("avg_meetings_per_day", 0.0), 1),
            "avg_productivity_score": round(baseline.get("avg_productivity_score", 0.0), 1),
        },
        "feature_deltas": feature_deltas,
    }
