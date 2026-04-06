from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
    from .insights_engine import generate_comparative_insights
except ImportError:
    from insights_engine import generate_comparative_insights


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


def _safe_number(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_hour(hour: int) -> str:
    suffix = "AM" if hour < 12 else "PM"
    normalized = hour % 12 or 12
    return f"{normalized} {suffix}"


def _format_hour_window(hours: list[int]) -> str:
    if not hours:
        return ""
    ordered = sorted(hours)
    start = ordered[0]
    end = ordered[-1] + 1
    return f"{_format_hour(start)} - {_format_hour(end)}"


def _metric_average(events: list[dict[str, Any]], feature: str) -> float:
    return _avg([_safe_number((event.get("features") or {}).get(feature)) for event in events])


def compute_time_of_day_patterns(events: list[dict[str, Any]]) -> dict[str, list[int]]:
    hourly_scores: dict[int, list[float]] = {}
    for event in events:
        score = _safe_number(event.get("productivity_score"))
        timestamp = _parse_timestamp(event.get("timestamp"))
        hourly_scores.setdefault(timestamp.hour, []).append(score)

    if not hourly_scores:
        return {"best_hours": [], "worst_hours": []}

    ranked_hours = sorted(
        ((hour, _avg(scores)) for hour, scores in hourly_scores.items()),
        key=lambda item: item[1],
    )
    worst_hours = sorted([hour for hour, _ in ranked_hours[:2]])
    best_hours = sorted([hour for hour, _ in ranked_hours[-2:]])
    return {"best_hours": best_hours, "worst_hours": worst_hours}


def _compute_personal_baseline(events: list[dict[str, Any]]) -> dict[str, Any]:
    recent_events = events[-30:] if events else []
    hourly_distribution: dict[int, float] = {}
    for hour in range(24):
        scores = [
            _safe_number(event.get("productivity_score"))
            for event in recent_events
            if _parse_timestamp(event.get("timestamp")).hour == hour
        ]
        if scores:
            hourly_distribution[hour] = round(_avg(scores), 1)

    return {
        "avg_focus_score": round(_metric_average(recent_events, "focus_score"), 1),
        "avg_distractions": round(_metric_average(recent_events, "distractions"), 1),
        "avg_meetings": round(_metric_average(recent_events, "meetings_per_day"), 1),
        "avg_productivity_score": round(
            _avg([_safe_number(event.get("productivity_score")) for event in recent_events]),
            1,
        ),
        "hourly_productivity_distribution": hourly_distribution,
    }


def _build_headline(score_change: float) -> tuple[str, str]:
    if score_change < -5:
        return f"You are {abs(score_change):.0f}% less focused than yesterday", "declining"
    if score_change > 5:
        return f"You are {abs(score_change):.0f}% more productive than yesterday", "improving"
    return "You are maintaining consistent productivity", "stable"


def _reason_from_delta(feature: str, delta_info: dict[str, Any]) -> str | None:
    delta = _safe_number(delta_info.get("delta"))
    delta_pct = _safe_number(delta_info.get("delta_pct"))
    label = FEATURE_LABELS.get(feature, feature.replace("_", " "))

    if feature == "focus_score" and abs(delta) >= 1:
        direction = "fell" if delta < 0 else "improved"
        return f"Focus score {direction} by {abs(delta):.0f} points"
    if feature == "task_completion_rate" and abs(delta) >= 0.03:
        direction = "dropped" if delta < 0 else "improved"
        return f"Task completion {direction} by {abs(delta) * 100:.0f}%"
    if feature in {"meetings_per_day", "commits_per_day", "bugs_fixed"} and abs(delta) >= 0.2:
        direction = "increased" if delta > 0 else "decreased"
        return f"{label.capitalize()} {direction} by {abs(delta):.1f}"
    if abs(delta_pct) >= 8:
        direction = "increased" if delta > 0 else "decreased"
        return f"{label.capitalize()} {direction} by {abs(delta_pct):.0f}%"
    return None


def _build_reasons(
    comparative_insight: dict[str, Any],
    current_event: dict[str, Any],
    baseline_memory: dict[str, Any],
) -> tuple[str, str]:
    drivers = [str(item) for item in comparative_insight.get("drivers") or [] if item]
    feature_deltas = comparative_insight.get("feature_deltas") or {}
    reason_candidates = list(drivers)

    ranked_features = sorted(
        feature_deltas.items(),
        key=lambda item: abs(_safe_number((item[1] or {}).get("delta_pct"))),
        reverse=True,
    )
    for feature, delta_info in ranked_features:
        reason = _reason_from_delta(feature, delta_info or {})
        if reason and reason not in reason_candidates:
            reason_candidates.append(reason)

    current_features = current_event.get("features") or {}
    current_distractions = _safe_number(current_features.get("distractions"))
    current_meetings = _safe_number(current_features.get("meetings_per_day"))
    current_focus = _safe_number(current_features.get("focus_score"))

    if (
        baseline_memory.get("avg_distractions", 0.0) > 0
        and current_distractions > baseline_memory["avg_distractions"] * 1.25
    ):
        reason_candidates.insert(
            0,
            f"Distractions increased by {abs(((current_distractions - baseline_memory['avg_distractions']) / baseline_memory['avg_distractions']) * 100):.0f}%",
        )

    if current_meetings > baseline_memory.get("avg_meetings", 0.0) and current_focus < baseline_memory.get("avg_focus_score", 0.0):
        reason_candidates.append(
            f"Meetings increased by {abs(current_meetings - baseline_memory.get('avg_meetings', 0.0)):.1f}"
        )

    primary = reason_candidates[0] if reason_candidates else "Your core work signals are near your usual baseline"
    secondary = (
        reason_candidates[1]
        if len(reason_candidates) > 1
        else "The rest of your signals are relatively stable"
    )
    return primary, secondary


def _build_action_items(
    comparative_insight: dict[str, Any],
    current_event: dict[str, Any],
    baseline_memory: dict[str, Any],
) -> list[str]:
    items: list[str] = []
    feature_deltas = comparative_insight.get("feature_deltas") or {}
    alerts = [str(item) for item in comparative_insight.get("alerts") or [] if item]
    current_features = current_event.get("features") or {}
    score_change = _safe_number(comparative_insight.get("score_change"))

    distractions_delta = _safe_number((feature_deltas.get("distractions") or {}).get("delta"))
    meetings_delta = _safe_number((feature_deltas.get("meetings_per_day") or {}).get("delta"))
    coding_delta = _safe_number((feature_deltas.get("coding_hours") or {}).get("delta"))
    focus_delta = _safe_number((feature_deltas.get("focus_score") or {}).get("delta"))
    current_score = _safe_number(current_event.get("productivity_score"))

    if distractions_delta > 0 or _safe_number(current_features.get("distractions")) > baseline_memory.get("avg_distractions", 0):
        items.append("Reduce interruptions (try focus mode)")
    if meetings_delta > 0 and focus_delta < 0:
        items.append("Batch meetings and protect deep work")
    if coding_delta > 0 and current_score < baseline_memory.get("avg_productivity_score", current_score):
        items.append("You are busy but not productive - reduce context switching")
    if any("Momentum is building" in alert for alert in alerts):
        items.append("You're on a productive streak - protect this block")
    if score_change < 0:
        items.append("Block 90 minutes for deep work")
    if distractions_delta > 0:
        items.append("Silence notifications")
    if meetings_delta > 0 or focus_delta < 0:
        items.append("Avoid context switching")
    if not items:
        items.append("Keep your next block focused on one important task")

    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped[:3]


def _build_pattern_insight(events: list[dict[str, Any]]) -> tuple[str, dict[str, list[int]]]:
    patterns = compute_time_of_day_patterns(events)
    best_hours = patterns.get("best_hours") or []
    worst_hours = patterns.get("worst_hours") or []

    if best_hours:
        return f"Your best work time is {_format_hour_window(best_hours)}", patterns
    if worst_hours:
        return f"Your lowest-energy stretch is {_format_hour_window(worst_hours)}", patterns
    return "Devora is still learning your best work rhythm", patterns


def _build_intensity(trend: str, current_score: float, alerts: list[str]) -> str:
    if trend == "declining" or any("busy, but not productive" in alert for alert in alerts):
        return "warning"
    if trend == "improving" and current_score >= 75:
        return "optimize"
    return "nudge"


def generate_coaching_feedback(
    current_event: dict[str, Any],
    past_events: list[dict[str, Any]],
) -> dict[str, Any]:
    comparative_insight = current_event.get("personal_insight")
    if not comparative_insight:
        comparative_insight = generate_comparative_insights(current_event, past_events)

    history = [*past_events, current_event]
    baseline_memory = _compute_personal_baseline(history)
    headline, trend = _build_headline(_safe_number(comparative_insight.get("score_change")))
    primary_reason, secondary_reason = _build_reasons(
        comparative_insight,
        current_event,
        baseline_memory,
    )
    action_items = _build_action_items(
        comparative_insight,
        current_event,
        baseline_memory,
    )
    pattern_insight, time_patterns = _build_pattern_insight(history[-60:])
    alerts = [str(item) for item in comparative_insight.get("alerts") or [] if item]
    current_score = _safe_number(current_event.get("productivity_score"))

    return {
        "headline": headline,
        "primary_reason": primary_reason,
        "secondary_reason": secondary_reason,
        "action_items": action_items,
        "pattern_insight": pattern_insight,
        "trend": trend,
        "intensity": _build_intensity(trend, current_score, alerts),
        "baseline_memory": baseline_memory,
        "time_patterns": time_patterns,
    }
