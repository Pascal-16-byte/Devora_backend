from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _safe_number(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _format_hour(hour: int) -> str:
    suffix = "AM" if hour < 12 else "PM"
    normalized = hour % 12 or 12
    return f"{normalized} {suffix}"


def _format_hour_range(hours: list[int]) -> str:
    if not hours:
        return "Not enough data yet"
    ordered = sorted(hours)
    return f"{_format_hour(ordered[0])} - {_format_hour((ordered[-1] + 1) % 24)}"


def _extract_distraction_value(event: dict[str, Any]) -> float:
    features = event.get("features") or {}
    feature_distractions = _safe_number(features.get("distractions"))
    app_usage = event.get("app_usage") or {}
    distraction_minutes = _safe_number(((app_usage.get("categories") or {}).get("distraction") or {}).get("minutes"))
    return max(feature_distractions, distraction_minutes)


def generate_behavioral_patterns(events: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_events = events[-60:] if events else []
    hourly_scores: dict[int, list[float]] = {}
    hourly_distractions: dict[int, list[float]] = {}
    weekday_scores: dict[int, list[float]] = {}

    for event in ordered_events:
        timestamp = _parse_timestamp(event.get("timestamp"))
        hour = timestamp.hour
        weekday = timestamp.weekday()
        hourly_scores.setdefault(hour, []).append(_safe_number(event.get("productivity_score")))
        hourly_distractions.setdefault(hour, []).append(_extract_distraction_value(event))
        weekday_scores.setdefault(weekday, []).append(_safe_number(event.get("productivity_score")))

    ranked_hours = sorted(
        ((hour, _avg(scores)) for hour, scores in hourly_scores.items()),
        key=lambda item: item[1],
    )
    best_hours = sorted([hour for hour, _ in ranked_hours[-2:]]) if ranked_hours else []
    worst_hours = sorted([hour for hour, _ in ranked_hours[:2]]) if ranked_hours else []

    ranked_distraction_hours = sorted(
        ((hour, _avg(values)) for hour, values in hourly_distractions.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    distraction_peak_hour = ranked_distraction_hours[0][0] if ranked_distraction_hours else None

    ranked_weekdays = sorted(
        ((weekday, _avg(scores)) for weekday, scores in weekday_scores.items()),
        key=lambda item: item[1],
    )
    best_day = WEEKDAY_NAMES[ranked_weekdays[-1][0]] if ranked_weekdays else None
    worst_day = WEEKDAY_NAMES[ranked_weekdays[0][0]] if ranked_weekdays else None

    insights: list[str] = []
    if best_hours:
        if max(best_hours) < 12:
            insights.append("You work best in the morning")
        elif min(best_hours) >= 12:
            insights.append("Your strongest work tends to happen later in the day")
        else:
            insights.append(f"You work best between {_format_hour_range(best_hours)}")
    if distraction_peak_hour is not None:
        if distraction_peak_hour >= 12:
            insights.append("Afternoons show high distraction")
        else:
            insights.append("Distractions spike earlier in the day")
    if best_day and worst_day and best_day != worst_day:
        insights.append(f"{best_day} is usually your strongest day")

    return {
        "best_hours": best_hours,
        "worst_hours": worst_hours,
        "best_day": best_day,
        "worst_day": worst_day,
        "distraction_peak_hour": distraction_peak_hour,
        "best_time_message": (
            f"You work best between {_format_hour_range(best_hours)}"
            if best_hours
            else "Devora is still learning your best work window"
        ),
        "risk_zone_message": (
            f"You are most distracted around {_format_hour(distraction_peak_hour)}"
            if distraction_peak_hour is not None
            else "Devora is still learning your distraction risk zones"
        ),
        "insights": insights[:3],
    }
