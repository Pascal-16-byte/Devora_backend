from __future__ import annotations

from typing import Any


def _safe_number(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rolling_average(events: list[dict[str, Any]], size: int) -> float:
    if not events:
        return 0.0
    window = events[-size:]
    scores = [_safe_number(event.get("productivity_score")) for event in window]
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def _build_message(trend: str, delta_30min: float, anomaly_detected: bool) -> str:
    if anomaly_detected:
        return "Productivity dropped sharply in the latest block"
    if trend == "declining":
        return "Your productivity is dropping over the last 30 minutes"
    if trend == "improving":
        return "Your productivity is improving over the last 30 minutes"
    if abs(delta_30min) < 2:
        return "Your productivity is stable over the last 30 minutes"
    return "Your productivity is shifting slightly over the last 30 minutes"


def generate_temporal_insights(events: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_events = events[-60:] if events else []
    last_5_avg = _rolling_average(ordered_events, 5)
    last_15_avg = _rolling_average(ordered_events, 15)
    last_30_avg = _rolling_average(ordered_events, 30)

    if last_5_avg < last_15_avg < last_30_avg:
        trend = "declining"
    elif last_5_avg > last_15_avg > last_30_avg:
        trend = "improving"
    else:
        trend = "stable"

    delta_30min = round(last_5_avg - last_15_avg, 2)

    recent_scores = [_safe_number(event.get("productivity_score")) for event in ordered_events[-2:]]
    anomaly_detected = (
        len(recent_scores) == 2
        and recent_scores[-2] - recent_scores[-1] > 25
    )

    return {
        "trend": trend,
        "last_5_avg": last_5_avg,
        "last_15_avg": last_15_avg,
        "last_30_avg": last_30_avg,
        "delta_30min": delta_30min,
        "anomaly_detected": anomaly_detected,
        "message": _build_message(trend, delta_30min, anomaly_detected),
    }
