from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

DEFAULT_ACTION = "Keep your current focus block protected"
MAX_ALERTS = 2

RULE_PRIORITIES = {
    "distraction": 0,
    "momentum": 1,
    "focus": 2,
    "context_switch": 3,
    "time_risk": 4,
}

SEVERITY_RANK = {
    "high": 0,
    "medium": 1,
}


def _safe_number(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []

    normalized: list[int] = []
    for value in values:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _parse_timestamp_hour(value: str | None) -> int | None:
    if not value:
        return None

    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).hour


def _extract_features(current_features: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    payload = current_features or {}
    features = payload.get("features", payload)
    app_usage = payload.get("app_usage") or {}
    timestamp = payload.get("timestamp")
    return features, app_usage, timestamp


def _event_distractions(event: dict[str, Any]) -> float:
    features = (event or {}).get("features") or {}
    return _safe_number(features.get("distractions"))


def _event_score(event: dict[str, Any]) -> float:
    return _safe_number((event or {}).get("productivity_score"))


def _append_rule(triggered: list[dict[str, str]], rule_id: str, severity: str, alert: str, action: str) -> None:
    triggered.append(
        {
            "rule_id": rule_id,
            "severity": severity,
            "alert": alert,
            "action": action,
        }
    )


def generate_proactive_alerts(
    current_features: dict[str, Any] | None,
    recent_events: list[dict[str, Any]] | None,
    pattern_insight: dict[str, Any] | None,
) -> dict[str, Any]:
    features, app_usage, timestamp = _extract_features(current_features)
    events = recent_events or []
    pattern = pattern_insight or {}
    triggered: list[dict[str, str]] = []

    has_distractions = "distractions" in features
    distractions = _safe_number(features.get("distractions")) if has_distractions else 0.0
    recent_distractions = [_event_distractions(event) for event in events[-3:]]
    distraction_trend_up = (
        len(recent_distractions) == 3
        and recent_distractions[0] < recent_distractions[1] < recent_distractions[2]
    )
    if (has_distractions and distractions > 6) or distraction_trend_up:
        _append_rule(
            triggered,
            "distraction",
            "high",
            "You are entering a distraction zone",
            "Close distraction apps and start 30 min focus block",
        )

    has_focus_score = "focus_score" in features
    focus_score = _safe_number(features.get("focus_score")) if has_focus_score else 0.0
    if has_focus_score and focus_score < 50:
        _append_rule(
            triggered,
            "focus",
            "high",
            "Your focus is dropping",
            "Take a short break or reset environment",
        )

    context_switches = _safe_number(app_usage.get("context_switches"))
    active_seconds = max(_safe_number(app_usage.get("active_seconds")), 0.0)
    switch_rate = context_switches / max(active_seconds / 300.0, 1.0)
    if context_switches >= 8 or switch_rate >= 3:
        _append_rule(
            triggered,
            "context_switch",
            "medium",
            "Too many task switches detected",
            "Stick to one task for next 45 mins",
        )

    current_hour = _parse_timestamp_hour(timestamp)
    worst_hours = _safe_int_list(pattern.get("worst_hours"))
    if current_hour is not None and current_hour in worst_hours:
        _append_rule(
            triggered,
            "time_risk",
            "medium",
            "This is usually your low productivity window",
            "Reduce meetings / protect focus",
        )

    recent_scores = [_event_score(event) for event in events[-3:]]
    if len(recent_scores) == 3 and recent_scores[0] > recent_scores[1] > recent_scores[2]:
        _append_rule(
            triggered,
            "momentum",
            "high",
            "Your productivity trend is declining",
            "Pause and reset your plan",
        )

    if not triggered:
        return {
            "risk_level": "low",
            "alerts": [],
            "recommended_action": DEFAULT_ACTION,
        }

    triggered.sort(
        key=lambda rule: (
            SEVERITY_RANK.get(rule["severity"], 99),
            RULE_PRIORITIES.get(rule["rule_id"], 99),
        )
    )

    alerts = [rule["alert"] for rule in triggered[:MAX_ALERTS]]
    top_rule = triggered[0]
    risk_level = "high" if any(rule["severity"] == "high" for rule in triggered) else "medium"

    return {
        "risk_level": risk_level,
        "alerts": alerts,
        "recommended_action": top_rule["action"],
    }
