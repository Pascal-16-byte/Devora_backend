"""
Map raw machine activity into the existing Devora ML feature schema.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

try:
    from .context_engine import analyze_context, extract_domain_from_title, is_browser_app
except ImportError:
    from context_engine import analyze_context, extract_domain_from_title, is_browser_app


APP_RULES = {
    "coding": [
        "code",
        "cursor",
        "pycharm",
        "idea",
        "webstorm",
        "sublime",
        "notepad++",
        "terminal",
        "powershell",
        "cmd",
        "windows terminal",
        "devenv",
        "rider",
        "vim",
        "nvim",
        "git",
        "docker",
        "postman",
    ],
    "distraction": [
        "youtube",
        "netflix",
        "spotify",
        "discord",
        "twitter",
        "x ",
        "instagram",
        "facebook",
        "reddit",
        "prime video",
        "twitch",
        "anime",
        "watch",
        "movie",
        "music",
        "reel",
        "shorts",
        "game",
        "steam",
    ],
    "communication": [
        "slack",
        "teams",
        "zoom",
        "meet",
        "outlook",
        "mail",
        "telegram",
        "whatsapp",
        "skype",
        "webex",
    ],
}

TITLE_HINTS = {
    "coding": [
        "pull request",
        "github",
        "gitlab",
        "stack overflow",
        "stackoverflow",
        "documentation",
        "docs",
        "localhost",
        ".py",
        ".js",
        ".ts",
        ".java",
    ],
    "distraction": [
        "youtube",
        "netflix",
        "music",
        "twitch",
        "highlights",
        "trailer",
        "anime",
        "watch",
        "movie",
        "prime video",
        "reel",
        "shorts",
    ],
    "communication": ["meeting", "call", "huddle", "zoom", "teams", "standup", "slack", "mail", "inbox"],
}

DOMAIN_RULES = {
    "coding": [
        "github",
        "gitlab",
        "bitbucket",
        "stack overflow",
        "stackoverflow",
        "docs",
        "documentation",
        "readthedocs",
        "notion",
        "codepen",
        "chatgpt",
        "openai",
        "localhost",
        "127.0.0.1",
    ],
    "distraction": [
        "youtube",
        "netflix",
        "prime video",
        "instagram",
        "twitter",
        "x",
        "facebook",
        "reddit",
        "twitch",
        "hotstar",
    ],
    "communication": [
        "gmail",
        "mail",
        "outlook",
        "slack",
        "discord",
        "whatsapp",
        "teams",
        "meet",
        "zoom",
    ],
}

DEBUG_KEYWORDS = ["bug", "issue", "error", "traceback", "debug", "exception", "fix"]
DEFAULT_SLEEP_HOURS = float(os.getenv("DEVINSIGHT_DEFAULT_SLEEP_HOURS", "7.0"))
DOMAIN_PRIORITY = ("coding", "communication", "distraction")


def _normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def _score_rule_matches(text: str, rule_map: dict[str, list[str]]) -> dict[str, int]:
    return {
        category: sum(1 for keyword in keywords if keyword in text)
        for category, keywords in rule_map.items()
    }


def _best_scored_category(text: str, rule_map: dict[str, list[str]]) -> str | None:
    if not text:
        return None

    scores = _score_rule_matches(text, rule_map)
    best_score = max(scores.values(), default=0)
    if best_score <= 0:
        return None

    tied_categories = [category for category, score in scores.items() if score == best_score]
    for category in DOMAIN_PRIORITY:
        if category in tied_categories:
            return category
    return tied_categories[0]


def categorize_app(app_name: str | None, window_title: str | None = None) -> str:
    normalized_app = _normalize(app_name)
    normalized_title = _normalize(window_title)
    context = analyze_context(normalized_app, normalized_title)
    intent = context.get("intent")

    if intent in {"coding", "learning"}:
        return "coding"
    if intent == "distraction":
        return "distraction"
    if intent == "communication":
        return "communication"

    for category, keywords in APP_RULES.items():
        if any(keyword in normalized_app for keyword in keywords):
            return category

    if is_browser_app(normalized_app):
        domain_context = extract_domain_from_title(normalized_title)
        domain_category = _best_scored_category(domain_context, DOMAIN_RULES)
        if domain_category:
            return domain_category

        title_category = _best_scored_category(normalized_title, TITLE_HINTS)
        if title_category:
            return title_category

        return "other"

    title_category = _best_scored_category(normalized_title, TITLE_HINTS)
    if title_category:
        return title_category

    return "other"


@dataclass
class SessionFeatureState:
    session_seconds: float = 0.0
    active_seconds: float = 0.0
    idle_seconds: float = 0.0
    coding_seconds: float = 0.0
    learning_seconds: float = 0.0
    distraction_seconds: float = 0.0
    communication_seconds: float = 0.0
    other_seconds: float = 0.0
    deep_work_seconds: float = 0.0
    debug_seconds: float = 0.0
    distraction_weighted_seconds: float = 0.0
    keystrokes: int = 0
    mouse_clicks: int = 0
    mouse_scrolls: int = 0
    context_switches: int = 0
    distraction_switches: int = 0
    last_app_name: str | None = None
    app_usage_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    category_usage_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    intent_usage_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))


def update_session_state(state: SessionFeatureState, snapshot: dict[str, Any], elapsed_seconds: float) -> SessionFeatureState:
    app_name = snapshot.get("app_name") or "Unknown"
    window_title = snapshot.get("window_title") or ""
    category = snapshot.get("category") or categorize_app(app_name, window_title)
    context = snapshot.get("context") or analyze_context(app_name, window_title)
    intent = str(context.get("intent") or "neutral")
    intent_confidence = float(context.get("confidence") or 0.3)
    is_idle = bool(snapshot.get("is_idle"))
    activity_counts = snapshot.get("input_activity") or {}
    interaction_count = (
        int(activity_counts.get("keystrokes", 0))
        + int(activity_counts.get("mouse_clicks", 0))
        + int(activity_counts.get("mouse_scrolls", 0))
    )

    state.session_seconds += max(elapsed_seconds, 0.0)
    if is_idle:
        state.idle_seconds += elapsed_seconds
    else:
        state.active_seconds += elapsed_seconds

    if state.last_app_name and state.last_app_name != app_name:
        state.context_switches += 1
        if category == "distraction":
            state.distraction_switches += 1

    state.last_app_name = app_name
    state.app_usage_seconds[app_name] += elapsed_seconds
    state.category_usage_seconds[category] += elapsed_seconds
    state.intent_usage_seconds[intent] += elapsed_seconds

    if category == "coding":
        state.coding_seconds += elapsed_seconds
    elif category == "distraction":
        state.distraction_seconds += elapsed_seconds
    elif category == "communication":
        state.communication_seconds += elapsed_seconds
    else:
        state.other_seconds += elapsed_seconds

    if intent == "learning":
        state.learning_seconds += elapsed_seconds

    if intent == "distraction":
        state.distraction_weighted_seconds += elapsed_seconds * max(0.3, min(intent_confidence, 1.0))

    if (
        not is_idle
        and intent in {"coding", "learning"}
        and intent_confidence >= 0.6
        and (interaction_count > 0 or category == "coding")
    ):
        state.deep_work_seconds += elapsed_seconds

    normalized_title = _normalize(window_title)
    if any(keyword in normalized_title for keyword in DEBUG_KEYWORDS):
        state.debug_seconds += elapsed_seconds

    state.keystrokes += int(activity_counts.get("keystrokes", 0))
    state.mouse_clicks += int(activity_counts.get("mouse_clicks", 0))
    state.mouse_scrolls += int(activity_counts.get("mouse_scrolls", 0))
    return state


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _estimate_focus_score(state: SessionFeatureState, avg_cpu_percent: float) -> float:
    active = max(state.active_seconds, 1.0)
    coding_ratio = _safe_ratio(state.coding_seconds, active)
    learning_ratio = _safe_ratio(state.learning_seconds, active)
    deep_work_ratio = _safe_ratio(state.deep_work_seconds, active)
    distraction_ratio = _safe_ratio(state.distraction_seconds, active)
    communication_ratio = _safe_ratio(state.communication_seconds, active)
    context_penalty = min(state.context_switches / max(active / 300, 1.0), 10.0)
    activity_bonus = min((state.keystrokes / max(active / 60, 1.0)) * 2.5, 20.0)
    cpu_bonus = min(avg_cpu_percent * 0.35, 12.0)

    score = (
        38
        + coding_ratio * 28
        + learning_ratio * 10
        + deep_work_ratio * 18
        - distraction_ratio * 26
        - communication_ratio * 8
        - context_penalty * 2.5
        + activity_bonus
        + cpu_bonus
    )
    return round(max(0.0, min(score, 100.0)), 1)


def build_prediction_features(
    state: SessionFeatureState,
    latest_snapshot: dict[str, Any],
    avg_cpu_percent: float = 0.0,
    personalization_context: dict[str, float] | None = None,
) -> dict[str, float]:
    session_hours = state.session_seconds / 3600
    coding_hours = round(state.coding_seconds / 3600, 2)
    learning_time = round(state.learning_seconds / 3600, 2)
    distraction_minutes = state.distraction_seconds / 60
    communication_minutes = state.communication_seconds / 60
    idle_minutes = state.idle_seconds / 60
    deep_work_ratio = round(_safe_ratio(state.deep_work_seconds, max(state.active_seconds, 1.0)), 3)
    distraction_intensity = round(
        min(
            30.0,
            state.distraction_switches + (state.distraction_weighted_seconds / 60.0) / 5.0,
        ),
        2,
    )
    focus_score = _estimate_focus_score(state, avg_cpu_percent)

    commits_per_day = round(
        min(
            12.0,
            (state.keystrokes / 450.0)
            + (state.coding_seconds / 5400.0)
            + (state.learning_seconds / 7200.0)
            + (state.context_switches / 25.0),
        ),
        2,
    )
    lines_of_code = round(
        min(
            4000.0,
            (state.keystrokes * 1.7)
            + (state.coding_seconds / 12.0)
            - (state.mouse_clicks * 0.4),
        ),
        1,
    )
    bugs_fixed = round(
        min(
            15.0,
            (state.debug_seconds / 2700.0) + (state.communication_seconds / 7200.0),
        ),
        2,
    )
    distractions = round(
        max(distraction_intensity, min(30.0, state.distraction_switches + distraction_minutes / 6.0)),
        2,
    )
    task_completion_rate = round(
        max(
            0.0,
            min(
                1.0,
                (focus_score / 100.0) * 0.55
                + _safe_ratio(state.coding_seconds, max(state.active_seconds, 1.0)) * 0.35
                + deep_work_ratio * 0.15
                + max(0.0, 1 - _safe_ratio(state.distraction_seconds, max(state.active_seconds, 1.0))) * 0.10,
            ),
        ),
        3,
    )
    meetings_per_day = round(min(12.0, communication_minutes / 30.0), 2)
    break_time = round(min(480.0, idle_minutes + max(0.0, state.other_seconds / 120.0)), 1)
    personalization_context = personalization_context or {}

    return {
        "coding_hours": coding_hours,
        "learning_time": learning_time,
        "commits_per_day": commits_per_day,
        "lines_of_code": max(0.0, lines_of_code),
        "bugs_fixed": bugs_fixed,
        "sleep_hours": DEFAULT_SLEEP_HOURS,
        "distractions": distractions,
        "deep_work_ratio": deep_work_ratio,
        "distraction_intensity": distraction_intensity,
        "task_completion_rate": task_completion_rate,
        "meetings_per_day": meetings_per_day,
        "break_time": break_time,
        "focus_score": focus_score,
        "avg_focus_last_7_days": float(
            personalization_context.get("avg_focus_last_7_days", focus_score)
        ),
        "avg_distraction_last_7_days": float(
            personalization_context.get("avg_distraction_last_7_days", distractions)
        ),
        "personal_baseline_score": float(
            personalization_context.get("personal_baseline_score", focus_score)
        ),
    }


def serialize_usage_summary(
    state: SessionFeatureState,
    latest_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latest_snapshot = latest_snapshot or {}
    top_apps = sorted(
        (
            {
                "app_name": app_name,
                "seconds": round(seconds, 1),
                "minutes": round(seconds / 60, 1),
            }
            for app_name, seconds in state.app_usage_seconds.items()
        ),
        key=lambda item: item["seconds"],
        reverse=True,
    )

    categories = {
        category: {
            "seconds": round(seconds, 1),
            "minutes": round(seconds / 60, 1),
            "hours": round(seconds / 3600, 2),
        }
        for category, seconds in state.category_usage_seconds.items()
    }

    return {
        "session_seconds": round(state.session_seconds, 1),
        "session_hours": round(state.session_seconds / 3600, 2),
        "active_seconds": round(state.active_seconds, 1),
        "idle_seconds": round(state.idle_seconds, 1),
        "coding_seconds": round(state.coding_seconds, 1),
        "learning_seconds": round(state.learning_seconds, 1),
        "distraction_seconds": round(state.distraction_seconds, 1),
        "communication_seconds": round(state.communication_seconds, 1),
        "other_seconds": round(state.other_seconds, 1),
        "deep_work_seconds": round(state.deep_work_seconds, 1),
        "context_switches": state.context_switches,
        "keystrokes": state.keystrokes,
        "mouse_clicks": state.mouse_clicks,
        "mouse_scrolls": state.mouse_scrolls,
        "intents": {
            intent: {
                "seconds": round(seconds, 1),
                "minutes": round(seconds / 60, 1),
                "hours": round(seconds / 3600, 2),
            }
            for intent, seconds in state.intent_usage_seconds.items()
        },
        "top_apps": top_apps[:8],
        "categories": categories,
        "active_app": latest_snapshot.get("app_name"),
        "active_window": latest_snapshot.get("window_title"),
    }
