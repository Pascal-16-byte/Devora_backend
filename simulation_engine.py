"""
Scenario-based what-if simulation helpers for productivity planning.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


Predictor = Callable[[dict[str, float]], dict[str, Any]]

_FEATURE_LIMITS: dict[str, tuple[float, float]] = {
    "coding_hours": (0.0, 24.0),
    "learning_time": (0.0, 24.0),
    "commits_per_day": (0.0, 50.0),
    "lines_of_code": (0.0, 5000.0),
    "bugs_fixed": (0.0, 50.0),
    "sleep_hours": (0.0, 12.0),
    "distractions": (0.0, 30.0),
    "deep_work_ratio": (0.0, 1.0),
    "distraction_intensity": (0.0, 30.0),
    "task_completion_rate": (0.0, 1.0),
    "meetings_per_day": (0.0, 20.0),
    "break_time": (0.0, 480.0),
    "focus_score": (0.0, 100.0),
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_feature(feature_name: str, value: float) -> float:
    if feature_name not in _FEATURE_LIMITS:
        return value

    min_value, max_value = _FEATURE_LIMITS[feature_name]
    return max(min_value, min(max_value, value))


def _normalized_input(base_input: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in base_input.items():
        normalized[key] = _clamp_feature(key, _safe_float(value))
    return normalized


def _with_updates(base_input: dict[str, float], **updates: float) -> dict[str, float]:
    scenario = dict(base_input)
    for key, value in updates.items():
        scenario[key] = _clamp_feature(key, float(value))
    return scenario


def _build_scenarios(base_input: dict[str, float]) -> list[dict[str, Any]]:
    current = dict(base_input)
    less_distraction = _with_updates(
        base_input,
        distractions=base_input.get("distractions", 0.0) * 0.5,
        distraction_intensity=max(
            0.0,
            base_input.get("distraction_intensity", base_input.get("distractions", 0.0)) * 0.55,
        ),
    )
    more_focus = _with_updates(
        base_input,
        focus_score=base_input.get("focus_score", 0.0) + 15.0,
        deep_work_ratio=max(base_input.get("deep_work_ratio", 0.0), 0.65),
    )
    better_sleep = _with_updates(
        base_input,
        sleep_hours=base_input.get("sleep_hours", 0.0) + 1.5,
        focus_score=base_input.get("focus_score", 0.0) + 5.0,
    )
    deep_work_mode = _with_updates(
        base_input,
        coding_hours=base_input.get("coding_hours", 0.0) + 2.0,
        distractions=max(0.0, base_input.get("distractions", 0.0) - 3.0),
        distraction_intensity=max(
            0.0,
            base_input.get("distraction_intensity", base_input.get("distractions", 0.0)) - 4.0,
        ),
        deep_work_ratio=max(base_input.get("deep_work_ratio", 0.0), 0.75),
    )

    return [
        {
            "name": "Current",
            "input": current,
            "explanation": "This is your current plan and acts as the baseline.",
        },
        {
            "name": "Less Distraction",
            "input": less_distraction,
            "explanation": "Reducing distractions improves focus stability and protects deep work.",
        },
        {
            "name": "More Focus",
            "input": more_focus,
            "explanation": "A higher focus score usually improves sustained attention and task momentum.",
        },
        {
            "name": "Better Sleep",
            "input": better_sleep,
            "explanation": "More recovery often improves cognitive stamina and lowers mental drag.",
        },
        {
            "name": "Deep Work Mode",
            "input": deep_work_mode,
            "explanation": "More maker time with fewer interruptions creates a stronger deep work window.",
        },
    ]


def run_what_if_analysis(
    base_input: dict[str, Any],
    predict_model: Predictor,
    *,
    top_recommendations: int = 2,
) -> dict[str, Any]:
    normalized_input = _normalized_input(base_input)
    scenario_definitions = _build_scenarios(normalized_input)

    results: list[dict[str, Any]] = []
    for scenario in scenario_definitions:
        prediction = predict_model(dict(scenario["input"]))
        results.append(
            {
                "name": scenario["name"],
                "score": round(_safe_float(prediction.get("productivity_score")), 1),
                "level": str(prediction.get("productivity_level", "Unknown")),
                "explanation": scenario["explanation"],
            }
        )

    current_result = next((item for item in results if item["name"] == "Current"), results[0])
    current_score = current_result["score"]
    current_level = current_result["level"]

    for result in results:
        result["improvement"] = round(result["score"] - current_score, 1)

    best_result = max(results, key=lambda item: (item["score"], item["improvement"]))
    recommendations = sorted(
        (
            item
            for item in results
            if item["name"] != "Current" and item["improvement"] > 0
        ),
        key=lambda item: item["improvement"],
        reverse=True,
    )[:max(1, top_recommendations)]

    return {
        "scenarios": results,
        "best_scenario": best_result["name"],
        "best_score": best_result["score"],
        "current_score": current_score,
        "current_level": current_level,
        "improvement": round(best_result["score"] - current_score, 1),
        "recommendations": [
            {
                "name": item["name"],
                "improvement": item["improvement"],
                "explanation": item["explanation"],
            }
            for item in recommendations
        ],
    }
