import unittest
from unittest.mock import patch

import numpy as np

from backend.app import ProgrammerMetrics, _build_prediction_response


FEATURES = [
    "coding_hours",
    "commits_per_day",
    "lines_of_code",
    "bugs_fixed",
    "sleep_hours",
    "distractions",
    "task_completion_rate",
    "meetings_per_day",
    "break_time",
    "focus_score",
]


class _FakeScaler:
    def transform(self, values):
        return values


class _FakeModel:
    classes_ = np.array([0, 1, 2])

    def predict(self, values):
        return np.array([1])

    def predict_proba(self, values):
        return np.array([[0.2, 0.6, 0.2]])


class _FakeLabelEncoder:
    classes_ = np.array(["Low", "Medium", "High"])

    def inverse_transform(self, values):
        mapping = {0: "Low", 1: "Medium", 2: "High"}
        return np.array([mapping[value] for value in values])

    def transform(self, values):
        mapping = {"Low": 0, "Medium": 1, "High": 2}
        return np.array([mapping[value] for value in values])


class AppProactiveIntegrationTests(unittest.TestCase):
    def test_build_prediction_response_includes_proactive_and_current_event(self) -> None:
        captured_recent_events = []

        def _capture_proactive(current_features, recent_events, pattern_insight):
            del current_features, pattern_insight
            captured_recent_events.extend(recent_events)
            return {
                "risk_level": "high",
                "alerts": ["Your productivity trend is declining"],
                "recommended_action": "Pause and reset your plan",
            }

        metrics = ProgrammerMetrics(
            coding_hours=4,
            commits_per_day=3,
            lines_of_code=180,
            bugs_fixed=2,
            sleep_hours=7,
            distractions=7,
            task_completion_rate=0.55,
            meetings_per_day=2,
            break_time=40,
            focus_score=46,
        )

        historical_events = [
            {"id": "old-1", "productivity_score": 84, "features": {"distractions": 3}},
            {"id": "old-2", "productivity_score": 71, "features": {"distractions": 5}},
        ]

        with (
            patch("backend.app.get_bundle", return_value={
                "model": _FakeModel(),
                "scaler": _FakeScaler(),
                "label_encoder": _FakeLabelEncoder(),
                "features": FEATURES,
            }),
            patch("backend.app.get_shap_values", return_value=[np.zeros((1, len(FEATURES))) for _ in range(3)]),
            patch("backend.app.generate_explanation", return_value={
                "explanation": "Explanation",
                "feature_importance": {"focus_score": 0.4},
                "feature_contributions": {"focus_score": -0.2},
            }),
            patch("backend.app.generate_local_plot", return_value="/static/example.png"),
            patch("backend.app.compute_productivity_score", return_value=63.0),
            patch("backend.app.save_prediction"),
            patch("backend.app.get_latest_activity_log", return_value={"app_usage": {"context_switches": 9, "active_seconds": 900}}),
            patch("backend.app.get_recent_prediction_events", return_value=historical_events),
            patch("backend.app.generate_comparative_insights", return_value={"summary": "comparison"}),
            patch("backend.app.generate_temporal_insights", return_value={"trend": "declining"}),
            patch("backend.app.generate_behavioral_patterns", return_value={"worst_hours": [15]}),
            patch("backend.app.generate_coaching_feedback", return_value={"headline": "coach"}),
            patch("backend.app.generate_proactive_alerts", side_effect=_capture_proactive),
            patch("backend.app.save_prediction_event"),
        ):
            response = _build_prediction_response(metrics, source="manual")

        self.assertEqual(response.proactive["risk_level"], "high")
        self.assertEqual(response.proactive["recommended_action"], "Pause and reset your plan")
        self.assertEqual(len(captured_recent_events), 3)
        self.assertEqual(captured_recent_events[-1]["productivity_score"], 63.0)
        self.assertIn("features", captured_recent_events[-1])


if __name__ == "__main__":
    unittest.main()
