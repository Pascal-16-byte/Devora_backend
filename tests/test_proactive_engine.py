import unittest

from backend.proactive_engine import DEFAULT_ACTION, generate_proactive_alerts


class ProactiveEngineTests(unittest.TestCase):
    def test_returns_default_when_no_rules_fire(self) -> None:
        result = generate_proactive_alerts(
            {"features": {"distractions": 2, "focus_score": 74}},
            [],
            {},
        )

        self.assertEqual(result["risk_level"], "low")
        self.assertEqual(result["alerts"], [])
        self.assertEqual(result["recommended_action"], DEFAULT_ACTION)

    def test_triggers_distraction_spike_from_value(self) -> None:
        result = generate_proactive_alerts(
            {"features": {"distractions": 7, "focus_score": 70}},
            [],
            {},
        )

        self.assertEqual(result["risk_level"], "high")
        self.assertIn("You are entering a distraction zone", result["alerts"])
        self.assertEqual(result["recommended_action"], "Close distraction apps and start 30 min focus block")

    def test_triggers_focus_drop(self) -> None:
        result = generate_proactive_alerts(
            {"features": {"distractions": 2, "focus_score": 45}},
            [],
            {},
        )

        self.assertEqual(result["risk_level"], "high")
        self.assertIn("Your focus is dropping", result["alerts"])

    def test_triggers_context_switch_overload_with_rate_based_signal(self) -> None:
        result = generate_proactive_alerts(
            {
                "features": {"distractions": 2, "focus_score": 72},
                "app_usage": {"context_switches": 7, "active_seconds": 600},
            },
            [],
            {},
        )

        self.assertEqual(result["risk_level"], "medium")
        self.assertEqual(result["alerts"], ["Too many task switches detected"])
        self.assertEqual(result["recommended_action"], "Stick to one task for next 45 mins")

    def test_triggers_time_risk_from_worst_hours(self) -> None:
        result = generate_proactive_alerts(
            {
                "features": {"distractions": 2, "focus_score": 72},
                "timestamp": "2026-04-05T15:00:00+00:00",
            },
            [],
            {"worst_hours": [14, 15]},
        )

        self.assertEqual(result["risk_level"], "medium")
        self.assertEqual(result["alerts"], ["This is usually your low productivity window"])

    def test_triggers_momentum_loss_from_last_three_scores(self) -> None:
        recent_events = [
            {"productivity_score": 82, "features": {"distractions": 3}},
            {"productivity_score": 71, "features": {"distractions": 3}},
            {"productivity_score": 63, "features": {"distractions": 3}},
        ]

        result = generate_proactive_alerts(
            {"features": {"distractions": 5, "focus_score": 68}},
            recent_events,
            {},
        )

        self.assertEqual(result["risk_level"], "high")
        self.assertIn("Your productivity trend is declining", result["alerts"])
        self.assertEqual(result["recommended_action"], "Pause and reset your plan")

    def test_prioritizes_high_severity_rules_and_limits_alerts(self) -> None:
        recent_events = [
            {"productivity_score": 86, "features": {"distractions": 4}},
            {"productivity_score": 78, "features": {"distractions": 5}},
            {"productivity_score": 65, "features": {"distractions": 7}},
        ]

        result = generate_proactive_alerts(
            {
                "features": {"distractions": 7, "focus_score": 42},
                "app_usage": {"context_switches": 12, "active_seconds": 900},
                "timestamp": "2026-04-05T15:00:00+00:00",
            },
            recent_events,
            {"worst_hours": [15]},
        )

        self.assertEqual(result["risk_level"], "high")
        self.assertLessEqual(len(result["alerts"]), 2)
        self.assertEqual(result["alerts"][0], "You are entering a distraction zone")
        self.assertEqual(result["recommended_action"], "Close distraction apps and start 30 min focus block")

    def test_handles_sparse_inputs_without_crashing(self) -> None:
        result = generate_proactive_alerts({}, [{"productivity_score": 55}], None)

        self.assertEqual(result["risk_level"], "low")
        self.assertEqual(result["alerts"], [])
        self.assertEqual(result["recommended_action"], DEFAULT_ACTION)


if __name__ == "__main__":
    unittest.main()
