import asyncio
import unittest
from unittest.mock import patch

from backend.app import SimulationRequest, simulate
from backend.simulation_engine import run_what_if_analysis


def _fake_predictor(scenario):
    score = round(
        42
        + float(scenario.get("coding_hours", 0)) * 3.0
        + float(scenario.get("focus_score", 0)) * 0.18
        + float(scenario.get("sleep_hours", 0)) * 1.5
        - float(scenario.get("distractions", 0)) * 2.2,
        1,
    )
    level = "High" if score >= 80 else "Medium" if score >= 60 else "Low"
    return {
        "productivity_score": score,
        "productivity_level": level,
    }


class SimulationEngineTests(unittest.TestCase):
    def test_run_what_if_analysis_returns_best_scenario_and_top_recommendations(self) -> None:
        base_input = {
            "coding_hours": 5.0,
            "focus_score": 60.0,
            "sleep_hours": 6.5,
            "distractions": 6.0,
            "task_completion_rate": 0.7,
            "meetings_per_day": 2.0,
        }

        result = run_what_if_analysis(base_input, _fake_predictor)

        self.assertEqual(result["scenarios"][0]["name"], "Current")
        self.assertEqual(result["best_scenario"], "Deep Work Mode")
        self.assertGreater(result["improvement"], 0)
        self.assertLessEqual(len(result["recommendations"]), 2)
        self.assertEqual(result["recommendations"][0]["name"], "Deep Work Mode")

    def test_simulate_route_returns_scenario_response(self) -> None:
        payload = SimulationRequest(
            coding_hours=6,
            commits_per_day=4,
            lines_of_code=220,
            bugs_fixed=2,
            sleep_hours=7,
            distractions=5,
            task_completion_rate=0.72,
            meetings_per_day=2,
            break_time=35,
            focus_score=68,
            user_id="user-123",
        )

        def _summary_stub(metrics, user_id=None):
            self.assertEqual(user_id, "user-123")
            score = round(50 + metrics.coding_hours * 2 - metrics.distractions * 1.5 + metrics.focus_score * 0.1, 1)
            level = "High" if score >= 80 else "Medium" if score >= 60 else "Low"
            return {
                "productivity_score": score,
                "productivity_level": level,
            }

        with patch("backend.app._predict_summary", side_effect=_summary_stub):
            response = asyncio.run(simulate(payload))

        self.assertEqual(response.scenarios[0].name, "Current")
        self.assertTrue(any(item.name == response.best_scenario for item in response.scenarios))
        self.assertEqual(response.current_score, response.scenarios[0].score)
        self.assertLessEqual(len(response.recommendations), 2)


if __name__ == "__main__":
    unittest.main()
