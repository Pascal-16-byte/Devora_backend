import unittest
from pathlib import Path
from uuid import uuid4

from backend import database


class PredictionEventPersistenceTests(unittest.TestCase):
    def test_proactive_round_trip(self) -> None:
        original_db_path = database.DB_PATH

        temp_db_path = Path(__file__).resolve().parent / f"test_devinsight_{uuid4().hex}.db"
        events = []
        database.DB_PATH = temp_db_path
        try:
            database.init_db()
            database.save_prediction_event(
                {
                    "id": "evt-1",
                    "timestamp": "2026-04-05T15:00:00+00:00",
                    "source": "manual",
                    "productivity_level": "Medium",
                    "productivity_score": 64.0,
                    "probabilities": {"Low": 12.0, "Medium": 68.0, "High": 20.0},
                    "features": {"focus_score": 48, "distractions": 7},
                    "feature_importance": {"focus_score": 0.4},
                    "feature_contributions": {"focus_score": -0.2},
                    "proactive": {
                        "risk_level": "high",
                        "alerts": ["You are entering a distraction zone"],
                        "recommended_action": "Close distraction apps and start 30 min focus block",
                    },
                }
            )

            events = database.get_recent_prediction_events(limit=5)
        finally:
            database.DB_PATH = original_db_path
            temp_db_path.unlink(missing_ok=True)

        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0]["proactive"],
            {
                "risk_level": "high",
                "alerts": ["You are entering a distraction zone"],
                "recommended_action": "Close distraction apps and start 30 min focus block",
            },
        )


if __name__ == "__main__":
    unittest.main()
