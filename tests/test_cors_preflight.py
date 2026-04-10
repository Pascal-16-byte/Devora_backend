import unittest

from fastapi.testclient import TestClient

from backend.app import app


class CorsPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_tracking_session_preflight_succeeds(self) -> None:
        response = self.client.options(
            "/tracking/session",
            headers={
                "Origin": "https://devora-frontend.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        self.assertIn(response.status_code, (200, 204))
        self.assertEqual(response.headers["access-control-allow-origin"], "*")
        self.assertIn("POST", response.headers["access-control-allow-methods"])

    def test_activity_logs_preflight_succeeds(self) -> None:
        response = self.client.options(
            "/activity/logs",
            headers={
                "Origin": "https://devora-frontend.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        self.assertIn(response.status_code, (200, 204))
        self.assertEqual(response.headers["access-control-allow-origin"], "*")

    def test_realtime_predict_preflight_succeeds(self) -> None:
        response = self.client.options(
            "/realtime/predict",
            headers={
                "Origin": "https://devora-frontend.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        self.assertIn(response.status_code, (200, 204))
        self.assertEqual(response.headers["access-control-allow-origin"], "*")


if __name__ == "__main__":
    unittest.main()
