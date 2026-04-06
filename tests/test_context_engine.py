import unittest

from backend.context_engine import analyze_context
from backend.feature_mapper import categorize_app


class ContextEngineTests(unittest.TestCase):
    def test_youtube_tutorial_is_learning_and_maps_to_coding(self) -> None:
        context = analyze_context("brave.exe", "How to build FastAPI tutorial - YouTube - Brave")

        self.assertEqual(context["intent"], "learning")
        self.assertGreaterEqual(context["confidence"], 0.6)
        self.assertEqual(
            categorize_app("brave.exe", "How to build FastAPI tutorial - YouTube - Brave"),
            "coding",
        )

    def test_youtube_shorts_is_distraction(self) -> None:
        context = analyze_context("chrome.exe", "Funny cat shorts - YouTube - Google Chrome")

        self.assertEqual(context["intent"], "distraction")
        self.assertEqual(
            categorize_app("chrome.exe", "Funny cat shorts - YouTube - Google Chrome"),
            "distraction",
        )

    def test_stackoverflow_question_is_coding(self) -> None:
        context = analyze_context("chrome.exe", "python traceback fix - Stack Overflow - Google Chrome")

        self.assertEqual(context["intent"], "coding")
        self.assertGreaterEqual(context["confidence"], 0.6)

    def test_chatgpt_workspace_is_coding_not_communication(self) -> None:
        context = analyze_context("brave.exe", "ChatGPT - ML Project - Brave")

        self.assertEqual(context["intent"], "coding")
        self.assertGreaterEqual(context["confidence"], 0.6)
        self.assertEqual(
            categorize_app("brave.exe", "ChatGPT - ML Project - Brave"),
            "coding",
        )

    def test_meeting_window_is_communication(self) -> None:
        context = analyze_context("Teams.exe", "Sprint Planning Meeting | Microsoft Teams")

        self.assertEqual(context["intent"], "communication")
        self.assertEqual(categorize_app("Teams.exe", "Sprint Planning Meeting | Microsoft Teams"), "communication")


if __name__ == "__main__":
    unittest.main()
