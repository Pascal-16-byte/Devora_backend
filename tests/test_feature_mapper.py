import unittest

from backend.feature_mapper import categorize_app, extract_domain_from_title


class FeatureMapperTests(unittest.TestCase):
    def test_extract_domain_from_browser_style_title(self) -> None:
        self.assertEqual(extract_domain_from_title("ChatGPT - OpenAI - Google Chrome"), "chatgpt")
        self.assertEqual(extract_domain_from_title("Inbox - user@gmail.com - Gmail - Google Chrome"), "gmail")

    def test_browser_tabs_use_domain_rules_before_title_hints(self) -> None:
        category = categorize_app("Google Chrome", "YouTube - Coding tutorial - Google Chrome")
        self.assertEqual(category, "distraction")

    def test_unknown_browser_tabs_fall_back_to_other(self) -> None:
        category = categorize_app("Google Chrome", "Some random tab - Google Chrome")
        self.assertEqual(category, "other")

    def test_browser_productivity_sites_are_not_treated_as_distraction(self) -> None:
        self.assertEqual(categorize_app("Google Chrome", "GitHub - repo name - Google Chrome"), "coding")
        self.assertEqual(categorize_app("Google Chrome", "ChatGPT - OpenAI - Google Chrome"), "coding")
        self.assertEqual(categorize_app("Google Chrome", "Inbox - user@gmail.com - Gmail - Google Chrome"), "communication")

    def test_non_browser_classification_still_uses_existing_rules(self) -> None:
        self.assertEqual(categorize_app("Code.exe", "main.py - Visual Studio Code"), "coding")
        self.assertEqual(categorize_app("Slack.exe", "Daily Standup | Slack"), "communication")


if __name__ == "__main__":
    unittest.main()
