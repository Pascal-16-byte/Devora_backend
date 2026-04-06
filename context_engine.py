"""
Lightweight context analysis for tracker activity.
"""

from __future__ import annotations

import re
from typing import Any


BROWSER_APP_KEYWORDS = ("chrome", "edge", "firefox", "brave", "opera", "safari")
BROWSER_TITLE_SUFFIXES = (
    "google chrome",
    "chrome",
    "microsoft edge",
    "edge",
    "mozilla firefox",
    "firefox",
    "brave",
    "opera",
    "safari",
)
TITLE_PART_SPLIT_RE = re.compile(r"\s[-|:]\s")
URL_TOKEN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?((?:localhost|127\.0\.0\.1|[a-z0-9-]+(?:\.[a-z0-9-]+)+))"
)

KNOWN_DOMAIN_ALIASES: dict[str, set[str]] = {
    "youtube": {"youtube", "youtu.be"},
    "github": {"github", "github.com"},
    "gitlab": {"gitlab", "gitlab.com"},
    "bitbucket": {"bitbucket", "bitbucket.org"},
    "chatgpt": {"chatgpt", "chat.openai.com", "openai"},
    "claude": {"claude", "anthropic"},
    "stackoverflow": {"stackoverflow", "stack overflow", "stackoverflow.com"},
    "localhost": {"localhost", "127.0.0.1"},
    "slack": {"slack", "slack.com"},
    "teams": {"teams", "teams.microsoft.com"},
    "zoom": {"zoom", "zoom.us"},
    "meet": {"meet", "meet.google.com", "google meet"},
    "docs": {"docs", "documentation", "readthedocs"},
    "netflix": {"netflix", "netflix.com"},
}

INTENT_RULES: dict[str, dict[str, set[str]]] = {
    "learning": {
        "app": {"udemy", "coursera", "pluralsight"},
        "domain": {"youtube", "docs", "readthedocs", "developer.mozilla", "freecodecamp"},
        "strong": {
            "tutorial",
            "course",
            "how to",
            "guide",
            "documentation",
            "docs",
            "lecture",
            "walkthrough",
            "training",
            "learn",
        },
        "weak": {"article", "reference", "example", "explainer", "notes"},
    },
    "coding": {
        "app": {
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
        },
        "domain": {
            "github",
            "gitlab",
            "bitbucket",
            "chatgpt",
            "openai",
            "claude",
            "anthropic",
            "copilot",
            "stackoverflow",
            "stack overflow",
            "localhost",
            "127.0.0.1",
            "vercel",
            "render",
        },
        "strong": {
            "pull request",
            "merge request",
            "issue",
            "commit",
            "chatgpt",
            "claude",
            "copilot",
            "localhost",
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".java",
            ".go",
            ".rs",
        },
        "weak": {"repo", "branch", "stack overflow", "debug", "fix", "api", "backend"},
    },
    "distraction": {
        "app": {"netflix", "spotify", "steam"},
        "domain": {"youtube", "netflix", "spotify", "twitch", "instagram", "reddit"},
        "strong": {
            "reels",
            "shorts",
            "memes",
            "music",
            "netflix",
            "gaming",
            "trailer",
            "movie",
            "playlist",
            "stream",
            "funny",
            "highlights",
        },
        "weak": {"watch", "video", "live", "reddit", "twitch"},
    },
    "communication": {
        "app": {"slack", "teams", "zoom", "outlook", "mail", "discord", "telegram"},
        "domain": {"slack", "teams", "zoom", "meet", "gmail", "outlook"},
        "strong": {
            "meeting",
            "call",
            "zoom",
            "teams",
            "slack",
            "standup",
            "huddle",
            "inbox",
            "calendar",
        },
        "weak": {"message", "email", "mail", "thread"},
    },
}

INTENT_PRIORITY = ("communication", "coding", "learning", "distraction")


def normalize_text(text: str | None) -> str:
    return (text or "").strip().lower()


def is_browser_app(app_name: str | None) -> bool:
    normalized_app = normalize_text(app_name)
    return any(browser in normalized_app for browser in BROWSER_APP_KEYWORDS)


def _title_candidates(title: str | None) -> list[str]:
    normalized_title = normalize_text(title)
    if not normalized_title:
        return []

    candidates = [part.strip() for part in TITLE_PART_SPLIT_RE.split(normalized_title) if part.strip()]
    candidates = [part for part in candidates if part not in BROWSER_TITLE_SUFFIXES]
    return candidates or [normalized_title]


def extract_domain_from_title(title: str | None) -> str:
    candidates = _title_candidates(title)
    if not candidates:
        return ""

    for candidate in candidates:
        url_match = URL_TOKEN_RE.search(candidate)
        if url_match:
            raw_domain = url_match.group(1)
            if raw_domain in {"localhost", "127.0.0.1"}:
                return raw_domain

            labels = [label for label in raw_domain.split(".") if label]
            if labels:
                return labels[-2] if len(labels) >= 2 else labels[0]

    for candidate in candidates:
        for canonical, aliases in KNOWN_DOMAIN_ALIASES.items():
            if any(alias in candidate for alias in aliases):
                return canonical

    return candidates[0]


def _collect_matches(text: str, keywords: set[str]) -> list[str]:
    if not text:
        return []
    return sorted(keyword for keyword in keywords if keyword in text)


def _append_evidence(evidence: list[str], label: str, matches: list[str]) -> int:
    if not matches:
        return 0
    evidence.append(f"{label}: {', '.join(matches[:3])}")
    return len(matches)


def analyze_context(app_name: str | None, window_title: str | None) -> dict[str, Any]:
    normalized_app = normalize_text(app_name)
    normalized_title = normalize_text(window_title)
    domain = extract_domain_from_title(normalized_title)
    combined_text = " ".join(part for part in (normalized_app, normalized_title, domain) if part)

    scores: dict[str, int] = {}
    evidence_by_intent: dict[str, list[str]] = {}

    for intent, rules in INTENT_RULES.items():
        evidence: list[str] = []
        score = 0

        app_matches = _collect_matches(normalized_app, rules["app"])
        if app_matches:
            score += 4 + len(app_matches)
            _append_evidence(evidence, "app", app_matches)

        domain_matches = _collect_matches(domain, rules["domain"])
        if domain_matches:
            score += 4 + len(domain_matches)
            _append_evidence(evidence, "domain", domain_matches)

        strong_matches = _collect_matches(combined_text, rules["strong"])
        if strong_matches:
            score += 5 + (2 * len(strong_matches))
            _append_evidence(evidence, "keywords", strong_matches)

        weak_matches = _collect_matches(combined_text, rules["weak"])
        if weak_matches:
            score += 2 + len(weak_matches)
            _append_evidence(evidence, "signals", weak_matches)

        if domain == "youtube" and intent == "learning" and strong_matches:
            score += 3
            evidence.append("youtube learning context")
        if domain == "youtube" and intent == "distraction" and strong_matches:
            score += 3
            evidence.append("youtube entertainment context")

        scores[intent] = score
        evidence_by_intent[intent] = evidence

    top_score = max(scores.values(), default=0)
    if top_score <= 0:
        return {
            "intent": "neutral",
            "confidence": 0.3,
            "reason": "No strong contextual signals found.",
        }

    top_intents = [intent for intent, score in scores.items() if score == top_score]
    intent = next((candidate for candidate in INTENT_PRIORITY if candidate in top_intents), top_intents[0])
    second_best = max((score for label, score in scores.items() if label != intent), default=0)
    margin = top_score - second_best

    if top_score >= 9 or margin >= 4:
        confidence = 0.9
    elif top_score >= 4:
        confidence = 0.6
    else:
        confidence = 0.3

    evidence = evidence_by_intent.get(intent) or []
    reason = "; ".join(evidence[:3]) if evidence else "Fallback intent classification."
    return {
        "intent": intent,
        "confidence": confidence,
        "reason": reason,
    }
