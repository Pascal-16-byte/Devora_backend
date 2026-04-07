"""
Realtime system tracker for Devora.

Run with:
    python tracker.py

The tracker samples active-window/process activity every few seconds, stores it
locally, converts it into the existing Devora feature schema, and sends the
derived features to the FastAPI backend only when the activity meaningfully
changes, with a 25-second fallback heartbeat.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import sqlite3
import subprocess
from shutil import which
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil
import requests

try:
    import win32gui
    import win32process
except ImportError:  # pragma: no cover - Windows-only dependency
    win32gui = None
    win32process = None

try:
    from pynput import keyboard, mouse
except ImportError as exc:  # pragma: no cover - system dependency
    raise RuntimeError("tracker.py requires pynput for idle detection.") from exc

try:
    from .context_engine import analyze_context
    from .feature_mapper import (
        SessionFeatureState,
        build_prediction_features,
        categorize_app,
        serialize_usage_summary,
        update_session_state,
    )
except ImportError:
    from context_engine import analyze_context
    from feature_mapper import (
        SessionFeatureState,
        build_prediction_features,
        categorize_app,
        serialize_usage_summary,
        update_session_state,
    )

BASE_DIR = Path(__file__).resolve().parent
LOCAL_DB_PATH = BASE_DIR / "tracker_local.db"
LOCAL_JSON_PATH = BASE_DIR / "tracker_predictions.jsonl"
DEFAULT_API_BASE = "https://devora-backend-bo7f.onrender.com"
LOGGER = logging.getLogger("tracker")
UNKNOWN_WINDOW_INFO: tuple[str, str, int | None] = ("Unknown", "Unknown Window", None)
_LAST_WINDOW_INFO: tuple[str, str, int | None] = UNKNOWN_WINDOW_INFO


class InputActivityMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.last_input_at = time.time()
        self.keystrokes = 0
        self.mouse_clicks = 0
        self.mouse_scrolls = 0
        self._keyboard_listener: keyboard.Listener | None = None
        self._mouse_listener: mouse.Listener | None = None

    def _touch(self, kind: str) -> None:
        with self._lock:
            self.last_input_at = time.time()
            if kind == "key":
                self.keystrokes += 1
            elif kind == "click":
                self.mouse_clicks += 1
            elif kind == "scroll":
                self.mouse_scrolls += 1

    def start(self) -> None:
        self._keyboard_listener = keyboard.Listener(on_press=lambda key: self._touch("key"))
        self._mouse_listener = mouse.Listener(
            on_click=lambda x, y, button, pressed: self._touch("click") if pressed else None,
            on_scroll=lambda x, y, dx, dy: self._touch("scroll"),
            on_move=lambda x, y: None,
        )
        self._keyboard_listener.daemon = True
        self._mouse_listener.daemon = True
        self._keyboard_listener.start()
        self._mouse_listener.start()

    def stop(self) -> None:
        if self._keyboard_listener:
            self._keyboard_listener.stop()
        if self._mouse_listener:
            self._mouse_listener.stop()

    def get_idle_seconds(self) -> float:
        with self._lock:
            return max(0.0, time.time() - self.last_input_at)

    def drain_activity_counts(self) -> dict[str, int]:
        with self._lock:
            counts = {
                "keystrokes": self.keystrokes,
                "mouse_clicks": self.mouse_clicks,
                "mouse_scrolls": self.mouse_scrolls,
            }
            self.keystrokes = 0
            self.mouse_clicks = 0
            self.mouse_scrolls = 0
            return counts


class LocalTrackerStore:
    def __init__(self, db_path: Path = LOCAL_DB_PATH, json_path: Path = LOCAL_JSON_PATH) -> None:
        self.db_path = db_path
        self.json_path = json_path
        self._init_db()

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_activity (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    app_name TEXT,
                    category TEXT,
                    context_json TEXT,
                    window_title TEXT,
                    cpu_percent REAL,
                    memory_mb REAL,
                    idle_seconds REAL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    productivity_level TEXT,
                    productivity_score REAL,
                    explanation TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(raw_activity)").fetchall()
            }
            if "context_json" not in columns:
                conn.execute("ALTER TABLE raw_activity ADD COLUMN context_json TEXT")
            conn.commit()

    def save_activity(self, snapshot: dict[str, Any]) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO raw_activity (
                    id, timestamp, app_name, category, context_json, window_title,
                    cpu_percent, memory_mb, idle_seconds, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    snapshot["timestamp"],
                    snapshot.get("app_name"),
                    snapshot.get("category"),
                    json.dumps(snapshot.get("context") or {}, ensure_ascii=True),
                    snapshot.get("window_title"),
                    snapshot.get("cpu_percent"),
                    snapshot.get("memory_mb"),
                    snapshot.get("idle_seconds"),
                    json.dumps(snapshot, ensure_ascii=True),
                ),
            )
            conn.commit()

    def save_prediction(self, prediction: dict[str, Any]) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    id, timestamp, productivity_level, productivity_score, explanation, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction.get("prediction_id") or str(uuid4()),
                    datetime.now(timezone.utc).isoformat(),
                    prediction.get("productivity_level"),
                    prediction.get("productivity_score"),
                    prediction.get("explanation"),
                    json.dumps(prediction, ensure_ascii=True),
                ),
            )
            conn.commit()

        with self.json_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(prediction, ensure_ascii=True) + "\n")


@dataclass
class TrackerConfig:
    api_base_url: str = DEFAULT_API_BASE
    sample_interval_seconds: float = 2.0
    send_interval_seconds: float = 25.0
    idle_threshold_seconds: float = 120.0
    focus_delta_threshold: float = 5.0
    privacy_enabled: bool = True
    tracker_id: str = field(default_factory=lambda: uuid4().hex[:12])


def _trim_window_title(title: str, enabled: bool) -> str:
    if enabled:
        return title
    return "[hidden]"


def _safe_process_name(pid: int | None) -> str:
    if not pid:
        return "Unknown"

    try:
        return psutil.Process(pid).name() or "Unknown"
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return "Unknown"


def _remember_window_info(info: tuple[str, str, int | None]) -> tuple[str, str, int | None]:
    global _LAST_WINDOW_INFO

    app_name, window_title, pid = info
    if app_name != "Unknown" or window_title != "Unknown Window" or pid is not None:
        _LAST_WINDOW_INFO = info
    return info


def _fallback_window_info(reason: str) -> tuple[str, str, int | None]:
    if _LAST_WINDOW_INFO != UNKNOWN_WINDOW_INFO:
        LOGGER.warning("Active window detection failed (%s); using cached window info.", reason)
        return _LAST_WINDOW_INFO

    LOGGER.warning("Active window detection failed (%s); using unknown window fallback.", reason)
    return UNKNOWN_WINDOW_INFO


def _run_command(command: list[str], timeout: float = 1.0) -> str | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.warning("Command failed for active window detection (%s): %s", command[0], exc)
        return None

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        LOGGER.warning("Command returned non-zero for active window detection (%s): %s", command[0], stderr or completed.returncode)
        return None

    return completed.stdout.strip()


def _parse_xprop_value(raw_output: str | None) -> str | None:
    if not raw_output:
        return None

    match = re.search(r"=\s*(.+)$", raw_output, re.MULTILINE)
    if not match:
        return None

    value = match.group(1).strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value or None


def _get_active_window_windows() -> tuple[str, str, int | None]:
    if win32gui is None or win32process is None:
        return _fallback_window_info("pywin32 is unavailable")

    hwnd = win32gui.GetForegroundWindow()
    if not hwnd:
        return _fallback_window_info("no foreground window handle")

    window_title = win32gui.GetWindowText(hwnd) or "Unknown Window"
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    return _remember_window_info((_safe_process_name(pid), window_title, pid or None))


def _get_active_window_mac() -> tuple[str, str, int | None]:
    try:
        from AppKit import NSWorkspace
        from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly
    except ImportError:
        return _fallback_window_info("Quartz/AppKit is unavailable")

    try:
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return _fallback_window_info("no frontmost macOS application")

        pid = int(app.processIdentifier()) if app.processIdentifier() else None
        app_name = app.localizedName() or _safe_process_name(pid)
        window_title = "Unknown Window"

        windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, 0) or []
        for window in windows:
            owner_pid = window.get("kCGWindowOwnerPID")
            if pid is not None and owner_pid != pid:
                continue
            if window.get("kCGWindowLayer", 0) != 0:
                continue

            name = window.get("kCGWindowName")
            if name:
                window_title = name
                break

        return _remember_window_info((app_name or "Unknown", window_title, pid))
    except Exception as exc:  # pragma: no cover - platform-specific runtime behavior
        return _fallback_window_info(str(exc))


def _get_active_window_linux() -> tuple[str, str, int | None]:
    if which("xprop") is None:
        return _fallback_window_info("xprop is unavailable")

    root_output = _run_command(["xprop", "-root", "_NET_ACTIVE_WINDOW"])
    if not root_output:
        return _fallback_window_info("unable to read _NET_ACTIVE_WINDOW")

    match = re.search(r"window id # (0x[0-9a-fA-F]+)", root_output)
    if not match:
        return _fallback_window_info("unable to parse active window id")

    window_id = match.group(1)
    if window_id.lower() == "0x0":
        return _fallback_window_info("desktop is active window")

    title_output = _run_command(["xprop", "-id", window_id, "WM_NAME"])
    pid_output = _run_command(["xprop", "-id", window_id, "_NET_WM_PID"])

    window_title = _parse_xprop_value(title_output) or "Unknown Window"

    pid: int | None = None
    pid_match = re.search(r"=\s*(\d+)", pid_output or "")
    if pid_match:
        pid = int(pid_match.group(1))

    app_name = _safe_process_name(pid)
    info = (app_name, window_title, pid)

    if info == UNKNOWN_WINDOW_INFO:
        return _fallback_window_info("linux window details unavailable")

    return _remember_window_info(info)


def get_active_window_info() -> tuple[str, str, int | None]:
    system = platform.system()

    try:
        if system == "Windows":
            return _get_active_window_windows()
        if system == "Darwin":
            return _get_active_window_mac()
        if system == "Linux":
            return _get_active_window_linux()
        return _fallback_window_info(f"unsupported platform: {system}")
    except Exception as exc:  # pragma: no cover - defensive guard for tracker loop stability
        return _fallback_window_info(str(exc))


def _collect_process_metrics(limit: int = 6) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            memory_info = proc.info.get("memory_info")
            processes.append(
                {
                    "pid": proc.info["pid"],
                    "name": proc.info["name"] or "Unknown",
                    "cpu_percent": proc.cpu_percent(interval=None),
                    "memory_mb": round((memory_info.rss / (1024 * 1024)) if memory_info else 0.0, 1),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    processes.sort(key=lambda item: (item["cpu_percent"], item["memory_mb"]), reverse=True)
    return processes[:limit]


def _prime_cpu_counters() -> None:
    for proc in psutil.process_iter(["pid"]):
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def build_snapshot(
    config: TrackerConfig,
    activity_monitor: InputActivityMonitor,
    state: SessionFeatureState,
    elapsed_seconds: float,
) -> tuple[dict[str, Any], float]:
    app_name, window_title, pid = get_active_window_info()
    context = analyze_context(app_name, window_title)
    category = categorize_app(app_name, window_title)
    idle_seconds = activity_monitor.get_idle_seconds()
    is_idle = idle_seconds >= config.idle_threshold_seconds
    input_activity = activity_monitor.drain_activity_counts()

    cpu_percent = 0.0
    memory_mb = 0.0
    if pid:
        try:
            process = psutil.Process(pid)
            cpu_percent = process.cpu_percent(interval=None)
            memory_mb = round(process.memory_info().rss / (1024 * 1024), 1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    snapshot = {
        "tracker_id": config.tracker_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "app_name": app_name,
        "app_display_name": app_name.replace(".exe", ""),
        "category": category,
        "active_window": app_name,
        "window_title": _trim_window_title(window_title, config.privacy_enabled),
        "context": context,
        "cpu_percent": round(cpu_percent, 2),
        "memory_mb": memory_mb,
        "idle_seconds": round(idle_seconds, 1),
        "is_idle": is_idle,
        "input_activity": input_activity,
        "process_metrics": _collect_process_metrics(),
    }
    update_session_state(state, snapshot, elapsed_seconds)
    usage_summary = serialize_usage_summary(state, snapshot)
    snapshot.update(
        {
            "session_seconds": usage_summary["session_seconds"],
            "coding_seconds": usage_summary["coding_seconds"],
            "distraction_seconds": usage_summary["distraction_seconds"],
            "communication_seconds": usage_summary["communication_seconds"],
            "break_seconds": round(state.idle_seconds, 1),
            "app_usage": usage_summary,
        }
    )
    return snapshot, cpu_percent


def send_json(session: requests.Session, method: str, url: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = session.request(method=method, url=url, json=payload, timeout=10)
        response.raise_for_status()
        if response.content:
            return response.json()
        return None
    except requests.RequestException as exc:
        print(f"[tracker] request failed for {url}: {exc}")
        return None


def _build_activity_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "tracker_id": snapshot.get("tracker_id"),
        "timestamp": snapshot.get("timestamp"),
        "app_name": snapshot.get("app_name"),
        "app_display_name": snapshot.get("app_display_name"),
        "category": snapshot.get("category"),
        "active_window": snapshot.get("active_window"),
        "window_title": snapshot.get("window_title"),
        "context": snapshot.get("context") or {},
        "cpu_percent": snapshot.get("cpu_percent"),
        "memory_mb": snapshot.get("memory_mb"),
        "idle_seconds": snapshot.get("idle_seconds"),
        "is_idle": snapshot.get("is_idle"),
        "session_seconds": snapshot.get("session_seconds"),
        "coding_seconds": snapshot.get("coding_seconds"),
        "distraction_seconds": snapshot.get("distraction_seconds"),
        "communication_seconds": snapshot.get("communication_seconds"),
        "break_seconds": snapshot.get("break_seconds"),
        "app_usage": snapshot.get("app_usage") or {},
        "process_metrics": (snapshot.get("process_metrics") or [])[:3],
    }


def should_send_update(
    previous_snapshot: dict[str, Any] | None,
    current_snapshot: dict[str, Any],
    previous_features: dict[str, Any] | None,
    current_features: dict[str, Any],
    *,
    seconds_since_last_send: float,
    fallback_interval_seconds: float,
    focus_delta_threshold: float,
) -> bool:
    if previous_snapshot is None or previous_features is None:
        return True

    if seconds_since_last_send >= fallback_interval_seconds:
        return True

    if previous_snapshot.get("app_name") != current_snapshot.get("app_name"):
        return True

    if previous_snapshot.get("category") != current_snapshot.get("category"):
        return True

    if bool(previous_snapshot.get("is_idle")) != bool(current_snapshot.get("is_idle")):
        return True

    if abs(
        float(previous_features.get("focus_score", 0.0))
        - float(current_features.get("focus_score", 0.0))
    ) >= focus_delta_threshold:
        return True

    if abs(
        float(previous_features.get("distractions", 0.0))
        - float(current_features.get("distractions", 0.0))
    ) >= 2.0:
        return True

    if abs(
        float(previous_features.get("deep_work_ratio", 0.0))
        - float(current_features.get("deep_work_ratio", 0.0))
    ) >= 0.08:
        return True

    if abs(
        float(previous_features.get("task_completion_rate", 0.0))
        - float(current_features.get("task_completion_rate", 0.0))
    ) >= 0.08:
        return True

    return False


def run_tracker(config: TrackerConfig) -> None:
    print(f"[tracker] starting Devora tracker with id={config.tracker_id}")
    print(f"[tracker] backend={config.api_base_url}, sample_interval={config.sample_interval_seconds}s, send_interval={config.send_interval_seconds}s")

    store = LocalTrackerStore()
    activity_monitor = InputActivityMonitor()
    activity_monitor.start()
    state = SessionFeatureState()
    session = requests.Session()
    _prime_cpu_counters()

    last_sample_at = time.time()
    last_send_at = 0.0
    recent_cpu: list[float] = []
    latest_snapshot: dict[str, Any] | None = None
    last_sent_snapshot: dict[str, Any] | None = None
    last_sent_features: dict[str, Any] | None = None

    try:
        while True:
            now = time.time()
            elapsed = max(now - last_sample_at, config.sample_interval_seconds)
            last_sample_at = now

            latest_snapshot, cpu_percent = build_snapshot(config, activity_monitor, state, elapsed)
            recent_cpu.append(cpu_percent)
            recent_cpu = recent_cpu[-12:]
            store.save_activity(latest_snapshot)
            feature_payload = build_prediction_features(
                state=state,
                latest_snapshot=latest_snapshot,
                avg_cpu_percent=sum(recent_cpu) / max(len(recent_cpu), 1),
            )

            if latest_snapshot and should_send_update(
                last_sent_snapshot,
                latest_snapshot,
                last_sent_features,
                feature_payload,
                seconds_since_last_send=now - last_send_at,
                fallback_interval_seconds=config.send_interval_seconds,
                focus_delta_threshold=config.focus_delta_threshold,
            ):
                activity_payload = _build_activity_payload(latest_snapshot)
                send_json(session, "POST", f"{config.api_base_url}/activity/logs", activity_payload)
                predict_payload = {
                    **feature_payload,
                    "tracker_id": config.tracker_id,
                    "source": "tracker",
                }
                prediction = send_json(session, "POST", f"{config.api_base_url}/realtime/predict", predict_payload)
                if prediction:
                    enriched_prediction = {
                        **prediction,
                        "tracker_id": config.tracker_id,
                        "features": feature_payload,
                        "activity_snapshot": latest_snapshot,
                    }
                    store.save_prediction(enriched_prediction)
                    print(
                        "[tracker] "
                        f"{prediction['productivity_level']} score={prediction['productivity_score']} "
                        f"active_app={latest_snapshot['app_name']}"
                    )
                last_send_at = now
                last_sent_snapshot = latest_snapshot
                last_sent_features = dict(feature_payload)

            time.sleep(config.sample_interval_seconds)
    except KeyboardInterrupt:
        print("[tracker] stopping tracker")
    finally:
        activity_monitor.stop()


def parse_args() -> TrackerConfig:
    parser = argparse.ArgumentParser(description="Devora realtime tracker")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--send-interval", type=float, default=25.0)
    parser.add_argument("--idle-threshold", type=float, default=120.0)
    parser.add_argument("--tracker_id", default=None)
    parser.add_argument("--disable-privacy", action="store_true", help="Send full window titles")
    args = parser.parse_args()

    return TrackerConfig(
        api_base_url=args.api_base_url,
        sample_interval_seconds=args.sample_interval,
        send_interval_seconds=args.send_interval,
        idle_threshold_seconds=args.idle_threshold,
        privacy_enabled=not args.disable_privacy,
        tracker_id=args.tracker_id or uuid4().hex[:12],
    )


if __name__ == "__main__":
    run_tracker(parse_args())
