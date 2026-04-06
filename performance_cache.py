"""
Lightweight cache helpers with optional Redis support.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol

try:  # pragma: no cover - optional dependency
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None


def _json_default(value: Any) -> str:
    return str(value)


def stable_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def stable_hash(value: Any) -> str:
    return sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def build_cache_key(namespace: str, payload: Any) -> str:
    return f"{namespace}:{stable_hash(payload)}"


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None:
        ...

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        ...

    def invalidate_prefix(self, prefix: str) -> None:
        ...


@dataclass
class _MemoryEntry:
    expires_at: float
    value: Any


class MemoryTTLCache:
    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: dict[str, _MemoryEntry] = {}

    def _purge_expired_unlocked(self) -> None:
        now = time.time()
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if entry.expires_at <= now
        ]
        for key in expired_keys:
            self._entries.pop(key, None)

        overflow = len(self._entries) - self.max_entries
        if overflow > 0:
            oldest_keys = sorted(
                self._entries.items(),
                key=lambda item: item[1].expires_at,
            )[:overflow]
            for key, _ in oldest_keys:
                self._entries.pop(key, None)

    def get(self, key: str) -> Any | None:
        with self._lock:
            self._purge_expired_unlocked()
            entry = self._entries.get(key)
            if not entry:
                return None
            if entry.expires_at <= time.time():
                self._entries.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        with self._lock:
            self._entries[key] = _MemoryEntry(
                expires_at=time.time() + max(ttl_seconds, 1),
                value=value,
            )
            self._purge_expired_unlocked()

    def invalidate_prefix(self, prefix: str) -> None:
        with self._lock:
            matching_keys = [key for key in self._entries if key.startswith(prefix)]
            for key in matching_keys:
                self._entries.pop(key, None)


class RedisTTLCache:
    def __init__(self, redis_url: str) -> None:
        if redis is None:
            raise RuntimeError("redis package is not installed")

        self._client = redis.from_url(redis_url, decode_responses=True)
        self._client.ping()

    def get(self, key: str) -> Any | None:
        payload = self._client.get(key)
        if payload is None:
            return None
        return json.loads(payload)

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self._client.setex(key, max(ttl_seconds, 1), stable_json_dumps(value))

    def invalidate_prefix(self, prefix: str) -> None:
        cursor = 0
        pattern = f"{prefix}*"
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=200)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break


class CacheManager:
    def __init__(self, backend: CacheBackend) -> None:
        self.backend = backend

    def get(self, key: str) -> Any | None:
        return self.backend.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self.backend.set(key, value, ttl_seconds)

    def get_or_set(self, key: str, ttl_seconds: int, factory) -> Any:
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value

        value = factory()
        self.set(key, value, ttl_seconds)
        return value

    def invalidate_prefix(self, prefix: str) -> None:
        self.backend.invalidate_prefix(prefix)


def create_cache_manager() -> CacheManager:
    redis_url = os.getenv("DEVORA_REDIS_URL")
    if redis_url:
        try:  # pragma: no cover - depends on local redis availability
            return CacheManager(RedisTTLCache(redis_url))
        except Exception:
            pass

    max_entries = int(os.getenv("DEVORA_MEMORY_CACHE_SIZE", "1024"))
    return CacheManager(MemoryTTLCache(max_entries=max_entries))
