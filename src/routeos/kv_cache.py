"""
Adaptive KV-Cache Manager.

Instead of storing every token in the key-value cache, RouteOS uses
the KVRetentionRouter's signal to decide which tokens deserve full
cache entries and which can be evicted or compressed.

Strategy:
  - HIGH importance  → keep full K/V vectors
  - LOW importance   → evict (rolling window of last N positions retained)
  - MEDIUM (future)  → quantise to int8 and store compressed
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class CacheEntry:
    position: int
    key: torch.Tensor    # (num_heads, head_dim)
    value: torch.Tensor  # (num_heads, head_dim)
    importance: float    # router score 0-1
    retained: bool = True


@dataclass
class CacheStats:
    total_tokens: int = 0
    retained_tokens: int = 0
    evicted_tokens: int = 0
    compression_ratio: float = 1.0

    @property
    def retention_rate(self) -> float:
        if self.total_tokens == 0:
            return 1.0
        return self.retained_tokens / self.total_tokens

    @property
    def memory_saved_pct(self) -> float:
        return (1.0 - self.compression_ratio) * 100


class KVCacheManager:
    """
    Manages the KV cache with adaptive retention.

    Tokens routed as important are kept indefinitely (up to max_size).
    Tokens routed as unimportant are evicted via a rolling window,
    keeping only the last `window_size` positions for positional context.

    Example:
        cache = KVCacheManager(max_size=2048, importance_threshold=0.4)
        cache.add(position=42, key=k, value=v, importance=0.8)  # kept
        cache.add(position=43, key=k2, value=v2, importance=0.1)  # evicted
        keys, values = cache.get_active_kv()
    """

    def __init__(
        self,
        max_size: int = 2048,
        importance_threshold: float = 0.4,
        window_size: int = 128,
    ) -> None:
        self.max_size = max_size
        self.importance_threshold = importance_threshold
        self.window_size = window_size

        self._important: list[CacheEntry] = []
        self._window: deque[CacheEntry] = deque(maxlen=window_size)
        self._stats = CacheStats()

    def add(
        self,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: float,
    ) -> bool:
        """Add a token to the cache. Returns True if retained, False if evicted."""
        entry = CacheEntry(position=position, key=key, value=value, importance=importance)
        self._stats.total_tokens += 1

        if importance >= self.importance_threshold:
            if len(self._important) >= self.max_size:
                # Evict the least important cached entry to make room
                self._important.sort(key=lambda e: e.importance)
                self._important.pop(0)
            self._important.append(entry)
            self._stats.retained_tokens += 1
            return True
        else:
            # Low importance: only keep in rolling window for positional context
            self._window.append(entry)
            self._stats.evicted_tokens += 1
            entry.retained = False
            self._update_compression_ratio()
            return False

    def get_active_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve concatenated K, V tensors for all active cache entries.

        Returns: keys (n, num_heads, head_dim), values (n, num_heads, head_dim)
        """
        entries = sorted(
            list(self._important) + list(self._window),
            key=lambda e: e.position,
        )
        if not entries:
            raise ValueError("Cache is empty")
        keys = torch.stack([e.key for e in entries])
        values = torch.stack([e.value for e in entries])
        return keys, values

    def clear(self) -> None:
        self._important.clear()
        self._window.clear()
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def _update_compression_ratio(self) -> None:
        total = self._stats.total_tokens
        retained = self._stats.retained_tokens + len(self._window)
        self._stats.compression_ratio = retained / total if total > 0 else 1.0

    def __len__(self) -> int:
        return len(self._important) + len(self._window)

    def __repr__(self) -> str:
        s = self._stats
        return (
            f"KVCacheManager(retained={s.retained_tokens}, evicted={s.evicted_tokens}, "
            f"retention_rate={s.retention_rate:.1%}, window={len(self._window)})"
        )
