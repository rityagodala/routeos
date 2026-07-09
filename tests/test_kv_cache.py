"""Tests for KV cache manager."""

import pytest
import torch
from routeos.kv_cache import KVCacheManager


@pytest.fixture
def cache() -> KVCacheManager:
    return KVCacheManager(max_size=10, importance_threshold=0.5, window_size=5)


def _dummy_kv(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(4, 16)  # (num_heads=4, head_dim=16)


def test_high_importance_token_retained(cache):
    retained = cache.add(0, _dummy_kv(0), _dummy_kv(1), importance=0.9)
    assert retained is True
    assert cache.stats.retained_tokens == 1


def test_low_importance_token_evicted(cache):
    retained = cache.add(0, _dummy_kv(0), _dummy_kv(1), importance=0.1)
    assert retained is False
    assert cache.stats.evicted_tokens == 1


def test_get_active_kv_returns_tensors(cache):
    cache.add(0, _dummy_kv(0), _dummy_kv(1), importance=0.9)
    keys, values = cache.get_active_kv()
    assert keys.shape[0] >= 1
    assert values.shape[0] >= 1


def test_empty_cache_raises(cache):
    with pytest.raises(ValueError, match="Cache is empty"):
        cache.get_active_kv()


def test_clear_resets_stats(cache):
    cache.add(0, _dummy_kv(0), _dummy_kv(1), importance=0.9)
    cache.clear()
    assert cache.stats.total_tokens == 0
    assert len(cache) == 0


def test_retention_rate_calculation(cache):
    cache.add(0, _dummy_kv(0), _dummy_kv(1), importance=0.9)  # kept
    cache.add(1, _dummy_kv(2), _dummy_kv(3), importance=0.1)  # evicted
    # retention_rate counts important tokens only
    assert cache.stats.retained_tokens == 1
    assert cache.stats.evicted_tokens == 1
