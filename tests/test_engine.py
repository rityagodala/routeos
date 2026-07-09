"""Tests for the main RouteOS inference engine."""

import pytest
from routeos.engine import RouteOSEngine, EngineConfig, InferenceResult


@pytest.fixture
def engine() -> RouteOSEngine:
    cfg = EngineConfig(hidden_dim=64, num_experts=4, top_k_experts=2, max_kv_size=128)
    return RouteOSEngine(config=cfg)


def test_generate_returns_result(engine):
    result = engine.generate("Hello world", max_new_tokens=10)
    assert isinstance(result, InferenceResult)


def test_generate_correct_token_count(engine):
    result = engine.generate("test", max_new_tokens=20)
    assert result.tokens_generated == 20


def test_latency_is_positive(engine):
    result = engine.generate("test", max_new_tokens=5)
    assert result.latency_ms > 0


def test_cost_vs_baseline_in_range(engine):
    result = engine.generate("test", max_new_tokens=10)
    assert 0.0 <= result.cost_vs_baseline <= 1.0


def test_expert_utilisation_keys(engine):
    result = engine.generate("test", max_new_tokens=10)
    assert len(result.expert_utilisation) > 0
    for v in result.expert_utilisation.values():
        assert 0.0 <= v <= 1.0


def test_benchmark_aggregates_multiple_prompts(engine):
    prompts = ["prompt A", "prompt B", "prompt C"]
    summary = engine.benchmark(prompts, max_new_tokens=5)
    assert summary["num_prompts"] == 3
    assert "avg_latency_ms" in summary
    assert "avg_cost_vs_baseline" in summary


def test_reset_clears_kv_cache(engine):
    engine.generate("first", max_new_tokens=5)
    engine.reset()
    assert engine.kv_cache.stats.total_tokens == 0
