"""Tests for token router module."""

import pytest
import torch
from routeos.router import TokenRouter, AttentionMode, RoutingDecision


@pytest.fixture
def router() -> TokenRouter:
    return TokenRouter(hidden_dim=64, num_experts=4, top_k=2)


def test_router_returns_routing_decision(router):
    hidden = torch.randn(1, 64)
    decision, aux_loss = router(hidden)
    assert isinstance(decision, RoutingDecision)


def test_expert_indices_within_bounds(router):
    hidden = torch.randn(1, 64)
    decision, _ = router(hidden)
    for idx in decision.expert_indices:
        assert 0 <= idx < 4


def test_expert_weights_sum_to_one(router):
    hidden = torch.randn(1, 64)
    decision, _ = router(hidden)
    assert abs(decision.expert_weights.sum().item() - 1.0) < 1e-5


def test_attention_mode_is_valid(router):
    hidden = torch.randn(1, 64)
    decision, _ = router(hidden)
    assert decision.attention_mode in (AttentionMode.FULL, AttentionMode.SPARSE)


def test_complexity_score_in_range(router):
    hidden = torch.randn(1, 64)
    decision, _ = router(hidden)
    assert 0.0 <= decision.complexity_score <= 1.0


def test_aux_loss_is_scalar(router):
    hidden = torch.randn(1, 64)
    _, aux_loss = router(hidden)
    assert aux_loss.dim() == 0


def test_router_batch_independence(router):
    """Router should handle different hidden states independently."""
    h1 = torch.zeros(1, 64)
    h2 = torch.ones(1, 64)
    d1, _ = router(h1)
    d2, _ = router(h2)
    # Decisions may differ; just verify both are valid
    assert isinstance(d1, RoutingDecision)
    assert isinstance(d2, RoutingDecision)
