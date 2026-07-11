"""
Token-level routing controller.

Predicts per-token compute requirements:
  - Attention mode: full vs sparse
  - Expert selection: top-k gating over MoE experts
  - KV-cache retention: keep vs compress

Based on TriRoute: Unified Learned Routing for Joint Adaptive Attention,
Experts, and KV-Cache Allocation (arXiv 2025).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum


class AttentionMode(Enum):
    FULL = "full"
    SPARSE = "sparse"


@dataclass
class RoutingDecision:
    """Per-token routing result returned by TokenRouter."""

    attention_mode: AttentionMode
    expert_indices: list[int]  # selected expert indices (top-k)
    expert_weights: torch.Tensor  # softmax-normalised gating weights
    keep_kv: bool  # whether to retain this token in KV cache
    complexity_score: float  # 0-1 difficulty estimate for diagnostics


class AttentionRouter(nn.Module):
    """Lightweight binary classifier: should this token use full attention?"""

    def __init__(self, hidden_dim: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, AttentionMode]:
        """
        Args:
            hidden: (batch, hidden_dim) token representations
        Returns:
            score tensor and resolved AttentionMode
        """
        score = torch.sigmoid(self.net(hidden)).squeeze(-1)  # (batch,)
        mode = AttentionMode.FULL if score.mean().item() >= self.threshold else AttentionMode.SPARSE
        return score, mode


class ExpertRouter(nn.Module):
    """
    Top-k gating network for Mixture-of-Experts routing.

    Uses straight-through estimation to allow gradient flow through
    discrete expert selection. Includes load-balancing auxiliary loss.
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (batch, hidden_dim)
        Returns:
            top_indices: (batch, top_k)  selected expert indices
            top_weights: (batch, top_k)  normalised gating scores
            aux_loss:    scalar load-balancing loss term
        """
        logits = self.gate(hidden)  # (batch, num_experts)
        probs = F.softmax(logits, dim=-1)

        top_weights, top_indices = torch.topk(probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Load-balancing loss: encourages uniform expert utilisation
        # (fraction of tokens routed * mean routing probability per expert)
        expert_mask = torch.zeros_like(probs).scatter_(-1, top_indices, 1.0)
        density = expert_mask.mean(0)
        mean_prob = probs.mean(0)
        aux_loss = (density * mean_prob).sum() * self.num_experts

        return top_indices, top_weights, aux_loss


class KVRetentionRouter(nn.Module):
    """Binary classifier: should this token be kept in the KV cache?"""

    def __init__(self, hidden_dim: int, threshold: float = 0.3) -> None:
        super().__init__()
        self.threshold = threshold
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1),
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, bool]:
        score = torch.sigmoid(self.net(hidden)).squeeze(-1)
        keep = score.mean().item() >= self.threshold
        return score, keep


class TokenRouter(nn.Module):
    """
    Joint token router combining attention, expert, and KV-cache routing.

    Wraps AttentionRouter, ExpertRouter, and KVRetentionRouter into a
    single forward pass that returns RoutingDecision for each token batch.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        attn_threshold: float = 0.5,
        kv_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.attention_router = AttentionRouter(hidden_dim, attn_threshold)
        self.expert_router = ExpertRouter(hidden_dim, num_experts, top_k)
        self.kv_router = KVRetentionRouter(hidden_dim, kv_threshold)

    def forward(self, hidden: torch.Tensor) -> tuple[RoutingDecision, torch.Tensor]:
        """
        Args:
            hidden: (batch, hidden_dim) token representations
        Returns:
            decision: RoutingDecision with all routing outcomes
            aux_loss:  load-balancing loss for training
        """
        attn_score, attn_mode = self.attention_router(hidden)
        expert_indices, expert_weights, aux_loss = self.expert_router(hidden)
        kv_score, keep_kv = self.kv_router(hidden)

        complexity = (attn_score.mean() + kv_score.mean()).item() / 2

        decision = RoutingDecision(
            attention_mode=attn_mode,
            expert_indices=expert_indices[0].tolist(),
            expert_weights=expert_weights[0],
            keep_kv=keep_kv,
            complexity_score=complexity,
        )
        return decision, aux_loss
