"""
Mixture-of-Experts (MoE) layer.

Each expert is a small feed-forward network specialised for a domain
(e.g., coding, reasoning, factual recall). The ExpertRouter selects
top-k experts per token; this module executes those experts and
combines their outputs via the gating weights.

Implements load-balanced sparse MoE following Switch Transformer /
Mixtral conventions, extended with TriRoute joint routing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class Expert(nn.Module):
    """Single FFN expert with configurable hidden expansion."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """
    Sparse MoE layer: routes each token to top-k experts.

    Usage:
        moe = MixtureOfExperts(hidden_dim=512, num_experts=8, top_k=2)
        output, aux_loss = moe(hidden_states, expert_indices, expert_weights)
    """

    EXPERT_LABELS: dict[int, str] = {
        0: "coding",
        1: "reasoning",
        2: "factual_recall",
        3: "math",
        4: "language",
        5: "summarisation",
        6: "instruction_following",
        7: "general",
    }

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        expert_hidden = hidden_dim * expert_hidden_multiplier
        self.experts = nn.ModuleList(
            [Expert(hidden_dim, expert_hidden, hidden_dim) for _ in range(num_experts)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            hidden:         (batch, hidden_dim)
            expert_indices: (batch, top_k) from ExpertRouter
            expert_weights: (batch, top_k) normalised gating weights
        Returns:
            output:     (batch, hidden_dim) weighted combination of expert outputs
            stats:      dict with per-expert utilisation for monitoring
        """
        batch_size = hidden.shape[0]
        output = torch.zeros_like(hidden)
        expert_hits: dict[int, int] = {i: 0 for i in range(self.num_experts)}

        for b in range(batch_size):
            for k in range(self.top_k):
                idx = expert_indices[b, k].item()
                weight = expert_weights[b, k]
                expert_out = self.experts[idx](hidden[b].unsqueeze(0)).squeeze(0)
                output[b] += weight * self.dropout(expert_out)
                expert_hits[idx] += 1

        utilisation = {
            self.EXPERT_LABELS.get(i, str(i)): expert_hits[i] / (batch_size * self.top_k)
            for i in range(self.num_experts)
        }
        return output, utilisation

    def expert_label(self, idx: int) -> str:
        return self.EXPERT_LABELS.get(idx, f"expert_{idx}")
