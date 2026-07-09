"""
Prometheus-compatible metrics for RouteOS.

Tracks latency, throughput, cost ratio, and expert utilisation.
Exposes a /metrics endpoint when used with the FastAPI server.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestRecord:
    timestamp: float
    tokens: int
    latency_ms: float
    cost_ratio: float


class InferenceMetrics:
    """
    In-process metrics store. In production, swap out for
    prometheus_client counters/histograms.
    """

    def __init__(self) -> None:
        self._records: list[RequestRecord] = []
        self._start_time = time.time()

    def record(self, tokens: int, latency_ms: float, cost_ratio: float) -> None:
        self._records.append(
            RequestRecord(
                timestamp=time.time(),
                tokens=tokens,
                latency_ms=latency_ms,
                cost_ratio=cost_ratio,
            )
        )

    def summary(self) -> dict[str, Any]:
        if not self._records:
            return {"total_requests": 0}

        n = len(self._records)
        avg_latency = sum(r.latency_ms for r in self._records) / n
        avg_tps = sum(r.tokens / (r.latency_ms / 1000) for r in self._records) / n
        avg_cost = sum(r.cost_ratio for r in self._records) / n
        uptime_s = time.time() - self._start_time

        return {
            "total_requests": n,
            "total_tokens": sum(r.tokens for r in self._records),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_tokens_per_sec": round(avg_tps, 1),
            "avg_cost_vs_baseline": round(avg_cost, 3),
            "uptime_seconds": round(uptime_s, 1),
            "estimated_cost_savings_pct": round((1 - avg_cost) * 100, 1),
        }

    def prometheus_text(self) -> str:
        """Minimal Prometheus text format exposition."""
        s = self.summary()
        lines = [
            f'routeos_total_requests {s.get("total_requests", 0)}',
            f'routeos_avg_latency_ms {s.get("avg_latency_ms", 0)}',
            f'routeos_avg_tokens_per_sec {s.get("avg_tokens_per_sec", 0)}',
            f'routeos_cost_vs_baseline {s.get("avg_cost_vs_baseline", 1.0)}',
        ]
        return "\n".join(lines)
