"""
FastAPI gateway for RouteOS.

Exposes:
  POST /generate    — run adaptive inference
  GET  /metrics     — Prometheus metrics
  GET  /benchmark   — run benchmark suite
  GET  /health      — liveness probe
  GET  /routing/{request_id} — inspect routing decisions for a request

Run with:
    uvicorn routeos.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from routeos.engine import RouteOSEngine, EngineConfig, InferenceResult

app = FastAPI(
    title="RouteOS",
    description="Adaptive Small Language Model Inference Engine",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton engine (in production: use dependency injection with a pool)
_engine = RouteOSEngine(config=EngineConfig())
_request_log: dict[str, InferenceResult] = {}


# --- Request / Response models ---

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.01, le=4.0)


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float
    cost_vs_baseline: float
    kv_retention_rate: float
    top_experts: list[str]


class BenchmarkRequest(BaseModel):
    prompts: list[str] = Field(default_factory=lambda: [
        "Explain transformer attention mechanisms",
        "Write a Python quicksort function",
        "What is the capital of France?",
    ])
    max_new_tokens: int = Field(default=128, ge=1, le=512)


# --- Endpoints ---

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "RouteOS v0.1.0"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    request_id = str(uuid.uuid4())[:8]
    _engine.reset()
    result = _engine.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    _request_log[request_id] = result

    top_experts = sorted(
        result.expert_utilisation.items(), key=lambda x: x[1], reverse=True
    )[:3]

    return GenerateResponse(
        request_id=request_id,
        text=result.text,
        tokens_generated=result.tokens_generated,
        latency_ms=round(result.latency_ms, 2),
        tokens_per_second=round(result.tokens_per_second, 1),
        cost_vs_baseline=round(result.cost_vs_baseline, 3),
        kv_retention_rate=round(result.kv_cache_stats["retention_rate"], 3),
        top_experts=[e[0] for e in top_experts],
    )


@app.get("/routing/{request_id}")
def get_routing(request_id: str) -> dict[str, Any]:
    if request_id not in _request_log:
        raise HTTPException(status_code=404, detail="Request not found")
    result = _request_log[request_id]
    decisions = result.routing_decisions[:10]  # first 10 tokens
    return {
        "request_id": request_id,
        "sample_routing": [
            {
                "token": i,
                "attention_mode": d.attention_mode.value,
                "experts": d.expert_indices,
                "keep_kv": d.keep_kv,
                "complexity": round(d.complexity_score, 3),
            }
            for i, d in enumerate(decisions)
        ],
        "expert_utilisation": result.expert_utilisation,
        "kv_cache": result.kv_cache_stats,
    }


@app.post("/benchmark")
def benchmark(req: BenchmarkRequest) -> dict[str, Any]:
    _engine.reset()
    return _engine.benchmark(req.prompts, max_new_tokens=req.max_new_tokens)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return _engine.metrics.summary()


@app.get("/metrics/prometheus", response_class=None)
def metrics_prometheus():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(_engine.metrics.prometheus_text(), media_type="text/plain")
