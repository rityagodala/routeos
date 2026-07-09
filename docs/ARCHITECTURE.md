# RouteOS Architecture

## Overview

RouteOS implements TriRoute's joint routing mechanism: every token independently
decides how much compute it deserves across three dimensions.

```
Input Token Hidden State
         │
    ┌────▼────┐
    │ TokenRouter │
    └────┬────┘
         │
    ┌────┴──────────────────────┐
    ▼                           ▼                          ▼
AttentionRouter          ExpertRouter              KVRetentionRouter
    │                           │                          │
Full / Sparse           Top-k Expert                Keep / Evict
 Attention               Selection                   KV Entry
```

## Components

### TokenRouter (`router.py`)
Joint controller that runs all three sub-routers in a single forward pass.
Returns a `RoutingDecision` dataclass with all routing outcomes.

### MixtureOfExperts (`moe.py`)
8 specialised FFN experts (coding, reasoning, math, etc.).
Top-2 gating with load-balancing auxiliary loss.

### KVCacheManager (`kv_cache.py`)
Importance-based cache: high-importance tokens kept full,
low-importance tokens survive only in a rolling window.

### RouteOSEngine (`engine.py`)
Orchestrates the full pipeline. Attach a HuggingFace model via
`engine.model` to get real text generation.

### FastAPI Gateway (`api.py`)
Production API with `/generate`, `/metrics`, `/benchmark`, `/routing/{id}`.

## Deployment

```
Docker → RouteOS FastAPI (port 8000)
            │
     Redis KV store (optional, for distributed cache)
            │
     GPU Worker pool (vLLM backend)
            │
     Prometheus + Grafana (port 3000)
```

## Benchmarking

```bash
# Compare RouteOS vs vanilla baseline
uv run python scripts/benchmark.py \
  --model meta-llama/Llama-3.1-8B \
  --num-prompts 100 \
  --max-tokens 256
```
