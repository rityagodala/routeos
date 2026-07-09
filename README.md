# RouteOS

**The Kubernetes of AI Inference** — a production inference engine that dynamically decides how much computation every token deserves.

Instead of running every request through the entire model, RouteOS implements TriRoute's joint routing mechanism to intelligently allocate attention, experts, and memory based on token difficulty.

## Research Basis

Inspired by:
> **TriRoute: Unified Learned Routing for Joint Adaptive Attention, Experts, and KV-Cache Allocation**

The paper demonstrates that routing attention sparsity, expert selection, and KV-cache retention jointly (rather than independently) achieves better compute/accuracy trade-offs than any single routing dimension alone.

## The Problem

Production LLM inference is expensive and uniform:

- Every token gets **full attention** regardless of complexity
- Every token activates **all model parameters** (dense models) or random experts (vanilla MoE)
- Every token gets **stored in KV cache** regardless of importance

RouteOS fixes all three simultaneously.

## How It Works

```
User Prompt
     │
FastAPI Gateway  ← POST /generate
     │
RouteOSEngine
     │
TokenRouter ──────────────────────────────────────────┐
     │                                                  │
AttentionRouter        ExpertRouter          KVRetentionRouter
"Barack Obama was     "def quicksort" →      "the" → EVICT
 born in" → FULL       experts: coding,      "Obama" → KEEP
                        reasoning
```

| Token type | Attention | Experts | KV Cache |
|---|---|---|---|
| Complex / rare | Full | 2 specialised | Retained |
| Common / simple | Sparse | 1 general | Evicted |

## Install

```bash
pip install routeos
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add routeos
```

## Quick Start

### Python API

```python
from routeos.engine import RouteOSEngine, EngineConfig

engine = RouteOSEngine(config=EngineConfig(
    hidden_dim=4096,
    num_experts=8,
    top_k_experts=2,
))

result = engine.generate("Explain transformer attention", max_new_tokens=256)

print(f"Cost vs baseline:  {result.cost_vs_baseline:.1%}")
print(f"KV retention rate: {result.kv_cache_stats['retention_rate']:.1%}")
print(f"Top experts used:  {list(result.expert_utilisation.keys())[:3]}")
```

### FastAPI Server

```bash
uvicorn routeos.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "max_new_tokens": 200}'
```

```bash
curl http://localhost:8000/metrics
```

### Benchmark: RouteOS vs Vanilla Llama

```python
from routeos.engine import RouteOSEngine, EngineConfig

engine = RouteOSEngine(config=EngineConfig())
summary = engine.benchmark(
    prompts=["Explain attention", "Write a sort function", "What is Paris?"],
    max_new_tokens=128,
)
print(summary)
# {'avg_cost_vs_baseline': 0.61, 'avg_tokens_per_second': 847, ...}
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system diagram.

```
User
 │
FastAPI Gateway (routeos.api)
 │
RouteOSEngine (routeos.engine)
 │
TokenRouter → MixtureOfExperts + KVCacheManager
 │
PyTorch Model Server (attach HuggingFace / vLLM model)
 │
Redis KV Cache (distributed)     Prometheus + Grafana
```

## Key Metrics

| Metric | Target |
|---|---|
| Cost vs vanilla baseline | ≤ 65% |
| KV cache compression | 30–60% |
| Accuracy degradation | < 1% on MMLU |
| Latency overhead (routing) | < 2ms per request |

## Development

```bash
git clone https://github.com/yourusername/routeos
cd routeos
uv sync --all-extras
uv run pytest tests/ -v
uv run ruff check src/ tests/
```

## Tech Stack

PyTorch 2.x · HuggingFace Transformers · vLLM · FastAPI · Redis · Prometheus · Kubernetes · Triton (optional GPU kernels)

## License

MIT
