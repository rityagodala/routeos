# RouteOS — Product Requirements Document

**Author:** Portfolio Project  
**Inspired by:** TriRoute: Unified Learned Routing for Joint Adaptive Attention, Experts, and KV-Cache Allocation

---

## 1. Problem Statement

Production LLM inference applies identical compute to every token regardless of complexity:

- **Full attention** computed for every token, even trivial filler words
- **All experts** (or all parameters in dense models) consulted uniformly
- **Entire KV cache** retained for every token, even positionally insignificant ones

This wastes GPU compute on easy tokens and inflates latency and cost for every request.

## 2. Vision

RouteOS is the **inference routing layer** that sits between the gateway and the model, making per-token compute decisions in real time. Target: achieve **Llama-level accuracy at 60–65% of the compute cost**.

## 3. Core Features

| Feature | Description |
|---|---|
| Attention routing | Binary full/sparse attention per token via learned classifier |
| MoE expert routing | Top-k gating with load-balancing loss over 8 specialised experts |
| KV cache routing | Importance-scored retention; low-importance tokens evicted to rolling window |
| Joint routing | All three decisions made in a single TokenRouter forward pass |
| FastAPI gateway | Streaming inference API with latency + cost telemetry |
| Metrics | Prometheus-compatible /metrics endpoint; Grafana dashboard |

## 4. Architecture

```
User → FastAPI (routeos.api)
         │
    RouteOSEngine (routeos.engine)
         │
    TokenRouter  ──── per-token in O(hidden_dim)
    │       │       │
  Attn   Expert    KV
  Router  Router  Router
         │
    MixtureOfExperts (routeos.moe)
    KVCacheManager   (routeos.kv_cache)
```

## 5. Implementation Plan

### Phase 1 — Core Routing (complete)
- `TokenRouter`: joint attention + expert + KV routing in one forward pass
- `MixtureOfExperts`: 8-expert layer with load-balancing auxiliary loss
- `KVCacheManager`: importance-threshold retention + rolling window

### Phase 2 — Engine & API (complete)
- `RouteOSEngine`: orchestrates routing across generation steps
- `FastAPI` gateway: `/generate`, `/metrics`, `/benchmark`, `/routing/{id}`
- `InferenceMetrics`: Prometheus-compatible telemetry

### Phase 3 — Model Integration
- Hook routing into HuggingFace model forward pass at each layer
- vLLM PagedAttention integration for KV cache backend
- CUDA Graph capture for routing network (sub-1ms overhead)

### Phase 4 — Production
- Docker + Kubernetes deployment manifests
- Redis distributed KV cache
- Grafana dashboard with routing visualisation
- Benchmarks vs Llama 3.1 8B baseline on MMLU, HumanEval, GSM8K

## 6. Benchmarking Plan

Compare `Llama 3.1 8B` baseline vs `Llama 3.1 8B + RouteOS`:

| Metric | Baseline | RouteOS Target |
|---|---|---|
| Tokens/sec | 100 | 150+ |
| GPU memory | 16 GB | 10–12 GB |
| Cost/1M tokens | $1.00 | $0.62 |
| MMLU accuracy | 68.4% | ≥ 67.8% |
