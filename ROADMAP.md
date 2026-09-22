# MiroFish-Offline Roadmap

## Current State (v0.5.0)

Fully local fork running on Neo4j 5.18 CE + Ollama (Qwen3:8b). All cloud dependencies removed. Full pipeline: upload text → knowledge graph → entity extraction → synthetic persona generation → multi-platform simulation → report generation. KV cache Q8 quantization enabled for memory-constrained hardware. Tiered model routing (fast/quality), semantic response pool, agent memory, benchmark + export tooling.

### Completed (Sessions 1-5)
- [x] Removed all Zep Cloud dependencies — replaced with local Neo4j graph storage
- [x] Ollama integration via OpenAI-compatible endpoint
- [x] System One action router — fast archetype-weighted action sampling, skips LLM for non-text actions (~60-70% LLM call reduction)
- [x] Agent memory persistence — SQLite-backed action journal, injected into LLM context
- [x] Response pool — TopoSim-inspired text reuse with variation (~30% reuse rate for CREATE_POST)
- [x] 12 synthetic persona templates with behavioral archetypes (Lurker, Amplifier, Contributor, Debater)
- [x] Custom agent archetypes with speech profiles and emotional states
- [x] Ontology generator cleanup (chunked sampling, JSON retry, improved logging)
- [x] Neo4j 5.18 bump (relationship vector search support)
- [x] Report agent with tool-calling (insight_forge, panorama_search, quick_search, interview_agents)
- [x] Docker Compose setup for Neo4j + Ollama
- [x] Frontend UI translated to English
- [x] KV cache Q8 quantization (`OLLAMA_KV_CACHE_TYPE=q8_0`) — halves KV cache memory
- [x] README overhaul — proper credits, removed bloat images, updated repo URL
- [x] .env.example updated with all performance tuning vars

---

## Completed (v0.5.0) — Inference Optimization & Export

### Optimization
- [x] Tiered model router — qwen3:0.6b for lightweight tasks (NER, action fallbacks), qwen3:8b for text generation only
- [x] Response Pool v2 — semantic similarity via nomic-embed-text for cross-archetype response sharing
- [x] Memory injection fix — replace summary each round instead of stacking duplicates in context

### Tooling
- [x] Benchmarking suite — automated script logging tok/s, System One routing %, timing, memory per run
- [x] Export simulation as structured JSON for external analysis and reproducibility

---

## Near Term

### v0.6.0 — Model Upgrades & Search
- [ ] Upgrade to Qwen 3.6+ for native MTP speculative decoding (~2x tok/s, zero code changes)
- [ ] Tune hybrid search weights (currently 0.7 vector / 0.3 BM25) — make configurable per graph
- [ ] Support vLLM-MLX as alternative backend for continuous batching on Apple Silicon
- [ ] Support multiple embedding models (e.g., mxbai-embed-large, bge-m3 for multilingual)

### v0.7.0 — Stability & Observability
- [ ] Fix `camel-oasis` / `camel-ai` compatibility with Python 3.12+ (currently requires <3.12)
- [ ] Add Docker Compose GPU auto-detection (fallback to CPU-only Ollama)
- [ ] Connection resilience: auto-reconnect to Neo4j on transient failures
- [ ] Add `/api/status` endpoint showing Neo4j connection state, Ollama model availability, and disk usage

---

## Mid Term

### v0.8.0 — Enhanced Simulation
- [ ] Real-time simulation dashboard with WebSocket updates
- [ ] Temporal graph: track how entity relationships evolve across simulation rounds
- [ ] Graph diff: compare two simulation runs side-by-side
- [ ] Multi-language simulation support

### v0.9.0 — Graph Intelligence
- [ ] Community detection (Louvain/Leiden) to auto-identify entity clusters
- [ ] Graph visualization improvements: force-directed layout, filtering by entity type
- [ ] Graph-aware reranking: boost results connected to the query entity

---

## Long Term

### v1.0.0 — Production Ready
- [ ] Authentication & multi-user support
- [ ] Graph versioning: snapshot and restore graph states
- [ ] Plugin system for custom NER extractors, search strategies, and report templates
- [ ] Comprehensive test suite (unit + integration + E2E)

---

## Hardware Tiers

| Tier | RAM | GPU VRAM | Recommended Model | Expected Performance |
|------|-----|----------|-------------------|---------------------|
| Minimal | 8 GB | — (CPU only) | qwen3:1.7b | Slow, basic quality |
| Light | 16 GB | 6-8 GB | qwen3:4b | Usable for small graphs |
| Standard | 16 GB | Apple M-series | qwen3:8b + Q8 KV cache | Good for most use cases |
| Power | 32+ GB | 16+ GB VRAM | qwen3:14b+ | Full quality, fast |

---

## Contributing

This project is AGPL-3.0 licensed. Contributions welcome — especially around:
- Python 3.12+ compatibility for CAMEL-AI / OASIS
- Additional embedding model support
- Simulation quality improvements
- Documentation and tutorials
