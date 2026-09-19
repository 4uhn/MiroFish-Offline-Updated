# MiroFish-Offline Roadmap

## Current State (v0.4.0)

Fully local fork running on Neo4j 5.18 CE + Ollama (Qwen3:8b). All cloud dependencies removed. Full pipeline: upload text → knowledge graph → entity extraction → synthetic persona generation → multi-platform simulation → report generation.

### Completed (Sessions 1-4)
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

---

## Near Term

### v0.5.0 — Stability & Observability
- [ ] Fix `camel-oasis` / `camel-ai` compatibility with Python 3.12+ (currently requires <3.12)
- [ ] Add Docker Compose GPU auto-detection (fallback to CPU-only Ollama)
- [ ] Connection resilience: auto-reconnect to Neo4j on transient failures
- [ ] Add `/api/status` endpoint showing Neo4j connection state, Ollama model availability, and disk usage
- [ ] Real-time simulation dashboard with WebSocket updates

### v0.6.0 — Search & Multi-Model
- [ ] Tune hybrid search weights (currently 0.7 vector / 0.3 BM25) — make configurable per graph
- [ ] Model router: assign different Ollama models to different tasks (fast model for NER, large model for reports)
- [ ] Support vLLM and llama.cpp as alternative backends alongside Ollama
- [ ] Support multiple embedding models (e.g., mxbai-embed-large, bge-m3 for multilingual)

---

## Mid Term

### v0.7.0 — Enhanced Simulation
- [ ] Multi-language simulation support (agents can interact in different languages)
- [ ] Export simulation transcripts as structured JSON for external analysis
- [ ] Temporal graph: track how entity relationships evolve across simulation rounds
- [ ] Graph diff: compare two simulation runs side-by-side

### v0.8.0 — Graph Intelligence
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
- [ ] Performance benchmarks: document throughput (texts/min) and latency per hardware tier

---

## Hardware Tiers

| Tier | RAM | GPU VRAM | Recommended Model | Expected Performance |
|------|-----|----------|-------------------|---------------------|
| Minimal | 8 GB | — (CPU only) | qwen3:1.7b | Slow, basic quality |
| Light | 16 GB | 6-8 GB | qwen3:4b | Usable for small graphs |
| Standard | 32 GB | 12-16 GB | qwen3:8b | Good for most use cases |
| Power | 64 GB | 24+ GB | qwen3:14b+ | Full quality, fast |

---

## Contributing

This project is AGPL-3.0 licensed. Contributions welcome — especially around:
- Python 3.12+ compatibility for CAMEL-AI / OASIS
- Additional embedding model support
- Simulation quality improvements
- Documentation and tutorials
