<div align="center">

# MiroFish-Offline (Updated)

**Fully local multi-agent simulation engine — no cloud APIs required.**

*Simulate public opinion, market sentiment, and social dynamics entirely on your hardware.*

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat-square)](./LICENSE)

</div>

## What is this?

MiroFish is a multi-agent simulation engine: upload any document (press release, policy draft, financial report), and it generates AI agents with unique personalities that simulate the public reaction on social media. Posts, arguments, opinion shifts — hour by hour.

This fork makes it **fully local and fully English**, optimized for Apple Silicon (M2 Pro 16GB tested):

| Feature | Original MiroFish | This Fork |
|---|---|---|
| Language | Chinese UI | **English UI** (1,000+ strings translated) |
| Graph DB | Zep Cloud | **Neo4j CE 5.18** |
| LLM | DashScope / OpenAI API | **Ollama** (qwen3:8b) |
| Embeddings | Zep Cloud | **nomic-embed-text** via Ollama |
| Cloud dependency | API keys required | **Zero** |
| Action routing | Every action = LLM call | **System One** (60-70% skip LLM) |
| Agent behavior | Uniform | **4 archetypes** (Lurker, Amplifier, Contributor, Debater) |
| Agent memory | Stateless | **Action journal** with SQLite persistence |
| Model routing | Single model | **Tiered** (fast 0.6b for NER, 8b for generation) |
| Response sharing | None | **Semantic pool** (embedding-matched cross-archetype reuse) |
| KV cache | Default f16 | **Q8 quantized** (halves memory, negligible quality loss) |

## How it works

1. **Graph Build** — Extracts entities (people, companies, events) and relationships from your document. Builds a knowledge graph via Neo4j with chunked sampling for large documents.
2. **Ontology Generation** — LLM designs entity types and relationship schemas from your source material (10 entity types, 6-10 relationship types).
3. **Env Setup** — Generates agent personas with behavioral archetypes, speech profiles, emotional states, and weighted action distributions.
4. **Simulation** — Agents interact on simulated Twitter/Reddit platforms. The **System One router** handles non-text actions (likes, follows, reposts) instantly without LLM calls. Only text-generating actions (posts, comments, quotes) use the LLM. The **response pool** reuses adapted posts between similar agents for further LLM savings.
5. **Report** — A ReportAgent analyzes the post-simulation environment, interviews agents, searches the knowledge graph, and generates a structured analysis.
6. **Interaction** — Chat with any agent from the simulated world. Full memory and personality persists.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Flask API                       │
│     graph.py   simulation.py   report.py        │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              Service Layer                       │
│  EntityReader   GraphTools   OntologyGenerator   │
│  ReportAgent    SimulationConfigGenerator        │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│          Simulation Engine (OASIS)               │
│  ┌─────────────────────────────────────┐        │
│  │  System One Router                   │        │
│  │  ┌───────────┐  ┌────────────────┐  │        │
│  │  │ Archetype  │  │ Response Pool  │  │        │
│  │  │ Weights    │  │ (text reuse)   │  │        │
│  │  └─────┬─────┘  └───────┬────────┘  │        │
│  │        │                │            │        │
│  │   ManualAction     ManualAction      │        │
│  │   (instant)        (semantic match)  │        │
│  │        │                │            │        │
│  │        └────────┬───────┘            │        │
│  │                 │                    │        │
│  │            LLMAction                 │        │
│  │            (Ollama)                  │        │
│  └─────────────────────────────────────┘        │
│  Agent Memory Store (SQLite)                     │
│  Tiered Model Router (0.6b→NER, 8b→generation)  │
└──────────────────┬──────────────────────────────┘
                   │
            ┌──────▼──────┐
            │  Neo4j CE   │
            │  5.18       │
            └─────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose (recommended), **or**
- Python 3.11+, Node.js 18+, Neo4j 5.18+, Ollama

### Option A: Docker

```bash
git clone https://github.com/4uhn/MiroFish-Offline-Updated.git
cd MiroFish-Offline-Updated
cp .env.example .env

docker compose up -d

# Pull models into Ollama
docker exec mirofish-ollama ollama pull qwen3:8b
docker exec mirofish-ollama ollama pull nomic-embed-text
```

Open `http://localhost:3000`.

### Option B: Manual (recommended for Apple Silicon)

**1. Start Neo4j**

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/mirofish123 \
  neo4j:5.18-community
```

**2. Start Ollama & pull models**

```bash
ollama serve &
ollama pull qwen3:8b           # LLM (best tool-calling stability)
ollama pull nomic-embed-text   # Embeddings (768d)
```

**3. Configure & run backend**

```bash
cp .env.example .env
# Edit .env: set LLM_PROVIDER=ollama, LLM_MODEL_NAME=qwen3:8b

cd backend
pip install -r requirements.txt
python run.py
```

**4. Run frontend**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Configuration

All settings in `.env` (copy from `.env.example`):

```bash
# LLM provider: "ollama" or "groq"
LLM_PROVIDER=ollama
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=qwen3:8b

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=mirofish123

# Embeddings
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BASE_URL=http://localhost:11434
```

**Optional: Tiered model routing** (speeds up graph building ~5x):

```bash
ollama pull qwen3:0.6b
# Then in .env:
LLM_FAST_MODEL_NAME=qwen3:0.6b
LLM_FAST_NUM_CTX=2048
```

Works with any OpenAI-compatible API — swap Ollama for any other provider by changing `LLM_BASE_URL` and `LLM_API_KEY`.

## LLM Call Reduction

The key optimization for local hardware. Three layers stack:

| Layer | What it does | Savings |
|---|---|---|
| **System One Router** | Non-text actions (like, follow, repost) resolved instantly via archetype-weighted sampling | ~60-70% of all decisions |
| **Tiered Model Router** | NER/extraction tasks routed to qwen3:0.6b (~5x faster); text generation stays on qwen3:8b | ~5x faster graph building |
| **Semantic Response Pool** | CREATE_POST reused cross-archetype via nomic-embed-text similarity matching, with text variation | ~30% of remaining text actions |
| **Agent Memory** | Tracks all actions; replaces summary each round (no stacking) so agents stay contextually grounded | Better output quality |

Net result: **~75-80% of all agent decisions avoid LLM calls entirely.**

### Behavioral Archetypes

Each agent is assigned one of four archetypes that control their action distribution:

| Archetype | Posts | Likes | Follows | Reposts | Does Nothing |
|---|---|---|---|---|---|
| **Lurker** | 3% | 15% | 5% | 2% | 75% |
| **Amplifier** | 5% | 30% | 10% | 35% | 20% |
| **Contributor** | 25% | 20% | 10% | 10% | 35% |
| **Debater** | 30% | 10% | 5% | 15% | 40% |

### Inference Optimization

Additional optimizations for memory-constrained hardware (16GB):

| Technique | Effect |
|---|---|
| **KV Cache Q8** (`OLLAMA_KV_CACHE_TYPE=q8_0`) | Halves KV cache memory with negligible quality loss |
| **Context window 2048** (`OLLAMA_NUM_CTX=2048`) | Sufficient for social media posts, saves ~4x memory vs 8192 |
| **Parallel requests 3** (`OLLAMA_NUM_PARALLEL=3`) | Balances throughput vs memory on 16GB |
| **Keep alive** (`OLLAMA_KEEP_ALIVE=-1`) | Model stays loaded, avoids reload latency between rounds |

## Hardware Requirements

| Component | Minimum (qwen3:8b) | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| GPU/NPU | Apple M-series or 8GB VRAM | 16+ GB VRAM |
| Disk | 15 GB | 30 GB |
| CPU | 4 cores | 8+ cores |

Tested on: M2 Pro 16GB (MacBook Pro). CPU-only mode works but is slower.

## Tools

```bash
# Benchmark LLM throughput, embedding speed, pool efficiency
python backend/scripts/benchmark.py
python backend/scripts/benchmark.py --skip-embedding -o bench.json

# Export a completed simulation as structured JSON
python backend/scripts/export_simulation.py backend/uploads/simulations/<sim_id>
python backend/scripts/export_simulation.py <sim_dir> --pretty --no-posts

# Test LLM readiness before running a simulation
python backend/scripts/test_llm_readiness.py
```

## Use Cases

- **PR crisis testing** — simulate the public reaction to a press release before publishing
- **Market sentiment** — feed financial news and observe simulated social response
- **Policy impact analysis** — test draft regulations against simulated public opinion
- **Research** — multi-agent behavior studies with configurable archetype distributions

## License

AGPL-3.0 — same as the original MiroFish project. See [LICENSE](./LICENSE).

## Credits

This project builds on the work of several open-source projects:

- **[MiroFish](https://github.com/666ghj/MiroFish)** by [666ghj](https://github.com/666ghj) — the original multi-agent simulation engine
- **[MiroFish-Offline](https://github.com/nikmcfly/MiroFish-Offline)** by [nikmcfly](https://github.com/nikmcfly) — the English fork with local-only stack that this project is based on
- **[OASIS](https://github.com/camel-ai/oasis)** (CAMEL-AI) — the underlying simulation engine

**This fork adds:** System One action routing (inspired by [TypeSafe/Jev](https://github.com/jevhub)), tiered model routing, behavioral archetypes, agent memory persistence, semantic response pooling (nomic-embed-text), benchmarking suite, simulation export, KV cache optimization, and inference tuning for Apple Silicon.
