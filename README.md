<div align="center">

<img src="./static/image/mirofish-offline-banner.png" alt="MiroFish Offline" width="100%"/>

# MiroFish-Offline

**Fully local multi-agent simulation engine — no cloud APIs required.**

*Simulate public opinion, market sentiment, and social dynamics entirely on your hardware.*

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat-square)](./LICENSE)

</div>

## What is this?

MiroFish is a multi-agent simulation engine: upload any document (press release, policy draft, financial report), and it generates AI agents with unique personalities that simulate the public reaction on social media. Posts, arguments, opinion shifts — hour by hour.

This fork makes it **fully local and fully English**, optimized for Apple Silicon (M2 Pro 16GB tested):

| Feature | Original MiroFish | MiroFish-Offline |
|---|---|---|
| Language | Chinese UI | **English UI** (1,000+ strings translated) |
| Graph DB | Zep Cloud | **Neo4j CE 5.18** |
| LLM | DashScope / OpenAI API | **Ollama** (qwen3:8b) |
| Embeddings | Zep Cloud | **nomic-embed-text** via Ollama |
| Cloud dependency | API keys required | **Zero** |
| Action routing | Every action = LLM call | **System One** (60-70% skip LLM) |
| Agent behavior | Uniform | **4 archetypes** (Lurker, Amplifier, Contributor, Debater) |
| Agent memory | Stateless | **Action journal** with SQLite persistence |
| Response sharing | None | **TopoSim-inspired** archetype pooling |

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
│  │   (instant)        (from pool)       │        │
│  │        │                │            │        │
│  │        └────────┬───────┘            │        │
│  │                 │                    │        │
│  │            LLMAction                 │        │
│  │            (Ollama)                  │        │
│  └─────────────────────────────────────┘        │
│  Agent Memory Store (SQLite)                     │
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
git clone https://github.com/44jch/MiroFish-Offline.git
cd MiroFish-Offline
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

Works with any OpenAI-compatible API — swap Ollama for any other provider by changing `LLM_BASE_URL` and `LLM_API_KEY`.

## LLM Call Reduction

The key optimization for local hardware. Three layers stack:

| Layer | What it does | Savings |
|---|---|---|
| **System One Router** | Non-text actions (like, follow, repost) resolved instantly via archetype-weighted sampling | ~60-70% of all decisions |
| **Response Pool** | CREATE_POST reused from same-archetype agent with text variation (synonym swap, sentence reorder) | ~30% of remaining text actions |
| **Agent Memory** | Tracks all actions; injects summary before LLM calls so agents stay contextually grounded | Better output quality |

Net result: **~75-80% of all agent decisions avoid LLM calls entirely.**

### Behavioral Archetypes

Each agent is assigned one of four archetypes that control their action distribution:

| Archetype | Posts | Likes | Follows | Reposts | Does Nothing |
|---|---|---|---|---|---|
| **Lurker** | 3% | 15% | 5% | 2% | 75% |
| **Amplifier** | 5% | 30% | 10% | 35% | 20% |
| **Contributor** | 25% | 20% | 10% | 10% | 35% |
| **Debater** | 30% | 10% | 5% | 15% | 40% |

## Hardware Requirements

| Component | Minimum (qwen3:8b) | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| GPU/NPU | Apple M-series or 8GB VRAM | 16+ GB VRAM |
| Disk | 15 GB | 30 GB |
| CPU | 4 cores | 8+ cores |

Tested on: M2 Pro 16GB (MacBook Pro). CPU-only mode works but is slower.

## Use Cases

- **PR crisis testing** — simulate the public reaction to a press release before publishing
- **Market sentiment** — feed financial news and observe simulated social response
- **Policy impact analysis** — test draft regulations against simulated public opinion
- **Research** — multi-agent behavior studies with configurable archetype distributions

## License

AGPL-3.0 — same as the original MiroFish project. See [LICENSE](./LICENSE).

## Credits

Fork of [MiroFish](https://github.com/666ghj/MiroFish) by [666ghj](https://github.com/666ghj). Simulation engine powered by [OASIS](https://github.com/camel-ai/oasis) (CAMEL-AI).

**This fork adds:** full English translation, local-only operation (Neo4j CE + Ollama), System One action routing, behavioral archetypes, agent memory persistence, TopoSim-inspired response pooling, and ontology generation improvements.
