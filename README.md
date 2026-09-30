# MiroFish-Offline+

**On-device multi-agent social simulation. No cloud APIs; runs on a 16 GB Mac.**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat-square)](./LICENSE)

## What this is

You give it a document describing a situation (a crisis, an announcement, a policy) and a question in plain English. It builds a cast of about 30 AI agents (residents, journalists, officials, organisations) and lets them react to the situation hour by hour on a simulated Twitter and Reddit. Then a report agent writes up what happened, using only what the agents actually posted.

It's a tool for exploring scenarios, not a forecast. Every model, database and embedding runs on your own machine: no API keys, no cloud.

It's an extended fork of [MiroFish-Offline](https://github.com/nikmcfly/MiroFish-Offline), which is itself a fork of [MiroFish](https://github.com/666ghj/MiroFish). The web app walks you through five steps:

1. **Graph build.** The LLM designs an ontology for your document. Its entities and relationships are then extracted into a Neo4j knowledge graph.
2. **Environment setup.** Agent personas are generated from the graph, along with a scenario clock and the dated events from your document.
3. **Simulation.** Agents post, comment, like and repost on both platforms (via [OASIS](https://github.com/camel-ai/oasis)). Each round, every agent sees the time, the facts released so far and its own feed.
4. **Report.** A report agent searches the graph, reads the agents' posts, interviews agents, and writes an analysis.
5. **Interaction.** You can chat with the report agent or any individual agent, or send a survey to every agent.

## Quick start (macOS)

### Prerequisites

- An Apple Silicon Mac with **16 GB RAM or more**.
- [Docker Desktop](https://www.docker.com/products/docker-desktop/), for Neo4j.
- [Ollama](https://ollama.com/download).
- **Python 3.11.** Python 3.12 won't work, because OASIS and CAMEL require <3.12. Install it with `brew install python@3.11`.
- **Node.js 18 or newer.** Install it with `brew install node`.
- About 15 GB of free disk space for the models and the Python dependencies.

### 1. Clone and configure

```bash
git clone https://github.com/4uhn/MiroFish-Offline-Updated.git
cd MiroFish-Offline-Updated
cp .env.example .env
```

Open `.env` and set `NEO4J_PASSWORD` to a password of your choice. Leave everything else as it is. Each setting is explained in `.env.example`.

### 2. Start Neo4j

```bash
docker compose up -d neo4j
```

This starts Neo4j 5.18 on `localhost:7687`, using the password from `.env`.

### 3. Start Ollama with the right settings

> **Important:** the Ollama desktop app starts its own server in the background with default settings, and your settings below would then be silently ignored. Quit it first: use the llama icon in the menu bar, then **Quit Ollama**. Check with `pgrep -fl ollama`, which should print nothing.

In a terminal that you keep open:

```bash
OLLAMA_CONTEXT_LENGTH=8192 \
OLLAMA_NUM_PARALLEL=3 \
OLLAMA_KV_CACHE_TYPE=q8_0 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KEEP_ALIVE=-1 \
ollama serve
```

In a second terminal, download the two models (about 5.5 GB):

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Once a model has loaded, `ollama ps` should show a context of `8192`. If you change `OLLAMA_CONTEXT_LENGTH`, set `OLLAMA_NUM_CTX` in `.env` to the same value, because the backend sizes its prompts from it.

### 4. Start the backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # first install is large (OASIS pulls in PyTorch)
python run.py
```

The API listens on `http://127.0.0.1:5001`.

### 5. Start the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**.

> `docker-compose.yml` also defines containers for the whole stack, but its Ollama service needs an NVIDIA GPU and Docker on macOS can't use the Mac's GPU. On a Mac, run Ollama natively as above.

## Try the example

`examples/bristol-water/` contains a ready-made scenario: the first week of a lead contamination crisis in Bristol's water supply. The scenario is **fictional**. It names real organisations, but none of the events happened.

1. On the home page, upload `examples/bristol-water/seed.md` (PDF, Markdown and text files all work).
2. Paste the contents of `examples/bristol-water/prompt.md` into the simulation requirement box, then click **Start Engine**.
3. Follow the five steps. On a 16 GB M2 Pro, a full run takes about 40 minutes.

**Don't reload or close the page during the simulation step.** Reopening the page restarts the simulation from the beginning.

## Design notes

- **Everything runs on one model: qwen3:8b.** A 16 GB machine fits one 8B model with three parallel 8K-token slots. A second model would mean a second set of behaviours to validate.
- **Qwen3's hidden reasoning is turned off.** Qwen3 was silently "thinking" on every call. Turning it off cut an agent decision from 35.7 s to 3.0 s.
- **Most actions don't use the LLM.** A dual-process "System One" router samples likes, reposts and do-nothing from each agent's behavioural archetype (lurker, amplifier, contributor or debater). Only posts and comments go to the LLM. In our test runs, about half of all agent decisions skipped the LLM.
- **Agents amplify similar posts instead of copying them.** When an agent is about to write something close to a post from a similar agent, it reposts or upvotes that post instead of generating near-duplicate text. It never copies another agent's words.
- **The scenario unfolds on a clock.** Dated events from the document are released at their simulated hour. Facts containing numbers that aren't in the document are dropped.
- **Graph memory is grounded and scoped to the run.** A relationship extracted from agent activity is kept only if it is backed by what that agent actually did or wrote. Each run's graph edges are tagged with that run, so one run never leaks into another.
- **The report can't invent numbers or quotes.** Counts (who said what, how often) are computed in code and handed to the model. Every quote is checked word for word against real agent posts and interview answers, and must be credited to the right speaker; a quote that fails is removed.
- **Evidence comes before conclusions.** Report sections are framed as questions, and the summary is written last, from what the sections found.
- **Code enforces integrity, not style.** Code checks only whether quotes are real, numbers are measured and prompts fit the context window. The model's wording isn't patched with extra rules: those patches kept growing and would be unnecessary with a larger model.

## Limitations

- **This is not a forecast.** Each scenario has been run once, with no variance across runs and no validation against real-world outcomes.
- **An 8B model has a ceiling.** Reports can overclaim ("trust fluctuated", "gained traction") without the numbers to back it. Interview answers are sometimes presented as fact. One articulate agent can supply most of the quotes.
- **Agents are similar.** They share templated openers, sometimes converge on the same view, and occasionally invent personal details or give wrong advice.
- **Known issue: an event can be dropped.** The step that turns the document into a scenario can drop an event the question asks about; in the Bristol example it was the CEO's apology. No agent then sees that event, but the report may still describe reactions to it. The fix would be to check that every event named in the question is present in the scenario. It hasn't been made yet.
- **Tested on one machine.** Development and testing used a single 16 GB M2 Pro MacBook Pro.

## License and credits

Licensed under the **GNU Affero General Public License v3.0** (see [LICENSE](./LICENSE)), the same license as the projects it builds on:

- [**MiroFish**](https://github.com/666ghj/MiroFish) by 666ghj: the original multi-agent simulation and prediction engine.
- [**MiroFish-Offline**](https://github.com/nikmcfly/MiroFish-Offline) by nikmcfly: the English, local-only fork (Neo4j and Ollama instead of cloud services) that this project extends.
