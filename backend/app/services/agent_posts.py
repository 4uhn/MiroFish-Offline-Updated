"""
Verbatim agent posts for the report agent.

The graph holds NER-extracted facts: paraphrases like "Kwame Patel criticized
Bristol Water for delays". In Run 9 the search tools labelled these as quotable,
so the report quoted paraphrases as if agents had said them, while the words
agents actually wrote (in the platform databases) never reached the report
writer. This index reads what agents wrote on both platforms so the tools can
show real, attributable text next to the facts.
"""

import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..config import Config

logger = logging.getLogger('mirofish.agent_posts')


@dataclass
class AgentPost:
    agent_id: int
    name: str
    profession: str
    platform: str       # "Twitter" or "Reddit"
    kind: str           # "post", "quote" or "comment"
    content: str
    is_public: bool     # synthetic member of the public, not an institution/official

    def to_text(self) -> str:
        role = f", {self.profession}" if self.profession and self.is_public else ""
        return f'{self.name}{role} ({self.platform} {self.kind}): "{self.content}"'


# Twitter/Reddit share OASIS's schema. Reposts copy no text, so only
# originals, quote text and comments are authored words.
_QUERIES = (
    ("post", "SELECT u.agent_id, p.content FROM post p JOIN user u ON u.user_id = p.user_id "
             "WHERE p.original_post_id IS NULL AND p.content != ''"),
    ("quote", "SELECT u.agent_id, p.quote_content FROM post p JOIN user u ON u.user_id = p.user_id "
              "WHERE p.quote_content IS NOT NULL AND p.quote_content != ''"),
    ("comment", "SELECT u.agent_id, c.content FROM comment c JOIN user u ON u.user_id = c.user_id "
                "WHERE c.content IS NOT NULL AND c.content != ''"),
)


class AgentPostIndex:
    """What agents wrote in one simulation, searchable by embedding similarity."""

    _cache: Dict[str, "AgentPostIndex"] = {}
    _cache_lock = threading.Lock()

    def __init__(self, simulation_id: str, embedding_service=None):
        self.simulation_id = simulation_id
        self._embedder = embedding_service
        self._embeddings: Optional[List[List[float]]] = None
        self._embed_lock = threading.Lock()
        self.posts: List[AgentPost] = self._load()

    @classmethod
    def for_simulation(cls, simulation_id: str) -> "AgentPostIndex":
        """Shared, lazily built index per simulation (the platform DBs don't change after the run)."""
        with cls._cache_lock:
            index = cls._cache.get(simulation_id)
            if index is None:
                from ..storage.embedding_service import EmbeddingService
                index = cls(simulation_id, EmbeddingService())
                cls._cache[simulation_id] = index
            return index

    def _sim_dir(self) -> str:
        return os.path.join(Config.OASIS_SIMULATION_DATA_DIR, self.simulation_id)

    def _load_people(self) -> Dict[int, Dict[str, str]]:
        people: Dict[int, Dict[str, str]] = {}
        public = public_agent_ids(self.simulation_id)
        path = os.path.join(self._sim_dir(), "reddit_profiles.json")
        try:
            with open(path, encoding="utf-8") as f:
                for p in json.load(f):
                    people[int(p["user_id"])] = {
                        "name": p.get("name") or p.get("username", ""),
                        "profession": p.get("profession", ""),
                        "is_public": int(p["user_id"]) in public,
                    }
        except (OSError, ValueError, KeyError) as e:
            logger.warning(f"AgentPostIndex: could not read profiles for {self.simulation_id}: {e}")
        return people

    def _load(self) -> List[AgentPost]:
        people = self._load_people()
        posts: List[AgentPost] = []
        seen = set()
        for platform in ("twitter", "reddit"):
            db = os.path.join(self._sim_dir(), f"{platform}_simulation.db")
            if not os.path.exists(db):
                continue
            try:
                conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                try:
                    for kind, sql in _QUERIES:
                        for agent_id, text in conn.execute(sql):
                            text = " ".join((text or "").split())
                            key = text.lower()
                            if len(text) < 20 or key in seen:
                                continue
                            seen.add(key)
                            person = people.get(agent_id, {})
                            posts.append(AgentPost(
                                agent_id=agent_id,
                                name=person.get("name") or f"Agent {agent_id}",
                                profession=person.get("profession", ""),
                                platform=platform.capitalize(),
                                kind=kind,
                                content=text,
                                is_public=bool(person.get("is_public")),
                            ))
                finally:
                    conn.close()
            except sqlite3.Error as e:
                logger.warning(f"AgentPostIndex: failed reading {db}: {e}")
        logger.info(f"AgentPostIndex: {len(posts)} authored posts/comments for {self.simulation_id}")
        return posts

    def counts_by_agent(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for p in self.posts:
            counts[p.agent_id] = counts.get(p.agent_id, 0) + 1
        return counts

    def stats_text(self, keywords: Optional[List[str]] = None) -> str:
        """Counts of what agents wrote, for checking claims about reach.

        Run 10's report said the campaigners' 2019 warnings "gained traction";
        0 of ~95 agent texts mentioned them. Counts only, no quotable text.
        A keyword matches a text when each of its words starts a word there
        ("warning" matches "warnings"). Texts copied word for word by several
        agents are counted once.
        """
        posts = self.posts
        if not posts:
            return "No agent posts or comments were found for this simulation."

        def agents(ps):
            return len({p.agent_id for p in ps})

        public = [p for p in posts if p.is_public]
        inst = [p for p in posts if not p.is_public]
        lines = [
            "### Simulation statistics (counted from the platform databases; counts only, nothing here is quotable)",
            f"Authored texts (posts, quotes, comments): {len(posts)} by {agents(posts)} agents",
            f"- Members of the public: {len(public)} texts by {agents(public)} agents",
            f"- Institutions and officials: {len(inst)} texts by {agents(inst)} agents",
        ]
        by_kind: Dict[str, int] = {}
        for p in posts:
            key = f"{p.platform} {p.kind}s"
            by_kind[key] = by_kind.get(key, 0) + 1
        lines.append("By platform: " + ", ".join(f"{k} {v}" for k, v in sorted(by_kind.items())))
        names = {p.agent_id: p.name for p in posts}
        top = sorted(self.counts_by_agent().items(), key=lambda kv: -kv[1])[:8]
        lines.append("Most active authors: " + ", ".join(f"{names[a]} ({n})" for a, n in top))

        keywords = [k.strip() for k in (keywords or []) if k and k.strip()]
        if keywords:
            # Run 14's report turned "3 public" (agents) into "3 public texts",
            # read "Bristol Water: 5 public" as 5 agents blaming it, and
            # "regulators: 0" as regulators never coming up (the Environment
            # Agency was in 9 texts); each count now says what it counts
            lines.append("Mentions (texts containing the words, the agents who wrote them and how many of those "
                         "agents are members of the public; a mention is not blame or support, and 0 means no text "
                         "uses these words, not that the topic never came up under other names):")
            for kw in keywords[:10]:
                words = [w for w in re.findall(r"\w+", kw.lower())]
                if not words:
                    continue
                pats = [re.compile(rf"\b{re.escape(w)}") for w in words]
                hits = [p for p in posts if all(pat.search(p.content.lower()) for pat in pats)]
                lines.append(f"- \"{kw}\": {len(hits)} texts by {agents(hits)} agents, "
                             f"{agents([p for p in hits if p.is_public])} of them public agents")
        return "\n".join(lines)

    def _ensure_embeddings(self) -> bool:
        with self._embed_lock:
            if self._embeddings is not None:
                return True
            if not self._embedder or not self.posts:
                return False
            try:
                self._embeddings = self._embedder.embed_batch([p.content for p in self.posts])
                return True
            except Exception as e:
                logger.warning(f"AgentPostIndex: embedding {len(self.posts)} posts failed: {e}")
                return False

    def search(self, query: str, k: int = 6, exclude: Optional[set] = None) -> List[AgentPost]:
        """Top-k posts most similar to the query.

        At most 2 per agent and at most half from institutions: official
        statements match topic queries most closely and would otherwise crowd
        out the public voices the report is about.

        Returns [] (and logs) when embeddings are unavailable, rather than
        substituting unranked posts.
        """
        if not query or not self._ensure_embeddings():
            return []
        try:
            q = self._embedder.embed(query)
        except Exception as e:
            logger.warning(f"AgentPostIndex: embedding query failed: {e}")
            return []
        scored = sorted(
            ((_cosine(q, e), p) for e, p in zip(self._embeddings, self.posts)),
            key=lambda x: x[0], reverse=True,
        )
        picked: List[AgentPost] = []
        per_agent: Dict[int, int] = {}
        max_institutional = (k + 1) // 2
        institutional = 0
        for _, p in scored:
            if exclude and p.content in exclude:
                continue
            if per_agent.get(p.agent_id, 0) >= 2:
                continue
            if not p.is_public:
                if institutional >= max_institutional:
                    continue
                institutional += 1
            per_agent[p.agent_id] = per_agent.get(p.agent_id, 0) + 1
            picked.append(p)
            if len(picked) >= k:
                break
        return picked


def public_agent_ids(simulation_id: str) -> set:
    """Agent ids of synthetic members of the public (entity_type Synthetic_*) in simulation_config.json."""
    path = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id, "simulation_config.json")
    try:
        with open(path, encoding="utf-8") as f:
            configs = json.load(f).get("agent_configs", [])
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read agent types for {simulation_id}: {e}")
        return set()
    return {
        int(a["agent_id"]) for a in configs
        if str(a.get("entity_type", "")).startswith("Synthetic_")
    }


# Trace rows that are not the agent doing anything in the simulation.
_NON_ACTIONS = ("sign_up", "refresh", "interview", "do_nothing")


def active_agent_ids(simulation_id: str) -> Optional[set]:
    """Agent ids with at least one real action (post, comment, like, repost...) on either platform.

    In the Run 11 smoke test qwen3:8b invented posts for an agent that had
    only signed up, whatever the interview prompt said, so agents with
    nothing on record are not interviewed. None when no platform DB could
    be read, so callers can tell "nobody acted" from "unknown".
    """
    sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
    placeholders = ",".join("?" * len(_NON_ACTIONS))
    active: set = set()
    read_any = False
    for platform in ("twitter", "reddit"):
        db = os.path.join(sim_dir, f"{platform}_simulation.db")
        if not os.path.exists(db):
            continue
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    f"SELECT DISTINCT user_id FROM trace WHERE action NOT IN ({placeholders})",
                    _NON_ACTIONS,
                )
                active.update(int(r[0]) for r in rows)
                read_any = True
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.warning(f"active_agent_ids: failed reading {db}: {e}")
    return active if read_any else None


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
