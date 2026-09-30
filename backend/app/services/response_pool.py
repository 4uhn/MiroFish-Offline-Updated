"""
Response Pool v4 — voice-gated amplification with semantic topic matching.

When an agent is routed to write (System 2), the pool may instead have it
amplify a post or comment already written by a same-voice peer: a repost on
Twitter, an upvote on Reddit. Voice is the synthetic persona template plus
stance (e.g. two skeptical residents). Institutional agents have no voice, so
they neither feed nor draw from the pool. Within a voice, entries are ranked
by embedding similarity to the agent's interested topics, and an agent is
never handed its own text or the same item twice.

v3 re-posted the peer's text under the second agent's name with cosmetic
variation. In Run 9 that put one healthcare worker's "I just came from a
shift..." under another's name, which is a fabricated first-person claim, and
on Reddit the pool never fired because only posts were pooled while Reddit
agents write comments. Amplifying the original keeps the LLM saving without
inventing speech, and works for comments.
"""

import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger('mirofish.response_pool')


@dataclass
class PoolEntry:
    content: str
    archetype: str
    voice: str
    kind: str                       # "post" or "comment"
    item_id: int                    # post_id or comment_id in the platform DB
    source_agent_id: Optional[int] = None
    embedding: Optional[List[float]] = None
    amplify_count: int = 0


class ResponsePool:
    """Pool of LLM-written posts/comments that same-voice peers may amplify."""

    _MAX_AMPLIFY_PER_ENTRY = 2

    def __init__(
        self,
        amplify_probability: float = 0.3,
        max_pool_size: int = 32,
        min_content_length: int = 30,
        embedding_service=None,
        dedup_threshold: float = 0.90,
    ):
        self._entries: List[PoolEntry] = []
        self._amplify_prob = amplify_probability
        self._max_pool_size = max_pool_size
        self._min_content_length = min_content_length
        self._embedder = embedding_service
        self._dedup_threshold = dedup_threshold
        self._amplified: Set[Tuple[int, str, int]] = set()
        self._amplify_count = 0
        self._miss_count = 0
        self._dedup_count = 0
        self._semantic_matches = 0
        self._no_candidate_count = 0

    def add(
        self,
        archetype: str,
        content: str,
        kind: str,
        item_id: Optional[int],
        voice: Optional[str] = None,
        source_agent_id: Optional[int] = None,
    ):
        """Add an LLM-written post or comment. Items without a voice (institutions) are not pooled.

        A near-duplicate of an existing entry is counted in stats['deduped'] and
        not added, so the pool does not fill with one talking point.
        """
        if not voice or item_id is None:
            return
        if len(content.strip()) < self._min_content_length:
            return

        content = content.strip()
        embedding = self._embed(content)

        if embedding:
            for entry in self._entries:
                if entry.embedding and _cosine_sim(embedding, entry.embedding) > self._dedup_threshold:
                    self._dedup_count += 1
                    return

        self._entries.append(PoolEntry(
            content=content, archetype=archetype, voice=voice, kind=kind,
            item_id=int(item_id), source_agent_id=source_agent_id, embedding=embedding,
        ))
        if len(self._entries) > self._max_pool_size:
            self._entries.pop(0)

    def try_amplify(
        self,
        archetype: str,
        query: Optional[str] = None,
        voice: Optional[str] = None,
        agent_id: Optional[int] = None,
    ) -> Optional[PoolEntry]:
        """Return a same-voice peer's item for this agent to amplify, or None to fall through to the LLM.

        Args:
            archetype: Agent's behavioral archetype (kept for stats/compat)
            query: Topic string for semantic ranking (e.g. joined interested_topics)
            voice: Agent's voice key; None (institutions) disables the pool
            agent_id: Acting agent, so it never amplifies its own item or one it already amplified
        """
        if not voice or not self._entries:
            return None

        if random.random() > self._amplify_prob:
            self._miss_count += 1
            return None

        entry = self._select_entry(voice, agent_id, query)
        if entry is None:
            self._no_candidate_count += 1
            return None

        entry.amplify_count += 1
        self._amplify_count += 1
        self._amplified.add((agent_id, entry.kind, entry.item_id))
        return entry

    def _select_entry(
        self, voice: str, agent_id: Optional[int], query: Optional[str] = None,
    ) -> Optional[PoolEntry]:
        """Pick a same-voice entry from another agent, ranked by topic similarity when possible."""
        available = [
            e for e in self._entries
            if e.voice == voice
            and e.source_agent_id != agent_id
            and e.amplify_count < self._MAX_AMPLIFY_PER_ENTRY
            and (agent_id, e.kind, e.item_id) not in self._amplified
        ]
        if not available:
            return None

        if query and self._embedder:
            query_emb = self._embed(query)
            if query_emb:
                scored = [
                    (_cosine_sim(query_emb, e.embedding), e)
                    for e in available if e.embedding
                ]
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    best_score, best_entry = random.choice(scored[:3])
                    if best_score > 0.3:
                        self._semantic_matches += 1
                        return best_entry

        return random.choice(available)

    def _embed(self, text: str) -> Optional[List[float]]:
        if not self._embedder:
            return None
        try:
            result = self._embedder.embed(text)
            if result and len(result) > 0:
                return result
            logger.warning("Embedding returned empty vector for: %s", text[:60])
            return None
        except Exception as e:
            logger.warning("Embedding failed: %s (text: %s)", e, text[:60])
            return None

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "pooled": len(self._entries),
            "amplified": self._amplify_count,
            "skipped": self._miss_count,
            "deduped": self._dedup_count,
            "semantic_matches": self._semantic_matches,
            "no_candidate": self._no_candidate_count,
        }


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
