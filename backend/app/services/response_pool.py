"""
Response Sharing Pool v2 — Semantic similarity for cross-archetype sharing.

V1: Archetype-siloed pools with string-level text variation.
V2: Unified pool with embedding-based semantic dedup and topic matching.
Posts from any archetype can be reused by any other when relevant.

Falls back to v1 behavior when embedding service is unavailable.
Only applies to CREATE_POST. Comments and quotes require too much context.
"""

import random
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger('mirofish.response_pool')


@dataclass
class PoolEntry:
    content: str
    archetype: str
    embedding: Optional[List[float]] = None
    reuse_count: int = 0


class ResponsePool:
    """Unified pool of LLM-generated posts with semantic matching."""

    def __init__(
        self,
        reuse_probability: float = 0.3,
        max_pool_size: int = 32,
        min_content_length: int = 30,
        embedding_service=None,
        dedup_threshold: float = 0.90,
    ):
        self._entries: List[PoolEntry] = []
        self._reuse_prob = reuse_probability
        self._max_pool_size = max_pool_size
        self._min_content_length = min_content_length
        self._embedder = embedding_service
        self._dedup_threshold = dedup_threshold
        self._reuse_count = 0
        self._miss_count = 0
        self._dedup_count = 0
        self._semantic_matches = 0

    def add(self, archetype: str, content: str):
        """Add an LLM-generated response to the pool with semantic dedup."""
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
            content=content, archetype=archetype, embedding=embedding,
        ))
        if len(self._entries) > self._max_pool_size:
            self._entries.pop(0)

    def try_reuse(
        self,
        archetype: str,
        query: Optional[str] = None,
    ) -> Optional[str]:
        """Try to reuse a pooled response with text variation.

        Args:
            archetype: Agent's behavioral archetype
            query: Optional topic string for semantic matching (e.g. joined interested_topics)
        """
        if not self._entries:
            return None

        if random.random() > self._reuse_prob:
            self._miss_count += 1
            return None

        base = self._select_entry(archetype, query)
        if not base:
            return None

        varied = vary_text(base)
        if _word_overlap(base, varied) > 0.85:
            varied = vary_text(base)

        self._reuse_count += 1
        return varied

    _MAX_REUSE_PER_ENTRY = 2

    def _select_entry(self, archetype: str, query: Optional[str] = None) -> Optional[str]:
        """Select the best pool entry for reuse."""
        available = [e for e in self._entries if e.reuse_count < self._MAX_REUSE_PER_ENTRY]
        if not available:
            return None

        # Semantic matching: find the most topic-relevant post
        if query and self._embedder:
            query_emb = self._embed(query)
            if query_emb:
                scored = []
                for entry in available:
                    if entry.embedding:
                        score = _cosine_sim(query_emb, entry.embedding)
                        scored.append((score, entry))
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top = scored[:min(3, len(scored))]
                    best_score, best_entry = random.choice(top)
                    if best_score > 0.3:
                        self._semantic_matches += 1
                        best_entry.reuse_count += 1
                        return best_entry.content

        # Fallback: prefer same archetype, allow cross-archetype
        same = [e for e in available if e.archetype == archetype]
        if same and random.random() < 0.7:
            chosen = random.choice(same)
        else:
            chosen = random.choice(available)
        chosen.reuse_count += 1
        return chosen.content

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
            "reused": self._reuse_count,
            "skipped": self._miss_count,
            "deduped": self._dedup_count,
            "semantic_matches": self._semantic_matches,
        }


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ═══════════════════════════════════════════════════════════════
# Text variation — lightweight string transforms (unchanged from v1)
# ═══════════════════════════════════════════════════════════════

_FILLERS = [
    "honestly, ", "tbh, ", "I think ", "ngl, ", "just saying, ",
    "for real though, ", "look, ", "seriously, ", "can't believe ",
    "important: ", "worth noting: ", "reminder: ",
]

_OPENER_GROUPS = [
    ["Breaking:", "Update:", "Alert:", "Just in:"],
    ["Critical update:", "Important:", "PSA:", "Heads up:"],
    ["This is huge.", "This matters.", "Pay attention.", "Don't ignore this."],
]

_SYNONYMS = {
    "important": ["crucial", "vital", "critical", "essential"],
    "shows": ["reveals", "indicates", "demonstrates", "confirms"],
    "says": ["states", "reports", "notes", "claims"],
    "warns": ["cautions", "alerts", "advises", "urges"],
    "supports": ["backs", "endorses", "champions", "advocates for"],
    "help": ["assist", "aid", "support"],
    "big": ["major", "significant", "substantial"],
    "bad": ["concerning", "troubling", "alarming", "worrying"],
    "good": ["positive", "encouraging", "promising"],
    "people": ["individuals", "citizens", "the public", "everyone"],
    "need": ["must", "should", "have to"],
    "very": ["extremely", "incredibly", "particularly"],
    "problem": ["issue", "concern", "challenge"],
    "said": ["stated", "mentioned", "noted", "commented"],
    "confirms": ["reports", "announces", "verifies", "reveals"],
    "linked": ["connected", "tied", "related", "associated"],
    "available": ["accessible", "open", "ready", "offered"],
    "immediate": ["urgent", "prompt", "swift"],
    "seek": ["get", "pursue", "request", "obtain"],
    "active": ["operational", "running", "live", "ongoing"],
    "cases": ["instances", "incidents", "occurrences"],
    "set": ["established", "launched", "opened"],
    "highlights": ["emphasizes", "stresses", "underlines", "points out"],
    "recognizing": ["identifying", "spotting", "detecting"],
    "always": ["make sure to", "be sure to", "remember to"],
    "appear": ["emerge", "develop", "show up", "manifest"],
    "treatment": ["care", "intervention", "medication"],
    "impact": ["effect", "consequence", "influence"],
    "address": ["tackle", "handle", "deal with", "respond to"],
    "working": ["collaborating", "cooperating", "coordinating"],
    "ensure": ["guarantee", "make sure", "verify"],
}

_HASHTAG_SETS = [
    ["#StaySafe", "#PublicHealth", "#HealthAlert"],
    ["#Breaking", "#News", "#Update"],
    ["#Awareness", "#SpreadTheWord", "#ShareThis"],
    ["#Community", "#Together", "#Support"],
]


def vary_text(text: str) -> str:
    """Apply lightweight string transformations to make text read as written by a different person."""
    sentences = _split_sentences(text)

    transforms = [
        _swap_opener,
        _shuffle_middle_sentences,
        _swap_synonyms,
        _swap_synonyms,
        _vary_punctuation,
        _toggle_filler,
        _vary_hashtags,
        _add_reaction_prefix,
    ]
    random.shuffle(transforms)
    n_transforms = random.randint(3, min(5, len(transforms)))
    for transform in transforms[:n_transforms]:
        sentences = transform(sentences)

    result = " ".join(s.strip() for s in sentences if s.strip())
    return result


def _split_sentences(text: str) -> List[str]:
    hashtag_match = re.search(r'((?:\s*#\w+)+)\s*$', text)
    hashtags = hashtag_match.group(0).strip() if hashtag_match else ""
    body = text[:hashtag_match.start()].strip() if hashtag_match else text

    sentences = re.split(r'(?<=[.!?])\s+', body)
    sentences = [s for s in sentences if s.strip()]

    if hashtags:
        sentences.append(hashtags)

    return sentences


def _swap_opener(sentences: List[str]) -> List[str]:
    if not sentences:
        return sentences

    first = sentences[0]
    for group in _OPENER_GROUPS:
        for opener in group:
            if first.startswith(opener):
                replacement = random.choice([o for o in group if o != opener])
                sentences[0] = replacement + first[len(opener):]
                return sentences
    return sentences


def _shuffle_middle_sentences(sentences: List[str]) -> List[str]:
    if len(sentences) <= 2:
        return sentences

    has_hashtag_end = sentences[-1].startswith("#")
    last = sentences[-1] if has_hashtag_end else None
    body = sentences[:-1] if has_hashtag_end else list(sentences)

    if len(body) <= 2:
        return sentences

    middle = body[1:]
    random.shuffle(middle)
    result = [body[0]] + middle
    if last is not None:
        result.append(last)
    return result


def _swap_synonyms(sentences: List[str]) -> List[str]:
    result = []
    swaps_done = 0
    max_swaps = random.randint(1, 2)

    for s in sentences:
        if swaps_done >= max_swaps:
            result.append(s)
            continue

        words = s.split()
        for i, word in enumerate(words):
            clean = word.lower().strip(".,!?;:'\"")
            if clean in _SYNONYMS and swaps_done < max_swaps:
                replacement = random.choice(_SYNONYMS[clean])
                if word[0].isupper():
                    replacement = replacement.capitalize()
                trailing = ""
                for ch in reversed(word):
                    if ch in ".,!?;:'\"":
                        trailing = ch + trailing
                    else:
                        break
                words[i] = replacement + trailing
                swaps_done += 1
        result.append(" ".join(words))

    return result


def _vary_punctuation(sentences: List[str]) -> List[str]:
    result = []
    for s in sentences:
        if s.startswith("#"):
            result.append(s)
            continue
        if random.random() < 0.3:
            if s.endswith("."):
                s = s[:-1] + random.choice(["!", ".", "..."])
            elif s.endswith("!"):
                s = s[:-1] + random.choice([".", "!!"])
        result.append(s)
    return result


def _toggle_filler(sentences: List[str]) -> List[str]:
    if not sentences:
        return sentences

    first = sentences[0]

    for filler in _FILLERS:
        if first.lower().startswith(filler.lower()):
            sentences[0] = first[len(filler):].strip()
            if sentences[0]:
                sentences[0] = sentences[0][0].upper() + sentences[0][1:]
            return sentences

    if random.random() < 0.3:
        filler = random.choice(_FILLERS)
        if first[:2].isupper():
            sentences[0] = filler + first
        else:
            sentences[0] = filler + first[0].lower() + first[1:]

    return sentences


def _add_reaction_prefix(sentences: List[str]) -> List[str]:
    if not sentences or sentences[0].startswith("#"):
        return sentences

    prefixes = [
        "Wow, ", "This is concerning — ", "Just saw this: ",
        "Can't ignore this — ", "Everyone needs to know: ",
        "Sharing this — ", "Seriously, ", "So ",
        "This just in: ", "FYI: ", "Whoa, ",
    ]

    first_lower = sentences[0].lower()
    if any(first_lower.startswith(f.lower()) for f in _FILLERS):
        return sentences
    for group in _OPENER_GROUPS:
        if any(first_lower.startswith(o.lower()) for o in group):
            return sentences

    if random.random() < 0.4:
        prefix = random.choice(prefixes)
        first = sentences[0]
        if first[:2].isupper():
            sentences[0] = prefix + first
        else:
            sentences[0] = prefix + first[0].lower() + first[1:]

    return sentences


def _vary_hashtags(sentences: List[str]) -> List[str]:
    if not sentences:
        return sentences

    last = sentences[-1]
    if last.startswith("#"):
        if random.random() < 0.4:
            new_set = random.choice(_HASHTAG_SETS)
            count = min(len(last.split()), len(new_set))
            sentences[-1] = " ".join(random.sample(new_set, count))
        elif random.random() < 0.3:
            sentences = sentences[:-1]
    else:
        if random.random() < 0.2:
            tag_set = random.choice(_HASHTAG_SETS)
            count = random.randint(1, 2)
            sentences.append(" ".join(random.sample(tag_set, count)))

    return sentences


def _word_overlap(a: str, b: str) -> float:
    """Quick word-overlap similarity check."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / max(len(words_a), len(words_b))
