"""
Response Sharing Pool — TopoSim-inspired response reuse for similar agents.

When multiple agents of the same archetype need to generate posts, the first
agent's LLM response is pooled. Subsequent agents of the same archetype have
a configurable chance of reusing a varied version of a pooled response instead
of making a new LLM call.

Text variation is pure string manipulation — no LLM call needed. This produces
posts that convey similar information but read as written by a different person.

Only applies to CREATE_POST. Comments and quotes require too much context to reuse.
"""

import random
import re
import logging
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger('mirofish.response_pool')

# Filler phrases that can be inserted or removed
_FILLERS = [
    "honestly, ", "tbh, ", "I think ", "ngl, ", "just saying, ",
    "for real though, ", "look, ", "seriously, ", "can't believe ",
    "important: ", "worth noting: ", "reminder: ",
]

# Opening swaps — each group contains interchangeable openers
_OPENER_GROUPS = [
    ["Breaking:", "Update:", "Alert:", "Just in:"],
    ["Critical update:", "Important:", "PSA:", "Heads up:"],
    ["This is huge.", "This matters.", "Pay attention.", "Don't ignore this."],
]

# Common word synonyms for swap
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

# Hashtag variations
_HASHTAG_SETS = [
    ["#StaySafe", "#PublicHealth", "#HealthAlert"],
    ["#Breaking", "#News", "#Update"],
    ["#Awareness", "#SpreadTheWord", "#ShareThis"],
    ["#Community", "#Together", "#Support"],
]


class ResponsePool:
    """Pool of recent LLM-generated posts per archetype for reuse with variation."""

    def __init__(
        self,
        reuse_probability: float = 0.3,
        max_pool_size: int = 8,
        min_content_length: int = 30,
    ):
        self._pool: Dict[str, List[str]] = defaultdict(list)
        self._reuse_prob = reuse_probability
        self._max_pool_size = max_pool_size
        self._min_content_length = min_content_length
        self._reuse_count = 0
        self._miss_count = 0

    def add(self, archetype: str, content: str):
        """Add an LLM-generated response to the pool."""
        if len(content.strip()) < self._min_content_length:
            return
        pool = self._pool[archetype]
        pool.append(content.strip())
        if len(pool) > self._max_pool_size:
            pool.pop(0)

    def try_reuse(self, archetype: str) -> Optional[str]:
        """Try to reuse a pooled response with text variation.

        Returns varied text on success, None if pool is empty or
        the random check doesn't pass.
        """
        pool = self._pool.get(archetype, [])
        if not pool:
            return None

        if random.random() > self._reuse_prob:
            self._miss_count += 1
            return None

        base = random.choice(pool)
        varied = vary_text(base)

        # Don't reuse if variation is too similar to the original
        if _similarity(base, varied) > 0.85:
            varied = vary_text(base)

        self._reuse_count += 1
        logger.debug(f"Reusing pooled {archetype} response ({len(base)} -> {len(varied)} chars)")
        return varied

    @property
    def stats(self) -> Dict[str, int]:
        total_pooled = sum(len(v) for v in self._pool.values())
        return {
            "pooled": total_pooled,
            "reused": self._reuse_count,
            "skipped": self._miss_count,
        }


def vary_text(text: str) -> str:
    """Apply lightweight string transformations to make text read as written by a different person."""
    sentences = _split_sentences(text)

    transforms = [
        _swap_opener,
        _shuffle_middle_sentences,
        _swap_synonyms,
        _swap_synonyms,  # double chance — synonyms are the most effective transform
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
    """Split text into sentences, preserving hashtags as a separate final element."""
    # Extract trailing hashtags
    hashtag_match = re.search(r'((?:\s*#\w+)+)\s*$', text)
    hashtags = hashtag_match.group(0).strip() if hashtag_match else ""
    body = text[:hashtag_match.start()].strip() if hashtag_match else text

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', body)
    sentences = [s for s in sentences if s.strip()]

    if hashtags:
        sentences.append(hashtags)

    return sentences


def _swap_opener(sentences: List[str]) -> List[str]:
    """Replace opening phrase with a variation."""
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
    """Shuffle middle sentences (keep first and last in place)."""
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
    """Swap 1-2 common words with synonyms."""
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
                # Preserve original capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                # Preserve trailing punctuation
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
    """Vary sentence-ending punctuation."""
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
    """Add or remove a filler phrase from the first sentence."""
    if not sentences:
        return sentences

    first = sentences[0]

    # Try to remove an existing filler
    for filler in _FILLERS:
        if first.lower().startswith(filler.lower()):
            sentences[0] = first[len(filler):].strip()
            if sentences[0]:
                sentences[0] = sentences[0][0].upper() + sentences[0][1:]
            return sentences

    # Or add one (30% chance)
    if random.random() < 0.3:
        filler = random.choice(_FILLERS)
        # Don't lowercase acronyms (all-caps words like NHS, UKHSA)
        if first[:2].isupper():
            sentences[0] = filler + first
        else:
            sentences[0] = filler + first[0].lower() + first[1:]

    return sentences


def _add_reaction_prefix(sentences: List[str]) -> List[str]:
    """Add a conversational reaction prefix to make the post feel personal."""
    if not sentences or sentences[0].startswith("#"):
        return sentences

    prefixes = [
        "Wow, ", "This is concerning — ", "Just saw this: ",
        "Can't ignore this — ", "Everyone needs to know: ",
        "Sharing this — ", "Seriously, ", "So ",
        "This just in: ", "FYI: ", "Whoa, ",
    ]

    # Only add if no filler/opener already present
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
    """Add, remove, or swap hashtags."""
    if not sentences:
        return sentences

    # Find hashtag element
    last = sentences[-1]
    if last.startswith("#"):
        if random.random() < 0.4:
            # Swap with a different set
            new_set = random.choice(_HASHTAG_SETS)
            count = min(len(last.split()), len(new_set))
            sentences[-1] = " ".join(random.sample(new_set, count))
        elif random.random() < 0.3:
            # Remove hashtags entirely
            sentences = sentences[:-1]
    else:
        # Maybe add hashtags (20% chance)
        if random.random() < 0.2:
            tag_set = random.choice(_HASHTAG_SETS)
            count = random.randint(1, 2)
            sentences.append(" ".join(random.sample(tag_set, count)))

    return sentences


def _similarity(a: str, b: str) -> float:
    """Quick word-overlap similarity check."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / max(len(words_a), len(words_b))
