"""
Entity name normalisation and alias matching.

NER on a long seed document names the same organisation several ways:
"Environment Agency", "Environmental Agency", "EnvironmentalAgency";
"Drinking Water Inspectorate (DWI)" and "DWI"; "NHS Bristol ICB" and
"NHS Bristol, North Somerset and South Gloucestershire ICB". These helpers
decide when two names refer to the same entity, so graph nodes and agent
selection are not split across aliases.
"""

import re
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Set

_PAREN_RE = re.compile(r"\(([^)]*)\)")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_STOPWORDS = {"the", "of", "and", "for", "&", "a", "an", "in", "at", "on"}

# Extra words that turn an organisation into a group of people
# ("Bristol Water" vs "Bristol Water customers" are different entities).
GROUP_NOUNS = {
    "customers", "residents", "staff", "users", "members", "employees", "workers",
    "customer", "resident", "user", "member", "employee", "worker", "patient",
    "family", "household",
    "patients", "families", "households", "people", "community", "communities",
    "campaigners", "supporters", "critics", "officials", "spokesperson", "ceo",
    "chair", "director", "site", "sites", "report", "statement",
}

# Legal-form words that can end a name without changing what it refers to.
CORPORATE_SUFFIXES = {"ltd", "limited", "plc", "inc", "llc", "llp", "co", "corp", "corporation"}

# Compact keys this similar are treated as spelling variants
# ("environmentagency" vs "environmentalagency" = 0.94).
_FUZZY_THRESHOLD = 0.9


def name_tokens(name: str) -> List[str]:
    """Lowercase word tokens with parentheticals, punctuation and stopwords removed."""
    text = _PAREN_RE.sub(" ", name or "")
    text = _CAMEL_RE.sub(" ", text).lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return [t for t in text.split() if t not in _STOPWORDS]


def compact_key(name: str) -> str:
    """Canonical key for exact alias comparison ("Environment  Agency" -> "environmentagency")."""
    return "".join(name_tokens(name))


def acronyms(name: str) -> Set[str]:
    """Acronyms a name is known by: explicit "(DWI)" parentheticals, all-caps names, and initials."""
    found: Set[str] = set()
    for inner in _PAREN_RE.findall(name or ""):
        inner = inner.strip()
        if inner.isupper() and 2 <= len(inner) <= 6:
            found.add(inner.lower())
    stripped = _PAREN_RE.sub("", name or "").strip()
    if stripped.isupper() and " " not in stripped and 2 <= len(stripped) <= 6:
        found.add(stripped.lower())
    tokens = name_tokens(name)
    if len(tokens) >= 2:
        found.add("".join(t[0] for t in tokens))
        # All-caps words keep every letter: "UK Health Security Agency" -> "ukhsa"
        words = [w for w in re.findall(r"[A-Za-z]+", stripped) if w.lower() not in _STOPWORDS]
        found.add("".join(w.lower() if w.isupper() else w[0].lower() for w in words))
    return found


def same_entity(a: str, b: str) -> bool:
    """True when two entity names are aliases of the same real-world entity."""
    ka, kb = compact_key(a), compact_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    if SequenceMatcher(None, ka, kb).ratio() >= _FUZZY_THRESHOLD:
        return True

    # "DWI" vs "Drinking Water Inspectorate (DWI)"
    if ka in acronyms(b) or kb in acronyms(a):
        return True

    # "NHS Bristol ICB" vs "NHS Bristol, North Somerset and South Gloucestershire ICB".
    # The shorter name needs two or more tokens so "Bristol" does not swallow "Bristol Water".
    ta, tb = name_tokens(a), name_tokens(b)
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(short) >= 2 and set(short) <= set(long_) and short[0] == long_[0]:
        extra = set(long_) - set(short)
        if extra & GROUP_NOUNS:
            return False
        # The last word names what the entity is, so it must survive: in Run 10
        # "Eastville Chemical Works Legacy Group" (campaigners) merged into
        # "Eastville Chemical Works" (the polluter). Only a corporate suffix
        # may follow ("Persimmon Homes Ltd").
        return short[-1] == long_[-1] or extra <= CORPORATE_SUFFIXES
    return False


def find_alias(name: str, candidates: Iterable[str]) -> Optional[str]:
    """Return the first candidate that is an alias of name, or None."""
    for c in candidates:
        if same_entity(name, c):
            return c
    return None


def is_mentioned(name: str, text: str) -> bool:
    """True when text names the entity: its full name, a spelling variant,
    its acronym ("the DWI"), or a leading part of two or more words ("NHS Bristol"
    for "NHS Bristol ICB", "the Secretary of State" for "Secretary of State for
    Environment").

    A match followed by a group noun names something else ("the former
    Eastville Chemical Works site" is a place, not the works or its
    campaigners), as in same_entity.
    """
    tokens = name_tokens(name)
    if not tokens:
        return False
    # Acronyms count only when the text writes them in capitals ("the DWI")
    caps = {w.lower() for w in re.findall(r"\b[A-Z]{2,6}\b", text or "")}
    if caps & acronyms(name):
        return True
    # Unlike names, text keeps its parentheticals: a prompt may list the
    # institutions in brackets.
    words = name_tokens((text or "").replace("(", " ").replace(")", " "))
    n = len(tokens)
    key = "".join(tokens)
    for i in range(len(words)):
        # Longest leading part of the name that appears here
        k = 0
        while k < n and i + k < len(words) and words[i + k] == tokens[k]:
            k += 1
        if k == n or k >= 2:
            following = words[i + k] if i + k < len(words) else ""
            if following not in GROUP_NOUNS:
                return True
        # Spelling variants of the whole name ("Environmental Agency")
        if n >= 2 and SequenceMatcher(None, "".join(words[i:i + n]), key).ratio() >= _FUZZY_THRESHOLD:
            following = words[i + n] if i + n < len(words) else ""
            if following not in GROUP_NOUNS:
                return True
    return False
