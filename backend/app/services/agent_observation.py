"""
What a simulation agent sees, and what it may write.

Two patches to OASIS, installed once per simulation process:

Compact observation
    OASIS shows an agent its feed as json.dumps(posts, indent=4): every post
    with every comment, each comment carrying comment_id, post_id, user_id,
    created_at and score on separate indented lines. The largest Run 10 Reddit
    feed (five posts with 20/11/8/7/1 comments) measured 7,987 tokens on
    qwen3:8b, and Ollama cuts any prompt over num_ctx=8192 to its last ~4K
    tokens, which drops the system prompt and the persona. The JSON also gave
    agents the wall-clock created_at (Reddit) and only numeric user ids, so
    no one could tell who had said what.

    render_posts writes the same feed as text under a fixed character budget:
    author names, scores, the post and its most relevant comments (the top
    comment and the latest ones), with a count of the rest. The five-post
    worst case fits in ~4,000 characters (~1K tokens).

Copy guard
    qwen3:8b copies text from its prompt. In Run 10 the Reddit comment "I'm a
    parent, and I'm terrified..." appeared four times under different agents,
    some of them not parents. A create_post, create_comment or quote_post whose
    text is mostly (by word 4-gram containment) an existing post or comment is
    turned into the endorsement it amounts to: a like on Reddit, a repost on
    Twitter. Repeating one's own text becomes do_nothing. The guard is armed
    only inside an agent's LLM turn, so seed posts and scheduled events, which
    go through the same tool methods, are never converted.

No package-relative imports: the simulation scripts import this module by
bare name.
"""

import contextvars
import functools
import logging
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("mirofish.agent_observation")

# ---------------------------------------------------------------------------
# Compact observation
# ---------------------------------------------------------------------------

OBS_MAX_CHARS = 4000
POST_CHARS = 320
COMMENT_CHARS = 160
COMMENTS_PER_POST = 3

_USER_REF_RE = re.compile(r"\bUser (\d+)\b")


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _score(item: Dict[str, Any]) -> int:
    if isinstance(item.get("score"), int):
        return item["score"]
    return int(item.get("num_likes") or 0) - int(item.get("num_dislikes") or 0)


def _pick_comments(comments: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """The top-scored comment (if anyone upvoted it) and the latest ones, oldest first."""
    if k <= 0 or not comments:
        return []
    ordered = sorted(comments, key=lambda c: c.get("comment_id", 0))
    chosen = ordered[-k:]
    top = max(ordered, key=_score)
    if _score(top) > 0 and top not in chosen:
        chosen = [top] + ordered[-(k - 1):] if k > 1 else [top]
    return sorted(chosen, key=lambda c: c.get("comment_id", 0))


def render_posts(
    posts: List[Dict[str, Any]],
    viewer_id: int,
    names: Dict[int, str],
    max_chars: int = OBS_MAX_CHARS,
) -> str:
    """The feed as short text blocks, within max_chars.

    Detail is removed in steps until the feed fits: fewer comments per post,
    then shorter posts.
    """
    def who(uid) -> str:
        return "you" if uid == viewer_id else names.get(uid, f"user {uid}")

    def render(k: int, post_chars: int) -> str:
        blocks = []
        for p in posts:
            content = _USER_REF_RE.sub(lambda m: names.get(int(m.group(1)), m.group(0)),
                                       p.get("content") or "")
            comments = p.get("comments") or []
            head = f"[post {p.get('post_id')}] {who(p.get('user_id'))} (score {_score(p)}"
            head += f", {len(comments)} comments)" if comments else ")"
            lines = [f'{head}: "{_clip(content, post_chars)}"']
            shown = _pick_comments(comments, k)
            for c in shown:
                lines.append(f'  - [comment {c.get("comment_id")}] {who(c.get("user_id"))} '
                             f'(score {_score(c)}): "{_clip(c.get("content"), COMMENT_CHARS)}"')
            if len(comments) > len(shown):
                lines.append(f"  (+{len(comments) - len(shown)} more comments)")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    text = ""
    for post_chars in (POST_CHARS, POST_CHARS // 2):
        for k in range(COMMENTS_PER_POST, -1, -1):
            text = render(k, post_chars)
            if len(text) <= max_chars:
                return text
    return text[:max_chars].rstrip() + "…"


def install_compact_observation(names: Dict[int, str]) -> None:
    """Make every OASIS agent see its feed as render_posts text.

    names maps agent id to display name. OASIS user ids equal agent ids, and
    both platforms in a parallel run share the same agents, so one mapping
    serves both.
    """
    from oasis.social_agent.agent_environment import SocialEnvironment

    async def get_posts_env(self) -> str:
        posts = await self.action.refresh()
        if not posts.get("success") or not posts.get("posts"):
            return "After refreshing, there are no existing posts."
        feed = render_posts(posts["posts"], self.action.agent_id, names)
        return f"After refreshing, you see these posts:\n{feed}\n"

    SocialEnvironment.get_posts_env = get_posts_env


# ---------------------------------------------------------------------------
# Copy guard
# ---------------------------------------------------------------------------

# Share of the new text's word 4-grams that already appear in one existing
# text. Quoting a phrase from a post stays well below this; a paraphrase that
# keeps whole sentences does not.
COPY_CONTAINMENT = 0.6
_SHINGLE = 4
# Texts shorter than this many words are compared whole ("Same here" is not a copy).
_MIN_WORDS = 8

_IN_LLM_TURN: contextvars.ContextVar[bool] = contextvars.ContextVar("in_llm_turn", default=False)


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _shingles(words: List[str]) -> Set[Tuple[str, ...]]:
    return {tuple(words[i:i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)}


@dataclass
class _Text:
    kind: str       # "post" or "comment"
    item_id: int
    post_id: int    # the post itself, or the post a comment belongs to
    author: int
    shingles: Set[Tuple[str, ...]]


class CopyGuard:
    """Detects posts and comments that copy an existing text on one platform."""

    def __init__(self, db_path: str, platform: str, threshold: float = COPY_CONTAINMENT):
        self.db_path = db_path
        self.platform = platform
        self.threshold = threshold
        self._cache: Dict[Tuple[str, int], _Text] = {}
        self.stats: Counter = Counter()

    def _load(self) -> List[_Text]:
        """Every authored text on the platform so far (reposts carry none)."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=10)
        try:
            rows = [("post", pid, pid, uid, q or c) for pid, uid, c, q in conn.execute(
                "SELECT post_id, user_id, content, quote_content FROM post "
                "WHERE original_post_id IS NULL OR quote_content IS NOT NULL")]
            rows += [("comment", cid, pid, uid, c) for cid, pid, uid, c in conn.execute(
                "SELECT comment_id, post_id, user_id, content FROM comment")]
        finally:
            conn.close()
        texts = []
        for kind, item_id, post_id, author, content in rows:
            key = (kind, item_id)
            if key not in self._cache:
                self._cache[key] = _Text(kind, item_id, post_id, author, _shingles(_words(content)))
            texts.append(self._cache[key])
        return texts

    def find_copy(self, text: str) -> Optional[Tuple[_Text, float]]:
        """The existing text that text copies, with its containment, or None."""
        words = _words(text)
        if len(words) < _MIN_WORDS:
            return None
        new = _shingles(words)
        best, best_score = None, 0.0
        for t in self._load():
            if not t.shingles:
                continue
            score = len(new & t.shingles) / len(new)
            if score > best_score:
                best, best_score = t, score
        if best is not None and best_score >= self.threshold:
            return best, best_score
        return None

    async def convert(self, action, source: _Text, score: float, attempted: str, text: str):
        """Perform the endorsement a copy amounts to, instead of the copy."""
        agent_id = action.agent_id
        if source.author == agent_id:
            self.stats["self_repeats"] += 1
            result = await action.do_nothing()
            outcome = "do_nothing"
        elif source.kind == "comment" and self.platform == "reddit":
            result = await action.like_comment(comment_id=source.item_id)
            outcome = f"like_comment {source.item_id}"
        elif self.platform == "twitter":
            result = await action.repost(post_id=source.post_id)
            outcome = f"repost {source.post_id}"
        else:
            result = await action.like_post(post_id=source.post_id)
            outcome = f"like_post {source.post_id}"
        self.stats["copies_converted"] += 1
        logger.info(
            f"[{self.platform}] copy guard: agent {agent_id} {attempted} copied "
            f"{source.kind} {source.item_id} by {source.author} ({score:.0%}) -> {outcome}: "
            f"{_clip(text, 90)!r}"
        )
        return result


def _guarded(method_name: str, text_arg: str):
    """Wrap SocialAction.<method_name> so an LLM-written copy is converted."""
    from oasis.social_agent.agent_action import SocialAction

    original = getattr(SocialAction, method_name)

    @functools.wraps(original)
    async def wrapper(self, *args, **kwargs):
        guard: Optional[CopyGuard] = getattr(self, "_copy_guard", None)
        if guard is not None and _IN_LLM_TURN.get():
            bound = dict(zip(original.__code__.co_varnames[1:], args), **kwargs)
            text = bound.get(text_arg) or ""
            try:
                match = guard.find_copy(text)
            except sqlite3.Error as e:
                logger.warning(f"copy guard could not read {guard.db_path}: {e}")
                match = None
            if match:
                return await guard.convert(self, match[0], match[1], method_name, text)
        return await original(self, *args, **kwargs)

    setattr(SocialAction, method_name, wrapper)


_installed = False


def install_copy_guard() -> None:
    """Wrap the text-writing actions and mark each agent's LLM turn. Idempotent.

    Must run after any other SocialAction patch (the create_comment kwargs
    shim) so the guard sees the final signature, and before agents are built,
    since CAMEL builds each agent's tool list from these methods.
    """
    global _installed
    if _installed:
        return
    from oasis.social_agent.agent import SocialAgent

    _guarded("create_post", "content")
    _guarded("create_comment", "content")
    _guarded("quote_post", "quote_content")

    original_turn = SocialAgent.perform_action_by_llm

    @functools.wraps(original_turn)
    async def perform_action_by_llm(self, *args, **kwargs):
        token = _IN_LLM_TURN.set(True)
        try:
            return await original_turn(self, *args, **kwargs)
        finally:
            _IN_LLM_TURN.reset(token)

    SocialAgent.perform_action_by_llm = perform_action_by_llm
    _installed = True


def attach_copy_guard(agent_graph, guard: CopyGuard) -> None:
    """Give every agent on one platform that platform's guard."""
    for _, agent in agent_graph.get_agents():
        agent.env.action._copy_guard = guard
