"""
Round-context injection for simulation agents.

Before an agent makes a System Two (LLM) decision, we inject a SINGLE user
message that carries what a small local model needs to produce varied,
grounded, conversational output instead of repetitive boilerplate:

  0. NOW / PUBLICLY KNOWN    — the scenario date and time, and the facts public
                               by then (scenario.known_facts_block), so agents
                               do not invent dates and figures
  1. YOUR RECENT ACTIVITY   — what this agent already did (continuity)
  2. WHAT OTHERS ARE SAYING  — a digest of *other* agents' recent posts, so the
                               agent reacts to and builds on the live
                               conversation instead of posting into a vacuum
                               (this is the main fix for convergent, repetitive
                               posts — every agent otherwise only sees the seed
                               event and independently rephrases it)
  3. STYLE guidance          — an explicit instruction not to echo stock openers
                               or repeat points already made in the feed

The whole block is injected as one user message tagged with
``ROUND_CONTEXT_MARKER``. Re-injecting surgically removes the previous tagged
block first, so per-round context does not grow unbounded across a run.
"""

import os
from typing import Any, Dict, List, Optional

# Every injected block starts with this line. Used both to identify the block
# for surgical replacement and to keep the three ingredients under one tag.
ROUND_CONTEXT_MARKER = "[ROUND CONTEXT]"

_STYLE_GUIDANCE = (
    "STYLE: write in your own voice. Do not introduce yourself or state your job, "
    "and do not open with stock phrases (\"As a...\", \"I can't believe...\", "
    "\"This is a wake-up call...\", \"Stay safe everyone\"). Do not repeat what "
    "others already said: react to it, answer it, or add something new. "
    "Your own life, feelings and experiences are yours to describe, but every "
    "number, date, time, place, phone number, link or official name you mention "
    "must come from the facts or posts above. If you are not sure of a detail, "
    "leave it out or say you don't know."
)

# The Reddit observation (agent_observation.render_posts) already lists each
# post with its comments, so the digest keeps only the instruction to reply.
# A generic "react to this" header pulled Reddit agents toward new top-level
# posts (A/B: 12/12 comments without round context, 8/12 with it), which
# leaves threads without discussion.
_REDDIT_THREAD_DIRECTIVE = (
    "Join a thread in your feed: reply with create_comment to the post you have "
    "most to say about, using its post id. Only start a new post if none of them fits."
)


def build_feed_digest(
    feed_posts: Optional[List[Dict[str, Any]]],
    exclude_user_id: int,
    max_posts: int = 5,
    max_chars: int = 180,
    platform: str = "twitter",
    names: Optional[Dict[int, str]] = None,
) -> Optional[str]:
    """Format a short digest of what *other* agents recently posted.

    ``feed_posts`` are OASIS refresh() posts: dicts with user_id, content,
    num_likes, etc. The acting agent's own posts are excluded, near-duplicate
    openers are collapsed, and each snippet is truncated to keep the injected
    context small.

    On Twitter an agent's own refresh shows only 2-4 posts, so the digest
    widens what it sees. On Reddit the agent's observation already shows the
    posts, so only the reply directive is returned.
    """
    if not feed_posts:
        return None
    if platform.lower() == "reddit":
        return _REDDIT_THREAD_DIRECTIVE

    names = names or {}
    seen_openers = set()
    lines: List[str] = []
    for p in feed_posts:
        uid = p.get("user_id")
        if uid == exclude_user_id:
            continue
        content = (p.get("content") or "").strip()
        if not content:
            continue
        opener = content[:50].lower()
        if opener in seen_openers:
            continue
        seen_openers.add(opener)

        snippet = " ".join(content.split())
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "…"
        likes = p.get("num_likes")
        tag = f" ({likes} likes)" if isinstance(likes, int) and likes > 0 else ""
        lines.append(f'- {names.get(uid, f"user {uid}")}{tag}: "{snippet}"')
        if len(lines) >= max_posts:
            break

    if not lines:
        return None
    return "WHAT OTHERS ARE SAYING RIGHT NOW (react to this, don't echo it):\n" + "\n".join(lines)


# Run 9: 16 of ~40 institutional posts were "We are closely monitoring the
# situation" boilerplate. Run 10 replaced that with "say one concrete thing: a
# specific action with a time or place, a number", and 31 of 59 institutional
# texts then invented dates ("October 15th, 2026"), places ("River Wye") and
# links. Institutions now get concreteness from the fact sheet instead.
_INSTITUTION_STYLE = (
    "You post for an organisation: write as \"we\". Say what your organisation "
    "knows, is doing or advises, using only the facts above. Do not invent dates, "
    "times, figures, locations, phone numbers, links or names. If something is not "
    "known yet, say so plainly. Answer criticism in your feed directly. No generic "
    "reassurance (\"We are closely monitoring the situation\", \"working closely "
    "with partners\", \"we take this seriously\")."
)


def build_round_context(
    memory_summary: Optional[str],
    feed_digest: Optional[str],
    include_style_guidance: bool = True,
    institutional: bool = False,
    facts_block: Optional[str] = None,
) -> Optional[str]:
    """Assemble the combined round-context block, or None if empty.

    facts_block (scenario.known_facts_block) gives the scenario date and time
    and what is publicly known by then. Style guidance is only added when there
    is something to react to, so brand-new agents in an empty world don't get a
    bare instruction with no context.
    """
    parts: List[str] = []
    if facts_block:
        parts.append(facts_block)
    if memory_summary:
        parts.append(memory_summary)
    if feed_digest:
        parts.append(feed_digest)
    if include_style_guidance and parts:
        parts.append(_STYLE_GUIDANCE)
        if institutional:
            parts.append(_INSTITUTION_STYLE)

    if not parts:
        return None
    return ROUND_CONTEXT_MARKER + "\n" + "\n\n".join(parts)


# Past observe/act turns an agent's LLM call replays verbatim. OASIS agents are
# CAMEL ChatAgents whose token_limit defaults to 999,999,999 for Ollama models,
# so without a cap every past feed dump and create_post call is resent each
# round. In Run 8 prompts reached p99 ~7.1K tokens, two exceeded 8192 and were
# cut by Ollama to their last ~4K tokens (dropping the persona), and agents
# copied their own earlier tool calls word for word. The agent's longer-term
# history reaches the model through the memory summary in the round context.
# Replaying even one turn resends a stale feed (Run 10's largest Reddit feed
# alone was ~8K tokens), so the default replays none: the memory summary
# already lists what the agent did.
DEFAULT_HISTORY_TURNS = int(os.environ.get("SIM_AGENT_HISTORY_TURNS", "0"))


def _record_role(record: Dict[str, Any]) -> str:
    role = record.get("role_at_backend")
    return getattr(role, "value", role) or ""


def _record_content(record: Dict[str, Any]) -> str:
    content = (record.get("message") or {}).get("content")
    return content if isinstance(content, str) else ""


def _is_round_context(record: Dict[str, Any]) -> bool:
    return _record_content(record).startswith(ROUND_CONTEXT_MARKER)


def trim_agent_history(mem_list: List[Dict[str, Any]], keep_turns: int) -> None:
    """Keep the leading system prompt and only the last keep_turns turns, in place.

    A turn starts at a user message (the OASIS environment prompt, or an
    interview question) and runs until the next one, so a tool call is never
    separated from its result.
    """
    head = 1 if mem_list and _record_role(mem_list[0]) == "system" else 0
    turn_starts = [
        i for i in range(head, len(mem_list))
        if _record_role(mem_list[i]) == "user" and not _is_round_context(mem_list[i])
    ]
    if len(turn_starts) <= keep_turns:
        return
    cut = turn_starts[-keep_turns] if keep_turns > 0 else len(mem_list)
    del mem_list[head:cut]


def inject_round_context(agent, content: Optional[str], keep_turns: int = DEFAULT_HISTORY_TURNS) -> None:
    """Trim the agent's replayed history and add this round's context block.

    The block is written as a USER message. CAMEL's ScoreBasedContextCreator
    only sends the first SYSTEM record and silently drops later ones, so the
    SYSTEM-role injection used before Run 9 never reached the model.
    The previous round's block (matched by ROUND_CONTEXT_MARKER) is removed
    first, so blocks do not stack. With content=None only the trim runs.
    """
    from camel.messages import BaseMessage
    from camel.types import OpenAIBackendRole

    try:
        mem_list = agent.memory._chat_history_block.storage.memory_list
        mem_list[:] = [r for r in mem_list if not _is_round_context(r)]
        trim_agent_history(mem_list, keep_turns)
    except (AttributeError, TypeError):
        pass

    if content:
        agent.update_memory(
            message=BaseMessage.make_user_message(role_name="User", content=content),
            role=OpenAIBackendRole.USER,
        )
