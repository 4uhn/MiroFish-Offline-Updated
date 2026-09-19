"""
System One Action Router — fast action selection without LLM calls.

Inspired by TypeSafe Jev's "System One" architecture: separate the
structured decision (what action to take) from text generation (what to write).
Non-text actions (like, follow, repost, do_nothing) are resolved instantly
via archetype-weighted sampling, bypassing the LLM entirely.

Only actions that require text generation (CREATE_POST, CREATE_COMMENT,
QUOTE_POST) are forwarded to the LLM (System Two).
"""

import random
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from oasis import ActionType, LLMAction, ManualAction

try:
    from .agent_archetypes import (
        BehavioralArchetype,
        ARCHETYPE_ACTION_WEIGHTS,
        TWITTER_ACTION_MAPPING,
        REDDIT_ACTION_MAPPING,
    )
except ImportError:
    from agent_archetypes import (
        BehavioralArchetype,
        ARCHETYPE_ACTION_WEIGHTS,
        TWITTER_ACTION_MAPPING,
        REDDIT_ACTION_MAPPING,
    )

logger = logging.getLogger('mirofish.system_one')

ACTIONS_REQUIRING_TEXT = {"CREATE_POST", "CREATE_COMMENT", "QUOTE_POST"}

ACTION_TYPE_MAP = {at.name.upper(): at for at in ActionType}


def _resolve_action_type(name: str) -> ActionType:
    """Convert action name string to OASIS ActionType enum."""
    key = name.upper()
    if key in ACTION_TYPE_MAP:
        return ACTION_TYPE_MAP[key]
    return ActionType.DO_NOTHING


def route_agent_action(
    agent,
    agent_id: int,
    archetype: str,
    platform: str,
    available_actions: List[str],
    feed_posts: Optional[List[Dict[str, Any]]] = None,
    agent_graph=None,
) -> Union[LLMAction, ManualAction]:
    """Decide an agent's action using System One (fast path) or System Two (LLM).

    Returns ManualAction for non-text actions (instant, no LLM call),
    or LLMAction for text-generating actions (CREATE_POST, etc.).
    """
    try:
        archetype_enum = BehavioralArchetype(archetype)
    except ValueError:
        archetype_enum = BehavioralArchetype.CONTRIBUTOR

    action_name = _sample_action(archetype_enum, platform, available_actions)

    if action_name in ACTIONS_REQUIRING_TEXT:
        logger.debug(f"Agent {agent_id} -> System Two (LLM): {action_name}")
        return LLMAction()

    logger.debug(f"Agent {agent_id} -> System One (fast): {action_name}")
    action_args = _build_action_args(
        action_name, agent_id, feed_posts, agent_graph
    )

    if action_args is None:
        return ManualAction(
            action_type=ActionType.DO_NOTHING,
            action_args={},
        )

    return ManualAction(
        action_type=_resolve_action_type(action_name),
        action_args=action_args,
    )


def _sample_action(
    archetype: BehavioralArchetype,
    platform: str,
    available_actions: List[str],
) -> str:
    """Sample action from archetype distribution, constrained to available actions."""
    weights = ARCHETYPE_ACTION_WEIGHTS[archetype].copy()

    mapping = TWITTER_ACTION_MAPPING if platform == "twitter" else REDDIT_ACTION_MAPPING

    platform_weights: Dict[str, float] = {}
    for generic_action, weight in weights.items():
        platform_action = mapping.get(generic_action, generic_action)
        platform_weights[platform_action] = (
            platform_weights.get(platform_action, 0) + weight
        )

    platform_weights = {
        k: v for k, v in platform_weights.items() if k in available_actions
    }

    if not platform_weights:
        return "DO_NOTHING"

    actions = list(platform_weights.keys())
    probs = list(platform_weights.values())
    total = sum(probs)
    probs = [p / total for p in probs]

    return random.choices(actions, weights=probs, k=1)[0]


def _build_action_args(
    action_name: str,
    agent_id: int,
    feed_posts: Optional[List[Dict[str, Any]]],
    agent_graph=None,
) -> Optional[Dict[str, Any]]:
    """Build the action_args dict for a ManualAction.

    Returns None if the action can't be executed (e.g. no posts to like).
    """
    if action_name == "DO_NOTHING":
        return {}

    if action_name in ("LIKE_POST", "DISLIKE_POST"):
        post = _pick_random_post(feed_posts, agent_id)
        if not post:
            return None
        return {"post_id": post["post_id"]}

    if action_name == "LIKE_COMMENT":
        comment = _pick_random_comment(feed_posts)
        if not comment:
            return None
        return {"comment_id": comment["comment_id"]}

    if action_name == "DISLIKE_COMMENT":
        comment = _pick_random_comment(feed_posts)
        if not comment:
            return None
        return {"comment_id": comment["comment_id"]}

    if action_name == "REPOST":
        post = _pick_random_post(feed_posts, agent_id)
        if not post:
            return None
        return {"post_id": post["post_id"]}

    if action_name == "FOLLOW":
        target_id = _pick_follow_target(feed_posts, agent_id, agent_graph)
        if target_id is None:
            return None
        return {"followee_id": target_id}

    return {}


def _pick_random_post(
    feed_posts: Optional[List[Dict[str, Any]]],
    exclude_agent_id: int,
) -> Optional[Dict[str, Any]]:
    """Pick a random post from the feed, excluding the agent's own posts."""
    if not feed_posts:
        return None
    candidates = [
        p for p in feed_posts
        if p.get("user_id") != exclude_agent_id and p.get("post_id") is not None
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _pick_random_comment(
    feed_posts: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Pick a random comment from posts in the feed."""
    if not feed_posts:
        return None
    comments = []
    for post in feed_posts:
        for c in post.get("comments", []):
            if c.get("comment_id") is not None:
                comments.append(c)
    if not comments:
        return None
    return random.choice(comments)


def _pick_follow_target(
    feed_posts: Optional[List[Dict[str, Any]]],
    exclude_agent_id: int,
    agent_graph=None,
) -> Optional[int]:
    """Pick a user to follow from the feed."""
    if not feed_posts:
        return None
    user_ids = {
        p["user_id"] for p in feed_posts
        if p.get("user_id") is not None and p["user_id"] != exclude_agent_id
    }
    if not user_ids:
        return None
    return random.choice(list(user_ids))


async def fetch_agent_feed(agent) -> List[Dict[str, Any]]:
    """Fetch the agent's current feed/timeline from the platform.

    Returns a list of post dicts with post_id, user_id, content, comments, etc.
    """
    try:
        result = await agent.env.action.refresh()
        if result.get("success") and result.get("posts"):
            return result["posts"]
    except Exception as e:
        logger.warning(f"Failed to fetch feed for agent: {e}")
    return []


def build_archetype_lookup(config: Dict[str, Any]) -> Dict[int, str]:
    """Build agent_id -> archetype mapping from simulation config."""
    lookup = {}
    for agent_cfg in config.get("agent_configs", []):
        agent_id = agent_cfg.get("agent_id", -1)
        lookup[agent_id] = agent_cfg.get("archetype", "contributor")
    return lookup
