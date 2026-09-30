"""
Agent Memory Store — lightweight action journal for simulation agents.

Tracks all agent actions (both System One fast-path and System Two LLM)
so that when an agent does get an LLM call, it has context about its
recent behavior. Without this, System One ManualActions are invisible
to the agent's chat history.

Storage: in-memory dict per simulation, with optional SQLite persistence.
"""

import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger('mirofish.agent_memory')


@dataclass
class ActionRecord:
    round_num: int
    action_type: str
    target_id: Optional[int] = None
    content_preview: Optional[str] = None
    via_system_one: bool = False


class AgentMemoryStore:
    """Per-agent action journal for a single simulation run."""

    def __init__(self, db_path: Optional[str] = None, max_history: int = 20):
        self._history: Dict[int, List[ActionRecord]] = defaultdict(list)
        self._max_history = max_history
        self._db_path = db_path

        if db_path:
            self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id INTEGER NOT NULL,
                round_num INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                target_id INTEGER,
                content_preview TEXT,
                via_system_one INTEGER DEFAULT 0
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_memory_agent "
            "ON agent_memory(agent_id)"
        )
        conn.commit()
        conn.close()

    def record(
        self,
        agent_id: int,
        round_num: int,
        action_type: str,
        target_id: Optional[int] = None,
        content_preview: Optional[str] = None,
        via_system_one: bool = False,
    ):
        rec = ActionRecord(
            round_num=round_num,
            action_type=action_type,
            target_id=target_id,
            content_preview=content_preview[:80] if content_preview else None,
            via_system_one=via_system_one,
        )
        history = self._history[agent_id]
        history.append(rec)
        if len(history) > self._max_history:
            history.pop(0)

        if self._db_path:
            self._persist(agent_id, rec)

    def _persist(self, agent_id: int, rec: ActionRecord):
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO agent_memory "
                "(agent_id, round_num, action_type, target_id, content_preview, via_system_one) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (agent_id, rec.round_num, rec.action_type,
                 rec.target_id, rec.content_preview, int(rec.via_system_one)),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to persist agent memory: {e}")

    def get_recent(self, agent_id: int, n: int = 8) -> List[ActionRecord]:
        return self._history[agent_id][-n:]

    def get_summary(self, agent_id: int, n: int = 8) -> Optional[str]:
        """Natural language summary of recent actions, suitable for LLM context injection."""
        recent = self.get_recent(agent_id, n)
        if not recent:
            return None

        lines = []
        for rec in recent:
            line = _format_action(rec)
            if line:
                lines.append(f"- {line}")

        if not lines:
            return None

        return (
            "YOUR RECENT ACTIVITY (actions you took in previous rounds):\n"
            + "\n".join(lines)
            + "\n\nKeep your views consistent with this, but do not repeat or rephrase "
            "a post you already made. If you post again, say something new: react to "
            "a new development, answer someone, or add a detail you have not mentioned."
        )

    def agent_count(self) -> int:
        return len(self._history)

    def total_records(self) -> int:
        return sum(len(v) for v in self._history.values())


def _format_action(rec: ActionRecord) -> Optional[str]:
    """Format a single action record as natural language."""
    t = rec.action_type.upper()

    if t == "DO_NOTHING":
        return None

    if t == "LIKE_POST":
        if rec.content_preview:
            return f'Round {rec.round_num}: Upvoted a post: "{rec.content_preview}..."'
        return f"Round {rec.round_num}: Liked a post" + (
            f" (post #{rec.target_id})" if rec.target_id else ""
        )
    if t == "DISLIKE_POST":
        return f"Round {rec.round_num}: Disliked a post" + (
            f" (post #{rec.target_id})" if rec.target_id else ""
        )
    if t == "LIKE_COMMENT":
        if rec.content_preview:
            return f'Round {rec.round_num}: Upvoted a comment: "{rec.content_preview}..."'
        return f"Round {rec.round_num}: Liked a comment"
    if t == "DISLIKE_COMMENT":
        return f"Round {rec.round_num}: Disliked a comment"
    if t == "FOLLOW":
        return f"Round {rec.round_num}: Followed user #{rec.target_id}"
    if t == "REPOST":
        if rec.content_preview:
            return f'Round {rec.round_num}: Reposted someone else\'s post: "{rec.content_preview}..."'
        return f"Round {rec.round_num}: Reposted a post" + (
            f" (post #{rec.target_id})" if rec.target_id else ""
        )
    if t == "CREATE_POST":
        preview = f': "{rec.content_preview}..."' if rec.content_preview else ""
        return f"Round {rec.round_num}: Created a post{preview}"
    if t == "CREATE_COMMENT":
        preview = f': "{rec.content_preview}..."' if rec.content_preview else ""
        return f"Round {rec.round_num}: Commented on a post{preview}"
    if t == "QUOTE_POST":
        preview = f': "{rec.content_preview}..."' if rec.content_preview else ""
        return f"Round {rec.round_num}: Quoted a post{preview}"

    return f"Round {rec.round_num}: {t}"
