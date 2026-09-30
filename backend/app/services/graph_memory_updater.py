"""
Graph memory update service
Dynamically updates agent activities from simulations into the Neo4j graph.

Replaces zep_graph_memory_updater.py — Zep client replaced by GraphStorage.
"""

import os
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from ..utils.logger import get_logger
from ..storage import GraphStorage

logger = get_logger('mirofish.graph_memory_updater')


@dataclass
class AgentActivity:
    """Agent activity record"""
    platform: str           # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str        # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str

    def to_episode_text(self) -> str:
        """
        Convert an activity to a natural language text description.

        Uses natural language format so the NER extractor can extract entities and relations.
        """
        action_descriptions = {
            "CREATE_POST": self._describe_create_post,
            "LIKE_POST": self._describe_like_post,
            "DISLIKE_POST": self._describe_dislike_post,
            "REPOST": self._describe_repost,
            "QUOTE_POST": self._describe_quote_post,
            "FOLLOW": self._describe_follow,
            "CREATE_COMMENT": self._describe_create_comment,
            "LIKE_COMMENT": self._describe_like_comment,
            "DISLIKE_COMMENT": self._describe_dislike_comment,
            "SEARCH_POSTS": self._describe_search,
            "SEARCH_USER": self._describe_search_user,
            "MUTE": self._describe_mute,
        }

        describe_func = action_descriptions.get(self.action_type, self._describe_generic)
        description = describe_func()

        return f"{self.agent_name}: {description}"

    def grounding(self) -> Dict[str, Any]:
        """What this activity lets the graph record about its agent.

        own_words is the text the agent wrote; acted_on names the users whose
        post, comment or account the agent acted on. Someone else's post
        quoted inside the line is excluded: in Run 12 NER read Sophie Evans's
        post inside "Ryan Thompson: reposted Sophie Evans's post" as Ryan
        blaming the Environment Agency, which neither of them named.
        """
        args = self.action_args
        own_words = {
            "CREATE_POST": args.get("content", ""),
            "CREATE_COMMENT": args.get("content", ""),
            "QUOTE_POST": args.get("quote_content", "") or args.get("content", ""),
        }.get(self.action_type, "")
        acted_on = [
            args.get(key, "")
            for key in ("post_author_name", "original_author_name",
                        "comment_author_name", "target_user_name")
        ]
        return {
            "actor": self.agent_name,
            "own_words": own_words,
            "acted_on": [name for name in acted_on if name],
        }

    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        if content:
            return f'posted: "{content}"'
        return "posted"

    def _describe_like_post(self) -> str:
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        if post_content and post_author:
            return f'liked {post_author}\'s post: "{post_content}"'
        elif post_content:
            return f'liked a post: "{post_content}"'
        elif post_author:
            return f"liked a post by {post_author}"
        return "liked a post"

    def _describe_dislike_post(self) -> str:
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        if post_content and post_author:
            return f'downvoted {post_author}\'s post: "{post_content}"'
        elif post_content:
            return f'downvoted a post: "{post_content}"'
        elif post_author:
            return f"downvoted a post by {post_author}"
        return "downvoted a post"

    def _describe_repost(self) -> str:
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        if original_content and original_author:
            return f'reposted {original_author}\'s post: "{original_content}"'
        elif original_content:
            return f'reposted a post: "{original_content}"'
        elif original_author:
            return f"reposted a post by {original_author}"
        return "reposted a post"

    def _describe_quote_post(self) -> str:
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        quote_content = self.action_args.get("quote_content", "") or self.action_args.get("content", "")
        base = ""
        if original_content and original_author:
            base = f'quoted {original_author}\'s post "{original_content}"'
        elif original_content:
            base = f'quoted a post "{original_content}"'
        elif original_author:
            base = f"quoted a post by {original_author}"
        else:
            base = "quoted a post"
        if quote_content:
            base += f', and commented: "{quote_content}"'
        return base

    def _describe_follow(self) -> str:
        target_user_name = self.action_args.get("target_user_name", "")
        if target_user_name:
            return f'followed user "{target_user_name}"'
        return "followed a user"

    def _describe_create_comment(self) -> str:
        content = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        if content:
            if post_content and post_author:
                return f'commented on {post_author}\'s post "{post_content}": "{content}"'
            elif post_content:
                return f'commented on post "{post_content}": "{content}"'
            elif post_author:
                return f'commented on {post_author}\'s post: "{content}"'
            return f'commented: "{content}"'
        return "posted a comment"

    def _describe_like_comment(self) -> str:
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        if comment_content and comment_author:
            return f'liked {comment_author}\'s comment: "{comment_content}"'
        elif comment_content:
            return f'liked a comment: "{comment_content}"'
        elif comment_author:
            return f"liked a comment by {comment_author}"
        return "liked a comment"

    def _describe_dislike_comment(self) -> str:
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        if comment_content and comment_author:
            return f'downvoted {comment_author}\'s comment: "{comment_content}"'
        elif comment_content:
            return f'downvoted a comment: "{comment_content}"'
        elif comment_author:
            return f"downvoted a comment by {comment_author}"
        return "downvoted a comment"

    def _describe_search(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f'searched for "{query}"' if query else "performed a search"

    def _describe_search_user(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f'searched for user "{query}"' if query else "searched for a user"

    def _describe_mute(self) -> str:
        target_user_name = self.action_args.get("target_user_name", "")
        if target_user_name:
            return f'muted user "{target_user_name}"'
        return "muted a user"

    def _describe_generic(self) -> str:
        return f"performed {self.action_type} action"


class GraphMemoryUpdater:
    """
    Graph memory updater (via GraphStorage / Neo4j)

    Monitors simulation action log files and updates new agent activities into the graph in real time.
    Groups by platform and sends a batch to the graph every BATCH_SIZE activities.
    """

    BATCH_SIZE = 5

    PLATFORM_DISPLAY_NAMES = {
        'twitter': 'World 1',
        'reddit': 'World 2',
    }

    SEND_INTERVAL = 0.5
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    # Concurrent NER calls once the simulation has ended; match OLLAMA_NUM_PARALLEL.
    DRAIN_WORKERS = int(os.environ.get('GRAPH_MEMORY_DRAIN_WORKERS', '3'))

    def __init__(self, graph_id: str, storage: GraphStorage, simulation_id: Optional[str] = None):
        """
        Initialize the updater.

        Args:
            graph_id: Graph ID
            storage: GraphStorage instance (injected)
            simulation_id: Run the activities belong to; tagged on what they write
        """
        self.graph_id = graph_id
        self.storage = storage
        self.simulation_id = simulation_id

        self._activity_queue: Queue = Queue()

        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            'twitter': [],
            'reddit': [],
        }
        self._buffer_lock = threading.Lock()

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self._stats_lock = threading.Lock()
        self._total_activities = 0
        self._total_sent = 0
        self._total_items_sent = 0
        self._failed_count = 0
        self._failed_items = 0
        self._skipped_count = 0

        logger.info(f"GraphMemoryUpdater initialized: graph_id={graph_id}, batch_size={self.BATCH_SIZE}")

    def _get_platform_display_name(self, platform: str) -> str:
        return self.PLATFORM_DISPLAY_NAMES.get(platform.lower(), platform)

    def start(self):
        """Start the background worker thread."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"GraphMemoryUpdater-{self.graph_id[:8]}"
        )
        self._worker_thread.start()
        logger.info(f"GraphMemoryUpdater started: graph_id={self.graph_id}")

    def stop(self, timeout: Optional[float] = 10):
        """Stop the worker after it has sent everything already queued.

        The worker drains the queue and the partial platform buffers itself
        (see _drain_remaining). Pass timeout=None to wait for the full drain;
        with a timeout, a worker that is still busy finishes on its own thread.
        """
        self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning(
                f"GraphMemoryUpdater still draining after {timeout}s: graph_id={self.graph_id}, "
                f"pending={self.pending_count()}"
            )
            return

        logger.info(f"GraphMemoryUpdater stopped: graph_id={self.graph_id}, "
                     f"total_activities={self._total_activities}, "
                     f"batches_sent={self._total_sent}, "
                     f"items_sent={self._total_items_sent}, "
                     f"failed={self._failed_count}, "
                     f"skipped={self._skipped_count}")

    def pending_count(self) -> int:
        """Activities not yet written to (or given up on for) the graph, including in-flight batches."""
        with self._stats_lock:
            return self._total_activities - self._total_items_sent - self._failed_items

    def add_activity(self, activity: AgentActivity):
        """Add an agent activity to the queue."""
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return

        with self._stats_lock:
            self._total_activities += 1
        self._activity_queue.put(activity)
        logger.debug(f"Added activity to queue: {activity.agent_name} - {activity.action_type}")

    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        """Add an activity from dictionary data."""
        if "event_type" in data:
            return

        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )

        self.add_activity(activity)

    def _worker_loop(self):
        """Send full batches one at a time while the simulation runs, then drain the rest."""
        while self._running:
            try:
                try:
                    activity = self._activity_queue.get(timeout=1)
                except Empty:
                    continue

                platform = activity.platform.lower()
                batch = None
                with self._buffer_lock:
                    buffer = self._platform_buffers.setdefault(platform, [])
                    buffer.append(activity)
                    if len(buffer) >= self.BATCH_SIZE:
                        batch = buffer[:self.BATCH_SIZE]
                        del buffer[:self.BATCH_SIZE]

                if batch:
                    self._send_batch_activities(batch, platform)
                    time.sleep(self.SEND_INTERVAL)

            except Exception as e:
                logger.error(f"Worker loop exception: {e}")
                time.sleep(1)

        self._drain_remaining()

    def _drain_remaining(self):
        """Send everything still queued or buffered, DRAIN_WORKERS batches at a time.

        While the simulation runs, one NER call at a time keeps the graph from
        competing harder with the agents for Ollama. Once it has ended, Ollama
        is otherwise idle, and a serial drain left 2 of its 3 slots unused
        while the report waited (Run 9: ~25 batches, ~12 min). Neo4jStorage
        serialises the graph writes, so only the NER calls overlap.
        """
        while True:
            with self._buffer_lock:
                while True:
                    try:
                        activity = self._activity_queue.get_nowait()
                    except Empty:
                        break
                    self._platform_buffers.setdefault(activity.platform.lower(), []).append(activity)

                batches = []
                for platform, buffer in self._platform_buffers.items():
                    for i in range(0, len(buffer), self.BATCH_SIZE):
                        batches.append((buffer[i:i + self.BATCH_SIZE], platform))
                    buffer.clear()

            if not batches:
                return

            workers = max(1, min(self.DRAIN_WORKERS, len(batches)))
            logger.info(
                f"Draining {sum(len(b) for b, _ in batches)} activities in {len(batches)} batches "
                f"with {workers} parallel NER calls: graph_id={self.graph_id}"
            )
            started = time.time()
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="GraphMemoryDrain") as pool:
                for batch, platform in batches:
                    pool.submit(self._send_batch_activities, batch, platform)
            logger.info(f"Drained {len(batches)} batches in {time.time() - started:.0f}s: graph_id={self.graph_id}")

    def _send_batch_activities(self, activities: List[AgentActivity], platform: str):
        """
        Send a batch of activities to the graph (combined into one text, triggering NER via add_text).
        """
        if not activities:
            return

        episode_texts = [activity.to_episode_text() for activity in activities]
        combined_text = "\n".join(episode_texts)

        for attempt in range(self.MAX_RETRIES):
            try:
                self.storage.add_text(
                    self.graph_id, combined_text, source="simulation",
                    simulation_id=self.simulation_id,
                    grounding=[activity.grounding() for activity in activities],
                )

                with self._stats_lock:
                    self._total_sent += 1
                    self._total_items_sent += len(activities)
                display_name = self._get_platform_display_name(platform)
                logger.info(f"Successfully sent batch of {len(activities)} {display_name} activities to graph {self.graph_id}")
                logger.debug(f"Batch content preview: {combined_text[:200]}...")
                return

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"Failed to send batch to graph (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Failed to send batch to graph after {self.MAX_RETRIES} retries: {e}")
                    with self._stats_lock:
                        self._failed_count += 1
                        self._failed_items += len(activities)

class GraphMemoryManager:
    """
    Manages graph memory updaters for multiple simulations.

    Each simulation can have its own updater instance.
    NOTE: create_updater() requires a GraphStorage instance — must be passed in.
    """

    _updaters: Dict[str, GraphMemoryUpdater] = {}
    _draining: Dict[str, GraphMemoryUpdater] = {}
    _lock = threading.Lock()

    @classmethod
    def create_updater(
        cls, simulation_id: str, graph_id: str, storage: GraphStorage
    ) -> GraphMemoryUpdater:
        """
        Create a graph memory updater for a simulation.

        Args:
            simulation_id: Simulation ID
            graph_id: Graph ID
            storage: GraphStorage instance
        """
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()

            updater = GraphMemoryUpdater(graph_id, storage, simulation_id=simulation_id)
            updater.start()
            cls._updaters[simulation_id] = updater

            logger.info(f"Created graph memory updater: simulation_id={simulation_id}, graph_id={graph_id}")
            return updater

    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[GraphMemoryUpdater]:
        """Get the updater for a simulation."""
        return cls._updaters.get(simulation_id)

    @classmethod
    def finish_updater(cls, simulation_id: str):
        """Drain a finished simulation's updater on a background thread.

        Called as soon as every platform has logged simulation_end. The OASIS
        process stays alive afterwards to answer interviews, so waiting for
        process exit (the old trigger) meant the last activities were never
        written and the report read a partial graph.
        """
        with cls._lock:
            updater = cls._updaters.pop(simulation_id, None)
            if updater is None:
                return
            cls._draining[simulation_id] = updater
        threading.Thread(
            target=cls._drain, args=(simulation_id, updater), daemon=True,
            name=f"GraphMemoryDrain-{simulation_id[-8:]}",
        ).start()

    @classmethod
    def _drain(cls, simulation_id: str, updater: GraphMemoryUpdater):
        try:
            updater.stop(timeout=None)
            logger.info(f"Graph memory drained: simulation_id={simulation_id}")
        finally:
            with cls._lock:
                cls._draining.pop(simulation_id, None)

    @classmethod
    def pending_count(cls, simulation_id: str) -> int:
        """Activities for this simulation not yet in the graph (0 when no updater exists)."""
        updater = cls._updaters.get(simulation_id) or cls._draining.get(simulation_id)
        return updater.pending_count() if updater else 0

    @classmethod
    def wait_until_drained(
        cls,
        simulation_id: str,
        timeout: float,
        on_wait: Optional[Callable[[int], None]] = None,
        poll_interval: float = 5.0,
    ) -> bool:
        """Block until the simulation's activity has been written to the graph.

        Returns True when nothing is pending, False on timeout. on_wait is
        called with the pending count on each poll.
        """
        deadline = time.time() + timeout
        while simulation_id in cls._updaters or simulation_id in cls._draining:
            if time.time() >= deadline:
                return False
            if on_wait:
                on_wait(cls.pending_count(simulation_id))
            time.sleep(poll_interval)
        return True

    _stop_all_done = False

    @classmethod
    def stop_all(cls):
        """Stop all updaters."""
        if cls._stop_all_done:
            return
        cls._stop_all_done = True

        with cls._lock:
            if cls._updaters:
                for simulation_id, updater in list(cls._updaters.items()):
                    try:
                        updater.stop()
                    except Exception as e:
                        logger.error(f"Failed to stop updater: simulation_id={simulation_id}, error={e}")
                cls._updaters.clear()
            logger.info("Stopped all graph memory updaters")
