"""
OASIS dual-platform parallel simulation preset script
Run Twitter and Reddit simulations simultaneously with the same configuration file

Features:
- Dual-platform (Twitter + Reddit) parallel simulation
- Keep environment running after simulation completes (enter wait mode)
- Support Interview commands via IPC
- Support single Agent interview and batch interview
- Support remote environment shutdown command

Usage:
    python run_parallel_simulation.py --config simulation_config.json
    python run_parallel_simulation.py --config simulation_config.json --no-wait  # Close immediately after completion
    python run_parallel_simulation.py --config simulation_config.json --twitter-only
    python run_parallel_simulation.py --config simulation_config.json --reddit-only

Log structure:
    sim_xxx/
    ├── twitter/
    │   └── actions.jsonl    # Twitter platform action log
    ├── reddit/
    │   └── actions.jsonl    # Reddit platform action log
    ├── simulation.log       # Main simulation process log
    └── run_state.json       # Run state (for API queries)
"""

# ============================================================

# ============================================================
import sys
import os

if sys.platform == 'win32':

    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    

    import builtins
    _original_open = builtins.open
    
    def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None, 
                   newline=None, closefd=True, opener=None):

        if encoding is None and 'b' not in mode:
            encoding = 'utf-8'
        return _original_open(file, mode, buffering, encoding, errors, 
                              newline, closefd, opener)
    
    builtins.open = _utf8_open

import argparse
import asyncio
import contextlib
import io
import json
import logging
import random
import signal
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

_shutdown_event = None
_cleanup_done = False

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

from dotenv import load_dotenv
_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)
    print(f"Loaded environment configuration: {_env_file}")
else:

    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)
        print(f"Loaded environment configuration: {_backend_env}")

class MaxTokensWarningFilter(logging.Filter):
    """Filter out camel-ai max_tokens warnings (we intentionally don't set max_tokens to let the model decide)"""
    
    def filter(self, record):

        if "max_tokens" in record.getMessage() and "Invalid or missing" in record.getMessage():
            return False
        return True

logging.getLogger().addFilter(MaxTokensWarningFilter())

def disable_oasis_logging():

    oasis_loggers = [
        "social.agent",
        "social.twitter", 
        "social.rec",
        "oasis.env",
        "table",
    ]
    
    for logger_name in oasis_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.handlers.clear()
        logger.propagate = False

class _RetryOnlyFilter(logging.Filter):
    def filter(self, record):
        return "Retrying request" in record.getMessage()


def log_model_retries():
    """Surface the OpenAI SDK's otherwise silent retries of agent model calls in simulation.log."""
    retry_log = logging.getLogger("openai._base_client")
    retry_log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] Agent model call timed out or failed: %(message)s", "%H:%M:%S"))
    handler.addFilter(_RetryOnlyFilter())
    retry_log.addHandler(handler)
    retry_log.propagate = False


def init_logging_for_simulation(simulation_dir: str):

    disable_oasis_logging()
    log_model_retries()
    

    old_log_dir = os.path.join(simulation_dir, "log")
    if os.path.exists(old_log_dir):
        import shutil
        shutil.rmtree(old_log_dir, ignore_errors=True)

from action_logger import SimulationLogManager, PlatformActionLogger

try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph,
        generate_reddit_agent_graph
    )
except ImportError as e:
    print(f"Error: Missing dependency {e}")
    print("Install with: pip install oasis-ai camel-ai")
    sys.exit(1)

# Monkey-patch SocialAction methods to tolerate extra kwargs from LLM tool calls
# (e.g. create_comment receives created_at which it doesn't accept).
# functools.wraps is required: CAMEL builds the tool schema from __name__,
# __doc__ and the signature, so without it the LLM sees a tool called
# "_create_comment_compat" with no description and never picks it.
import functools
from oasis.social_agent.agent_action import SocialAction
_orig_create_comment = SocialAction.create_comment
@functools.wraps(_orig_create_comment)
async def _create_comment_compat(self, post_id: int, content: str, **_kwargs):
    return await _orig_create_comment(self, post_id, content)
SocialAction.create_comment = _create_comment_compat

sys.path.insert(0, os.path.join(_backend_dir, 'app', 'services'))
from system_one_router import (
    route_agent_action,
    fetch_agent_feed,
    build_archetype_lookup,
    build_voice_lookup,
)
from agent_memory import AgentMemoryStore
from response_pool import ResponsePool
from round_context import build_feed_digest, build_round_context, inject_round_context
from scenario import SimClock, build_clock, known_facts_block, load_facts, parse_datetime
from agent_observation import (
    CopyGuard,
    attach_copy_guard,
    install_compact_observation,
    install_copy_guard,
)
# After the create_comment shim above, before any agent is built
install_copy_guard()

# Embedding service for response-pool topic matching
_pool_embedder = None
try:
    from app.storage.embedding_service import EmbeddingService
    _pool_embedder = EmbeddingService()
    if _pool_embedder.health_check():
        print("Response pool (amplify): semantic matching enabled (nomic-embed-text)")
    else:
        _pool_embedder = None
        print("Response pool: embedding unavailable, matching by voice only")
except Exception as e:
    print(f"Response pool: semantic matching unavailable ({e}), matching by voice only")

TWITTER_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.REPOST,
    ActionType.FOLLOW,
    ActionType.DO_NOTHING,
    ActionType.QUOTE_POST,
]

# Reddit available actions (excluding INTERVIEW, which can only be triggered via ManualAction)
# Trimmed list: removed SEARCH_POSTS, SEARCH_USER, TREND, REFRESH, MUTE to favor
# interaction-heavy actions (comments, likes, follows) over passive browsing.
REDDIT_ACTIONS = [
    ActionType.LIKE_POST,
    ActionType.DISLIKE_POST,
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_COMMENT,
    ActionType.DISLIKE_COMMENT,
    ActionType.DO_NOTHING,
    ActionType.FOLLOW,
]

IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"

class CommandType:
    """Command type constants"""
    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"

# OASIS answers interviews using the agent's normal memory, which is a run of
# post/comment tool calls. Without framing, qwen3 imitates that history and
# answers with <tool_call>{"name": "create_post", ...} text or a new post
# (about half of Run 8's interview replies). In Run 10, 29 of 40 answers were
# observer summaries of the feed, so the frame asks for the agent's own
# position and limits it to what it did and what was public.
_INTERVIEW_FRAME = (
    "[PRIVATE INTERVIEW - this is not a social media action]\n"
    "A researcher is interviewing you about the events above. Answer as yourself "
    "in plain prose (3-6 sentences): as 'we' if you speak for an organisation, "
    "otherwise in the first person. "
    "Give your own position: what you did or decided and why, who you hold "
    "responsible, whom you trust or distrust now, and what you will do next. "
    "Do not summarise what other people posted or describe the discussion from outside. "
    "Start with the substance: do not open by stating your role or job "
    "(no 'As a ...', 'As part of ...', 'As the ...'). "
    "Mention only actions of yours listed under YOUR RECENT ACTIVITY and only "
    "facts listed as publicly known. "
    "If a question does not apply to you, say so briefly and answer the rest. "
    "Do not write a post, do not ask the interviewer questions, and do not output "
    "tool calls, JSON or <tool_call> tags.\n\n"
    "Questions: "
)


_NO_ACTIVITY_SUMMARY = (
    "YOUR RECENT ACTIVITY: none. You did not post, comment or react during this period. "
    "If asked what you did, say so plainly; do not describe your job or background instead."
)

# Facts shown to interviewees: all that were public at the end, up to this many
_INTERVIEW_FACTS = 10


def frame_interview_prompt(prompt: str) -> str:
    if prompt.startswith(_INTERVIEW_FRAME):
        return prompt
    return _INTERVIEW_FRAME + prompt


class ParallelIPCHandler:
    
    def __init__(
        self,
        simulation_dir: str,
        twitter_env=None,
        twitter_agent_graph=None,
        reddit_env=None,
        reddit_agent_graph=None,
        twitter_memory: Optional[AgentMemoryStore] = None,
        reddit_memory: Optional[AgentMemoryStore] = None,
        interview_facts: Optional[Dict[str, Optional[str]]] = None,
    ):
        self.simulation_dir = simulation_dir
        self.memory = {"twitter": twitter_memory, "reddit": reddit_memory}
        # Per platform: the scenario facts public when its simulation ended
        self.interview_facts = interview_facts or {}
        self.twitter_env = twitter_env
        self.twitter_agent_graph = twitter_agent_graph
        self.reddit_env = reddit_env
        self.reddit_agent_graph = reddit_agent_graph
        
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        

        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
    
    def update_status(self, status: str):
        """Update environment status"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
                "twitter_available": self.twitter_env is not None,
                "reddit_available": self.reddit_env is not None,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def poll_command(self) -> Optional[Dict[str, Any]]:
        """Poll for pending commands"""
        if not os.path.exists(self.commands_dir):
            return None
        

        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))
        
        command_files.sort(key=lambda x: x[1])
        
        for filepath, _ in command_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
        
        return None
    
    def send_response(self, command_id: str, status: str, result: Dict = None, error: str = None):
        """Send response"""
        response = {
            "command_id": command_id,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        response_file = os.path.join(self.responses_dir, f"{command_id}.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        

        command_file = os.path.join(self.commands_dir, f"{command_id}.json")
        try:
            os.remove(command_file)
        except OSError:
            pass
    
    def _prepare_for_interview(self, agent, agent_id: int, platform: str) -> None:
        """Replace the agent's replayed tool-call history with a plain summary.

        Replaying raw create_post/like_post tool calls made qwen3 answer about
        half of Run 8's interviews with <tool_call> text. The agent keeps its
        persona and gets its own activity summary as context instead, plus the
        facts public at the end of the run. An agent that did nothing is told
        so: in Run 10 such agents answered by reciting their persona.
        """
        store = self.memory.get(platform)
        summary = store.get_summary(agent_id, n=12) if store else None
        if summary is None:
            summary = _NO_ACTIVITY_SUMMARY
        ctx = build_round_context(summary, None, include_style_guidance=False,
                                  facts_block=self.interview_facts.get(platform))
        inject_round_context(agent, ctx, keep_turns=0)

    def _get_env_and_graph(self, platform: str):
        if platform == "twitter" and self.twitter_env:
            return self.twitter_env, self.twitter_agent_graph, "twitter"
        elif platform == "reddit" and self.reddit_env:
            return self.reddit_env, self.reddit_agent_graph, "reddit"
        else:
            return None, None, None
    
    async def _interview_single_platform(self, agent_id: int, prompt: str, platform: str) -> Dict[str, Any]:
        env, agent_graph, actual_platform = self._get_env_and_graph(platform)
        
        if not env or not agent_graph:
            return {"platform": platform, "error": f"{platform} platform unavailable"}
        
        try:
            agent = agent_graph.get_agent(agent_id)
            self._prepare_for_interview(agent, agent_id, actual_platform)
            interview_action = ManualAction(
                action_type=ActionType.INTERVIEW,
                action_args={"prompt": frame_interview_prompt(prompt)}
            )
            actions = {agent: interview_action}
            await env.step(actions)
            
            result = self._get_interview_result(agent_id, actual_platform)
            result["platform"] = actual_platform
            return result
            
        except Exception as e:
            return {"platform": platform, "error": str(e)}
    
    async def handle_interview(self, command_id: str, agent_id: int, prompt: str, platform: str = None) -> bool:

        if platform in ("twitter", "reddit"):
            result = await self._interview_single_platform(agent_id, prompt, platform)
            
            if "error" in result:
                self.send_response(command_id, "failed", error=result["error"])
                print(f"  Interview: agent_id={agent_id}, platform={platform}, error={result['error']}")
                return False
            else:
                self.send_response(command_id, "completed", result=result)
                print(f"  Interview: agent_id={agent_id}, platform={platform}")
                return True
        

        if not self.twitter_env and not self.reddit_env:
            self.send_response(command_id, "failed", error="No available simulation environment")
            return False
        
        results = {
            "agent_id": agent_id,
            "prompt": prompt,
            "platforms": {}
        }
        success_count = 0
        

        tasks = []
        platforms_to_interview = []
        
        if self.twitter_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "twitter"))
            platforms_to_interview.append("twitter")
        
        if self.reddit_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "reddit"))
            platforms_to_interview.append("reddit")
        

        platform_results = await asyncio.gather(*tasks)
        
        for platform_name, platform_result in zip(platforms_to_interview, platform_results):
            results["platforms"][platform_name] = platform_result
            if "error" not in platform_result:
                success_count += 1
        
        if success_count > 0:
            self.send_response(command_id, "completed", result=results)
            print(f"  Interview: agent_id={agent_id}, ={success_count}/{len(platforms_to_interview)}")
            return True
        else:
            errors = [f"{p}: {r.get('error', '')}" for p, r in results["platforms"].items()]
            self.send_response(command_id, "failed", error="; ".join(errors))
            print(f"  Interview: agent_id={agent_id}, ")
            return False
    
    async def handle_batch_interview(self, command_id: str, interviews: List[Dict], platform: str = None) -> bool:

        twitter_interviews = []
        reddit_interviews = []
        both_platforms_interviews = []
        
        for interview in interviews:
            item_platform = interview.get("platform", platform)
            if item_platform == "twitter":
                twitter_interviews.append(interview)
            elif item_platform == "reddit":
                reddit_interviews.append(interview)
            else:

                both_platforms_interviews.append(interview)
        

        if both_platforms_interviews:
            if self.twitter_env:
                twitter_interviews.extend(both_platforms_interviews)
            if self.reddit_env:
                reddit_interviews.extend(both_platforms_interviews)
        
        results = {}
        

        if twitter_interviews and self.twitter_env:
            try:
                twitter_actions = {}
                for interview in twitter_interviews:
                    agent_id = interview.get("agent_id")
                    prompt = interview.get("prompt", "")
                    try:
                        agent = self.twitter_agent_graph.get_agent(agent_id)
                        self._prepare_for_interview(agent, agent_id, "twitter")
                        twitter_actions[agent] = ManualAction(
                            action_type=ActionType.INTERVIEW,
                            action_args={"prompt": frame_interview_prompt(prompt)}
                        )
                    except Exception as e:
                        print(f"  Warning: Unable to get Twitter Agent {agent_id}: {e}")
                
                if twitter_actions:
                    await self.twitter_env.step(twitter_actions)
                    
                    for interview in twitter_interviews:
                        agent_id = interview.get("agent_id")
                        result = self._get_interview_result(agent_id, "twitter")
                        result["platform"] = "twitter"
                        results[f"twitter_{agent_id}"] = result
            except Exception as e:
                print(f"  TwitterInterview: {e}")
        

        if reddit_interviews and self.reddit_env:
            try:
                reddit_actions = {}
                for interview in reddit_interviews:
                    agent_id = interview.get("agent_id")
                    prompt = interview.get("prompt", "")
                    try:
                        agent = self.reddit_agent_graph.get_agent(agent_id)
                        self._prepare_for_interview(agent, agent_id, "reddit")
                        reddit_actions[agent] = ManualAction(
                            action_type=ActionType.INTERVIEW,
                            action_args={"prompt": frame_interview_prompt(prompt)}
                        )
                    except Exception as e:
                        print(f"  Warning: Unable to get Reddit Agent {agent_id}: {e}")
                
                if reddit_actions:
                    await self.reddit_env.step(reddit_actions)
                    
                    for interview in reddit_interviews:
                        agent_id = interview.get("agent_id")
                        result = self._get_interview_result(agent_id, "reddit")
                        result["platform"] = "reddit"
                        results[f"reddit_{agent_id}"] = result
            except Exception as e:
                print(f"  RedditInterview: {e}")
        
        if results:
            self.send_response(command_id, "completed", result={
                "interviews_count": len(results),
                "results": results
            })
            print(f"  Interview: {len(results)} Agent")
            return True
        else:
            self.send_response(command_id, "failed", error="No successful interviews")
            return False
    
    @staticmethod
    def _extract_content_from_response(response):
        if not response:
            return response
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
                response = parsed
            except (json.JSONDecodeError, TypeError):
                return response
        if isinstance(response, dict):
            if "arguments" in response and isinstance(response["arguments"], dict):
                return response["arguments"].get("content", response["arguments"].get("text", str(response["arguments"])))
            if "content" in response:
                return response["content"]
            if "response" in response:
                return response["response"]
        return response

    def _get_interview_result(self, agent_id: int, platform: str) -> Dict[str, Any]:
        """Get the latest Interview result from database"""
        db_path = os.path.join(self.simulation_dir, f"{platform}_simulation.db")

        result = {
            "agent_id": agent_id,
            "response": None,
            "timestamp": None
        }

        if not os.path.exists(db_path):
            return result

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()


            cursor.execute("""
                SELECT user_id, info, created_at
                FROM trace
                WHERE action = ? AND user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (ActionType.INTERVIEW.value, agent_id))

            row = cursor.fetchone()
            if row:
                user_id, info_json, created_at = row
                try:
                    info = json.loads(info_json) if info_json else {}
                    raw_response = info.get("response", info)
                    result["response"] = self._extract_content_from_response(raw_response)
                    result["timestamp"] = created_at
                except json.JSONDecodeError:
                    result["response"] = info_json

            conn.close()

        except Exception as e:
            print(f"  Interview: {e}")

        return result
    
    async def process_commands(self) -> bool:
        command = self.poll_command()
        if not command:
            return True
        
        command_id = command.get("command_id")
        command_type = command.get("command_type")
        args = command.get("args", {})
        
        print(f"\nIPC: {command_type}, id={command_id}")
        
        if command_type == CommandType.INTERVIEW:
            await self.handle_interview(
                command_id,
                args.get("agent_id", 0),
                args.get("prompt", ""),
                args.get("platform")
            )
            return True
            
        elif command_type == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(
                command_id,
                args.get("interviews", []),
                args.get("platform")
            )
            return True
            
        elif command_type == CommandType.CLOSE_ENV:
            print("Received close environment command")
            self.send_response(command_id, "completed", result={"message": "Environment will close"})
            return False
        
        else:
            self.send_response(command_id, "failed", error=f"Unknown command type: {command_type}")
            return True

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

FILTERED_ACTIONS = {'refresh', 'sign_up'}

ACTION_TYPE_MAP = {
    'create_post': 'CREATE_POST',
    'like_post': 'LIKE_POST',
    'dislike_post': 'DISLIKE_POST',
    'repost': 'REPOST',
    'quote_post': 'QUOTE_POST',
    'follow': 'FOLLOW',
    'mute': 'MUTE',
    'create_comment': 'CREATE_COMMENT',
    'like_comment': 'LIKE_COMMENT',
    'dislike_comment': 'DISLIKE_COMMENT',
    'search_posts': 'SEARCH_POSTS',
    'search_user': 'SEARCH_USER',
    'trend': 'TREND',
    'do_nothing': 'DO_NOTHING',
    'interview': 'INTERVIEW',
}

def get_agent_names_from_config(config: Dict[str, Any]) -> Dict[int, str]:
    agent_names = {}
    agent_configs = config.get("agent_configs", [])

    for agent_config in agent_configs:
        agent_id = agent_config.get("agent_id")
        entity_name = agent_config.get("entity_name", f"Agent_{agent_id}")
        if agent_id is not None:
            agent_names[agent_id] = entity_name

    return agent_names

def build_topics_lookup(simulation_dir: str) -> Dict[int, str]:
    """Build agent_id -> query string for semantic pool matching.

    Topics live in the profile files, not in simulation_config.json's
    agent_configs. reddit_profiles.json carries interested_topics; the Twitter
    CSV drops them, so Twitter-only runs use the persona text instead.
    Profile user_id == OASIS agent_id on both platforms.
    """
    lookup: Dict[int, str] = {}
    reddit_path = os.path.join(simulation_dir, "reddit_profiles.json")
    twitter_path = os.path.join(simulation_dir, "twitter_profiles.csv")
    if os.path.exists(reddit_path):
        with open(reddit_path, 'r', encoding='utf-8') as f:
            for p in json.load(f):
                topics = p.get("interested_topics") or []
                if p.get("user_id") is not None and topics:
                    lookup[int(p["user_id"])] = " ".join(topics)
    elif os.path.exists(twitter_path):
        import csv
        with open(twitter_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                persona = (row.get("user_char") or "").strip()
                if row.get("user_id") not in (None, "") and persona:
                    lookup[int(row["user_id"])] = persona[:500]
    return lookup


def fetch_new_actions_from_db(
    db_path: str,
    last_rowid: int,
    agent_names: Dict[int, str]
) -> Tuple[List[Dict[str, Any]], int]:
    actions = []
    new_last_rowid = last_rowid
    
    if not os.path.exists(db_path):
        return actions, new_last_rowid
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        

        cursor.execute("""
            SELECT rowid, user_id, action, info
            FROM trace
            WHERE rowid > ?
            ORDER BY rowid ASC
        """, (last_rowid,))
        
        for rowid, user_id, action, info_json in cursor.fetchall():

            new_last_rowid = rowid
            

            if action in FILTERED_ACTIONS:
                continue
            

            try:
                action_args = json.loads(info_json) if info_json else {}
            except json.JSONDecodeError:
                action_args = {}
            

            simplified_args = {}
            if 'content' in action_args:
                simplified_args['content'] = action_args['content']
            if 'post_id' in action_args:
                simplified_args['post_id'] = action_args['post_id']
            if 'comment_id' in action_args:
                simplified_args['comment_id'] = action_args['comment_id']
            if 'quoted_id' in action_args:
                simplified_args['quoted_id'] = action_args['quoted_id']
            if 'new_post_id' in action_args:
                simplified_args['new_post_id'] = action_args['new_post_id']
            if 'follow_id' in action_args:
                simplified_args['follow_id'] = action_args['follow_id']
            if 'query' in action_args:
                simplified_args['query'] = action_args['query']
            if 'like_id' in action_args:
                simplified_args['like_id'] = action_args['like_id']
            if 'dislike_id' in action_args:
                simplified_args['dislike_id'] = action_args['dislike_id']
            

            action_type = ACTION_TYPE_MAP.get(action, action.upper())
            

            _enrich_action_context(cursor, action_type, simplified_args, agent_names)
            
            actions.append({
                'agent_id': user_id,
                'agent_name': agent_names.get(user_id, f'Agent_{user_id}'),
                'action_type': action_type,
                'action_args': simplified_args,
            })
        
        conn.close()
    except Exception as e:
        print(f"Failed to read Database actions: {e}")
    
    return actions, new_last_rowid

def _enrich_action_context(
    cursor,
    action_type: str,
    action_args: Dict[str, Any],
    agent_names: Dict[int, str]
) -> None:
    try:

        if action_type in ('LIKE_POST', 'DISLIKE_POST'):
            post_id = action_args.get('post_id')
            if post_id:
                post_info = _get_post_info(cursor, post_id, agent_names)
                if post_info:
                    action_args['post_content'] = post_info.get('content', '')
                    action_args['post_author_name'] = post_info.get('author_name', '')
        

        elif action_type == 'REPOST':
            new_post_id = action_args.get('new_post_id')
            if new_post_id:

                cursor.execute("""
                    SELECT original_post_id FROM post WHERE post_id = ?
                """, (new_post_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    original_post_id = row[0]
                    original_info = _get_post_info(cursor, original_post_id, agent_names)
                    if original_info:
                        action_args['original_content'] = original_info.get('content', '')
                        action_args['original_author_name'] = original_info.get('author_name', '')
        

        elif action_type == 'QUOTE_POST':
            quoted_id = action_args.get('quoted_id')
            new_post_id = action_args.get('new_post_id')
            
            if quoted_id:
                original_info = _get_post_info(cursor, quoted_id, agent_names)
                if original_info:
                    action_args['original_content'] = original_info.get('content', '')
                    action_args['original_author_name'] = original_info.get('author_name', '')
            

            if new_post_id:
                cursor.execute("""
                    SELECT quote_content FROM post WHERE post_id = ?
                """, (new_post_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    action_args['quote_content'] = row[0]
        

        elif action_type == 'FOLLOW':
            follow_id = action_args.get('follow_id')
            if follow_id:

                cursor.execute("""
                    SELECT followee_id FROM follow WHERE follow_id = ?
                """, (follow_id,))
                row = cursor.fetchone()
                if row:
                    followee_id = row[0]
                    target_name = _get_user_name(cursor, followee_id, agent_names)
                    if target_name:
                        action_args['target_user_name'] = target_name
        

        elif action_type == 'MUTE':

            target_id = action_args.get('user_id') or action_args.get('target_id')
            if target_id:
                target_name = _get_user_name(cursor, target_id, agent_names)
                if target_name:
                    action_args['target_user_name'] = target_name
        

        elif action_type in ('LIKE_COMMENT', 'DISLIKE_COMMENT'):
            comment_id = action_args.get('comment_id')
            if comment_id:
                comment_info = _get_comment_info(cursor, comment_id, agent_names)
                if comment_info:
                    action_args['comment_content'] = comment_info.get('content', '')
                    action_args['comment_author_name'] = comment_info.get('author_name', '')
        

        elif action_type == 'CREATE_COMMENT':
            post_id = action_args.get('post_id')
            if post_id:
                post_info = _get_post_info(cursor, post_id, agent_names)
                if post_info:
                    action_args['post_content'] = post_info.get('content', '')
                    action_args['post_author_name'] = post_info.get('author_name', '')
    
    except Exception as e:

        print(f"Failed to enrich action context: {e}")

def _get_post_info(
    cursor,
    post_id: int,
    agent_names: Dict[int, str]
) -> Optional[Dict[str, str]]:
    try:
        cursor.execute("""
            SELECT p.content, p.user_id, u.agent_id
            FROM post p
            LEFT JOIN user u ON p.user_id = u.user_id
            WHERE p.post_id = ?
        """, (post_id,))
        row = cursor.fetchone()
        if row:
            content = row[0] or ''
            user_id = row[1]
            agent_id = row[2]
            

            author_name = ''
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:

                cursor.execute("SELECT name, user_name FROM user WHERE user_id = ?", (user_id,))
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ''
            
            return {'content': content, 'author_name': author_name}
    except Exception:
        pass
    return None

def _get_user_name(
    cursor,
    user_id: int,
    agent_names: Dict[int, str]
) -> Optional[str]:
    try:
        cursor.execute("""
            SELECT agent_id, name, user_name FROM user WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        if row:
            agent_id = row[0]
            name = row[1]
            user_name = row[2]
            

            if agent_id is not None and agent_id in agent_names:
                return agent_names[agent_id]
            return name or user_name or ''
    except Exception:
        pass
    return None

def _get_comment_info(
    cursor,
    comment_id: int,
    agent_names: Dict[int, str]
) -> Optional[Dict[str, str]]:
    try:
        cursor.execute("""
            SELECT c.content, c.user_id, u.agent_id
            FROM comment c
            LEFT JOIN user u ON c.user_id = u.user_id
            WHERE c.comment_id = ?
        """, (comment_id,))
        row = cursor.fetchone()
        if row:
            content = row[0] or ''
            user_id = row[1]
            agent_id = row[2]
            

            author_name = ''
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:

                cursor.execute("SELECT name, user_name FROM user WHERE user_id = ?", (user_id,))
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ''
            
            return {'content': content, 'author_name': author_name}
    except Exception:
        pass
    return None

def create_model(config: Dict[str, Any], use_boost: bool = False):

    boost_api_key = os.environ.get("LLM_BOOST_API_KEY", "")
    boost_base_url = os.environ.get("LLM_BOOST_BASE_URL", "")
    boost_model = os.environ.get("LLM_BOOST_MODEL_NAME", "")
    has_boost_config = bool(boost_api_key)
    

    if use_boost and has_boost_config:

        llm_api_key = boost_api_key
        llm_base_url = boost_base_url
        llm_model = boost_model or os.environ.get("LLM_MODEL_NAME", "")
        config_label = "[LLM]"
    else:

        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")
        config_label = "[LLM]"
    

    if not llm_model:
        llm_model = config.get("llm_model", "gpt-4o-mini")
    

    if llm_api_key:
        os.environ["OPENAI_API_KEY"] = llm_api_key
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("API Key .env LLM_API_KEY")
    
    if llm_base_url:
        os.environ["OPENAI_API_BASE_URL"] = llm_base_url
    
    print(f"{config_label} model={llm_model}, base_url={llm_base_url[:40] if llm_base_url else ''}...")
    
    # Build model config — disable thinking for qwen3 models (thinking
    # consumes token budget and breaks tool calling with low max_tokens).
    model_config = {}
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        model_config["extra_body"] = {
            # Ollama /v1 ignores `think`; this is the switch it honours (see llm_client.py)
            "reasoning_effort": "none",
        }

    # Sampling diversity for content generation. CAMEL leaves temperature=None,
    # so Ollama falls back to qwen3's low default (~0.6) which drives repetitive
    # posts. Bump it (and top_p) to spread CREATE_POST/comment output. Tunable
    # via env so it can be dialed back for a more deterministic run.
    model_config["temperature"] = float(os.environ.get("SIM_GEN_TEMPERATURE", "0.9"))
    model_config["top_p"] = float(os.environ.get("SIM_GEN_TOP_P", "0.95"))

    # CAMEL's default is a 180s timeout with 3 silent SDK retries. With ~16
    # agents queued on 3 Ollama slots a call can wait longer than that, so it
    # was cut off (Ollama logs a 500 at 3m0s) and re-queued from scratch.
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=llm_model,
        model_config_dict=model_config if model_config else None,
        timeout=float(os.environ.get("SIM_MODEL_TIMEOUT", "600")),
        max_retries=int(os.environ.get("SIM_MODEL_MAX_RETRIES", "1")),
    )

def get_active_agents_for_round(
    env,
    config: Dict[str, Any],
    current_hour: int,
    round_num: int
) -> List:
    """Select which agents are active this round based on time-of-day and agent config.

    Guarantees at least 1 agent per round to avoid empty rounds.
    """
    time_config = config.get("time_config", {})
    agent_configs = config.get("agent_configs", [])

    base_min = time_config.get("agents_per_hour_min", 5)
    base_max = time_config.get("agents_per_hour_max", 20)

    peak_hours = time_config.get("peak_hours", [18, 19, 20, 21])
    off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
    morning_hours = time_config.get("morning_hours", [6, 7, 8])
    work_hours = time_config.get("work_hours", list(range(9, 18)))

    if current_hour in peak_hours:
        multiplier = time_config.get("peak_activity_multiplier", 1.5)
    elif current_hour in off_peak_hours:
        multiplier = time_config.get("off_peak_activity_multiplier", 0.15)
    elif current_hour in morning_hours:
        multiplier = time_config.get("morning_activity_multiplier", 0.4)
    elif current_hour in work_hours:
        multiplier = time_config.get("work_activity_multiplier", 0.7)
    else:
        multiplier = 0.5  # Night hours

    target_count = max(1, int(random.uniform(base_min, base_max) * multiplier))

    # Build candidate pool: agents whose active_hours include this hour, filtered by activity_level
    candidates = []
    all_agent_ids = []
    for cfg in agent_configs:
        agent_id = cfg.get("agent_id", 0)
        active_hours = cfg.get("active_hours", list(range(8, 23)))
        activity_level = cfg.get("activity_level", 0.5)
        all_agent_ids.append(agent_id)

        if current_hour not in active_hours:
            continue

        if random.random() < activity_level:
            candidates.append(agent_id)

    # Guarantee at least 1 candidate — pick a random agent if pool is empty
    if not candidates and all_agent_ids:
        candidates = [random.choice(all_agent_ids)]

    selected_ids = random.sample(
        candidates,
        min(target_count, len(candidates))
    ) if candidates else []

    active_agents = []
    for agent_id in selected_ids:
        try:
            agent = env.agent_graph.get_agent(agent_id)
            active_agents.append((agent_id, agent))
        except Exception:
            pass

    return active_agents

class PlatformSimulation:
    """Platform simulation result container"""
    def __init__(self):
        self.env = None
        self.agent_graph = None
        self.total_actions = 0
        self.memory_store: Optional[AgentMemoryStore] = None
        self.clock: Optional[SimClock] = None
        self.facts: List[Any] = []
        self.final_time: Optional[datetime] = None  # scenario time of the last round run

    def interview_facts_block(self) -> Optional[str]:
        """The facts public when the simulation ended, for interviews."""
        if self.clock is None or self.final_time is None or not self.facts:
            return None
        return known_facts_block(self.facts, self.final_time, self.clock, _INTERVIEW_FACTS, random.Random(0))

_TEXT_ACTIONS = ('CREATE_POST', 'CREATE_COMMENT', 'QUOTE_POST')

# Chance that an agent routed to write amplifies a same-voice peer instead
_POOL_AMPLIFY_PROB = float(os.environ.get("SIM_POOL_AMPLIFY_PROB", "0.3"))


def _action_text(action_type: str, action_args: Dict[str, Any]) -> str:
    """Return the text an agent wrote for an action (quotes store it in quote_content)."""
    if action_type == 'QUOTE_POST':
        return (action_args.get('quote_content') or '').strip()
    return (action_args.get('content') or '').strip()


# Event configs written before planned_rounds existed were planned against
# OASIS_DEFAULT_MAX_ROUNDS, which defaults to 10.
_LEGACY_PLANNED_ROUNDS = 10


def schedule_events(
    event_config: Dict[str, Any],
    total_rounds: int,
    clock: Optional[SimClock] = None,
    log_info=print,
) -> Dict[int, List[Dict[str, Any]]]:
    """Map round number (1-based) -> scheduled events that fire in that round.

    An event with a scenario time ("at") fires in the first round whose clock
    time reaches it, so it lines up with the facts agents are shown; an event
    after the simulated period is dropped. Older configs give trigger_round,
    planned against event_config['planned_rounds'], and are rescaled when the
    run is longer or shorter so the narrative arc keeps its shape. Events
    without a poster agent or content are dropped.
    """
    events = event_config.get("scheduled_events") or []
    planned = event_config.get("planned_rounds") or _LEGACY_PLANNED_ROUNDS
    schedule: Dict[int, List[Dict[str, Any]]] = {}
    for event in events:
        if event.get("poster_agent_id") is None or not (event.get("content") or "").strip():
            continue
        when = parse_datetime(event.get("at")) if clock and clock.dated else None
        if when is not None:
            if when > clock.end:
                log_info(f"Scheduled event after the simulated period ({when:%d %b %H:%M}) dropped: "
                         f"{event.get('description', '')[:60]}")
                continue
            trigger = clock.round_for(when)
        else:
            try:
                trigger = int(event.get("trigger_round", 1))
            except (TypeError, ValueError):
                continue
            if planned != total_rounds:
                trigger = round(trigger * total_rounds / planned)
        trigger = min(max(trigger, 1), total_rounds)
        schedule.setdefault(trigger, []).append(event)
    return schedule


def _add_manual_action(actions: Dict[Any, Any], agent, action: ManualAction) -> None:
    """Queue a manual action; an agent with several this round gets a list (OASIS runs each)."""
    if agent not in actions:
        actions[agent] = action
    elif isinstance(actions[agent], list):
        actions[agent].append(action)
    else:
        actions[agent] = [actions[agent], action]


def apply_scheduled_events(
    events: List[Dict[str, Any]],
    env,
    actions: Dict[Any, Any],
    scripted_agents: set,
    log_info,
) -> None:
    """Add this round's scheduled event posts to actions.

    The poster agent posts the event text instead of acting on its own this
    round, whether or not it was scheduled to be active. Two events for one
    agent in the same round are both posted.
    """
    for event in events:
        agent_id = event["poster_agent_id"]
        try:
            agent = env.agent_graph.get_agent(agent_id)
        except Exception as e:
            log_info(f"Scheduled event skipped, agent {agent_id} not found: {e}")
            continue
        _add_manual_action(actions, agent, ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": event["content"].strip()},
        ))
        scripted_agents.add(agent_id)
        log_info(f"Scheduled event fired for agent {agent_id}: {event.get('description', '')[:80]}")


def pool_amplify_action(entry, platform: str) -> ManualAction:
    """Turn a pool hit into an amplification of the peer's original item:
    a repost on Twitter, an upvote on Reddit. The agent never re-posts the
    peer's words under its own name."""
    if entry.kind == 'comment':
        return ManualAction(action_type=ActionType.LIKE_COMMENT, action_args={"comment_id": entry.item_id})
    if platform == 'twitter':
        return ManualAction(action_type=ActionType.REPOST, action_args={"post_id": entry.item_id})
    return ManualAction(action_type=ActionType.LIKE_POST, action_args={"post_id": entry.item_id})


def log_round_actions(
    actual_actions: List[Dict[str, Any]],
    round_num: int,
    action_logger: Optional['PlatformActionLogger'],
    memory_store: AgentMemoryStore,
    response_pool: ResponsePool,
    voice_lookup: Dict[int, Optional[str]],
    archetype_lookup: Dict[int, str],
    scripted_agents: set,
    seen_posts: set,
    counters: Dict[str, int],
) -> int:
    """Log the actions OASIS committed this round, record them in agent memory,
    and add LLM-written posts and comments to the response pool.

    Every committed action is logged, so actions.jsonl matches the platform DB.
    Verbatim repeat posts, comments and quotes (of any earlier text, by
    anyone) are logged as well and counted in
    counters['verbatim_repeats'] instead of being hidden. Posts or comments with
    no text are the only actions skipped, and they are counted in
    counters['empty_text'].

    Args:
        scripted_agents: agent ids whose post this round was a scheduled event,
            not LLM-written. Their posts are recorded as System One in memory
            and are not added to the pool.

    Returns:
        Number of actions logged this round.
    """
    logged = 0
    for action_data in actual_actions:
        action_type = action_data.get('action_type', '')
        action_args = action_data.get('action_args', {})
        a_id = action_data['agent_id']
        text = _action_text(action_type, action_args)

        if action_type in ('CREATE_POST', 'CREATE_COMMENT') and not text:
            counters['empty_text'] += 1
            continue

        is_repeat = False
        if action_type in ('CREATE_POST', 'CREATE_COMMENT', 'QUOTE_POST'):
            # Comments count too: Run 10's copies were Reddit comments
            key = " ".join(text[:200].lower().split())
            is_repeat = key in seen_posts
            if is_repeat:
                counters['verbatim_repeats'] += 1
            seen_posts.add(key)

        if action_logger:
            action_logger.log_action(
                round_num=round_num,
                agent_id=a_id,
                agent_name=action_data['agent_name'],
                action_type=action_type,
                action_args=action_args,
            )
        logged += 1

        if action_type in _TEXT_ACTIONS:
            scripted = action_type == 'CREATE_POST' and a_id in scripted_agents
            memory_store.record(
                agent_id=a_id,
                round_num=round_num,
                action_type=action_type,
                content_preview=text,
                via_system_one=scripted,
            )
            if action_type in ('CREATE_POST', 'CREATE_COMMENT') and not scripted and not is_repeat:
                kind = 'post' if action_type == 'CREATE_POST' else 'comment'
                response_pool.add(
                    archetype_lookup.get(a_id, "contributor"), text,
                    kind=kind, item_id=action_args.get(f'{kind}_id'),
                    voice=voice_lookup.get(a_id), source_agent_id=a_id,
                )
    return logged


# Facts shown per agent per round (fresh developments are always shown on top
# of these). Institutions get more because they are asked to post official
# updates from them; residents get fewer so they do not all recite one list.
_FACTS_PER_INSTITUTION = int(os.environ.get("SIM_FACTS_INSTITUTION", "6"))
_FACTS_PER_INDIVIDUAL = int(os.environ.get("SIM_FACTS_INDIVIDUAL", "4"))

_PLATFORMS = {
    "twitter": {
        "label": "Twitter",
        "profile_file": "twitter_profiles.csv",
        "db_file": "twitter_simulation.db",
        "memory_file": "twitter_agent_memory.db",
        "graph": generate_twitter_agent_graph,
        "actions": TWITTER_ACTIONS,
        "platform_type": oasis.DefaultPlatformType.TWITTER,
        "use_boost": False,
    },
    "reddit": {
        "label": "Reddit",
        "profile_file": "reddit_profiles.json",
        "db_file": "reddit_simulation.db",
        "memory_file": "reddit_agent_memory.db",
        "graph": generate_reddit_agent_graph,
        "actions": REDDIT_ACTIONS,
        "platform_type": oasis.DefaultPlatformType.REDDIT,
        "use_boost": True,
    },
}


async def run_platform_simulation(
    platform: str,
    config: Dict[str, Any],
    simulation_dir: str,
    action_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None,
) -> PlatformSimulation:
    """Run one platform's simulation: seed posts, then clock-driven rounds."""
    spec = _PLATFORMS[platform]
    label = spec["label"]
    result = PlatformSimulation()

    def log_info(msg):
        if main_logger:
            main_logger.info(f"[{label}] {msg}")
        else:
            print(f"[{label}] {msg}")

    log_info("Initializing...")
    model = create_model(config, use_boost=spec["use_boost"])

    profile_path = os.path.join(simulation_dir, spec["profile_file"])
    if not os.path.exists(profile_path):
        log_info(f"Error: Profile file does not exist: {profile_path}")
        return result

    # OASIS prints every Reddit profile to stdout while building system prompts
    with contextlib.redirect_stdout(io.StringIO()):
        result.agent_graph = await spec["graph"](
            profile_path=profile_path,
            model=model,
            available_actions=spec["actions"],
        )

    agent_names = get_agent_names_from_config(config)
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')
    install_compact_observation(agent_names)

    db_path = os.path.join(simulation_dir, spec["db_file"])
    if os.path.exists(db_path):
        os.remove(db_path)

    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=spec["platform_type"],
        database_path=db_path,
        semaphore=30,
    )
    await result.env.reset()
    copy_guard = CopyGuard(db_path, platform)
    attach_copy_guard(result.agent_graph, copy_guard)
    log_info("Environment started")

    if action_logger:
        action_logger.log_simulation_start(config)

    # Rounds and the scenario clock
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 60)
    total_rounds = (total_hours * 60) // minutes_per_round
    if max_rounds is not None and max_rounds > 0 and max_rounds < total_rounds:
        log_info(f"Rounds: {max_rounds} of the configured {total_rounds} (max_rounds)")
        total_rounds = max_rounds
    clock = build_clock(config, total_rounds)
    facts = load_facts(config)
    result.clock, result.facts, result.final_time = clock, facts, clock.start
    log_info(f"Clock: {clock.label(clock.at(1))} to {clock.label(clock.at(total_rounds))}, "
             f"{clock.step_description}; {len(facts)} scenario facts")

    total_actions = 0
    last_rowid = 0

    memory_store = AgentMemoryStore(db_path=os.path.join(simulation_dir, spec["memory_file"]))
    result.memory_store = memory_store
    response_pool = ResponsePool(amplify_probability=_POOL_AMPLIFY_PROB, embedding_service=_pool_embedder)
    voice_lookup = build_voice_lookup(config)
    log_counters = {'verbatim_repeats': 0, 'empty_text': 0}
    seen_texts: set = set()
    archetype_lookup = build_archetype_lookup(config)

    # Round 0: seed posts. One agent may have several, so each gets a list.
    event_config = config.get("event_config", {})
    if action_logger:
        action_logger.log_round_start(0, clock.start.hour, clock.start.isoformat())
    initial_actions: Dict[Any, Any] = {}
    seed_count = 0
    for post in event_config.get("initial_posts", []):
        agent_id = post.get("poster_agent_id")
        content = (post.get("content") or "").strip()
        if agent_id is None or not content:
            log_info(f"Skipping initial post without poster or content: {content[:60]!r}")
            continue
        try:
            agent = result.env.agent_graph.get_agent(agent_id)
        except Exception as e:
            log_info(f"Skipping initial post, agent {agent_id} not found: {e}")
            continue
        _add_manual_action(initial_actions, agent, ManualAction(
            action_type=ActionType.CREATE_POST, action_args={"content": content}))
        seed_count += 1
    if initial_actions:
        await result.env.step(initial_actions)
        log_info(f"Published {seed_count} initial posts from {len(initial_actions)} agents")
    # Log the seeds as committed, from the DB like every other round. They go
    # into the posters' memory too: in the Run 11 smoke test a seed poster was
    # told at interview time that it had done nothing.
    seed_actions, last_rowid = fetch_new_actions_from_db(db_path, last_rowid, agent_names)
    seed_count = log_round_actions(
        seed_actions, 0, action_logger, memory_store, response_pool,
        voice_lookup, archetype_lookup, {a['agent_id'] for a in seed_actions},
        seen_texts, log_counters,
    )
    total_actions += seed_count
    if action_logger:
        action_logger.log_round_end(0, seed_count)

    start_time = datetime.now()
    available = [a.name.upper() for a in spec["actions"]]
    s1_skipped = 0
    s2_calls = 0

    event_schedule = schedule_events(event_config, total_rounds, clock, log_info)
    log_info(f"Scheduled events: {sum(len(v) for v in event_schedule.values())} at rounds {sorted(event_schedule)}")
    pool_amplified = 0
    topics_lookup = build_topics_lookup(simulation_dir)

    for round_num in range(1, total_rounds + 1):
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(f"Received shutdown signal, stopping at round {round_num}")
            break

        now = clock.at(round_num)
        result.final_time = now
        active_agents = get_active_agents_for_round(result.env, config, now.hour, round_num)

        if action_logger:
            action_logger.log_round_start(round_num, now.hour, now.isoformat())

        actions: Dict[Any, Any] = {}
        scripted_agents: set = set()
        apply_scheduled_events(event_schedule.get(round_num, []), result.env, actions, scripted_agents, log_info)

        if not active_agents and not actions:
            if action_logger:
                action_logger.log_round_end(round_num, 0)
            continue

        # System One routing: fetch feed once, then route each agent
        feed_posts = await fetch_agent_feed(active_agents[0][1]) if active_agents else []

        for agent_id, agent in active_agents:
            if agent_id in scripted_agents:
                continue  # posting a scheduled event this round
            archetype = archetype_lookup.get(agent_id, "contributor")
            action = route_agent_action(
                agent=agent,
                agent_id=agent_id,
                archetype=archetype,
                platform=platform,
                available_actions=available,
                feed_posts=feed_posts,
                agent_graph=result.agent_graph,
            )
            if isinstance(action, LLMAction):
                # Amplify a same-voice peer's item instead of spending an LLM call
                entry = response_pool.try_amplify(
                    archetype, query=topics_lookup.get(agent_id),
                    voice=voice_lookup.get(agent_id), agent_id=agent_id,
                )
                if entry:
                    amp_action = pool_amplify_action(entry, platform)
                    actions[agent] = amp_action
                    pool_amplified += 1
                    s1_skipped += 1
                    memory_store.record(
                        agent_id=agent_id,
                        round_num=round_num,
                        action_type=amp_action.action_type.name,
                        target_id=entry.item_id,
                        content_preview=entry.content,
                        via_system_one=True,
                    )
                else:
                    actions[agent] = action
                    s2_calls += 1
                    institutional = voice_lookup.get(agent_id) is None
                    facts_block = known_facts_block(
                        facts, now, clock,
                        _FACTS_PER_INSTITUTION if institutional else _FACTS_PER_INDIVIDUAL,
                        random.Random(round_num * 100003 + agent_id),
                    ) if clock.dated or facts else None
                    ctx = build_round_context(
                        memory_store.get_summary(agent_id),
                        build_feed_digest(feed_posts, exclude_user_id=agent_id,
                                          platform=platform, names=agent_names),
                        institutional=institutional,
                        facts_block=facts_block,
                    )
                    inject_round_context(agent, ctx)
            else:
                actions[agent] = action
                s1_skipped += 1
                if isinstance(action, ManualAction):
                    target = (
                        action.action_args.get("post_id")
                        or action.action_args.get("followee_id")
                        or action.action_args.get("comment_id")
                    )
                    memory_store.record(
                        agent_id=agent_id,
                        round_num=round_num,
                        action_type=action.action_type.name if hasattr(action.action_type, 'name') else str(action.action_type),
                        target_id=target,
                        via_system_one=True,
                    )

        await result.env.step(actions)

        actual_actions, last_rowid = fetch_new_actions_from_db(db_path, last_rowid, agent_names)
        round_action_count = log_round_actions(
            actual_actions, round_num, action_logger, memory_store, response_pool,
            voice_lookup, archetype_lookup, scripted_agents,
            seen_texts, log_counters,
        )
        total_actions += round_action_count

        if action_logger:
            action_logger.log_round_end(round_num, round_action_count)

        if round_num % 5 == 0:
            log_info(f"{clock.label(now)} - Round {round_num}/{total_rounds} "
                     f"({round_num / total_rounds * 100:.0f}%)")

    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)

    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    total_decisions = s1_skipped + s2_calls
    s1_pct = (s1_skipped / max(total_decisions, 1)) * 100
    pool_stats = response_pool.stats
    log_info(
        f"{label} done: {elapsed:.1f}s, {total_actions} actions, "
        f"System One: {s1_skipped}/{total_decisions} ({s1_pct:.0f}%) decisions skipped LLM, "
        f"pool: amplified={pool_amplified}, deduped={pool_stats['deduped']}, semantic={pool_stats['semantic_matches']}, "
        f"no_candidate={pool_stats['no_candidate']}, "
        f"copy_guard: converted={copy_guard.stats['copies_converted']} (self={copy_guard.stats['self_repeats']}), "
        f"verbatim_repeats={log_counters['verbatim_repeats']}, empty_text={log_counters['empty_text']}, "
        f"memory: {memory_store.total_records()} records for {memory_store.agent_count()} agents"
    )
    return result


async def run_twitter_simulation(config, simulation_dir, action_logger=None, main_logger=None, max_rounds=None):
    return await run_platform_simulation("twitter", config, simulation_dir, action_logger, main_logger, max_rounds)


async def run_reddit_simulation(config, simulation_dir, action_logger=None, main_logger=None, max_rounds=None):
    return await run_platform_simulation("reddit", config, simulation_dir, action_logger, main_logger, max_rounds)


async def main():
    parser = argparse.ArgumentParser(description='Run the OASIS Twitter and Reddit simulations in parallel')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to simulation_config.json (its directory is the simulation directory)'
    )
    parser.add_argument(
        '--twitter-only',
        action='store_true',
        help='Run only the Twitter simulation'
    )
    parser.add_argument(
        '--reddit-only',
        action='store_true',
        help='Run only the Reddit simulation'
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=None,
        help='Stop after this many rounds (default: the configured total)'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        default=False,
        help='Exit when the rounds finish instead of waiting for interview commands'
    )
    
    args = parser.parse_args()
    

    global _shutdown_event
    _shutdown_event = asyncio.Event()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file does not exist: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    simulation_dir = os.path.dirname(args.config) or "."
    wait_for_commands = not args.no_wait
    

    init_logging_for_simulation(simulation_dir)
    

    log_manager = SimulationLogManager(simulation_dir)
    twitter_logger = log_manager.get_twitter_logger()
    reddit_logger = log_manager.get_reddit_logger()
    
    log_manager.info("=" * 60)
    log_manager.info("OASIS dual-platform parallel simulation")
    log_manager.info(f"Configuration file: {args.config}")
    log_manager.info(f"Simulation ID: {config.get('simulation_id', 'unknown')}")
    log_manager.info(f"Wait mode: {'Enabled' if wait_for_commands else 'Disabled'}")
    log_manager.info("=" * 60)
    
    time_config = config.get("time_config", {})
    total_hours = time_config.get('total_simulation_hours', 72)
    minutes_per_round = time_config.get('minutes_per_round', 30)
    config_total_rounds = (total_hours * 60) // minutes_per_round
    
    log_manager.info(f"Simulation parameters:")
    log_manager.info(f"  - Total simulation duration: {total_hours} hours")
    log_manager.info(f"  - Time per round: {minutes_per_round} minutes")
    log_manager.info(f"  - Configured total rounds: {config_total_rounds}")
    if args.max_rounds:
        log_manager.info(f"  - Maximum rounds limit: {args.max_rounds}")
        if args.max_rounds < config_total_rounds:
            log_manager.info(f"  - Actual execution rounds: {args.max_rounds} (Truncated)")
    log_manager.info(f"  - Number of agents: {len(config.get('agent_configs', []))}")
    
    log_manager.info("Log structure:")
    log_manager.info(f"  - Main log: simulation.log")
    log_manager.info(f"  - Twitter actions: twitter/actions.jsonl")
    log_manager.info(f"  - Reddit actions: reddit/actions.jsonl")
    log_manager.info("=" * 60)
    
    start_time = datetime.now()
    

    twitter_result: Optional[PlatformSimulation] = None
    reddit_result: Optional[PlatformSimulation] = None
    
    if args.twitter_only:
        twitter_result = await run_twitter_simulation(config, simulation_dir, twitter_logger, log_manager, args.max_rounds)
    elif args.reddit_only:
        reddit_result = await run_reddit_simulation(config, simulation_dir, reddit_logger, log_manager, args.max_rounds)
    else:

        results = await asyncio.gather(
            run_twitter_simulation(config, simulation_dir, twitter_logger, log_manager, args.max_rounds),
            run_reddit_simulation(config, simulation_dir, reddit_logger, log_manager, args.max_rounds),
        )
        twitter_result, reddit_result = results
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    log_manager.info("=" * 60)
    log_manager.info(f"Simulation loop completed! Total time: {total_elapsed:.1f} seconds")
    

    if wait_for_commands:
        log_manager.info("")
        log_manager.info("=" * 60)
        log_manager.info("Enter wait mode - environment keeps running")
        log_manager.info("Supported commands: interview, batch_interview, close_env")
        log_manager.info("=" * 60)
        

        ipc_handler = ParallelIPCHandler(
            simulation_dir=simulation_dir,
            twitter_env=twitter_result.env if twitter_result else None,
            twitter_agent_graph=twitter_result.agent_graph if twitter_result else None,
            reddit_env=reddit_result.env if reddit_result else None,
            reddit_agent_graph=reddit_result.agent_graph if reddit_result else None,
            twitter_memory=twitter_result.memory_store if twitter_result else None,
            reddit_memory=reddit_result.memory_store if reddit_result else None,
            interview_facts={
                "twitter": twitter_result.interview_facts_block() if twitter_result else None,
                "reddit": reddit_result.interview_facts_block() if reddit_result else None,
            },
        )
        ipc_handler.update_status("alive")
        

        try:
            while not _shutdown_event.is_set():
                should_continue = await ipc_handler.process_commands()
                if not should_continue:
                    break

                try:
                    await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                    break
                except asyncio.TimeoutError:
                    pass
        except KeyboardInterrupt:
            print("\nReceived interrupt signal")
        except asyncio.CancelledError:
            print("\nTask was cancelled")
        except Exception as e:
            print(f"\nError processing command: {e}")
        
        log_manager.info("\nClose environment...")
        ipc_handler.update_status("stopped")
    

    if twitter_result and twitter_result.env:
        await twitter_result.env.close()
        log_manager.info("[Twitter] ")
    
    if reddit_result and reddit_result.env:
        await reddit_result.env.close()
        log_manager.info("[Reddit] ")
    
    log_manager.info("=" * 60)
    log_manager.info(f"All completed!")
    log_manager.info(f"Log files:")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'simulation.log')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'twitter', 'actions.jsonl')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'reddit', 'actions.jsonl')}")
    log_manager.info("=" * 60)

def setup_signal_handlers():
    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\nReceived {sig_name} signal, exiting...")
        
        if not _cleanup_done:
            _cleanup_done = True

            if _shutdown_event:
                _shutdown_event.set()
        

        else:
            print("Force exit...")
            sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    setup_signal_handlers()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted")
    except SystemExit:
        pass
    finally:

        try:
            from multiprocessing import resource_tracker
            resource_tracker._resource_tracker._stop()
        except Exception:
            pass
        print("Simulation process exited")
