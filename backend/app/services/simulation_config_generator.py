"""
Simulation Config Intelligent Generator
Uses LLM to automatically generate detailed simulation parameters based on
simulation requirements, document content, and knowledge graph information.
Fully automated with no manual parameter setup required.

Uses a step-by-step generation strategy to avoid failures from generating
overly long content in a single pass:
1. Generate time configuration
2. Extract the scenario brief (start time, place, dated facts) from the document
3. Generate Agent configurations in batches
4. Generate event configuration, written for named agents from the roster
5. Generate platform configuration
"""

import json
import math
import os
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import ollama_extra_body
from .entity_reader import EntityNode
from .scenario import date_in_source, ground_fact, numbers_in_source, parse_datetime

logger = get_logger('mirofish.simulation_config')

@dataclass
class AgentActivityConfig:
    """Activity configuration for a single Agent"""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str

    # Activity level (0.0-1.0)
    activity_level: float = 0.5  # Overall activity level

    # Posting frequency (expected posts per hour)
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0

    # Active hours (24-hour format, 0-23)
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))

    # Response speed (reaction delay to trending events, in simulated minutes)
    response_delay_min: int = 5
    response_delay_max: int = 60

    # Sentiment bias (-1.0 to 1.0, negative to positive)
    sentiment_bias: float = 0.0

    # Stance (attitude towards specific topics)
    stance: str = "neutral"  # supportive, opposing, neutral, observer

    # Influence weight (determines probability of posts being seen by other Agents)
    influence_weight: float = 1.0

    # Behavioral archetype (lurker, amplifier, contributor, debater)
    # Used by the System One router to decide actions without an LLM call
    archetype: str = "contributor"


@dataclass
class TimeSimulationConfig:
    """Time simulation configuration (based on typical UK activity patterns)"""
    # Total simulation duration (in simulated hours)
    total_simulation_hours: int = 72  # Default: simulate 72 hours (3 days)

    # Time per round (in simulated minutes) - default 60 minutes (1 hour), accelerated time flow
    minutes_per_round: int = 60

    # Range of Agents activated per hour
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20

    # Peak hours (evening 18-21, most active time)
    peak_hours: List[int] = field(default_factory=lambda: [18, 19, 20, 21])
    peak_activity_multiplier: float = 1.5

    # Off-peak hours (midnight to early morning 0-5, almost no activity)
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  # Extremely low activity in early hours

    # Morning hours
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4

    # Work hours
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Event configuration"""
    # Initial events (trigger events at simulation start)
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)

    # Scheduled events (events triggered at specific times)
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)

    # Trending topic keywords
    hot_topics: List[str] = field(default_factory=list)

    # Narrative direction
    narrative_direction: str = ""

    # Round count the scheduled events were planned against. The runner
    # rescales trigger_round when the simulation runs for a different count.
    planned_rounds: int = 0


@dataclass
class PlatformConfig:
    """Platform-specific configuration"""
    platform: str  # twitter or reddit

    # Recommendation algorithm weights
    recency_weight: float = 0.4  # Recency/freshness
    popularity_weight: float = 0.3  # Popularity
    relevance_weight: float = 0.3  # Relevance

    # Viral threshold (number of interactions before triggering spread)
    viral_threshold: int = 10

    # Echo chamber strength (degree of similar viewpoint clustering)
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Complete simulation parameter configuration"""
    # Basic information
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    # Time configuration
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)

    # Agent configuration list
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)

    # Event configuration
    event_config: EventConfig = field(default_factory=EventConfig)

    # Scenario brief: {"start", "location", "facts": [{"text", "known_from"}]},
    # read by the runner's scenario clock and fact sheet (scenario.py)
    scenario: Dict[str, Any] = field(default_factory=dict)

    # Platform configuration
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None

    # LLM configuration
    llm_model: str = ""
    llm_base_url: str = ""

    # Generation metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # LLM reasoning explanation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "scenario": self.scenario,
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    Simulation Config Intelligent Generator

    Uses LLM to analyse simulation requirements, document content, and
    knowledge graph entity information to automatically generate optimal
    simulation parameter configurations.

    Uses a step-by-step generation strategy:
    1. Generate time and event configurations (lightweight)
    2. Generate Agent configurations in batches (10-20 per batch)
    3. Generate platform configuration
    """

    # Maximum context character count
    MAX_CONTEXT_LENGTH = 50000
    # Number of Agents per batch
    AGENTS_PER_BATCH = 15

    # Context truncation lengths per step (character count)
    TIME_CONFIG_CONTEXT_LENGTH = 10000   # Time configuration
    EVENT_CONFIG_CONTEXT_LENGTH = 8000   # Event configuration
    ENTITY_SUMMARY_LENGTH = 300          # Entity summary
    AGENT_SUMMARY_LENGTH = 300           # Entity summary in Agent config
    ENTITIES_PER_TYPE_DISPLAY = 20       # Number of entities displayed per type

    # Activity range for graph entities, which all post in the institutional
    # voice. Run 10 gave "Local businesses" 0.9 and a chemical works 0.8, and
    # 8 institutions wrote 59 of ~95 texts, drowning out the residents.
    INSTITUTION_ACTIVITY = (0.1, 0.3)
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        synthetic_personas: Optional[List[Dict[str, Any]]] = None,
    ) -> SimulationParameters:
        """
        Intelligently generate complete simulation configuration (step by step)

        Args:
            simulation_id: Simulation ID
            project_id: Project ID
            graph_id: Graph ID
            simulation_requirement: Simulation requirement description
            document_text: Original document content
            entities: Filtered entity list
            enable_twitter: Whether to enable Twitter
            enable_reddit: Whether to enable Reddit
            progress_callback: Progress callback function(current_step, total_steps, message)
            synthetic_personas: Synthetic individual personas to include in the simulation

        Returns:
            SimulationParameters: Complete simulation parameters
        """
        synthetic_personas = synthetic_personas or []
        total_agents = len(entities) + len(synthetic_personas)
        logger.info(f"Starting intelligent simulation config generation: simulation_id={simulation_id}, entities={len(entities)}, synthetic={len(synthetic_personas)}")

        # Calculate total number of steps
        num_entity_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        num_synthetic_batches = math.ceil(len(synthetic_personas) / self.AGENTS_PER_BATCH) if synthetic_personas else 0
        num_batches = num_entity_batches + num_synthetic_batches
        total_steps = 4 + num_batches  # Time + scenario + N Agent batches + events + platform
        current_step = 0
        
        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")
        
        # 1. Build base context information
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )
        
        reasoning_parts = []
        
        # ========== Step 1: Generate time configuration ==========
        report_progress(1, "Generating time configuration...")
        # Activity bounds are per agent in the simulation, synthetic personas included
        time_config_result = self._generate_time_config(context, total_agents)
        time_config = self._parse_time_config(time_config_result, total_agents)
        reasoning_parts.append(f"Time config: {time_config_result.get('reasoning', 'Success')}")

        # ========== Step 2: Scenario brief (start time, place, dated facts) ==========
        report_progress(2, "Extracting scenario timeline and facts...")
        scenario = self._generate_scenario(document_text, simulation_requirement, time_config)
        reasoning_parts.append(
            f"Scenario: start={scenario.get('start') or 'undated'}, {len(scenario.get('facts', []))} facts"
        )

        # ========== Steps 3-N: Generate Agent configurations in batches ==========
        all_agent_configs = []
        for batch_idx in range(num_entity_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]

            report_progress(
                3 + batch_idx,
                f"Generating Agent configs ({start_idx + 1}-{end_idx}/{total_agents})..."
            )

            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement
            )
            all_agent_configs.extend(batch_configs)

        # Generate configs for synthetic individual personas
        personas_by_agent: Dict[int, Dict[str, Any]] = {}
        if synthetic_personas:
            synthetic_start_idx = len(entities)
            for batch_idx in range(num_synthetic_batches):
                start_idx = batch_idx * self.AGENTS_PER_BATCH
                end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(synthetic_personas))
                batch_personas = synthetic_personas[start_idx:end_idx]
                agent_start_idx = synthetic_start_idx + start_idx

                report_progress(
                    3 + num_entity_batches + batch_idx,
                    f"Generating synthetic agent configs ({agent_start_idx + 1}-{agent_start_idx + len(batch_personas)}/{total_agents})..."
                )

                batch_configs = self._generate_synthetic_agent_configs_batch(
                    context=context,
                    personas=batch_personas,
                    start_idx=agent_start_idx,
                    simulation_requirement=simulation_requirement,
                )
                all_agent_configs.extend(batch_configs)
                for i, persona in enumerate(batch_personas):
                    personas_by_agent[agent_start_idx + i] = persona

        reasoning_parts.append(f"Agent config: successfully generated {len(all_agent_configs)}")

        # ========== Events, written for named agents ==========
        report_progress(3 + num_batches, "Generating initial posts and scheduled events...")
        # Use the actual max rounds the simulation will run (from env config)
        actual_max_rounds = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
        computed_rounds = time_config.total_simulation_hours // max(1, time_config.minutes_per_round // 60)
        total_rounds = min(computed_rounds, actual_max_rounds) if actual_max_rounds > 0 else computed_rounds
        roster = self._build_roster(all_agent_configs, personas_by_agent)
        event_config_result = self._generate_event_config(
            simulation_requirement, document_text, roster, scenario, time_config, total_rounds
        )
        event_config = self._parse_event_config(event_config_result, roster, document_text, scenario)
        event_config.planned_rounds = total_rounds
        reasoning_parts.append(
            f"Event config: {len(event_config.initial_posts)} initial posts, "
            f"{len(event_config.scheduled_events)} scheduled events"
        )

        # ========== Final step: Generate platform configuration ==========
        report_progress(total_steps, "Generating platform configuration...")
        twitter_config = None
        reddit_config = None
        
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )
        
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )
        
        # Build final parameters
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            scenario=scenario,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )
        
        logger.info(f"Simulation config generation complete: {len(params.agent_configs)} Agent configs")
        
        return params
    
    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:
        """Build LLM context, truncated to maximum length"""

        # Entity summary
        entity_summary = self._summarize_entities(entities)

        # Build context
        context_parts = [
            f"## Simulation Requirement\n{simulation_requirement}",
            f"\n## Entity Information ({len(entities)} entities)\n{entity_summary}",
        ]

        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500  # Reserve 500 chars

        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(document truncated)"
            context_parts.append(f"\n## Original Document Content\n{doc_text}")
        
        return "\n".join(context_parts)
    
    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Generate entity summary"""
        lines = []

        # Group by type
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)
        
        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)} entities)")
            # Use configured display count and summary length
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... and {len(type_entities) - display_count} more")
        
        return "\n".join(lines)
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """LLM call with retry logic and JSON repair"""
        
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    extra_body=ollama_extra_body(self.base_url),
                    temperature=0.7 - (attempt * 0.1)  # Lower temperature on each retry
                    # No max_tokens set, let LLM generate freely
                )
                
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                
                # Check if output was truncated
                if finish_reason == 'length':
                    logger.warning(f"LLM output truncated (attempt {attempt+1})")
                    content = self._fix_truncated_json(content)
                
                # Try to parse JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse failed (attempt {attempt+1}): {str(e)[:80]}")

                    # Try to fix JSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed
                    
                    last_error = e
                    
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))
        
        raise last_error or Exception("LLM call failed")
    
    def _fix_truncated_json(self, content: str) -> str:
        """Fix truncated JSON"""
        content = content.strip()

        # Count unclosed brackets
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # Check for unclosed strings
        if content and content[-1] not in '",}]':
            content += '"'
        
        # Close brackets
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Try to fix configuration JSON"""
        import re

        # Fix truncated content
        content = self._fix_truncated_json(content)
        
        # Extract JSON portion
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            # Remove newlines within strings
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s
            
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)
            
            try:
                return json.loads(json_str)
            except:
                # Try removing all control characters
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass
        
        return None
    
    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """Generate time configuration"""
        # Use configured context truncation length
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]

        # Calculate maximum allowed value (90% of agent count, within the compute budget)
        max_agents_allowed = min(max(1, int(num_entities * 0.9)), self._agents_per_hour_budget())

        prompt = f"""Based on the following simulation requirements, generate a time simulation configuration.

{context_truncated}

## Task
Generate a time configuration JSON.

### General principles (for reference only; adjust flexibly based on the specific event and participant demographics):
- Activity patterns should follow typical daily routines
- 0-5 AM: almost no activity (activity multiplier 0.05)
- 6-8 AM: gradually increasing activity (activity multiplier 0.4)
- 9 AM-5 PM: moderate activity during work hours (activity multiplier 0.7)
- 6-9 PM: evening peak hours (activity multiplier 1.5)
- 10-11 PM: activity declining (activity multiplier 0.5)
- General pattern: low activity late night, gradual morning increase, moderate during work, evening peak
- **Important**: The example values below are for reference only. You should adjust the specific time periods based on the nature of the event and the characteristics of participants.
  - For example: student demographics may peak at 9-11 PM; media active all day; official institutions only during work hours
  - For example: breaking news may trigger late-night discussion, so off_peak_hours can be shortened

### Return JSON format (no markdown)

Example:
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 3,
    "agents_per_hour_max": {max_agents_allowed},
    "peak_hours": [18, 19, 20, 21],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17],
    "reasoning": "Explanation of the time configuration for this event"
}}

Field descriptions:
- total_simulation_hours (int): Total simulation duration, 24-168 hours; shorter for breaking events, longer for ongoing topics
- minutes_per_round (int): Duration per round, 30-120 minutes, recommended 60 minutes
- agents_per_hour_min (int): Minimum Agents activated per hour (range: 1-{max_agents_allowed})
- agents_per_hour_max (int): Maximum Agents activated per hour (range: 1-{max_agents_allowed})
- peak_hours (int array): Peak hours, adjust based on participant demographics
- off_peak_hours (int array): Off-peak hours, typically late night/early morning
- morning_hours (int array): Morning hours
- work_hours (int array): Work hours
- reasoning (string): Brief explanation of why this configuration was chosen"""

        system_prompt = "You are a social media simulation expert. Respond in English only. Return pure JSON format. Time configuration should follow typical daily activity patterns."
        
        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Time config LLM generation failed: {e}, using default config")
            return self._get_default_time_config(num_entities)

    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Get default time configuration (typical daily activity patterns)"""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  # 1 hour per round, accelerated time flow
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [18, 19, 20, 21],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17],
            "reasoning": "Using default activity pattern configuration (1 hour per round)"
        }
    
    @staticmethod
    def _agents_per_hour_budget() -> int:
        """Most agents activated per simulated hour: a compute budget, not a realism setting.

        Every active agent is one decision per round, and on a 16GB M2 Pro with
        3 decode slots that sets the simulation's wall time. Raise
        SIM_MAX_AGENTS_PER_HOUR on faster hardware.
        """
        return max(1, int(os.environ.get('SIM_MAX_AGENTS_PER_HOUR', '8')))

    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
        """Parse time configuration result and validate agents_per_hour values do not exceed total agent count"""
        # Get raw values
        agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
        agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))
        
        # Validate and correct: ensure values do not exceed total agent count
        if agents_per_hour_min > num_entities:
            logger.warning(f"agents_per_hour_min ({agents_per_hour_min}) exceeds total Agent count ({num_entities}), corrected")
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(f"agents_per_hour_max ({agents_per_hour_max}) exceeds total Agent count ({num_entities}), corrected")
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

        budget = self._agents_per_hour_budget()
        if agents_per_hour_max > budget:
            logger.info(f"agents_per_hour_max {agents_per_hour_max} capped to SIM_MAX_AGENTS_PER_HOUR={budget}")
            agents_per_hour_max = budget
            agents_per_hour_min = min(agents_per_hour_min, budget)

        # Ensure min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= max, corrected to {agents_per_hour_min}")
        
        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),  # Default 1 hour per round
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [18, 19, 20, 21]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,  # Almost no one active in early hours
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )
    
    # Document characters given to the scenario extraction (qwen3:8b runs with an 8K context)
    SCENARIO_DOC_LENGTH = 12000
    MIN_FACTS = 8
    MAX_FACTS = 14
    MAX_INITIAL_POSTS = 6

    def _generate_scenario(
        self,
        document_text: str,
        simulation_requirement: str,
        time_config: TimeSimulationConfig,
    ) -> Dict[str, Any]:
        """Extract the scenario brief: start time, place and dated facts.

        Run 10 agents had no clock and no fact sheet, so qwen3:8b invented
        dates, places and links. The brief gives the runner both (scenario.py).
        Every fact is checked against the document (ground_fact) and every
        date must appear in it (date_in_source); anything that fails is
        dropped, not repaired. Without a document date the run is undated.
        """
        doc = (document_text or "")[:self.SCENARIO_DOC_LENGTH]
        if not doc.strip():
            logger.warning("No document text: the scenario is undated and has no facts")
            return {"start": None, "location": "", "facts": []}

        prompt = f"""Read the document and extract the situation a social media simulation starts from.

## Document
{doc}

## Simulation requirement
{simulation_requirement}

## Task
The simulation covers {time_config.total_simulation_hours} hours from its start. Return JSON (no markdown):
{{
    "start": "YYYY-MM-DD HH:MM",
    "location": "place name",
    "facts": [
        {{"text": "One fact in one sentence.", "known_from": "YYYY-MM-DD HH:MM"}},
        {{"text": "A fact public before the start.", "known_from": null}}
    ]
}}

Fields:
- start: the date and time of the first public announcement (a notice, alert or statement to the public), as the document dates it. Internal findings the public was not told about happen before the start. null if the document gives no calendar date.
- location: the town or area, written exactly as in the document.
- facts: {self.MIN_FACTS} to {self.MAX_FACTS} facts, each one sentence under 25 words.
  - Copy names, numbers, places and dates exactly as the document writes them. Add nothing the document does not say.
  - Cover what happened, who is affected, the official advice, where to get help, the cause, and each later development.
  - known_from: when the fact became public, from the document's timeline. Use the time of day if the document gives one, otherwise 09:00. null if it was public before the start or the document does not say when."""

        system_prompt = ("You extract facts from documents. Respond in English only. Return pure JSON. "
                         "Never add a name, number, date or place that is not in the document.")
        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Scenario extraction failed: {e}; the scenario is undated and has no facts")
            return {"start": None, "location": "", "facts": []}
        return self._parse_scenario(result, document_text)

    @staticmethod
    def _parse_scenario(result: Dict[str, Any], document_text: str) -> Dict[str, Any]:
        """Validate the extracted brief against the document."""
        start = parse_datetime(result.get("start"))
        if start is not None and not date_in_source(start, document_text):
            logger.warning(f"Scenario start {result.get('start')!r} is not a date in the document; run is undated")
            start = None

        location = (result.get("location") or "").strip() if isinstance(result.get("location"), str) else ""
        if location and location.lower() not in document_text.lower():
            logger.warning(f"Scenario location {location!r} is not in the document; dropped")
            location = ""

        facts: List[Dict[str, Any]] = []
        seen = set()
        for f in result.get("facts") or []:
            if not isinstance(f, dict):
                continue
            text = re.sub(r"\s+", " ", (f.get("text") or "")).strip() if isinstance(f.get("text"), str) else ""
            if not text or text.lower() in seen:
                continue
            if not ground_fact(text, document_text):
                logger.warning(f"Scenario fact dropped, a number or name is not in the document: {text!r}")
                continue
            raw_when = f.get("known_from")
            when = parse_datetime(raw_when)
            if raw_when and when is None:
                logger.warning(f"Scenario fact dropped, unreadable known_from {raw_when!r}: {text!r}")
                continue
            if when is not None:
                if start is None:
                    logger.warning(f"Scenario fact dropped, dated fact in an undated run: {text!r}")
                    continue
                if not date_in_source(when, document_text):
                    logger.warning(f"Scenario fact dropped, known_from {raw_when!r} is not a date in the document: {text!r}")
                    continue
                if when <= start:
                    when = None  # public from the start
            seen.add(text.lower())
            facts.append({"text": text, "known_from": when.strftime("%Y-%m-%d %H:%M") if when else None})

        if len(facts) > SimulationConfigGenerator.MAX_FACTS:
            facts = facts[:SimulationConfigGenerator.MAX_FACTS]
        if len(facts) < SimulationConfigGenerator.MIN_FACTS:
            logger.warning(f"Scenario has only {len(facts)} grounded facts (wanted {SimulationConfigGenerator.MIN_FACTS}+)")
        dated = sum(1 for f in facts if f["known_from"])
        logger.info(f"Scenario: start={start or 'undated'}, location={location or '-'}, "
                    f"{len(facts)} facts ({dated} revealed during the run)")
        return {
            "start": start.strftime("%Y-%m-%d %H:%M") if start else None,
            "location": location,
            "facts": facts,
        }

    @staticmethod
    def _build_roster(
        agent_configs: List[AgentActivityConfig],
        personas_by_agent: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Agents that can author a post, labelled for the event prompt.

        Graph entities are institutions (the runner gives them the institutional
        voice); synthetic personas are individuals.
        """
        roster = []
        for a in agent_configs:
            persona = personas_by_agent.get(a.agent_id)
            if persona is None:
                kind, detail = "institution", a.entity_type
            else:
                kind = "individual"
                detail = persona.get("role") or a.entity_type.replace("Synthetic_", "").replace("_", " ")
                if persona.get("age"):
                    detail = f"{detail}, {persona['age']}"
            roster.append({
                "agent_id": a.agent_id,
                "name": a.entity_name,
                "entity_type": a.entity_type,
                "kind": kind,
                "role": (persona or {}).get("role", ""),
                "label": f"{a.entity_name} ({kind}: {detail})",
            })
        return roster

    def _generate_event_config(
        self,
        simulation_requirement: str,
        document_text: str,
        roster: List[Dict[str, Any]],
        scenario: Dict[str, Any],
        time_config: TimeSimulationConfig,
        total_rounds: int,
    ) -> Dict[str, Any]:
        """Generate seed posts and scheduled posts, each by a named roster agent.

        Run 10 asked for a poster_type and matched it to an agent afterwards:
        a resident's "I have a child under five" went to Bristol Water and one
        agent posted four of the nine seeds. Naming the poster in the prompt
        lets the model write in that agent's voice. In a dated run each
        scheduled event is a dated fact from the scenario, posted at its time.
        """
        start = parse_datetime(scenario.get("start"))
        facts = scenario.get("facts") or []
        known = [f["text"] for f in facts if not f.get("known_from")]
        later = [f for f in facts if f.get("known_from")]
        roster_lines = "\n".join(f"- {r['label']}" for r in roster)

        if start is not None:
            end = start + timedelta(hours=time_config.total_simulation_hours)
            timeline = (
                f"The simulation starts {start:%d %B %Y %H:%M} and ends {end:%d %B %Y %H:%M}.\n"
                "Known at the start:\n" + "\n".join(f"- {t}" for t in known) +
                "\nBecomes public later:\n" +
                ("\n".join(f"- [{f['known_from']}] {f['text']}" for f in later) or "- (nothing)")
            )
            event_shape = ('{{"at": "YYYY-MM-DD HH:MM", "description": "Brief event description", '
                           '"poster": "Name from the roster", "content": "Post text"}}').format()
            event_rule = ("scheduled_events: one post for each development under \"Becomes public later\", "
                          "with \"at\" set to its time. The poster is the organisation that announced it, "
                          "or a journalist or resident reporting it.")
        else:
            timeline = "Known facts:\n" + "\n".join(f"- {f['text']}" for f in facts)
            event_shape = ('{{"trigger_round": 1, "description": "Brief event description", '
                           '"poster": "Name from the roster", "content": "Post text"}}').format()
            event_rule = (f"scheduled_events: 3 to 6 later developments from the document, spread over rounds "
                          f"1 to {total_rounds} in the order the document gives them.")

        prompt = f"""Write the opening posts and the scheduled posts for a social media simulation.

Simulation requirement: {simulation_requirement}

## Document
{(document_text or "")[:self.EVENT_CONFIG_CONTEXT_LENGTH]}

## Timeline
{timeline}

## Roster (every poster must be one of these names, copied exactly)
{roster_lines}

## Task
Return JSON (no markdown):
{{
    "hot_topics": ["keyword1", "keyword2"],
    "narrative_direction": "How the discussion is likely to develop",
    "initial_posts": [
        {{"poster": "Name from the roster", "content": "Post text"}}
    ],
    "scheduled_events": [
        {event_shape}
    ],
    "reasoning": "Brief explanation"
}}

Rules:
- initial_posts: 3 to {self.MAX_INITIAL_POSTS} posts at the start, from different posters: at least one institution and at least two individuals. Use only what is known at the start.
- {event_rule}
- An institution writes as "we" and states only facts it knows. An individual writes in the first person about their own situation.
- Every number, date, place and name must come from the document. No links unless the document gives them.
- Copy each poster's name exactly from the roster, without the part in brackets."""

        system_prompt = ("You are a public opinion analysis expert. Respond in English only. Return pure JSON. "
                         "Each poster must be a name from the roster.")
        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Event config LLM generation failed: {e}; the simulation starts without seed posts")
            return {"hot_topics": [], "narrative_direction": "", "initial_posts": [], "scheduled_events": []}

    _FIRST_PERSON = re.compile(r"\b(?:I|I'm|I've|I'd|I'll)\b|\b(?:my|me|mine|myself)\b", re.IGNORECASE)
    _CHILD_WORDS = re.compile(r"\b(?:child|children|kids?|son|daughter|baby|toddler)\b", re.IGNORECASE)
    _URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

    @classmethod
    def _post_problem(cls, content: str, document_text: str) -> Optional[str]:
        """Why a generated post cannot be used, or None."""
        if not numbers_in_source(content, document_text):
            return "a number is not in the document"
        for url in cls._URL.findall(content):
            if url.rstrip(".,)").lower() not in document_text.lower():
                return f"link {url!r} is not in the document"
        return None

    def _parse_event_config(
        self,
        result: Dict[str, Any],
        roster: List[Dict[str, Any]],
        document_text: str,
        scenario: Dict[str, Any],
    ) -> EventConfig:
        """Resolve each post's poster by name and drop posts that fail the checks.

        A poster that is not on the roster is dropped, not matched to someone
        similar. An institution given a first-person post ("my son") gets it
        moved to an individual, preferring parents for posts about children.
        """
        by_name = {self._norm_name(r["name"]): r for r in roster}
        individuals = [r for r in roster if r["kind"] == "individual"]
        # Posts the model already gave each agent, so a moved post goes to
        # someone without one of their own
        usage: Dict[int, int] = {}
        for post in (result.get("initial_posts") or []) + (result.get("scheduled_events") or []):
            r = by_name.get(self._norm_name(str(post.get("poster") or ""))) if isinstance(post, dict) else None
            if r:
                usage[r["agent_id"]] = usage.get(r["agent_id"], 0) + 1

        def resolve(post: Dict[str, Any], what: str) -> Optional[Dict[str, Any]]:
            content = (post.get("content") or "").strip() if isinstance(post.get("content"), str) else ""
            if not content:
                return None
            name = str(post.get("poster") or "")
            # The roster line itself ("Name (institution: Type)") is accepted;
            # a name with its own brackets matches first.
            author = (by_name.get(self._norm_name(name))
                      or by_name.get(self._norm_name(re.sub(r"\s*\([^)]*\)\s*$", "", name))))
            if author is None:
                logger.warning(f"{what} dropped, poster {post.get('poster')!r} is not on the roster: {content[:60]!r}")
                return None
            problem = self._post_problem(content, document_text)
            if problem:
                logger.warning(f"{what} dropped, {problem}: {content[:80]!r}")
                return None
            if author["kind"] == "institution" and self._FIRST_PERSON.search(content):
                if not individuals:
                    logger.warning(f"{what} dropped, first-person post by institution {author['name']}: {content[:60]!r}")
                    return None
                wants_parent = bool(self._CHILD_WORDS.search(content))
                candidates = sorted(
                    individuals,
                    key=lambda r: (not (wants_parent and "parent" in r["role"].lower()), usage.get(r["agent_id"], 0)),
                )
                logger.info(f"{what}: first-person post moved from {author['name']} to {candidates[0]['name']}")
                author = candidates[0]
                usage[author["agent_id"]] = usage.get(author["agent_id"], 0) + 1
            return {
                "content": content,
                "poster_name": author["name"],
                "poster_type": author["entity_type"],
                "poster_agent_id": author["agent_id"],
            }

        initial_posts = []
        for post in result.get("initial_posts") or []:
            if isinstance(post, dict) and (resolved := resolve(post, "Initial post")):
                initial_posts.append(resolved)
        if len(initial_posts) > self.MAX_INITIAL_POSTS:
            logger.info(f"{len(initial_posts)} initial posts, keeping the first {self.MAX_INITIAL_POSTS}")
            initial_posts = initial_posts[:self.MAX_INITIAL_POSTS]
        kinds = {r["kind"] for r in roster if r["agent_id"] in {p["poster_agent_id"] for p in initial_posts}}
        if kinds != {"institution", "individual"}:
            logger.warning(f"Initial posts come only from: {sorted(kinds) or 'nobody'}")

        start = parse_datetime(scenario.get("start"))
        scheduled_events = []
        for event in result.get("scheduled_events") or []:
            if not isinstance(event, dict):
                continue
            desc = str(event.get("description") or "")
            if start is not None:
                when = parse_datetime(event.get("at"))
                if when is None or not date_in_source(when, document_text):
                    logger.warning(f"Scheduled event dropped, time {event.get('at')!r} is not a document date: {desc[:60]!r}")
                    continue
                if when <= start:
                    logger.warning(f"Scheduled event dropped, {when} is not after the start {start}: {desc[:60]!r}")
                    continue
                timing = {"at": when.strftime("%Y-%m-%d %H:%M")}
            else:
                try:
                    timing = {"trigger_round": max(1, int(event.get("trigger_round")))}
                except (TypeError, ValueError):
                    logger.warning(f"Scheduled event dropped, no trigger_round: {desc[:60]!r}")
                    continue
            resolved = resolve(event, "Scheduled event")
            if resolved:
                scheduled_events.append({**timing, "description": desc, **resolved})
        scheduled_events.sort(key=lambda e: e.get("at") or e.get("trigger_round"))

        for p in initial_posts:
            logger.info(f"Initial post by {p['poster_name']} (agent {p['poster_agent_id']}): {p['content'][:60]!r}")
        for e in scheduled_events:
            logger.info(f"Scheduled event at {e.get('at') or 'round ' + str(e.get('trigger_round'))} "
                        f"by {e['poster_name']} (agent {e['poster_agent_id']}): {e['content'][:60]!r}")

        return EventConfig(
            initial_posts=initial_posts,
            scheduled_events=scheduled_events,
            hot_topics=result.get("hot_topics") or [],
            narrative_direction=result.get("narrative_direction") or "",
        )

    @staticmethod
    def _norm_name(name: str) -> str:
        return re.sub(r"\s+", " ", (name or "").strip().strip('"').lower())

    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """Generate Agent configurations in batches"""

        # Build entity information (using configured summary length)
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })
        
        prompt = f"""Based on the following information, generate social media activity configurations for each entity.

Simulation requirement: {simulation_requirement}

## Entity List
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Task
Generate activity configurations for each entity, noting:
- **Activity patterns should follow typical daily routines**: almost no activity from 0-5 AM, most active during evening 6-9 PM
- These entities are organisations and officials; individual residents are simulated separately. Every activity_level must be 0.1-0.3.
- **Official institutions** (University/GovernmentAgency): active during work hours (9-17), slow response (60-240 min), high influence (2.5-3.0)
- **Media** (MediaOutlet): active all day (8-23), fast response (5-30 min), high influence (2.0-2.5)
- **Businesses and community groups**: active during the day and early evening (8-20), moderate response (30-120 min), influence (1.0-1.5)
- **Public figures/experts**: medium-high influence (1.5-2.0)

Return JSON format (no markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <must match input>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <posting frequency>,
            "comments_per_hour": <commenting frequency>,
            "active_hours": [<list of active hours, following typical daily patterns>],
            "response_delay_min": <minimum response delay in minutes>,
            "response_delay_max": <maximum response delay in minutes>,
            "sentiment_bias": <-1.0 to 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <influence weight>
        }},
        ...
    ]
}}"""

        system_prompt = "You are a social media behaviour analysis expert. Respond in English only. Return pure JSON. Configurations should follow typical daily activity patterns."
        
        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"Agent config batch LLM generation failed: {e}, using rule-based generation")
            llm_configs = {}
        
        # Build AgentActivityConfig objects
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})
            
            # If LLM did not generate, use rule-based generation
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)
            
            activity_level = cfg.get("activity_level", 0.2)
            clamped = min(max(float(activity_level), self.INSTITUTION_ACTIVITY[0]), self.INSTITUTION_ACTIVITY[1])
            if clamped != activity_level:
                logger.info(f"{entity.name}: activity_level {activity_level} clamped to {clamped}")

            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=clamped,
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)
        
        return configs
    
    def _generate_synthetic_agent_configs_batch(
        self,
        context: str,
        personas: List[Dict[str, Any]],
        start_idx: int,
        simulation_requirement: str,
    ) -> List[AgentActivityConfig]:
        """Generate Agent activity configurations for synthetic individual personas via LLM.

        Uses persona metadata (role, archetype, emotional state, stance) to prompt
        the LLM for realistic, differentiated activity parameters.  Falls back to
        archetype-based defaults if the LLM call fails.
        """

        persona_list = []
        for i, p in enumerate(personas):
            persona_list.append({
                "agent_id": start_idx + i,
                "name": p["name"],
                "role": p["role"],
                "age": p["age"],
                "archetype": p["archetype"],
                "emotional_state": p["emotional_state"],
                "stance": p["stance"],
                "susceptible_to_misinfo": p.get("susceptible_to_misinfo", False),
            })

        prompt = f"""Based on the following information, generate realistic social media activity configurations for each INDIVIDUAL person (not an institution).

Simulation requirement: {simulation_requirement}

## Individual Personas
```json
{json.dumps(persona_list, ensure_ascii=False, indent=2)}
```

## Task
Generate activity configurations for each individual person. These are REAL PEOPLE, not organizations. Their behavior should follow empirical social media patterns:

- **Lurker archetype**: Very low activity (0.05-0.15), rarely posts, mostly reads. Posts 0-0.1/hour, comments 0.1-0.3/hour. DO_NOTHING most of the time.
- **Amplifier archetype**: Moderate activity (0.3-0.5), mostly shares/reposts others' content. Posts 0.1-0.3/hour, comments 0.3-0.8/hour.
- **Contributor archetype**: High activity (0.5-0.8), creates original content. Posts 0.3-0.8/hour, comments 0.5-1.0/hour.
- **Debater archetype**: High activity (0.6-0.9), engages in arguments and replies. Posts 0.2-0.5/hour, comments 1.0-2.0/hour. Very responsive.

Activity patterns by role:
- **Parents**: Active mornings (7-9), lunch (12-13), and evenings (19-22). Not active during work hours.
- **Students**: Active late morning (10-12), afternoons, and late evenings (20-24). Night owls.
- **Journalists**: Active early morning through evening (7-22). Very fast response times (1-10 min).
- **Local residents**: Sporadic, mainly evenings (18-22). Low overall activity.

Emotional state affects behavior:
- **angry** people post MORE frequently and with stronger sentiment bias (negative)
- **worried/fearful** people comment more than they post (seeking reassurance)
- **calm/curious** people have balanced activity and neutral sentiment

Influence weight for individuals should be LOW (0.3-1.2) — they are not institutions.

IMPORTANT: Make each person DISTINCT. A 52-year-old angry parent posts very differently from a 34-year-old calm journalist. Vary the parameters meaningfully.

Return JSON format (no markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <must match input>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <posting frequency>,
            "comments_per_hour": <commenting frequency>,
            "active_hours": [<list of active hours>],
            "response_delay_min": <min delay in minutes>,
            "response_delay_max": <max delay in minutes>,
            "sentiment_bias": <-1.0 to 1.0>,
            "stance": "<supportive/opposing/neutral/observer/skeptical/fearful/angry>",
            "influence_weight": <influence weight, typically 0.3-1.2 for individuals>
        }},
        ...
    ]
}}"""

        system_prompt = (
            "You are a social media behaviour analyst specializing in crisis communication dynamics. "
            "You understand how different demographics behave online during public health and environmental crises. "
            "Respond in English only. Return pure JSON. "
            "Make each person's config DISTINCT — avoid giving everyone the same parameters."
        )

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"Synthetic agent config LLM generation failed: {e}, using archetype-based fallback")
            llm_configs = {}

        configs = []
        for i, persona in enumerate(personas):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})

            if not cfg:
                cfg = self._synthetic_config_fallback(persona)

            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=f"synthetic_{agent_id}",
                entity_name=persona["name"],
                entity_type=f"Synthetic_{persona.get('template', 'individual')}",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.3),
                comments_per_hour=cfg.get("comments_per_hour", 0.8),
                active_hours=cfg.get("active_hours", list(range(18, 23))),
                response_delay_min=cfg.get("response_delay_min", 1),
                response_delay_max=cfg.get("response_delay_max", 15),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", persona.get("stance", "neutral")),
                influence_weight=cfg.get("influence_weight", 0.8),
                archetype=persona.get("archetype", "contributor"),
            )
            configs.append(config)

        return configs

    def _synthetic_config_fallback(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """Archetype-based fallback config for a synthetic persona when LLM fails."""
        archetype = persona.get("archetype", "contributor")
        role = persona.get("role", "").lower()
        emotional = persona.get("emotional_state", "calm")

        # Base config by archetype
        archetype_defaults = {
            "lurker": {
                "activity_level": 0.1,
                "posts_per_hour": 0.05,
                "comments_per_hour": 0.15,
                "response_delay_min": 10,
                "response_delay_max": 60,
            },
            "amplifier": {
                "activity_level": 0.4,
                "posts_per_hour": 0.15,
                "comments_per_hour": 0.5,
                "response_delay_min": 3,
                "response_delay_max": 20,
            },
            "contributor": {
                "activity_level": 0.6,
                "posts_per_hour": 0.5,
                "comments_per_hour": 0.8,
                "response_delay_min": 2,
                "response_delay_max": 15,
            },
            "debater": {
                "activity_level": 0.7,
                "posts_per_hour": 0.3,
                "comments_per_hour": 1.5,
                "response_delay_min": 1,
                "response_delay_max": 10,
            },
        }

        cfg = archetype_defaults.get(archetype, archetype_defaults["contributor"]).copy()

        # Active hours by role
        if "parent" in role:
            cfg["active_hours"] = [7, 8, 9, 12, 13, 19, 20, 21, 22]
        elif "student" in role:
            cfg["active_hours"] = [10, 11, 12, 13, 14, 20, 21, 22, 23]
        elif "journalist" in role:
            cfg["active_hours"] = list(range(7, 23))
        else:
            cfg["active_hours"] = [9, 10, 11, 12, 18, 19, 20, 21, 22]

        # Emotional state affects sentiment
        sentiment_map = {"angry": -0.6, "fearful": -0.3, "worried": -0.2, "calm": 0.0, "curious": 0.1}
        cfg["sentiment_bias"] = sentiment_map.get(emotional, 0.0)
        cfg["stance"] = persona.get("stance", "neutral")
        cfg["influence_weight"] = 0.8

        return cfg

    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """Generate a single Agent config based on rules (typical daily patterns)"""
        entity_type = (entity.get_entity_type() or "Unknown").lower()

        if entity_type in ["university", "governmentagency", "ngo"]:
            # Official institutions: active during work hours, low frequency, high influence
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),  # 9:00-17:59
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet"]:
            # Media: active all day, moderate frequency, high influence
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),  # 7:00-23:59
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:
            # Experts/professors: active during work + evening hours, moderate frequency
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),  # 8:00-21:59
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:
            # Students: mainly evening, high frequency
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # Morning + evening
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni"]:
            # Alumni: mainly evening
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  # Lunch break + evening
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:
            # General public: evening peak
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # Daytime + evening
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
    

