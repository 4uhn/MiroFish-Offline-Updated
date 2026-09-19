"""
Simulation Config Intelligent Generator
Uses LLM to automatically generate detailed simulation parameters based on
simulation requirements, document content, and knowledge graph information.
Fully automated with no manual parameter setup required.

Uses a step-by-step generation strategy to avoid failures from generating
overly long content in a single pass:
1. Generate time configuration
2. Generate event configuration
3. Generate Agent configurations in batches
4. Generate platform configuration
"""

import json
import math
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .entity_reader import EntityNode

logger = get_logger('mirofish.simulation_config')

# Default activity schedule configuration (UK timezone)
DEFAULT_ACTIVITY_SCHEDULE = {
    # Dead hours (almost no activity)
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # Morning hours (gradually waking up)
    "morning_hours": [6, 7, 8],
    # Work hours
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17],
    # Evening peak hours (most active)
    "peak_hours": [18, 19, 20, 21],
    # Night hours (activity declining)
    "night_hours": [22, 23],
    # Activity multipliers
    "activity_multipliers": {
        "dead": 0.05,      # Late night, almost no one active
        "morning": 0.4,    # Morning, gradually increasing
        "work": 0.7,       # Work hours, moderate activity
        "peak": 1.5,       # Evening peak
        "night": 0.5       # Late night, declining
    }
}


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
        total_steps = 3 + num_batches  # Time config + Event config + N Agent batches + Platform config
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
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(f"Time config: {time_config_result.get('reasoning', 'Success')}")

        # ========== Step 2: Generate event configuration ==========
        report_progress(2, "Generating event configuration and trending topics...")
        # Use the actual max rounds the simulation will run (from env config)
        actual_max_rounds = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
        computed_rounds = time_config.total_simulation_hours // max(1, time_config.minutes_per_round // 60)
        total_rounds = min(computed_rounds, actual_max_rounds) if actual_max_rounds > 0 else computed_rounds
        event_config_result = self._generate_event_config(context, simulation_requirement, entities, total_rounds)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"Event config: {event_config_result.get('reasoning', 'Success')}")

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

        reasoning_parts.append(f"Agent config: successfully generated {len(all_agent_configs)}")

        # ========== Assign publisher Agents to initial posts ==========
        logger.info("Assigning suitable publisher Agents to initial posts...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        scheduled_count = len([e for e in event_config.scheduled_events if e.get("poster_agent_id") is not None])
        reasoning_parts.append(f"Post assignment: {assigned_count} initial + {scheduled_count} scheduled events assigned")

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
        import re
        
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

        # Calculate maximum allowed value (90% of agent count)
        max_agents_allowed = max(1, int(num_entities * 0.9))

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
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
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
    
    def _generate_event_config(
        self,
        context: str,
        simulation_requirement: str,
        entities: List[EntityNode],
        total_rounds: int = 10
    ) -> Dict[str, Any]:
        """Generate event configuration"""

        # Get available entity types for LLM reference
        entity_types_available = list(set(
            e.get_entity_type() or "Unknown" for e in entities
        ))

        # List representative entity names for each type
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)

        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}"
            for t, examples in type_examples.items()
        ])

        # Use configured context truncation length
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""Based on the following simulation requirements, generate an event configuration.

Simulation requirement: {simulation_requirement}

{context_truncated}

## Available Entity Types and Examples
{type_info}

## Simulation Structure
The simulation will run for {total_rounds} rounds total. Use trigger_round to schedule events at specific rounds during the simulation to create a compelling narrative arc.

## Task
Generate an event configuration JSON:
- Extract trending topic keywords
- Describe the narrative development direction
- Design initial post content; **each post must specify a poster_type (publisher type)**
- Design 3-6 scheduled events spread across the simulation rounds to drive narrative progression. Each event needs a trigger_round (1 to {total_rounds}) indicating when it fires. Space them out to create phases: early reaction, mid-simulation escalation, and late resolution/aftermath.

**Important**: poster_type must be selected from the "Available Entity Types" listed above, so that initial posts can be assigned to the appropriate Agent for publishing.
For example: official statements should be published by Official/University types, news by MediaOutlet, student opinions by Student.

Return JSON format (no markdown):
{{
    "hot_topics": ["keyword1", "keyword2", ...],
    "narrative_direction": "<description of narrative development direction>",
    "initial_posts": [
        {{"content": "Post content in English", "poster_type": "Entity type (must be from available types)"}},
        ...
    ],
    "scheduled_events": [
        {{"trigger_round": 2, "description": "Brief event description", "poster_type": "Entity type", "content": "Post content triggered by this event"}},
        {{"trigger_round": 5, "description": "Mid-simulation escalation", "poster_type": "Entity type", "content": "Post content for escalation"}},
        {{"trigger_round": 8, "description": "Late resolution event", "poster_type": "Entity type", "content": "Post content for resolution"}}
    ],
    "reasoning": "<brief explanation>"
}}"""

        system_prompt = "You are a public opinion analysis expert. Respond in English only. Return pure JSON format. Note that poster_type must exactly match an available entity type."
        
        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Event config LLM generation failed: {e}, using default config")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Using default configuration"
            }
    
    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """Parse event configuration result"""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=result.get("scheduled_events", []),
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )
    
    def _match_agent_for_type(
        self,
        poster_type: str,
        agents_by_type: Dict[str, List[AgentActivityConfig]],
        agent_configs: List[AgentActivityConfig],
        used_indices: Dict[str, int]
    ) -> int:
        """Match an agent_id for a given poster_type using direct match, aliases, or fallback."""
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }

        poster_type_lower = poster_type.lower()

        # 1. Direct match
        if poster_type_lower in agents_by_type:
            agents = agents_by_type[poster_type_lower]
            idx = used_indices.get(poster_type_lower, 0) % len(agents)
            used_indices[poster_type_lower] = idx + 1
            return agents[idx].agent_id

        # 2. Match using aliases
        for alias_key, aliases in type_aliases.items():
            if poster_type_lower in aliases or alias_key == poster_type_lower:
                for alias in aliases:
                    if alias in agents_by_type:
                        agents = agents_by_type[alias]
                        idx = used_indices.get(alias, 0) % len(agents)
                        used_indices[alias] = idx + 1
                        return agents[idx].agent_id

        # 3. Fallback: highest influence agent
        logger.warning(f"No matching Agent found for type '{poster_type}', using highest influence Agent")
        if agent_configs:
            sorted_agents = sorted(agent_configs, key=lambda a: a.influence_weight, reverse=True)
            return sorted_agents[0].agent_id
        return 0

    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        Assign suitable publisher Agents to initial posts and scheduled events.

        Match the most appropriate agent_id based on each post's poster_type.
        """
        # Build agent index by entity type
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)

        used_indices: Dict[str, int] = {}

        # Assign agents to initial posts
        if event_config.initial_posts:
            updated_posts = []
            for post in event_config.initial_posts:
                poster_type = post.get("poster_type", "Unknown")
                content = post.get("content", "")
                matched_agent_id = self._match_agent_for_type(
                    poster_type, agents_by_type, agent_configs, used_indices
                )
                updated_posts.append({
                    "content": content,
                    "poster_type": poster_type,
                    "poster_agent_id": matched_agent_id
                })
                logger.info(f"Initial post assignment: poster_type='{poster_type}' -> agent_id={matched_agent_id}")
            event_config.initial_posts = updated_posts

        # Assign agents to scheduled events
        if event_config.scheduled_events:
            updated_events = []
            for event in event_config.scheduled_events:
                poster_type = event.get("poster_type", "Unknown")
                matched_agent_id = self._match_agent_for_type(
                    poster_type, agents_by_type, agent_configs, used_indices
                )
                updated_events.append({
                    "trigger_round": event.get("trigger_round", 1),
                    "description": event.get("description", ""),
                    "content": event.get("content", ""),
                    "poster_type": poster_type,
                    "poster_agent_id": matched_agent_id
                })
                logger.info(f"Scheduled event assignment: round={event.get('trigger_round')}, poster_type='{poster_type}' -> agent_id={matched_agent_id}")
            event_config.scheduled_events = updated_events

        return event_config
    
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
- **Official institutions** (University/GovernmentAgency): low activity (0.1-0.3), active during work hours (9-17), slow response (60-240 min), high influence (2.5-3.0)
- **Media** (MediaOutlet): moderate activity (0.4-0.6), active all day (8-23), fast response (5-30 min), high influence (2.0-2.5)
- **Individuals** (Student/Person/Alumni): high activity (0.6-0.9), mainly active in evenings (18-23), fast response (1-15 min), low influence (0.8-1.2)
- **Public figures/experts**: moderate activity (0.4-0.6), medium-high influence (1.5-2.0)

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
            
            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
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
    

