"""
OASIS Simulation Manager
Orchestrates Twitter and Reddit dual-platform parallel simulations.
Uses preset scripts with LLM-generated configuration parameters.
"""

import os
import json
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.logger import get_logger
from .entity_reader import EntityReader, FilteredEntities, EntityNode
from .oasis_profile_generator import OasisProfileGenerator, OasisAgentProfile
from .simulation_config_generator import SimulationConfigGenerator, SimulationParameters
from .agent_archetypes import generate_synthetic_personas

logger = get_logger('mirofish.simulation')


class SimulationStatus(str, Enum):
    CREATED = "created"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class PlatformType(str, Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"


@dataclass
class SimulationState:
    simulation_id: str
    project_id: str
    graph_id: str
    
    enable_twitter: bool = True
    enable_reddit: bool = True
    status: SimulationStatus = SimulationStatus.CREATED

    entities_count: int = 0
    profiles_count: int = 0
    entity_types: List[str] = field(default_factory=list)

    config_generated: bool = False
    config_reasoning: str = ""

    current_round: int = 0
    twitter_status: str = "not_started"
    reddit_status: str = "not_started"

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Full state as dict."""
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "enable_twitter": self.enable_twitter,
            "enable_reddit": self.enable_reddit,
            "status": self.status.value,
            "entities_count": self.entities_count,
            "profiles_count": self.profiles_count,
            "entity_types": self.entity_types,
            "config_generated": self.config_generated,
            "config_reasoning": self.config_reasoning,
            "current_round": self.current_round,
            "twitter_status": self.twitter_status,
            "reddit_status": self.reddit_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }
    
    def to_simple_dict(self) -> Dict[str, Any]:
        """Simplified state dict for API responses."""
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "status": self.status.value,
            "entities_count": self.entities_count,
            "profiles_count": self.profiles_count,
            "entity_types": self.entity_types,
            "config_generated": self.config_generated,
            "error": self.error,
        }


class SimulationManager:
    """
    Orchestrates the full simulation preparation pipeline:
    1. Read and filter entities from the knowledge graph
    2. Generate OASIS Agent Profiles (with synthetic personas)
    3. LLM-generate simulation config (timing, events, activity)
    4. Save all files needed by the runner scripts
    """
    SIMULATION_DATA_DIR = os.path.join(
        os.path.dirname(__file__), 
        '../../uploads/simulations'
    )
    
    def __init__(self):
        os.makedirs(self.SIMULATION_DATA_DIR, exist_ok=True)
        self._simulations: Dict[str, SimulationState] = {}
    
    def _get_simulation_dir(self, simulation_id: str) -> str:
        sim_dir = os.path.join(self.SIMULATION_DATA_DIR, simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        return sim_dir
    
    def _save_simulation_state(self, state: SimulationState):
        sim_dir = self._get_simulation_dir(state.simulation_id)
        state_file = os.path.join(sim_dir, "state.json")
        
        state.updated_at = datetime.now().isoformat()
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        
        self._simulations[state.simulation_id] = state
    
    def _load_simulation_state(self, simulation_id: str) -> Optional[SimulationState]:
        if simulation_id in self._simulations:
            return self._simulations[simulation_id]
        
        sim_dir = self._get_simulation_dir(simulation_id)
        state_file = os.path.join(sim_dir, "state.json")
        
        if not os.path.exists(state_file):
            return None
        
        with open(state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        state = SimulationState(
            simulation_id=simulation_id,
            project_id=data.get("project_id", ""),
            graph_id=data.get("graph_id", ""),
            enable_twitter=data.get("enable_twitter", True),
            enable_reddit=data.get("enable_reddit", True),
            status=SimulationStatus(data.get("status", "created")),
            entities_count=data.get("entities_count", 0),
            profiles_count=data.get("profiles_count", 0),
            entity_types=data.get("entity_types", []),
            config_generated=data.get("config_generated", False),
            config_reasoning=data.get("config_reasoning", ""),
            current_round=data.get("current_round", 0),
            twitter_status=data.get("twitter_status", "not_started"),
            reddit_status=data.get("reddit_status", "not_started"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            error=data.get("error"),
        )
        
        self._simulations[simulation_id] = state
        return state
    
    def create_simulation(
        self,
        project_id: str,
        graph_id: str,
        enable_twitter: bool = True,
        enable_reddit: bool = True,
    ) -> SimulationState:
        import uuid
        simulation_id = f"sim_{uuid.uuid4().hex[:12]}"
        
        state = SimulationState(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=enable_twitter,
            enable_reddit=enable_reddit,
            status=SimulationStatus.CREATED,
        )
        
        self._save_simulation_state(state)
        logger.info(f"Created simulation: {simulation_id}, project={project_id}, graph={graph_id}")
        
        return state
    
    def prepare_simulation(
        self,
        simulation_id: str,
        simulation_requirement: str,
        document_text: str,
        defined_entity_types: Optional[List[str]] = None,
        use_llm_for_profiles: bool = True,
        progress_callback: Optional[callable] = None,
        parallel_profile_count: int = 2,
        storage: 'GraphStorage' = None,
    ) -> SimulationState:
        """Prepare the full simulation: read entities, generate profiles, build config."""
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation not found: {simulation_id}")
        
        try:
            state.status = SimulationStatus.PREPARING
            self._save_simulation_state(state)
            
            sim_dir = self._get_simulation_dir(simulation_id)
            
            # ========== Phase 1: Read and filter entities ==========
            if progress_callback:
                progress_callback("reading", 0, "In progress: Graph...")

            if not storage:
                raise ValueError("storage (GraphStorage) is required for prepare_simulation")
            reader = EntityReader(storage)
            
            if progress_callback:
                progress_callback("reading", 30, "In progress: Reading...")
            
            filtered = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=defined_entity_types,
                enrich_with_edges=True
            )
            
            state.entities_count = filtered.filtered_count
            state.entity_types = list(filtered.entity_types)
            
            if progress_callback:
                progress_callback(
                    "reading", 100, 
                    f"complete，{filtered.filtered_count} Entity",
                    current=filtered.filtered_count,
                    total=filtered.filtered_count
                )
            
            if filtered.filtered_count == 0:
                state.status = SimulationStatus.FAILED
                state.error = "No matching entities found. Check that the knowledge graph was built correctly."
                self._save_simulation_state(state)
                return state

            # Cap institutional entities (leave room for synthetic personas)
            MAX_AGENT_ENTITIES = int(os.environ.get('MAX_AGENT_ENTITIES', '30'))
            max_institutional = min(8, len(filtered.entities))  # Cap institutions, prioritize synthetic individuals
            if len(filtered.entities) > max_institutional:
                logger.info(
                    f"Capping institutional entities from {len(filtered.entities)} to {max_institutional}"
                )
                filtered.entities = filtered.entities[:max_institutional]
                filtered.filtered_count = len(filtered.entities)

            # Generate synthetic individual personas (students, parents, residents)
            # to fill the simulation with real people, not just institutions
            synthetic_personas = generate_synthetic_personas(
                simulation_requirement=simulation_requirement,
                num_institutional_agents=len(filtered.entities),
                target_total_agents=MAX_AGENT_ENTITIES,
            )
            if synthetic_personas:
                logger.info(f"Generated {len(synthetic_personas)} synthetic personas")
                synth_path = os.path.join(sim_dir, "synthetic_personas.json")
                with open(synth_path, 'w', encoding='utf-8') as f:
                    serializable = []
                    for sp in synthetic_personas:
                        sp_copy = {k: v for k, v in sp.items() if k != 'speech_profile'}
                        sp_copy['speech_profile'] = {
                            'formality': sp['speech_profile'].formality,
                            'avg_length': sp['speech_profile'].avg_length,
                            'vocabulary_constraints': sp['speech_profile'].vocabulary_constraints,
                        }
                        serializable.append(sp_copy)
                    json.dump(serializable, f, ensure_ascii=False, indent=2)

            state.entities_count = len(filtered.entities) + len(synthetic_personas)

            # ========== Phase 2: Generate Agent Profiles ==========
            total_entities = len(filtered.entities) + len(synthetic_personas)
            
            if progress_callback:
                progress_callback(
                    "generating_profiles", 0, 
                    "StartingGenerating...",
                    current=0,
                    total=total_entities
                )
            
            generator = OasisProfileGenerator(storage=storage, graph_id=state.graph_id)
            
            def profile_progress(current, total, msg):
                if progress_callback:
                    progress_callback(
                        "generating_profiles", 
                        int(current / total * 100), 
                        msg,
                        current=current,
                        total=total,
                        item_name=msg
                    )
            
            # Set up realtime save path (prefer Reddit JSON format)
            realtime_output_path = None
            realtime_platform = "reddit"
            if state.enable_reddit:
                realtime_output_path = os.path.join(sim_dir, "reddit_profiles.json")
                realtime_platform = "reddit"
            elif state.enable_twitter:
                realtime_output_path = os.path.join(sim_dir, "twitter_profiles.csv")
                realtime_platform = "twitter"
            
            profiles = generator.generate_profiles_from_entities(
                entities=filtered.entities,
                use_llm=use_llm_for_profiles,
                progress_callback=profile_progress,
                graph_id=state.graph_id,
                parallel_count=parallel_profile_count,
                realtime_output_path=realtime_output_path,
                output_platform=realtime_platform
            )

            # Add synthetic individual personas (no LLM needed — pre-built profiles)
            from .agent_archetypes import (
                build_agent_system_prompt, EmotionalState,
                get_speech_profile_for_entity_type,
            )
            if synthetic_personas:
                next_id = len(profiles)
                for sp in synthetic_personas:
                    speech = sp['speech_profile']
                    emotional = EmotionalState(sp['emotional_state'])
                    persona_text = (
                        f"{sp['name']} is a {sp['age']}-year-old {sp['gender']} {sp['role']} "
                        f"in Canterbury, Kent. Stance: {sp['stance']}. "
                        f"\n\n--- BEHAVIORAL INSTRUCTIONS ---\n"
                        f"{build_agent_system_prompt(speech, emotional, is_institutional=False, confirmation_bias=sp.get('susceptible_to_misinfo', False))}"
                    )
                    profile = OasisAgentProfile(
                        user_id=next_id,
                        user_name=sp['name'].lower().replace(' ', '_') + f"_{random.randint(100,999)}",
                        name=sp['name'],
                        bio=f"{sp['role'].title()}, {sp['age']}, Canterbury",
                        persona=persona_text,
                        age=sp['age'],
                        gender=sp['gender'],
                        mbti=random.choice(["ENFP", "INFP", "ISTP", "ESFJ", "INTP", "ENFJ", "ISTJ", "ESTP"]),
                        country="United Kingdom",
                        profession=sp['role'],
                        interested_topics=["health", "local news", "university life"],
                        karma=random.randint(100, 3000),
                        source_entity_type=sp.get('template', 'synthetic'),
                    )
                    profiles.append(profile)
                    next_id += 1
                logger.info(f"Added {len(synthetic_personas)} synthetic persona profiles")

            # Enrich institutional profiles with speech/behavioral instructions
            for profile in profiles:
                if profile.source_entity_type and profile.source_entity_type not in (
                    'synthetic', 'anxious_student', 'calm_student', 'angry_parent',
                    'worried_parent', 'local_resident', 'skeptic_resident', 'journalist'
                ):
                    speech = get_speech_profile_for_entity_type(profile.source_entity_type or 'Organization')
                    behavioral_suffix = (
                        f"\n\n--- BEHAVIORAL INSTRUCTIONS ---\n"
                        f"{build_agent_system_prompt(speech, EmotionalState.CALM, is_institutional=True)}"
                    )
                    profile.persona = profile.persona + behavioral_suffix

            state.profiles_count = len(profiles)

            # Save profiles (Twitter=CSV, Reddit=JSON as required by OASIS)
            if progress_callback:
                progress_callback(
                    "generating_profiles", 95, 
                    "SavingProfilefile...",
                    current=total_entities,
                    total=total_entities
                )
            
            if state.enable_reddit:
                generator.save_profiles(
                    profiles=profiles,
                    file_path=os.path.join(sim_dir, "reddit_profiles.json"),
                    platform="reddit"
                )
            
            if state.enable_twitter:
                generator.save_profiles(
                    profiles=profiles,
                    file_path=os.path.join(sim_dir, "twitter_profiles.csv"),
                    platform="twitter"
                )
            
            if progress_callback:
                progress_callback(
                    "generating_profiles", 100, 
                    f"complete，{len(profiles)} Profile",
                    current=len(profiles),
                    total=len(profiles)
                )
            
            # ========== Phase 3: LLM-generate simulation config ==========
            if progress_callback:
                progress_callback(
                    "generating_config", 0, 
                    "In progress: Simulation...",
                    current=0,
                    total=3
                )
            
            config_generator = SimulationConfigGenerator()
            
            if progress_callback:
                progress_callback(
                    "generating_config", 30, 
                    "In progress: LLMGeneratingConfiguration...",
                    current=1,
                    total=3
                )
            
            sim_params = config_generator.generate_config(
                simulation_id=simulation_id,
                project_id=state.project_id,
                graph_id=state.graph_id,
                simulation_requirement=simulation_requirement,
                document_text=document_text,
                entities=filtered.entities,
                enable_twitter=state.enable_twitter,
                enable_reddit=state.enable_reddit,
                synthetic_personas=synthetic_personas,
            )
            
            if progress_callback:
                progress_callback(
                    "generating_config", 70, 
                    "In progress: SavingConfigurationfile...",
                    current=2,
                    total=3
                )
            
            config_path = os.path.join(sim_dir, "simulation_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(sim_params.to_json())
            
            state.config_generated = True
            state.config_reasoning = sim_params.generation_reasoning
            
            if progress_callback:
                progress_callback(
                    "generating_config", 100, 
                    "ConfigurationGeneratingcomplete",
                    current=3,
                    total=3
                )
            
            state.status = SimulationStatus.READY
            self._save_simulation_state(state)
            
            logger.info(f"Simulation ready: {simulation_id}, "
                       f"entities={state.entities_count}, profiles={state.profiles_count}")
            
            return state
            
        except Exception as e:
            logger.error(f"Simulation preparation failed: {simulation_id}, error={str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            state.status = SimulationStatus.FAILED
            state.error = str(e)
            self._save_simulation_state(state)
            raise
    
    def get_simulation(self, simulation_id: str) -> Optional[SimulationState]:
        return self._load_simulation_state(simulation_id)
    
    def list_simulations(self, project_id: Optional[str] = None) -> List[SimulationState]:
        simulations = []
        
        if os.path.exists(self.SIMULATION_DATA_DIR):
            for sim_id in os.listdir(self.SIMULATION_DATA_DIR):
                sim_path = os.path.join(self.SIMULATION_DATA_DIR, sim_id)
                if sim_id.startswith('.') or not os.path.isdir(sim_path):
                    continue
                
                state = self._load_simulation_state(sim_id)
                if state:
                    if project_id is None or state.project_id == project_id:
                        simulations.append(state)
        
        return simulations
    
    def get_profiles(self, simulation_id: str, platform: str = "reddit") -> List[Dict[str, Any]]:
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation not found: {simulation_id}")
        
        sim_dir = self._get_simulation_dir(simulation_id)
        profile_path = os.path.join(sim_dir, f"{platform}_profiles.json")
        
        if not os.path.exists(profile_path):
            return []
        
        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_simulation_config(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_run_instructions(self, simulation_id: str) -> Dict[str, str]:
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
        
        return {
            "simulation_dir": sim_dir,
            "scripts_dir": scripts_dir,
            "config_file": config_path,
            "commands": {
                "twitter": f"python {scripts_dir}/run_twitter_simulation.py --config {config_path}",
                "reddit": f"python {scripts_dir}/run_reddit_simulation.py --config {config_path}",
                "parallel": f"python {scripts_dir}/run_parallel_simulation.py --config {config_path}",
            },
            "instructions": (
                f"1. condaconda activate MiroFish\n"
                f"2. Simulation ({scripts_dir}):\n"
                f"   - Twitter: python {scripts_dir}/run_twitter_simulation.py --config {config_path}\n"
                f"   - Reddit: python {scripts_dir}/run_reddit_simulation.py --config {config_path}\n"
                f"   - python {scripts_dir}/run_parallel_simulation.py --config {config_path}"
            )
        }
