"""
Agent Behavioral Archetypes & Synthetic Persona Generation

Based on "From Who They Are to How They Act" (arXiv 2601.15114, Jan 2026):
- Behavioral archetypes control ACTION DISTRIBUTION (what agents do)
- Personas control CONTENT STYLE (how agents write)

Key insight: Sample action type BEFORE prompting the LLM for content.
This prevents the LLM's bias toward CREATE_POST.
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class BehavioralArchetype(str, Enum):
    LURKER = "lurker"
    AMPLIFIER = "amplifier"
    CONTRIBUTOR = "contributor"
    DEBATER = "debater"


# Action probability distributions per archetype
# Values are weights (will be normalized). Actions: post, reply, repost, like, do_nothing
ARCHETYPE_ACTION_WEIGHTS = {
    BehavioralArchetype.LURKER: {
        "CREATE_POST": 2,
        "CREATE_COMMENT": 5,
        "REPOST": 3,
        "LIKE_POST": 15,
        "LIKE_COMMENT": 10,
        "DO_NOTHING": 65,
    },
    BehavioralArchetype.AMPLIFIER: {
        "CREATE_POST": 5,
        "CREATE_COMMENT": 15,
        "REPOST": 35,
        "LIKE_POST": 25,
        "LIKE_COMMENT": 10,
        "DO_NOTHING": 10,
    },
    BehavioralArchetype.CONTRIBUTOR: {
        "CREATE_POST": 30,
        "CREATE_COMMENT": 30,
        "REPOST": 10,
        "LIKE_POST": 15,
        "LIKE_COMMENT": 5,
        "DO_NOTHING": 10,
    },
    BehavioralArchetype.DEBATER: {
        "CREATE_POST": 15,
        "CREATE_COMMENT": 45,
        "REPOST": 5,
        "LIKE_POST": 10,
        "LIKE_COMMENT": 5,
        "DO_NOTHING": 20,
    },
}

# Twitter-specific mapping (Twitter uses QUOTE_POST instead of CREATE_COMMENT for debate)
TWITTER_ACTION_MAPPING = {
    "CREATE_POST": "CREATE_POST",
    "CREATE_COMMENT": "QUOTE_POST",
    "REPOST": "REPOST",
    "LIKE_POST": "LIKE_POST",
    "LIKE_COMMENT": "LIKE_POST",
    "DO_NOTHING": "DO_NOTHING",
}

# Reddit-specific mapping
REDDIT_ACTION_MAPPING = {
    "CREATE_POST": "CREATE_POST",
    "CREATE_COMMENT": "CREATE_COMMENT",
    "REPOST": "CREATE_COMMENT",  # Reddit doesn't have repost, comment instead
    "LIKE_POST": "LIKE_POST",
    "LIKE_COMMENT": "LIKE_COMMENT",
    "DO_NOTHING": "DO_NOTHING",
}


class EmotionalState(str, Enum):
    CALM = "calm"
    CURIOUS = "curious"
    WORRIED = "worried"
    FEARFUL = "fearful"
    ANGRY = "angry"
    PANICKED = "panicked"


@dataclass
class SpeechProfile:
    """Controls HOW an agent writes (tone, vocabulary, length)."""
    formality: str  # "formal", "casual", "very_casual"
    avg_length: str  # "short" (1-2 sentences), "medium" (3-5), "long" (paragraph)
    uses_emoji: bool
    uses_slang: bool
    uses_hashtags: bool
    tone_keywords: List[str]  # e.g. ["worried", "demanding", "clinical"]
    example_phrases: List[str]  # exemplar phrases this type would use
    vocabulary_constraints: str  # free-text instruction for the LLM


@dataclass
class SyntheticPersona:
    """A synthesized individual agent persona."""
    name: str
    age: int
    gender: str
    role: str  # e.g. "university student", "concerned parent"
    archetype: BehavioralArchetype
    emotional_state: EmotionalState
    speech_profile: SpeechProfile
    backstory: str  # 1-2 sentences of personal context
    stance: str  # "fearful", "skeptical", "trusting", "angry", "neutral"
    susceptible_to_misinfo: bool


# Persona templates for common demographic groups in crisis scenarios
PERSONA_TEMPLATES: Dict[str, Dict] = {
    "anxious_student": {
        "role": "university student",
        "age_range": (18, 23),
        "gender_pool": ["male", "female", "non-binary"],
        "archetype": BehavioralArchetype.DEBATER,
        "emotional_state": EmotionalState.FEARFUL,
        "stance": "fearful",
        "susceptible_to_misinfo": True,
        "speech_profile": SpeechProfile(
            formality="very_casual",
            avg_length="short",
            uses_emoji=True,
            uses_slang=True,
            uses_hashtags=False,
            tone_keywords=["scared", "confused", "urgent"],
            example_phrases=[
                "wtf is going on",
                "should I be worried??",
                "ngl this is terrifying",
                "has anyone else heard about",
            ],
            vocabulary_constraints=(
                "Write like a scared 20-year-old student. Use informal language, "
                "abbreviations (ngl, tbh, fr, lowkey), short sentences. "
                "Express genuine fear and confusion. Ask questions. "
                "You may share unverified rumors you've heard from friends."
            ),
        ),
    },
    "calm_student": {
        "role": "university student",
        "age_range": (19, 24),
        "gender_pool": ["male", "female"],
        "archetype": BehavioralArchetype.AMPLIFIER,
        "emotional_state": EmotionalState.CURIOUS,
        "stance": "trusting",
        "susceptible_to_misinfo": False,
        "speech_profile": SpeechProfile(
            formality="casual",
            avg_length="medium",
            uses_emoji=True,
            uses_slang=True,
            uses_hashtags=True,
            tone_keywords=["informative", "reassuring", "practical"],
            example_phrases=[
                "here's what the uni said",
                "just got the vaccine, wasn't bad",
                "sharing this for anyone who needs it",
            ],
            vocabulary_constraints=(
                "Write like a level-headed university student sharing useful info. "
                "Casual but not panicky. Share official guidance. Encourage others."
            ),
        ),
    },
    "angry_parent": {
        "role": "parent of university student",
        "age_range": (42, 58),
        "gender_pool": ["male", "female"],
        "archetype": BehavioralArchetype.DEBATER,
        "emotional_state": EmotionalState.ANGRY,
        "stance": "angry",
        "susceptible_to_misinfo": True,
        "speech_profile": SpeechProfile(
            formality="casual",
            avg_length="medium",
            uses_emoji=False,
            uses_slang=False,
            uses_hashtags=True,
            tone_keywords=["demanding", "accusatory", "protective"],
            example_phrases=[
                "Why weren't we told sooner?",
                "Heads need to roll for this",
                "My child is at risk and nobody is doing anything",
                "I want answers from the university NOW",
            ],
            vocabulary_constraints=(
                "Write as an angry, protective parent. Ask accusatory questions. "
                "Demand accountability from institutions. Express frustration. "
                "Use strong language but not profanity. Tag institutions."
            ),
        ),
    },
    "worried_parent": {
        "role": "parent of university student",
        "age_range": (40, 55),
        "gender_pool": ["male", "female"],
        "archetype": BehavioralArchetype.AMPLIFIER,
        "emotional_state": EmotionalState.WORRIED,
        "stance": "fearful",
        "susceptible_to_misinfo": True,
        "speech_profile": SpeechProfile(
            formality="casual",
            avg_length="medium",
            uses_emoji=False,
            uses_slang=False,
            uses_hashtags=False,
            tone_keywords=["worried", "seeking reassurance", "sharing"],
            example_phrases=[
                "Does anyone know if the vaccine is available?",
                "I'm so worried about my daughter",
                "Please share this with other parents",
            ],
            vocabulary_constraints=(
                "Write as a worried parent seeking information. "
                "Share posts from authorities. Ask practical questions. "
                "Express concern without anger. Seek community support."
            ),
        ),
    },
    "healthcare_worker": {
        "role": "healthcare worker",
        "age_range": (28, 55),
        "gender_pool": ["male", "female", "non-binary"],
        "archetype": BehavioralArchetype.CONTRIBUTOR,
        "emotional_state": EmotionalState.WORRIED,
        "stance": "trusting",
        "susceptible_to_misinfo": False,
        "speech_profile": SpeechProfile(
            formality="casual",
            avg_length="medium",
            uses_emoji=False,
            uses_slang=False,
            uses_hashtags=True,
            tone_keywords=["informed", "exhausted", "authoritative"],
            example_phrases=[
                "I work in A&E and I can tell you",
                "Please listen to the official guidance",
                "What I'm seeing on the front line",
                "The public needs to understand",
            ],
            vocabulary_constraints=(
                "Write as an off-duty healthcare worker posting on personal time. "
                "You have clinical knowledge but keep it accessible. "
                "Frustrated by misinformation. Tired but caring. "
                "Reference what you see at work without violating patient confidentiality."
            ),
        ),
    },
    "concerned_teacher": {
        "role": "school teacher",
        "age_range": (27, 50),
        "gender_pool": ["male", "female"],
        "archetype": BehavioralArchetype.AMPLIFIER,
        "emotional_state": EmotionalState.WORRIED,
        "stance": "fearful",
        "susceptible_to_misinfo": False,
        "speech_profile": SpeechProfile(
            formality="casual",
            avg_length="medium",
            uses_emoji=False,
            uses_slang=False,
            uses_hashtags=True,
            tone_keywords=["concerned", "practical", "community-minded"],
            example_phrases=[
                "As a teacher I need to know",
                "What do I tell my students?",
                "Parents are asking me questions I can't answer",
                "Our school hasn't told us anything",
            ],
            vocabulary_constraints=(
                "Write as a concerned teacher. Worried about students and community. "
                "Practical, wants clear guidance. Shares useful information. "
                "Frustrated by lack of communication from authorities."
            ),
        ),
    },
    "activist": {
        "role": "community activist",
        "age_range": (22, 40),
        "gender_pool": ["male", "female", "non-binary"],
        "archetype": BehavioralArchetype.DEBATER,
        "emotional_state": EmotionalState.ANGRY,
        "stance": "skeptical",
        "susceptible_to_misinfo": True,
        "speech_profile": SpeechProfile(
            formality="casual",
            avg_length="medium",
            uses_emoji=False,
            uses_slang=False,
            uses_hashtags=True,
            tone_keywords=["passionate", "systemic", "call-to-action"],
            example_phrases=[
                "This is what happens when we defund",
                "Thread: here's what they're not telling you",
                "We need to organize NOW",
                "The system failed us again",
            ],
            vocabulary_constraints=(
                "Write as a community activist. Frame issues as systemic failures. "
                "Call for collective action. Challenge official narratives. "
                "Passionate but articulate. Use threads to build arguments."
            ),
        ),
    },
    "retiree": {
        "role": "retired professional",
        "age_range": (60, 78),
        "gender_pool": ["male", "female"],
        "archetype": BehavioralArchetype.LURKER,
        "emotional_state": EmotionalState.WORRIED,
        "stance": "neutral",
        "susceptible_to_misinfo": True,
        "speech_profile": SpeechProfile(
            formality="formal",
            avg_length="medium",
            uses_emoji=False,
            uses_slang=False,
            uses_hashtags=False,
            tone_keywords=["measured", "experienced", "cautious"],
            example_phrases=[
                "In my experience",
                "I've seen situations like this before",
                "What concerns me most is",
                "We need calm heads right now",
            ],
            vocabulary_constraints=(
                "Write as a retired professional. Measured, slightly formal tone. "
                "Draw on life experience. Occasionally share unverified information "
                "from WhatsApp or Facebook groups. Genuinely concerned about community."
            ),
        ),
    },
    "office_worker": {
        "role": "office worker",
        "age_range": (25, 45),
        "gender_pool": ["male", "female", "non-binary"],
        "archetype": BehavioralArchetype.LURKER,
        "emotional_state": EmotionalState.CURIOUS,
        "stance": "neutral",
        "susceptible_to_misinfo": False,
        "speech_profile": SpeechProfile(
            formality="casual",
            avg_length="short",
            uses_emoji=True,
            uses_slang=False,
            uses_hashtags=False,
            tone_keywords=["casual", "mildly interested", "distracted"],
            example_phrases=[
                "Just saw this, is it real?",
                "Interesting",
                "Wait what's happening?",
                "Following this",
            ],
            vocabulary_constraints=(
                "Write as a regular office worker casually scrolling social media. "
                "Mildly interested, not deeply engaged. Short reactions. "
                "Occasionally shares things without reading the full article. "
                "Represents the passive majority of social media users."
            ),
        ),
    },
    "local_resident": {
        "role": "local resident",
        "age_range": (30, 65),
        "gender_pool": ["male", "female"],
        "archetype": BehavioralArchetype.LURKER,
        "emotional_state": EmotionalState.CURIOUS,
        "stance": "neutral",
        "susceptible_to_misinfo": False,
        "speech_profile": SpeechProfile(
            formality="casual",
            avg_length="short",
            uses_emoji=False,
            uses_slang=False,
            uses_hashtags=False,
            tone_keywords=["observational", "matter-of-fact", "local"],
            example_phrases=[
                "Saw long queues at the clinic today",
                "Hope the students are ok",
                "Anyone know if this affects non-students too?",
            ],
            vocabulary_constraints=(
                "Write as a local resident observing the situation. "
                "Short, factual observations. Mild concern. "
                "Occasionally ask questions about personal risk."
            ),
        ),
    },
    "skeptic_resident": {
        "role": "local resident",
        "age_range": (35, 60),
        "gender_pool": ["male", "female"],
        "archetype": BehavioralArchetype.DEBATER,
        "emotional_state": EmotionalState.ANGRY,
        "stance": "skeptical",
        "susceptible_to_misinfo": True,
        "speech_profile": SpeechProfile(
            formality="casual",
            avg_length="medium",
            uses_emoji=False,
            uses_slang=False,
            uses_hashtags=True,
            tone_keywords=["skeptical", "conspiratorial", "anti-authority"],
            example_phrases=[
                "Convenient timing for more vaccine mandates",
                "Why aren't they telling us the full story?",
                "Follow the money",
                "I'll wait for actual evidence before panicking",
            ],
            vocabulary_constraints=(
                "Write as a skeptical resident who distrusts authorities. "
                "Question official narratives. Hint at cover-ups. "
                "Not outright conspiracy theorist but deeply skeptical. "
                "May share alternative interpretations of events."
            ),
        ),
    },
    "journalist": {
        "role": "local journalist",
        "age_range": (28, 45),
        "gender_pool": ["male", "female"],
        "archetype": BehavioralArchetype.CONTRIBUTOR,
        "emotional_state": EmotionalState.CALM,
        "stance": "neutral",
        "susceptible_to_misinfo": False,
        "speech_profile": SpeechProfile(
            formality="formal",
            avg_length="medium",
            uses_emoji=False,
            uses_slang=False,
            uses_hashtags=True,
            tone_keywords=["investigative", "questioning", "factual"],
            example_phrases=[
                "BREAKING:",
                "Sources tell me",
                "I've asked [authority] for comment",
                "Thread: Here's what we know so far",
            ],
            vocabulary_constraints=(
                "Write as a journalist. Break news, ask hard questions, "
                "challenge official statements, cite sources. "
                "Professional but not institutional. Push for transparency."
            ),
        ),
    },
}

# Institutional speech profiles (for entities extracted from the document)
INSTITUTIONAL_SPEECH_PROFILES: Dict[str, SpeechProfile] = {
    "HealthAgency": SpeechProfile(
        formality="formal",
        avg_length="long",
        uses_emoji=False,
        uses_slang=False,
        uses_hashtags=True,
        tone_keywords=["clinical", "authoritative", "measured"],
        example_phrases=[
            "We can confirm that",
            "The risk to the general public remains low",
            "We advise all individuals to",
        ],
        vocabulary_constraints=(
            "Write as a health authority. Use measured, clinical language. "
            "Reference guidelines and data. Avoid speculation. "
            "Project calm authority. Include actionable advice."
        ),
    ),
    "University": SpeechProfile(
        formality="formal",
        avg_length="medium",
        uses_emoji=False,
        uses_slang=False,
        uses_hashtags=True,
        tone_keywords=["supportive", "informative", "pastoral"],
        example_phrases=[
            "We are supporting our students by",
            "Updated guidance is available at",
            "Student welfare remains our priority",
        ],
        vocabulary_constraints=(
            "Write as a university communications team. Supportive, pastoral tone. "
            "Reference student services, updated guidance. Balance transparency with calm."
        ),
    ),
    "Organization": SpeechProfile(
        formality="formal",
        avg_length="medium",
        uses_emoji=False,
        uses_slang=False,
        uses_hashtags=True,
        tone_keywords=["professional", "neutral", "informative"],
        example_phrases=[
            "We can confirm",
            "Our position is",
            "We are working with",
        ],
        vocabulary_constraints=(
            "Write as a professional organization. Clear, concise, factual. "
            "Avoid emotional language. Reference partnerships and official sources."
        ),
    ),
    "MediaOutlet": SpeechProfile(
        formality="formal",
        avg_length="medium",
        uses_emoji=False,
        uses_slang=False,
        uses_hashtags=True,
        tone_keywords=["breaking", "factual", "attention-grabbing"],
        example_phrases=[
            "BREAKING:",
            "JUST IN:",
            "We understand that",
            "More to follow",
        ],
        vocabulary_constraints=(
            "Write as a news outlet. Factual, attention-grabbing headlines. "
            "Attribution to sources. Concise. Include 'more to follow' for developing stories."
        ),
    ),
    "NGO": SpeechProfile(
        formality="formal",
        avg_length="medium",
        uses_emoji=False,
        uses_slang=False,
        uses_hashtags=True,
        tone_keywords=["advocating", "supportive", "educational"],
        example_phrases=[
            "Know the signs:",
            "If you or someone you know",
            "Our helpline is available",
        ],
        vocabulary_constraints=(
            "Write as a charity/NGO. Educational, supportive, action-oriented. "
            "Provide resources and helpline numbers. Advocate for affected groups."
        ),
    ),
    "GovernmentOfficial": SpeechProfile(
        formality="formal",
        avg_length="long",
        uses_emoji=False,
        uses_slang=False,
        uses_hashtags=False,
        tone_keywords=["authoritative", "reassuring", "policy-focused"],
        example_phrases=[
            "I have been briefed on",
            "The government is taking action",
            "I want to reassure the public",
        ],
        vocabulary_constraints=(
            "Write as a government official. Authoritative, reassuring. "
            "Reference policy actions, cross-agency coordination. "
            "Project competence and control."
        ),
    ),
    "Nightclub": SpeechProfile(
        formality="casual",
        avg_length="short",
        uses_emoji=True,
        uses_slang=False,
        uses_hashtags=True,
        tone_keywords=["brief", "cooperative", "community-minded"],
        example_phrases=[
            "Closed tonight. Stay safe everyone",
            "We're working with health authorities",
            "Will update when we know more",
        ],
        vocabulary_constraints=(
            "Write as a nightclub/venue. Keep it SHORT and simple. "
            "Not corporate-speak — you're a local venue, not a hospital. "
            "Brief updates, community spirit. Casual but responsible."
        ),
    ),
}


def sample_action_for_archetype(
    archetype: BehavioralArchetype,
    platform: str = "twitter",
    available_actions: Optional[List[str]] = None
) -> str:
    """
    Sample an action type based on archetype's behavioral distribution.

    This is called BEFORE the LLM generates content — decoupling action choice
    from content generation prevents bias toward CREATE_POST.
    """
    weights = ARCHETYPE_ACTION_WEIGHTS[archetype].copy()

    # Map generic actions to platform-specific actions
    mapping = TWITTER_ACTION_MAPPING if platform == "twitter" else REDDIT_ACTION_MAPPING

    # Build platform-specific weights
    platform_weights: Dict[str, float] = {}
    for generic_action, weight in weights.items():
        platform_action = mapping.get(generic_action, generic_action)
        platform_weights[platform_action] = platform_weights.get(platform_action, 0) + weight

    # Filter to only available actions if specified
    if available_actions:
        platform_weights = {k: v for k, v in platform_weights.items() if k in available_actions}

    if not platform_weights:
        return "DO_NOTHING"

    actions = list(platform_weights.keys())
    probs = list(platform_weights.values())
    total = sum(probs)
    probs = [p / total for p in probs]

    return random.choices(actions, weights=probs, k=1)[0]


def get_speech_profile_for_entity_type(entity_type: str) -> SpeechProfile:
    """Get the speech profile for an institutional entity type."""
    return INSTITUTIONAL_SPEECH_PROFILES.get(
        entity_type,
        INSTITUTIONAL_SPEECH_PROFILES["Organization"]
    )


def generate_synthetic_personas(
    simulation_requirement: str,
    num_institutional_agents: int,
    target_total_agents: int = 15,
) -> List[Dict]:
    """
    Generate synthetic individual personas to fill the simulation with real people.

    Called when the entity extraction only produces institutional agents.
    Analyzes the simulation requirement to determine what demographics should be present.

    Returns a list of persona configs ready for profile generation.
    """
    num_synthetic = target_total_agents - num_institutional_agents
    if num_synthetic <= 0:
        return []

    requirement_lower = simulation_requirement.lower()

    # Determine which persona templates are relevant based on the requirement text
    relevant_templates = []

    # Topic-specific templates
    if any(w in requirement_lower for w in ["student", "university", "campus", "college"]):
        relevant_templates.append(("anxious_student", 3))
        relevant_templates.append(("calm_student", 2))

    if any(w in requirement_lower for w in ["parent", "family", "child", "son", "daughter"]):
        relevant_templates.append(("angry_parent", 2))
        relevant_templates.append(("worried_parent", 2))

    if any(w in requirement_lower for w in ["resident", "local", "community", "neighbour"]):
        relevant_templates.append(("local_resident", 2))
        relevant_templates.append(("skeptic_resident", 2))

    if any(w in requirement_lower for w in ["journalist", "media", "press", "reporter"]):
        relevant_templates.append(("journalist", 2))

    if any(w in requirement_lower for w in ["health", "hospital", "doctor", "nurse", "nhs", "vaccine", "disease", "outbreak"]):
        relevant_templates.append(("healthcare_worker", 2))

    if any(w in requirement_lower for w in ["school", "teacher", "education", "classroom"]):
        relevant_templates.append(("concerned_teacher", 2))

    if any(w in requirement_lower for w in ["protest", "activist", "organize", "rights", "justice", "inequality"]):
        relevant_templates.append(("activist", 2))

    # Always-present background population (the silent majority)
    relevant_templates.append(("office_worker", 3))
    relevant_templates.append(("retiree", 2))
    relevant_templates.append(("local_resident", 1))

    # If only background templates matched, add a general engaged mix too
    if all(t[0] in ("office_worker", "retiree", "local_resident") for t in relevant_templates):
        relevant_templates.extend([
            ("anxious_student", 2),
            ("worried_parent", 2),
            ("journalist", 1),
            ("healthcare_worker", 1),
            ("activist", 1),
        ])

    # Expand templates to fill the target count
    personas = []
    template_pool = []
    for template_name, count in relevant_templates:
        template_pool.extend([template_name] * count)

    # Names pool for generating diverse names
    first_names = [
        "Emma", "James", "Priya", "Mohammed", "Sarah", "Tom", "Aisha", "Liam",
        "Chloe", "Daniel", "Fatima", "Oliver", "Sophie", "Ryan", "Mei", "Jack",
        "Hannah", "Kwame", "Isla", "Marcus", "Zara", "Ethan", "Nia", "Callum",
    ]
    last_names = [
        "Smith", "Patel", "Jones", "Williams", "Brown", "Ahmed", "Taylor",
        "Chen", "Wilson", "Davies", "Kumar", "Evans", "Johnson", "Khan",
        "Martin", "O'Brien", "Thompson", "Garcia", "Lee", "Mitchell",
    ]

    used_names = set()

    for i in range(num_synthetic):
        template_name = template_pool[i % len(template_pool)]
        template = PERSONA_TEMPLATES[template_name]

        # Generate a unique name
        while True:
            fname = random.choice(first_names)
            lname = random.choice(last_names)
            full_name = f"{fname} {lname}"
            if full_name not in used_names:
                used_names.add(full_name)
                break

        age = random.randint(*template["age_range"])
        gender = random.choice(template["gender_pool"])

        persona_config = {
            "name": full_name,
            "age": age,
            "gender": gender,
            "role": template["role"],
            "archetype": template["archetype"].value,
            "emotional_state": template["emotional_state"].value,
            "stance": template["stance"],
            "susceptible_to_misinfo": template["susceptible_to_misinfo"],
            "speech_profile": template["speech_profile"],
            "is_synthetic": True,
            "template": template_name,
        }
        personas.append(persona_config)

    return personas


def build_agent_system_prompt(
    speech_profile: SpeechProfile,
    emotional_state: EmotionalState,
    is_institutional: bool,
    confirmation_bias: bool = False,
) -> str:
    """
    Build the behavioral system prompt for an agent.
    Appended to the OASIS agent's system prompt to control their output style.
    """
    parts = []

    parts.append(f"COMMUNICATION STYLE: {speech_profile.vocabulary_constraints}")

    if not is_institutional:
        parts.append(f"CURRENT EMOTIONAL STATE: {emotional_state.value.upper()}")

        if emotional_state in (EmotionalState.FEARFUL, EmotionalState.PANICKED):
            parts.append(
                "You are feeling genuinely scared. This affects how you write — "
                "shorter sentences, more questions, more urgency."
            )
        elif emotional_state == EmotionalState.ANGRY:
            parts.append(
                "You are frustrated and angry. You may blame institutions, "
                "ask accusatory questions, or demand action."
            )

    if confirmation_bias:
        parts.append(
            "COGNITIVE BIAS: You have confirmation bias. You tend to believe "
            "information that matches your existing fears or beliefs, and may "
            "share unverified information if it feels emotionally true to you."
        )

    if speech_profile.uses_emoji:
        parts.append("You occasionally use emojis in your posts.")
    if speech_profile.uses_slang:
        parts.append("You use internet slang and abbreviations (ngl, tbh, fr, lowkey, etc).")

    parts.append(f"POST LENGTH: Keep your posts {speech_profile.avg_length}.")

    return "\n".join(parts)
