"""
Agent Behavioral Archetypes & Synthetic Persona Generation

Based on "From Who They Are to How They Act" (arXiv 2601.15114, Jan 2026):
- Behavioral archetypes control ACTION DISTRIBUTION (what agents do)
- Personas control CONTENT STYLE (how agents write)

Action types are sampled from the archetype weights before any LLM call
(system_one_router.py), which avoids the LLM's bias toward CREATE_POST.
"""

import re
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
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
    vocabulary_constraints: str  # free-text instruction for the LLM


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
        vocabulary_constraints=(
            "Write as a nightclub/venue. Keep it SHORT and simple. "
            "Not corporate-speak — you're a local venue, not a hospital. "
            "Brief updates, community spirit. Casual but responsible."
        ),
    ),
}


def get_speech_profile_for_entity_type(entity_type: str) -> SpeechProfile:
    """Get the speech profile for an institutional entity type."""
    return INSTITUTIONAL_SPEECH_PROFILES.get(
        entity_type,
        INSTITUTIONAL_SPEECH_PROFILES["Organization"]
    )


def _extract_scenario_context(simulation_requirement: str) -> Dict[str, str]:
    """Extract location and topic keywords from the simulation requirement text."""
    text = simulation_requirement

    location = "the local area"
    loc_patterns = [
        r'\bin\s+([A-Z][a-z]+(?:(?:\s+(?:upon|on|le|la|de|in)\s+|\s+)[A-Z][a-z]+)*(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?)',
        r'(?:city|town|region|area|borough|county)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Water|Council|Hospital|University|School|NHS|Trust|Borough)',
        r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ]
    stop_words = {
        "The", "This", "That", "These", "Those", "What", "How", "Why", "When",
        "Where", "Who", "Which", "Each", "Every", "Some", "Any", "All", "Most",
        "Many", "Several", "Both", "Few", "Other", "Such", "Create", "Simulate",
        "Generate", "Run", "Start", "Social", "Media", "Public", "Health",
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
    }
    for pattern in loc_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if match.split()[0] not in stop_words and len(match) > 2:
                location = match
                break
        if location != "the local area":
            break

    topic_keywords = []
    topic_map = {
        "water": ["water safety", "public health", "local infrastructure"],
        "contamination": ["contamination", "environmental health", "public safety"],
        "meningitis": ["meningitis", "public health", "university health"],
        "outbreak": ["disease outbreak", "public health", "epidemiology"],
        "flood": ["flooding", "emergency response", "climate"],
        "fire": ["fire safety", "emergency services", "local news"],
        "crime": ["crime", "public safety", "law enforcement"],
        "housing": ["housing", "local planning", "community"],
        "transport": ["transport", "infrastructure", "commuting"],
        "school": ["education", "school safety", "local community"],
        "nhs": ["NHS", "healthcare", "public health"],
        "vaccine": ["vaccination", "public health", "medical science"],
        "protest": ["activism", "civil rights", "community organizing"],
        "pollution": ["pollution", "environment", "public health"],
    }
    req_lower = simulation_requirement.lower()
    for keyword, topics in topic_map.items():
        if keyword in req_lower:
            topic_keywords.extend(topics)
    if not topic_keywords:
        topic_keywords = ["local news", "current events", "community"]
    topic_keywords = list(dict.fromkeys(topic_keywords))[:4]

    return {"location": location, "topics": topic_keywords}


def generate_synthetic_personas(
    simulation_requirement: str,
    num_institutional_agents: int,
    target_total_agents: int = 15,
    location: Optional[str] = None,
) -> List[Dict]:
    """
    Generate synthetic individual personas to fill the simulation with real people.

    Called when the entity extraction only produces institutional agents.
    Analyzes the simulation requirement to determine what demographics should be present.

    Args:
        location: Location extracted from knowledge graph entities. Falls back to
                  regex extraction from simulation_requirement if not provided.

    Returns a list of persona configs ready for profile generation.
    """
    num_synthetic = target_total_agents - num_institutional_agents
    if num_synthetic <= 0:
        return []

    scenario = _extract_scenario_context(simulation_requirement)
    if location:
        scenario["location"] = location
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

    # Names pool split by gender for consistent name-gender pairing
    male_first_names = [
        "James", "Mohammed", "Tom", "Liam", "Daniel", "Oliver", "Ryan",
        "Jack", "Kwame", "Marcus", "Ethan", "Callum",
    ]
    female_first_names = [
        "Emma", "Priya", "Sarah", "Aisha", "Chloe", "Fatima", "Sophie",
        "Mei", "Hannah", "Isla", "Zara", "Nia",
    ]
    neutral_first_names = [
        "Alex", "Jordan", "Sam", "Robin", "Casey", "Morgan",
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

        age = random.randint(*template["age_range"])
        gender = random.choice(template["gender_pool"])

        if gender == "male":
            name_pool = male_first_names
        elif gender == "female":
            name_pool = female_first_names
        else:
            name_pool = neutral_first_names

        # Generate a unique name
        while True:
            fname = random.choice(name_pool)
            lname = random.choice(last_names)
            full_name = f"{fname} {lname}"
            if full_name not in used_names:
                used_names.add(full_name)
                break

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
            "location": scenario["location"],
            "interested_topics": scenario["topics"],
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
        # Without this, about 1 in 6 posts opened with "As a parent/journalist, ..."
        # (Run 8), because the style line above describes the role.
        parts.append(
            "Speak as yourself. Do not announce your role or job (no \"As a parent...\", "
            "\"As a local journalist...\"); let it show through what you notice and care about."
        )
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
