"""
Ontology Generation Service
Analyze text content and generate entity and relationship type definitions for social simulation.
"""

import logging
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient

logger = logging.getLogger('mirofish.ontology')


# System prompt for ontology generation
ONTOLOGY_SYSTEM_PROMPT = """You are a professional knowledge graph ontology design expert. Your task is to analyze the given text content and simulation requirements, and design entity types and relationship types suitable for **social media opinion simulation**.

**IMPORTANT: You must output valid JSON data only. Do not output any other content.**

## Core Task Background

We are building a **social media opinion simulation system**. In this system:
- Each entity is an "account" or "agent" that can post, interact, and spread information on social media
- Entities influence each other through reposting, commenting, and responding
- We need to simulate how various parties react to public opinion events and how information propagates

Therefore, **entities must be real-world agents that can post and interact on social media**:

**Acceptable entity types**:
- Specific individuals (public figures, parties involved, opinion leaders, experts, ordinary people)
- Companies and enterprises (including their official accounts)
- Organizations and institutions (universities, associations, NGOs, unions, etc.)
- Government departments and regulatory bodies
- Media organizations (newspapers, TV stations, self-media, websites)
- Social media platforms themselves
- Representatives of specific groups (e.g., alumni associations, fan clubs, advocacy groups)

**NOT acceptable**:
- Abstract concepts (e.g., "public opinion", "sentiment", "trends")
- Topics/themes (e.g., "academic integrity", "education reform")
- Viewpoints/stances (e.g., "supporters", "opponents")

## Output Format

Output in JSON format with the following structure:

```json
{
    "entity_types": [
        {
            "name": "Entity type name (English, PascalCase)",
            "description": "Brief description (English, max 100 characters)",
            "attributes": [
                {
                    "name": "attribute_name (English, snake_case)",
                    "type": "text",
                    "description": "Attribute description"
                }
            ],
            "examples": ["Example entity 1", "Example entity 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Relationship type name (English, UPPER_SNAKE_CASE)",
            "description": "Brief description (English, max 100 characters)",
            "source_targets": [
                {"source": "SourceEntityType", "target": "TargetEntityType"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brief analysis summary of the text content"
}
```

## Design Guidelines (CRITICAL!)

### 1. Entity Type Design - Must Be Strictly Followed

**Quantity requirement: You must output exactly 10 entity types**

**Hierarchy requirement (must include both specific types and fallback types)**:

Your 10 entity types must include the following levels:

A. **Fallback types (required, must be the last 2 in the list)**:
   - `Person`: The fallback type for any individual person. When a person does not belong to any more specific person type, they should be classified here.
   - `Organization`: The fallback type for any organization. When an organization does not belong to any more specific organization type, it should be classified here.

B. **Specific types (8 types, designed based on the text content)**:
   - Design more specific types based on the key roles that appear in the text
   - Example: If the text involves an academic event, you might have `Student`, `Professor`, `University`
   - Example: If the text involves a business event, you might have `Company`, `CEO`, `Employee`
   - A type is a CATEGORY that several entities could belong to, never one specific named entity.
     Name types after the role, not the instance: a water utility called "Thames Water" is a
     `WaterCompany`, not `ThamesWater`; "Leeds City Council" is a `LocalCouncil`, not `LeedsCityCouncil`.
     The specific names belong in `examples`.

**Why fallback types are needed**:
- The text will contain various people, such as "elementary school teachers", "bystanders", "anonymous netizens"
- If there is no specific type that matches, they should be classified under `Person`
- Similarly, small organizations, ad-hoc groups, etc. should be classified under `Organization`

**Design principles for specific types**:
- Identify the most frequently mentioned or key role types from the text
- Each specific type should have clear boundaries to avoid overlap
- The description must clearly explain how this type differs from the fallback type

### 2. Relationship Type Design

- Quantity: 6-10 types
- Relationships should reflect real connections in social media interactions
- Ensure that the source_targets of relationships cover the entity types you defined

### 3. Attribute Design

- 1-3 key attributes per entity type
- **NOTE**: Attribute names must NOT use `name`, `uuid`, `group_id`, `created_at`, `summary` (these are system reserved words)
- Recommended: `full_name`, `title`, `role`, `position`, `location`, `description`, etc.

## Entity Type Reference

**Person types (specific)**:
- Student: A student at an educational institution
- Professor: A professor or academic scholar
- Journalist: A reporter or journalist
- Celebrity: A celebrity or internet influencer
- Executive: A corporate executive or senior manager
- Official: A government official
- Lawyer: A legal professional
- Doctor: A medical professional

**Person types (fallback)**:
- Person: Any individual person (used when no specific person type applies)

**Organization types (specific)**:
- University: A college or university
- Company: A corporation or business enterprise
- GovernmentAgency: A government department or agency
- MediaOutlet: A media organization
- Hospital: A hospital or medical institution
- School: An elementary or secondary school
- NGO: A non-governmental organization

**Organization types (fallback)**:
- Organization: Any organization (used when no specific organization type applies)

## Relationship Type Reference

- WORKS_FOR: Employed by or works at
- STUDIES_AT: Studies or is enrolled at
- AFFILIATED_WITH: Is affiliated with or belongs to
- REPRESENTS: Acts as a representative of
- REGULATES: Has regulatory authority over
- REPORTS_ON: Covers or reports on
- COMMENTS_ON: Comments on or discusses
- RESPONDS_TO: Responds or reacts to
- SUPPORTS: Supports or endorses
- OPPOSES: Opposes or is against
- COLLABORATES_WITH: Collaborates or works together with
- COMPETES_WITH: Competes against

You MUST respond in English only. All descriptions, examples, and analysis should be in English.
"""


class OntologyGenerator:
    """
    Ontology Generator
    Analyzes text content and generates entity and relationship type definitions.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate ontology definitions.

        Args:
            document_texts: List of document texts
            simulation_requirement: Description of simulation requirements
            additional_context: Additional context

        Returns:
            Ontology definition (entity_types, edge_types, etc.)
        """
        user_message = self._build_user_message(
            document_texts,
            simulation_requirement,
            additional_context
        )

        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        total_chars = sum(len(m["content"]) for m in messages)
        logger.info(f"Generating ontology — {len(document_texts)} docs, {total_chars:,} chars to LLM")

        # Local LLMs can be flaky with JSON — retry once on parse failure
        last_error = None
        for attempt in range(2):
            try:
                result = self.llm_client.chat_json(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=4096
                )
                break
            except ValueError as e:
                last_error = e
                if attempt == 0:
                    logger.warning(f"JSON parse failed, retrying: {e}")
                    continue
                raise last_error

        result = self._validate_and_process(result)
        logger.info(
            f"Ontology generated: {len(result.get('entity_types', []))} entities, "
            f"{len(result.get('edge_types', []))} edges"
        )
        return result
    
    # ~20K chars of document text fits within 8192 token context alongside
    # the system prompt (~1200 tokens) and output (~2000 tokens).
    MAX_TEXT_LENGTH_FOR_LLM = 20000

    # Number of evenly-spaced chunks to sample from long documents.
    SAMPLE_CHUNKS = 5

    def _sample_long_text(self, text: str) -> str:
        """Sample representative chunks from across a long document.

        Instead of truncating (which loses everything after the cutoff),
        this takes evenly-spaced chunks so the LLM sees the full arc of
        the document — intro, middle sections, and conclusion.
        """
        if len(text) <= self.MAX_TEXT_LENGTH_FOR_LLM:
            return text

        original_length = len(text)
        chunk_size = self.MAX_TEXT_LENGTH_FOR_LLM // self.SAMPLE_CHUNKS
        step = (original_length - chunk_size) / max(self.SAMPLE_CHUNKS - 1, 1)

        chunks = []
        for i in range(self.SAMPLE_CHUNKS):
            start = int(i * step)
            end = start + chunk_size
            chunk = text[start:end]

            # Trim to the nearest sentence boundary to avoid mid-word cuts
            if i < self.SAMPLE_CHUNKS - 1:
                for delim in ('.', '!', '?', '\n'):
                    last = chunk.rfind(delim)
                    if last > chunk_size // 2:
                        chunk = chunk[:last + 1]
                        break

            chunks.append(chunk.strip())

        sampled = "\n\n[...]\n\n".join(chunks)
        logger.info(
            f"Sampled {self.SAMPLE_CHUNKS} chunks from {original_length:,} char document "
            f"(reduced to {len(sampled):,} chars)"
        )
        return sampled

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Build the user message."""
        combined_text = "\n\n---\n\n".join(document_texts)
        combined_text = self._sample_long_text(combined_text)

        message = f"""## Simulation Requirement

{simulation_requirement}

## Document Content

{combined_text}
"""

        if additional_context:
            message += f"""
## Additional Notes

{additional_context}
"""

        message += """
Based on the above content, design entity types and relationship types suitable for social opinion simulation.

**Rules that MUST be followed**:
1. You must output exactly 10 entity types
2. The last 2 must be fallback types: Person (individual fallback) and Organization (organization fallback)
3. The first 8 should be specific types designed based on the text content. Each type is a category (e.g. WaterCompany, RegulatoryBody), never a single named organisation or person; put named entities in `examples`
4. All entity types must be real-world agents that can post on social media, not abstract concepts
5. Attribute names must NOT use reserved words like name, uuid, group_id — use full_name, org_name, etc. instead
"""

        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and post-process results."""
        if "entity_types" not in result:
            logger.warning("LLM returned no entity_types — using empty list")
            result["entity_types"] = []
        if "edge_types" not in result:
            logger.warning("LLM returned no edge_types — will generate defaults")
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # Validate entity types
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Ensure description does not exceed 100 characters
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # Validate relationship types
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        # Zep API limits: max 10 custom entity types, max 10 custom edge types
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10
        
        # Fallback type definitions
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        # Check if fallback types already exist
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        # Fallback types to add
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            # If adding would exceed 10, remove some existing types
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Calculate how many to remove
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Remove from the end (preserve the more important specific types at the front)
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            # Add fallback types
            result["entity_types"].extend(fallbacks_to_add)
        
        # If edge_types is empty, generate sensible defaults based on entity types
        if not result["edge_types"]:
            entity_names = {e["name"] for e in result["entity_types"]}
            default_edges = [
                {"name": "AFFILIATED_WITH", "description": "Entity is affiliated with or belongs to an organization.", "source_targets": [], "attributes": []},
                {"name": "RESPONDS_TO", "description": "Entity responds to or reacts to another entity's actions.", "source_targets": [], "attributes": []},
                {"name": "REPORTS_ON", "description": "Entity reports on or covers another entity.", "source_targets": [], "attributes": []},
                {"name": "COLLABORATES_WITH", "description": "Entity collaborates or works with another entity.", "source_targets": [], "attributes": []},
                {"name": "SUPPORTS", "description": "Entity supports or endorses another entity.", "source_targets": [], "attributes": []},
                {"name": "OPPOSES", "description": "Entity opposes or criticizes another entity.", "source_targets": [], "attributes": []},
            ]
            result["edge_types"] = default_edges
            logger.info(f"Generated {len(default_edges)} default edge types (LLM produced none)")

        # Final safeguard: ensure limits are not exceeded (defensive programming)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    

