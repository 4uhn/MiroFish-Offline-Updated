"""
NER/RE Extractor — entity and relation extraction via local LLM

Replaces Zep Cloud's built-in NER/RE pipeline.
Uses LLMClient.chat_json() with a structured prompt to extract
entities and relations from text chunks, guided by the graph's ontology.

Tiered model routing: uses the fast model (e.g. qwen3:0.6b) for extraction
when available, with automatic fallback to the quality model on failure.
"""

import logging
from typing import Dict, Any, List, Optional

from ..utils.llm_client import LLMClient
from ..utils.model_router import get_router, TaskType

logger = logging.getLogger('mirofish.ner_extractor')

# System prompt template for NER/RE extraction
_SYSTEM_PROMPT = """You are a Named Entity Recognition and Relation Extraction system.
Given a text and an ontology (entity types + relation types), extract all entities and relations.

ONTOLOGY:
{ontology_description}

RULES:
1. Only extract entity types and relation types defined in the ontology.
2. Normalize entity names: strip whitespace, use canonical form (e.g., "Jack Ma" not "ma jack").
3. Each entity must have: name, type (from ontology), and optional attributes.
4. Each relation must have: source entity name, target entity name, type (from ontology), and a fact sentence describing the relationship.
5. If no entities or relations are found, return empty lists.
6. Be precise — only extract what is explicitly stated or strongly implied in the text.

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "...", "type": "...", "attributes": {{"key": "value"}}}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "type": "...", "fact": "..."}}
  ]
}}"""

_USER_PROMPT = """Extract entities and relations from the following text:

{text}"""


class NERExtractor:
    """Extract entities and relations from text using local LLM.

    Uses the fast model tier (e.g. qwen3:0.6b) when available. If the fast
    model fails to produce valid JSON after retries, falls back to the
    quality model (e.g. qwen3:8b) for one final attempt.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, max_retries: int = 2):
        router = get_router()
        self.llm = llm_client or router.get_client(TaskType.EXTRACTION)
        self._quality_fallback = (
            router.get_quality_client() if router.fast_available else None
        )
        self.max_retries = max_retries
        self._fast_calls = 0
        self._fallback_calls = 0

    def extract(self, text: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities and relations from text, guided by ontology.

        Args:
            text: Input text chunk
            ontology: Dict with 'entity_types' and 'relation_types' from graph

        Returns:
            Dict with 'entities' and 'relations' lists
        """
        if not text or not text.strip():
            return {"entities": [], "relations": []}

        ontology_desc = self._format_ontology(ontology)
        system_msg = _SYSTEM_PROMPT.format(ontology_description=ontology_desc)
        user_msg = _USER_PROMPT.format(text=text.strip())

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.llm.chat_json(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                )
                self._fast_calls += 1
                return self._validate_and_clean(result, ontology)

            except ValueError as e:
                last_error = e
                logger.warning(
                    "NER extraction failed (attempt %d/%d): invalid JSON — %s",
                    attempt + 1, self.max_retries + 1, e,
                )
            except Exception as e:
                last_error = e
                logger.error("NER extraction error: %s", e)
                if attempt >= self.max_retries:
                    break

        if self._quality_fallback is not None:
            logger.info("Fast model failed NER — falling back to quality model")
            try:
                result = self._quality_fallback.chat_json(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                )
                self._fallback_calls += 1
                return self._validate_and_clean(result, ontology)
            except Exception as e:
                logger.error("Quality fallback also failed: %s", e)

        logger.error(
            "NER extraction failed after all attempts: %s", last_error
        )
        return {"entities": [], "relations": []}

    @property
    def routing_stats(self) -> Dict[str, int]:
        return {
            "fast_calls": self._fast_calls,
            "fallback_calls": self._fallback_calls,
        }

    def _format_ontology(self, ontology: Dict[str, Any]) -> str:
        """Format ontology dict into readable text for the LLM prompt."""
        parts = []

        entity_types = ontology.get("entity_types", [])
        if entity_types:
            parts.append("Entity Types:")
            for et in entity_types:
                if isinstance(et, dict):
                    name = et.get("name", str(et))
                    desc = et.get("description", "")
                    attrs = et.get("attributes", [])
                    line = f"  - {name}"
                    if desc:
                        line += f": {desc}"
                    if attrs:
                        attr_names = [a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in attrs]
                        line += f" (attributes: {', '.join(attr_names)})"
                    parts.append(line)
                else:
                    parts.append(f"  - {et}")

        relation_types = ontology.get("relation_types", ontology.get("edge_types", []))
        if relation_types:
            parts.append("\nRelation Types:")
            for rt in relation_types:
                if isinstance(rt, dict):
                    name = rt.get("name", str(rt))
                    desc = rt.get("description", "")
                    source_targets = rt.get("source_targets", [])
                    line = f"  - {name}"
                    if desc:
                        line += f": {desc}"
                    if source_targets:
                        st_strs = [f"{st.get('source', '?')} → {st.get('target', '?')}" for st in source_targets]
                        line += f" ({', '.join(st_strs)})"
                    parts.append(line)
                else:
                    parts.append(f"  - {rt}")

        if not parts:
            parts.append("No specific ontology defined. Extract all entities and relations you find.")

        return "\n".join(parts)

    def _validate_and_clean(
        self, result: Dict[str, Any], ontology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and normalize LLM output."""
        entities = result.get("entities", [])
        relations = result.get("relations", [])

        # Get valid type names from ontology
        valid_entity_types = set()
        for et in ontology.get("entity_types", []):
            if isinstance(et, dict):
                valid_entity_types.add(et.get("name", "").strip())
            else:
                valid_entity_types.add(str(et).strip())

        valid_relation_types = set()
        for rt in ontology.get("relation_types", ontology.get("edge_types", [])):
            if isinstance(rt, dict):
                valid_relation_types.add(rt.get("name", "").strip())
            else:
                valid_relation_types.add(str(rt).strip())

        # Clean entities
        cleaned_entities = []
        seen_names = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            etype = str(entity.get("type", "Entity")).strip()
            if not name:
                continue

            # Deduplicate by normalized name
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)

            # If ontology has types, warn but keep entities with unknown types
            if valid_entity_types and etype not in valid_entity_types:
                logger.debug(f"Entity '{name}' has type '{etype}' not in ontology, keeping anyway")

            cleaned_entities.append({
                "name": name,
                "type": etype,
                "attributes": entity.get("attributes", {}),
            })

        # Clean relations
        cleaned_relations = []
        entity_names_lower = {e["name"].lower() for e in cleaned_entities}
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source", "")).strip()
            target = str(relation.get("target", "")).strip()
            rtype = str(relation.get("type", "RELATED_TO")).strip()
            fact = str(relation.get("fact", "")).strip()

            if not source or not target:
                continue

            # Ensure source and target entities exist
            # (they might not if LLM hallucinated a relation without the entity)
            if source.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": source,
                    "type": "Entity",
                    "attributes": {},
                })
                entity_names_lower.add(source.lower())

            if target.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": target,
                    "type": "Entity",
                    "attributes": {},
                })
                entity_names_lower.add(target.lower())

            cleaned_relations.append({
                "source": source,
                "target": target,
                "type": rtype,
                "fact": fact or f"{source} {rtype} {target}",
            })

        return {
            "entities": cleaned_entities,
            "relations": cleaned_relations,
        }
