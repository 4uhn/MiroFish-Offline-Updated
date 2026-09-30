"""
NER/RE Extractor — entity and relation extraction via local LLM

Replaces Zep Cloud's built-in NER/RE pipeline.
Uses LLMClient.chat_json() with a structured prompt to extract
entities and relations from text chunks, guided by the graph's ontology.

Entity types are constrained to the ontology at decode time (JSON schema enum)
and again in validation. Anything that is not an ontology actor (places,
substances, events) is typed "Entity": it stays in the graph for search but
gets no type label, so it can never become a simulation agent.

A smaller extraction model (qwen3:0.6b) was tried and dropped: it cut graph
build from ~27 min to <1 min but mistyped entities (chemicals as Person,
council areas as the water company), and those entities became agents.
"""

import logging
import os
from typing import Dict, Any, List, Optional

from ..utils.llm_client import LLMClient
from ..utils.entity_names import compact_key, find_alias, is_mentioned, same_entity

logger = logging.getLogger('mirofish.ner_extractor')

# Type for extracted things that are not ontology actors (no Neo4j type label)
GENERIC_ENTITY_TYPE = "Entity"
GENERIC_RELATION_TYPE = "RELATED_TO"

# System prompt template for NER/RE extraction
_SYSTEM_PROMPT = """You are a Named Entity Recognition and Relation Extraction system.
Given a text and an ontology (entity types + relation types), extract all entities and relations.

ONTOLOGY:
{ontology_description}

RULES:
1. Give every entity exactly one type from the ontology's Entity Types. If an entity is not an actor that fits any of them (a place, postcode, substance, event, document, concept), use the type "Entity". Facilities and sites (treatment works, reservoirs, campuses, buildings, distribution points) are "Entity" unless the text shows them acting as an organisation. Never invent new types.
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


def ground_extraction(
    extraction: Dict[str, Any], grounding: List[Dict[str, Any]], text: str
) -> Dict[str, Any]:
    """Keep only what a batch of agent activity supports.

    Each grounding item is one activity: its actor, the actor's own words and
    the users the actor acted on. A relation stands only when its source is
    an actor and its target is named in that actor's own words or is someone
    the actor acted on. An entity stands only when the text names it.

    In Run 12, NER on these batches invented relation targets: 7 "X blames
    the Environment Agency" edges, one for Ryan Thompson, whose only activity
    was reposting a post that never named the agency; and "Local campaigners
    blame Bristol Water" from a post that named no one. The report then
    repeated them as findings.
    """
    def supported(relation: Dict[str, Any]) -> bool:
        target = relation["target"]
        return any(
            same_entity(relation["source"], item["actor"])
            and (is_mentioned(target, item.get("own_words", ""))
                 or any(same_entity(target, name) for name in item.get("acted_on", [])))
            for item in grounding
        )

    relations = [r for r in extraction.get("relations", []) if supported(r)]
    endpoints = {r["source"].lower() for r in relations} | {r["target"].lower() for r in relations}
    entities = [
        e for e in extraction.get("entities", [])
        if e["name"].lower() in endpoints or is_mentioned(e["name"], text)
    ]

    dropped_relations = [r for r in extraction.get("relations", []) if r not in relations]
    dropped_entities = [e["name"] for e in extraction.get("entities", []) if e not in entities]
    if dropped_relations:
        logger.info("NER grounding: dropped %d unsupported relations: %s", len(dropped_relations),
                    "; ".join(r["fact"][:80] for r in dropped_relations[:10]))
    if dropped_entities:
        logger.info("NER grounding: dropped %d entities the text does not name: %s",
                    len(dropped_entities), ", ".join(dropped_entities[:10]))
    return {"entities": entities, "relations": relations}


class NERExtractor:
    """Extract entities and relations from text using local LLM."""

    def __init__(self, llm_client: Optional[LLMClient] = None, max_retries: int = 2):
        # Graph memory NER shares Ollama with the running simulation, so a
        # request can sit in Ollama's queue for minutes. With the SDK's default
        # 300s timeout and hidden retries, Run 9 lost 10 min to two silent
        # timeouts on one chunk. Wait longer and retry in the logged loop below.
        self.llm = llm_client or LLMClient(
            timeout=float(os.environ.get("NER_REQUEST_TIMEOUT", "900")), max_retries=0)
        self.max_retries = max_retries

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

        schema = self._build_schema(ontology)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.llm.chat_json(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                    schema=schema,
                    schema_name="ner_extraction",
                )
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

        logger.error(
            "NER extraction failed after all attempts: %s", last_error
        )
        return {"entities": [], "relations": []}

    @staticmethod
    def _entity_type_names(ontology: Dict[str, Any]) -> List[str]:
        names = []
        for et in ontology.get("entity_types", []):
            name = (et.get("name", "") if isinstance(et, dict) else str(et)).strip()
            if name and name not in names:
                names.append(name)
        return names

    @staticmethod
    def _relation_type_names(ontology: Dict[str, Any]) -> List[str]:
        names = []
        for rt in ontology.get("relation_types", ontology.get("edge_types", [])):
            name = (rt.get("name", "") if isinstance(rt, dict) else str(rt)).strip()
            if name and name not in names:
                names.append(name)
        return names

    def _build_schema(self, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """JSON schema for extraction output; entity and relation types are enums when the ontology defines them."""
        type_names = self._entity_type_names(ontology)
        entity_type = {"type": "string"}
        if type_names:
            entity_type = {"type": "string", "enum": type_names + [GENERIC_ENTITY_TYPE]}
        relation_names = self._relation_type_names(ontology)
        relation_type = {"type": "string"}
        if relation_names:
            relation_type = {"type": "string", "enum": relation_names + [GENERIC_RELATION_TYPE]}
        return {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": entity_type,
                            "attributes": {"type": "object"},
                        },
                        "required": ["name", "type"],
                    },
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "type": relation_type,
                            "fact": {"type": "string"},
                        },
                        "required": ["source", "target", "type", "fact"],
                    },
                },
            },
            "required": ["entities", "relations"],
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

        # Canonical ontology type names, looked up case-insensitively
        canonical_types = {n.lower(): n for n in self._entity_type_names(ontology)}
        retyped = []

        canonical_relations = {n.lower(): n for n in self._relation_type_names(ontology)}
        remapped_relations = []

        # Names that are schema vocabulary, not entities ("RegulatoryBody", "WORKS_FOR")
        schema_keys = {
            compact_key(n)
            for n in list(canonical_types.values()) + list(canonical_relations.values())
            + [GENERIC_ENTITY_TYPE, GENERIC_RELATION_TYPE]
        }
        dropped_names = set()
        # alias name (lowercase) -> the first spelling seen in this chunk
        alias_of: Dict[str, str] = {}

        # Clean entities
        cleaned_entities = []
        seen_names = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            etype = str(entity.get("type", GENERIC_ENTITY_TYPE)).strip()
            if not name:
                continue

            if compact_key(name) in schema_keys:
                dropped_names.add(name.lower())
                continue

            # Deduplicate by normalized name, then by alias
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            alias = find_alias(name, [e["name"] for e in cleaned_entities])
            if alias:
                alias_of[name_lower] = alias
                continue
            seen_names.add(name_lower)

            # Off-ontology types become generic entities: kept for graph search,
            # but never labelled, so they cannot become agents. This also keeps
            # LLM-invented strings out of the Cypher label.
            if canonical_types:
                canonical = canonical_types.get(etype.lower())
                if canonical is None and etype != GENERIC_ENTITY_TYPE:
                    retyped.append(f"{name}:{etype}")
                etype = canonical or GENERIC_ENTITY_TYPE

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
            if source.lower() in dropped_names or target.lower() in dropped_names:
                continue
            if compact_key(source) in schema_keys or compact_key(target) in schema_keys:
                continue
            source = alias_of.get(source.lower(), source)
            target = alias_of.get(target.lower(), target)
            if source.lower() == target.lower():
                continue

            if canonical_relations:
                canonical_rel = canonical_relations.get(rtype.lower())
                if canonical_rel is None:
                    remapped_relations.append(rtype)
                    canonical_rel = GENERIC_RELATION_TYPE
                rtype = canonical_rel

            # Ensure source and target entities exist
            # (they might not if LLM hallucinated a relation without the entity)
            if source.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": source,
                    "type": GENERIC_ENTITY_TYPE,
                    "attributes": {},
                })
                entity_names_lower.add(source.lower())

            if target.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": target,
                    "type": GENERIC_ENTITY_TYPE,
                    "attributes": {},
                })
                entity_names_lower.add(target.lower())

            cleaned_relations.append({
                "source": source,
                "target": target,
                "type": rtype,
                "fact": fact or f"{source} {rtype} {target}",
            })

        if remapped_relations:
            logger.info("NER: %d off-ontology relation types set to %s: %s",
                        len(remapped_relations), GENERIC_RELATION_TYPE,
                        ", ".join(remapped_relations[:10]))
        if dropped_names:
            logger.info("NER: dropped %d entities named after schema types: %s",
                        len(dropped_names), ", ".join(sorted(dropped_names)[:10]))
        if alias_of:
            logger.info("NER: merged %d aliases: %s", len(alias_of),
                        ", ".join(f"{a}->{b}" for a, b in list(alias_of.items())[:10]))

        if retyped:
            logger.info("NER: %d off-ontology entity types set to %s: %s",
                        len(retyped), GENERIC_ENTITY_TYPE, ", ".join(retyped[:10]))

        return {
            "entities": cleaned_entities,
            "relations": cleaned_relations,
        }
