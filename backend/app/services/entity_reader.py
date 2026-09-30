"""
Entity reading and filtering service.
Reads nodes from Neo4j graph, filters out meaningful entity type nodes.

Replaces zep_entity_reader.py — all Zep Cloud calls replaced by GraphStorage.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

from ..utils.logger import get_logger
from ..utils.entity_names import is_mentioned, name_tokens, same_entity
from ..storage import GraphStorage
from ..storage.graph_storage import SEED_ONLY, in_run_scope

logger = get_logger('mirofish.entity_reader')

@dataclass
class EntityNode:
    """Entity node data structure"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    related_edges: List[Dict[str, Any]] = field(default_factory=list)

    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        """Get entity type (exclude default Entity label)"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None

@dataclass
class FilteredEntities:
    """Filtered entity set"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }

def _type_mentioned(entity_type: Optional[str], text: str) -> bool:
    """True when text names the entity's type as a kind of actor
    ("local campaigners" for EnvironmentalCampaigner)."""
    type_tokens = name_tokens(entity_type or "")
    if not type_tokens:
        return False
    head = type_tokens[-1]
    return any(w == head or w.rstrip("s") == head for w in name_tokens(text))


def select_agent_entities(
    entities: List[EntityNode], cap: int, requirement: Optional[str] = None
) -> List[EntityNode]:
    """Choose which graph entities become institutional agents.

    Aliases ("NHS Bristol ICB" / "NHS Bristol, North Somerset and South
    Gloucestershire ICB") are merged into the best-connected name, and their
    edges are pooled. Groups are then ranked by degree, taking the
    best-connected entity of each type first so that one heavily discussed
    type cannot crowd out the others. The remaining slots go by degree.

    Replaces a plain entities[:cap], whose graph-read order dropped the two
    hubs in Run 8 (Environment Agency, 64 edges; Bristol Water, 61) and
    included NHS ICB twice under different names.

    When the simulation requirement is given, entities it names (under any
    alias) are chosen first, then one entity for each actor type it names.
    """
    groups: List[List[EntityNode]] = []
    for entity in sorted(entities, key=lambda e: len(e.related_edges), reverse=True):
        for group in groups:
            if same_entity(entity.name, group[0].name):
                group.append(entity)
                break
        else:
            groups.append([entity])

    merged: List[EntityNode] = []
    names: Dict[int, List[str]] = {}
    for group in groups:
        head = group[0]
        names[id(head)] = [e.name for e in group]
        if len(group) > 1:
            seen = {(e.get("edge_name"), e.get("fact")) for e in head.related_edges}
            for alias in group[1:]:
                for edge in alias.related_edges:
                    key = (edge.get("edge_name"), edge.get("fact"))
                    if key not in seen:
                        seen.add(key)
                        head.related_edges.append(edge)
            logger.info(
                f"Merged entity aliases into '{head.name}': "
                f"{[a.name for a in group[1:]]}"
            )
        merged.append(head)

    ranked = sorted(merged, key=lambda e: len(e.related_edges), reverse=True)
    chosen: List[EntityNode] = []
    seen_types: Set[str] = set()

    # Entities the user's prompt names come first. In Run 10 the prompt listed
    # the Environment Agency, NHS Bristol and the local campaigners, and
    # degree ranking dropped all three.
    if requirement:
        named = [e for e in ranked if any(is_mentioned(n, requirement) for n in names[id(e)])]
        # "NHS" and "NHS Bristol" both match "NHS Bristol": keep the more specific one
        token_lists = {id(e): name_tokens(e.name) for e in named}
        named = [
            e for e in named
            if not any(
                o is not e and len(token_lists[id(o)]) > len(token_lists[id(e)])
                and token_lists[id(o)][:len(token_lists[id(e)])] == token_lists[id(e)]
                for o in named
            )
        ]
        type_named = [
            e for e in ranked
            if e not in named and _type_mentioned(e.get_entity_type(), requirement)
        ]
        for entity in named + type_named:
            etype = entity.get_entity_type() or "Entity"
            # A type the prompt names only as a type gets one agent
            if entity in type_named and etype in seen_types:
                continue
            if len(chosen) < cap:
                chosen.append(entity)
                seen_types.add(etype)
        if chosen:
            logger.info(f"Named in the simulation requirement: {[e.name for e in chosen]}")

    for entity in ranked:
        if entity in chosen:
            continue
        etype = entity.get_entity_type() or "Entity"
        if etype not in seen_types and len(chosen) < cap:
            chosen.append(entity)
            seen_types.add(etype)
    for entity in ranked:
        if len(chosen) >= cap:
            break
        if entity not in chosen:
            chosen.append(entity)

    return sorted(chosen, key=lambda e: len(e.related_edges), reverse=True)


# Entity types that represent locations, objects, or abstract concepts — NOT social media agents.
# These are excluded from simulation agent generation even if the ontology includes them.
NON_AGENT_ENTITY_TYPES = {
    "location", "place", "address", "postalcode", "postcode", "zipcode",
    "building", "facility", "venue", "residence", "resident", "hall",
    "street", "road", "area", "district", "region", "city", "country",
    "campus", "dormitory", "clinic", "ward",
    "event", "incident", "topic", "concept", "policy", "legislation",
    "date", "time", "period", "timeline",
    "document", "report", "publication", "article",
    "product", "service", "technology", "tool",
    "disease", "condition", "symptom", "treatment", "vaccine",
}

class EntityReader:
    """
    Entity reader and filter service (via GraphStorage / Neo4j)

    Main functions:
    1. Read all nodes from the knowledge graph
    2. Filter nodes that have defined entity types (custom labels beyond just "Entity")
    3. Get related edges and associated node info for each entity
    """

    def __init__(self, storage: GraphStorage):
        self.storage = storage

    @staticmethod
    def _is_garbage_entity_name(name: str) -> bool:
        """Heuristic check: is this entity name a fragment, generic word, or non-entity?"""
        import re
        stripped = name.strip()
        # Too short (single word under 3 chars) or empty
        if len(stripped) < 2:
            return True
        # Pure number or year (e.g., "2015", "100")
        if re.match(r'^\d+$', stripped):
            return True
        # Generic common words that are not named entities
        GENERIC_WORDS = {
            "staff", "residents", "students", "people", "young people",
            "children", "adults", "families", "parents", "teachers",
            "workers", "members", "visitors", "patients", "users",
            "public", "community", "population", "audience", "group",
            "friday", "saturday", "sunday", "monday", "tuesday",
            "wednesday", "thursday", "weekend", "morning", "evening",
        }
        if stripped.lower() in GENERIC_WORDS:
            return True
        # Single common word (not a proper noun) — allow if capitalized like a name
        if len(stripped.split()) == 1 and stripped.islower():
            return True
        # Country/region names used as standalone entities (too broad)
        OVERLY_BROAD = {"uk", "us", "usa", "eu", "china", "india", "france", "germany", "england", "scotland", "wales"}
        if stripped.lower() in OVERLY_BROAD:
            return True
        return False

    @staticmethod
    def _normalize_name_for_dedup(name: str) -> str:
        """Normalize entity name for deduplication comparison."""
        import re
        n = name.lower().strip()
        # Remove parenthetical suffixes like (NHS), (LocalGovernment), (University), etc.
        n = re.sub(r'\s*\([^)]*\)\s*$', '', n)
        # Remove common prefixes/suffixes
        n = re.sub(r'^(the|a|an)\s+', '', n)
        # Remove punctuation
        n = re.sub(r'[^\w\s]', '', n)
        # Collapse whitespace
        n = re.sub(r'\s+', ' ', n).strip()
        return n

    @staticmethod
    def _looks_like_location(name: str) -> bool:
        """Heuristic check: does this entity name look like a location/address rather than a person/org?"""
        import re
        # UK/US postcode pattern (e.g. "CT1 3NG", "Canterbury CT2 7NZ")
        if re.search(r'\b[A-Z]{1,2}\d{1,2}\s*\d[A-Z]{2}\b', name):
            return True
        # US zip code
        if re.search(r'\b\d{5}(-\d{4})?\b', name) and len(name) < 20:
            return True
        # Pure street/road names (but not organisations with these words)
        if re.search(r'\b(Road|Street|Lane|Avenue|Drive|Close|Way|Crescent|Place|Terrace)\b', name, re.IGNORECASE):
            if not re.search(r'\b(Council|Authority|Agency|Foundation|Institute|Hospital|University|College|School)\b', name, re.IGNORECASE):
                return True
        # Building/residence names that end with typical building suffixes
        if re.search(r'\b(Halls?\s+of\s+Residence|Building|Campus|Dormitory|Clinic|Hall)\s*$', name, re.IGNORECASE):
            return True
        return False

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        logger.info(f"Getting all nodes in graph {graph_id}...")
        nodes = self.storage.get_all_nodes(graph_id)
        logger.info(f"Got {len(nodes)} nodes total")
        return nodes

    def get_location_names(self, graph_id: str) -> List[str]:
        """Extract location/place entity names from the graph for scenario context."""
        location_types = {"location", "place", "city", "region", "area", "district", "country"}
        all_nodes = self.get_all_nodes(graph_id)
        locations = []
        for node in all_nodes:
            labels = node.get("labels", [])
            custom_labels = [la for la in labels if la not in ["Entity", "Node"]]
            for label in custom_labels:
                if label.lower() in location_types:
                    name = node.get("name", "").strip()
                    if name and len(name) > 2 and not self._is_garbage_entity_name(name):
                        locations.append(name)
                    break
        return locations

    # Agents are prepared from the seed document alone. On a reused graph,
    # earlier runs' agent claims ("Bristol City Council collaborates with the
    # Environment Agency") would otherwise become entity context for profiles.

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        logger.info(f"Getting all edges in graph {graph_id}...")
        edges = [e for e in self.storage.get_all_edges(graph_id) if in_run_scope(e, SEED_ONLY)]
        logger.info(f"Got {len(edges)} seed edges")
        return edges

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        try:
            return [e for e in self.storage.get_node_edges(node_uuid) if in_run_scope(e, SEED_ONLY)]
        except Exception as e:
            logger.warning(f"Failed to get edges for node {node_uuid}: {str(e)}")
            return []

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        logger.info(f"Starting to filter entities in graph {graph_id}...")

        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)

        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []

        node_map = {n["uuid"]: n for n in all_nodes}

        filtered_entities = []
        entity_types_found: Set[str] = set()
        _seen_names: Dict[str, str] = {}  # normalized_name -> original_name for deduplication

        for node in all_nodes:
            labels = node.get("labels", [])

            # Filter: must have custom labels beyond just "Entity"/"Node"
            custom_labels = [la for la in labels if la not in ["Entity", "Node"]]

            if not custom_labels:
                continue

            # Agents come from the seed document. Entities first named in an
            # earlier run's agent posts (synthetic residents, Run 10's invented
            # "River Wye") stay in the graph for the report but must not
            # become institutions when the graph is prepared again.
            if node.get("source") == "simulation":
                logger.debug(f"Skipping simulation-created entity: {node.get('name', '?')}")
                continue

            # If specific types requested, check for match
            if defined_entity_types:
                matching_labels = [la for la in custom_labels if la in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            # Skip non-agent entity types (locations, objects, concepts, etc.)
            if entity_type.lower() in NON_AGENT_ENTITY_TYPES:
                logger.debug(f"Skipping non-agent entity: {node.get('name', '?')} (type: {entity_type})")
                continue

            # Skip entities whose names look like addresses/postcodes
            name = node.get("name", "")
            if self._looks_like_location(name):
                logger.debug(f"Skipping location-like entity: {name} (type: {entity_type})")
                continue

            # Skip garbage entity names (fragments, generic words, numbers)
            if self._is_garbage_entity_name(name):
                logger.debug(f"Skipping garbage entity name: {name} (type: {entity_type})")
                continue

            # Deduplication: skip if a similar entity name was already added
            dedup_key = self._normalize_name_for_dedup(name)
            if dedup_key in _seen_names:
                existing = _seen_names[dedup_key]
                # Keep the one with the longer/more specific name
                if len(name) <= len(existing):
                    logger.debug(f"Skipping duplicate entity: '{name}' (already have '{existing}')")
                    continue
                else:
                    # Replace the shorter one — remove it from filtered_entities
                    filtered_entities = [e for e in filtered_entities if self._normalize_name_for_dedup(e.name) != dedup_key]
                    logger.debug(f"Replacing entity '{existing}' with longer name '{name}'")
            _seen_names[dedup_key] = name

            # Also check if name is a substring/acronym of an already-seen entity
            skip_as_acronym = False
            for seen_key, seen_name in list(_seen_names.items()):
                if seen_key == dedup_key:
                    continue
                # Check if current name is an acronym of a seen name (e.g., "JCVI" vs "Joint Committee on Vaccination and Immunisation")
                if len(name) <= 6 and name.isupper():
                    seen_words = seen_name.split()
                    acronym = ''.join(w[0] for w in seen_words if w[0].isupper())
                    if name.upper() == acronym.upper():
                        logger.debug(f"Skipping acronym entity '{name}' (already have '{seen_name}')")
                        skip_as_acronym = True
                        break
                # Check reverse: if a seen name is an acronym of current name
                if len(seen_name) <= 6 and seen_name.isupper():
                    current_words = name.split()
                    acronym = ''.join(w[0] for w in current_words if w[0].isupper())
                    if seen_name.upper() == acronym.upper():
                        # Remove the acronym, keep the full name
                        filtered_entities = [e for e in filtered_entities if e.name != seen_name]
                        del _seen_names[seen_key]
                        _seen_names[dedup_key] = name
                        logger.debug(f"Replacing acronym entity '{seen_name}' with full name '{name}'")
                        break
            if skip_as_acronym:
                continue

            entity_types_found.add(entity_type)

            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
            )

            if enrich_with_edges:
                related_edges = []
                related_node_uuids: Set[str] = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge.get("fact", ""),
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge.get("fact", ""),
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges

                related_nodes = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        related_node = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": related_node["uuid"],
                            "name": related_node["name"],
                            "labels": related_node.get("labels", []),
                            "summary": related_node.get("summary", ""),
                        })

                entity.related_nodes = related_nodes

            filtered_entities.append(entity)

        logger.info(f"Filter completed: total nodes {total_count}, matched {len(filtered_entities)}, "
                     f"entity types: {entity_types_found}")

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str
    ) -> Optional[EntityNode]:
        try:
            # Get the node directly by UUID (O(1) lookup)
            node = self.storage.get_node(entity_uuid)
            if not node:
                return None

            # Get edges for this node (O(degree) via Cypher)
            edges = [e for e in self.storage.get_node_edges(entity_uuid) if in_run_scope(e, SEED_ONLY)]

            # Process related edges and collect related node UUIDs
            related_edges = []
            related_node_uuids: Set[str] = set()

            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge.get("fact", ""),
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge.get("fact", ""),
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])

            # Fetch related nodes individually (avoids loading ALL nodes)
            related_nodes = []
            for related_uuid in related_node_uuids:
                related_node = self.storage.get_node(related_uuid)
                if related_node and in_run_scope(related_node, SEED_ONLY):
                    related_nodes.append({
                        "uuid": related_node["uuid"],
                        "name": related_node["name"],
                        "labels": related_node.get("labels", []),
                        "summary": related_node.get("summary", ""),
                    })

            return EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
                related_edges=related_edges,
                related_nodes=related_nodes,
            )

        except Exception as e:
            logger.error(f"Failed to get entity {entity_uuid}: {str(e)}")
            return None

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges
        )
        return result.entities
