"""
Neo4jStorage — Neo4j Community Edition implementation of GraphStorage.

Replaces all Zep Cloud API calls with local Neo4j Cypher queries.
Includes: CRUD, NER/RE-based text ingestion, hybrid search, retry logic.
"""

import json
import time
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import (
    TransientError,
    ServiceUnavailable,
    SessionExpired,
)

from ..config import Config
from .graph_storage import GraphStorage
from .embedding_service import EmbeddingService
from .ner_extractor import NERExtractor, ground_extraction
from ..utils.entity_names import find_alias
from .search_service import SearchService
from . import neo4j_schema

logger = logging.getLogger('mirofish.neo4j_storage')


class Neo4jStorage(GraphStorage):
    """Neo4j CE implementation of the GraphStorage interface."""

    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1  # seconds

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
        ner_extractor: Optional[NERExtractor] = None,
    ):
        self._uri = uri or Config.NEO4J_URI
        self._user = user or Config.NEO4J_USER
        self._password = password or Config.NEO4J_PASSWORD

        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        self._embedding = embedding_service or EmbeddingService()
        self._ner = ner_extractor or NERExtractor()
        self._search = SearchService(self._embedding)
        # add_text runs NER concurrently during the graph memory drain; the
        # alias lookup + entity merge that follows must not interleave.
        self._write_lock = threading.Lock()

        # Initialize schema (indexes, constraints)
        self._ensure_schema()

    def close(self):
        """Close the Neo4j driver connection."""
        self._driver.close()

    def _ensure_schema(self):
        """Create indexes and constraints if they don't exist."""
        with self._driver.session() as session:
            for query in neo4j_schema.ALL_SCHEMA_QUERIES:
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"Schema query warning (may already exist): {e}")

    # ----------------------------------------------------------------
    # Retry wrapper
    # ----------------------------------------------------------------

    def _call_with_retry(self, func, *args, **kwargs):
        """
        Execute a function with retry on Neo4j transient errors.
        Replaces 3 different retry patterns from the Zep codebase.
        """
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (TransientError, ServiceUnavailable, SessionExpired) as e:
                last_error = e
                wait = self.RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(
                    f"Neo4j transient error (attempt {attempt + 1}/{self.MAX_RETRIES}), "
                    f"retrying in {wait}s: {e}"
                )
                time.sleep(wait)
            except Exception:
                raise

        raise last_error  # type: ignore

    # ----------------------------------------------------------------
    # Graph lifecycle
    # ----------------------------------------------------------------

    def create_graph(self, name: str, description: str = "") -> str:
        graph_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        def _create(tx):
            tx.run(
                """
                CREATE (g:Graph {
                    graph_id: $graph_id,
                    name: $name,
                    description: $description,
                    ontology_json: '{}',
                    created_at: $created_at
                })
                """,
                graph_id=graph_id,
                name=name,
                description=description,
                created_at=now,
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _create)

        logger.info(f"Created graph '{name}' with id {graph_id}")
        return graph_id

    def delete_graph(self, graph_id: str) -> None:
        def _delete(tx):
            # Delete all entities and their relationships
            tx.run(
                "MATCH (n {graph_id: $gid}) DETACH DELETE n",
                gid=graph_id,
            )
            # Delete graph node
            tx.run(
                "MATCH (g:Graph {graph_id: $gid}) DELETE g",
                gid=graph_id,
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _delete)
        logger.info(f"Deleted graph {graph_id}")

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        def _set(tx):
            tx.run(
                """
                MATCH (g:Graph {graph_id: $gid})
                SET g.ontology_json = $ontology_json
                """,
                gid=graph_id,
                ontology_json=json.dumps(ontology, ensure_ascii=False),
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _set)

    def get_ontology(self, graph_id: str) -> Dict[str, Any]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (g:Graph {graph_id: $gid}) RETURN g.ontology_json AS oj",
                gid=graph_id,
            )
            record = result.single()
            if record and record["oj"]:
                return json.loads(record["oj"])
            return {}

    # ----------------------------------------------------------------
    # Add data (NER → nodes/edges)
    # ----------------------------------------------------------------

    def _resolve_aliases(
        self, graph_id: str, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Rename extracted entities to an existing graph node's name when they are aliases.

        Nodes MERGE on the lowercase name, so without this "Environmental Agency" in
        one chunk and "Environment Agency" in another became two nodes (Run 8 had
        3 for the Environment Agency and 3 for the DWI).
        """
        with self._driver.session() as session:
            existing = [
                r["name"] for r in session.run(
                    "MATCH (n:Entity {graph_id: $gid}) RETURN n.name AS name", gid=graph_id
                ) if r["name"]
            ]
        if not existing:
            return entities, relations
        existing_lower = {n.lower() for n in existing}

        renamed: Dict[str, str] = {}
        resolved_entities = []
        for entity in entities:
            name = entity["name"]
            if name.lower() not in existing_lower:
                alias = find_alias(name, existing)
                if alias:
                    renamed[name.lower()] = alias
                    entity = {**entity, "name": alias}
            if any(e["name"].lower() == entity["name"].lower() for e in resolved_entities):
                continue
            resolved_entities.append(entity)

        if not renamed:
            return entities, relations
        logger.info(f"[add_text] Resolved {len(renamed)} aliases to existing nodes: {renamed}")

        resolved_relations = []
        for rel in relations:
            src = renamed.get(rel["source"].lower(), rel["source"])
            tgt = renamed.get(rel["target"].lower(), rel["target"])
            if src.lower() != tgt.lower():
                resolved_relations.append({**rel, "source": src, "target": tgt})
        return resolved_entities, resolved_relations

    def add_text(
        self,
        graph_id: str,
        text: str,
        source: str = "document",
        simulation_id: Optional[str] = None,
        grounding: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Process text: NER/RE → batch embed → create nodes/edges → return episode_id.

        source is "document" for the seed text or "simulation" for agent
        activity. It is stored on the episode, on each relation and on entities
        the text creates, so the report can tell seed facts from things agents
        made up: Run 10's graph held "River Wye" and "Civic Health Centre",
        which only agents' posts mentioned.

        simulation_id is stored on the episode and relations, and added to
        each entity's simulation_ids, so a report reads only its own run when
        a graph is reused. grounding filters the extraction to what the
        activities support (see ground_extraction).
        """
        episode_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Get ontology for NER guidance
        ontology = self.get_ontology(graph_id)

        # Extract entities and relations
        logger.info(f"[add_text] Starting NER extraction for chunk ({len(text)} chars)...")
        extraction = self._ner.extract(text, ontology)
        if grounding is not None:
            extraction = ground_extraction(extraction, grounding, text)
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        logger.info(
            f"[add_text] NER done: {len(entities)} entities, {len(relations)} relations"
        )

        with self._write_lock:
            return self._write_extraction(
                graph_id, text, episode_id, now, entities, relations, source, simulation_id
            )

    def _write_extraction(
        self, graph_id: str, text: str, episode_id: str, now: str,
        entities: List[Dict[str, Any]], relations: List[Dict[str, Any]],
        source: str = "document",
        simulation_id: Optional[str] = None,
    ) -> str:
        """Resolve aliases, embed and write one NER result. Caller holds _write_lock."""
        entities, relations = self._resolve_aliases(graph_id, entities, relations)

        # --- Batch embed all texts at once ---
        entity_summaries = [f"{e['name']} ({e['type']})" for e in entities]
        fact_texts = [r.get("fact", f"{r['source']} {r['type']} {r['target']}") for r in relations]
        all_texts_to_embed = entity_summaries + fact_texts

        all_embeddings: list = []
        if all_texts_to_embed:
            logger.info(f"[add_text] Batch-embedding {len(all_texts_to_embed)} texts...")
            try:
                all_embeddings = self._embedding.embed_batch(all_texts_to_embed)
            except Exception as e:
                logger.warning(f"[add_text] Batch embedding failed, falling back to empty: {e}")
                all_embeddings = [[] for _ in all_texts_to_embed]

        entity_embeddings = all_embeddings[:len(entities)]
        relation_embeddings = all_embeddings[len(entities):]
        logger.info(f"[add_text] Embedding done, writing to Neo4j...")

        with self._driver.session() as session:
            # Create episode node
            def _create_episode(tx):
                tx.run(
                    """
                    CREATE (ep:Episode {
                        uuid: $uuid,
                        graph_id: $graph_id,
                        data: $data,
                        processed: true,
                        source: $source,
                        simulation_id: $simulation_id,
                        created_at: $created_at
                    })
                    """,
                    uuid=episode_id,
                    graph_id=graph_id,
                    data=text,
                    source=source,
                    simulation_id=simulation_id,
                    created_at=now,
                )

            self._call_with_retry(session.execute_write, _create_episode)

            # MERGE entities (upsert by graph_id + name + primary label)
            entity_uuid_map: Dict[str, str] = {}  # name_lower -> uuid
            for idx, entity in enumerate(entities):
                ename = entity["name"]
                etype = entity["type"]
                attrs = entity.get("attributes", {})
                summary_text = entity_summaries[idx]
                embedding = entity_embeddings[idx] if idx < len(entity_embeddings) else []

                e_uuid = str(uuid.uuid4())
                entity_uuid_map[ename.lower()] = e_uuid

                def _merge_entity(tx, _uuid=e_uuid, _name=ename, _type=etype,
                                  _attrs=attrs, _embedding=embedding,
                                  _summary=summary_text, _now=now):
                    # MERGE by graph_id + lowercase name to deduplicate. Agent
                    # activity never rewrites a seed entity's attributes or
                    # embedding: the next prepare builds profiles from them.
                    result = tx.run(
                        """
                        MERGE (n:Entity {graph_id: $gid, name_lower: $name_lower})
                        ON CREATE SET
                            n.uuid = $uuid,
                            n.name = $name,
                            n.summary = $summary,
                            n.attributes_json = $attrs_json,
                            n.embedding = $embedding,
                            n.source = $source,
                            n.simulation_ids = CASE WHEN $sim IS NULL THEN [] ELSE [$sim] END,
                            n.created_at = $now
                        ON MATCH SET
                            n.summary = CASE WHEN n.summary = '' OR n.summary IS NULL
                                THEN $summary ELSE n.summary END,
                            n.attributes_json = CASE
                                WHEN $source = 'simulation' AND coalesce(n.source, 'document') <> 'simulation'
                                THEN n.attributes_json ELSE $attrs_json END,
                            n.embedding = CASE
                                WHEN $source = 'simulation' AND coalesce(n.source, 'document') <> 'simulation'
                                THEN n.embedding ELSE $embedding END,
                            n.simulation_ids = CASE
                                WHEN $sim IS NULL OR $sim IN coalesce(n.simulation_ids, [])
                                THEN n.simulation_ids ELSE coalesce(n.simulation_ids, []) + $sim END
                        RETURN n.uuid AS uuid
                        """,
                        gid=graph_id,
                        name_lower=_name.lower(),
                        uuid=_uuid,
                        name=_name,
                        summary=_summary,
                        attrs_json=json.dumps(_attrs, ensure_ascii=False),
                        embedding=_embedding,
                        source=source,
                        sim=simulation_id,
                        now=_now,
                    )
                    record = result.single()
                    return record["uuid"] if record else _uuid

                actual_uuid = self._call_with_retry(session.execute_write, _merge_entity)
                entity_uuid_map[ename.lower()] = actual_uuid

                # Add the entity type label only if the node has none yet. Each
                # episode's NER may type a known entity differently; adding every
                # guess left Run 9's Eastville Chemical Works labelled
                # ConstructionCompany + EnvironmentalGroup + Organization.
                if etype and etype != "Entity":
                    etype = etype.replace("`", "")
                    try:
                        def _add_label(tx, _name_lower=ename.lower()):
                            tx.run(
                                f"MATCH (n:Entity {{graph_id: $gid, name_lower: $nl}}) "
                                f"WHERE all(l IN labels(n) WHERE l IN ['Entity', 'Node']) "
                                f"SET n:`{etype}`",
                                gid=graph_id,
                                nl=_name_lower,
                            )
                        self._call_with_retry(session.execute_write, _add_label)
                    except Exception as e:
                        logger.warning(f"Failed to add label '{etype}' to '{ename}': {e}")

            # Create relations. A fact already recorded between the same two
            # entities by the same source and run gains this episode instead of
            # a copy: Run 12 held "Local campaigners blame Bristol Water for
            # ignoring their warnings" three times, which read as three voices.
            merged = 0
            for idx, relation in enumerate(relations):
                source_name = relation["source"]
                target_name = relation["target"]
                rtype = relation["type"]
                fact = relation["fact"]

                source_uuid = entity_uuid_map.get(source_name.lower())
                target_uuid = entity_uuid_map.get(target_name.lower())

                if not source_uuid or not target_uuid:
                    logger.warning(
                        f"Skipping relation {source_name}->{target_name}: "
                        f"entity not found in extraction results"
                    )
                    continue

                fact_embedding = relation_embeddings[idx] if idx < len(relation_embeddings) else []
                r_uuid = str(uuid.uuid4())

                def _write_relation(tx, _r_uuid=r_uuid, _source_uuid=source_uuid,
                                    _target_uuid=target_uuid, _rtype=rtype,
                                    _fact=fact, _fact_emb=fact_embedding,
                                    _episode_id=episode_id, _now=now):
                    existing = tx.run(
                        """
                        MATCH (:Entity {uuid: $src_uuid})-[r:RELATION {graph_id: $gid, name: $name}]->(:Entity {uuid: $tgt_uuid})
                        WHERE toLower(r.fact) = toLower($fact)
                          AND coalesce(r.source, 'document') = $source
                          AND coalesce(r.simulation_id, '') = coalesce($sim, '')
                        WITH r LIMIT 1
                        SET r.episode_ids = CASE WHEN $episode_id IN coalesce(r.episode_ids, [])
                            THEN r.episode_ids ELSE coalesce(r.episode_ids, []) + $episode_id END
                        RETURN r.uuid AS uuid
                        """,
                        src_uuid=_source_uuid,
                        tgt_uuid=_target_uuid,
                        gid=graph_id,
                        name=_rtype,
                        fact=_fact,
                        source=source,
                        sim=simulation_id,
                        episode_id=_episode_id,
                    ).single()
                    if existing:
                        return False
                    tx.run(
                        """
                        MATCH (src:Entity {uuid: $src_uuid})
                        MATCH (tgt:Entity {uuid: $tgt_uuid})
                        CREATE (src)-[r:RELATION {
                            uuid: $uuid,
                            graph_id: $gid,
                            name: $name,
                            fact: $fact,
                            fact_embedding: $fact_embedding,
                            attributes_json: '{}',
                            episode_ids: [$episode_id],
                            source: $source,
                            simulation_id: $sim,
                            created_at: $now,
                            valid_at: null,
                            invalid_at: null,
                            expired_at: null
                        }]->(tgt)
                        """,
                        src_uuid=_source_uuid,
                        tgt_uuid=_target_uuid,
                        uuid=_r_uuid,
                        gid=graph_id,
                        name=_rtype,
                        fact=_fact,
                        fact_embedding=_fact_emb,
                        episode_id=_episode_id,
                        source=source,
                        sim=simulation_id,
                        now=_now,
                    )
                    return True

                if not self._call_with_retry(session.execute_write, _write_relation):
                    merged += 1

            if merged:
                logger.info(f"[add_text] {merged} relations repeated an existing fact; episode added to it")

        logger.info(f"[add_text] Chunk done: episode={episode_id}")
        return episode_id

    # ----------------------------------------------------------------
    # Read nodes
    # ----------------------------------------------------------------

    def get_all_nodes(self, graph_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
        def _read(tx):
            result = tx.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                RETURN n, labels(n) AS labels
                ORDER BY n.created_at DESC
                LIMIT $limit
                """,
                gid=graph_id,
                limit=limit,
            )
            return [self._node_to_dict(record["n"], record["labels"]) for record in result]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_node(self, uuid: str) -> Optional[Dict[str, Any]]:
        def _read(tx):
            result = tx.run(
                "MATCH (n:Entity {uuid: $uuid}) RETURN n, labels(n) AS labels",
                uuid=uuid,
            )
            record = result.single()
            if record:
                return self._node_to_dict(record["n"], record["labels"])
            return None

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """O(1) Cypher — NOT full scan + filter like the old Zep code."""
        def _read(tx):
            result = tx.run(
                """
                MATCH (n:Entity {uuid: $uuid})-[r:RELATION]-(m:Entity)
                RETURN r, startNode(r).uuid AS src_uuid, endNode(r).uuid AS tgt_uuid
                """,
                uuid=node_uuid,
            )
            return [
                self._edge_to_dict(record["r"], record["src_uuid"], record["tgt_uuid"])
                for record in result
            ]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_nodes_by_label(self, graph_id: str, label: str) -> List[Dict[str, Any]]:
        def _read(tx):
            # Dynamic label in query (safe — label comes from ontology, not user input)
            query = f"""
                MATCH (n:Entity:`{label}` {{graph_id: $gid}})
                RETURN n, labels(n) AS labels
            """
            result = tx.run(query, gid=graph_id)
            return [self._node_to_dict(record["n"], record["labels"]) for record in result]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    # ----------------------------------------------------------------
    # Read edges
    # ----------------------------------------------------------------

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        def _read(tx):
            result = tx.run(
                """
                MATCH (src:Entity)-[r:RELATION {graph_id: $gid}]->(tgt:Entity)
                RETURN r, src.uuid AS src_uuid, tgt.uuid AS tgt_uuid
                ORDER BY r.created_at DESC
                """,
                gid=graph_id,
            )
            return [
                self._edge_to_dict(record["r"], record["src_uuid"], record["tgt_uuid"])
                for record in result
            ]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    # ----------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        simulation_id: Optional[str] = None,
    ):
        """
        Hybrid search — returns results matching the scope.

        Returns a dict with 'edges' and/or 'nodes' lists
        (callers like zep_tools will wrap into SearchResult).
        simulation_id limits simulation facts to one run (see in_run_scope).
        """
        result = {"edges": [], "nodes": [], "query": query}

        with self._driver.session() as session:
            if scope in ("edges", "both"):
                result["edges"] = self._search.search_edges(
                    session, graph_id, query, limit, simulation_id
                )

            if scope in ("nodes", "both"):
                result["nodes"] = self._search.search_nodes(
                    session, graph_id, query, limit, simulation_id
                )

        return result

    # ----------------------------------------------------------------
    # Graph info
    # ----------------------------------------------------------------

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        Full graph dump with enriched edge format (for frontend).
        Includes derived fields: fact_type, source_node_name, target_node_name.
        """
        def _read(tx):
            # Get all nodes
            node_result = tx.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                RETURN n, labels(n) AS labels
                """,
                gid=graph_id,
            )
            nodes = []
            node_map: Dict[str, str] = {}  # uuid -> name
            for record in node_result:
                nd = self._node_to_dict(record["n"], record["labels"])
                nodes.append(nd)
                node_map[nd["uuid"]] = nd["name"]

            # Get all edges with source/target node names (JOIN)
            edge_result = tx.run(
                """
                MATCH (src:Entity)-[r:RELATION {graph_id: $gid}]->(tgt:Entity)
                RETURN r, src.uuid AS src_uuid, tgt.uuid AS tgt_uuid,
                       src.name AS src_name, tgt.name AS tgt_name
                """,
                gid=graph_id,
            )
            edges = []
            for record in edge_result:
                ed = self._edge_to_dict(record["r"], record["src_uuid"], record["tgt_uuid"])
                # Enriched fields for frontend
                ed["fact_type"] = ed["name"]
                ed["source_node_name"] = record["src_name"] or ""
                ed["target_node_name"] = record["tgt_name"] or ""
                # Legacy alias
                ed["episodes"] = ed.get("episode_ids", [])
                edges.append(ed)

            return {
                "graph_id": graph_id,
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
            }

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    # ----------------------------------------------------------------
    # Dict conversion helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _node_to_dict(node, labels: List[str]) -> Dict[str, Any]:
        """Convert Neo4j node to the standard node dict format."""
        props = dict(node)
        attrs_json = props.pop("attributes_json", "{}")
        try:
            attributes = json.loads(attrs_json) if attrs_json else {}
        except (json.JSONDecodeError, TypeError):
            attributes = {}

        # Remove internal fields from dict
        props.pop("embedding", None)
        props.pop("name_lower", None)

        return {
            "uuid": props.get("uuid", ""),
            "name": props.get("name", ""),
            "labels": [l for l in labels if l != "Entity"] if labels else [],
            "summary": props.get("summary", ""),
            "attributes": attributes,
            "created_at": props.get("created_at"),
            # None on graphs built before provenance was recorded
            "source": props.get("source"),
            "simulation_ids": props.get("simulation_ids") or [],
        }

    @staticmethod
    def _edge_to_dict(rel, source_uuid: str, target_uuid: str) -> Dict[str, Any]:
        """Convert Neo4j relationship to the standard edge dict format."""
        props = dict(rel)
        attrs_json = props.pop("attributes_json", "{}")
        try:
            attributes = json.loads(attrs_json) if attrs_json else {}
        except (json.JSONDecodeError, TypeError):
            attributes = {}

        # Remove internal fields
        props.pop("fact_embedding", None)

        episode_ids = props.get("episode_ids", [])
        if episode_ids and not isinstance(episode_ids, list):
            episode_ids = [str(episode_ids)]

        return {
            "uuid": props.get("uuid", ""),
            "name": props.get("name", ""),
            "fact": props.get("fact", ""),
            "source_node_uuid": source_uuid,
            "target_node_uuid": target_uuid,
            "attributes": attributes,
            "created_at": props.get("created_at"),
            "valid_at": props.get("valid_at"),
            "invalid_at": props.get("invalid_at"),
            "expired_at": props.get("expired_at"),
            "episode_ids": episode_ids,
            # None on graphs built before provenance was recorded
            "source": props.get("source"),
            "simulation_id": props.get("simulation_id"),
        }
