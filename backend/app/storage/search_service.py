"""
SearchService — hybrid search (vector + keyword) over Neo4j graph data.

Replaces Zep Cloud's built-in search with reranker.
Scoring: 0.7 * vector_score + 0.3 * keyword_score (BM25 via fulltext index).
"""

import logging
from typing import List, Dict, Any, Optional

from neo4j import Session as Neo4jSession

from .embedding_service import EmbeddingService

logger = logging.getLogger('mirofish.search')

# Run scope (see graph_storage.in_run_scope): with $sim set, simulation facts
# must come from that run. Graphs built before runs were tagged have no
# simulation_id, so their simulation facts drop out of scoped reads.
_EDGE_IN_SCOPE = """($sim IS NULL OR coalesce(relationship.source, 'document') <> 'simulation'
       OR relationship.simulation_id = $sim)"""
_NODE_IN_SCOPE = """($sim IS NULL OR coalesce(node.source, 'document') <> 'simulation'
       OR $sim IN coalesce(node.simulation_ids, []))"""

# Cypher for vector search on edges (facts). The index spans every graph, so
# the nearest $candidates are filtered to this graph and run before the limit:
# asking for only $limit neighbours left a graph holding a fifth of the
# database with about a fifth of its results.
_VECTOR_SEARCH_EDGES = """
CALL db.index.vector.queryRelationships('fact_embedding', $candidates, $query_vector)
YIELD relationship, score
WHERE relationship.graph_id = $graph_id AND """ + _EDGE_IN_SCOPE + """
RETURN relationship AS r, startNode(relationship).uuid AS src, endNode(relationship).uuid AS tgt, score
ORDER BY score DESC
LIMIT $limit
"""

# Cypher for vector search on nodes (entities)
_VECTOR_SEARCH_NODES = """
CALL db.index.vector.queryNodes('entity_embedding', $candidates, $query_vector)
YIELD node, score
WHERE node.graph_id = $graph_id AND """ + _NODE_IN_SCOPE + """
RETURN node AS n, score
ORDER BY score DESC
LIMIT $limit
"""

# Cypher for fulltext (BM25) search on edges
_FULLTEXT_SEARCH_EDGES = """
CALL db.index.fulltext.queryRelationships('fact_fulltext', $query_text)
YIELD relationship, score
WHERE relationship.graph_id = $graph_id AND """ + _EDGE_IN_SCOPE + """
RETURN relationship AS r, startNode(relationship).uuid AS src, endNode(relationship).uuid AS tgt, score
ORDER BY score DESC
LIMIT $limit
"""

# Cypher for fulltext search on nodes
_FULLTEXT_SEARCH_NODES = """
CALL db.index.fulltext.queryNodes('entity_fulltext', $query_text)
YIELD node, score
WHERE node.graph_id = $graph_id AND """ + _NODE_IN_SCOPE + """
RETURN node AS n, score
ORDER BY score DESC
LIMIT $limit
"""


class SearchService:
    """Hybrid search combining vector similarity and keyword matching."""

    VECTOR_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3
    # Nearest neighbours fetched from the shared vector index before filtering
    VECTOR_CANDIDATES = 500

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding = embedding_service

    def search_edges(
        self,
        session: Neo4jSession,
        graph_id: str,
        query: str,
        limit: int = 10,
        simulation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search edges (facts/relations) using hybrid scoring.

        Returns list of dicts with edge properties + 'score'.
        """
        query_vector = self.embedding.embed(query)

        # Vector search
        vector_results = self._run_edge_vector_search(
            session, graph_id, query_vector, limit * 2, simulation_id
        )

        # Keyword search
        keyword_results = self._run_edge_keyword_search(
            session, graph_id, query, limit * 2, simulation_id
        )

        # Merge and rank
        merged = self._merge_results(
            vector_results, keyword_results, key="uuid", limit=limit
        )
        return merged

    def search_nodes(
        self,
        session: Neo4jSession,
        graph_id: str,
        query: str,
        limit: int = 10,
        simulation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search nodes (entities) using hybrid scoring.

        Returns list of dicts with node properties + 'score'.
        """
        query_vector = self.embedding.embed(query)

        vector_results = self._run_node_vector_search(
            session, graph_id, query_vector, limit * 2, simulation_id
        )

        keyword_results = self._run_node_keyword_search(
            session, graph_id, query, limit * 2, simulation_id
        )

        merged = self._merge_results(
            vector_results, keyword_results, key="uuid", limit=limit
        )
        return merged

    @staticmethod
    def _edge_row(record) -> Dict[str, Any]:
        """Edge properties plus its endpoints.

        Endpoint UUIDs are not relationship properties, so returning only the
        relationship left every search hit without a source or target, and
        InsightForge printed its relationship chains with blank names.
        """
        return {
            **dict(record["r"]),
            "uuid": record["r"]["uuid"],
            "source_node_uuid": record["src"],
            "target_node_uuid": record["tgt"],
            "_score": record["score"],
        }

    def _run_edge_vector_search(
        self, session: Neo4jSession, graph_id: str, query_vector: List[float], limit: int,
        simulation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run vector similarity search on edge fact_embedding."""
        try:
            result = session.run(
                _VECTOR_SEARCH_EDGES,
                graph_id=graph_id,
                query_vector=query_vector,
                limit=limit,
                candidates=max(limit, self.VECTOR_CANDIDATES),
                sim=simulation_id,
            )
            return [
                self._edge_row(record)
                for record in result
            ]
        except Exception as e:
            logger.warning(f"Vector edge search failed (index may not exist yet): {e}")
            return []

    def _run_edge_keyword_search(
        self, session: Neo4jSession, graph_id: str, query: str, limit: int,
        simulation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run fulltext (BM25) search on edge fact + name."""
        try:
            # Escape special Lucene characters in query
            safe_query = self._escape_lucene(query)
            result = session.run(
                _FULLTEXT_SEARCH_EDGES,
                graph_id=graph_id,
                query_text=safe_query,
                limit=limit,
                sim=simulation_id,
            )
            return [
                self._edge_row(record)
                for record in result
            ]
        except Exception as e:
            logger.warning(f"Keyword edge search failed: {e}")
            return []

    def _run_node_vector_search(
        self, session: Neo4jSession, graph_id: str, query_vector: List[float], limit: int,
        simulation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run vector similarity search on entity embedding."""
        try:
            result = session.run(
                _VECTOR_SEARCH_NODES,
                graph_id=graph_id,
                query_vector=query_vector,
                limit=limit,
                candidates=max(limit, self.VECTOR_CANDIDATES),
                sim=simulation_id,
            )
            return [
                {**dict(record["n"]), "uuid": record["n"]["uuid"], "_score": record["score"]}
                for record in result
            ]
        except Exception as e:
            logger.warning(f"Vector node search failed: {e}")
            return []

    def _run_node_keyword_search(
        self, session: Neo4jSession, graph_id: str, query: str, limit: int,
        simulation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run fulltext search on entity name + summary."""
        try:
            safe_query = self._escape_lucene(query)
            result = session.run(
                _FULLTEXT_SEARCH_NODES,
                graph_id=graph_id,
                query_text=safe_query,
                limit=limit,
                sim=simulation_id,
            )
            return [
                {**dict(record["n"]), "uuid": record["n"]["uuid"], "_score": record["score"]}
                for record in result
            ]
        except Exception as e:
            logger.warning(f"Keyword node search failed: {e}")
            return []

    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        key: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Merge vector and keyword results with weighted scoring.

        Normalizes scores to [0, 1] range before combining.
        """
        # Normalize vector scores
        v_max = max((r["_score"] for r in vector_results), default=1.0) or 1.0
        v_scores = {r[key]: r["_score"] / v_max for r in vector_results}

        # Normalize keyword scores
        k_max = max((r["_score"] for r in keyword_results), default=1.0) or 1.0
        k_scores = {r[key]: r["_score"] / k_max for r in keyword_results}

        # Build combined result map
        all_items: Dict[str, Dict[str, Any]] = {}
        for r in vector_results:
            all_items[r[key]] = {k: v for k, v in r.items() if k != "_score"}
        for r in keyword_results:
            if r[key] not in all_items:
                all_items[r[key]] = {k: v for k, v in r.items() if k != "_score"}

        # Calculate hybrid scores
        scored = []
        for uid, item in all_items.items():
            v = v_scores.get(uid, 0.0)
            k = k_scores.get(uid, 0.0)
            combined = self.VECTOR_WEIGHT * v + self.KEYWORD_WEIGHT * k
            item["score"] = combined
            scored.append(item)

        # Sort by combined score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    @staticmethod
    def _escape_lucene(query: str) -> str:
        """Escape special Lucene query characters."""
        special = r'+-&|!(){}[]^"~*?:\/'
        result = []
        for ch in query:
            if ch in special:
                result.append('\\')
            result.append(ch)
        return ''.join(result)
