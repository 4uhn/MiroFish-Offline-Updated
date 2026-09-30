"""
GraphStorage — abstract interface for graph storage backends.

All Zep Cloud calls are replaced by this abstraction.
Current implementation: Neo4jStorage (neo4j_storage.py).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

# Run scope for reads that must see no simulation facts at all, such as
# building agent profiles before this graph's next run.
SEED_ONLY = "__seed_only__"


def in_run_scope(item: Dict[str, Any], simulation_id: Optional[str]) -> bool:
    """True when a node or edge dict belongs in a read scoped to simulation_id.

    Seed facts are always in scope. Simulation facts are in scope only for the
    run that wrote them (edges carry simulation_id, nodes the simulation_ids
    that mentioned them). simulation_id=None means unscoped.
    """
    if simulation_id is None or item.get("source") != "simulation":
        return True
    return (item.get("simulation_id") == simulation_id
            or simulation_id in (item.get("simulation_ids") or []))


class GraphStorage(ABC):
    """Abstract interface for graph storage backends."""

    # --- Graph lifecycle ---

    @abstractmethod
    def create_graph(self, name: str, description: str = "") -> str:
        """Create a new graph. Returns graph_id."""

    @abstractmethod
    def delete_graph(self, graph_id: str) -> None:
        """Delete a graph and all its nodes/edges."""

    @abstractmethod
    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        """Store ontology (entity types + relation types) for a graph."""

    @abstractmethod
    def get_ontology(self, graph_id: str) -> Dict[str, Any]:
        """Retrieve stored ontology for a graph."""

    # --- Add data ---

    @abstractmethod
    def add_text(
        self,
        graph_id: str,
        text: str,
        source: str = "document",
        simulation_id: Optional[str] = None,
        grounding: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Process text: NER/RE → create nodes/edges → return episode_id.
        This is synchronous (unlike Zep Cloud's async episodes).
        source is "document" (seed text) or "simulation" (agent activity).
        simulation_id tags what a run writes. grounding (one item per agent
        activity: actor, own_words, acted_on) limits what NER may record.
        """

    # --- Read nodes ---

    @abstractmethod
    def get_all_nodes(self, graph_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
        """Get all nodes in a graph (with optional limit)."""

    @abstractmethod
    def get_node(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Get a single node by UUID."""

    @abstractmethod
    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """Get all edges connected to a node (O(1) via Cypher, not full scan)."""

    @abstractmethod
    def get_nodes_by_label(self, graph_id: str, label: str) -> List[Dict[str, Any]]:
        """Get nodes filtered by entity type label."""

    # --- Read edges ---

    @abstractmethod
    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """Get all edges in a graph."""

    # --- Search ---

    @abstractmethod
    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        simulation_id: Optional[str] = None,
    ):
        """
        Hybrid search (vector + keyword) over graph data.

        Args:
            graph_id: Graph to search in
            query: Search query text
            limit: Max results
            scope: "edges", "nodes", or "both"
            simulation_id: Limit simulation facts to this run (see in_run_scope);
                SEED_ONLY excludes them all, None applies no run filter

        Returns:
            Dict with 'edges' and/or 'nodes' lists (wrapped by GraphToolsService into SearchResult)
        """

    # --- Graph info ---

    @abstractmethod
    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        Get full graph data (enriched format for frontend).

        Returns dict with:
            graph_id, nodes, edges, node_count, edge_count
        Edge dicts include derived fields: fact_type, source_node_name, target_node_name
        """
