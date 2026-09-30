"""
Graph Retrieval Tools Service
Encapsulates graph search, node reading, edge querying, and other tools for use by the Report Agent.

Replaces zep_tools.py — all Zep Cloud calls replaced by GraphStorage.

Core Retrieval Tools (Optimized):
1. InsightForge (Deep Insight Retrieval) - Most powerful hybrid retrieval, auto-generates sub-queries for multi-dimensional search
2. PanoramaSearch (Broad Search) - Get the full picture, including expired content
3. QuickSearch (Simple Search) - Fast retrieval
"""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..storage import GraphStorage
from ..storage.graph_storage import in_run_scope

logger = get_logger('mirofish.graph_tools')

# Graph content extracted from agents' posts during the run rather than from
# the seed document (storage records source="simulation"). Run 10's agents
# invented a "River Wye" and a "Civic Health Centre"; NER stored them next to
# the seed facts and the report could not tell them apart.
AGENT_CLAIM_TAG = "[agent claim]"


def with_origin(text: str, source: Optional[str]) -> str:
    """Prefix text that came from simulated agents' posts with AGENT_CLAIM_TAG."""
    if text and source == "simulation":
        return f"{AGENT_CLAIM_TAG} {text}"
    return text


@dataclass
class SearchResult:
    """Search result"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }

    def to_text(self) -> str:
        """Convert to text format for LLM comprehension"""
        text_parts = [f"Search query: {self.query}", f"Found {self.total_count} relevant items"]

        if self.facts:
            text_parts.append("\n### Relevant Facts (machine-extracted summaries of simulation activity: paraphrase them, never put them in quotation marks):")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Node information"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    source: Optional[str] = None  # "document" or "simulation"; None on older graphs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "source": self.source,
        }

    def to_text(self) -> str:
        """Convert to text format"""
        entity_type = next((la for la in self.labels if la not in ["Entity", "Node"]), "Unknown type")
        return with_origin(f"Entity: {self.name} (Type: {entity_type})\nSummary: {self.summary}", self.source)


@dataclass
class EdgeInfo:
    """Edge information"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Temporal info (may be absent in Neo4j — kept for interface compat)
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    source: Optional[str] = None  # "document" or "simulation"; None on older graphs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source": self.source,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }

    def to_text(self, include_temporal: bool = False) -> str:
        """Convert to text format"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Relationship: {source} --[{self.name}]--> {target}\nFact: {with_origin(self.fact, self.source)}"

        if include_temporal:
            valid_at = self.valid_at or "Unknown"
            invalid_at = self.invalid_at or "Present"
            base_text += f"\nValidity: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (Expired: {self.expired_at})"

        return base_text

    @property
    def is_expired(self) -> bool:
        """Whether this edge has expired"""
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        """Whether this edge has been invalidated"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    Deep Insight Retrieval Result (InsightForge)
    Contains retrieval results from multiple sub-queries, along with comprehensive analysis
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]

    # Retrieval results by dimension
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)

    # Statistics
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }

    def to_text(self) -> str:
        """Convert to detailed text format for LLM comprehension"""
        text_parts = [
            f"## Future Prediction Deep Analysis",
            f"Analysis query: {self.query}",
            f"Prediction scenario: {self.simulation_requirement}",
            f"\n### Prediction Data Statistics",
            f"- Relevant prediction facts: {self.total_facts}",
            f"- Entities involved: {self.total_entities}",
            f"- Relationship chains: {self.total_relationships}"
        ]

        if self.sub_queries:
            text_parts.append(f"\n### Analyzed Sub-queries")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")

        if self.semantic_facts:
            text_parts.append(f"\n### [Key Facts] (machine-extracted summaries of simulation activity: paraphrase them, never put them in quotation marks)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f'{i}. {fact}')

        if self.entity_insights:
            text_parts.append(f"\n### [Core Entities]")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Entity')})")
                if entity.get('summary'):
                    text_parts.append(f"  Summary: {entity.get('summary')}")
                if entity.get('related_facts'):
                    text_parts.append(f"  Related facts: {len(entity.get('related_facts', []))}")

        if self.relationship_chains:
            text_parts.append(f"\n### [Relationship Chains]")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")

        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Broad Search Result (Panorama)
    Contains all relevant information, including expired content
    """
    query: str

    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)

    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }

    def to_text(self) -> str:
        """Convert to text format (full version, no truncation)"""
        text_parts = [
            f"## Broad Search Results (Future Panoramic View)",
            f"Query: {self.query}",
            f"\n### Statistics",
            f"- Total nodes: {self.total_nodes}",
            f"- Total edges: {self.total_edges}",
            f"- Currently active facts: {self.active_count}",
            f"- Historical/expired facts: {self.historical_count}"
        ]

        if self.active_facts:
            text_parts.append(f"\n### [Currently Active Facts] (machine-extracted summaries of simulation activity: paraphrase them, never put them in quotation marks)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. {fact}')

        if self.historical_facts:
            text_parts.append(f"\n### [Historical/Expired Facts] (earlier extracted summaries, superseded; paraphrase only)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. {fact}')

        if self.all_nodes:
            text_parts.append(f"\n### [Entities Involved]")
            for node in self.all_nodes:
                entity_type = next((la for la in node.labels if la not in ["Entity", "Node"]), "Entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")

        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Single agent interview result"""
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        text += f"_Bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Key Quotes:**\n"
            for quote in self.key_quotes:
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    Interview Result
    Contains interview responses from multiple simulated agents
    """
    interview_topic: str
    interview_questions: List[str]

    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)

    selection_reasoning: str = ""
    summary: str = ""

    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }

    def to_text(self) -> str:
        """Convert to detailed text format for LLM comprehension and report citation"""
        text_parts = [
            "## In-Depth Interview Report",
            f"**Interview Topic:** {self.interview_topic}",
            f"**Interviews Conducted:** {self.interviewed_count} / {self.total_agents} simulated agents",
            "These are answers agents gave when interviewed after the simulation, not posts they made "
            "during it ([Twitter] and [Reddit] mark which platform's copy of the agent answered). An "
            "answer is the agent's own account: report what it says the agent did or felt as its claim "
            "(\"told interviewers it had...\"), not as an observed event, and credit quotes as "
            "(Name, role, interview). Five answers are five views, not a measure of how widespread a view was.",
            "\n### Rationale for Interviewee Selection",
            self.selection_reasoning or "(Auto-selected)",
            "\n---",
            "\n### Interview Transcripts",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(No interview records)\n\n---")

        # No LLM-written summary of the answers: in Run 13 the section writer
        # paraphrased it in place of the transcripts, keeping its anonymous
        # credits ("a council representative stated") and generalisations
        # ("broad agreement") and losing the interview framing.
        if self.summary:
            text_parts.append(f"\n{self.summary}")

        return "\n".join(text_parts)


class GraphToolsService:
    """
    Graph Retrieval Tools Service (via GraphStorage / Neo4j)

    [Core Retrieval Tools - Optimized]
    1. insight_forge - Deep insight retrieval (most powerful, auto-generates sub-queries, multi-dimensional search)
    2. panorama_search - Broad search (get the full picture, including expired content)
    3. quick_search - Simple search (fast retrieval)
    4. interview_agents - In-depth interviews (interview simulated agents, obtain multi-perspective views)

    [Basic Tools]
    - search_graph - Graph semantic search
    - get_all_nodes - Get all nodes in the graph
    - get_all_edges - Get all edges in the graph (with temporal info)
    - get_node_detail - Get detailed node information
    - get_node_edges - Get edges related to a node
    - get_entities_by_type - Get entities by type
    - get_entity_summary - Get relationship summary for an entity
    """

    def __init__(
        self,
        storage: GraphStorage,
        llm_client: Optional[LLMClient] = None,
        simulation_id: Optional[str] = None,
    ):
        self.storage = storage
        self._llm_client = llm_client
        # Run scope for every graph read: simulation facts from other runs of
        # this graph stay out of the report (see in_run_scope). None = unscoped.
        self.simulation_id = simulation_id
        # simulation_id -> {agent_id: times interviewed}, so later report
        # sections hear from agents earlier sections did not
        self._interview_counts: Dict[str, Dict[int, int]] = {}
        logger.info("GraphToolsService initialized")

    @property
    def llm(self) -> LLMClient:
        """Lazy-initialize the LLM client"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def _scoped(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Nodes or edges from a storage read, limited to this service's run scope."""
        return [item for item in items if in_run_scope(item, self.simulation_id)]

    # ========== Basic Tools ==========

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Graph semantic search (hybrid: vector + BM25 via Neo4j)

        Args:
            graph_id: Graph ID
            query: Search query
            limit: Number of results to return
            scope: Search scope, "edges" or "nodes" or "both"

        Returns:
            SearchResult
        """
        if not query or not query.strip():
            logger.error("Graph search called with empty query — this should be caught upstream. Falling back to local search.")
            return self._local_search(graph_id, "overview", limit, scope)

        logger.info(f"Graph search: graph_id={graph_id}, query={query[:50]}...")

        try:
            search_results = self.storage.search(
                graph_id=graph_id,
                query=query,
                limit=limit,
                scope=scope,
                simulation_id=self.simulation_id,
            )

            facts = []
            edges = []
            nodes = []

            # Parse edge results
            if hasattr(search_results, 'edges'):
                edge_list = search_results.edges
            elif isinstance(search_results, dict) and 'edges' in search_results:
                edge_list = search_results['edges']
            else:
                edge_list = []

            for edge in edge_list:
                if isinstance(edge, dict):
                    fact = edge.get('fact', '')
                    if fact:
                        facts.append(with_origin(fact, edge.get('source')))
                    edges.append({
                        "uuid": edge.get('uuid', ''),
                        "name": edge.get('name', ''),
                        "fact": fact,
                        "source_node_uuid": edge.get('source_node_uuid', ''),
                        "target_node_uuid": edge.get('target_node_uuid', ''),
                    })

            # Parse node results
            if hasattr(search_results, 'nodes'):
                node_list = search_results.nodes
            elif isinstance(search_results, dict) and 'nodes' in search_results:
                node_list = search_results['nodes']
            else:
                node_list = []

            for node in node_list:
                if isinstance(node, dict):
                    nodes.append({
                        "uuid": node.get('uuid', ''),
                        "name": node.get('name', ''),
                        "labels": node.get('labels', []),
                        "summary": node.get('summary', ''),
                    })
                    summary = node.get('summary', '')
                    if summary:
                        facts.append(with_origin(f"[{node.get('name', '')}]: {summary}", node.get('source')))

            logger.info(f"Search complete: found {len(facts)} relevant facts")

            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )

        except Exception as e:
            logger.warning(f"Graph search failed, falling back to local search: {str(e)}")
            return self._local_search(graph_id, query, limit, scope)

    def _local_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Local keyword matching search (fallback)
        """
        logger.info(f"Using local search: query={query[:30]}...")

        facts = []
        edges_result = []
        nodes_result = []

        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        def match_score(text: str) -> int:
            if not text:
                return 0
            text_lower = text.lower()
            if query_lower in text_lower:
                return 100
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score

        try:
            if scope in ["edges", "both"]:
                all_edges = self._scoped(self.storage.get_all_edges(graph_id))
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.get("fact", "")) + match_score(edge.get("name", ""))
                    if score > 0:
                        scored_edges.append((score, edge))

                scored_edges.sort(key=lambda x: x[0], reverse=True)

                for score, edge in scored_edges[:limit]:
                    fact = edge.get("fact", "")
                    if fact:
                        facts.append(with_origin(fact, edge.get("source")))
                    edges_result.append({
                        "uuid": edge.get("uuid", ""),
                        "name": edge.get("name", ""),
                        "fact": fact,
                        "source_node_uuid": edge.get("source_node_uuid", ""),
                        "target_node_uuid": edge.get("target_node_uuid", ""),
                    })

            if scope in ["nodes", "both"]:
                all_nodes = self._scoped(self.storage.get_all_nodes(graph_id))
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.get("name", "")) + match_score(node.get("summary", ""))
                    if score > 0:
                        scored_nodes.append((score, node))

                scored_nodes.sort(key=lambda x: x[0], reverse=True)

                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.get("uuid", ""),
                        "name": node.get("name", ""),
                        "labels": node.get("labels", []),
                        "summary": node.get("summary", ""),
                    })
                    summary = node.get("summary", "")
                    if summary:
                        facts.append(with_origin(f"[{node.get('name', '')}]: {summary}", node.get('source')))

            logger.info(f"Local search complete: found {len(facts)} relevant facts")

        except Exception as e:
            logger.error(f"Local search failed: {str(e)}")

        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """Get all nodes in the graph"""
        logger.info(f"Getting all nodes for graph {graph_id}...")

        raw_nodes = self._scoped(self.storage.get_all_nodes(graph_id))

        result = []
        for node in raw_nodes:
            result.append(NodeInfo(
                uuid=node.get("uuid", ""),
                name=node.get("name", ""),
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
                source=node.get("source"),
            ))

        logger.info(f"Retrieved {len(result)} nodes")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """Get all edges in the graph (with temporal info)"""
        logger.info(f"Getting all edges for graph {graph_id}...")

        raw_edges = self._scoped(self.storage.get_all_edges(graph_id))

        result = []
        for edge in raw_edges:
            edge_info = EdgeInfo(
                uuid=edge.get("uuid", ""),
                name=edge.get("name", ""),
                fact=edge.get("fact", ""),
                source_node_uuid=edge.get("source_node_uuid", ""),
                target_node_uuid=edge.get("target_node_uuid", ""),
                source=edge.get("source"),
            )

            if include_temporal:
                edge_info.created_at = edge.get("created_at")
                edge_info.valid_at = edge.get("valid_at")
                edge_info.invalid_at = edge.get("invalid_at")
                edge_info.expired_at = edge.get("expired_at")

            result.append(edge_info)

        logger.info(f"Retrieved {len(result)} edges")
        return result

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """Get detailed information for a single node"""
        logger.info(f"Getting node detail: {node_uuid[:8]}...")

        try:
            node = self.storage.get_node(node_uuid)
            if not node:
                return None

            return NodeInfo(
                uuid=node.get("uuid", ""),
                name=node.get("name", ""),
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
                source=node.get("source"),
            )
        except Exception as e:
            logger.error(f"Failed to get node detail: {str(e)}")
            return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Get all edges related to a node

        Optimized: uses storage.get_node_edges() (O(degree) Cypher)
        instead of loading ALL edges and filtering.
        """
        logger.info(f"Getting edges for node {node_uuid[:8]}...")

        try:
            raw_edges = self._scoped(self.storage.get_node_edges(node_uuid))

            result = []
            for edge in raw_edges:
                result.append(EdgeInfo(
                    uuid=edge.get("uuid", ""),
                    name=edge.get("name", ""),
                    fact=edge.get("fact", ""),
                    source_node_uuid=edge.get("source_node_uuid", ""),
                    target_node_uuid=edge.get("target_node_uuid", ""),
                    created_at=edge.get("created_at"),
                    valid_at=edge.get("valid_at"),
                    invalid_at=edge.get("invalid_at"),
                    expired_at=edge.get("expired_at"),
                    source=edge.get("source"),
                ))

            logger.info(f"Found {len(result)} edges related to node")
            return result

        except Exception as e:
            logger.warning(f"Failed to get node edges: {str(e)}")
            return []

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str
    ) -> List[NodeInfo]:
        """Get entities by type"""
        logger.info(f"Getting entities of type {entity_type}...")

        # Use optimized label-based query from storage
        raw_nodes = self._scoped(self.storage.get_nodes_by_label(graph_id, entity_type))

        result = []
        for node in raw_nodes:
            result.append(NodeInfo(
                uuid=node.get("uuid", ""),
                name=node.get("name", ""),
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
                source=node.get("source"),
            ))

        logger.info(f"Found {len(result)} entities of type {entity_type}")
        return result

    def get_entity_summary(
        self,
        graph_id: str,
        entity_name: str
    ) -> Dict[str, Any]:
        """Get relationship summary for a specified entity"""
        logger.info(f"Getting relationship summary for entity {entity_name}...")

        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )

        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break

        related_edges = []
        if entity_node:
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)

        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """Get graph statistics"""
        logger.info(f"Getting statistics for graph {graph_id}...")

        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1

        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1

        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }

    def get_simulation_context(
        self,
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Get simulation-related context information"""
        logger.info(f"Getting simulation context: {simulation_requirement[:50]}...")

        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )

        stats = self.get_graph_statistics(graph_id)

        all_nodes = self.get_all_nodes(graph_id)

        entities = []
        for node in all_nodes:
            custom_labels = [la for la in node.labels if la not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities)
        }

    # ========== Core Retrieval Tools (Optimized) ==========

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        [InsightForge - Deep Insight Retrieval]

        The most powerful hybrid retrieval function, automatically decomposing queries for multi-dimensional search:
        1. Use LLM to decompose the query into multiple sub-queries
        2. Perform semantic search for each sub-query
        3. Extract related entities and obtain their detailed information
        4. Trace relationship chains
        5. Integrate all results to generate deep insights
        """
        logger.info(f"InsightForge deep insight retrieval: {query[:50]}...")

        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )

        # Step 1: Use LLM to generate sub-queries
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(f"Generated {len(sub_queries)} sub-queries")

        # Step 2: Perform semantic search for each sub-query
        all_facts = []
        all_edges = []
        seen_facts = set()

        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )

            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)

            all_edges.extend(search_result.edges)

        # Also search for the original query
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        # Step 3: Extract related entity UUIDs from edges
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)

        # Get related entity details
        entity_insights = []
        node_map = {}

        for uuid in list(entity_uuids):
            if not uuid:
                continue
            try:
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((la for la in node.labels if la not in ["Entity", "Node"]), "Entity")

                    related_facts = [
                        f for f in all_facts
                        if node.name.lower() in f.lower()
                    ]

                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts
                    })
            except Exception as e:
                logger.debug(f"Failed to get node {uuid}: {e}")
                continue

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        # Step 4: Build relationship chains
        relationship_chains = []
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')

                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]

                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)

        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)

        logger.info(f"InsightForge complete: {result.total_facts} facts, {result.total_entities} entities, {result.total_relationships} relationships")
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """Use LLM to generate sub-queries"""
        system_prompt = """You are a professional question analysis expert. Your task is to decompose a complex question into multiple sub-questions that can be independently observed in a simulated world. Respond in English only.

Requirements:
1. Each sub-question should be specific enough to find related agent behaviors or events in the simulated world
2. Sub-questions should cover different dimensions of the original question (e.g., who, what, why, how, when, where)
3. Sub-questions should be relevant to the simulation scenario
4. Return in JSON format: {"sub_queries": ["sub-question 1", "sub-question 2", ...]}"""

        user_prompt = f"""Simulation background:
{simulation_requirement}

{f"Report context: {report_context[:500]}" if report_context else ""}

Please decompose the following question into {max_queries} sub-questions:
{query}

Return the sub-question list in JSON format."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            sub_queries = response.get("sub_queries", [])
            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning(f"Failed to generate sub-queries: {str(e)}, using defaults")
            return [
                query,
                f"Key participants in {query}",
                f"Causes and impacts of {query}",
                f"Development process of {query}"
            ][:max_queries]

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        [PanoramaSearch - Broad Search]

        Get a full-picture view, including all relevant content and historical/expired information.
        """
        logger.info(f"PanoramaSearch broad search: {query[:50]}...")

        result = PanoramaResult(query=query)

        # Get all nodes
        all_nodes = self.get_all_nodes(graph_id)
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        # Get all edges (with temporal info)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        # Classify facts
        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue

            is_historical = edge.is_expired or edge.is_invalid

            if is_historical:
                valid_at = edge.valid_at or "Unknown"
                invalid_at = edge.invalid_at or edge.expired_at or "Unknown"
                fact_with_time = f"[{valid_at} - {invalid_at}] {with_origin(edge.fact, edge.source)}"
                historical_facts.append(fact_with_time)
            else:
                active_facts.append(with_origin(edge.fact, edge.source))

        # Sort by relevance to the query
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score

        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(f"PanoramaSearch complete: {result.active_count} active, {result.historical_count} historical")
        return result

    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        [QuickSearch - Simple Search]
        Fast, lightweight retrieval tool.
        """
        logger.info(f"QuickSearch simple search: {query[:50]}...")

        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )

        logger.info(f"QuickSearch complete: {result.total_count} results")
        return result

    _TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
    _TEXT_ARG_KEYS = ('content', 'text', 'body', 'message', 'reply', 'quote_content')

    @classmethod
    def _clean_tool_call_response(cls, response: str) -> str:
        """Recover the agent's words from a reply written as a tool call.

        Handles qwen3's <tool_call>{"name": ..., "arguments": {...}}</tool_call>
        text and bare {"tool_name"/"name": ..., "arguments": {...}} JSON. Returns
        the text argument if there is one, "" for a textless call such as
        join_group, and the input unchanged when it is not a tool call.
        """
        if not response:
            return ""
        text = response.strip()
        match = cls._TOOL_CALL_RE.search(text)
        if match:
            payload = match.group(1)
        elif text.startswith('{') and ('"arguments"' in text[:200]):
            payload = text
        else:
            return response

        try:
            data = json.loads(payload)
            args = data.get('arguments', {}) if isinstance(data, dict) else {}
            if isinstance(args, str):
                args = json.loads(args)
            for key in cls._TEXT_ARG_KEYS:
                if isinstance(args, dict) and args.get(key):
                    return str(args[key]).strip()
            return ""
        except (json.JSONDecodeError, TypeError, AttributeError):
            m = re.search(r'"(?:content|message|text)"\s*:\s*"((?:[^"\\]|\\.)*)"', payload)
            if m:
                return m.group(1).replace('\\n', '\n').replace('\\"', '"').strip()
            return ""

    @classmethod
    def _extract_interview_text(cls, raw: Any) -> str:
        """Turn an IPC interview result into the agent's answer text.

        With no platform given, the result is {"platforms": {"twitter": {...},
        "reddit": {...}}}; Run 8 passed str() of that whole dict to the summary.
        Each platform's answer is cleaned, and the answers are joined with a
        platform label.
        """
        if isinstance(raw, dict) and isinstance(raw.get('platforms'), dict):
            parts = []
            for platform, res in raw['platforms'].items():
                answer = cls._clean_tool_call_response((res or {}).get('response') or '')
                if answer.strip():
                    parts.append(f"[{platform.capitalize()}] {answer.strip()}")
            return "\n\n".join(parts)
        if isinstance(raw, dict):
            return cls._clean_tool_call_response(raw.get('response') or '').strip()
        return cls._clean_tool_call_response(str(raw or '')).strip()

    @staticmethod
    def _normalise_for_quote_match(text: str) -> str:
        text = text.replace('\u2019', "'").replace('\u2018', "'")
        text = re.sub(r'[^a-z0-9\' ]+', ' ', text.lower())
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _key_quotes(answer: str, limit: int = 2) -> List[str]:
        """Verbatim sentences from an answer, for the report to cite."""
        body = re.sub(r'^\[(Twitter|Reddit)\]\s*', '', answer, flags=re.MULTILINE)
        sentences = re.split(r'(?<=[.!?])\s+', body)
        picked = [s.strip() for s in sentences if len(s.split()) >= 6 and '?' not in s]
        return picked[:limit]

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """Call OASIS interview API to get first-person agent responses."""
        from ..services.simulation_runner import SimulationRunner

        profiles = self._load_agent_profiles(simulation_id)
        if not profiles:
            return InterviewResult(
                interview_topic=interview_requirement,
                interview_questions=[],
                summary="No agent profiles found for this simulation.",
                total_agents=0,
                interviewed_count=0
            )

        selected_agents, selected_indices, reasoning = self._select_agents_for_interview(
            simulation_id, profiles, interview_requirement, simulation_requirement, max_agents
        )
        counts = self._interview_counts.setdefault(simulation_id, {})
        for agent_profile, agent_idx in zip(selected_agents, selected_indices):
            aid = int(agent_profile.get("user_id", agent_idx))
            counts[aid] = counts.get(aid, 0) + 1

        questions = custom_questions or self._generate_interview_questions(
            interview_requirement, simulation_requirement
        )

        interviews = []
        env_alive = SimulationRunner.check_env_alive(simulation_id)

        for agent_profile, agent_idx in zip(selected_agents, selected_indices):
            agent_name = agent_profile.get("realname", agent_profile.get("name", agent_profile.get("username", f"Agent_{agent_idx}")))
            agent_role = agent_profile.get("profession", "Participant")
            agent_bio = agent_profile.get("bio", "")[:150]
            agent_id = agent_profile.get("user_id", agent_idx)

            combined_question = " ".join(questions)

            response_text = ""
            if env_alive:
                try:
                    result = SimulationRunner.interview_agent(
                        simulation_id=simulation_id,
                        agent_id=agent_id,
                        prompt=combined_question,
                        timeout=120.0
                    )
                    if result.get("success"):
                        response_text = self._extract_interview_text(result.get("result", ""))
                        if not response_text:
                            logger.warning(f"Interview with {agent_name} returned no answer text")
                except Exception as e:
                    logger.warning(f"Live interview failed for {agent_name}: {e}")

            if not response_text:
                if env_alive:
                    response_text = f"[{agent_name} gave no usable answer to the interview.]"
                else:
                    response_text = f"[Simulation environment not available — {agent_name} could not be interviewed live.]"
                key_quotes = []
            else:
                key_quotes = self._key_quotes(response_text)

            interview = AgentInterview(
                agent_name=agent_name,
                agent_role=agent_role,
                agent_bio=agent_bio,
                question=combined_question,
                response=response_text,
                key_quotes=key_quotes,
            )
            interviews.append(interview)

        return InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=questions,
            selected_agents=[{"name": a.get("realname", a.get("name", "?")), "role": a.get("profession", "?")} for a in selected_agents],
            interviews=interviews,
            selection_reasoning=reasoning,
            total_agents=len(profiles),
            interviewed_count=len(interviews)
        )

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Load agent profile files for a simulation"""
        import os
        import csv

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )

        profiles = []

        # Try reading Reddit JSON format first
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"Loaded {len(profiles)} profiles from reddit_profiles.json")
                return profiles
            except Exception as e:
                logger.warning(f"Failed to read reddit_profiles.json: {e}")

        # Try reading Twitter CSV format
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "Unknown"
                        })
                logger.info(f"Loaded {len(profiles)} profiles from twitter_profiles.csv")
                return profiles
            except Exception as e:
                logger.warning(f"Failed to read twitter_profiles.csv: {e}")

        return profiles

    _MAX_INSTITUTIONS_PER_INTERVIEW = 2
    _MAX_SAME_TYPE_PER_INTERVIEW = 2

    def _select_agents_for_interview(
        self,
        simulation_id: str,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """Pick interviewees: the LLM ranks by relevance, code enforces the mix.

        In Run 9 the LLM picked the same four institutions plus one resident
        for every section. Now each agent is shown as an institution or a
        member of the public, with how much it wrote and how often it has
        been interviewed; code then caps institutions and same-type agents
        and prefers agents not yet interviewed.
        """
        from .agent_posts import AgentPostIndex, active_agent_ids, public_agent_ids

        public_ids = public_agent_ids(simulation_id)
        post_counts = AgentPostIndex.for_simulation(simulation_id).counts_by_agent()
        interviewed = self._interview_counts.get(simulation_id, {})
        types = self._agent_entity_types(simulation_id)

        def agent_id(i: int) -> int:
            return int(profiles[i].get("user_id", i))

        # Only agents that did something: asked about their actions, an agent
        # with none on record makes them up, whatever the prompt says.
        active = active_agent_ids(simulation_id)
        if active is None:
            logger.warning(f"Interview selection: no platform DB for {simulation_id}; not filtering by activity")
            candidates = list(range(len(profiles)))
        else:
            candidates = [i for i in range(len(profiles)) if agent_id(i) in active]
            if len(candidates) < len(profiles):
                logger.info(f"Interview selection: {len(profiles) - len(candidates)} of {len(profiles)} "
                            f"agents took no action and are not interviewed")
            if len(candidates) < max_agents:
                logger.info(f"Interview selection: only {len(candidates)} active agents for {max_agents} slots")

        agent_summaries = []
        for i in candidates:
            profile = profiles[i]
            aid = agent_id(i)
            agent_summaries.append({
                "index": i,
                "name": profile.get("name") or profile.get("realname") or profile.get("username", f"Agent_{i}"),
                "kind": "member of the public" if aid in public_ids else "institution/official",
                "profession": profile.get("profession", "Unknown"),
                "bio": profile.get("bio", "")[:160],
                "posts_written": post_counts.get(aid, 0),
                "times_interviewed": interviewed.get(aid, 0),
            })

        system_prompt = """You are planning interviews for a report on a simulated public-opinion event. Rank the agents who would give the most useful, varied first-person answers on the interview topic. Respond in English only.

Ranking criteria:
1. Relevance of the agent's position to the interview topic
2. A mix of members of the public and institutions/officials, and of opposing views
3. Agents who wrote more during the simulation have more to draw on
4. Prefer agents with times_interviewed = 0 when they are comparably relevant

Return JSON:
{
    "ranked_indices": [agent indices, most suitable first, at least """ + str(max_agents * 2) + """ if available],
    "reasoning": "One or two sentences on the mix you chose"
}"""

        user_prompt = f"""Interview topic:
{interview_requirement}

Simulation background:
{simulation_requirement if simulation_requirement else "Not provided"}

Agents ({len(agent_summaries)} total):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}"""

        ranked: List[int] = []
        reasoning = ""
        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            raw = response.get("ranked_indices") or response.get("selected_indices") or []
            allowed = set(candidates)
            ranked = [int(i) for i in raw if isinstance(i, (int, str)) and str(i).isdigit() and int(i) in allowed]
            reasoning = response.get("reasoning", "")
        except Exception as e:
            logger.warning(f"LLM interview ranking failed, ranking by activity instead: {e}")

        # Agents the LLM did not rank follow, most active first
        rest = sorted(
            (i for i in candidates if i not in ranked),
            key=lambda i: -post_counts.get(agent_id(i), 0),
        )
        order = list(dict.fromkeys(ranked)) + rest
        # Stable sort: fewer interviews first, LLM rank within ties
        order.sort(key=lambda i: interviewed.get(agent_id(i), 0))

        chosen: List[int] = []
        institutions = 0
        per_type: Dict[str, int] = {}
        for i in order:
            aid = agent_id(i)
            is_public = aid in public_ids
            etype = types.get(aid, "")
            if not is_public and institutions >= self._MAX_INSTITUTIONS_PER_INTERVIEW:
                continue
            if is_public and per_type.get(etype, 0) >= self._MAX_SAME_TYPE_PER_INTERVIEW:
                continue
            chosen.append(i)
            institutions += 0 if is_public else 1
            per_type[etype] = per_type.get(etype, 0) + 1
            if len(chosen) >= max_agents:
                break

        logger.info(
            f"Interview selection: {[profiles[i].get('name', i) for i in chosen]} "
            f"({institutions} institution(s); LLM ranked {len(ranked)})"
        )
        return [profiles[i] for i in chosen], chosen, reasoning or "Ranked by relevance, mixed by agent type"

    @staticmethod
    def _agent_entity_types(simulation_id: str) -> Dict[int, str]:
        import os
        path = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id, "simulation_config.json")
        try:
            with open(path, encoding="utf-8") as f:
                return {
                    int(a["agent_id"]): str(a.get("entity_type", ""))
                    for a in json.load(f).get("agent_configs", [])
                }
        except (OSError, ValueError, KeyError) as e:
            logger.warning(f"Could not read agent types for {simulation_id}: {e}")
            return {}

    _ROLE_ADDRESS = re.compile(r'^\s*(as|for) (a|an|the) [^,?]{1,60},\s*', re.IGNORECASE)

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
    ) -> List[str]:
        """Generate one question list that every interviewee can answer.

        The same questions go to residents and institutions alike. In Run 9 the
        prompt listed interviewee roles and asked for role-specific questions,
        which produced "As a journalist, ..." questions sent to the NHS and a
        56% rate of answers opening with "As a ...". In Run 10 the questions
        asked what interviewees saw or noticed, and 29 of 40 answers were
        summaries of the feed, so the questions now ask for their own position.
        """
        system_prompt = """You are a research interviewer. Generate 3-4 interview questions on the topic. Respond in English only.

The SAME questions are put to every interviewee: residents, workers, journalists, companies and public bodies. So each question must make sense for any of them.

Question requirements:
1. Ask for the interviewee's own position, one of these per question: what they did or decided and why; who they hold responsible; whom they trust or distrust now and why; what they will do next
2. Never ask what they saw, noticed, observed or heard others say, and never ask them to describe the public reaction
3. Never address a role or group ("As a journalist...", "For the council...", "How do residents...")
4. Use "you" and ask about their own actions and views
5. Under 35 words each, no background preamble

Return JSON: {"questions": ["question 1", "question 2", ...]}"""

        user_prompt = f"""Interview topic: {interview_requirement}

Simulation background: {simulation_requirement if simulation_requirement else "Not provided"}"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            questions = [q.strip() for q in response.get("questions", []) if isinstance(q, str) and q.strip()]
        except Exception as e:
            logger.warning(f"Failed to generate interview questions: {e}")
            questions = []

        if not questions:
            return [
                f"What did you do or decide about {interview_requirement}, and why?",
                "Who do you hold responsible, and whom do you trust or distrust now?",
                "What will you do next?",
            ]

        cleaned = []
        for q in questions:
            stripped = self._ROLE_ADDRESS.sub('', q)
            if stripped != q:
                logger.warning(f"Interview question addressed a role; removed the address: {q[:80]}")
                stripped = stripped[:1].upper() + stripped[1:]
            cleaned.append(stripped)
        return cleaned
