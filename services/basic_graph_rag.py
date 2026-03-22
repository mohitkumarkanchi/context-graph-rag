"""
Basic Graph RAG service.

Stateless pipeline — every query is independent. No memory
of previous turns, no coreference resolution, no investigation
chain. This is the baseline that the context pipeline improves upon.

LangGraph pipeline nodes:
    1. extract_entities   — Pull entity names from the query
    2. retrieve_subgraph  — Find matching nodes and traverse N hops
    3. generate_response  — Feed subgraph context to LLM

This pipeline works well for single-shot questions like
"What oil does CNC Mill M-400 need?" but fails on follow-ups
like "When was it last serviced?" because it has no idea
what "it" refers to.
"""

import logging
import time
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from domain.enums import RAGMode
from domain.models import ContextState, PipelineState, SubGraph
from repositories.graph_repo import GraphRepository
from repositories.llm_repo import LLMRepository
from services.rag_factory import BaseRAGService

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# LangGraph state type
#
# LangGraph requires a TypedDict for state. We bridge
# between this and our Pydantic PipelineState at the
# entry/exit points of the graph.
# ─────────────────────────────────────────────────────────


class BasicGraphState(TypedDict):
    """LangGraph state schema for the basic pipeline."""

    # Input fields
    query: str
    session_id: str
    turn_number: int

    # Extraction output
    extracted_entities: list[str]

    # Retrieval output
    retrieved_subgraph_context: str
    retrieved_entity_count: int
    retrieved_relationship_count: int

    # Generation output
    response: str
    sources: list[str]

    # Timing metrics
    retrieval_time_ms: float
    generation_time_ms: float


class BasicGraphRAGService(BaseRAGService):
    """
    Stateless Graph RAG pipeline.

    Each query goes through extract → retrieve → generate
    with no awareness of previous conversation turns.
    Implements BaseRAGService so it's interchangeable
    with the context pipeline via the factory.
    """

    def __init__(
        self,
        graph_repo: GraphRepository,
        llm_repo: LLMRepository,
    ) -> None:
        """
        Initialize with required repositories.

        Args:
            graph_repo: Knowledge graph for retrieval.
            llm_repo: LLM for extraction and generation.
        """
        self._graph_repo = graph_repo
        self._llm_repo = llm_repo
        self._pipeline = self._build_pipeline()

        logger.info("Basic Graph RAG service initialized")

    # ─────────────────────────────────────────────────────
    # BaseRAGService interface
    # ─────────────────────────────────────────────────────

    async def query(self, state: PipelineState) -> PipelineState:
        """
        Run the basic RAG pipeline on a user query.

        Converts the Pydantic PipelineState to a LangGraph
        TypedDict, runs the graph, and converts back.

        Args:
            state: Pipeline state with query and session_id set.

        Returns:
            Updated state with response and retrieval metadata.
        """
        # Convert Pydantic model → LangGraph TypedDict
        initial_state: BasicGraphState = {
            "query": state.query,
            "session_id": state.session_id,
            "turn_number": state.turn_number,
            "extracted_entities": [],
            "retrieved_subgraph_context": "",
            "retrieved_entity_count": 0,
            "retrieved_relationship_count": 0,
            "response": "",
            "sources": [],
            "retrieval_time_ms": 0.0,
            "generation_time_ms": 0.0,
        }

        # Run the LangGraph pipeline
        result = await self._pipeline.ainvoke(initial_state)

        # Map result back to PipelineState
        state.extracted_entities = result["extracted_entities"]
        state.response = result["response"]
        state.sources = result["sources"]
        state.retrieval_time_ms = result["retrieval_time_ms"]
        state.generation_time_ms = result["generation_time_ms"]

        # Reconstruct SubGraph metadata for evaluation
        state.retrieved_subgraph = SubGraph(
            entities=[],
            relationships=[],
        )

        logger.info(
            "Basic RAG query completed: %d entities extracted, "
            "%d entities retrieved, %.0fms retrieval, %.0fms generation",
            len(state.extracted_entities),
            result["retrieved_entity_count"],
            state.retrieval_time_ms,
            state.generation_time_ms,
        )

        return state

    def get_mode(self) -> RAGMode:
        """Return RAGMode.BASIC."""
        return RAGMode.BASIC

    def get_context_state(self, session_id: str) -> Optional[ContextState]:
        """
        Always returns None — basic pipeline has no context.

        Args:
            session_id: Ignored.

        Returns:
            None.
        """
        return None

    def get_context_graph_json(self, session_id: str) -> dict:
        """
        Returns empty graph — basic pipeline has no context.

        Args:
            session_id: Ignored.

        Returns:
            Empty nodes/links dict.
        """
        return {"nodes": [], "links": []}

    # ─────────────────────────────────────────────────────
    # Pipeline construction
    # ─────────────────────────────────────────────────────

    def _build_pipeline(self) -> StateGraph:
        """
        Build the LangGraph state graph for the basic pipeline.

        Graph structure:
            START → extract_entities
                  → retrieve_subgraph
                  → generate_response
                  → END

        Returns:
            Compiled StateGraph ready for invocation.
        """
        graph = StateGraph(BasicGraphState)

        # Register nodes
        graph.add_node("extract_entities", self._node_extract_entities)
        graph.add_node("retrieve_subgraph", self._node_retrieve_subgraph)
        graph.add_node("generate_response", self._node_generate_response)

        # Linear flow: extract → retrieve → generate
        graph.set_entry_point("extract_entities")
        graph.add_edge("extract_entities", "retrieve_subgraph")
        graph.add_edge("retrieve_subgraph", "generate_response")
        graph.add_edge("generate_response", END)

        return graph.compile()

    # ─────────────────────────────────────────────────────
    # Pipeline nodes
    # ─────────────────────────────────────────────────────

    async def _node_extract_entities(
        self,
        state: BasicGraphState,
    ) -> dict[str, Any]:
        """
        Node 1: Extract entity names from the user query.

        Uses the LLM to identify manufacturing-domain entities
        mentioned in the query. These become the seed nodes
        for subgraph retrieval.

        Falls back to using the raw query as a search term
        if the LLM extraction returns nothing.

        Args:
            state: Current pipeline state.

        Returns:
            Dict with updated extracted_entities list.
        """
        query = state["query"]

        logger.debug("Extracting entities from: '%s'", query)
        entities = await self._llm_repo.extract_entities(query)

        # Fallback: use query itself as a search term
        if not entities:
            logger.debug(
                "LLM extraction returned nothing, "
                "using raw query as search term"
            )
            entities = [query]

        logger.debug("Extracted entities: %s", entities)
        return {"extracted_entities": entities}

    async def _node_retrieve_subgraph(
        self,
        state: BasicGraphState,
    ) -> dict[str, Any]:
        """
        Node 2: Retrieve relevant subgraph from knowledge graph.

        Takes extracted entity names and resolves them to graph
        nodes via two strategies:
            1. Exact name match (fast, precise)
            2. Scored search fallback (fuzzy, broader)

        Then traverses N hops from matched nodes via BFS to
        build the context subgraph for generation.

        Args:
            state: Current pipeline state with extracted_entities.

        Returns:
            Dict with subgraph context string, sources, and
            retrieval metadata.
        """
        start = time.perf_counter()

        extracted = state["extracted_entities"]
        seed_ids: list[str] = []
        sources: list[str] = []

        for entity_name in extracted:
            # Strategy 1: exact name match
            entity = self._graph_repo.get_entity_by_name(entity_name)
            if entity:
                seed_ids.append(entity.id)
                sources.append(
                    f"{entity.name} ({entity.entity_type.value})"
                )
                continue

            # Strategy 2: scored search (top 3 per term)
            search_results = self._graph_repo.search_entities(entity_name)
            for result in search_results[:3]:
                if result.id not in seed_ids:
                    seed_ids.append(result.id)
                    sources.append(
                        f"{result.name} ({result.entity_type.value})"
                    )

        # BFS traversal from seed nodes
        if seed_ids:
            subgraph = self._graph_repo.get_subgraph(seed_ids=seed_ids)
            context_str = subgraph.to_context_string()
            entity_count = len(subgraph.entities)
            rel_count = len(subgraph.relationships)
        else:
            context_str = "No relevant entities found in the knowledge graph."
            entity_count = 0
            rel_count = 0

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "Retrieved subgraph: %d entities, %d relationships "
            "from seeds %s (%.1fms)",
            entity_count,
            rel_count,
            seed_ids,
            elapsed_ms,
        )

        return {
            "retrieved_subgraph_context": context_str,
            "retrieved_entity_count": entity_count,
            "retrieved_relationship_count": rel_count,
            "sources": sources,
            "retrieval_time_ms": elapsed_ms,
        }

    async def _node_generate_response(
        self,
        state: BasicGraphState,
    ) -> dict[str, Any]:
        """
        Node 3: Generate answer from retrieved context.

        Passes the subgraph context string to the LLM with
        a system prompt that instructs it to answer using
        only the provided knowledge graph information.

        No session context is passed — this is the stateless
        pipeline. The LLM sees only the current query and the
        retrieved subgraph.

        Args:
            state: Current pipeline state with subgraph context.

        Returns:
            Dict with the generated response string and timing.
        """
        start = time.perf_counter()

        response = await self._llm_repo.generate_response(
            query=state["query"],
            subgraph_context=state["retrieved_subgraph_context"],
            context_summary=None,  # No context — stateless pipeline
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug("Generated response in %.1fms", elapsed_ms)

        return {
            "response": response,
            "generation_time_ms": elapsed_ms,
        }