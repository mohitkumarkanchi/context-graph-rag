"""
Context Graph RAG service.

Stateful pipeline — maintains a session context graph that
accumulates entities, coreference resolutions, and RCA
investigation chains across conversation turns.

LangGraph pipeline nodes:
    1. load_context       — Load or create session context state
    2. resolve_coref      — Resolve pronouns using context ("it" → M-400)
    3. augment_query      — Rewrite vague queries into explicit ones
    4. extract_entities   — Pull entity names from augmented query
    5. retrieve_subgraph  — Retrieve with context-aware re-ranking
    6. generate_response  — Generate answer with full context
    7. update_context     — Write new entities/edges back to context graph

The key difference from the basic pipeline is steps 1–3 and 7:
context is loaded before retrieval, used to resolve ambiguity,
and updated after generation. This creates a feedback loop where
each turn enriches the context for the next turn.
"""

import logging
import time
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from domain.enums import RAGMode
from domain.models import ContextState, PipelineState, SubGraph
from repositories.context_repo import ContextRepository
from repositories.graph_repo import GraphRepository
from repositories.llm_repo import LLMRepository
from services.rag_factory import BaseRAGService

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# LangGraph state type
# ─────────────────────────────────────────────────────────

from typing import TypedDict


class ContextGraphState(TypedDict):
    """
    LangGraph state schema for the context pipeline.

    Extends the basic state with context-specific fields
    for coreference resolution, query augmentation, and
    RCA investigation tracking.
    """
    # Input
    query: str
    session_id: str
    turn_number: int

    # Context (loaded from session store)
    context_summary: str
    resolved_references: dict[str, str]
    recent_entity_ids: list[str]

    # Augmentation
    augmented_query: str

    # Extraction & retrieval
    extracted_entities: list[str]
    seed_entity_ids: list[str]
    retrieved_subgraph_context: str
    retrieved_entity_count: int
    retrieved_relationship_count: int

    # Generation
    response: str
    sources: list[str]
    rca_metadata: dict[str, Any]

    # Timing
    retrieval_time_ms: float
    generation_time_ms: float


class ContextGraphRAGService(BaseRAGService):
    """
    Stateful Graph RAG pipeline with session context.

    Each query flows through 7 nodes that collectively
    load context, resolve ambiguity, retrieve with awareness
    of the conversation history, generate a contextual response,
    and update the session state for the next turn.
    """

    def __init__(
        self,
        graph_repo: GraphRepository,
        context_repo: ContextRepository,
        llm_repo: LLMRepository,
    ) -> None:
        """
        Initialize with all three repositories.

        The context_repo is what makes this pipeline stateful —
        it's not used by the basic pipeline at all.

        Args:
            graph_repo: Knowledge graph for retrieval.
            context_repo: Session context store.
            llm_repo: LLM for all language tasks.
        """
        self._graph_repo = graph_repo
        self._context_repo = context_repo
        self._llm_repo = llm_repo
        self._pipeline = self._build_pipeline()

        logger.info("Context Graph RAG service initialized")

    # ─────────────────────────────────────────────────────
    # BaseRAGService interface
    # ─────────────────────────────────────────────────────

    async def query(self, state: PipelineState) -> PipelineState:
        """
        Run the context-aware RAG pipeline.

        Converts Pydantic state → LangGraph TypedDict, runs
        the 7-node pipeline, and converts back. The context
        repository is mutated during execution (new entities,
        edges, coreference resolutions are recorded).

        Args:
            state: Pipeline state with query, session_id, and
                turn_number populated.

        Returns:
            Updated state with contextual response and metadata.
        """
        # Convert Pydantic model → LangGraph TypedDict
        initial_state: ContextGraphState = {
            "query": state.query,
            "session_id": state.session_id,
            "turn_number": state.turn_number,
            "context_summary": "",
            "resolved_references": {},
            "recent_entity_ids": [],
            "augmented_query": "",
            "extracted_entities": [],
            "seed_entity_ids": [],
            "retrieved_subgraph_context": "",
            "retrieved_entity_count": 0,
            "retrieved_relationship_count": 0,
            "response": "",
            "sources": [],
            "rca_metadata": {},
            "retrieval_time_ms": 0.0,
            "generation_time_ms": 0.0,
        }

        # Run the LangGraph pipeline
        result = await self._pipeline.ainvoke(initial_state)

        # Map result back to PipelineState
        state.extracted_entities = result["extracted_entities"]
        state.augmented_query = result["augmented_query"]
        state.response = result["response"]
        state.sources = result["sources"]
        state.retrieval_time_ms = result["retrieval_time_ms"]
        state.generation_time_ms = result["generation_time_ms"]

        # Attach context state for the API response
        state.context_state = self._context_repo.get(state.session_id)

        logger.info(
            "Context RAG query completed (turn %d): "
            "augmented='%s', %d entities retrieved, "
            "%.0fms retrieval, %.0fms generation",
            state.turn_number,
            result["augmented_query"][:80],
            result["retrieved_entity_count"],
            state.retrieval_time_ms,
            state.generation_time_ms,
        )

        return state

    def get_mode(self) -> RAGMode:
        """Return RAGMode.CONTEXT."""
        return RAGMode.CONTEXT

    def get_context_state(self, session_id: str) -> Optional[ContextState]:
        """
        Get the accumulated context state for a session.

        Args:
            session_id: The session to look up.

        Returns:
            ContextState if session exists, None otherwise.
        """
        return self._context_repo.get(session_id)

    def get_context_graph_json(self, session_id: str) -> dict:
        """
        Get the session context graph as D3-compatible JSON.

        Args:
            session_id: The session to serialize.

        Returns:
            Dict with nodes and links for frontend visualization.
        """
        return self._context_repo.to_json_graph(session_id)

    # ─────────────────────────────────────────────────────
    # Pipeline construction
    # ─────────────────────────────────────────────────────

    def _build_pipeline(self) -> StateGraph:
        """
        Build the 7-node LangGraph state graph.

        Graph structure:
            START → load_context
                  → resolve_coref
                  → augment_query
                  → extract_entities
                  → retrieve_subgraph
                  → generate_response
                  → update_context
                  → END

        All nodes are sequential — no branching. Each node
        reads from and writes to the shared ContextGraphState.

        Returns:
            Compiled StateGraph ready for invocation.
        """
        graph = StateGraph(ContextGraphState)

        # Add all 7 nodes
        graph.add_node("load_context", self._node_load_context)
        graph.add_node("resolve_coref", self._node_resolve_coref)
        graph.add_node("augment_query", self._node_augment_query)
        graph.add_node("extract_entities", self._node_extract_entities)
        graph.add_node("retrieve_subgraph", self._node_retrieve_subgraph)
        graph.add_node("generate_response", self._node_generate_response)
        graph.add_node("update_context", self._node_update_context)

        # Linear flow through all nodes
        graph.set_entry_point("load_context")
        graph.add_edge("load_context", "resolve_coref")
        graph.add_edge("resolve_coref", "augment_query")
        graph.add_edge("augment_query", "extract_entities")
        graph.add_edge("extract_entities", "retrieve_subgraph")
        graph.add_edge("retrieve_subgraph", "generate_response")
        graph.add_edge("generate_response", "update_context")
        graph.add_edge("update_context", END)

        return graph.compile()

    # ─────────────────────────────────────────────────────
    # Pipeline nodes
    # ─────────────────────────────────────────────────────

    async def _node_load_context(
        self,
        state: ContextGraphState,
    ) -> dict[str, Any]:
        """
        Node 1: Load existing session context.

        Retrieves (or creates) the session's ContextState and
        extracts the summary string, resolved references, and
        recently discussed entity IDs. These feed into the
        coreference resolution and retrieval steps.

        Args:
            state: Current pipeline state.

        Returns:
            Dict with context_summary, resolved_references,
            and recent_entity_ids.
        """
        session_id = state["session_id"]

        context_state = self._context_repo.get_or_create(session_id)

        context_summary = context_state.to_context_string()
        resolved_refs = dict(context_state.resolved_references)
        recent_ids = context_state.get_recent_entities(last_n_turns=3)

        logger.debug(
            "Loaded context for session %s: %d entities tracked, "
            "%d resolved refs, %d recent entities",
            session_id,
            len(context_state.entity_ids),
            len(resolved_refs),
            len(recent_ids),
        )

        return {
            "context_summary": context_summary,
            "resolved_references": resolved_refs,
            "recent_entity_ids": recent_ids,
        }

    async def _node_resolve_coref(
        self,
        state: ContextGraphState,
    ) -> dict[str, Any]:
        """
        Node 2: Resolve pronouns and vague references.

        If the query contains ambiguous references like "it",
        "that machine", "those parts", this node uses the LLM
        and session context to resolve them to specific entity IDs.

        On turn 1 (no context yet), this is a no-op.

        Args:
            state: Current pipeline state with context loaded.

        Returns:
            Dict with updated resolved_references.
        """
        context_summary = state["context_summary"]

        # Skip coreference resolution on the first turn
        # (no context to resolve against)
        if not state["recent_entity_ids"]:
            logger.debug("First turn — skipping coreference resolution")
            return {"resolved_references": {}}

        # Ask LLM to resolve references
        new_refs = await self._llm_repo.resolve_coreferences(
            query=state["query"],
            context_summary=context_summary,
        )

        # Merge with existing resolutions (new ones take precedence)
        merged_refs = {**state["resolved_references"], **new_refs}

        # Record resolutions in the context repository
        session_id = state["session_id"]
        turn = state["turn_number"]
        for pronoun, entity_id in new_refs.items():
            self._context_repo.resolve_reference(
                session_id=session_id,
                pronoun=pronoun,
                entity_id=entity_id,
                turn_number=turn,
            )

        if new_refs:
            logger.debug("Resolved coreferences: %s", new_refs)

        return {"resolved_references": merged_refs}

    async def _node_augment_query(
        self,
        state: ContextGraphState,
    ) -> dict[str, Any]:
        """
        Node 3: Rewrite the query using resolved references.

        Transforms vague queries into explicit ones:
            "When was it last serviced?"
            → "When was CNC Mill M-400 last serviced?"

        If no coreferences were resolved, the original query
        passes through unchanged.

        Args:
            state: Current pipeline state with resolved references.

        Returns:
            Dict with the augmented_query string.
        """
        resolved_refs = state["resolved_references"]

        augmented = await self._llm_repo.augment_query(
            query=state["query"],
            resolved_refs=resolved_refs,
            context_summary=state["context_summary"],
        )

        logger.debug(
            "Query augmentation: '%s' → '%s'",
            state["query"],
            augmented,
        )

        return {"augmented_query": augmented}

    async def _node_extract_entities(
        self,
        state: ContextGraphState,
    ) -> dict[str, Any]:
        """
        Node 4: Extract entities from the augmented query.

        Uses the augmented (explicit) query rather than the
        original, so entity extraction benefits from coreference
        resolution. For example, "When was CNC Mill M-400 last
        serviced?" will extract "CNC Mill M-400" where the
        original "When was it last serviced?" would not.

        Args:
            state: Current pipeline state with augmented query.

        Returns:
            Dict with extracted_entities list.
        """
        # Use augmented query if available, fall back to original
        query = state["augmented_query"] or state["query"]

        entities = await self._llm_repo.extract_entities(query)

        # Fall back to query as search term if extraction fails
        if not entities:
            logger.debug(
                "Entity extraction returned nothing, "
                "using augmented query as search term"
            )
            entities = [query]

        logger.debug("Extracted entities: %s", entities)
        return {"extracted_entities": entities}

    async def _node_retrieve_subgraph(
        self,
        state: ContextGraphState,
    ) -> dict[str, Any]:
        """
        Node 5: Context-aware subgraph retrieval.

        This is where the context pipeline diverges most from
        the basic pipeline. In addition to retrieving based on
        extracted entities, it also:

        1. Includes recently discussed entities as additional
           seed nodes (so the context neighborhood stays in scope)
        2. Merges the new retrieval with context-relevant subgraphs
        3. Prioritizes entities in the RCA investigation chain

        Args:
            state: Current pipeline state with extracted entities
                and context.

        Returns:
            Dict with subgraph context, sources, and metadata.
        """
        start = time.perf_counter()

        extracted = state["extracted_entities"]
        recent_ids = state["recent_entity_ids"]

        # ── Step 1: Resolve extracted names to graph node IDs ──

        seed_ids: list[str] = []
        sources: list[str] = []

        for entity_name in extracted:
            # Try exact name match
            entity = self._graph_repo.get_entity_by_name(entity_name)
            if entity:
                seed_ids.append(entity.id)
                sources.append(
                    f"{entity.name} ({entity.entity_type.value})"
                )
                continue

            # Fall back to scored search
            search_results = self._graph_repo.search_entities(entity_name)
            for result in search_results[:3]:
                if result.id not in seed_ids:
                    seed_ids.append(result.id)
                    sources.append(
                        f"{result.name} ({result.entity_type.value})"
                    )

        # ── Step 2: Add recently discussed entities as seeds ──
        #
        # This is the key context-aware enhancement. By including
        # entities from recent turns, the retrieval stays anchored
        # to the conversation's focus area.

        for recent_id in recent_ids:
            if recent_id not in seed_ids:
                seed_ids.append(recent_id)
                entity = self._graph_repo.get_entity(recent_id)
                if entity:
                    sources.append(
                        f"{entity.name} ({entity.entity_type.value}) [context]"
                    )

        # ── Step 3: Retrieve subgraph with expanded seeds ──

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
            "Context-aware retrieval: %d seeds (%d from context), "
            "%d entities, %d rels (%.1fms)",
            len(seed_ids),
            len(recent_ids),
            entity_count,
            rel_count,
            elapsed_ms,
        )

        return {
            "seed_entity_ids": seed_ids,
            "retrieved_subgraph_context": context_str,
            "retrieved_entity_count": entity_count,
            "retrieved_relationship_count": rel_count,
            "sources": sources,
            "retrieval_time_ms": elapsed_ms,
        }

    async def _node_generate_response(
        self,
        state: ContextGraphState,
    ) -> dict[str, Any]:
        """
        Node 6: Generate context-aware response.

        Passes both the subgraph context AND the session context
        summary to the LLM. This allows the model to:
        - Reference previously discussed entities naturally
        - Maintain continuity with earlier answers
        - Proactively surface related information
        - Build on the RCA investigation chain

        Also attempts RCA-specific analysis if there's an
        active investigation chain in the context.

        Args:
            state: Current pipeline state with subgraph and context.

        Returns:
            Dict with response string and RCA metadata.
        """
        start = time.perf_counter()

        context_summary = state["context_summary"]
        subgraph_context = state["retrieved_subgraph_context"]

        # Check if there's an active RCA investigation
        investigation = self._context_repo.get_investigation_summary(
            state["session_id"]
        )
        has_investigation = (
            investigation.get("status") == "active"
            and investigation.get("suspected", [])
        )

        if has_investigation:
            # Use RCA-specific generation for structured output
            rca_result = await self._llm_repo.analyze_rca_step(
                query=state["augmented_query"] or state["query"],
                subgraph_context=subgraph_context,
                investigation_summary=investigation,
            )
            response = rca_result["answer"]
            rca_metadata = rca_result
        else:
            # Standard generation with context
            response = await self._llm_repo.generate_response(
                query=state["augmented_query"] or state["query"],
                subgraph_context=subgraph_context,
                context_summary=context_summary,
            )
            rca_metadata = {}

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "Generated context-aware response in %.1fms "
            "(rca_active=%s)",
            elapsed_ms,
            has_investigation,
        )

        return {
            "response": response,
            "rca_metadata": rca_metadata,
            "generation_time_ms": elapsed_ms,
        }

    async def _node_update_context(
        self,
        state: ContextGraphState,
    ) -> dict[str, Any]:
        """
        Node 7: Update the session context graph.

        Records what happened in this turn:
        - Entities that were discussed (from extraction + retrieval)
        - New RCA causal links (from RCA metadata)
        - Entities flagged for follow-up
        - Advances the turn counter

        This is the feedback loop — the context written here
        feeds into Node 1 (load_context) on the next turn.

        Args:
            state: Fully populated pipeline state.

        Returns:
            Empty dict (all mutations go to the context repo).
        """
        session_id = state["session_id"]
        turn = state["turn_number"]

        # ── Record discussed entities ──

        # Combine extracted entities resolved to graph IDs
        discussed_ids = list(set(state["seed_entity_ids"]))
        if discussed_ids:
            self._context_repo.add_discussed_entities(
                session_id=session_id,
                entity_ids=discussed_ids,
                turn_number=turn,
            )

        # ── Process RCA metadata if present ──

        rca = state.get("rca_metadata", {})

        # Record suspected causes
        for cause in rca.get("suspected_causes", []):
            source = cause.get("source", "")
            target = cause.get("target", "")
            if source and target:
                self._context_repo.mark_suspected_cause(
                    session_id=session_id,
                    source_id=source,
                    target_id=target,
                    turn_number=turn,
                )

        # Record ruled-out causes
        for ruled in rca.get("ruled_out", []):
            source = ruled.get("source", "")
            target = ruled.get("target", "")
            if source and target:
                self._context_repo.rule_out_cause(
                    session_id=session_id,
                    source_id=source,
                    target_id=target,
                    turn_number=turn,
                    reason=ruled.get("reason", ""),
                )

        # Record follow-ups
        for follow in rca.get("follow_ups", []):
            entity_id = follow.get("entity", "")
            if entity_id:
                self._context_repo.mark_follow_up(
                    session_id=session_id,
                    entity_id=entity_id,
                    turn_number=turn,
                    reason=follow.get("reason", ""),
                )

        # Track any new entities mentioned in the RCA response
        for entity_id in rca.get("new_entities", []):
            self._context_repo.add_discussed_entities(
                session_id=session_id,
                entity_ids=[entity_id],
                turn_number=turn,
            )

        logger.debug(
            "Context updated for session %s turn %d: "
            "%d entities, %d suspected, %d ruled out, %d follow-ups",
            session_id,
            turn,
            len(discussed_ids),
            len(rca.get("suspected_causes", [])),
            len(rca.get("ruled_out", [])),
            len(rca.get("follow_ups", [])),
        )

        # Return empty — all state changes went to context_repo
        return {}