"""
FastAPI route definitions.

All HTTP endpoints are defined here and grouped by concern.
Each route delegates to the appropriate service — no business
logic lives in this file. The router receives dependencies
(factory, repos, evaluation service) via a setup function
called at app startup.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    ChatRequest,
    ChatResponse,
    ContextGraphResponse,
    ContextSummaryResponse,
    EvalComparisonResponse,
    EvalScenarioRequest,
    EvalScenarioResponse,
    EvalSingleRequest,
    EvalTurnResponse,
    HealthResponse,
    KnowledgeGraphResponse,
    SubGraphRequest,
)
from domain.enums import RAGMode
from domain.models import PipelineState
from repositories.context_repo import ContextRepository
from repositories.graph_repo import GraphRepository
from repositories.llm_repo import LLMRepository
from services.evaluation import EvaluationService
from services.rag_factory import RAGFactory

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Router instance
# ─────────────────────────────────────────────────────────

router = APIRouter()

# ─────────────────────────────────────────────────────────
# Service references (injected at startup via setup())
# ─────────────────────────────────────────────────────────

_factory: Optional[RAGFactory] = None
_graph_repo: Optional[GraphRepository] = None
_context_repo: Optional[ContextRepository] = None
_llm_repo: Optional[LLMRepository] = None
_eval_service: Optional[EvaluationService] = None

# Tracks auto-incrementing turn numbers per session
_session_turns: dict[str, int] = {}


def setup(
    factory: RAGFactory,
    graph_repo: GraphRepository,
    context_repo: ContextRepository,
    llm_repo: LLMRepository,
    eval_service: EvaluationService,
) -> None:
    """
    Inject service dependencies into the router module.

    Called once during app startup after all services are
    initialized. This avoids circular imports and keeps
    route functions clean.

    Args:
        factory: RAG pipeline factory.
        graph_repo: Knowledge graph repository.
        context_repo: Session context repository.
        llm_repo: LLM interaction repository.
        eval_service: Side-by-side evaluation service.
    """
    global _factory, _graph_repo, _context_repo, _llm_repo, _eval_service
    _factory = factory
    _graph_repo = graph_repo
    _context_repo = context_repo
    _llm_repo = llm_repo
    _eval_service = eval_service
    logger.info("Router dependencies injected")


def _get_factory() -> RAGFactory:
    """Retrieve the factory, raising if not initialized."""
    if _factory is None:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Server is starting up.",
        )
    return _factory


def _get_next_turn(session_id: str) -> int:
    """
    Auto-increment and return the next turn number for a session.

    Args:
        session_id: The session to track.

    Returns:
        The next turn number (starts at 1).
    """
    current = _session_turns.get(session_id, 0)
    next_turn = current + 1
    _session_turns[session_id] = next_turn
    return next_turn


# ─────────────────────────────────────────────────────────
# Chat endpoints
# ─────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a query to a RAG pipeline",
    description=(
        "Sends a user query through the selected RAG pipeline "
        "(basic or context). For context mode, maintains session "
        "state across turns."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.

    Creates or retrieves the appropriate RAG service via the
    factory, builds a PipelineState, runs the pipeline, and
    returns the formatted response.
    """
    factory = _get_factory()

    # Resolve session ID and turn number
    session_id = request.session_id or str(uuid.uuid4())
    turn_number = request.turn_number or _get_next_turn(session_id)

    # Get the appropriate service
    service = factory.create(request.mode)

    # Build pipeline state
    state = PipelineState(
        query=request.query,
        session_id=session_id,
        turn_number=turn_number,
    )

    # Run the pipeline
    try:
        state = await service.query(state)
    except Exception as e:
        logger.error(
            "Pipeline error (mode=%s, session=%s): %s",
            request.mode.value,
            session_id,
            str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(e)}",
        )

    return ChatResponse(
        response=state.response,
        mode=request.mode,
        session_id=session_id,
        turn_number=turn_number,
        extracted_entities=state.extracted_entities,
        augmented_query=state.augmented_query,
        sources=state.sources,
        retrieval_time_ms=state.retrieval_time_ms,
        generation_time_ms=state.generation_time_ms,
    )


# ─────────────────────────────────────────────────────────
# Context graph endpoints
# ─────────────────────────────────────────────────────────

@router.get(
    "/context/{session_id}/graph",
    response_model=ContextGraphResponse,
    summary="Get session context graph",
    description=(
        "Returns the session's context graph in D3-compatible "
        "format for frontend visualization."
    ),
)
async def get_context_graph(session_id: str) -> ContextGraphResponse:
    """
    Retrieve the context graph for a session.

    Returns the accumulated entities, edges, and metadata
    that the context pipeline has built across turns.
    """
    if _context_repo is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    graph_data = _context_repo.to_json_graph(session_id)

    return ContextGraphResponse(
        session_id=session_id,
        turn_count=graph_data.get("turn_count", 0),
        nodes=graph_data.get("nodes", []),
        links=graph_data.get("links", []),
    )


@router.get(
    "/context/{session_id}/summary",
    response_model=ContextSummaryResponse,
    summary="Get RCA investigation summary",
    description=(
        "Returns a structured summary of the current RCA "
        "investigation state for a session."
    ),
)
async def get_context_summary(session_id: str) -> ContextSummaryResponse:
    """
    Retrieve the investigation summary for a session.

    Includes suspected causes, ruled-out entities, and
    items flagged for follow-up.
    """
    if _context_repo is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    summary = _context_repo.get_investigation_summary(session_id)
    return ContextSummaryResponse(**summary)


@router.delete(
    "/context/{session_id}",
    summary="Reset a session",
    description="Deletes all context state for a session.",
)
async def reset_session(session_id: str) -> dict:
    """
    Clear a session's context state.

    Used when the user wants to start a fresh investigation
    or the frontend resets.
    """
    if _context_repo is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    deleted = _context_repo.delete(session_id)

    # Also reset turn counter
    _session_turns.pop(session_id, None)

    return {
        "session_id": session_id,
        "deleted": deleted,
        "message": "Session reset" if deleted else "Session not found",
    }


# ─────────────────────────────────────────────────────────
# Evaluation endpoints
# ─────────────────────────────────────────────────────────

@router.post(
    "/evaluate/single",
    response_model=EvalComparisonResponse,
    summary="Compare a single query across both pipelines",
    description=(
        "Runs the same query through both basic and context "
        "RAG pipelines and returns side-by-side results."
    ),
)
async def evaluate_single(request: EvalSingleRequest) -> EvalComparisonResponse:
    """
    Single-query side-by-side comparison.

    Useful for testing individual questions and seeing
    how the two pipelines differ in their responses.
    """
    if _eval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    comparison = await _eval_service.compare_single(
        query=request.query,
        session_id=request.session_id,
        turn_number=request.turn_number,
    )

    return EvalComparisonResponse(
        turn_number=comparison.turn_number,
        query=comparison.query,
        basic=EvalTurnResponse(
            response=comparison.basic.response,
            augmented_query=comparison.basic.augmented_query,
            extracted_entities=comparison.basic.extracted_entities,
            sources=comparison.basic.sources,
            retrieval_time_ms=comparison.basic.retrieval_time_ms,
            generation_time_ms=comparison.basic.generation_time_ms,
        ),
        context=EvalTurnResponse(
            response=comparison.context.response,
            augmented_query=comparison.context.augmented_query,
            extracted_entities=comparison.context.extracted_entities,
            sources=comparison.context.sources,
            retrieval_time_ms=comparison.context.retrieval_time_ms,
            generation_time_ms=comparison.context.generation_time_ms,
        ),
    )


@router.post(
    "/evaluate/scenario",
    response_model=EvalScenarioResponse,
    summary="Run the full RCA scenario evaluation",
    description=(
        "Runs a multi-turn conversation through both pipelines. "
        "Defaults to the 6-turn RCA scenario if no queries provided."
    ),
)
async def evaluate_scenario(
    request: EvalScenarioRequest,
) -> EvalScenarioResponse:
    """
    Multi-turn scenario evaluation.

    This is the main demo endpoint. Runs 6 turns of the
    bad-batch investigation through both pipelines and
    returns a detailed comparison report.
    """
    if _eval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    report = await _eval_service.run_scenario(
        queries=request.queries,
        scenario_name=request.scenario_name,
        session_id=request.session_id,
    )

    # Convert internal TurnComparison → API response model
    turns = []
    for turn in report.turns:
        turns.append(
            EvalComparisonResponse(
                turn_number=turn.turn_number,
                query=turn.query,
                basic=EvalTurnResponse(
                    response=turn.basic.response,
                    augmented_query=turn.basic.augmented_query,
                    extracted_entities=turn.basic.extracted_entities,
                    sources=turn.basic.sources,
                    retrieval_time_ms=turn.basic.retrieval_time_ms,
                    generation_time_ms=turn.basic.generation_time_ms,
                ),
                context=EvalTurnResponse(
                    response=turn.context.response,
                    augmented_query=turn.context.augmented_query,
                    extracted_entities=turn.context.extracted_entities,
                    sources=turn.context.sources,
                    retrieval_time_ms=turn.context.retrieval_time_ms,
                    generation_time_ms=turn.context.generation_time_ms,
                ),
            )
        )

    return EvalScenarioResponse(
        scenario_name=report.scenario_name,
        session_id=report.session_id,
        turns=turns,
        summary=report.summary,
        total_time_ms=report.total_time_ms,
        basic_total_retrieval_ms=report.basic_total_retrieval_ms,
        basic_total_generation_ms=report.basic_total_generation_ms,
        context_total_retrieval_ms=report.context_total_retrieval_ms,
        context_total_generation_ms=report.context_total_generation_ms,
    )


# ─────────────────────────────────────────────────────────
# Knowledge graph endpoints
# ─────────────────────────────────────────────────────────

@router.get(
    "/graph",
    response_model=KnowledgeGraphResponse,
    summary="Get the full knowledge graph",
    description=(
        "Returns the entire static manufacturing knowledge "
        "graph in D3-compatible format."
    ),
)
async def get_knowledge_graph() -> KnowledgeGraphResponse:
    """
    Retrieve the full knowledge graph for visualization.

    Returns all entities and relationships. For large graphs,
    use POST /graph/subgraph instead.
    """
    if _graph_repo is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    graph_data = _graph_repo.to_json_graph()

    return KnowledgeGraphResponse(
        nodes=graph_data["nodes"],
        links=graph_data["links"],
        node_count=_graph_repo.node_count,
        edge_count=_graph_repo.edge_count,
    )


@router.post(
    "/graph/subgraph",
    response_model=KnowledgeGraphResponse,
    summary="Extract a subgraph from seed entities",
    description=(
        "Retrieves a subgraph by traversing N hops from "
        "the specified seed entity IDs."
    ),
)
async def get_subgraph(request: SubGraphRequest) -> KnowledgeGraphResponse:
    """
    Retrieve a focused subgraph for visualization or debugging.

    Useful for inspecting what the retrieval step sees
    for a given set of seed entities.
    """
    if _graph_repo is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    subgraph = _graph_repo.get_subgraph(
        seed_ids=request.seed_ids,
        max_hops=request.max_hops,
        max_entities=request.max_entities,
    )

    graph_data = _graph_repo.to_json_graph(subgraph=subgraph)

    return KnowledgeGraphResponse(
        nodes=graph_data["nodes"],
        links=graph_data["links"],
        node_count=len(subgraph.entities),
        edge_count=len(subgraph.relationships),
    )


# ─────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Reports the status of all system components.",
)
async def health_check() -> HealthResponse:
    """
    Check the health of all system components.

    Verifies that the knowledge graph is loaded, the LLM
    is reachable, and reports active session count.
    """
    # Graph status
    graph_loaded = _graph_repo is not None and _graph_repo.node_count > 0
    node_count = _graph_repo.node_count if _graph_repo else 0
    edge_count = _graph_repo.edge_count if _graph_repo else 0

    # LLM status
    llm_health = {"status": "unknown", "model": "", "latency_ms": 0.0}
    if _llm_repo is not None:
        llm_health = await _llm_repo.health_check()

    # Session count
    active_sessions = (
        len(_context_repo.list_sessions())
        if _context_repo
        else 0
    )

    # Overall status
    overall = "healthy"
    if not graph_loaded:
        overall = "degraded"
    if llm_health.get("status") != "healthy":
        overall = "degraded"

    return HealthResponse(
        status=overall,
        graph_loaded=graph_loaded,
        graph_node_count=node_count,
        graph_edge_count=edge_count,
        llm_status=llm_health.get("status", "unknown"),
        llm_model=llm_health.get("model", ""),
        llm_latency_ms=llm_health.get("latency_ms", 0.0),
        active_sessions=active_sessions,
    )