"""
API request and response schemas.

Pydantic models that define the contract between the
frontend and backend. FastAPI uses these for automatic
request validation, response serialization, and OpenAPI
documentation generation.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from domain.enums import RAGMode


# ─────────────────────────────────────────────────────────
# Chat endpoints
# ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Request body for POST /chat.

    The frontend sends the user's query along with the
    RAG mode and session tracking info. The session_id
    is required for the context pipeline to maintain state
    across turns.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question.",
        examples=["What machines are on Assembly Line A?"],
    )
    mode: RAGMode = Field(
        default=RAGMode.CONTEXT,
        description="Which RAG pipeline to use.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Session identifier for context tracking. "
            "Auto-generated if not provided."
        ),
    )
    turn_number: Optional[int] = Field(
        default=None,
        description=(
            "Conversation turn number. Auto-incremented "
            "if not provided."
        ),
    )


class ChatResponse(BaseModel):
    """
    Response body for POST /chat.

    Returns the generated answer along with metadata about
    what the pipeline did — extracted entities, sources,
    timing, and (for context mode) the augmented query and
    context state.
    """
    response: str = Field(
        ...,
        description="The generated answer.",
    )
    mode: RAGMode = Field(
        ...,
        description="Which pipeline produced this response.",
    )
    session_id: str = Field(
        ...,
        description="Session identifier (useful for subsequent requests).",
    )
    turn_number: int = Field(
        ...,
        description="Which conversation turn this was.",
    )

    # Pipeline metadata
    extracted_entities: list[str] = Field(
        default_factory=list,
        description="Entity names extracted from the query.",
    )
    augmented_query: Optional[str] = Field(
        default=None,
        description=(
            "The rewritten query after coreference resolution. "
            "Only populated in context mode."
        ),
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Knowledge graph entities used as retrieval sources.",
    )

    # Timing
    retrieval_time_ms: float = Field(
        default=0.0,
        description="Time spent on subgraph retrieval (milliseconds).",
    )
    generation_time_ms: float = Field(
        default=0.0,
        description="Time spent on LLM generation (milliseconds).",
    )


# ─────────────────────────────────────────────────────────
# Context graph endpoints
# ─────────────────────────────────────────────────────────

class ContextGraphResponse(BaseModel):
    """
    Response body for GET /context/{session_id}/graph.

    Returns the session's context graph in D3-compatible
    format for the frontend force-directed visualization.
    """
    session_id: str
    turn_count: int = 0
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    links: list[dict[str, Any]] = Field(default_factory=list)


class ContextSummaryResponse(BaseModel):
    """
    Response body for GET /context/{session_id}/summary.

    Returns a structured summary of the RCA investigation
    state for a given session.
    """
    session_id: str
    status: str = "no_session"
    turn_count: int = 0
    entities_tracked: int = 0
    suspected: list[dict[str, Any]] = Field(default_factory=list)
    ruled_out: list[dict[str, Any]] = Field(default_factory=list)
    follow_ups: list[dict[str, Any]] = Field(default_factory=list)
    context_string: str = ""


# ─────────────────────────────────────────────────────────
# Evaluation endpoints
# ─────────────────────────────────────────────────────────

class EvalSingleRequest(BaseModel):
    """
    Request body for POST /evaluate/single.

    Runs a single query through both pipelines for comparison.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to evaluate.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for context tracking.",
    )
    turn_number: int = Field(
        default=1,
        description="Which conversation turn this is.",
    )


class EvalTurnResponse(BaseModel):
    """
    Single turn result from one pipeline.

    Used within EvalComparisonResponse to show what each
    pipeline did for the same query.
    """
    response: str
    augmented_query: Optional[str] = None
    extracted_entities: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0


class EvalComparisonResponse(BaseModel):
    """
    Response body for POST /evaluate/single.

    Side-by-side results from both pipelines for one query.
    """
    turn_number: int
    query: str
    basic: EvalTurnResponse
    context: EvalTurnResponse


class EvalScenarioRequest(BaseModel):
    """
    Request body for POST /evaluate/scenario.

    Runs a multi-turn scenario. If no queries provided,
    uses the default RCA scenario.
    """
    queries: Optional[list[str]] = Field(
        default=None,
        description=(
            "List of queries in conversation order. "
            "Defaults to the RCA scenario if not provided."
        ),
    )
    scenario_name: Optional[str] = Field(
        default=None,
        description="Human-readable scenario name.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID prefix.",
    )


class EvalScenarioResponse(BaseModel):
    """
    Response body for POST /evaluate/scenario.

    Full multi-turn evaluation report with per-turn
    comparisons and an overall summary.
    """
    scenario_name: str
    session_id: str
    turns: list[EvalComparisonResponse] = Field(default_factory=list)
    summary: str = ""
    total_time_ms: float = 0.0

    # Aggregate timing
    basic_total_retrieval_ms: float = 0.0
    basic_total_generation_ms: float = 0.0
    context_total_retrieval_ms: float = 0.0
    context_total_generation_ms: float = 0.0


# ─────────────────────────────────────────────────────────
# Knowledge graph endpoints
# ─────────────────────────────────────────────────────────

class KnowledgeGraphResponse(BaseModel):
    """
    Response body for GET /graph.

    Returns the full static knowledge graph or a filtered
    subgraph in D3-compatible format.
    """
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    links: list[dict[str, Any]] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0


class SubGraphRequest(BaseModel):
    """
    Request body for POST /graph/subgraph.

    Retrieves a subgraph starting from seed entities.
    """
    seed_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Entity IDs to start traversal from.",
        examples=[["machine_m400", "line_a"]],
    )
    max_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum BFS traversal depth.",
    )
    max_entities: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Maximum entities to return.",
    )


# ─────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """
    Response body for GET /health.

    Reports the status of all system components.
    """
    status: str = "healthy"
    graph_loaded: bool = False
    graph_node_count: int = 0
    graph_edge_count: int = 0
    llm_status: str = "unknown"
    llm_model: str = ""
    llm_latency_ms: float = 0.0
    active_sessions: int = 0