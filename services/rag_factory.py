"""
RAG pipeline factory.

Implements the factory pattern to create the appropriate
RAG service based on the requested mode. This keeps the
API layer decoupled from pipeline implementation details —
the router just calls `factory.create(mode)` and gets back
a service that implements `BaseRAGService`.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from domain.enums import RAGMode
from domain.models import ContextState, PipelineState, SubGraph
from repositories.context_repo import ContextRepository
from repositories.graph_repo import GraphRepository
from repositories.llm_repo import LLMRepository

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Abstract base service
# ─────────────────────────────────────────────────────────

class BaseRAGService(ABC):
    """
    Abstract interface for RAG pipelines.

    Both BasicGraphRAGService and ContextGraphRAGService
    implement this interface, so the API layer and evaluation
    service can treat them interchangeably.
    """

    @abstractmethod
    async def query(self, state: PipelineState) -> PipelineState:
        """
        Run the full RAG pipeline on a user query.

        Takes a PipelineState with at minimum `query` and
        `session_id` populated. Returns the same state object
        with `response`, `retrieved_subgraph`, and other
        fields populated by the pipeline.

        Args:
            state: The pipeline state to process.

        Returns:
            Updated PipelineState with response and metadata.
        """
        ...

    @abstractmethod
    def get_mode(self) -> RAGMode:
        """
        Return which RAG mode this service implements.

        Returns:
            RAGMode.BASIC or RAGMode.CONTEXT.
        """
        ...

    @abstractmethod
    def get_context_state(self, session_id: str) -> Optional[ContextState]:
        """
        Get the current context state for a session.

        Basic pipeline always returns None.
        Context pipeline returns the accumulated state.

        Args:
            session_id: The session to look up.

        Returns:
            ContextState or None.
        """
        ...

    @abstractmethod
    def get_context_graph_json(self, session_id: str) -> dict:
        """
        Get the context graph as D3-compatible JSON.

        Basic pipeline returns an empty graph.
        Context pipeline returns the session's context graph.

        Args:
            session_id: The session to serialize.

        Returns:
            Dict with "nodes" and "links" lists.
        """
        ...


# ─────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────

class RAGFactory:
    """
    Factory that creates RAG service instances.

    Holds references to the shared repositories and injects
    them into the appropriate service class. The repositories
    are created once at app startup and shared across all
    service instances.

    Usage:
        factory = RAGFactory(graph_repo, context_repo, llm_repo)
        basic_service = factory.create(RAGMode.BASIC)
        context_service = factory.create(RAGMode.CONTEXT)
    """

    def __init__(
        self,
        graph_repo: GraphRepository,
        context_repo: ContextRepository,
        llm_repo: LLMRepository,
    ) -> None:
        """
        Initialize the factory with shared repositories.

        Args:
            graph_repo: The knowledge graph repository.
            context_repo: The session context repository.
            llm_repo: The LLM interaction repository.
        """
        self._graph_repo = graph_repo
        self._context_repo = context_repo
        self._llm_repo = llm_repo

        # Cache service instances — one per mode
        self._services: dict[RAGMode, BaseRAGService] = {}

        logger.info("RAG factory initialized")

    def create(self, mode: RAGMode) -> BaseRAGService:
        """
        Create or retrieve a RAG service for the given mode.

        Services are cached after first creation — calling
        `create(RAGMode.BASIC)` twice returns the same instance.
        This is safe because services are stateless (state lives
        in the repositories, not the service).

        Args:
            mode: Which RAG pipeline to create.

        Returns:
            A BaseRAGService implementation.

        Raises:
            ValueError: If the mode is not recognized.
        """
        # Return cached instance if available
        if mode in self._services:
            return self._services[mode]

        # Import here to avoid circular imports
        # (services import from this module for BaseRAGService)
        from services.basic_graph_rag import BasicGraphRAGService
        from services.context_graph_rag import ContextGraphRAGService

        if mode == RAGMode.BASIC:
            service = BasicGraphRAGService(
                graph_repo=self._graph_repo,
                llm_repo=self._llm_repo,
            )
        elif mode == RAGMode.CONTEXT:
            service = ContextGraphRAGService(
                graph_repo=self._graph_repo,
                context_repo=self._context_repo,
                llm_repo=self._llm_repo,
            )
        else:
            raise ValueError(f"Unknown RAG mode: {mode}")

        # Cache for future calls
        self._services[mode] = service

        logger.info("Created RAG service: %s", mode.value)
        return service

    def create_both(self) -> tuple[BaseRAGService, BaseRAGService]:
        """
        Convenience method to create both services at once.

        Used by the evaluation service which needs to run
        the same query through both pipelines.

        Returns:
            Tuple of (basic_service, context_service).
        """
        basic = self.create(RAGMode.BASIC)
        context = self.create(RAGMode.CONTEXT)
        return basic, context