"""
Evaluation service.

Runs the same multi-turn conversation through both RAG pipelines
side by side and produces a structured comparison report. This is
the core of the demo — it makes the difference between stateless
and stateful RAG visually obvious.

Supports two modes:
    1. Single query comparison — same question through both pipelines
    2. Scenario evaluation  — a scripted multi-turn conversation
       (like the 6-turn RCA scenario) run through both pipelines
"""

import logging
import time
import uuid
from typing import Optional

from domain.enums import RAGMode
from domain.models import PipelineState
from services.rag_factory import BaseRAGService, RAGFactory

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Data structures for evaluation results
# ─────────────────────────────────────────────────────────

from pydantic import BaseModel, Field


class TurnResult(BaseModel):
    """Result of a single turn from one pipeline."""
    turn_number: int
    query: str
    augmented_query: Optional[str] = None
    response: str
    extracted_entities: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0


class TurnComparison(BaseModel):
    """Side-by-side comparison of a single turn."""
    turn_number: int
    query: str
    basic: TurnResult
    context: TurnResult
    analysis: str = ""


class ScenarioReport(BaseModel):
    """
    Full evaluation report for a multi-turn scenario.

    Contains per-turn comparisons and an overall summary
    highlighting where context RAG outperforms basic RAG.
    """
    scenario_name: str
    session_id: str
    turns: list[TurnComparison] = Field(default_factory=list)
    summary: str = ""
    total_time_ms: float = 0.0
    basic_total_retrieval_ms: float = 0.0
    basic_total_generation_ms: float = 0.0
    context_total_retrieval_ms: float = 0.0
    context_total_generation_ms: float = 0.0


# ─────────────────────────────────────────────────────────
# Predefined RCA scenario
# ─────────────────────────────────────────────────────────

RCA_SCENARIO = {
    "name": "Manufacturing RCA — Bad Batch B-442",
    "description": (
        "An operator discovers defective parts coming off Assembly Line A. "
        "Over 6 turns, they trace the root cause from the defect back to "
        "a specific supplier batch, discovering that other machines are "
        "also at risk."
    ),
    "queries": [
        # Turn 1 — Both pipelines handle this fine
        (
            "We're seeing defective parts coming off Assembly Line A. "
            "What machines are on that line?"
        ),
        # Turn 2 — Basic starts losing context ("those machines")
        (
            "Check the sensor data for those machines — "
            "any anomalies recently?"
        ),
        # Turn 3 — Basic can't resolve "it"
        "When was it last serviced?",
        # Turn 4 — Basic has no idea what "that service" means
        "What parts were replaced during that service?",
        # Turn 5 — Basic completely lost ("that batch")
        "Did that batch go to any other machines?",
        # Turn 6 — Basic cannot connect anything
        "Are those machines showing any early warning signs?",
    ],
}


# ─────────────────────────────────────────────────────────
# Evaluation service
# ─────────────────────────────────────────────────────────

class EvaluationService:
    """
    Runs side-by-side comparisons between RAG pipelines.

    Uses the RAGFactory to obtain both pipeline instances
    and runs identical queries through each, collecting
    results and timing for comparison.
    """

    def __init__(self, factory: RAGFactory) -> None:
        """
        Initialize with a RAG factory.

        Args:
            factory: Factory to create both pipeline instances.
        """
        self._factory = factory
        self._basic: BaseRAGService
        self._context: BaseRAGService
        self._basic, self._context = factory.create_both()

        logger.info("Evaluation service initialized")

    # ─────────────────────────────────────────────────────
    # Single query comparison
    # ─────────────────────────────────────────────────────

    async def compare_single(
        self,
        query: str,
        session_id: Optional[str] = None,
        turn_number: int = 1,
    ) -> TurnComparison:
        """
        Run a single query through both pipelines and compare.

        Creates independent PipelineState objects for each
        pipeline so they don't interfere with each other.

        Args:
            query: The user question to evaluate.
            session_id: Session ID for context pipeline.
                Generated if not provided.
            turn_number: Which turn this is in the conversation.

        Returns:
            TurnComparison with results from both pipelines.
        """
        sid = session_id or str(uuid.uuid4())

        # ── Run basic pipeline ──

        basic_state = PipelineState(
            query=query,
            session_id=f"{sid}_basic",
            turn_number=turn_number,
        )
        basic_state = await self._basic.query(basic_state)
        basic_result = self._state_to_turn_result(basic_state)

        # ── Run context pipeline ──

        context_state = PipelineState(
            query=query,
            session_id=f"{sid}_context",
            turn_number=turn_number,
        )
        context_state = await self._context.query(context_state)
        context_result = self._state_to_turn_result(context_state)

        # ── Build comparison ──

        comparison = TurnComparison(
            turn_number=turn_number,
            query=query,
            basic=basic_result,
            context=context_result,
        )

        logger.info(
            "Single comparison done (turn %d): basic=%.0fms, context=%.0fms",
            turn_number,
            basic_result.retrieval_time_ms + basic_result.generation_time_ms,
            context_result.retrieval_time_ms + context_result.generation_time_ms,
        )

        return comparison

    # ─────────────────────────────────────────────────────
    # Multi-turn scenario evaluation
    # ─────────────────────────────────────────────────────

    async def run_scenario(
        self,
        queries: Optional[list[str]] = None,
        scenario_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ScenarioReport:
        """
        Run a full multi-turn scenario through both pipelines.

        Executes each query sequentially, advancing the turn
        counter. The basic pipeline treats each turn independently.
        The context pipeline accumulates state across turns.

        If no queries are provided, runs the predefined RCA scenario.

        Args:
            queries: List of user questions in conversation order.
                Defaults to RCA_SCENARIO queries.
            scenario_name: Human-readable name for the report.
            session_id: Session ID prefix. Generated if not provided.

        Returns:
            ScenarioReport with per-turn comparisons and summary.
        """
        # Use defaults if not provided
        scenario = queries or RCA_SCENARIO["queries"]
        name = scenario_name or RCA_SCENARIO["name"]
        sid = session_id or str(uuid.uuid4())

        logger.info(
            "Starting scenario evaluation: '%s' (%d turns)",
            name,
            len(scenario),
        )

        start = time.perf_counter()
        turns: list[TurnComparison] = []

        # ── Run each turn sequentially ──
        #
        # Important: the context pipeline uses the same session ID
        # across all turns so context accumulates. The basic pipeline
        # gets a different session ID per turn (or the same — doesn't
        # matter since it's stateless).

        for i, query in enumerate(scenario, start=1):
            logger.info("Running turn %d/%d: '%s'", i, len(scenario), query[:60])

            # Basic pipeline — fresh state each turn
            basic_state = PipelineState(
                query=query,
                session_id=f"{sid}_basic",
                turn_number=i,
            )
            basic_state = await self._basic.query(basic_state)

            # Context pipeline — same session, advancing turns
            context_state = PipelineState(
                query=query,
                session_id=f"{sid}_context",
                turn_number=i,
            )
            context_state = await self._context.query(context_state)

            # Build comparison for this turn
            comparison = TurnComparison(
                turn_number=i,
                query=query,
                basic=self._state_to_turn_result(basic_state),
                context=self._state_to_turn_result(context_state),
            )
            turns.append(comparison)

        total_ms = (time.perf_counter() - start) * 1000

        # ── Build summary report ──

        report = ScenarioReport(
            scenario_name=name,
            session_id=sid,
            turns=turns,
            total_time_ms=total_ms,
            basic_total_retrieval_ms=sum(
                t.basic.retrieval_time_ms for t in turns
            ),
            basic_total_generation_ms=sum(
                t.basic.generation_time_ms for t in turns
            ),
            context_total_retrieval_ms=sum(
                t.context.retrieval_time_ms for t in turns
            ),
            context_total_generation_ms=sum(
                t.context.generation_time_ms for t in turns
            ),
        )

        # Generate the summary analysis
        report.summary = self._generate_summary(report)

        logger.info(
            "Scenario evaluation completed: '%s' — %d turns in %.1fs",
            name,
            len(turns),
            total_ms / 1000,
        )

        return report

    async def run_rca_scenario(
        self,
        session_id: Optional[str] = None,
    ) -> ScenarioReport:
        """
        Convenience method to run the predefined RCA scenario.

        This is the main demo entry point — runs the 6-turn
        bad-batch investigation through both pipelines.

        Args:
            session_id: Optional session ID prefix.

        Returns:
            ScenarioReport for the RCA scenario.
        """
        return await self.run_scenario(
            queries=RCA_SCENARIO["queries"],
            scenario_name=RCA_SCENARIO["name"],
            session_id=session_id,
        )

    # ─────────────────────────────────────────────────────
    # Report generation
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _generate_summary(report: ScenarioReport) -> str:
        """
        Generate a human-readable summary of the evaluation.

        Highlights key differences between the two pipelines
        and identifies which turns showed the biggest gap.

        Args:
            report: The completed scenario report.

        Returns:
            Multi-line summary string.
        """
        lines = [
            f"Scenario: {report.scenario_name}",
            f"Total turns: {len(report.turns)}",
            f"Total time: {report.total_time_ms / 1000:.1f}s",
            "",
            "Timing comparison:",
            f"  Basic   — retrieval: {report.basic_total_retrieval_ms:.0f}ms, "
            f"generation: {report.basic_total_generation_ms:.0f}ms",
            f"  Context — retrieval: {report.context_total_retrieval_ms:.0f}ms, "
            f"generation: {report.context_total_generation_ms:.0f}ms",
            "",
            "Per-turn analysis:",
        ]

        for turn in report.turns:
            # Determine if context had an advantage this turn
            basic_entities = len(turn.basic.extracted_entities)
            context_entities = len(turn.context.extracted_entities)
            had_augmentation = (
                turn.context.augmented_query
                and turn.context.augmented_query != turn.query
            )

            advantage = "neutral"
            if had_augmentation:
                advantage = "context advantage (query augmented)"
            elif context_entities > basic_entities:
                advantage = "context advantage (more entities)"

            lines.append(
                f"  Turn {turn.turn_number}: {advantage}"
            )
            if had_augmentation:
                lines.append(
                    f"    Augmented: '{turn.context.augmented_query}'"
                )

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _state_to_turn_result(state: PipelineState) -> TurnResult:
        """
        Convert a PipelineState to a TurnResult for reporting.

        Extracts only the fields needed for comparison,
        keeping the report objects clean and serializable.

        Args:
            state: Completed pipeline state.

        Returns:
            TurnResult with response and metadata.
        """
        return TurnResult(
            turn_number=state.turn_number,
            query=state.query,
            augmented_query=state.augmented_query,
            response=state.response,
            extracted_entities=state.extracted_entities,
            sources=state.sources,
            retrieval_time_ms=state.retrieval_time_ms,
            generation_time_ms=state.generation_time_ms,
        )