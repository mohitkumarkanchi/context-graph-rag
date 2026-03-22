"""
Context graph repository.

Manages per-session context graphs that accumulate state
across conversation turns. This is the key differentiator
between basic Graph RAG (stateless) and context Graph RAG
(stateful).

Each session maintains its own ContextState, which tracks:
    - Entities the user has discussed
    - Coreference resolutions ("it" → machine_m400)
    - RCA investigation chain (suspected / ruled out)
    - Turn-by-turn edge history for recency weighting
"""

import logging
from typing import Optional

from domain.enums import ContextEdgeType
from domain.models import ContextEdge, ContextState

logger = logging.getLogger(__name__)


class ContextRepository:
    """
    In-memory store for session context graphs.

    One ContextState per session_id. Each state grows with
    every conversation turn — entities get added, coreferences
    get resolved, and the RCA investigation chain extends.

    Thread safety note: This implementation is not thread-safe.
    For production, wrap mutations in a lock or use an async-safe
    store. Fine for a single-user demo.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ContextState] = {}

    # ─────────────────────────────────────────────────────
    # Session lifecycle
    # ─────────────────────────────────────────────────────

    def get_or_create(self, session_id: str) -> ContextState:
        """
        Retrieve an existing session context or create a new one.

        This is the main entry point — called at the start of
        every turn in the context RAG pipeline.

        Args:
            session_id: Unique session identifier.

        Returns:
            The session's ContextState (new or existing).
        """
        if session_id not in self._sessions:
            logger.info("Creating new context session: %s", session_id)
            self._sessions[session_id] = ContextState(
                session_id=session_id,
            )
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[ContextState]:
        """
        Retrieve a session context without creating one.

        Args:
            session_id: Unique session identifier.

        Returns:
            The ContextState if it exists, None otherwise.
        """
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """
        Remove a session context entirely.

        Used when a user starts a fresh investigation or
        the session expires.

        Args:
            session_id: Session to delete.

        Returns:
            True if the session existed and was deleted.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Deleted context session: %s", session_id)
            return True
        return False

    def list_sessions(self) -> list[str]:
        """
        List all active session IDs.

        Returns:
            List of session ID strings.
        """
        return list(self._sessions.keys())

    # ─────────────────────────────────────────────────────
    # Entity tracking
    # ─────────────────────────────────────────────────────

    def add_discussed_entities(
        self,
        session_id: str,
        entity_ids: list[str],
        turn_number: int,
    ) -> ContextState:
        """
        Record that entities were discussed in this turn.

        Creates a DISCUSSED edge from each entity to a virtual
        "turn_N" node, and adds the entity to the session's
        tracked entity list.

        Args:
            session_id: The active session.
            entity_ids: IDs of entities mentioned or retrieved.
            turn_number: Current conversation turn number.

        Returns:
            Updated ContextState.
        """
        state = self.get_or_create(session_id)

        for entity_id in entity_ids:
            # Track the entity
            state.add_entity(entity_id)

            # Create a DISCUSSED edge with turn metadata
            edge = ContextEdge(
                source_id=entity_id,
                target_id=f"turn_{turn_number}",
                edge_type=ContextEdgeType.DISCUSSED,
                turn_number=turn_number,
            )
            state.add_edge(edge)

        # Advance the turn counter
        state.turn_count = max(state.turn_count, turn_number)

        logger.debug(
            "Session %s turn %d: added %d discussed entities",
            session_id,
            turn_number,
            len(entity_ids),
        )

        return state

    # ─────────────────────────────────────────────────────
    # Coreference resolution
    # ─────────────────────────────────────────────────────

    def resolve_reference(
        self,
        session_id: str,
        pronoun: str,
        entity_id: str,
        turn_number: int,
    ) -> ContextState:
        """
        Record a coreference resolution.

        When the user says "it", "that machine", "those parts",
        etc., the pipeline resolves the pronoun to a specific
        entity. This method records both the resolution mapping
        and a RESOLVED_TO edge in the context graph.

        Args:
            session_id: The active session.
            pronoun: The ambiguous reference (e.g. "it", "that").
            entity_id: The resolved entity ID.
            turn_number: Current conversation turn.

        Returns:
            Updated ContextState.
        """
        state = self.get_or_create(session_id)

        # Store the resolution mapping for future lookups
        state.resolve_reference(pronoun, entity_id)

        # Create a RESOLVED_TO edge for graph traceability
        edge = ContextEdge(
            source_id=f"ref_{pronoun.lower()}",
            target_id=entity_id,
            edge_type=ContextEdgeType.RESOLVED_TO,
            turn_number=turn_number,
            metadata={"original_text": pronoun},
        )
        state.add_edge(edge)

        logger.debug(
            "Session %s: resolved '%s' → %s",
            session_id,
            pronoun,
            entity_id,
        )

        return state

    # ─────────────────────────────────────────────────────
    # RCA investigation tracking
    # ─────────────────────────────────────────────────────

    def mark_suspected_cause(
        self,
        session_id: str,
        source_id: str,
        target_id: str,
        turn_number: int,
        confidence: float = 1.0,
    ) -> ContextState:
        """
        Mark an entity as a suspected root cause.

        Adds a SUSPECTED_CAUSE edge from the symptom/effect
        to the suspected cause. This builds the investigation
        chain that the context RAG uses to prioritize retrieval.

        Example chain over turns:
            defect_001 → machine_m400 (T2)
            machine_m400 → maint_jan15 (T3)
            maint_jan15 → batch_b442 (T4)

        Args:
            source_id: The symptom or downstream entity.
            target_id: The suspected upstream cause.
            turn_number: Current conversation turn.
            confidence: How confident the suspicion is (0.0–1.0).

        Returns:
            Updated ContextState.
        """
        state = self.get_or_create(session_id)

        edge = ContextEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=ContextEdgeType.SUSPECTED_CAUSE,
            turn_number=turn_number,
            confidence=confidence,
        )
        state.add_edge(edge)

        # Ensure both entities are tracked
        state.add_entity(source_id)
        state.add_entity(target_id)

        logger.debug(
            "Session %s: suspected cause %s → %s (conf: %.2f)",
            session_id,
            source_id,
            target_id,
            confidence,
        )

        return state

    def rule_out_cause(
        self,
        session_id: str,
        source_id: str,
        target_id: str,
        turn_number: int,
        reason: str = "",
    ) -> ContextState:
        """
        Mark an entity as ruled out from the investigation.

        Adds a RULED_OUT edge, which tells the retrieval
        pipeline to deprioritize this path in future turns.

        Args:
            source_id: The symptom entity.
            target_id: The entity being ruled out.
            turn_number: Current conversation turn.
            reason: Optional explanation for ruling it out.

        Returns:
            Updated ContextState.
        """
        state = self.get_or_create(session_id)

        edge = ContextEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=ContextEdgeType.RULED_OUT,
            turn_number=turn_number,
            metadata={"reason": reason} if reason else {},
        )
        state.add_edge(edge)

        logger.debug(
            "Session %s: ruled out %s → %s",
            session_id,
            source_id,
            target_id,
        )

        return state

    def mark_follow_up(
        self,
        session_id: str,
        entity_id: str,
        turn_number: int,
        reason: str = "",
    ) -> ContextState:
        """
        Flag an entity for follow-up investigation.

        Used when the pipeline discovers something worth
        investigating but hasn't been asked about yet.
        For example: "Batch B-442 also went to L-200 and
        G-150 — these may need inspection."

        Args:
            entity_id: Entity to flag for follow-up.
            turn_number: Current conversation turn.
            reason: Why this entity needs follow-up.

        Returns:
            Updated ContextState.
        """
        state = self.get_or_create(session_id)

        edge = ContextEdge(
            source_id=entity_id,
            target_id=f"follow_up_t{turn_number}",
            edge_type=ContextEdgeType.FOLLOW_UP,
            turn_number=turn_number,
            metadata={"reason": reason} if reason else {},
        )
        state.add_edge(edge)
        state.add_entity(entity_id)

        logger.debug(
            "Session %s: flagged %s for follow-up",
            session_id,
            entity_id,
        )

        return state

    # ─────────────────────────────────────────────────────
    # Query helpers
    # ─────────────────────────────────────────────────────

    def get_recent_entity_ids(
        self,
        session_id: str,
        last_n_turns: int = 3,
    ) -> list[str]:
        """
        Get entity IDs discussed in the most recent N turns.

        Used by the context RAG pipeline to weight retrieval
        toward recently discussed entities.

        Args:
            session_id: The active session.
            last_n_turns: How many turns back to look.

        Returns:
            List of entity IDs (may be empty if session
            doesn't exist).
        """
        state = self.get(session_id)
        if not state:
            return []
        return state.get_recent_entities(last_n_turns)

    def get_resolved_references(
        self,
        session_id: str,
    ) -> dict[str, str]:
        """
        Get all coreference resolutions for a session.

        Returns a mapping of pronoun → entity_id, e.g.:
            {"it": "machine_m400", "that batch": "batch_b442"}

        Args:
            session_id: The active session.

        Returns:
            Dict of pronoun → entity_id mappings.
        """
        state = self.get(session_id)
        if not state:
            return {}
        return dict(state.resolved_references)

    def get_investigation_summary(self, session_id: str) -> dict:
        """
        Get a summary of the current RCA investigation state.

        Returns a structured dict suitable for including
        in the LLM prompt or returning via the API.

        Args:
            session_id: The active session.

        Returns:
            Dict with suspected causes, ruled out entities,
            follow-ups, and the full context string.
        """
        state = self.get(session_id)
        if not state:
            return {
                "session_id": session_id,
                "status": "no_session",
                "suspected": [],
                "ruled_out": [],
                "follow_ups": [],
                "context_string": "",
            }

        suspected = [
            {"source": e.source_id, "target": e.target_id, "turn": e.turn_number}
            for e in state.edges
            if e.edge_type == ContextEdgeType.SUSPECTED_CAUSE
        ]
        ruled_out = [
            {
                "source": e.source_id,
                "target": e.target_id,
                "turn": e.turn_number,
                "reason": e.metadata.get("reason", ""),
            }
            for e in state.edges
            if e.edge_type == ContextEdgeType.RULED_OUT
        ]
        follow_ups = [
            {
                "entity": e.source_id,
                "turn": e.turn_number,
                "reason": e.metadata.get("reason", ""),
            }
            for e in state.edges
            if e.edge_type == ContextEdgeType.FOLLOW_UP
        ]

        return {
            "session_id": session_id,
            "status": "active",
            "turn_count": state.turn_count,
            "entities_tracked": len(state.entity_ids),
            "suspected": suspected,
            "ruled_out": ruled_out,
            "follow_ups": follow_ups,
            "context_string": state.to_context_string(),
        }

    # ─────────────────────────────────────────────────────
    # Serialization (for frontend graph viz)
    # ─────────────────────────────────────────────────────

    def to_json_graph(self, session_id: str) -> dict:
        """
        Convert session context graph to D3-compatible JSON.

        Output format matches GraphRepository.to_json_graph()
        so the frontend can render both graphs with the same
        visualization code.

        Args:
            session_id: The session to serialize.

        Returns:
            Dict with "nodes" and "links" lists.
            Empty graph if session doesn't exist.
        """
        state = self.get(session_id)
        if not state:
            return {"nodes": [], "links": []}

        # Build node list from tracked entity IDs
        nodes = [
            {"id": eid, "type": "context_entity"}
            for eid in state.entity_ids
        ]

        # Add virtual turn nodes for edges that reference them
        turn_nodes = {
            e.target_id
            for e in state.edges
            if e.target_id.startswith("turn_")
            or e.target_id.startswith("follow_up_")
        }
        for tid in turn_nodes:
            nodes.append({"id": tid, "type": "turn_marker"})

        # Add reference nodes
        ref_nodes = {
            e.source_id
            for e in state.edges
            if e.source_id.startswith("ref_")
        }
        for rid in ref_nodes:
            nodes.append({"id": rid, "type": "reference"})

        # Build link list
        links = [
            {
                "source": e.source_id,
                "target": e.target_id,
                "type": e.edge_type.value,
                "turn": e.turn_number,
                "confidence": e.confidence,
            }
            for e in state.edges
        ]

        return {
            "session_id": session_id,
            "turn_count": state.turn_count,
            "nodes": nodes,
            "links": links,
        }