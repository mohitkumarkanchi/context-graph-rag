"""
Domain models for the Graph RAG demo.
Pydantic models — validation, serialization, and FastAPI compatibility.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field

from domain.enums import (
    AlertSeverity,
    ContextEdgeType,
    EntityType,
    MaintenanceType,
    RelationType,
)


# ─────────────────────────────────────────────────────────
# Knowledge graph primitives
# ─────────────────────────────────────────────────────────

class Entity(BaseModel):
    """A node in the knowledge graph."""
    id: str
    name: str
    entity_type: EntityType
    properties: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}

    def __hash__(self) -> int:
        return hash(self.id)


class Relationship(BaseModel):
    """An edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def key(self) -> str:
        return f"{self.source_id}--{self.relation_type.value}-->{self.target_id}"


class SubGraph(BaseModel):
    """
    A slice of the knowledge graph returned by retrieval.
    Contains the entities and relationships relevant to a query.
    """
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    @computed_field
    @property
    def entity_ids(self) -> set[str]:
        return {e.id for e in self.entities}

    def to_context_string(self) -> str:
        """Serialize subgraph into a text block for LLM context."""
        lines = []
        for e in self.entities:
            props = ", ".join(f"{k}: {v}" for k, v in e.properties.items())
            lines.append(f"[{e.entity_type.value}] {e.name} ({props})")
        for r in self.relationships:
            lines.append(
                f"  {r.source_id} --{r.relation_type.value}--> {r.target_id}"
            )
        return "\n".join(lines)

    def merge(self, other: "SubGraph") -> "SubGraph":
        """Combine two subgraphs, deduplicating by id/key."""
        seen_entities = {e.id for e in self.entities}
        seen_rels = {r.key for r in self.relationships}

        merged_entities = list(self.entities)
        merged_rels = list(self.relationships)

        for e in other.entities:
            if e.id not in seen_entities:
                merged_entities.append(e)
                seen_entities.add(e.id)

        for r in other.relationships:
            if r.key not in seen_rels:
                merged_rels.append(r)
                seen_rels.add(r.key)

        return SubGraph(entities=merged_entities, relationships=merged_rels)


# ─────────────────────────────────────────────────────────
# Context graph (session state)
# ─────────────────────────────────────────────────────────

class ContextEdge(BaseModel):
    """An edge in the session context graph."""
    source_id: str
    target_id: str
    edge_type: ContextEdgeType
    turn_number: int
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextState(BaseModel):
    """
    The accumulated session context graph.
    Grows with each conversation turn.
    """
    session_id: str
    entity_ids: list[str] = Field(default_factory=list)
    edges: list[ContextEdge] = Field(default_factory=list)
    turn_count: int = 0
    resolved_references: dict[str, str] = Field(default_factory=dict)

    def add_entity(self, entity_id: str) -> None:
        if entity_id not in self.entity_ids:
            self.entity_ids.append(entity_id)

    def add_edge(self, edge: ContextEdge) -> None:
        self.edges.append(edge)

    def resolve_reference(self, pronoun: str, entity_id: str) -> None:
        """Track coreference resolution: 'it' → 'machine_m400'."""
        self.resolved_references[pronoun.lower()] = entity_id

    def get_recent_entities(self, last_n_turns: int = 3) -> list[str]:
        """Get entity IDs discussed in the last N turns."""
        if not self.edges:
            return list(self.entity_ids)
        min_turn = max(0, self.turn_count - last_n_turns)
        recent_ids = set()
        for edge in self.edges:
            if edge.turn_number >= min_turn:
                recent_ids.add(edge.source_id)
                recent_ids.add(edge.target_id)
        return list(recent_ids & set(self.entity_ids))

    def get_investigation_chain(self) -> list[ContextEdge]:
        """Get the RCA causal chain built so far."""
        return [
            e for e in self.edges
            if e.edge_type in (
                ContextEdgeType.SUSPECTED_CAUSE,
                ContextEdgeType.RULED_OUT,
            )
        ]

    def to_context_string(self) -> str:
        """Serialize context state for LLM prompt."""
        lines = [f"Session: {self.session_id} | Turn: {self.turn_count}"]

        if self.entity_ids:
            lines.append(f"Entities discussed: {', '.join(self.entity_ids)}")

        if self.resolved_references:
            refs = ", ".join(
                f'"{k}" → {v}'
                for k, v in self.resolved_references.items()
            )
            lines.append(f"Resolved references: {refs}")

        chain = self.get_investigation_chain()
        if chain:
            lines.append("Investigation chain:")
            for e in chain:
                status = (
                    "suspected"
                    if e.edge_type == ContextEdgeType.SUSPECTED_CAUSE
                    else "ruled out"
                )
                lines.append(f"  {e.source_id} → {e.target_id} ({status})")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# Manufacturing domain models
# ─────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    """A single sensor measurement."""
    sensor_id: str
    machine_id: str
    metric: str
    value: float
    unit: str
    timestamp: datetime
    is_anomaly: bool = False


class MaintenanceRecord(BaseModel):
    """A maintenance event with full details."""
    event_id: str
    machine_id: str
    technician_id: str
    maintenance_type: MaintenanceType
    date: datetime
    description: str
    parts_replaced: list[str] = Field(default_factory=list)


class DefectReport(BaseModel):
    """A quality defect detected on the line."""
    defect_id: str
    machine_id: str
    description: str
    severity: AlertSeverity
    detected_at: datetime
    batch_affected: Optional[str] = None


# ─────────────────────────────────────────────────────────
# LangGraph pipeline state
# ─────────────────────────────────────────────────────────

class PipelineState(BaseModel):
    """
    State object passed through LangGraph nodes.
    Both basic and context pipelines use this,
    but context pipeline populates context_state.
    """
    query: str
    session_id: str
    turn_number: int = 0

    # Extraction
    extracted_entities: list[str] = Field(default_factory=list)

    # Retrieval
    retrieved_subgraph: Optional[SubGraph] = None

    # Context (only used by context pipeline)
    context_state: Optional[ContextState] = None
    augmented_query: Optional[str] = None

    # Generation
    response: str = ""
    sources: list[str] = Field(default_factory=list)

    # Evaluation metadata
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0