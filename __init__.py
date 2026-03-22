# ─────────────────────────────────────────────────────────
# domain/__init__.py
# ─────────────────────────────────────────────────────────
"""Domain layer — enums, models, and core types."""

from domain.enums import (
    AlertSeverity,
    ContextEdgeType,
    EntityType,
    MaintenanceType,
    RAGMode,
    RelationType,
)
from domain.models import (
    ContextEdge,
    ContextState,
    DefectReport,
    Entity,
    MaintenanceRecord,
    PipelineState,
    Relationship,
    SensorReading,
    SubGraph,
)

__all__ = [
    "AlertSeverity",
    "ContextEdge",
    "ContextEdgeType",
    "ContextState",
    "DefectReport",
    "Entity",
    "EntityType",
    "MaintenanceRecord",
    "MaintenanceType",
    "PipelineState",
    "RAGMode",
    "Relationship",
    "RelationType",
    "SensorReading",
    "SubGraph",
]


# ─────────────────────────────────────────────────────────
# data/__init__.py
# ─────────────────────────────────────────────────────────
"""Data layer — synthetic manufacturing data generation."""

from data.synthetic_factory import SyntheticFactory

__all__ = ["SyntheticFactory"]


# ─────────────────────────────────────────────────────────
# repositories/__init__.py
# ─────────────────────────────────────────────────────────
"""Repository layer — graph, context, and LLM operations."""

from repositories.context_repo import ContextRepository
from repositories.graph_repo import GraphRepository
from repositories.llm_repo import LLMRepository

__all__ = [
    "ContextRepository",
    "GraphRepository",
    "LLMRepository",
]


# ─────────────────────────────────────────────────────────
# services/__init__.py
# ─────────────────────────────────────────────────────────
"""Service layer — RAG pipelines, factory, and evaluation."""

from services.rag_factory import BaseRAGService, RAGFactory

__all__ = [
    "BaseRAGService",
    "RAGFactory",
]


# ─────────────────────────────────────────────────────────
# api/__init__.py
# ─────────────────────────────────────────────────────────
"""API layer — FastAPI routes and WebSocket handlers."""


# ─────────────────────────────────────────────────────────
# tests/__init__.py
# ─────────────────────────────────────────────────────────
"""Test suite."""