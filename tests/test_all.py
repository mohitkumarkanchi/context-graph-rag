"""
All-in-one test script.

Tests each layer bottom-up: domain → data → repositories →
services → API. Run with:

    uv run pytest tests/test_all.py -v
    uv run python tests/test_all.py       # standalone mode
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from httpx import ASGITransport, AsyncClient

from config import get_settings
from data.synthetic_factory import SyntheticFactory
from domain.enums import EntityType, RAGMode, RelationType, ContextEdgeType
from domain.models import Entity, Relationship, SubGraph, ContextState, PipelineState
from repositories.graph_repo import GraphRepository
from repositories.context_repo import ContextRepository


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    """Generate synthetic data once for all tests."""
    factory = SyntheticFactory()
    return factory.build()


@pytest.fixture(scope="module")
def graph_repo(synthetic_data):
    """Loaded graph repo shared across tests."""
    entities, relationships = synthetic_data
    repo = GraphRepository()
    repo.load(entities, relationships)
    return repo


@pytest.fixture
def context_repo():
    """Fresh context repo per test."""
    return ContextRepository()


# ─────────────────────────────────────────────────────────
# Domain layer
# ─────────────────────────────────────────────────────────

class TestDomain:
    """Basic domain model sanity checks."""

    def test_entity_creation(self):
        e = Entity(id="m1", name="Mill", entity_type=EntityType.MACHINE)
        assert e.id == "m1"
        assert e.entity_type == EntityType.MACHINE

    def test_entity_hashable(self):
        e = Entity(id="m1", name="Mill", entity_type=EntityType.MACHINE)
        assert {e, e} == {e}

    def test_relationship_key(self):
        r = Relationship(
            source_id="a",
            target_id="b",
            relation_type=RelationType.LOCATED_IN,
        )
        assert r.key == "a--located_in-->b"

    def test_subgraph_merge_deduplicates(self):
        e1 = Entity(id="a", name="A", entity_type=EntityType.MACHINE)
        e2 = Entity(id="b", name="B", entity_type=EntityType.SENSOR)
        sg1 = SubGraph(entities=[e1])
        sg2 = SubGraph(entities=[e1, e2])
        merged = sg1.merge(sg2)
        assert len(merged.entities) == 2

    def test_pipeline_state_defaults(self):
        state = PipelineState(query="test", session_id="s1")
        assert state.turn_number == 0
        assert state.response == ""
        assert state.extracted_entities == []


# ─────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────

class TestSyntheticData:
    """Verify the generated manufacturing data is complete."""

    def test_entity_count(self, synthetic_data):
        entities, _ = synthetic_data
        assert len(entities) > 40

    def test_relationship_count(self, synthetic_data):
        _, relationships = synthetic_data
        assert len(relationships) > 60

    def test_has_key_machines(self, synthetic_data):
        entities, _ = synthetic_data
        ids = {e.id for e in entities}
        assert "machine_m400" in ids
        assert "machine_l200" in ids
        assert "machine_g150" in ids

    def test_has_bad_batch(self, synthetic_data):
        entities, _ = synthetic_data
        ids = {e.id for e in entities}
        assert "batch_b442" in ids

    def test_has_defect(self, synthetic_data):
        entities, _ = synthetic_data
        defects = [e for e in entities if e.entity_type == EntityType.DEFECT]
        assert len(defects) >= 1

    def test_has_rca_chain_relationships(self, synthetic_data):
        _, rels = synthetic_data
        rel_types = {r.relation_type for r in rels}
        assert RelationType.TRIGGERED_ALERT in rel_types
        assert RelationType.PERFORMED_ON in rel_types
        assert RelationType.FROM_BATCH in rel_types
        assert RelationType.INSTALLED_IN in rel_types


# ─────────────────────────────────────────────────────────
# Graph repository
# ─────────────────────────────────────────────────────────

class TestGraphRepo:
    """Test knowledge graph operations."""

    def test_node_count(self, graph_repo):
        assert graph_repo.node_count > 40

    def test_edge_count(self, graph_repo):
        assert graph_repo.edge_count > 60

    def test_get_entity_by_id(self, graph_repo):
        e = graph_repo.get_entity("machine_m400")
        assert e is not None
        assert e.name == "CNC Mill M-400"

    def test_get_entity_by_name(self, graph_repo):
        e = graph_repo.get_entity_by_name("M-400")
        assert e is not None
        assert e.id == "machine_m400"

    def test_search_entities(self, graph_repo):
        results = graph_repo.search_entities("vibration sensor")
        assert len(results) > 0
        assert any("vibration" in e.name.lower() for e in results)

    def test_get_entities_by_type(self, graph_repo):
        machines = graph_repo.get_entities_by_type(EntityType.MACHINE)
        assert len(machines) >= 7

    def test_get_neighbors(self, graph_repo):
        neighbors = graph_repo.get_neighbors("machine_m400")
        assert len(neighbors) > 0
        neighbor_types = {e.entity_type for e, _ in neighbors}
        assert EntityType.SENSOR in neighbor_types or EntityType.PLC in neighbor_types

    def test_get_subgraph(self, graph_repo):
        sg = graph_repo.get_subgraph(seed_ids=["machine_m400"], max_hops=1)
        assert len(sg.entities) > 1
        assert len(sg.relationships) > 0

    def test_subgraph_respects_max_entities(self, graph_repo):
        sg = graph_repo.get_subgraph(
            seed_ids=["machine_m400"],
            max_hops=3,
            max_entities=5,
        )
        assert len(sg.entities) <= 5

    def test_trace_supply_chain(self, graph_repo):
        sg = graph_repo.trace_supply_chain("part_bearing_m400")
        entity_ids = {e.id for e in sg.entities}
        # Should find the part, batch, supplier, and sibling parts
        assert "batch_b442" in entity_ids
        assert "supplier_precision" in entity_ids

    def test_trace_supply_chain_finds_siblings(self, graph_repo):
        sg = graph_repo.trace_supply_chain("part_bearing_m400")
        entity_ids = {e.id for e in sg.entities}
        # Sibling parts from same batch
        assert "part_bearing_l200" in entity_ids
        assert "part_bearing_g150" in entity_ids

    def test_trace_machine_rca(self, graph_repo):
        sg = graph_repo.trace_machine_to_root_cause("machine_m400")
        entity_ids = {e.id for e in sg.entities}
        assert "machine_m400" in entity_ids
        assert "batch_b442" in entity_ids
        assert len(sg.entities) > 5

    def test_to_json_graph(self, graph_repo):
        data = graph_repo.to_json_graph()
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) == graph_repo.node_count


# ─────────────────────────────────────────────────────────
# Context repository
# ─────────────────────────────────────────────────────────

class TestContextRepo:
    """Test session context graph operations."""

    def test_create_session(self, context_repo):
        state = context_repo.get_or_create("sess_1")
        assert state.session_id == "sess_1"
        assert state.turn_count == 0

    def test_add_discussed_entities(self, context_repo):
        context_repo.add_discussed_entities("s1", ["machine_m400"], turn_number=1)
        state = context_repo.get("s1")
        assert "machine_m400" in state.entity_ids
        assert state.turn_count == 1

    def test_resolve_reference(self, context_repo):
        context_repo.get_or_create("s2")
        context_repo.resolve_reference("s2", "it", "machine_m400", turn_number=2)
        refs = context_repo.get_resolved_references("s2")
        assert refs["it"] == "machine_m400"

    def test_mark_suspected_cause(self, context_repo):
        context_repo.get_or_create("s3")
        context_repo.mark_suspected_cause(
            "s3", "defect_001", "machine_m400", turn_number=1,
        )
        summary = context_repo.get_investigation_summary("s3")
        assert len(summary["suspected"]) == 1

    def test_rule_out_cause(self, context_repo):
        context_repo.get_or_create("s4")
        context_repo.rule_out_cause(
            "s4", "defect_001", "machine_m200", turn_number=1, reason="no anomalies",
        )
        summary = context_repo.get_investigation_summary("s4")
        assert len(summary["ruled_out"]) == 1

    def test_recent_entities(self, context_repo):
        context_repo.add_discussed_entities("s5", ["a", "b"], turn_number=1)
        context_repo.add_discussed_entities("s5", ["c"], turn_number=5)
        recent = context_repo.get_recent_entity_ids("s5", last_n_turns=2)
        assert "c" in recent

    def test_delete_session(self, context_repo):
        context_repo.get_or_create("s6")
        assert context_repo.delete("s6") is True
        assert context_repo.get("s6") is None

    def test_to_json_graph(self, context_repo):
        context_repo.add_discussed_entities("s7", ["machine_m400"], turn_number=1)
        data = context_repo.to_json_graph("s7")
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) > 0


# ─────────────────────────────────────────────────────────
# API (no LLM required — tests structure only)
# ─────────────────────────────────────────────────────────

class TestAPI:
    """
    Test API endpoints without LLM dependency.

    Uses the FastAPI test client to verify that routes
    exist, validate requests, and return correct shapes.
    The /health and /graph endpoints work without Ollama.
    """

    @pytest.fixture(autouse=True)
    async def setup_app(self, graph_repo):
        """Set up a test app with graph loaded but no LLM."""
        from main import app
        from api import router as api_router
        from repositories.context_repo import ContextRepository
        from services.rag_factory import RAGFactory

        context_repo = ContextRepository()

        # Inject repos without LLM (chat will fail but graph won't)
        api_router.setup(
            factory=None,  # No factory — chat endpoints will 503
            graph_repo=graph_repo,
            context_repo=context_repo,
            llm_repo=None,
            eval_service=None,
        )
        # Override the module-level refs directly for graph endpoints
        api_router._graph_repo = graph_repo
        api_router._context_repo = context_repo

        transport = ASGITransport(app=app)
        self.client = AsyncClient(transport=transport, base_url="http://test")

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        resp = await self.client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["graph_node_count"] > 0

    @pytest.mark.asyncio
    async def test_knowledge_graph_endpoint(self):
        resp = await self.client.get("/api/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) > 0
        assert len(data["links"]) > 0

    @pytest.mark.asyncio
    async def test_subgraph_endpoint(self):
        resp = await self.client.post(
            "/api/graph/subgraph",
            json={"seed_ids": ["machine_m400"], "max_hops": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_count"] > 0

    @pytest.mark.asyncio
    async def test_chat_returns_503_without_factory(self):
        resp = await self.client.post(
            "/api/chat",
            json={"query": "test", "mode": "basic"},
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_context_graph_empty_session(self):
        resp = await self.client.get("/api/context/nonexistent/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []

    @pytest.mark.asyncio
    async def test_chat_validates_empty_query(self):
        resp = await self.client.post(
            "/api/chat",
            json={"query": "", "mode": "basic"},
        )
        assert resp.status_code == 422  # Pydantic validation error


# ─────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────

def run_standalone():
    """
    Run core tests without pytest (no LLM needed).

    Useful for quick verification after setup:
        uv run python tests/test_all.py
    """
    print("=" * 50)
    print("Context Graph RAG — Quick Test")
    print("=" * 50)

    # ── Synthetic data ──
    print("\n[1/4] Generating synthetic data...")
    factory = SyntheticFactory()
    entities, rels = factory.build()
    print(f"  ✓ {len(entities)} entities, {len(rels)} relationships")

    # ── Graph repo ──
    print("\n[2/4] Loading knowledge graph...")
    repo = GraphRepository()
    repo.load(entities, rels)
    print(f"  ✓ {repo.node_count} nodes, {repo.edge_count} edges")

    # ── Key lookups ──
    print("\n[3/4] Testing graph operations...")

    m400 = repo.get_entity("machine_m400")
    assert m400 is not None, "M-400 not found"
    print(f"  ✓ Found: {m400.name}")

    sg = repo.get_subgraph(seed_ids=["machine_m400"], max_hops=2)
    print(f"  ✓ Subgraph: {len(sg.entities)} entities, {len(sg.relationships)} rels")

    supply = repo.trace_supply_chain("part_bearing_m400")
    supply_ids = {e.id for e in supply.entities}
    assert "batch_b442" in supply_ids, "Bad batch not in supply chain"
    assert "part_bearing_l200" in supply_ids, "Sibling part not found"
    print(f"  ✓ Supply chain: found batch B-442 and sibling parts")

    rca = repo.trace_machine_to_root_cause("machine_m400")
    print(f"  ✓ RCA trace: {len(rca.entities)} entities in causal chain")

    # ── Context repo ──
    print("\n[4/4] Testing context repository...")
    ctx = ContextRepository()
    ctx.add_discussed_entities("test", ["machine_m400"], turn_number=1)
    ctx.resolve_reference("test", "it", "machine_m400", turn_number=2)
    ctx.mark_suspected_cause("test", "defect_001", "machine_m400", turn_number=2)

    refs = ctx.get_resolved_references("test")
    assert refs["it"] == "machine_m400"
    summary = ctx.get_investigation_summary("test")
    assert len(summary["suspected"]) == 1
    print(f"  ✓ Context: coreference resolved, RCA chain started")

    graph_json = ctx.to_json_graph("test")
    assert len(graph_json["nodes"]) > 0
    print(f"  ✓ Context graph JSON: {len(graph_json['nodes'])} nodes")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
    print("\nNext: start Ollama and run the server:")
    print("  ollama pull llama3.2")
    print("  uv run python main.py")


if __name__ == "__main__":
    run_standalone()