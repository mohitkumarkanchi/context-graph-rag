"""
Knowledge graph repository.

Wraps NetworkX with domain-aware operations for
entity lookup, traversal, and subgraph extraction.
Follows PEP 8 style conventions throughout.
"""

import logging
from typing import Optional

import networkx as nx

from config import get_settings
from domain.enums import EntityType, RelationType
from domain.models import Entity, Relationship, SubGraph

logger = logging.getLogger(__name__)


class GraphRepository:
    """
    Manages the static manufacturing knowledge graph.

    This is a read-only repository — after initial load via
    `load()`, the graph is not mutated. All retrieval methods
    return domain model objects (Entity, Relationship, SubGraph),
    keeping NetworkX as an internal implementation detail.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}
        self._settings = get_settings()

    # ─────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────

    def load(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> None:
        """
        Populate the graph from synthetic factory output.

        Each entity becomes a node keyed by `entity.id`.
        Each relationship becomes a directed edge with
        `relation_type` stored as edge data.

        Args:
            entities: List of domain Entity objects.
            relationships: List of domain Relationship objects.
        """
        for entity in entities:
            self._entities[entity.id] = entity
            self._graph.add_node(
                entity.id,
                name=entity.name,
                entity_type=entity.entity_type.value,
                **entity.properties,
            )

        for rel in relationships:
            self._graph.add_edge(
                rel.source_id,
                rel.target_id,
                relation_type=rel.relation_type.value,
                **rel.properties,
            )

        logger.info(
            "Graph loaded: %d nodes, %d edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    @property
    def node_count(self) -> int:
        """Return total number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Return total number of edges in the graph."""
        return self._graph.number_of_edges()

    # ─────────────────────────────────────────────────────
    # Entity lookup
    # ─────────────────────────────────────────────────────

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Retrieve a single entity by its exact ID.

        Args:
            entity_id: The unique identifier (e.g. "machine_m400").

        Returns:
            The Entity if found, None otherwise.
        """
        return self._entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """
        Find first entity whose name contains the query string.

        Performs case-insensitive substring matching.
        Returns the first match found — not deterministic
        if multiple entities share a substring.

        Args:
            name: Substring to search for (e.g. "M-400").

        Returns:
            Matching Entity or None.
        """
        name_lower = name.lower()
        for entity in self._entities.values():
            if name_lower in entity.name.lower():
                return entity
        return None

    def search_entities(self, query: str) -> list[Entity]:
        """
        Score-based search across entity names, types, and properties.

        Splits the query into terms and scores each entity
        by how many terms appear in its searchable text.
        Name matches receive a 2x boost over property matches.

        Args:
            query: Free-text search string (e.g. "CNC vibration line A").

        Returns:
            List of entities sorted by relevance score (descending).
        """
        terms = query.lower().split()
        scored_results: list[tuple[int, Entity]] = []

        for entity in self._entities.values():
            # Build a flat string of all searchable content
            searchable = (
                f"{entity.name} "
                f"{entity.entity_type.value} "
                f"{' '.join(str(v) for v in entity.properties.values())}"
            ).lower()

            score = 0
            for term in terms:
                if term in searchable:
                    score += 1
                # Boost direct name matches
                if term in entity.name.lower():
                    score += 2

            if score > 0:
                scored_results.append((score, entity))

        # Sort highest score first
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [entity for _, entity in scored_results]

    def get_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """
        Retrieve all entities of a given type.

        Args:
            entity_type: The EntityType enum value to filter by.

        Returns:
            List of matching entities (unordered).
        """
        return [
            e for e in self._entities.values()
            if e.entity_type == entity_type
        ]

    # ─────────────────────────────────────────────────────
    # Neighbor traversal
    # ─────────────────────────────────────────────────────

    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",
        relation_types: Optional[list[RelationType]] = None,
    ) -> list[tuple[Entity, Relationship]]:
        """
        Get immediate neighbors of an entity.

        Returns a list of (neighbor_entity, connecting_relationship)
        tuples. Can be filtered by direction and relationship type.

        Args:
            entity_id: The node to get neighbors for.
            direction: One of "out", "in", or "both".
            relation_types: Optional filter — only include edges
                whose type is in this list.

        Returns:
            List of (Entity, Relationship) tuples.
        """
        if entity_id not in self._graph:
            return []

        results: list[tuple[Entity, Relationship]] = []

        # Collect outgoing edges: entity_id → target
        if direction in ("out", "both"):
            for _, target, data in self._graph.out_edges(entity_id, data=True):
                rel = self._edge_data_to_relationship(
                    source_id=entity_id,
                    target_id=target,
                    data=data,
                )
                if relation_types and rel.relation_type not in relation_types:
                    continue
                target_entity = self._entities.get(target)
                if target_entity:
                    results.append((target_entity, rel))

        # Collect incoming edges: source → entity_id
        if direction in ("in", "both"):
            for source, _, data in self._graph.in_edges(entity_id, data=True):
                rel = self._edge_data_to_relationship(
                    source_id=source,
                    target_id=entity_id,
                    data=data,
                )
                if relation_types and rel.relation_type not in relation_types:
                    continue
                source_entity = self._entities.get(source)
                if source_entity:
                    results.append((source_entity, rel))

        return results

    # ─────────────────────────────────────────────────────
    # Subgraph extraction (core RAG retrieval)
    # ─────────────────────────────────────────────────────

    def get_subgraph(
        self,
        seed_ids: list[str],
        max_hops: Optional[int] = None,
        max_entities: Optional[int] = None,
    ) -> SubGraph:
        """
        Extract a subgraph via BFS from seed entities.

        This is the core retrieval operation used by both
        the basic and context RAG pipelines. Starting from
        one or more seed nodes, it traverses outgoing and
        incoming edges up to `max_hops` depth, collecting
        all discovered entities and relationships.

        Args:
            seed_ids: Starting node IDs for traversal.
            max_hops: Maximum BFS depth (defaults to config value).
            max_entities: Cap on total entities returned
                (defaults to config value).

        Returns:
            SubGraph containing discovered entities and
            deduplicated relationships.
        """
        hops = max_hops or self._settings.graph_hop_depth
        limit = max_entities or self._settings.max_subgraph_entities

        visited_nodes: set[str] = set()
        collected_entities: list[Entity] = []
        collected_rels: list[Relationship] = []

        # Initialize BFS frontier with valid seed nodes
        frontier = {
            sid for sid in seed_ids
            if sid in self._graph
        }

        for _ in range(hops):
            if not frontier:
                break

            next_frontier: set[str] = set()

            for node_id in frontier:
                if node_id in visited_nodes:
                    continue
                visited_nodes.add(node_id)

                # Collect the entity
                entity = self._entities.get(node_id)
                if entity:
                    collected_entities.append(entity)

                # Check entity cap
                if len(collected_entities) >= limit:
                    break

                # Traverse outgoing edges
                for _, target, data in self._graph.out_edges(node_id, data=True):
                    rel = self._edge_data_to_relationship(
                        source_id=node_id,
                        target_id=target,
                        data=data,
                    )
                    collected_rels.append(rel)
                    if target not in visited_nodes:
                        next_frontier.add(target)

                # Traverse incoming edges
                for source, _, data in self._graph.in_edges(node_id, data=True):
                    rel = self._edge_data_to_relationship(
                        source_id=source,
                        target_id=node_id,
                        data=data,
                    )
                    collected_rels.append(rel)
                    if source not in visited_nodes:
                        next_frontier.add(source)

            # Stop early if we hit the entity cap
            if len(collected_entities) >= limit:
                break

            frontier = next_frontier

        # Deduplicate relationships by their composite key
        unique_rels = self._deduplicate_relationships(collected_rels)

        logger.debug(
            "Subgraph extracted: %d entities, %d relationships from seeds %s",
            len(collected_entities),
            len(unique_rels),
            seed_ids,
        )

        return SubGraph(
            entities=collected_entities,
            relationships=unique_rels,
        )

    # ─────────────────────────────────────────────────────
    # RCA-specific traversals
    # ─────────────────────────────────────────────────────

    def trace_supply_chain(self, part_id: str) -> SubGraph:
        """
        Trace a part's supply chain and find sibling parts.

        From a given part, follows the chain:
            part → batch → supplier
        Then finds all other parts from the same batch.
        This is critical for the RCA scenario where a bad
        batch (B-442) affects multiple machines.

        Args:
            part_id: The part entity ID to trace from.

        Returns:
            SubGraph containing the part, its batch, the
            supplier, and all sibling parts from that batch.
        """
        entities: list[Entity] = []
        rels: list[Relationship] = []

        # Start with the part itself
        part = self._entities.get(part_id)
        if not part:
            return SubGraph()
        entities.append(part)

        # Part → batch
        batch_id: Optional[str] = None
        for neighbor, rel in self.get_neighbors(
            part_id,
            direction="out",
            relation_types=[RelationType.FROM_BATCH],
        ):
            entities.append(neighbor)
            rels.append(rel)
            batch_id = neighbor.id

        if not batch_id:
            return SubGraph(entities=entities, relationships=rels)

        # Batch → supplier
        for neighbor, rel in self.get_neighbors(
            batch_id,
            direction="out",
            relation_types=[RelationType.BATCH_OWNED_BY],
        ):
            entities.append(neighbor)
            rels.append(rel)

        # Find sibling parts from the same batch
        # (other parts that also came from this batch)
        for neighbor, rel in self.get_neighbors(
            batch_id,
            direction="in",
            relation_types=[RelationType.FROM_BATCH],
        ):
            if neighbor.id != part_id:
                entities.append(neighbor)
                rels.append(rel)

                # Also get where the sibling part is installed
                for machine, install_rel in self.get_neighbors(
                    neighbor.id,
                    direction="out",
                    relation_types=[RelationType.INSTALLED_IN],
                ):
                    entities.append(machine)
                    rels.append(install_rel)

        return SubGraph(
            entities=entities,
            relationships=self._deduplicate_relationships(rels),
        )

    def trace_machine_to_root_cause(self, machine_id: str) -> SubGraph:
        """
        Full RCA traversal from a machine back to supplier.

        Follows the chain:
            machine → alerts → sensors
            machine → maintenance events → technician
            maintenance → parts replaced → batch → supplier
            batch → sibling parts → other machines

        This is the complete causal chain the context graph
        RAG builds incrementally over multiple conversation turns.

        Args:
            machine_id: The machine to investigate.

        Returns:
            SubGraph containing the full RCA chain.
        """
        # Start with a deep subgraph from the machine
        machine_subgraph = self.get_subgraph(
            seed_ids=[machine_id],
            max_hops=3,
            max_entities=50,
        )

        # For each part found, trace its supply chain
        part_entities = [
            e for e in machine_subgraph.entities
            if e.entity_type == EntityType.PART
        ]

        combined = machine_subgraph
        for part in part_entities:
            supply_chain = self.trace_supply_chain(part.id)
            combined = combined.merge(supply_chain)

        logger.debug(
            "RCA trace for %s: %d entities, %d relationships",
            machine_id,
            len(combined.entities),
            len(combined.relationships),
        )

        return combined

    # ─────────────────────────────────────────────────────
    # Serialization (for frontend graph viz)
    # ─────────────────────────────────────────────────────

    def to_json_graph(self, subgraph: Optional[SubGraph] = None) -> dict:
        """
        Convert graph (or subgraph) to a JSON-serializable dict.

        Output format is compatible with D3 force-directed graph:
            {
                "nodes": [{"id": ..., "name": ..., "type": ...}, ...],
                "links": [{"source": ..., "target": ..., "type": ...}, ...]
            }

        Args:
            subgraph: If provided, serialize only this subgraph.
                If None, serialize the entire knowledge graph.

        Returns:
            Dict with "nodes" and "links" lists.
        """
        if subgraph:
            nodes = [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.entity_type.value,
                    "properties": e.properties,
                }
                for e in subgraph.entities
            ]
            links = [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "type": r.relation_type.value,
                }
                for r in subgraph.relationships
            ]
        else:
            nodes = [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.entity_type.value,
                    "properties": e.properties,
                }
                for e in self._entities.values()
            ]
            links = [
                {
                    "source": u,
                    "target": v,
                    "type": d.get("relation_type", "unknown"),
                }
                for u, v, d in self._graph.edges(data=True)
            ]

        return {"nodes": nodes, "links": links}

    # ─────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _edge_data_to_relationship(
        source_id: str,
        target_id: str,
        data: dict,
    ) -> Relationship:
        """
        Convert NetworkX edge data dict to a domain Relationship.

        Separates the `relation_type` key from the rest of
        the edge attributes, which become `properties`.
        """
        return Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=RelationType(data["relation_type"]),
            properties={
                k: v for k, v in data.items()
                if k != "relation_type"
            },
        )

    @staticmethod
    def _deduplicate_relationships(
        rels: list[Relationship],
    ) -> list[Relationship]:
        """
        Remove duplicate relationships by composite key.

        Each relationship's key is:
            "{source_id}--{relation_type}-->{target_id}"
        """
        seen: set[str] = set()
        unique: list[Relationship] = []
        for rel in rels:
            if rel.key not in seen:
                unique.append(rel)
                seen.add(rel.key)
        return unique