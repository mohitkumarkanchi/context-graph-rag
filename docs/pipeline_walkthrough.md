# Pipeline Walkthrough — How Data Flows Through the Code

A technical reference showing exactly how a user query moves through the codebase, which functions get called, what data structures are created, and where state is read and written.

---

## Table of contents

1. [Startup: building the knowledge graph](#1-startup-building-the-knowledge-graph)
2. [Basic Graph RAG pipeline](#2-basic-graph-rag-pipeline)
3. [Context Graph RAG pipeline](#3-context-graph-rag-pipeline)
4. [Side-by-side evaluation pipeline](#4-side-by-side-evaluation-pipeline)
5. [Data structures reference](#5-data-structures-reference)
6. [Code path for the RCA scenario](#6-code-path-for-the-rca-scenario)

---

## 1. Startup: building the knowledge graph

When the server starts, `main.py` runs the lifespan handler. Here's the exact sequence:

### Step 1: Generate synthetic data

```
main.py → lifespan()
  └→ SyntheticFactory().build()
       └→ _build_plants()           → 2 Plant entities
       └→ _build_assembly_lines()   → 3 AssemblyLine entities + LOCATED_IN, CONTAINS edges
       └→ _build_machines()         → 8 Machine entities + LOCATED_IN, CONTAINS edges
       └→ _build_sensors()          → 12 Sensor/PLC entities + MONITORED_BY, READS_FROM edges
       └→ _build_personnel()        → 6 Technician/Operator entities + OPERATES edges
       └→ _build_suppliers_and_batches() → 3 Suppliers + 4 Batches + BATCH_OWNED_BY edges
       └→ _build_parts()            → 6 Part entities + FROM_BATCH, INSTALLED_IN, SUPPLIED_BY edges
       └→ _build_maintenance_events() → 5 events + PERFORMED_ON, PERFORMED_BY, REPLACED_PART edges
       └→ _build_alerts_and_defects() → 3 Alerts + 1 Defect + TRIGGERED_ALERT, ALERT_FOR, DEFECT_ON edges
       └→ _build_materials()        → 3 Material entities + PROCESSES edges
       └→ _build_process_logs()     → 4 ProcessLog entities + LOGGED_BY edges
       
  Returns: (~50 entities, ~120 relationships)
```

Each `_build_*` method calls two helpers:
- `_add_entity(id, name, entity_type, properties)` → creates an `Entity` pydantic model, appends to list
- `_add_rel(source_id, target_id, relation_type, **props)` → creates a `Relationship` model, appends to list

### Step 2: Load into NetworkX

```
main.py → lifespan()
  └→ GraphRepository().load(entities, relationships)
       └→ For each entity:
            self._entities[entity.id] = entity          # dict lookup table
            self._graph.add_node(entity.id, **props)    # NetworkX node
       └→ For each relationship:
            self._graph.add_edge(src, tgt, relation_type=..., **props)  # NetworkX edge
```

After this, the knowledge graph is a NetworkX `DiGraph` with ~50 nodes and ~120 directed edges, plus a parallel `dict[str, Entity]` for fast ID lookups.

### Step 3: Initialize remaining services

```
main.py → lifespan()
  └→ ContextRepository()     # empty dict, ready for sessions
  └→ LLMRepository()         # ChatOllama connection to Ollama
  └→ RAGFactory(graph_repo, context_repo, llm_repo)
  └→ EvaluationService(factory)
  └→ router.setup(...)       # inject into FastAPI routes
  └→ ws_module.setup(...)    # inject into WebSocket handler
```

---

## 2. Basic Graph RAG pipeline

When a query hits `POST /api/chat` with `mode: "basic"`:

```
api/router.py → chat()
  └→ factory.create(RAGMode.BASIC)
       └→ Returns BasicGraphRAGService (cached singleton)
  └→ service.query(PipelineState)
```

### Inside BasicGraphRAGService.query()

```python
# 1. Convert Pydantic → LangGraph TypedDict
initial_state = BasicGraphState(
    query="What machines are on Assembly Line A?",
    session_id="abc_basic",
    turn_number=1,
    extracted_entities=[],
    retrieved_subgraph_context="",
    ...
)

# 2. Run the 3-node LangGraph pipeline
result = await self._pipeline.ainvoke(initial_state)

# 3. Convert back to PipelineState
state.response = result["response"]
```

### Node 1: extract_entities

```
Input:  query = "What machines are on Assembly Line A?"

Code path:
  _node_extract_entities()
    └→ llm_repo.extract_entities(query)
         └→ System prompt: "You are an entity extraction system..."
         └→ LLM returns: ["Assembly Line A"]
         └→ _parse_json_list() strips markdown fences, parses JSON

Output: extracted_entities = ["Assembly Line A"]
```

If the LLM returns nothing (extraction fails), fallback kicks in:
```python
if not entities:
    entities = [query]  # use the raw query as a search term
```

### Node 2: retrieve_subgraph

```
Input:  extracted_entities = ["Assembly Line A"]

Code path:
  _node_retrieve_subgraph()
    │
    ├→ For "Assembly Line A":
    │    └→ graph_repo.get_entity_by_name("Assembly Line A")
    │         └→ Scans _entities dict, case-insensitive substring match
    │         └→ Returns Entity(id="line_a", name="Assembly Line A", ...)
    │    └→ seed_ids = ["line_a"]
    │    └→ sources = ["Assembly Line A (assembly_line)"]
    │
    └→ graph_repo.get_subgraph(seed_ids=["line_a"], max_hops=2)
         │
         └→ BFS traversal:
              Hop 0: Visit line_a
                Outgoing: line_a → plant_01 (LOCATED_IN)
                          line_a → machine_m400 (CONTAINS)
                          line_a → machine_m200 (CONTAINS)
                          line_a → machine_l200 (CONTAINS)
                Incoming: plant_01 → line_a (CONTAINS)
                
              Hop 1: Visit plant_01, machine_m400, machine_m200, machine_l200
                Each machine fans out to sensors, operators, etc.
                
         └→ Deduplicate relationships by composite key
         └→ Return SubGraph(entities=[...], relationships=[...])
         
    └→ subgraph.to_context_string()
         └→ Serializes into text:
              "[assembly_line] Assembly Line A (description: CNC machining...)"
              "[machine] CNC Mill M-400 (spindle_speed_rpm: 12000, ...)"
              "  line_a --contains--> machine_m400"
              ...

Output: retrieved_subgraph_context = "..." (text block)
        retrieved_entity_count = 15
        sources = ["Assembly Line A (assembly_line)"]
```

### Node 3: generate_response

```
Input:  query = "What machines are on Assembly Line A?"
        retrieved_subgraph_context = "[assembly_line] Assembly Line A..."

Code path:
  _node_generate_response()
    └→ llm_repo.generate_response(
            query=query,
            subgraph_context=subgraph_context,
            context_summary=None,          # ← THIS IS THE KEY: no context
       )
         └→ System prompt:
              "You are a manufacturing plant assistant.
               Answer using ONLY the provided knowledge graph context.
               
               Knowledge graph context:
               [assembly_line] Assembly Line A ...
               [machine] CNC Mill M-400 ...
               ..."
         └→ LLM generates response

Output: response = "Assembly Line A contains three machines: CNC Mill M-400, 
                    CNC Mill M-200, and Lathe L-200. M-400 is..."
```

### What gets returned to the API

```python
ChatResponse(
    response="Assembly Line A contains three machines...",
    mode="basic",
    session_id="abc_basic",
    turn_number=1,
    extracted_entities=["Assembly Line A"],
    augmented_query=None,         # basic pipeline never augments
    sources=["Assembly Line A (assembly_line)"],
    retrieval_time_ms=12.3,
    generation_time_ms=1847.5,
)
```

---

## 3. Context Graph RAG pipeline

Same query, but `mode: "context"`. The pipeline has 7 nodes instead of 3.

```
api/router.py → chat()
  └→ factory.create(RAGMode.CONTEXT)
       └→ Returns ContextGraphRAGService (cached singleton)
  └→ service.query(PipelineState)
```

### Turn 1: "What machines are on Assembly Line A?"

On the first turn, the context pipeline behaves almost identically to basic. The difference is that it WRITES context at the end.

```
Node 1 (load_context):
  context_repo.get_or_create("abc_context")
  → Creates new ContextState(session_id="abc_context", entity_ids=[], edges=[], turn_count=0)
  → context_summary = "Session: abc_context | Turn: 0"
  → recent_entity_ids = []

Node 2 (resolve_coref):
  recent_entity_ids is empty → SKIP (no context to resolve against)
  → resolved_references = {}

Node 3 (augment_query):
  No resolved references → returns original query unchanged
  → augmented_query = "What machines are on Assembly Line A?"

Node 4 (extract_entities):
  Same as basic pipeline
  → extracted_entities = ["Assembly Line A"]

Node 5 (retrieve_subgraph):
  Same as basic, BUT also includes recent_entity_ids as extra seeds.
  On turn 1, recent_entity_ids is empty, so no difference.
  → Same subgraph as basic pipeline

Node 6 (generate_response):
  Same as basic, BUT passes context_summary to the LLM.
  On turn 1, context_summary is nearly empty, so minimal difference.
  → Similar response to basic pipeline

Node 7 (update_context):    ← THIS IS WHERE CONTEXT GETS WRITTEN
  context_repo.add_discussed_entities(
      "abc_context",
      ["line_a", "machine_m400", "machine_m200", "machine_l200"],
      turn_number=1,
  )
  
  For each entity ID:
    state.add_entity(entity_id)           # adds to entity_ids list
    state.add_edge(ContextEdge(
        source_id=entity_id,
        target_id="turn_1",
        edge_type=ContextEdgeType.DISCUSSED,
        turn_number=1,
    ))
  
  state.turn_count = 1
```

After turn 1, the ContextState looks like:
```python
ContextState(
    session_id="abc_context",
    entity_ids=["line_a", "machine_m400", "machine_m200", "machine_l200"],
    edges=[
        ContextEdge(source="line_a", target="turn_1", type=DISCUSSED, turn=1),
        ContextEdge(source="machine_m400", target="turn_1", type=DISCUSSED, turn=1),
        ContextEdge(source="machine_m200", target="turn_1", type=DISCUSSED, turn=1),
        ContextEdge(source="machine_l200", target="turn_1", type=DISCUSSED, turn=1),
    ],
    turn_count=1,
    resolved_references={},
)
```

### Turn 3: "When was it last serviced?"

This is where context shines. By now the context has entities from turns 1 and 2.

```
Node 1 (load_context):
  Reads existing ContextState
  → context_summary = "Session: abc_context | Turn: 2
                        Entities discussed: line_a, machine_m400, machine_m200, machine_l200, sensor_m400_vib
                        Resolved references: "those machines" → machine_m400, machine_m200, machine_l200"
  → recent_entity_ids = [machine_m400, sensor_m400_vib, ...]

Node 2 (resolve_coref):
  llm_repo.resolve_coreferences(
      query="When was it last serviced?",
      context_summary="...Entities discussed: ...machine_m400..."
  )
  → LLM returns: {"it": "machine_m400"}
  
  context_repo.resolve_reference("abc_context", "it", "machine_m400", turn_number=3)
  → state.resolved_references["it"] = "machine_m400"
  → New edge: ContextEdge(source="ref_it", target="machine_m400", type=RESOLVED_TO, turn=3)

Node 3 (augment_query):
  llm_repo.augment_query(
      query="When was it last serviced?",
      resolved_refs={"it": "machine_m400"},
      context_summary="..."
  )
  → LLM returns: "When was CNC Mill M-400 last serviced?"

Node 4 (extract_entities):
  Extracts from AUGMENTED query, not original
  → "When was CNC Mill M-400 last serviced?" → ["CNC Mill M-400"]

Node 5 (retrieve_subgraph):
  seed_ids from extraction: ["machine_m400"]
  PLUS recent_entity_ids: ["machine_m400", "sensor_m400_vib", ...]
  → Combined seeds retrieve M-400 neighborhood including maintenance events
  
  graph_repo.get_subgraph(seed_ids=["machine_m400", "sensor_m400_vib", ...])
  → BFS finds: maint_m400_jan15, tech_ravi, part_bearing_m400, ...

Node 6 (generate_response):
  LLM receives:
    - Query: "When was CNC Mill M-400 last serviced?"
    - Subgraph context: includes maintenance event details
    - Session context: knows this is turn 3 of an investigation
  
  → "CNC Mill M-400 was last serviced on January 15, 2025.
     Technician Ravi Kumar performed preventive maintenance,
     replacing the spindle bearing with a part from Batch B-442
     (Precision Parts Co.)."

Node 7 (update_context):
  Adds: maint_m400_jan15, tech_ravi to discussed entities
  If RCA metadata present: adds SUSPECTED_CAUSE edges
```

### Key difference: what the LLM sees

On turn 3, here's what each pipeline sends to the LLM:

**Basic pipeline prompt:**
```
System: You are a manufacturing plant assistant. Answer using ONLY
the provided knowledge graph context.

Knowledge graph context:
(empty or random — "it" couldn't be resolved to any entity)

User: When was it last serviced?
```

**Context pipeline prompt:**
```
System: You are a manufacturing plant assistant. Answer using ONLY
the provided knowledge graph context.

Knowledge graph context:
[machine] CNC Mill M-400 (spindle_speed_rpm: 12000, status: degraded, ...)
[maintenance_event] M-400 Bearing Replacement (date: 2025-01-15, ...)
[technician] Ravi Kumar (specialization: CNC maintenance, ...)
[part] Spindle Bearing #M400-SB (installed_date: 2025-01-15, ...)
  machine_m400 --monitored_by--> sensor_m400_vib
  maint_m400_jan15 --performed_on--> machine_m400
  maint_m400_jan15 --performed_by--> tech_ravi
  maint_m400_jan15 --replaced_part--> part_bearing_m400

Conversation context:
Session: abc_context | Turn: 2
Entities discussed: line_a, machine_m400, machine_m200, machine_l200, sensor_m400_vib
Resolved references: "it" → machine_m400

User: When was CNC Mill M-400 last serviced?
```

The context pipeline gives the LLM everything it needs. The basic pipeline gives it nothing.

---

## 4. Side-by-side evaluation pipeline

When `POST /api/evaluate/scenario` is called:

```
api/router.py → evaluate_scenario()
  └→ eval_service.run_scenario(queries=None)  # uses default RCA scenario
       │
       ├→ factory.create_both()
       │    └→ Returns (BasicGraphRAGService, ContextGraphRAGService)
       │
       └→ For each query (6 turns):
            │
            ├→ basic_state = PipelineState(query=q, session_id="abc_basic", turn=i)
            ├→ basic_service.query(basic_state)     # stateless — each turn independent
            │
            ├→ context_state = PipelineState(query=q, session_id="abc_context", turn=i)
            ├→ context_service.query(context_state)  # stateful — same session across turns
            │
            └→ TurnComparison(basic=result1, context=result2)
       
       └→ _generate_summary(report)
       └→ Return ScenarioReport
```

Critical detail: the basic pipeline uses `session_id="abc_basic"` and the context pipeline uses `session_id="abc_context"`. They never share context. The context pipeline's session accumulates state across all 6 turns. The basic pipeline's session is irrelevant since it's stateless.

---

## 5. Data structures reference

### Entity (knowledge graph node)

```python
Entity(
    id="machine_m400",                    # unique ID used in graph
    name="CNC Mill M-400",               # human-readable name
    entity_type=EntityType.MACHINE,       # enum for typing
    properties={                          # flexible key-value attrs
        "manufacturer": "Haas Automation",
        "spindle_speed_rpm": 12000,
        "status": "degraded",
    },
)
```

### Relationship (knowledge graph edge)

```python
Relationship(
    source_id="machine_m400",
    target_id="line_a",
    relation_type=RelationType.LOCATED_IN,
    properties={},
)
# key = "machine_m400--located_in-->line_a"
```

### SubGraph (retrieval result)

```python
SubGraph(
    entities=[Entity(...), Entity(...), ...],
    relationships=[Relationship(...), ...],
)
# .entity_ids → {"machine_m400", "line_a", ...}
# .to_context_string() → serialized text for LLM
# .merge(other) → combine two subgraphs, deduplicate
```

### ContextEdge (context graph edge)

```python
ContextEdge(
    source_id="machine_m400",
    target_id="turn_1",
    edge_type=ContextEdgeType.DISCUSSED,
    turn_number=1,
    confidence=1.0,
    metadata={},
)
```

### ContextState (full session state)

```python
ContextState(
    session_id="abc_context",
    entity_ids=["machine_m400", "line_a", ...],
    edges=[ContextEdge(...), ...],
    turn_count=3,
    resolved_references={"it": "machine_m400", "that batch": "batch_b442"},
)
# .get_recent_entities(last_n_turns=3) → entity IDs from recent turns
# .get_investigation_chain() → SUSPECTED_CAUSE and RULED_OUT edges
# .to_context_string() → serialized text for LLM prompt
```

### PipelineState (flows through LangGraph)

```python
PipelineState(
    query="When was it last serviced?",
    session_id="abc_context",
    turn_number=3,
    extracted_entities=["CNC Mill M-400"],
    augmented_query="When was CNC Mill M-400 last serviced?",
    retrieved_subgraph=SubGraph(...),
    context_state=ContextState(...),
    response="M-400 was last serviced on January 15...",
    sources=["CNC Mill M-400 (machine)", ...],
    retrieval_time_ms=15.2,
    generation_time_ms=2103.4,
)
```

---

## 6. Code path for the RCA scenario

Here's the exact file and function path for the full 6-turn RCA scenario:

```
User clicks "Run scenario" in frontend
  └→ app.js: runScenario()
       └→ fetch("POST /api/evaluate/scenario", body={})

  └→ api/router.py: evaluate_scenario()
       └→ services/evaluation.py: EvaluationService.run_scenario()
            │
            ├→ services/rag_factory.py: RAGFactory.create_both()
            │    ├→ services/basic_graph_rag.py: BasicGraphRAGService()
            │    └→ services/context_graph_rag.py: ContextGraphRAGService()
            │
            └→ For turn 1..6:
                 │
                 ├→ BASIC PATH:
                 │    services/basic_graph_rag.py: query()
                 │      └→ _node_extract_entities()
                 │           └→ repositories/llm_repo.py: extract_entities()
                 │      └→ _node_retrieve_subgraph()
                 │           └→ repositories/graph_repo.py: get_entity_by_name()
                 │           └→ repositories/graph_repo.py: get_subgraph()
                 │      └→ _node_generate_response()
                 │           └→ repositories/llm_repo.py: generate_response()
                 │
                 └→ CONTEXT PATH:
                      services/context_graph_rag.py: query()
                        └→ _node_load_context()
                             └→ repositories/context_repo.py: get_or_create()
                        └→ _node_resolve_coref()
                             └→ repositories/llm_repo.py: resolve_coreferences()
                             └→ repositories/context_repo.py: resolve_reference()
                        └→ _node_augment_query()
                             └→ repositories/llm_repo.py: augment_query()
                        └→ _node_extract_entities()
                             └→ repositories/llm_repo.py: extract_entities()
                        └→ _node_retrieve_subgraph()
                             └→ repositories/graph_repo.py: get_entity_by_name()
                             └→ repositories/graph_repo.py: search_entities()
                             └→ repositories/graph_repo.py: get_subgraph()
                        └→ _node_generate_response()
                             └→ repositories/context_repo.py: get_investigation_summary()
                             └→ repositories/llm_repo.py: generate_response()
                                  OR llm_repo.py: analyze_rca_step()
                        └→ _node_update_context()
                             └→ repositories/context_repo.py: add_discussed_entities()
                             └→ repositories/context_repo.py: mark_suspected_cause()
                             └→ repositories/context_repo.py: mark_follow_up()

  └→ services/evaluation.py: _generate_summary()
  └→ api/router.py: converts to EvalScenarioResponse
  └→ Frontend renders turn cards with side-by-side comparison
```

Every arrow is a real function call you can trace in the codebase. The repository layer is the boundary — nothing above it touches NetworkX, Ollama, or in-memory session dicts directly.