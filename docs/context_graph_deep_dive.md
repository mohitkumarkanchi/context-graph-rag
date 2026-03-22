# Context Graph RAG — Deep Dive

A comprehensive guide to understanding context graphs, how they differ from traditional Graph RAG, when to use them, and exactly how the context graph gets populated turn by turn in this project.

---

## Table of contents

1. [What is Graph RAG?](#1-what-is-graph-rag)
2. [The problem: statelessness](#2-the-problem-statelessness)
3. [What a context graph adds](#3-what-a-context-graph-adds)
4. [How the context graph gets populated (step by step)](#4-how-the-context-graph-gets-populated)
5. [The RCA scenario walkthrough](#5-the-rca-scenario-walkthrough)
6. [Context graph use cases beyond RAG](#6-context-graph-use-cases-beyond-rag)
7. [When to use context graph vs SQL](#7-when-to-use-context-graph-vs-sql)
8. [Architecture and design decisions](#8-architecture-and-design-decisions)

---

## 1. What is Graph RAG?

Traditional RAG (Retrieval Augmented Generation) works like this: take a user question, search a vector database for relevant chunks, stuff those chunks into a prompt, and ask the LLM to answer.

Graph RAG replaces the vector database with a knowledge graph. Instead of searching for similar text chunks, it:

1. **Extracts entities** from the user's question (e.g., "CNC Mill M-400")
2. **Finds those entities** as nodes in a knowledge graph
3. **Traverses the graph** N hops outward to collect related entities and relationships
4. **Serializes that subgraph** into text and feeds it to the LLM as context
5. **Generates an answer** grounded in the graph's structured knowledge

The advantage over vector RAG is that relationships are explicit. The knowledge graph doesn't just know that "M-400" and "Assembly Line A" appear near each other in some document — it knows that M-400 is `LOCATED_IN` Assembly Line A, which is `LOCATED_IN` Plant Floor 1. These typed, directional relationships make retrieval far more precise.

---

## 2. The problem: statelessness

Basic Graph RAG treats every query independently. The pipeline has no memory:

```
Turn 1: "What machines are on Assembly Line A?"
→ Extracts "Assembly Line A" → retrieves machines → answers correctly

Turn 2: "Check the sensor data for those machines"
→ Extracts... what? "those machines" is not an entity name.
→ The pipeline has no idea what "those" refers to.
→ Returns generic or irrelevant sensor data.
```

This is the fundamental limitation. In a real investigation, every question builds on the previous answers. Humans maintain a mental model of the conversation — they know that "it" means the machine they just asked about, that "that batch" means the supplier batch mentioned two turns ago. Basic Graph RAG has no such mental model.

The symptoms get worse with each turn:

- **Turn 1**: Both pipelines work identically
- **Turn 2**: Basic starts failing on pronouns ("those machines")
- **Turn 3**: Basic can't resolve "it" at all
- **Turn 4–6**: Basic is essentially answering random questions because it has zero conversational context

---

## 3. What a context graph adds

A context graph is a **per-session, dynamic graph** that sits alongside the static knowledge graph. It doesn't replace the knowledge graph — it augments it with conversational state.

### What the knowledge graph knows (static, pre-built)

```
Machine M-400 --LOCATED_IN--> Assembly Line A
Machine M-400 --MONITORED_BY--> Sensor VS-401
Sensor VS-401 --TRIGGERED_ALERT--> Alert M-400 Vibration
Maintenance Jan 15 --PERFORMED_ON--> Machine M-400
Maintenance Jan 15 --REPLACED_PART--> Spindle Bearing #M400-SB
Spindle Bearing #M400-SB --FROM_BATCH--> Batch B-442
Batch B-442 --BATCH_OWNED_BY--> Precision Parts Co.
```

These relationships exist before any conversation happens. They encode domain knowledge.

### What the context graph knows (dynamic, per-session)

```
machine_m400 --DISCUSSED--> turn_1
machine_l200 --DISCUSSED--> turn_1
machine_m200 --DISCUSSED--> turn_1
ref_"it" --RESOLVED_TO--> machine_m400       (from turn 3)
ref_"that batch" --RESOLVED_TO--> batch_b442  (from turn 5)
defect_001 --SUSPECTED_CAUSE--> machine_m400  (from turn 2)
machine_m400 --SUSPECTED_CAUSE--> maint_jan15 (from turn 3)
maint_jan15 --SUSPECTED_CAUSE--> batch_b442   (from turn 4)
machine_l200 --FOLLOW_UP--> follow_up_t5      (from turn 5)
machine_g150 --FOLLOW_UP--> follow_up_t5      (from turn 5)
```

These relationships are created during the conversation. They encode what the user has discussed, what pronouns resolve to, what the investigation chain looks like, and what needs follow-up.

### The three capabilities context provides

**Coreference resolution**: "it" → machine_m400, "that batch" → batch_b442, "those machines" → [machine_l200, machine_g150]. Without this, any follow-up question with a pronoun fails.

**Retrieval re-ranking**: When retrieving from the knowledge graph, recently discussed entities get boosted as additional seed nodes. So even if the current query doesn't explicitly mention M-400, the retrieval still includes M-400's neighborhood because the conversation is about M-400.

**Investigation chain tracking**: The context graph tracks suspected causes, ruled-out entities, and items flagged for follow-up. This lets the LLM build on the investigation rather than starting from scratch each turn.

---

## 4. How the context graph gets populated

This is the core mechanism. The context graph starts empty and grows through the 7-node LangGraph pipeline that runs on every turn.

### The 7-node pipeline

```
Query arrives
    │
    ▼
[1] LOAD CONTEXT
    Read the session's ContextState from memory.
    Extract: context_summary, resolved_references, recent_entity_ids.
    On turn 1, everything is empty — that's fine.
    │
    ▼
[2] RESOLVE COREFERENCES
    Send the query + context_summary to the LLM.
    Ask: "Are there any pronouns or vague references? Map them to entity IDs."
    Example: query="When was it last serviced?", context says machine_m400 was
    discussed → LLM returns {"it": "machine_m400"}.
    Record the resolution in the context repo.
    Skipped on turn 1 (no context to resolve against).
    │
    ▼
[3] AUGMENT QUERY
    Rewrite the query by replacing pronouns with entity names.
    "When was it last serviced?" → "When was CNC Mill M-400 last serviced?"
    This augmented query is what the rest of the pipeline uses.
    │
    ▼
[4] EXTRACT ENTITIES
    Send the augmented query to the LLM for entity extraction.
    "When was CNC Mill M-400 last serviced?" → ["CNC Mill M-400"]
    The augmented query makes extraction far more reliable than
    trying to extract from "When was it last serviced?"
    │
    ▼
[5] RETRIEVE SUBGRAPH (context-aware)
    Resolve extracted entity names to graph node IDs.
    ALSO add recently discussed entity IDs as extra seed nodes.
    Run BFS traversal from ALL seed nodes (extracted + context).
    This means the retrieval stays anchored to the conversation topic
    even if the current query is vague.
    │
    ▼
[6] GENERATE RESPONSE
    Send the query + subgraph context + session context to the LLM.
    The session context tells the LLM what's been discussed before,
    so it can reference previous answers and build continuity.
    If there's an active RCA investigation, use the RCA-specific
    prompt that also returns structured metadata.
    │
    ▼
[7] UPDATE CONTEXT  ← THIS IS WHERE THE CONTEXT GRAPH GROWS
    After generation, write back to the context repo:
    
    a) DISCUSSED entities: all entity IDs that were retrieved
       or mentioned get a DISCUSSED edge to the current turn.
       
    b) Coreference resolutions: already recorded in step 2,
       but any new ones from the LLM response are added.
       
    c) RCA metadata (from the structured LLM response):
       - suspected_causes → SUSPECTED_CAUSE edges
       - ruled_out → RULED_OUT edges  
       - follow_ups → FOLLOW_UP edges
       - new_entities → added to tracked entity list
    
    d) Turn counter is advanced.
    
    ALL of this feeds back into step 1 on the NEXT turn.
```

### The feedback loop

This is the key insight: **step 7 writes state that step 1 reads on the next turn**. Each turn enriches the context, which makes the next turn's coreference resolution, retrieval, and generation better.

```
Turn 1: Empty context → extract "Assembly Line A" → retrieve machines
         → WRITE: discussed [line_a, machine_m400, machine_l200, machine_m200]

Turn 2: READ context → knows those machines → resolve "those" → augment query
         → better retrieval → WRITE: discussed [sensor_m400_vib], suspected [defect→m400]

Turn 3: READ context → knows M-400 is the focus → resolve "it" → augment query
         → retrieve maintenance → WRITE: discussed [maint_jan15, tech_ravi]

Turn 4: READ context → knows the Jan 15 maintenance → resolve "that service"
         → retrieve parts → WRITE: discussed [part_bearing_m400, batch_b442]

Turn 5: READ context → knows batch B-442 → resolve "that batch"
         → trace supply chain → find sibling parts
         → WRITE: discussed [machine_l200, machine_g150], follow_up [l200, g150]

Turn 6: READ context → knows L-200 and G-150 from turn 5 → resolve "those machines"
         → check their sensors → find early warnings
         → WRITE: complete causal chain in context
```

### What the context graph looks like at turn 6

Nodes (14 tracked entities):
```
line_a, machine_m400, machine_l200, machine_m200, machine_g150,
sensor_m400_vib, maint_jan15, tech_ravi, part_bearing_m400,
batch_b442, supplier_precision, part_bearing_l200, part_bearing_g150,
defect_001
```

Edges (accumulated across all turns):
```
DISCUSSED edges:     14 entities × their respective turns
RESOLVED_TO edges:   "it"→m400, "those machines"→[m400,l200,m200],
                     "that service"→maint_jan15, "that batch"→b442
SUSPECTED_CAUSE:     defect_001→m400→maint_jan15→batch_b442
FOLLOW_UP:           machine_l200, machine_g150
```

This accumulated subgraph IS the investigation. It's the complete causal chain that a human investigator would have built mentally. The context graph makes it explicit and machine-readable.

---

## 5. The RCA scenario walkthrough

Here's exactly what happens at each turn, showing what basic RAG sees vs what context RAG sees.

### Turn 1: "We're seeing defective parts on Assembly Line A. What machines are on that line?"

| | Basic RAG | Context RAG |
|---|---|---|
| Context loaded | None | None (first turn) |
| Coreference | Skipped | Skipped (no context) |
| Augmented query | N/A | Same as original |
| Extracted entities | "Assembly Line A" | "Assembly Line A" |
| Seed nodes | [line_a] | [line_a] |
| Retrieval | Line A + connected machines | Same |
| Response | Lists M-400, M-200, L-200 | Same |
| Context written | Nothing | DISCUSSED: line_a, machine_m400, machine_m200, machine_l200 |

Both work identically. The difference starts next turn.

### Turn 2: "Check the sensor data for those machines — any anomalies?"

| | Basic RAG | Context RAG |
|---|---|---|
| Context loaded | None | entities=[line_a, m400, m200, l200] |
| Coreference | Can't resolve "those machines" | "those machines" → [m400, m200, l200] |
| Augmented query | "Check the sensor data for those machines" (unchanged) | "Check sensor data for CNC Mill M-400, CNC Mill M-200, and Lathe L-200" |
| Extracted entities | "those machines" (fails or random) | "CNC Mill M-400", "CNC Mill M-200", "Lathe L-200" |
| Seed nodes | [] or random | [machine_m400, machine_m200, machine_l200] |
| Retrieval | Wrong or empty subgraph | Sensors for all three machines |
| Response | Vague or incorrect | "M-400 vibration at 7.2mm/s (threshold 4.5) — critical. L-200 trending up at 3.9mm/s." |
| Context written | Nothing | DISCUSSED: sensors. SUSPECTED: defect→m400 |

Basic RAG has already lost the thread. Context RAG found the anomaly.

### Turn 3: "When was it last serviced?"

| | Basic RAG | Context RAG |
|---|---|---|
| Context loaded | None | knows M-400 is the focus |
| Coreference | "it" = ??? | "it" → machine_m400 |
| Augmented query | "When was it last serviced?" (unchanged) | "When was CNC Mill M-400 last serviced?" |
| Response | Random maintenance record or "I don't know which machine" | "M-400 was serviced on January 15, 2025 by technician Ravi Kumar. It was a preventive maintenance — spindle bearing replacement." |

### Turn 4: "What parts were replaced during that service?"

| | Basic RAG | Context RAG |
|---|---|---|
| Coreference | "that service" = ??? | "that service" → maint_jan15 |
| Augmented query | Unchanged | "What parts were replaced during the M-400 maintenance on January 15?" |
| Response | Generic or wrong | "Spindle bearing #M400-SB was replaced, sourced from Batch B-442 supplied by Precision Parts Co." |

### Turn 5: "Did that batch go to any other machines?"

| | Basic RAG | Context RAG |
|---|---|---|
| Coreference | "that batch" = ??? | "that batch" → batch_b442 |
| Retrieval | Nothing relevant | Traces: B-442 → sibling parts → L-200, G-150 |
| Response | Complete failure | "Yes — Batch B-442 also supplied bearings to Lathe L-200 (installed Jan 18) and Grinder G-150 (installed Jan 20)." |
| Context written | Nothing | FOLLOW_UP: machine_l200, machine_g150 |

### Turn 6: "Are those machines showing any early warning signs?"

| | Basic RAG | Context RAG |
|---|---|---|
| Coreference | "those machines" = ??? | "those machines" → [machine_l200, machine_g150] |
| Retrieval | Nothing relevant | Sensors for L-200 and G-150 |
| Response | Cannot answer | "L-200 vibration is at 3.9mm/s, approaching the 4.5 threshold — recommend inspection. G-150 is at 2.8mm/s with threshold 3.0 — also trending up. Both received bearings from the same batch B-442 that caused the M-400 failure." |

By turn 6, the context graph has the complete causal chain. Basic RAG was lost by turn 2.

---

## 6. Context graph use cases beyond RAG

The pattern of "accumulate state as a graph across interactions" applies far beyond conversational Q&A.

### Fraud detection

A single transaction looks fine. But a context graph connecting events across a user session reveals patterns:

```
Session graph after 10 minutes:
  login_new_device --FOLLOWED_BY--> address_change
  address_change --FOLLOWED_BY--> large_transfer
  large_transfer --AMOUNT--> $47,000
  large_transfer --DESTINATION--> new_account (never seen before)
```

Each event alone is innocent. The context graph connecting them reveals a suspicious pattern. A stateless system evaluating each transaction independently would miss it.

### Healthcare patient journeys

A patient visits multiple specialists over months. Each doctor sees their own slice:

```
Context graph across visits:
  visit_jan (cardiology) --PRESCRIBED--> Drug A
  visit_mar (neurology) --SYMPTOM--> headaches
  Drug A --KNOWN_SIDE_EFFECT--> headaches
  visit_mar --FOLLOW_UP--> check Drug A interaction
```

A context graph accumulating symptoms, prescriptions, and test results across visits can surface that the new symptom might be a side effect from a drug prescribed by a different doctor 3 months ago. No single doctor's records contain this insight.

### Incident management / root cause analysis

This is exactly what our demo does. In manufacturing:

```
Context graph during investigation:
  defect_rate_spike --OBSERVED_ON--> Machine X
  Machine X --LAST_MAINTAINED--> Event Jan 15
  Event Jan 15 --REPLACED--> Part from Batch B-442
  Batch B-442 --ALSO_SUPPLIED--> Machine Y, Machine Z
  Machine Y --SENSOR_TREND--> vibration increasing
```

The context graph traces sensor readings → maintenance events → operator shifts → material batch changes to find causal chains. Each step in the investigation adds nodes and edges.

### Recommendation systems with session awareness

Traditional recommendations are stateless — they don't know what you browsed 5 minutes ago:

```
Session context graph:
  user --VIEWED--> Product A (running shoes, $120)
  user --VIEWED--> Product B (running shoes, $95)
  user --DISMISSED--> Product C (casual shoes, $80)
  user --LINGERED_ON--> Product B (45 seconds)
  user --COMPARED--> Product A vs Product B
```

By the 5th product viewed, the context graph understands: the user wants running shoes in the $90–120 range and prefers the cheaper option. A stateless recommender would keep showing casual shoes.

### Compliance and audit trails

Regulations often care about sequences and relationships:

```
Context graph for a disbursement:
  approval --TIMESTAMP--> Jan 10
  disbursement --TIMESTAMP--> Jan 12
  approval --BY--> Manager A
  disbursement --REQUESTED_BY--> Employee B
  Manager A --REPORTS_TO--> Director (not Employee B ✓)
  review_period = 48 hours (Jan 10 → Jan 12 ✓)
```

"Did the approval happen before the disbursement? Was the approver different from the requester? Did the review window meet the 48-hour requirement?" A context graph encodes these temporal and relational constraints naturally. SQL can answer each question individually but connecting them requires complex multi-table joins.

### Multi-agent orchestration

When multiple AI agents collaborate on a task, a shared context graph acts as their working memory:

```
Shared context graph:
  Agent A (researcher) --FOUND--> 5 relevant papers
  Agent A --SUMMARIZED--> key findings
  Agent B (planner) --READ--> findings from Agent A
  Agent B --CREATED--> action plan with 3 steps
  Agent C (executor) --EXECUTING--> step 1
  Agent C --BLOCKED_BY--> missing data
  Agent C --REQUESTED--> Agent A re-research topic X
```

The context graph is how agents stay coherent without passing enormous text blobs back and forth. Each agent reads from and writes to the shared graph.

---

## 7. Why not just use a SQL database?

This is the right question to ask, and the honest answer is: sometimes SQL *is* the better choice. But for context graphs specifically, SQL has fundamental limitations that go beyond performance.

### The surface-level answer: JOINs don't scale with depth

Imagine the RCA scenario at turn 5. The context pipeline needs to answer: "Did that batch go to any other machines?" This requires traversing:

```
batch_b442 ← FROM_BATCH ← part_bearing_l200 → INSTALLED_IN → machine_l200
batch_b442 ← FROM_BATCH ← part_bearing_g150 → INSTALLED_IN → machine_g150
```

In a graph, this is a 2-hop traversal from `batch_b442`. Takes microseconds.

In SQL, this is:

```sql
SELECT m.name
FROM parts p1
JOIN batches b ON p1.batch_id = b.id
JOIN parts p2 ON p2.batch_id = b.id AND p2.id != p1.id
JOIN machines m ON p2.installed_in = m.id
WHERE b.id = 'B-442';
```

Two JOINs for 2 hops. Manageable. But the RCA trace from machine back to supplier is 5+ hops:

```
machine → maintenance_event → part → batch → supplier
                                  ↓
                            other_parts → other_machines → their_sensors
```

In SQL, that's 7+ JOINs, and each additional hop multiplies the query complexity. In a graph, it's `get_subgraph(seed_ids=["machine_m400"], max_hops=4)` — same function regardless of depth.

### The deeper answer: schema rigidity vs dynamic relationships

This is the real problem. The knowledge graph has a fixed schema — you could model it in SQL tables. But the *context graph* creates new relationship types at runtime:

```
Turn 1: DISCUSSED edges (entity → turn marker)
Turn 2: RESOLVED_TO edges ("those machines" → specific entity IDs)
Turn 3: SUSPECTED_CAUSE edges (defect → machine)
Turn 4: SUSPECTED_CAUSE edges (machine → maintenance event)
Turn 5: FOLLOW_UP edges (flagging machines for inspection)
Turn 6: RULED_OUT edges (eliminating non-causes)
```

These edge types emerge from the conversation. They weren't defined at schema design time. In a SQL database, you'd have three options:

**Option A: A table per relationship type.** Create `discussed`, `resolved_to`, `suspected_cause`, `ruled_out`, `follow_up` tables. But what happens when the LLM discovers a new relationship type during conversation? You'd need to run DDL (CREATE TABLE) at runtime. That's not how SQL databases are meant to work.

**Option B: A generic edges table (EAV pattern).**

```sql
CREATE TABLE context_edges (
    source_id VARCHAR,
    target_id VARCHAR,
    edge_type VARCHAR,
    turn_number INT,
    confidence FLOAT,
    metadata JSON
);
```

This works — but now you've built a graph database with extra steps. Every "traversal" is a self-join on this table. You've lost all the benefits of SQL (type safety, foreign keys, query optimization) and gained none of the benefits of a graph (native traversal, path-finding, community detection). This is essentially what people mean when they say "EAV is a poorly optimized graph database pretending to be relational."

**Option C: Store everything as JSON.** Dump the context state as a JSON blob per session. Simple to store, impossible to query. "Find all sessions where machine_m400 was a suspected cause" requires deserializing every session's JSON.

### What about the knowledge graph — could THAT be SQL?

Yes, actually. The static knowledge graph (machines, sensors, parts, suppliers) has a fixed schema and could live in PostgreSQL tables:

```sql
machines (id, name, line_id, manufacturer, spindle_speed, ...)
sensors (id, name, machine_id, metric, threshold, ...)
maintenance_events (id, machine_id, technician_id, date, type, ...)
parts (id, name, batch_id, machine_id, installed_date, ...)
batches (id, supplier_id, manufactured_date, quantity, ...)
```

For the static knowledge graph, SQL is perfectly fine — and in production you'd probably use it. The schema is known, the relationships are predefined, and you get ACID transactions for writes.

The context graph is where SQL falls apart. It's dynamic, session-scoped, relationship-heavy, and needs to support arbitrary-depth traversal. That's a graph problem.

### The production architecture: both

In the real world, you wouldn't pick one or the other. You'd use:

```
SQL (PostgreSQL, etc.)          → System of record
  - Machine specs               - ACID writes
  - Maintenance schedules        - Aggregations (AVG cycle time, SUM defects)
  - Sensor readings (time-series)- Reporting dashboards
  - Inventory / procurement      - Known queries, fixed schema

Knowledge Graph (Neo4j, etc.)   → Reasoning layer
  - Entity relationships         - Multi-hop traversal
  - RCA causal chains            - Path finding
  - Cross-domain connections     - Pattern matching

Context Graph (in-memory / Redis) → Session state
  - Per-session investigation    - Dynamic edge types
  - Coreference resolutions      - Real-time read/write per turn
  - Suspected/ruled-out chains   - Ephemeral (dies with session)
```

Our demo uses NetworkX for both graphs because it's zero-setup and perfect for learning. But the architecture cleanly separates the concerns — swapping NetworkX for Neo4j or SQL for PostgreSQL doesn't change any service-layer code.

### Decision framework

Ask these five questions about your use case:

| Question | If yes → |
|---|---|
| Are relationships the primary thing you query? | Context graph |
| Does your schema evolve dynamically at runtime? | Context graph |
| Do you need multi-hop reasoning (3+ hops)? | Context graph |
| Is your data tabular with known columns and simple lookups? | SQL |
| Do you need ACID transactions and strong consistency? | SQL |
| Do you need structured storage AND relational reasoning? | Both |

### Real-world mapping

**SQL only**: Inventory management, payroll, order tracking, reporting dashboards, CRUD applications. The schema is fixed, queries are known, relationships are shallow.

**Context graph only**: Conversational AI session state, fraud pattern detection across events, knowledge exploration, multi-agent shared working memory. Relationships are the point, schema is dynamic, depth is unbounded.

**Both (the most common production pattern)**:
- Manufacturing RCA — SQL stores sensor readings and maintenance records, graph traces causal chains across them
- Healthcare patient journeys — SQL stores medical records, graph connects symptoms and prescriptions across specialist visits
- Compliance systems — SQL stores events and timestamps, graph validates temporal and relational rules across them
- Supply chain risk — SQL stores procurement data, graph traces batch exposure across production lines

---

## 8. Architecture and design decisions

### Why LangGraph for the pipeline

LangGraph is purpose-built for stateful, graph-based agent workflows. It gives us:

- **Native state management** that persists across pipeline nodes
- **Graph-based workflow definition** where each node is a processing step
- **Built-in checkpointing** so you can snapshot and replay investigations
- **Tight integration with LangChain's** retrieval and LLM tools

Each step in our RAG pipeline becomes a node in a LangGraph graph, and the context state flows naturally through LangGraph's state management.

### Why NetworkX over Neo4j

For a demo project, NetworkX gives us zero setup, in-memory operation, and pure Python. The knowledge graph is small enough (~60 nodes, ~150 edges) that graph database overhead isn't justified. In production with thousands of machines and millions of sensor readings, you'd switch to Neo4j, Amazon Neptune, or similar.

### Why the factory pattern

The `RAGFactory.create(mode)` pattern lets the API layer be completely agnostic about which pipeline it's talking to. Both services implement the same `BaseRAGService` interface with the same `query(state)` method. The evaluation service calls both through this interface. If you add a third pipeline variant later, you just add another branch in the factory — nothing else changes.

### Why Pydantic everywhere

Using Pydantic for domain models, API schemas, and config creates consistency across all layers. An `Entity` created in the synthetic factory, stored in the graph repo, serialized through the API, and rendered in the frontend all use the same validated model. No manual serialization, no schema drift between layers.

### The TypedDict bridge

LangGraph requires TypedDict for its state schema, but the rest of the codebase uses Pydantic. The service layer converts between the two at the pipeline entry/exit points. This keeps LangGraph as an internal implementation detail — if you swapped LangGraph for a different orchestrator, only the service files would change.

---

## Summary

The context graph is not a replacement for the knowledge graph. It's a **session-level working memory** that makes multi-turn interactions coherent. It tracks what's been discussed, resolves ambiguous references, and builds investigation chains incrementally.

The key mechanism is the **feedback loop**: each turn's pipeline reads from the context graph at the start and writes to it at the end. This creates a compounding effect where each turn enriches the context for the next.

In our manufacturing demo, this turns a basic Q&A system into an investigation assistant that can trace a defective part back through the maintenance event, to the specific bearing, to the supplier batch, and then forward to find other machines at risk — all through natural conversation with pronouns and vague references that a stateless system can't handle.