# Context Graph RAG

A side-by-side comparison of **basic Graph RAG** vs **Graph RAG with a context graph**, demonstrating how stateful context improves multi-turn conversations and root cause analysis in manufacturing.


![Alt text](context-graph-rag/assets/display_image.png)





## What is this?

Traditional Graph RAG extracts entities and relationships from documents, stores them in a knowledge graph, and retrieves relevant subgraphs as context for an LLM. It works great for single-shot questions but **falls apart on multi-turn conversations** — every query starts fresh with no memory of what came before.

**Context Graph RAG** adds a session-level context graph that accumulates state across conversation turns. It tracks which entities have been discussed, resolves pronouns like "it" and "that machine" to specific entities, and builds an investigation chain for root cause analysis.

This project demonstrates the difference using a **manufacturing domain** — machines, sensors, maintenance events, supplier batches, and defect reports — with a scripted 6-turn RCA scenario that makes the gap between the two approaches impossible to miss.

## The RCA demo scenario

An operator discovers defective parts coming off Assembly Line A. Over 6 turns, they trace the root cause:

| Turn | Query | Basic RAG | Context RAG |
|------|-------|-----------|-------------|
| 1 | "We're seeing defective parts on Assembly Line A. What machines are on that line?" | Lists machines correctly | Same — both handle this fine |
| 2 | "Check the sensor data for those machines — any anomalies?" | Doesn't know what "those machines" means | Resolves "those" → machines from T1. Finds vibration spike on M-400 |
| 3 | "When was it last serviced?" | "It" is ambiguous — may return wrong machine | "It" = M-400. Finds maintenance on Jan 15 by Ravi Kumar |
| 4 | "What parts were replaced during that service?" | No context for "that service" | Finds spindle bearing from Batch B-442 (Precision Parts Co.) |
| 5 | "Did that batch go to any other machines?" | Complete failure — no idea which batch | Batch B-442 → also went to Lathe L-200 and Grinder G-150 |
| 6 | "Are those machines showing early warning signs?" | Cannot connect anything | L-200 vibration trending up. Proactively warns about G-150 too |

By turn 6, the context graph has built a complete causal chain:

```
Defective parts → M-400 vibration anomaly → Jan 15 maintenance
→ Bearing from Batch B-442 → Same batch in L-200 and G-150
→ L-200 showing early warning signs
```

Basic Graph RAG lost the thread at turn 2.

## Where context graphs shine (beyond RAG)

Context graphs are useful well beyond conversational Q&A:

- **Fraud detection** — A single transaction looks fine. A context graph connecting login from new device + address change + large transfer within 10 minutes reveals a suspicious pattern.
- **Healthcare patient journeys** — Accumulates symptoms, prescriptions, and test results across specialist visits. Can surface that a new symptom might be a side effect from a drug prescribed 3 months ago by a different doctor.
- **Incident management / RCA** — Traces sensor readings → maintenance events → operator shifts → material batch changes to find causal chains. This is what our demo does.
- **Recommendation systems** — Tracks what a user has browsed, rejected, compared, and lingered on during a session. By the 5th product, it understands the emerging preference pattern.
- **Compliance and audit trails** — Encodes temporal and relational constraints: "Did the approval happen before the disbursement? Was the approver different from the requester?"
- **Multi-agent orchestration** — A shared context graph acts as working memory for multiple AI agents collaborating on a task.

## When to use a context graph vs SQL

| Signal | SQL | Context graph | Both |
|--------|-----|---------------|------|
| Fixed schema, known queries | ✓ | | |
| ACID transactions critical | ✓ | | |
| Simple aggregations (COUNT, SUM) | ✓ | | |
| Relationships > 2 JOINs deep | | ✓ | |
| Schema changes per session | | ✓ | |
| Conversational state / memory | | ✓ | |
| Root cause / causal chain analysis | | ✓ | |
| Multi-agent shared working memory | | ✓ | |
| Structured ops data + AI reasoning | | | ✓ |
| Compliance with temporal rules | | | ✓ |
| Recommendation with session context | | | ✓ |

**SQL struggles when relationships are the primary thing you're querying.** "Find all entities within 3 hops of Machine M-400" is a simple graph traversal but an ugly self-join repeated 3 times in SQL.

**The deeper issue is schema rigidity.** In a context graph, new relationship types emerge dynamically — `discussed`, `compared_with`, `suspected_cause_of` — during a conversation. In SQL you'd need to predefine all possible relationship types or use an EAV pattern, which is essentially a poorly optimized graph database.

**In production, you'd probably use both.** SQL for structured operational data (machine specs, maintenance schedules, inventory) and a context graph layered on top for relational reasoning and session state.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Frontend (HTML + JS + D3 graph viz)            │
│  Split-pane chat │ Live context graph │ Diff    │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│  API Layer (FastAPI)                            │
│  POST /chat │ GET /graph │ POST /evaluate │ WS  │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│  Service Layer (LangGraph pipelines)            │
│                                                 │
│  RAGFactory.create(mode) ──┬── BasicGraphRAG    │
│                            └── ContextGraphRAG  │
│  EvaluationService (side-by-side comparison)    │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│  Repository Layer                               │
│  GraphRepo (NetworkX) │ ContextRepo │ LLMRepo   │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│  Data Layer                                     │
│  Knowledge Graph (in-memory) │ Synthetic Factory│
└─────────────────────────────────────────────────┘
```

### Pipeline comparison

**Basic Graph RAG** (3 nodes, stateless):
```
Query → Extract entities → Retrieve subgraph → Generate response
```

**Context Graph RAG** (7 nodes, stateful):
```
Query → Load context → Resolve coreferences → Augment query
      → Extract entities → Retrieve subgraph (context-aware)
      → Generate response → Update context graph
                                    ↑
                                    └── feeds next turn
```

## Project structure

```
context-graph-rag/
├── domain/
│   ├── enums.py              # EntityType, RelationType, RAGMode, etc.
│   └── models.py             # Pydantic models: Entity, SubGraph, ContextState, etc.
├── data/
│   └── synthetic_factory.py  # Generates manufacturing knowledge graph
├── repositories/
│   ├── graph_repo.py         # NetworkX knowledge graph operations
│   ├── context_repo.py       # Session context graph management
│   └── llm_repo.py           # ChatOllama wrapper with structured prompts
├── services/
│   ├── rag_factory.py        # Factory pattern + BaseRAGService ABC
│   ├── basic_graph_rag.py    # Stateless LangGraph pipeline (3 nodes)
│   ├── context_graph_rag.py  # Stateful LangGraph pipeline (7 nodes)
│   └── evaluation.py         # Side-by-side comparison runner
├── api/
│   ├── router.py             # FastAPI REST endpoints
│   ├── schemas.py            # Pydantic request/response models
│   └── websocket.py          # WebSocket streaming endpoint
├── frontend/                 # HTML + JS + D3 (built later)
├── tests/
├── config.py                 # Pydantic settings (.env reader)
├── main.py                   # FastAPI app entry point
├── pyproject.toml            # uv project config
└── .env                      # Environment variables
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Ollama](https://ollama.ai/) running locally

### Installation

```bash
# Clone and enter the project
cd context-graph-rag

# Install dependencies
uv sync

# Pull the LLM model
ollama pull llama3.2

# Verify setup
uv run python -c "
import fastapi, langchain, langgraph, networkx, ollama
print('All packages installed successfully!')
"
```

### Running

```bash
# Start the server
uv run uvicorn main:app --reload

# Or directly
uv run python main.py
```

The server starts at `http://localhost:8000`:
- **API docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:8000 (once frontend is built)
- **WebSocket**: ws://localhost:8000/ws/chat

### Quick test

```bash
# Health check
curl http://localhost:8000/api/health | python -m json.tool

# Single chat query (context mode)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What machines are on Assembly Line A?", "mode": "context"}'

# Run the full RCA scenario evaluation
curl -X POST http://localhost:8000/api/evaluate/scenario \
  -H "Content-Type: application/json" \
  -d '{}' | python -m json.tool

# Get the full knowledge graph (for visualization)
curl http://localhost:8000/api/graph | python -m json.tool
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Send a query to basic or context RAG pipeline |
| GET | `/api/context/{session_id}/graph` | Get session context graph (D3 format) |
| GET | `/api/context/{session_id}/summary` | Get RCA investigation summary |
| DELETE | `/api/context/{session_id}` | Reset a session |
| POST | `/api/evaluate/single` | Compare one query across both pipelines |
| POST | `/api/evaluate/scenario` | Run multi-turn RCA scenario evaluation |
| GET | `/api/graph` | Get the full knowledge graph |
| POST | `/api/graph/subgraph` | Extract a subgraph from seed entities |
| GET | `/api/health` | System health check |
| WS | `/ws/chat` | WebSocket for streaming chat |

## Tech stack

| Layer | Technology | Why |
|-------|------------|-----|
| LLM | Ollama + llama3.2 | Local, free, no API keys |
| Graph store | NetworkX | Zero setup, in-memory, perfect for demos |
| Pipeline orchestration | LangGraph | Native state management, graph-based workflows |
| LLM integration | LangChain + ChatOllama | Async-native, structured prompts |
| API framework | FastAPI | Async, auto-docs, Pydantic integration |
| Data validation | Pydantic | Shared models across all layers |
| Package manager | uv | Fast, modern Python project management |

## Design patterns

- **Factory pattern** — `RAGFactory.create(mode)` returns the right pipeline without the caller knowing the implementation. Both services implement `BaseRAGService`.
- **Repository pattern** — All data access (graph, context, LLM) goes through repository classes. No direct NetworkX or Ollama calls in service or API code.
- **Layered architecture** — Domain → Data → Repository → Service → API. Each layer only depends on the ones below it.
- **Pydantic everywhere** — Domain models, API schemas, and config all use Pydantic for validation and serialization consistency.

## License

MIT