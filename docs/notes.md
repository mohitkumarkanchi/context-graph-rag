A context graph gives your RAG system three things it doesn't have today:

→ It resolves "it", "that machine", "those parts" to actual entities
→ It tracks what's been investigated, suspected, and ruled out
→ It makes every retrieval sharper because context anchors the search

Chat history can't do this. It's flat text. A context graph is a structured, growing map of the conversation — entities as nodes, relationships as edges, investigation state as metadata.

I built two pipelines side by side to show the difference:

Basic Graph RAG (3 steps): extract → retrieve → generate. Stateless.
Context Graph RAG (7 steps): load context → resolve coreferences → augment query → extract → retrieve with context seeds → generate with investigation state → update context graph.

The 4 extra steps create a feedback loop. Each turn reads from the context graph and writes back. The graph grows. Retrieval sharpens.

Tested on a 6-turn manufacturing RCA scenario — tracing defective parts to a bad supplier batch:

By turn 3, basic was guessing.
By turn 6, context had traced: Defect → Machine → Maintenance → Part → Batch B-442 → 2 other machines at risk.

14 entities tracked, 4 coreference resolutions, a 4-link causal chain — all built through conversation.

Three takeaways:

1. Coreference resolution is the unlock. "It" → "CNC Mill M-400" changes everything downstream.
2. Recently discussed entities as retrieval seeds is surprisingly powerful — context anchors vague queries.
3. The pattern is domain-agnostic — manufacturing, fraud detection, healthcare, compliance, multi-agent systems.

Open source, fully local (Python + LangGraph + Ollama + NetworkX).

What domain would you apply this to?

#AI #RAG #GraphRAG #KnowledgeGraph #LangGraph #Python #BuildInPublic