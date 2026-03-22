"""
LLM repository.

Wraps ChatOllama with domain-specific prompt methods for
entity extraction, coreference resolution, and RAG generation.
All LLM interactions go through this single class — no direct
Ollama calls anywhere else in the codebase.
"""

import json
import logging
import time
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings

logger = logging.getLogger(__name__)


class LLMRepository:
    """
    Manages all LLM interactions via ChatOllama.

    Provides structured methods for each LLM task:
        - Entity extraction from user queries
        - Coreference resolution using session context
        - RAG answer generation from subgraph context
        - RCA analysis and causal chain reasoning

    Each method constructs a focused system prompt, calls
    the model, and parses the response into domain objects.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            request_timeout=settings.ollama_request_timeout,
        )
        self._model_name = settings.ollama_model
        logger.info("LLM repository initialized with model: %s", self._model_name)

    # ─────────────────────────────────────────────────────
    # Entity extraction
    # ─────────────────────────────────────────────────────

    async def extract_entities(self, query: str) -> list[str]:
        """
        Extract entity names and identifiers from a user query.

        Asks the LLM to identify manufacturing-domain entities
        (machines, parts, people, locations, etc.) mentioned
        in the query. Returns normalized search terms that can
        be matched against the knowledge graph.

        Args:
            query: The raw user question.

        Returns:
            List of extracted entity name strings.
            Empty list if extraction fails.
        """
        system_prompt = (
            "You are an entity extraction system for a manufacturing plant. "
            "Extract all entity names from the user's query. Entities include: "
            "machines, assembly lines, plants, sensors, technicians, operators, "
            "parts, suppliers, batches, defects, alerts, and materials.\n\n"
            "Return ONLY a JSON array of strings. No explanation, no markdown.\n"
            "Example: [\"CNC Mill M-400\", \"Assembly Line A\", \"Ravi Kumar\"]\n"
            "If no entities found, return: []"
        )

        response_text = await self._invoke(system_prompt, query)
        return self._parse_json_list(response_text)

    # ─────────────────────────────────────────────────────
    # Coreference resolution
    # ─────────────────────────────────────────────────────

    async def resolve_coreferences(
        self,
        query: str,
        context_summary: str,
    ) -> dict[str, str]:
        """
        Resolve pronouns and vague references using session context.

        Given the current query and a summary of what's been
        discussed so far, identifies references like "it", "that
        machine", "those parts" and maps them to specific entity IDs.

        Args:
            query: The current user question.
            context_summary: Output of ContextState.to_context_string().

        Returns:
            Dict mapping reference text → entity_id.
            Example: {"it": "machine_m400", "that batch": "batch_b442"}
            Empty dict if no coreferences found.
        """
        system_prompt = (
            "You are a coreference resolution system. Given a user query and "
            "the conversation context, identify any pronouns or vague references "
            "(like 'it', 'that machine', 'those', 'the same batch', etc.) and "
            "map them to specific entity IDs from the context.\n\n"
            "Context:\n"
            f"{context_summary}\n\n"
            "Return ONLY a JSON object mapping reference → entity_id.\n"
            "Example: {\"it\": \"machine_m400\", \"that batch\": \"batch_b442\"}\n"
            "If no coreferences found, return: {}"
        )

        response_text = await self._invoke(system_prompt, query)
        return self._parse_json_dict(response_text)

    # ─────────────────────────────────────────────────────
    # Query augmentation
    # ─────────────────────────────────────────────────────

    async def augment_query(
        self,
        query: str,
        resolved_refs: dict[str, str],
        context_summary: str,
    ) -> str:
        """
        Rewrite a vague query into an explicit one using context.

        Replaces pronouns with entity names and adds relevant
        context so the retrieval step can find the right subgraph.

        Example:
            Input:  "When was it last serviced?"
            Output: "When was CNC Mill M-400 last serviced?"

        Args:
            query: The original user question.
            resolved_refs: Coreference mappings from resolve_coreferences().
            context_summary: Current session context.

        Returns:
            The augmented query string. Returns original query
            if no augmentation is needed.
        """
        # If no references to resolve, return as-is
        if not resolved_refs:
            return query

        system_prompt = (
            "You are a query rewriting system. Rewrite the user's query by "
            "replacing all pronouns and vague references with their specific "
            "entity names. Keep the question's intent identical.\n\n"
            "Coreference mappings:\n"
            f"{json.dumps(resolved_refs, indent=2)}\n\n"
            "Conversation context:\n"
            f"{context_summary}\n\n"
            "Return ONLY the rewritten query as plain text. No explanation."
        )

        augmented = await self._invoke(system_prompt, query)

        # Fallback to original if LLM returns empty or garbage
        if not augmented or len(augmented) < 5:
            return query

        logger.debug("Query augmented: '%s' → '%s'", query, augmented)
        return augmented

    # ─────────────────────────────────────────────────────
    # RAG generation
    # ─────────────────────────────────────────────────────

    async def generate_response(
        self,
        query: str,
        subgraph_context: str,
        context_summary: Optional[str] = None,
    ) -> str:
        """
        Generate an answer using retrieved subgraph as context.

        This is the main generation method used by both RAG
        pipelines. The basic pipeline passes only subgraph_context.
        The context pipeline also passes context_summary for
        richer, state-aware answers.

        Args:
            query: The user's question (possibly augmented).
            subgraph_context: Serialized subgraph from retrieval
                (output of SubGraph.to_context_string()).
            context_summary: Optional session context for the
                stateful pipeline.

        Returns:
            The generated answer string.
        """
        # Build system prompt with available context
        system_parts = [
            "You are a manufacturing plant assistant. Answer the user's "
            "question using ONLY the provided knowledge graph context. "
            "Be specific — cite machine names, part numbers, dates, and "
            "measurements when available.",
            "",
            "Knowledge graph context:",
            subgraph_context,
        ]

        # Add session context for the stateful pipeline
        if context_summary:
            system_parts.extend([
                "",
                "Conversation context (what has been discussed so far):",
                context_summary,
                "",
                "Use the conversation context to:",
                "- Provide continuity with previous answers",
                "- Reference previously discussed entities naturally",
                "- Flag related information the user hasn't asked about yet",
                "- Build on the RCA investigation chain if one is in progress",
            ])

        system_parts.extend([
            "",
            "If the context doesn't contain enough information to answer, "
            "say so clearly. Do not fabricate information.",
        ])

        system_prompt = "\n".join(system_parts)
        return await self._invoke(system_prompt, query)

    # ─────────────────────────────────────────────────────
    # RCA-specific generation
    # ─────────────────────────────────────────────────────

    async def analyze_rca_step(
        self,
        query: str,
        subgraph_context: str,
        investigation_summary: dict,
    ) -> dict:
        """
        Generate an RCA-aware response with structured metadata.

        Beyond answering the question, this method asks the LLM
        to identify new suspected causes, entities to rule out,
        and entities flagged for follow-up investigation.

        Args:
            query: The user's RCA-related question.
            subgraph_context: Serialized retrieval context.
            investigation_summary: Output of
                ContextRepository.get_investigation_summary().

        Returns:
            Dict with keys:
                - "answer": The natural language response.
                - "suspected_causes": List of {source, target} dicts.
                - "ruled_out": List of {source, target, reason} dicts.
                - "follow_ups": List of {entity, reason} dicts.
                - "new_entities": List of entity IDs to track.
        """
        system_prompt = (
            "You are an RCA (root cause analysis) assistant for a "
            "manufacturing plant. Answer the user's question and also "
            "provide structured investigation metadata.\n\n"
            "Knowledge graph context:\n"
            f"{subgraph_context}\n\n"
            "Current investigation state:\n"
            f"{json.dumps(investigation_summary, indent=2)}\n\n"
            "Return a JSON object with these keys:\n"
            "- \"answer\": Your natural language response to the user.\n"
            "- \"suspected_causes\": Array of {\"source\": \"...\", \"target\": \"...\"} "
            "for any new causal links discovered.\n"
            "- \"ruled_out\": Array of {\"source\": \"...\", \"target\": \"...\", "
            "\"reason\": \"...\"} for eliminated causes.\n"
            "- \"follow_ups\": Array of {\"entity\": \"...\", \"reason\": \"...\"} "
            "for things worth investigating next.\n"
            "- \"new_entities\": Array of entity ID strings discussed in this response.\n\n"
            "Return ONLY valid JSON. No markdown, no explanation outside the JSON."
        )

        response_text = await self._invoke(system_prompt, query)

        # Parse structured response
        try:
            result = json.loads(self._clean_json_response(response_text))
            # Ensure all expected keys exist
            return {
                "answer": result.get("answer", response_text),
                "suspected_causes": result.get("suspected_causes", []),
                "ruled_out": result.get("ruled_out", []),
                "follow_ups": result.get("follow_ups", []),
                "new_entities": result.get("new_entities", []),
            }
        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, treat entire response as the answer
            logger.warning("Failed to parse RCA JSON response, using raw text")
            return {
                "answer": response_text,
                "suspected_causes": [],
                "ruled_out": [],
                "follow_ups": [],
                "new_entities": [],
            }

    # ─────────────────────────────────────────────────────
    # Health check
    # ─────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """
        Verify the Ollama connection is working.

        Sends a minimal prompt and measures round-trip time.

        Returns:
            Dict with "status", "model", and "latency_ms".
        """
        start = time.perf_counter()
        try:
            response = await self._llm.ainvoke(
                [HumanMessage(content="Reply with OK")]
            )
            latency = (time.perf_counter() - start) * 1000
            return {
                "status": "healthy",
                "model": self._model_name,
                "latency_ms": round(latency, 1),
                "response": response.content.strip(),
            }
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.error("LLM health check failed: %s", str(e))
            return {
                "status": "unhealthy",
                "model": self._model_name,
                "latency_ms": round(latency, 1),
                "error": str(e),
            }

    # ─────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────

    async def _invoke(self, system_prompt: str, user_message: str) -> str:
        """
        Send a system + user message pair to the LLM.

        All LLM calls in this class route through this method,
        making it easy to add logging, retries, or caching later.

        Args:
            system_prompt: The system instruction.
            user_message: The user's input.

        Returns:
            The model's response text (stripped of whitespace).
        """
        start = time.perf_counter()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        response = await self._llm.ainvoke(messages)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "LLM call completed in %.1fms (%d chars response)",
            elapsed_ms,
            len(response.content),
        )

        return response.content.strip()

    @staticmethod
    def _clean_json_response(text: str) -> str:
        """
        Strip markdown code fences from LLM JSON output.

        LLMs often wrap JSON in ```json ... ``` blocks even
        when told not to. This removes those wrappers.

        Args:
            text: Raw LLM response text.

        Returns:
            Cleaned text ready for json.loads().
        """
        cleaned = text.strip()

        # Remove ```json ... ``` wrappers
        if cleaned.startswith("```"):
            # Find the end of the first line (```json or ```)
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return cleaned.strip()

    def _parse_json_list(self, text: str) -> list[str]:
        """
        Parse LLM response as a JSON list of strings.

        Handles common LLM quirks: markdown fences, extra
        whitespace, and non-list responses.

        Args:
            text: Raw LLM response text.

        Returns:
            List of strings. Empty list on parse failure.
        """
        try:
            parsed = json.loads(self._clean_json_response(text))
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return []
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse JSON list from: %s", text[:200])
            return []

    def _parse_json_dict(self, text: str) -> dict[str, str]:
        """
        Parse LLM response as a JSON dict of string → string.

        Args:
            text: Raw LLM response text.

        Returns:
            Dict of string mappings. Empty dict on parse failure.
        """
        try:
            parsed = json.loads(self._clean_json_response(text))
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
            return {}
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse JSON dict from: %s", text[:200])
            return {}