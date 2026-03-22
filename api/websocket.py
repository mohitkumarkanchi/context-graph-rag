"""
WebSocket endpoint for streaming chat responses.

Provides a persistent connection for the frontend to send
queries and receive responses in real time. Supports both
basic and context RAG modes over the same connection.

Protocol:
    Client → Server (JSON):
        {
            "type": "query",
            "query": "What machines are on Line A?",
            "mode": "context",
            "session_id": "abc-123",        // optional
            "turn_number": 1                 // optional
        }

    Server → Client (JSON):
        {
            "type": "status",
            "step": "extracting_entities",
            "message": "Extracting entities from query..."
        }
        {
            "type": "response",
            "data": { ...ChatResponse fields... }
        }
        {
            "type": "context_update",
            "data": { ...context graph JSON... }
        }
        {
            "type": "error",
            "message": "Pipeline execution failed: ..."
        }
"""

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from domain.enums import RAGMode
from domain.models import PipelineState
from repositories.context_repo import ContextRepository
from services.rag_factory import RAGFactory

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Router instance
# ─────────────────────────────────────────────────────────

ws_router = APIRouter()

# ─────────────────────────────────────────────────────────
# Service references (injected at startup)
# ─────────────────────────────────────────────────────────

_factory: Optional[RAGFactory] = None
_context_repo: Optional[ContextRepository] = None

# Turn tracking per session (mirrors router.py)
_session_turns: dict[str, int] = {}


def setup(
    factory: RAGFactory,
    context_repo: ContextRepository,
) -> None:
    """
    Inject dependencies into the WebSocket module.

    Called once during app startup alongside router.setup().

    Args:
        factory: RAG pipeline factory.
        context_repo: Session context repository.
    """
    global _factory, _context_repo
    _factory = factory
    _context_repo = context_repo
    logger.info("WebSocket dependencies injected")


def _get_next_turn(session_id: str) -> int:
    """Auto-increment and return next turn number."""
    current = _session_turns.get(session_id, 0)
    next_turn = current + 1
    _session_turns[session_id] = next_turn
    return next_turn


# ─────────────────────────────────────────────────────────
# Connection manager
# ─────────────────────────────────────────────────────────

class ConnectionManager:
    """
    Tracks active WebSocket connections.

    Provides methods to broadcast messages to all connected
    clients or send to a specific connection. Also handles
    graceful disconnect cleanup.
    """

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept a new WebSocket connection.

        Assigns a unique connection ID and stores the socket.

        Args:
            websocket: The incoming WebSocket connection.

        Returns:
            The assigned connection ID.
        """
        await websocket.accept()
        conn_id = str(uuid.uuid4())[:8]
        self._connections[conn_id] = websocket
        logger.info("WebSocket connected: %s", conn_id)
        return conn_id

    def disconnect(self, conn_id: str) -> None:
        """
        Remove a connection from the manager.

        Args:
            conn_id: The connection to remove.
        """
        self._connections.pop(conn_id, None)
        logger.info("WebSocket disconnected: %s", conn_id)

    async def send_json(self, conn_id: str, data: dict) -> None:
        """
        Send a JSON message to a specific connection.

        Silently ignores if the connection no longer exists.

        Args:
            conn_id: Target connection.
            data: Dict to serialize and send.
        """
        ws = self._connections.get(conn_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception as e:
                logger.warning(
                    "Failed to send to %s: %s", conn_id, str(e)
                )
                self.disconnect(conn_id)

    @property
    def active_count(self) -> int:
        """Return the number of active connections."""
        return len(self._connections)


# Singleton manager instance
manager = ConnectionManager()


# ─────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────

@ws_router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """
    Main WebSocket endpoint for streaming chat.

    Maintains a persistent connection. The client sends
    query messages, and the server responds with status
    updates, the final response, and context graph updates.
    """
    conn_id = await manager.connect(websocket)

    # Send welcome message
    await manager.send_json(conn_id, {
        "type": "connected",
        "connection_id": conn_id,
        "message": "Connected to Graph RAG WebSocket",
    })

    try:
        while True:
            # Wait for client message
            raw = await websocket.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json(conn_id, {
                    "type": "error",
                    "message": "Invalid JSON message",
                })
                continue

            # Route by message type
            msg_type = message.get("type", "")

            if msg_type == "query":
                await _handle_query(conn_id, message)
            elif msg_type == "ping":
                await manager.send_json(conn_id, {"type": "pong"})
            else:
                await manager.send_json(conn_id, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        manager.disconnect(conn_id)
    except Exception as e:
        logger.error("WebSocket error (%s): %s", conn_id, str(e))
        manager.disconnect(conn_id)


# ─────────────────────────────────────────────────────────
# Query handler
# ─────────────────────────────────────────────────────────

async def _handle_query(conn_id: str, message: dict) -> None:
    """
    Process a query message from the client.

    Runs the RAG pipeline and sends status updates at each
    major step so the frontend can show progress.

    Args:
        conn_id: The connection to respond to.
        message: The parsed query message.
    """
    if _factory is None:
        await manager.send_json(conn_id, {
            "type": "error",
            "message": "Service not initialized",
        })
        return

    # ── Parse request fields ──

    query = message.get("query", "").strip()
    if not query:
        await manager.send_json(conn_id, {
            "type": "error",
            "message": "Empty query",
        })
        return

    mode_str = message.get("mode", "context")
    try:
        mode = RAGMode(mode_str)
    except ValueError:
        await manager.send_json(conn_id, {
            "type": "error",
            "message": f"Invalid mode: {mode_str}. Use 'basic' or 'context'.",
        })
        return

    session_id = message.get("session_id") or str(uuid.uuid4())
    turn_number = message.get("turn_number") or _get_next_turn(session_id)

    # ── Send status: starting ──

    await manager.send_json(conn_id, {
        "type": "status",
        "step": "starting",
        "message": f"Processing query with {mode.value} pipeline...",
        "session_id": session_id,
        "turn_number": turn_number,
    })

    # ── Get the service and run the pipeline ──

    service = _factory.create(mode)

    state = PipelineState(
        query=query,
        session_id=session_id,
        turn_number=turn_number,
    )

    try:
        # Send status: extracting
        await manager.send_json(conn_id, {
            "type": "status",
            "step": "extracting_entities",
            "message": "Extracting entities from query...",
        })

        # Run the full pipeline
        state = await service.query(state)

        # Send status: complete
        await manager.send_json(conn_id, {
            "type": "status",
            "step": "complete",
            "message": "Response generated",
        })

    except Exception as e:
        logger.error(
            "Pipeline error via WebSocket (mode=%s, session=%s): %s",
            mode.value,
            session_id,
            str(e),
        )
        await manager.send_json(conn_id, {
            "type": "error",
            "message": f"Pipeline execution failed: {str(e)}",
        })
        return

    # ── Send the response ──

    await manager.send_json(conn_id, {
        "type": "response",
        "data": {
            "response": state.response,
            "mode": mode.value,
            "session_id": session_id,
            "turn_number": turn_number,
            "extracted_entities": state.extracted_entities,
            "augmented_query": state.augmented_query,
            "sources": state.sources,
            "retrieval_time_ms": state.retrieval_time_ms,
            "generation_time_ms": state.generation_time_ms,
        },
    })

    # ── Send context graph update (context mode only) ──

    if mode == RAGMode.CONTEXT and _context_repo is not None:
        context_graph = _context_repo.to_json_graph(session_id)
        investigation = _context_repo.get_investigation_summary(session_id)

        await manager.send_json(conn_id, {
            "type": "context_update",
            "data": {
                "graph": context_graph,
                "investigation": investigation,
            },
        })