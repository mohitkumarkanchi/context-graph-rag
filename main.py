"""
Application entry point.

Wires together all layers — repositories, services, and routes —
and starts the FastAPI server. Uses lifespan events for clean
startup and shutdown.

Run with:
    uv run uvicorn main:app --reload
    uv run python main.py
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import router as api_router
from api import websocket as ws_module
from config import get_settings
from data.synthetic_factory import SyntheticFactory
from repositories.context_repo import ContextRepository
from repositories.graph_repo import GraphRepository
from repositories.llm_repo import LLMRepository
from services.evaluation import EvaluationService
from services.rag_factory import RAGFactory

# ─────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Lifespan — startup and shutdown
# ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup:
        1. Generate synthetic manufacturing data
        2. Load it into the knowledge graph
        3. Initialize all repositories
        4. Create the RAG factory and evaluation service
        5. Inject dependencies into routers

    Shutdown:
        Logs a clean shutdown message. In-memory state
        is discarded (sessions, context graphs).
    """
    logger.info("=" * 60)
    logger.info("Starting Context Graph RAG Demo")
    logger.info("=" * 60)

    # ── Step 1: Generate synthetic data ──

    logger.info("Generating synthetic manufacturing data...")
    factory = SyntheticFactory()
    entities, relationships = factory.build()
    logger.info(
        "Generated %d entities and %d relationships",
        len(entities),
        len(relationships),
    )

    # ── Step 2: Initialize repositories ──

    logger.info("Initializing repositories...")

    graph_repo = GraphRepository()
    graph_repo.load(entities, relationships)

    context_repo = ContextRepository()

    logger.info("Initializing LLM repository (model: %s)...", settings.ollama_model)
    llm_repo = LLMRepository()

    # ── Step 3: Create services ──

    logger.info("Creating RAG factory and services...")
    rag_factory = RAGFactory(
        graph_repo=graph_repo,
        context_repo=context_repo,
        llm_repo=llm_repo,
    )

    eval_service = EvaluationService(factory=rag_factory)

    # ── Step 4: Inject dependencies into routers ──

    logger.info("Injecting dependencies into routes...")

    api_router.setup(
        factory=rag_factory,
        graph_repo=graph_repo,
        context_repo=context_repo,
        llm_repo=llm_repo,
        eval_service=eval_service,
    )

    ws_module.setup(
        factory=rag_factory,
        context_repo=context_repo,
    )

    # ── Step 5: Verify LLM connection ──

    logger.info("Checking LLM connection...")
    health = await llm_repo.health_check()
    if health["status"] == "healthy":
        logger.info(
            "LLM ready: %s (%.0fms latency)",
            health["model"],
            health["latency_ms"],
        )
    else:
        logger.warning(
            "LLM health check failed: %s — "
            "the app will start but queries may fail",
            health.get("error", "unknown"),
        )

    logger.info("=" * 60)
    logger.info("Server ready at http://%s:%d", settings.api_host, settings.api_port)
    logger.info("API docs at http://%s:%d/docs", settings.api_host, settings.api_port)
    logger.info("=" * 60)

    # Hand control to the app
    yield

    # ── Shutdown ──

    logger.info("Shutting down Context Graph RAG Demo")
    logger.info(
        "Active sessions at shutdown: %d",
        len(context_repo.list_sessions()),
    )


# ─────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Sets up CORS, includes all routers, and mounts the
    static files directory for the frontend.

    Returns:
        Configured FastAPI app instance.
    """
    app = FastAPI(
        title="Context Graph RAG Demo",
        description=(
            "A side-by-side comparison of basic Graph RAG vs "
            "Graph RAG with a context graph. Demonstrates how "
            "stateful context improves multi-turn conversations "
            "and root cause analysis in manufacturing."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS ──
    #
    # Allow all origins in dev mode so the frontend
    # can reach the API regardless of how it's served.

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──

    app.include_router(
        api_router.router,
        prefix="/api",
        tags=["api"],
    )
    app.include_router(
        ws_module.ws_router,
        tags=["websocket"],
    )

    # ── Static files (frontend) ──
    #
    # Mount AFTER API routes so /api/* takes priority.
    # Use check_dir=False so the app starts even if
    # the frontend directory doesn't exist yet.

    import os
    if os.path.isdir("frontend"):
        app.mount(
            "/",
            StaticFiles(directory="frontend", html=True),
            name="frontend",
        )
        logger.info("Frontend static files mounted from ./frontend/")
    else:
        logger.info(
            "No frontend directory found — "
            "API-only mode (use /docs for testing)"
        )

    return app


# ─────────────────────────────────────────────────────────
# App instance
# ─────────────────────────────────────────────────────────

app = create_app()


# ─────────────────────────────────────────────────────────
# Direct execution
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )