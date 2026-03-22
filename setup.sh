# Core framework
uv add fastapi uvicorn[standard] websockets

# LangChain + LangGraph
uv add langchain langchain-community langgraph

# Ollama integration
uv add langchain-ollama ollama

# Graph store
uv add networkx

# Data & utilities
uv add pydantic pydantic-settings python-dotenv

# Rich console output (for CLI testing)
uv add rich

# ─────────────────────────────────────────────────────────
# 4. Dev dependencies
# ─────────────────────────────────────────────────────────
uv add --dev pytest pytest-asyncio httpx ruff

# ─────────────────────────────────────────────────────────
# 5. Create folder structure
# ─────────────────────────────────────────────────────────
mkdir -p domain data repositories services api tests

touch domain/__init__.py domain/enums.py domain/models.py
touch data/__init__.py data/synthetic_factory.py
touch repositories/__init__.py repositories/graph_repo.py repositories/context_repo.py repositories/llm_repo.py
touch services/__init__.py services/rag_factory.py services/basic_graph_rag.py services/context_graph_rag.py services/evaluation.py
touch api/__init__.py api/router.py api/schemas.py api/websocket.py
touch tests/__init__.py tests/test_graph_repo.py tests/test_rca_scenario.py
touch config.py main.py .env

# ─────────────────────────────────────────────────────────
# 6. Setup .env file
# ─────────────────────────────────────────────────────────
cat > .env << 'EOF'
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
GRAPH_HOP_DEPTH=2
LOG_LEVEL=INFO
EOF

# ─────────────────────────────────────────────────────────
# 7. Make sure Ollama is running with the model
# ─────────────────────────────────────────────────────────
ollama pull llama3.2

# ─────────────────────────────────────────────────────────
# 8. Verify everything
# ─────────────────────────────────────────────────────────
uv run python -c "
import fastapi, langchain, langgraph, networkx, ollama
print('All packages installed successfully!')
print(f'  FastAPI:    {fastapi.__version__}')
print(f'  NetworkX:   {networkx.__version__}')
print(f'  LangChain:  {langchain.__version__}')