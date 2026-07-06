# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

SuperBizAgent is an enterprise-grade intelligent oncall operations assistant with two core capabilities:
- **RAG Chat**: Multi-turn conversation with vector-retrieval augmented knowledge base (LangChain + LangGraph + Milvus)
- **AIOps Diagnosis**: Automated fault diagnosis using a Plan-Execute-Replan pattern with MCP tool integration

## Tech Stack

- **Framework**: FastAPI + LangChain + LangGraph
- **LLM**: Alibaba Cloud DashScope (Qwen models) via OpenAI-compatible API
- **Vector DB**: Milvus (Docker)
- **Tool Protocol**: MCP (Model Context Protocol) via `langchain-mcp-adapters` + `fastmcp`
- **Logging**: Loguru
- **Package Manager**: uv (preferred) or pip

## Common Commands

### Setup & Lifecycle
```bash
make init              # Full initialization: Docker (Milvus) → start services → upload docs
make start             # Start all services (CLS MCP + Monitor MCP + FastAPI)
make stop              # Stop all services
make restart           # Restart all services
make dev               # Dev mode with hot-reload (foreground, port 9900)
make run               # Production mode (foreground, port 9900)
```

On Windows without `make`, use `start-windows.bat` / `stop-windows.bat`.

### Dependency Management
```bash
uv pip install -e .            # Install production deps
uv pip install -e ".[dev]"     # Install dev deps
```

### Code Quality
```bash
make format            # Ruff import sort + Black formatting
make lint              # Ruff check
make fix               # Ruff auto-fix + format
make test              # pytest with coverage
make test-quick        # pytest without coverage
make check-all         # format + lint + test
```

### Individual Test
```bash
pytest tests/test_<name>.py -v
```

### Docker (Milvus)
```bash
docker compose -f vector-database.yml up -d     # Start Milvus
docker compose -f vector-database.yml down       # Stop Milvus
```

### MCP Servers (separate processes)
```bash
python mcp_servers/cls_server.py      # CLS log service (port 8003)
python mcp_servers/monitor_server.py  # Monitor service (port 8004)
```

## Architecture

### Service Topology

Three independent services run together:
1. **FastAPI app** (`app/main.py`, port 9900) — the main API server
2. **CLS MCP Server** (`mcp_servers/cls_server.py`, port 8003) — log query tools
3. **Monitor MCP Server** (`mcp_servers/monitor_server.py`, port 8004) — monitoring data tools

The FastAPI app connects to MCP servers as a client via `langchain-mcp-adapters`.

### Request Flow

**RAG Chat** (`/api/chat`, `/api/chat_stream`):
- `app/api/chat.py` → `RagAgentService` (`app/services/rag_agent_service.py`)
- Uses LangGraph `create_agent` with tools: `retrieve_knowledge` (Milvus vector search) + `get_current_time` + MCP tools
- Session state managed by `MemorySaver` checkpointer (thread_id = session_id)
- Streaming uses `agent.astream()` with `stream_mode="messages"`

**AIOps Diagnosis** (`/api/aiops`):
- `app/api/aiops.py` → `AIOpsService` (`app/services/aiops_service.py`)
- LangGraph `StateGraph` with three nodes: `planner` → `executor` → `replanner`
- State defined in `app/agent/aiops/state.py` as `PlanExecuteState`
- `planner` generates diagnostic steps, `executor` calls MCP tools, `replanner` evaluates and decides next action
- Streaming via `graph.astream()` with `stream_mode="updates"`

### Key Design Patterns

- **LLM Initialization**: `LLMFactory` (`app/core/llm_factory.py`) creates `ChatOpenAI` instances pointing to DashScope's OpenAI-compatible endpoint. The RAG service uses `ChatQwen` from `langchain-qwq` directly instead.
- **MCP Client**: Singleton pattern in `app/agent/mcp_client.py`. Uses `MultiServerMCPClient` with a retry interceptor (exponential backoff). Configured via `config.mcp_servers` dict from `.env`.
- **Config**: Pydantic Settings (`app/config.py`) loads from `.env`. All env vars are lowercase in code (e.g., `DASHSCOPE_API_KEY` → `config.dashscope_api_key`).
- **Vector Pipeline**: Document upload (`/api/upload`) → `DocumentSplitterService` splits text → `VectorEmbeddingService` generates embeddings → `VectorIndexService` stores in Milvus.

### Directory Layout

- `app/api/` — FastAPI route handlers (chat, aiops, file upload, health)
- `app/services/` — Business logic (RAG agent, AIOps, vector operations)
- `app/agent/` — Agent internals (MCP client, AIOps planner/executor/replanner)
- `app/core/` — Infrastructure (LLM factory, Milvus client)
- `app/tools/` — LangChain tools (knowledge retrieval, time)
- `app/models/` — Pydantic request/response models
- `mcp_servers/` — Standalone MCP server implementations (FastMCP)
- `aiops-docs/` — Markdown knowledge base docs (uploaded to Milvus on init)
- `static/` — Frontend (HTML/JS/CSS)

## Environment Configuration

Required in `.env`:
```
DASHSCOPE_API_KEY=<your-key>
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-max
MILVUS_HOST=localhost
MILVUS_PORT=19530
RAG_TOP_K=3
CHUNK_MAX_SIZE=800
CHUNK_OVERLAP=100
```

MCP server URLs are also configurable in `.env` (default: `http://localhost:8003/mcp` for CLS, `http://localhost:8004/mcp` for Monitor).

## Code Style

- Python 3.11 target, line length 100
- Formatting: Black + Ruff (isort)
- Linting: Ruff (E, W, F, I, C, B, UP rules)
- Pre-commit hooks configured (see `.pre-commit-config.yaml`)
- `__init__.py` files ignore unused import warnings (F401)
