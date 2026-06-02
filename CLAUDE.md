# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**super-biz-agent-py** — 基于 LangChain 的智能业务代理系统，支持 RAG 知识库和 AIOps 智能运维。项目使用 FastAPI 提供 HTTP/SSE API，通过 LangGraph 编排多 Agent 工作流，对接 DashScope (Qwen) 作为 LLM 后端，使用 Milvus 作为向量数据库实现 RAG，并通过 MCP 协议连接外部运维工具（CLS 日志服务、监控服务、Prometheus）。

## Development Commands

```bash
# 环境管理（使用 uv）
uv sync              # 安装主依赖
uv sync --dev        # 安装主依赖 + 开发依赖
uv venv              # 创建虚拟环境
uv add <package>     # 添加依赖
uv add --dev <pkg>   # 添加开发依赖

# 运行应用（默认端口 9900）
uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 9900 --reload

# 运行 MCP Server（需先于主应用启动）
uv run python mcp_servers/cls_server.py      # CLS 日志服务，端口 8003
uv run python mcp_servers/monitor_server.py   # 监控服务，端口 8004

# 代码检查
uv run ruff check .
uv run ruff check --fix .
uv run black .
uv run isort .
uv run mypy app/

# 测试
uv run pytest
uv run pytest tests/test_xxx.py            # 单个测试文件
uv run pytest tests/test_xxx.py::test_fn   # 单个测试函数
uv run pytest -k "keyword"                 # 按关键字过滤

# Python 版本要求: >=3.11, <3.14
```

## Architecture

```
app/
├── main.py                    # FastAPI 入口，lifespan 管理 Milvus 连接生命周期
├── config.py                  # pydantic-settings 配置，从 .env 加载
├── core/
│   ├── llm_factory.py         # LLM 工厂，ChatOpenAI 通过 OpenAI 兼容模式调 DashScope
│   └── milvus_client.py       # Milvus 客户端单例，管理连接/Collection/索引
├── agent/
│   ├── aiops/                 # Plan-Execute-Replan Agent（核心工作流）
│   │   ├── state.py           #   PlanExecuteState 状态定义（input, plan, past_steps, response）
│   │   ├── planner.py         #   Planner 节点：先 RAG 检索经验，再制定步骤计划
│   │   ├── executor.py        #   Executor 节点：ToolNode 自动执行工具调用
│   │   ├── replanner.py       #   Replanner 节点：continue / replan / respond 三路决策
│   │   └── utils.py           #   format_tools_description 工具描述格式化
│   └── mcp_client.py          # MCP 客户端单例，含重试拦截器和 transport 建议
├── api/
│   ├── chat.py                # /api/chat, /api/chat_stream (SSE), /api/chat/clear, /api/chat/session/{id}
│   ├── aiops.py               # /api/aiops (SSE 流式诊断)
│   ├── file.py                # /api/upload, /api/index_directory（上传文件→自动建索引）
│   └── health.py              # /health（含 Milvus 连通性检查）
├── models/                    # Pydantic 请求/响应模型
├── services/
│   ├── rag_agent_service.py   # RAG Agent 服务，用 langchain.agents.create_agent + ChatQwen
│   ├── aiops_service.py       # AIOps 服务，构建 StateGraph(planner→executor→replanner) 工作流
│   ├── vector_store_manager.py    # LangChain Milvus VectorStore 封装，add_documents / delete_by_source
│   ├── vector_index_service.py    # 文件→分片→向量化→入库的索引流水线
│   ├── vector_embedding_service.py # DashScope Embeddings (OpenAI 兼容模式, text-embedding-v4, 1024维)
│   ├── vector_search_service.py   # 原生 Milvus ORM 向量检索
│   └── document_splitter_service.py # Markdown 标题分割 + RecursiveCharacterTextSplitter 二次分割 + 小片段合并
├── tools/
│   ├── knowledge_tool.py      # retrieve_knowledge: RAG 检索工具（response_format="content_and_artifact"）
│   ├── time_tool.py           # get_current_time: 时区时间工具
│   └── query_metrics_alerts.py # query_prometheus_alerts: Prometheus /api/v1/alerts 告警查询
└── utils/
    └── logger.py              # loguru 日志配置

mcp_servers/                   # 独立 MCP Server 进程（FastMCP streamable-http）
├── cls_server.py              # CLS 日志服务 (端口 8003): search_log, get_topic_info 等
└── monitor_server.py          # 监控服务 (端口 8004): query_cpu_metrics, query_memory_metrics 等

aiops-docs/                    # RAG 知识源，告警处理方案文档（CPU/内存/磁盘/服务不可用/响应慢）
static/                        # 前端静态文件
uploads/                       # 用户上传文件目录（txt/md）
```

### 核心工作流：Plan-Execute-Replan

AIOps 诊断的核心是 LangGraph StateGraph，状态为 `PlanExecuteState`：

```
planner → executor → replanner ──(continue)──→ executor ──→ replanner ──→ ... ──→ END
                        ↑              │                            │
                        └──(replan)────┘                            │
                        ↑                                           │
                        └───────────(respond / plan为空)────────────┘
```

- **Planner**：先调用 `retrieve_knowledge` 从 RAG 知识库检索经验文档，再结合本地工具 + MCP 工具列表生成步骤计划
- **Executor**：取计划首步骤，LLM 绑定所有工具决定是否调用，ToolNode 自动执行工具，结果追加到 `past_steps`
- **Replanner**：三路决策——`respond`（信息充足，生成最终报告）、`continue`（计划合理继续）、`replan`（调整计划，新步骤数 ≤ 剩余步骤数）
- **安全限制**：最大 8 步强制结束；≥5 步禁止 replan；指数退避重试 MCP 工具调用

### RAG Agent（对话场景）

`rag_agent_service` 使用 `langchain.agents.create_agent` + `ChatQwen`，工具集与 AIOps 相同（本地工具 + MCP 工具），通过 `MemorySaver` 维护会话历史，支持 SSE 流式输出。

### LLM 后端

| 用途 | 后端 | 模型 | 入口 |
|------|------|------|------|
| RAG 对话 / AIOps Agent | DashScope | qwen-max (config.rag_model) | ChatQwen (langchain-qwq) |
| 备用/流式对话 | DashScope | qwen-max (config.dashscope_model) | ChatOpenAI (langchain-openai, OpenAI 兼容模式) |
| Embedding | DashScope | text-embedding-v4 | DashScopeEmbeddings (OpenAI 兼容模式, 1024 维) |

### Milvus 向量数据库

- Collection: `biz`，向量维度 1024，索引 IVF_FLAT (L2)
- 字段: `id` (VARCHAR PK), `vector` (FLOAT_VECTOR), `content` (VARCHAR), `metadata` (JSON)
- `milvus_client.py` 含 pymilvus MilvusClient ORM 别名补丁，解决 langchain_milvus 内部创建的 `cm-{id}` 别名未注册问题

### MCP 服务

MCP 客户端为全局单例（`MultiServerMCPClient`），从 `config.mcp_servers` 读取配置，支持三种 transport：
- `stdio` — 本地进程通信
- `sse` — 远程 SSE 端点（腾讯云等 `/sse/` 端点）
- `streamable-http` — 本地 FastMCP 服务

`mcp_client.py` 内置 `retry_interceptor`（指数退避，最多 3 次）和 `suggest_mcp_transport`（URL/transport 不匹配时警告）。

### 文档索引流程

1. 上传文件 → `file.py` 保存到 `uploads/`
2. `document_splitter_service` 智能分割：Markdown 按 H1/H2 标题分割，再 RecursiveCharacterTextSplitter 二次分割，最后合并 <300 字符的小片段
3. `vector_embedding_service` (DashScope text-embedding-v4) 批量向量化
4. `vector_store_manager` 写入 Milvus `biz` collection

## API 路由

| 路由 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查（含 Milvus 连通性） |
| `/api/chat` | POST | 快速对话（RAG Agent，一次性返回） |
| `/api/chat_stream` | POST | 流式对话（SSE） |
| `/api/chat/clear` | POST | 清空会话历史 |
| `/api/chat/session/{id}` | GET | 查询会话历史 |
| `/api/aiops` | POST | AIOps 诊断（SSE 流式） |
| `/api/upload` | POST | 上传文件并自动建向量索引 |
| `/api/index_directory` | POST | 索引指定目录下所有文件 |

## Key Dependencies

- `langchain` + `langgraph` — Agent 框架与 Plan-Execute-Replan 工作流编排
- `langchain-qwq` — Qwen 模型原生集成（ChatQwen，用于 Agent 推理）
- `langchain-openai` — OpenAI 兼容模式（ChatOpenAI，用于 LLMFactory 流式对话）
- `langchain-milvus` — Milvus 向量存储
- `langchain-mcp-adapters` — MCP 工具适配（MultiServerMCPClient + 拦截器）
- `langchain-text-splitters` — 文档分割
- `fastmcp` — MCP 服务端实现（cls_server / monitor_server）
- `dashscope` — 阿里云 DashScope SDK
- `pymilvus` — Milvus Python SDK
- `sse-starlette` — SSE 流式响应
- `pydantic-settings` — 基于 `.env` 的配置管理

## Configuration

所有配置通过 `.env` 文件管理（`pydantic-settings` 加载），主要分组：
- `APP_*` — 应用基础配置（端口 9900）
- `DASHSCOPE_*` — 阿里云 DashScope API
- `OPENAI_*` — 百度千帆 API（兼容 OpenAI 接口，LLMFactory 中使用）
- `MILVUS_*` — Milvus 向量数据库连接
- `RAG_*` / `CHUNK_*` — RAG 检索与分块参数
- `MCP_*` — MCP 外部工具服务（cls / monitor）
- `PROMETHEUS_*` — Prometheus 告警查询

## 重要约定

- 所有 Service 和核心组件使用**全局单例**模式（模块级实例化）
- Milvus 连接在 `VectorStoreManager.__init__` 时提前建立（早于 FastAPI lifespan），因为模块导入时就需要访问
- Agent 工具集统一为 `DEFAULT_LOCAL_AGENT_TOOLS + MCP 工具`，RAG Agent 和 AIOps Agent 共享
- MCP Server 是独立进程，需先于主应用启动
- 文件上传仅支持 `.txt` 和 `.md`，最大 10MB
