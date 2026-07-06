# 第 5 章 RAG 对话服务

## 5.1 ChatQwen vs ChatOpenAI

项目用了两种 LLM 接入方式：

| 组件 | 接入方式 | 原因 |
|------|---------|------|
| RAG Agent | `ChatQwen`（langchain-qwq） | 原生集成，流式支持更好 |
| AIOps Agent | `ChatQwen`（同上） | 统一使用 |
| LLMFactory | `ChatOpenAI`（langchain-openai） | OpenAI 兼容模式，方便切换提供商 |

## 5.2 RAG Agent 初始化

```python
# app/services/rag_agent_service.py（关键部分）
class RagAgentService:
    def __init__(self, streaming: bool = True):
        self.model = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=streaming,
        )
        self.tools = [retrieve_knowledge, get_current_time]  # 本地工具
        self.mcp_tools = []  # MCP 工具（延迟加载）
        self.checkpointer = MemorySaver()  # 会话持久化
        self.agent = None
        self._agent_initialized = False

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()
        self.mcp_tools = mcp_tools

        # 合并所有工具
        all_tools = self.tools + self.mcp_tools

        # 创建 LangGraph Agent
        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )
        self._agent_initialized = True
```

**要点 — 延迟初始化**：Agent 在首次查询时才真正构建，因为 MCP 工具需要异步获取。这确保启动时 MCP 服务器不必已就绪，只要首次查询时可用即可。

## 5.3 MCP 客户端（单例 + 重试拦截器）

```python
# app/agent/mcp_client.py
_mcp_client: Optional[MultiServerMCPClient] = None


async def retry_interceptor(request, handler, max_retries=3, delay=1.0):
    """MCP 工具调用重试拦截器（指数退避）"""
    last_error = None
    for attempt in range(max_retries):
        try:
            result = await handler(request)
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # 1s, 2s, 4s
                await asyncio.sleep(wait_time)

    # 所有重试失败，返回错误结果（不抛异常）
    return CallToolResult(
        content=[TextContent(type="text", text=f"工具 {request.name} 重试失败: {last_error}")],
        isError=True
    )


async def get_mcp_client_with_retry():
    """获取带重试的 MCP 客户端（单例）"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MultiServerMCPClient(
            config.mcp_servers,  # {"cls": {...}, "monitor": {...}}
            tool_interceptors=[retry_interceptor],
        )
    return _mcp_client
```

**要点**：
- **单例模式**：整个应用共享一个 MCP 客户端
- **重试拦截器**：指数退避（1s → 2s → 4s），失败返回错误结果而非抛异常，保证 Agent 流程不被中断
- **拦截器链**：`retry_interceptor` 始终在链头，自定义拦截器在其后

## 5.4 会话管理

```python
# MemorySaver 用 thread_id 隔离不同会话
config_dict = {"configurable": {"thread_id": session_id}}

# 查询
result = await self.agent.ainvoke(input=agent_input, config=config_dict)

# 流式
async for token, metadata in self.agent.astream(
    input=agent_input, config=config_dict, stream_mode="messages"
):
    ...

# 获取会话历史
checkpoint_tuple = self.checkpointer.get(config_dict)
messages = checkpoint_tuple.checkpoint["channel_values"]["messages"]

# 清空会话
self.checkpointer.delete_thread(session_id)
```

## 5.5 流式输出（SSE）

API 层使用 `sse-starlette` 将 Agent 流式输出转为 SSE：

```python
# app/api/chat.py
@router.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        async for chunk in rag_agent_service.query_stream(request.question, session_id=request.id):
            if chunk["type"] == "content":
                yield {"event": "message", "data": json.dumps({"type": "content", "data": chunk["data"]})}
            elif chunk["type"] == "complete":
                yield {"event": "message", "data": json.dumps({"type": "done"})}
            # ...

    return EventSourceResponse(event_generator())
```

**RAG 服务的流式实现**：

```python
async def query_stream(self, question, session_id):
    async for token, metadata in self.agent.astream(
        input=agent_input, config=config_dict, stream_mode="messages"
    ):
        if type(token).__name__ in ("AIMessage", "AIMessageChunk"):
            content_blocks = getattr(token, 'content_blocks', None)
            if content_blocks:
                for block in content_blocks:
                    if block.get('type') == 'text':
                        yield {"type": "content", "data": block.get('text', '')}
    yield {"type": "complete"}
```

**要点**：
- RAG 用 `stream_mode="messages"`，AIOps 用 `stream_mode="updates"`，两者不同
- RAG 的流式内容在 `AIMessageChunk.content_blocks` 中，需要遍历提取 `type == "text"` 的块
