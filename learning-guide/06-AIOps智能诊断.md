# 第 6 章 AIOps 智能诊断（核心）

这是项目最有价值的部分。基于 LangGraph 实现 **Plan-Execute-Replan** 模式。

## 6.1 模式原理

```
用户输入（任务描述）
       │
       ▼
  ┌─────────┐
  │ Planner  │  制定 4-6 步排查计划
  └────┬────┘
       │
       ▼
  ┌─────────┐     ┌───────────┐
  │ Executor │────▶│ Replanner │
  │ 调用工具  │     │ 评估结果   │
  └─────────┘     └─────┬─────┘
       ▲                │
       │    ┌───────────┼───────────┐
       │    │           │           │
       │  continue    replan     respond
       │    │        (新计划)    (出报告)
       │    │           │           │
       └────┘           └────┘      ▼
                                最终报告
```

## 6.2 状态设计

```python
# app/agent/aiops/state.py
import operator
from typing import List, TypedDict, Annotated


class PlanExecuteState(TypedDict):
    input: str                                        # 用户任务（只设一次）
    plan: List[str]                                   # 剩余步骤（覆盖更新）
    past_steps: Annotated[List[tuple], operator.add]  # 已执行步骤（追加更新！）
    response: str                                     # 最终报告（非空=终止信号）
```

**最关键的设计**：`past_steps` 使用 `Annotated[List[tuple], operator.add]`。

这意味着每个节点返回 `{"past_steps": [(task, result)]}` 时，LangGraph 不是覆盖，而是追加到列表。否则每次执行后历史就被清空了。

| 字段 | 更新语义 | 原因 |
|------|---------|------|
| `input` | 设一次 | 任务不变 |
| `plan` | 覆盖 | Planner 设全量，Executor 弹出首元素，Replanner 替换 |
| `past_steps` | 追加（operator.add） | 执行历史只能累积 |
| `response` | 覆盖 | 非空即终止 |

## 6.3 Planner — 制定计划

```python
# app/agent/aiops/planner.py
class Plan(BaseModel):
    steps: List[str] = Field(description="完成任务所需的不同步骤")


planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """作为一个专家级别的规划者，将复杂任务分解为可执行的步骤。

可用工具列表：
{tools_description}

注意：你的职责是制定计划，实际工具调用由 Executor 执行。

{experience_context}

计划应该：
- 将任务分解为逻辑独立的步骤
- 每个步骤明确使用哪些工具，最好提供工具参数
- 步骤之间有清晰依赖关系
- 如果有相关经验文档，请参考其中的方法"""),
    ("placeholder", "{messages}"),
])


async def planner(state: PlanExecuteState) -> Dict[str, Any]:
    input_text = state.get("input", "")

    # 关键：先查知识库，获取相关经验
    context_str = await retrieve_knowledge.ainvoke({"query": input_text})

    # 获取所有工具（本地 + MCP）
    mcp_client = await get_mcp_client_with_retry()
    mcp_tools = await mcp_client.get_tools()
    all_tools = [get_current_time, retrieve_knowledge] + mcp_tools
    tools_description = format_tools_description(all_tools)

    # LLM 结构化输出
    llm = ChatQwen(model=config.rag_model, api_key=config.dashscope_api_key, temperature=0)
    planner_chain = planner_prompt | llm.with_structured_output(Plan)

    plan_result = await planner_chain.ainvoke({
        "messages": [("user", input_text)],
        "tools_description": tools_description,
        "experience_context": experience_context or "",
    })

    return {"plan": plan_result.steps}
```

**要点**：
- **RAG 增强**：Planner 在制定计划前先查知识库，让计划基于组织经验而非纯 LLM 推理
- **结构化输出**：`llm.with_structured_output(Plan)` 让 LLM 直接返回 `Plan` 对象，无需手动解析
- **工具信息注入**：将可用工具列表注入 prompt，让计划能指明用哪个工具
- **容错**：异常时返回默认 3 步计划 `["收集信息", "分析数据", "生成报告"]`

## 6.4 Executor — 执行步骤（三阶段工具调用协议）

```python
# app/agent/aiops/executor.py
async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    plan = state.get("plan", [])
    task = plan[0]  # 取出第一步

    # 准备工具
    mcp_client = await get_mcp_client_with_retry()
    mcp_tools = await mcp_client.get_tools()
    all_tools = [get_current_time, retrieve_knowledge] + mcp_tools

    llm = ChatQwen(model=config.rag_model, temperature=0)
    llm_with_tools = llm.bind_tools(all_tools)
    tool_node = ToolNode(all_tools)  # LangGraph 内置工具执行器

    messages = [
        SystemMessage(content="你是一个能力强大的助手，负责执行具体任务步骤..."),
        HumanMessage(content=f"请执行以下任务: {task}")
    ]

    # ── 阶段1: LLM 决定是否调用工具 ──
    llm_response = await llm_with_tools.ainvoke(messages)

    if llm_response.tool_calls:
        # ── 阶段2: ToolNode 自动执行工具调用 ──
        messages.append(llm_response)
        tool_messages = await tool_node.ainvoke({"messages": messages})

        # ── 阶段3: LLM 综合工具结果生成最终答案 ──
        messages.extend(tool_messages["messages"])
        final_response = await llm_with_tools.ainvoke(messages)
        result = final_response.content
    else:
        result = llm_response.content

    # 返回状态更新
    return {
        "plan": plan[1:],                # 弹出已执行的步骤
        "past_steps": [(task, result)],  # 追加到执行历史
    }
```

**三阶段协议**：

```
阶段1: LLM 决策
  输入: SystemMessage + HumanMessage(当前步骤)
  输出: AIMessage（可能含 tool_calls）

阶段2: ToolNode 执行
  输入: messages + AIMessage(含 tool_calls)
  输出: ToolMessage（工具返回值）

阶段3: LLM 综合
  输入: 所有 messages + ToolMessage
  输出: 最终自然语言答案
```

**为什么不直接用 Agent 循环？** 固定三阶段确保每步只做一轮工具调用，防止 Agent 陷入无限工具调用循环。

## 6.5 Replanner — 评估与调整

```python
# app/agent/aiops/replanner.py
class Act(BaseModel):
    action: str   # "continue" | "replan" | "respond"
    new_steps: List[str] = Field(default_factory=list)


async def replanner(state: PlanExecuteState) -> Dict[str, Any]:
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    # 硬限制：最多 8 步
    if len(past_steps) >= 8:
        return await _generate_response(state, llm)

    if plan:
        # LLM 决策
        replanner_chain = replanner_prompt | llm.with_structured_output(Act)
        act = await replanner_chain.ainvoke({...})

        if act.action == "respond":
            # 信息充足，生成最终报告
            return await _generate_response(state, llm)
        elif act.action == "replan":
            # 调整计划（严格限制：新步骤数 <= 剩余步骤数）
            if len(past_steps) >= 5:
                return await _generate_response(state, llm)  # 已执行5步，禁止 replan
            return {"plan": act.new_steps[:len(plan)]}  # 截断到剩余步骤数
        else:  # continue
            return {}  # 不修改状态
    else:
        # 计划为空，必须出报告
        return await _generate_response(state, llm)
```

**三优先级决策**：

| 优先级 | 决策 | 触发条件 |
|--------|------|---------|
| 1（最高） | `respond` | 信息充足，或已执行 >= 5 步 |
| 2 | `continue` | 剩余计划合理且必要 |
| 3（最低） | `replan` | 原计划有严重问题（已执行 >= 5 步时禁止） |

**多层防无限循环机制**：
- 硬限制：`past_steps >= 8` 强制出报告
- 二次检查：`past_steps >= 5` 禁止 replan
- 新步骤数不能超过当前剩余步骤数

**Replanner 的决策如何影响图路由**：不直接传 `action`，而是通过状态变更间接表达：

| action | 状态变更 | should_continue 判断 |
|--------|---------|---------------------|
| `respond` | 设置 `response`（非空） | `response` 非空 → `END` |
| `replan` | 设置 `plan`（新步骤） | `plan` 非空 → `EXECUTOR` |
| `continue` | 返回 `{}`（无变更） | `plan` 非空 → `EXECUTOR` |

这种解耦设计让条件边函数不需要理解 `Act` 模型，只需检查 `response` 和 `plan`。

## 6.6 状态图编译

```python
# app/services/aiops_service.py
class AIOpsService:
    def _build_graph(self):
        workflow = StateGraph(PlanExecuteState)

        # 添加节点
        workflow.add_node("planner", planner)
        workflow.add_node("executor", executor)
        workflow.add_node("replanner", replanner)

        # 设置入口
        workflow.set_entry_point("planner")

        # 无条件边
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "replanner")

        # 条件边：replanner 的输出根据状态决定走向
        def should_continue(state):
            if state.get("response"):
                return END
            if state.get("plan"):
                return "executor"
            return END

        workflow.add_conditional_edges(
            "replanner", should_continue,
            {"executor": "executor", END: END}
        )

        return workflow.compile(checkpointer=MemorySaver())
```

**图拓扑**：

```
[START] → planner → executor → replanner ──┐
                        ▲                   │
                        │    should_continue │
                        │    ┌───────────────┤
                        │    │               │
                        └────┤ plan非空      ├── response非空 → [END]
                             └───────────────┘
```

## 6.7 流式事件格式化

```python
async def execute(self, user_input, session_id):
    initial_state = {"input": user_input, "plan": [], "past_steps": [], "response": ""}
    config_dict = {"configurable": {"thread_id": session_id}}

    async for event in self.graph.astream(input=initial_state, config=config_dict, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "planner":
                yield {"type": "plan", "stage": "plan_created", "plan": node_output.get("plan", [])}
            elif node_name == "executor":
                yield {"type": "step_complete", "stage": "step_executed", ...}
            elif node_name == "replanner":
                yield {"type": "report" if node_output.get("response") else "status", ...}

    yield {"type": "complete", "response": final_response}
```

**要点**：`stream_mode="updates"` 每个节点执行完输出一次状态增量，适合展示进度。
