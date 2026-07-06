# 第 7 章 MCP 服务器

## 7.1 FastMCP 快速搭建

使用 `fastmcp` 框架，几行代码就能定义一个 MCP 服务器：

```python
from fastmcp import FastMCP

mcp = FastMCP("服务名")

@mcp.tool()
def my_tool(param: str) -> dict:
    """工具描述"""
    return {"result": "..."}

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8003)
```

**`streamable-http` 传输方式**：客户端通过 HTTP POST 调用工具，支持长连接和流式响应。URL 格式为 `http://host:port/mcp`。

## 7.2 CLS 日志服务（5 个工具）

```python
# mcp_servers/cls_server.py
mcp = FastMCP("CLS")

@mcp.tool()
def get_current_timestamp() -> int:
    """获取当前时间戳（毫秒）"""
    return int(datetime.now().timestamp() * 1000)

@mcp.tool()
def get_region_code_by_name(region_name: str) -> dict:
    """根据区域名称获取区域代码"""
    # 映射: 北京→ap-beijing, 上海→ap-shanghai, 广州→ap-guangzhou

@mcp.tool()
def get_topic_info_by_name(topic_name: str, region_code: str = None) -> dict:
    """根据名称查询日志主题"""

@mcp.tool()
def search_topic_by_service_name(service_name: str, region_code: str = None, fuzzy: bool = True) -> dict:
    """根据服务名搜索日志主题"""

@mcp.tool()
def search_log(topic_id: str, start_time: int, end_time: int, query: str = None, limit: int = 100) -> dict:
    """搜索日志"""
    # 只对 topic-001 返回模拟数据：每分钟生成 1 条 INFO 日志

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8003)
```

**要点**：
- CLS 工具使用**毫秒时间戳**（`int`），与 Monitor 的字符串时间格式不同
- `search_log` 是核心工具，动态生成日志模拟数据
- 所有工具加了 `@log_tool_call` 装饰器记录调用日志

## 7.3 Monitor 监控服务（2 个工具）

```python
# mcp_servers/monitor_server.py
mcp = FastMCP("Monitor")

@mcp.tool()
def query_cpu_metrics(service_name: str, start_time: str = None, end_time: str = None, interval: str = "1m") -> dict:
    """查询 CPU 使用率指标"""
    # 模拟数据：指数增长（10% → 95%），超过 80% 触发告警

@mcp.tool()
def query_memory_metrics(service_name: str, start_time: str = None, end_time: str = None, interval: str = "1m") -> dict:
    """查询内存使用率指标"""
    # 模拟数据：线性增长（30% → 85%），超过 70% 触发告警

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8004)
```

**模拟数据设计**：
- CPU：指数增长模式，模拟突发压力
- 内存：线性增长模式，模拟内存泄漏
- 都加了随机噪声（CPU ±2%，内存 ±1%），让数据更真实
- 统计信息包含 avg、max、min、p95 + 告警判定

**要点**：时间参数用 `"YYYY-MM-DD HH:MM:SS"` 字符串格式，默认查最近 1 小时。
