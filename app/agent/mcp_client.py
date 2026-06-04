"""
MCP 客户端管理
提供全局单例的 MCP 客户端，避免重复初始化
"""

import asyncio
from typing import Optional, Dict, Any, List, Union

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import CallToolResult, TextContent
from loguru import logger

_mcp_client: Optional[MultiServerMCPClient] = None

def format_exception_chain(exc: BaseException) -> str:

    sub_exceptions = getattr(exc, "exceptions", None)

    if sub_exceptions is not None:
        lines = [str(exc)]
        for i, sub in enumerate(sub_exceptions):
            lines.append(f"[{i}] {format_exception_chain(sub)}")

        return "\n".join(lines)

    msg = f"{type(exc).__name__}: {exc}"
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return f"{msg}\n  caused by: {format_exception_chain(cause)}"
    return msg


async def load_mcp_tools_safe(
        client: MultiServerMCPClient,
) -> tuple[list[Union[BaseTool, Any]], str | None ]:
    """加载 MCP 工具；失败时返回空列表与可读错误信息，不向上抛出。"""
    try:
        tools = await client.get_tools()
        return tools, None
    except BaseException as e:
        return [], format_exception_chain(e)


async def retry_interceptor(
    request: MCPToolCallRequest,
    handler,
    max_retries: int = 3,
    delay: float = 1.0,
):
    """MCP 工具调用重试拦截器
    
    当工具调用失败时，使用指数退避策略自动重试。
    如果所有重试都失败，返回包含错误信息的结果而不是抛出异常。
    
    MCPToolCallRequest 结构：
    - name: str - 工具名称
    - args: dict[str, Any] - 工具参数
    - server_name: str - 服务器名称
    
    Args:
        request: MCP 工具调用请求
        handler: 实际的工具调用处理器
        max_retries: 最大重试次数（默认3次）
        delay: 初始延迟时间（秒，默认1秒）
    
    Returns:
        CallToolResult: 工具调用结果或错误信息
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(
                f"调用 MCP 工具: {request.name} "
                f"(服务器: {request.server_name}, 第 {attempt + 1}/{max_retries} 次尝试)"
            )
            result = await handler(request)
            logger.info(f"MCP 工具 {request.name} 调用成功")
            return result
            
        except Exception as e:
            last_error = e
            logger.warning(
                f"MCP 工具 {request.name} 调用失败 "
                f"(第 {attempt + 1}/{max_retries} 次): {str(e)}"
            )
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # 指数退避
                logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                await asyncio.sleep(wait_time)
    
    # 所有重试都失败，返回错误结果而不是抛出异常
    error_msg = f"工具 {request.name} 在 {max_retries} 次重试后仍然失败: {str(last_error)}"
    logger.error(error_msg)
    return CallToolResult(
        content=[TextContent(type="text", text=error_msg)],
        isError=True
    )


def _create_mcp_client(
    servers: Dict[str, Dict[str, str]],
    tool_interceptors: Optional[List] = None
) -> MultiServerMCPClient:
    """
    创建 MCP 客户端实例
    
    Args:
        servers: MCP 服务器配置
        tool_interceptors: 工具拦截器列表
    
    Returns:
        MultiServerMCPClient: 未初始化的客户端实例
    """
    # MultiServerMCPClient 的第一个参数直接接收 servers 配置字典
    # 格式: {server_name: {"transport": "...", "url": "..."}}
    kwargs: Dict[str, Any] = {}
    
    if tool_interceptors:
        kwargs["tool_interceptors"] = tool_interceptors
    
    # 第一个参数是 servers 配置，直接传递
    return MultiServerMCPClient(servers, **kwargs)  # type: ignore[arg-type]
from app.config import config

# 使用配置文件中定义的完整 MCP 服务器配置
DEFAULT_MCP_SERVERS = config.mcp_servers


async def get_mcp_client(
    servers: Optional[Dict[str, Dict[str, str]]] = None,
    tool_interceptors: Optional[List] = None,
    force_new: bool = False
) -> MultiServerMCPClient:
    """
    获取或初始化 MCP 客户端（不带重试拦截器）
    
    这是一个单例模式，确保整个应用只有一个 MCP 客户端实例（除非 force_new=True）
    
    从 langchain-mcp-adapters 0.1.0 开始，MultiServerMCPClient 不再支持作为上下文管理器使用。
    直接创建实例即可使用。
    
    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义工具拦截器列表
        force_new: 是否强制创建新实例（用于特殊场景，如需要不同配置）
    
    Returns:
        MultiServerMCPClient: MCP 客户端实例
    """
    global _mcp_client
    
    # 如果请求新实例，直接创建并返回（不缓存）
    if force_new:
        logger.info("创建新的 MCP 客户端实例（非单例）")
        client = _create_mcp_client(
            servers or DEFAULT_MCP_SERVERS, 
            tool_interceptors
        )
        # 不再需要 __aenter__()，直接返回即可
        return client
    
    # 单例模式：如果已存在，直接返回
    if _mcp_client is None:
        logger.info("初始化全局 MCP 客户端...")
        _mcp_client = _create_mcp_client(
            servers or DEFAULT_MCP_SERVERS, 
            tool_interceptors
        )
        # 不再需要 __aenter__()，直接使用即可
        logger.info("全局 MCP 客户端初始化完成")
    
    return _mcp_client


async def get_mcp_client_with_retry(
    servers: Optional[Dict[str, Dict[str, str]]] = None,
    tool_interceptors: Optional[List] = None,
    force_new: bool = False
) -> MultiServerMCPClient:
    """
    获取或初始化带重试功能的 MCP 客户端
    
    这是一个单例模式，确保整个应用只有一个 MCP 客户端实例（除非 force_new=True）
    重试拦截器会自动添加到拦截器列表的开头
    
    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义工具拦截器列表（会在重试拦截器之后添加）
        force_new: 是否强制创建新实例（用于特殊场景，如需要不同配置）
    
    Returns:
        MultiServerMCPClient: 带重试功能的 MCP 客户端实例
    """
    # 构建拦截器列表：重试拦截器在最前面
    interceptors = [retry_interceptor]
    if tool_interceptors:
        interceptors.extend(tool_interceptors)
    
    return await get_mcp_client(
        servers=servers,
        tool_interceptors=interceptors,
        force_new=force_new
    )
def suggest_mcp_transport(url: str, transport: str) -> str | None:
    """URL 与 transport 明显不匹配时给出建议（不自动改写配置）。"""
    lower_url = url.lower()
    if "/sse" in lower_url and transport.replace("_", "-") in (
        "streamable-http",
        "http",
    ):
        return (
            f"MCP URL 含 /sse/ 但 transport={transport!r}，"
            "腾讯云等托管端点应使用 transport=sse"
        )
    if transport == "sse" and "/mcp" in lower_url and "/sse" not in lower_url:
        return (
            f"MCP URL 为本地 FastMCP 路径但 transport={transport!r}，"
            "本地服务通常应使用 transport=streamable-http"
        )
    return None
