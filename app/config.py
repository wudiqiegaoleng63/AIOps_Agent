"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置
    mcp_cls_enabled: bool = True
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_enabled: bool = True
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"
    mcp_ssh_log_transport: str = "streamable-http"
    mcp_ssh_log_url: str = "http://localhost:8005/mcp"

    # SSH 日志查询配置
    ssh_log_enabled: bool = False
    ssh_log_host: str = ""
    ssh_log_port: int = 22
    ssh_log_user: str = "root"
    ssh_log_password: str = ""
    ssh_log_key_file: str = ""
    ssh_log_timeout: int = 15
    ssh_log_allowed_paths: str = (
        "/var/log/syslog,"
        "/var/log/auth.log,"
        "/var/log/kern.log,"
        "/var/lib/docker/containers/*/*-json.log"
    )

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        servers = {}
        if self.mcp_cls_enabled:
            servers["cls"] = {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            }
        if self.mcp_monitor_enabled:
            servers["monitor"] = {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        if self.ssh_log_enabled:
            if self.mcp_ssh_log_transport == "stdio":
                servers["ssh_log"] = {
                    "transport": "stdio",
                    "command": ".venv\\Scripts\\python.exe",
                    "args": ["mcp_servers/ssh_log_server.py", "--stdio"],
                }
            else:
                servers["ssh_log"] = {
                    "transport": self.mcp_ssh_log_transport,
                    "url": self.mcp_ssh_log_url,
                }
        return servers


# 全局配置实例
config = Settings()
