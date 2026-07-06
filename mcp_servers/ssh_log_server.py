"""SSH 日志查询 MCP Server.

通过 SSH 连接远程服务器，提供系统状态、Docker 日志和常见日志文件查询工具。
敏感连接信息从 .env 读取，不写入代码。
"""

import fnmatch
import json
import os
import shlex
import sys
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import Any

import paramiko
from fastmcp import FastMCP
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class SSHLogSettings(BaseSettings):
    """SSH 日志服务配置."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

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
    def allowed_patterns(self) -> list[str]:
        return [p.strip() for p in self.ssh_log_allowed_paths.split(",") if p.strip()]


settings = SSHLogSettings()
mcp = FastMCP("SSHLog")


def _configured() -> bool:
    return bool(settings.ssh_log_host and settings.ssh_log_user)


def _safe_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _is_allowed_path(path: str) -> bool:
    normalized = str(PurePosixPath(path))
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in settings.allowed_patterns)


@contextmanager
def _ssh_client():
    if not _configured():
        raise RuntimeError("SSH 日志服务未配置，请设置 SSH_LOG_HOST/SSH_LOG_USER")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    kwargs: dict[str, Any] = {
        "hostname": settings.ssh_log_host,
        "port": settings.ssh_log_port,
        "username": settings.ssh_log_user,
        "timeout": settings.ssh_log_timeout,
        "banner_timeout": settings.ssh_log_timeout,
        "auth_timeout": settings.ssh_log_timeout,
    }
    if settings.ssh_log_key_file:
        kwargs["key_filename"] = os.path.expanduser(settings.ssh_log_key_file)
    else:
        kwargs["password"] = settings.ssh_log_password

    try:
        client.connect(**kwargs)
        yield client
    finally:
        client.close()


def _run(command: str, timeout: int | None = None) -> dict[str, Any]:
    logger.info(f"SSH 执行命令: {command}")
    with _ssh_client() as client:
        stdin, stdout, stderr = client.exec_command(
            command,
            timeout=timeout or settings.ssh_log_timeout,
            get_pty=False,
        )
        stdin.close()
        exit_status = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")

    return {
        "exit_status": exit_status,
        "stdout": out,
        "stderr": err,
        "ok": exit_status == 0,
    }


@mcp.tool()
def list_log_files() -> dict[str, Any]:
    """列出允许查询的远程日志文件及大小."""
    patterns = " ".join(shlex.quote(pattern) for pattern in settings.allowed_patterns)
    command = (
        "for pattern in "
        + patterns
        + "; do "
        + 'for file in $pattern; do [ -f "$file" ] && '
        + 'stat -c "%n\\t%s\\t%y" "$file"; done; '
        + "done"
    )
    result = _run(command)
    files = []
    if result["stdout"]:
        for line in result["stdout"].splitlines():
            line = line.replace("\\t", "\t")
            parts = line.split("\t", 2)
            if len(parts) == 3:
                files.append({"path": parts[0], "size_bytes": int(parts[1]), "modified": parts[2]})
    return {"configured_patterns": settings.allowed_patterns, "files": files, "raw": result}


@mcp.tool()
def tail_log(file_path: str, lines: int = 100) -> dict[str, Any]:
    """查看指定日志文件最后 N 行."""
    if not _is_allowed_path(file_path):
        return {"ok": False, "error": f"不允许查询该路径: {file_path}"}

    line_count = _safe_int(lines, 1, 1000)
    command = f"tail -n {line_count} -- {shlex.quote(file_path)}"
    result = _run(command)
    return {"file_path": file_path, "lines": line_count, **result}


@mcp.tool()
def search_log(keyword: str, file_path: str = "", limit: int = 100) -> dict[str, Any]:
    """在日志中按关键词搜索，默认搜索所有允许路径."""
    if not keyword.strip():
        return {"ok": False, "error": "keyword 不能为空"}

    max_count = _safe_int(limit, 1, 500)
    quoted_keyword = shlex.quote(keyword)

    if file_path:
        if not _is_allowed_path(file_path):
            return {"ok": False, "error": f"不允许查询该路径: {file_path}"}
        targets = shlex.quote(file_path)
    else:
        targets = " ".join(shlex.quote(pattern) for pattern in settings.allowed_patterns)

    command = f"grep -RInhi --binary-files=without-match -m {max_count} -- {quoted_keyword} {targets} 2>/dev/null || true"
    result = _run(command, timeout=max(settings.ssh_log_timeout, 30))
    return {"keyword": keyword, "file_path": file_path or "allowed_paths", "limit": max_count, **result}


@mcp.tool()
def search_error_logs(limit: int = 100) -> dict[str, Any]:
    """搜索常见错误日志关键字，如 ERROR、Exception、Traceback、500."""
    max_count = _safe_int(limit, 1, 500)
    targets = " ".join(shlex.quote(pattern) for pattern in settings.allowed_patterns)
    pattern = "'ERROR|Error|Exception|Traceback|CRITICAL|FATAL| 500 | 502 | 503 | 504 '"
    command = f"grep -RInEh --binary-files=without-match -m {max_count} {pattern} {targets} 2>/dev/null || true"
    result = _run(command, timeout=max(settings.ssh_log_timeout, 30))
    return {"limit": max_count, **result}


@mcp.tool()
def search_docker_logs(container_name: str = "", keyword: str = "", tail: int = 200) -> dict[str, Any]:
    """查询 Docker 容器日志；可指定容器名和关键词."""
    tail_count = _safe_int(tail, 1, 2000)

    if container_name:
        command = f"docker logs --tail={tail_count} {shlex.quote(container_name)} 2>&1"
    else:
        command = "docker ps -a --format '{{.Names}}\\t{{.Status}}\\t{{.Image}}'"

    if keyword.strip():
        command = f"{command} | grep -i -- {shlex.quote(keyword)} || true"

    result = _run(command, timeout=max(settings.ssh_log_timeout, 30))
    return {
        "container_name": container_name or "all",
        "keyword": keyword,
        "tail": tail_count,
        **result,
    }


@mcp.tool()
def get_system_status() -> dict[str, Any]:
    """获取远程服务器基础状态：负载、磁盘、内存、端口和 Docker 容器."""
    command = r"""
printf '%s\n' '--- hostname ---'
hostnamectl
printf '%s\n' '--- uptime ---'
uptime
printf '%s\n' '--- disk ---'
df -h
printf '%s\n' '--- memory ---'
free -h
printf '%s\n' '--- listening ports ---'
ss -tulpn
printf '%s\n' '--- docker containers ---'
docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}' 2>/dev/null || true
"""
    result = _run(command, timeout=max(settings.ssh_log_timeout, 30))
    return result


@mcp.tool()
def get_recent_security_events(lines: int = 100) -> dict[str, Any]:
    """查看 SSH 登录、安全认证相关日志."""
    file_path = "/var/log/auth.log"
    line_count = _safe_int(lines, 1, 500)
    if not _is_allowed_path(file_path):
        return {"ok": False, "error": f"不允许查询该路径: {file_path}"}
    command = f"tail -n {line_count} -- {shlex.quote(file_path)}"
    result = _run(command)
    return {"file_path": file_path, "lines": line_count, **result}


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8005, path="/mcp")
