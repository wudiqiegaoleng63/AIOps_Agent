"""Prometheus 告警查询工具

通过 Prometheus HTTP API `GET /api/v1/alerts` 拉取当前规则产生的告警列表
（含 pending / firing 等状态）。每条告警由「完整 labels」唯一标识，与 Prometheus
文档一致；不得仅用 `alertname` 去重，否则多实例同名规则会被错误合并。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
from langchain_core.tools import tool
from loguru import logger

from app.config import config


ALERTS_API_PATH = "/api/v1/alerts"


COMMON_LABEL_KEYS = ("alertname", "severity", "instance", "job", "namespace", "pod")


def _parse_active_at(active_at_str: str) -> datetime | None:
    """将 Prometheus 返回的 activeAt（RFC3339 或带 Z 后缀）解析为 UTC 时间。"""
    if not active_at_str:
        return None
    try:
        s = active_at_str.replace("Z", "+00:00", 1)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None
    
def _labels_identity(labels: dict[str, Any]) -> str:
    """告警唯一键：完整 labels 的 JSON（键排序），用于去重或合并重复项。"""
    return json.dumps(labels, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def calculate_duration(active_at_str: str) -> str:
    """根据 activeAt 计算相对当前 UTC 的已持续时长（人类可读短文本）。"""
    active_at = _parse_active_at(active_at_str)
    if active_at is None:
        return "unknown"
    now = datetime.now(timezone.utc)
    delta = now - active_at
    total_seconds = max(0, int(delta.total_seconds()))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}h{minutes}m{seconds}s"
    if minutes > 0:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"

def query_prometheus_alerts_api() -> tuple[dict[str, Any], str | None]:
    """请求 `GET {prometheus_base_url}/api/v1/alerts`。

    返回 (JSON 体, 错误信息)。成功时第二项为 None；HTTP 或 JSON 解析失败时第一项为空 dict。
    """
    base_url = config.prometheus_base_url.rstrip("/")
    api_url = f"{base_url}{ALERTS_API_PATH}"
    logger.info("Querying Prometheus alerts: {}", api_url)
    try:
        with httpx.Client(timeout=config.prometheus_request_timeout) as client:
            resp = client.get(api_url)
            resp.raise_for_status()
            body = resp.json()
    except httpx.HTTPError as e:
        return {}, f"failed to query Prometheus alerts: {e}"
    except json.JSONDecodeError as e:
        return {}, f"failed to parse response: {e}"
    return body, None


def _pick_common_labels(labels: dict[str, Any]) -> dict[str, Any]:
    """从 labels 中提取常用维度，减少 Agent 阅读整表 labels 的成本。"""
    out: dict[str, Any] = {}
    for k in COMMON_LABEL_KEYS:
        if k == "alertname":
            continue
        v = labels.get(k)
        if v is not None and v != "":
            out[k] = v
    return out

def _simplify_alerts(result: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """将 Prometheus `data.alerts` 转为简化列表，并按 activeAt 从新到旧排序。

    返回 (simplified_alerts, state_counts)。
    """
    data = result.get("data") or {}
    alerts = data.get("alerts") or []
    if not isinstance(alerts, list):
        return [], {}

    simplified: list[dict[str, Any]] = []
    # 若上游偶发重复推送完全相同 labels 的条目，只保留一条（按 labels 身份去重）
    seen_identity: set[str] = set()
    state_counts: dict[str, int] = {}

    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        labels = alert.get("labels") or {}
        annotations = alert.get("annotations") or {}
        if not isinstance(labels, dict):
            labels = {}
        if not isinstance(annotations, dict):
            annotations = {}

        identity = _labels_identity(labels)
        if identity in seen_identity:
            continue
        seen_identity.add(identity)

        state = str(alert.get("state", "") or "")
        state_counts[state] = state_counts.get(state, 0) + 1

        active_at = str(alert.get("activeAt", "") or "")
        alert_name = str(labels.get("alertname", "") or "")

        simplified.append(
            {
                "alert_name": alert_name,
                "labels": labels,
                "common_labels": _pick_common_labels(labels),
                "description": str(annotations.get("description", "") or ""),
                "summary": str(annotations.get("summary", "") or ""),
                "state": state,
                "active_at": active_at,
                "duration": calculate_duration(active_at),
            }
        )

    # 「最新」：按 activeAt 降序；无法解析的时间排在最后，便于人工扫列表
    def sort_key(item: dict[str, Any]) -> tuple[int, float]:
        dt = _parse_active_at(str(item.get("active_at", "")))
        if dt is None:
            return (1, 0.0)
        # (0, -timestamp) 保证新在前；用负的 timestamp 避免 tuple 比较问题
        return (0, -dt.timestamp())

    simplified.sort(key=sort_key)
    return simplified, state_counts


@tool
def query_prometheus_alerts() -> str:
    """查询 Prometheus 服务端当前活动告警（HTTP GET /api/v1/alerts）。

    适用场景：用户关心「有没有告警」「哪些规则在 firing/pending」「最近触发了什么告警」
    「排查监控告警」「和 Prometheus 告警规则相关的现状」等运维/可观测性问题；无需用户
    提供参数，直接调用即可拉取服务端已聚合的告警列表。

    行为说明：向配置项 `prometheus_base_url` 指向的 Prometheus 拉取告警；结果按激活时间
    从新到旧排序；每条包含 alert 名称、labels、常见维度摘要、描述/摘要注解、状态与
    持续时长等。返回 JSON 字符串，含 success、alerts、state_counts 等字段。

    注意：这是 Prometheus 内置告警 API，不是执行 PromQL 指标查询，也不是 Alertmanager
    的通知/静默接口；若需查指标曲线请用 MCP 或其它指标工具。

    Returns:
        str: JSON 字符串。成功时含告警列表与状态统计；失败时含 success=false 与 error。
    """
    result, err = query_prometheus_alerts_api()
    if err:
        out = {
            "success": False,
            "error": err,
            "message": "Failed to query Prometheus alerts",
        }
        return json.dumps(out, ensure_ascii=False, indent=2)

    if result.get("status") != "success":
        err_msg = result.get("error") or result.get("errorType") or "Prometheus returned non-success status"
        out = {
            "success": False,
            "error": str(err_msg),
            "message": "Failed to query Prometheus alerts",
        }
        return json.dumps(out, ensure_ascii=False, indent=2)

    simplified, state_counts = _simplify_alerts(result)
    out = {
        "success": True,
        "alerts": simplified,
        "state_counts": state_counts,
        "total": len(simplified),
        "message": f"已获取 {len(simplified)} 条告警（按 activeAt 从新到旧），状态分布: {state_counts}",
    }
    logger.info("Prometheus alerts query completed: {} alerts, states={}", len(simplified), state_counts)
    return json.dumps(out, ensure_ascii=False, indent=2)
