"""把领域异常映射成封闭错误码（mcp-error-semantics 组 4）。

**为什么需要一层包装**：`ComputeError` / `LedgerError` / `ContractError` 都是
普通异常，fastmcp 捕获后会把它包成一段文本，`structuredContent` 为空——
调用方只能读字符串。本模块把每个工具函数包一层，捕获这些已知异常并返回
`ToolResult(is_error=True)`，让错误码走结构化通道。

**为什么不改领域异常本身**：计算层的异常是纯 Python 语义（离线测试不装
fastmcp 也要能 import），把 MCP 的返回形状塞进去会让计算层依赖传输层，
破坏 `test_offline_boundary` 钉住的封闭性。

**未知异常照常抛出**：不把 `ZeroDivisionError` 之类伪装成 `internal_error`
的成功响应。真正的 bug 应当带着堆栈出现在日志里，而不是变成一条结构化的
"服务内部错误"让调用方以为可以重试。

映射规则：

| 领域异常 | 码 | 判据 |
|---|---|---|
| `ComputeError` | `invalid_input` | 输入序列/参数违反契约 |
| `ContractError` | `invalid_input` | Rating/TargetWeights 契约违规 |
| `LedgerError`（含 UnknownDecision / RevisionConflict / SettlementRefused） | `invalid_input` | 账本调用违规（引用了不存在的 decision、结算窗不足等） |
| `FileNotFoundError` / 账本路径不可写 | `upstream_unavailable` | 本地存储不可用（调用方无从修正，可重试） |

**为什么账本路径问题算 upstream 而不是 invalid_input**：`db_path` 由部署
配置或环境变量给出，调用方传错参数的情况远少于"这台机器的目录权限不对"。
把它归到 `invalid_input` 会让调用方去改一个它根本控制不了的参数。
"""
from __future__ import annotations

import functools
import sqlite3
from typing import Any, Callable

from .errors import INTERNAL_ERROR, INVALID_INPUT, UPSTREAM_UNAVAILABLE, error_result


def _classify(exc: BaseException) -> tuple[str, str, dict[str, Any]] | None:
    """返回 (code, message, extra) 或 None（表示不认识，应照常抛出）。"""
    from .compute.base import ComputeError
    from .contracts.models import ContractError
    from .ledger.errors import LedgerError

    if isinstance(exc, ComputeError):
        return INVALID_INPUT, str(exc), {}
    if isinstance(exc, ContractError):
        # ContractError 自带 path/reason：把它们提出来，调用方不必解析文本。
        return INVALID_INPUT, str(exc), {
            "field": getattr(exc, "path", None),
            "reason": getattr(exc, "reason", None),
        }
    if isinstance(exc, LedgerError):
        extra: dict[str, Any] = {}
        missing = getattr(exc, "missing_trading_days", None)
        if missing is not None:
            extra["missing_trading_days"] = missing
        return INVALID_INPUT, str(exc), extra
    if isinstance(exc, (sqlite3.OperationalError, OSError)):
        # 本地账本不可用：路径不存在 / 目录不可写 / 只读文件系统 / 库被锁死。
        # OSError 覆盖 PermissionError / FileNotFoundError / IsADirectoryError
        # 以及"只读文件系统"这类 errno 特例——它们都由同一个原因引起
        # （这台机器的部署配置不对），对调用方是同一类处置。
        return UPSTREAM_UNAVAILABLE, f"本地账本不可用：{type(exc).__name__}: {exc}", {
            "upstream": "local_sqlite_ledger",
        }
    return None


def map_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """包一层：已知领域异常 → 结构化错误码；未知异常照常抛。"""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — 分类后未知异常会重新抛出
            classified = _classify(exc)
            if classified is None:
                raise
            code, message, extra = classified
            return error_result(code, message, **extra)

    return wrapper
