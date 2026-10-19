"""调用者标识解析（ledger-tenancy-and-settlement）。

**为什么 caller 不能是入参**：调用者标识用于隔离。若它能由入参声明，任何调用方
都能声称自己是别人——那不叫隔离，叫"请自觉遵守"。故只从**调用上下文**解析。

解析顺序（与 `fd-paper-trading` 的凭据解析同构）：

1. HTTP 传输：从原始 ASGI 请求读 `Authorization`（fastmcp 的 `get_http_headers()`
   会剥掉 Authorization，必须读原始 request——same trap as facet-ops-mcp）。
2. stdio / 无请求上下文：用 `FD_QUANT_CALLER` 环境变量（部署侧配置的固定身份，
   适用于单机单用户的本地账本场景）。
3. 都没有：回落到 `"anonymous"`。

**为什么回落而不是报错**：账本此前是单用户的，本地 `fd_quant_ledger.db` 场景
根本没有凭据。报错误会让既有的本地用法全部失效；而回落到 `anonymous` 让
"没有身份基础设施"显式可见——所有记录都归 anonymous，运维一眼能看出该配
`FD_QUANT_CALLER`。

**取的是标识而非凭据本身**：不记录 token 原文（那是凭据，不是身份）。取
Bearer 的**稳定指纹**前 12 位十六进制，既能区分调用者，又不落凭据。
"""

from __future__ import annotations

import hashlib
import os

#: 无身份信息时使用的显式占位符。刻意不用空串——空串看起来像"解析失败"，
#: 而 anonymous 是一个可查询、可统计的真实取值。
ANONYMOUS = "anonymous"

#: 旧记录迁移时使用的归属（见任务 2.4）。
LEGACY = "legacy"

_ENV_VAR = "FD_QUANT_CALLER"


def _fingerprint(token: str) -> str:
    """Bearer 的稳定指纹：取 SHA-256 前 12 位十六进制。

    不记录 token 原文——那是凭据，落库即泄露面。指纹只用于区分与统计，
    且同一 token 恒得同一指纹（可复核）。
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def resolve_caller() -> str:
    """从调用上下文解析调用者标识；不接受入参声明。"""
    try:
        from fastmcp.server.dependencies import get_http_request
    except Exception:  # noqa: BLE001 — 计算层离线测试不装 fastmcp
        get_http_request = None  # type: ignore[assignment]

    if get_http_request is not None:
        try:
            request = get_http_request()
        except Exception:  # noqa: BLE001 — 无请求上下文（stdio）不是错误
            request = None
        if request is not None:
            auth = (request.headers.get("Authorization") or "").strip()
            if auth:
                raw = auth.split(" ", 1)[1] if " " in auth else auth
                return _fingerprint(raw)

    env = (os.environ.get(_ENV_VAR) or "").strip()
    if env:
        return env
    return ANONYMOUS
