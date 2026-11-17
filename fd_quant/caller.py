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
import hmac
import os
import re
import time
from typing import Mapping, Optional

#: 无身份信息时使用的显式占位符。刻意不用空串——空串看起来像"解析失败"，
#: 而 anonymous 是一个可查询、可统计的真实取值。
ANONYMOUS = "anonymous"

#: 旧记录迁移时使用的归属（见任务 2.4）。
LEGACY = "legacy"

_ENV_VAR = "FD_QUANT_CALLER"

# ── 签名头（seal-ledger-path-and-caller-isolation 3.1）─────────────────────
#
# 为什么用 HMAC 而不是 JWT：本仓有三道离线边界闸（test_offline_boundary /
# test_ledger_offline / test_family_conformance）只许「数值库 + 传输层」，
# `pyjwt[crypto]` 过不去（2026-11-15 出片 CI 因此红过）。`hmac` 与 `hashlib`
# 同属标准库，零新依赖。
#
# 头名与签法是对外契约的一半：`line-proxy` 侧按同一段字符串签，所以拼接顺序
# （`<id>.<ts>`，不含头名、不含密钥）不能改。

#: 客户身份头。值形如 `<customer_id>.<unix_ts>.<hex hmac_sha256>`。
CALLER_HEADER = "X-Fd-Caller"

#: 验签密钥的环境变量。未配 ⇒ **整条头轨不生效**（等价于"没有这条轨"），
#: 而不是"有头就拒"——滚动升级期间旧 pod 不该因为多出一个头就 401。
CALLER_HMAC_ENV = "FD_QUANT_CALLER_HMAC_KEY"

#: 时钟偏移窗口（秒）。头随请求走，不需要独立寿命。
CALLER_CLOCK_SKEW_S = 300

#: 原因码：稳定且互不相同（错误映射与告警按它们分流，不按异常类名）。
REASON_MALFORMED = "caller_signature_malformed"
REASON_BAD_ID = "caller_id_invalid"
REASON_EXPIRED = "caller_signature_expired"
REASON_MISMATCH = "caller_signature_mismatch"

#: `customer_id` 字符白名单（1–128）。允许 `.` 是为了邮箱，因此**必须额外禁 `..`**：
#: 否则一个纯点号 id 会带着路径穿越的味道流到下游目录名里。
_ID_RE = re.compile(r"^[A-Za-z0-9@._-]{1,128}$")


class CallerSignatureError(ValueError):
    """身份头验签失败。**抛出而不是回落**——回落会把一次伪造变成一次合法的匿名写入。

    `reason` 是稳定字符串而非异常类名：调用方与错误映射按它分流，重命名异常类
    不该改变对外原因码。
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail or reason
        super().__init__(f"{reason}: {self.detail}" if detail else reason)


def _hmac_key(env: Optional[Mapping[str, str]] = None) -> str:
    e = os.environ if env is None else env
    return (e.get(CALLER_HMAC_ENV) or "").strip()


def sign_caller(customer_id: str, ts: int, key: str) -> str:
    """按对外契约签名：`hmac_sha256(key, f"{customer_id}.{ts}")` 的十六进制。

    **网关侧照这个函数实现**（本变更只定线侧契约，不改 `line-proxy`）。放在这里
    而不是测试里，是为了让「签」与「验」共用同一段拼接——两份各写一遍，迟早有
    一份把分隔符改了。
    """
    msg = f"{customer_id}.{ts}".encode("utf-8")
    return hmac.new(key.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def verify_caller_header(
    value: Optional[str],
    *,
    now: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """验签一个 `X-Fd-Caller` 值：通过返回 `customer_id`，不适用时返回 `None`。

    `None` 的两种含义**都不是失败**，都是"这条轨不适用于本次请求"：头不存在/为空、
    密钥未配。失败一律抛 `CallerSignatureError`（原因码见 `REASON_*`）。

    检查顺序「形状 → id 合法性 → 时钟窗 → 签名」是刻意的：形状错了没必要算 HMAC；
    超窗的请求也不该被报成"签名错误"，否则排查会朝错方向走。
    """
    raw = (value or "").strip()
    key = _hmac_key(env)
    if not raw or not key:
        return None

    parts = raw.rsplit(".", 2)
    if len(parts) != 3:
        raise CallerSignatureError(REASON_MALFORMED, "期望 <customer_id>.<unix_ts>.<hex>")
    customer_id, ts_raw, sig = parts

    if not _ID_RE.match(customer_id) or customer_id in {".", ".."} or ".." in customer_id:
        raise CallerSignatureError(REASON_BAD_ID, "customer_id 只许 [A-Za-z0-9@._-] 且不得含 `..`")

    try:
        ts = int(ts_raw)
    except (TypeError, ValueError):
        raise CallerSignatureError(REASON_MALFORMED, "unix_ts 不是整数") from None

    current = time.time() if now is None else float(now)
    if abs(current - ts) > CALLER_CLOCK_SKEW_S:
        raise CallerSignatureError(
            REASON_EXPIRED, f"时间戳偏移 {abs(current - ts):.0f}s 超出 ±{CALLER_CLOCK_SKEW_S}s"
        )

    expected = sign_caller(customer_id, ts, key)
    if not hmac.compare_digest(expected, sig.lower()):
        raise CallerSignatureError(REASON_MISMATCH, "签名不匹配")
    return customer_id


def _request_header(request: object, name: str) -> Optional[str]:
    """从请求头取一个值。Starlette 的 `Headers` 大小写不敏感，普通 dict 不是——
    两种都认，免得夹具与线上走两条路径（夹具能绕过的那条路径等于没验）。"""
    headers = getattr(request, "headers", None)
    if not headers:
        return None
    for key in (name, name.lower(), name.upper()):
        try:
            got = headers.get(key)
        except Exception:  # noqa: BLE001 — 非 Mapping 形态直接放弃
            return None
        if got:
            return got
    return None


def _fingerprint(token: str) -> str:
    """Bearer 的稳定指纹：取 SHA-256 前 12 位十六进制。

    不记录 token 原文——那是凭据，落库即泄露面。指纹只用于区分与统计，
    且同一 token 恒得同一指纹（可复核）。

    **已知局限**（本变更的动机之一）：注册表在出口把每个客户的 `Authorization`
    换成同一把行内令牌，故这个指纹对全体客户同值。它保留为「没升级到签名头时的
    兜底」，让未配 `FD_QUANT_CALLER_HMAC_KEY` 的部署行为与今天逐字相同（5.5 的
    行为中性判据）；客户级隔离靠签名头，不靠它。
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def resolve_caller() -> str:
    """从调用上下文解析调用者标识；不接受入参声明。

    顺序：**签名头**（验过才用，验不过抛）→ Bearer 指纹（未升级部署的兜底）
    → `FD_QUANT_CALLER` env → `anonymous`。

    为什么把指纹留在签名头之后而不是删掉：删掉会让"没配密钥的部署"从"全客户
    同名"直接变成"全客户 anonymous"——那不是修隔离，是把两种不同的现状混成
    一个，且与 5.5 的行为中性判据冲突。头是**增量**（design Risks 最后一条）。
    """
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
            # 验签头优先：验不过**抛**，不静默退化到下面的指纹/匿名。
            signed = verify_caller_header(_request_header(request, CALLER_HEADER))
            if signed:
                return signed
            auth = (request.headers.get("Authorization") or "").strip()
            if auth:
                raw = auth.split(" ", 1)[1] if " " in auth else auth
                return _fingerprint(raw)

    env = (os.environ.get(_ENV_VAR) or "").strip()
    if env:
        return env
    return ANONYMOUS
