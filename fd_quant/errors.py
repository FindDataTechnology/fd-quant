"""错误码的封闭集合与统一返回形状（mcp-error-semantics）。

**为什么要有封闭集合**：调用方需要按码分支——"重试一次"、"改参数"、"换个
数据源"、"放弃"是四种不同的动作。把这一切都塞进一段中文消息里，等于要求
调用方解析自然语言，而那在措辞变化时会静默失效（不会报错，只会走错分支）。

**为什么码是英文常量而不是中文短语**：码是**接口**，会被写进调用方代码与
判断分支；中文描述在 `message` 里，是给人读的。两者都要有，但不能互换。

**为什么用 ToolResult(is_error=True) 而不是抛异常**：抛异常时 fastmcp 会把
消息包成一段文本，`structuredContent` 为空——调用方只能读字符串。用
`ToolResult` 可以让 `isError=true` 与结构化错误体**同时**到达客户端：平台
侧按 isError 判定失败（凭据失效等场景依赖这个信号），模型侧按 code 分支。

码的语义边界：

| 码 | 含义 | 调用方该做什么 |
|---|---|---|
| `invalid_input` | 入参违反契约 | 改参数，别重试 |
| `not_found` | 查询有效但无匹配 | 换条件或接受空结果 |
| `not_wired` | 能力尚未提供 | 别重试，换路径 |
| `upstream_unavailable` | 依赖的外部系统不可达/报错 | 可重试，或降级 |
| `internal_error` | 本服务自身故障 | 上报，别盲目重试 |

`degraded` 不是错误码，是**成功响应里的标记**：用于"确实降级返回了结果"的
场景（如检索无结果但明确说明原因），使调用方能区分"真的没有"与"取不到"。
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

#: 封闭集合。新增码必须在这里登记——`test_error_codes.py` 会比对
#: 各服务实际使用的码集合与它，出现未登记的码即失败。
INVALID_INPUT = "invalid_input"
NOT_FOUND = "not_found"
NOT_WIRED = "not_wired"
UPSTREAM_UNAVAILABLE = "upstream_unavailable"
INTERNAL_ERROR = "internal_error"

CODES: tuple[str, ...] = (
    INVALID_INPUT,
    NOT_FOUND,
    NOT_WIRED,
    UPSTREAM_UNAVAILABLE,
    INTERNAL_ERROR,
)

#: 码 → 是否值得重试。调用方可以据此自动决定重试策略，而不必硬编码码表。
RETRYABLE: dict[str, bool] = {
    INVALID_INPUT: False,
    NOT_FOUND: False,
    NOT_WIRED: False,
    UPSTREAM_UNAVAILABLE: True,
    INTERNAL_ERROR: False,
}


def error_body(
    code: str,
    message: str,
    *,
    field: Optional[str] = None,
    upstream: Optional[str] = None,
    retryable: Optional[bool] = None,
    **extra: Any,
) -> dict[str, Any]:
    """构造结构化错误体。

    `field` 用于入参非法（指出哪个参数），`upstream` 用于上游不可用（指出
    是哪个依赖）——两者都是"错误含定位信息"这条要求的载体，不能只写在
    message 里靠人读。
    """
    if code not in CODES:
        raise ValueError(f"未登记的错误码 {code!r}；合法值：{', '.join(CODES)}")
    body: dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": RETRYABLE[code] if retryable is None else retryable,
    }
    if field is not None:
        body["field"] = field
    if upstream is not None:
        body["upstream"] = upstream
    body.update(extra)
    return body


def error_result(
    code: str,
    message: str,
    *,
    field: Optional[str] = None,
    upstream: Optional[str] = None,
    retryable: Optional[bool] = None,
    **extra: Any,
):
    """返回 `ToolResult(is_error=True)`，让 isError 与结构化错误体同时到达。

    延迟导入 fastmcp：计算层的离线测试不装 fastmcp 也能 import 本模块。
    """
    from fastmcp.tools.base import ToolResult

    return ToolResult(
        structured_content={"error": error_body(
            code, message, field=field, upstream=upstream,
            retryable=retryable, **extra)},
        is_error=True,
    )


def degraded_note(reason: str, **extra: Any) -> dict[str, Any]:
    """成功响应里的降级标记：区分"真的没有"与"取不到"。"""
    return {"degraded": True, "degraded_reason": reason, **extra}


def codes_used_in(module_globals: Iterable[Any]) -> set[str]:
    """测试辅助：从模块里收集实际用到的码（用于封闭性核对）。"""
    return {v for v in module_globals if isinstance(v, str) and v in CODES}
