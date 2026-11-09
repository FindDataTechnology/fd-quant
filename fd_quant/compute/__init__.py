"""工具注册中心：六族计算模块 → MCP 工具面的唯一汇总点。

每个族模块导出 `TOOLS: list[ToolSpec]`；本模块只做汇总，不含计算逻辑。
`server.py` 与 `scripts/gen_surface.py`（manifest 生成）与
`scripts/selfcheck_surface.py`（manifest ↔ 运行时对拍）都从 `all_tools()` 读，
因此"工具面"只有一个事实源。

族的 import 是**显式列出**的（不是目录扫描）：漏掉一个族会在 selfcheck 里
以"manifest 有声明而运行时没有"的形式失败，而不是静默消失。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

#: 计算族的缺省行为属性：纯函数，无副作用，重复调用结果相同，不触达外部世界。
#: `openWorldHint=False` 是**事实**而非保守声明——计算层被
#: `tests/test_offline_boundary.py` 钉住不得 import sqlite3/网络/LLM。
_READONLY_ANNOTATIONS: dict[str, Any] = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

# 族分组 → 面向 tool-surface.json 的 level 标注
GROUP_LEVELS: dict[str, str] = {
    "technical_indicators": "commercial",
    "portfolio_statistics": "commercial",
    "fundamental_ratios": "commercial",
    "valuation_operators": "commercial",
    "regime_detection": "commercial",
    "factor_statistics": "commercial",
    "validation": "commercial",
    "decision_consistency": "commercial",
}


@dataclass(frozen=True)
class ToolSpec:
    """一个计算工具的完整声明：名字、分组、自足描述、实现。

    `description` 缺省取函数 docstring——工具的函数名即工具名，docstring 即
    描述，两者都是 MCP 面向调用方的唯一事实源，复制一份到声明里必然会漂移。
    描述必须自足：输入形状、参数默认值、单位、`as_of` 要求、warmup 语义、
    返回形状；由 `tests/test_tool_surface.py` 断言要素齐备。

    `annotations` 是 MCP 协议层的行为属性（`readOnlyHint` / `destructiveHint` /
    `idempotentHint` / `openWorldHint`），决定客户端能否自动批准。缺省为
    **只读 + 幂等**：计算族全部是无副作用的纯函数，把它们标成可写会让客户端
    对 46 个纯计算工具逐个弹确认；账本族必须显式覆盖（见 `ledger/tools.py`）。
    """

    name: str
    group: str
    fn: Callable[..., dict]
    description: str = ""
    data_sources: tuple[str, ...] = ()
    notes: Optional[str] = None
    level: Optional[str] = None  # None → 取 GROUP_LEVELS[group]
    tool_annotations: dict[str, Any] = field(
        default_factory=lambda: dict(_READONLY_ANNOTATIONS)
    )

    @property
    def effective_level(self) -> str:
        return self.level or GROUP_LEVELS.get(self.group, "commercial")

    @property
    def doc(self) -> str:
        return (self.description or (self.fn.__doc__ or "")).strip()


_MODULES = (
    ".technical",
    ".portfolio",
    ".fundamental",
    ".valuation",
    ".regime",
    ".factor",
    ".validation",
    # 决策一致性族（add-decision-consistency-and-premise-watch）：两个纯计算工具，
    # 只对传入的决策记录/前提作答，不取数、不读账本。独立成族而不塞进 validation：
    # validation 管"输入序列的口径"，这一族管"多个结论之间是否自相矛盾、前提是否
    # 还可监控"——判据不同，混在一组会让族描述失真。
    ".consistency",
    ".premise_watch",
)

#: 允许缺失的族（并行开发期：模块尚未落地时注册中心仍可 import）。
#: 一旦六个族全部落地，`missing_families()` 返回空——由 selfcheck 断言。
_ALLOW_MISSING = True


def all_tools() -> list[ToolSpec]:
    """汇总六族工具。族模块缺失时跳过（开发期宽容；由 selfcheck 把关完整性）。"""
    import importlib

    collected: list[ToolSpec] = []
    seen: set[str] = set()
    for mod_name in _MODULES:
        try:
            mod = importlib.import_module(mod_name, package=__package__)
        except ModuleNotFoundError:
            continue
        for spec in getattr(mod, "TOOLS", ()):  # noqa: B009
            if spec.name in seen:
                raise RuntimeError(f"duplicate tool name across families: {spec.name}")
            seen.add(spec.name)
            collected.append(spec)
    return collected


def missing_families() -> list[str]:
    """尚未落地的族模块（完整发布前必须为空）。"""
    import importlib

    out: list[str] = []
    for mod_name in _MODULES:
        try:
            importlib.import_module(mod_name, package=__package__)
        except ModuleNotFoundError:
            out.append(mod_name.lstrip("."))
    return out


def tools_by_group() -> dict[str, list[ToolSpec]]:
    out: dict[str, list[ToolSpec]] = {}
    for spec in all_tools():
        out.setdefault(spec.group, []).append(spec)
    return out
