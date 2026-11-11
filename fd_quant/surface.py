"""fd-quant MCP 工具面的**单一汇总点**：计算面七族 + 决策账本族。

为什么需要这一层：`fd_quant.compute` 是**纯计算**层，它的族模块被
`tests/test_offline_boundary.py::test_compute_layer_package_list_is_closed` 与一条
禁止 import 扫描钉住（不得出现 sqlite3 / 网络 / LLM 客户端）。账本**必须**持有本地
SQLite（spec: 存储原子性与并发安全），所以它物理上不住在 `fd_quant/compute/` 里，
而是 `fd_quant/ledger/`；本模块是两者唯一的汇合处——server、`scripts/gen_surface.py`
（上牌 manifest 生成器）、`scripts/selfcheck_surface.py`（双向对拍）都从这里读，
于是"能列出"与"能服务"仍是同一份清单，且计算层的封闭性不被账本打破。

工具面的方向性依赖：`ledger → compute`（账本的工具声明复用 `ToolSpec`），
`surface → {compute, ledger}`。**计算层不 import 账本**——这条由上面的闭合断言保证。
"""
from __future__ import annotations

from typing import Any

from .compute import (
    GROUP_LEVELS,
    ToolSpec,
    all_tools as compute_tools,
    missing_families as compute_missing_families,
)
from .ledger import TOOLS as LEDGER_TOOLS
from .portfolio import PORTFOLIO_TOOLS
from .risk_tools import RISK_TOOLS

#: 账本族的暴露级别（`ToolSpec.effective_level` 缺省即 "commercial"，
#: 这里显式登记，使"账本面是否可被 pack 引用"是一个可见的决定而不是默认值）。
#: commercial 的理由：P5 决策复盘 pack 与 P4 投委会是既定消费方（proposal.md 的
#: 契约下游），它们要引用这四个工具做落账 / 结算 / 点内检索 / 聚合。
EXTRA_GROUP_LEVELS: dict[str, str] = {
    "decision_ledger": "commercial",
    # 组合构建：纯确定性函数（不取数、不联网、不落库），与计算面同级可对外。
    "portfolio_construction": "commercial",
    # 组合风控闸：纯函数、只读幂等，与组合构建同级。
    "portfolio_risk": "commercial",
}

#: 全量分组的暴露级别：计算面七族（来自 compute 层）+ 账本族（本层登记）。
SURFACE_GROUP_LEVELS: dict[str, str] = {**GROUP_LEVELS, **EXTRA_GROUP_LEVELS}


def all_surface_tools() -> list[ToolSpec]:
    """服务实际注册的全部工具（计算面七族在前，账本族在后；顺序稳定）。

    重名即报错：两个族同名说明工具面有两个事实源，宁可在这里炸也不要在 MCP
    上让后注册的静默覆盖先注册的。
    """
    out: list[ToolSpec] = list(compute_tools())
    seen = {spec.name for spec in out}
    for spec in LEDGER_TOOLS + PORTFOLIO_TOOLS + RISK_TOOLS:
        if spec.name in seen:
            raise RuntimeError(f"duplicate tool name across groups: {spec.name}")
        seen.add(spec.name)
        out.append(spec)
    return out


def tools_by_group() -> dict[str, list[ToolSpec]]:
    out: dict[str, list[ToolSpec]] = {}
    for spec in all_surface_tools():
        out.setdefault(spec.group, []).append(spec)
    return out


def missing_families() -> list[str]:
    """尚未落地的**计算面**族（账本族不在此列：它是独立分组，不是计算族）。"""
    return compute_missing_families()


# 2026-11-11 删掉三个零引用的助手（`group_level` / `ledger_group_present` / `find_tool`）：
# 前两个分别与 `ToolSpec.effective_level`、`scripts/selfcheck_surface.py` 里的反向上牌自检
# 重复实现同一件事；`find_tool` 是没人调的线性查找。留着它们的代价不是几行代码，而是
# 下一个读者会从两份实现里挑一份——然后另一份就悄悄漂移了。
