"""组合风控的 MCP 工具外壳（与纯判定分离）。

## 为什么拆开

`risk.py` 是**纯判定**：不依赖框架、不 import MCP、不碰 ToolSpec。交易仓要字节镜像它
（三方共用同一实现），而镜像件不能拖着 `fd_quant.compute` 的依赖走——那会把整条框架链
带进一个只需要算术的仓。

工具注册属传输层，放这里；判定留在 `risk.py`。这也让"改判据"与"改工具面"在 diff 里分得开。
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .compute import ToolSpec
from .risk import evaluate_portfolio_gate


def evaluate_portfolio_gate_tool(
    weights: Mapping[str, float],
    thresholds: Optional[Mapping[str, Any]] = None,
    sectors: Optional[Mapping[str, str]] = None,
    equity_series: Optional[Sequence[Sequence[Any]]] = None,
    as_of: Optional[str] = None,
    price_as_of_max: Optional[str] = None,
) -> dict:
    """组合级风控判定的对外入口（纯函数，不取数、不联网、不落库）。

    三态返回是本工具的核心：`status` 取 ok / breached / **unavailable**，
    且 `breached` 同为三态（true / false / **null**）。调用方（尤其执行路径）必须把
    `unavailable` 当作拒绝条件，**不得**读成"未触达"。

    Input: weights（{symbol: 权重}，非负、Σ ≤ 1，拒绝而不归一化）,
      thresholds（四阈值 + 可选 max_staleness_days；缺任一项即 unavailable）,
      sectors（{symbol: 行业}；缺归属时该判据报不可判定并点名，不归入默认行业）,
      equity_series（[(日期, 权益)]，用于回撤与单日亏损；缺则该项不可判定）,
      as_of / price_as_of_max（两端都给且配了 max_staleness_days 才判行情陈旧）。
    Returns: status / breached / items（触达项，含 kind、actual、limit、subject）/
      undetermined（不可判定项，含 reason 与涉及标的）/ inputs（含缺失阈值清单）。
      判据只报"触达了什么"，不输出任何处置指令。
    """
    return evaluate_portfolio_gate(
        weights=weights, thresholds=thresholds, sectors=sectors,
        equity_series=equity_series, as_of=as_of, price_as_of_max=price_as_of_max,
    )


_RISK_READ = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

RISK_TOOLS: list[ToolSpec] = [
    ToolSpec("evaluate_portfolio_gate", "portfolio_risk", evaluate_portfolio_gate_tool,
             tool_annotations=_RISK_READ),
]
