"""决策账本（`quant-decision-ledger`）：追加式记录 + 到期结算 alpha + 反思 + 点内检索。

分层与本层的对外面：

- `store.LedgerStore` —— 唯一的落库入口：`record` / `settle` / `get` / `retrieve` /
  `stats_by_pack`（SQLite 本地单文件，路径可配，见 `store.default_db_path`）。
- `settlement` —— 结算的**纯函数**层：基准映射、序列对齐、覆盖检查、收益与 alpha 算术。
  不碰库，故"持有期未满被拒"与"alpha 怎么算"可以脱离存储单独验证。
- `tools.TOOLS` —— MCP 工具组（`record_decision` / `settle_decision` /
  `retrieve_decisions` / `summarize_pack_performance`），由 `fd_quant.surface` 汇总进
  server 与上牌 manifest。
- `errors` —— 违规类型（`LedgerError` / `UnknownDecision` / `RevisionConflict` /
  `SettlementRefused`），文本一律点名缺的是什么。

纪律（三条，都是 spec 的硬要求）：

1. **账本不取数**：价格与基准序列由调用方传入，本层不联网、不调 LLM。
2. **账本不广播**：没有任何 push / notify / broadcast 接口；反思只存、只能被检索带出，
   注入决策是消费侧（P5/P4）的行为。
3. **追加式**：已落记录不就地改写，修订产生新行并以 `revises` 指回原记录。
"""
from __future__ import annotations

from .errors import LedgerError, RevisionConflict, SettlementRefused, UnknownDecision
from .settlement import (
    DEFAULT_BENCHMARK_MAP,
    DEFAULT_HOLDING_DAYS,
    is_iso_date,
    resolve_benchmark,
    settlement_window,
)
from .store import DEFAULT_DB_FILENAME, LedgerStore, default_db_path
from .tools import TOOLS, GROUP

__all__ = [
    "DEFAULT_BENCHMARK_MAP",
    "DEFAULT_DB_FILENAME",
    "DEFAULT_HOLDING_DAYS",
    "GROUP",
    "LedgerError",
    "LedgerStore",
    "RevisionConflict",
    "SettlementRefused",
    "TOOLS",
    "UnknownDecision",
    "default_db_path",
    "is_iso_date",
    "resolve_benchmark",
    "settlement_window",
]
