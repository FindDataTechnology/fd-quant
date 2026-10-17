"""决策账本的违规类型（自带可操作信息，供调用方自修）。

放在独立模块是为了不让 `store` 与 `settlement` 互相 import（存储要调结算的纯
函数，结算不能反向依赖存储）。契约实例违规不在这里：那类错误由契约层自己的
`ContractError`（带字段路径与原因）表达，账本不包装它。

风格与计算面一致：错误文本必须点名**缺的东西**（哪个 decision_id、哪个日期、
缺几个交易日），而不是只说"非法"。这样 MCP 调用方能在同一轮对话里改参数重试。
"""
from __future__ import annotations

from typing import Optional


class LedgerError(ValueError):
    """账本输入或状态违规（路径错误 / 键不匹配 / 修订分叉 / 结算窗不足）。"""


class UnknownDecision(LedgerError):
    """引用的 `decision_id` 不在账本里（或 `revises` 指向了不存在的记录）。"""


class RevisionConflict(LedgerError):
    """同一 tip 已有一条**不同内容**的修订：账本是线性链，不能分叉。

    文本给出既有修订的 `decision_id`——调用方要改的是那条 tip 的最新记录，
    不是再挂一条兄弟记录（账本不保留"哪个分支是真的"的裁决权）。
    """


class SettlementRefused(LedgerError):
    """结算被拒（结算窗覆盖不足 / 入场日不在序列里 / 端点价格缺失）。

    `missing_trading_days` 让调用方不必解析文本就能知道要继续取多少天数据。
    """

    def __init__(self, message: str, *, missing_trading_days: Optional[int] = None):
        super().__init__(message)
        self.missing_trading_days = missing_trading_days
