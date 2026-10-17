"""跨族共用的确定性测试数据。

刻意**不用** `numpy.random`：随机数的默认位流随 numpy 版本演进，
用它生成的"夹具"会在某次升级后静默换一套数，使断言从"验证实现"退化成
"验证运气"。这里用一个写死的线性同余发生器，序列在任何一个 Python/numpy
版本上都逐位相同。

所有夹具都返回 `(dates, values...)` 形态，日期从 `2025-01-01` 起按自然日递增
（跳过周末不是必需的：计算面只要求严格递增的 ISO 日期）。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

_MASK = (1 << 31) - 1


class LCG:
    """最小标准线性同余发生器（Numerical Recipes 参数），逐位可复现。"""

    def __init__(self, seed: int = 12345) -> None:
        self.state = seed & _MASK

    def next_u01(self) -> float:
        self.state = (1103515245 * self.state + 12345) & _MASK
        return self.state / _MASK

    def next_normal(self) -> float:
        """Box–Muller（单值）：均 0 方差 1 的伪正态。"""
        import math

        u1 = max(self.next_u01(), 1e-12)
        u2 = self.next_u01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def dates(n: int, start: str = "2025-01-01") -> list[str]:
    """n 个严格递增的 ISO 日期（自然日）。"""
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


def as_of(n: int, start: str = "2025-01-01") -> str:
    """恰好覆盖 n 个日期序列的 as_of（= 最后一个日期）。"""
    return dates(n, start)[-1]


def of_dates_max(ds: list[str]) -> str:
    """给定日期列表时的 as_of（= 最大日期）——序列由夹具生成时的便利入口。"""
    return max(ds)


# ── 技术指标用：OHLCV ────────────────────────────────────────────


def ohlcv(n: int = 120, seed: int = 7, start: str = "2025-01-01") -> dict:
    """确定性 OHLCV：随机游走收盘价，高低开在收盘附近，量为正。

    返回 `{"dates": [...], "open": [...], "high": [...], "low": [...],
    "close": [...], "volume": [...]}`。high ≥ max(open,close) 且
    low ≤ min(open,close)，因此 ATR/KDJ/DMI 的输入是合法的。
    """
    rng = LCG(seed)
    ds = dates(n, start)
    close: list[float] = []
    open_: list[float] = []
    high: list[float] = []
    low: list[float] = []
    vol: list[float] = []

    px = 20.0
    prev = px
    for _ in range(n):
        px = max(1.0, px * (1.0 + 0.012 * rng.next_normal()))
        # 开盘 = 昨收附近的小跳空
        o = max(0.5, prev * (1.0 + 0.004 * rng.next_normal()))
        hi = max(o, px) * (1.0 + 0.006 * abs(rng.next_normal()) + 0.001)
        lo = min(o, px) * (1.0 - 0.006 * abs(rng.next_normal()) - 0.001)
        v = 1_000_000.0 * (1.0 + 0.5 * abs(rng.next_normal()))
        open_.append(round(o, 4))
        high.append(round(hi, 4))
        low.append(round(lo, 4))
        close.append(round(px, 4))
        vol.append(round(v, 2))
        prev = px

    return {"dates": ds, "open": open_, "high": high, "low": low,
            "close": close, "volume": vol}


def monotone_up(n: int = 60, start: str = "2025-01-01") -> dict:
    """只涨收盘价（RSI 趋近 100 的经典夹具），K 线形状合法。"""
    ds = dates(n, start)
    close = [10.0 + 0.5 * i for i in range(n)]
    return {
        "dates": ds, "open": [c - 0.2 for c in close], "close": close,
        "high": [c + 0.3 for c in close], "low": [c - 0.4 for c in close],
        "volume": [1_000_000.0 + 1000.0 * i for i in range(n)],
    }


def monotone_down(n: int = 60, start: str = "2025-01-01") -> dict:
    ds = dates(n, start)
    close = [60.0 - 0.5 * i for i in range(n)]
    return {
        "dates": ds, "open": [c + 0.2 for c in close], "close": close,
        "high": [c + 0.4 for c in close], "low": [c - 0.3 for c in close],
        "volume": [500_000.0 + 500.0 * i for i in range(n)],
    }


def with_nan(values: list[Optional[float]], positions: tuple[int, ...]) -> list:
    out = list(values)
    for p in positions:
        out[p] = None
    return out


# ── 组合统计用：权益曲线与收益序列 ───────────────────────────────


def equity_updown(up: int = 30, down: int = 20, start: str = "2025-01-01"):
    """先涨后跌：手算最大回撤的幅度与起止日唯一确定。

    涨幅段每日 +2%（30 日），跌幅段每日 -2.5%（20 日）。
    峰值在索引 up-1，谷底在末位。
    """
    n = up + down
    ds = dates(n, start)
    vals = [100.0]
    for _ in range(up - 1):
        vals.append(vals[-1] * 1.02)
    for _ in range(down):
        vals.append(vals[-1] * 0.975)
    return ds, vals


def returns(n: int = 250, seed: int = 11, mu: float = 0.0004, sigma: float = 0.01):
    """确定性日收益序列（小数，非百分数）。"""
    rng = LCG(seed)
    return [mu + sigma * rng.next_normal() for _ in range(n)]


def equity_from_returns(rets: list[float], start: str = "2025-01-01", base: float = 100.0):
    ds = dates(len(rets) + 1, start)
    vals = [base]
    for r in rets:
        vals.append(vals[-1] * (1.0 + r))
    return ds, vals


# ── 估值算子用：三表与假设 ───────────────────────────────────────


def three_statements() -> dict:
    """一组自洽的三表输入（量级为亿元，内部一致：资产=负债+权益）。"""
    return {
        "revenue": 1000.0,
        "cost_of_goods_sold": 600.0,
        "gross_profit": 400.0,
        "operating_income": 180.0,
        "net_income": 130.0,
        "interest_expense": 20.0,
        "income_tax": 30.0,
        "total_assets": 2000.0,
        "total_equity": 1200.0,
        "total_liabilities": 800.0,
        "current_assets": 700.0,
        "current_liabilities": 350.0,
        "inventory": 150.0,
        "accounts_receivable": 120.0,
        "accounts_payable": 90.0,
        "cash_from_operations": 210.0,
        "capital_expenditure": 80.0,
        "ebit": 180.0,
        "revenue_prior": 900.0,
        "net_income_prior": 110.0,
        "accounts_receivable_prior": 75.0,
        "inventory_prior": 140.0,
        "gross_margin_prior": 0.36,
    }


def dcf_inputs() -> dict:
    return {
        "free_cash_flows": [100.0, 110.0, 121.0, 133.1, 146.4],
        "discount_rate": 0.09,
        "terminal_growth": 0.03,
        "shares_outstanding": 50.0,
        "net_debt": 200.0,
        "exit_multiple": 12.0,
        "exit_metric": 200.0,
    }
