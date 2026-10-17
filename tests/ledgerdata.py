"""账本测试夹具：确定性价格序列与契约实例构造器。

**为什么单独一个文件而不改 `quantdata.py`**：`quantdata.py` 是计算面全族的共用夹具
（硬约束：不得修改），而账本需要的是"交易日轴 + 两条价格序列 + 契约实例"这一组与
指标无关的输入。放在这里使账本测试自足，且将来账本的夹具可以独立演进。

日期轴用**连续自然日**：账本不内置交易日历（持有期按"传入序列的日期集合"计交易日），
故夹具只需要严格递增的 ISO 日期；用自然日可以让"错位对齐"这类用例的意图一眼可见，
不必先在脑内换算交易日。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Sequence

from fd_quant.contracts.models import Evidence, Producer, Rating, TargetWeights

#: 账本测试统一用的默认产出方（与 fd-quant fixtures 的 pack 命名一致）。
PACK = "pack-equity-research"
SKILL = "equity-research-thesis"


def dates(n: int, start: str = "2025-01-06") -> list[str]:
    """n 个严格递增的 ISO 日期（自然日；是否交易日由调用方序列隐含）。"""
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


def shift(iso: str, days: int) -> str:
    return (date.fromisoformat(iso) + timedelta(days=days)).isoformat()


def series(values: Sequence[Optional[float]], start: str = "2025-01-06") -> dict:
    """价格序列夹具：`{"dates": [...], "prices": [...]}`（None 保留为缺失）。"""
    return {"dates": dates(len(values), start), "prices": list(values)}


def flat_then_move(p0: float, p1: float, n: int, *, move_at: Optional[int] = None) -> dict:
    """前段恒定 p0，从 `move_at`（默认最后一个点）起跳到 p1——收益手算唯一确定。"""
    at = n - 1 if move_at is None else move_at
    values = [p0] * at + [p1] * (n - at)
    return series(values)


def rating(
    as_of: str,
    subject: str,
    *,
    side: str = "buy",
    conviction: Optional[float] = 0.6,
    price_target: Optional[float] = None,
    basis: str = "构造判定：现金流稳定、估值处于历史中位。",
    evidence: Optional[list[tuple[str, float, str]]] = None,
    pack: str = PACK,
    skill: str = SKILL,
    run_id: Optional[str] = None,
) -> Rating:
    """构造一份**合法** Rating（默认无点位、无引用，故不必凑证据链）。"""
    evs: list[Evidence] = []
    refs: list[str] = []
    for ev_id, value, unit in (evidence or []):
        evs.append(Evidence(evidence_id=ev_id, label=f"构造证据 {ev_id}", value=value,
                            unit=unit, as_of=as_of, source=f"test:{ev_id}"))
        refs.append(ev_id)
    if price_target is not None and not refs:
        evs.append(Evidence(evidence_id="ev-pt", label="构造点位证据", value=price_target,
                            unit="CNY/share", as_of=as_of, source="test:ev-pt"))
        refs.append("ev-pt")
    return Rating(
        as_of=as_of, subject=subject, side=side, conviction=conviction,
        price_target=price_target, target_evidence_ids=refs, basis=basis,
        evidence=evs, time_horizon="6-12 个月",
        producer=Producer(pack=pack, skill=skill, run_id=run_id),
    )


def target_weights(as_of: str, *, pack: str = PACK, skill: str = SKILL) -> TargetWeights:
    """构造一份合法的组合权重（与 Rating 关联存储用）。"""
    return TargetWeights(
        as_of=as_of, universe=["600519.SS", "000001.SZ"],
        weights={"600519.SS": 0.5, "000001.SZ": 0.3}, cash_floor=0.2,
        producer=Producer(pack=pack, skill=skill),
    )


def sequential_id_factory(prefix: str = "dec-test"):
    """确定性 `decision_id` 生成器（`LedgerStore(id_factory=...)` 用）。

    默认的 uuid4 派生 id 在**并发**下是对的选择（无需协调），但它使"同一条演练复跑
    两次逐位一致"变成不可能——报告里全是变化的 id。需要一个可比对的产物时注入本
    生成器：与"注入时钟"同一手法，把不确定性收到一个接缝上。
    """
    counter = {"n": 0}

    def factory() -> str:
        counter["n"] += 1
        return f"{prefix}-{counter['n']:04d}"

    return factory
