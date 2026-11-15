"""`settle_decision` 的描述必须自足——尤其是「窗口够不够」这条，它只能用一次调用学会。

为什么有这条测试（2026-11-15）：`record_decision` 的实测是 **17 次调用 / 16 次被拒**
（见 `tests/test_record_decision_docstring.py`），拒因逐个不同、一次拒收才暴露一个字段。
`settle_decision` 的失败形态更贵——它不在字段名上错，而在**窗口长度**上错：
只传「入场日 + 出场日」两个点看起来完全正确，实际会被
`holding period not covered … missing N trading days` 拒掉，而且状态保持 `pending`，
**报告里却像是结过了**。首个到期日是 2026-11-20，这类错误会在最贵的那一天出现。

所以这里钉三件事：① 描述里必须有可复制的调用范例与「≥ holding_days+1 个对齐点」这条；
② 范例的形状必须真能结算（端到端跑一遍）；③ 两个典型误用必须真被拒，且拒收文本点名缺口。
"""
from __future__ import annotations

import pytest

from fd_quant.ledger.tools import record_decision, settle_decision

ENTRY = "2026-11-13"
# 入场日 + 5 个后续交易日（holding_days=5 ⇒ 需要 6 个对齐点）
DATES = ["2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
SYMBOL = [10.00, 10.20, 10.15, 10.40, 10.55, 10.80]
BENCH = [3800.0, 3812.0, 3805.0, 3820.0, 3828.0, 3841.0]


def _payload() -> dict:
    return {
        "schema_version": "1",
        "as_of": ENTRY,
        "subject": "600519.SS",
        "side": "buy",
        "conviction": 0.6,
        "price_target": None,
        "target_evidence_ids": [],
        "basis": "测试用判定",
        "time_horizon": "3M",
        "evidence": [],
        "producer": {"pack": "投研晨会", "skill": "invest-decision-log"},
    }


def _record(db: str) -> str:
    out = record_decision(
        as_of=ENTRY,
        pack="投研晨会",
        skill="invest-decision-log",
        payload=_payload(),
        symbol="600519.SS",
        db_path=db,
    )
    return out["result"]["decision_id"]


# ── 描述本身 ────────────────────────────────────────────────────────────────

def test_docstring_carries_a_copyable_call() -> None:
    doc = settle_decision.__doc__ or ""
    assert "settle_decision(" in doc, "描述里必须有可复制的调用范例（不是指向仓内文件）"
    for arg in ("decision_id=", "as_of=", "symbol_dates", "symbol_prices",
                "benchmark_dates", "benchmark_prices", "benchmark_symbol="):
        assert arg in doc, f"范例缺参数 {arg}"


def test_docstring_states_the_window_requirement() -> None:
    doc = settle_decision.__doc__ or ""
    assert "holding_days + 1" in doc, "必须写出「入场日 + holding_days 个后续交易日」这条"
    assert "交集" in doc, "必须写出日期轴取两条序列的交集（任何一边缺一天就少一天）"
    assert "结算口径日" in doc, "必须写清 as_of 是结算口径日，不是入场日"


# ── 范例形状真能结算 ────────────────────────────────────────────────────────

def test_example_shape_settles_end_to_end(tmp_path) -> None:
    db = str(tmp_path / "ledger.db")
    did = _record(db)
    out = settle_decision(
        decision_id=did,
        as_of="2026-11-20",
        symbol_dates=DATES,
        symbol_prices=SYMBOL,
        benchmark_dates=DATES,
        benchmark_prices=BENCH,
        benchmark_symbol="000300.SH",
        holding_days=5,
        db_path=db,
    )
    r = out["result"]
    assert r["created"] is True
    assert r["status"] == "settled"
    assert r["settlement_date"] == "2026-11-20"       # 入场日 + 5 个交易日
    assert r["holding_days"] == 5
    # 10.00 → 10.80 = +8%；3800 → 3841 ≈ +1.079%；alpha = raw − bench
    assert r["raw_return"] == pytest.approx(0.08)
    assert r["benchmark_return"] == pytest.approx((3841.0 - 3800.0) / 3800.0)
    assert r["alpha"] == pytest.approx(0.08 - (3841.0 - 3800.0) / 3800.0)


def test_settlement_is_idempotent(tmp_path) -> None:
    db = str(tmp_path / "ledger.db")
    did = _record(db)
    kw = dict(
        decision_id=did, as_of="2026-11-20",
        symbol_dates=DATES, symbol_prices=SYMBOL,
        benchmark_dates=DATES, benchmark_prices=BENCH,
        benchmark_symbol="000300.SH", holding_days=5, db_path=db,
    )
    settle_decision(**kw)
    again = settle_decision(**kw)["result"]
    assert again["created"] is False
    assert again["reason"] == "duplicate_settlement"
    assert again["counts"]["settlements"] == 1, "重复结算不许插第二行"


# ── 两个典型误用必须真被拒 ──────────────────────────────────────────────────

def test_two_points_are_refused_with_the_missing_count(tmp_path) -> None:
    """只给入场日与出场日——最像「对的」那种写法，必须被拒且点名缺几天。"""
    db = str(tmp_path / "ledger.db")
    did = _record(db)
    with pytest.raises(Exception) as ei:
        settle_decision(
            decision_id=did, as_of="2026-11-20",
            symbol_dates=[ENTRY, "2026-11-20"], symbol_prices=[10.00, 10.80],
            benchmark_dates=[ENTRY, "2026-11-20"], benchmark_prices=[3800.0, 3841.0],
            benchmark_symbol="000300.SH", holding_days=5, db_path=db,
        )
    text = str(ei.value)
    assert "missing 4 trading days" in text, f"应点名缺 4 个交易日，实际：{text[:200]}"


def test_future_price_point_is_refused(tmp_path) -> None:
    db = str(tmp_path / "ledger.db")
    did = _record(db)
    with pytest.raises(Exception) as ei:
        settle_decision(
            decision_id=did, as_of="2026-11-20",
            symbol_dates=DATES + ["2026-11-23"], symbol_prices=SYMBOL + [11.00],
            benchmark_dates=DATES, benchmark_prices=BENCH,
            benchmark_symbol="000300.SH", holding_days=5, db_path=db,
        )
    assert "2026-11-23" in str(ei.value), "越界日期要被点名，不许静默截断"


def test_refusal_leaves_the_decision_pending(tmp_path) -> None:
    """被拒 = 一行不落、状态仍是 pending——「半窗 alpha 比没有 alpha 更坏」。"""
    from fd_quant.ledger.tools import list_pending_decisions

    db = str(tmp_path / "ledger.db")
    did = _record(db)
    with pytest.raises(Exception):
        settle_decision(
            decision_id=did, as_of="2026-11-20",
            symbol_dates=[ENTRY, "2026-11-20"], symbol_prices=[10.00, 10.80],
            benchmark_dates=[ENTRY, "2026-11-20"], benchmark_prices=[3800.0, 3841.0],
            benchmark_symbol="000300.SH", holding_days=5, db_path=db,
        )
    pending = list_pending_decisions(as_of="2026-11-20", db_path=db)
    ids = [p["decision_id"] for p in pending["result"]["items"]]
    assert did in ids, "被拒的结算不许把决策变成 settled"
