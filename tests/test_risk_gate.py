"""组合级风控闸（change3 task 3；change4 spec portfolio-risk-gate）。

四条纪律各配拒绝用例：
1. 阈值缺失 → `unavailable`，**不是** false；
2. 权重非法 → 拒绝并原样报错，绝不归一化；
3. 行业归属缺失 → 该项"不可判定"并点名，不归入默认行业；
4. 行情陈旧 → 触达并报两侧日期。

`unavailable` 与 `breached=false` 的区分是本模块的核心：前者表示"问不出答案"，
执行侧必须当拒绝条件；后者才是"问过了，没触达"。
"""
from __future__ import annotations

import pytest

from fd_quant.risk import (
    STATUS_BREACHED,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    THRESHOLD_KEYS,
    RiskGateError,
    Thresholds,
    evaluate_portfolio_gate,
)

GOOD = {"per_symbol_max": 0.3, "industry_max": 0.5, "drawdown_max": 0.15,
        "daily_loss_max": 0.05}


def _gate(**kw):
    base = dict(weights={"A": 0.2, "B": 0.2}, thresholds=GOOD,
                sectors={"A": "tech", "B": "fin"},
                equity_series=[("2026-06-01", 100.0), ("2026-06-02", 101.0)])
    base.update(kw)
    return evaluate_portfolio_gate(**base)


# ------------------------------------------------------- 3.1 阈值读取


def test_missing_thresholds_are_unavailable_not_false():
    """缺配置 → unavailable，绝不是"未触达"。这是本模块最重要的区分。"""
    for missing in THRESHOLD_KEYS:
        partial = {k: v for k, v in GOOD.items() if k != missing}
        out = _gate(thresholds=partial)
        assert out["status"] == STATUS_UNAVAILABLE, missing
        assert missing in out["inputs"]["missing"], missing
        assert out["items"] == []


def test_no_thresholds_at_all_is_unavailable():
    out = _gate(thresholds=None)
    assert out["status"] == STATUS_UNAVAILABLE
    assert set(out["inputs"]["missing"]) == set(THRESHOLD_KEYS)


def test_all_present_and_within_limits_is_ok():
    out = _gate()
    assert out["status"] == STATUS_OK
    assert out["items"] == []


def test_thresholds_object_and_mapping_are_equivalent():
    a = _gate(thresholds=Thresholds(0.3, 0.5, 0.15, 0.05))
    b = _gate(thresholds=GOOD)
    assert a["status"] == b["status"] == STATUS_OK


@pytest.mark.parametrize("bad", [0, -0.1, 1.5, "x", True])
def test_invalid_threshold_value_is_an_input_error(bad):
    with pytest.raises(RiskGateError):
        _gate(thresholds={**GOOD, "per_symbol_max": bad})


# ------------------------------------------------------- 3.3 不归一化


def test_negative_weight_is_refused():
    with pytest.raises(RiskGateError, match="negative"):
        _gate(weights={"A": -0.1})


def test_sum_over_one_is_refused_and_not_rescaled():
    with pytest.raises(RiskGateError, match="exceeds 1"):
        _gate(weights={"A": 0.7, "B": 0.5})


def test_non_numeric_weight_is_refused():
    with pytest.raises(RiskGateError, match="must be a number"):
        _gate(weights={"A": "0.2"})


# ------------------------------------------------------- 触达（单票 / 行业）


def test_per_symbol_breach_is_named():
    out = _gate(weights={"A": 0.45, "B": 0.05})
    assert out["status"] == STATUS_BREACHED
    hit = [i for i in out["items"] if i["kind"] == "per_symbol"]
    assert hit and hit[0]["subject"] == "A"
    assert hit[0]["actual"] == pytest.approx(0.45)
    assert hit[0]["limit"] == pytest.approx(0.3)


def test_industry_breach_is_named():
    out = _gate(weights={"A": 0.3, "C": 0.26}, sectors={"A": "tech", "C": "tech"})
    hit = [i for i in out["items"] if i["kind"] == "industry_concentration"]
    assert hit and hit[0]["subject"] == "tech"
    assert hit[0]["actual"] == pytest.approx(0.56)
    assert hit[0]["limit"] == pytest.approx(0.5)


def test_industry_exactly_at_limit_does_not_breach():
    """边界：恰好等于上限不算触达（判据是严格大于）。把这条钉住，免得后来人
    用 >= 改掉它而没人发现。"""
    out = _gate(weights={"A": 0.25, "C": 0.25},
                sectors={"A": "tech", "C": "tech"})
    assert not [i for i in out["items"] if i["kind"] == "industry_concentration"]


def test_industry_just_over_limit_breaches():
    out = _gate(weights={"A": 0.26, "C": 0.25}, sectors={"A": "tech", "C": "tech"})
    hit = [i for i in out["items"] if i["kind"] == "industry_concentration"]
    assert hit and hit[0]["actual"] > hit[0]["limit"]


def test_unknown_sector_is_undetermined_not_defaulted():
    """缺归属 → 不可判定并点名；不许把它归进某个默认行业。"""
    out = _gate(weights={"A": 0.2, "ZZZ": 0.2}, sectors={"A": "tech"})
    und = [u for u in out["undetermined"] if u["kind"] == "industry_concentration"]
    assert und and "ZZZ" in und[0]["subjects"]
    # 其余判据照常给出结论
    assert any(i["kind"] == "per_symbol" for i in out["items"]) is False
    assert out["status"] == STATUS_OK


def test_missing_sector_map_is_undetermined():
    out = _gate(sectors=None)
    assert any(u["kind"] == "industry_concentration" for u in out["undetermined"])


# ------------------------------------------------------- 回撤与单日亏损


def test_drawdown_breach():
    series = [("2026-06-01", 100.0), ("2026-06-02", 120.0), ("2026-06-03", 100.0)]
    out = _gate(equity_series=series)
    hit = [i for i in out["items"] if i["kind"] == "drawdown"]
    assert hit and hit[0]["actual"] == pytest.approx(20.0 / 120.0)


def test_daily_loss_breach():
    series = [("2026-06-01", 100.0), ("2026-06-02", 100.0), ("2026-06-03", 90.0)]
    out = _gate(equity_series=series)
    hit = [i for i in out["items"] if i["kind"] == "daily_loss"]
    assert hit and hit[0]["actual"] == pytest.approx(0.10)


def test_missing_equity_series_is_undetermined_not_zero():
    out = _gate(equity_series=None)
    assert any("drawdown" in u["kind"] for u in out["undetermined"])
    assert not any(i["kind"] in ("drawdown", "daily_loss") for i in out["items"])


# ------------------------------------------------------- 3.4 行情陈旧


def test_stale_price_breaches_and_reports_both_dates():
    out = _gate(thresholds={**GOOD, "max_staleness_days": 5},
                as_of="2026-07-20", price_as_of_max="2026-07-10")
    hit = [i for i in out["items"] if i["kind"] == "price_staleness"]
    assert hit
    assert hit[0]["actual"] == 10.0 and hit[0]["limit"] == 5.0
    assert "2026-07-20" in hit[0]["note"] and "2026-07-10" in hit[0]["note"]


def test_fresh_price_does_not_breach():
    out = _gate(thresholds={**GOOD, "max_staleness_days": 30},
                as_of="2026-07-20", price_as_of_max="2026-07-10")
    assert not [i for i in out["items"] if i["kind"] == "price_staleness"]


def test_staleness_limit_unset_means_not_checked():
    """未配上限＝显式选择不判；但不等于"新鲜"——输出里不出现该判据即可。"""
    out = _gate(as_of="2026-07-20", price_as_of_max="2026-01-01")
    assert not [i for i in out["items"] if i["kind"] == "price_staleness"]


def test_staleness_without_dates_is_undetermined():
    out = _gate(thresholds={**GOOD, "max_staleness_days": 5})
    assert any(u["kind"] == "price_staleness" for u in out["undetermined"])


# ------------------------------------------------------- 纯函数性


def test_pure_and_deterministic():
    a = _gate()
    b = _gate()
    assert a == b


def test_items_carry_no_disposal_instructions():
    """判据只回答"触达了什么"，不回答"该怎么办"。"""
    out = _gate(weights={"A": 0.9}, sectors={"A": "tech"})
    assert out["status"] == STATUS_BREACHED
    text = repr(out["items"])
    for word in ("liquidate", "清仓", "减半", "pause", "暂停", "reduce"):
        assert word not in text


# ------------------------------------------------------- 3.5 阈值留痕


def test_threshold_change_is_appended_with_old_new_and_reason(tmp_path):
    from fd_quant.ledger.store import LedgerStore

    st = LedgerStore(tmp_path / "l.db")
    first = st.record_risk_thresholds(caller="c1", thresholds=GOOD, reason="初版",
                                      now="2026-07-01T00:00:00Z")
    assert first["previous"] is None
    second = st.record_risk_thresholds(
        caller="c1", thresholds={**GOOD, "per_symbol_max": 0.2}, reason="收紧单票",
        now="2026-07-02T00:00:00Z")

    # 旧值 / 新值 / 时点 / 原因 四样都在
    assert second["previous"]["thresholds"]["per_symbol_max"] == pytest.approx(0.3)
    assert second["current"]["thresholds"]["per_symbol_max"] == pytest.approx(0.2)
    assert second["current"]["recorded_at"] == "2026-07-02T00:00:00Z"
    assert second["previous"]["reason"] == "初版"
    assert second["current"]["reason"] == "收紧单票"


def test_history_is_not_rewritten(tmp_path):
    """改阈值不改写历史：第一行逐字不变。"""
    from fd_quant.ledger.store import LedgerStore

    st = LedgerStore(tmp_path / "l.db")
    st.record_risk_thresholds(caller="c1", thresholds=GOOD, reason="初版")
    st.record_risk_thresholds(caller="c1", thresholds={**GOOD, "daily_loss_max": 0.02},
                             reason="收紧日损")
    import sqlite3

    con = sqlite3.connect(st.db_path)
    rows = con.execute("SELECT reason, thresholds FROM risk_thresholds ORDER BY id").fetchall()
    con.close()
    assert len(rows) == 2
    assert rows[0][0] == "初版"
    assert '"per_symbol_max":0.3' in rows[0][1].replace(" ", "")


def test_partial_thresholds_cannot_be_recorded():
    """只写一半的阈值表读出来是"不可用"，与"从没配过"无法区分——故直接拒绝落账。"""
    from fd_quant.ledger import LedgerError
    from fd_quant.ledger.store import LedgerStore
    import tempfile, os

    st = LedgerStore(os.path.join(tempfile.mkdtemp(), "l.db"))
    with pytest.raises(LedgerError, match="missing"):
        st.record_risk_thresholds(caller="c1", thresholds={"per_symbol_max": 0.3},
                                  reason="半套")


def test_latest_is_none_when_never_configured(tmp_path):
    """没配过返回 None，**不返回空默认值**——"没配"与"配了个安全值"是两件事。"""
    from fd_quant.ledger.store import LedgerStore

    st = LedgerStore(tmp_path / "l.db")
    assert st.latest_risk_thresholds(caller="c1") is None


def test_latest_returns_most_recent(tmp_path):
    from fd_quant.ledger.store import LedgerStore

    st = LedgerStore(tmp_path / "l.db")
    st.record_risk_thresholds(caller="c1", thresholds=GOOD, reason="初版")
    st.record_risk_thresholds(caller="c1", thresholds={**GOOD, "industry_max": 0.4},
                             reason="收紧行业")
    got = st.latest_risk_thresholds(caller="c1")
    assert got["thresholds"]["industry_max"] == pytest.approx(0.4)
    assert got["reason"] == "收紧行业"


# --------------------------------------------- change4 1.1/2.2 形状对齐


def test_breached_is_tri_state_never_folding_unavailable():
    """`breached` 三态：true / false / **null**。null 表示判据不可用，
    绝不折叠成 false —— 那会让"问不出答案"看起来像"问过了没触达"。"""
    good = {"per_symbol_max": 0.45, "industry_max": 0.9, "drawdown_max": 0.5,
            "daily_loss_max": 0.5}
    ok = evaluate_portfolio_gate(weights={"A": 0.2}, thresholds=good, sectors={"A": "x"},
                                 equity_series=[("2026-06-01", 100.0),
                                                ("2026-06-02", 100.0)])
    assert ok["breached"] is False and ok["status"] == STATUS_OK

    br = evaluate_portfolio_gate(weights={"A": 0.9}, thresholds=good, sectors={"A": "x"},
                                 equity_series=[("2026-06-01", 100.0),
                                                ("2026-06-02", 100.0)])
    assert br["breached"] is True and br["status"] == STATUS_BREACHED

    un = evaluate_portfolio_gate(weights={"A": 0.2}, thresholds={"per_symbol_max": 0.45},
                                 sectors={"A": "x"})
    assert un["breached"] is None and un["status"] == STATUS_UNAVAILABLE


def test_items_is_a_list_never_none():
    """未触达时 items == []，不是 null 也不是省略。"""
    out = evaluate_portfolio_gate(
        weights={"A": 0.2},
        thresholds={"per_symbol_max": 0.45, "industry_max": 0.9,
                    "drawdown_max": 0.5, "daily_loss_max": 0.5},
        sectors={"A": "x"}, equity_series=[("2026-06-01", 100.0), ("2026-06-02", 100.0)],
    )
    assert out["items"] == [] and isinstance(out["items"], list)


# ------------------- 边界补钉：三个判据各自的"恰好等于上限"（change4 3.4）


def test_per_symbol_exactly_at_limit_does_not_breach():
    """单票恰好等于上限不触达（判据是严格大于）。

    **这条是补的，而且是翻转实验抓出来的**：原先只有行业集中度有"恰好等于上限"的用例，
    单票没有——于是把 `>` 改成 `>=` 时，全套测试仍然全绿。没有这条，那条判据的边界
    就没有任何东西钉着，"改一处"也不会有人发现。
    """
    out = _gate(weights={"A": 0.3}, thresholds={**GOOD, "per_symbol_max": 0.3})
    assert not [i for i in out["items"] if i["kind"] == "per_symbol"]
    assert out["breached"] is False


def test_per_symbol_just_over_limit_breaches():
    out = _gate(weights={"A": 0.3000001}, thresholds={**GOOD, "per_symbol_max": 0.3})
    assert [i for i in out["items"] if i["kind"] == "per_symbol"]


def test_drawdown_exactly_at_limit_does_not_breach():
    """回撤恰好等于上限不触达。构造：峰值 100、谷值 90 → 回撤 0.10；上限 0.10。"""
    out = _gate(equity_series=[("2026-06-01", 100.0), ("2026-06-02", 90.0)],
                thresholds={**GOOD, "drawdown_max": 0.10})
    assert not [i for i in out["items"] if i["kind"] == "drawdown"]


def test_daily_loss_exactly_at_limit_does_not_breach():
    """单日亏损恰好等于上限不触达。构造：100 → 95，跌幅 0.05；上限 0.05。"""
    out = _gate(equity_series=[("2026-06-01", 100.0), ("2026-06-02", 95.0)],
                thresholds={**GOOD, "daily_loss_max": 0.05})
    assert not [i for i in out["items"] if i["kind"] == "daily_loss"]
