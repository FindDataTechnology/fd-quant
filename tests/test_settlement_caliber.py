"""结算入口的口径断言（add-price-supply-and-caliber task 4.5）。

三态结论是这条的核心：comparable / 不可比 / 未检查。**未检查不是可比**——
两侧都没 ref 是历史记录，两侧只有一侧有 ref 是缺口，两者都不该被读成"口径一致"。
"""
from __future__ import annotations

import pytest

from fd_quant.contracts.models import ContractError, PriceSeriesRef
from fd_quant.ledger.errors import LedgerError, SettlementRefused
from fd_quant.ledger.store import LedgerStore

DATES = ["2026-06-30", "2026-07-01", "2026-07-02"]
PROD = {"pack": "p", "skill": "k"}


def _rating():
    return {"schema_version": "1", "as_of": "2026-06-30", "subject": "600519",
            "side": "buy", "basis": "t",
            "evidence": [{"evidence_id": "e", "label": "l", "value": 1, "unit": "u",
                          "as_of": "2026-06-30", "source": "s"}],
            "producer": PROD}


def _ref(source="astock_daily", anchor="2026-07-10", as_of_max="2026-07-10"):
    return PriceSeriesRef.create(source_id=source, symbol="600519", adjustment="qfq",
                                 adjustment_anchor=anchor, as_of_max=as_of_max)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv("PRICE_CALIBER_ENFORCEMENT", raising=False)
    return LedgerStore(tmp_path / "l.db")


def _record(store, ref=None):
    return store.record(caller="c1", as_of="2026-06-30", symbol="600519", pack="p",
                        skill="k", payload=_rating(), series_ref=ref)["decision_id"]


def _settle(store, did, ref=None):
    return store.settle(decision_id=did, symbol_dates=DATES,
                        symbol_prices=[10.0, 11.0, 12.0],
                        benchmark_dates=DATES, benchmark_prices=[100.0, 100.0, 101.0],
                        as_of="2026-07-02", benchmark_symbol="000300",
                        holding_days=2, series_ref=ref)


# ------------------------------------------------------- 可比 / 不可比


def test_same_caliber_is_comparable(store):
    out = _settle(store, _record(store, _ref()), _ref())
    cal = out["caliber"]
    assert cal["checked"] is True and cal["comparable"] is True
    assert cal["source_id"] == "astock_daily"
    assert "warning" not in cal


def test_different_source_is_not_comparable_and_warns(store):
    out = _settle(store, _record(store, _ref("astock_daily")), _ref("akshare"))
    cal = out["caliber"]
    assert cal["checked"] is True and cal["comparable"] is False
    assert cal["warning"] is True
    assert "source_id" in cal["difference"]
    # 关键：warn 下结算仍然完成——口径不一致标记为不可比，不是拦住结算
    assert out["created"] is True


def test_different_anchor_is_not_comparable(store):
    """同源但锚点不同 → 仍是不同口径（实测两源同日整差 1.00 元的根因）。"""
    out = _settle(store, _record(store, _ref(anchor="2026-07-10")),
                  _ref(anchor="2026-08-06"))
    cal = out["caliber"]
    assert cal["comparable"] is False
    assert "adjustment_anchor" in cal["difference"]


def test_different_watermark_is_not_comparable(store):
    out = _settle(store, _record(store, _ref(as_of_max="2026-07-10")),
                  _ref(as_of_max="2026-07-30"))
    assert out["caliber"]["comparable"] is False
    assert "as_of_max" in out["caliber"]["difference"]


# ------------------------------------------------------- 缺口不等于一致


def test_only_one_side_has_ref_is_not_comparable(store):
    """两侧只有一侧有 ref → 不可比。把它当"一致"等于默认缺失那侧同源。"""
    out = _settle(store, _record(store, _ref()), None)
    cal = out["caliber"]
    assert cal["checked"] is True and cal["comparable"] is False
    assert "only on the record side" in cal["reason"]


def test_only_settlement_side_has_ref_is_not_comparable(store):
    out = _settle(store, _record(store, None), _ref())
    cal = out["caliber"]
    assert cal["comparable"] is False
    assert "only on the settlement side" in cal["reason"]


def test_neither_side_has_ref_is_unchecked_not_comparable(store):
    """历史记录（两侧都没 ref）→ checked:false，comparable:null。
    **不读成 true**：没检查过不等于一致。"""
    out = _settle(store, _record(store, None), None)
    cal = out["caliber"]
    assert cal["checked"] is False
    assert cal["comparable"] is None


# ------------------------------------------------------- refuse 开关


def test_refuse_mode_blocks_mismatch(store, monkeypatch):
    monkeypatch.setenv("PRICE_CALIBER_ENFORCEMENT", "refuse")
    did = _record(store, _ref("astock_daily"))
    with pytest.raises(SettlementRefused, match="not comparable"):
        _settle(store, did, _ref("akshare"))
    # 被拒后记录仍是 pending，不落半行
    assert store.get(did)["status"] == "pending"


def test_refuse_mode_blocks_one_sided_ref(store, monkeypatch):
    monkeypatch.setenv("PRICE_CALIBER_ENFORCEMENT", "refuse")
    with pytest.raises(SettlementRefused, match="only on the record side"):
        _settle(store, _record(store, _ref()), None)


def test_refuse_mode_allows_identical_ref(store, monkeypatch):
    monkeypatch.setenv("PRICE_CALIBER_ENFORCEMENT", "refuse")
    out = _settle(store, _record(store, _ref()), _ref())
    assert out["created"] is True and out["caliber"]["comparable"] is True


def test_refuse_mode_allows_legacy_records_without_refs(store, monkeypatch):
    """两侧都没 ref 时不拦——否则历史记录永远结不了账。"""
    monkeypatch.setenv("PRICE_CALIBER_ENFORCEMENT", "refuse")
    out = _settle(store, _record(store, None), None)
    assert out["created"] is True and out["caliber"]["checked"] is False


def test_invalid_switch_value_is_refused(store, monkeypatch):
    monkeypatch.setenv("PRICE_CALIBER_ENFORCEMENT", "maybe")
    with pytest.raises(LedgerError, match="must be 'warn' or 'refuse'"):
        _settle(store, _record(store, _ref()), _ref())


# ------------------------------------------------------- 落账侧校验与读回


def test_invalid_ref_is_rejected_at_record_time(store):
    """不合法 ref 当场拒绝，不留"看着有 ref 其实缺字段"的行。"""
    with pytest.raises(ContractError):
        _record(store, {"source_id": "x", "symbol": "y", "adjustment": "qfq",
                        "as_of_max": "2026-07-10"})  # 缺 anchor 与 fingerprint


def test_recorded_ref_is_readable_back(store):
    did = _record(store, _ref())
    got = store.get(did)
    assert got["series_ref"]["source_id"] == "astock_daily"
    assert len(got["series_ref"]["fingerprint"]) == 64
