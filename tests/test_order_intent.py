"""OrderIntent 出口与对账（change4 tasks 4.1-4.4）。

四条硬约束各配拒绝用例：出处必填、限价必带价、paper_only 不可关、判据不可用不许导出。
对账部分钉住那个最容易被漏的第三态（未触发）与孤儿。
"""
from __future__ import annotations

import pytest

from fd_quant.order_intent import (
    ALLOWED_CHANNELS,
    FILLED,
    NOT_TRIGGERED,
    ORPHAN,
    REJECTED,
    ChannelNotAvailable,
    OrderIntent,
    OrderIntentError,
    check_order_intent,
    export_order_intent,
    reconcile,
)

OK_VERDICT = {"status": "ok", "items": [], "breached": False}
PRODUCER = {"pack": "wire-equity-research", "skill": "portfolio-construction"}


def _intent(**over):
    base = dict(
        schema_version="1", intent_id="oi-1", as_of="2026-06-30", account_id="acc-1",
        symbol="600519", side="buy", order_type="market", quantity=100.0,
        derived_from=["2026-06-30-eq"], risk_verdict=OK_VERDICT, producer=PRODUCER,
        paper_only=True,
    )
    base.update(over)
    return base


# ------------------------------------------------------- 4.1 形状与出处


def test_valid_intent_passes():
    assert check_order_intent(_intent()) == []


@pytest.mark.parametrize("missing", ["intent_id", "as_of", "account_id", "symbol",
                                     "derived_from", "risk_verdict", "paper_only"])
def test_missing_required_is_rejected(missing):
    d = _intent()
    d.pop(missing)
    assert any(e["path"] == missing for e in check_order_intent(d))


def test_empty_derived_from_is_rejected():
    """没有出处的订单意图无法回答"这笔单是谁的决定"。"""
    errs = check_order_intent(_intent(derived_from=[]))
    assert any(e["path"] == "derived_from" and "traceable" in e["reason"] for e in errs)


def test_limit_order_without_price_is_rejected():
    errs = check_order_intent(_intent(order_type="limit"))
    assert any(e["path"] == "limit_price" for e in errs)


def test_limit_order_with_price_passes():
    assert check_order_intent(_intent(order_type="limit", limit_price=1500.0)) == []


def test_quantity_or_target_weight_required():
    d = _intent()
    d.pop("quantity")
    errs = check_order_intent(d)
    assert any(e["path"] == "quantity" for e in errs)
    # 给 target_weight 就合法（此时 quantity 缺省，不该再报缺）
    assert check_order_intent(_intent(target_weight=0.4, **{"quantity": None})) == []


def test_unknown_side_and_order_type_are_rejected():
    assert any(e["path"] == "side" for e in check_order_intent(_intent(side="hold")))
    assert any(e["path"] == "order_type"
               for e in check_order_intent(_intent(order_type="stop")))


def test_paper_only_cannot_be_switched_off():
    """paper_only 恒真：不存在把它设为假的合法路径。"""
    errs = check_order_intent(_intent(paper_only=False))
    assert any(e["path"] == "paper_only" for e in errs)


def test_dataclass_always_serialises_paper_only_true():
    it = OrderIntent(intent_id="x", as_of="2026-06-30", account_id="a", symbol="S",
                     side="buy", order_type="market", quantity=1.0,
                     derived_from=["k"], risk_verdict=OK_VERDICT, producer=PRODUCER,
                     paper_only=False)
    assert it.to_dict()["paper_only"] is True


# ------------------------------------------------------- 4.2 判据前置


def test_unavailable_verdict_blocks_export():
    """判据不可用 → 不许导出（宁可不下单，也不在不知道风险偏好时下单）。"""
    bad = {"status": "unavailable", "inputs": {"missing": ["industry_max"]}}
    with pytest.raises(OrderIntentError, match="unavailable"):
        export_order_intent(_intent(risk_verdict=bad))


def test_ok_verdict_exports_on_simulated_channel():
    out = export_order_intent(_intent())
    assert out["exported"] is True and out["channel"] == "simulated"
    assert out["paper_only"] is True


def test_breached_verdict_is_not_exported_by_the_gate_holder():
    """闸触达时执行岗本就不该构造意图；若构造了，契约允许它存在（记录被拦），
    但导出必须由调用方自己拦住 —— 这里钉住"导出本身只拒不可用"。"""
    breached = {"status": "breached", "items": [{"kind": "per_symbol"}], "breached": True}
    out = export_order_intent(_intent(risk_verdict=breached))
    assert out["exported"] is True


# ------------------------------------------------------- 4.3 通道


def test_real_channel_is_not_provided():
    assert "real" not in ALLOWED_CHANNELS
    with pytest.raises(ChannelNotAvailable, match="not provided"):
        export_order_intent(_intent(), channel="real")


def test_no_real_adapter_exists_in_the_module():
    import fd_quant.order_intent as m

    names = [n for n in dir(m) if not n.startswith("_")]
    assert not [n for n in names if "Real" in n or "Live" in n or "Broker" in n]


# ------------------------------------------------------- 4.4 三态对账


def test_reconcile_covers_three_states():
    intents = [_intent(intent_id="a"), _intent(intent_id="b"), _intent(intent_id="c")]
    out = reconcile(intents, {
        "a": {"state": FILLED},
        "b": {"state": REJECTED, "reason": "limit price unreachable"},
        "c": {"state": NOT_TRIGGERED, "reason": "never crossed; cancelled at close"},
    })
    assert out["counts"] == {FILLED: 1, REJECTED: 1, NOT_TRIGGERED: 1, ORPHAN: 0}
    assert out["accounted"] == 3
    assert out["details"]["c"]["reason"].startswith("never crossed")


def test_orphan_intent_is_reported_and_excluded_from_volume():
    intents = [_intent(intent_id="a"), _intent(intent_id="ghost")]
    out = reconcile(intents, {"a": {"state": FILLED}})
    assert out["orphans"] == ["ghost"]
    assert out["counts"][ORPHAN] == 1
    # 成交量只算三态之和，孤儿不计入
    assert out["accounted"] == 1


def test_intent_without_id_is_an_error():
    with pytest.raises(OrderIntentError):
        reconcile([{"symbol": "600519"}], {})


# ------------------------------------------- 4.1 后半句：出处必须可解析


def test_provenance_resolves_when_the_source_is_known():
    from fd_quant.order_intent import check_order_intent_provenance

    out = check_order_intent_provenance(_intent(), ["2026-06-30-eq"])
    assert out["checked"] is True and out["valid"] is True
    assert out["resolved"] == ["2026-06-30-eq"]


def test_unknown_source_is_named_not_just_counted():
    from fd_quant.order_intent import check_order_intent_provenance

    out = check_order_intent_provenance(_intent(derived_from=["gone-1"]), ["2026-06-30-eq"])
    assert out["valid"] is False
    assert out["unresolved"] == ["gone-1"]
    assert any("gone-1" in e["reason"] for e in out["errors"])


def test_export_refuses_a_dangling_origin():
    """spec：「不得进入执行」必须发生在这道门上，而不是事后审计才发现。"""
    with pytest.raises(OrderIntentError, match="does not resolve"):
        export_order_intent(_intent(derived_from=["gone-1"]),
                            known_source_ids=["2026-06-30-eq"])


def test_export_without_a_source_set_does_not_claim_it_checked():
    out = export_order_intent(_intent())
    assert out["provenance"] == {"checked": False}


def test_export_records_the_provenance_it_checked():
    out = export_order_intent(_intent(), known_source_ids=["2026-06-30-eq"])
    assert out["provenance"]["checked"] is True
    assert out["provenance"]["valid"] is True
