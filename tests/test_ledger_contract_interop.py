"""契约联调（task 4.3）：payload 即 `Rating` 实例，落账与查回后经契约校验通过。

跨 change 对拍：批 0.2 的契约层是 `payload` 的形状权威。账本不重新定义形状——
存的就是契约实例，查回后必须仍过 `validate()`，且**往返逐字相等**（可选字段省略
语义是契约的保证，账本若在存储时补了 null 就破坏了它）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import ledgerdata as ld
from fd_quant.contracts.models import ContractError, validate
from fd_quant.contracts.unsourced import unsourced_fields
from fd_quant.ledger import LedgerStore

#: 单用户本地账本场景的测试调用者（隔离新增维度后必须显式给出）
CALLER = "test-caller"

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rating" / "valid"
SYMBOL = "600519.SS"


@pytest.fixture()
def store(tmp_path) -> LedgerStore:
    return LedgerStore(tmp_path / "ledger.db")


def _fixture_payloads() -> list[tuple[str, dict]]:
    return [(p.name, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(FIXTURES.glob("*.json"))]


# ── 4.3 payload 即 Rating，落账查回过校验 ───────────────────────


@pytest.mark.parametrize(("name", "payload"), _fixture_payloads(),
                         ids=[n for n, _p in _fixture_payloads()])
def test_rating_fixtures_round_trip_through_the_ledger(store, name, payload):
    """批 0.2 的合法 Rating 夹具逐份落账、查回、再过契约校验（跨 change 对拍）。

    用**契约层的真实夹具**而不是自己造的样例：夹具是"契约认为合法"的定义集，
    账本若在其中某一份上失手（例如把空的 target_evidence_ids 存成 null），
    这条会直接红，而自造样例很可能恰好绕开那一份。
    """
    before = validate(payload, kind="rating")
    assert before["valid"], (name, before["errors"])

    r = store.record(caller=CALLER, as_of=payload["as_of"], symbol=payload["subject"],
                     pack=payload["producer"]["pack"], skill=payload["producer"]["skill"],
                     payload=payload)
    assert r["created"] is True, name

    item = store.get(r["decision_id"])
    round_tripped = item["payload"]
    after = validate(round_tripped, kind="rating")
    assert after["valid"], (name, after["errors"])

    # 往返逐字相等（省略语义不被存储层"补 null"）
    assert round_tripped == payload, name
    assert json.dumps(round_tripped, sort_keys=True) == json.dumps(payload, sort_keys=True)
    # 无出处数字检测在往返后结论不变
    assert unsourced_fields(round_tripped, kind="rating") == unsourced_fields(payload, kind="rating")


def test_rating_instance_object_is_accepted_and_stored(store):
    """传 `Rating` **实例**（不是 dict）也可以：账本调 `to_dict()` 后存，形状同契约。"""
    from fd_quant.contracts.models import Producer, Rating

    rating = Rating(as_of="2025-01-06", subject=SYMBOL, side="strong_buy", basis="实例输入。",
                    producer=Producer(pack=ld.PACK, skill=ld.SKILL), conviction=0.9)
    r = store.record(caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                     payload=rating)
    item = store.get(r["decision_id"])
    assert item["payload"] == rating.to_dict()
    assert validate(item["payload"], kind="rating")["valid"]


def test_stored_payload_is_validated_at_the_boundary_not_at_read_time(store):
    """非法 payload 在**写入**时即被拒（账本里不可能存着非法契约实例）。"""
    bad = ld.rating("2025-01-06", SYMBOL).to_dict()
    bad["conviction"] = 1.5   # 契约要求 [0, 1]
    with pytest.raises(ContractError) as ei:
        store.record(caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                     payload=bad)
    assert ei.value.path == "conviction"
    assert store.row_counts()["decisions"] == 0


def test_review_side_survives_the_round_trip(store):
    """review（"证据不足，不确定"是合法判定，不是错误）在账本里逐字保留、不被降级。"""
    payload = ld.rating("2025-01-06", SYMBOL, side="review", conviction=None,
                        basis="证据集不足以支撑方向判断。").to_dict()
    r = store.record(caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                     payload=payload)
    got = store.get(r["decision_id"])["payload"]
    assert got["side"] == "review"
    assert got["price_target"] is None and got["conviction"] is None
    assert validate(got, kind="rating")["valid"]


def test_evidence_references_are_stored_and_readable(store):
    """引用数组（evidence）落账可查：缺省取 payload.target_evidence_ids，也可显式覆盖。"""
    payload = ld.rating("2025-01-06", SYMBOL, price_target=1500.0,
                        evidence=[("ev-dcf", 1500.0, "CNY/share")]).to_dict()
    r = store.record(caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                     payload=payload)
    assert store.get(r["decision_id"])["evidence"] == ["ev-dcf"]

    other = ld.rating("2025-01-07", SYMBOL).to_dict()
    other["as_of"] = "2025-01-07"
    r2 = store.record(caller=CALLER, as_of="2025-01-07", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                      payload=other, evidence=["ref-a", "ref-b"])
    assert store.get(r2["decision_id"])["evidence"] == ["ref-a", "ref-b"]


# ── TargetWeights 关联（design.md 未禁止；本层支持）──────────────


def test_target_weights_can_be_associated_and_round_trips(store):
    """spec 的"可选关联 TargetWeights"落地形态：同一决策行内的第二列。

    存的是完整契约实例（不是 id 引用），故权重快照与判定同条记录、同一时刻可查回。
    """
    tw = ld.target_weights("2025-01-06")
    r = store.record(caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                     payload=ld.rating("2025-01-06", SYMBOL), target_weights=tw)
    item = store.get(r["decision_id"])
    assert item["target_weights"] == tw.to_dict()
    assert validate(item["target_weights"], kind="target_weights")["valid"]
    assert item["target_weights"]["weights"] == {"600519.SS": 0.5, "000001.SZ": 0.3}
    assert item["target_weights"]["cash_floor"] == 0.2


def test_illegal_target_weights_are_refused(store):
    """非法 TargetWeights（Σweights > 1）被拒——契约边界不做静默归一化，账本也不。"""
    bad = ld.target_weights("2025-01-06").to_dict()
    bad["weights"] = {"600519.SS": 0.8, "000001.SZ": 0.7}
    with pytest.raises(ContractError) as ei:
        store.record(caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                     payload=ld.rating("2025-01-06", SYMBOL), target_weights=bad)
    assert ei.value.path == "weights"
    assert store.row_counts()["decisions"] == 0


def test_records_without_target_weights_keep_the_column_null(store):
    """没关联权重时该列为 null（"没给"与"给了空权重"是两件事）。"""
    r = store.record(caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                     payload=ld.rating("2025-01-06", SYMBOL))
    assert store.get(r["decision_id"])["target_weights"] is None


def test_target_weights_survives_a_revision(store):
    """修订可以改权重而不动判定：两代各自的权重快照都查得回（历史诚实）。"""
    tw1 = ld.target_weights("2025-01-06")
    v1 = store.record(caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                      payload=ld.rating("2025-01-06", SYMBOL), target_weights=tw1)
    tw2 = ld.target_weights("2025-01-06")
    tw2.weights = {"600519.SS": 0.2, "000001.SZ": 0.1}
    tw2.cash_floor = 0.7
    v2 = store.record(caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                      payload=ld.rating("2025-01-06", SYMBOL), target_weights=tw2,
                      revises=v1["decision_id"])
    assert v2["created"] is True
    assert store.get(v1["decision_id"])["target_weights"] == tw1.to_dict()
    assert store.get(v2["decision_id"])["target_weights"] == tw2.to_dict()
