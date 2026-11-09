"""PriceSeriesRef 契约（add-price-supply-and-caliber task 4.1）。

这个契约的存在理由是**拒绝**：没有水位、没有锚点、口径越界的引用都必须被挡在
契约边界，而不是被下游各自猜。指纹则要能自证"输入变了"——水位前移必换指纹。
"""
from __future__ import annotations

import json

import pytest

from fd_quant.contracts.models import (
    PRICE_ADJUSTMENTS,
    SUPPORTED_SCHEMA_VERSION,
    ContractError,
    PriceSeriesRef,
    compute_price_series_fingerprint,
    check_price_series_ref,
    load_schema,
    validate,
    validate_price_series_ref,
)


def _valid(**over):
    base = dict(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        source_id="astock_daily",
        symbol="600036",
        adjustment="qfq",
        adjustment_anchor="2026-07-10",
        as_of_max="2026-07-10",
        fingerprint="a" * 64,
    )
    base.update(over)
    return base


# ------------------------------------------------------------ 形状与拒绝


def test_valid_ref_passes():
    assert check_price_series_ref(_valid()) == []
    assert validate_price_series_ref(_valid())["valid"] is True


@pytest.mark.parametrize("missing", ["source_id", "symbol", "adjustment", "as_of_max", "fingerprint"])
def test_missing_required_field_is_rejected(missing):
    d = _valid()
    d.pop(missing)
    errs = check_price_series_ref(d)
    assert any(e["path"] == missing for e in errs), errs


def test_as_of_max_is_mandatory_and_must_be_a_date():
    """没有水位的引用无法回答"这批价格到哪天为止"——而那是陈旧度判定的唯一依据。"""
    errs = check_price_series_ref(_valid(as_of_max=None))
    assert any(e["path"] == "as_of_max" for e in errs)
    errs = check_price_series_ref(_valid(as_of_max="2026/07/10"))
    assert any(e["path"] == "as_of_max" for e in errs)


def test_qfq_without_anchor_is_rejected():
    d = _valid()
    d.pop("adjustment_anchor")
    errs = check_price_series_ref(d)
    assert any(e["path"] == "adjustment_anchor" for e in errs)


def test_index_adjustment_is_not_applicable_not_blank():
    assert check_price_series_ref(_valid(adjustment="not_applicable", adjustment_anchor=None)) == []
    # 空口径不许冒充"不适用"
    errs = check_price_series_ref(_valid(adjustment="", adjustment_anchor=None))
    assert any(e["path"] == "adjustment" for e in errs)


def test_unknown_adjustment_is_rejected():
    errs = check_price_series_ref(_valid(adjustment="adj"))
    assert any(e["path"] == "adjustment" for e in errs)
    assert "not_applicable" in " ".join(PRICE_ADJUSTMENTS)


def test_unknown_field_is_rejected():
    errs = check_price_series_ref(_valid(extra_note="hello"))
    assert any(e["path"] == "extra_note" for e in errs)


def test_bad_fingerprint_length_is_rejected():
    errs = check_price_series_ref(_valid(fingerprint="abc"))
    assert any(e["path"] == "fingerprint" for e in errs)


def test_wrong_schema_version_is_rejected():
    errs = check_price_series_ref(_valid(schema_version="9"))
    assert any(e["path"] == "schema_version" for e in errs)


def test_row_count_must_be_non_negative_int():
    assert check_price_series_ref(_valid(row_count=0)) == []
    assert any(e["path"] == "row_count" for e in check_price_series_ref(_valid(row_count=-1)))
    assert any(e["path"] == "row_count" for e in check_price_series_ref(_valid(row_count="400")))


# ------------------------------------------------------------ 指纹


def test_fingerprint_is_stable_for_identical_inputs():
    a = compute_price_series_fingerprint(
        source_id="astock_daily", symbol="600036", adjustment="qfq",
        as_of_max="2026-07-10", adjustment_anchor="2026-07-10", row_count=400,
    )
    b = compute_price_series_fingerprint(
        source_id="astock_daily", symbol="600036", adjustment="qfq",
        as_of_max="2026-07-10", adjustment_anchor="2026-07-10", row_count=400,
    )
    assert a == b and len(a) == 64


def test_fingerprint_changes_when_watermark_moves():
    a = PriceSeriesRef.create(source_id="s", symbol="600036", adjustment="qfq",
                              adjustment_anchor="2026-07-10", as_of_max="2026-07-10")
    b = PriceSeriesRef.create(source_id="s", symbol="600036", adjustment="qfq",
                              adjustment_anchor="2026-07-10", as_of_max="2026-07-11")
    assert a.fingerprint != b.fingerprint


def test_fingerprint_changes_when_anchor_changes():
    """换锚点就是换口径：指纹必须变，否则"换过锚点"这件事在数据上不可见。"""
    a = PriceSeriesRef.create(source_id="s", symbol="600036", adjustment="qfq",
                              adjustment_anchor="2026-07-10", as_of_max="2026-07-10")
    b = PriceSeriesRef.create(source_id="s", symbol="600036", adjustment="qfq",
                              adjustment_anchor="2026-08-06", as_of_max="2026-07-10")
    assert a.fingerprint != b.fingerprint


# ------------------------------------------------------------ 实例与往返


def test_create_computes_its_own_fingerprint():
    r = PriceSeriesRef.create(source_id="astock_daily", symbol="600036", adjustment="qfq",
                              adjustment_anchor="2026-07-10", as_of_max="2026-07-10")
    assert r.fingerprint == compute_price_series_fingerprint(
        source_id="astock_daily", symbol="600036", adjustment="qfq",
        as_of_max="2026-07-10", adjustment_anchor="2026-07-10",
    )
    assert validate_price_series_ref(r)["valid"] is True


def test_roundtrip_is_field_identical():
    r = PriceSeriesRef.create(source_id="s", symbol="600036", adjustment="qfq",
                              adjustment_anchor="2026-07-10", as_of_max="2026-07-10",
                              window_start="2025-09-23", window_end="2026-07-10",
                              row_count=400)
    assert PriceSeriesRef.from_json(r.to_json()).to_dict() == r.to_dict()


def test_from_dict_refuses_invalid():
    with pytest.raises(ContractError):
        PriceSeriesRef.from_dict(_valid(as_of_max="nope"))


def test_matches_series_compares_caliber_and_fingerprint():
    a = PriceSeriesRef.create(source_id="s", symbol="600036", adjustment="qfq",
                              adjustment_anchor="2026-07-10", as_of_max="2026-07-10")
    b = PriceSeriesRef.create(source_id="s", symbol="600036", adjustment="qfq",
                              adjustment_anchor="2026-07-10", as_of_max="2026-07-10")
    c = PriceSeriesRef.create(source_id="t", symbol="600036", adjustment="qfq",
                              adjustment_anchor="2026-07-10", as_of_max="2026-07-10")
    assert a.matches_series(b) is True
    assert a.matches_series(c) is False


# ------------------------------------------------------------ 统一入口


def test_validate_sniffs_ref_kind():
    assert validate(_valid(), kind=None)["valid"] is True
    assert validate({"schema_version": "1", "source_id": "s", "fingerprint": "a" * 64,
                     "symbol": "x", "adjustment": "none", "as_of_max": "2026-01-01"})["valid"] is True


def test_validate_accepts_instance():
    r = PriceSeriesRef.create(source_id="s", symbol="600036", adjustment="none",
                              as_of_max="2026-07-10")
    assert validate(r)["valid"] is True


def test_load_schema_returns_the_file():
    s = load_schema("price_series_ref")
    assert s["title"] == "PriceSeriesRef"
    assert set(s["required"]) == {"schema_version", "source_id", "symbol", "adjustment", "as_of_max", "fingerprint"}
    # schema 与 Python 侧必须同判：未知字段一律拒绝
    assert s["additionalProperties"] is False


def test_schema_and_python_agree_on_adjustment_enum():
    s = load_schema("price_series_ref")
    assert tuple(s["properties"]["adjustment"]["enum"]) == PRICE_ADJUSTMENTS


def test_json_schema_parses():
    load_schema("price_series_ref")  # raises if malformed
    assert json.dumps(load_schema("price_series_ref"))
