"""Model-layer semantics: the rules JSON Schema cannot express, plus the
round-trip / strictness / determinism guarantees the spec calls out."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from fd_quant.contracts import (  # noqa: E402
    ContractError, Evidence, Producer, Rating, TargetWeights,
    validate, validate_rating, validate_target_weights,
)
from fd_quant.contracts.unsourced import unsourced_fields  # noqa: E402

PROD = {"pack": "p", "skill": "s"}


def _rating(**over) -> dict:
    base = {
        "schema_version": "1", "as_of": "2026-10-16", "subject": "600519.SS",
        "side": "buy", "conviction": 0.5, "price_target": None,
        "target_evidence_ids": [], "basis": "b", "time_horizon": None,
        "evidence": [], "producer": PROD,
    }
    base.update(over)
    return base


def _tw(**over) -> dict:
    base = {
        "schema_version": "1", "as_of": "2026-10-16", "universe": ["A", "B"],
        "weights": {"A": 0.5, "B": 0.3}, "cash_floor": 0.2,
        "constraints": {}, "evidence": [], "producer": PROD,
    }
    base.update(over)
    return base


# ── Rating semantics ────────────────────────────────────────────


def test_null_target_is_legal_with_a_real_side():
    """扣值不扣结论: keep the direction, withhold the number."""
    r = validate_rating(_rating(side="buy", price_target=None, conviction=0.6))
    assert r["valid"], r["errors"]


def test_review_requires_null_target_and_conviction():
    ok = validate_rating(_rating(side="review", price_target=None, conviction=None))
    assert ok["valid"], ok["errors"]
    bad = validate_rating(_rating(side="review", price_target=120.0, conviction=None))
    assert not bad["valid"]
    assert any(e["path"] == "price_target" for e in bad["errors"])
    bad2 = validate_rating(_rating(side="review", price_target=None, conviction=0.3))
    assert not bad2["valid"]
    assert any(e["path"] == "conviction" for e in bad2["errors"])


def test_review_is_not_silently_downgraded_by_roundtrip():
    inst = Rating.from_dict(_rating(side="review", price_target=None, conviction=None))
    again = Rating.from_json(inst.to_json())
    assert again.side == "review"
    assert again.price_target is None and again.conviction is None


def test_target_requires_resolvable_evidence():
    ev = {"evidence_id": "ev1", "label": "l", "value": 1.0, "unit": None,
          "as_of": "2026-10-16", "source": "src"}
    ok = validate_rating(_rating(price_target=100.0, target_evidence_ids=["ev1"], evidence=[ev]))
    assert ok["valid"], ok["errors"]
    dangling = validate_rating(_rating(price_target=100.0, target_evidence_ids=["evX"], evidence=[ev]))
    assert not dangling["valid"]
    assert any("dangling" in e["reason"] for e in dangling["errors"])
    missing = validate_rating(_rating(price_target=100.0, target_evidence_ids=[], evidence=[ev]))
    assert not missing["valid"]
    assert any(e["path"] == "target_evidence_ids" for e in missing["errors"])


def test_duplicate_evidence_id_rejected():
    ev = {"evidence_id": "dup", "label": "l", "value": 1.0, "unit": None,
          "as_of": "2026-10-16", "source": "src"}
    r = validate_rating(_rating(evidence=[ev, dict(ev)]))
    assert not r["valid"]
    assert any("duplicate" in e["reason"] for e in r["errors"])


def test_unsourced_detection_lists_price_target():
    assert unsourced_fields(_rating(price_target=100.0, target_evidence_ids=[])) == ["price_target"]
    assert unsourced_fields(_rating(price_target=100.0, target_evidence_ids=["gone"])) == \
        ["target_evidence_ids[gone]"]
    assert unsourced_fields(_rating(price_target=None)) == []


# ── TargetWeights invariants (rejection, never normalisation) ────


@pytest.mark.parametrize("sum_val,tol_case", [
    (1.0, "exact"),
    (1.0 + 5e-10, "inside tolerance"),
])
def test_weight_sum_within_tolerance_accepted(sum_val, tol_case):
    r = validate_target_weights(_tw(universe=["A"], weights={"A": sum_val}, cash_floor=0.0))
    assert r["valid"], (tol_case, r["errors"])


@pytest.mark.parametrize("sum_val", [1.0 + 1e-8, 1.2])
def test_weight_sum_over_tolerance_rejected_without_normalising(sum_val):
    data = _tw(universe=["A"], weights={"A": sum_val}, cash_floor=0.0)
    r = validate_target_weights(data)
    assert not r["valid"]
    # rejection, not silent rescale: the error states the sum
    assert any("Σweights" in e["reason"] for e in r["errors"])
    # the instance itself is untouched (no normalisation side effect)
    assert data["weights"]["A"] == sum_val


def test_negative_weight_rejected_including_tiny():
    assert not validate_target_weights(_tw(weights={"A": -0.1, "B": 0.3}))["valid"]
    r = validate_target_weights(_tw(universe=["A"], weights={"A": -1e-9}, cash_floor=0.0))
    assert not r["valid"]
    assert any("non-negative" in e["reason"] for e in r["errors"])


def test_zero_weight_is_fine():
    assert validate_target_weights(_tw(universe=["A"], weights={"A": 0.0}, cash_floor=0.0))["valid"]


def test_cash_floor_plus_weights_conflict_rejected():
    r = validate_target_weights(_tw(universe=["A"], weights={"A": 0.9}, cash_floor=0.2))
    assert not r["valid"]
    assert any("cash_floor + Σweights" in e["reason"] for e in r["errors"])


def test_symbol_outside_universe_rejected():
    r = validate_target_weights(_tw(universe=["A"], weights={"Z": 0.1}, cash_floor=0.0))
    assert not r["valid"]
    assert any("not in universe" in e["reason"] for e in r["errors"])


# ── round-trip + strictness + determinism ───────────────────────


def _semantically_equal(a: dict, b: dict) -> bool:
    """Compare contracts allowing the optional-field omission/null equivalence.

    The spec requires round-trip equivalence; an optional field may be written
    as absent or as explicit null (both are legal), and the model normalises to
    "absent". So equivalence is compared after folding null-valued optional
    keys away on both sides.
    """
    optional = ("time_horizon",)

    def fold(d: dict) -> dict:
        out = dict(d)
        for k in optional:
            if out.get(k, None) is None:
                out.pop(k, None)
        prod = out.get("producer")
        if isinstance(prod, dict) and prod.get("run_id", None) is None:
            out["producer"] = {k: v for k, v in prod.items() if k != "run_id"}
        return out

    return fold(a) == fold(b)


def test_rating_roundtrip_equivalence():
    raw = _rating(side="buy", conviction=0.6, price_target=123.4,
                  target_evidence_ids=["e1"],
                  evidence=[{"evidence_id": "e1", "label": "l", "value": 123.4,
                             "unit": "CNY", "as_of": "2026-10-16", "source": "s"}])
    inst = Rating.from_dict(raw)
    back = Rating.from_json(inst.to_json()).to_dict()
    assert _semantically_equal(back, raw), (back, raw)


def test_target_weights_roundtrip_equivalence():
    raw = _tw(constraints={"max_weight_per_symbol": 0.4},
              evidence=[{"evidence_id": "e1", "label": "l", "value": 1,
                         "unit": None, "as_of": "2026-10-16", "source": "s"}])
    inst = TargetWeights.from_dict(raw)
    back = TargetWeights.from_json(inst.to_json()).to_dict()
    assert _semantically_equal(back, raw), (back, raw)


def test_explicit_null_and_omitted_optional_are_both_valid():
    """Both spellings of 'no time horizon' are legal — the contract must not
    make one of them an accident."""
    with_null = validate_rating(_rating(time_horizon=None))
    omitted = _rating()
    omitted.pop("time_horizon")
    without = validate_rating(omitted)
    assert with_null["valid"] and without["valid"]


def test_unknown_field_rejected_both_contracts():
    r1 = validate(_rating(extra_note="x"))
    assert not r1["valid"] and any(e["path"] == "extra_note" for e in r1["errors"])
    r2 = validate(_tw(oops=1))
    assert not r2["valid"] and any(e["path"] == "oops" for e in r2["errors"])


def test_unsupported_version_rejected_with_supported_named():
    for bad in ("9", None, 1):
        r = validate(_rating(schema_version=bad))
        assert not r["valid"]
        assert any('"1"' in e["reason"] for e in r["errors"])


def test_validate_sniffs_kind():
    assert validate(_rating())["valid"]
    assert validate(_tw())["valid"]
    assert not validate({"nothing": "useful"})["valid"]


def test_validate_accepts_json_string_and_model():
    assert validate(_rating().__class__ and Rating.from_dict(_rating()))["valid"]
    import json as _json
    assert validate(_json.dumps(_tw()))["valid"]


def test_validation_deterministic_and_structured():
    data = _rating(side="review", price_target=5.0, conviction=0.2)
    a, b = validate_rating(data), validate_rating(data)
    assert a == b
    assert a["valid"] is False and len(a["errors"]) >= 2
    assert all("path" in e and "reason" in e for e in a["errors"])


def test_contract_error_carries_path():
    with pytest.raises(ContractError) as ei:
        Rating.from_dict(_rating(side="nope"))
    assert ei.value.path == "side"
    assert ei.value.as_dict()["reason"]
