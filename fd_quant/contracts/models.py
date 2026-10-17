"""契约模型与校验（纯本地、确定性）。

Python 侧与 `schemas/*.json` 是同一形状的两份表达，由 tests/test_contract_parity.py
对拍钉住（同一样例集必须两侧同判）。校验拒绝而不自动归一化：违规输入由生产者修正，
契约边界不做静默缩放或补值。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

SUPPORTED_SCHEMA_VERSION = "1"

RATING_SIDES = ("strong_buy", "buy", "hold", "sell", "strong_sell", "review")
REVIEW_SIDE = "review"

#: 权重不变式的浮点容差（Σweights ≤ 1 与 cash_floor + Σweights ≤ 1）。
WEIGHT_SUM_TOLERANCE = 1e-9

_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ContractError(Exception):
    """A contract violation, carrying a field path and a human-readable reason."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")

    def as_dict(self) -> dict:
        return {"path": self.path, "reason": self.reason}


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        return False
    try:
        _date.fromisoformat(value)
    except ValueError:
        return False
    return True


# ── shared sub-shapes ───────────────────────────────────────────


@dataclass
class Evidence:
    """One traceable number: value + unit + period + source, with a stable id."""

    evidence_id: str
    label: str
    value: Any
    unit: Optional[str]
    as_of: str
    source: str

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "as_of": self.as_of,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Evidence":
        errs = _check_evidence(d, "evidence")
        if errs:
            raise ContractError(errs[0]["path"], errs[0]["reason"])
        return cls(
            evidence_id=d["evidence_id"], label=d["label"], value=d["value"],
            unit=d.get("unit"), as_of=d["as_of"], source=d["source"],
        )


@dataclass
class Producer:
    """Who produced the instance — attribution for the decision ledger."""

    pack: str
    skill: str
    run_id: Optional[str] = None

    def to_dict(self) -> dict:
        # Omit an unset optional field rather than emitting `run_id: null`:
        # round-trip equivalence is a contract guarantee, and adding a key the
        # input did not have would break it.
        d = {"pack": self.pack, "skill": self.skill}
        if self.run_id is not None:
            d["run_id"] = self.run_id
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Producer":
        errs = _check_producer(d, "producer")
        if errs:
            raise ContractError(errs[0]["path"], errs[0]["reason"])
        return cls(pack=d["pack"], skill=d["skill"], run_id=d.get("run_id"))


_EVIDENCE_FIELDS = ("evidence_id", "label", "value", "unit", "as_of", "source")
_PRODUCER_FIELDS = ("pack", "skill", "run_id")


def _check_evidence(d: Any, path: str) -> list[dict]:
    errs: list[dict] = []
    if not isinstance(d, Mapping):
        return [{"path": path, "reason": "must be an object"}]
    for k in d:
        if k not in _EVIDENCE_FIELDS:
            errs.append({"path": f"{path}.{k}", "reason": "unknown field"})
    for k in _EVIDENCE_FIELDS:
        if k not in d:
            errs.append({"path": f"{path}.{k}", "reason": "required field missing"})
    if isinstance(d.get("evidence_id"), str) and not d["evidence_id"]:
        errs.append({"path": f"{path}.evidence_id", "reason": "must be a non-empty string"})
    if not isinstance(d.get("evidence_id"), (str, type(None))):
        errs.append({"path": f"{path}.evidence_id", "reason": "must be a string"})
    if "as_of" in d and not _is_iso_date(d.get("as_of")):
        errs.append({"path": f"{path}.as_of", "reason": "must be an ISO date (YYYY-MM-DD)"})
    if "source" in d and not (isinstance(d.get("source"), str) and d["source"]):
        errs.append({"path": f"{path}.source", "reason": "must be a non-empty string"})
    if "label" in d and not (isinstance(d.get("label"), str) and d["label"]):
        errs.append({"path": f"{path}.label", "reason": "must be a non-empty string"})
    return errs


def _check_producer(d: Any, path: str) -> list[dict]:
    errs: list[dict] = []
    if not isinstance(d, Mapping):
        return [{"path": path, "reason": "must be an object"}]
    for k in d:
        if k not in _PRODUCER_FIELDS:
            errs.append({"path": f"{path}.{k}", "reason": "unknown field"})
    for k in ("pack", "skill"):
        if k not in d:
            errs.append({"path": f"{path}.{k}", "reason": "required field missing"})
        elif not (isinstance(d[k], str) and d[k]):
            errs.append({"path": f"{path}.{k}", "reason": "must be a non-empty string"})
    if "run_id" in d and d["run_id"] is not None and not isinstance(d["run_id"], str):
        errs.append({"path": f"{path}.run_id", "reason": "must be a string or null"})
    return errs


# ── Rating ──────────────────────────────────────────────────────

_RATING_FIELDS = (
    "schema_version", "as_of", "subject", "side", "conviction", "price_target",
    "target_evidence_ids", "basis", "time_horizon", "evidence", "producer",
)


def check_rating(d: Any) -> list[dict]:
    """Validate a Rating (dict or JSON string). Returns [] when valid."""
    errs: list[dict] = []
    if isinstance(d, (str, bytes)):
        try:
            d = json.loads(d)
        except (TypeError, ValueError) as e:
            return [{"path": "$", "reason": f"invalid JSON: {e}"}]
    if not isinstance(d, Mapping):
        return [{"path": "$", "reason": "must be an object"}]

    for k in d:
        if k not in _RATING_FIELDS:
            errs.append({"path": k, "reason": "unknown field"})
    if d.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errs.append({
            "path": "schema_version",
            "reason": f'must be "{SUPPORTED_SCHEMA_VERSION}" '
                      f"(got {d.get('schema_version')!r}); unsupported versions are rejected",
        })
    for k in ("as_of", "subject", "side", "basis"):
        if k not in d:
            errs.append({"path": k, "reason": "required field missing"})
    if "as_of" in d and not _is_iso_date(d.get("as_of")):
        errs.append({"path": "as_of", "reason": "must be an ISO date (YYYY-MM-DD)"})
    if "subject" in d and not (isinstance(d.get("subject"), str) and d["subject"]):
        errs.append({"path": "subject", "reason": "must be a non-empty string"})
    if "basis" in d and not (isinstance(d.get("basis"), str) and d["basis"]):
        errs.append({"path": "basis", "reason": "must be a non-empty string"})
    if "side" in d and d["side"] not in RATING_SIDES:
        errs.append({
            "path": "side",
            "reason": f"must be one of {list(RATING_SIDES)} (got {d['side']!r})",
        })
    conv = d.get("conviction", None)
    if conv is not None and not (isinstance(conv, (int, float)) and 0 <= float(conv) <= 1):
        errs.append({"path": "conviction", "reason": "must be a number in [0, 1] or null"})
    pt = d.get("price_target", None)
    if pt is not None and not isinstance(pt, (int, float)):
        errs.append({"path": "price_target", "reason": "must be a number or null"})

    # review is an explicit state: no point target, no conviction.
    if d.get("side") == REVIEW_SIDE:
        if pt is not None:
            errs.append({
                "path": "price_target",
                "reason": "must be null when side is 'review' (review means no judgement)",
            })
        if conv is not None:
            errs.append({
                "path": "conviction",
                "reason": "must be null when side is 'review' (review means no judgement)",
            })

    # evidence array shape + unique ids
    ev = d.get("evidence", [])
    ids: list[str] = []
    if not isinstance(ev, list):
        errs.append({"path": "evidence", "reason": "must be an array"})
        ev = []
    for i, item in enumerate(ev):
        errs.extend(_check_evidence(item, f"evidence[{i}]"))
        if isinstance(item, Mapping) and isinstance(item.get("evidence_id"), str):
            ids.append(item["evidence_id"])
    dupes = {i for i in ids if ids.count(i) > 1}
    for dup in sorted(dupes):
        errs.append({"path": "evidence", "reason": f"duplicate evidence_id: {dup}"})

    # target references must resolve
    refs = d.get("target_evidence_ids", [])
    if refs is None:
        refs = []
    if not isinstance(refs, list):
        errs.append({"path": "target_evidence_ids", "reason": "must be an array of strings"})
        refs = []
    if pt is not None and not refs:
        errs.append({
            "path": "target_evidence_ids",
            "reason": "must be non-empty when price_target is set (a number needs a source)",
        })
    for r in refs:
        if not isinstance(r, str) or not r:
            errs.append({"path": "target_evidence_ids", "reason": "entries must be non-empty strings"})
        elif r not in ids:
            errs.append({"path": "target_evidence_ids", "reason": f"dangling evidence reference: {r}"})

    if "producer" in d:
        errs.extend(_check_producer(d["producer"], "producer"))
    else:
        errs.append({"path": "producer", "reason": "required field missing"})
    if "time_horizon" in d and d["time_horizon"] is not None and not isinstance(d["time_horizon"], str):
        errs.append({"path": "time_horizon", "reason": "must be a string or null"})
    return errs


@dataclass
class Rating:
    """A single-subject directional judgement. See schemas/rating.schema.json."""

    as_of: str
    subject: str
    side: str
    basis: str
    producer: Producer
    evidence: list[Evidence] = field(default_factory=list)
    conviction: Optional[float] = None
    price_target: Optional[float] = None
    target_evidence_ids: list[str] = field(default_factory=list)
    time_horizon: Optional[str] = None
    schema_version: str = SUPPORTED_SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "as_of": self.as_of,
            "subject": self.subject,
            "side": self.side,
            "conviction": self.conviction,
            "price_target": self.price_target,
            "target_evidence_ids": list(self.target_evidence_ids),
            "basis": self.basis,
            "evidence": [
                e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence
            ],
            "producer": self.producer.to_dict() if isinstance(self.producer, Producer) else self.producer,
        }
        if self.time_horizon is not None:
            d["time_horizon"] = self.time_horizon
        return d

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, **kw)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Rating":
        errs = check_rating(d)
        if errs:
            raise ContractError(errs[0]["path"], errs[0]["reason"])
        return cls(
            as_of=d["as_of"], subject=d["subject"], side=d["side"], basis=d["basis"],
            producer=Producer.from_dict(d["producer"]),
            evidence=[Evidence.from_dict(e) for e in d.get("evidence", [])],
            conviction=d.get("conviction"),
            price_target=d.get("price_target"),
            target_evidence_ids=list(d.get("target_evidence_ids") or []),
            time_horizon=d.get("time_horizon"),
        )

    @classmethod
    def from_json(cls, s: str) -> "Rating":
        return cls.from_dict(json.loads(s))


# ── TargetWeights ───────────────────────────────────────────────

_TW_FIELDS = (
    "schema_version", "as_of", "universe", "weights", "cash_floor",
    "constraints", "evidence", "producer",
)
_TW_CONSTRAINT_FIELDS = ("max_weight_per_symbol", "max_weight_per_sector")


def check_target_weights(d: Any) -> list[dict]:
    """Validate TargetWeights (dict or JSON string). Returns [] when valid.

    Invariants are enforced by rejection, never by normalisation: negative
    weights, sums over 1, or symbols outside the universe are refused, not
    silently rescaled.
    """
    errs: list[dict] = []
    if isinstance(d, (str, bytes)):
        try:
            d = json.loads(d)
        except (TypeError, ValueError) as e:
            return [{"path": "$", "reason": f"invalid JSON: {e}"}]
    if not isinstance(d, Mapping):
        return [{"path": "$", "reason": "must be an object"}]

    for k in d:
        if k not in _TW_FIELDS:
            errs.append({"path": k, "reason": "unknown field"})
    if d.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errs.append({
            "path": "schema_version",
            "reason": f'must be "{SUPPORTED_SCHEMA_VERSION}" '
                      f"(got {d.get('schema_version')!r}); unsupported versions are rejected",
        })
    for k in ("as_of", "universe", "weights", "cash_floor", "producer"):
        if k not in d:
            errs.append({"path": k, "reason": "required field missing"})
    if "as_of" in d and not _is_iso_date(d.get("as_of")):
        errs.append({"path": "as_of", "reason": "must be an ISO date (YYYY-MM-DD)"})

    universe = d.get("universe", [])
    if not isinstance(universe, list) or not universe:
        errs.append({"path": "universe", "reason": "must be a non-empty array"})
        universe = []
    else:
        if len(set(universe)) != len(universe):
            errs.append({"path": "universe", "reason": "must not contain duplicates"})
        for i, s in enumerate(universe):
            if not (isinstance(s, str) and s):
                errs.append({"path": f"universe[{i}]", "reason": "must be a non-empty string"})

    weights = d.get("weights", {})
    total = 0.0
    if not isinstance(weights, Mapping):
        errs.append({"path": "weights", "reason": "must be an object of symbol → number"})
        weights = {}
    for sym, w in weights.items():
        if not isinstance(w, (int, float)) or isinstance(w, bool):
            errs.append({"path": f"weights.{sym}", "reason": "must be a number"})
            continue
        if float(w) < 0:
            errs.append({"path": f"weights.{sym}", "reason": f"must be non-negative (got {w})"})
        if universe and sym not in universe:
            errs.append({"path": f"weights.{sym}", "reason": "symbol is not in universe"})
        total += float(w)

    cash_floor = d.get("cash_floor", 0.0)
    if not isinstance(cash_floor, (int, float)) or isinstance(cash_floor, bool):
        errs.append({"path": "cash_floor", "reason": "must be a number in [0, 1]"})
    else:
        if not (0 <= float(cash_floor) <= 1):
            errs.append({"path": "cash_floor", "reason": "must be in [0, 1]"})
        combined = total + float(cash_floor)
        if total > 1 + WEIGHT_SUM_TOLERANCE:
            errs.append({
                "path": "weights",
                "reason": f"Σweights must be ≤ 1 (got {total}); rejected, not normalised",
            })
        elif combined > 1 + WEIGHT_SUM_TOLERANCE:
            errs.append({
                "path": "cash_floor",
                "reason": f"cash_floor + Σweights must be ≤ 1 "
                          f"(got {combined}); rejected, not normalised",
            })

    cons = d.get("constraints", {})
    if not isinstance(cons, Mapping):
        errs.append({"path": "constraints", "reason": "must be an object"})
    else:
        for k in cons:
            if k not in _TW_CONSTRAINT_FIELDS:
                errs.append({"path": f"constraints.{k}", "reason": "unknown field"})
        for k, v in cons.items():
            if k in _TW_CONSTRAINT_FIELDS and v is not None:
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0 <= float(v) <= 1):
                    errs.append({"path": f"constraints.{k}", "reason": "must be a number in [0, 1] or null"})

    ev = d.get("evidence", [])
    ids: list[str] = []
    if not isinstance(ev, list):
        errs.append({"path": "evidence", "reason": "must be an array"})
        ev = []
    for i, item in enumerate(ev):
        errs.extend(_check_evidence(item, f"evidence[{i}]"))
        if isinstance(item, Mapping) and isinstance(item.get("evidence_id"), str):
            ids.append(item["evidence_id"])
    dupes = {i for i in ids if ids.count(i) > 1}
    for dup in sorted(dupes):
        errs.append({"path": "evidence", "reason": f"duplicate evidence_id: {dup}"})

    if "producer" in d:
        errs.extend(_check_producer(d["producer"], "producer"))
    else:
        errs.append({"path": "producer", "reason": "required field missing"})
    return errs


@dataclass
class TargetWeights:
    """Target portfolio weights. See schemas/target-weights.schema.json."""

    as_of: str
    universe: list[str]
    weights: dict[str, float]
    producer: Producer
    cash_floor: float = 0.0
    constraints: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    schema_version: str = SUPPORTED_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "as_of": self.as_of,
            "universe": list(self.universe),
            "weights": dict(self.weights),
            "cash_floor": self.cash_floor,
            "constraints": dict(self.constraints),
            "evidence": [
                e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence
            ],
            "producer": self.producer.to_dict() if isinstance(self.producer, Producer) else self.producer,
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, **kw)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TargetWeights":
        errs = check_target_weights(d)
        if errs:
            raise ContractError(errs[0]["path"], errs[0]["reason"])
        return cls(
            as_of=d["as_of"], universe=list(d["universe"]),
            weights={k: float(v) for k, v in d["weights"].items()},
            producer=Producer.from_dict(d["producer"]),
            cash_floor=float(d.get("cash_floor", 0.0)),
            constraints=dict(d.get("constraints") or {}),
            evidence=[Evidence.from_dict(e) for e in d.get("evidence", [])],
        )

    @classmethod
    def from_json(cls, s: str) -> "TargetWeights":
        return cls.from_dict(json.loads(s))


# ── unified entry points ────────────────────────────────────────


def validate(instance: Any, kind: Optional[str] = None) -> dict:
    """Validate either contract. Returns {"valid": bool, "errors": [...]}.

    Each error carries a field path and a human-readable reason (never a bare
    "invalid"). Pure local and deterministic — no network, no DB.
    """
    d = instance
    if isinstance(d, (str, bytes)):
        try:
            d = json.loads(d)
        except (TypeError, ValueError) as e:
            return {"valid": False, "errors": [{"path": "$", "reason": f"invalid JSON: {e}"}]}
    if isinstance(d, (Rating, TargetWeights)):
        d = d.to_dict()

    if kind is None:
        kind = _sniff_kind(d)
        if kind is None:
            return {"valid": False, "errors": [{
                "path": "$",
                "reason": "cannot tell which contract this is "
                          "(expected rating fields or universe/weights fields)",
            }]}
    if kind == "rating":
        errs = check_rating(d)
    elif kind in ("target_weights", "target-weights", "weights"):
        errs = check_target_weights(d)
    else:
        return {"valid": False, "errors": [{"path": "$", "reason": f"unknown kind: {kind}"}]}
    return {"valid": not errs, "errors": errs}


def _sniff_kind(d: Any) -> Optional[str]:
    if not isinstance(d, Mapping):
        return None
    if "side" in d or "subject" in d:
        return "rating"
    if "universe" in d and "weights" in d:
        return "target_weights"
    return None


def validate_rating(instance: Any) -> dict:
    return validate(instance, kind="rating")


def validate_target_weights(instance: Any) -> dict:
    return validate(instance, kind="target_weights")


def load_schema(kind: str) -> dict:
    """Load the JSON Schema for a contract kind (for parity tests / consumers)."""
    name = {
        "rating": "rating.schema.json",
        "target_weights": "target-weights.schema.json",
    }.get(kind)
    if name is None:
        raise ContractError("kind", f"unknown contract kind: {kind}")
    return json.loads((_SCHEMA_DIR / name).read_text(encoding="utf-8"))
