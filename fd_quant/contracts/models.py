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


# ── PriceSeriesRef: 价格序列的出处引用 ──────────────────────────
#
# 为什么需要这个契约（openspec change: add-price-supply-and-caliber）：
# 研究取价、交易行情、结算取价是三条独立路径，各自解析源。没有一条共同的引用，
# "研报里的涨幅、回测里的收益、账本里的 alpha"就是三个互不可比的数字，而这个差别
# 不会自己喊出来。引用把"这一串价格是哪来的、哪天的、什么复权口径"变成可断言的字段。
#
# 指纹（fingerprint）的用途：同一 (源, 标的, 口径, 窗口, 水位) 必得同一指纹；水位前移
# 指纹随之改变。这样"两次结果不一致"可以被归因为"输入变了"而不是靠肉眼比数字。

PRICE_ADJUSTMENTS = ("qfq", "hfq", "none", "not_applicable")

_PSR_REQUIRED = ("source_id", "symbol", "adjustment", "as_of_max", "fingerprint")
_PSR_FIELDS = frozenset(
    _PSR_REQUIRED
    + (
        "schema_version",
        "adjustment_anchor",
        "db_source",
        "window_start",
        "window_end",
        "row_count",
        "producer",
    )
)


def compute_price_series_fingerprint(
    *,
    source_id: str,
    symbol: str,
    adjustment: str,
    as_of_max: str,
    adjustment_anchor: Optional[str] = None,
    db_source: Optional[str] = None,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    row_count: Optional[int] = None,
) -> str:
    """序列指纹：sha256 over canonical JSON of the identifying fields.

    ``as_of_max`` 参与计算，所以底层水位前移一定换指纹——这正是"输入变了"的
    可机器判定形态。不含浮点，避免跨平台的表示差异。
    """
    import hashlib

    payload = {
        "source_id": source_id,
        "db_source": db_source if db_source is not None else source_id,
        "symbol": symbol,
        "adjustment": adjustment,
        "adjustment_anchor": adjustment_anchor,
        "as_of_max": as_of_max,
        "window_start": window_start,
        "window_end": window_end,
        "row_count": row_count,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def check_price_series_ref(d: Any) -> list[dict]:
    """Validate PriceSeriesRef (dict or JSON string). Returns [] when valid."""
    errs: list[dict] = []
    if isinstance(d, (str, bytes)):
        try:
            d = json.loads(d)
        except (TypeError, ValueError) as e:
            return [{"path": "$", "reason": f"invalid JSON: {e}"}]
    if not isinstance(d, Mapping):
        return [{"path": "$", "reason": "must be an object"}]

    for k in d:
        if k not in _PSR_FIELDS:
            errs.append({"path": k, "reason": "unknown field"})
    if d.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errs.append({
            "path": "schema_version",
            "reason": f'must be "{SUPPORTED_SCHEMA_VERSION}" '
                      f"(got {d.get('schema_version')!r}); unsupported versions are rejected",
        })
    for k in _PSR_REQUIRED:
        if k not in d:
            errs.append({"path": k, "reason": "required field missing"})

    for k in ("source_id", "symbol", "fingerprint"):
        if k in d and (not isinstance(d.get(k), str) or not d.get(k)):
            errs.append({"path": k, "reason": "must be a non-empty string"})

    if "adjustment" in d and d.get("adjustment") not in PRICE_ADJUSTMENTS:
        errs.append({
            "path": "adjustment",
            "reason": f"must be one of {PRICE_ADJUSTMENTS} (got {d.get('adjustment')!r})",
        })

    # as_of_max 必填且必须是日期：没有水位的"序列引用"是无意义的——它无法回答
    # "这批价格到哪天为止"，而那正是陈旧度判定的唯一依据。
    if "as_of_max" in d and not _is_iso_date(d.get("as_of_max")):
        errs.append({"path": "as_of_max", "reason": "must be an ISO date (YYYY-MM-DD)"})

    for k in ("window_start", "window_end"):
        if d.get(k) is not None and not _is_iso_date(d.get(k)):
            errs.append({"path": k, "reason": "must be an ISO date (YYYY-MM-DD)"})

    if d.get("adjustment") in ("qfq", "hfq") and not d.get("adjustment_anchor"):
        errs.append({
            "path": "adjustment_anchor",
            "reason": f"required when adjustment is {d.get('adjustment')!r}: without an "
                      "anchor the reader cannot tell whether two batches are comparable",
        })

    if "row_count" in d and d.get("row_count") is not None:
        rc = d.get("row_count")
        if not isinstance(rc, int) or isinstance(rc, bool) or rc < 0:
            errs.append({"path": "row_count", "reason": "must be a non-negative integer"})

    if "producer" in d and d.get("producer") is not None:
        errs.extend(_check_producer(d.get("producer"), "producer"))

    fp = d.get("fingerprint")
    if isinstance(fp, str) and fp and len(fp) != 64:
        errs.append({
            "path": "fingerprint",
            "reason": "must be a 64-char sha256 hex digest "
                      f"(got length {len(fp)})",
        })

    return errs


@dataclass
class PriceSeriesRef:
    """A price series' provenance. See schemas/price-series-ref.schema.json."""

    source_id: str
    symbol: str
    adjustment: str
    as_of_max: str
    fingerprint: str
    adjustment_anchor: Optional[str] = None
    db_source: Optional[str] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    row_count: Optional[int] = None
    producer: Optional[Producer] = None
    schema_version: str = SUPPORTED_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        symbol: str,
        adjustment: str,
        as_of_max: str,
        adjustment_anchor: Optional[str] = None,
        db_source: Optional[str] = None,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
        row_count: Optional[int] = None,
        producer: Optional[Producer] = None,
    ) -> "PriceSeriesRef":
        """构造一个自洽实例：指纹由字段算出，不接受外部传入。"""
        fp = compute_price_series_fingerprint(
            source_id=source_id,
            symbol=symbol,
            adjustment=adjustment,
            as_of_max=as_of_max,
            adjustment_anchor=adjustment_anchor,
            db_source=db_source,
            window_start=window_start,
            window_end=window_end,
            row_count=row_count,
        )
        return cls(
            source_id=source_id,
            symbol=symbol,
            adjustment=adjustment,
            as_of_max=as_of_max,
            fingerprint=fp,
            adjustment_anchor=adjustment_anchor,
            db_source=db_source,
            window_start=window_start,
            window_end=window_end,
            row_count=row_count,
            producer=producer,
        )

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "symbol": self.symbol,
            "adjustment": self.adjustment,
            "as_of_max": self.as_of_max,
            "fingerprint": self.fingerprint,
        }
        if self.adjustment_anchor is not None:
            d["adjustment_anchor"] = self.adjustment_anchor
        if self.db_source is not None:
            d["db_source"] = self.db_source
        if self.window_start is not None:
            d["window_start"] = self.window_start
        if self.window_end is not None:
            d["window_end"] = self.window_end
        if self.row_count is not None:
            d["row_count"] = self.row_count
        if self.producer is not None:
            d["producer"] = (
                self.producer.to_dict()
                if isinstance(self.producer, Producer)
                else self.producer
            )
        return d

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, **kw)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PriceSeriesRef":
        errs = check_price_series_ref(d)
        if errs:
            raise ContractError(errs[0]["path"], errs[0]["reason"])
        return cls(
            source_id=d["source_id"],
            symbol=d["symbol"],
            adjustment=d["adjustment"],
            as_of_max=d["as_of_max"],
            fingerprint=d["fingerprint"],
            adjustment_anchor=d.get("adjustment_anchor"),
            db_source=d.get("db_source"),
            window_start=d.get("window_start"),
            window_end=d.get("window_end"),
            row_count=d.get("row_count"),
            producer=(
                Producer.from_dict(d["producer"]) if d.get("producer") else None
            ),
        )

    @classmethod
    def from_json(cls, s: str) -> "PriceSeriesRef":
        return cls.from_dict(json.loads(s))

    def matches_series(self, other: "PriceSeriesRef") -> bool:
        """两条引用是否指向同一批数据（口径 + 水位 + 指纹全同）。"""
        return (
            self.source_id == other.source_id
            and (self.db_source or self.source_id) == (other.db_source or other.source_id)
            and self.adjustment == other.adjustment
            and self.adjustment_anchor == other.adjustment_anchor
            and self.as_of_max == other.as_of_max
            and self.fingerprint == other.fingerprint
        )


# ── PortfolioIntent: 评级到仓位的派生契约 ──────────────────────
#
# 为什么要有第三个契约（openspec change: add-portfolio-intent-contract）：
# Rating 说的是"这只怎么样"，TargetWeights 说的是"各拿多少"，中间那一步——按什么规则
# 从前者得到后者——原先只存在于人的临场算术里。临场算术无法审计、不可复现，不同人算出的
# 权重不可比。本契约把那一步变成显式规则 + 显式出处，使第三方只凭实例就能照着重算。
#
# 它刻意**只允许三种规则**：三种都不需要估计未来参数。任何需要估计预期收益或协方差的
# "最优权重"，在没有样本外验证体系时就是样本内拟合——契约直接不接受。

RULE_KINDS = ("equal_weight", "conviction_weighted", "target_upside_weighted")

#: 入选的 side。hold/sell 不配仓，review 表示无法形成判定，二者都不进权重。
SELECTED_SIDES = ("strong_buy", "buy")

#: 排除原因：固定枚举，因为它要能被跨期聚合（"哪个原因在吃掉候选池"）。
EXCLUSION_REASONS = (
    "side_not_selected",
    "below_conviction_floor",
    "negative_upside",
    "outside_universe",
    "data_gap",
    "user_excluded",
)

_PI_REQUIRED = ("as_of", "universe", "rules", "weights", "cash_floor", "derived_from", "producer")
_PI_FIELDS = frozenset(
    _PI_REQUIRED + ("schema_version", "rebalance", "excluded", "rationale", "evidence")
)
_RULE_KEYS = frozenset(
    {"kind", "conviction_floor", "cash_floor", "price_reference", "price_reference_as_of"}
)


def _check_rules(rules: Any) -> list[dict]:
    errs: list[dict] = []
    if not isinstance(rules, Mapping):
        return [{"path": "rules", "reason": "must be an object"}]

    for k in rules:
        if k not in _RULE_KEYS:
            errs.append({"path": f"rules.{k}", "reason": "unknown field"})

    kind = rules.get("kind")
    if kind not in RULE_KINDS:
        errs.append({
            "path": "rules.kind",
            "reason": f"must be one of {RULE_KINDS} (got {kind!r}); optimising rules "
                      "(mean-variance, risk parity, Black-Litterman, IC-fitted weights) are "
                      "not accepted — without out-of-sample validation an 'optimal' weight "
                      "is in-sample fitting",
        })
        return errs

    if kind in ("conviction_weighted", "target_upside_weighted") and rules.get("conviction_floor") is None:
        errs.append({
            "path": "rules.conviction_floor",
            "reason": f"required for kind={kind!r}",
        })
    if rules.get("cash_floor") is None:
        errs.append({"path": "rules.cash_floor", "reason": f"required for kind={kind!r}"})
    for key in ("cash_floor", "conviction_floor"):
        v = rules.get(key)
        if v is not None and (
            not isinstance(v, (int, float)) or isinstance(v, bool) or not (0 <= float(v) <= 1)
        ):
            errs.append({"path": f"rules.{key}", "reason": "must be a number in [0, 1]"})

    if kind == "target_upside_weighted":
        pr = rules.get("price_reference")
        if not isinstance(pr, Mapping) or not pr:
            errs.append({
                "path": "rules.price_reference",
                "reason": f"required for kind={kind!r} and must be a non-empty object of "
                          "symbol → reference price (the denominator of upside)",
            })
        else:
            for sym, v in pr.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool) or float(v) <= 0:
                    errs.append({
                        "path": f"rules.price_reference.{sym}",
                        "reason": "must be a positive number",
                    })
        asof = rules.get("price_reference_as_of")
        if asof is not None and not _is_iso_date(asof):
            errs.append({
                "path": "rules.price_reference_as_of",
                "reason": "must be an ISO date (YYYY-MM-DD)",
            })

    return errs


def _check_excluded(excluded: Any, derived_from: list[str]) -> list[dict]:
    errs: list[dict] = []
    if not isinstance(excluded, list):
        return [{"path": "excluded", "reason": "must be an array"}]

    seen: set[str] = set()
    for i, item in enumerate(excluded):
        path = f"excluded[{i}]"
        if not isinstance(item, Mapping):
            errs.append({"path": path, "reason": "must be an object with id + reason"})
            continue
        rid = item.get("id")
        if not isinstance(rid, str) or not rid:
            errs.append({"path": f"{path}.id", "reason": "must be a non-empty string"})
        elif rid in seen:
            errs.append({"path": f"{path}.id", "reason": f"duplicate excluded id: {rid}"})
        else:
            seen.add(rid)
            if derived_from and rid not in derived_from:
                errs.append({
                    "path": f"{path}.id",
                    "reason": f"{rid!r} is not in derived_from — an exclusion can only "
                              "refer to an input",
                })
        reason = item.get("reason")
        if reason not in EXCLUSION_REASONS:
            errs.append({
                "path": f"{path}.reason",
                "reason": f"must be one of {EXCLUSION_REASONS} (got {reason!r}); free text "
                          "belongs in rationale, not here — this field is aggregated across periods",
            })
    return errs


def check_portfolio_intent(d: Any) -> list[dict]:
    """Validate PortfolioIntent (dict or JSON string). Returns [] when valid."""
    errs: list[dict] = []
    if isinstance(d, (str, bytes)):
        try:
            d = json.loads(d)
        except (TypeError, ValueError) as e:
            return [{"path": "$", "reason": f"invalid JSON: {e}"}]
    if not isinstance(d, Mapping):
        return [{"path": "$", "reason": "must be an object"}]

    for k in d:
        if k not in _PI_FIELDS:
            errs.append({"path": k, "reason": "unknown field"})
    if d.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errs.append({
            "path": "schema_version",
            "reason": f'must be "{SUPPORTED_SCHEMA_VERSION}" '
                      f"(got {d.get('schema_version')!r}); unsupported versions are rejected",
        })
    for k in _PI_REQUIRED:
        if k not in d:
            errs.append({"path": k, "reason": "required field missing"})
    if "as_of" in d and not _is_iso_date(d.get("as_of")):
        errs.append({"path": "as_of", "reason": "must be an ISO date (YYYY-MM-DD)"})

    errs.extend(_check_rules(d.get("rules")))

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

    derived_from = d.get("derived_from")
    if not isinstance(derived_from, list) or not derived_from:
        errs.append({
            "path": "derived_from",
            "reason": "must be a non-empty array of source ids: weights without a "
                      "traceable origin cannot be reviewed",
        })
        derived_from = []
    else:
        for i, rid in enumerate(derived_from):
            if not (isinstance(rid, str) and rid):
                errs.append({"path": f"derived_from[{i}]", "reason": "must be a non-empty string"})
        if len(set(derived_from)) != len(derived_from):
            errs.append({"path": "derived_from", "reason": "must not contain duplicates"})

    # 权重：与 TargetWeights 同一套不变式（拒绝而不归一化）。
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
        if total > 1 + WEIGHT_SUM_TOLERANCE:
            errs.append({
                "path": "weights",
                "reason": f"Σweights must be ≤ 1 (got {total}); rejected, not normalised",
            })
        elif total + float(cash_floor) > 1 + WEIGHT_SUM_TOLERANCE:
            errs.append({
                "path": "cash_floor",
                "reason": f"cash_floor + Σweights must be ≤ 1 "
                          f"(got {total + float(cash_floor)}); rejected, not normalised",
            })

    errs.extend(_check_excluded(d.get("excluded", []), list(derived_from)))

    # 计数守恒的另一半：权重非空但每个来源都被排除，说明权重的出处断了。
    excluded_ids = {
        i["id"] for i in (d.get("excluded") or [])
        if isinstance(i, Mapping) and isinstance(i.get("id"), str)
    }
    selected = [rid for rid in derived_from if rid not in excluded_ids]
    if derived_from and not errs and not selected and weights:
        errs.append({
            "path": "weights",
            "reason": "weights are non-empty but every derived_from id is excluded — "
                      "the weights have no selected input",
        })

    reb = d.get("rebalance")
    if reb is not None:
        if not isinstance(reb, Mapping):
            errs.append({"path": "rebalance", "reason": "must be an object or null"})
        elif "every_n_bars" in reb:
            v = reb["every_n_bars"]
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                errs.append({
                    "path": "rebalance.every_n_bars",
                    "reason": "must be a positive integer",
                })

    r = d.get("rationale")
    if r is not None and not isinstance(r, str):
        errs.append({"path": "rationale", "reason": "must be a string"})

    ev = d.get("evidence", [])
    if not isinstance(ev, list):
        errs.append({"path": "evidence", "reason": "must be an array"})
    else:
        ids: list[str] = []
        for i, item in enumerate(ev):
            errs.extend(_check_evidence(item, f"evidence[{i}]"))
            if isinstance(item, Mapping) and isinstance(item.get("evidence_id"), str):
                ids.append(item["evidence_id"])
        for dup in sorted({i for i in ids if ids.count(i) > 1}):
            errs.append({"path": "evidence", "reason": f"duplicate evidence_id: {dup}"})

    if "producer" not in d:
        errs.append({"path": "producer", "reason": "required field missing"})
    else:
        errs.extend(_check_producer(d.get("producer"), "producer"))

    return errs


@dataclass
class PortfolioIntent:
    """How a set of judgements became a set of holdings. See schemas/portfolio-intent.schema.json."""

    as_of: str
    universe: list[str]
    rules: dict[str, Any]
    weights: dict[str, float]
    producer: Producer
    cash_floor: float = 0.0
    derived_from: list[str] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    rebalance: Optional[dict] = None
    rationale: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    schema_version: str = SUPPORTED_SCHEMA_VERSION

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "as_of": self.as_of,
            "universe": list(self.universe),
            "rules": dict(self.rules),
            "weights": dict(self.weights),
            "cash_floor": self.cash_floor,
            "derived_from": list(self.derived_from),
            "excluded": [dict(e) for e in self.excluded],
            "rationale": self.rationale,
            "evidence": [
                e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence
            ],
            "producer": (
                self.producer.to_dict() if isinstance(self.producer, Producer) else self.producer
            ),
        }
        if self.rebalance is not None:
            d["rebalance"] = dict(self.rebalance)
        return d

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, **kw)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PortfolioIntent":
        errs = check_portfolio_intent(d)
        if errs:
            raise ContractError(errs[0]["path"], errs[0]["reason"])
        return cls(
            as_of=d["as_of"],
            universe=list(d["universe"]),
            rules=dict(d["rules"]),
            weights={k: float(v) for k, v in d["weights"].items()},
            producer=Producer.from_dict(d["producer"]),
            cash_floor=float(d.get("cash_floor", 0.0)),
            derived_from=list(d["derived_from"]),
            excluded=[dict(e) for e in d.get("excluded", [])],
            rebalance=dict(d["rebalance"]) if d.get("rebalance") is not None else None,
            rationale=d.get("rationale", "") or "",
            evidence=[Evidence.from_dict(e) for e in d.get("evidence", [])],
        )

    @classmethod
    def from_json(cls, s: str) -> "PortfolioIntent":
        return cls.from_dict(json.loads(s))

    def to_target_weights(self) -> "TargetWeights":
        """无损导出为可执行权重。执行侧零改动——它继续只认 TargetWeights。"""
        return TargetWeights(
            as_of=self.as_of,
            universe=list(self.universe),
            weights=dict(self.weights),
            producer=self.producer,
            cash_floor=self.cash_floor,
            constraints={},
            evidence=list(self.evidence),
        )


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
    if isinstance(d, (Rating, TargetWeights, PriceSeriesRef, PortfolioIntent)):
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
    elif kind in ("decision", "decision_v2", "decision-v2"):
        from fd_quant.contracts.decision import check_decision
        errs = check_decision(d)
    elif kind in ("target_weights", "target-weights", "weights"):
        errs = check_target_weights(d)
    elif kind in ("price_series_ref", "price-series-ref", "price_ref"):
        errs = check_price_series_ref(d)
    elif kind in ("portfolio_intent", "portfolio-intent", "intent"):
        errs = check_portfolio_intent(d)
    else:
        return {"valid": False, "errors": [{"path": "$", "reason": f"unknown kind: {kind}"}]}
    return {"valid": not errs, "errors": errs}


def _sniff_kind(d: Any) -> Optional[str]:
    if not isinstance(d, Mapping):
        return None
    # 先认 PriceSeriesRef：它有 source_id + fingerprint 这一对不同名组合，不会与另两者撞。
    if "fingerprint" in d and "source_id" in d:
        return "price_series_ref"
    # PortfolioIntent 有 rules + derived_from 这一对；TargetWeights 没有 rules。
    if "rules" in d and "derived_from" in d:
        return "portfolio_intent"
    # Decision v2 有 subject_kind；Rating 的 subject 只是个字符串。先认 v2，
    # 否则带 subject 的 v2 实例会被当成 Rating 去校验，报一堆缺 side/price_target。
    if "subject_kind" in d:
        return "decision"
    if "side" in d or "subject" in d:
        return "rating"
    if "universe" in d and "weights" in d:
        return "target_weights"
    return None


def validate_rating(instance: Any) -> dict:
    return validate(instance, kind="rating")


def validate_target_weights(instance: Any) -> dict:
    return validate(instance, kind="target_weights")


def validate_price_series_ref(instance: Any) -> dict:
    return validate(instance, kind="price_series_ref")


def validate_portfolio_intent(instance: Any) -> dict:
    return validate(instance, kind="portfolio_intent")


def load_schema(kind: str) -> dict:
    """Load the JSON Schema for a contract kind (for parity tests / consumers)."""
    name = {
        "rating": "rating.schema.json",
        "target_weights": "target-weights.schema.json",
        "price_series_ref": "price-series-ref.schema.json",
        "portfolio_intent": "portfolio-intent.schema.json",
    }.get(kind)
    if name is None:
        raise ContractError("kind", f"unknown contract kind: {kind}")
    return json.loads((_SCHEMA_DIR / name).read_text(encoding="utf-8"))
