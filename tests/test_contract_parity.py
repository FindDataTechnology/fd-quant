"""Parity test: JSON Schema 与 Python 模型必须同判。

The two expressions of the same shape drift unless something pins them: this
file runs one fixture set through both sides and asserts they agree (valid
fixtures pass both; each invalid fixture is refused by both, with the same
field path). That is the whole point of the contract layer — a pack author
validating with either tool must get the same answer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from fd_quant.contracts import models as M  # noqa: E402

jsonschema = pytest.importorskip("jsonschema")
from jsonschema import Draft202012Validator  # noqa: E402

FIXTURES = _ROOT / "tests" / "fixtures"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixtures(kind: str, sub: str) -> list[Path]:
    d = FIXTURES / kind / sub
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*.json"))


def _schema_validator(kind: str) -> Draft202012Validator:
    return Draft202012Validator(M.load_schema(kind))


def _schema_errors(kind: str, data: dict) -> list[str]:
    v = _schema_validator(kind)
    out = []
    for e in v.iter_errors(data):
        path = "/".join(str(p) for p in e.absolute_path)
        if not path and e.validator == "additionalProperties" and isinstance(e.instance, dict):
            # jsonschema reports "additionalProperties" at the *parent* (empty
            # path here) and names the offending key inside the message; surface
            # the key so paths are comparable with the model layer's notation.
            for k in e.instance:
                if k in (e.message or ""):
                    path = k
                    break
        out.append(path or "$")
    return out


def _model_errors(kind: str, data: dict) -> list[str]:
    errs = M.check_rating(data) if kind == "rating" else M.check_target_weights(data)
    return [e["path"] for e in errs]


# ── valid fixtures: both sides accept ───────────────────────────


@pytest.mark.parametrize("kind", ["rating", "target-weights"])
def test_valid_fixtures_pass_both_sides(kind):
    key = "rating" if kind == "rating" else "target_weights"
    files = _fixtures(kind, "valid")
    assert files, f"no valid fixtures for {kind}"
    for f in files:
        data = _load(f)
        s_err = _schema_errors(key, data)
        m_err = _model_errors(key, data)
        assert not s_err, f"{f.name}: schema rejected a valid fixture: {s_err}"
        assert not m_err, f"{f.name}: model rejected a valid fixture: {m_err}"


# ── invalid fixtures: shape must be refused by BOTH sides;
#    relational (cross-field arithmetic / referential) is model-layer duty ──
#
# JSON Schema cannot express "Σweights ≤ 1", "cash_floor + Σweights ≤ 1", or
# "every target_evidence_ids entry resolves inside evidence" — those are
# cross-field relations, not shapes. So each invalid fixture declares which
# layer it exercises:
#   shape      → both the schema and the model must refuse it
#   relational → the unified validation entry point (model layer) must refuse it;
#                the schema may accept it (documented limitation, see design.md D1)


@pytest.mark.parametrize("kind", ["rating", "target-weights"])
def test_invalid_fixtures_refused_by_correct_layers(kind):
    key = "rating" if kind == "rating" else "target_weights"
    files = _fixtures(kind, "invalid")
    assert files, f"no invalid fixtures for {kind}"
    for f in files:
        data = _load(f)
        expect = data.pop("__expect_path__", None)
        layer = data.pop("__parity__", "shape")
        s_err = _schema_errors(key, data)
        m_err = _model_errors(key, data)
        # The model layer is the contract's统一校验入口: it must refuse both kinds.
        assert m_err, f"{f.name}: model accepted an invalid fixture"
        if layer == "shape":
            assert s_err, f"{f.name}: schema accepted a shape-invalid fixture"
        else:
            assert layer == "relational", f"{f.name}: unknown __parity__ layer {layer}"
        if expect:
            head = expect.split(".")[0].split("[")[0]
            assert any(head in p for p in m_err), \
                f"{f.name}: model path {m_err} does not mention {expect}"
            if layer == "shape":
                assert any(head in p for p in s_err), \
                    f"{f.name}: schema path {s_err} does not mention {expect}"


def test_relational_fixtures_are_the_schemas_known_gap():
    """Keep the schema/model split honest: each relational fixture is exactly a
    case the schema cannot express. If a future schema *does* refuse one, this
    test fails and the fixture should be re-classified as `shape`."""
    for kind in ("rating", "target-weights"):
        key = "rating" if kind == "rating" else "target_weights"
        relational = [
            f for f in _fixtures(kind, "invalid")
            if _load(f).get("__parity__") == "relational"
        ]
        assert relational, f"{kind}: expected at least one relational fixture"
        for f in relational:
            data = _load(f)
            data.pop("__expect_path__", None)
            data.pop("__parity__", None)
            assert not _schema_errors(key, data), (
                f"{f.name}: schema now refuses a fixture classified as relational — "
                f"re-classify it as 'shape'"
            )


def test_every_invalid_fixture_declares_expected_path_and_layer():
    for kind in ("rating", "target-weights"):
        for f in _fixtures(kind, "invalid"):
            data = _load(f)
            assert "__expect_path__" in data, f"{f.name}: missing __expect_path__ annotation"
            assert data.get("__parity__") in ("shape", "relational"), \
                f"{f.name}: missing or unknown __parity__ annotation"


# ── determinism ─────────────────────────────────────────────────


def test_validation_is_deterministic():
    for kind in ("rating", "target-weights"):
        for f in _fixtures(kind, "valid") + _fixtures(kind, "invalid"):
            data = _load(f)
            data.pop("__expect_path__", None)
            key = "rating" if kind == "rating" else "target_weights"
            a = M.validate(data, kind=key)
            b = M.validate(data, kind=key)
            assert a == b, f"{f.name}: validation is not deterministic"


def test_validation_is_local_only():
    """No network/DB imports in the contract layer (pure shapes and rules)."""
    src = (M.__file__,)
    text = Path(M.__file__).read_text(encoding="utf-8")
    for banned in ("requests", "httpx", "sqlalchemy", "urllib.request", "socket"):
        assert banned not in text, f"contract layer must not reference {banned}"
