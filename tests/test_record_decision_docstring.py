"""`record_decision` 的工具描述必须**自足**——形状写在描述里，不指向读不到的文件。

为什么有这条测试（2026-11-15 实测）：行业岗在 7 天里调用 `record_decision` **17 次，
16 次被拒、1 次成功**，拒因逐个不同且最后才凑对：

    schema_version: must be "1" (got '1.0')      ×4
    conviction: must be a number in [0, 1] or null ×4
    evidence[0].quote: unknown field              ×4
    action: unknown field                         ×1
    schema_version: must be "1" (got None)        ×1
    side: required field missing                  ×1
    side: must be one of [...];                       ×1
    ⇒ created=true                                ×1

根因不在模型：`payload: Mapping[str, Any]` 在 MCP 的 inputSchema 里只是「对象」，
而 docstring 当时只写「字段同 `schemas/rating.schema.json`」——**调用方读不到那个文件**。
于是唯一的发现途径就是拒收，一次拒收一个字段。

这条测试把「描述里的范例」变成可执行的断言：范例必须真能落账；反过来，
描述里点名的「常见拒收」也必须真的被拒（不然描述会漂成一份过期的说明书）。
"""
from __future__ import annotations

import json
import re

import pytest

from fd_quant.contracts.models import _RATING_FIELDS, RATING_SIDES
from fd_quant.ledger.tools import record_decision

_EXAMPLE_RE = re.compile(r"\{\s*\"schema_version\".*?\n\s*\}", re.S)


def _example() -> dict:
    doc = record_decision.__doc__ or ""
    m = _EXAMPLE_RE.search(doc)
    assert m, "record_decision 的 docstring 里必须有一段可复制的 payload 范例"
    return json.loads(m.group(0))


def test_docstring_states_the_whole_whitelist() -> None:
    doc = record_decision.__doc__ or ""
    for field in _RATING_FIELDS:
        assert field in doc, f"白名单字段 {field} 没写进描述——调用方只能靠拒收发现它"


def test_docstring_states_the_side_enum() -> None:
    doc = record_decision.__doc__ or ""
    for side in RATING_SIDES:
        assert side in doc, f"side 取值 {side} 没写进描述"


def test_docstring_does_not_point_at_an_unreadable_file() -> None:
    doc = record_decision.__doc__ or ""
    assert "schemas/rating.schema.json" not in doc, (
        "描述不能再让调用方去看仓内文件——它读不到；形状要写在描述里"
    )


def test_example_records_successfully(tmp_path) -> None:
    """范例必须真能落账（与 W1 提示词包同一条纪律：范例要被真校验器跑过）。"""
    payload = _example()
    db = str(tmp_path / "ledger.db")
    # 契约要求 payload 与入参逐字一致，所以按入参回填
    payload["as_of"] = "2026-11-13"
    payload["subject"] = "600519.SS"
    payload["producer"] = {"pack": "投研晨会", "skill": "invest-decision-log"}
    out = record_decision(
        as_of="2026-11-13",
        pack="投研晨会",
        skill="invest-decision-log",
        payload=payload,
        symbol="600519.SS",
        db_path=db,
    )
    assert out["result"]["created"] is True, out
    assert out["result"]["status"] == "pending"


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda p: p.update(schema_version="1.0"), "schema_version"),
        (lambda p: p.update(schema_version=1), "schema_version"),
        (lambda p: p.pop("schema_version"), "schema_version"),
        (lambda p: p.update(conviction=60), "conviction"),
        (lambda p: p.update(evidence=[{"quote": "x"}]), "evidence[0].quote"),
        (lambda p: p.update(action="buy"), "action"),
        (lambda p: p.pop("side"), "side"),
        (lambda p: p.update(side="buy_strong"), "side"),
        (lambda p: p.pop("basis"), "basis"),
    ],
)
def test_documented_rejections_really_reject(tmp_path, mutate, field) -> None:
    """描述里点名的每个「常见拒收」都必须真的以那个 field 被拒。"""
    payload = _example()
    payload["as_of"] = "2026-11-13"
    payload["subject"] = "600519.SS"
    payload["producer"] = {"pack": "投研晨会", "skill": "invest-decision-log"}
    mutate(payload)
    with pytest.raises(Exception) as ei:  # ContractError 或映射后的异常
        record_decision(
            as_of="2026-11-13",
            pack="投研晨会",
            skill="invest-decision-log",
            payload=payload,
            symbol="600519.SS",
            db_path=str(tmp_path / "ledger.db"),
        )
    text = str(ei.value)
    assert field in text, f"期望拒收点名 {field}，实际：{text[:200]}"
