"""错误码契约（mcp-error-semantics 组 4）。

四件事必须成立，缺一条调用方就得靠解析中文决定怎么办：

1. **失败带码**：每个失败响应的 `structuredContent.error.code` 属于封闭集合。
2. **四类可区分**：`invalid_input` / `not_found` / `not_wired` /
   `upstream_unavailable` 各有实测样例，且走不同的码。
3. **码之外有定位信息**：入参非法给 `field`，上游不可用给 `upstream`。
4. **未知异常不伪装**：真正的 bug 必须带堆栈抛出，而不是变成一条"内部错误"
   让调用方以为可以重试。

另有一条**反向**断言：本服务实际用到的码集合 ⊆ `errors.CODES`。新增一个
没登记的码会让这条红——封闭集合的意义就在这里。
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from fd_quant import errors as E
from fd_quant.error_mapping import map_errors
from fd_quant.errors import CODES, RETRYABLE, error_body, error_result


# ── 码表本身 ───────────────────────────────────────────────────

def test_codes_are_closed_and_cover_the_four_classes():
    for required in (E.INVALID_INPUT, E.NOT_FOUND, E.NOT_WIRED, E.UPSTREAM_UNAVAILABLE):
        assert required in CODES
    assert len(set(CODES)) == len(CODES), "码表有重复"


def test_retryable_matches_semantics():
    """只有上游不可用值得重试：其余三类重试都是浪费。"""
    assert RETRYABLE[E.UPSTREAM_UNAVAILABLE] is True
    for code in (E.INVALID_INPUT, E.NOT_FOUND, E.NOT_WIRED, E.INTERNAL_ERROR):
        assert RETRYABLE[code] is False, f"{code} 不应标为可重试"


def test_unregistered_code_is_rejected():
    """未登记的码构造即报错——封闭集合靠这条真的闭合。"""
    with pytest.raises(ValueError):
        error_body("something_else", "msg")


def test_error_body_carries_locators():
    body = error_body(E.INVALID_INPUT, "as_of 不是 ISO 日期", field="as_of")
    assert body["code"] == E.INVALID_INPUT
    assert body["field"] == "as_of"
    assert body["retryable"] is False

    body2 = error_body(E.UPSTREAM_UNAVAILABLE, "网关不可达", upstream="llm_gateway")
    assert body2["upstream"] == "llm_gateway"
    assert body2["retryable"] is True


# ── 领域异常 → 码（映射表逐类实测）────────────────────────────

def test_compute_error_maps_to_invalid_input():
    from fd_quant.compute.base import ComputeError

    @map_errors
    def tool():
        raise ComputeError("as_of must be an ISO date")

    r = tool()
    assert r.is_error is True
    assert r.structured_content["error"]["code"] == E.INVALID_INPUT
    # 原中文描述必须保留——码是给程序看的，消息是给人看的
    assert "ISO date" in r.structured_content["error"]["message"]


def test_contract_error_maps_to_invalid_input_with_field():
    from fd_quant.contracts.models import ContractError

    @map_errors
    def tool():
        raise ContractError("side", "must be one of strong_buy/...")

    r = tool()
    err = r.structured_content["error"]
    assert err["code"] == E.INVALID_INPUT
    assert err["field"] == "side"


def test_ledger_error_maps_to_invalid_input():
    from fd_quant.ledger.errors import UnknownDecision

    @map_errors
    def tool():
        raise UnknownDecision("no decision 'x' in the ledger")

    r = tool()
    assert r.structured_content["error"]["code"] == E.INVALID_INPUT


def test_settlement_refusal_carries_missing_days():
    """结算窗不足时，缺多少交易日要作为字段给出，而不是只写在文本里。"""
    from fd_quant.ledger.errors import SettlementRefused

    @map_errors
    def tool():
        raise SettlementRefused("missing 3 trading days", missing_trading_days=3)

    r = tool()
    err = r.structured_content["error"]
    assert err["code"] == E.INVALID_INPUT
    assert err["missing_trading_days"] == 3


def test_local_storage_failure_maps_to_upstream_unavailable():
    """账本路径不可用归到 upstream，不归 invalid_input。

    `db_path` 由部署配置给出，调用方改不动它；把它当入参错误会让人去改一个
    自己控制不了的参数。
    """
    @map_errors
    def tool():
        raise OSError(30, "Read-only file system")

    r = tool()
    err = r.structured_content["error"]
    assert err["code"] == E.UPSTREAM_UNAVAILABLE
    assert err["upstream"] == "local_sqlite_ledger"
    assert err["retryable"] is True


def test_unknown_exception_is_re_raised_not_disguised():
    """真正的 bug 必须带堆栈抛出——伪装成结构化错误会让调用方以为可重试。"""
    @map_errors
    def tool():
        raise ZeroDivisionError("real bug")

    with pytest.raises(ZeroDivisionError):
        tool()


# ── 真实 MCP 往返：码与 isError 同时到达 ──────────────────────

def _call(name: str, args: dict):
    from fastmcp import Client

    from fd_quant.server import app

    async def go():
        async with Client(app) as c:
            return await c.call_tool(name, args, raise_on_error=False)

    return asyncio.run(go())


def test_invalid_input_over_the_wire():
    r = _call("compute_rsi", {"dates": ["2025-01-01"], "values": [1.0], "as_of": "bad"})
    assert r.is_error is True
    assert r.structured_content["error"]["code"] == E.INVALID_INPUT


def test_upstream_unavailable_over_the_wire():
    r = _call("retrieve_decisions",
              {"symbol": "X", "as_of": "2025-01-01",
               "db_path": "/proc/no-such-dir/ledger.db"})
    assert r.is_error is True
    err = r.structured_content["error"]
    assert err["code"] == E.UPSTREAM_UNAVAILABLE
    assert err["upstream"] == "local_sqlite_ledger"


def test_contract_violation_over_the_wire_carries_field():
    db = str(Path(tempfile.mkdtemp()) / "l.db")
    r = _call("record_decision", {
        "as_of": "2025-01-02", "symbol": "600519.SS", "pack": "p", "skill": "s",
        "payload": {"schema_version": "1"}, "db_path": db})
    assert r.is_error is True
    err = r.structured_content["error"]
    assert err["code"] == E.INVALID_INPUT
    assert err["field"], "契约违规必须指出是哪个字段"


def test_a_successful_call_is_not_an_error_result():
    """成功路径不能被错误包装误伤（is_error 必须为 False）。"""
    r = _call("summarize_drawdown",
              {"dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
               "values": [100.0, 99.0, 101.0], "as_of": "2025-01-03"})
    assert r.is_error is False
    # 成功路径走 outputSchema，structuredContent 才是载荷；r.data 是 fastmcp 的
    # Root 包装，不是 dict。
    assert "max_drawdown" in (r.structured_content or {})


# ── 封闭性：服务里用到的码都在表内 ───────────────────────────

def test_all_codes_used_in_the_service_are_registered():
    """扫源码里出现的码常量，必须全部登记在 CODES 内。

    这条防的是"某处写了一个新码但忘了登记"——那种码调用方没有分支，
    会落进默认处理，而这正是标准化想消除的东西。
    """
    import re

    src = Path(__file__).resolve().parents[1] / "fd_quant"
    found: set[str] = set()
    pattern = re.compile(r"\b(invalid_input|not_found|not_wired|upstream_unavailable|internal_error)\b")
    for py in src.rglob("*.py"):
        found |= set(pattern.findall(py.read_text(encoding="utf-8")))
    unregistered = found - set(CODES)
    assert unregistered == set(), f"源码里出现了未登记的码：{sorted(unregistered)}"
