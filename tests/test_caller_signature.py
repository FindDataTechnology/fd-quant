"""客户身份头验签（seal-ledger-path-and-caller-isolation 3.1 / 3.2）。

判据逐条对应任务原文，按「能红」写：

- 3.1 好签名通过 / 坏签名拒 / 超窗拒 / 密钥未配时忽略该头 / customer_id 非法拒；
  **原因码稳定且互不相同**（四码各一条用例，且断言字符串常量本身）。
- 3.2 解析链：签名头 → Bearer 指纹（未升级兜底）→ env → anonymous；
  **验签失败抛出、不返回 anonymous**；无头时照旧回落（不被本项误伤）。

为什么把指纹留在链上：`tests/test_ledger_tenancy.py` 断言过「Bearer 落指纹不落原文」，
且 5.5 要求未配密钥的部署与回滚片行为同形。删掉指纹会把「全客户同名」直接变成
「全客户 anonymous」，那是把两种现状混成一个，不是修隔离。
"""

from __future__ import annotations

import sys
import types

import pytest

from fd_quant import caller as caller_mod
from fd_quant.caller import (
    ANONYMOUS,
    CALLER_CLOCK_SKEW_S,
    CALLER_HMAC_ENV,
    REASON_BAD_ID,
    REASON_EXPIRED,
    REASON_MALFORMED,
    REASON_MISMATCH,
    CallerSignatureError,
    sign_caller,
    verify_caller_header,
)

KEY = "unit-test-hmac-key"
ENV = {CALLER_HMAC_ENV: KEY}
CID = "zhang@example.com"
NOW = 1_700_000_000


def _header(cid: str = CID, ts: int = NOW, key: str = KEY) -> str:
    return f"{cid}.{ts}.{sign_caller(cid, ts, key)}"


def _reason(value: str | None, *, now: float = NOW, env=None):
    with pytest.raises(CallerSignatureError) as exc:
        verify_caller_header(value, now=now, env=ENV if env is None else env)
    return exc.value.reason


# ── 3.1 规则层 ──────────────────────────────────────────────────────────────


def test_good_signature_returns_customer_id():
    assert verify_caller_header(_header(), now=NOW, env=ENV) == CID


def test_bad_signature_is_rejected():
    forged = f"{CID}.{NOW}." + "0" * 64
    assert _reason(forged) == REASON_MISMATCH


def test_signature_for_another_timestamp_is_rejected():
    """签名与时戳必须成对：拿 A 时刻的签名配 B 时刻的时戳（窗口内）也要拒。"""
    sig = sign_caller(CID, NOW, KEY)
    assert _reason(f"{CID}.{NOW + 1}.{sig}") == REASON_MISMATCH


def test_window_boundary_is_inclusive_and_one_second_out_is_not():
    """±300 秒是**含边界**的：正好 300 通过，301 拒。

    只测「超窗拒」而不测边界，等于没钉住阈值——把窗口改成 ±1 秒也会全绿。
    """
    for offset in (-CALLER_CLOCK_SKEW_S, CALLER_CLOCK_SKEW_S):
        assert verify_caller_header(_header(ts=NOW + offset), now=NOW, env=ENV) == CID
    for offset in (-CALLER_CLOCK_SKEW_S - 1, CALLER_CLOCK_SKEW_S + 1):
        assert _reason(_header(ts=NOW + offset)) == REASON_EXPIRED


def test_header_is_ignored_when_key_is_not_configured():
    """未配密钥 ⇒ 整条轨不生效（**不抛**）。滚动升级期间旧 pod 不该因此 401。"""
    assert verify_caller_header(_header(), now=NOW, env={}) is None
    assert verify_caller_header("total-garbage", now=NOW, env={}) is None


def test_missing_or_empty_header_is_not_an_error():
    assert verify_caller_header(None, now=NOW, env=ENV) is None
    assert verify_caller_header("   ", now=NOW, env=ENV) is None


@pytest.mark.parametrize(
    "value",
    ["only-two.parts", "a.b.c.d.e", f"{CID}.not-a-number.deadbeef"],
)
def test_malformed_header_reason(value):
    assert _reason(value) == REASON_MALFORMED


@pytest.mark.parametrize(
    "cid",
    ["../x", "a b", "", ".", "..", "a" * 129, "user/../etc", "名字"],
)
def test_illegal_customer_id_reason(cid):
    """非法 id 的判断必须先于验签：否则攻击者靠"签名对不上"永远看不到 id 的问题。"""
    assert _reason(f"{cid}.{NOW}.deadbeef") == REASON_BAD_ID


def test_reason_codes_are_stable_and_distinct():
    """四个原因码必须是**不同的固定字符串**——错误映射与告警按它们分流，
    重命名异常类不该改变对外行为。"""
    codes = [REASON_MALFORMED, REASON_BAD_ID, REASON_EXPIRED, REASON_MISMATCH]
    assert len(set(codes)) == 4
    assert codes == [
        "caller_signature_malformed",
        "caller_id_invalid",
        "caller_signature_expired",
        "caller_signature_mismatch",
    ]


def test_signature_is_lowercase_hex_of_hmac_sha256():
    """签法是对外契约：`hmac_sha256(key, "<id>.<ts>")` 的十六进制。

    网关侧照同一段字符串签，所以这里钉住的是**跨仓契约**，不是内部实现。
    """
    import hashlib
    import hmac

    expected = hmac.new(KEY.encode(), f"{CID}.{NOW}".encode(), hashlib.sha256).hexdigest()
    assert sign_caller(CID, NOW, KEY) == expected
    assert expected == expected.lower() and len(expected) == 64


# ── 3.2 解析链 ──────────────────────────────────────────────────────────────


def _install_request(monkeypatch, headers: dict):
    class _Request:
        pass

    req = _Request()
    req.headers = headers
    fake = types.ModuleType("fastmcp.server.dependencies")
    fake.get_http_request = lambda: req
    monkeypatch.setitem(sys.modules, "fastmcp.server.dependencies", fake)
    return req


def test_resolve_caller_uses_verified_header(monkeypatch):
    """验过的头优先于 env 与指纹——这是客户级隔离真正生效的那一跳。

    时戳用**真实时钟**签（`resolve_caller()` 内部读 `time.time()`，夹具另造一个
    `now` 会让"窗口"变成两个来源，那正是这条判据要防的东西）。
    """
    import time

    monkeypatch.setenv(CALLER_HMAC_ENV, KEY)
    monkeypatch.setenv("FD_QUANT_CALLER", "env-identity")
    _install_request(
        monkeypatch,
        {
            caller_mod.CALLER_HEADER: _header(ts=int(time.time())),
            "Authorization": "Bearer sk-abc",
        },
    )
    assert caller_mod.resolve_caller() == CID


def test_resolve_caller_raises_on_forged_header_and_never_returns_anonymous(monkeypatch):
    """任务的硬判据：伪造头**报错**，且**不**返回 `anonymous`。

    回落成 anonymous 会让攻击者拿到一次合法的匿名写入——200 就是"成功"的错觉。
    """
    import time

    monkeypatch.setenv(CALLER_HMAC_ENV, KEY)
    # 时戳必须**在窗内**、签名必须**错**——否则撞上的是「超窗」那条原因码，
    # 断言就变成了在测另一条判据（第一次写这条用例时正是这么错的）。
    forged = f"{CID}.{int(time.time())}." + "0" * 64
    _install_request(monkeypatch, {caller_mod.CALLER_HEADER: forged})
    with pytest.raises(CallerSignatureError) as exc:
        caller_mod.resolve_caller()
    assert exc.value.reason == REASON_MISMATCH
    assert exc.value.reason != ANONYMOUS


def test_resolve_caller_without_header_still_falls_back(monkeypatch):
    """无头时不被本项误伤：仍按 指纹 → env → anonymous 回落。"""
    monkeypatch.setenv(CALLER_HMAC_ENV, KEY)
    monkeypatch.setenv("FD_QUANT_CALLER", "deployed-identity")
    _install_request(monkeypatch, {})
    assert caller_mod.resolve_caller() == "deployed-identity"

    monkeypatch.delenv("FD_QUANT_CALLER", raising=False)
    assert caller_mod.resolve_caller() == ANONYMOUS


def test_bearer_fingerprint_still_precedes_env(monkeypatch):
    """未升级部署（无签名头、无密钥）行为必须与今天逐字相同：指纹优先于 env。"""
    monkeypatch.delenv(CALLER_HMAC_ENV, raising=False)
    monkeypatch.setenv("FD_QUANT_CALLER", "deployed-identity")
    _install_request(monkeypatch, {"Authorization": "Bearer sk-abc"})
    got = caller_mod.resolve_caller()
    assert got == caller_mod._fingerprint("sk-abc"), "指纹档位被静默跳过了"
    assert got != "deployed-identity"
