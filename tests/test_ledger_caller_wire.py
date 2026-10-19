"""调用者隔离的**线上形态**验证（组 2 补强）：真起 HTTP server，用两个 Bearer 走 MCP。

`tests/test_ledger_tenancy.py` 验的是 store 层（直接调 `LedgerStore`），隔离逻辑正确
不等于**身份真的传到了那一层**——HTTP 传输下 `caller` 要靠 `Authorization` 头解析，
而这条链路上有已知的坑（fastmcp 的 `get_http_headers()` 会剥掉 Authorization，必须
读原始 request）。本文件把那段链路整条跑一遍。

被 `@pytest.mark.external` 标记：它起本地端口、起子进程，但**不联网**；默认用
`-m "not external"` 排除，需要时显式跑（与 repo 内既有约定一致）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _rpc(port: int, session: str | None, token: str, method: str, params: dict, _id: int = 1):
    """发一条 MCP JSON-RPC（streamable-http），返回 (session_id, response)。

    刻意**用标准库 urllib 而不是 mcp SDK**：SDK 会替我们做对很多事，而这里要验的
    正是"不靠 SDK 的客户端也能被正确识别身份"——真实调用方不一定是 SDK。
    """
    body = json.dumps({"jsonrpc": "2.0", "id": _id, "method": method, "params": params}).encode()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {token}",
    }
    if session:
        headers["mcp-session-id"] = session
    req = urllib.request.Request(f"http://127.0.0.1:{port}/mcp", data=body, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        raw, new = resp.read().decode(), resp.headers.get("mcp-session-id")
    except urllib.error.HTTPError as exc:
        raw, new = exc.read().decode(), exc.headers.get("mcp-session-id")
    out = None
    for line in raw.splitlines():
        if line.startswith("data:"):
            try:
                data = json.loads(line[5:].strip())
                if "result" in data or "error" in data:
                    out = data
            except ValueError:
                pass
    return (new or session), (out or {"raw": raw[:400]})


def _result(resp: dict) -> dict:
    return ((resp.get("result") or {}).get("structuredContent") or {}).get("result", {})


def _rating(as_of: str, subject: str) -> dict:
    return {
        "schema_version": "1", "as_of": as_of, "subject": subject, "side": "buy",
        "conviction": 0.6, "price_target": None, "target_evidence_ids": [],
        "basis": "线上形态隔离实测。", "evidence": [], "time_horizon": "6-12 个月",
        "producer": {"pack": "pack-x", "skill": "skill-y", "run_id": None},
    }


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    port = 8231
    db = tmp_path_factory.mktemp("wire") / "ledger.db"
    env = dict(os.environ, FD_QUANT_LEDGER_DB=str(db))
    proc = subprocess.Popen(
        [sys.executable, "-m", "fd_quant.server", "--http", str(port)],
        cwd=str(REPO), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except Exception:  # noqa: BLE001 — 启动竞态，轮询到就绪为止
                time.sleep(0.5)
        else:
            pytest.fail("fd-quant HTTP server did not become ready")
        yield port, db
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _session(port: int, token: str) -> str:
    sid, _ = _rpc(port, None, token, "initialize",
                  {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "wire-test", "version": "1"}})
    assert sid, "streamable-http 必须回 mcp-session-id"
    return sid


@pytest.mark.external
def test_two_bearers_get_separate_ledgers_over_http(server):
    """两个 Bearer 走同一台 HTTP server：同键各成功、互不顶掉、互不可见。

    这是组 2 的**线上形态**验收：store 层隔离了，但身份若没从 Authorization 头传到
    store 层，两个调用者仍会互撞（都解析成 anonymous）。
    """
    port, db = server
    sid_a = _session(port, "token-AAA")
    sid_b = _session(port, "token-BBB")
    args = {"as_of": "2025-01-06", "symbol": "600519.SS", "pack": "pack-x",
            "skill": "skill-y", "payload": _rating("2025-01-06", "600519.SS")}

    _, ra = _rpc(port, sid_a, "token-AAA", "tools/call",
                 {"name": "record_decision", "arguments": args})
    _, rb = _rpc(port, sid_b, "token-BBB", "tools/call",
                 {"name": "record_decision", "arguments": args})
    a, b = _result(ra), _result(rb)

    assert a.get("created") is True and b.get("created") is True, (a, b)
    assert a["decision_id"] != b["decision_id"]
    # `counts` 是**全库**行数（跨调用者），不是某个调用者的行数——账本刻意如此：
    # 行数快照要能回答"这次操作有没有让库里多一行"，若按调用者裁剪就失去了对照意义。
    assert b["counts"]["decisions"] == 2, b["counts"]

    # 同一 Bearer 重复提交仍被拒——隔离不能把幂等键放掉
    _, ra2 = _rpc(port, sid_a, "token-AAA", "tools/call",
                  {"name": "record_decision", "arguments": args})
    a2 = _result(ra2)
    assert a2.get("created") is False and a2.get("reason") == "duplicate_key", a2
    assert a2["decision_id"] == a["decision_id"]


@pytest.mark.external
def test_bearer_plaintext_never_reaches_the_database(server):
    """凭据不落库：库里只有 12 位指纹，token 原文不在任何字节里。

    这是"取身份而非取凭据"的可证伪形式——直接扫库文件的字节。
    """
    port, db = server
    sid = _session(port, "secret-token-ZZZ")
    args = {"as_of": "2025-02-03", "symbol": "000333.SZ", "pack": "pack-x",
            "skill": "skill-y", "payload": _rating("2025-02-03", "000333.SZ")}
    _, r = _rpc(port, sid, "secret-token-ZZZ", "tools/call",
                {"name": "record_decision", "arguments": args})
    assert _result(r).get("created") is True, r

    raw = db.read_bytes()
    assert b"secret-token-ZZZ" not in raw, "Bearer 原文落进了账本文件"

    import sqlite3

    con = sqlite3.connect(db)
    callers = {row[0] for row in con.execute("SELECT caller FROM decisions")}
    con.close()
    assert "secret-token-ZZZ" not in callers
    # 指纹：SHA-256 前 12 位十六进制，长度固定、可复核
    import hashlib

    expected = hashlib.sha256(b"secret-token-ZZZ").hexdigest()[:12]
    assert expected in callers, (expected, callers)
