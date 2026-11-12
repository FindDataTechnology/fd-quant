"""共享 bearer 门的判据（change converge-line-identity-and-keys 8.6）。

两层：
1. **门本身**用真 ASGI scope 直测——不装 fastmcp 也能跑，故 CI 永远验得到。
2. **接线**见各仓服务入口旁的调用点；本文件只钉门的判据与豁免面。
"""
from __future__ import annotations

import asyncio

import pytest

from fd_quant.auth import ENV_TOKEN, EXEMPT_PATHS, BearerAuthMiddleware, configured_token


def _run(mw, headers, *, path="/mcp", scope_type="http"):
    """喂一个 scope 进去，返回 (status, body, reached_downstream)。"""
    reached = []
    sent = []

    async def downstream(scope, receive, send):
        reached.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": scope_type,
        "path": path,
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
    }
    asyncio.run(mw(downstream)(scope, receive, send))

    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body, bool(reached)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV_TOKEN, raising=False)
    yield


def test_no_token_means_no_gate():
    status, _, reached = _run(BearerAuthMiddleware, {})
    assert (status, reached) == (200, True)
    assert configured_token() == ""


def test_matching_bearer_passes(monkeypatch):
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    status, body, reached = _run(BearerAuthMiddleware, {"authorization": "Bearer s3cret"})
    assert (status, reached, body) == (200, True, b"ok")


def test_missing_header_rejected(monkeypatch):
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    status, body, reached = _run(BearerAuthMiddleware, {})
    assert status == 401
    assert reached is False, "门必须在进 MCP 处理之前拦住"
    assert b"unauthorized" in body


def test_wrong_token_rejected(monkeypatch):
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    status, _, reached = _run(BearerAuthMiddleware, {"authorization": "Bearer wrong"})
    assert (status, reached) == (401, False)


def test_wrong_scheme_rejected(monkeypatch):
    """裸 token（无 Bearer 前缀）不算——判据是整串恒时比较，不做宽容解析。"""
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    for value in ("s3cret", "Basic s3cret", "Bearer  s3cret"):
        status, _, reached = _run(BearerAuthMiddleware, {"authorization": value})
        assert (status, reached) == (401, False), value


def test_health_paths_exempt(monkeypatch):
    """探针永远不带钥匙：豁免路径必须放行，否则部署被自己的门打死。"""
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    for path in EXEMPT_PATHS:
        status, _, reached = _run(BearerAuthMiddleware, {}, path=path)
        assert (status, reached) == (200, True), path


def test_non_health_paths_not_exempt(monkeypatch):
    """豁免只认精确路径：带前缀/后缀的不是健康面。"""
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    for path in ("/healthz/x", "/api/v1/healthz", "/mcp", "/"):
        status, _, reached = _run(BearerAuthMiddleware, {}, path=path)
        assert (status, reached) == (401, False), path


def test_lifespan_passes_through(monkeypatch):
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    status, _, reached = _run(BearerAuthMiddleware, {}, scope_type="lifespan")
    assert (status, reached) == (200, True)


def test_token_reread_per_request(monkeypatch):
    """改旗/轮换不需要重启：同一中间件类，改 env 后立刻生效。"""
    monkeypatch.setenv(ENV_TOKEN, "one")
    assert _run(BearerAuthMiddleware, {"authorization": "Bearer one"})[0] == 200
    monkeypatch.setenv(ENV_TOKEN, "two")
    assert _run(BearerAuthMiddleware, {"authorization": "Bearer one"})[0] == 401
    assert _run(BearerAuthMiddleware, {"authorization": "Bearer two"})[0] == 200
