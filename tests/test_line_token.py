"""门 K 换发令牌的离线校验（fd-quant 落点）

原型：`fd-info/tests/test_line_token.py`（8.7），逐条同形。原（converge-line-identity-and-keys 8.x（本线 info））。

只测 `fd_quant.line_token` 本身：不装数据库、不起 app、不联网。
接线（票进来能不能过到路由）在 `test_line_token_track.py`，两层分开——
规则层绿而接线层断，是这类改动最常见的假绿。

断言一律认**稳定原因码**（`LineTokenRejected.reason`），不认文案：
文案会随人改，原因码是契约。
"""
from __future__ import annotations

import time

import pytest

jwt = pytest.importorskip("jwt")

from fd_quant.line_token import (  # noqa: E402
    ENV_ISSUER,
    ENV_JWKS_URI,
    ENV_LINE,
    ENV_PEM,
    LINE_NAME,
    MAX_TTL_SECONDS,
    LineTokenRejected,
    LineTokenSettings,
    settings_from_env,
    verify_line_token,
)

ISSUER = "https://platform.finddatatech.cloud"


def _keypair(tag: str = "line"):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        .decode()
    )
    return key, pem


_PRIV, _PUB = _keypair()


def _settings(**over) -> LineTokenSettings:
    base = dict(
        issuer=ISSUER, line=LINE_NAME, public_key_pem=_PUB, max_ttl_seconds=MAX_TTL_SECONDS
    )
    base.update(over)
    return LineTokenSettings(**base)


def mint(
    *,
    sub: str = "caller-1",
    aud: str = LINE_NAME,
    source: str = LINE_NAME,
    iss: str = ISSUER,
    scope: str = "line:read line:write",
    ttl: int = 300,
    email: str | None = None,
    drop: tuple[str, ...] = (),
    key=None,
) -> str:
    now = int(time.time())
    payload = {
        "iss": iss,
        "aud": aud,
        "source": source,
        "sub": sub,
        "jti": f"jti-{sub}-{now}",
        "scope": scope,
        "iat": now,
        "exp": now + ttl,
    }
    if email:
        payload["email"] = email
    for claim in drop:
        payload.pop(claim, None)
    return jwt.encode(payload, key or _PRIV, algorithm="RS256")


def test_valid_token_yields_identity():
    ident = verify_line_token(mint(email="a@b.c"), settings=_settings())
    assert ident.subject == "caller-1"
    assert ident.email == "a@b.c"
    assert ident.scope == ("line:read", "line:write")
    assert ident.source == LINE_NAME
    assert ident.jti
    assert ident.can_write is True


def test_read_only_scope_cannot_write():
    """`can_write` 只认 `line:write`：半枚票不该在写面上等于一把客户级写钥匙。"""
    ident = verify_line_token(mint(scope="line:read"), settings=_settings())
    assert ident.can_write is False


def test_not_configured_when_issuer_or_key_missing():
    for settings in (
        _settings(issuer=""),
        _settings(public_key_pem=""),
        _settings(line=""),
    ):
        with pytest.raises(LineTokenRejected) as exc:
            verify_line_token(mint(), settings=settings)
        assert exc.value.reason == LineTokenRejected.NOT_CONFIGURED


def test_empty_token_is_malformed():
    with pytest.raises(LineTokenRejected) as exc:
        verify_line_token("   ", settings=_settings())
    assert exc.value.reason == LineTokenRejected.TOKEN_MALFORMED


def test_garbage_token_is_malformed():
    with pytest.raises(LineTokenRejected) as exc:
        verify_line_token("not-a-jwt", settings=_settings())
    assert exc.value.reason == LineTokenRejected.TOKEN_MALFORMED


def test_signed_by_another_key_is_bad_signature():
    """原因码必须指「签错了钥」，不能报成 token_malformed。

    PyJWT 的 `InvalidSignatureError` 是 `DecodeError` 的子类，except 顺序反了就会把
    它吞成 token_malformed —— 排查会往"票格式"上跑，而真问题是钥匙不对。
    """
    other, _ = _keypair("line-other")
    with pytest.raises(LineTokenRejected) as exc:
        verify_line_token(mint(key=other), settings=_settings())
    assert exc.value.reason == LineTokenRejected.BAD_SIGNATURE


def test_wrong_issuer_rejected():
    with pytest.raises(LineTokenRejected) as exc:
        verify_line_token(mint(iss=ISSUER + "/wrong"), settings=_settings())
    assert exc.value.reason == LineTokenRejected.WRONG_ISSUER


def test_wrong_audience_rejected():
    with pytest.raises(LineTokenRejected) as exc:
        verify_line_token(mint(aud="wire"), settings=_settings())
    assert exc.value.reason == LineTokenRejected.WRONG_AUDIENCE


def test_other_line_source_rejected_even_with_right_audience():
    """跨线复用票的闸：签名与 aud 都合法、但票是给别的线发的，必须拒。

    这条是计量归属的边界——放过去就会把 wire 的用量记到 erp 头上。
    """
    with pytest.raises(LineTokenRejected) as exc:
        verify_line_token(mint(source="wire"), settings=_settings())
    assert exc.value.reason == LineTokenRejected.WRONG_SOURCE


def test_expired_rejected():
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": ISSUER,
            "aud": LINE_NAME,
            "source": LINE_NAME,
            "sub": "caller-1",
            "jti": "j1",
            "scope": "line:read",
            "iat": now - 1200,
            "exp": now - 600,
        },
        _PRIV,
        algorithm="RS256",
    )
    with pytest.raises(LineTokenRejected) as exc:
        verify_line_token(token, settings=_settings())
    assert exc.value.reason == LineTokenRejected.EXPIRED


def test_ttl_over_ceiling_rejected():
    """PyJWT 只管有没有过期，不管给的时间太长——短命约束要线侧自己守。"""
    with pytest.raises(LineTokenRejected) as exc:
        verify_line_token(mint(ttl=MAX_TTL_SECONDS + 1), settings=_settings())
    assert exc.value.reason == LineTokenRejected.TTL_TOO_LONG


@pytest.mark.parametrize("claim", ["sub", "jti", "exp", "source"])
def test_missing_claim_rejected(claim):
    with pytest.raises(LineTokenRejected) as exc:
        verify_line_token(mint(drop=(claim,)), settings=_settings())
    assert exc.value.reason == LineTokenRejected.MISSING_CLAIM


def test_jwks_uri_path_works():
    """PEM 为空时走 JWKS：工厂注入假件（离线、不联网）。"""
    from types import SimpleNamespace

    settings = _settings(public_key_pem="", jwks_uri=ISSUER + "/jwks")
    ident = verify_line_token(
        mint(),
        settings=settings,
        jwks_client_factory=lambda url: SimpleNamespace(
            get_signing_key_from_jwt=lambda token: SimpleNamespace(key=_PUB)
        ),
    )
    assert ident.subject == "caller-1"


def test_settings_from_env_defaults_line_to_line():
    """缺省线名恒为本线：让"这条线是谁"随环境漂移，等于让 aud/source 判定可配。"""
    assert settings_from_env({}).line == LINE_NAME
    assert settings_from_env({}).configured is False
    s = settings_from_env({ENV_ISSUER: ISSUER, ENV_PEM: _PUB, ENV_LINE: "  "})
    assert s.line == LINE_NAME and s.configured is True
    s2 = settings_from_env({ENV_ISSUER: ISSUER, ENV_JWKS_URI: ISSUER + "/jwks"})
    assert s2.configured is True  # JWKS 也能让设置成立
