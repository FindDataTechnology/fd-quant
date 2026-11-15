"""门 K 换发令牌的线侧离线校验（converge-line-identity-and-keys 8.x，本线 info）。

网关把客户的**词元** `sk-` 换成一枚 `aud`=本线（quant）、带 `source`、`exp` ≤600 秒的
JWT（paas `gateway/line-token-exchange.js`）。本模块只做**离线**校验：公钥来自本地
PEM 或 JWKS，**不回调网关、不逐请求问计费**——网关挂了只影响"换新票"，已经换到手里
的票照样能用。这是「别人也能只使用那条子线、而不和平台绑定」在本线侧的可执行形态。

第五份落点，**逻辑逐条对齐** `fd-lex`（8.2）、`fd-wire`（8.1）、`fd-erp-ledger-mcp`（8.5）、
`fd-info`（8.7）四份，只改线的名字与文档。四份若分叉，以「门 K 票的形状」为准——那是 paas 网关发的，
不是线侧定的。

与门 A 成员级断言（本仓没有那扇门）的三处差别，都不是笔误：

1. **TTL 上限 600 秒而不是 300**：门 K 说的是"这一小段时间内这个客户可以打这条线"，
   门 A 说的是"这一次代做"。容忍窗口本就不同。
2. **不做 `jti` 一次性**。同一枚票在有效期内会被多次调用复用；一次性会让第二次调用
   就 401。一次性只属于门 A。
3. **校验 `source`**。签名合法、`aud` 合法，但票是给别的线签的，仍然要拒——`source`
   既是计量归属的依据，也是跨线复用票的闸。

**本模块不解析客户身份**（这是本线与 wire 的实质差别，不是没做完）：本仓
`surfaces.account=false`，没有"客户"这张表可查，租户范围由平台在转发头里
注入后由本线业务代码判定（本模块不碰这件事）。所以这里只把网关解析出的**计费侧主体**（`sub`）原样交出去，
调用方按需记日志/归因；**绝不据此建档**（身份永不自动创建）。

拒绝一律给稳定原因码，对外只说"令牌不对"这一层（`wrong_source` 与 `wrong_audience`
在响应里同码，不告诉调用方是哪一步没过——家族既有口径）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "MAX_TTL_SECONDS",
    "ENV_ISSUER",
    "ENV_LINE",
    "ENV_PEM",
    "ENV_JWKS_URI",
    "LineTokenIdentity",
    "LineTokenRejected",
    "LineTokenSettings",
    "settings_from_env",
    "verify_line_token",
]

#: 门 K 令牌的有效期上限（秒）。网关侧同样硬夹 600，两边各守一次。
MAX_TTL_SECONDS = 600

#: 环境变量名与家族一致（wire/lex 同族，运维只记一套写法，换线只换前缀）。
ENV_ISSUER = "FD_QUANT_LINE_TOKEN_ISSUER"
ENV_LINE = "FD_QUANT_LINE_TOKEN_LINE"
ENV_PEM = "FD_QUANT_LINE_TOKEN_PUBLIC_KEY_PEM"
ENV_JWKS_URI = "FD_QUANT_LINE_TOKEN_JWKS_URI"

#: 本线的名字。写成常量而不是读 env 的缺省值——票的 `aud`/`source` 都必须是它，
#: 让缺省值随环境漂移等于让"这条线是谁"变成可配置项。
LINE_NAME = "quant"


class LineTokenRejected(Exception):
    """校验失败。``reason`` 是稳定码，供调用方与测试断言（不要断言文案）。"""

    NOT_CONFIGURED = "not_configured"
    TOKEN_MALFORMED = "token_malformed"
    BAD_SIGNATURE = "bad_signature"
    WRONG_ISSUER = "wrong_issuer"
    WRONG_AUDIENCE = "wrong_audience"
    WRONG_SOURCE = "wrong_source"
    EXPIRED = "expired"
    TTL_TOO_LONG = "ttl_too_long"
    MISSING_CLAIM = "missing_claim"

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class LineTokenIdentity:
    """已校验的门 K 身份。

    ``subject`` 是网关解析出的**计费侧主体**（sub2api 用户 id 的字符串形态），
    **不是** Logto 用户名，也**不是**本仓的租户 id——本仓没有客户表（见模块头注）。
    叫错名字会让人去拿它查一张不存在的表，然后得出"票是好的但解析不到客户"的错觉。
    """

    subject: str
    email: str | None
    scope: tuple[str, ...]
    source: str
    expires_at: int
    jti: str

    @property
    def can_write(self) -> bool:
        """`line:write` 才允许变更类动作（与家族口径一致：只读票不开写面）。"""
        return "line:write" in self.scope


@dataclass(frozen=True)
class LineTokenSettings:
    issuer: str
    line: str = LINE_NAME
    public_key_pem: str = ""
    jwks_uri: str = ""
    algorithms: tuple[str, ...] = ("ES256", "RS256", "EdDSA")
    max_ttl_seconds: int = MAX_TTL_SECONDS

    @property
    def configured(self) -> bool:
        return bool(self.issuer and self.line and (self.jwks_uri or self.public_key_pem))


def settings_from_env(env: dict[str, str] | None = None) -> LineTokenSettings:
    """从环境读设置。**每次调用都重读**——轮换公钥不需要重启进程（与 bearer 门同）。"""
    e = os.environ if env is None else env
    return LineTokenSettings(
        issuer=(e.get(ENV_ISSUER) or "").strip(),
        line=(e.get(ENV_LINE) or LINE_NAME).strip() or LINE_NAME,
        public_key_pem=(e.get(ENV_PEM) or "").strip(),
        jwks_uri=(e.get(ENV_JWKS_URI) or "").strip(),
    )


def verify_line_token(
    token: str,
    *,
    settings: LineTokenSettings,
    jwks_client_factory: Callable[[str], Any] | None = None,
) -> LineTokenIdentity:
    """离线校验一枚门 K 令牌。失败一律抛 :class:`LineTokenRejected`。

    顺序与家族一致：先验签（最贵的一步先做，后面每步都便宜），再判 iss/aud，
    再判 `source`，最后判有效期窗口。**不写任何状态**——门 K 的票可复用，
    所以这里没有重放台账。
    """
    if not settings.configured:
        raise LineTokenRejected(LineTokenRejected.NOT_CONFIGURED, "issuer/line/key incomplete")
    if not isinstance(token, str) or not token.strip():
        raise LineTokenRejected(LineTokenRejected.TOKEN_MALFORMED, "empty token")

    try:
        import jwt as pyjwt
    except ImportError as exc:  # pragma: no cover - 依赖缺失属部署错误
        raise LineTokenRejected(LineTokenRejected.NOT_CONFIGURED, "PyJWT not installed") from exc

    key: Any
    if settings.public_key_pem:
        key = settings.public_key_pem
    elif settings.jwks_uri and jwks_client_factory is not None:
        try:
            key = jwks_client_factory(settings.jwks_uri).get_signing_key_from_jwt(token).key
        except Exception as exc:
            raise LineTokenRejected(
                LineTokenRejected.BAD_SIGNATURE, f"jwks lookup failed: {type(exc).__name__}"
            ) from exc
    else:
        raise LineTokenRejected(LineTokenRejected.NOT_CONFIGURED, "no verification key source")

    try:
        claims = pyjwt.decode(
            token,
            key,
            algorithms=list(settings.algorithms),
            issuer=settings.issuer,
            audience=settings.line,
            options={"verify_exp": True, "verify_aud": True, "verify_iss": True},
        )
    except pyjwt.ExpiredSignatureError as exc:
        raise LineTokenRejected(LineTokenRejected.EXPIRED) from exc
    except pyjwt.InvalidAudienceError as exc:
        raise LineTokenRejected(LineTokenRejected.WRONG_AUDIENCE, f"expected {settings.line}") from exc
    except pyjwt.InvalidIssuerError as exc:
        raise LineTokenRejected(LineTokenRejected.WRONG_ISSUER, f"expected {settings.issuer}") from exc
    except pyjwt.InvalidSignatureError as exc:
        # 必须排在 DecodeError 之前：InvalidSignatureError 是 DecodeError 的子类，
        # 顺序反了会把它报成 token_malformed——原因码指错地方，排查就会往"票格式"
        # 上跑，而真问题是签错了钥（lex 侧 2026-11-12 由跨线用例撞出，此处照抄）。
        raise LineTokenRejected(LineTokenRejected.BAD_SIGNATURE, type(exc).__name__) from exc
    except pyjwt.DecodeError as exc:
        raise LineTokenRejected(LineTokenRejected.TOKEN_MALFORMED) from exc
    except Exception as exc:  # 算法不符等
        raise LineTokenRejected(LineTokenRejected.BAD_SIGNATURE, type(exc).__name__) from exc

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise LineTokenRejected(LineTokenRejected.MISSING_CLAIM, "sub")
    jti = claims.get("jti")
    if not isinstance(jti, str) or not jti.strip():
        raise LineTokenRejected(LineTokenRejected.MISSING_CLAIM, "jti")
    exp = claims.get("exp")
    iat = claims.get("iat")
    if not isinstance(exp, int):
        raise LineTokenRejected(LineTokenRejected.MISSING_CLAIM, "exp")

    if isinstance(iat, int) and exp - iat > settings.max_ttl_seconds:
        # PyJWT 只管"有没有过期"，不管"给的时间太长"——短命约束要线侧自己守。
        raise LineTokenRejected(
            LineTokenRejected.TTL_TOO_LONG, f"{exp - iat}s > {settings.max_ttl_seconds}s"
        )

    source = claims.get("source")
    if not isinstance(source, str) or not source.strip():
        raise LineTokenRejected(LineTokenRejected.MISSING_CLAIM, "source")
    if source.strip() != settings.line:
        raise LineTokenRejected(LineTokenRejected.WRONG_SOURCE, f"expected {settings.line}")

    scope_value = claims.get("scope") or ""
    scope = tuple(str(scope_value).split()) if scope_value else ()
    email = claims.get("email")
    return LineTokenIdentity(
        subject=subject.strip(),
        email=str(email) if isinstance(email, str) and email else None,
        scope=scope,
        source=source.strip(),
        expires_at=exp,
        jti=jti.strip(),
    )
