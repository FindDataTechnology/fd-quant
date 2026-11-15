"""HTTP 层共享 bearer 门——本线自有的常命钥匙（change converge-line-identity-and-keys 8.6）。

门只在 ``MCP_BEARER_TOKEN`` 非空时**存在**：未配置时服务入口仍然走原来的
``*.run(transport="http")``，一个中间件都不挂（与接入前逐字节一致）；配置后
每个 http 请求必须带 ``Authorization: Bearer <token>``，恒时比较，否则在进入
MCP 处理之前就 401。stdio 不经过这里。

``/health`` 与 ``/healthz`` **永远豁免**：探针不带客户钥匙，把它们一起门掉等于
把部署打死（本族有些线把 /healthz 和 /mcp 挂在同一个 app 上）。健康面只回状态与
工具数，不是数据面。

形状与 ``fd-open-data-mcp`` / ``fd-cn-report`` / ``fd-info`` / ``fd-text-core`` 的门
逐行同族，运维只需记一个变量名（``MCP_BEARER_TOKEN``）。它给的是「摘掉壹座后本线
仍能被合法消费」这条能力，而**不是**客户级身份：一把钥匙全客户共用，谁在用由外层
（网关/registry）记录。每客户一把钥匙需要钥匙库 + 吊销面，属 8.x 的后续工作。
"""

from __future__ import annotations

import hmac
import os

from . import line_token

ENV_TOKEN = "MCP_BEARER_TOKEN"

#: 校验通过的门 K 身份放这里（ASGI scope 键）。命名带线名，免得同一进程里
#: 两条线的身份串味。**本仓不解析客户身份**，`sub` 只原样交出去供归因，
#: 绝不据此建档（身份永不自动创建）。
SCOPE_IDENTITY = "quant.line_identity"

# 健康面豁免：探针永远不带钥匙，且只回状态，不是数据面。
EXEMPT_PATHS = (
    "/health",
    "/healthz",
    # 线自描述端点（change converge-line-identity-and-keys 1.4）：公开元数据，
    # 未持有本线凭据的调用方也要能读到「这条线认哪些钥匙」。必须与
    # ``wellknown.register()`` 挂的路由**成对**存在——只加路由不加这里，
    # 配了令牌的线会对匿名发现请求回 401。
    "/.well-known/line-manifest.json",
)


def configured_token() -> str:
    """每请求重读：改旗/轮换不需要重启进程。"""
    return os.environ.get(ENV_TOKEN, "").strip()


def gate_enabled() -> bool:
    """门存在与否 = 「共享钥匙」或「词元轨」任一被配置。

    两者都空时**一个中间件都不挂**，与接入前逐字节一致。本仓的启动入口
    必须用这个谓词而不是 `configured_token()`：只开词元轨的部署（客户拿 `sk-`
    换票直调、不再共用一把线钥匙）如果用旧谓词，中间件根本不会被挂上——数据面
    会对任何匿名请求放行，而日志里一行异常都没有。
    """
    return bool(configured_token()) or line_token.settings_from_env().configured


class BearerAuthMiddleware:
    """ASGI 中间件：配了共享钥匙或词元轨就要求匹配的 bearer，两者都未配则不设门。

    只拦 ``http`` scope——lifespan 必须放过去，否则应用起不来；健康面按
    ``EXEMPT_PATHS`` 豁免。

    两条轨道，**顺序是刻意的**：先做共享钥匙的恒时比较（便宜、无解析），不匹配才
    去验词元票（要验签）。反过来会让每一发共享钥匙都白付一次 JWT 解析，而共享钥匙
    是本线最常用的那条路（平台代调也是它）。
    """

    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return self.asgi_app(scope, receive, send)
        shared = configured_token()
        line_settings = line_token.settings_from_env()
        if not shared and not line_settings.configured:
            return self.asgi_app(scope, receive, send)
        if (scope.get("path") or "") in EXEMPT_PATHS:
            return self.asgi_app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        provided = headers.get(b"authorization", b"")

        # ① 线自有共享钥匙（回退轨）：恒时比较。
        if shared:
            expected = ("Bearer " + shared).encode("utf-8", "surrogateescape")
            if hmac.compare_digest(provided, expected):
                return self.asgi_app(scope, receive, send)

        # ② 门 K：网关换发来的短命票，离线验签。失败**不解释**（与家族同口径：
        #    只说"令牌不对"，不告诉调用方是签名、iss、aud 还是 source 没过）。
        if line_settings.configured:
            raw = provided.decode("utf-8", "surrogateescape")
            bearer = raw[len("Bearer ") :] if raw.startswith("Bearer ") else ""
            if bearer.strip():
                try:
                    ident = line_token.verify_line_token(bearer.strip(), settings=line_settings)
                except line_token.LineTokenRejected:
                    ident = None
                if ident is not None:
                    scope[SCOPE_IDENTITY] = ident
                    return self.asgi_app(scope, receive, send)

        async def _reject(receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b"Bearer"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b'{"error": "unauthorized"}'})

        return _reject(receive, send)
