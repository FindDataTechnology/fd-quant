"""fd-quant MCP 服务入口：**计算面七族 + 决策账本族**（stdio + HTTP 双传输）。

启动：
    python -m fd_quant.server                # stdio（MCP 默认）
    python -m fd_quant.server --http 8080    # HTTP（streamable-http）

与 `fd_quant.compute.server` 的分工（两个入口都保留，各有正当用途）：

- `fd_quant.server`（本文件）：**服务上牌的那个面**。工具集取
  `fd_quant.surface.all_surface_tools()` = 计算面七族 + 账本四工具，与
  `tool-surface.json` 是同一份清单（`scripts/gen_surface.py` 与
  `scripts/selfcheck_surface.py` 也读同一个汇总点）。
- `fd_quant.compute.server`：**纯计算面入口**，工具集只有 `fd_quant.compute.all_tools()`。
  它不含账本，故可跑在读不到本地盘的沙箱里；账本要持有本地 SQLite，本模块只把
  `fd_quant.ledger` 拉进来，`fd_quant/compute/` 内不出现任何账本或 sqlite 依赖
  （计算层的封闭性由 `tests/test_offline_boundary.py` 断言，未被本改动打破）。

两类工具的纪律不同，`instructions` 里显式分开写，免得调用方把账本也当成纯计算：
计算工具只对传入序列作答；账本工具把记录落到本地库（路径 `FD_QUANT_LEDGER_DB` 或
`db_path` 参数），但同样**不取数、不联网**——价格与基准序列一律由调用方传入。
"""
from __future__ import annotations

import argparse
import sys

from fastmcp import FastMCP

from .surface import all_surface_tools, missing_families

SERVER_NAME = "fd-quant"

INSTRUCTIONS = (
    "fd-quant 量化能力面，两组工具：\n"
    "① 确定性计算面（技术指标 / 组合统计 / 基本面比率 / 估值算子 / regime 检测 / "
    "因子统计 / 口径校验）：只对传入的序列与参数作答——不取数、不读库、不联网、不调 LLM。"
    "每个工具要求 as_of（ISO 日期）并在输入序列上执行：含 as_of 之后的数据点会被拒绝并"
    "点名该日期。返回统一带 evidence 六字段与输入的起止日期。\n"
    "② 决策账本（record_decision / settle_decision / retrieve_decisions / "
    "summarize_pack_performance）：把决策追加式落到本地 SQLite（幂等键 "
    "as_of + symbol + pack + skill，重复提交被拒并返回既有 decision_id），到期按传入的价格"
    "序列结算原始收益与相对基准 alpha（基准按市场后缀映射，可显式覆盖），按 pack 聚合口径"
    "统计。账本同样不取数、不联网——价格与基准序列由调用方传入；持有期按交易日计，交易日历"
    "来自传入序列的日期集合。账本无任何推送/广播接口：反思只存、只随检索带出。"
)


def build_server() -> FastMCP:
    app = FastMCP(name=SERVER_NAME, instructions=INSTRUCTIONS)

    missing = missing_families()
    if missing:
        # 不静默：缺族时在 stderr 留痕，免得"工具面少了一族"被当成正常。
        print(
            f"[{SERVER_NAME}] WARNING: compute families not yet present: {', '.join(missing)}",
            file=sys.stderr,
        )

    for spec in all_surface_tools():
        app.tool(spec.fn, name=spec.name, description=spec.doc)

    @app.custom_route("/health", methods=["GET"])
    async def health(_request):  # noqa: ANN001 — FastMCP 传 Request，此处不用
        """探针端点：只回答"进程活着且工具面已装配"。

        **为什么不拿 `/mcp` 当探针**：它只接 POST，GET 应答 400/405 是协议正常
        形态。拿它当探针会把"服务健康"读成"服务挂了"——本仓在网关侧踩过同类
        判读错误（裸 405 曾被误当故障信号）。
        """
        from starlette.responses import JSONResponse

        from .surface import missing_families, tools_by_group

        groups = {g: len(ts) for g, ts in tools_by_group().items()}
        payload = {
            "status": "ok",
            "server": SERVER_NAME,
            "tools": sum(groups.values()),
            "groups": groups,
        }
        # 缺族不是致命（开发期），但必须在探针里可见——否则"少了一族"会被当成健康。
        missing = missing_families()
        if missing:
            payload["missing_families"] = missing
            payload["status"] = "degraded"
        return JSONResponse(payload, status_code=200 if not missing else 503)

    return app


app = build_server()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=SERVER_NAME)
    parser.add_argument("--http", type=int, default=None, metavar="PORT",
                        help="serve streamable-http on PORT instead of stdio")
    parser.add_argument("--host", default="127.0.0.1",
                        help="HTTP bind host (default 127.0.0.1)")
    parser.add_argument("--list-tools", action="store_true",
                        help="print the tool inventory and exit")
    args = parser.parse_args(argv)

    if args.list_tools:
        for spec in all_surface_tools():
            print(f"{spec.group:22s} {spec.name}")
        return 0

    if args.http:
        app.run(transport="http", host=args.host, port=args.http)
        return 0

    app.run()  # stdio
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
