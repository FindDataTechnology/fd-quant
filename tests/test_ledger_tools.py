"""账本工具面（task 5.1）：命名、描述自足、两个显式假设、manifest 双向、无推送形状。

这份文件同时是"账本工具**不是**纯计算工具"这条判断的可执行形式：它们过计算面的
通用质量条款（自足描述：Input: / Returns: / as_of / 单位 / 形状），但**不**被计算面
的那些条款约束——"不读库"这类计算面硬约束对账本不成立（账本就是本地库），
所以账本有自己的族模块与自己的闸门。
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import ledgerdata as ld
from fd_quant.compute import ToolSpec
from fd_quant.ledger import tools as ledger_tools
from fd_quant.surface import EXTRA_GROUP_LEVELS, SURFACE_GROUP_LEVELS, all_surface_tools

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "tool-surface.json"

LEDGER = ledger_tools.TOOLS
IDS = [s.name for s in LEDGER]
# 动词白名单：账本工具的动词必须是这几类之一。"list" 于 add-investment-trading-fleet
# 加入——待结算列举是一个独立的动作（"该结什么"），既不是 retrieve 的点内检索
# （那个只看已结算），也不是 summarize 的聚合。
ASSERTED_VERBS = {"record", "settle", "retrieve", "list", "summarize"}


# ── 命名与注册 ──────────────────────────────────────────────────


def test_ledger_group_registers_six_tools():
    """六个动作各一个工具：落账 / 单票结算 / 组合结算 / 检索 / 待结算列举 / 聚合。

    组合结算是独立工具而不是 settle_decision 的一个分支：两者的输入形状
    （单序列 vs 多腿）与返回字段（成交价 vs 加权收益+现金）都不同，塞进一个
    工具只会让签名变成"两种互斥用法的并集"。
    """
    assert {s.name for s in LEDGER} == {
        "record_decision", "settle_decision", "settle_portfolio", "retrieve_decisions",
        "list_pending_decisions", "summarize_pack_performance",
    }
    assert all(s.group == "decision_ledger" for s in LEDGER)


def test_ledger_tools_appear_in_the_surface_registry():
    """账本工具进得了服务工具面（`all_surface_tools`）——否则"注册进 server"是空话。"""
    surface = {s.name for s in all_surface_tools()}
    for spec in LEDGER:
        assert spec.name in surface
    assert SURFACE_GROUP_LEVELS["decision_ledger"] == "commercial"
    # 封闭集合断言：新增一个对外可见的族必须显式改这里（否则"某族悄悄
    # 变成 commercial"不会有人发现）。portfolio_construction 于
    # add-portfolio-intent-contract 加入——三件纯确定性、只读、幂等的工具。
    assert EXTRA_GROUP_LEVELS == {
        "decision_ledger": "commercial",
        "portfolio_construction": "commercial",
        # add-investment-trading-fleet：组合风控闸（纯函数、只读幂等）
        "portfolio_risk": "commercial",
    }


def test_ledger_tool_names_use_the_extended_verb_whitelist():
    """`动词_名词`，动词落在合规闸门的白名单里（含新增的 record/settle/retrieve）。"""
    for spec in LEDGER:
        assert "_" in spec.name
        verb = spec.name.split("_")[0]
        assert verb in ASSERTED_VERBS, spec.name


def test_ledger_tools_are_toolspec_instances_with_a_group_level():
    for spec in LEDGER:
        assert isinstance(spec, ToolSpec)
        assert spec.group == "decision_ledger"
        assert spec.effective_level == "commercial"
        assert spec.doc, f"{spec.name}: empty description"


# ── 描述自足（含两个显式假设）───────────────────────────────────


@pytest.mark.parametrize("spec", LEDGER, ids=IDS)
def test_ledger_description_is_self_sufficient(spec):
    """task 5.1：描述自足——Input: / Returns: / as_of / 单位 / 形状齐备（计算面同款条款）。"""
    doc = spec.doc
    assert len(doc) >= 120, f"{spec.name}: description too thin ({len(doc)} chars)"
    low = doc.lower()
    assert "input:" in low, f"{spec.name}: no Input: section"
    assert "return" in low, f"{spec.name}: no Returns: section"
    assert "as_of" in low, f"{spec.name}: does not mention as_of"


@pytest.mark.parametrize("spec", LEDGER, ids=IDS)
def test_ledger_description_states_the_two_required_assumptions(spec):
    """task 5.1：逐工具断言描述含两个显式假设（序列由调用方传入 + 持有期按交易日计）。"""
    doc = spec.doc
    assert "账本不取数" in doc, f"{spec.name}: missing the 'ledger never fetches' assumption"
    assert "由调用方传入" in doc, f"{spec.name}: missing 'series supplied by the caller'"
    assert "持有期按交易日计" in doc, (
        f"{spec.name}: missing the 'holding period counted in trading days' assumption"
    )
    assert "交易日历" in doc, f"{spec.name}: missing the trading-calendar caveat"


def test_tool_descriptions_mention_the_db_path_default_and_env_var():
    """库路径的默认值与环境变量必须在描述里（否则调用方不知道记录落在哪）。"""
    for spec in LEDGER:
        assert "db_path" in spec.doc, spec.name
        assert "FD_QUANT_LEDGER_DB" in spec.doc, spec.name


def test_each_tool_documents_its_units_and_defaults():
    """单位与默认值回显：结算/聚合工具写清小数口径，检索工具写清默认 N/M。"""
    settle = _spec("settle_decision").doc
    assert "0.01 = 1%" in settle and "alpha" in settle
    assert "默认 5" in settle  # 默认持有期
    retrieve = _spec("retrieve_decisions").doc
    assert "默认 5" in retrieve and "默认 3" in retrieve  # N / M 默认
    stats = _spec("summarize_pack_performance").doc
    assert "win_rate" in stats and "samples" in stats


def test_aggregation_tool_does_not_claim_to_return_reflection_text():
    """聚合工具是数字视图：描述里不承诺返回反思（反思只在检索里带出）。"""
    doc = _spec("summarize_pack_performance").doc
    returns = doc.split("Returns:", 1)[1]
    assert "reflection" not in returns


# ── as_of 与调用形状 ────────────────────────────────────────────


@pytest.mark.parametrize("spec", LEDGER, ids=IDS)
def test_as_of_is_a_required_kwarg(spec):
    sig = inspect.signature(spec.fn)
    assert "as_of" in sig.parameters, f"{spec.name}: no as_of parameter"
    assert sig.parameters["as_of"].default is inspect.Parameter.empty, (
        f"{spec.name}: as_of must be required (no default)"
    )


@pytest.mark.parametrize("spec", LEDGER, ids=IDS)
def test_missing_as_of_is_refused(spec, tmp_path):
    """缺 as_of 即拒绝（点名 as_of），不会写出任何行。"""
    from fd_quant.ledger import LedgerError, SettlementRefused

    kw = _kwargs_for(spec, tmp_path)
    kw["as_of"] = None
    with pytest.raises((LedgerError, SettlementRefused), match="as_of"):
        spec.fn(**kw)


@pytest.mark.parametrize("spec", LEDGER, ids=IDS)
def test_every_tool_runs_with_zero_configuration_and_is_json_serialisable(spec, tmp_path,
                                                                         monkeypatch):
    """零配置可用（db_path 由参数给）且返回可 JSON 序列化（allow_nan=False）。"""
    monkeypatch.delenv("FD_QUANT_LEDGER_DB", raising=False)
    r = spec.fn(**_kwargs_for(spec, tmp_path))
    assert isinstance(r, dict)
    assert r["tool"] == spec.name
    assert r["db_path"].endswith(".db")
    assert "result" in r
    json.dumps(r, allow_nan=False, ensure_ascii=False)


def test_tools_default_to_the_env_var_when_db_path_is_omitted(tmp_path, monkeypatch):
    """不传 db_path 时走 `FD_QUANT_LEDGER_DB`（零配置部署形态）。"""
    monkeypatch.setenv("FD_QUANT_LEDGER_DB", str(tmp_path / "env-ledger.db"))
    r = ledger_tools.retrieve_decisions(symbol="600519.SS", as_of="2025-01-01")
    assert r["db_path"] == str(tmp_path / "env-ledger.db")


def test_a_recorded_decision_is_visible_to_the_other_tools_through_one_db(tmp_path):
    """四工具在同一条 db_path 上串起来（工具面是端到端演练的调用形态）。"""
    db = str(tmp_path / "shared.db")
    rec = ledger_tools.record_decision(
        as_of="2025-01-06", symbol="600519.SS", pack=ld.PACK, skill=ld.SKILL,
        payload=ld.rating("2025-01-06", "600519.SS").to_dict(), db_path=db,
    )
    did = rec["result"]["decision_id"]
    sym = ld.flat_then_move(100.0, 110.0, 6)
    bench = ld.flat_then_move(3000.0, 3060.0, 6)
    settled = ledger_tools.settle_decision(
        decision_id=did, symbol_dates=sym["dates"], symbol_prices=sym["prices"],
        benchmark_dates=bench["dates"], benchmark_prices=bench["prices"],
        as_of=sym["dates"][-1], db_path=db,
    )
    assert settled["result"]["status"] == "settled"
    found = ledger_tools.retrieve_decisions(symbol="600519.SS", as_of="2025-12-31",
                                            db_path=db)
    assert found["result"]["counts"]["same_symbol"] == 1
    stats = ledger_tools.summarize_pack_performance(as_of="2025-12-31", db_path=db)
    assert stats["result"]["rows"][0]["samples"] == 1


def test_settlement_refusal_reaches_the_caller_as_a_named_error(tmp_path):
    """持有期未满时工具抛出的错误仍带缺口交易日数（调用方能在同一轮里补齐数据）。"""
    from fd_quant.ledger import SettlementRefused

    db = str(tmp_path / "short.db")
    rec = ledger_tools.record_decision(
        as_of="2025-01-06", symbol="600519.SS", pack=ld.PACK, skill=ld.SKILL,
        payload=ld.rating("2025-01-06", "600519.SS").to_dict(), db_path=db,
    )
    three = ld.series([100.0, 101.0, 102.0])
    with pytest.raises(SettlementRefused) as ei:
        ledger_tools.settle_decision(
            decision_id=rec["result"]["decision_id"], symbol_dates=three["dates"],
            symbol_prices=three["prices"], benchmark_dates=three["dates"],
            benchmark_prices=three["prices"], as_of=three["dates"][-1], db_path=db,
        )
    assert ei.value.missing_trading_days == 3
    assert "missing 3 trading days" in str(ei.value)


def test_no_ledger_tool_is_named_like_a_push_or_broadcast():
    """task 3.4 在工具面的形式：不存在 push/notify/broadcast 形状的工具。"""
    banned = ("push", "notify", "broadcast", "subscribe", "emit", "send", "publish")
    for spec in LEDGER:
        low = spec.name.lower()
        assert not any(word in low for word in banned), spec.name


# ── tool-surface.json 与 manifest 生成器 ────────────────────────


def test_manifest_declares_the_ledger_group_with_all_four_tools():
    """task 5.1：`tool-surface.json` 更新——账本组在 manifest 里、工具名逐条对齐。"""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    groups = {g["group"]: g for g in manifest["tool_groups"]}
    assert "decision_ledger" in groups, "manifest lacks the decision_ledger group"
    group = groups["decision_ledger"]
    assert group["level"] == "commercial"
    assert group["tools"] == sorted(spec.name for spec in LEDGER)
    assert group["description"], "group description must be filled"
    assert "序列由调用方传入" in group["description"] or "由调用方传入" in group["description"]
    # 顶层计数把账本算进去
    total = sum(len(g["tools"]) for g in manifest["tool_groups"])
    assert manifest["registry"]["expected_tool_count"] == total
    assert "decision_ledger" in manifest["description"] or "账本" in manifest["description"]


def test_manifest_matches_the_runtime_surface_bidirectionally():
    """双向：manifest ⊆ runtime 且 runtime ⊆ manifest（与 selfcheck 同一断言，跑在 pytest 里）。"""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    man = {t for g in manifest["tool_groups"] for t in g["tools"]}
    run = {s.name for s in all_surface_tools()}
    assert man == run, (sorted(run - man), sorted(man - run))


def test_gen_surface_check_is_green_and_selfcheck_passes():
    """`gen_surface.py --check` 无漂移 + `selfcheck_surface.py` PASS（task 5.1 的验证条件）。"""
    gen = subprocess.run([sys.executable, str(REPO / "scripts" / "gen_surface.py"), "--check"],
                         capture_output=True, text=True, cwd=str(REPO))
    assert gen.returncode == 0, (gen.stdout, gen.stderr)
    assert "一致" in gen.stdout

    chk = subprocess.run([sys.executable, str(REPO / "scripts" / "selfcheck_surface.py"),
                          "--gen-check"],
                         capture_output=True, text=True, cwd=str(REPO))
    assert chk.returncode == 0, (chk.stdout, chk.stderr)
    assert "selfcheck 通过" in chk.stdout
    assert "方向 1" in chk.stdout


def test_server_registers_both_groups_through_the_surface_aggregator():
    """server 入口注册的是计算面 + 账本（同一份 `all_surface_tools`），且计算层不含账本。"""
    from fd_quant.surface import all_surface_tools as agg
    from fd_quant.compute import all_tools as compute_only

    names = {s.name for s in agg()}
    ledger_names = {s.name for s in LEDGER}
    assert ledger_names <= names
    assert compute_only() and not (ledger_names & {s.name for s in compute_only()})

    src = (REPO / "fd_quant" / "server.py").read_text(encoding="utf-8")
    assert "all_surface_tools" in src
    assert "from .surface import" in src


def test_mcp_server_lists_and_calls_the_ledger_tools(tmp_path):
    """真起一次 MCP server（fastmcp 内存态）：工具**列得出来**且**调得动**。

    这条比"注册表里有这四个名字"强一档——它走的是 MCP 自己的 `list_tools` 与
    `call_tool`，故"能列出"与"能服务"是同一份清单这件事在这里被实测。
    """
    fastmcp = pytest.importorskip("fastmcp", reason="server extra not installed")
    import asyncio

    from fd_quant.server import build_server

    async def scenario():
        app = build_server()
        tools = {t.name: t for t in await app.list_tools()}
        for spec in LEDGER:
            assert spec.name in tools, spec.name
            doc = tools[spec.name].description or ""
            assert "账本不取数" in doc and "持有期按交易日计" in doc, spec.name
        db = str(tmp_path / "mcp.db")
        payload = ld.rating("2025-01-06", "600519.SS").to_dict()
        result = await app.call_tool("record_decision", {
            "as_of": "2025-01-06", "symbol": "600519.SS", "pack": ld.PACK,
            "skill": ld.SKILL, "payload": payload, "db_path": db,
        })
        text = getattr(result, "content", [None])[0]
        body = json.loads(getattr(text, "text", None) or json.dumps(result))
        return body

    body = asyncio.run(scenario())
    assert body["result"]["created"] is True
    assert body["result"]["reason"] == "created"
    assert body["tool"] == "record_decision"
    assert fastmcp is not None


def test_compute_layer_stays_closed_after_the_ledger_lands():
    """硬约束回归：账本不侵入 `fd_quant/compute/`（无 sqlite / 无 ledger import）。"""
    import ast

    root = REPO / "fd_quant" / "compute"
    hits: list[str] = []
    for py in sorted(root.glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for n in names:
                base = n.split(".")[0]
                if base in ("sqlite3", "ledger") or n.startswith("fd_quant.ledger"):
                    hits.append(f"{py.name}:{getattr(node, 'lineno', '?')}: {n}")
    assert not hits, "compute layer imported ledger/sqlite:\n" + "\n".join(hits)


# ── 辅助 ────────────────────────────────────────────────────────


def _spec(name: str) -> ToolSpec:
    for spec in LEDGER:
        if spec.name == name:
            return spec
    raise AssertionError(f"no ledger tool named {name}")


def _kwargs_for(spec: ToolSpec, tmp_path: Path) -> dict:
    """每个工具一组最小合法参数（与 selfcheck 一样：按签名给，不猜）。"""
    db = str(tmp_path / f"{spec.name}.db")
    seed = {
        "record_decision": {
            "as_of": "2025-01-06", "symbol": "600519.SS", "pack": ld.PACK, "skill": ld.SKILL,
            "payload": ld.rating("2025-01-06", "600519.SS").to_dict(),
        },
        "settle_decision": {
            "decision_id": "dec-seed", "as_of": "2025-01-11",
            "symbol_dates": ld.dates(6), "symbol_prices": [100.0] * 5 + [110.0],
            "benchmark_dates": ld.dates(6), "benchmark_prices": [3000.0] * 6,
        },
        "settle_portfolio": {
            "decision_id": "pf-seed", "as_of": "2025-01-11", "benchmark_symbol": "000300.SS",
            "legs": {"600519.SS": [ld.dates(6), [100.0] * 5 + [110.0]]},
            "benchmark_dates": ld.dates(6), "benchmark_prices": [3000.0] * 6,
        },
        "list_pending_decisions": {"as_of": "2025-01-11"},
        "retrieve_decisions": {"symbol": "600519.SS", "as_of": "2025-01-11"},
        "summarize_pack_performance": {"as_of": "2025-01-11"},
    }[spec.name]
    if spec.name == "settle_decision":
        # 先落一条决策，再拿真 id（结算工具需要一个存在的 decision）
        rec = ledger_tools.record_decision(
            as_of="2025-01-06", symbol="600519.SS", pack=ld.PACK, skill=ld.SKILL,
            payload=ld.rating("2025-01-06", "600519.SS").to_dict(), db_path=db,
        )
        seed = {**seed, "decision_id": rec["result"]["decision_id"]}
    if spec.name == "settle_portfolio":
        # 组合同理：先落一条**组合**记录（单票记录会被 settle_portfolio 拒绝）
        from fd_quant.portfolio import build_intent

        intent = build_intent(
            [{"as_of": "2025-01-06", "subject": "600519.SS", "side": "buy", "conviction": 0.8}],
            {"kind": "equal_weight", "cash_floor": 0.2}, as_of="2025-01-06",
            producer={"pack": ld.PACK, "skill": ld.SKILL},
        )
        rec = ledger_tools.record_decision(
            as_of="2025-01-06", portfolio_key="2025-01-06-eq", pack=ld.PACK,
            skill=ld.SKILL, payload=intent.to_dict(), record_kind="portfolio", db_path=db,
        )
        seed = {**seed, "decision_id": rec["result"]["decision_id"]}
    return {**seed, "db_path": db}
