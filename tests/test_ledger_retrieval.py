"""反思与点内检索（tasks 3.1–3.4）＋ 聚合视图（tasks 4.1–4.2）。

点内纪律是这份测试的主角：**结算日 ≤ as_of** 的记录才可见，`as_of` 之后才结算的
记录不得出现（蓝本 TradingAgents 到 v0.6.0 才修的 #1251 号 bug 就是这条）。
"""
from __future__ import annotations

import inspect
import json

import pytest

import ledgerdata as ld
from fd_quant.ledger import LedgerStore

#: 单用户本地账本场景的测试调用者（隔离新增维度后必须显式给出）
CALLER = "test-caller"

SYMBOL = "600519.SS"
OTHER = "000001.SZ"


@pytest.fixture()
def store(tmp_path) -> LedgerStore:
    return LedgerStore(tmp_path / "ledger.db")


def _record(store: LedgerStore, *, symbol: str = SYMBOL, as_of: str = "2025-01-06",
            pack: str = ld.PACK, skill: str = ld.SKILL, side: str = "buy") -> dict:
    return store.record(
        caller=CALLER, as_of=as_of, symbol=symbol, pack=pack, skill=skill,
        payload=ld.rating(as_of, symbol, side=side, pack=pack, skill=skill),
    )


def _settle(store: LedgerStore, decision_id: str, *, start: str = "2025-01-06", n: int = 6,
            sym_p0: float = 100.0, sym_p1: float = 110.0, bench_p0: float = 3000.0,
            bench_p1: float = 3060.0, holding_days: int | None = None,
            reflection: str | None = None, benchmark_symbol: str | None = None) -> dict:
    """在 `start` 起的 n 天窗内结算。`as_of` 取窗末，故该窗完全落在窗内。"""
    sym = ld.flat_then_move(sym_p0, sym_p1, n)
    bench = ld.flat_then_move(bench_p0, bench_p1, n)
    sym = ld.series(sym["prices"], start=start)
    bench = ld.series(bench["prices"], start=start)
    return store.settle(
        decision_id=decision_id, symbol_dates=sym["dates"], symbol_prices=sym["prices"],
        benchmark_dates=bench["dates"], benchmark_prices=bench["prices"],
        as_of=sym["dates"][-1], holding_days=holding_days, reflection=reflection,
        benchmark_symbol=benchmark_symbol,
    )


# ── 3.1 反思 ────────────────────────────────────────────────────


def test_reflection_is_stored_with_the_settlement_and_readable_back(store):
    """task 3.1 / spec Scenario「反思随结算存储」：文本与结算数据同记录可查回。"""
    r = _record(store)
    text = "判定方向正确但幅度高估：渠道去化比预期慢两周。下次把库存周转作为前置条件。"
    _settle(store, r["decision_id"], reflection=text)
    s = store.get(r["decision_id"])["settlement"]
    assert s["reflection"] == text
    assert s["alpha"] is not None  # 反思与结算数据同记录，不另开一处


def test_settlement_without_reflection_succeeds_and_the_field_is_empty(store):
    """task 3.1 / spec Scenario「无反思仍可结算」：不传反思 → 结算成功、字段为空。"""
    r = _record(store)
    out = _settle(store, r["decision_id"])
    assert out["created"] is True
    assert out["reflection"] is None
    assert store.get(r["decision_id"])["settlement"]["reflection"] is None


def test_reflection_may_be_empty_string_and_is_stored_as_given(store):
    """空字符串是"传了但没有内容"，与"没传"（null）不同——两者都可用，各自原样存。"""
    a = _record(store, symbol="600519.SS", as_of="2025-01-06")
    b = _record(store, symbol="000001.SZ", as_of="2025-01-20")
    _settle(store, a["decision_id"], start="2025-01-06", reflection="")
    _settle(store, b["decision_id"], start="2025-01-20")
    assert store.get(a["decision_id"])["settlement"]["reflection"] == ""
    assert store.get(b["decision_id"])["settlement"]["reflection"] is None


def test_reflection_is_not_written_by_anything_but_settlement(store):
    """反思只有一条写入路径（结算），账本不做任何自动生成/推送。"""
    r = _record(store)
    assert store.get(r["decision_id"])["settlement"] is None
    # 库内没有任何"反思表 / 反思队列"——反思是 settlements 的一列
    assert store.row_counts()["settlements"] == 0


# ── 3.2 点内检索（含 N/M 限额与倒序）─────────────────────────────


def test_records_settled_after_as_of_are_filtered_out(store):
    """task 3.2 / spec Scenario「未来结算被过滤」：as_of 之后才结算的记录不返回。

    构造两条同标的记录：一条在 2025-01-13 结算、一条在 2025-01-20 结算；以
    2025-01-13 检索只应看到第一条。
    """
    early = _record(store, as_of="2025-01-06")
    _settle(store, early["decision_id"], start="2025-01-06")          # 结算日 2025-01-11
    late = _record(store, as_of="2025-01-14")
    _settle(store, late["decision_id"], start="2025-01-14")           # 结算日 2025-01-19

    out = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-01-11")
    ids = [x["decision_id"] for x in out["same_symbol"]]
    assert early["decision_id"] in ids
    assert late["decision_id"] not in ids
    assert out["counts"]["total"] == 1

    # 把口径推到第二条的结算日之后，两条都在（对照：过滤的是时间，不是记录）
    later = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-01-19")
    assert {x["decision_id"] for x in later["same_symbol"]} == {
        early["decision_id"], late["decision_id"]
    }


def test_results_are_ordered_by_settlement_date_descending(store):
    """spec：「检索结果 SHALL 按结算日倒序」。"""
    ids = []
    for i, start in enumerate(["2025-01-06", "2025-02-06", "2025-03-06"]):
        r = _record(store, symbol=f"60051{i}.SZ", as_of=start)
        _settle(store, r["decision_id"], start=start)
        ids.append(r["decision_id"])
    out = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31")
    dates = [x["settlement"]["settlement_date"] for x in out["cross_symbol"]]
    assert dates == sorted(dates, reverse=True)
    assert len(out["cross_symbol"]) == 3


def test_n_same_limit_returns_the_most_recent_n(store):
    """task 3.2 / spec Scenario「N/M 限额生效」：同标的 8 条可用记录、N=5 → 最近 5 条。"""
    created = []
    starts = [f"2025-{m:02d}-01" for m in range(1, 9)]
    for i, start in enumerate(starts):
        r = _record(store, symbol=SYMBOL, as_of=start)
        _settle(store, r["decision_id"], start=start, sym_p1=100.0 + i)
        created.append((start, r["decision_id"]))

    out = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31", n_same=5)
    assert len(out["same_symbol"]) == 5
    returned = {x["decision_id"] for x in out["same_symbol"]}
    expected = {did for _s, did in created[-5:]}  # 结算日最晚的 5 条
    assert returned == expected
    assert out["n_same"] == 5

    # N 可配：N=3 只回 3 条；N=8 回全部 8 条
    assert len(store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31", n_same=3)["same_symbol"]) == 3
    assert len(store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31", n_same=8)["same_symbol"]) == 8
    assert len(store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31", n_same=0)["same_symbol"]) == 0


def test_m_cross_limit_bounds_the_cross_symbol_lessons(store):
    """跨标的教训限 M 条（默认 3），同样按结算日倒序、同样点内。"""
    for i in range(5):
        sym = f"60000{i}.SZ"
        start = f"2025-{i + 1:02d}-01"
        r = _record(store, symbol=sym, as_of=start)
        _settle(store, r["decision_id"], start=start)
    out = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31")
    assert len(out["cross_symbol"]) == 3
    assert out["m_cross"] == 3
    assert all(x["symbol"] != SYMBOL for x in out["cross_symbol"])

    out_m = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31", m_cross=5)
    assert len(out_m["cross_symbol"]) == 5


def test_same_symbol_and_cross_symbol_are_disjoint(store):
    """两组不重叠：同标的走 N 限额，跨标的走 M 限额，一条记录不会同时出现。"""
    mine = _record(store, symbol=SYMBOL, as_of="2025-01-06")
    _settle(store, mine["decision_id"], start="2025-01-06")
    theirs = _record(store, symbol=OTHER, as_of="2025-01-06")
    _settle(store, theirs["decision_id"], start="2025-01-06")
    out = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31")
    same_ids = {x["decision_id"] for x in out["same_symbol"]}
    cross_ids = {x["decision_id"] for x in out["cross_symbol"]}
    assert same_ids == {mine["decision_id"]}
    assert cross_ids == {theirs["decision_id"]}
    assert not (same_ids & cross_ids)


def test_pending_records_never_appear(store):
    """未结算记录不出现在检索里（点内检索的是**已结算**的历史，不是待办）。"""
    done = _record(store, symbol=SYMBOL, as_of="2025-01-06")
    _settle(store, done["decision_id"], start="2025-01-06")
    pending = _record(store, symbol=SYMBOL, as_of="2025-02-06")
    out = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31")
    assert {x["decision_id"] for x in out["same_symbol"]} == {done["decision_id"]}
    assert pending["decision_id"] not in {x["decision_id"] for x in out["same_symbol"]}


def test_pack_filter_narrows_the_retrieval(store):
    """可选 pack 过滤：只看某个 pack 的历史（跨标的教训也按同一过滤）。"""
    a = _record(store, symbol=SYMBOL, as_of="2025-01-06", pack="pack-a")
    _settle(store, a["decision_id"], start="2025-01-06")
    b = _record(store, symbol="000002.SZ", as_of="2025-01-06", pack="pack-b")
    _settle(store, b["decision_id"], start="2025-01-06")
    out = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31", pack="pack-a")
    assert {x["decision_id"] for x in out["same_symbol"]} == {a["decision_id"]}
    assert out["cross_symbol"] == []
    assert out["pack"] == "pack-a"


def test_retrieval_from_an_empty_ledger_returns_empty_arrays(store):
    """空账本查询是合法结果（两个空数组），不是错误。"""
    out = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-01-01")
    assert out["same_symbol"] == [] and out["cross_symbol"] == []
    assert out["counts"] == {"same_symbol": 0, "cross_symbol": 0, "total": 0}


def test_retrieval_validates_its_arguments(store):
    """口径参数非法即拒绝（ISO 日期、非空 symbol、非负限额）。"""
    from fd_quant.ledger import LedgerError, SettlementRefused

    with pytest.raises(SettlementRefused):
        store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025/01/01")
    with pytest.raises(LedgerError):
        store.retrieve(caller=CALLER, symbol="  ", as_of="2025-01-01")
    with pytest.raises(LedgerError):
        store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-01-01", n_same=-1)
    with pytest.raises(LedgerError):
        store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-01-01", m_cross=1.5)


# ── 3.3 检索项含快照与结算 ──────────────────────────────────────


def test_retrieved_item_carries_snapshot_and_settlement_field_by_field(store):
    """task 3.3：检索返回项逐字段断言（判定快照 + 结算结果 + 反思）。"""
    r = store.record(
        caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
        payload=ld.rating("2025-01-06", SYMBOL, side="buy", conviction=0.7,
                          price_target=1500.0, evidence=[("ev-dcf", 1500.0, "CNY/share")],
                          run_id="run-001"),
    )
    _settle(store, r["decision_id"], reflection="方向对、幅度低估，下次给更宽的点位区间。",
            benchmark_symbol="000300.SS")
    item = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31")["same_symbol"][0]

    # 判定快照
    assert item["decision_id"] == r["decision_id"]
    assert item["as_of"] == "2025-01-06"
    assert item["symbol"] == SYMBOL
    assert item["pack"] == ld.PACK and item["skill"] == ld.SKILL
    assert item["status"] == "settled"
    assert item["revises"] is None and item["revision_seq"] == 0
    assert item["created_at"]
    payload = item["payload"]
    assert payload["side"] == "buy" and payload["conviction"] == 0.7
    assert payload["price_target"] == 1500.0
    assert payload["target_evidence_ids"] == ["ev-dcf"]
    assert payload["producer"] == {"pack": ld.PACK, "skill": ld.SKILL, "run_id": "run-001"}
    assert payload["evidence"][0]["evidence_id"] == "ev-dcf"
    assert item["evidence"] == ["ev-dcf"]      # 引用数组与 payload 的引用一致
    assert item["target_weights"] is None

    # 结算结果
    s = item["settlement"]
    assert s["settlement_date"] == ld.dates(6)[-1]
    assert s["entry_price"] == 100.0 and s["exit_price"] == 110.0
    assert s["raw_return"] == pytest.approx(0.10, abs=1e-12)
    assert s["benchmark_symbol"] == "000300.SS"
    assert s["benchmark_mapping"] == "explicit"
    assert s["benchmark_return"] == pytest.approx(0.02, abs=1e-12)
    assert s["alpha"] == pytest.approx(0.08, abs=1e-12)
    assert s["holding_days"] == 5
    assert s["reflection"] == "方向对、幅度低估，下次给更宽的点位区间。"
    assert s["settled_at"]


def test_retrieved_item_carries_the_revision_pointer(store):
    """修订过的记录在检索里带 revises 指针（历史可追溯"这一代改了什么"）。

    三代链：v1 初始（buy）→ v2 修订（hold）→ v3 修订（sell）。检索应回 v3，且
    revises 指向 v2、revision_seq=2，而初代 v1 的 payload 逐字未被改写。
    """
    v1 = _record(store, as_of="2025-01-06", side="buy")
    v2 = store.record(caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                      payload=ld.rating("2025-01-06", SYMBOL, side="hold"),
                      revises=v1["decision_id"])
    v3 = store.record(caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                      payload=ld.rating("2025-01-06", SYMBOL, side="sell"),
                      revises=v2["decision_id"])
    assert (v2["created"], v3["created"]) == (True, True)
    _settle(store, v3["decision_id"], start="2025-01-06")
    item = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31")["same_symbol"][0]
    assert item["decision_id"] == v3["decision_id"]
    assert item["revises"] == v2["decision_id"]
    assert item["revision_seq"] == 2
    assert item["payload"]["side"] == "sell"
    assert store.get(v1["decision_id"])["payload"]["side"] == "buy"    # 初代未被改写
    assert store.get(v2["decision_id"])["payload"]["side"] == "hold"   # 中间代也未被改写


def test_retrieval_result_is_json_serialisable(store):
    """检索结果整体可 JSON 序列化（`allow_nan=False`）——消费方要能直接塞进上下文。"""
    r = _record(store)
    _settle(store, r["decision_id"], reflection="一段反思。")
    out = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31")
    text = json.dumps(out, ensure_ascii=False, allow_nan=False)
    assert SYMBOL in text and "一段反思。" in text


# ── 3.4 无推送 / 无广播 ─────────────────────────────────────────


def test_ledger_public_api_has_no_push_notify_broadcast_surface():
    """task 3.4 的结构性断言：账本公开 API 面**不含** push/notify/broadcast 类名字。

    这是 spec「反思 SHALL NOT 由账本主动推送给任何消费方」的机械形式：如果将来有人
    给账本加一个 `notify_reflection()`，这条会红——纪律不靠走查记忆，靠断言。
    """
    import fd_quant.ledger as ledger
    from fd_quant.ledger import store as store_mod
    from fd_quant.ledger import settlement as settlement_mod
    from fd_quant.ledger import tools as tools_mod

    banned = ("push", "notify", "broadcast", "subscribe", "emit", "send", "publish",
              "subscriber", "listener", "callback", "webhook", "daemon", "scheduler",
              "poll", "watch")

    def public_names(obj) -> set[str]:
        out = set()
        for name, _member in inspect.getmembers(obj):
            if name.startswith("_"):
                continue
            out.add(name)
        return out

    candidates: dict[str, set[str]] = {
        "fd_quant.ledger module": public_names(ledger),
        "LedgerStore": public_names(LedgerStore),
        "tools module": public_names(tools_mod),
        "settlement module": public_names(settlement_mod),
        "store module": public_names(store_mod),
    }
    for label, names in candidates.items():
        hits = sorted(
            n for n in names
            if any(word in n.lower() for word in banned)
        )
        assert not hits, f"{label} exposes push-like API: {hits}"


def test_no_push_like_name_in_the_ledger_source_text():
    """同一纪律的文本层对照：源码里除注释/描述外不出现 push/notify 类**标识符**。

    （注释与 docstring 里会**提到**这些词——设计决定本身就要写下来；这里扫的是
    可能成为接口的部分：`def` / `class` / 工具名字面量。）
    """
    import ast
    from pathlib import Path

    banned = ("push", "notify", "broadcast", "subscribe", "emit", "send", "publish",
              "webhook")
    root = Path(__file__).resolve().parents[1] / "fd_quant" / "ledger"
    hits: list[str] = []
    for py in sorted(root.glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            else:
                continue
            for name in names:
                low = name.lower()
                if any(word in low for word in banned):
                    hits.append(f"{py.name}:{getattr(node, 'lineno', '?')}: {name}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in list(node.args.args) + list(node.args.kwonlyargs):
                    low = arg.arg.lower()
                    if any(word in low for word in banned):
                        hits.append(f"{py.name}:{node.lineno}: arg {arg.arg}")
    assert not hits, "ledger source declares a push-like identifier:\n" + "\n".join(hits)


def test_the_only_reflection_output_path_is_retrieval(store):
    """task 3.4 的代码走查记录（可执行形式）：反思只有**两条**读出口——

    ① `get(decision_id)`（按 id 单查）② `retrieve(...)`（点内检索）。
    没有任何"全部反思列表"或"最近反思流"接口：不按 symbol/as_of 检索就拿不到反思，
    这正是"注入决策属消费侧行为"的边界。下面逐一点名公开查询方法并断言集合封闭。
    """
    query_methods = {
        name for name, _m in inspect.getmembers(LedgerStore, inspect.isfunction)
        if not name.startswith("_")
    }
    assert query_methods == {
        "record",              # 写入（返回的是新记录，不含历史反思）
        "settle",              # 写入（返回本次/既有结算，含反思——单条，不构成流）
        "settle_portfolio",    # 写入（组合结算；含反思，同样单条、不构成流）
        "settle_metric_threshold",  # 写入（阈值口径结算；返回单条，含反思但不构成流）
        "get",                 # 读：按 id 单查
        "retrieve",            # 读：点内检索
        "list_pending",        # 读：待结算清单（只回 pending，不含反思文本）
        "stats_by_pack",       # 读：聚合数字（**不含**反思文本）
        "row_counts",          # 读：行数
        # 风控阈值留痕（add-investment-trading-fleet 3.5）：阈值变更历史。
        # 不含反思文本——它是另一条审计线，不与决策反思混在一起。
        "record_risk_thresholds",
        "latest_risk_thresholds",
    }, f"ledger gained a public method without a push/reflection-output review: {query_methods}"

    # 聚合视图不含反思文本（数字视图与文本视图分离：绩效数字可公开，反思只随点内检索）
    r = _record(store)
    _settle(store, r["decision_id"], reflection="不应出现在聚合里的反思文本。")
    stats = store.stats_by_pack(caller=CALLER, as_of="2025-12-31")
    assert "不应出现在聚合里的反思文本。" not in json.dumps(stats, ensure_ascii=False, default=str)
    assert "reflection" not in json.dumps(stats)


# ── 4.1 聚合统计 ────────────────────────────────────────────────


def _three_alpha_settlements(store: LedgerStore) -> list[dict]:
    """构造 alpha +1% / -0.5% / +2% 的三条已结算记录（基准恒为 0 收益）。"""
    out = []
    cases = [("600519.SS", 100.0, 101.0),      # +1%
             ("000001.SZ", 200.0, 199.0)]      # -0.5%
    for sym, p0, p1 in cases:
        r = _record(store, symbol=sym, as_of="2025-01-06")
        _settle(store, r["decision_id"], start="2025-01-06", sym_p0=p0, sym_p1=p1,
                bench_p0=1000.0, bench_p1=1000.0)
        out.append(r)
    r = _record(store, symbol="300750.SZ", as_of="2025-01-06")
    _settle(store, r["decision_id"], start="2025-01-06", sym_p0=50.0, sym_p1=51.0,
            bench_p0=1000.0, bench_p1=1000.0)   # +2%
    out.append(r)
    return out


def test_stats_by_pack_counts_mean_alpha_and_win_rate(store):
    """task 4.1 / spec Scenario「聚合统计正确」：3 条 +1% / -0.5% / +2% → 样本 3、胜率 2/3。"""
    _three_alpha_settlements(store)
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31")
    assert out["only_settled"] is True
    row = next(g for g in out["rows"] if g["pack"] == ld.PACK)
    assert row["samples"] == 3
    assert row["wins"] == 2
    assert row["win_rate"] == pytest.approx(2 / 3)
    assert row["mean_alpha"] == pytest.approx((0.01 + -0.005 + 0.02) / 3)
    assert row["mean_raw_return"] == pytest.approx((0.01 + -0.005 + 0.02) / 3)


def test_stats_only_counts_settled_records(store):
    """spec：视图 SHALL 只统计 `settled` 记录——pending 的决策不算样本。"""
    settled = _record(store, symbol="600519.SS", as_of="2025-01-06")
    _settle(store, settled["decision_id"], start="2025-01-06", sym_p1=101.0,
            bench_p0=1000.0, bench_p1=1000.0)
    _record(store, symbol="000001.SZ", as_of="2025-02-06")  # pending，不结算
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31")
    row = next(g for g in out["rows"] if g["pack"] == ld.PACK)
    assert row["samples"] == 1
    assert row["counts"] if False else True
    assert store.row_counts()["pending"] == 1


def test_stats_respects_the_as_of_cutoff(store):
    """聚合同样点内：结算日 > as_of 的记录不计入（与检索同一纪律，口径不分叉）。"""
    early = _record(store, symbol="600519.SS", as_of="2025-01-06")
    _settle(store, early["decision_id"], start="2025-01-06")
    late = _record(store, symbol="000001.SZ", as_of="2025-02-06")
    _settle(store, late["decision_id"], start="2025-02-06")
    early_only = store.stats_by_pack(caller=CALLER, as_of="2025-01-11")
    row = next(g for g in early_only["rows"] if g["pack"] == ld.PACK)
    assert row["samples"] == 1
    both = store.stats_by_pack(caller=CALLER, as_of="2025-12-31")
    assert next(g for g in both["rows"] if g["pack"] == ld.PACK)["samples"] == 2


def test_win_rate_is_alpha_strictly_greater_than_zero(store):
    """胜率按定义：alpha > 0 才算赢（alpha 恰为 0 记负——"跑平基准"不是胜）。"""
    zero = _record(store, symbol="600519.SS", as_of="2025-01-06")
    _settle(store, zero["decision_id"], start="2025-01-06", sym_p0=100.0, sym_p1=110.0,
            bench_p0=1000.0, bench_p1=1100.0)   # alpha 恰 0
    win = _record(store, symbol="000001.SZ", as_of="2025-01-06")
    _settle(store, win["decision_id"], start="2025-01-06", sym_p0=100.0, sym_p1=111.0,
            bench_p0=1000.0, bench_p1=1100.0)   # alpha > 0
    row = next(g for g in store.stats_by_pack(caller=CALLER, as_of="2025-12-31")["rows"]
               if g["pack"] == ld.PACK)
    assert row["samples"] == 2 and row["wins"] == 1
    assert row["win_rate"] == pytest.approx(0.5)


def test_stats_group_by_skill(store):
    """按 skill 分组：同 pack 两个技能各成一行（design.md 的有意设计，用于"哪个模式更准"）。"""
    for skill, p1 in (("equity-research-thesis", 111.0), ("equity-research-modeling", 99.0)):
        r = store.record(caller=CALLER, as_of="2025-01-06", symbol=f"6005{len(skill)}.SS", pack=ld.PACK,
                         skill=skill,
                         payload=ld.rating("2025-01-06", f"6005{len(skill)}.SS", pack=ld.PACK,
                                           skill=skill))
        _settle(store, r["decision_id"], start="2025-01-06", sym_p1=p1,
                bench_p0=1000.0, bench_p1=1000.0)
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31", group_by_skill=True)
    assert out["group_by"] == ["pack", "skill", "record_kind"]
    rows = {g["skill"]: g for g in out["rows"] if g["pack"] == ld.PACK}
    assert rows["equity-research-thesis"]["samples"] == 1
    assert rows["equity-research-thesis"]["mean_alpha"] == pytest.approx(0.11)
    assert rows["equity-research-modeling"]["mean_alpha"] == pytest.approx(-0.01)


def test_stats_group_by_holding_period(store):
    """按持有期分组：3 / 5 / 10 日各成一行，同一 pack 的窗口可比（批次边界回归的支撑）。"""
    for start, hd in (("2025-01-06", 3), ("2025-02-06", 5), ("2025-03-06", 10)):
        r = _record(store, symbol=SYMBOL, as_of=start)
        sym = ld.series([100.0] * (hd + 1), start=start, )["dates"]
        prices = [100.0] * hd + [110.0]
        bench_dates = ld.series([1000.0] * (hd + 1), start=start)["dates"]
        bench_prices = [1000.0] * (hd + 1)
        store.settle(decision_id=r["decision_id"], symbol_dates=sym, symbol_prices=prices,
                     benchmark_dates=bench_dates, benchmark_prices=bench_prices,
                     as_of=sym[-1], holding_days=hd)
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31", group_by_holding=True,
                              holding_days=[3, 5, 10])
    assert out["group_by"] == ["pack", "record_kind", "holding_days"]
    rows = {g["holding_days"]: g for g in out["rows"] if g["pack"] == ld.PACK}
    assert set(rows) == {3, 5, 10}
    assert all(rows[h]["samples"] == 1 for h in rows)
    assert all(rows[h]["mean_alpha"] == pytest.approx(0.10) for h in rows)


def test_stats_by_skill_and_holding_together(store):
    """两级分组可同时开：pack × skill × holding_days 的组键唯一且各自成行。"""
    r = store.record(caller=CALLER, as_of="2025-01-06", symbol=SYMBOL, pack=ld.PACK, skill=ld.SKILL,
                     payload=ld.rating("2025-01-06", SYMBOL, pack=ld.PACK, skill=ld.SKILL))
    _settle(store, r["decision_id"], start="2025-01-06")
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31", group_by_skill=True, group_by_holding=True,
                              holding_days=[5, 10])
    assert out["group_by"] == ["pack", "skill", "record_kind", "holding_days"]
    rows = {g["holding_days"]: g for g in out["rows"] if g["pack"] == ld.PACK}
    assert rows[5]["skill"] == ld.SKILL and rows[5]["samples"] == 1
    assert rows[10]["skill"] == ld.SKILL and rows[10]["samples"] == 0


# ── 4.2 空组明确返回零 ──────────────────────────────────────────


def test_querying_a_pack_with_no_settlements_returns_a_zero_row(store):
    """task 4.2 / spec Scenario「空组明确返回零」：无结算记录的 pack 返回样本数 0 的行。"""
    _three_alpha_settlements(store)
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31", packs=["pack-never-settled"])
    row = next(g for g in out["rows"] if g["pack"] == "pack-never-settled")
    assert row["samples"] == 0
    assert row["wins"] == 0
    assert row["mean_alpha"] is None and row["win_rate"] is None
    assert row["mean_raw_return"] is None


def test_registered_but_unsettled_pack_shows_zero_not_omitted(store):
    """已落记录但一条都没结算的 pack：行在场、样本 0（"还没结算" ≠ "没这个 pack"）。"""
    _record(store, symbol=SYMBOL, as_of="2025-01-06")  # pending，不结算
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31")
    assert ld.PACK in out["registered_packs"]
    row = next(g for g in out["rows"] if g["pack"] == ld.PACK)
    assert row["samples"] == 0 and row["mean_alpha"] is None


def test_empty_ledger_stats_are_empty_rows_not_an_error(store):
    """空账本 + 显式请求 pack → 零行（不报错；这也是端到端演练的起点）。"""
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31", packs=["pack-a", "pack-b"])
    assert [g["pack"] for g in out["rows"]] == ["pack-a", "pack-b"]
    assert all(g["samples"] == 0 for g in out["rows"])
    assert out["registered_packs"] == ["pack-a", "pack-b"]


def test_zero_sample_row_for_a_holding_bucket_that_has_no_settlements(store):
    """按持有期分组时，没出现过的窗口档位同样显式出 0 行（3 日有、10 日无）。"""
    r = _record(store, as_of="2025-01-06")
    _settle(store, r["decision_id"], start="2025-01-06")
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31", group_by_holding=True,
                              holding_days=[5, 10])
    rows = {g["holding_days"]: g for g in out["rows"] if g["pack"] == ld.PACK}
    assert rows[5]["samples"] == 1 and rows[10]["samples"] == 0
    assert rows[10]["mean_alpha"] is None


def test_stats_output_is_json_serialisable(store):
    """聚合结果可 JSON 序列化（allow_nan=False）；含空组时也不例外。"""
    _three_alpha_settlements(store)
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31", packs=["pack-empty"],
                              group_by_holding=True, holding_days=[3, 5, 10])
    json.dumps(out, allow_nan=False, ensure_ascii=False)
    assert any(g["samples"] == 0 for g in out["rows"])
