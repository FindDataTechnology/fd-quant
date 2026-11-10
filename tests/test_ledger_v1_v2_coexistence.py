"""v1 记录（Rating / PortfolioIntent）与 v2 记录（Decision）同表共存。

两套形状来自两条不同的线：
- v1 是本 change 一直在用的（`record_kind` + `symbol`/`portfolio_key`，三列 v2 字段全 NULL）
- v2 由另一条线引入（`subject_kind` + `subject_key` + `action` + 前提/结算规则）

表里有一条 CHECK：v1 三列全空、v2 三列全给，**不许"半套"**。这条 CHECK 是两套形状能
共存的前提，本文件测的就是这个前提在真实写入下成立，且**我的三条查询不因 v2 行而崩**。

为什么值得单独测：两个会话各自都测了自己的形状，但没人测"混在一张表里"。而这条接缝
一旦破，症状会是"聚合少了行"或"检索看不到记录"——都不是报错，是最难发现的那种。
"""
from __future__ import annotations

import pytest

from fd_quant.contracts.decision import Decision
from fd_quant.ledger.store import LedgerStore

RATING = {
    "schema_version": "1", "as_of": "2026-06-30", "subject": "600519", "side": "buy",
    "basis": "v1 判定",
    "evidence": [{"evidence_id": "e", "label": "l", "value": 1, "unit": "u",
                  "as_of": "2026-06-30", "source": "s"}],
}

PRECONDITION = {"claim": "收盘价大于零", "metric_ref": "price.close",
                "comparator": "gt", "threshold": 0.0, "as_of": "2026-06-30"}


@pytest.fixture
def ledger(tmp_path):
    return LedgerStore(tmp_path / "co.db")


def _record_v1(store, pack="p", producer=None):
    payload = {**RATING, "producer": producer or {"pack": pack, "skill": "k"}}
    return store.record(caller="c1", as_of="2026-06-30", symbol="600519", pack=pack,
                        skill="k", payload=payload)


def _record_v2(store, pack="p2"):
    d = Decision(as_of="2026-06-30", subject="600519", subject_kind="instrument",
                 action="buy", basis="v2 判定", settlement_rule_id="alpha_vs_benchmark",
                 preconditions=[PRECONDITION],
                 evidence=[{"evidence_id": "e1", "label": "l", "value": 1, "unit": "u",
                            "as_of": "2026-06-30", "source": "s"}],
                 producer={"pack": pack, "skill": "k"})
    return store.record(caller="c1", as_of="2026-06-30", symbol="600519", pack=pack,
                        skill="k", payload=d.to_dict())


def test_both_shapes_land_in_the_same_table(ledger):
    assert _record_v1(ledger)["created"] is True
    assert _record_v2(ledger)["created"] is True
    assert ledger.row_counts()["decisions"] == 2


def test_v1_rows_leave_v2_columns_null(ledger):
    """v1 三列全空是 CHECK 允许的一半；若被写成"半套"，约束会拒绝。"""
    import sqlite3

    _record_v1(ledger)
    con = sqlite3.connect(ledger.db_path)
    row = con.execute("SELECT subject_kind, subject_key, action FROM decisions").fetchone()
    con.close()
    assert row == (None, None, None)


def test_v2_payload_is_validated_as_v2_but_columns_stay_null(ledger):
    """**跨线已知缺口（记录在案，不是断言通过）**。

    现状：`_check_payload` 会把带 `subject_kind` 的 payload 按 v2 `Decision` 校验，
    但**没有任何写入路径**把 `subject_kind` / `subject_key` / `action` 落进列里——
    全仓只有 CHECK 约束、迁移语句（写字面 NULL）与校验分派提到这三个名字。

    后果：v2 payload 落库后，行在**列**上仍是 v1 形状（三列全 NULL）。CHECK 允许
    （全 NULL 是 v1 分支），所以**不报错**；但任何按 `subject_kind` 过滤/分组的
    读路径会看不到这些行。

    本条把现状钉住，作用有二：
    1. 若该线的作者补上写入路径，这条会红——届时改成断言非 NULL 即可（是"提醒"不是"阻碍"）；
    2. 若有人以为"v2 已经端到端可用"，这条会告诉他列是空的。

    语义（`subject_key` 该归一化成什么）属该线作者的决定，本方案不代拟。
    """
    import sqlite3

    _record_v2(ledger)
    con = sqlite3.connect(ledger.db_path)
    row = con.execute(
        "SELECT subject_kind, subject_key, action FROM decisions"
    ).fetchone()
    con.close()
    assert row == (None, None, None), (
        "v2 列已被写入——说明该线补上了写入路径；请把本测试改为断言非 NULL，"
        f"并核对 subject_key 的语义。实际：{row}"
    )


def test_v2_payload_still_round_trips_through_retrieval(ledger):
    """列是 NULL，但 payload 本身完整保留——所以"数据没丢"，只是列没同步。"""
    _record_v2(ledger)
    out = ledger.retrieve(caller="c1", symbol="600519", as_of="2026-07-20")
    assert out["same_symbol"] == []          # 未结算
    # 直接读 payload 确认 v2 内容在
    import sqlite3
    con = sqlite3.connect(ledger.db_path)
    raw = con.execute("SELECT payload FROM decisions").fetchone()[0]
    con.close()
    assert '"subject_kind"' in raw and '"action"' in raw


def test_list_pending_handles_both_shapes(ledger):
    """待结算清单是定时复盘的输入：v2 行也必须在清单里，不能因形状不同被漏掉。"""
    _record_v1(ledger)
    _record_v2(ledger)
    out = ledger.list_pending(caller="c1", today="2026-07-20")
    assert out["pagination"]["total"] == 2
    packs = {i["pack"] for i in out["items"]}
    assert packs == {"p", "p2"}
    # 每条都带 record_kind 与 subject（复盘岗据此选结算入口）
    for item in out["items"]:
        assert item["record_kind"] in ("single", "portfolio")
        assert item["subject"] == "600519"


def test_retrieve_handles_both_shapes(ledger):
    """点内检索不因 v2 行报错（未结算时结果为空是正确行为）。"""
    _record_v1(ledger)
    _record_v2(ledger)
    out = ledger.retrieve(caller="c1", symbol="600519", as_of="2026-07-20")
    assert out["same_symbol"] == []  # 两条都还没结算


def test_stats_by_pack_counts_both_shapes_as_zero_sample_groups(ledger):
    """聚合里两个 pack 都成组出现、样本为 0（未结算）——不因形状不同而少组。"""
    _record_v1(ledger)
    _record_v2(ledger)
    stats = ledger.stats_by_pack(caller="c1", as_of="2026-07-20", packs=["p", "p2"])
    groups = {r["pack"] for r in stats["rows"]}
    assert groups == {"p", "p2"}
    assert all(r["samples"] == 0 for r in stats["rows"])


def test_coexistence_survives_a_reopen(ledger, tmp_path):
    """重开库（触发全部迁移）后两种形状仍在——迁移链不得只认一种。"""
    _record_v1(ledger)
    _record_v2(ledger)
    again = LedgerStore(ledger.db_path)
    assert again.row_counts()["decisions"] == 2
    out = again.list_pending(caller="c1", today="2026-07-20")
    assert out["pagination"]["total"] == 2
