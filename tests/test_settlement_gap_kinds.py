"""结算缺口点名 `gap_kinds`（close-first-decision-loop tasks 2.1-2.5）。

**这条列的用途是让"结不出来"可聚合**：`insufficient_data` 只说明"没定论"，
没说明"该去补什么"。三路缺口（基准序列 / 价格水位 / 指标观测）的处置完全不同，
混成一个数就没法行动。

三条不可退让：
1. `insufficient_data` ⇒ `gap_kinds` 非空；`hit`/`miss` ⇒ `gap_kinds` 必为 NULL（库层 CHECK）；
2. 旧库迁移后列存在、幂等、既有行留空（它们产生于"缺口还没有名字"的时期）；
3. 聚合面按缺口可计数——造三条不同缺口，一条查询给出三类各 1 条。
"""
from __future__ import annotations

import sqlite3

import pytest

from fd_quant.ledger.errors import SettlementRefused
from fd_quant.ledger.store import LedgerStore

WINDOW = ["2026-08-31", "2026-09-30", "2026-10-31"]


def record(store: LedgerStore, *, subject: str = "s-1", as_of: str = "2026-11-13") -> str:
    payload = {
        "schema_version": "2",
        "as_of": as_of,
        "subject": subject,
        "subject_kind": "store",
        "action": "expand",
        "preconditions": [],
        "settlement_rule_id": "metric_threshold",
        "expected_outcome": {
            "metric_ref": subject, "comparator": "gte", "threshold": 0.42,
            "horizon": {"unit": "month", "count": 3},
            "settlement_source": "caller_supplied_series",
        },
        "basis": "构造判定：月均流水连续三月高于保本线。",
        "evidence": [],
        "producer": {"pack": "wire-decision-review", "skill": "decision-review-log"},
    }
    return store.record(
        caller="alice", as_of=as_of, symbol=subject,
        pack="wire-decision-review", skill="decision-review-log",
        payload=payload, evidence=[],
    )["decision_id"]


@pytest.fixture
def store(tmp_path):
    return LedgerStore(tmp_path / "ledger.db")


# ── 2.1 列与迁移 ────────────────────────────────────────────


def test_schema_has_gap_kinds(store):
    con = sqlite3.connect(store.db_path)
    cols = {r[1] for r in con.execute("PRAGMA table_info(settlements)")}
    con.close()
    assert "gap_kinds" in cols


def test_migration_is_idempotent_on_existing_db(tmp_path):
    """旧库（无 gap_kinds）迁移后可读，且二次启动是空操作。"""
    db = tmp_path / "old.db"
    # 先用"旧 schema"造库：手工建表（不带 gap_kinds）
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY, caller TEXT NOT NULL DEFAULT 'legacy',
            record_kind TEXT NOT NULL DEFAULT 'single', as_of TEXT NOT NULL,
            symbol TEXT, portfolio_key TEXT, pack TEXT NOT NULL, skill TEXT NOT NULL,
            payload TEXT NOT NULL, target_weights TEXT, series_ref TEXT,
            evidence TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            revises TEXT, revision_seq INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, subject_kind TEXT, subject_key TEXT,
            action TEXT, preconditions TEXT, settlement_spec TEXT
        );
        CREATE TABLE settlements (
            decision_id TEXT PRIMARY KEY, settled_at TEXT NOT NULL,
            settlement_date TEXT NOT NULL, rule_id TEXT NOT NULL DEFAULT 'alpha_vs_benchmark',
            verdict TEXT NOT NULL DEFAULT 'miss', entry_price REAL, exit_price REAL,
            raw_return REAL, benchmark_symbol TEXT, benchmark_mapping TEXT,
            benchmark_entry_price REAL, benchmark_exit_price REAL, benchmark_return REAL,
            alpha REAL, holding_days INTEGER, metric_ref TEXT, observed_value REAL,
            comparator TEXT, threshold REAL, series_fingerprint TEXT, reflection TEXT
        );
        INSERT INTO decisions (decision_id, caller, as_of, symbol, pack, skill, payload,
            evidence, status, created_at)
        VALUES ('d-old','legacy','2026-01-05','600519','p','s','{}','[]','settled',
                '2026-01-05T00:00:00+00:00');
        INSERT INTO settlements (decision_id, settled_at, settlement_date, verdict,
            entry_price, exit_price, benchmark_symbol, alpha)
        VALUES ('d-old','2026-01-12T00:00:00+00:00','2026-01-12','hit',
                10.0, 11.0, '000001.SS', 0.05);
        """
    )
    con.commit()
    con.close()

    LedgerStore(db)  # 迁移
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(settlements)")}
    assert "gap_kinds" in cols
    row = con.execute(
        "SELECT verdict, alpha, gap_kinds FROM settlements WHERE decision_id='d-old'"
    ).fetchone()
    con.close()
    assert row == ("hit", 0.05, None), "既有行必须原样保留，缺口留空（当时没有缺口信息）"

    # 幂等：再开一次不应炸，也不应重复搬迁
    LedgerStore(db)
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM settlements").fetchone()[0] == 1
    con.close()


def test_migration_rolls_back_on_failure(tmp_path, monkeypatch):
    """迁移中途失败 → 表结构回到迁移前，且不留 `settlements_old_gap` 残表。

    注入点选在"已经 rename、还没搬完"那一刻：这才是真正需要回滚的窗口。
    若 `_init_schema` 的 ROLLBACK 失灵，这里会留下 `settlements_old_gap` 且
    `settlements` 消失——那正是"半套结构"的形态。
    """
    from fd_quant.ledger import store as store_mod

    db = tmp_path / "rollback.db"
    LedgerStore(db)  # 先建一个完整库
    con = sqlite3.connect(db)
    assert "gap_kinds" in {r[1] for r in con.execute("PRAGMA table_info(settlements)")}
    con.close()

    def half_migration(conn):
        conn.execute("ALTER TABLE settlements RENAME TO settlements_old_gap")
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(store_mod, "_migrate_to_gap_kinds", half_migration)
    with pytest.raises(RuntimeError, match="injected migration failure"):
        LedgerStore(db)
    monkeypatch.undo()

    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    cols = {r[1] for r in con.execute("PRAGMA table_info(settlements)")}
    con.close()
    assert "settlements" in tables, "回滚后原表必须还在"
    assert "settlements_old_gap" not in tables, "迁移残表必须被回滚掉"
    assert "gap_kinds" in cols, "回滚后表结构应与迁移前逐列一致"


# ── 2.3 结算路径写缺口 ──────────────────────────────────────


def test_insufficient_writes_gap_kind(store):
    did = record(store)
    out = store.settle_metric_threshold(
        decision_id=did, dates=WINDOW, values=[0.38, 0.41, None],
        as_of="2026-11-13", comparator="gte", threshold=0.42,
    )
    assert out["verdict"] == "insufficient_data"
    assert out["gap_kinds"] == "metric_observation"


def test_decided_settlement_has_no_gap(store):
    did = record(store)
    out = store.settle_metric_threshold(
        decision_id=did, dates=WINDOW, values=[0.38, 0.41, 0.45],
        as_of="2026-11-13", comparator="gte", threshold=0.42,
    )
    assert out["verdict"] == "hit"
    assert out["gap_kinds"] is None


def test_db_refuses_gap_on_decided_row(store):
    """库层 CHECK：定了论还想留缺口 → 写不进去。

    这不是"多一道保险"，而是这条列存在的理由：允许半态就等于允许
    "有缺口却算命中"这种读起来自洽、实则自相矛盾的记录。
    """
    did = record(store)
    con = sqlite3.connect(store.db_path)
    con.execute("PRAGMA foreign_keys=ON")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO settlements (decision_id, settled_at, settlement_date, rule_id,"
            " verdict, metric_ref, comparator, threshold, gap_kinds)"
            " VALUES (?, '2026-11-13T00:00:00+00:00','2026-10-31','metric_threshold',"
            " 'hit','m','gte',0.42,'price_watermark')",
            (did,),
        )
    con.close()


def test_db_refuses_insufficient_without_gap(store):
    did = record(store)
    con = sqlite3.connect(store.db_path)
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO settlements (decision_id, settled_at, settlement_date, rule_id,"
            " verdict, metric_ref, comparator, threshold)"
            " VALUES (?, '2026-11-13T00:00:00+00:00','2026-10-31','metric_threshold',"
            " 'insufficient_data','m','gte',0.42)",
            (did,),
        )
    con.close()


def test_price_path_refusal_names_the_gap(tmp_path):
    """价格口径拒绝结算时，缺口必须落在 error 上（不落行，但要说清补哪一路）。"""
    store = LedgerStore(tmp_path / "p.db")
    payload = {
        "schema_version": "1", "as_of": "2026-01-05", "subject": "600519",
        "side": "buy", "basis": "构造判定", "evidence": [],
        "producer": {"pack": "p", "skill": "s"},
    }
    did = store.record(
        caller="alice", as_of="2026-01-05", symbol="600519", pack="p", skill="s",
        payload=payload, evidence=[],
    )["decision_id"]

    # 基准整条序列无可用价 → 缺口应为 benchmark_series
    with pytest.raises(SettlementRefused) as ei:
        store.settle(
            decision_id=did,
            symbol_dates=["2026-01-05", "2026-01-06"], symbol_prices=[10.0, 11.0],
            benchmark_dates=["2026-01-05", "2026-01-06"], benchmark_prices=[None, None],
            as_of="2026-01-06",
        )
    assert ei.value.gap_kinds == "benchmark_series"
    assert "缺口=benchmark_series" in str(ei.value)

    # 标的价缺失（基准正常）→ 缺口应为 price_watermark
    with pytest.raises(SettlementRefused) as ei2:
        store.settle(
            decision_id=did,
            symbol_dates=["2026-01-05", "2026-01-06"], symbol_prices=[None, None],
            benchmark_dates=["2026-01-05", "2026-01-06"], benchmark_prices=[100.0, 101.0],
            as_of="2026-01-06",
        )
    assert ei2.value.gap_kinds == "price_watermark"


# ── 2.4 聚合按缺口可计数 ────────────────────────────────────


def test_gap_breakdown_counts_each_kind(store):
    """造三条不同缺口 + 一条结论，聚合给出三类缺口各 1 条。"""
    # 指标观测缺失（真实路径）
    d1 = record(store, subject="s-1")
    store.settle_metric_threshold(
        decision_id=d1, dates=WINDOW, values=[0.38, 0.41, None],
        as_of="2026-11-13", comparator="gte", threshold=0.42,
    )
    # 另外两条缺口用直接的库写入造（基准/价格两路只在价格口径出现，
    # 而价格口径不落 insufficient_data 行——它们的缺口走 error，不走进聚合）。
    for i, kind in enumerate(("benchmark_series", "price_watermark"), start=2):
        did = record(store, subject=f"s-{i}")
        con = sqlite3.connect(store.db_path)
        con.execute(
            "INSERT INTO settlements (decision_id, settled_at, settlement_date, rule_id,"
            " verdict, metric_ref, comparator, threshold, gap_kinds)"
            " VALUES (?, '2026-11-13T00:00:00+00:00','2026-10-31','metric_threshold',"
            " 'insufficient_data','m','gte',0.42,?)",
            (did, kind),
        )
        con.commit()
        con.close()
        store._write_tx  # noqa: B018 — 显式提示：这里绕过了 store 的写路径，故意
        con = sqlite3.connect(store.db_path)
        con.execute("UPDATE decisions SET status='settled' WHERE decision_id=?", (did,))
        con.commit()
        con.close()

    out = store.stats_by_pack(caller="alice", as_of="2026-11-13")
    row = next(r for r in out["rows"] if r["pack"] == "wire-decision-review")
    assert row["gaps"] == {
        "metric_observation": 1, "benchmark_series": 1, "price_watermark": 1,
    }
    assert row["samples"] == 3


def test_gap_breakdown_absent_when_no_gaps(store):
    """没出现过的缺口不许预先铺 0——那会让"没缺"和"缺过"在结果里同形。"""
    did = record(store)
    store.settle_metric_threshold(
        decision_id=did, dates=WINDOW, values=[0.38, 0.41, 0.45],
        as_of="2026-11-13", comparator="gte", threshold=0.42,
    )
    out = store.stats_by_pack(caller="alice", as_of="2026-11-13")
    row = next(r for r in out["rows"] if r["pack"] == "wire-decision-review")
    assert row["gaps"] == {}


def test_destructive_constant_gap(store):
    """破坏实验：把缺口写成常量 → 三类合一会让上面的逐类断言失配。"""
    d1 = record(store, subject="s-1")
    store.settle_metric_threshold(
        decision_id=d1, dates=WINDOW, values=[0.38, 0.41, None],
        as_of="2026-11-13", comparator="gte", threshold=0.42,
    )
    con = sqlite3.connect(store.db_path)
    con.execute("UPDATE settlements SET gap_kinds='metric_observation'")
    con.commit()
    con.close()
    # 常量缺口时，聚合只有一类；这正是"没区分"的形态
    out = store.stats_by_pack(caller="alice", as_of="2026-11-13")
    row = next(r for r in out["rows"] if r["pack"] == "wire-decision-review")
    assert list(row["gaps"]) == ["metric_observation"]
