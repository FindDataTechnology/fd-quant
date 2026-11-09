"""账本组合记录（add-portfolio-intent-contract tasks 3.1-3.5）。

三件必须成立的事：
1. 既有单体账本**逐条等价**（旧幂等键在新索引下不变）；
2. 两类记录共存但主体字段互斥，冲突要给出可读错误；
3. 聚合不混算——组合 alpha 与单票 alpha 进同一个胜率，那个胜率就没有主语。
"""
from __future__ import annotations

import sqlite3

import pytest

from fd_quant.ledger.store import LedgerStore
from fd_quant.portfolio import build_intent

RATING = {
    "schema_version": "1", "as_of": "2026-06-30", "subject": "600519", "side": "buy",
    "basis": "test",
    "evidence": [{"evidence_id": "e1", "label": "l", "value": 1, "unit": "u",
                  "as_of": "2026-06-30", "source": "s"}],
    "producer": {"pack": "p", "skill": "k"},
}


@pytest.fixture
def store(tmp_path):
    return LedgerStore(tmp_path / "ledger.db")


def _intent(**rules):
    r = {"kind": "equal_weight", "cash_floor": 0.2}
    r.update(rules)
    return build_intent(
        [{"as_of": "2026-06-30", "subject": "600519", "side": "buy", "conviction": 0.8}],
        r, as_of="2026-06-30", producer={"pack": "p", "skill": "k"},
    )


# ------------------------------------------------- 3.1 列与迁移


def test_columns_exist_on_a_fresh_db(store):
    con = sqlite3.connect(store.db_path)
    cols = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
    con.close()
    assert {"record_kind", "portfolio_key"} <= cols


def test_existing_single_rows_survive_and_default_to_single(tmp_path):
    """先按旧 schema 造一行，再用新代码打开：行不能丢，且判为 single。"""
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY, caller TEXT NOT NULL DEFAULT 'anonymous',
            as_of TEXT NOT NULL, symbol TEXT NOT NULL, pack TEXT NOT NULL, skill TEXT NOT NULL,
            payload TEXT NOT NULL, target_weights TEXT, evidence TEXT NOT NULL,
            status TEXT NOT NULL, revises TEXT, revision_seq INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    con.execute(
        "INSERT INTO decisions (decision_id, caller, as_of, symbol, pack, skill, payload,"
        " evidence, status, revision_seq, created_at)"
        " VALUES ('d-old','c1','2026-06-30','600519','p','k','{}','[]','pending',0,'2026-06-30')"
    )
    con.commit()
    con.close()

    st = LedgerStore(db)
    con = sqlite3.connect(db)
    row = con.execute("SELECT record_kind, portfolio_key, symbol FROM decisions").fetchone()
    con.close()
    assert row == ("single", None, "600519")
    assert st.row_counts()["decisions"] == 1


def test_migration_is_idempotent(tmp_path):
    db = tmp_path / "m.db"
    LedgerStore(db).row_counts()
    LedgerStore(db).row_counts()  # 第二次打开不应报错或重复搬行
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    con.close()


# ------------------------------------------------- 3.2 幂等键


def test_portfolio_idempotency_key_uses_portfolio_key(store):
    a = store.record(caller="c1", as_of="2026-06-30", portfolio_key="p-1", pack="p",
                     skill="k", payload=_intent(), record_kind="portfolio")
    b = store.record(caller="c1", as_of="2026-06-30", portfolio_key="p-1", pack="p",
                     skill="k", payload=_intent(), record_kind="portfolio")
    assert a["created"] is True
    assert b["created"] is False and b["reason"] == "duplicate_key"
    assert store.row_counts()["decisions"] == 1


def test_same_subject_text_in_two_kinds_are_distinct_records(store):
    """symbol 与 portfolio_key 撞成同一个字符串时，两类记录仍互不覆盖。"""
    s = store.record(caller="c1", as_of="2026-06-30", symbol="X", pack="p", skill="k",
                     payload={**RATING, "subject": "X"})
    p = store.record(caller="c1", as_of="2026-06-30", portfolio_key="X", pack="p", skill="k",
                     payload=_intent(), record_kind="portfolio")
    assert s["created"] and p["created"]
    assert s["decision_id"] != p["decision_id"]
    assert store.row_counts()["decisions"] == 2


def test_portfolio_and_single_coexist_for_same_as_of(store):
    store.record(caller="c1", as_of="2026-06-30", symbol="600519", pack="p", skill="k",
                 payload=RATING)
    store.record(caller="c1", as_of="2026-06-30", portfolio_key="p-1", pack="p", skill="k",
                 payload=_intent(), record_kind="portfolio")
    assert store.row_counts()["decisions"] == 2


# ------------------------------------------------- 3.3 主体一致性


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        (dict(record_kind="portfolio", portfolio_key="p-1", symbol="600519"), "must not carry symbol"),
        (dict(record_kind="portfolio", portfolio_key=None), "requires portfolio_key"),
        (dict(record_kind="single", symbol=None), "requires symbol"),
        (dict(record_kind="single", symbol="600519", portfolio_key="p-1"), "must not carry portfolio_key"),
        (dict(record_kind="weird", symbol="600519"), "must be 'single' or 'portfolio'"),
    ],
)
def test_subject_field_conflicts_are_refused_with_readable_errors(store, kwargs, needle):
    from fd_quant.ledger.errors import LedgerError

    payload = _intent() if kwargs.get("record_kind") == "portfolio" else RATING
    with pytest.raises(LedgerError, match=needle):
        store.record(caller="c1", as_of="2026-06-30", pack="p", skill="k",
                     payload=payload, **kwargs)


def test_db_check_constraint_backs_the_python_check(store):
    """绕过 Python 直接插库也要被拒——约束是最后一道，不是唯一一道。"""
    con = sqlite3.connect(store.db_path)
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO decisions (decision_id, caller, record_kind, as_of, symbol,"
            " portfolio_key, pack, skill, payload, evidence, status, revision_seq, created_at)"
            " VALUES ('x','c1','portfolio','2026-06-30','600519','p-1','p','k','{}','[]',"
            "'pending',0,'now')"
        )
    con.close()


# ------------------------------------------------- 3.5 payload


def test_portfolio_payload_must_be_an_intent_not_a_rating(store):
    from fd_quant.contracts.models import ContractError

    with pytest.raises(ContractError):
        store.record(caller="c1", as_of="2026-06-30", portfolio_key="p-1", pack="p",
                     skill="k", payload=RATING, record_kind="portfolio")


def test_single_payload_must_be_a_rating_not_an_intent(store):
    from fd_quant.contracts.models import ContractError

    with pytest.raises(ContractError):
        store.record(caller="c1", as_of="2026-06-30", symbol="600519", pack="p",
                     skill="k", payload=_intent())


def test_producer_must_match_the_record_key(store):
    from fd_quant.contracts.models import ContractError

    with pytest.raises(ContractError):
        store.record(caller="c1", as_of="2026-06-30", portfolio_key="p-1", pack="other",
                     skill="k", payload=_intent(), record_kind="portfolio")


def test_payload_roundtrips_through_the_ledger(store):
    intent = _intent()
    out = store.record(caller="c1", as_of="2026-06-30", portfolio_key="p-1", pack="p",
                       skill="k", payload=intent, record_kind="portfolio")
    got = store.get(out["decision_id"])
    assert got is not None
    assert got["payload"]["rules"]["kind"] == "equal_weight"
    assert got["payload"]["weights"] == intent.weights
