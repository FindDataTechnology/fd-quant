"""结算演练的基准必须是**指数**（2026-11-14 订正）。

事故形态：`settlement_e2e.py` 把基准表写成 `scraw_000001_daily`，而 000001 在 A 股同时
对应上证综指（`000001.SH`）与平安银行（`000001.SZ`），DAAS 那张表装的是后者
（实测 2026-11-06 收盘 11.81，综指在 4000 上下）。于是这条演练一直拿一只银行股当"大盘"：
算术自洽、`benchmark_symbol` 归因行还写着综指——**两条都不会报错**。

这组测试守的是那条护栏：判据取库内自述的口径（`scraw_configs.columns_json.source`），
不从表名推断，因为表名正是骗过所有人的东西。
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_script():
    name = "settlement_e2e_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/settlement_e2e.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e2e = _load_script()


def _db(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """造一个只有口径记录的库：`(code, url, source)`。"""
    p = tmp_path / "daas.db"
    con = sqlite3.connect(str(p))
    con.execute("CREATE TABLE scraw_configs (id INTEGER PRIMARY KEY, url TEXT, name TEXT, columns_json TEXT)")
    con.execute("CREATE TABLE scraw_%s_daily (date TEXT PRIMARY KEY, Close REAL)" % "000300")
    for i, (code, url, source) in enumerate(rows, 1):
        con.execute(
            "INSERT INTO scraw_configs (id,url,name,columns_json) VALUES (?,?,?,?)",
            (i, url, f"astock_{code}", json.dumps({"source": source})),
        )
    con.commit()
    con.close()
    return p


def test_index_table_is_accepted(tmp_path):
    db = _db(tmp_path, [("000300", "akshare:index_zh_a_hist", "index_zh_a_hist")])
    assert e2e.assert_index_table(str(db), "scraw_000300_daily") == "index_zh_a_hist"


def test_stock_table_is_refused_and_names_the_collision(tmp_path):
    """核心用例：000001 那张表是个股，必须拒绝，且理由要点到"两个 000001"这层。

    只报"来源不对"不够——下一个人看到表名还是 000001 就会以为是综指，再改回来一次。
    """
    db = _db(tmp_path, [("000001", "akshare:stock_zh_a_hist", "stock_zh_a_hist")])
    with pytest.raises(SystemExit) as exc:
        e2e.assert_index_table(str(db), "scraw_000001_daily")
    msg = str(exc.value)
    assert "stock_zh_a_hist" in msg
    assert "上证综指" in msg and "平安银行" in msg


def test_missing_caliber_record_is_refused(tmp_path):
    """没有口径记录 = 证明不了它是指数。fail-closed，不默认放行。"""
    db = _db(tmp_path, [])
    with pytest.raises(SystemExit) as exc:
        e2e.assert_index_table(str(db), "scraw_000300_daily")
    assert "无法证明它是指数" in str(exc.value)


def test_default_benchmark_is_an_index_code_not_the_colliding_one():
    """默认值本身就是判据：不许再退回 000001 这类在两地都存在的代码。"""
    assert e2e.BENCHMARK_TABLE == "scraw_000300_daily"
    assert e2e.BENCHMARK_TABLE != "scraw_000001_daily"
    assert e2e.BENCHMARK_SYMBOL.endswith(".SH"), "基准标的要显式带市场，别靠后缀映射推"


def test_benchmark_symbol_is_declared_explicitly_in_the_settlement():
    """基准标的必须显式传给账本，让记录标 `mapping=explicit`。

    映射推出来的基准与真正喂进去的序列是不是同一条，账本自己无从校验——
    这正是这次事故能静默通过的原因。
    """
    assert e2e.BENCHMARK_SYMBOL.startswith("000300")
