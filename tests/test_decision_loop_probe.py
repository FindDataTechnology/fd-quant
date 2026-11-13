"""决策闭环探针（close-first-decision-loop tasks 1.1-1.5）。

**这个文件测的不是探针"能跑"，而是它"能红"。** 一条判据如果在坏状态上也是绿的，
它就只是装饰。每个正向断言都配一条破坏实验：把被测的那半改坏，断言对应判据变红。

判据清单（与 spec `decision-loop-closure` 逐条对位）：
- 存在性：`verdict ∈ {hit, miss}` 才算闭合，`insufficient_data` 不算
- 新鲜度：超期报「停摆」，不是「本期无决策」
- 可归因：pack / skill / record_kind 不能空
- 水位：**独立于**口径一致；拿不到语义库报 `unavailable` 且不计为通过
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "decision_loop_probe.py"


def _load_probe():
    return _load_module(PROBE, "decision_loop_probe")


def _load_module(path: Path, name: str):
    """按路径加载脚本模块。

    必须先塞进 `sys.modules`：`@dataclass` 需要回查 `sys.modules[cls.__module__]`，
    否则在 3.12 上报 `'NoneType' object has no attribute '__dict__'`——
    这条不是本仓的 bug，是"用 importlib 加载带 dataclass 的脚本"的固定姿势。
    """
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


# ── 夹具：直接建库（不经过 LedgerStore，才能造出"坏状态"）────────────


def make_ledger(path: Path, rows: list[dict], *, settlements: list[dict]) -> Path:
    """按最小 schema 造一个账本：只有探针读的列，够它判定。"""
    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY,
            caller TEXT NOT NULL,
            record_kind TEXT,
            as_of TEXT NOT NULL,
            symbol TEXT,
            pack TEXT,
            skill TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE settlements (
            decision_id TEXT PRIMARY KEY,
            settled_at TEXT NOT NULL,
            settlement_date TEXT NOT NULL,
            rule_id TEXT NOT NULL,
            verdict TEXT NOT NULL,
            gap_kinds TEXT,
            reflection TEXT
        );
        """
    )
    for r in rows:
        con.execute(
            "INSERT INTO decisions (decision_id, caller, record_kind, as_of, symbol, pack,"
            " skill, status, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (r["decision_id"], r.get("caller", "alice"), r.get("record_kind", "single"),
             r["as_of"], r.get("symbol"), r.get("pack", "wire-decision-review"),
             r.get("skill", "decision-review-log"), r.get("status", "settled"),
             r["created_at"]),
        )
    for s in settlements:
        con.execute(
            "INSERT INTO settlements (decision_id, settled_at, settlement_date, rule_id,"
            " verdict, gap_kinds, reflection) VALUES (?,?,?,?,?,?,?)",
            (s["decision_id"], s.get("settled_at", "2026-11-10T00:00:00+00:00"),
             s.get("settlement_date", "2026-11-10"), s.get("rule_id", "metric_threshold"),
             s["verdict"], s.get("gap_kinds"), s.get("reflection")),
        )
    con.commit()
    con.close()
    return path


def run_probe(*args: str) -> tuple[int, dict]:
    """跑探针并解析 JSON；返回 (退出码, 结果字典)。"""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = probe.run(list(args) + ["--json"])
    payload = json.loads(buf.getvalue())
    # 探针的 run() 会 sys.exit? 不会：它返回退出码，由 __main__ 转成 sys.exit。
    return (rc if isinstance(rc, int) else 0), payload


def codes(payload: dict) -> set[str]:
    return {v["code"] for v in payload.get("violations", [])}


# ── 判据 1：存在性 ───────────────────────────────────────────


def test_empty_ledger_is_not_started_not_failure(tmp_path):
    """生产账本 0 行的真实形态（2026-11-13 实测）。

    它必须是 `not_started`（退出码 3），**不是**失败：这条线还没有基线，
    "停摆"与"未闭合"都无从谈起。把它判成失败，闸就常年红，最后没人再看。
    """
    db = make_ledger(tmp_path / "empty.db", [], settlements=[])
    rc, out = run_probe("--db", str(db), "--as-of", "2026-11-13")
    assert rc == 3, out
    assert out["state"] == "not_started"
    assert out["not_started"] is True
    assert codes(out) == set(), "没开始不是违规——不许伪造违规码"
    assert out["checked"]["existence"]["state"] == "not_started"
    # 三判据都标"未判定"，不许出现空集合上的"通过"（那是空真）
    assert out["checked"]["freshness"]["judged"] is False
    assert out["checked"]["water_level"]["judged"] is False
    assert out["checked"]["water_level"]["state"] == "not_applicable"


def test_strict_turns_not_started_into_a_failure(tmp_path):
    """需要把"没开始"当失败用的场合（本应已有真实使用的环境）：--strict。"""
    db = make_ledger(tmp_path / "empty.db", [], settlements=[])
    rc, out = run_probe("--db", str(db), "--as-of", "2026-11-13", "--strict")
    assert rc == 1, out
    assert out["state"] == "violation"
    assert "NOT_STARTED" in codes(out)


def test_recorded_but_never_settled_is_a_violation_not_not_started(tmp_path):
    """边界：**有落账但一条结算都没有** → 是违规（入口在跑，结算没跑），不是"没开始"。

    这一条保证 `not_started` 只在 `decisions == 0` 时成立；否则"落账了但永远不结算"
    会被伪装成"还没开始"，而这两者的处置方向相反（一个查调度入口，一个查结算口径）。
    """
    db = make_ledger(
        tmp_path / "nosettle.db",
        [{"decision_id": "d1", "as_of": "2026-11-10", "symbol": "600519",
          "created_at": "2026-11-10T00:00:00+00:00"}],
        settlements=[],
    )
    rc, out = run_probe("--db", str(db), "--as-of", "2026-11-13")
    assert rc == 1, out
    assert out["state"] == "violation"
    assert out["not_started"] is False
    assert "LOOP_NEVER_CLOSED" in codes(out)


def test_only_insufficient_does_not_count_as_closed(tmp_path):
    """**破坏实验的核心**：有结算行但结论都是"没定论"，不许判为闭合。"""
    db = make_ledger(
        tmp_path / "ins.db",
        [{"decision_id": "d1", "as_of": "2026-11-10", "symbol": "600519",
          "created_at": "2026-11-10T00:00:00+00:00"}],
        settlements=[{"decision_id": "d1", "verdict": "insufficient_data",
                      "gap_kinds": "benchmark_series"}],
    )
    rc, out = run_probe("--db", str(db), "--as-of", "2026-11-13")
    assert rc == 1
    assert "LOOP_NEVER_CLOSED" in codes(out)
    detail = next(v["detail"] for v in out["violations"] if v["code"] == "LOOP_NEVER_CLOSED")
    assert "insufficient_data" in detail, "必须点名有几条没定论，而不是只报 0"


def test_hit_and_miss_both_are_closed(tmp_path):
    db = make_ledger(
        tmp_path / "ok.db",
        [
            {"decision_id": "d1", "as_of": "2026-11-10", "symbol": "600519",
             "created_at": "2026-11-10T00:00:00+00:00"},
            {"decision_id": "d2", "as_of": "2026-11-11", "symbol": "000333",
             "created_at": "2026-11-11T00:00:00+00:00"},
        ],
        settlements=[
            {"decision_id": "d1", "verdict": "hit"},
            {"decision_id": "d2", "verdict": "miss"},
        ],
    )
    rc, out = run_probe("--db", str(db), "--as-of", "2026-11-13")
    assert rc == 0, out
    assert out["checked"]["existence"]["decided"] == 2


def test_existence_goes_red_when_only_insufficient_but_code_changed(tmp_path):
    """破坏实验：把"有结论"判定放宽成"有结算行"，必须红。

    直接改源码副本（不动原文件），确认这条断言真的绑在 verdict 上，
    而不是绑在"文件存在"上。
    """
    src = PROBE.read_text(encoding="utf-8")
    broken = src.replace(
        'DECIDED_VERDICTS = ("hit", "miss")',
        'DECIDED_VERDICTS = ("hit", "miss", "insufficient_data")',
    ).replace(
        "WHERE s.verdict IN ('hit','miss')",
        "WHERE s.verdict IN ('hit','miss','insufficient_data')",
    )
    assert broken != src, "破坏实验没改到任何东西：断言已失效"
    tmp_src = tmp_path / "probe_broken.py"
    tmp_src.write_text(broken, encoding="utf-8")
    mod = _load_module(tmp_src, "probe_broken")

    db = make_ledger(
        tmp_path / "ins2.db",
        [{"decision_id": "d1", "as_of": "2026-11-10", "symbol": "600519",
          "created_at": "2026-11-10T00:00:00+00:00"}],
        settlements=[{"decision_id": "d1", "verdict": "insufficient_data",
                      "gap_kinds": "benchmark_series"}],
    )
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.run(["--db", str(db), "--as-of", "2026-11-13", "--json"])
    out = json.loads(buf.getvalue())
    assert rc == 0 and "LOOP_NEVER_CLOSED" not in codes(out), (
        "放宽后的探针应当把 insufficient_data 当成闭合 —— 这说明正向用例绑在了 verdict 上"
    )


# ── 判据 2：新鲜度 ───────────────────────────────────────────


def test_stale_ledger_says_stalled_not_quiet(tmp_path):
    db = make_ledger(
        tmp_path / "stale.db",
        [{"decision_id": "d1", "as_of": "2026-09-01", "symbol": "600519",
          "created_at": "2026-09-01T00:00:00+00:00"}],
        settlements=[{"decision_id": "d1", "verdict": "hit"}],
    )
    rc, out = run_probe("--db", str(db), "--as-of", "2026-11-13", "--days", "14")
    assert rc == 1
    assert "LOOP_STALLED" in codes(out)
    detail = next(v["detail"] for v in out["violations"] if v["code"] == "LOOP_STALLED")
    assert "停摆" in detail and "2026-09-01" in detail


def test_fresh_ledger_passes_freshness(tmp_path):
    db = make_ledger(
        tmp_path / "fresh.db",
        [{"decision_id": "d1", "as_of": "2026-11-12", "symbol": "600519",
          "created_at": "2026-11-12T00:00:00+00:00"}],
        settlements=[{"decision_id": "d1", "verdict": "hit"}],
    )
    rc, out = run_probe("--db", str(db), "--as-of", "2026-11-13", "--days", "14")
    assert rc == 0, out
    assert out["checked"]["freshness"]["ok"] is True


# ── 判据 3：可归因 ───────────────────────────────────────────


def test_unattributable_settlement_is_red(tmp_path):
    db = make_ledger(
        tmp_path / "noattr.db",
        [{"decision_id": "d1", "as_of": "2026-11-12", "symbol": "600519",
          "pack": "", "skill": "decision-review-log",
          "created_at": "2026-11-12T00:00:00+00:00"}],
        settlements=[{"decision_id": "d1", "verdict": "hit"}],
    )
    rc, out = run_probe("--db", str(db), "--as-of", "2026-11-13")
    assert rc == 1
    assert "SETTLEMENT_UNATTRIBUTABLE" in codes(out)


# ── 判据 4：水位（独立于口径一致）────────────────────────────


def test_water_level_unavailable_is_not_a_pass(tmp_path, monkeypatch):
    """拿不到语义库时报 `unavailable`，且**不计为通过**。"""
    monkeypatch.delenv("FD_OPEN_DATA_DSN", raising=False)
    monkeypatch.delenv("FD_OPEN_DATA_MCP_DATABASE_URL", raising=False)
    db = make_ledger(
        tmp_path / "nod.db",
        [{"decision_id": "d1", "as_of": "2026-11-12", "symbol": "600519",
          "created_at": "2026-11-12T00:00:00+00:00"}],
        settlements=[{"decision_id": "d1", "verdict": "hit"}],
    )
    rc, out = run_probe("--db", str(db), "--as-of", "2026-11-13")
    wl = out["checked"]["water_level"]
    assert wl["state"] == "unavailable"
    assert wl["judged"] is False
    # 未判定不产生违规（探针要能在只有账本的机器上跑），但也不许写成 ok
    assert wl.get("ok") is not True


def test_water_level_stale_while_sources_agree_is_red(tmp_path, monkeypatch):
    """**规格的核心场景**：三路一致地指向同一个过旧来源 → 水位红。

    口径判据（price_caliber_gate 那一条）会绿，因为它只问"三路是否同源"。
    本判据必须独立地变红，否则"一致地错"永远查不出来。
    """
    db = make_ledger(
        tmp_path / "stale_price.db",
        [{"decision_id": "d1", "as_of": "2026-11-12", "symbol": "600519.SS",
          "created_at": "2026-11-12T00:00:00+00:00"}],
        settlements=[{"decision_id": "d1", "verdict": "hit"}],
    )
    monkeypatch.setenv("FD_OPEN_DATA_DSN", "postgresql://fake/db")
    monkeypatch.setattr(
        probe, "read_water_level",
        lambda symbols: {
            "state": "ok",
            "reason": "",
            "symbols": {"600519": {"price_as_of_max": "2026-09-18",
                                   "last_fetch_at": "2026-09-19T00:00:00"}},
        },
    )
    rc, out = run_probe("--db", str(db), "--as-of", "2026-11-13",
                        "--max-stale-days", "7")
    assert rc == 1
    assert "PRICE_WATERMARK_STALE" in codes(out)
    wl = out["checked"]["water_level"]
    assert wl["judged"] is True and wl["ok"] is False
    assert wl["stale"][0][0] == "600519" and wl["stale"][0][2] > 7


def test_supply_never_fetched_is_its_own_finding(tmp_path, monkeypatch):
    """供给没产出 ≠ 水位过旧：两者的修法完全不同，必须分开点名。"""
    db = make_ledger(
        tmp_path / "nofetch.db",
        [{"decision_id": "d1", "as_of": "2026-11-12", "symbol": "000333.SZ",
          "created_at": "2026-11-12T00:00:00+00:00"}],
        settlements=[{"decision_id": "d1", "verdict": "hit"}],
    )
    monkeypatch.setenv("FD_OPEN_DATA_DSN", "postgresql://fake/db")
    monkeypatch.setattr(
        probe, "read_water_level",
        lambda symbols: {
            "state": "ok", "reason": "",
            "symbols": {"000333": {"price_as_of_max": None, "last_fetch_at": None}},
        },
    )
    rc, out = run_probe("--db", str(db), "--as-of", "2026-11-13")
    assert rc == 1
    found = codes(out)
    assert "SUPPLY_NOT_PRODUCING" in found
    assert "PRICE_NEVER_OBSERVED" in found


def test_symbol_normalization_maps_suffixed_codes():
    assert probe.normalize_symbol("600519.SS") == "600519"
    assert probe.normalize_symbol("000333.SZ") == "000333"
    assert probe.normalize_symbol("600519") == "600519"
    assert probe.normalize_symbol("0700.HK") == "0700"


# ── 参数与致命路径 ───────────────────────────────────────────


def test_missing_db_is_exit_2(tmp_path):
    rc, out = run_probe("--db", str(tmp_path / "nope.db"))
    assert rc == 2
    assert "账本不存在" in out["fatal"]


def test_unknown_mode_rejected(tmp_path):
    db = make_ledger(tmp_path / "x.db", [], settlements=[])
    with pytest.raises(SystemExit):
        probe.run(["--db", str(db), "--mode", "nonsense"])


def test_as_of_parsing_is_deterministic(tmp_path):
    """`--as-of` 让判定可复现：不依赖运行当天。"""
    db = make_ledger(
        tmp_path / "det.db",
        [{"decision_id": "d1", "as_of": "2026-11-01", "symbol": "600519",
          "created_at": "2026-11-01T00:00:00+00:00"}],
        settlements=[{"decision_id": "d1", "verdict": "hit"}],
    )
    _, a = run_probe("--db", str(db), "--as-of", "2026-11-05", "--days", "10")
    _, b = run_probe("--db", str(db), "--as-of", "2026-11-05", "--days", "10")
    assert a == b
    _, late = run_probe("--db", str(db), "--as-of", "2026-12-31", "--days", "10")
    assert "LOOP_STALLED" in codes(late), "同一账本换个判定日必须能变红"


def test_old_ledger_without_gap_column_still_runs(tmp_path):
    """**探针必须能读旧账本**：它最该能跑的时刻正是"迁移还没上/刚回滚"的时候。

    旧表没有 `gap_kinds` 列时，缺口统计跳过并显式标注，
    不得因此整体退出——那会把"读不了"伪装成"跑不起来"（退出码 2）。
    """
    db = tmp_path / "old_schema.db"
    con = sqlite3.connect(str(db))
    con.executescript(
        """
        CREATE TABLE decisions (
            decision_id TEXT PRIMARY KEY, caller TEXT NOT NULL, record_kind TEXT,
            as_of TEXT NOT NULL, symbol TEXT, pack TEXT, skill TEXT,
            status TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE settlements (
            decision_id TEXT PRIMARY KEY, settled_at TEXT NOT NULL,
            settlement_date TEXT NOT NULL, rule_id TEXT NOT NULL,
            verdict TEXT NOT NULL, reflection TEXT
        );
        INSERT INTO decisions (decision_id, caller, record_kind, as_of, symbol, pack,
            skill, status, created_at)
        VALUES ('d1','alice','single','2026-11-12','600519','p','s','settled',
                '2026-11-12T00:00:00+00:00');
        INSERT INTO settlements (decision_id, settled_at, settlement_date, rule_id,
            verdict) VALUES ('d1','2026-11-12T00:00:00+00:00','2026-11-12',
                'metric_threshold','hit');
        """
    )
    con.commit()
    con.close()

    rc, out = run_probe("--db", str(db), "--as-of", "2026-11-13")
    assert rc == 0, out
    assert out["checked"]["ledger_schema"]["gap_column_present"] is False
    assert out["checked"]["existence"]["decided"] == 1


def test_date_import_used_for_water_judgement():
    """水位判据用的是 `date`（日粒度），不是 datetime——避免边界漂移。"""
    assert isinstance(probe.dt.date.today(), date)


# ── 供给归因：三层判定（含 2026-11-13 我自己踩过的误判）──────────────


@pytest.mark.parametrize(
    "attempts,observations,latest,policies,runs,open_age,expected",
    [
        # **我踩过的坑**：批量脚本路径不写 fetch_log ⇒ attempts=0，
        # 但策略存在且每天在跑（只是失败）。不许判成 no_policy。
        (0, 120182, "2026-09-18",
         [{"id": 2659, "enabled": True}], [{"status": "failed"}], None,
         "dispatch_failing"),
        # 有启用策略、无失败记录、但有超期 open run ⇒ 卡死挡派发
        (0, 120182, "2026-09-18",
         [{"id": 2659, "enabled": True}], [], 13099, "dispatch_blocked"),
        # 策略成功但没有新观测 ⇒ 抓到了没写进来
        (0, 120182, "2026-09-18",
         [{"id": 2659, "enabled": True}], [{"status": "success"}], None,
         "zero_yield"),
        # 真·没有策略也没有抓取 ⇒ 才是 no_policy
        (0, 0, None, [], [], None, "no_policy"),
        # 无策略但有抓取尝试 ⇒ 走适配器路径的判定
        (7, 0, None, [], [], None, "fetch_failing"),
        # 新鲜观测 ⇒ producing
        (0, 500, "2026-11-12",
         [{"id": 5, "enabled": True}], [{"status": "success"}], None, "producing"),
        # 卡死优先级高于失败：两者同时成立时报 dispatch_blocked（根因更靠前）
        (0, 0, None, [{"id": 2659, "enabled": True}],
         [{"status": "failed"}], 99999, "dispatch_blocked"),
        # 禁用策略不算"有策略"
        (0, 0, None, [{"id": 2659, "enabled": False}], [], None, "no_policy"),
    ],
)
def test_classify_supply(attempts, observations, latest, policies, runs, open_age, expected):
    got = probe.classify_supply(
        attempts=attempts, observations=observations, latest_observation=latest,
        policies=policies, runs=runs, open_run_age_seconds=open_age,
        now=date(2026, 11, 13), max_stale_days=7,
    )
    assert got == expected


def test_classify_supply_never_says_no_policy_when_policy_exists():
    """回归：这条断言就是"我判错过一次"的化石。

    attempts=0 绝不等于"没派发"——批量路径不写 fetch_log。只要存在启用策略，
    归因就不能是 no_policy。
    """
    out = probe.classify_supply(
        attempts=0, observations=120182, latest_observation="2026-09-18",
        policies=[{"id": 2659, "enabled": True}],
        runs=[{"status": "failed"}, {"status": "failed"}],
        open_run_age_seconds=None, now=date(2026, 11, 13), max_stale_days=7,
    )
    assert out != "no_policy"


def test_attribution_readings_cover_every_enum_branch():
    """每个可能返回的归因都必须有可读解释——否则输出里会出现空白的"原因"。"""
    for kind in ("producing", "dispatch_blocked", "dispatch_failing",
                 "zero_yield", "fetch_failing", "no_policy"):
        assert probe.ATTRIBUTION_READINGS.get(kind)


# ── 卡死阈值派生（unblock-silent-dispatch-stalls task 4.1）──────────────


def test_stuck_threshold_derives_from_executor_deadline(monkeypatch):
    """阈值必须**派生**自执行侧那个 env，不另写一个数字。

    两份阈值一旦漂移：①判定比执行侧紧 → Job 还活着、行已判卡死 ⇒ 双跑；
    ②判定比执行侧松（或不判）→ Job 早死了、行永不关 ⇒ 就是那条策略永久停更的缺陷。
    """
    monkeypatch.delenv(probe.JOB_DEADLINE_ENV, raising=False)
    assert probe.open_run_stuck_seconds() == (
        probe.DEFAULT_JOB_DEADLINE_SECONDS + probe.RECONCILER_TICK_SECONDS
    )
    monkeypatch.setenv(probe.JOB_DEADLINE_ENV, "600")
    assert probe.open_run_stuck_seconds() == 600 + probe.RECONCILER_TICK_SECONDS


def test_stuck_threshold_survives_garbage_env(monkeypatch):
    """env 写坏不许把判据变成 0（那会让每一条 open run 立刻被判卡死）。"""
    for bad in ("abc", "0", "-5", ""):
        monkeypatch.setenv(probe.JOB_DEADLINE_ENV, bad)
        assert probe.open_run_stuck_seconds() == (
            probe.DEFAULT_JOB_DEADLINE_SECONDS + probe.RECONCILER_TICK_SECONDS
        ), f"env={bad!r} 时阈值必须回落到默认"


def test_classify_uses_the_derived_threshold(monkeypatch):
    """classify_supply 必须走派生函数，而不是某个常量快照。

    把执行侧上限调到很小 → 同一个 open run 年龄应从"未超龄"翻成 dispatch_blocked。
    """
    kwargs = dict(
        attempts=0, observations=0, latest_observation=None,
        policies=[{"id": 2659, "enabled": True}], runs=[],
        open_run_age_seconds=1200, now=date(2026, 11, 13), max_stale_days=7,
    )
    monkeypatch.delenv(probe.JOB_DEADLINE_ENV, raising=False)
    assert probe.classify_supply(**kwargs) != "dispatch_blocked", "20min 远未超龄"
    monkeypatch.setenv(probe.JOB_DEADLINE_ENV, "60")
    assert probe.classify_supply(**kwargs) == "dispatch_blocked", "上限 60s+900s 后 20min 已超龄"
