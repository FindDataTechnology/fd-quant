"""端到端演练的可执行断言（task 5.2）：脚本跑得通、离线、复跑逐位一致。

为什么把 `scripts/ledger_e2e.py` 也纳入测试：它是一条**跨模块**的链路，只靠人工运行
会在别处改动后静默失效（例如检索限额改了、结算拒绝文本改了）。这里跑的是脚本的
`run()` 本身（不是重写一遍链路），故它与报告是同一份事实。

离线证据来自 `test_ledger_offline.py`（同一份断言在封网下也跑）；本文件只钉行为。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPORT = (REPO.parent / "openspec" / "changes" / "fd-quant-decision-ledger"
          / "reports" / "ledger-e2e.md")


def _load_e2e():
    """按路径加载演练脚本（不是包，故不能直接 import）。"""
    spec = importlib.util.spec_from_file_location("ledger_e2e", REPO / "scripts" / "ledger_e2e.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "tests"))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def e2e():
    return _load_e2e()


@pytest.fixture(scope="module")
def steps(e2e, tmp_path_factory):
    db = tmp_path_factory.mktemp("e2e") / "ledger-e2e.db"
    return e2e.run(db)


def test_e2e_covers_every_required_step(steps):
    """五条决策（含一次修订）/ 幂等拒绝 / 提前结算被拒 / 全部结算 / 检索 / 聚合都在。"""
    assert steps["1_recorded_initial"]["count"] == 4
    assert steps["2_revision"]["revision_seq"] == 1
    assert steps["3_duplicate_refused"]["created"] is False
    assert steps["3_duplicate_refused"]["rows_unchanged"] is True
    assert steps["4_premature_refused"]["refused"] is True
    assert steps["4_premature_refused"]["missing_trading_days"] == 3
    assert steps["5_settlements"]["count"] == 5
    assert steps["6_settlement_idempotent"]["created"] is False
    assert len(steps["7_point_in_time"]["cuts"]) == 3
    assert steps["8_aggregate"]["by_pack"]
    assert all(c["valid"] for c in steps["9_contract_roundtrip"])


def test_e2e_revision_leaves_the_original_row_untouched(steps):
    """修订语义在整条链路上仍成立（原行 payload 与 created_at 不变）。"""
    assert steps["2_revision"]["original_payload_unchanged"] is True
    assert steps["2_revision"]["revision_row_counts"]["decisions"] == 5
    assert steps["2_revision"]["revises"] == steps["2_revision"]["original_decision_id"]


def test_e2e_point_in_time_discipline_holds_across_cutoffs(steps):
    """点内纪律：每个口径的可见结算日都不晚于该口径；未来结算的那条结构性缺席。

    被过滤的那条是**跨标的**（300750.SZ 相对 600519.SS），故在 `cross_symbol` 组里找它。
    """
    for cut in steps["7_point_in_time"]["cuts"]:
        assert all(d <= cut["as_of"] for d in cut["settlement_dates"]), cut
    hidden = steps["7_point_in_time"]["future_settlement_hidden"]
    full = steps["7_point_in_time"]["full_history"]
    assert hidden["decision_ids"], hidden
    late_visible = {x["decision_id"] for x in full["same_symbol"] + full["cross_symbol"]}
    assert set(hidden["decision_ids"]) <= late_visible, (hidden, late_visible)
    # 它出现在跨标的组（不是同标的组）——口径与校验都要诚实
    assert set(hidden["decision_ids"]) <= {x["decision_id"] for x in full["cross_symbol"]}
    assert not (set(hidden["decision_ids"]) & {x["decision_id"] for x in full["same_symbol"]})
    assert hidden["absent_from_cutoffs"] == ["2025-01-08", "2025-01-12"]


def test_e2e_aggregate_includes_an_explicit_zero_group(steps):
    """聚合里包含一个显式零样本组（空组返回零，不是省略）。"""
    zero_rows = [r for r in steps["8_aggregate"]["empty_group"] if r["samples"] == 0]
    assert zero_rows and all(r["mean_alpha"] is None for r in zero_rows)


def test_e2e_is_reproducible_bitwise(e2e, steps, tmp_path):
    """复跑逐位一致：注入确定性 id 生成器 + 显式时间戳后，指纹必须相同。

    这条是"报告里的数字是确定的"的机械保证：若哪天有 uuid4 或当前时刻渗进被比较的
    结果，指纹就会漂，测试先红而不是报告先烂。
    """
    again = e2e.run(tmp_path / "second.db")
    assert e2e._fingerprint(again) == e2e._fingerprint(steps)


def test_e2e_result_is_json_serialisable(steps):
    json.dumps(steps, allow_nan=False, ensure_ascii=False, sort_keys=True)


def test_report_exists_and_carries_the_fingerprint(e2e, steps):
    """报告落盘且带指纹（`--check` 的可核对性依赖它）。"""
    if not REPORT.exists():
        pytest.skip("report not generated in this checkout")
    text = REPORT.read_text(encoding="utf-8")
    assert "决策账本端到端演练" in text
    assert e2e._fingerprint(steps) in text
    assert "netblock" in text  # 复跑段说明了断网跑法
    assert "ledger_e2e.py --check" in text


def test_e2e_check_mode_verifies_the_committed_report():
    """`--check` 对已落盘报告回 0（报告与当前实现一致）。"""
    import subprocess

    if not REPORT.exists():
        pytest.skip("report not generated in this checkout")
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "ledger_e2e.py"), "--check"],
                          capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "OK" in proc.stdout


def test_e2e_script_uses_only_the_callers_series(  ):
    """账本不取数：演练脚本自带价格源（`_SeriesSource`），不 import 任何数据客户端。"""
    import ast

    src = (REPO / "scripts" / "ledger_e2e.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"requests", "httpx", "urllib", "socket", "aiohttp", "yfinance", "akshare",
              "tushare", "openai", "anthropic", "litellm", "daas"}
    assert not (imported & banned), sorted(imported & banned)
    assert "_SeriesSource" in src
