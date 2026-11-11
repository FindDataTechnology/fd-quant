"""执行岗"下单前强制过闸"的链路进 CI（change3 task 3.2 的机制链，不是真跑）。

为什么纳入 pytest：`scripts/verify_executor_gate_chain.py` 是一条**跨模块**的链路
（风险门 → 订单意图导出 → 舰队种子），只在本地手跑的话，任何一处改动都能让它
静默失效。这里跑脚本自己的三段（不是重写一遍判据），所以测试与报告是同一份事实。

**它证明的是机制链**：缺阈值时门会拒、该判定下意图导不出去、执岗稿点名了这条条件。
**不证明**真实回合零订单 —— 那需要舰队实化 + 模型跑一轮，属编成侧（见
`openspec/changes/add-investment-trading-fleet` 的 §6 卡点表）。

写法与 `tests/test_ledger_e2e.py` 一致：按路径加载（`scripts/` 不是包），再跑 CLI 一次。
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "verify_executor_gate_chain.py"

#: 舰队种子住在**工作区**的另一个仓（`paas/`），不在本仓里。
#: 单仓 CI 检出的是 `fd-quant/` 本身，`REPO.parent` 下没有 `paas/`——所以第 ③ 段
#: （执岗稿是否把"阈值未配置"列为不行动条件）在单仓 CI 里**读不到对象**。
#:
#: 处理原则：**不假装跑过**。种子在 → 正常跑；种子不在 → `skip` 并写明该在哪跑。
#: 另有一条测试钉住"种子读不到时脚本 exit 2 且点名路径"——那才是这条链在缺少对象时
#: 应当有的行为（"跑不完整"绝不等于"没有违规"）。
SEED = REPO.parent / "paas/docs/fleet-templates/invest-trading-fleet.seed.json"
_NEEDS_SEED = pytest.mark.skipif(
    not SEED.exists(),
    reason=(f"舰队种子不在本仓检出范围内（{SEED}）；第③段属工作区级检查，"
            "在 finddata 根跑 `python fd-quant/scripts/verify_executor_gate_chain.py`"),
)


def _load_script():
    """按路径加载校验脚本（不是包，故不能直接 import）。"""
    spec = importlib.util.spec_from_file_location("verify_executor_gate_chain", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


# --------------------------------------------------- 三段各自可独立失败

def test_gate_refuses_when_a_threshold_is_missing(script):
    """缺阈值 → unavailable 且**点名缺项**；四项齐全未触达时必须是 ok。"""
    assert script._gate_refuses_when_a_threshold_is_missing() == []


def test_order_cannot_be_exported_without_a_usable_verdict(script):
    """unavailable 判定下订单意图导不出去 —— "没配阈值还下单"在契约层做不到。"""
    assert script._order_cannot_be_exported_without_a_usable_verdict() == []


@_NEEDS_SEED
def test_executor_prompt_lists_thresholds_unset_as_a_no_go(script):
    """执岗 persona 与节奏 prompt 把"阈值未配置"列为不行动条件。"""
    assert script._executor_prompt_lists_thresholds_unset_as_a_no_go(
        script.DEFAULT_SEED) == []


# --------------------------------------------------- 端到端与边界

@_NEEDS_SEED
def test_chain_passes_in_the_workspace():
    """在工作区里跑一次完整链路：exit 0，且三段全绿。"""
    proc = subprocess.run([sys.executable, str(SCRIPT)],
                          cwd=str(REPO), env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "三段全绿" in proc.stdout


def test_missing_seed_exits_2_not_0(script, tmp_path):
    """种子读不到时是"跑不完整"（exit 2），**不是**"没有问题"。"""
    assert script.main(["--seed", str(tmp_path / "nope.json")]) == 2
