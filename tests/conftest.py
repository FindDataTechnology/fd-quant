"""pytest 路径接缝：仓库根（`import fd_quant`）与 tests 目录（`import quantdata`）。

两处都在这里加一次，测试文件里就不再各写 sys.path 体操——`conftest.py` 是
pytest 唯一保证先于测试收集导入的模块。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── 账本落点隔离（seal-ledger-path-and-caller-isolation 2.1）─────────────────
#
# 分库后账本落 `<root>/<slug>-<sha8>/ledger.db`；`root` 未配时回退**当前工作目录**，
# 于是「在仓里跑 pytest」会让所有未显式设根的用例共享 `<repo>/anonymous-…/ledger.db`，
# 一条用例写进去的行被下一条读到——失败顺序还取决于收集顺序，看起来像随机。
#
# 治法是让每条用例有自己的空根，而不是逐条打补丁：给 `FD_QUANT_LEDGER_ROOT` 一个
# 每用例专属的临时目录（autouse ⇒ 忘记设的用例也被兜住）。显式设了该变量的用例
# 照常覆盖它（monkeypatch.setenv 在后，仍生效）。
import tempfile  # noqa: E402

_ROOT_KEY = "FD_QUANT_LEDGER_ROOT"


@pytest.fixture(autouse=True)
def _isolated_ledger_root(monkeypatch):
    root = tempfile.mkdtemp(prefix="fdq-root-")
    monkeypatch.setenv(_ROOT_KEY, root)
    yield root
