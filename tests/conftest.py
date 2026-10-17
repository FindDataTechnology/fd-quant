"""pytest 路径接缝：仓库根（`import fd_quant`）与 tests 目录（`import quantdata`）。

两处都在这里加一次，测试文件里就不再各写 sys.path 体操——`conftest.py` 是
pytest 唯一保证先于测试收集导入的模块。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
