#!/usr/bin/env python3
"""把根目录的孤立旧账本搬到 `<root>/legacy/ledger.db`（任务 4.1 的可执行入口）。

用法：

    python scripts/migrate_ledger_to_legacy.py --root /data            # 真搬
    python scripts/migrate_ledger_to_legacy.py --root /data --dry-run  # 只看会做什么

**先停服务再跑**：迁移是移动文件，运行中的连接会指向被移走的 inode。

为什么要有这个脚本而不是只留 `store.migrate_stray_ledger()`：迁移是**部署动作**，
它必须能在不 import 整个工具面（fastmcp 等）的前提下跑起来——镜像里的迁移
常常发生在依赖装好之前。所以这里只 import 账本存储层。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许从仓根直接跑（`python scripts/xxx.py`）：把仓根放进 sys.path，
# 而不是要求调用方先 `pip install -e .`。同 `scripts/ledger_e2e.py` 的做法。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fd_quant.ledger.store import migrate_stray_ledger  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把孤立旧账本迁到 <root>/legacy/ledger.db")
    parser.add_argument("--root", default=None,
                        help="账本根目录；缺省按 FD_QUANT_LEDGER_ROOT → 旧变量父目录 → CWD")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要做什么，不动文件")
    args = parser.parse_args(argv)

    outcome = migrate_stray_ledger(args.root, dry_run=args.dry_run)
    # 结局打到 stdout：这是一个人手动跑的命令，不是 stdio 服务。
    print(outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
