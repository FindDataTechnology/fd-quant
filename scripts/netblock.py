"""pytest 插件：封掉一切出网路径（task 9.3 的运行时证据）。

用 `-p` 加载本插件：
    PYTHONPATH=scripts .venv/bin/python -m pytest tests/ -p netblock

**为什么封 connect 而不是替换 socket 类**：替换 `socket.socket` 会破坏标准库
`ssl` 模块自身的 `class SSLSocket(socket)` 继承（实测 `TypeError: function()
argument 'code' must be code, not str`），使整个解释器在导入期就崩。封住
`connect` / `connect_ex` / `create_connection` / DNS 解析足以让任何真实出网
尝试失败——这正是要证的事实。

用 `pytest_configure` 打补丁（导入完成后）而不是 `sitecustomize`（导入期），
后者会在 ssl 自己 import 时就动手。
"""
from __future__ import annotations

import socket

_ORIGINAL: dict[str, object] = {}


def _deny(*a, **k):
    raise RuntimeError("NETWORK BLOCKED: this run must not touch the network")


def pytest_configure(config):
    _ORIGINAL["socket.connect"] = socket.socket.connect
    _ORIGINAL["socket.connect_ex"] = socket.socket.connect_ex
    _ORIGINAL["create_connection"] = socket.create_connection
    _ORIGINAL["getaddrinfo"] = socket.getaddrinfo
    _ORIGINAL["gethostbyname"] = socket.gethostbyname

    socket.socket.connect = _deny
    socket.socket.connect_ex = _deny
    socket.create_connection = _deny
    socket.getaddrinfo = _deny
    socket.gethostbyname = _deny


def pytest_unconfigure(config):
    socket.socket.connect = _ORIGINAL["socket.connect"]
    socket.socket.connect_ex = _ORIGINAL["socket.connect_ex"]
    socket.create_connection = _ORIGINAL["create_connection"]
    socket.getaddrinfo = _ORIGINAL["getaddrinfo"]
    socket.gethostbyname = _ORIGINAL["gethostbyname"]


def main() -> int:
    """独立自检：确认闸门真的会拦（反向验证，不是"相信它会拦"）。"""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    probe = (
        "import socket, sys\n"
        "sys.path.insert(0, %r)\n"
        "import netblock\n"
        "netblock.pytest_configure(None)\n"
        "try:\n"
        "    socket.create_connection(('example.com', 80), timeout=2)\n"
        "    print('FAIL: 出网未被拦截')\n"
        "    sys.exit(1)\n"
        "except RuntimeError as e:\n"
        "    print('OK: 闸门生效 —', e)\n"
    ) % str(Path(__file__).resolve().parent)
    p = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    print(p.stdout.strip() or p.stderr.strip()[-400:])
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
