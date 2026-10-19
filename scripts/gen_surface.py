#!/usr/bin/env python3
"""tool-surface.json 生成器：从运行时工具面导出上牌 manifest。

**为什么用生成器而不是手写 manifest**：手写的清单会与代码漂移——加了工具忘了
登记，或登记了已删除的工具。`mcp-surface-contract` 要求 manifest 随代码同步维护，
把"同步"落成一条命令（而不是一句纪律）是唯一可靠的做法。

事实源取 `fd_quant.surface.all_surface_tools()`（计算面七族 + 决策账本族的汇总点，
也就是 server 实际注册的那份清单），故新增/删除工具后重跑本脚本即可，manifest 不会漂。

账本族（`decision_ledger`）与计算面七族的纪律不同：计算工具只对传入序列作答；
账本工具持有本地 SQLite（追加式记录 + 结算）。这两件事在 manifest 里分别写成
组级 description 与顶层 description，免得读者把"账本落库"误当成"计算面读了库"。

用法：
    python3 scripts/gen_surface.py            # 写 tool-surface.json
    python3 scripts/gen_surface.py --check    # 只比对，不写（CI / selfcheck 用）
退出码：0 一致或已写入 / 1 --check 下发现漂移。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "tool-surface.json"

SERVER_NAME = "fd-quant"
ENTRY_PATH = "/fd-quant-compute"

#: 族分组 → 面向读者的组说明（工具名由 all_surface_tools() 提供，不在此重复）
GROUP_DESCRIPTIONS = {
    "technical_indicators": "自实现技术指标（不依赖 ta-lib）：均线族 / 摆动族 / 波动族 / 量价族。warmup 以 null 表示，每指标回带 params 与 warmup。",
    "portfolio_statistics": "组合统计：年化收益与波动、回撤（含起止日期）、风险调整收益（Sharpe/Sortino/Calmar）、相对基准度量（IR/TE/Beta）、集中度与换手。无风险利率为显式输入，不内置隐藏假设。",
    "fundamental_ratios": "基本面比率与红旗检测：盈利 / 成长 / 偿债 / 效率 / 现金流五类 + DuPont 三因子分解 + 规则化红旗（阈值可配）。输入为快照映射（三表科目）。",
    "valuation_operators": "估值算子：DCF（WACC 分步、两种终值法）、DDM、可比倍数（winsurize 去极值）、Monte Carlo（百分位带、种子可传以复现）、双变量敏感性网格、多方法合成（分歧超阈值则扣值不扣结论）。",
    "regime_detection": "regime 双速检测：慢速（趋势 + 回撤 + 波动率 Z）输出 risk_on/neutral/risk_off 与逐判据证据；快速（N 日冲击 + 波动率跳变）输出是否触发与降仓建议证据。波动率序列由调用方提供，本层不取数。",
    "factor_statistics": "因子统计原子：IC（Pearson 与 Spearman）、按因子分组的等权组合收益、单因子暴露回归（系数 / t 值 / R²）。不内置任何因子定义或因子库，因子与收益均由调用方传入。",
    "validation": "口径与反偷看护栏：walk-forward 窗口切分（测试窗严格晚于其训练窗）、无未来函数检查器（越界报错并点名首个越界点）、序列口径体检（缺失率与 warmup）。",
    "decision_ledger": "决策账本（本组**有状态**：本地 SQLite 单文件，路径由 db_path 参数或 FD_QUANT_LEDGER_DB 指定）：追加式决策记录（幂等键 as_of+symbol+pack+skill，重复提交被拒并返回既有 decision_id；修订产生新行并以 revises 指回）、到期结算（原始收益 + 相对基准 alpha；基准按后缀映射 .SS→000001.SS / .SZ→399001.SZ / .HK→^HSI / .T→^N225 / 无后缀→SPY，可显式覆盖）、反思随结算存储但**绝不推送**、点内检索（只回结算日 ≤ as_of，同标的最多 5 条 + 跨标的教训最多 3 条）、按 pack 聚合（样本数 / 平均 alpha / 胜率 / 原始收益均值，空组显式返回 0）。与计算面同一纪律：不取数、不联网——价格与基准序列由调用方传入，持有期按交易日计。",
}

DESCRIPTION_TEMPLATE = (
    "fd-quant 量化能力面，**两组**共 {count} 工具：① 确定性计算面七族（技术指标 / 组合统计 / "
    "基本面比率 / 估值算子 / regime 检测 / 因子统计 / 口径校验）——**只对传入的序列与参数作答，"
    "不取数、不读库、不联网、不调 LLM**：每个工具要求 as_of（ISO 日期）并在输入序列上执行，含 "
    "as_of 之后的数据点会被拒绝并点名该日期（不静默截断）；返回统一带 evidence 六字段与输入起止"
    "口径；同输入重跑逐位一致（固定归约顺序，不含非确定并行归约）。② 决策账本四工具（落账 / 结算 / "
    "点内检索 / 聚合）——**有状态**：把决策追加式落到本地 SQLite（db_path 参数或 FD_QUANT_LEDGER_DB），"
    "同样不取数、不联网（价格与基准序列由调用方传入，持有期按交易日计），无任何推送 / 广播接口。"
)

#: 共享实例（计算面无状态；账本面见 CONSUMPTION_NOTE 的状态模型说明）
HOSTING = ["shared-instance"]

CONSUMPTION_NOTE = (
    "计算面：纯计算、无状态、无凭据——同一输入恒得同一输出，故无需逐用户隔离，"
    "共享实例即可服务全部调用者。数据由调用方从各自通道取来后传入——本服务不做任何取数，"
    "因此不参与「价格只走本地通道」等数据路径约束。"
    "账本面（decision_ledger 组）：**有状态**——记录写本地单文件 SQLite，"
    "路径由 db_path 参数或 FD_QUANT_LEDGER_DB 指定（默认工作目录 fd_quant_ledger.db）。"
    "它同样不取数、不联网、无凭据，且无任何推送/广播接口；"
    "「多实例共享同一账本」不在本层支持（属部署形态问题，单机本地库是 v1 的边界）。"
)

#: 尚无 pack 技能引用本服务（消费方 pack 未建）。空表是诚实值，不是待办占位：
#: 一旦有技能进 pack，在此登记并由 selfcheck 反向校验。
SKILLS_ELIGIBLE: list[str] = []


def build_manifest() -> dict:
    sys.path.insert(0, str(REPO))
    from fd_quant.surface import all_surface_tools, tools_by_group

    groups = []
    for name, specs in sorted(tools_by_group().items()):
        groups.append(
            {
                "group": name,
                "level": specs[0].effective_level,
                "tools": sorted(s.name for s in specs),
                "description": GROUP_DESCRIPTIONS.get(name, ""),
            }
        )

    total = len(all_surface_tools())
    from fd_quant.output_schemas import load as _load_output_schemas, undeclared

    declared_outputs = sorted(_load_output_schemas()["tools"])
    return {
        "schema": "mcp-tool-surface/v1",
        "server": SERVER_NAME,
        "entry_path": ENTRY_PATH,
        "product_line": "wire",
        "hosting": HOSTING,
        # 工具数从运行时派生：写死在描述里必然随族增长而漂，而描述也是清单的一部分。
        "description": DESCRIPTION_TEMPLATE.format(count=total),
        "skills_eligible_for_packs": SKILLS_ELIGIBLE,
        # 行为属性同样从声明表派生（mcp-tool-annotations）：清单里再手抄一份
        # 就等于给协议层属性造了第二个事实源，两边必然漂。
        "tool_annotations": {
            spec.name: dict(spec.tool_annotations) for spec in all_surface_tools()
        },
        # 输出结构同样派生（mcp-result-contract）：`undeclared` 显式列出未声明的
        # 工具，使"没有声明"这件事在清单里可见，而不是静默空缺。
        # 命名语序约定（mcp-tool-naming）：写进生成器而不是手抄进 manifest，
        # 否则每次重新生成都会把它冲掉——而这类"被冲掉的字段"不会有任何提示。
        "naming_convention": {
            "order": "verb_noun",
            "applies_to": "new_tools",
            "note": ("新工具一律 `动词_名词`。既有名词序工具**不改名**——改名会破坏已发布 "
                     "pack 的引用（一个技能文本里点名的工具名消失即整个套件不可用）；"
                     "它们在清单中登记为 historical，改名的唯一正当路径是兼容期机制。"),
            "historical": "see tool_naming_legacy",
            "prefix_policy": ("跨服务同名工具不得靠前缀区分（前缀是服务标识而非工具语义）；"
                              "同名消解靠描述里写明部署与数据源差异。"),
        },
        "output_structure": {
            "declared": declared_outputs,
            "undeclared": undeclared(),
            "source": "schemas/outputs.json（由 scripts/gen_output_schemas.py 从运行时实测生成）",
            "note": "required = 多组输入下键集合的交集；类型取并集（null 是"
                    "「该值不存在」的正式表达）；payload/target_weights 引用契约真源 "
                    "schemas/rating.schema.json 与 target-weights.schema.json。",
        },
        "tool_groups": groups,
        "registry": {
            "entry_path": ENTRY_PATH,
            "expected_tool_count": total,
            "consumption_note": CONSUMPTION_NOTE,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="只比对现有 manifest，不写入（发现漂移退出码 1）")
    args = ap.parse_args()

    fresh = build_manifest()
    text = json.dumps(fresh, ensure_ascii=False, indent=2) + "\n"

    if args.check:
        if not MANIFEST.exists():
            print(f"✗ manifest 缺失：{MANIFEST}", file=sys.stderr)
            return 1
        cur = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if cur != fresh:
            cur_tools = {t for g in cur.get("tool_groups", []) for t in g.get("tools", [])}
            fresh_tools = {t for g in fresh["tool_groups"] for t in g["tools"]}
            added = sorted(fresh_tools - cur_tools)
            removed = sorted(cur_tools - fresh_tools)
            print("✗ manifest 与运行时工具面漂移：", file=sys.stderr)
            if added:
                print(f"  运行时有而 manifest 无（需登记）：{added}", file=sys.stderr)
            if removed:
                print(f"  manifest 有而运行时无（需删除）：{removed}", file=sys.stderr)
            if not added and not removed:
                print("  工具集相同但其它字段有差异（description / 分组 / 计数）", file=sys.stderr)
            return 1
        print(f"✓ manifest 与运行时一致（{fresh['registry']['expected_tool_count']} 工具）")
        return 0

    MANIFEST.write_text(text, encoding="utf-8")
    print(f"✓ 已写入 {MANIFEST.relative_to(REPO.parent)}")
    for g in fresh["tool_groups"]:
        print(f"    {g['group']:24s} {len(g['tools']):2d} 工具  [{g['level']}]")
    print(f"  合计 {fresh['registry']['expected_tool_count']} 工具")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
