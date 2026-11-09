"""账本 MCP 工具组：落账 / 结算 / 点内检索 / 聚合（工具名即函数名，`动词_名词`）。

工具是**账本唯一被外部（MCP 调用方）触达的面**：模型在同一轮对话里拿到结构化结果
或点名缺什么的错误，因此描述必须自足——输入形状、参数默认值、单位、`as_of` 要求、
返回形状，以及账本层的两条显式假设（**序列由调用方传入** = 账本不取数；
**持有期按交易日计** = 交易日历由传入序列的日期集合隐式给出）。

数据库路径三态：显式 `db_path` → 环境变量 `FD_QUANT_LEDGER_DB` → 当前工作目录
`fd_quant_ledger.db`。每个工具都带上 `db_path`，故同一台机器上多份账本可并存
（测试因此零配置，只在 env 或参数上传一个临时文件）。

**无推送**：这四个工具里没有任何"推送 / 广播 / 订阅 / 通知"形状的动作——反思只存、
只随检索带出；把它注入决策者是消费侧（P5/P4）的行为（spec: 反思存储与不自动广播）。
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..caller import resolve_caller
from ..compute import ToolSpec
from .store import LedgerStore
from .settlement import DEFAULT_HOLDING_DAYS, check_iso_date

GROUP = "decision_ledger"

#: 账本层两条显式假设（逐工具写入描述；由 tests/test_ledger_tools.py 断言存在）。
ASSUMPTIONS = (
    "假设：账本不取数——任何价格 / 基准序列都由调用方传入；"
    "持有期按交易日计，交易日历由传入序列的日期集合隐式给出（账本不内置日历）。"
)


def _store(db_path: Optional[str]) -> LedgerStore:
    """按调用参数（或 env / 默认路径）打开账本。短生命周期：调用结束即不再持有连接。"""
    return LedgerStore(db_path)


def _envelope(tool: str, as_of: str, store: LedgerStore, result: dict) -> dict:
    """账本工具的统一外壳：工具名 + 口径 + 生效的库路径 + 结果。

    与计算面的 `result_envelope` 刻意不同：账本工具的主角不是数值序列，故不造
    `input_start/input_end`，也不编造 evidence——价格来源在结算的
    `benchmark_symbol` / `*_return` 里，判定来源在 `payload` / `evidence` 里。
    `db_path` 进外壳是因为"这条记录落在哪个库"是调用方必须能核对的事实。
    """
    return {
        "tool": tool,
        "as_of": as_of,
        "db_path": str(store.db_path),
        "result": result,
    }


def record_decision(
    as_of: str,
    pack: str,
    skill: str,
    payload: Mapping[str, Any],
    symbol: Optional[str] = None,
    portfolio_key: Optional[str] = None,
    record_kind: str = "single",
    db_path: Optional[str] = None,
    target_weights: Optional[Mapping[str, Any]] = None,
    evidence: Optional[Sequence[str]] = None,
    revises: Optional[str] = None,
    now: Optional[str] = None,
) -> dict:
    """追加一条决策记录到本地账本（幂等键 caller + as_of + symbol + pack + skill）。

    落账即审计：`payload` 是契约实例（`Rating` JSON 对象，字段同 schemas/rating.schema.json），
    账本原样存下并在查回时逐字给予；`payload.as_of` 必须等于 `as_of`、`payload.subject`
    必须等于 `symbol`、`payload.producer.pack/skill` 必须等于 `pack`/`skill`（不一致即拒绝
    并同时点名两边——账本不替调用方挑一个当真）。

    调用者隔离：`caller` **不是入参**——它由调用上下文解析（HTTP 用 Authorization 的
    Bearer 指纹，stdio 用环境变量 FD_QUANT_CALLER，都没有则是 anonymous）。故调用方
    无法声称自己是别人：伪造的入参身份不存在，因为根本没有这个入参。

    幂等：同一 (caller, as_of, symbol, pack, skill) 重复提交**被拒绝但返回既有 decision_id**
    （返回里 created=false、reason="duplicate_key"），记录总数逐字不变——这是"每个决策只
    算一次账"的屏障，重复记录会静默污染胜率与平均 alpha。修订：传 revises=<既有 decision_id>
    发一条**新记录**（created=true、revises 指回、revision_seq 递增），被修订记录逐字不改；
    修订必须指向同键的当前 tip（链是线性的），内容与 tip 逐字相同则幂等返回 tip。

    组合形态（add-portfolio-intent-contract）：`record_kind="portfolio"` + `portfolio_key`
    （该期组合的稳定键），此时**不传 symbol**、payload 是 `PortfolioIntent`；单票仍传
    symbol + Rating。主体字段与种类必须一致（同时给或同时缺都被拒绝）。幂等键因此是
    (caller, record_kind, as_of, subject_key, pack, skill)，subject_key 单体取 symbol、
    组合取 portfolio_key——同一 as_of 下两者互不覆盖。

    Input: as_of (ISO 日期, 必填; 也是幂等键的一部分), symbol (标的，如 600519.SS；组合记录不传),
      pack / skill (产出方标识，必须与 payload.producer 一致), payload (Rating 契约对象, 必填),
      db_path (可选；缺省取环境变量 FD_QUANT_LEDGER_DB，再缺省为工作目录 fd_quant_ledger.db),
      target_weights (可选 TargetWeights 契约对象，随判定一起存), evidence (可选引用 id 数组；
      缺省取 payload.target_evidence_ids), revises (可选：被修订的 decision_id), now (可选 ISO
      时间戳，仅用于可复现测试；缺省记当前 UTC 时刻)。
    Returns: tool / as_of / db_path / result（result 含 created (bool；false 即重复提交被拒)、
      reason (created | duplicate_key | revision_created | revision_duplicate)、decision_id
      (既有或新建的标识)、status (pending | settled)、revises、revision_seq (0=初始记录)、
      created_at、counts.decisions / counts.settlements / counts.pending / counts.settled ——
      行数可用于核对"重复提交未产生第二行"）。单位：无（本工具不做算术）。
      假设：账本不取数——任何价格 / 基准序列都由调用方传入；持有期按交易日计，交易日历由传入序列的日期集合隐式给出（账本不内置日历）。
    """
    store = _store(db_path)
    result = store.record(
        caller=resolve_caller(), as_of=as_of, symbol=symbol, pack=pack, skill=skill,
        payload=payload, record_kind=record_kind, portfolio_key=portfolio_key,
        target_weights=target_weights, evidence=evidence, revises=revises, now=now,
    )
    return _envelope("record_decision", as_of, store, result)


def settle_decision(
    decision_id: str,
    symbol_dates: Sequence[str],
    symbol_prices: Sequence[Optional[float]],
    benchmark_dates: Sequence[str],
    benchmark_prices: Sequence[Optional[float]],
    as_of: str,
    db_path: Optional[str] = None,
    reflection: Optional[str] = None,
    benchmark_symbol: Optional[str] = None,
    holding_days: Optional[int] = None,
    now: Optional[str] = None,
) -> dict:
    """到期结算一条决策：算原始收益、基准收益与 alpha，并落库（幂等）。

    **序列由调用方传入**——账本不取数、不联网（与计算面同一纪律）：标的与基准两条
    (日期, 价格) 序列都由消费侧从自己的数据通道取来后传入。**持有期按交易日计**：
    交易日历由传入序列的日期集合隐式给出（账本不内置日历），持有期 5 表示入场日之后
    第 5 个对齐交易日出场。

    对齐与覆盖：日期轴 = 标的日期 ∩ 基准日期（两边都有价的日子才算交易日）；入场日 =
    该决策的 as_of，必须落在轴上；从入场日到轴末尾的点数必须 ≥ holding_days + 1，否则
    **拒绝结算并报缺口交易日数**（error 文本含 "missing N trading days"，同时 error 带
    missing_trading_days 字段），不落任何行、状态保持 pending——半窗 alpha 比没有 alpha
    更坏。价格缺失（null）或非正的端点拒绝（点名日期）。晚于 as_of 的数据点拒绝并点名首个
    越界日期（不静默截断）。

    基准：显式 benchmark_symbol 永远优先；否则按市场后缀映射（.SS→000001.SS、
    .SZ→399001.SZ、.HK→^HSI、.T→^N225、无后缀→SPY，映射表可配），结果里的
    benchmark_mapping 说明基准是显式点的还是映射推出来的（可审计）。

    幂等：已结算的 decision 重复结算返回既有结果（created=false、reason="duplicate_settlement"），
    不插入第二条结算行；本次传了与既有不同的 reflection 时既有行不被改写，返回
    reflection_ignored=true 明示未生效（追加语义）。reflection 可选（2–4 句由调用方生成），
    不传不影响结算。

    Input: decision_id (必填，record_decision 返回的那个), symbol_dates / symbol_prices
      (标的 (ISO 日期, 价格) 序列，等长、日期严格递增), benchmark_dates / benchmark_prices
      (基准序列，同形状), as_of (ISO 日期，必填：结算口径日期，晚于它的价格点被拒),
      db_path (可选；缺省取环境变量 FD_QUANT_LEDGER_DB，再缺省为工作目录 fd_quant_ledger.db),
      reflection (可选字符串；只在首次结算可写入),
      benchmark_symbol (可选：显式基准代码，优先于后缀映射), holding_days (可选正整数，
      默认 5 个交易日), now (可选 ISO 时间戳，仅用于可复现测试)。
    Returns: tool / as_of / db_path / result（result 含 created、reason、decision_id、
      decision_as_of (入场日)、symbol / pack / skill、status（结算后为 settled）、settled_at、
      settlement_date (出场日 = 结算日)、entry_price / exit_price、raw_return (标的持有期简单
      收益，小数：0.01 = 1%)、benchmark_symbol / benchmark_mapping、benchmark_entry_price /
      benchmark_exit_price、benchmark_return (基准同期收益，小数)、alpha (= raw_return −
      benchmark_return，小数)、holding_days (交易日数)、reflection (可为 null)、
      reflection_ignored (bool)、counts (行数快照，核对未插入第二行)）。
      假设：账本不取数——任何价格 / 基准序列都由调用方传入；持有期按交易日计，交易日历由传入序列的日期集合隐式给出（账本不内置日历）。
    """
    store = _store(db_path)
    result = store.settle(
        decision_id=decision_id, symbol_dates=symbol_dates, symbol_prices=symbol_prices,
        benchmark_dates=benchmark_dates, benchmark_prices=benchmark_prices, as_of=as_of,
        reflection=reflection, benchmark_symbol=benchmark_symbol, holding_days=holding_days,
        now=now,
    )
    return _envelope("settle_decision", as_of, store, result)


def settle_portfolio(
    decision_id: str,
    legs: Mapping[str, Sequence[Any]],
    benchmark_dates: Sequence[str],
    benchmark_prices: Sequence[Optional[float]],
    benchmark_symbol: str,
    as_of: str,
    db_path: Optional[str] = None,
    reflection: Optional[str] = None,
    holding_days: Optional[int] = None,
    now: Optional[str] = None,
) -> dict:
    """到期结算一条**组合**记录：算加权收益、现金部分与相对基准 alpha，并落库（幂等）。

    与 `settle_decision` 的三处不同，都是组合语义要求的：

    1. **整期多腿**：`legs` 是 `{symbol: [日期序列, 价格序列]}` 映射，必须覆盖该期
       `PortfolioIntent.weights` 的全部标的——少一条腿即拒绝（不把缺输入读成"没持仓"），
       多一条腿同样拒绝（没有权重的腿不该出现在结算里）。
    2. **日期轴 = 全部标的日期 ∩ 基准日期**。各腿各用各的轴算出来的是几段不同持有期收益
       的加权和，那不是"一个组合在某一期的收益"。
    3. **现金单独报**：`cash_weight = 1 − Σ权重`，收益记零，**不摊进标的收益**。

    **权重不另传**：取自记录的 payload（`PortfolioIntent.weights`）。允许另传就会有两个
    权重来源，而两份迟早不一致，"这笔单依据哪份权重"也就没有唯一答案。

    整期原子：任一腿不覆盖持有期 → 整次拒绝并报缺口交易日数，不落任何行、状态保持
    pending。幂等：已结算再结算返回既有结果（created=false）。

    Input: decision_id (必填，组合记录的 id), legs (必填，`{symbol: [dates, prices]}`),
      benchmark_dates / benchmark_prices (基准序列), benchmark_symbol (必填：组合没有单一
      标的可做后缀映射), as_of (ISO 日期), db_path (可选；缺省取环境变量
      FD_QUANT_LEDGER_DB，再缺省为工作目录 fd_quant_ledger.db), reflection (可选，只在首次可写),
      holding_days (可选正整数，默认 5 个交易日), now (可选，仅测试用)。
    Returns: tool / as_of / db_path / result（created、reason、decision_id、settlement 含
      raw_return (= Σ 权重 × 各腿收益，小数)、invested_weight、cash_weight、cash_return
      (恒 0)、benchmark_symbol / benchmark_return、alpha (= raw_return − benchmark_return)、
      holding_days、leg_returns (逐腿收益)、reflection、reflection_ignored、counts）。
      假设：账本不取数——任何价格 / 基准序列都由调用方传入；权重取自记录 payload，
      不另传（另传就有两个权重来源）；持有期按交易日计，交易日历由传入序列的日期集合
      隐式给出（账本不内置日历）。
    """
    store = _store(db_path)
    parsed: dict[str, tuple[Sequence[Any], Sequence[Any]]] = {}
    for sym, pair in dict(legs).items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise LedgerError(
                f"legs[{sym!r}] must be [dates, prices] (got {type(pair).__name__})"
            )
        parsed[str(sym)] = (pair[0], pair[1])
    result = store.settle_portfolio(
        decision_id=decision_id, legs=parsed, benchmark_dates=benchmark_dates,
        benchmark_prices=benchmark_prices, as_of=as_of, benchmark_symbol=benchmark_symbol,
        reflection=reflection, holding_days=holding_days, now=now,
    )
    return _envelope("settle_portfolio", as_of, store, result)


def retrieve_decisions(
    symbol: str,
    as_of: str,
    db_path: Optional[str] = None,
    n_same: int = 5,
    m_cross: int = 3,
    pack: Optional[str] = None,
) -> dict:
    """点内检索历史决策与教训：只回**结算日不晚于 as_of** 的记录，按结算日倒序。

    点内（point-in-time）纪律落在查询层而不是调用方自觉：`as_of` 之后才结算的记录
    **不会**出现在结果里（历史回放看到未来结算是这条纪律要防的具体 bug）。结果分两组：
    同标的 `same_symbol`（最多 n_same 条）与跨标的 `cross_symbol` 教训（最多 m_cross 条）。
    每条记录带判定快照（payload / evidence / 修订指针）**与**结算结果（含 reflection，若存在）。
    空账本或没有匹配记录时两个数组为空——"没有历史"是合法结果，不是错误。

    Input: symbol (必填，如 600519.SS), as_of (ISO 日期，必填：检索口径日；只回结算日 ≤ 它的记录),
      db_path (可选；缺省取环境变量 FD_QUANT_LEDGER_DB，再缺省为工作目录 fd_quant_ledger.db),
      n_same (同标的最多返回条数，默认 5，可配；0 表示不回),
      m_cross (跨标的教训最多返回条数，默认 3，可配), pack (可选：只看某个 pack 的记录)。
    Returns: tool / as_of / db_path / result（result 含 symbol、n_same / m_cross (回显)、
      pack、same_symbol 与 cross_symbol（各为数组，元素含 decision_id / as_of / symbol /
      pack / skill / status / revises / revision_seq / created_at / payload (Rating 对象) /
      target_weights / evidence (引用 id 数组) / settlement (settled_at / settlement_date /
      entry_price / exit_price / raw_return / benchmark_symbol / benchmark_mapping /
      benchmark_return / alpha / holding_days / reflection)）、counts（两组的条数与合计））。
      单位：returns 与 alpha 为小数（0.01 = 1%）。
      假设：账本不取数——任何价格 / 基准序列都由调用方传入；持有期按交易日计，交易日历由传入序列的日期集合隐式给出（账本不内置日历）。
    """
    store = _store(db_path)
    result = store.retrieve(caller=resolve_caller(), symbol=symbol, as_of=as_of, n_same=n_same,
                            m_cross=m_cross, pack=pack)
    return _envelope("retrieve_decisions", as_of, store, result)


def list_pending_decisions(
    as_of: str,
    pack: Optional[str] = None,
    record_kind: Optional[str] = None,
    as_of_from: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    holding_days: Optional[int] = None,
    db_path: Optional[str] = None,
) -> dict:
    """列出**待结算**记录（分页 + 总数），供定时复盘逐条结算。

    为什么需要它：`retrieve_decisions` 只回已结算且结算日 ≤ as_of 的记录（点内纪律），
    看不到待结算清单。此前做到期复盘只能靠"落账时自己记住 decision_id"——对定时任务
    等于没有。本工具补上这条唯一的"该结什么"入口。

    **未到期的记录照样在清单里**，带 `due_estimate` 与 `matured=false`。隐去它们会让
    "没东西要结"与"有 N 条还没到期"变成同一个输出，而两者的处置完全不同。

    到期日 `due_estimate` 是**自然日粗估**（as_of + holding_days 天）：精确交易日以结算时
    传入序列的日期集合为准，账本不内置日历。

    Input: as_of (ISO 日期，必填：本次复盘的参考日，用于判断哪些已到期),
      pack / record_kind (可选过滤；record_kind 取 single | portfolio),
      as_of_from (可选：只要 as_of ≥ 该日的记录), limit / offset (分页，默认 100 / 0),
      holding_days (可选正整数，默认取本库配置的持有期),
      db_path (可选；缺省取环境变量 FD_QUANT_LEDGER_DB，再缺省为工作目录 fd_quant_ledger.db)。
    Returns: tool / as_of / db_path / caller / filters / pagination（limit、offset、
      total、returned）/ holding_days / today / items（每条含 decision_id、as_of、subject、
      record_kind、pack、skill、status、payload、due_estimate、holding_days、matured、
      missing_trading_days_estimate、counts）。仅在调用者自身范围内生效。
      假设：账本不取数——任何价格 / 基准序列都由调用方传入；持有期按交易日计，交易日历由传入序列的日期集合隐式给出（账本不内置日历）。
    """
    store = _store(db_path)
    a = check_iso_date(as_of, name="as_of")
    result = store.list_pending(
        caller=resolve_caller(), pack=pack, record_kind=record_kind,
        as_of_from=as_of_from, limit=limit, offset=offset,
        holding_days=holding_days, today=a,
    )
    return _envelope("list_pending_decisions", a, store, result)


def summarize_pack_performance(
    as_of: str,
    db_path: Optional[str] = None,
    packs: Optional[Sequence[str]] = None,
    group_by_skill: bool = False,
    group_by_holding: bool = False,
    holding_days: Optional[Sequence[int]] = None,
) -> dict:
    """按 pack 汇总结算表现：样本数、平均 alpha、胜率、原始收益均值（只统计已结算）。

    只统计 status=settled **且结算日 ≤ as_of** 的记录（结算口径先于 as_of，与检索同一
    点内纪律）；未结算的决策只贡献"这个 pack 存在"。**零样本的组显式返回 samples=0**
    （比值字段为 null）而不是被省略——"还没结算"与"没有这个 pack"含义完全不同。

    口径（判定"哪个 pack 更准"用的就是这几个数）：alpha = raw_return − benchmark_return
    （小数）；win_rate = alpha > 0 的记录占比（alpha 恰为 0 记负，不算赢）；mean_alpha /
    mean_raw_return 为组内算术均值。

    Input: as_of (ISO 日期，必填：只统计结算日 ≤ 它的记录), db_path (可选；缺省取环境变量
      FD_QUANT_LEDGER_DB，再缺省为工作目录 fd_quant_ledger.db),
      packs (可选 pack 名数组；给了就保证这些 pack 各有一行——没有结算记录时 samples=0),
      group_by_skill (可选 bool，默认 false：组细化为 pack × skill), group_by_holding
      (可选 bool，默认 false：再按持有期分档), holding_days (可选持有期档位数组，默认只有本库
      配置的持有期；仅在 group_by_holding=true 时有意义，用于把 3/5/10 日并列比较)。
    Returns: tool / as_of / db_path / result（result 含 only_settled (恒 true)、group_by
      (生效的分组维度数组)、holding_days (档位，未按持有期分组时为 null)、registered_packs
      (账本里出现过的 pack)、rows（每行含 pack、skill (未按 skill 分组时为 null)、
      holding_days (未按持有期分组时无此键)、samples、wins、mean_alpha、win_rate、
      mean_raw_return；零样本行的比值为 null）、counts（行数与纳入统计的已结算记录数））。
      单位：mean_alpha / mean_raw_return / win_rate 为小数（0.02 = 2%）。
      假设：账本不取数——任何价格 / 基准序列都由调用方传入；持有期按交易日计，交易日历由传入序列的日期集合隐式给出（账本不内置日历）。
    """
    store = _store(db_path)
    result = store.stats_by_pack(
        caller=resolve_caller(), as_of=as_of, packs=packs, group_by_skill=group_by_skill,
        group_by_holding=group_by_holding, holding_days=holding_days,
    )
    return _envelope("summarize_pack_performance", as_of, store, result)


#: 账本族的协议层行为属性。四个工具都写本地 SQLite，但**写入语义不同**，
#: 这个区别必须在协议层可见——否则客户端只能把「检索历史决策」和「落一条
#: 决策」当成同一类东西，对前者弹确认、或对后者静默自动批准。
#:
#: - `record_decision` / `settle_decision`：写库。**非幂等**——重复提交同一个
#:   决策会被幂等键拒绝并返回既有 id，但「被拒」与「成功」是两种结果，
#:   自动重试不安全。`destructiveHint=False`：只追加，不改写既有行。
#: - `retrieve_decisions` / `summarize_pack_performance`：只读，幂等。
_LEDGER_WRITE = {
    "readOnlyHint": False,
    "destructiveHint": False,   # 追加式，不就地改写
    "idempotentHint": False,    # 重复提交被拒 ≠ 幂等成功
    "openWorldHint": False,     # 不取数、不联网
}
_LEDGER_READ = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

TOOLS: list[ToolSpec] = [
    ToolSpec("record_decision", GROUP, record_decision, tool_annotations=_LEDGER_WRITE),
    ToolSpec("settle_decision", GROUP, settle_decision, tool_annotations=_LEDGER_WRITE),
    ToolSpec("settle_portfolio", GROUP, settle_portfolio, tool_annotations=_LEDGER_WRITE),
    ToolSpec("retrieve_decisions", GROUP, retrieve_decisions, tool_annotations=_LEDGER_READ),
    ToolSpec("list_pending_decisions", GROUP, list_pending_decisions,
             tool_annotations=_LEDGER_READ),
    ToolSpec("summarize_pack_performance", GROUP, summarize_pack_performance,
             tool_annotations=_LEDGER_READ),
]

#: 默认持有期在描述之外也对调用方可见（`summarize_pack_performance` 的分档默认值）。
DEFAULT_HOLDING = DEFAULT_HOLDING_DAYS
