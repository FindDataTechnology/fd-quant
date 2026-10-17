"""校验族测试：逐 spec Scenario 一个用例 + 性质断言（不变式）+ 拒绝 + 确定性。

最关键的两条性质（本族存在的理由，也是批 1.2 回测引擎依赖的性质）：

1. **每个测试窗的数据严格晚于其训练窗**（`test_start_i == train_end_i + 1` 且日期更大）；
2. **注入的未来数据被抓**，错误文本点名越界**日期与下标**。

"注入夹具"按 spec 措辞（`#### Scenario: 注入的未来函数被抓`）就地构造：一段声明
`as_of` 为某日的序列，其中却含该日之后的数据点。spec 措辞逐条引用在用例 docstring 里。
夹具用 `quantdata.dates` 造真日期序列；数值用简单等差数列（本族不做数值计算，数值只用于
满足"成对等长"的入参契约）。
"""
from __future__ import annotations

import json

import pytest

import quantdata as qd
from fd_quant.compute.base import ComputeError, assert_bitequal
from fd_quant.compute.validation import (
    TOOLS,
    check_observation_sufficiency,
    check_series_as_of_bounds,
    split_holdout,
    split_walk_forward_windows,
    validate_no_lookahead,
)


def _series(n: int, *, start: str = "2025-01-01") -> tuple[list[str], list[float]]:
    ds = qd.dates(n, start)
    return ds, [float(i) for i in range(n)]


def _injected_future_fixture():
    """注入夹具：序列声称 as_of=第 100 日，却含第 101..110 日的点。

    构造方式就是 spec Scenario 描述的那件事——不是修改被测代码，而是给它一份
    "在 as_of 时刻不可能知道的数据"。首个越界点的下标是 100（越界点共 10 个）。
    """
    ds = qd.dates(110)
    values = [100.0 + i for i in range(110)]
    declared_as_of = ds[99]              # 声称的信息截止
    future_indices = list(range(100, 110))
    return ds, values, declared_as_of, future_indices


# ── spec Scenario：窗口切分不重叠且有序 ───────────────────────────


def test_walk_forward_200_days_60_20_20():
    """spec Scenario:「以训练 60/测试 20/步长 20 切分 200 日范围 → 窗口列表按时间有序，
    每个测试窗前于其后任一训练窗，且相邻窗口符合步长」；tasks 7.1 逐条同义。

    **spec 文本此处有一处不自洽（已在报告里点名）**：该句若读作"每个测试窗早于**其后**
    的训练窗"，在 60/20/20 下不可能成立——窗口是无缝平铺的，窗口 k+1 的训练窗（起点
    20(k+1)）必然与窗口 k 的测试窗（20k+60 .. 20k+79）交叠。与 Requirement 正文
    「窗口列表 SHALL 保证每个测试窗口的数据严格晚于其训练窗口」一致的读法是"每个测试窗
    晚于**其（及此前）**训练窗"，这也是本用例断言的口径。
    """
    ds = qd.dates(200)
    values = [float(i) for i in range(200)]
    r = split_walk_forward_windows(ds, values, ds[-1], 60, 20, 20)

    # 窗口数 = (200 - 80) // 20 + 1 = 7
    assert r["window_count"] == 7 and r["n_windows"] == 7
    assert r["train_window"] == 60 and r["test_window"] == 20 and r["step"] == 20
    assert r["point_count"] == 200 and r["unassigned_tail"] == 0

    ws = r["windows"]
    assert [w["index"] for w in ws] == list(range(7))
    for w in ws:
        # 本窗口的测试窗严格晚于本窗口的训练窗（Requirement 正文的那条性质）
        assert w["test_start_i"] > w["train_end_i"]
        assert w["test_start"] > w["train_end"]
        # 也晚于**所有在此之前的**训练窗（时间有序的直接推论）
        for earlier in ws[:w["index"]]:
            assert w["test_start_i"] > earlier["train_end_i"]
            assert w["test_start"] > earlier["train_end"]
    for a, b in zip(ws, ws[1:]):
        # 按时间有序：整体（训练起点）单调前进
        assert a["train_start_i"] < b["train_start_i"]
        assert a["train_end_i"] < b["train_end_i"] < b["test_end_i"]
        assert b["train_start_i"] - a["train_start_i"] == 20  # 相邻窗口符合步长
        assert b["train_start"] > a["train_start"]
    for w in ws:
        assert w["train_start_i"] == w["index"] * 20
        assert w["train_end_i"] == w["index"] * 20 + 59
        assert w["test_start_i"] == w["index"] * 20 + 60
        assert w["test_end_i"] == w["index"] * 20 + 79
        assert w["train_size"] == 60 and w["test_size"] == 20
    assert r["invariant"] == "test_strictly_after_train"


def test_every_test_window_strictly_follows_its_train_window():
    """本模块的核心不变式：测试窗数据**严格晚于**训练窗——日期与下标两侧都断言。

    下标侧：test_start_i == train_end_i + 1（无缝、无重叠）；日期侧：test_start > train_end。
    容差为 0：这是构造性质，不是近似。
    """
    for n, tw, tew, st in ((200, 60, 20, 20), (60, 30, 15, 5), (37, 10, 3, 7),
                           (100, 20, 20, 60), (80, 60, 20, 20)):
        ds = qd.dates(n)
        r = split_walk_forward_windows(ds, [float(i) for i in range(n)], ds[-1], tw, tew, st)
        assert r["window_count"] == (n - tw - tew) // st + 1
        assert r["invariant"] == "test_strictly_after_train"
        starts = [w["train_start_i"] for w in r["windows"]]
        assert starts == sorted(starts) and starts == list(range(starts[0], starts[-1] + 1, st))
        for w in r["windows"]:
            assert w["test_start_i"] == w["train_end_i"] + 1
            assert w["test_start"] > w["train_end"]
            assert w["train_start_i"] <= w["train_end_i"] < w["test_start_i"] <= w["test_end_i"]
            assert w["train_end_i"] - w["train_start_i"] + 1 == tw
            assert w["test_end_i"] - w["test_start_i"] + 1 == tew
            # 窗口内日期与下标一一对应（没有跳号或错位）
            assert ds[w["train_start_i"]] == w["train_start"]
            assert ds[w["train_end_i"]] == w["train_end"]
            assert ds[w["test_start_i"]] == w["test_start"]
            assert ds[w["test_end_i"]] == w["test_end"]


def test_window_dates_come_from_input_series():
    """窗口边界日期必须是输入序列里的真日期，不是现推的（否则与数据对不上）。"""
    ds, vs = _series(150, start="2024-03-01")
    r = split_walk_forward_windows(ds, vs, max(ds), 60, 20, 20)
    for w in r["windows"]:
        for k in ("train_start", "train_end", "test_start", "test_end"):
            assert w[k] in ds, f"{k}={w[k]} 不在输入日期序列里"


def test_windows_are_indexed_from_zero_and_ordered():
    ds, vs = _series(300)
    r = split_walk_forward_windows(ds, vs, max(ds), 80, 20, 20)
    idx = [w["index"] for w in r["windows"]]
    assert idx == list(range(len(idx))), "index 应从 0 起连续"
    starts = [w["train_start_i"] for w in r["windows"]]
    assert starts == sorted(starts), "窗口应按起点升序（前推，不回头）"


def test_windows_never_cover_data_beyond_as_of():
    """as_of 是轴的口径截止：任何窗口边界都不得晚于 as_of（构造性保证，逐窗断言）。"""
    ds = qd.dates(150)
    as_of = ds[120]
    probe = ds[:121]  # 轴上合法的部分
    r = split_walk_forward_windows(probe, [float(i) for i in range(121)], as_of, 60, 20, 20)
    assert r["as_of"] == as_of
    for w in r["windows"]:
        assert w["test_end"] <= as_of
    # 越界轴直接拒绝并点名首个越界日期
    with pytest.raises(ComputeError) as ei:
        split_walk_forward_windows(ds, [1.0] * 150, as_of, 60, 20, 20)
    assert ds[121] in str(ei.value)


def test_overlapping_step_creates_more_windows():
    """步长小于测试窗 → 窗口重叠 → 数量变多（参数确实生效）。"""
    ds, vs = _series(300)
    a = split_walk_forward_windows(ds, vs, max(ds), 100, 40, 40)
    b = split_walk_forward_windows(ds, vs, max(ds), 100, 40, 10)
    assert b["window_count"] > a["window_count"]
    for w in b["windows"]:  # 步长仍严格生效，不变式不因重叠而破
        assert w["test_start_i"] == w["train_end_i"] + 1


def test_adjacent_windows_skip_when_step_exceeds_test_window():
    """step > test_window 时相邻测试窗之间跳过 bar（调用方自选的滚动节奏，允许）。"""
    ds = qd.dates(100)
    r = split_walk_forward_windows(ds, [1.0] * 100, ds[-1], 20, 20, 60)
    assert r["window_count"] == 2
    a, b = r["windows"]
    assert b["train_start_i"] - a["train_start_i"] == 60
    assert b["train_start_i"] > a["test_end_i"]  # 中间跳过了 40 个 bar
    assert b["test_end_i"] <= 99


def test_start_end_narrow_the_axis_before_splitting():
    """start/end 闭区间裁剪：裁剪后再切，窗口下标落在原轴坐标系上且范围被尊重。"""
    ds = qd.dates(200)
    r = split_walk_forward_windows(ds, [1.0] * 200, ds[-1], 20, 10, 10,
                                   start=ds[50], end=ds[119])
    assert r["range_start"] == ds[50] and r["range_end"] == ds[119]
    assert r["window_count"] == (70 - 30) // 10 + 1
    for w in r["windows"]:
        assert ds[50] <= w["train_start"] and w["test_end"] <= ds[119]
        assert w["train_start_i"] >= 50 and w["test_end_i"] <= 119


def test_unassigned_tail_is_reported():
    """末尾放不下一个窗口的 bar 必须被点名（unassigned_tail），不静默丢弃。"""
    ds, vs = _series(101)
    r = split_walk_forward_windows(ds, vs, max(ds), 60, 20, 20)
    assert r["unassigned_tail"] == 101 - (r["windows"][-1]["test_end_i"] + 1)
    assert r["unassigned_tail"] == 1
    assert r["coverage_start"] == ds[0] and r["coverage_end"] == ds[99]


def test_default_windows_match_the_spec_scenario():
    """默认窗口 = spec Scenario 的 60/20（步长取 test_window），不传参数即得该口径。

    默认值刻意取小：任何一个"够切一刀"的序列都跑得动，且能逐字复现 spec 的
    200 日 / 60 / 20 / 20 用例；放不下时报错而**不静默缩放**（缩放等于替调用方发明参数）。
    """
    ds, vs = _series(200)
    r = split_walk_forward_windows(ds, vs, max(ds))
    assert (r["train_window"], r["test_window"], r["step"]) == (60, 20, 20)
    assert r["window_count"] == 7
    # 恰好放得下一个窗口的短序列也直接可用
    short, sv = _series(80)
    r2 = split_walk_forward_windows(short, sv, max(short))
    assert r2["window_count"] >= 1
    assert r2["train_window"] + r2["test_window"] <= 80
    # 再短一点（默认窗放不下）→ 显式拒绝，不缩放
    tiny, tv = _series(40)
    with pytest.raises(ComputeError, match=r"holds 40 bar\(s\)"):
        split_walk_forward_windows(tiny, tv, max(tiny))


def test_explicit_window_larger_than_series_is_refused():
    """显式传入的窗口放不下 → 报错点名所有相关数字（不返回空表）。"""
    ds, vs = _series(50)
    with pytest.raises(ComputeError, match="shrink the windows or widen the range") as ei:
        split_walk_forward_windows(ds, vs, max(ds), 200, 20)
    msg = str(ei.value)
    assert "50 bar(s)" in msg and "train_window=200" in msg and "test_window=20" in msg


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, None])
def test_invalid_window_params_refused(bad):
    ds, vs = _series(300)
    with pytest.raises(ComputeError):
        split_walk_forward_windows(ds, vs, max(ds), bad, 20)


def test_bad_start_end_are_refused_with_dates():
    ds = qd.dates(200)
    with pytest.raises(ComputeError, match="start must be an ISO date"):
        split_walk_forward_windows(ds, [1.0] * 200, ds[-1], 60, 20, 20, start="2025/01/01")
    with pytest.raises(ComputeError, match="is after end"):
        split_walk_forward_windows(ds, [1.0] * 200, ds[-1], 60, 20, 20,
                                   start=ds[100], end=ds[50])
    with pytest.raises(ComputeError, match="after the last date on the axis"):
        split_walk_forward_windows(ds, [1.0] * 200, ds[-1], 60, 20, 20, start="2030-01-01")
    with pytest.raises(ComputeError, match="before the first date on the axis"):
        split_walk_forward_windows(ds, [1.0] * 200, ds[-1], 60, 20, 20, end="2020-01-01")


# ── spec Scenario：注入的未来函数被抓 ────────────────────────────


def test_validate_no_lookahead_passes_on_legal_series():
    """spec:对照面——合法序列（全部不晚于 as_of）通过，且回带口径。

    注意本工具**不接受**轴上的越界点（那正是它要抓的东西），因此"合法输入"本身就是
    已被剪到 as_of 的序列：传 50 个点、as_of=末位即合法；若传 50 个点而 as_of 在第 30 位，
    那就是注入的违法夹具，必须报错（见下一个用例）。
    """
    ds = qd.dates(50)
    values = [float(i) for i in range(50)]
    r = validate_no_lookahead(ds, values, ds[-1])
    assert r["ok"] is True
    assert r["declared_as_of"] == ds[-1]
    assert r["point_count"] == 50 and r["points_checked"] == 50
    assert r["future_points"] == 0 and r["lookahead_points"] == 0
    assert r["first_future_date"] is None and r["first_future_index"] is None
    assert r["first_date"] == ds[0] and r["last_date"] == ds[-1]
    assert r["is_monotonic"] is True
    assert json.dumps(r, allow_nan=False)
    # 合法序列的全前缀在中间口径上也合法（as_of 是闭边界）
    mid = ds[29]
    r2 = validate_no_lookahead(ds[:30], values[:30], mid)
    assert r2["ok"] is True and r2["point_count"] == 30 and r2["last_date"] == mid


def test_as_of_equal_to_last_date_passes():
    """as_of 取到序列末位（含）合法——边界是闭的。"""
    ds, vs = _series(30)
    r = validate_no_lookahead(ds, vs, ds[-1])
    assert r["ok"] is True and r["last_date"] == ds[-1] and r["point_count"] == 30


def test_span_and_monotonic_reported():
    ds, vs = _series(10, start="2025-01-01")
    r = validate_no_lookahead(ds, vs, max(ds))
    assert r["is_monotonic"] is True
    assert r["span_days"] == 9  # 10 个自然日连排，首末相差 9 天
    assert r["first_date"] == "2025-01-01" and r["last_date"] == "2025-01-10"


def test_injected_future_data_is_caught_with_date_and_index():
    """spec Scenario:「对一个在 as_of 之后仍含数据点的夹具运行验证器 → 报错并指明越界日期
    （不通过）」；tasks 7.2 同义。这里额外断言**下标**也被点名（错误必须可操作）。"""
    ds, values, declared_as_of, future_indices = _injected_future_fixture()
    with pytest.raises(ComputeError) as ei:
        validate_no_lookahead(ds, values, declared_as_of)
    msg = str(ei.value)
    first = future_indices[0]
    assert ds[first] in msg                       # 首个越界日期
    assert f"index {first}" in msg                # 以及它的下标
    assert declared_as_of in msg                  # 声明的口径
    assert ds[future_indices[-1]] in msg          # 最新越界日期
    assert "lookahead detected" in msg
    assert str(len(future_indices)) in msg        # 违规点数


def test_future_data_is_refused_and_first_date_named():
    """越界必须报错，且点名**第一个**越界日期（调用方要剪掉的就是那一位）。"""
    ds, vs = _series(40)
    with pytest.raises(ComputeError) as ei:
        validate_no_lookahead(ds, vs, ds[10])
    msg = str(ei.value)
    assert ds[11] in msg, f"错误未点名首个越界日期 {ds[11]}"
    assert ds[10] in msg, "错误应回显声明的 as_of（ds[10]）以便对照"
    assert "index 11" in msg, "错误应给出首个越界点的下标"
    assert "No future data is dropped silently" in msg, "错误应承诺不静默丢弃（全仓纪律）"


def test_error_message_offers_both_ways_out():
    """护栏错误必须可操作：给出"剪输入"与"抬 as_of"两条出路，且不声称已静默截断。"""
    ds, values, declared_as_of, future_indices = _injected_future_fixture()
    with pytest.raises(ComputeError) as ei:
        validate_no_lookahead(ds, values, declared_as_of)
    msg = str(ei.value)
    assert f"slice the input to points <= {declared_as_of}" in msg
    latest = ds[future_indices[-1]]
    assert f"raise as_of to >= {latest}" in msg
    assert "No future data is dropped silently" in msg


def test_validator_does_not_modify_the_input():
    """设计 D7：验证器只做检查、**不改数据**（输入对象的每个元素都保持原样）。"""
    ds, values, declared_as_of, _ = _injected_future_fixture()
    ds_copy, values_copy = list(ds), list(values)
    with pytest.raises(ComputeError):
        validate_no_lookahead(ds, values, declared_as_of)
    assert ds == ds_copy and values == values_copy
    legal = qd.dates(20)
    r = validate_no_lookahead(legal, [1.0] * 20, legal[-1])
    assert legal == qd.dates(20) and r["ok"] is True


def test_point_as_of_declares_per_point_information_cutoff():
    """泄漏的可检形态：点的日期合法，但它的值用了更晚才可能知道的信息 → 报错并点名。

    这是"在 as_of 时刻使用了未来数据"的直接形态（不是"轴上有未来点"，而是"该点的
    信息口径晚于 as_of"）——回测引擎靠它抓"滚动窗口里混进了修订后的财务数"这类泄漏。
    """
    ds = qd.dates(30)
    values = [float(i) for i in range(30)]
    declared = ds[19]
    # 全部等于 as_of 的逐点口径（无泄漏）→ 通过
    ok = validate_no_lookahead(ds[:20], values[:20], declared, point_as_of=[declared] * 20)
    assert ok["ok"] is True
    # 让 index 17 的点"用了 as_of 之后的信息"
    leaky = [ds[i] for i in range(20)]
    leaky[17] = ds[25]
    with pytest.raises(ComputeError) as ei:
        validate_no_lookahead(ds[:20], values[:20], declared, point_as_of=leaky)
    msg = str(ei.value)
    assert ds[17] in msg and "index 17" in msg and ds[25] in msg
    assert "point_as_of" in msg and "lookahead detected" in msg


def test_point_as_of_length_must_match():
    ds = qd.dates(10)
    with pytest.raises(ComputeError, match="point_as_of length 5 != series length 10"):
        validate_no_lookahead(ds, [1.0] * 10, ds[-1], point_as_of=ds[:5])
    with pytest.raises(ComputeError, match=r"point_as_of\[2\] is not an ISO date"):
        validate_no_lookahead(ds, [1.0] * 10, ds[-1], point_as_of=["2025-01-01"] * 2 + ["bad"] + [ds[-1]] * 7)


def test_name_labels_the_series_that_offended():
    """多序列调用时错误文本能分辨是哪一条（`name` 进文本与 params）。"""
    ds = qd.dates(10)
    with pytest.raises(ComputeError, match="factor_a"):
        validate_no_lookahead(ds, [1.0] * 10, ds[4], name="factor_a")
    r = validate_no_lookahead(ds[:5], [1.0] * 5, ds[4], name="factor_a")
    assert r["params"]["name"] == "factor_a"
    with pytest.raises(ComputeError, match="non-empty string"):
        validate_no_lookahead(ds[:5], [1.0] * 5, ds[4], name="  ")


def test_missing_values_are_counted_not_treated_as_lookahead():
    """缺失不是未来数据：null 只计数，不报错（护栏不该把"没有数"当成"偷看未来"）。"""
    ds = qd.dates(20)
    values = qd.with_nan([float(i) for i in range(20)], (3, 7, 8))
    r = validate_no_lookahead(ds, values, ds[-1])
    assert r["ok"] is True and r["missing_points"] == 3


@pytest.mark.parametrize("bad_dates", [
    ["2025-01-02", "2025-01-01"],                    # 递减
    ["2025-01-01", "2025-01-01"],                    # 重复（非严格递增）
    ["2025-01-01", "not-a-date"],                    # 非日期
    ["2025-01-01", "2025-1-2"],                      # 非 ISO 补零形态
])
def test_illegal_date_sequences_refused(bad_dates):
    with pytest.raises(ComputeError):
        validate_no_lookahead(bad_dates, [1.0] * len(bad_dates), "2025-12-31")


# ── check_series_as_of_bounds ────────────────────────────────────


def test_bounds_reports_missing_and_warmup():
    ds, _ = _series(12)
    vs = [None, None, None] + [float(i) for i in range(3, 12)]
    r = check_series_as_of_bounds(ds, vs, max(ds))
    assert r["point_count"] == 12
    assert r["missing_count"] == 3
    assert r["warmup"] == 3, "warmup = 首个非缺失点之前有几个 null"
    assert r["is_complete"] is False
    assert abs(r["missing_ratio"] - 3 / 12) < 1e-12


def test_bounds_complete_series():
    ds, vs = _series(8)
    r = check_series_as_of_bounds(ds, vs, max(ds))
    assert r["is_complete"] is True and r["missing_count"] == 0 and r["warmup"] == 0


def test_bounds_all_missing_gives_null_warmup():
    ds, _ = _series(5)
    r = check_series_as_of_bounds(ds, [None] * 5, max(ds))
    assert r["warmup"] is None, "全缺失时 warmup 无定义，应为 null 而非 0"
    assert r["missing_ratio"] == 1.0


def test_bounds_refuses_future_like_peers():
    ds, vs = _series(20)
    with pytest.raises(ComputeError, match="after as_of"):
        check_series_as_of_bounds(ds, vs, ds[5])


# ── 拒绝语义（跨工具） ───────────────────────────────────────────


@pytest.mark.parametrize("fn", [split_walk_forward_windows, validate_no_lookahead,
                               check_series_as_of_bounds])
def test_missing_as_of_is_refused(fn):
    ds = qd.dates(20)
    with pytest.raises(ComputeError, match="as_of"):
        fn(ds, [1.0] * 20, None)


@pytest.mark.parametrize("fn", [split_walk_forward_windows, validate_no_lookahead,
                               check_series_as_of_bounds])
def test_length_mismatch_and_empty_are_refused(fn):
    ds = qd.dates(20)
    with pytest.raises(ComputeError, match="length"):
        fn(ds, [1.0] * 19, ds[-1])
    with pytest.raises(ComputeError, match="empty"):
        fn([], [], ds[-1])


@pytest.mark.parametrize("fn", [split_walk_forward_windows, validate_no_lookahead,
                               check_series_as_of_bounds])
def test_unsorted_dates_are_refused(fn):
    ds = list(qd.dates(20))
    ds[5], ds[6] = ds[6], ds[5]
    with pytest.raises(ComputeError, match="strictly increasing"):
        fn(ds, [1.0] * 20, max(ds))


def test_non_numeric_values_are_refused():
    ds = qd.dates(20)
    with pytest.raises(ComputeError, match=r"values\[3\] is not numeric"):
        validate_no_lookahead(ds, [1.0] * 3 + ["x"] + [1.0] * 16, ds[-1])


# ── 外壳 / evidence / 确定性 ─────────────────────────────────────


def test_tools_are_declared_with_the_validation_group():
    names = [t.name for t in TOOLS]
    assert names == ["split_walk_forward_windows", "split_holdout", "validate_no_lookahead",
                     "check_observation_sufficiency", "check_series_as_of_bounds"]
    for t in TOOLS:
        assert t.group == "validation"
        assert t.name.split("_")[0] in {"split", "validate", "check"}


def test_envelope_reports_the_axis_it_split():
    """切分器外壳回带的是**轴本身**的起止与长度（真实数字，不是 None）。"""
    ds = qd.dates(200)
    r = split_walk_forward_windows(ds, [1.0] * 200, ds[-1], 60, 20, 20)
    assert r["as_of"] == ds[-1]
    assert r["input_start"] == ds[0] and r["input_end"] == ds[-1]
    assert r["input_length"] == 200
    assert isinstance(r["warmup"], dict) and r["warmup"] == {}
    ids = [e["evidence_id"] for e in r["evidence"]]
    assert ids and len(ids) == len(set(ids))
    for e in r["evidence"]:
        assert set(e) == {"evidence_id", "label", "value", "unit", "as_of", "source"}
        assert e["as_of"] == ds[-1]
        assert "split_walk_forward_windows" in e["source"]


def test_envelope_reports_the_validated_series():
    ds = qd.dates(20)
    r = validate_no_lookahead(ds, [1.0] * 20, ds[-1])
    assert r["input_start"] == ds[0] and r["input_end"] == ds[-1]
    assert r["input_length"] == 20
    assert all("validate_no_lookahead" in e["source"] for e in r["evidence"])
    assert r["evidence"]


@pytest.mark.parametrize("fn,args", [
    (split_walk_forward_windows, "windows"),
    (validate_no_lookahead, "ok"),
    (check_series_as_of_bounds, "point_count"),
])
def test_bitwise_determinism(fn, args):
    """spec 1.1：同输入两次调用逐位一致（本族不做浮点归约，确定性是构造性的）。"""
    ds, vs = _series(120)
    kw = {"as_of": max(ds)}
    if fn is split_walk_forward_windows:
        kw.update(train_window=60, test_window=20, step=20)
    assert_bitequal(fn(ds, vs, **kw), fn(ds, vs, **kw), what=fn.__name__)


@pytest.mark.parametrize("fn", [
    split_walk_forward_windows, validate_no_lookahead, check_series_as_of_bounds,
])
def test_envelope_and_json_safe(fn):
    ds, vs = _series(120)
    kw = {"as_of": max(ds)}
    if fn is split_walk_forward_windows:
        kw.update(train_window=60, test_window=20, step=20)
    r = fn(ds, vs, **kw)
    assert r["as_of"] == kw["as_of"]
    assert r["input_start"] == ds[0] and r["input_end"] == ds[-1]
    assert r["input_length"] == len(ds)
    assert r["evidence"], "校验族也必须带 evidence（溯源不能因为不产生数值就省掉）"
    json.dumps(r, allow_nan=False)  # 契约：恒可 json.dumps


def test_no_randomness_in_module():
    """本族只做口径与结构校验：源码里不出现随机源（确定性是构造性的）。"""
    import fd_quant.compute.validation as mod

    src = open(mod.__file__, encoding="utf-8").read()
    for banned in ("np.random", "numpy.random", "import random", "from random"):
        assert banned not in src, f"unexpected in validation module: {banned}"


# ── 窗口条目别名键 / coverage（新增键的规范面） ───────────────────


def test_splitter_works_from_the_date_axis_alone():
    """切分只依赖日期轴：`values=None` 时用恒等数列走 `check_series`，结果与传数值一致。

    切分是下标结构（窗口 = 下标区间），数值不参与——但日期形状与越界口径仍然必须走
    全仓同一套 `check_series`（ISO / 严格递增 / 越界点名首个日期），否则本族会悄悄
    长出自己的第二套日期规则。
    """
    ds, vs = _series(200)
    with_values = split_walk_forward_windows(ds, vs, max(ds), 60, 20, 20)
    bare = split_walk_forward_windows(ds, None, max(ds), 60, 20, 20)
    assert bare["windows"] == with_values["windows"]
    assert bare["window_count"] == with_values["window_count"] == 7
    assert bare["input_length"] == 200 and bare["input_start"] == ds[0]
    # 越界日期即便没有 values 也被抓（护栏不因省了数值而失效）
    with pytest.raises(ComputeError) as ei:
        split_walk_forward_windows(ds, None, ds[150], 60, 20, 20)
    assert ds[151] in str(ei.value)
    # 非严格递增的日期轴同样被抓（日期规则与 check_series 逐字一致）
    bad = list(ds)
    bad[5], bad[6] = bad[6], bad[5]
    with pytest.raises(ComputeError, match="strictly increasing"):
        split_walk_forward_windows(bad, None, max(bad), 60, 20, 20)
    assert bare["params"]["short_sample"] == "refuse"


def test_window_entries_carry_both_key_spellings_with_equal_values():
    """窗口条目同时带规范键（`*_i` / 裸日期）与别名键（`*_idx` / `*_date` / `window`）。

    两套键名由同一处代码同时写入，故值必然相等；调用方按自己习惯取用，不会出现
    "两套值不一致"这种需要判断的情形。
    """
    ds, vs = _series(200)
    r = split_walk_forward_windows(ds, vs, max(ds), 60, 20, 20)
    for w in r["windows"]:
        assert w["window"] == w["index"]
        assert w["train_start_idx"] == w["train_start_i"]
        assert w["train_end_idx"] == w["train_end_i"]
        assert w["test_start_idx"] == w["test_start_i"]
        assert w["test_end_idx"] == w["test_end_i"]
        assert w["train_start_date"] == w["train_start"] == ds[w["train_start_idx"]]
        assert w["train_end_date"] == w["train_end"] == ds[w["train_end_idx"]]
        assert w["test_start_date"] == w["test_start"] == ds[w["test_start_idx"]]
        assert w["test_end_date"] == w["test_end"] == ds[w["test_end_idx"]]


def test_coverage_counts_observations_covered_by_any_test_window():
    """coverage = 被任一测试窗覆盖的观测数（重叠的测试窗按并集计一次）。

    注意 coverage **不等于** point_count - unassigned_tail：窗口 0 的训练段（这里
    下标 0..59）也既不在测试窗里、也不在尾部。三者关系是
    `coverage + 首个测试窗之前的训练观测数 + unassigned_tail == point_count`。
    """
    ds, vs = _series(200)
    # 无缝平铺：测试窗 [60,79] [80,99] ... [180,199] → 并集 140 个观测
    a = split_walk_forward_windows(ds, vs, max(ds), 60, 20, 20)
    assert a["coverage"] == 140
    assert a["coverage_ratio"] == 140 / 200
    first_test_start = min(w["test_start_i"] for w in a["windows"])
    assert first_test_start == 60
    assert a["coverage"] + first_test_start + a["unassigned_tail"] == a["point_count"]
    # step=10 < test_window=20：测试窗两两重叠，coverage 是并集而不是各窗之和
    b = split_walk_forward_windows(ds, vs, max(ds), 60, 20, 10)
    naive_sum = sum(w["test_size"] for w in b["windows"])
    assert naive_sum > b["coverage"], "重叠场景下朴素求和必须大于并集（否则本条没测到东西）"
    assert b["coverage"] == max(w["test_end_i"] for w in b["windows"]) - min(
        w["test_start_i"] for w in b["windows"]) + 1


# ── 短样本策略：默认拒绝 / 显式 empty ────────────────────────────


def test_short_sample_refuses_by_default_and_answers_when_asked():
    """短样本默认拒绝（不缩放、不返回空表），显式 `short_sample="empty"` 则回答案而非异常。

    "这个范围能切几个窗"是个合法问题，答案可以是"0 个，因为…"——但**必须是显式选择**：
    默认静默返回空表会让回测安静地跑 0 个窗口，那比报错危险得多。
    """
    ds, vs = _series(40)
    with pytest.raises(ComputeError) as ei:
        split_walk_forward_windows(ds, vs, max(ds), 60, 20)
    assert "40 bar(s)" in str(ei.value)

    r = split_walk_forward_windows(ds, vs, max(ds), 60, 20, short_sample="empty")
    assert r["windows"] == [] and r["window_count"] == 0 and r["n_windows"] == 0
    assert isinstance(r["reason"], str) and r["reason"]
    assert "40 bar(s)" in r["reason"], "reason 必须点名成因数字（否则调用方不知道改哪个参数）"
    assert "train_window=60" in r["reason"] and "test_window=20" in r["reason"]
    assert r["coverage"] == 0 and r["coverage_ratio"] == 0.0
    assert r["unassigned_tail"] == 40 and r["point_count"] == 40
    assert r["coverage_start"] is None and r["coverage_end"] is None
    assert r["params"]["short_sample"] == "empty"
    json.dumps(r, allow_nan=False)
    # 空表外壳的键集与正常路径一致：调用方不必按"有没有窗口"分支处理两套形状
    assert set(r) == set(split_walk_forward_windows(*_series(200), max(_series(200)[0]),
                                                   60, 20, 20))
    # 非法取值点名两个合法选项
    with pytest.raises(ComputeError, match="short_sample must be"):
        split_walk_forward_windows(ds, vs, max(ds), 60, 20, short_sample="silent")


def test_default_short_sample_is_refuse():
    """默认值必须是 "refuse"（静默空表是危险的默认）。"""
    ds, vs = _series(200)
    r = split_walk_forward_windows(ds, vs, max(ds), 60, 20, 20)
    assert r["params"]["short_sample"] == "refuse"
    assert r["reason"] is None, "窗口非空时 reason 恒为 null（不是空字符串）"


# ── 截断策略：放不下的尾部被丢弃，不产出半窗 ─────────────────────


def test_truncation_drops_the_partial_tail_window():
    """截断选择：末尾放不下完整"训练+测试"的观测被**丢弃**（绝不产出半窗）。

    具体：101 个观测 / 60 / 20 / 20 → 窗口 0（训练 0..59 / 测试 60..79）、窗口 1
    （训练 20..79 / 测试 80..99）；下标 100 那个观测凑不出第三窗（需要到下标 140），
    于是被**丢弃**并点名为 unassigned_tail=1。半窗（训练 60 + 测试不足 20）不产出：
    那会让"每个窗口的 test_size 都等于 test_window"这条性质失效。
    """
    ds, vs = _series(101)
    r = split_walk_forward_windows(ds, vs, max(ds), 60, 20, 20)
    assert r["window_count"] == 2, "第三窗放不下 → 整个丢弃（不是拿剩下的 1 个观测凑半窗）"
    for w in r["windows"]:
        assert w["train_size"] == 60 and w["test_size"] == 20, "截断后仍无半窗"
        assert w["test_end_i"] <= 100, "窗口不得越出样本"
    assert (r["windows"][1]["train_start_idx"], r["windows"][1]["test_end_idx"]) == (20, 99)
    assert r["unassigned_tail"] == 101 - 100 == 1, "丢弃的尾部观测必须被点名"
    assert r["coverage"] == 40, "两个测试窗 [60,79] [80,99] 的并集"
    # 被丢弃的观测不属于任何窗口（既不在训练窗也不在测试窗）
    covered = {i for w in r["windows"]
               for i in range(w["train_start_idx"], w["test_end_idx"] + 1)}
    assert 100 not in covered
    # 尾部从 1 缩到 0（101 → 100）：窗口数不变，tail 归零
    ds2, vs2 = _series(100)
    r2 = split_walk_forward_windows(ds2, vs2, max(ds2), 60, 20, 20)
    assert r2["window_count"] == 2 and r2["unassigned_tail"] == 0


def test_exact_train_plus_test_yields_exactly_one_window():
    """边界：恰好 `train_window + test_window` 个观测 → 恰好一个窗口（不早退、不报错）。"""
    for n, tw, tew in ((80, 60, 20), (75, 60, 15), (50, 25, 25), (2, 1, 1)):
        ds = qd.dates(n)
        r = split_walk_forward_windows(ds, [1.0] * n, ds[-1], tw, tew)
        assert r["window_count"] == 1, f"{n} 个观测 / {tw}+{tew} 应恰好一窗"
        w = r["windows"][0]
        assert w["train_start_idx"] == 0
        assert w["train_end_idx"] == tw - 1
        assert w["test_start_idx"] == tw
        assert w["test_end_idx"] == n - 1
        assert w["test_start_idx"] == w["train_end_idx"] + 1
        assert r["unassigned_tail"] == 0 and r["coverage"] == tew
    # 再少一个观测 → 放不下（默认拒绝）
    ds = qd.dates(79)
    with pytest.raises(ComputeError, match="holds 79 bar"):
        split_walk_forward_windows(ds, [1.0] * 79, ds[-1], 60, 20)


def test_window_count_formula_matches_hand_computation():
    """窗口数 = (参与切分的 bar 数 - train_window - test_window) // step + 1（逐例手算）。"""
    for n, tw, tew, st, expected in (
        (200, 60, 20, 20, 7),     # (200-80)//20 + 1 = 6 + 1
        (200, 60, 20, 10, 13),    # (200-80)//10 + 1 = 12 + 1
        (200, 60, 20, 30, 5),     # (200-80)//30 + 1 = 4 + 1
        (100, 50, 20, 30, 2),     # (100-70)//30 + 1 = 1 + 1
        (80, 60, 20, 1, 1),       # (80-80)//1 + 1  = 0 + 1
        (121, 60, 20, 20, 3),     # (121-80)//20 + 1 = 2 + 1
        (1000, 252, 63, 63, 11),  # (1000-315)//63 + 1 = 10 + 1（本族默认口径）
    ):
        ds = qd.dates(n)
        r = split_walk_forward_windows(ds, [1.0] * n, ds[-1], tw, tew, st)
        hand = (n - tw - tew) // st + 1
        assert hand == expected, f"手算有误：n={n} {tw}/{tew}/{st}"
        assert r["window_count"] == expected, f"n={n} {tw}/{tew}/{st}"
        assert len(r["windows"]) == expected
        assert [w["window"] for w in r["windows"]] == list(range(expected))
        assert [w["train_start_idx"] for w in r["windows"]] == [
            k * st for k in range(expected)]


# ── validate_no_lookahead：无 values / declared_as_of / windows ──


def test_no_lookahead_accepts_a_date_axis_alone():
    """只给日期轴即可校验（值不是必需的）：值缺省时不做数值形状检查，空值不计数。"""
    ds = qd.dates(50)
    r = validate_no_lookahead(ds, None, ds[-1])
    assert r["ok"] is True and r["values_provided"] is False
    assert r["point_count"] == 50 and r["points_checked"] == 50 == r["checked_points"]
    assert r["missing_points"] == 0
    assert r["input_length"] == 50
    json.dumps(r, allow_nan=False)
    # 越界日期即便没有 values 也要抓（护栏不因省了数值而失效）
    with pytest.raises(ComputeError) as ei:
        validate_no_lookahead(ds, None, ds[10])
    assert ds[11] in str(ei.value)


def test_declared_as_of_is_the_tighter_cutoff():
    """`declared_as_of`（调用方**声称**的口径日）与 `as_of` 取更早者作有效截止日。

    声称的截止日更早时，夹在两者之间的点仍是"声称口径之后的数据"——用工具的 as_of
    盖过它等于替调用方把声称改成事实。
    """
    ds = qd.dates(30)
    vs = [1.0] * 30
    r = validate_no_lookahead(ds[:20], vs[:20], ds[19], declared_as_of=ds[19])
    assert r["ok"] is True
    assert r["declared_as_of"] == ds[19] and r["requested_as_of"] == ds[19]
    # 声称口径更早 → 该口径之后的点被拒（即便没超过工具自己的 as_of）
    with pytest.raises(ComputeError) as ei:
        validate_no_lookahead(ds[:20], vs[:20], ds[19], declared_as_of=ds[9])
    msg = str(ei.value)
    assert ds[10] in msg and ds[9] in msg, "错误应点名首个越界日期与生效的截止日"
    # 工具 as_of 更早时，不得被声称口径抬高
    with pytest.raises(ComputeError) as ei2:
        validate_no_lookahead(ds[:20], vs[:20], ds[9], declared_as_of=ds[19])
    assert ds[10] in str(ei2.value)
    with pytest.raises(ComputeError, match="declared_as_of must be an ISO date"):
        validate_no_lookahead(ds[:5], vs[:5], ds[4], declared_as_of="2025/01/01")


def test_headroom_days_measures_observation_to_cutoff_distance():
    """headroom_days = 末个观测到有效截止日的自然日数（0 = 数据正好用到截止日）。

    这是"这段数据离口径日还有多远"的读数：本工具不因"数据比口径旧"报错（过期的点
    不是未来数据），但把距离报出来，调用方才能判断新鲜度。as_of 是**闭边界**，故
    观测日 == 截止日时 headroom = 0 且合法。
    """
    ds, vs = _series(50)
    assert validate_no_lookahead(ds, vs, ds[-1])["headroom_days"] == 0
    # 数据到 ds[30] 为止、口径日是 ds[40] → 富余 10 天（数据更旧，合法）
    r = validate_no_lookahead(ds[:31], vs[:31], ds[40])
    assert r["ok"] is True and r["last_date"] == ds[30] and r["declared_as_of"] == ds[40]
    assert r["headroom_days"] == 10
    assert r["span_days"] == 30
    # 口径日在轴之外（数据全部更旧）也合法：富余 = 末位到该日期的自然日数
    full = validate_no_lookahead(ds, vs, "2025-03-31")
    assert full["headroom_days"] == 40, "ds[49]=2025-02-19 距 2025-03-31 是 40 天"
    assert full["ok"] is True and full["last_date"] == ds[-1]
    # declared_as_of 与 as_of 相同时，headroom 仍以有效截止日为基准
    tight = validate_no_lookahead(ds[:31], vs[:31], ds[30], declared_as_of=ds[30])
    assert tight["headroom_days"] == 0
    # headroom 也进 evidence，可被上游记录
    assert any(e["label"] == "headroom_days" for e in r["evidence"])


def test_no_lookahead_verifies_a_window_plan():
    """`windows` 入参：同一个工具既验原始序列，也逐窗验窗口计划（测试区间严格晚于训练区间）。"""
    ds, vs = _series(200)
    plan = split_walk_forward_windows(ds, vs, max(ds), 60, 20, 20)
    r = validate_no_lookahead(ds, vs, max(ds), windows=plan["windows"])
    assert r["ok"] is True
    assert r["checked_windows"] == 7
    assert r["window_invariant"] == "test_strictly_after_train"
    assert r["params"]["windows_checked"] == 7
    # 别名键（*_idx / *_date）也要被认——调用方直接回传切分器输出即可
    aliased = [{k: w[k] for k in ("train_start_idx", "train_end_idx",
                                 "test_start_idx", "test_end_idx")} for w in plan["windows"]]
    assert validate_no_lookahead(ds, vs, max(ds), windows=aliased)["checked_windows"] == 7
    dated = [{k: w[k] for k in ("train_start_date", "train_end_date",
                                "test_start_date", "test_end_date")} for w in plan["windows"]]
    assert validate_no_lookahead(ds, vs, max(ds), windows=dated)["checked_windows"] == 7
    # 未传 windows → 该键为 null（不是 0/False 这种看起来像"检查过了"的值）
    plain = validate_no_lookahead(ds, vs, max(ds))
    assert plain["checked_windows"] == 0 and plain["window_invariant"] is None


def test_leaky_window_plan_is_caught():
    """注入的泄漏窗口计划：测试区间与训练区间重叠 → 报错点名窗口序号与两个区间。"""
    ds, vs = _series(200)
    plan = split_walk_forward_windows(ds, vs, max(ds), 60, 20, 20)
    bad = [dict(plan["windows"][0])]
    bad[0]["test_start_i"] = bad[0]["train_end_i"]        # 测试窗从训练窗末位开始
    bad[0]["test_start_idx"] = bad[0]["train_end_i"]
    with pytest.raises(ComputeError) as ei:
        validate_no_lookahead(ds, vs, max(ds), windows=bad)
    msg = str(ei.value)
    assert "windows[0]" in msg and "lookahead in the window plan" in msg
    assert "test_start=59" in msg and "train_end=59" in msg
    # 完全重叠（测试窗 == 训练窗）同样被抓
    overlap = [dict(plan["windows"][0])]
    overlap[0].update(test_start_i=0, test_start_idx=0, test_end_i=59, test_end_idx=59)
    with pytest.raises(ComputeError, match="lookahead in the window plan"):
        validate_no_lookahead(ds, vs, max(ds), windows=overlap)


def test_window_plan_off_the_axis_and_malformed_entries_are_refused():
    """窗口计划的坏输入：边界日期不在被检轴上 / 条目不是 dict / 缺键 / 边界倒置 → 全部报错点名。

    "边界日期不在轴上"是关键一条：窗口若描述的是另一段历史，这份计划根本无从复核——
    静默放过等于给调用方一个"检查通过"的假信号。
    """
    ds, vs = _series(120)
    plan = split_walk_forward_windows(ds, vs, max(ds), 60, 20, 20)
    # 只有日期拼法、且日期不在被检轴上（下标拼法全部缺席，无从归一）
    off_axis = [{"train_start_date": "2024-01-01", "train_end_date": ds[59],
                 "test_start_date": ds[60], "test_end_date": ds[79]}]
    with pytest.raises(ComputeError, match="not on the validated series axis"):
        validate_no_lookahead(ds, vs, max(ds), windows=off_axis)
    # 条目不是 dict
    with pytest.raises(ComputeError, match="must be a dict"):
        validate_no_lookahead(ds, vs, max(ds), windows=["not-a-dict"])
    # 缺键：四个边界一个都没有（错误文本列出可接受的拼法）
    with pytest.raises(ComputeError, match=r"missing train_start_i") as ei:
        validate_no_lookahead(ds, vs, max(ds), windows=[{"window": 0}])
    assert "train_start_idx" in str(ei.value) and "train_start_date" in str(ei.value)
    # 边界倒置（训练起点晚于终点）
    with pytest.raises(ComputeError, match="boundaries are inverted"):
        validate_no_lookahead(ds, vs, max(ds), windows=[{
            "train_start_i": 10, "train_end_i": 5, "test_start_i": 6, "test_end_i": 20}])
    # 布尔不是下标（bool 是 int 的子类，必须显式拒绝）
    with pytest.raises(ComputeError, match="must be an int or ISO date"):
        validate_no_lookahead(ds, vs, max(ds), windows=[{
            "train_start_i": True, "train_end_i": 59, "test_start_i": 60,
            "test_end_i": 79}])


# ── check_observation_sufficiency ────────────────────────────────


def test_sufficiency_reports_enough_and_short():
    """充分性闸门：够 → sufficient=true 且 shortfall=0；不够 → 点名还差多少。"""
    ds, vs = _series(300)
    r = check_observation_sufficiency(ds, max(ds), vs)
    assert r["sufficient"] is True
    assert r["usable_observations"] == 300 == r["observation_count"]
    assert r["required_observations"] == 252, "默认门槛 = 252（约一个交易年）"
    assert r["shortfall"] == 0 and r["surplus"] == 48
    assert r["reason"] is None
    assert r["first_date"] == ds[0] and r["last_date"] == ds[-1] and r["span_days"] == 299
    assert all(e["evidence_id"] != "" for e in r["evidence"])
    assert any(e["label"] == "sufficient" for e in r["evidence"])

    sds, svs = _series(30)
    short = check_observation_sufficiency(sds, max(sds), svs)
    assert short["sufficient"] is False
    assert short["usable_observations"] == 30 and short["shortfall"] == 222
    assert "30 usable" in short["reason"] and "252" in short["reason"]
    assert short["surplus"] == 0 and short["ratio"] == 30 / 252
    json.dumps(short, allow_nan=False)


def test_sufficiency_counts_only_non_null_points():
    """缺失点不参与统计：300 个点里 100 个 null 不算"够 252 个观测"。"""
    ds = qd.dates(300)
    vs = [float(i) for i in range(300)]
    for i in range(0, 200, 2):          # 100 个 null
        vs[i] = None
    r = check_observation_sufficiency(ds, max(ds), vs)
    assert r["point_count"] == 300 and r["missing_count"] == 100
    assert r["usable_observations"] == 200
    assert r["sufficient"] is False and r["shortfall"] == 52
    assert "100 of 300 points are null" in r["reason"]
    # 不传 values → 按日期点数计（不假装知道缺失）
    r2 = check_observation_sufficiency(ds, max(ds))
    assert r2["usable_observations"] == 300 and r2["missing_count"] == 0
    assert r2["params"]["values_provided"] is False


def test_sufficiency_threshold_boundaries_and_bad_params():
    """门槛边界：恰好等于 required → 够；少一个 → 不够；required=0 → 恒够且 ratio=null。"""
    ds, vs = _series(10)
    assert check_observation_sufficiency(ds, max(ds), vs, 10)["sufficient"] is True
    assert check_observation_sufficiency(ds, max(ds), vs, 11)["shortfall"] == 1
    zero = check_observation_sufficiency(ds, max(ds), vs, 0)
    assert zero["sufficient"] is True and zero["ratio"] is None and zero["surplus"] == 10
    for bad in (-1, 1.5, True, "252", None):
        with pytest.raises(ComputeError, match="required_observations must be a non-negative"):
            check_observation_sufficiency(ds, max(ds), vs, bad)


# ── split_holdout ────────────────────────────────────────────────


def test_holdout_is_the_last_portion_with_hand_computed_size():
    """留出集恒为**最后一段**：100 个点切 20% → 训练 0..79、留出 80..99（手算）。"""
    ds, vs = _series(100)
    r = split_holdout(ds, max(ds), holdout_fraction=0.2)
    assert r["train_start_idx"] == 0 and r["train_end_idx"] == 79
    assert r["test_start_idx"] == 80 and r["test_end_idx"] == 99
    assert r["train_size"] == 80 and r["test_size"] == 20
    assert r["n_train"] == 80 and r["n_test"] == 20
    assert r["train_start_date"] == ds[0] and r["train_end_date"] == ds[79]
    assert r["test_start_date"] == ds[80] and r["test_end_date"] == ds[99]
    assert r["test_end_date"] == max(ds), "留出集必须到轴末位（最后一段，不是随机抽）"
    assert r["holdout_fraction"] == 0.2 and r["train_fraction"] == 0.8
    assert r["embargo"] == 0 and r["embargo_dropped"] == 0
    assert r["reason"] is None and r["warmup"] == {}
    json.dumps(r, allow_nan=False)
    # 训练区间与留出区间不重叠且留出在后
    assert r["train_end_idx"] < r["test_start_idx"]
    assert r["test_size"] < r["point_count"]
    # 占比越小留出越短；floor 用精确整数算术（0.2 × 100 恒为 20，不受 float 末位影响）
    for frac, expected in ((0.1, 10), (0.2, 20), (0.25, 25), (0.5, 50)):
        assert split_holdout(ds, max(ds), holdout_fraction=frac)["test_size"] == expected


def test_holdout_embargo_shifts_the_split_by_exactly_that_many():
    """embargo 的语义：训练与留出之间**丢弃** n 个观测（防重叠期泄露），留出起点不动。"""
    ds, vs = _series(100)
    base = split_holdout(ds, max(ds), vs, 0.2)
    for emb in (0, 1, 5, 20):
        r = split_holdout(ds, max(ds), vs, 0.2, emb)
        assert r["embargo"] == emb
        assert r["test_start_idx"] == base["test_start_idx"] == 80, "embargo 不动留出集"
        assert r["test_end_idx"] == base["test_end_idx"] == 99
        assert r["test_size"] == base["test_size"] == 20
        assert r["train_end_idx"] == 80 - emb - 1, "训练尾部恰好被切掉 embargo 个观测"
        assert r["train_size"] == 80 - emb
        assert r["embargo_dropped"] == emb
        assert r["train_size"] + emb + r["test_size"] == 100, "训练+丢弃+留出 = 全部观测"
        assert r["train_end_date"] == ds[r["train_end_idx"]]
        assert r["test_start_date"] == ds[80]
    # 丢弃的观测恰好位于训练末位与留出首位之间
    r = split_holdout(ds, max(ds), vs, 0.2, 3)
    assert ds[r["train_end_idx"] + 1:r["test_start_idx"]] == ds[77:80]
    assert (r["test_start_idx"] - r["train_end_idx"] - 1) == 3


def test_holdout_defaults_and_refusals():
    """默认 holdout_fraction=0.2；非法参数与切不动的占比都报错点名数字。"""
    ds, vs = _series(100)
    assert split_holdout(ds, max(ds))["test_size"] == 20, "默认 0.2"
    assert split_holdout(ds, max(ds))["params"]["holdout_fraction"] == 0.2
    for bad in (0.0, 1.0, -0.1, 1.5, float("nan"), float("inf"), True, "0.2"):
        with pytest.raises(ComputeError, match="holdout_fraction"):
            split_holdout(ds, max(ds), vs, bad)
    for bad in (-1, 1.5, True, "3"):
        with pytest.raises(ComputeError, match="embargo must be a non-negative integer"):
            split_holdout(ds, max(ds), vs, 0.2, bad)
    # 占比太小 → 留出集为空（floor 后 0 个点）→ 报错而不是给一个空留出集
    tds, tvs = _series(4)
    with pytest.raises(ComputeError, match=r"floor\(4 x 0\.2\) = 0"):
        split_holdout(tds, max(tds), tvs, 0.2)
    # embargo 吃掉全部训练观测 → 报错并给出可用的上界
    with pytest.raises(ComputeError, match="no training observations") as ei:
        split_holdout(ds, max(ds), vs, 0.2, 80)
    assert "reduce embargo" in str(ei.value) and "<= 79" in str(ei.value)


def test_holdout_embargo_dropped_is_capped_by_available_gap():
    """embargo 大于可用空隙时：embargo_dropped 报"实际丢弃数"，训练集缩到 1 个也不越界。"""
    ds, vs = _series(10)
    r = split_holdout(ds, max(ds), vs, 0.2, 7)     # 留出 2 个 → 训练能留 1 个（丢弃 7）
    assert r["test_start_idx"] == 8 and r["train_end_idx"] == 0
    assert r["train_size"] == 1 and r["embargo_dropped"] == 7
    with pytest.raises(ComputeError, match="no training observations"):
        split_holdout(ds, max(ds), vs, 0.2, 8)


# ── 注入的未来数据：每个工具都拒绝并点名首个越界日期 ─────────────


def test_injected_future_data_is_refused_by_every_date_axis_tool():
    """spec Scenario「对一个在 as_of 之后仍含数据点的夹具运行验证器 → 报错并指明越界日期」：
    本族**每一个**吃日期轴的工具都必须抓到它（护栏不能只装在其中一个入口）。"""
    ds, values, declared_as_of, future_indices = _injected_future_fixture()
    first, last = ds[future_indices[0]], ds[future_indices[-1]]
    calls = {
        "split_walk_forward_windows": lambda: split_walk_forward_windows(
            ds, values, declared_as_of, 60, 20, 20),
        "split_holdout": lambda: split_holdout(ds, declared_as_of, values, 0.2),
        "validate_no_lookahead": lambda: validate_no_lookahead(ds, values, declared_as_of),
        "check_observation_sufficiency": lambda: check_observation_sufficiency(
            ds, declared_as_of, values),
        "check_series_as_of_bounds": lambda: check_series_as_of_bounds(
            ds, values, declared_as_of),
        "validate_no_lookahead (bare axis)": lambda: validate_no_lookahead(
            ds, None, declared_as_of),
    }
    for label, call in calls.items():
        with pytest.raises(ComputeError) as ei:
            call()
        msg = str(ei.value)
        assert first in msg, f"{label}: 未点名首个越界日期 {first}"
        assert last in msg or str(len(future_indices)) in msg, (
            f"{label}: 未点名违规规模（最新越界日期或违例点数）"
        )
        assert declared_as_of in msg, f"{label}: 未回显声明的 as_of 以供对照"


# ── 不改数据（spec 设计 D7：验证器只做检查） ─────────────────────


def test_no_tool_mutates_its_input():
    """设计 D7 / spec「不改数据」：**每一个**工具跑完后输入列表逐位不变（含越界报错路径）。

    "改数据"的形态不止"换掉某个值"，还包括排序、截断、去重——本族全部工具都不做这些。
    深拷贝比较（`list(...)` 只浅拷贝，故再对窗口表用 JSON 快照比），逐位相等而非近似。
    """
    ds, values, declared_as_of, _ = _injected_future_fixture()
    ds_copy, values_copy = list(ds), list(values)

    r1 = split_walk_forward_windows(ds, values, ds[-1], 60, 20, 20)
    assert ds == ds_copy and values == values_copy
    assert json.dumps(r1, allow_nan=False)

    r2 = split_walk_forward_windows(ds[:40], values[:40], ds[39], 60, 20, 20,
                                    short_sample="empty")
    assert ds[:40] == ds_copy[:40] and values[:40] == values_copy[:40]
    assert json.dumps(r2, allow_nan=False)

    r3 = split_holdout(ds, ds[-1], values, 0.2, 3)
    assert ds == ds_copy and values == values_copy
    assert json.dumps(r3, allow_nan=False)

    r4 = validate_no_lookahead(ds, values, ds[-1])
    assert ds == ds_copy and values == values_copy
    assert json.dumps(r4, allow_nan=False)

    r5 = check_observation_sufficiency(ds, ds[-1], values)
    assert ds == ds_copy and values == values_copy
    assert json.dumps(r5, allow_nan=False)

    r6 = check_series_as_of_bounds(ds, values, ds[-1])
    assert ds == ds_copy and values == values_copy
    assert json.dumps(r6, allow_nan=False)

    # 越界报错路径同样不改输入（抛错也不能"顺手修剪"）
    for fn, args in ((split_walk_forward_windows, (ds, values, declared_as_of)),
                     (split_holdout, (ds, declared_as_of, values)),
                     (validate_no_lookahead, (ds, values, declared_as_of)),
                     (check_observation_sufficiency, (ds, declared_as_of, values)),
                     (check_series_as_of_bounds, (ds, values, declared_as_of))):
        with pytest.raises(ComputeError):
            fn(*args)
        assert ds == ds_copy and values == values_copy, f"{fn.__name__}: 报错路径修改了输入"

    # 窗口计划：校验 `windows` 也不得改动调用方手里的窗口 dict
    pds, pvs = _series(120)
    plan = split_walk_forward_windows(pds, pvs, max(pds), 60, 20, 20)
    snapshot = json.dumps(plan["windows"], sort_keys=True)
    validate_no_lookahead(pds, pvs, max(pds), windows=plan["windows"])
    assert json.dumps(plan["windows"], sort_keys=True) == snapshot
    # 数值也不得被原地改成 NaN / None（验证器不"修正"缺失）
    assert all(v is not None for v in values_copy)

# ── 确定性（新增工具逐一覆盖） ───────────────────────────────────


@pytest.mark.parametrize("fn,args", [
    (split_walk_forward_windows, ("dates_values_as_of", 60, 20, 20)),
    (split_holdout, ("dates_as_of_values", 0.2, 5)),
    (validate_no_lookahead, ("dates_values_as_of",)),
    (check_observation_sufficiency, ("dates_as_of_values",)),
])
def test_new_tools_are_bitwise_deterministic(fn, args):
    """确定性：同输入两次调用逐位一致（本族无浮点归约与随机源，故是构造性的）。"""
    ds, vs = _series(200)
    if args[0] == "dates_values_as_of":
        call = lambda: fn(ds, vs, max(ds), *args[1:])          # noqa: E731
    else:
        call = lambda: fn(ds, max(ds), vs, *args[1:])          # noqa: E731
    assert_bitequal(call(), call(), what=fn.__name__)


def test_new_tools_are_json_safe_and_evidence_bearing():
    """新工具同样带非空 evidence 与统一外壳（校验族也不能省溯源）。"""
    ds, vs = _series(300)
    for r in (split_walk_forward_windows(ds, vs, max(ds), 60, 20, 20),
              split_holdout(ds, max(ds), vs, 0.2, 5),
              validate_no_lookahead(ds, vs, max(ds)),
              check_observation_sufficiency(ds, max(ds), vs)):
        assert r["as_of"] == max(ds)
        assert r["input_start"] == ds[0] and r["input_end"] == ds[-1]
        assert r["input_length"] == 300
        assert r["evidence"] and len({e["evidence_id"] for e in r["evidence"]}) == len(
            r["evidence"])
        assert all(e["source"].startswith("fd-quant:") for e in r["evidence"])
        json.dumps(r, allow_nan=False)
