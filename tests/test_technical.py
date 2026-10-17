"""技术指标族测试：定义对拍、warmup 纪律、NaN 传播、确定性。

"对拍独立实现"在各用例内就地写一份递推（不调用被测代码的任何内部函数），
这是唯一能发现"实现与定义一起写错"的方式。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import quantdata as qd
from fd_quant.compute import base
from fd_quant.compute.technical import (
    compute_atr,
    compute_bollinger,
    compute_cci,
    compute_dmi,
    compute_ema,
    compute_kdj,
    compute_macd,
    compute_obv,
    compute_rsi,
    compute_sma,
)


def _finite(xs):
    return [x for x in xs if x is not None]


# ── SMA / EMA ────────────────────────────────────────────────────


def test_sma_warmup_is_null_then_value():
    """SMA(20) 于 100 日序列：前 19 位 null，第 20 位起为数值。"""
    n = 100
    ds = qd.dates(n)
    vals = [float(i) for i in range(n)]
    r = compute_sma(ds, vals, as_of=qd.as_of(n), window=20)
    out = r["value"]
    assert len(out) == n
    assert out[:19] == [None] * 19
    assert all(v is not None for v in out[19:])
    assert r["warmup"]["value"] == 19
    assert r["params"]["window"] == 20
    # 第 20 位 = 前 20 个数的均值
    assert out[19] == pytest.approx(sum(vals[:20]) / 20, abs=1e-12)


def test_sma_partial_window_shorter_than_input():
    """窗口长于序列：全 null，不报错。"""
    r = compute_sma(qd.dates(5), [1.0, 2, 3, 4, 5], as_of=qd.as_of(5), window=10)
    assert r["value"] == [None] * 5
    assert r["warmup"]["value"] == 5


def test_ema_seed_is_simple_mean_then_recursion():
    """EMA 对拍：种子 = 前 window 个简单均值，其后 a=2/(w+1)。"""
    n = 40
    w = 10
    ds = qd.dates(n)
    vals = [float(i % 7) + 1.0 for i in range(n)]
    r = compute_ema(ds, vals, as_of=qd.as_of(n), window=w)
    out = r["value"]
    assert out[: w - 1] == [None] * (w - 1)

    # 独立递推
    seed = sum(vals[:w]) / w
    expect = [seed]
    a = 2.0 / (w + 1.0)
    for x in vals[w:]:
        expect.append(a * x + (1 - a) * expect[-1])
    for got, exp in zip(out[w - 1:], expect):
        assert got == pytest.approx(exp, abs=1e-12, rel=1e-12)


# ── RSI ──────────────────────────────────────────────────────────


def test_rsi_monotone_up_approaches_100_with_wilder_recursion():
    """只涨序列 RSI 趋近 100，且与 Wilder 递推逐点一致。"""
    n = 60
    d = qd.monotone_up(n)
    r = compute_rsi(d["dates"], d["close"], as_of=qd.as_of(n), window=14)
    out = r["value"]
    assert out[:14] == [None] * 14
    assert out[14] is not None
    # 全涨：avg_loss = 0 → 定义上取 100
    assert all(v == pytest.approx(100.0) for v in _finite(out))

    # 对拍：构造带跌的序列，独立 Wilder 递推
    rng_vals = [10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 12.5, 13.0, 12.0, 13.5,
                14.0, 13.0, 14.5, 15.0, 14.0, 15.5, 16.0, 15.5, 16.5, 17.0]
    ds = qd.dates(len(rng_vals))
    got = compute_rsi(ds, rng_vals, as_of=qd.as_of(len(rng_vals)), window=14)["value"]

    gains, losses = [], []
    for i in range(1, len(rng_vals)):
        ch = rng_vals[i] - rng_vals[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    ag = sum(gains[:14]) / 14
    al = sum(losses[:14]) / 14
    expect = [None] * 14 + [100.0 - 100.0 / (1.0 + ag / al)]
    for i in range(14, len(gains)):
        ag = (ag * 13 + gains[i]) / 14
        al = (al * 13 + losses[i]) / 14
        expect.append(100.0 - 100.0 / (1.0 + ag / al))
    assert got[14:] == pytest.approx(expect[14:], abs=1e-12)


def test_rsi_monotone_down_approaches_zero():
    n = 40
    d = qd.monotone_down(n)
    out = compute_rsi(d["dates"], d["close"], as_of=qd.as_of(n), window=14)["value"]
    assert out[:14] == [None] * 14
    assert all(v == pytest.approx(0.0) for v in _finite(out))


def test_rsi_flat_series_is_neutral_50():
    """全平：无涨无跌（g=l=0），定义为中性 50，不是 0 也不是 NaN。"""
    n = 30
    out = compute_rsi(qd.dates(n), [5.0] * n, as_of=qd.as_of(n), window=14)["value"]
    assert all(v == pytest.approx(50.0) for v in _finite(out))


# ── MACD ─────────────────────────────────────────────────────────


def test_macd_shapes_and_identities():
    d = qd.ohlcv(120, seed=3)
    n = len(d["dates"])
    r = compute_macd(d["dates"], d["close"], as_of=qd.of_dates_max(d["dates"]))
    assert len(r["dif"]) == len(r["dea"]) == len(r["histogram"]) == n
    # 柱 = (DIF - DEA) × scale
    for i in range(n):
        if r["dif"][i] is None or r["dea"][i] is None:
            assert r["histogram"][i] is None
            continue
        assert r["histogram"][i] == pytest.approx(r["dif"][i] - r["dea"][i], abs=1e-12)
    # warmup 递增：dif 最早，histogram 最晚
    assert r["warmup"]["dif"] == 25  # slow(26)-1
    assert r["warmup"]["dea"] == 33  # slow+signal-2


def test_macd_rejects_fast_ge_slow():
    d = qd.ohlcv(60)
    with pytest.raises(base.ComputeError, match="must be <"):
        compute_macd(d["dates"], d["close"], as_of=qd.of_dates_max(d["dates"]),
                     fast=26, slow=12)


def test_macd_histogram_scale_is_explicit():
    """A 股软件常用 2.0 倍柱——显式传入时柱恰为 2 倍，不隐式假设。"""
    d = qd.ohlcv(80)
    a = qd.of_dates_max(d["dates"])
    r1 = compute_macd(d["dates"], d["close"], as_of=a)
    r2 = compute_macd(d["dates"], d["close"], as_of=a, histogram_scale=2.0)
    for x, y in zip(r2["histogram"], r1["histogram"]):
        if x is None:
            assert y is None
        else:
            assert x == pytest.approx(2.0 * y, abs=1e-12)


# ── 布林带 ───────────────────────────────────────────────────────


def test_bollinger_bands_shape_and_symmetry():
    d = qd.ohlcv(100, seed=5)
    n = len(d["dates"])
    r = compute_bollinger(d["dates"], d["close"], as_of=qd.of_dates_max(d["dates"]))
    assert len(r["upper"]) == len(r["middle"]) == len(r["lower"]) == n
    assert r["warmup"]["middle"] == 19
    for i in range(n):
        if r["middle"][i] is None:
            assert r["upper"][i] is None and r["lower"][i] is None
            continue
        # 上下轨关于中轨对称
        assert r["upper"][i] - r["middle"][i] == pytest.approx(
            r["middle"][i] - r["lower"][i], abs=1e-12)
        assert r["upper"][i] >= r["middle"][i] >= r["lower"][i]


def test_bollinger_matches_population_std():
    """对拍：窗口内总体标准差（除 w，非 w-1）。"""
    w = 20
    vals = [float((i * 7) % 13) + 10.0 for i in range(60)]
    ds = qd.dates(60)
    r = compute_bollinger(ds, vals, as_of=qd.as_of(60), window=w, num_std=2.0)
    win = vals[:w]
    m = sum(win) / w
    sd = math.sqrt(sum((x - m) ** 2 for x in win) / w)
    assert r["middle"][w - 1] == pytest.approx(m, abs=1e-12)
    assert r["upper"][w - 1] == pytest.approx(m + 2 * sd, abs=1e-12)
    assert r["lower"][w - 1] == pytest.approx(m - 2 * sd, abs=1e-12)


# ── ATR / KDJ ────────────────────────────────────────────────────


def test_atr_non_negative_and_warmup():
    d = qd.ohlcv(100, seed=9)
    r = compute_atr(d["dates"], d["high"], d["low"], d["close"],
                    as_of=qd.of_dates_max(d["dates"]), window=14)
    vals = _finite(r["value"])
    assert vals and all(v >= 0.0 for v in vals)
    # ATR 的 warmup 是 window-1：TR 首项只需高低价，不需要前收。
    assert r["warmup"]["value"] == 13
    assert r["value"][:13] == [None] * 13


def test_atr_matches_wilder_recursion():
    """对拍手算：TR 三条取最大，Wilder 递推。"""
    n = 20
    ds = qd.dates(n)
    high = [10.0 + i * 0.5 for i in range(n)]
    low = [9.0 + i * 0.5 for i in range(n)]
    close = [9.5 + i * 0.5 for i in range(n)]
    r = compute_atr(ds, high, low, close, as_of=qd.as_of(n), window=14)

    tr = [high[0] - low[0]]
    for i in range(1, n):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]),
                      abs(low[i] - close[i - 1])))
    expect = [None] * 13 + [sum(tr[:14]) / 14]
    for i in range(14, n):
        expect.append((expect[-1] * 13 + tr[i]) / 14)
    assert r["value"] == pytest.approx(expect, abs=1e-12, nan_ok=True)


def test_kdj_ranges_and_j_can_exceed():
    d = qd.ohlcv(120, seed=13)
    r = compute_kdj(d["dates"], d["high"], d["low"], d["close"],
                    as_of=qd.of_dates_max(d["dates"]))
    ks, ds_, js = _finite(r["k"]), _finite(r["d"]), _finite(r["j"])
    assert ks and ds_ and js
    assert all(0.0 <= v <= 100.0 for v in ks), f"K out of range: {[v for v in ks if v<0 or v>100][:5]}"
    assert all(0.0 <= v <= 100.0 for v in ds_)
    # J = 3K - 2D 恒等
    for k, dd, j in zip(r["k"], r["d"], r["j"]):
        if k is None:
            assert j is None
            continue
        assert j == pytest.approx(3 * k - 2 * dd, abs=1e-12)
    assert r["params"] == {"n_period": 9, "k_period": 3, "d_period": 3}
    assert r["warmup"]["k"] == 8


def test_kdj_rsv_at_high_yields_k_above_50():
    """持续收在区间最高 → RSV=100 → K/D 单调升向 100。"""
    n = 40
    ds = qd.dates(n)
    high = [10.0 + i for i in range(n)]
    low = [9.0 + i for i in range(n)]
    close = [10.0 + i for i in range(n)]  # 收在最高
    r = compute_kdj(ds, high, low, close, as_of=qd.as_of(n))
    ks = _finite(r["k"])
    assert ks[-1] > 50.0
    assert ks == sorted(ks)  # 单调不减


# ── OBV ──────────────────────────────────────────────────────────


def test_obv_accumulates_per_definition():
    """对拍：逐日按涨跌累加/扣减成交量，首日 0。"""
    n = 12
    ds = qd.dates(n)
    close = [10.0, 11.0, 11.0, 10.0, 10.5, 10.5, 11.5, 11.0, 12.0, 11.5, 11.5, 12.0]
    vol = [100.0 * (i + 1) for i in range(n)]
    r = compute_obv(ds, close, vol, as_of=qd.as_of(n))

    expect = [0.0]
    run = 0.0
    for i in range(1, n):
        if close[i] > close[i - 1]:
            run += vol[i]
        elif close[i] < close[i - 1]:
            run -= vol[i]
        expect.append(run)
    assert r["value"] == pytest.approx(expect, abs=1e-12)
    assert r["warmup"]["value"] == 0


# ── CCI / DMI ────────────────────────────────────────────────────


def test_cci_matches_independent_definition():
    w = 20
    n = 50
    ds = qd.dates(n)
    rng = qd.LCG(21)
    high, low, close = [], [], []
    px = 30.0
    for _ in range(n):
        px += 0.3 * rng.next_normal()
        close.append(px)
        high.append(px + 0.4)
        low.append(px - 0.4)
    r = compute_cci(ds, high, low, close, as_of=qd.as_of(n), window=w)

    tp = [(high[i] + low[i] + close[i]) / 3.0 for i in range(n)]
    expect = [None] * (w - 1)
    for i in range(w - 1, n):
        win = tp[i - w + 1:i + 1]
        m = sum(win) / w
        mad = sum(abs(x - m) for x in win) / w
        expect.append((tp[i] - m) / (0.015 * mad))
    assert r["value"] == pytest.approx(expect, abs=1e-10, nan_ok=True)


def test_dmi_identities_and_warmup():
    d = qd.ohlcv(150, seed=17)
    r = compute_dmi(d["dates"], d["high"], d["low"], d["close"],
                    as_of=qd.of_dates_max(d["dates"]))
    n = len(d["dates"])
    assert len(r["plus_di"]) == len(r["minus_di"]) == len(r["adx"]) == n
    finite_plus = _finite(r["plus_di"])
    assert finite_plus and all(0.0 <= v <= 100.0 for v in finite_plus)
    finite_adx = _finite(r["adx"])
    assert finite_adx and all(0.0 <= v <= 100.0 for v in finite_adx)
    # ADX 的 warmup 严格晚于 +DI
    assert r["warmup"]["adx"] > r["warmup"]["plus_di"]


def test_dmi_plus_di_dominant_in_uptrend():
    """单边上涨：+DI 应持续高于 -DI。"""
    n = 80
    ds = qd.dates(n)
    high = [10.0 + 0.5 * i for i in range(n)]
    low = [9.5 + 0.5 * i for i in range(n)]
    close = [9.8 + 0.5 * i for i in range(n)]
    r = compute_dmi(ds, high, low, close, as_of=qd.as_of(n))
    pairs = [(p, m) for p, m in zip(r["plus_di"], r["minus_di"])
             if p is not None and m is not None]
    assert pairs
    assert all(p > m for p, m in pairs)
    assert _finite(r["adx"])[-1] > 50.0  # 趋势明确 → ADX 高


# ── warmup / NaN 纪律（跨指标） ───────────────────────────────────


ALL_TOOLS = [
    ("sma", lambda d, a: compute_sma(d["dates"], d["close"], as_of=a)),
    ("ema", lambda d, a: compute_ema(d["dates"], d["close"], as_of=a)),
    ("rsi", lambda d, a: compute_rsi(d["dates"], d["close"], as_of=a)),
    ("macd", lambda d, a: compute_macd(d["dates"], d["close"], as_of=a)),
    ("bollinger", lambda d, a: compute_bollinger(d["dates"], d["close"], as_of=a)),
    ("atr", lambda d, a: compute_atr(d["dates"], d["high"], d["low"], d["close"], as_of=a)),
    ("kdj", lambda d, a: compute_kdj(d["dates"], d["high"], d["low"], d["close"], as_of=a)),
    ("obv", lambda d, a: compute_obv(d["dates"], d["close"], d["volume"], as_of=a)),
    ("cci", lambda d, a: compute_cci(d["dates"], d["high"], d["low"], d["close"], as_of=a)),
    ("dmi", lambda d, a: compute_dmi(d["dates"], d["high"], d["low"], d["close"], as_of=a)),
]


@pytest.mark.parametrize("name,fn", ALL_TOOLS, ids=[t[0] for t in ALL_TOOLS])
def test_every_indicator_declares_params_and_warmup(name, fn):
    """spec 2.4：每个指标返回含 params 与 warmup。"""
    d = qd.ohlcv(80, seed=23)
    a = qd.of_dates_max(d["dates"])
    r = fn(d, a)
    assert "params" in r and isinstance(r["params"], dict)
    assert "warmup" in r and isinstance(r["warmup"], dict) and r["warmup"]
    assert isinstance(r["evidence"], list) and len(r["evidence"]) >= 2
    assert r["as_of"] == a
    assert r["input_start"] == d["dates"][0] and r["input_end"] == d["dates"][-1]


@pytest.mark.parametrize("name,fn", ALL_TOOLS, ids=[t[0] for t in ALL_TOOLS])
def test_nan_input_never_raises_and_is_visible(name, fn):
    """spec 2.3：输入含 NaN 不抛异常；缺失处按惯例传播（输出 null）。"""
    d = qd.ohlcv(80, seed=29)
    a = qd.of_dates_max(d["dates"])
    for key in ("close", "high", "low", "volume"):
        if key in d:
            d[key] = qd.with_nan(d[key], (30, 31))
    r = fn(d, a)  # 不抛
    series_keys = [k for k in r if isinstance(r[k], list) and len(r[k]) == 80]
    assert series_keys, f"{name}: no series in result"
    # 注入点附近确有 null（局部传播可见）
    for k in series_keys:
        arr = r[k]
        assert arr[30] is None or arr[31] is None, f"{name}.{k}: NaN not propagated"


@pytest.mark.parametrize("name,fn", ALL_TOOLS, ids=[t[0] for t in ALL_TOOLS])
def test_bitwise_determinism(name, fn):
    """spec 1.2 / Scenario 同输入重跑逐位一致。"""
    d = qd.ohlcv(120, seed=31)
    a = qd.of_dates_max(d["dates"])
    r1, r2 = fn(d, a), fn(d, a)
    base.assert_bitequal(r1, r2, what=name)


@pytest.mark.parametrize("name,fn", ALL_TOOLS, ids=[t[0] for t in ALL_TOOLS])
def test_future_data_is_refused_naming_the_date(name, fn):
    """spec as_of：越界数据点名拒绝。"""
    d = qd.ohlcv(40, seed=37)
    as_of = d["dates"][30]  # 前 31 个合法
    with pytest.raises(base.ComputeError) as ei:
        fn(d, as_of)
    assert d["dates"][31] in str(ei.value)


def test_as_of_is_required():
    d = qd.ohlcv(10)
    with pytest.raises(base.ComputeError, match="as_of is required"):
        compute_sma(d["dates"], d["close"], as_of=None)  # type: ignore[arg-type]


def test_evidence_carries_source_and_unique_ids():
    """spec evidence：source 含工具与参数；id 单次响应内唯一。"""
    d = qd.ohlcv(60, seed=41)
    a = qd.of_dates_max(d["dates"])
    r = compute_atr(d["dates"], d["high"], d["low"], d["close"], as_of=a, window=14)
    ids = [e["evidence_id"] for e in r["evidence"]]
    assert len(ids) == len(set(ids))
    for e in r["evidence"]:
        assert e["source"].startswith("fd-quant:compute_atr(")
        assert "window=14" in e["source"]
        assert e["as_of"] == a
        assert set(e) == {"evidence_id", "label", "value", "unit", "as_of", "source"}
