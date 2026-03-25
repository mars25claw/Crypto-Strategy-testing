"""
Technical indicators library for the shared trading bot framework.

All indicator functions operate on numpy arrays and return numpy arrays (or floats
where noted). No pandas dependency. Edge cases with insufficient data produce NaN
values in the corresponding positions.
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average.

    EMA_today = Price * k + EMA_yesterday * (1 - k), where k = 2 / (period + 1).
    The first ``period - 1`` values are NaN; the EMA is seeded with the SMA of the
    first ``period`` values.
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period or period < 1:
        return out

    k = 2.0 / (period + 1)
    # Seed with SMA of the first `period` elements
    out[period - 1] = np.mean(data[:period])
    for i in range(period, n):
        out[i] = data[i] * k + out[i - 1] * (1.0 - k)
    return out


def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average using a cumulative-sum approach (O(n))."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period or period < 1:
        return out

    cs = np.cumsum(data)
    out[period - 1] = cs[period - 1] / period
    out[period:] = (cs[period:] - cs[:-period]) / period
    return out


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index with Wilder's smoothing.

    RSI = 100 - 100 / (1 + RS), RS = avg_gain / avg_loss.
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1 or period < 1:
        return out

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0.0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return out


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD indicator.

    Returns
    -------
    macd_line : np.ndarray
    signal_line : np.ndarray
    histogram : np.ndarray
    """
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    macd_line = fast_ema - slow_ema

    # Signal line is the EMA of the MACD line; we need to handle NaNs
    # Build a contiguous sub-array starting from the first valid MACD value.
    first_valid = slow - 1  # index where slow EMA becomes valid
    n = len(closes)
    signal_line = np.full(n, np.nan, dtype=np.float64)
    histogram = np.full(n, np.nan, dtype=np.float64)

    if n > first_valid:
        macd_valid = macd_line[first_valid:]
        sig = ema(macd_valid, signal)
        signal_line[first_valid:] = sig
        histogram[first_valid:] = macd_valid - sig

    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# ADX (+DI / -DI)
# ---------------------------------------------------------------------------

def adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average Directional Index.

    Returns
    -------
    adx : np.ndarray
    plus_di : np.ndarray
    minus_di : np.ndarray
    """
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)

    adx_out = np.full(n, np.nan, dtype=np.float64)
    plus_di_out = np.full(n, np.nan, dtype=np.float64)
    minus_di_out = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1 or period < 1:
        return adx_out, plus_di_out, minus_di_out

    # True Range, +DM, -DM for each bar (starting from index 1)
    tr = np.empty(n - 1, dtype=np.float64)
    plus_dm = np.empty(n - 1, dtype=np.float64)
    minus_dm = np.empty(n - 1, dtype=np.float64)

    for i in range(1, n):
        hi_diff = highs[i] - highs[i - 1]
        lo_diff = lows[i - 1] - lows[i]

        tr[i - 1] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        plus_dm[i - 1] = hi_diff if (hi_diff > lo_diff and hi_diff > 0) else 0.0
        minus_dm[i - 1] = lo_diff if (lo_diff > hi_diff and lo_diff > 0) else 0.0

    # Wilder's smoothing for first `period` bars
    atr_smooth = np.sum(tr[:period])
    plus_dm_smooth = np.sum(plus_dm[:period])
    minus_dm_smooth = np.sum(minus_dm[:period])

    def _di(dm_s, atr_s):
        return 100.0 * dm_s / atr_s if atr_s != 0.0 else 0.0

    pdi = _di(plus_dm_smooth, atr_smooth)
    mdi = _di(minus_dm_smooth, atr_smooth)

    idx = period  # output index (offset by 1 because tr starts at bar 1)
    plus_di_out[idx] = pdi
    minus_di_out[idx] = mdi

    di_sum = pdi + mdi
    dx_first = abs(pdi - mdi) / di_sum * 100.0 if di_sum != 0.0 else 0.0

    dx_values = [dx_first]

    for i in range(period, len(tr)):
        atr_smooth = atr_smooth - atr_smooth / period + tr[i]
        plus_dm_smooth = plus_dm_smooth - plus_dm_smooth / period + plus_dm[i]
        minus_dm_smooth = minus_dm_smooth - minus_dm_smooth / period + minus_dm[i]

        pdi = _di(plus_dm_smooth, atr_smooth)
        mdi = _di(minus_dm_smooth, atr_smooth)

        idx = i + 1
        plus_di_out[idx] = pdi
        minus_di_out[idx] = mdi

        di_sum = pdi + mdi
        dx = abs(pdi - mdi) / di_sum * 100.0 if di_sum != 0.0 else 0.0
        dx_values.append(dx)

    # ADX = Wilder-smoothed DX over `period` DX values
    # dx_values[0] corresponds to original index `period`, dx_values[j] -> index `period + j`
    if len(dx_values) >= period:
        adx_val = np.mean(dx_values[:period])
        first_adx_idx = 2 * period  # period (offset of dx[0]) + period (smoothing window)
        if first_adx_idx < n:
            adx_out[first_adx_idx] = adx_val
        for j in range(period, len(dx_values)):
            adx_val = (adx_val * (period - 1) + dx_values[j]) / period
            out_idx = period + j  # dx_values[j] corresponds to original index period+j
            if out_idx < n:
                adx_out[out_idx] = adx_val

    return adx_out, plus_di_out, minus_di_out


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range with Wilder's smoothing."""
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1 or period < 1:
        return out

    # True range from index 1 onward
    tr = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        tr[i - 1] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # First ATR is the SMA of the first `period` true-range values
    atr_val = np.mean(tr[:period])
    out[period] = atr_val

    for i in range(period, len(tr)):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        out[i + 1] = atr_val

    return out


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    closes: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands.

    Returns
    -------
    upper : np.ndarray
    middle : np.ndarray  (SMA)
    lower : np.ndarray
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = sma(closes, period)
    lower = np.full(n, np.nan, dtype=np.float64)

    if n < period or period < 1:
        return upper, middle, lower

    # Rolling std via the variance identity: E[X^2] - E[X]^2
    cs = np.cumsum(closes)
    cs2 = np.cumsum(closes ** 2)

    for i in range(period - 1, n):
        start = i - period + 1
        s = cs[i] - (cs[start - 1] if start > 0 else 0.0)
        s2 = cs2[i] - (cs2[start - 1] if start > 0 else 0.0)
        mean = s / period
        var = s2 / period - mean * mean
        # Clamp small negative values from floating-point error
        std = np.sqrt(max(var, 0.0))
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std

    return upper, middle, lower


# ---------------------------------------------------------------------------
# Volume indicators
# ---------------------------------------------------------------------------

def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """On-Balance Volume."""
    closes = np.asarray(closes, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    n = len(closes)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    out[0] = volumes[0]
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            out[i] = out[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            out[i] = out[i - 1] - volumes[i]
        else:
            out[i] = out[i - 1]
    return out


def obv_slope(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Slope of OBV over a rolling window using least-squares regression."""
    obv_arr = obv(closes, volumes)
    n = len(obv_arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period or period < 2:
        return out

    # Pre-compute sums for least-squares slope: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    x = np.arange(period, dtype=np.float64)
    sum_x = np.sum(x)
    sum_x2 = np.sum(x * x)
    denom = period * sum_x2 - sum_x * sum_x

    for i in range(period - 1, n):
        y = obv_arr[i - period + 1: i + 1]
        sum_y = np.sum(y)
        sum_xy = np.dot(x, y)
        out[i] = (period * sum_xy - sum_x * sum_y) / denom

    return out


def volume_average(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """Rolling average volume (SMA of volume)."""
    return sma(volumes, period)


# ---------------------------------------------------------------------------
# Hurst Exponent
# ---------------------------------------------------------------------------

def hurst_exponent(data: np.ndarray, max_lag: int = 100) -> float:
    """Hurst exponent via rescaled range (R/S) analysis.

    H < 0.5 : mean-reverting
    H ~ 0.5 : random walk
    H > 0.5 : trending

    Returns NaN if insufficient data.
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n < 20:
        return np.nan

    max_lag = min(max_lag, n // 2)
    if max_lag < 2:
        return np.nan

    lags = range(2, max_lag + 1)
    rs_values = []

    for lag in lags:
        # Split the series into non-overlapping sub-series of length `lag`
        num_chunks = n // lag
        if num_chunks == 0:
            continue

        rs_for_lag = []
        for c in range(num_chunks):
            chunk = data[c * lag: (c + 1) * lag]
            mean_c = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_c)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=0)
            if s > 0:
                rs_for_lag.append(r / s)

        if rs_for_lag:
            rs_values.append((np.log(lag), np.log(np.mean(rs_for_lag))))

    if len(rs_values) < 2:
        return np.nan

    log_lags = np.array([v[0] for v in rs_values])
    log_rs = np.array([v[1] for v in rs_values])

    # Linear regression: H is the slope
    coeffs = np.polyfit(log_lags, log_rs, 1)
    return float(coeffs[0])


# ---------------------------------------------------------------------------
# Z-Score
# ---------------------------------------------------------------------------

def z_score(data: np.ndarray, period: int = 50) -> np.ndarray:
    """Rolling Z-score: (value - rolling_mean) / rolling_std."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period or period < 2:
        return out

    rolling_mean = sma(data, period)

    cs2 = np.cumsum(data ** 2)
    cs = np.cumsum(data)

    for i in range(period - 1, n):
        start = i - period + 1
        s = cs[i] - (cs[start - 1] if start > 0 else 0.0)
        s2 = cs2[i] - (cs2[start - 1] if start > 0 else 0.0)
        mean = s / period
        var = s2 / period - mean * mean
        std = np.sqrt(max(var, 0.0))
        if std > 0:
            out[i] = (data[i] - mean) / std
        else:
            out[i] = 0.0

    return out


# ---------------------------------------------------------------------------
# Order Book Imbalance
# ---------------------------------------------------------------------------

def order_book_imbalance(bids_depth: np.ndarray, asks_depth: np.ndarray) -> float:
    """Bid/ask depth imbalance at the top N levels.

    Parameters
    ----------
    bids_depth : array-like
        Bid sizes at the top N price levels.
    asks_depth : array-like
        Ask sizes at the top N price levels.

    Returns
    -------
    float
        sum(bids) / sum(asks).  Returns NaN if asks sum to zero.
    """
    bids_sum = float(np.sum(np.asarray(bids_depth, dtype=np.float64)))
    asks_sum = float(np.sum(np.asarray(asks_depth, dtype=np.float64)))
    if asks_sum == 0.0:
        return np.nan
    return bids_sum / asks_sum


# ---------------------------------------------------------------------------
# Candle-shape helpers
# ---------------------------------------------------------------------------

def kline_body_ratio(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> np.ndarray:
    """|close - open| / (high - low).  Returns 0 where high == low (doji)."""
    opens = np.asarray(opens, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)

    body = np.abs(closes - opens)
    wick = highs - lows
    out = np.where(wick > 0.0, body / wick, 0.0)
    return out


def close_position_in_range(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> np.ndarray:
    """(close - low) / (high - low).  Returns 0.5 where high == low."""
    closes = np.asarray(closes, dtype=np.float64)
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)

    rng = highs - lows
    out = np.where(rng > 0.0, (closes - lows) / rng, 0.5)
    return out


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def volatility_ratio(
    closes: np.ndarray,
    short_window: int = 24,
    long_window: int = 168,
) -> np.ndarray:
    """Ratio of short-term to long-term return volatility.

    volatility_ratio = std(returns, short_window) / std(returns, long_window)
    """
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)

    if n < long_window + 1:
        return out

    returns = np.diff(np.log(closes))  # log returns, length n-1

    # Rolling std of returns
    for i in range(long_window - 1, len(returns)):
        long_std = np.std(returns[i - long_window + 1: i + 1], ddof=1)
        if i >= short_window - 1:
            short_std = np.std(returns[i - short_window + 1: i + 1], ddof=1)
            # +1 offset because returns[i] corresponds to closes[i+1]
            if long_std > 0:
                out[i + 1] = short_std / long_std
            else:
                out[i + 1] = np.nan

    return out


# ---------------------------------------------------------------------------
# Half-life (pairs trading)
# ---------------------------------------------------------------------------

def half_life(spread: np.ndarray) -> float:
    """Half-life of mean reversion from an AR(1) model.

    Fits: delta_spread_t = theta * (spread_{t-1} - mu) + epsilon
    Returns -ln(2) / ln(1 + theta).

    Returns NaN if the fit indicates no mean reversion (theta >= 0).
    """
    spread = np.asarray(spread, dtype=np.float64)
    n = len(spread)
    if n < 3:
        return np.nan

    y = np.diff(spread)  # delta spread
    x = spread[:-1] - np.mean(spread[:-1])

    # OLS: theta = (x . y) / (x . x)
    xx = np.dot(x, x)
    if xx == 0.0:
        return np.nan
    theta = np.dot(x, y) / xx

    if theta >= 0.0:
        return np.nan  # no mean reversion

    log_arg = 1.0 + theta
    if log_arg <= 0.0:
        return np.nan

    hl = -np.log(2.0) / np.log(log_arg)
    return float(hl)


# ---------------------------------------------------------------------------
# Rolling correlation
# ---------------------------------------------------------------------------

def correlation_rolling(
    series_a: np.ndarray,
    series_b: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """Rolling Pearson correlation between two series."""
    series_a = np.asarray(series_a, dtype=np.float64)
    series_b = np.asarray(series_b, dtype=np.float64)
    n = min(len(series_a), len(series_b))
    out = np.full(n, np.nan, dtype=np.float64)

    if n < window or window < 2:
        return out

    for i in range(window - 1, n):
        a = series_a[i - window + 1: i + 1]
        b = series_b[i - window + 1: i + 1]
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        a_dev = a - a_mean
        b_dev = b - b_mean
        num = np.dot(a_dev, b_dev)
        denom = np.sqrt(np.dot(a_dev, a_dev) * np.dot(b_dev, b_dev))
        if denom > 0:
            out[i] = num / denom
        else:
            out[i] = 0.0

    return out


# ===========================================================================
# IndicatorBuffer
# ===========================================================================

class IndicatorBuffer:
    """Fixed-size ring buffer of OHLCV candle data for a single symbol/timeframe.

    Parameters
    ----------
    max_size : int
        Maximum number of candles to retain (default 500).
    """

    __slots__ = (
        "max_size",
        "_timestamps",
        "_opens",
        "_highs",
        "_lows",
        "_closes",
        "_volumes",
    )

    def __init__(self, max_size: int = 500) -> None:
        self.max_size = max_size
        self._timestamps: list = []
        self._opens: list = []
        self._highs: list = []
        self._lows: list = []
        self._closes: list = []
        self._volumes: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_candle(self, candle: dict) -> None:
        """Append a candle dict with keys: timestamp, open, high, low, close, volume.

        If the buffer is full the oldest candle is discarded.
        """
        self._timestamps.append(candle.get("timestamp", 0))
        self._opens.append(float(candle.get("open", 0)))
        self._highs.append(float(candle.get("high", 0)))
        self._lows.append(float(candle.get("low", 0)))
        self._closes.append(float(candle.get("close", 0)))
        self._volumes.append(float(candle.get("volume", 0)))

        if len(self._closes) > self.max_size:
            self._timestamps.pop(0)
            self._opens.pop(0)
            self._highs.pop(0)
            self._lows.pop(0)
            self._closes.pop(0)
            self._volumes.pop(0)

    def get_closes(self) -> np.ndarray:
        return np.array(self._closes, dtype=np.float64)

    def get_highs(self) -> np.ndarray:
        return np.array(self._highs, dtype=np.float64)

    def get_lows(self) -> np.ndarray:
        return np.array(self._lows, dtype=np.float64)

    def get_opens(self) -> np.ndarray:
        return np.array(self._opens, dtype=np.float64)

    def get_volumes(self) -> np.ndarray:
        return np.array(self._volumes, dtype=np.float64)

    def get_timestamps(self) -> np.ndarray:
        return np.array(self._timestamps)

    def clear(self) -> None:
        """Clear all buffered data.  Useful as a memory-manager callback."""
        self._timestamps.clear()
        self._opens.clear()
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._volumes.clear()

    def __len__(self) -> int:
        return len(self._closes)

    def __repr__(self) -> str:
        return f"IndicatorBuffer(size={len(self)}, max_size={self.max_size})"
