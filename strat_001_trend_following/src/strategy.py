"""STRAT-001 Entry Logic — Sections 3.1 through 3.6.

Full signal generation pipeline:
  1. EMA 20/50 crossover on 4h (candle must CLOSE beyond crossover)
  2. EMA 200 trend filter (price above for LONG, below for SHORT)
  3. MACD confirmation (line vs signal, histogram positive AND increasing)
  4. RSI(14) 4h: LONG 50-75, SHORT 25-50; reject >80 LONG, <20 SHORT
  5. ADX(14): >25 required; 20-25 half size; <20 reject; +DI/-DI alignment
  6. Daily alignment: daily EMA(50) vs EMA(200) must support direction
  7. Volume: >=1.5x 20-period avg; 1.0-1.5x reduce 25%; <1.0x reject
  8. Manipulation check: >40% vol in single 60s window -> flag
  9. 15m pullback entry: wait for pullback to 15m EMA(20), enter on confirm
 10. Order: LIMIT -> 30s timeout -> cancel/replace -> 3 attempts -> MARKET
 11. Bracket orders (stop + TP) within 500ms of fill
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.indicators import (
    ema,
    macd,
    rsi,
    adx,
    atr,
    volume_average,
    IndicatorBuffer,
)
from shared.binance_client import BinanceClient, BinanceClientError
from shared.alerting import AlertLevel

from . import STRATEGY_ID

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


# ---------------------------------------------------------------------------
# Signal dataclasses
# ---------------------------------------------------------------------------

class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SignalStatus(str, Enum):
    PENDING_PULLBACK = "PENDING_PULLBACK"
    READY_TO_ENTER = "READY_TO_ENTER"
    ENTERING = "ENTERING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class EntrySignal:
    """Represents a validated crossover signal awaiting execution."""
    symbol: str
    direction: SignalDirection
    signal_time_ms: int
    crossover_price: float
    ema_fast: float
    ema_slow: float
    ema_200: float
    atr_value: float
    adx_value: float
    plus_di: float
    minus_di: float
    rsi_value: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    volume_ratio: float
    status: SignalStatus = SignalStatus.PENDING_PULLBACK
    size_multiplier: float = 1.0        # adjusted by ADX/volume
    manipulation_flagged: bool = False
    pullback_deadline_ms: int = 0
    cancel_price: float = 0.0           # cancel if price moves beyond this
    limit_attempts: int = 0
    current_order_id: Optional[int] = None
    filled_qty: float = 0.0
    filled_price: float = 0.0
    entry_start_ms: int = 0


@dataclass
class IndicatorSnapshot:
    """Snapshot of all indicator values at a moment in time."""
    ema_20: float = float("nan")
    ema_50: float = float("nan")
    ema_200: float = float("nan")
    macd_line: float = float("nan")
    macd_signal_line: float = float("nan")
    macd_histogram: float = float("nan")
    prev_macd_histogram: float = float("nan")
    rsi_value: float = float("nan")
    adx_value: float = float("nan")
    plus_di: float = float("nan")
    minus_di: float = float("nan")
    atr_value: float = float("nan")
    volume_ratio: float = float("nan")
    daily_ema_50: float = float("nan")
    daily_ema_200: float = float("nan")
    daily_close: float = float("nan")
    prev_ema_20: float = float("nan")
    prev_ema_50: float = float("nan")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_valid(v: float) -> bool:
    """Return True if v is a finite number (not NaN or Inf)."""
    return math.isfinite(v)


def _safe_last(arr: np.ndarray, fallback: float = float("nan")) -> float:
    """Return the last non-NaN value of an array, or *fallback*."""
    for i in range(len(arr) - 1, -1, -1):
        if np.isfinite(arr[i]):
            return float(arr[i])
    return fallback


def _safe_prev(arr: np.ndarray, offset: int = 1, fallback: float = float("nan")) -> float:
    """Return the (last - offset) non-NaN value."""
    idx = len(arr) - 1 - offset
    if 0 <= idx < len(arr) and np.isfinite(arr[idx]):
        return float(arr[idx])
    return fallback


# ---------------------------------------------------------------------------
# StrategyEngine
# ---------------------------------------------------------------------------

class TrendFollowingStrategy:
    """STRAT-001 entry signal generator and order manager.

    Parameters
    ----------
    config : dict
        Full ``strategy_params`` from config.yaml.
    client : BinanceClient
        Shared REST client for order placement.
    risk_manager : object
        Local :class:`~src.risk_manager.TrendRiskManager`.
    filters : object
        :class:`~src.filters.FilterEngine`.
    exit_manager : object
        :class:`~src.exit_manager.ExitManager`.
    alerter : object
        Shared :class:`~shared.alerting.AlertManager`.
    paper_engine : object | None
        :class:`~shared.paper_trading.PaperTradingEngine` if in paper mode.
    """

    TIMEFRAMES = ("1m", "15m", "4h", "1d")

    def __init__(
        self,
        config: Dict[str, Any],
        client: BinanceClient,
        risk_manager: Any,
        filters: Any,
        exit_manager: Any,
        alerter: Any,
        paper_engine: Any = None,
    ) -> None:
        self.cfg = config
        self.client = client
        self.risk_mgr = risk_manager
        self.filters = filters
        self.exit_mgr = exit_manager
        self.alerter = alerter
        self.paper = paper_engine

        # Indicator buffers: symbol -> timeframe -> IndicatorBuffer
        self.buffers: Dict[str, Dict[str, IndicatorBuffer]] = defaultdict(
            lambda: {tf: IndicatorBuffer(max_size=500) for tf in self.TIMEFRAMES}
        )

        # Pending entry signals: symbol -> EntrySignal
        self.pending_signals: Dict[str, EntrySignal] = {}

        # AggTrade volume tracking for manipulation detection
        # symbol -> list of (timestamp_s, qty) in current 4h candle
        self._agg_trade_buffer: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

        # Last crossover direction per symbol to detect changes
        self._last_cross_dir: Dict[str, Optional[str]] = {}

        # Previous indicator snapshots for anomaly detection
        self._prev_indicators: Dict[str, IndicatorSnapshot] = {}

        # Anomaly counters: symbol -> consecutive anomaly count
        self._anomaly_count: Dict[str, int] = defaultdict(int)

        # Running flag
        self._running = False

    # ======================================================================
    # Indicator calculation
    # ======================================================================

    def compute_indicators(self, symbol: str, timeframe: str) -> IndicatorSnapshot:
        """Calculate all indicators from the buffer for *symbol*/*timeframe*.

        Returns an IndicatorSnapshot.  Any indicator that cannot be computed
        (insufficient data) is left as NaN.
        """
        snap = IndicatorSnapshot()
        buf = self.buffers[symbol].get(timeframe)
        if buf is None or len(buf) < 50:
            return snap

        closes = buf.get_closes()
        highs = buf.get_highs()
        lows = buf.get_lows()
        volumes = buf.get_volumes()

        # EMA 20 / 50 / 200
        ema_20_arr = ema(closes, self.cfg.get("fast_ema_period", 20))
        ema_50_arr = ema(closes, self.cfg.get("slow_ema_period", 50))
        ema_200_arr = ema(closes, self.cfg.get("trend_ema_period", 200))

        snap.ema_20 = _safe_last(ema_20_arr)
        snap.ema_50 = _safe_last(ema_50_arr)
        snap.ema_200 = _safe_last(ema_200_arr)
        snap.prev_ema_20 = _safe_prev(ema_20_arr, 1)
        snap.prev_ema_50 = _safe_prev(ema_50_arr, 1)

        # MACD
        macd_line_arr, signal_arr, hist_arr = macd(
            closes,
            fast=self.cfg.get("macd_fast", 12),
            slow=self.cfg.get("macd_slow", 26),
            signal=self.cfg.get("macd_signal", 9),
        )
        snap.macd_line = _safe_last(macd_line_arr)
        snap.macd_signal_line = _safe_last(signal_arr)
        snap.macd_histogram = _safe_last(hist_arr)
        snap.prev_macd_histogram = _safe_prev(hist_arr, 1)

        # RSI
        rsi_arr = rsi(closes, self.cfg.get("rsi_period", 14))
        snap.rsi_value = _safe_last(rsi_arr)

        # ADX / +DI / -DI
        adx_arr, pdi_arr, mdi_arr = adx(
            highs, lows, closes, self.cfg.get("adx_period", 14)
        )
        snap.adx_value = _safe_last(adx_arr)
        snap.plus_di = _safe_last(pdi_arr)
        snap.minus_di = _safe_last(mdi_arr)

        # ATR
        atr_arr = atr(highs, lows, closes, self.cfg.get("atr_period", 14))
        snap.atr_value = _safe_last(atr_arr)

        # Volume ratio (current candle vol / 20-period avg)
        vol_avg_arr = volume_average(volumes, self.cfg.get("volume_avg_period", 20))
        vol_avg = _safe_last(vol_avg_arr)
        last_vol = float(volumes[-1]) if len(volumes) > 0 else 0.0
        if _is_valid(vol_avg) and vol_avg > 0:
            snap.volume_ratio = last_vol / vol_avg
        else:
            snap.volume_ratio = 0.0

        # Validate indicators for anomalies (Section 11.8)
        snap = self._validate_indicators(symbol, snap)

        return snap

    def compute_daily_indicators(self, symbol: str) -> Tuple[float, float, float]:
        """Return (daily_ema_50, daily_ema_200, daily_close) for *symbol*."""
        buf = self.buffers[symbol].get("1d")
        if buf is None or len(buf) < 200:
            return float("nan"), float("nan"), float("nan")

        closes = buf.get_closes()
        ema_50_arr = ema(closes, self.cfg.get("daily_fast_ema", 50))
        ema_200_arr = ema(closes, self.cfg.get("daily_slow_ema", 200))

        return _safe_last(ema_50_arr), _safe_last(ema_200_arr), float(closes[-1])

    def compute_15m_indicators(self, symbol: str) -> Tuple[float, float]:
        """Return (ema_20_15m, rsi_14_15m) for entry timing."""
        buf = self.buffers[symbol].get("15m")
        if buf is None or len(buf) < 20:
            return float("nan"), float("nan")

        closes = buf.get_closes()
        ema_20_arr = ema(closes, 20)
        rsi_arr = rsi(closes, 14)
        return _safe_last(ema_20_arr), _safe_last(rsi_arr)

    # ======================================================================
    # Indicator anomaly detection (Section 11.8)
    # ======================================================================

    def _validate_indicators(self, symbol: str, snap: IndicatorSnapshot) -> IndicatorSnapshot:
        """Validate indicator values; replace anomalies with previous valid values."""
        prev = self._prev_indicators.get(symbol)
        anomaly = False

        # RSI must be 0-100
        if not _is_valid(snap.rsi_value) or snap.rsi_value < 0 or snap.rsi_value > 100:
            logger.warning("Indicator anomaly %s: RSI=%.4f", symbol, snap.rsi_value)
            snap.rsi_value = prev.rsi_value if prev and _is_valid(prev.rsi_value) else 50.0
            anomaly = True

        # ADX must be >= 0
        if not _is_valid(snap.adx_value) or snap.adx_value < 0:
            logger.warning("Indicator anomaly %s: ADX=%.4f", symbol, snap.adx_value)
            snap.adx_value = prev.adx_value if prev and _is_valid(prev.adx_value) else 0.0
            anomaly = True

        # ATR must be > 0
        if not _is_valid(snap.atr_value) or snap.atr_value <= 0:
            logger.warning("Indicator anomaly %s: ATR=%.4f", symbol, snap.atr_value)
            snap.atr_value = prev.atr_value if prev and _is_valid(prev.atr_value) else 0.0
            anomaly = True

        # EMA values must be finite and positive
        for attr_name in ("ema_20", "ema_50", "ema_200"):
            val = getattr(snap, attr_name)
            if not _is_valid(val) or val <= 0:
                prev_val = getattr(prev, attr_name, float("nan")) if prev else float("nan")
                if _is_valid(prev_val):
                    setattr(snap, attr_name, prev_val)
                anomaly = True

        # MACD values must be finite
        for attr_name in ("macd_line", "macd_signal_line", "macd_histogram"):
            val = getattr(snap, attr_name)
            if not _is_valid(val):
                prev_val = getattr(prev, attr_name, 0.0) if prev else 0.0
                setattr(snap, attr_name, prev_val if _is_valid(prev_val) else 0.0)
                anomaly = True

        if anomaly:
            self._anomaly_count[symbol] += 1
            if self._anomaly_count[symbol] >= 3:
                logger.error(
                    "3 consecutive indicator anomalies for %s — halt required", symbol
                )
        else:
            self._anomaly_count[symbol] = 0

        self._prev_indicators[symbol] = snap
        return snap

    def is_halted_by_anomaly(self, symbol: str) -> bool:
        """Return True if 3+ consecutive anomalies occurred (Section 11.8)."""
        return self._anomaly_count.get(symbol, 0) >= 3

    def reset_anomaly_count(self, symbol: str) -> None:
        """Reset anomaly counter after successful recalculation from raw data."""
        self._anomaly_count[symbol] = 0

    # ======================================================================
    # Signal detection
    # ======================================================================

    def check_for_signal(self, symbol: str) -> Optional[EntrySignal]:
        """Evaluate 4h candle close for a crossover signal.

        Returns an :class:`EntrySignal` if all conditions are met, else None.
        This should be called on every closed 4h candle.
        """
        if self.is_halted_by_anomaly(symbol):
            logger.warning("Skipping signal check for %s — indicator anomaly halt", symbol)
            return None

        # Already have a pending signal for this symbol
        if symbol in self.pending_signals:
            return None

        snap = self.compute_indicators(symbol, "4h")
        if not all(_is_valid(v) for v in [
            snap.ema_20, snap.ema_50, snap.ema_200, snap.atr_value
        ]):
            return None

        # ------------------------------------------------------------------
        # Step 2: Detect crossover
        # ------------------------------------------------------------------
        prev_fast_above = (
            _is_valid(snap.prev_ema_20) and _is_valid(snap.prev_ema_50)
            and snap.prev_ema_20 > snap.prev_ema_50
        )
        curr_fast_above = snap.ema_20 > snap.ema_50

        crossover_long = not prev_fast_above and curr_fast_above
        crossover_short = prev_fast_above and not curr_fast_above

        if not crossover_long and not crossover_short:
            return None

        direction = SignalDirection.LONG if crossover_long else SignalDirection.SHORT

        # Candle must CLOSE beyond the crossover point
        buf_4h = self.buffers[symbol]["4h"]
        last_close = float(buf_4h.get_closes()[-1]) if len(buf_4h) > 0 else 0.0

        crossover_point = (snap.ema_20 + snap.ema_50) / 2.0
        if direction == SignalDirection.LONG and last_close < crossover_point:
            logger.info("%s LONG crossover detected but candle did not close above — rejected", symbol)
            return None
        if direction == SignalDirection.SHORT and last_close > crossover_point:
            logger.info("%s SHORT crossover detected but candle did not close below — rejected", symbol)
            return None

        # ------------------------------------------------------------------
        # Step 3: Trend filter — EMA 200
        # ------------------------------------------------------------------
        if direction == SignalDirection.LONG and last_close < snap.ema_200:
            logger.info("%s LONG rejected: price %.2f below EMA200 %.2f", symbol, last_close, snap.ema_200)
            return None
        if direction == SignalDirection.SHORT and last_close > snap.ema_200:
            logger.info("%s SHORT rejected: price %.2f above EMA200 %.2f", symbol, last_close, snap.ema_200)
            return None

        # ------------------------------------------------------------------
        # Condition A: MACD confirmation
        # ------------------------------------------------------------------
        if not _is_valid(snap.macd_line) or not _is_valid(snap.macd_signal_line):
            return None
        if not _is_valid(snap.macd_histogram) or not _is_valid(snap.prev_macd_histogram):
            return None

        if direction == SignalDirection.LONG:
            if snap.macd_line <= snap.macd_signal_line:
                logger.info("%s LONG rejected: MACD line below signal", symbol)
                return None
            if snap.macd_histogram <= 0:
                logger.info("%s LONG rejected: MACD histogram not positive", symbol)
                return None
            if snap.macd_histogram <= snap.prev_macd_histogram:
                logger.info("%s LONG rejected: MACD histogram not increasing", symbol)
                return None
        else:
            if snap.macd_line >= snap.macd_signal_line:
                logger.info("%s SHORT rejected: MACD line above signal", symbol)
                return None
            if snap.macd_histogram >= 0:
                logger.info("%s SHORT rejected: MACD histogram not negative", symbol)
                return None
            if abs(snap.macd_histogram) <= abs(snap.prev_macd_histogram):
                logger.info("%s SHORT rejected: MACD histogram abs not increasing", symbol)
                return None

        # ------------------------------------------------------------------
        # Condition B: RSI confirmation
        # ------------------------------------------------------------------
        if not _is_valid(snap.rsi_value):
            return None

        rsi_long_min = self.cfg.get("rsi_long_min", 50)
        rsi_long_max = self.cfg.get("rsi_long_max", 75)
        rsi_short_min = self.cfg.get("rsi_short_min", 25)
        rsi_short_max = self.cfg.get("rsi_short_max", 50)
        rsi_overbought = self.cfg.get("rsi_overbought", 80)
        rsi_oversold = self.cfg.get("rsi_oversold", 20)

        if direction == SignalDirection.LONG:
            if snap.rsi_value > rsi_overbought:
                logger.info("%s LONG rejected: RSI %.1f overbought (>%d)", symbol, snap.rsi_value, rsi_overbought)
                return None
            if not (rsi_long_min <= snap.rsi_value <= rsi_long_max):
                logger.info("%s LONG rejected: RSI %.1f not in [%d, %d]", symbol, snap.rsi_value, rsi_long_min, rsi_long_max)
                return None
        else:
            if snap.rsi_value < rsi_oversold:
                logger.info("%s SHORT rejected: RSI %.1f oversold (<%d)", symbol, snap.rsi_value, rsi_oversold)
                return None
            if not (rsi_short_min <= snap.rsi_value <= rsi_short_max):
                logger.info("%s SHORT rejected: RSI %.1f not in [%d, %d]", symbol, snap.rsi_value, rsi_short_min, rsi_short_max)
                return None

        # ------------------------------------------------------------------
        # Condition C: ADX confirmation
        # ------------------------------------------------------------------
        if not _is_valid(snap.adx_value):
            return None

        adx_strong = self.cfg.get("adx_strong", 25)
        adx_weak_low = self.cfg.get("adx_weak_low", 20)
        adx_reject = self.cfg.get("adx_reject_below", 20)

        if snap.adx_value < adx_reject:
            logger.info("%s rejected: ADX %.1f below %d (ranging)", symbol, snap.adx_value, adx_reject)
            return None

        size_mult = 1.0
        if adx_weak_low <= snap.adx_value < adx_strong:
            size_mult *= 0.5
            logger.info("%s ADX %.1f weak trend — 50%% size reduction", symbol, snap.adx_value)

        # +DI / -DI alignment
        if _is_valid(snap.plus_di) and _is_valid(snap.minus_di):
            if direction == SignalDirection.LONG and snap.plus_di <= snap.minus_di:
                logger.info("%s LONG rejected: +DI %.1f <= -DI %.1f", symbol, snap.plus_di, snap.minus_di)
                return None
            if direction == SignalDirection.SHORT and snap.minus_di <= snap.plus_di:
                logger.info("%s SHORT rejected: -DI %.1f <= +DI %.1f", symbol, snap.minus_di, snap.plus_di)
                return None

        # ------------------------------------------------------------------
        # Section 3.3: Daily timeframe alignment
        # ------------------------------------------------------------------
        d_ema50, d_ema200, d_close = self.compute_daily_indicators(symbol)
        if not _is_valid(d_ema50) or not _is_valid(d_ema200) or not _is_valid(d_close):
            logger.info("%s rejected: insufficient daily data", symbol)
            return None

        if direction == SignalDirection.LONG:
            if not (d_ema50 > d_ema200 or d_close > d_ema200):
                logger.info(
                    "%s LONG rejected: daily EMA50=%.2f not above EMA200=%.2f and price=%.2f not above EMA200",
                    symbol, d_ema50, d_ema200, d_close,
                )
                return None
        else:
            if not (d_ema50 < d_ema200 or d_close < d_ema200):
                logger.info(
                    "%s SHORT rejected: daily EMA50=%.2f not below EMA200=%.2f and price=%.2f not below EMA200",
                    symbol, d_ema50, d_ema200, d_close,
                )
                return None

        # ------------------------------------------------------------------
        # Section 3.4: Volume confirmation
        # ------------------------------------------------------------------
        vol_strong = self.cfg.get("volume_strong_multiplier", 1.5)
        vol_weak_low = self.cfg.get("volume_weak_low", 1.0)

        if snap.volume_ratio < vol_weak_low:
            logger.info("%s rejected: volume ratio %.2f below %.1f", symbol, snap.volume_ratio, vol_weak_low)
            return None

        if vol_weak_low <= snap.volume_ratio < vol_strong:
            size_mult *= 0.75
            logger.info("%s volume ratio %.2f weak — 25%% size reduction", symbol, snap.volume_ratio)

        # Manipulation check (Section 3.4)
        manipulation = self._check_volume_manipulation(symbol)

        # ------------------------------------------------------------------
        # Build the signal
        # ------------------------------------------------------------------
        now_ms = int(time.time() * 1000)
        max_pullback_candles = self.cfg.get("pullback_max_candles_4h", 3)
        pullback_deadline = now_ms + max_pullback_candles * 4 * 3600 * 1000  # 12h

        cancel_distance = self.cfg.get("pullback_cancel_atr_mult", 1.5) * snap.atr_value
        if direction == SignalDirection.LONG:
            cancel_price = last_close + cancel_distance
        else:
            cancel_price = last_close - cancel_distance

        signal = EntrySignal(
            symbol=symbol,
            direction=direction,
            signal_time_ms=now_ms,
            crossover_price=last_close,
            ema_fast=snap.ema_20,
            ema_slow=snap.ema_50,
            ema_200=snap.ema_200,
            atr_value=snap.atr_value,
            adx_value=snap.adx_value,
            plus_di=snap.plus_di,
            minus_di=snap.minus_di,
            rsi_value=snap.rsi_value,
            macd_line=snap.macd_line,
            macd_signal=snap.macd_signal_line,
            macd_histogram=snap.macd_histogram,
            volume_ratio=snap.volume_ratio,
            size_multiplier=size_mult,
            manipulation_flagged=manipulation,
            pullback_deadline_ms=pullback_deadline,
            cancel_price=cancel_price,
        )

        logger.info(
            "SIGNAL %s %s: EMA20=%.2f EMA50=%.2f EMA200=%.2f MACD=%.4f RSI=%.1f "
            "ADX=%.1f VolR=%.2f ATR=%.4f SizeMult=%.2f Manip=%s",
            symbol, direction.value, snap.ema_20, snap.ema_50, snap.ema_200,
            snap.macd_histogram, snap.rsi_value, snap.adx_value,
            snap.volume_ratio, snap.atr_value, size_mult, manipulation,
        )

        self._last_cross_dir[symbol] = direction.value
        return signal

    # ======================================================================
    # Volume manipulation detection (Section 3.4)
    # ======================================================================

    def record_agg_trade(self, symbol: str, timestamp_s: int, qty: float) -> None:
        """Record an aggregate trade for manipulation detection."""
        self._agg_trade_buffer[symbol].append((timestamp_s, qty))
        # Keep only last 4 hours of trades
        cutoff = timestamp_s - 4 * 3600
        buf = self._agg_trade_buffer[symbol]
        while buf and buf[0][0] < cutoff:
            buf.pop(0)

    def _check_volume_manipulation(self, symbol: str) -> bool:
        """Check if >40% of candle volume is concentrated in a single 60s window."""
        trades = self._agg_trade_buffer.get(symbol, [])
        if not trades:
            return False

        total_vol = sum(t[1] for t in trades)
        if total_vol <= 0:
            return False

        threshold_pct = self.cfg.get("volume_manipulation_pct", 40) / 100.0

        # Sliding 60-second windows
        window_vols: Dict[int, float] = defaultdict(float)
        for ts, qty in trades:
            window_key = ts // 60
            window_vols[window_key] += qty

        max_window_vol = max(window_vols.values()) if window_vols else 0.0
        if max_window_vol / total_vol > threshold_pct:
            logger.warning(
                "%s manipulation flag: %.1f%% of volume in single 60s window",
                symbol, (max_window_vol / total_vol) * 100,
            )
            return True
        return False

    def clear_agg_trades(self, symbol: str) -> None:
        """Clear the aggTrade buffer after candle close processing."""
        self._agg_trade_buffer[symbol].clear()

    # ======================================================================
    # 15m pullback entry (Section 3.5)
    # ======================================================================

    async def check_pullback_entry(
        self,
        symbol: str,
        signal: EntrySignal,
        current_price: float,
    ) -> bool:
        """Evaluate whether the 15m pullback condition is met.

        Returns True if the entry should proceed.
        """
        now_ms = int(time.time() * 1000)

        # Check if the setup has expired (12h)
        if now_ms > signal.pullback_deadline_ms:
            logger.info("%s pullback expired after 12h — cancelling signal", symbol)
            signal.status = SignalStatus.EXPIRED
            return False

        # Check if price has moved too far (>1.5x ATR)
        if signal.direction == SignalDirection.LONG:
            if current_price > signal.cancel_price:
                logger.info(
                    "%s LONG cancelled: price %.2f > cancel %.2f (1.5x ATR from signal)",
                    symbol, current_price, signal.cancel_price,
                )
                signal.status = SignalStatus.CANCELLED
                return False
        else:
            if current_price < signal.cancel_price:
                logger.info(
                    "%s SHORT cancelled: price %.2f < cancel %.2f (1.5x ATR from signal)",
                    symbol, current_price, signal.cancel_price,
                )
                signal.status = SignalStatus.CANCELLED
                return False

        # Get 15m indicators
        ema_20_15m, rsi_15m = self.compute_15m_indicators(symbol)
        if not _is_valid(ema_20_15m):
            return False

        buf_15m = self.buffers[symbol].get("15m")
        if buf_15m is None or len(buf_15m) < 2:
            return False

        closes_15m = buf_15m.get_closes()
        last_close = float(closes_15m[-1])
        prev_close = float(closes_15m[-2])

        if signal.direction == SignalDirection.LONG:
            # Wait for pullback to EMA20 on 15m
            pulled_back = prev_close <= ema_20_15m * 1.001
            # Confirm: candle closes above EMA20 in trade direction
            confirmed = last_close > ema_20_15m
            # RSI turning up
            rsi_confirm = _is_valid(rsi_15m) and rsi_15m > 50
        else:
            pulled_back = prev_close >= ema_20_15m * 0.999
            confirmed = last_close < ema_20_15m
            rsi_confirm = _is_valid(rsi_15m) and rsi_15m < 50

        if pulled_back and confirmed:
            logger.info(
                "%s 15m pullback entry confirmed: close=%.2f ema20_15m=%.2f rsi=%.1f",
                symbol, last_close, ema_20_15m, rsi_15m if _is_valid(rsi_15m) else 0,
            )
            return True

        # If near deadline and no pullback, enter on RSI confirmation only
        time_remaining_ms = signal.pullback_deadline_ms - now_ms
        if time_remaining_ms < 900_000 and rsi_confirm:  # last 15 minutes
            logger.info("%s near deadline — entering on RSI confirmation (%.1f)", symbol, rsi_15m)
            return True

        return False

    # ======================================================================
    # Order placement (Section 3.6)
    # ======================================================================

    async def execute_entry(
        self,
        signal: EntrySignal,
        position_size_qty: float,
        best_price: float,
    ) -> Optional[Dict[str, Any]]:
        """Execute entry order with LIMIT -> MARKET fallback logic.

        Returns the fill result dict or None if all attempts failed.
        """
        signal.status = SignalStatus.ENTERING
        signal.entry_start_ms = int(time.time() * 1000)
        symbol = signal.symbol
        side = "BUY" if signal.direction == SignalDirection.LONG else "SELL"
        max_attempts = self.cfg.get("max_limit_attempts", 3)
        timeout_s = self.cfg.get("limit_order_timeout_s", 30)

        for attempt in range(1, max_attempts + 1):
            try:
                # Place LIMIT order at best bid/ask
                logger.info(
                    "%s entry attempt %d/%d: LIMIT %s qty=%.6f price=%.2f",
                    symbol, attempt, max_attempts, side, position_size_qty, best_price,
                )

                if self.paper:
                    fill_result = self.paper.simulate_limit_order(
                        symbol=symbol,
                        side=side,
                        quantity=position_size_qty,
                        price=best_price,
                    )
                    if fill_result and fill_result.fill_quantity > 0:
                        return self._fill_to_dict(fill_result, symbol, side, "LIMIT", attempt)
                else:
                    order = await self.client.place_futures_order(
                        symbol=symbol,
                        side=side,
                        type="LIMIT",
                        quantity=position_size_qty,
                        price=best_price,
                        time_in_force="GTC",
                    )
                    signal.current_order_id = order.get("orderId")

                    # Wait for fill
                    filled = await self._wait_for_fill(symbol, signal.current_order_id, timeout_s)
                    if filled:
                        signal.filled_qty = filled.get("executedQty", 0)
                        signal.filled_price = filled.get("avgPrice", best_price)
                        return filled

                    # Cancel unfilled order
                    if signal.current_order_id:
                        try:
                            await self.client.cancel_futures_order(symbol, signal.current_order_id)
                        except BinanceClientError:
                            pass  # may already be cancelled/filled

                # Re-fetch best price for next attempt
                best_price = await self._get_best_price(symbol, side)

            except BinanceClientError as e:
                logger.error("%s LIMIT order error attempt %d: %s", symbol, attempt, e)
                signal.limit_attempts = attempt
                if attempt >= max_attempts:
                    break

        # Fall back to MARKET order after 3 failed LIMIT attempts
        logger.info("%s falling back to MARKET order: %s qty=%.6f", symbol, side, position_size_qty)
        try:
            if self.paper:
                fill_result = self.paper.simulate_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=position_size_qty,
                )
                if fill_result:
                    return self._fill_to_dict(fill_result, symbol, side, "MARKET", max_attempts + 1)
            else:
                order = await self.client.place_futures_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=position_size_qty,
                )
                return order
        except BinanceClientError as e:
            logger.error("%s MARKET order failed: %s", symbol, e)
            if self.alerter:
                await self.alerter.send(
                    f"ENTRY FAILED {symbol} {signal.direction.value}: {e}",
                    level=AlertLevel.CRITICAL,
                    strategy_id=STRATEGY_ID,
                )
            return None

    async def place_bracket_orders(
        self,
        symbol: str,
        direction: SignalDirection,
        entry_price: float,
        quantity: float,
        atr_value: float,
    ) -> Dict[str, Any]:
        """Place stop-loss and first TP within 500ms of fill (Section 3.6).

        Returns dict with 'stop_order' and 'tp_order' results.
        """
        start_ms = int(time.time() * 1000)
        hard_stop_mult = self.cfg.get("hard_stop_atr_mult", 2.0)
        tp1_mult = self.cfg.get("tp1_atr_mult", 1.0)
        tp1_pct = self.cfg.get("tp1_pct", 25) / 100.0
        bracket_deadline = self.cfg.get("bracket_order_deadline_ms", 500)

        if direction == SignalDirection.LONG:
            stop_price = entry_price - hard_stop_mult * atr_value
            tp_price = entry_price + tp1_mult * atr_value
            stop_side = "SELL"
        else:
            stop_price = entry_price + hard_stop_mult * atr_value
            tp_price = entry_price - tp1_mult * atr_value
            stop_side = "BUY"

        tp_qty = round(quantity * tp1_pct, 8)
        result = {"stop_order": None, "tp_order": None}

        # CRITICAL: Stop loss must be placed first — if it fails, retry up to 10 times
        for stop_attempt in range(1, 11):
            try:
                if self.paper:
                    result["stop_order"] = {"orderId": f"paper_stop_{symbol}", "stopPrice": stop_price}
                else:
                    result["stop_order"] = await self.client.place_futures_order(
                        symbol=symbol,
                        side=stop_side,
                        type="STOP_MARKET",
                        quantity=quantity,
                        stop_price=stop_price,
                        reduce_only=True,
                    )
                trade_logger.info(
                    "STOP_PLACED\tsymbol=%s\tprice=%.4f\tqty=%.6f\tattempt=%d",
                    symbol, stop_price, quantity, stop_attempt,
                )
                break
            except BinanceClientError as e:
                logger.error(
                    "%s stop loss placement attempt %d/10 failed: %s",
                    symbol, stop_attempt, e,
                )
                if stop_attempt >= 10:
                    # CRITICAL: Cannot place stop — close position immediately
                    logger.critical(
                        "%s CANNOT place stop loss after 10 attempts — closing position",
                        symbol,
                    )
                    if self.alerter:
                        await self.alerter.send(
                            f"STOP LOSS PLACEMENT FAILED {symbol} — emergency close",
                            level=AlertLevel.EMERGENCY,
                            strategy_id=STRATEGY_ID,
                        )
                    # Emergency market close
                    try:
                        if not self.paper:
                            close_side = "SELL" if direction == SignalDirection.LONG else "BUY"
                            await self.client.place_futures_order(
                                symbol=symbol,
                                side=close_side,
                                type="MARKET",
                                quantity=quantity,
                                reduce_only=True,
                            )
                    except Exception as close_err:
                        logger.critical("%s emergency close also failed: %s", symbol, close_err)
                    return result
                await asyncio.sleep(0.2)

        # Place TP1 order
        try:
            if self.paper:
                result["tp_order"] = {"orderId": f"paper_tp1_{symbol}", "stopPrice": tp_price}
            else:
                result["tp_order"] = await self.client.place_futures_order(
                    symbol=symbol,
                    side=stop_side,
                    type="TAKE_PROFIT_MARKET",
                    quantity=tp_qty,
                    stop_price=tp_price,
                    reduce_only=True,
                )
            trade_logger.info(
                "TP1_PLACED\tsymbol=%s\tprice=%.4f\tqty=%.6f",
                symbol, tp_price, tp_qty,
            )
        except BinanceClientError as e:
            logger.error("%s TP1 order failed: %s", symbol, e)

        elapsed_ms = int(time.time() * 1000) - start_ms
        if elapsed_ms > bracket_deadline:
            logger.warning(
                "%s bracket orders took %dms (deadline %dms)",
                symbol, elapsed_ms, bracket_deadline,
            )

        return result

    # ======================================================================
    # Helpers
    # ======================================================================

    async def _wait_for_fill(self, symbol: str, order_id: int, timeout_s: float) -> Optional[dict]:
        """Poll for order fill up to *timeout_s*."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                orders = await self.client.get_futures_open_orders(symbol=symbol)
                found = False
                for o in orders:
                    if o.get("orderId") == order_id:
                        if o.get("status") == "FILLED":
                            return o
                        found = True
                        break
                if not found:
                    # Order disappeared — check user trades
                    trades = await self.client.get_futures_user_trades(symbol=symbol, limit=5)
                    for t in trades:
                        if t.get("orderId") == order_id:
                            return {
                                "orderId": order_id,
                                "status": "FILLED",
                                "executedQty": float(t.get("qty", 0)),
                                "avgPrice": float(t.get("price", 0)),
                            }
            except Exception as e:
                logger.warning("Fill check error for %s: %s", symbol, e)

            await asyncio.sleep(1.0)
        return None

    async def _get_best_price(self, symbol: str, side: str) -> float:
        """Get the current best bid (for BUY) or ask (for SELL)."""
        try:
            depth = await self.client.get_futures_depth(symbol, limit=5)
            if side == "BUY" and depth.get("bids"):
                return float(depth["bids"][0][0])
            elif side == "SELL" and depth.get("asks"):
                return float(depth["asks"][0][0])
        except Exception as e:
            logger.warning("Best price fetch failed for %s: %s", symbol, e)
        return 0.0

    def _fill_to_dict(self, fill_result: Any, symbol: str, side: str, order_type: str, attempt: int) -> dict:
        """Convert a paper trading FillResult to a standard dict."""
        return {
            "orderId": f"paper_{int(time.time() * 1000)}",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "status": "FILLED",
            "executedQty": fill_result.fill_quantity,
            "avgPrice": fill_result.fill_price,
            "fees": fill_result.fees,
            "slippage_bps": fill_result.slippage_bps,
            "attempt": attempt,
            "timestamp_ms": fill_result.timestamp_ms,
        }

    # ======================================================================
    # WebSocket message handlers
    # ======================================================================

    async def on_kline(self, symbol: str, timeframe: str, kline_data: dict) -> None:
        """Handle incoming kline data from WS manager."""
        is_closed = kline_data.get("x", False)
        candle = {
            "timestamp": kline_data.get("t", 0),
            "open": float(kline_data.get("o", 0)),
            "high": float(kline_data.get("h", 0)),
            "low": float(kline_data.get("l", 0)),
            "close": float(kline_data.get("c", 0)),
            "volume": float(kline_data.get("v", 0)),
        }

        buf = self.buffers[symbol][timeframe]

        if is_closed:
            buf.add_candle(candle)

            # On 4h close, check for new signals
            if timeframe == "4h":
                signal = self.check_for_signal(symbol)
                if signal:
                    self.pending_signals[symbol] = signal
                # Clear agg trade buffer after 4h candle close
                self.clear_agg_trades(symbol)

        else:
            # Update the latest candle in-place for real-time price
            if len(buf) > 0:
                buf._highs[-1] = max(buf._highs[-1], candle["high"])
                buf._lows[-1] = min(buf._lows[-1], candle["low"])
                buf._closes[-1] = candle["close"]
                buf._volumes[-1] = candle["volume"]

    async def on_agg_trade(self, symbol: str, trade_data: dict) -> None:
        """Handle aggregated trade for volume tracking."""
        ts_ms = trade_data.get("T", 0)
        qty = float(trade_data.get("q", 0))
        self.record_agg_trade(symbol, ts_ms // 1000, qty)

    async def on_order_update(self, data: dict) -> None:
        """Handle ORDER_TRADE_UPDATE from user data stream."""
        symbol = data.get("s", "")
        order_id = data.get("i", 0)
        status = data.get("X", "")
        side = data.get("S", "")
        order_type = data.get("o", "")
        filled_qty = float(data.get("z", 0))
        avg_price = float(data.get("ap", 0))
        realized_pnl = float(data.get("rp", 0))

        trade_logger.info(
            "ORDER_UPDATE\tsymbol=%s\toid=%s\tstatus=%s\tside=%s\ttype=%s\t"
            "filled=%.6f\tavg_price=%.4f\trpnl=%.4f",
            symbol, order_id, status, side, order_type, filled_qty, avg_price, realized_pnl,
        )

        # Check if this is a fill for a pending entry
        if symbol in self.pending_signals:
            sig = self.pending_signals[symbol]
            if sig.current_order_id and sig.current_order_id == order_id and status == "FILLED":
                sig.status = SignalStatus.FILLED
                sig.filled_qty = filled_qty
                sig.filled_price = avg_price

    # ======================================================================
    # State serialization
    # ======================================================================

    def get_pending_signals_state(self) -> List[dict]:
        """Return pending signals as serializable dicts."""
        signals = []
        for sym, sig in self.pending_signals.items():
            signals.append({
                "symbol": sig.symbol,
                "direction": sig.direction.value,
                "status": sig.status.value,
                "signal_time_ms": sig.signal_time_ms,
                "crossover_price": sig.crossover_price,
                "atr_value": sig.atr_value,
                "adx_value": sig.adx_value,
                "rsi_value": sig.rsi_value,
                "size_multiplier": sig.size_multiplier,
                "manipulation_flagged": sig.manipulation_flagged,
                "pullback_deadline_ms": sig.pullback_deadline_ms,
                "filled_qty": sig.filled_qty,
                "filled_price": sig.filled_price,
            })
        return signals

    def get_indicator_state(self) -> Dict[str, Dict[str, dict]]:
        """Return current indicator snapshots for all symbols."""
        result: Dict[str, Dict[str, dict]] = {}
        for symbol in self.buffers:
            result[symbol] = {}
            for tf in self.TIMEFRAMES:
                if tf in ("4h", "1d"):
                    snap = self.compute_indicators(symbol, tf)
                    result[symbol][tf] = {
                        "ema_20": snap.ema_20,
                        "ema_50": snap.ema_50,
                        "ema_200": snap.ema_200,
                        "macd_histogram": snap.macd_histogram,
                        "rsi": snap.rsi_value,
                        "adx": snap.adx_value,
                        "plus_di": snap.plus_di,
                        "minus_di": snap.minus_di,
                        "atr": snap.atr_value,
                        "volume_ratio": snap.volume_ratio,
                    }
        return result
