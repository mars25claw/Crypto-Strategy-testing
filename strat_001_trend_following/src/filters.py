"""STRAT-001 Filters & Disqualifiers — Section 7.

Complete filter pipeline:
  7.1 Spread: <0.1%, re-check within 100ms of submission
  7.2 Whipsaw: 8h cooldown, 2 confirm candles, 3/week -> suspend 1 week
  7.3 Consecutive loss: 3->25%, 5->50%, 7->halt (delegated to risk_manager)
  7.4 Fee threshold: 2x ATR/price must >0.24%
  7.5 Correlation: >2 same-direction highly-correlated -> reject
  7.6 News/event: 2h before + 1h after high-impact events
  7.7 Hurst regime: <0.45 reject, 0.45-0.55 50%, >0.55 normal
  7.8 OI divergence: price highs + OI declining -> 50% size
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.indicators import hurst_exponent
from shared.risk_manager import RiskManager, CorrelationMatrix

from . import STRATEGY_ID
from .strategy import SignalDirection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter result
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Outcome of the filter pipeline."""
    passed: bool = True
    reject_reason: str = ""
    size_multiplier: float = 1.0
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Event calendar entry
# ---------------------------------------------------------------------------

@dataclass
class ScheduledEvent:
    """A known high-impact event on the calendar."""
    name: str
    timestamp_utc: float       # Unix timestamp
    impact: str = "HIGH"       # HIGH, MEDIUM
    pre_hours: float = 2.0
    post_hours: float = 1.0


# ---------------------------------------------------------------------------
# FilterEngine
# ---------------------------------------------------------------------------

class FilterEngine:
    """Section 7 filter pipeline for STRAT-001.

    Parameters
    ----------
    config : dict
        ``strategy_params`` from config.yaml.
    shared_risk : RiskManager
        Shared risk manager for cross-strategy checks.
    """

    def __init__(self, config: Dict[str, Any], shared_risk: RiskManager) -> None:
        self.cfg = config
        self.shared = shared_risk

        # Whipsaw tracking: symbol -> list of (timestamp_s, direction)
        self._stop_loss_events: Dict[str, List[Tuple[float, str]]] = defaultdict(list)

        # Whipsaw 2-candle confirmation: (symbol, direction) -> consecutive 4h candle count
        # in the new signal direction after a losing trade
        self._whipsaw_candle_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        # Track symbols that need 2-candle confirmation (set after a stop-loss)
        self._whipsaw_needs_confirmation: Dict[str, str] = {}  # symbol -> direction of loss

        # Whipsaw weekly counter: symbol -> count this week
        self._whipsaw_weekly: Dict[str, int] = defaultdict(int)
        self._whipsaw_weekly_reset_ts: float = 0.0

        # Instrument suspensions: symbol -> resume_timestamp
        self._suspended_until: Dict[str, float] = {}

        # Hurst exponents: symbol -> (value, last_calc_ts)
        self._hurst_cache: Dict[str, Tuple[float, float]] = {}

        # Event calendar
        self._events: List[ScheduledEvent] = []

        # Last spread values: symbol -> (bid, ask, timestamp)
        self._last_spread: Dict[str, Tuple[float, float, float]] = {}

        # OI tracking: symbol -> list of (timestamp, oi_value)
        self._oi_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        # Price highs tracking: symbol -> list of (timestamp, price)
        self._price_highs: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    # ======================================================================
    # Main filter pipeline
    # ======================================================================

    def evaluate(
        self,
        symbol: str,
        direction: SignalDirection,
        entry_price: float,
        atr_value: float,
        best_bid: float = 0.0,
        best_ask: float = 0.0,
    ) -> FilterResult:
        """Run the full Section 7 filter pipeline.

        Returns a :class:`FilterResult` indicating pass/fail and any
        size adjustments.
        """
        result = FilterResult()

        # 7.1 Spread filter
        spread_result = self._check_spread(symbol, best_bid, best_ask)
        if not spread_result[0]:
            result.passed = False
            result.reject_reason = spread_result[1]
            return result

        # Check suspension
        if self._is_suspended(symbol):
            result.passed = False
            result.reject_reason = f"{symbol} suspended due to whipsaw"
            return result

        # 7.2 Whipsaw protection
        whipsaw_result = self._check_whipsaw(symbol, direction.value)
        if not whipsaw_result[0]:
            result.passed = False
            result.reject_reason = whipsaw_result[1]
            return result

        # 7.4 Fee threshold
        fee_result = self._check_fee_threshold(atr_value, entry_price)
        if not fee_result[0]:
            result.passed = False
            result.reject_reason = fee_result[1]
            return result

        # 7.5 Correlation filter
        corr_result = self._check_correlation(symbol, direction.value)
        if not corr_result[0]:
            result.passed = False
            result.reject_reason = corr_result[1]
            return result

        # 7.6 News/event filter
        event_result = self._check_events()
        if not event_result[0]:
            result.passed = False
            result.reject_reason = event_result[1]
            return result

        # 7.7 Hurst regime filter
        hurst_result = self._check_hurst(symbol)
        if not hurst_result[0]:
            result.passed = False
            result.reject_reason = hurst_result[1]
            return result
        if hurst_result[2] < 1.0:
            result.size_multiplier *= hurst_result[2]
            result.warnings.append(f"Hurst={hurst_result[3]:.2f} -> {hurst_result[2]*100:.0f}% size")

        # 7.8 OI divergence
        oi_result = self._check_oi_divergence(symbol, direction)
        if oi_result[1] < 1.0:
            result.size_multiplier *= oi_result[1]
            result.warnings.append(f"OI_divergence -> {oi_result[1]*100:.0f}% size")

        return result

    # ======================================================================
    # 7.1 Spread filter
    # ======================================================================

    def update_spread(self, symbol: str, bid: float, ask: float) -> None:
        """Update the latest bid/ask from bookTicker stream."""
        self._last_spread[symbol] = (bid, ask, time.time())

    def _check_spread(self, symbol: str, bid: float, ask: float) -> Tuple[bool, str]:
        """Check if spread exceeds 0.1% of mid-price."""
        max_spread_pct = self.cfg.get("spread_max_pct", 0.1)

        # Use provided values or fall back to cached
        if bid <= 0 or ask <= 0:
            cached = self._last_spread.get(symbol)
            if cached:
                bid, ask, _ = cached
            else:
                return True, ""  # No data, allow

        if bid <= 0 or ask <= 0:
            return True, ""

        mid = (bid + ask) / 2.0
        spread_pct = ((ask - bid) / mid) * 100.0

        if spread_pct > max_spread_pct:
            return False, f"Spread {spread_pct:.3f}% > {max_spread_pct}% for {symbol}"

        return True, ""

    def recheck_spread_before_order(self, symbol: str) -> bool:
        """Re-check spread within 100ms of order submission (Section 7.1)."""
        cached = self._last_spread.get(symbol)
        if not cached:
            return True

        bid, ask, ts = cached
        # Must be recent (within 1 second)
        if time.time() - ts > 1.0:
            return True  # Stale data, allow but log

        max_spread_pct = self.cfg.get("spread_max_pct", 0.1)
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return True

        spread_pct = ((ask - bid) / mid) * 100.0
        if spread_pct > max_spread_pct:
            logger.warning(
                "%s spread check failed at order time: %.3f%% > %.1f%%",
                symbol, spread_pct, max_spread_pct,
            )
            return False
        return True

    # ======================================================================
    # 7.2 Whipsaw protection
    # ======================================================================

    def record_stop_loss(self, symbol: str, direction: str) -> None:
        """Record a stop-loss event for whipsaw tracking.

        After a losing trade, require 2 consecutive 4h candles closing in
        the new signal direction before allowing re-entry (Section 7.2).
        """
        now = time.time()
        self._stop_loss_events[symbol].append((now, direction))
        self._check_weekly_whipsaw(symbol)

        # Mark this symbol as needing 2-candle confirmation for any new direction
        self._whipsaw_needs_confirmation[symbol] = direction
        # Reset candle counts for all directions on this symbol
        for d in ("LONG", "SHORT"):
            self._whipsaw_candle_counts[(symbol, d)] = 0
        logger.info(
            "%s stop-loss recorded (%s) — requiring 2-candle confirmation for re-entry",
            symbol, direction,
        )

        # Prune old events (keep 30 days)
        cutoff = now - 30 * 86400
        self._stop_loss_events[symbol] = [
            e for e in self._stop_loss_events[symbol] if e[0] > cutoff
        ]

    def update_4h_candle(self, symbol: str, close_price: float, ema_20: float) -> None:
        """Update whipsaw 2-candle confirmation counters on each 4h candle close.

        Tracks consecutive 4h candles closing in a given direction (above EMA20 = LONG,
        below = SHORT). Resets count if the candle does not confirm the direction.

        Parameters
        ----------
        symbol : str
            The trading pair.
        close_price : float
            The 4h candle close price.
        ema_20 : float
            The 4h EMA(20) value at candle close.
        """
        if symbol not in self._whipsaw_needs_confirmation:
            return

        # Determine candle direction
        if close_price > ema_20:
            candle_dir = "LONG"
        elif close_price < ema_20:
            candle_dir = "SHORT"
        else:
            # Exactly at EMA — does not confirm either direction, reset both
            for d in ("LONG", "SHORT"):
                self._whipsaw_candle_counts[(symbol, d)] = 0
            return

        # Increment confirming direction, reset opposite
        self._whipsaw_candle_counts[(symbol, candle_dir)] += 1
        opposite = "SHORT" if candle_dir == "LONG" else "LONG"
        self._whipsaw_candle_counts[(symbol, opposite)] = 0

        count = self._whipsaw_candle_counts[(symbol, candle_dir)]
        if count >= 2:
            logger.info(
                "%s whipsaw 2-candle confirmation met for %s (%d candles)",
                symbol, candle_dir, count,
            )

    def _check_whipsaw(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """Check whipsaw cooldown: 8h minimum after stop on same direction,
        AND require 2 consecutive 4h candles closing in the new signal direction."""
        cooldown_candles = self.cfg.get("whipsaw_cooldown_candles_4h", 2)
        cooldown_s = cooldown_candles * 4 * 3600  # 8 hours
        now = time.time()

        events = self._stop_loss_events.get(symbol, [])
        for ts, evt_dir in reversed(events):
            if evt_dir == direction and (now - ts) < cooldown_s:
                remaining = cooldown_s - (now - ts)
                return False, (
                    f"Whipsaw cooldown: {symbol} {direction} stopped out "
                    f"{(now - ts) / 3600:.1f}h ago, {remaining / 3600:.1f}h remaining"
                )

        # 2-candle confirmation check: after a losing trade, require 2 consecutive
        # 4h candles closing in the new signal direction before re-entry
        if symbol in self._whipsaw_needs_confirmation:
            confirm_count = self._whipsaw_candle_counts.get((symbol, direction), 0)
            required = self.cfg.get("whipsaw_confirm_candles", 2)
            if confirm_count < required:
                return False, (
                    f"Whipsaw 2-candle confirmation: {symbol} {direction} has only "
                    f"{confirm_count}/{required} confirming 4h candles after loss"
                )
            # Confirmation met — clear the requirement for this symbol
            self._whipsaw_needs_confirmation.pop(symbol, None)

        return True, ""

    def _check_weekly_whipsaw(self, symbol: str) -> None:
        """Check if 3+ whipsaws in a week -> suspend instrument."""
        now = time.time()
        max_per_week = self.cfg.get("whipsaw_max_per_week", 3)
        suspend_days = self.cfg.get("whipsaw_suspend_days", 7)

        # Reset weekly counter on Monday
        if now - self._whipsaw_weekly_reset_ts > 7 * 86400:
            self._whipsaw_weekly.clear()
            self._whipsaw_weekly_reset_ts = now

        self._whipsaw_weekly[symbol] = self._whipsaw_weekly.get(symbol, 0) + 1

        if self._whipsaw_weekly[symbol] >= max_per_week:
            suspend_until = now + suspend_days * 86400
            self._suspended_until[symbol] = suspend_until
            logger.warning(
                "%s suspended for %d days: %d whipsaws this week",
                symbol, suspend_days, self._whipsaw_weekly[symbol],
            )

    def _is_suspended(self, symbol: str) -> bool:
        """Check if an instrument is currently suspended."""
        resume_ts = self._suspended_until.get(symbol, 0)
        if resume_ts > time.time():
            return True
        if resume_ts > 0:
            self._suspended_until.pop(symbol, None)
        return False

    # ======================================================================
    # 7.4 Fee threshold
    # ======================================================================

    def _check_fee_threshold(self, atr_value: float, entry_price: float) -> Tuple[bool, str]:
        """Check that expected profit exceeds 3x round-trip fees."""
        if entry_price <= 0 or atr_value <= 0:
            return True, ""

        # 2x ATR / price must > 0.24% (3x the 0.08% round-trip fee)
        hard_stop_mult = self.cfg.get("hard_stop_atr_mult", 2.0)
        profit_pct = (hard_stop_mult * atr_value / entry_price) * 100.0
        min_profit_pct = 0.24  # 3 x 0.08%

        if profit_pct < min_profit_pct:
            return False, (
                f"Fee threshold: 2xATR/price = {profit_pct:.3f}% < {min_profit_pct}% "
                f"(too low volatility)"
            )
        return True, ""

    # ======================================================================
    # 7.5 Correlation filter
    # ======================================================================

    def _check_correlation(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """Check cross-strategy correlation limits."""
        threshold = self.cfg.get("correlation_threshold", 0.75)
        max_same_dir = self.cfg.get("correlation_max_same_dir", 2)

        correlated_count = 0
        highly_correlated_symbols = self.shared.correlation_matrix.get_highly_correlated(
            symbol, threshold
        )

        for strat_id, sym_map in self.shared._positions.items():
            for sym, pos in sym_map.items():
                if sym in highly_correlated_symbols and pos.direction == direction.upper():
                    correlated_count += 1

        if correlated_count >= max_same_dir:
            return False, (
                f"Correlation filter: {correlated_count} {direction} positions "
                f"in highly correlated assets (threshold={threshold})"
            )
        return True, ""

    # ======================================================================
    # 7.6 News/event filter
    # ======================================================================

    def add_event(self, event: ScheduledEvent) -> None:
        """Add an event to the calendar."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.timestamp_utc)

    def load_events(self, events: List[Dict[str, Any]]) -> None:
        """Load events from a list of dicts."""
        for e in events:
            self._events.append(ScheduledEvent(
                name=e.get("name", "Unknown"),
                timestamp_utc=e.get("timestamp_utc", 0),
                impact=e.get("impact", "HIGH"),
                pre_hours=e.get("pre_hours", 2.0),
                post_hours=e.get("post_hours", 1.0),
            ))
        self._events.sort(key=lambda e: e.timestamp_utc)

    def _check_events(self) -> Tuple[bool, str]:
        """Check if we're within an event blackout window."""
        now = time.time()
        pre_hours = self.cfg.get("news_pre_event_hours", 2)
        post_hours = self.cfg.get("news_post_event_hours", 1)

        for event in self._events:
            event_start = event.timestamp_utc - (pre_hours * 3600)
            event_end = event.timestamp_utc + (post_hours * 3600)

            if event_start <= now <= event_end:
                return False, (
                    f"Event blackout: {event.name} at "
                    f"{datetime.fromtimestamp(event.timestamp_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
                )

        return True, ""

    def is_in_event_window(self) -> bool:
        """Check if currently in any event window (for stop tightening)."""
        now = time.time()
        for event in self._events:
            pre_hours = self.cfg.get("news_pre_event_hours", 2)
            post_hours = self.cfg.get("news_post_event_hours", 1)
            if event.timestamp_utc - (pre_hours * 3600) <= now <= event.timestamp_utc + (post_hours * 3600):
                return True
        return False

    # ======================================================================
    # 7.7 Hurst regime filter
    # ======================================================================

    def update_hurst(self, symbol: str, daily_closes: np.ndarray) -> float:
        """Recalculate Hurst exponent for *symbol* from 100-day daily closes."""
        if len(daily_closes) < 100:
            return float("nan")

        h = hurst_exponent(daily_closes[-100:])
        self._hurst_cache[symbol] = (h, time.time())
        logger.info("%s Hurst exponent updated: %.4f", symbol, h)
        return h

    def _check_hurst(self, symbol: str) -> Tuple[bool, str, float, float]:
        """Check Hurst exponent regime filter.

        Returns (passed, reason, size_multiplier, hurst_value).
        """
        cached = self._hurst_cache.get(symbol)
        if not cached:
            return True, "", 1.0, 0.5  # No data, allow

        h, calc_ts = cached
        if not np.isfinite(h):
            return True, "", 1.0, 0.5

        reject_below = self.cfg.get("hurst_reject_below", 0.45)
        reduce_below = self.cfg.get("hurst_reduce_below", 0.55)

        if h < reject_below:
            return False, f"Hurst={h:.3f} < {reject_below} (mean-reverting regime)", 0.0, h
        elif h < reduce_below:
            return True, "", 0.5, h  # Random walk, 50% size
        else:
            return True, "", 1.0, h  # Trending, normal size

    # ======================================================================
    # 7.8 OI divergence
    # ======================================================================

    def update_open_interest(self, symbol: str, oi_value: float, price: float) -> None:
        """Record open interest data point."""
        now = time.time()
        self._oi_history[symbol].append((now, oi_value))
        self._price_highs[symbol].append((now, price))

        # Keep only last 7 days
        cutoff = now - 7 * 86400
        self._oi_history[symbol] = [
            e for e in self._oi_history[symbol] if e[0] > cutoff
        ]
        self._price_highs[symbol] = [
            e for e in self._price_highs[symbol] if e[0] > cutoff
        ]

    def _check_oi_divergence(
        self, symbol: str, direction: SignalDirection
    ) -> Tuple[bool, float]:
        """Check for OI divergence.

        Returns (has_divergence, size_multiplier).
        """
        oi_data = self._oi_history.get(symbol, [])
        price_data = self._price_highs.get(symbol, [])

        if len(oi_data) < 10 or len(price_data) < 10:
            return False, 1.0

        oi_size_pct = self.cfg.get("oi_divergence_size_pct", 50) / 100.0

        # Get recent OI and price trends (last 24h vs previous 24h)
        now = time.time()
        recent_cutoff = now - 24 * 3600
        older_cutoff = now - 48 * 3600

        recent_oi = [v for t, v in oi_data if t > recent_cutoff]
        older_oi = [v for t, v in oi_data if older_cutoff < t <= recent_cutoff]
        recent_prices = [v for t, v in price_data if t > recent_cutoff]
        older_prices = [v for t, v in price_data if older_cutoff < t <= recent_cutoff]

        if not recent_oi or not older_oi or not recent_prices or not older_prices:
            return False, 1.0

        oi_rising = np.mean(recent_oi) > np.mean(older_oi)
        price_rising = max(recent_prices) > max(older_prices)
        price_falling = min(recent_prices) < min(older_prices)

        # Price highs + OI declining -> bearish divergence for longs
        if direction == SignalDirection.LONG and price_rising and not oi_rising:
            logger.info("%s OI divergence (bearish): price new highs, OI declining", symbol)
            return True, oi_size_pct

        # Price lows + OI declining -> shorts covering, bearish for shorts
        if direction == SignalDirection.SHORT and price_falling and not oi_rising:
            logger.info("%s OI divergence: price new lows, OI declining (shorts covering)", symbol)
            return True, oi_size_pct

        return False, 1.0

    # ======================================================================
    # Volume spike detection (Section 7.6 unexpected events)
    # ======================================================================

    def check_sudden_volume_spike(
        self, volumes: Dict[str, float], avg_volumes: Dict[str, float]
    ) -> bool:
        """Detect 3x+ volume spike across multiple assets simultaneously.

        Returns True if an unexpected event is detected.
        """
        spike_count = 0
        for symbol in volumes:
            vol = volumes[symbol]
            avg = avg_volumes.get(symbol, vol)
            if avg > 0 and vol / avg >= 3.0:
                spike_count += 1

        if spike_count >= 3:
            logger.warning(
                "Sudden volume spike detected across %d assets — 30min entry pause",
                spike_count,
            )
            return True
        return False

    # ======================================================================
    # State
    # ======================================================================

    def get_state(self) -> dict:
        """Return filter state for persistence/dashboard."""
        return {
            "suspended_instruments": {
                sym: datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                for sym, ts in self._suspended_until.items()
                if ts > time.time()
            },
            "whipsaw_weekly_counts": dict(self._whipsaw_weekly),
            "whipsaw_pending_confirmation": {
                sym: {
                    "lost_direction": d,
                    "long_candles": self._whipsaw_candle_counts.get((sym, "LONG"), 0),
                    "short_candles": self._whipsaw_candle_counts.get((sym, "SHORT"), 0),
                }
                for sym, d in self._whipsaw_needs_confirmation.items()
            },
            "hurst_exponents": {
                sym: {"value": h, "calc_ts": ts}
                for sym, (h, ts) in self._hurst_cache.items()
            },
            "upcoming_events": [
                {"name": e.name, "timestamp": e.timestamp_utc, "impact": e.impact}
                for e in self._events
                if e.timestamp_utc > time.time()
            ][:10],
        }
