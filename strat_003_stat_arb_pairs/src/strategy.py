"""Core pairs trading strategy logic.

Implements spread monitoring on 1h candles, Z-score signal generation,
entry confirmation (persistence, volume, regime, trend), simultaneous
dual-leg execution, all exit logic (mean reversion, opposite signal,
hard stop, dollar stop, time exit, cointegration breakdown),
rebalancing, and edge-case handling.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import numpy as np

from src.cointegration import CointegrationEngine, PairParameters

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PairPosition:
    """Tracks an open pair trade (both legs)."""

    pair_id: str
    asset_a: str
    asset_b: str
    direction: str            # "LONG_SPREAD" or "SHORT_SPREAD"
    entry_z_score: float
    entry_time: float
    hedge_ratio: float        # beta at entry
    half_life_hours: float

    # Leg A
    leg_a_side: str           # "BUY" or "SELL"
    leg_a_qty: float
    leg_a_entry_price: float

    # Leg B
    leg_b_side: str           # "BUY" or "SELL"
    leg_b_qty: float
    leg_b_entry_price: float

    # Fill times (defaults)
    leg_a_fill_time: float = 0.0
    leg_b_fill_time: float = 0.0

    # Running state
    current_z_score: float = 0.0
    peak_z_score: float = 0.0
    current_pnl: float = 0.0
    current_pnl_pct: float = 0.0
    rebalance_count: int = 0
    partial_exit_done: bool = False

    # Execution state
    is_fully_entered: bool = False
    pending_exit: bool = False
    exit_reason: str = ""


@dataclass
class SpreadState:
    """Maintains spread and Z-score history for a qualified pair."""

    asset_a: str
    asset_b: str
    spread_history: deque = field(default_factory=lambda: deque(maxlen=200))
    z_score_history: deque = field(default_factory=lambda: deque(maxlen=200))
    current_spread: float = 0.0
    current_z_score: float = 0.0
    rolling_mean: float = 0.0
    rolling_std: float = 0.0

    # Entry confirmation state
    consecutive_beyond_threshold: int = 0
    last_z_direction: float = 0.0  # Track if Z is moving toward mean

    # Volume tracking
    volume_a_history: deque = field(default_factory=lambda: deque(maxlen=25))
    volume_b_history: deque = field(default_factory=lambda: deque(maxlen=25))


@dataclass
class SignalEvent:
    """A detected signal awaiting confirmation."""

    asset_a: str
    asset_b: str
    direction: str            # "LONG_SPREAD" or "SHORT_SPREAD"
    z_score: float
    first_detection_time: float
    consecutive_count: int = 1
    confirmed: bool = False


# ---------------------------------------------------------------------------
# Strategy Engine
# ---------------------------------------------------------------------------

class PairsStrategy:
    """Core pairs trading strategy engine.

    Parameters
    ----------
    config : dict
        Strategy parameters from config.yaml strategy_params section.
    coint_engine : CointegrationEngine
        The cointegration testing engine with qualified pairs.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        coint_engine: CointegrationEngine,
    ) -> None:
        self._cfg = config
        self._coint = coint_engine

        # Thresholds from config
        self._z_entry = config.get("z_score_entry", 2.0)
        self._z_exit = config.get("z_score_exit", 0.5)
        self._z_hard_stop = config.get("z_score_hard_stop", 3.5)
        self._z_circuit = config.get("z_score_circuit_breaker", 5.0)
        self._z_opposite = config.get("z_score_opposite", 2.0)
        self._z_post_stop = config.get("z_score_post_stop_entry", 2.5)
        self._rolling_window = config.get("rolling_window", 50)
        self._z_persistence = config.get("z_persistence_bars", 2)
        self._vol_mult = config.get("volume_multiplier", 1.2)
        self._regime_pct = config.get("regime_change_pct", 10.0)
        self._regime_hours = config.get("regime_change_hours", 24)
        self._sim_window_ms = config.get("simultaneous_window_ms", 200)
        self._limit_timeout = config.get("limit_timeout_s", 60)
        self._hedge_correction = config.get("hedge_correction_threshold", 0.05)
        self._dollar_stop_pct = config.get("dollar_stop_pct", 1.5)
        self._time_exit_mult = config.get("time_exit_multiplier", 2.0)
        self._partial_exit_pct = config.get("partial_exit_pct", 50.0)
        self._breakdown_adf_p = config.get("cointegration_breakdown_adf_p", 0.10)
        self._breakdown_exit_h = config.get("breakdown_exit_hours", 4)
        self._beta_drift = config.get("beta_drift_threshold", 0.10)
        self._max_rebalances = config.get("max_rebalances_per_trade", 2)
        self._rebalance_max_pct = config.get("rebalance_max_pct", 20.0)
        self._min_spread_bps = config.get("min_spread_bps", 5)
        self._min_volume = config.get("min_24h_volume", 30_000_000)
        self._fee_mult = config.get("fee_multiplier", 3.0)
        self._taker_fee = config.get("taker_fee_pct", 0.04) / 100.0
        self._consec_loss_reduce = config.get("consecutive_loss_reduce", 3)
        self._consec_loss_halt = config.get("consecutive_loss_halt", 5)
        self._high_corr_regime = config.get("high_correlation_regime", 0.90)
        self._btc_flash_pct = config.get("btc_flash_pct", 8.0)
        self._btc_flash_window_h = config.get("btc_flash_window_hours", 4)
        self._btc_flash_halt_h = config.get("btc_flash_halt_hours", 4)
        self._spread_explosion_z = config.get("spread_explosion_z", 5.0)
        self._spread_explosion_halt_h = config.get("spread_explosion_halt_hours", 24)
        self._multi_stop_count = config.get("multi_stop_count", 3)
        self._multi_stop_window_h = config.get("multi_stop_window_hours", 24)
        self._multi_stop_halt_h = config.get("multi_stop_halt_hours", 48)

        # State
        self._positions: Dict[str, PairPosition] = {}  # pair_id -> PairPosition
        self._spread_states: Dict[str, SpreadState] = {}  # "A|B" -> SpreadState
        self._pending_signals: Dict[str, SignalEvent] = {}  # "A|B" -> SignalEvent
        self._stop_timestamps: List[float] = []  # timestamps of stop-loss events
        self._consecutive_losses: int = 0
        self._halted_until: float = 0.0
        self._btc_halt_until: float = 0.0

        # Price caches (updated by WS callbacks)
        self._prices_1h: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._prices_1m: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
        self._prices_1d: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._mark_prices: Dict[str, float] = {}
        self._book_tickers: Dict[str, Dict[str, float]] = {}  # symbol -> {bid, ask}
        self._volume_1h: Dict[str, deque] = defaultdict(lambda: deque(maxlen=25))
        self._ticker_24h: Dict[str, Dict[str, Any]] = {}

        # Callbacks for execution (set by main.py)
        self._execute_order_fn: Optional[Callable] = None
        self._get_equity_fn: Optional[Callable] = None
        self._risk_check_fn: Optional[Callable] = None

        # Performance tracking
        self._trade_history: List[Dict[str, Any]] = []
        self._alerts: List[Dict[str, Any]] = []

    # ======================================================================
    #  Public API - Callbacks registration
    # ======================================================================

    def set_execution_callbacks(
        self,
        execute_order: Callable,
        get_equity: Callable,
        risk_check: Callable,
    ) -> None:
        """Register callbacks for order execution, equity query, and risk checks."""
        self._execute_order_fn = execute_order
        self._get_equity_fn = get_equity
        self._risk_check_fn = risk_check

    # ======================================================================
    #  WebSocket data handlers
    # ======================================================================

    async def on_kline_1h(self, data: Dict[str, Any]) -> None:
        """Handle 1h kline updates — primary signal timeframe."""
        k = data.get("k", {})
        symbol = k.get("s", "")
        is_closed = k.get("x", False)
        close = float(k.get("c", 0))
        volume = float(k.get("v", 0))

        if not symbol or close <= 0:
            return

        self._prices_1h[symbol].append(close)
        self._volume_1h[symbol].append(volume)

        if is_closed:
            await self._update_spreads_and_check_signals(symbol)

    async def on_kline_1m(self, data: Dict[str, Any]) -> None:
        """Handle 1m kline updates — position monitoring."""
        k = data.get("k", {})
        symbol = k.get("s", "")
        close = float(k.get("c", 0))
        if symbol and close > 0:
            self._prices_1m[symbol].append(close)

        # Monitor active positions
        if k.get("x", False):
            await self._monitor_positions()

    async def on_kline_1d(self, data: Dict[str, Any]) -> None:
        """Handle daily kline updates."""
        k = data.get("k", {})
        symbol = k.get("s", "")
        close = float(k.get("c", 0))
        if symbol and close > 0:
            self._prices_1d[symbol].append(close)

    async def on_kline_5m(self, data: Dict[str, Any]) -> None:
        """Handle 5m kline updates — entry timing refinement."""
        # Used for execution timing; no persistent state needed
        pass

    async def on_book_ticker(self, data: Dict[str, Any]) -> None:
        """Handle book ticker updates — best bid/ask."""
        symbol = data.get("s", "")
        if symbol:
            self._book_tickers[symbol] = {
                "bid": float(data.get("b", 0)),
                "ask": float(data.get("a", 0)),
            }

    async def on_mark_price(self, data: Dict[str, Any]) -> None:
        """Handle mark price updates — PnL tracking."""
        symbol = data.get("s", "")
        price = float(data.get("p", 0))
        if symbol and price > 0:
            self._mark_prices[symbol] = price

        # Update PnL for active positions
        self._update_position_pnl()

    async def on_depth(self, data: Dict[str, Any]) -> None:
        """Handle depth updates — liquidity assessment."""
        # Stored for execution quality checks
        pass

    # ======================================================================
    #  Spread Monitoring & Signal Generation (Section 3.2, 3.3)
    # ======================================================================

    async def _update_spreads_and_check_signals(self, updated_symbol: str) -> None:
        """Recalculate spreads and Z-scores for all pairs involving updated_symbol."""
        for pair_params in self._coint.active_pairs:
            a, b = pair_params.asset_a, pair_params.asset_b
            if updated_symbol not in (a, b):
                continue

            key = f"{a}|{b}"
            state = self._spread_states.get(key)
            if state is None:
                state = SpreadState(asset_a=a, asset_b=b)
                self._spread_states[key] = state

            # Need close prices for both assets
            prices_a = list(self._prices_1h.get(a, []))
            prices_b = list(self._prices_1h.get(b, []))
            if len(prices_a) < 2 or len(prices_b) < 2:
                continue

            close_a = prices_a[-1]
            close_b = prices_b[-1]
            if close_a <= 0 or close_b <= 0:
                continue

            # Calculate spread: ln(Close_A) - beta * ln(Close_B)
            spread = math.log(close_a) - pair_params.hedge_ratio * math.log(close_b)
            state.spread_history.append(spread)
            state.current_spread = spread

            # Calculate Z-score using rolling 50-period
            if len(state.spread_history) >= self._rolling_window:
                recent = list(state.spread_history)[-self._rolling_window:]
                mu = float(np.mean(recent))
                sigma = float(np.std(recent))
                if sigma > 0:
                    z = (spread - mu) / sigma
                    state.current_z_score = z
                    state.rolling_mean = mu
                    state.rolling_std = sigma
                    state.z_score_history.append(z)

                    # Update volume tracking
                    vol_a = list(self._volume_1h.get(a, []))
                    vol_b = list(self._volume_1h.get(b, []))
                    if vol_a:
                        state.volume_a_history.append(vol_a[-1])
                    if vol_b:
                        state.volume_b_history.append(vol_b[-1])

                    # Check for signals
                    await self._check_signal(key, pair_params, state, z)

                    # Update active positions Z-score
                    self._update_position_z_scores(a, b, z)

    async def _check_signal(
        self,
        key: str,
        params: PairParameters,
        state: SpreadState,
        z: float,
    ) -> None:
        """Check for entry/exit signals on a pair."""
        # Check halts
        if self._is_halted():
            return

        # Circuit breaker: Z >= 5.0 => close all, halt 24h
        if abs(z) >= self._spread_explosion_z:
            await self._handle_spread_explosion(key, z)
            return

        # Check existing positions for exits
        for pos in list(self._positions.values()):
            if pos.asset_a == params.asset_a and pos.asset_b == params.asset_b:
                await self._check_exits(pos, z, params)
                return  # Already in this pair, don't check entry

        # No position - check entry
        entry_z = params.post_stop_entry_z if params.consecutive_stops > 0 else self._z_entry

        if z < -entry_z:
            await self._handle_entry_signal(key, params, state, z, "LONG_SPREAD")
        elif z > entry_z:
            await self._handle_entry_signal(key, params, state, z, "SHORT_SPREAD")
        else:
            # Z is within threshold, clear any pending signal
            self._pending_signals.pop(key, None)

    async def _handle_entry_signal(
        self,
        key: str,
        params: PairParameters,
        state: SpreadState,
        z: float,
        direction: str,
    ) -> None:
        """Process an entry signal with confirmation checks."""
        pending = self._pending_signals.get(key)

        if pending is None or pending.direction != direction:
            # New signal detected, start tracking
            self._pending_signals[key] = SignalEvent(
                asset_a=params.asset_a,
                asset_b=params.asset_b,
                direction=direction,
                z_score=z,
                first_detection_time=time.time(),
                consecutive_count=1,
            )
            logger.info(
                "Signal detected: %s/%s direction=%s Z=%.3f (awaiting confirmation)",
                params.asset_a, params.asset_b, direction, z,
            )
            return

        # Existing signal - increment consecutive count
        pending.consecutive_count += 1
        pending.z_score = z

        # Condition A: Z-score persistence (2 consecutive 1h closes)
        if pending.consecutive_count < self._z_persistence:
            return

        # Condition B: Volume confirmation
        if not self._check_volume_confirmation(state):
            logger.debug(
                "Signal %s/%s rejected: insufficient volume",
                params.asset_a, params.asset_b,
            )
            return

        # Condition C: No regime change
        if self._check_regime_change(params.asset_a) or \
           self._check_regime_change(params.asset_b):
            logger.info(
                "Signal %s/%s rejected: regime change detected",
                params.asset_a, params.asset_b,
            )
            self._pending_signals.pop(key, None)
            return

        # Condition D: Spread moving toward mean
        if not self._check_spread_turning(state, direction):
            # If diverging for 3+ bars, wait for reversal
            z_history = list(state.z_score_history)
            if len(z_history) >= 3:
                abs_z = [abs(zv) for zv in z_history[-3:]]
                if all(abs_z[i] > abs_z[i - 1] for i in range(1, len(abs_z))):
                    logger.debug(
                        "Signal %s/%s deferred: spread still diverging",
                        params.asset_a, params.asset_b,
                    )
                    return

        # Fee threshold check
        if not self._check_fee_threshold(params, z):
            logger.debug(
                "Signal %s/%s rejected: expected profit below fee threshold",
                params.asset_a, params.asset_b,
            )
            return

        # Spread filter (bid-ask)
        if not self._check_spread_filter(params.asset_a) or \
           not self._check_spread_filter(params.asset_b):
            logger.debug(
                "Signal %s/%s deferred: bid-ask spread too wide",
                params.asset_a, params.asset_b,
            )
            return

        # Volume filter (24h)
        if not self._check_volume_filter(params.asset_a) or \
           not self._check_volume_filter(params.asset_b):
            logger.debug(
                "Signal %s/%s rejected: 24h volume too low",
                params.asset_a, params.asset_b,
            )
            return

        # All confirmations passed - execute entry
        pending.confirmed = True
        logger.info(
            "Signal CONFIRMED: %s/%s direction=%s Z=%.3f",
            params.asset_a, params.asset_b, direction, z,
        )
        await self._execute_entry(params, state, z, direction)
        self._pending_signals.pop(key, None)

    # ======================================================================
    #  Entry Confirmation Helpers
    # ======================================================================

    def _check_volume_confirmation(self, state: SpreadState) -> bool:
        """At least one asset must have volume > 1.2x 20-period avg."""
        vol_a = list(state.volume_a_history)
        vol_b = list(state.volume_b_history)

        a_above = False
        b_above = False

        if len(vol_a) >= 2:
            avg_a = float(np.mean(vol_a[:-1][-20:])) if len(vol_a) > 1 else vol_a[0]
            if avg_a > 0 and vol_a[-1] > avg_a * self._vol_mult:
                a_above = True

        if len(vol_b) >= 2:
            avg_b = float(np.mean(vol_b[:-1][-20:])) if len(vol_b) > 1 else vol_b[0]
            if avg_b > 0 and vol_b[-1] > avg_b * self._vol_mult:
                b_above = True

        return a_above or b_above

    def _check_regime_change(self, symbol: str) -> bool:
        """Check if asset had > 10% move in last 24h."""
        prices = list(self._prices_1h.get(symbol, []))
        if len(prices) < 24:
            return False
        recent = prices[-24:]
        high = max(recent)
        low = min(recent)
        if low <= 0:
            return False
        move_pct = ((high - low) / low) * 100.0
        return move_pct > self._regime_pct

    def _check_spread_turning(self, state: SpreadState, direction: str) -> bool:
        """Check if spread is moving back toward the mean."""
        z_hist = list(state.z_score_history)
        if len(z_hist) < 2:
            return True  # Not enough data, allow entry

        current_z = z_hist[-1]
        prev_z = z_hist[-2]

        if direction == "LONG_SPREAD":
            # Z is negative, should be getting less negative (moving toward 0)
            return abs(current_z) < abs(prev_z)
        else:
            # Z is positive, should be getting less positive
            return abs(current_z) < abs(prev_z)

    def _check_fee_threshold(self, params: PairParameters, z: float) -> bool:
        """Expected reversion profit must exceed 3x round-trip fees."""
        # Expected spread reversion from entry to exit
        entry_z = abs(z)
        exit_z = self._z_exit
        expected_z_move = entry_z - exit_z

        if expected_z_move <= 0:
            return False

        # Convert to dollar terms (approximate)
        spread_move_pct = expected_z_move * params.spread_std * 100.0

        # Round-trip fees: 4 legs x taker fee
        total_fee_pct = 4 * self._taker_fee * 100.0

        return spread_move_pct > total_fee_pct * self._fee_mult

    def _check_spread_filter(self, symbol: str) -> bool:
        """Bid-ask spread must be < 0.05%."""
        bt = self._book_tickers.get(symbol)
        if bt is None:
            return True  # No data, allow
        bid, ask = bt.get("bid", 0), bt.get("ask", 0)
        if bid <= 0 or ask <= 0:
            return True
        mid = (bid + ask) / 2.0
        spread_bps = ((ask - bid) / mid) * 10000.0
        return spread_bps < self._min_spread_bps

    def _check_volume_filter(self, symbol: str) -> bool:
        """24h volume must be > $30M."""
        ticker = self._ticker_24h.get(symbol)
        if ticker is None:
            return True  # No data, allow
        vol = float(ticker.get("quoteVolume", 0))
        return vol >= self._min_volume

    # ======================================================================
    #  Execution (Section 3.4)
    # ======================================================================

    async def _execute_entry(
        self,
        params: PairParameters,
        state: SpreadState,
        z: float,
        direction: str,
    ) -> None:
        """Execute a pair trade entry: both legs within 200ms."""
        if self._execute_order_fn is None or self._get_equity_fn is None:
            logger.error("Execution callbacks not set, cannot enter trade")
            return

        equity = self._get_equity_fn()
        if equity <= 0:
            return

        # Position sizing (from risk_manager)
        if self._risk_check_fn:
            allowed, reason = self._risk_check_fn(params, direction, equity)
            if not allowed:
                logger.info(
                    "Entry blocked by risk manager: %s/%s - %s",
                    params.asset_a, params.asset_b, reason,
                )
                return

        # Calculate sizes
        sizes = self._calculate_position_sizes(params, equity)
        if sizes is None:
            return

        qty_a, qty_b, size_a_usdt, size_b_usdt = sizes

        # Determine sides
        if direction == "LONG_SPREAD":
            side_a = "BUY"   # Long A
            side_b = "SELL"  # Short B
        else:
            side_a = "SELL"  # Short A
            side_b = "BUY"   # Long B

        # Get best prices for limit orders
        bt_a = self._book_tickers.get(params.asset_a, {})
        bt_b = self._book_tickers.get(params.asset_b, {})
        price_a = bt_a.get("ask", 0) if side_a == "BUY" else bt_a.get("bid", 0)
        price_b = bt_b.get("ask", 0) if side_b == "BUY" else bt_b.get("bid", 0)

        if price_a <= 0 or price_b <= 0:
            logger.warning("Cannot get prices for entry, aborting")
            return

        pair_id = f"PAIR-{uuid.uuid4().hex[:8]}"

        # Place both orders simultaneously (within 200ms)
        order_a = {
            "symbol": params.asset_a,
            "side": side_a,
            "type": "LIMIT",
            "quantity": qty_a,
            "price": price_a,
            "pair_id": pair_id,
            "leg": "A",
        }
        order_b = {
            "symbol": params.asset_b,
            "side": side_b,
            "type": "LIMIT",
            "quantity": qty_b,
            "price": price_b,
            "pair_id": pair_id,
            "leg": "B",
        }

        trade_logger.info(
            "ENTRY\tpair=%s\t%s/%s\tdir=%s\tZ=%.3f\tbeta=%.4f\t"
            "qty_a=%.6f@%.4f\tqty_b=%.6f@%.4f",
            pair_id, params.asset_a, params.asset_b, direction, z,
            params.hedge_ratio, qty_a, price_a, qty_b, price_b,
        )

        # Execute both legs
        result_a, result_b = await asyncio.gather(
            self._execute_order_fn(order_a),
            self._execute_order_fn(order_b),
            return_exceptions=True,
        )

        # Handle one-leg-fill failure (Section 11.2)
        a_ok = not isinstance(result_a, Exception) and result_a is not None
        b_ok = not isinstance(result_b, Exception) and result_b is not None

        if not a_ok and not b_ok:
            logger.error("Both legs failed for pair %s", pair_id)
            self._add_alert("error", f"Entry failed: both legs failed for {params.asset_a}/{params.asset_b}")
            return

        if not a_ok or not b_ok:
            # One leg filled, other failed - handle within 120s
            failed_leg = "A" if not a_ok else "B"
            filled_leg = "B" if not a_ok else "A"
            logger.warning(
                "One-leg fill: pair %s leg %s filled, leg %s failed. "
                "Attempting recovery within 120s.",
                pair_id, filled_leg, failed_leg,
            )
            self._add_alert(
                "warning",
                f"One-leg fill on {params.asset_a}/{params.asset_b}: "
                f"leg {filled_leg} filled, retrying leg {failed_leg}",
            )
            # Retry the failed leg as MARKET
            retry_order = order_a if not a_ok else order_b
            retry_order["type"] = "MARKET"
            retry_order.pop("price", None)
            retry_result = await self._execute_order_fn(retry_order)
            if isinstance(retry_result, Exception) or retry_result is None:
                # Close the filled leg to avoid directional exposure
                close_order = order_b if not a_ok else order_a
                close_order["side"] = "SELL" if close_order["side"] == "BUY" else "BUY"
                close_order["type"] = "MARKET"
                close_order.pop("price", None)
                await self._execute_order_fn(close_order)
                logger.error("Failed to recover one-leg fill, closed filled leg")
                return
            # Update the result
            if not a_ok:
                result_a = retry_result
            else:
                result_b = retry_result

        # Verify hedge ratio
        fill_price_a = self._extract_fill_price(result_a, price_a)
        fill_price_b = self._extract_fill_price(result_b, price_b)

        actual_ratio = (qty_a * fill_price_a) / (qty_b * fill_price_b) if qty_b * fill_price_b > 0 else 0
        expected_ratio = params.hedge_ratio
        if expected_ratio > 0 and abs(actual_ratio - expected_ratio) / expected_ratio > self._hedge_correction:
            logger.info(
                "Hedge ratio off: actual=%.4f expected=%.4f, placing corrective order",
                actual_ratio, expected_ratio,
            )
            # Place corrective order on the smaller leg
            # (simplified: adjust leg B quantity)
            correction_qty = abs(qty_b * (expected_ratio / actual_ratio - 1.0)) if actual_ratio > 0 else 0
            if correction_qty > 0 and correction_qty < qty_b * 0.2:
                correction_side = side_b
                await self._execute_order_fn({
                    "symbol": params.asset_b,
                    "side": correction_side,
                    "type": "MARKET",
                    "quantity": correction_qty,
                    "pair_id": pair_id,
                    "leg": "B_correction",
                })

        # Create position record
        pos = PairPosition(
            pair_id=pair_id,
            asset_a=params.asset_a,
            asset_b=params.asset_b,
            direction=direction,
            entry_z_score=z,
            entry_time=time.time(),
            hedge_ratio=params.hedge_ratio,
            half_life_hours=params.half_life_days * 24.0,
            leg_a_side=side_a,
            leg_a_qty=qty_a,
            leg_a_entry_price=fill_price_a,
            leg_a_fill_time=time.time(),
            leg_b_side=side_b,
            leg_b_qty=qty_b,
            leg_b_entry_price=fill_price_b,
            leg_b_fill_time=time.time(),
            current_z_score=z,
            peak_z_score=z,
            is_fully_entered=True,
        )
        self._positions[pair_id] = pos

        self._add_alert(
            "info",
            f"Entered pair {params.asset_a}/{params.asset_b} {direction} at Z={z:.3f}",
        )

    def _calculate_position_sizes(
        self,
        params: PairParameters,
        equity: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Calculate position sizes based on hedge ratio and risk constraints.

        Returns (qty_a, qty_b, size_a_usdt, size_b_usdt) or None.
        """
        max_per_pair = equity * 0.05  # 5% per pair
        active_count = max(len(self._positions), 1)
        total_strategy = equity * 0.25
        capital_per_pair = min(max_per_pair, total_strategy / active_count)

        # Hedge-ratio-based sizing (Section 6.1)
        base_usdt = capital_per_pair / 2.0

        price_a = self._mark_prices.get(params.asset_a, 0)
        price_b = self._mark_prices.get(params.asset_b, 0)
        if price_a <= 0 or price_b <= 0:
            return None

        qty_a = base_usdt / price_a
        qty_b = (base_usdt * params.hedge_ratio) / price_b

        size_a = qty_a * price_a
        size_b = qty_b * price_b

        # Risk-based sizing (Section 6.2)
        max_loss = equity * 0.02  # 2% max loss per pair
        z_distance = self._z_hard_stop - self._z_entry
        if params.spread_std > 0 and z_distance > 0:
            max_spread_move = z_distance * params.spread_std
            risk_based_size = max_loss / max_spread_move if max_spread_move > 0 else capital_per_pair
            risk_based_per_leg = risk_based_size / 2.0
            # Use smaller of the two
            if risk_based_per_leg < base_usdt:
                scale = risk_based_per_leg / base_usdt
                qty_a *= scale
                qty_b *= scale
                size_a *= scale
                size_b *= scale

        # Marginal Hurst => reduced size
        if params.is_marginal:
            qty_a *= 0.7
            qty_b *= 0.7
            size_a *= 0.7
            size_b *= 0.7

        # Consecutive loss reduction
        if self._consecutive_losses >= self._consec_loss_reduce:
            qty_a *= 0.75
            qty_b *= 0.75
            size_a *= 0.75
            size_b *= 0.75

        if qty_a <= 0 or qty_b <= 0:
            return None

        return qty_a, qty_b, size_a, size_b

    # ======================================================================
    #  Exit Logic (Section 4)
    # ======================================================================

    async def _check_exits(
        self,
        pos: PairPosition,
        z: float,
        params: PairParameters,
    ) -> None:
        """Check all exit conditions for an open pair position."""
        if pos.pending_exit:
            return

        pos.current_z_score = z

        equity = self._get_equity_fn() if self._get_equity_fn else 0

        # 4.1 Mean reversion exit
        if pos.direction == "LONG_SPREAD" and z > -self._z_exit:
            await self._execute_exit(pos, "mean_reversion", z)
            return
        if pos.direction == "SHORT_SPREAD" and z < self._z_exit:
            await self._execute_exit(pos, "mean_reversion", z)
            return

        # 4.2 Opposite signal exit
        if pos.direction == "LONG_SPREAD" and z > self._z_opposite:
            await self._execute_exit(pos, "opposite_signal", z)
            return
        if pos.direction == "SHORT_SPREAD" and z < -self._z_opposite:
            await self._execute_exit(pos, "opposite_signal", z)
            return

        # 4.3 Hard stop: Z reaches +/- 3.5
        if abs(z) >= self._z_hard_stop:
            await self._execute_exit(pos, "hard_stop_z", z)
            self._record_stop(pos, params)
            return

        # 5.2 Dollar stop: combined PnL exceeds -1.5% equity
        if equity > 0 and pos.current_pnl < -(equity * self._dollar_stop_pct / 100.0):
            await self._execute_exit(pos, "dollar_stop", z)
            self._record_stop(pos, params)
            return

        # 4.4 Time-based exit
        max_duration_hours = pos.half_life_hours * self._time_exit_mult
        elapsed_hours = (time.time() - pos.entry_time) / 3600.0

        # Partial exit at half of max duration
        half_duration = max_duration_hours / 2.0
        if elapsed_hours >= half_duration and not pos.partial_exit_done:
            if abs(z) > 1.5:  # Z still beyond 1.5
                await self._execute_partial_exit(pos, self._partial_exit_pct / 100.0)
                pos.partial_exit_done = True

        if elapsed_hours >= max_duration_hours:
            await self._execute_exit(pos, "time_exit", z)
            return

        # 5.2 Trailing Z-score profit protection
        # If Z reverted past 0 and starts diverging again (re-crosses 0.5)
        if pos.direction == "LONG_SPREAD":
            if pos.peak_z_score > -self._z_exit and z < -self._z_exit:
                await self._execute_exit(pos, "trailing_z_protect", z)
                return
        if pos.direction == "SHORT_SPREAD":
            if pos.peak_z_score < self._z_exit and z > self._z_exit:
                await self._execute_exit(pos, "trailing_z_protect", z)
                return

        # Track peak Z (closest to 0)
        if pos.direction == "LONG_SPREAD":
            if z > pos.peak_z_score:
                pos.peak_z_score = z
        else:
            if z < pos.peak_z_score:
                pos.peak_z_score = z

        # 6.4 Rebalancing: check beta drift
        if pos.rebalance_count < self._max_rebalances:
            await self._check_rebalance(pos, params)

    async def _execute_exit(
        self,
        pos: PairPosition,
        reason: str,
        z: float,
    ) -> None:
        """Close both legs of a pair position within 200ms."""
        if self._execute_order_fn is None:
            return

        pos.pending_exit = True
        pos.exit_reason = reason

        # Close both legs simultaneously
        close_a = {
            "symbol": pos.asset_a,
            "side": "SELL" if pos.leg_a_side == "BUY" else "BUY",
            "type": "LIMIT",
            "quantity": pos.leg_a_qty,
            "price": self._get_close_price(pos.asset_a, pos.leg_a_side),
            "pair_id": pos.pair_id,
            "leg": "A_close",
            "reduce_only": True,
        }
        close_b = {
            "symbol": pos.asset_b,
            "side": "SELL" if pos.leg_b_side == "BUY" else "BUY",
            "type": "LIMIT",
            "quantity": pos.leg_b_qty,
            "price": self._get_close_price(pos.asset_b, pos.leg_b_side),
            "pair_id": pos.pair_id,
            "leg": "B_close",
            "reduce_only": True,
        }

        trade_logger.info(
            "EXIT\tpair=%s\t%s/%s\treason=%s\tentry_z=%.3f\texit_z=%.3f\tpnl=%.4f",
            pos.pair_id, pos.asset_a, pos.asset_b, reason,
            pos.entry_z_score, z, pos.current_pnl,
        )

        result_a, result_b = await asyncio.gather(
            self._execute_order_fn(close_a),
            self._execute_order_fn(close_b),
            return_exceptions=True,
        )

        # Handle failed close (convert to MARKET after 60s timeout)
        for leg_name, result, order in [("A", result_a, close_a), ("B", result_b, close_b)]:
            if isinstance(result, Exception) or result is None:
                logger.warning(
                    "Exit leg %s failed for pair %s, retrying as MARKET",
                    leg_name, pos.pair_id,
                )
                order["type"] = "MARKET"
                order.pop("price", None)
                await self._execute_order_fn(order)

        # Record trade
        self._record_trade(pos, reason, z)
        del self._positions[pos.pair_id]

        self._add_alert(
            "info",
            f"Exited pair {pos.asset_a}/{pos.asset_b} reason={reason} "
            f"Z={z:.3f} PnL={pos.current_pnl:.4f}",
        )

    async def _execute_partial_exit(
        self,
        pos: PairPosition,
        exit_fraction: float,
    ) -> None:
        """Partially exit a pair position, maintaining hedge ratio."""
        if self._execute_order_fn is None:
            return

        qty_a = pos.leg_a_qty * exit_fraction
        qty_b = pos.leg_b_qty * exit_fraction

        close_a = {
            "symbol": pos.asset_a,
            "side": "SELL" if pos.leg_a_side == "BUY" else "BUY",
            "type": "LIMIT",
            "quantity": qty_a,
            "price": self._get_close_price(pos.asset_a, pos.leg_a_side),
            "pair_id": pos.pair_id,
            "leg": "A_partial",
            "reduce_only": True,
        }
        close_b = {
            "symbol": pos.asset_b,
            "side": "SELL" if pos.leg_b_side == "BUY" else "BUY",
            "type": "LIMIT",
            "quantity": qty_b,
            "price": self._get_close_price(pos.asset_b, pos.leg_b_side),
            "pair_id": pos.pair_id,
            "leg": "B_partial",
            "reduce_only": True,
        }

        await asyncio.gather(
            self._execute_order_fn(close_a),
            self._execute_order_fn(close_b),
            return_exceptions=True,
        )

        # Adjust remaining quantities
        pos.leg_a_qty -= qty_a
        pos.leg_b_qty -= qty_b

        trade_logger.info(
            "PARTIAL_EXIT\tpair=%s\t%s/%s\tfraction=%.0f%%\tremaining_a=%.6f\tremaining_b=%.6f",
            pos.pair_id, pos.asset_a, pos.asset_b,
            exit_fraction * 100, pos.leg_a_qty, pos.leg_b_qty,
        )

    # ======================================================================
    #  TWAP Exit for Illiquid Legs (Section 4.3)
    # ======================================================================

    async def execute_twap_exit(
        self,
        pos: PairPosition,
        illiquid_leg: str,
        reason: str,
    ) -> None:
        """Execute TWAP exit when one leg has insufficient liquidity.

        If a leg has spread > 0.1% or depth < 1.5x position size, use TWAP
        over 4 hours: split into 8 sub-orders, 30 minutes apart.

        Parameters
        ----------
        pos : PairPosition
            The position to close.
        illiquid_leg : str
            Which leg is illiquid: "A" or "B".
        reason : str
            Exit reason for logging.
        """
        if self._execute_order_fn is None:
            return

        pos.pending_exit = True
        pos.exit_reason = f"twap_{reason}"

        # Close the LIQUID leg immediately at market
        if illiquid_leg == "A":
            liquid_order = {
                "symbol": pos.asset_b,
                "side": "SELL" if pos.leg_b_side == "BUY" else "BUY",
                "type": "MARKET",
                "quantity": pos.leg_b_qty,
                "pair_id": pos.pair_id,
                "leg": "B_close_market",
                "reduce_only": True,
            }
            illiquid_symbol = pos.asset_a
            illiquid_side = "SELL" if pos.leg_a_side == "BUY" else "BUY"
            illiquid_total_qty = pos.leg_a_qty
        else:
            liquid_order = {
                "symbol": pos.asset_a,
                "side": "SELL" if pos.leg_a_side == "BUY" else "BUY",
                "type": "MARKET",
                "quantity": pos.leg_a_qty,
                "pair_id": pos.pair_id,
                "leg": "A_close_market",
                "reduce_only": True,
            }
            illiquid_symbol = pos.asset_b
            illiquid_side = "SELL" if pos.leg_b_side == "BUY" else "BUY"
            illiquid_total_qty = pos.leg_b_qty

        # Close liquid leg immediately
        await self._execute_order_fn(liquid_order)

        # TWAP the illiquid leg: 8 sub-orders, 30 min apart (4 hours total)
        num_slices = 8
        slice_interval_s = 30 * 60  # 30 minutes
        slice_qty = illiquid_total_qty / num_slices

        logger.info(
            "TWAP EXIT: %s leg %s — %d slices of %.8f every %d min",
            illiquid_symbol, illiquid_leg, num_slices,
            slice_qty, slice_interval_s // 60,
        )

        trade_logger.info(
            "TWAP_EXIT_START\tpair=%s\tilliquid=%s\ttotal_qty=%.8f\t"
            "slices=%d\tinterval_min=%d\treason=%s",
            pos.pair_id, illiquid_symbol, illiquid_total_qty,
            num_slices, slice_interval_s // 60, reason,
        )

        remaining_qty = illiquid_total_qty
        for i in range(num_slices):
            if remaining_qty <= 0:
                break

            # Last slice gets whatever is left
            qty = min(slice_qty, remaining_qty)

            twap_order = {
                "symbol": illiquid_symbol,
                "side": illiquid_side,
                "type": "LIMIT",
                "quantity": qty,
                "price": self._get_close_price(illiquid_symbol, illiquid_side),
                "pair_id": pos.pair_id,
                "leg": f"twap_slice_{i+1}",
                "reduce_only": True,
            }

            try:
                result = await self._execute_order_fn(twap_order)
                if result is None or isinstance(result, Exception):
                    # Retry as MARKET
                    twap_order["type"] = "MARKET"
                    twap_order.pop("price", None)
                    await self._execute_order_fn(twap_order)

                remaining_qty -= qty
                logger.info(
                    "TWAP slice %d/%d executed: %.8f of %s (remaining: %.8f)",
                    i + 1, num_slices, qty, illiquid_symbol, remaining_qty,
                )
            except Exception as exc:
                logger.error("TWAP slice %d failed: %s", i + 1, exc)

            # Wait 30 minutes before next slice (unless last)
            if i < num_slices - 1 and remaining_qty > 0:
                await asyncio.sleep(slice_interval_s)

        # Record trade completion
        z = self._get_current_z(pos.asset_a, pos.asset_b)
        self._record_trade(pos, f"twap_{reason}", z)
        self._positions.pop(pos.pair_id, None)

        trade_logger.info(
            "TWAP_EXIT_COMPLETE\tpair=%s\t%s/%s\treason=%s",
            pos.pair_id, pos.asset_a, pos.asset_b, reason,
        )

    def check_leg_liquidity(self, pos: PairPosition) -> Optional[str]:
        """Check if either leg of a position has insufficient liquidity.

        Returns "A" or "B" if that leg is illiquid, None if both are OK.
        Illiquid = spread > 0.1% or depth < 1.5x position size.
        """
        for leg, symbol, qty in [
            ("A", pos.asset_a, pos.leg_a_qty),
            ("B", pos.asset_b, pos.leg_b_qty),
        ]:
            bt = self._book_tickers.get(symbol, {})
            bid = bt.get("bid", 0)
            ask = bt.get("ask", 0)

            if bid <= 0 or ask <= 0:
                continue

            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid * 100

            if spread_pct > 0.1:
                logger.info(
                    "Illiquid leg detected: %s %s spread=%.4f%% > 0.1%%",
                    leg, symbol, spread_pct,
                )
                return leg

            # TODO: check depth < 1.5x position when depth data available

        return None

    def _get_current_z(self, asset_a: str, asset_b: str) -> float:
        """Get current Z-score for a pair."""
        key = f"{asset_a}|{asset_b}"
        state = self._spread_states.get(key)
        if state:
            return state.current_z_score
        return 0.0

    # ======================================================================
    #  Cointegration Breakdown Exit (Section 4.5)
    # ======================================================================

    async def handle_cointegration_breakdown(
        self,
        asset_a: str,
        asset_b: str,
    ) -> None:
        """Handle a pair losing cointegration qualification.

        Close within 4 hours using LIMIT orders.
        """
        for pos in list(self._positions.values()):
            if pos.asset_a == asset_a and pos.asset_b == asset_b:
                logger.warning(
                    "Cointegration breakdown for %s/%s, closing pair %s within %dh",
                    asset_a, asset_b, pos.pair_id, self._breakdown_exit_h,
                )
                await self._execute_exit(pos, "cointegration_breakdown", pos.current_z_score)
                # Mark pair as broken
                self._coint.mark_pair_broken(asset_a, asset_b)

    # ======================================================================
    #  Rebalancing (Section 6.4)
    # ======================================================================

    async def _check_rebalance(
        self,
        pos: PairPosition,
        params: PairParameters,
    ) -> None:
        """Check if hedge ratio drift requires rebalancing."""
        if params.hedge_ratio <= 0 or pos.hedge_ratio <= 0:
            return

        drift = abs(params.hedge_ratio - pos.hedge_ratio) / pos.hedge_ratio
        if drift <= self._beta_drift:
            return

        # Rebalance needed
        logger.info(
            "Beta drift %.1f%% for pair %s (entry=%.4f, current=%.4f). Rebalancing.",
            drift * 100, pos.pair_id, pos.hedge_ratio, params.hedge_ratio,
        )

        price_b = self._mark_prices.get(pos.asset_b, 0)
        if price_b <= 0:
            return

        # Calculate new target quantity for leg B
        price_a = self._mark_prices.get(pos.asset_a, 0)
        if price_a <= 0:
            return

        target_qty_b = (pos.leg_a_qty * price_a * params.hedge_ratio) / price_b
        delta_b = target_qty_b - pos.leg_b_qty

        # Limit rebalance to 20% of position
        max_delta = pos.leg_b_qty * self._rebalance_max_pct / 100.0
        if abs(delta_b) > max_delta:
            delta_b = max_delta if delta_b > 0 else -max_delta

        if abs(delta_b) < 0.0001:
            return

        side = pos.leg_b_side if delta_b > 0 else ("SELL" if pos.leg_b_side == "BUY" else "BUY")

        if self._execute_order_fn:
            await self._execute_order_fn({
                "symbol": pos.asset_b,
                "side": side,
                "type": "MARKET",
                "quantity": abs(delta_b),
                "pair_id": pos.pair_id,
                "leg": "B_rebalance",
            })
            pos.leg_b_qty += delta_b
            pos.hedge_ratio = params.hedge_ratio
            pos.rebalance_count += 1

            trade_logger.info(
                "REBALANCE\tpair=%s\tnew_beta=%.4f\tdelta_b=%.6f\trebal_count=%d",
                pos.pair_id, params.hedge_ratio, delta_b, pos.rebalance_count,
            )

    # ======================================================================
    #  Circuit Breakers (Section 5.6)
    # ======================================================================

    async def _handle_spread_explosion(self, key: str, z: float) -> None:
        """Z-score >= 5.0: close ALL positions, halt 24h."""
        logger.critical(
            "SPREAD EXPLOSION: %s Z=%.3f, closing all pairs and halting for %dh",
            key, z, self._spread_explosion_halt_h,
        )
        self._halted_until = time.time() + self._spread_explosion_halt_h * 3600
        await self.close_all_positions("spread_explosion")
        self._add_alert(
            "critical",
            f"Spread explosion Z={z:.3f} on {key}. All positions closed. "
            f"Strategy halted for {self._spread_explosion_halt_h}h.",
        )

    def _record_stop(self, pos: PairPosition, params: PairParameters) -> None:
        """Record a stop-loss event and check multi-stop circuit breaker."""
        self._stop_timestamps.append(time.time())

        # Update pair stop count
        params.consecutive_stops += 1
        params.post_stop_entry_z = self._z_post_stop  # Raise entry threshold

        # Clean old stop timestamps
        cutoff = time.time() - self._multi_stop_window_h * 3600
        self._stop_timestamps = [t for t in self._stop_timestamps if t > cutoff]

        # Check multi-stop breaker: 3+ pairs stopped in 24h
        if len(self._stop_timestamps) >= self._multi_stop_count:
            logger.critical(
                "MULTI-STOP BREAKER: %d stops in %dh. Halting for %dh.",
                len(self._stop_timestamps),
                self._multi_stop_window_h,
                self._multi_stop_halt_h,
            )
            self._halted_until = time.time() + self._multi_stop_halt_h * 3600
            self._add_alert(
                "critical",
                f"Multi-stop circuit breaker: {len(self._stop_timestamps)} stops "
                f"in {self._multi_stop_window_h}h. Strategy halted for {self._multi_stop_halt_h}h.",
            )

    async def check_btc_flash(self) -> None:
        """Check if BTC moved > 8% in 4h (Section 7.4)."""
        prices = list(self._prices_1h.get("BTCUSDT", []))
        window = int(self._btc_flash_window_h)
        if len(prices) < window + 1:
            return

        recent = prices[-(window + 1):]
        high = max(recent)
        low = min(recent)
        if low <= 0:
            return

        move_pct = ((high - low) / low) * 100.0
        if move_pct > self._btc_flash_pct:
            self._btc_halt_until = time.time() + self._btc_flash_halt_h * 3600
            logger.warning(
                "BTC flash move %.1f%% in %dh. Halting pair trading for %dh.",
                move_pct, self._btc_flash_window_h, self._btc_flash_halt_h,
            )
            self._add_alert(
                "warning",
                f"BTC moved {move_pct:.1f}% in {self._btc_flash_window_h}h. "
                f"New entries halted for {self._btc_flash_halt_h}h.",
            )

    # ======================================================================
    #  Spread Mean/Variance Regime Shift (Section 11.3)
    # ======================================================================

    def check_regime_shift(self, key: str, params: PairParameters) -> bool:
        """Check if spread mean shifted by > 2 sigma over 7 days."""
        state = self._spread_states.get(key)
        if state is None or len(state.spread_history) < 168:  # 7 days * 24h
            return False

        recent_168 = list(state.spread_history)[-168:]
        recent_mu = float(np.mean(recent_168))
        old_mu = params.spread_mean

        if params.spread_std > 0:
            shift = abs(recent_mu - old_mu) / params.spread_std
            if shift > 2.0:
                logger.warning(
                    "Regime shift detected for %s: mu shifted by %.1f sigma",
                    key, shift,
                )
                return True
        return False

    # ======================================================================
    #  Position monitoring and PnL
    # ======================================================================

    async def _monitor_positions(self) -> None:
        """Periodic position monitoring (called on 1m close)."""
        for pos in list(self._positions.values()):
            # Check for illiquid leg (Section 11.5)
            for symbol in [pos.asset_a, pos.asset_b]:
                bt = self._book_tickers.get(symbol, {})
                bid = bt.get("bid", 0)
                ask = bt.get("ask", 0)
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                    spread_pct = ((ask - bid) / mid) * 100.0
                    if spread_pct > 0.2:
                        logger.warning(
                            "Illiquid leg detected: %s spread=%.3f%% for pair %s. "
                            "Using TWAP exit.",
                            symbol, spread_pct, pos.pair_id,
                        )
                        # Begin TWAP exit over 4 hours
                        await self._execute_exit(pos, "illiquid_twap", pos.current_z_score)
                        break

    def _update_position_pnl(self) -> None:
        """Update unrealized PnL for all open positions."""
        for pos in self._positions.values():
            price_a = self._mark_prices.get(pos.asset_a, pos.leg_a_entry_price)
            price_b = self._mark_prices.get(pos.asset_b, pos.leg_b_entry_price)

            # PnL for leg A
            if pos.leg_a_side == "BUY":
                pnl_a = (price_a - pos.leg_a_entry_price) * pos.leg_a_qty
            else:
                pnl_a = (pos.leg_a_entry_price - price_a) * pos.leg_a_qty

            # PnL for leg B
            if pos.leg_b_side == "BUY":
                pnl_b = (price_b - pos.leg_b_entry_price) * pos.leg_b_qty
            else:
                pnl_b = (pos.leg_b_entry_price - price_b) * pos.leg_b_qty

            pos.current_pnl = pnl_a + pnl_b
            entry_notional = (pos.leg_a_entry_price * pos.leg_a_qty +
                              pos.leg_b_entry_price * pos.leg_b_qty)
            pos.current_pnl_pct = (pos.current_pnl / entry_notional * 100.0) if entry_notional > 0 else 0.0

    def _update_position_z_scores(self, asset_a: str, asset_b: str, z: float) -> None:
        """Update Z-score for positions in this pair."""
        for pos in self._positions.values():
            if pos.asset_a == asset_a and pos.asset_b == asset_b:
                pos.current_z_score = z

    # ======================================================================
    #  Close all positions (kill switch / circuit breaker)
    # ======================================================================

    async def close_all_positions(self, reason: str) -> int:
        """Close all open pair positions. Returns count closed."""
        closed = 0
        for pos in list(self._positions.values()):
            await self._execute_exit(pos, reason, pos.current_z_score)
            closed += 1
        return closed

    # ======================================================================
    #  Halt checking
    # ======================================================================

    def _is_halted(self) -> bool:
        """Check if strategy is halted."""
        now = time.time()
        if now < self._halted_until:
            return True
        if now < self._btc_halt_until:
            return True
        if self._consecutive_losses >= self._consec_loss_halt:
            return True
        return False

    # ======================================================================
    #  Helpers
    # ======================================================================

    def _get_close_price(self, symbol: str, entry_side: str) -> float:
        """Get best exit price (opposite side of book)."""
        bt = self._book_tickers.get(symbol, {})
        if entry_side == "BUY":
            return bt.get("bid", self._mark_prices.get(symbol, 0))
        else:
            return bt.get("ask", self._mark_prices.get(symbol, 0))

    @staticmethod
    def _extract_fill_price(result: Any, fallback: float) -> float:
        """Extract fill price from order result."""
        if isinstance(result, dict):
            return float(result.get("avgPrice", result.get("price", fallback)))
        return fallback

    def _record_trade(self, pos: PairPosition, reason: str, exit_z: float) -> None:
        """Record completed trade for performance tracking."""
        is_win = pos.current_pnl > 0
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        entry_notional = (pos.leg_a_entry_price * pos.leg_a_qty +
                          pos.leg_b_entry_price * pos.leg_b_qty)
        fee_estimate = entry_notional * self._taker_fee * 4  # 4 legs

        self._trade_history.append({
            "trade_id": pos.pair_id,
            "symbol": f"{pos.asset_a}/{pos.asset_b}",
            "side": pos.direction,
            "entry_price": pos.leg_a_entry_price,
            "exit_price": self._mark_prices.get(pos.asset_a, 0),
            "quantity": pos.leg_a_qty,
            "pnl": pos.current_pnl - fee_estimate,
            "pnl_pct": pos.current_pnl_pct,
            "fees": fee_estimate,
            "entry_time_ms": int(pos.entry_time * 1000),
            "exit_time_ms": int(time.time() * 1000),
            "entry_z": pos.entry_z_score,
            "exit_z": exit_z,
            "exit_reason": reason,
            "hedge_ratio": pos.hedge_ratio,
            "rebalance_count": pos.rebalance_count,
            "half_life_hours": pos.half_life_hours,
        })

    def _add_alert(self, level: str, message: str) -> None:
        """Add an alert to the alert list."""
        self._alerts.append({
            "level": level,
            "message": message,
            "timestamp": time.time(),
            "strategy_id": "STRAT-003",
        })
        # Keep last 200 alerts
        if len(self._alerts) > 200:
            self._alerts = self._alerts[-200:]

    # ======================================================================
    #  Data access for dashboard / metrics
    # ======================================================================

    def get_positions(self) -> List[Dict[str, Any]]:
        """Return all open pair positions as dicts."""
        return [
            {
                "pair_id": p.pair_id,
                "asset_a": p.asset_a,
                "asset_b": p.asset_b,
                "direction": p.direction,
                "entry_z": p.entry_z_score,
                "current_z": p.current_z_score,
                "pnl": round(p.current_pnl, 4),
                "pnl_pct": round(p.current_pnl_pct, 4),
                "entry_time": p.entry_time,
                "duration_h": round((time.time() - p.entry_time) / 3600.0, 1),
                "hedge_ratio": p.hedge_ratio,
                "half_life_h": p.half_life_hours,
                "rebalance_count": p.rebalance_count,
                "leg_a": {"symbol": p.asset_a, "side": p.leg_a_side, "qty": p.leg_a_qty, "entry": p.leg_a_entry_price},
                "leg_b": {"symbol": p.asset_b, "side": p.leg_b_side, "qty": p.leg_b_qty, "entry": p.leg_b_entry_price},
            }
            for p in self._positions.values()
        ]

    def get_spread_states(self) -> List[Dict[str, Any]]:
        """Return current spread / Z-score state for all active pairs."""
        result = []
        for key, state in self._spread_states.items():
            result.append({
                "pair": key,
                "asset_a": state.asset_a,
                "asset_b": state.asset_b,
                "current_spread": round(state.current_spread, 6),
                "current_z": round(state.current_z_score, 4),
                "rolling_mean": round(state.rolling_mean, 6),
                "rolling_std": round(state.rolling_std, 6),
                "history_length": len(state.spread_history),
            })
        return result

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._trade_history[-limit:]

    def get_alerts(self) -> List[Dict[str, Any]]:
        return self._alerts[-50:]

    def get_strategy_specific_metrics(self) -> Dict[str, Any]:
        """Section 10.2 strategy-specific metrics."""
        trades = self._trade_history

        # Z-Score distribution at entry/exit
        entry_zs = [t.get("entry_z", 0) for t in trades]
        exit_zs = [t.get("exit_z", 0) for t in trades]

        # Mean reversion capture rate
        captures = []
        for t in trades:
            entry_z = abs(t.get("entry_z", 0))
            exit_z = abs(t.get("exit_z", 0))
            if entry_z > 0:
                capture = (entry_z - exit_z) / entry_z * 100.0
                captures.append(capture)

        # Half-life accuracy
        hl_accuracies = []
        for t in trades:
            predicted_hl_h = t.get("half_life_hours", 0)
            actual_duration_h = (t.get("exit_time_ms", 0) - t.get("entry_time_ms", 0)) / 3_600_000
            if predicted_hl_h > 0:
                hl_accuracies.append(actual_duration_h / predicted_hl_h)

        # Pair-level attribution
        pair_pnl: Dict[str, float] = defaultdict(float)
        pair_count: Dict[str, int] = defaultdict(int)
        stop_freq: Dict[str, int] = defaultdict(int)
        for t in trades:
            sym = t.get("symbol", "")
            pair_pnl[sym] += t.get("pnl", 0)
            pair_count[sym] += 1
            if t.get("exit_reason", "") in ("hard_stop_z", "dollar_stop"):
                stop_freq[sym] += 1

        # Beta drift
        beta_drifts = [t.get("rebalance_count", 0) for t in trades]

        # Cointegration stability
        total_pairs = len(self._coint.qualified_pairs)

        return {
            "z_score_entry_distribution": {
                "mean": round(float(np.mean(entry_zs)), 4) if entry_zs else 0,
                "std": round(float(np.std(entry_zs)), 4) if entry_zs else 0,
                "values": entry_zs[-50:],
            },
            "z_score_exit_distribution": {
                "mean": round(float(np.mean(exit_zs)), 4) if exit_zs else 0,
                "values": exit_zs[-50:],
            },
            "mean_reversion_capture_rate": round(float(np.mean(captures)), 2) if captures else 0,
            "half_life_accuracy": {
                "mean_ratio": round(float(np.mean(hl_accuracies)), 4) if hl_accuracies else 0,
                "values": hl_accuracies[-50:],
            },
            "cointegration_stability_score": total_pairs,
            "beta_drift_rebalances": sum(beta_drifts),
            "pair_level_attribution": {
                sym: {"pnl": round(pnl, 4), "trades": pair_count[sym]}
                for sym, pnl in sorted(pair_pnl.items(), key=lambda x: x[1], reverse=True)
            },
            "stop_loss_frequency_per_pair": dict(stop_freq),
            "net_directional_exposure": self._calculate_net_exposure(),
            "consecutive_losses": self._consecutive_losses,
            "halted_until": self._halted_until,
            "active_pairs": len(self._positions),
            "qualified_pairs": total_pairs,
        }

    def _calculate_net_exposure(self) -> Dict[str, float]:
        """Calculate net directional exposure across all pairs."""
        exposure: Dict[str, float] = defaultdict(float)
        for pos in self._positions.values():
            price_a = self._mark_prices.get(pos.asset_a, pos.leg_a_entry_price)
            price_b = self._mark_prices.get(pos.asset_b, pos.leg_b_entry_price)
            if pos.leg_a_side == "BUY":
                exposure[pos.asset_a] += pos.leg_a_qty * price_a
            else:
                exposure[pos.asset_a] -= pos.leg_a_qty * price_a
            if pos.leg_b_side == "BUY":
                exposure[pos.asset_b] += pos.leg_b_qty * price_b
            else:
                exposure[pos.asset_b] -= pos.leg_b_qty * price_b
        return {k: round(v, 2) for k, v in exposure.items()}

    # ======================================================================
    #  State persistence
    # ======================================================================

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize strategy state."""
        return {
            "positions": [
                {
                    "pair_id": p.pair_id,
                    "asset_a": p.asset_a,
                    "asset_b": p.asset_b,
                    "direction": p.direction,
                    "entry_z_score": p.entry_z_score,
                    "entry_time": p.entry_time,
                    "hedge_ratio": p.hedge_ratio,
                    "half_life_hours": p.half_life_hours,
                    "leg_a_side": p.leg_a_side,
                    "leg_a_qty": p.leg_a_qty,
                    "leg_a_entry_price": p.leg_a_entry_price,
                    "leg_b_side": p.leg_b_side,
                    "leg_b_qty": p.leg_b_qty,
                    "leg_b_entry_price": p.leg_b_entry_price,
                    "current_z_score": p.current_z_score,
                    "peak_z_score": p.peak_z_score,
                    "rebalance_count": p.rebalance_count,
                    "partial_exit_done": p.partial_exit_done,
                }
                for p in self._positions.values()
            ],
            "spread_history": {
                key: {
                    "spreads": list(s.spread_history)[-100:],
                    "z_scores": list(s.z_score_history)[-100:],
                }
                for key, s in self._spread_states.items()
            },
            "trade_history": self._trade_history[-200:],
            "consecutive_losses": self._consecutive_losses,
            "halted_until": self._halted_until,
            "btc_halt_until": self._btc_halt_until,
            "stop_timestamps": self._stop_timestamps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore strategy state from persistence."""
        self._consecutive_losses = state.get("consecutive_losses", 0)
        self._halted_until = state.get("halted_until", 0.0)
        self._btc_halt_until = state.get("btc_halt_until", 0.0)
        self._stop_timestamps = state.get("stop_timestamps", [])
        self._trade_history = state.get("trade_history", [])

        for pd in state.get("positions", []):
            try:
                pos = PairPosition(
                    pair_id=pd["pair_id"],
                    asset_a=pd["asset_a"],
                    asset_b=pd["asset_b"],
                    direction=pd["direction"],
                    entry_z_score=pd["entry_z_score"],
                    entry_time=pd["entry_time"],
                    hedge_ratio=pd["hedge_ratio"],
                    half_life_hours=pd["half_life_hours"],
                    leg_a_side=pd["leg_a_side"],
                    leg_a_qty=pd["leg_a_qty"],
                    leg_a_entry_price=pd["leg_a_entry_price"],
                    leg_b_side=pd["leg_b_side"],
                    leg_b_qty=pd["leg_b_qty"],
                    leg_b_entry_price=pd["leg_b_entry_price"],
                    current_z_score=pd.get("current_z_score", 0),
                    peak_z_score=pd.get("peak_z_score", 0),
                    rebalance_count=pd.get("rebalance_count", 0),
                    partial_exit_done=pd.get("partial_exit_done", False),
                    is_fully_entered=True,
                )
                self._positions[pos.pair_id] = pos
            except (KeyError, TypeError) as exc:
                logger.warning("Skipping malformed position state: %s", exc)

        for key, data in state.get("spread_history", {}).items():
            parts = key.split("|")
            if len(parts) == 2:
                ss = SpreadState(asset_a=parts[0], asset_b=parts[1])
                for s in data.get("spreads", []):
                    ss.spread_history.append(s)
                for z in data.get("z_scores", []):
                    ss.z_score_history.append(z)
                if ss.spread_history:
                    ss.current_spread = ss.spread_history[-1]
                if ss.z_score_history:
                    ss.current_z_score = ss.z_score_history[-1]
                self._spread_states[key] = ss
