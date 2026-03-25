"""Core mean reversion strategy logic for STRAT-004.

Implements:
- 3-signal entry (BB 4h +/-2.5 sigma, Z-score 4h +/-2.0, RSI 4h <25/>75)
  with 2-of-3 agreement required
- Signal strength sizing (3/3 = 100%, 2/3 = 75%, Hurst < 0.40 bonus +25%)
- 15m reversal confirmation (candle direction + volume + order book imbalance)
- 3-tranche take-profit exit with breakeven stop
- Hard stop, time exit, regime-change exit, anti-trend emergency exit
- Order placement: LIMIT -> 3 attempts -> MARKET, stop within 500ms
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.indicators import (
    bollinger_bands,
    z_score as compute_zscore,
    rsi as compute_rsi,
    sma,
    atr as compute_atr,
    order_book_imbalance,
    volume_average,
    IndicatorBuffer,
)

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


# ---------------------------------------------------------------------------
# Signal / position data types
# ---------------------------------------------------------------------------

class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class PositionPhase(str, Enum):
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"  # waiting for 15m reversal
    ENTERING = "ENTERING"                          # placing entry order
    ACTIVE = "ACTIVE"                              # position filled
    EXITING = "EXITING"                            # closing position
    CLOSED = "CLOSED"


@dataclass
class MeanReversionSignal:
    """Represents a detected mean reversion opportunity."""

    symbol: str
    direction: SignalDirection
    timestamp: float = field(default_factory=time.time)

    # Individual signal flags
    bb_triggered: bool = False
    zscore_triggered: bool = False
    rsi_triggered: bool = False

    # Values at signal time
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_middle: float = 0.0   # 20 SMA
    bb_std: float = 0.0      # current BB std dev
    zscore_value: float = 0.0
    rsi_value: float = 0.0
    close_price: float = 0.0

    # Confirmation tracking
    confirmation_candles_waited: int = 0
    confirmed: bool = False
    confirmed_at: float = 0.0
    confirmation_volume: float = 0.0
    confirmation_imbalance: float = 0.0

    # Expiry
    expiry_time: float = 0.0  # signal expires at this epoch

    @property
    def signal_count(self) -> int:
        return sum([self.bb_triggered, self.zscore_triggered, self.rsi_triggered])

    @property
    def signal_strength_pct(self) -> float:
        """Signal strength: 3/3 = 100%, 2/3 = 75%."""
        if self.signal_count >= 3:
            return 100.0
        elif self.signal_count >= 2:
            return 75.0
        return 0.0

    @property
    def is_valid(self) -> bool:
        return self.signal_count >= 2 and time.time() < self.expiry_time

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expiry_time

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "signal_count": self.signal_count,
            "strength_pct": self.signal_strength_pct,
            "bb": self.bb_triggered,
            "zscore": self.zscore_triggered,
            "rsi": self.rsi_triggered,
            "zscore_value": round(self.zscore_value, 4),
            "rsi_value": round(self.rsi_value, 2),
            "close_price": self.close_price,
            "bb_middle": round(self.bb_middle, 2),
            "confirmed": self.confirmed,
            "expired": self.is_expired,
        }


@dataclass
class MeanReversionPosition:
    """Tracks a live mean reversion position through its lifecycle."""

    symbol: str
    direction: SignalDirection
    phase: PositionPhase = PositionPhase.PENDING_CONFIRMATION
    signal: Optional[MeanReversionSignal] = None

    # Entry
    entry_price: float = 0.0
    entry_time: float = 0.0
    entry_quantity: float = 0.0
    remaining_quantity: float = 0.0
    leverage: int = 2
    entry_order_id: Optional[int] = None
    limit_attempts: int = 0

    # Stop
    stop_price: float = 0.0
    stop_order_id: Optional[int] = None
    breakeven_activated: bool = False
    original_stop_price: float = 0.0

    # Targets (mean reversion levels)
    target_sma: float = 0.0          # 20 SMA (mean)
    target_tranche1: float = 0.0     # 50% reversion
    target_tranche2: float = 0.0     # at SMA
    target_tranche3: float = 0.0     # opposite BB / Z +/-1.0

    # Take-profit order IDs
    tp1_order_id: Optional[int] = None
    tp2_order_id: Optional[int] = None
    tp3_order_id: Optional[int] = None

    # Tranche execution state
    tranche1_filled: bool = False
    tranche2_filled: bool = False
    tranche3_filled: bool = False

    # PnL
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # Timing
    created_at: float = field(default_factory=time.time)
    closed_at: float = 0.0
    close_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "phase": self.phase.value,
            "entry_price": self.entry_price,
            "remaining_qty": self.remaining_quantity,
            "stop_price": self.stop_price,
            "breakeven": self.breakeven_activated,
            "target_sma": round(self.target_sma, 2),
            "t1_filled": self.tranche1_filled,
            "t2_filled": self.tranche2_filled,
            "t3_filled": self.tranche3_filled,
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "realized_pnl": round(self.realized_pnl, 4),
            "holding_hours": round((time.time() - self.entry_time) / 3600, 1) if self.entry_time else 0,
            "close_reason": self.close_reason,
        }


# ---------------------------------------------------------------------------
# MeanReversionStrategy
# ---------------------------------------------------------------------------

class MeanReversionStrategy:
    """Core mean reversion strategy engine.

    Parameters
    ----------
    config : dict
        Strategy parameters from config.yaml.
    risk_manager :
        STRAT-004 risk manager instance.
    regime_classifier :
        Regime classifier instance.
    binance_client :
        Async Binance REST client.
    paper_engine :
        Paper trading engine (None for live).
    """

    STRATEGY_ID = "STRAT-004"

    def __init__(
        self,
        config: dict,
        risk_manager: Any = None,
        regime_classifier: Any = None,
        binance_client: Any = None,
        paper_engine: Any = None,
    ) -> None:
        self._config = config
        self._risk = risk_manager
        self._regime = regime_classifier
        self._client = binance_client
        self._paper = paper_engine

        # Config values
        self._bb_period: int = config.get("bb_period", 20)
        self._bb_std: float = config.get("bb_std", 2.5)
        self._zscore_period: int = config.get("zscore_period", 50)
        self._zscore_threshold: float = config.get("zscore_entry_threshold", 2.0)
        self._zscore_emergency: float = config.get("zscore_emergency_threshold", 4.0)
        self._rsi_period: int = config.get("rsi_period", 14)
        self._rsi_oversold: float = config.get("rsi_oversold", 25)
        self._rsi_overbought: float = config.get("rsi_overbought", 75)
        self._min_signals: int = config.get("min_signals_required", 2)

        # Confirmation
        self._vol_multiplier: float = config.get("volume_confirmation_multiplier", 1.0)
        self._vol_avg_period: int = config.get("volume_avg_period", 20)
        self._ob_long_threshold: float = config.get("order_book_imbalance_long", 1.2)
        self._ob_short_threshold: float = config.get("order_book_imbalance_short", 0.8)
        self._max_confirm_candles: int = config.get("max_confirmation_candles", 3)
        self._signal_expiry_hours: float = config.get("signal_expiry_hours", 8)
        self._reversion_expiry_pct: float = config.get("reversion_expiry_pct", 50.0)

        # Exit
        self._tranche1_pct: float = config.get("tranche1_pct", 40) / 100.0
        self._tranche1_reversion: float = config.get("tranche1_reversion_pct", 50) / 100.0
        self._tranche2_pct: float = config.get("tranche2_pct", 30) / 100.0
        self._tranche3_pct: float = config.get("tranche3_pct", 30) / 100.0
        self._opposite_z_target: float = config.get("opposite_zscore_target", 1.0)
        self._hard_stop_sigma: float = config.get("hard_stop_sigma", 1.0)
        self._hard_stop_z: float = config.get("hard_stop_zscore", 3.0)
        self._dollar_cap_pct: float = config.get("dollar_cap_pct", 1.5) / 100.0
        self._max_holding_days: float = config.get("max_holding_days", 5)
        self._progress_check_days: float = config.get("progress_check_days", 2.5)
        self._regime_exit_hours: float = config.get("regime_change_exit_hours", 2)

        # Order placement
        self._limit_timeout_s: int = config.get("limit_order_timeout_s", 30)
        self._max_limit_attempts: int = config.get("max_limit_attempts", 3)
        self._stop_max_ms: int = config.get("stop_placement_max_ms", 500)

        # Filters
        self._extreme_vol_atr_mult: float = config.get("extreme_vol_atr_multiplier", 2.0)
        self._extreme_vol_z_bump: float = config.get("extreme_vol_zscore_bump", 0.5)
        self._extreme_vol_size_red: float = config.get("extreme_vol_size_reduction", 0.5)
        self._fee_threshold_mult: float = config.get("fee_threshold_multiplier", 3.0)
        self._rt_fee_pct: float = config.get("round_trip_fee_pct", 0.08) / 100.0
        self._bb_width_min_pct: float = config.get("bb_width_min_pct", 2.0) / 100.0

        # Candle buffers: {symbol: {timeframe: IndicatorBuffer}}
        self._buffers: Dict[str, Dict[str, IndicatorBuffer]] = {}

        # Active signals awaiting confirmation
        self._pending_signals: Dict[str, MeanReversionSignal] = {}

        # Active positions
        self._positions: Dict[str, MeanReversionPosition] = {}

        # Order book snapshots (updated via WS)
        self._order_books: Dict[str, dict] = {}

        # Current prices (updated via WS)
        self._mark_prices: Dict[str, float] = {}
        self._best_bids: Dict[str, float] = {}
        self._best_asks: Dict[str, float] = {}

        # Metrics counters
        self._signals_generated: int = 0
        self._signals_confirmed: int = 0
        self._signals_expired: int = 0
        self._trades_completed: int = 0
        self._total_pnl: float = 0.0

    # ==================================================================
    # Buffer management
    # ==================================================================

    def init_buffers(self, instruments: List[str]) -> None:
        """Initialize candle buffers for all instruments and timeframes."""
        timeframes = ["1m", "15m", "4h", "1d"]
        for symbol in instruments:
            self._buffers[symbol] = {
                tf: IndicatorBuffer(max_size=500) for tf in timeframes
            }

    def get_buffer(self, symbol: str, timeframe: str) -> Optional[IndicatorBuffer]:
        """Return the buffer for a symbol/timeframe pair."""
        return self._buffers.get(symbol, {}).get(timeframe)

    # ==================================================================
    # WebSocket data handlers
    # ==================================================================

    async def on_kline(self, symbol: str, timeframe: str, kline: dict) -> None:
        """Handle an incoming kline (candle) update.

        Called by the WebSocket manager for every candle update.
        Only processes closed candles for signal generation.
        """
        buf = self.get_buffer(symbol, timeframe)
        if buf is None:
            return

        is_closed = kline.get("x", False)
        candle = {
            "timestamp": kline.get("t", 0),
            "open": kline.get("o", 0),
            "high": kline.get("h", 0),
            "low": kline.get("l", 0),
            "close": kline.get("c", 0),
            "volume": kline.get("v", 0),
        }

        if is_closed:
            buf.add_candle(candle)

            # Route to appropriate handler
            if timeframe == "4h":
                await self._on_4h_close(symbol)
            elif timeframe == "15m":
                await self._on_15m_close(symbol)
            elif timeframe == "1m":
                await self._on_1m_update(symbol)
            elif timeframe == "1d":
                # Daily close triggers regime reclassification
                # (handled in main.py scheduler, not here)
                pass
        else:
            # Live candle update — use for position monitoring on 1m
            if timeframe == "1m" and symbol in self._positions:
                current_price = float(kline.get("c", 0))
                if current_price > 0:
                    self._mark_prices[symbol] = current_price
                    await self._monitor_position(symbol, current_price)

    async def on_book_ticker(self, symbol: str, data: dict) -> None:
        """Handle best bid/ask ticker update."""
        self._best_bids[symbol] = float(data.get("b", 0))
        self._best_asks[symbol] = float(data.get("a", 0))

    async def on_depth(self, symbol: str, data: dict) -> None:
        """Handle order book depth update."""
        self._order_books[symbol] = data

    async def on_mark_price(self, symbol: str, data: dict) -> None:
        """Handle mark price update for PnL tracking."""
        price = float(data.get("p", 0))
        if price > 0:
            self._mark_prices[symbol] = price
            if symbol in self._positions:
                pos = self._positions[symbol]
                if pos.phase == PositionPhase.ACTIVE:
                    await self._update_position_pnl(symbol, price)

    async def on_agg_trade(self, symbol: str, data: dict) -> None:
        """Handle aggregate trade for volume spike detection."""
        # Volume tracking is handled via candle data primarily
        pass

    # ==================================================================
    # Signal generation (4h candle close)
    # ==================================================================

    async def _on_4h_close(self, symbol: str) -> None:
        """Evaluate entry signals on 4h candle close."""
        # Skip if instrument is not tradeable per regime
        if self._regime and not self._regime.is_tradeable(symbol):
            return

        # Skip if we already have a position or pending signal
        if symbol in self._positions or symbol in self._pending_signals:
            return

        buf = self.get_buffer(symbol, "4h")
        if buf is None or len(buf) < max(self._bb_period, self._zscore_period, self._rsi_period + 1):
            return

        closes = buf.get_closes()
        current_close = float(closes[-1])

        # Check filters first
        if not self._check_entry_filters(symbol, closes, buf):
            return

        # --- Signal A: Bollinger Band extreme ---
        upper, middle, lower = bollinger_bands(
            closes, period=self._bb_period, num_std=self._effective_bb_std(symbol),
        )
        bb_upper = float(upper[-1]) if not np.isnan(upper[-1]) else 0
        bb_lower = float(lower[-1]) if not np.isnan(lower[-1]) else 0
        bb_middle_val = float(middle[-1]) if not np.isnan(middle[-1]) else 0

        # Compute BB std for stop placement
        sma_arr = sma(closes, self._bb_period)
        sma_val = float(sma_arr[-1]) if not np.isnan(sma_arr[-1]) else 0
        bb_std_val = (bb_upper - sma_val) / self._effective_bb_std(symbol) if sma_val > 0 else 0

        bb_long = current_close < bb_lower
        bb_short = current_close > bb_upper

        # --- Signal B: Z-Score extreme ---
        z_arr = compute_zscore(closes, period=self._zscore_period)
        z_val = float(z_arr[-1]) if not np.isnan(z_arr[-1]) else 0
        z_threshold = self._effective_z_threshold(symbol)

        z_long = z_val < -z_threshold
        z_short = z_val > z_threshold

        # --- Signal C: RSI extreme ---
        rsi_arr = compute_rsi(closes, period=self._rsi_period)
        rsi_val = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 0

        rsi_long = rsi_val < self._rsi_oversold
        rsi_short = rsi_val > self._rsi_overbought

        # --- Evaluate: at least 2-of-3 must agree on direction ---
        long_count = sum([bb_long, z_long, rsi_long])
        short_count = sum([bb_short, z_short, rsi_short])

        signal: Optional[MeanReversionSignal] = None

        if long_count >= self._min_signals:
            signal = MeanReversionSignal(
                symbol=symbol,
                direction=SignalDirection.LONG,
                bb_triggered=bb_long,
                zscore_triggered=z_long,
                rsi_triggered=rsi_long,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_middle=bb_middle_val,
                bb_std=bb_std_val,
                zscore_value=z_val,
                rsi_value=rsi_val,
                close_price=current_close,
                expiry_time=time.time() + self._signal_expiry_hours * 3600,
            )

        elif short_count >= self._min_signals:
            signal = MeanReversionSignal(
                symbol=symbol,
                direction=SignalDirection.SHORT,
                bb_triggered=bb_short,
                zscore_triggered=z_short,
                rsi_triggered=rsi_short,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_middle=bb_middle_val,
                bb_std=bb_std_val,
                zscore_value=z_val,
                rsi_value=rsi_val,
                close_price=current_close,
                expiry_time=time.time() + self._signal_expiry_hours * 3600,
            )

        if signal is not None:
            self._signals_generated += 1
            self._pending_signals[symbol] = signal
            logger.info(
                "SIGNAL %s %s: %d/3 (BB=%s Z=%.2f RSI=%.1f) price=%.2f",
                symbol, signal.direction.value, signal.signal_count,
                signal.bb_triggered, z_val, rsi_val, current_close,
            )
            trade_logger.info(
                "SIGNAL\tsymbol=%s\tdir=%s\tcount=%d\tbb=%s\tz=%.4f\trsi=%.2f\tprice=%.4f",
                symbol, signal.direction.value, signal.signal_count,
                signal.bb_triggered, z_val, rsi_val, current_close,
            )

    # ==================================================================
    # Reversal confirmation (15m candle close)
    # ==================================================================

    async def _on_15m_close(self, symbol: str) -> None:
        """Check 15m reversal confirmation for pending signals."""
        signal = self._pending_signals.get(symbol)
        if signal is None or signal.confirmed:
            return

        # Check expiry
        if signal.is_expired:
            self._signals_expired += 1
            logger.info("Signal expired for %s", symbol)
            del self._pending_signals[symbol]
            return

        # Check if price has reverted > 50% before entry (opportunity diminished)
        current_price = self._mark_prices.get(symbol, 0)
        if current_price > 0 and self._check_reversion_expiry(signal, current_price):
            self._signals_expired += 1
            logger.info("Signal expired (50%% reversion before entry) for %s", symbol)
            del self._pending_signals[symbol]
            return

        signal.confirmation_candles_waited += 1

        # Max 3 candles (45 min) for confirmation
        if signal.confirmation_candles_waited > self._max_confirm_candles:
            self._signals_expired += 1
            logger.info("Signal expired (max confirmation candles) for %s", symbol)
            del self._pending_signals[symbol]
            return

        buf = self.get_buffer(symbol, "15m")
        if buf is None or len(buf) < self._vol_avg_period + 1:
            return

        closes_15m = buf.get_closes()
        opens_15m = buf.get_opens()
        volumes_15m = buf.get_volumes()

        last_close = float(closes_15m[-1])
        last_open = float(opens_15m[-1])
        last_volume = float(volumes_15m[-1])

        # Step 1: Candle direction must confirm reversal
        if signal.direction == SignalDirection.LONG:
            candle_confirms = last_close > last_open  # green candle
        else:
            candle_confirms = last_close < last_open  # red candle

        if not candle_confirms:
            return

        # Step 2: Volume > 1.0x 20-period average
        vol_avg = volume_average(volumes_15m, self._vol_avg_period)
        avg_vol = float(vol_avg[-1]) if not np.isnan(vol_avg[-1]) else 0
        vol_confirms = avg_vol > 0 and last_volume >= self._vol_multiplier * avg_vol

        if not vol_confirms:
            return

        # Step 3: Order book imbalance
        ob = self._order_books.get(symbol)
        ob_confirms = False
        imbalance_val = 0.0

        if ob:
            bids = ob.get("bids", ob.get("b", []))
            asks = ob.get("asks", ob.get("a", []))

            if bids and asks:
                bid_depths = np.array([float(b[1]) for b in bids[:10]])
                ask_depths = np.array([float(a[1]) for a in asks[:10]])
                imbalance_val = float(order_book_imbalance(bid_depths, ask_depths))

                if signal.direction == SignalDirection.LONG:
                    ob_confirms = imbalance_val > self._ob_long_threshold
                else:
                    ob_confirms = imbalance_val < self._ob_short_threshold

        if not ob_confirms:
            # Wait for next candle (up to max)
            logger.debug(
                "OB imbalance not confirmed for %s: %.3f (need %s)",
                symbol, imbalance_val,
                f">{self._ob_long_threshold}" if signal.direction == SignalDirection.LONG
                else f"<{self._ob_short_threshold}",
            )
            return

        # All 3 confirmations passed
        signal.confirmed = True
        signal.confirmed_at = time.time()
        signal.confirmation_volume = last_volume
        signal.confirmation_imbalance = imbalance_val
        self._signals_confirmed += 1

        logger.info(
            "CONFIRMED %s %s: vol=%.0f (avg=%.0f), OB_imbalance=%.3f",
            symbol, signal.direction.value, last_volume, avg_vol, imbalance_val,
        )
        trade_logger.info(
            "CONFIRMED\tsymbol=%s\tdir=%s\tvol=%.0f\timbalance=%.4f",
            symbol, signal.direction.value, last_volume, imbalance_val,
        )

        # Proceed to entry
        await self._execute_entry(symbol, signal)

    # ==================================================================
    # Entry execution
    # ==================================================================

    async def _execute_entry(self, symbol: str, signal: MeanReversionSignal) -> None:
        """Execute entry after confirmation: size, risk check, order placement."""
        # Compute position size
        equity = self._get_equity()
        if equity <= 0:
            logger.warning("Cannot enter: equity not available")
            return

        # Stop distance and sizing
        stop_price = self._compute_stop_price(signal)
        entry_price_est = self._mark_prices.get(symbol, signal.close_price)
        if entry_price_est <= 0:
            return

        stop_distance_pct = abs(stop_price - entry_price_est) / entry_price_est
        if stop_distance_pct <= 0:
            logger.warning("Invalid stop distance for %s", symbol)
            return

        # Position size = (equity * risk_pct) / stop_distance_pct
        risk_pct = self._config.get("risk_per_trade_pct", 1.0) / 100.0
        base_size = (equity * risk_pct) / stop_distance_pct

        # Signal strength adjustment
        strength_mult = signal.signal_strength_pct / 100.0

        # Regime multiplier (Hurst < 0.40 bonus)
        regime_mult = 1.0
        if self._regime:
            regime = self._regime.get_regime(symbol)
            regime_mult = regime.size_multiplier

        # Extreme volatility reduction
        vol_mult = self._check_extreme_volatility(symbol)

        # Final size
        size_usdt = base_size * strength_mult * regime_mult * vol_mult

        # Cap at max per trade
        max_per_trade = equity * (self._config.get("max_per_trade_pct", 4.0) / 100.0)
        size_usdt = min(size_usdt, max_per_trade)

        # Determine leverage
        leverage = self._compute_leverage(symbol)

        # Risk manager check
        if self._risk:
            allowed, reason = self._risk.check_entry_allowed(
                strategy_id=self.STRATEGY_ID,
                symbol=symbol,
                direction=signal.direction.value,
                size_usdt=size_usdt,
                leverage=leverage,
            )
            if not allowed:
                logger.info("Entry rejected by risk manager for %s: %s", symbol, reason)
                del self._pending_signals[symbol]
                return

        # Create position object
        position = MeanReversionPosition(
            symbol=symbol,
            direction=signal.direction,
            phase=PositionPhase.ENTERING,
            signal=signal,
            leverage=leverage,
        )

        # Compute targets
        self._compute_targets(position, signal, entry_price_est)

        self._positions[symbol] = position
        del self._pending_signals[symbol]

        # Place order
        await self._place_entry_order(position, size_usdt)

    async def _place_entry_order(
        self,
        position: MeanReversionPosition,
        size_usdt: float,
    ) -> None:
        """Place entry order: LIMIT -> 3 attempts -> MARKET."""
        symbol = position.symbol
        side = "BUY" if position.direction == SignalDirection.LONG else "SELL"

        # Estimate quantity from size_usdt and price
        price = self._mark_prices.get(symbol, 0)
        if price <= 0:
            logger.error("No price available for %s, aborting entry", symbol)
            del self._positions[symbol]
            return

        quantity = size_usdt / price

        for attempt in range(1, self._max_limit_attempts + 1):
            position.limit_attempts = attempt

            # Get best price for limit order
            if position.direction == SignalDirection.LONG:
                limit_price = self._best_bids.get(symbol, price)
            else:
                limit_price = self._best_asks.get(symbol, price)

            if limit_price <= 0:
                limit_price = price

            try:
                if self._paper:
                    # Paper trading fill
                    ob = self._order_books.get(symbol, {"bids": [], "asks": []})
                    result = self._paper.simulate_limit_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=limit_price,
                        current_price=price,
                        order_book_snapshot=ob,
                    )
                    if result is not None:
                        await self._on_entry_fill(position, result.fill_price, result.fill_quantity, result.fees)
                        return
                else:
                    order = await self._client.place_futures_order(
                        symbol=symbol,
                        side=side,
                        type="LIMIT",
                        quantity=quantity,
                        price=limit_price,
                        time_in_force="GTC",
                    )
                    position.entry_order_id = order.get("orderId")

                    # Wait for fill
                    await asyncio.sleep(self._limit_timeout_s)

                    # Check if filled (would be handled by user data stream in production)
                    # For now, fall through to next attempt
                    logger.info("Limit attempt %d/%d for %s", attempt, self._max_limit_attempts, symbol)

            except Exception as e:
                logger.error("Entry order failed for %s (attempt %d): %s", symbol, attempt, e)

        # After max LIMIT attempts, use MARKET
        logger.info("Falling back to MARKET order for %s", symbol)
        try:
            if self._paper:
                ob = self._order_books.get(symbol, {"bids": [], "asks": []})
                result = self._paper.simulate_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_book_snapshot=ob,
                )
                await self._on_entry_fill(position, result.fill_price, result.fill_quantity, result.fees)
            else:
                order = await self._client.place_futures_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=quantity,
                )
                fill_price = float(order.get("avgPrice", price))
                fill_qty = float(order.get("executedQty", quantity))
                await self._on_entry_fill(position, fill_price, fill_qty, 0)
        except Exception as e:
            logger.error("Market entry failed for %s: %s", symbol, e)
            del self._positions[symbol]

    async def _on_entry_fill(
        self,
        position: MeanReversionPosition,
        fill_price: float,
        fill_qty: float,
        fees: float,
    ) -> None:
        """Handle entry fill — update position, place stop, log."""
        position.entry_price = fill_price
        position.entry_quantity = fill_qty
        position.remaining_quantity = fill_qty
        position.entry_time = time.time()
        position.fees_paid += fees
        position.phase = PositionPhase.ACTIVE

        # Recompute targets based on actual fill
        if position.signal:
            self._compute_targets(position, position.signal, fill_price)

        # Place stop loss within 500ms
        await self._place_stop_order(position)

        # Record with risk manager
        if self._risk:
            self._risk.record_position_open(
                symbol=position.symbol,
                direction=position.direction.value,
                size_usdt=fill_price * fill_qty,
                entry_price=fill_price,
            )

        trade_logger.info(
            "ENTRY\tsymbol=%s\tdir=%s\tprice=%.4f\tqty=%.6f\tstop=%.4f\t"
            "target_sma=%.4f\tstrength=%d%%",
            position.symbol, position.direction.value, fill_price, fill_qty,
            position.stop_price, position.target_sma,
            int(position.signal.signal_strength_pct) if position.signal else 0,
        )

    async def _place_stop_order(self, position: MeanReversionPosition) -> None:
        """Place hard stop loss immediately after entry fill (within 500ms)."""
        start = time.time()
        symbol = position.symbol
        side = "SELL" if position.direction == SignalDirection.LONG else "BUY"

        try:
            if self._paper:
                # Paper mode: just track the stop price internally
                position.stop_order_id = -1
                position.original_stop_price = position.stop_price
            else:
                order = await self._client.place_futures_order(
                    symbol=symbol,
                    side=side,
                    type="STOP_MARKET",
                    quantity=position.remaining_quantity,
                    stop_price=position.stop_price,
                    reduce_only=True,
                )
                position.stop_order_id = order.get("orderId")
                position.original_stop_price = position.stop_price

            elapsed_ms = int((time.time() - start) * 1000)
            if elapsed_ms > self._stop_max_ms:
                logger.warning(
                    "Stop placement took %dms (limit %dms) for %s",
                    elapsed_ms, self._stop_max_ms, symbol,
                )
            else:
                logger.info("Stop placed for %s at %.4f (%dms)", symbol, position.stop_price, elapsed_ms)

        except Exception as e:
            logger.error("CRITICAL: Failed to place stop for %s: %s", symbol, e)
            # Emergency: close position if stop cannot be placed
            await self._emergency_close(position, "stop_placement_failed")

    # ==================================================================
    # Position monitoring (1m updates)
    # ==================================================================

    async def _on_1m_update(self, symbol: str) -> None:
        """Monitor positions on 1-minute candle updates."""
        if symbol not in self._positions:
            return

        position = self._positions[symbol]
        if position.phase != PositionPhase.ACTIVE:
            return

        current_price = self._mark_prices.get(symbol, 0)
        if current_price <= 0:
            return

        await self._monitor_position(symbol, current_price)

    async def _monitor_position(self, symbol: str, current_price: float) -> None:
        """Full position monitoring: stops, targets, time exits, regime changes."""
        position = self._positions.get(symbol)
        if position is None or position.phase != PositionPhase.ACTIVE:
            return

        # Update PnL tracking
        await self._update_position_pnl(symbol, current_price)

        # 1. Anti-trend emergency exit: Z >= 4.0
        await self._check_emergency_exit(position, current_price)
        if position.phase != PositionPhase.ACTIVE:
            return

        # 2. Hard stop check (paper mode)
        if self._paper:
            await self._check_paper_stop(position, current_price)
            if position.phase != PositionPhase.ACTIVE:
                return

        # 3. Tranche take-profits
        await self._check_take_profits(position, current_price)
        if position.phase != PositionPhase.ACTIVE:
            return

        # 4. Breakeven stop
        await self._check_breakeven_stop(position, current_price)

        # 5. Time exit
        await self._check_time_exit(position)
        if position.phase != PositionPhase.ACTIVE:
            return

        # 6. Regime change exit
        await self._check_regime_change_exit(position)

    async def _check_emergency_exit(
        self, position: MeanReversionPosition, current_price: float,
    ) -> None:
        """Anti-trend safeguard: Z >= 4.0 => emergency exit + blacklist."""
        buf = self.get_buffer(position.symbol, "4h")
        if buf is None or len(buf) < self._zscore_period:
            return

        closes = buf.get_closes()
        z_arr = compute_zscore(closes, period=self._zscore_period)
        z_val = float(z_arr[-1]) if not np.isnan(z_arr[-1]) else 0

        emergency = False
        if position.direction == SignalDirection.LONG and z_val <= -self._zscore_emergency:
            emergency = True
        elif position.direction == SignalDirection.SHORT and z_val >= self._zscore_emergency:
            emergency = True

        if emergency:
            logger.critical(
                "EMERGENCY EXIT %s: Z=%.2f beyond +/-%.1f threshold",
                position.symbol, z_val, self._zscore_emergency,
            )
            await self._close_position(position, current_price, "anti_trend_emergency_z4")

            # Blacklist instrument
            if self._risk:
                blacklist_days = self._config.get("blacklist_days", 7)
                self._risk.add_blacklist(position.symbol, blacklist_days)

    async def _check_paper_stop(
        self, position: MeanReversionPosition, current_price: float,
    ) -> None:
        """Check stop loss hit in paper mode."""
        triggered = False
        if position.direction == SignalDirection.LONG:
            triggered = current_price <= position.stop_price
        else:
            triggered = current_price >= position.stop_price

        if triggered:
            logger.info("Stop loss hit for %s at %.4f", position.symbol, current_price)
            await self._close_position(position, position.stop_price, "stop_loss")

    async def _check_take_profits(
        self, position: MeanReversionPosition, current_price: float,
    ) -> None:
        """Check and execute 3-tranche take profit scaling."""
        is_long = position.direction == SignalDirection.LONG

        # Tranche 1: 40% at 50% reversion
        if not position.tranche1_filled:
            t1_hit = (
                (is_long and current_price >= position.target_tranche1) or
                (not is_long and current_price <= position.target_tranche1)
            )
            if t1_hit:
                qty = position.entry_quantity * self._tranche1_pct
                await self._close_partial(position, qty, current_price, "tranche1_50pct_reversion")
                position.tranche1_filled = True
                # Move stop to breakeven
                await self._move_stop_to_breakeven(position)

        # Tranche 2: 30% at 20 SMA (full mean reversion)
        if not position.tranche2_filled and position.tranche1_filled:
            t2_hit = (
                (is_long and current_price >= position.target_tranche2) or
                (not is_long and current_price <= position.target_tranche2)
            )
            if t2_hit:
                qty = position.entry_quantity * self._tranche2_pct
                await self._close_partial(position, qty, current_price, "tranche2_sma_reversion")
                position.tranche2_filled = True

        # Tranche 3: 30% at opposite BB or Z +/-1.0
        if not position.tranche3_filled and position.tranche2_filled:
            t3_hit = (
                (is_long and current_price >= position.target_tranche3) or
                (not is_long and current_price <= position.target_tranche3)
            )
            if t3_hit:
                # Close all remaining
                await self._close_position(position, current_price, "tranche3_opposite_bb")

    async def _check_breakeven_stop(
        self, position: MeanReversionPosition, current_price: float,
    ) -> None:
        """Move stop to breakeven + fees at tranche 1 level."""
        if position.breakeven_activated or not position.tranche1_filled:
            return

        # Breakeven = entry price + fees (as fraction of entry)
        fee_buffer = position.entry_price * self._rt_fee_pct * 2
        if position.direction == SignalDirection.LONG:
            new_stop = position.entry_price + fee_buffer
        else:
            new_stop = position.entry_price - fee_buffer

        position.stop_price = new_stop
        position.breakeven_activated = True

        if not self._paper and self._client:
            try:
                # Cancel old stop and place new one
                if position.stop_order_id:
                    await self._client.cancel_futures_order(position.symbol, position.stop_order_id)

                side = "SELL" if position.direction == SignalDirection.LONG else "BUY"
                order = await self._client.place_futures_order(
                    symbol=position.symbol,
                    side=side,
                    type="STOP_MARKET",
                    quantity=position.remaining_quantity,
                    stop_price=new_stop,
                    reduce_only=True,
                )
                position.stop_order_id = order.get("orderId")
            except Exception as e:
                logger.error("Failed to move stop to breakeven for %s: %s", position.symbol, e)

        logger.info("Breakeven stop activated for %s at %.4f", position.symbol, new_stop)

    async def _check_time_exit(self, position: MeanReversionPosition) -> None:
        """Time-based exit: 5 days max, 50% close at 2.5 days if no progress."""
        if position.entry_time <= 0:
            return

        elapsed_days = (time.time() - position.entry_time) / 86400.0

        # 5-day hard exit
        if elapsed_days >= self._max_holding_days:
            current_price = self._mark_prices.get(position.symbol, position.entry_price)
            logger.info("Time exit (5d max) for %s", position.symbol)
            await self._close_position(position, current_price, "time_exit_5d")
            return

        # 2.5-day progress check: if tranche 1 not hit, close 50%
        if elapsed_days >= self._progress_check_days and not position.tranche1_filled:
            current_price = self._mark_prices.get(position.symbol, position.entry_price)
            qty_to_close = position.remaining_quantity * 0.5
            logger.info("Progress check exit (2.5d) for %s: closing 50%%", position.symbol)
            await self._close_partial(position, qty_to_close, current_price, "time_exit_2.5d_no_progress")

    async def _check_regime_change_exit(self, position: MeanReversionPosition) -> None:
        """Exit within 2h if regime changes to trending."""
        if self._regime is None:
            return

        if self._regime.check_regime_change_exit(position.symbol):
            regime = self._regime.get_regime(position.symbol)
            current_price = self._mark_prices.get(position.symbol, position.entry_price)
            logger.warning(
                "Regime change exit for %s: H=%.3f ADX=%.1f",
                position.symbol, regime.hurst, regime.adx_value,
            )
            await self._close_position(position, current_price, "regime_change_exit")

    # ==================================================================
    # Position closing
    # ==================================================================

    async def _close_partial(
        self,
        position: MeanReversionPosition,
        quantity: float,
        price: float,
        reason: str,
    ) -> None:
        """Close a partial quantity of the position."""
        quantity = min(quantity, position.remaining_quantity)
        if quantity <= 0:
            return

        symbol = position.symbol
        side = "SELL" if position.direction == SignalDirection.LONG else "BUY"

        try:
            if self._paper:
                ob = self._order_books.get(symbol, {"bids": [], "asks": []})
                result = self._paper.simulate_market_order(
                    symbol=symbol, side=side, quantity=quantity,
                    order_book_snapshot=ob,
                )
                pnl = self._calc_pnl(position, result.fill_price, quantity)
                position.realized_pnl += pnl
                position.fees_paid += result.fees
            else:
                await self._client.place_futures_order(
                    symbol=symbol, side=side, type="MARKET",
                    quantity=quantity, reduce_only=True,
                )
                pnl = self._calc_pnl(position, price, quantity)
                position.realized_pnl += pnl

            position.remaining_quantity -= quantity

            trade_logger.info(
                "PARTIAL_CLOSE\tsymbol=%s\tqty=%.6f\tprice=%.4f\treason=%s\tpnl=%.4f",
                symbol, quantity, price, reason, pnl,
            )

            # If nothing remaining, finalize
            if position.remaining_quantity <= 0:
                await self._finalize_close(position, reason)

        except Exception as e:
            logger.error("Partial close failed for %s: %s", symbol, e)

    async def _close_position(
        self,
        position: MeanReversionPosition,
        price: float,
        reason: str,
    ) -> None:
        """Close entire remaining position."""
        await self._close_partial(position, position.remaining_quantity, price, reason)

    async def _emergency_close(self, position: MeanReversionPosition, reason: str) -> None:
        """Emergency market close — used when stops fail to place."""
        price = self._mark_prices.get(position.symbol, position.entry_price)
        logger.critical("EMERGENCY CLOSE %s: %s", position.symbol, reason)
        await self._close_position(position, price, reason)

    async def _finalize_close(self, position: MeanReversionPosition, reason: str) -> None:
        """Finalize a fully closed position."""
        position.phase = PositionPhase.CLOSED
        position.closed_at = time.time()
        position.close_reason = reason
        position.remaining_quantity = 0

        symbol = position.symbol

        # Cancel any remaining stop/TP orders
        if not self._paper and self._client:
            try:
                await self._client.cancel_all_futures_orders(symbol)
            except Exception as e:
                logger.warning("Failed to cancel remaining orders for %s: %s", symbol, e)

        # Record with risk manager
        is_win = position.realized_pnl > 0
        if self._risk:
            self._risk.record_trade_closed(
                symbol=symbol,
                pnl=position.realized_pnl,
                is_win=is_win,
            )

        self._trades_completed += 1
        self._total_pnl += position.realized_pnl

        trade_logger.info(
            "CLOSE\tsymbol=%s\tdir=%s\tentry=%.4f\tpnl=%.4f\tfees=%.4f\t"
            "reason=%s\tholding_h=%.1f\twin=%s",
            symbol, position.direction.value, position.entry_price,
            position.realized_pnl, position.fees_paid, reason,
            (position.closed_at - position.entry_time) / 3600 if position.entry_time else 0,
            is_win,
        )

        # Remove from active positions
        self._positions.pop(symbol, None)

    # ==================================================================
    # Helper computations
    # ==================================================================

    def _compute_stop_price(self, signal: MeanReversionSignal) -> float:
        """Compute hard stop: 1.0 sigma beyond the entry extreme.

        For LONG at lower BB: stop = lower_BB - 1.0 * sigma
        For SHORT at upper BB: stop = upper_BB + 1.0 * sigma
        """
        sigma = signal.bb_std if signal.bb_std > 0 else abs(signal.close_price * 0.02)

        if signal.direction == SignalDirection.LONG:
            extreme = min(signal.bb_lower, signal.close_price)
            stop = extreme - self._hard_stop_sigma * sigma
        else:
            extreme = max(signal.bb_upper, signal.close_price)
            stop = extreme + self._hard_stop_sigma * sigma

        # Dollar cap: 1.5% max loss
        equity = self._get_equity()
        if equity > 0:
            max_loss_pct = self._dollar_cap_pct
            if signal.direction == SignalDirection.LONG:
                dollar_stop = signal.close_price * (1 - max_loss_pct)
                stop = max(stop, dollar_stop)
            else:
                dollar_stop = signal.close_price * (1 + max_loss_pct)
                stop = min(stop, dollar_stop)

        return stop

    def _compute_targets(
        self,
        position: MeanReversionPosition,
        signal: MeanReversionSignal,
        entry_price: float,
    ) -> None:
        """Compute take-profit targets based on Bollinger Bands and Z-score."""
        mean_target = signal.bb_middle  # 20 SMA
        position.target_sma = mean_target
        position.stop_price = self._compute_stop_price(signal)

        if signal.direction == SignalDirection.LONG:
            distance = mean_target - entry_price
            position.target_tranche1 = entry_price + distance * self._tranche1_reversion
            position.target_tranche2 = mean_target
            # Tranche 3: opposite (upper) BB or Z = +1.0
            position.target_tranche3 = signal.bb_upper
        else:
            distance = entry_price - mean_target
            position.target_tranche1 = entry_price - distance * self._tranche1_reversion
            position.target_tranche2 = mean_target
            # Tranche 3: opposite (lower) BB or Z = -1.0
            position.target_tranche3 = signal.bb_lower

    def _compute_leverage(self, symbol: str) -> int:
        """Determine leverage based on volatility."""
        buf = self.get_buffer(symbol, "4h")
        if buf is None or len(buf) < 50:
            return self._config.get("leverage_preferred", 2)

        closes = buf.get_closes()
        highs = buf.get_highs()
        lows = buf.get_lows()

        atr_arr = compute_atr(highs, lows, closes, period=14)
        valid = atr_arr[~np.isnan(atr_arr)]
        if len(valid) == 0:
            return self._config.get("leverage_preferred", 2)

        current_atr = float(valid[-1])
        current_price = float(closes[-1])

        atr_pct = (current_atr / current_price) * 100 if current_price > 0 else 0

        threshold = self._config.get("high_vol_atr_threshold", 4.0)
        if atr_pct > threshold:
            return self._config.get("leverage_high_vol", 1)

        return self._config.get("leverage_preferred", 2)

    def _effective_bb_std(self, symbol: str) -> float:
        """Return BB std dev, bumped in extreme volatility."""
        base = self._bb_std
        if self._is_extreme_volatility(symbol):
            base += self._extreme_vol_z_bump
        return base

    def _effective_z_threshold(self, symbol: str) -> float:
        """Return Z-score threshold, bumped in extreme volatility."""
        base = self._zscore_threshold
        if self._is_extreme_volatility(symbol):
            base += self._extreme_vol_z_bump
        return base

    def _is_extreme_volatility(self, symbol: str) -> bool:
        """Check if 4h ATR > 2x its 50-period average."""
        buf = self.get_buffer(symbol, "4h")
        if buf is None or len(buf) < 50:
            return False

        highs = buf.get_highs()
        lows = buf.get_lows()
        closes = buf.get_closes()

        atr_arr = compute_atr(highs, lows, closes, period=14)
        valid = atr_arr[~np.isnan(atr_arr)]
        if len(valid) < 50:
            return False

        current = float(valid[-1])
        avg = float(np.mean(valid[-50:]))
        return avg > 0 and current > self._extreme_vol_atr_mult * avg

    def _check_extreme_volatility(self, symbol: str) -> float:
        """Return size multiplier (1.0 normal, 0.5 extreme vol)."""
        if self._is_extreme_volatility(symbol):
            return self._extreme_vol_size_red
        return 1.0

    def _check_entry_filters(self, symbol: str, closes: np.ndarray, buf: IndicatorBuffer) -> bool:
        """Run all pre-signal filters. Returns True if entry is allowed."""
        # BB width filter
        upper, middle, lower = bollinger_bands(closes, period=self._bb_period, num_std=self._bb_std)
        if not np.isnan(upper[-1]) and not np.isnan(lower[-1]) and float(closes[-1]) > 0:
            bb_width = (float(upper[-1]) - float(lower[-1])) / float(closes[-1])
            if bb_width < self._bb_width_min_pct:
                logger.debug("BB width too narrow for %s: %.4f < %.4f", symbol, bb_width, self._bb_width_min_pct)
                return False

        # Fee threshold: expected move must exceed 3x fees
        sma_val = float(middle[-1]) if not np.isnan(middle[-1]) else 0
        current_price = float(closes[-1])
        if sma_val > 0 and current_price > 0:
            expected_move_pct = abs(sma_val - current_price) / current_price
            min_move = self._fee_threshold_mult * self._rt_fee_pct
            if expected_move_pct < min_move:
                logger.debug(
                    "Expected move too small for %s: %.4f%% < %.4f%%",
                    symbol, expected_move_pct * 100, min_move * 100,
                )
                return False

        # Risk manager filters (whipsaw, blacklist, etc.)
        if self._risk:
            if not self._risk.check_instrument_allowed(symbol):
                return False

        return True

    def _check_reversion_expiry(self, signal: MeanReversionSignal, current_price: float) -> bool:
        """Check if price has reverted > 50% before entry (opportunity expired)."""
        if signal.bb_middle <= 0 or signal.close_price <= 0:
            return False

        total_distance = abs(signal.bb_middle - signal.close_price)
        if total_distance <= 0:
            return False

        if signal.direction == SignalDirection.LONG:
            reverted = current_price - signal.close_price
        else:
            reverted = signal.close_price - current_price

        reversion_pct = (reverted / total_distance) * 100
        return reversion_pct >= self._reversion_expiry_pct

    def _calc_pnl(self, position: MeanReversionPosition, exit_price: float, quantity: float) -> float:
        """Calculate PnL for a partial close."""
        if position.direction == SignalDirection.LONG:
            return (exit_price - position.entry_price) * quantity
        else:
            return (position.entry_price - exit_price) * quantity

    async def _update_position_pnl(self, symbol: str, current_price: float) -> None:
        """Update unrealized PnL and excursion tracking."""
        position = self._positions.get(symbol)
        if position is None or position.entry_price <= 0:
            return

        pnl = self._calc_pnl(position, current_price, position.remaining_quantity)
        position.unrealized_pnl = pnl

        if pnl > position.max_favorable_excursion:
            position.max_favorable_excursion = pnl
        if pnl < position.max_adverse_excursion:
            position.max_adverse_excursion = pnl

        # Update paper trading PnL
        if self._paper:
            self._paper.update_position_pnl(symbol, current_price)

    async def _move_stop_to_breakeven(self, position: MeanReversionPosition) -> None:
        """Alias for breakeven stop activation after tranche 1."""
        # Already handled in _check_breakeven_stop, but called explicitly here
        current_price = self._mark_prices.get(position.symbol, 0)
        if current_price > 0:
            await self._check_breakeven_stop(position, current_price)

    def _get_equity(self) -> float:
        """Get current equity from paper engine or risk manager."""
        if self._paper:
            return self._paper.get_equity()
        if self._risk:
            return self._risk.get_current_equity()
        return 0.0

    # ==================================================================
    # Public API (for dashboard / main)
    # ==================================================================

    def get_positions(self) -> List[dict]:
        """Return all active positions as dicts."""
        return [p.to_dict() for p in self._positions.values()]

    def get_pending_signals(self) -> List[dict]:
        """Return pending (unconfirmed) signals."""
        return [s.to_dict() for s in self._pending_signals.values()]

    def get_metrics(self) -> dict:
        """Return strategy-specific metrics."""
        return {
            "signals_generated": self._signals_generated,
            "signals_confirmed": self._signals_confirmed,
            "signals_expired": self._signals_expired,
            "trades_completed": self._trades_completed,
            "total_pnl": round(self._total_pnl, 4),
            "active_positions": len(self._positions),
            "pending_signals": len(self._pending_signals),
            "win_rate": self._compute_win_rate(),
        }

    def _compute_win_rate(self) -> float:
        """Compute win rate from risk manager trade results."""
        if self._risk:
            results = self._risk.get_trade_results()
            if results:
                wins = sum(1 for r in results if r)
                return round(wins / len(results) * 100, 1)
        return 0.0

    def get_indicator_data(self, symbol: str) -> dict:
        """Return current indicator values for dashboard display."""
        buf = self.get_buffer(symbol, "4h")
        if buf is None or len(buf) < max(self._bb_period, self._zscore_period):
            return {}

        closes = buf.get_closes()
        upper, middle, lower = bollinger_bands(closes, self._bb_period, self._bb_std)
        z_arr = compute_zscore(closes, self._zscore_period)
        rsi_arr = compute_rsi(closes, self._rsi_period)

        return {
            "symbol": symbol,
            "close": float(closes[-1]) if len(closes) > 0 else 0,
            "bb_upper": float(upper[-1]) if not np.isnan(upper[-1]) else 0,
            "bb_middle": float(middle[-1]) if not np.isnan(middle[-1]) else 0,
            "bb_lower": float(lower[-1]) if not np.isnan(lower[-1]) else 0,
            "zscore": float(z_arr[-1]) if not np.isnan(z_arr[-1]) else 0,
            "rsi": float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 0,
            "bb_closes": closes[-50:].tolist() if len(closes) >= 50 else closes.tolist(),
            "bb_upper_arr": [float(x) if not np.isnan(x) else None for x in upper[-50:]],
            "bb_middle_arr": [float(x) if not np.isnan(x) else None for x in middle[-50:]],
            "bb_lower_arr": [float(x) if not np.isnan(x) else None for x in lower[-50:]],
            "zscore_arr": [float(x) if not np.isnan(x) else None for x in z_arr[-50:]],
        }
