"""STRAT-001 Exit Logic — Section 4.

Exit management for all open positions:
  4.1 Hard stop: 2.0x ATR(14) 4h, LOCKED at entry, STOP_MARKET within 500ms
  4.2 Trailing stop: activate at 1.0x ATR profit, trail 1.5x ATR, step 0.1x ATR
  4.3 Four-tranche TP: 25% at 1.0/2.0/3.0x ATR, 25% rides trailing
  4.4 Breakeven: entry + fees when 1.0x ATR reached AND first TP filled
  4.5 Time exit: max 10 days; 50% if no TP1 in 5d; close all if no TP2 in 10d
  4.6 Signal exit: reverse crossover -> close all; RSI extreme -> 50% + tighten
  4.7 Exit priority: Hard > Signal > Trailing > Time > TP
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from shared.binance_client import BinanceClient, BinanceClientError
from shared.alerting import AlertLevel

from . import STRATEGY_ID
from .strategy import SignalDirection, _is_valid

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

class ExitReason(str, Enum):
    HARD_STOP = "HARD_STOP"
    TRAILING_STOP = "TRAILING_STOP"
    SIGNAL_REVERSE = "SIGNAL_REVERSE"
    SIGNAL_RSI_EXTREME = "SIGNAL_RSI_EXTREME"
    SIGNAL_ADX_LOW = "SIGNAL_ADX_LOW"
    TIME_PARTIAL = "TIME_PARTIAL"
    TIME_FULL = "TIME_FULL"
    TP1 = "TP1"
    TP2 = "TP2"
    TP3 = "TP3"
    TP4_TRAILING = "TP4_TRAILING"
    BREAKEVEN = "BREAKEVEN"
    MANUAL = "MANUAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class TrailingStopState:
    """Tracks trailing stop state for a position."""
    active: bool = False
    activation_price: float = 0.0       # price at which trailing activates
    highest_favorable: float = 0.0      # best price since activation
    current_stop: float = 0.0           # current trailing stop price
    trail_distance: float = 0.0         # distance in price terms
    step_size: float = 0.0              # minimum step to update
    tightened: bool = False             # set after TP3
    stop_order_id: Optional[int] = None


@dataclass
class TakeProfitState:
    """Tracks take-profit tranche execution."""
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: float = 0.0
    tp1_filled: bool = False
    tp2_filled: bool = False
    tp3_filled: bool = False
    tp1_order_id: Optional[int] = None
    tp2_order_id: Optional[int] = None
    tp3_order_id: Optional[int] = None
    tp1_qty: float = 0.0
    tp2_qty: float = 0.0
    tp3_qty: float = 0.0
    tp4_qty: float = 0.0   # rides trailing


@dataclass
class ManagedPosition:
    """Full state for a managed position."""
    symbol: str
    direction: SignalDirection
    entry_price: float
    entry_time_ms: int
    total_quantity: float
    remaining_quantity: float
    atr_at_entry: float                 # LOCKED — does not recalculate
    hard_stop_price: float
    hard_stop_order_id: Optional[int] = None
    trailing: TrailingStopState = field(default_factory=TrailingStopState)
    tp: TakeProfitState = field(default_factory=TakeProfitState)
    breakeven_set: bool = False
    time_extension_used: bool = False
    scale_in_count: int = 0
    avg_entry_price: float = 0.0        # updated on scale-ins
    total_fees: float = 0.0
    leverage: int = 1
    last_price: float = 0.0
    unrealized_pnl: float = 0.0
    highest_pnl: float = 0.0
    r_multiple: float = 0.0

    def __post_init__(self):
        if self.avg_entry_price == 0.0:
            self.avg_entry_price = self.entry_price

    @property
    def initial_risk_distance(self) -> float:
        return self.atr_at_entry * 2.0  # hard stop distance

    @property
    def entry_age_hours(self) -> float:
        return (time.time() * 1000 - self.entry_time_ms) / (3600 * 1000)

    @property
    def entry_age_days(self) -> float:
        return self.entry_age_hours / 24.0

    def update_pnl(self, current_price: float) -> None:
        """Update unrealized PnL and R-multiple."""
        self.last_price = current_price
        if self.direction == SignalDirection.LONG:
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.remaining_quantity
        else:
            self.unrealized_pnl = (self.avg_entry_price - current_price) * self.remaining_quantity

        if self.initial_risk_distance > 0:
            pnl_per_unit = abs(current_price - self.avg_entry_price)
            self.r_multiple = pnl_per_unit / self.initial_risk_distance
            if self.unrealized_pnl < 0:
                self.r_multiple = -self.r_multiple

        if self.unrealized_pnl > self.highest_pnl:
            self.highest_pnl = self.unrealized_pnl


# ---------------------------------------------------------------------------
# ExitManager
# ---------------------------------------------------------------------------

class ExitManager:
    """Manages all exit logic for STRAT-001 positions.

    Parameters
    ----------
    config : dict
        ``strategy_params`` from config.yaml.
    client : BinanceClient
        Shared REST client.
    alerter : object
        Alerting system.
    paper_engine : object | None
        Paper trading engine, if in paper mode.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        client: BinanceClient,
        alerter: Any,
        paper_engine: Any = None,
        on_trade_close_callback: Any = None,
    ) -> None:
        self.cfg = config
        self.client = client
        self.alerter = alerter
        self.paper = paper_engine
        self._on_trade_close_callback = on_trade_close_callback

        # Active positions: symbol -> ManagedPosition
        self.positions: Dict[str, ManagedPosition] = {}

        # Closed trade records (for performance tracking)
        self._closed_trades: List[Dict[str, Any]] = []

    # ======================================================================
    # Position creation
    # ======================================================================

    def create_position(
        self,
        symbol: str,
        direction: SignalDirection,
        entry_price: float,
        quantity: float,
        atr_value: float,
        leverage: int = 1,
        fees: float = 0.0,
        stop_order_id: Optional[int] = None,
    ) -> ManagedPosition:
        """Create and register a new managed position."""
        hard_stop_mult = self.cfg.get("hard_stop_atr_mult", 2.0)

        if direction == SignalDirection.LONG:
            hard_stop = entry_price - hard_stop_mult * atr_value
        else:
            hard_stop = entry_price + hard_stop_mult * atr_value

        # Calculate TP prices
        tp1_mult = self.cfg.get("tp1_atr_mult", 1.0)
        tp2_mult = self.cfg.get("tp2_atr_mult", 2.0)
        tp3_mult = self.cfg.get("tp3_atr_mult", 3.0)
        tp1_pct = self.cfg.get("tp1_pct", 25) / 100.0
        tp2_pct = self.cfg.get("tp2_pct", 25) / 100.0
        tp3_pct = self.cfg.get("tp3_pct", 25) / 100.0
        tp4_pct = self.cfg.get("tp4_pct", 25) / 100.0

        if direction == SignalDirection.LONG:
            tp1_price = entry_price + tp1_mult * atr_value
            tp2_price = entry_price + tp2_mult * atr_value
            tp3_price = entry_price + tp3_mult * atr_value
        else:
            tp1_price = entry_price - tp1_mult * atr_value
            tp2_price = entry_price - tp2_mult * atr_value
            tp3_price = entry_price - tp3_mult * atr_value

        # Trailing stop parameters
        trail_activation = self.cfg.get("trailing_activation_atr", 1.0)
        trail_distance = self.cfg.get("trailing_distance_atr", 1.5)
        trail_step = self.cfg.get("trailing_step_atr", 0.1)

        if direction == SignalDirection.LONG:
            activation_price = entry_price + trail_activation * atr_value
        else:
            activation_price = entry_price - trail_activation * atr_value

        pos = ManagedPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time_ms=int(time.time() * 1000),
            total_quantity=quantity,
            remaining_quantity=quantity,
            atr_at_entry=atr_value,
            hard_stop_price=hard_stop,
            hard_stop_order_id=stop_order_id,
            trailing=TrailingStopState(
                active=False,
                activation_price=activation_price,
                trail_distance=trail_distance * atr_value,
                step_size=trail_step * atr_value,
            ),
            tp=TakeProfitState(
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                tp3_price=tp3_price,
                tp1_qty=round(quantity * tp1_pct, 8),
                tp2_qty=round(quantity * tp2_pct, 8),
                tp3_qty=round(quantity * tp3_pct, 8),
                tp4_qty=round(quantity * tp4_pct, 8),
            ),
            leverage=leverage,
            total_fees=fees,
        )

        self.positions[symbol] = pos
        trade_logger.info(
            "POSITION_CREATED\tsymbol=%s\tdir=%s\tentry=%.4f\tqty=%.6f\t"
            "stop=%.4f\ttp1=%.4f\ttp2=%.4f\ttp3=%.4f\tatr=%.4f",
            symbol, direction.value, entry_price, quantity,
            hard_stop, tp1_price, tp2_price, tp3_price, atr_value,
        )
        return pos

    # ======================================================================
    # Main tick processor — called on every price update
    # ======================================================================

    async def process_tick(
        self,
        symbol: str,
        current_price: float,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[ExitReason, float]]:
        """Evaluate all exit conditions for *symbol* at *current_price*.

        Returns (exit_reason, qty_closed) if an exit was triggered, else None.
        Exit conditions are evaluated in priority order:
          Hard Stop > Signal-Based > Trailing > Time-Based > TP
        """
        pos = self.positions.get(symbol)
        if pos is None:
            return None

        pos.update_pnl(current_price)

        # ------------------------------------------------------------------
        # 1. Hard stop check (should be server-side, this is safety net)
        # ------------------------------------------------------------------
        if self._is_hard_stop_hit(pos, current_price):
            qty = pos.remaining_quantity
            await self._execute_exit(pos, qty, ExitReason.HARD_STOP, current_price)
            return ExitReason.HARD_STOP, qty

        # ------------------------------------------------------------------
        # 2. Signal-based exits (Section 4.6)
        # ------------------------------------------------------------------
        if indicators:
            signal_exit = await self._check_signal_exits(pos, current_price, indicators)
            if signal_exit:
                return signal_exit

        # ------------------------------------------------------------------
        # 3. Trailing stop management (Section 4.2)
        # ------------------------------------------------------------------
        trailing_exit = await self._manage_trailing_stop(pos, current_price)
        if trailing_exit:
            return trailing_exit

        # ------------------------------------------------------------------
        # 4. Time-based exits (Section 4.5)
        # ------------------------------------------------------------------
        time_exit = await self._check_time_exits(pos, current_price, indicators)
        if time_exit:
            return time_exit

        # ------------------------------------------------------------------
        # 5. Take profit / breakeven management (Sections 4.3, 4.4)
        # ------------------------------------------------------------------
        tp_exit = await self._manage_take_profits(pos, current_price)
        if tp_exit:
            return tp_exit

        return None

    # ======================================================================
    # Hard stop (Section 4.1)
    # ======================================================================

    def _is_hard_stop_hit(self, pos: ManagedPosition, price: float) -> bool:
        """Client-side safety check for hard stop."""
        if pos.direction == SignalDirection.LONG:
            return price <= pos.hard_stop_price
        else:
            return price >= pos.hard_stop_price

    # ======================================================================
    # Signal-based exits (Section 4.6)
    # ======================================================================

    async def _check_signal_exits(
        self,
        pos: ManagedPosition,
        price: float,
        indicators: Dict[str, Any],
    ) -> Optional[Tuple[ExitReason, float]]:
        """Check for signal-based exit conditions."""
        symbol = pos.symbol

        # Reverse crossover — close all
        cross_reversed = indicators.get("cross_reversed", False)
        if cross_reversed:
            logger.info("%s reverse crossover detected — closing all", symbol)
            qty = pos.remaining_quantity
            await self._execute_exit(pos, qty, ExitReason.SIGNAL_REVERSE, price)
            return ExitReason.SIGNAL_REVERSE, qty

        # RSI extreme — close 50% + tighten trailing
        rsi_val = indicators.get("rsi", float("nan"))
        if _is_valid(rsi_val):
            if pos.direction == SignalDirection.LONG and rsi_val > 80:
                close_qty = round(pos.remaining_quantity * 0.5, 8)
                if close_qty > 0:
                    logger.info("%s RSI %.1f overbought — closing 50%%", symbol, rsi_val)
                    await self._execute_partial_exit(pos, close_qty, ExitReason.SIGNAL_RSI_EXTREME, price)
                    # Tighten trailing to 0.75x ATR
                    if pos.trailing.active:
                        pos.trailing.trail_distance = 0.75 * pos.atr_at_entry
                        await self._update_trailing_order(pos)
                    return ExitReason.SIGNAL_RSI_EXTREME, close_qty

            if pos.direction == SignalDirection.SHORT and rsi_val < 20:
                close_qty = round(pos.remaining_quantity * 0.5, 8)
                if close_qty > 0:
                    logger.info("%s RSI %.1f oversold — closing 50%%", symbol, rsi_val)
                    await self._execute_partial_exit(pos, close_qty, ExitReason.SIGNAL_RSI_EXTREME, price)
                    if pos.trailing.active:
                        pos.trailing.trail_distance = 0.75 * pos.atr_at_entry
                        await self._update_trailing_order(pos)
                    return ExitReason.SIGNAL_RSI_EXTREME, close_qty

        # ADX < 20 — close all (trend dissipated)
        adx_val = indicators.get("adx", float("nan"))
        if _is_valid(adx_val) and adx_val < 20:
            logger.info("%s ADX %.1f < 20 — closing all (trend dissipated)", symbol, adx_val)
            qty = pos.remaining_quantity
            await self._execute_exit(pos, qty, ExitReason.SIGNAL_ADX_LOW, price)
            return ExitReason.SIGNAL_ADX_LOW, qty

        return None

    # ======================================================================
    # Trailing stop (Section 4.2)
    # ======================================================================

    async def _manage_trailing_stop(
        self,
        pos: ManagedPosition,
        price: float,
    ) -> Optional[Tuple[ExitReason, float]]:
        """Manage trailing stop activation, updates, and execution."""
        ts = pos.trailing

        # Check activation
        if not ts.active:
            should_activate = False
            if pos.direction == SignalDirection.LONG:
                if price >= ts.activation_price:
                    should_activate = True
                    ts.highest_favorable = price
            else:
                if price <= ts.activation_price:
                    should_activate = True
                    ts.highest_favorable = price

            if should_activate:
                ts.active = True
                if pos.direction == SignalDirection.LONG:
                    ts.current_stop = price - ts.trail_distance
                else:
                    ts.current_stop = price + ts.trail_distance

                logger.info(
                    "%s trailing stop activated at price %.4f, stop=%.4f trail=%.4f",
                    pos.symbol, price, ts.current_stop, ts.trail_distance,
                )

                # Cancel original hard stop and place trailing
                await self._cancel_hard_stop(pos)
                await self._place_trailing_order(pos)
            return None

        # Update trailing stop
        if pos.direction == SignalDirection.LONG:
            if price > ts.highest_favorable:
                step_count = int((price - ts.highest_favorable) / ts.step_size) if ts.step_size > 0 else 0
                if step_count > 0:
                    ts.highest_favorable = price
                    new_stop = price - ts.trail_distance
                    if new_stop > ts.current_stop:
                        ts.current_stop = new_stop
                        await self._update_trailing_order(pos)

            # Check if trailing stop hit
            if price <= ts.current_stop:
                logger.info("%s trailing stop hit at %.4f", pos.symbol, price)
                qty = pos.remaining_quantity
                await self._execute_exit(pos, qty, ExitReason.TRAILING_STOP, price)
                return ExitReason.TRAILING_STOP, qty
        else:
            if price < ts.highest_favorable:
                step_count = int((ts.highest_favorable - price) / ts.step_size) if ts.step_size > 0 else 0
                if step_count > 0:
                    ts.highest_favorable = price
                    new_stop = price + ts.trail_distance
                    if new_stop < ts.current_stop:
                        ts.current_stop = new_stop
                        await self._update_trailing_order(pos)

            if price >= ts.current_stop:
                logger.info("%s trailing stop hit at %.4f", pos.symbol, price)
                qty = pos.remaining_quantity
                await self._execute_exit(pos, qty, ExitReason.TRAILING_STOP, price)
                return ExitReason.TRAILING_STOP, qty

        return None

    async def _place_trailing_order(self, pos: ManagedPosition) -> None:
        """Place TRAILING_STOP_MARKET or fallback STOP_MARKET."""
        if self.paper:
            pos.trailing.stop_order_id = None
            return

        symbol = pos.symbol
        side = "SELL" if pos.direction == SignalDirection.LONG else "BUY"

        # Calculate callback rate for TRAILING_STOP_MARKET
        callback_rate = (pos.trailing.trail_distance / pos.last_price) * 100.0
        callback_rate = round(max(0.1, min(5.0, callback_rate)), 1)

        try:
            order = await self.client.place_futures_order(
                symbol=symbol,
                side=side,
                type="TRAILING_STOP_MARKET",
                quantity=pos.remaining_quantity,
                callback_rate=callback_rate,
                reduce_only=True,
            )
            pos.trailing.stop_order_id = order.get("orderId")
            logger.info(
                "%s TRAILING_STOP_MARKET placed: callback=%.1f%% qty=%.6f",
                symbol, callback_rate, pos.remaining_quantity,
            )
        except BinanceClientError:
            # Fallback: use STOP_MARKET
            logger.info("%s falling back to STOP_MARKET for trailing", symbol)
            try:
                order = await self.client.place_futures_order(
                    symbol=symbol,
                    side=side,
                    type="STOP_MARKET",
                    quantity=pos.remaining_quantity,
                    stop_price=pos.trailing.current_stop,
                    reduce_only=True,
                )
                pos.trailing.stop_order_id = order.get("orderId")
            except BinanceClientError as e:
                logger.error("%s trailing STOP_MARKET failed: %s", symbol, e)

    async def _update_trailing_order(self, pos: ManagedPosition) -> None:
        """Cancel and replace trailing stop order (must complete <200ms)."""
        if self.paper:
            return

        start_ms = int(time.time() * 1000)
        symbol = pos.symbol
        side = "SELL" if pos.direction == SignalDirection.LONG else "BUY"
        cancel_replace_deadline = self.cfg.get("trailing_cancel_replace_ms", 200)

        # Cancel existing
        if pos.trailing.stop_order_id:
            try:
                await self.client.cancel_futures_order(symbol, pos.trailing.stop_order_id)
            except BinanceClientError:
                pass

        # Place new
        try:
            order = await self.client.place_futures_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                quantity=pos.remaining_quantity,
                stop_price=pos.trailing.current_stop,
                reduce_only=True,
            )
            pos.trailing.stop_order_id = order.get("orderId")
        except BinanceClientError as e:
            logger.error("%s trailing stop update failed: %s", symbol, e)

        elapsed = int(time.time() * 1000) - start_ms
        if elapsed > cancel_replace_deadline:
            logger.warning(
                "%s trailing stop cancel-replace took %dms (deadline %dms)",
                symbol, elapsed, cancel_replace_deadline,
            )

    async def _cancel_hard_stop(self, pos: ManagedPosition) -> None:
        """Cancel the original hard stop loss order."""
        if pos.hard_stop_order_id and not self.paper:
            try:
                await self.client.cancel_futures_order(pos.symbol, pos.hard_stop_order_id)
                logger.info("%s hard stop cancelled (trailing activated)", pos.symbol)
            except BinanceClientError:
                pass

    # ======================================================================
    # Take profit management (Sections 4.3, 4.4)
    # ======================================================================

    async def _manage_take_profits(
        self,
        pos: ManagedPosition,
        price: float,
    ) -> Optional[Tuple[ExitReason, float]]:
        """Check and execute TP tranches + breakeven logic."""
        tp = pos.tp

        # TP1: 25% at 1.0x ATR
        if not tp.tp1_filled:
            hit = (
                (pos.direction == SignalDirection.LONG and price >= tp.tp1_price)
                or (pos.direction == SignalDirection.SHORT and price <= tp.tp1_price)
            )
            if hit and tp.tp1_qty > 0 and tp.tp1_qty <= pos.remaining_quantity:
                await self._execute_partial_exit(pos, tp.tp1_qty, ExitReason.TP1, price)
                tp.tp1_filled = True

                # Move stop to breakeven (Section 4.4)
                await self._set_breakeven_stop(pos)

                return ExitReason.TP1, tp.tp1_qty

        # TP2: 25% at 2.0x ATR
        if tp.tp1_filled and not tp.tp2_filled:
            hit = (
                (pos.direction == SignalDirection.LONG and price >= tp.tp2_price)
                or (pos.direction == SignalDirection.SHORT and price <= tp.tp2_price)
            )
            if hit and tp.tp2_qty > 0 and tp.tp2_qty <= pos.remaining_quantity:
                await self._execute_partial_exit(pos, tp.tp2_qty, ExitReason.TP2, price)
                tp.tp2_filled = True

                # Activate trailing stop on remaining
                if not pos.trailing.active:
                    pos.trailing.active = True
                    pos.trailing.highest_favorable = price
                    if pos.direction == SignalDirection.LONG:
                        pos.trailing.current_stop = price - pos.trailing.trail_distance
                    else:
                        pos.trailing.current_stop = price + pos.trailing.trail_distance
                    await self._cancel_hard_stop(pos)
                    await self._place_trailing_order(pos)

                return ExitReason.TP2, tp.tp2_qty

        # TP3: 25% at 3.0x ATR
        if tp.tp2_filled and not tp.tp3_filled:
            hit = (
                (pos.direction == SignalDirection.LONG and price >= tp.tp3_price)
                or (pos.direction == SignalDirection.SHORT and price <= tp.tp3_price)
            )
            if hit and tp.tp3_qty > 0 and tp.tp3_qty <= pos.remaining_quantity:
                await self._execute_partial_exit(pos, tp.tp3_qty, ExitReason.TP3, price)
                tp.tp3_filled = True

                # Tighten trailing to 1.0x ATR on remaining 25%
                pos.trailing.trail_distance = 1.0 * pos.atr_at_entry
                pos.trailing.tightened = True
                await self._update_trailing_order(pos)

                return ExitReason.TP3, tp.tp3_qty

        return None

    async def _set_breakeven_stop(self, pos: ManagedPosition) -> None:
        """Move hard stop to breakeven = entry + round-trip fees (Section 4.4)."""
        # Estimate round-trip fees
        taker_fee = self.cfg.get("taker_fee_pct", 0.04) / 100.0 if self.paper else 0.0004
        fee_adjustment = pos.avg_entry_price * taker_fee * 2  # entry + exit

        if pos.direction == SignalDirection.LONG:
            breakeven_price = pos.avg_entry_price + fee_adjustment
        else:
            breakeven_price = pos.avg_entry_price - fee_adjustment

        pos.hard_stop_price = breakeven_price
        pos.breakeven_set = True

        # Replace the stop order
        if not self.paper and pos.hard_stop_order_id:
            side = "SELL" if pos.direction == SignalDirection.LONG else "BUY"
            try:
                await self.client.cancel_futures_order(pos.symbol, pos.hard_stop_order_id)
                order = await self.client.place_futures_order(
                    symbol=pos.symbol,
                    side=side,
                    type="STOP_MARKET",
                    quantity=pos.remaining_quantity,
                    stop_price=breakeven_price,
                    reduce_only=True,
                )
                pos.hard_stop_order_id = order.get("orderId")
            except BinanceClientError as e:
                logger.error("%s breakeven stop placement failed: %s", pos.symbol, e)

        logger.info(
            "%s breakeven stop set at %.4f (entry=%.4f + fees=%.4f)",
            pos.symbol, breakeven_price, pos.avg_entry_price, fee_adjustment,
        )

    # ======================================================================
    # Time-based exits (Section 4.5)
    # ======================================================================

    async def _check_time_exits(
        self,
        pos: ManagedPosition,
        price: float,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[ExitReason, float]]:
        """Check time-based exit conditions."""
        max_days = self.cfg.get("max_trade_days", 10)
        no_tp1_days = self.cfg.get("no_tp1_days", 5)
        extension_days = self.cfg.get("extension_days", 5)
        extension_adx = self.cfg.get("extension_adx_min", 30)
        no_tp1_close_pct = self.cfg.get("no_tp1_close_pct", 50) / 100.0

        age_days = pos.entry_age_days
        effective_max = max_days + (extension_days if pos.time_extension_used else 0)

        # Check for time extension eligibility
        if (
            not pos.time_extension_used
            and age_days >= max_days
            and pos.unrealized_pnl > 0
            and indicators
        ):
            adx_val = indicators.get("adx", 0)
            ema_20 = indicators.get("ema_20", 0)
            if _is_valid(adx_val) and adx_val > extension_adx:
                # Check price is on correct side of EMA20
                if (pos.direction == SignalDirection.LONG and price > ema_20) or \
                   (pos.direction == SignalDirection.SHORT and price < ema_20):
                    pos.time_extension_used = True
                    effective_max = max_days + extension_days
                    logger.info(
                        "%s time extension granted: profitable, ADX=%.1f, %d extra days",
                        pos.symbol, adx_val, extension_days,
                    )

        # 50% close if no TP1 after 5 days
        if age_days >= no_tp1_days and not pos.tp.tp1_filled:
            close_qty = round(pos.remaining_quantity * no_tp1_close_pct, 8)
            if close_qty > 0:
                logger.info(
                    "%s no TP1 after %.1f days — closing %.0f%%",
                    pos.symbol, age_days, no_tp1_close_pct * 100,
                )
                await self._execute_partial_exit(pos, close_qty, ExitReason.TIME_PARTIAL, price)
                return ExitReason.TIME_PARTIAL, close_qty

        # Full close after max duration
        if age_days >= effective_max:
            if not pos.tp.tp2_filled:
                logger.info(
                    "%s max duration %.1f days reached — closing all",
                    pos.symbol, age_days,
                )
                qty = pos.remaining_quantity
                await self._execute_exit(pos, qty, ExitReason.TIME_FULL, price)
                return ExitReason.TIME_FULL, qty

        return None

    # ======================================================================
    # Execution helpers
    # ======================================================================

    async def _execute_exit(
        self,
        pos: ManagedPosition,
        quantity: float,
        reason: ExitReason,
        price: float,
    ) -> None:
        """Execute a full position close."""
        await self._execute_partial_exit(pos, quantity, reason, price)
        # Cancel all remaining orders
        await self._cancel_all_position_orders(pos)
        # Remove from active positions
        self._record_closed_trade(pos, reason)
        self.positions.pop(pos.symbol, None)

    async def _execute_partial_exit(
        self,
        pos: ManagedPosition,
        quantity: float,
        reason: ExitReason,
        price: float,
    ) -> None:
        """Execute a partial position close."""
        symbol = pos.symbol
        side = "SELL" if pos.direction == SignalDirection.LONG else "BUY"
        quantity = min(quantity, pos.remaining_quantity)

        if quantity <= 0:
            return

        try:
            if self.paper:
                # Simulate market close
                logger.info(
                    "PAPER EXIT %s: %s qty=%.6f reason=%s price=%.4f",
                    symbol, side, quantity, reason.value, price,
                )
            else:
                await self.client.place_futures_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=quantity,
                    reduce_only=True,
                )
        except BinanceClientError as e:
            logger.error("%s exit order failed (%s): %s", symbol, reason.value, e)
            if self.alerter:
                await self.alerter.send(
                    f"EXIT ORDER FAILED {symbol} {reason.value}: {e}",
                    level=AlertLevel.CRITICAL,
                    strategy_id=STRATEGY_ID,
                )
            return

        pos.remaining_quantity -= quantity
        trade_logger.info(
            "EXIT\tsymbol=%s\treason=%s\tqty=%.6f\tprice=%.4f\tremaining=%.6f\t"
            "pnl=%.4f\tr_mult=%.2f\tage_hrs=%.1f",
            symbol, reason.value, quantity, price, pos.remaining_quantity,
            pos.unrealized_pnl, pos.r_multiple, pos.entry_age_hours,
        )

        # Update remaining orders quantities
        if pos.remaining_quantity > 0:
            # Adjust TP orders for new remaining size
            pass  # TP orders are already for fixed quantities

    async def _cancel_all_position_orders(self, pos: ManagedPosition) -> None:
        """Cancel all open orders associated with a position."""
        if self.paper:
            return

        order_ids = []
        if pos.hard_stop_order_id:
            order_ids.append(pos.hard_stop_order_id)
        if pos.trailing.stop_order_id:
            order_ids.append(pos.trailing.stop_order_id)
        for oid_attr in ("tp1_order_id", "tp2_order_id", "tp3_order_id"):
            oid = getattr(pos.tp, oid_attr, None)
            if oid:
                order_ids.append(oid)

        for oid in order_ids:
            try:
                await self.client.cancel_futures_order(pos.symbol, oid)
            except BinanceClientError:
                pass

    def _record_closed_trade(self, pos: ManagedPosition, reason: ExitReason) -> None:
        """Record a completed trade for performance tracking."""
        now_ms = int(time.time() * 1000)
        duration_ms = now_ms - pos.entry_time_ms

        record = {
            "symbol": pos.symbol,
            "direction": pos.direction.value,
            "entry_price": pos.entry_price,
            "avg_entry_price": pos.avg_entry_price,
            "exit_price": pos.last_price,
            "quantity": pos.total_quantity,
            "realized_pnl": pos.unrealized_pnl,  # At exit, unrealized becomes realized
            "fees": pos.total_fees,
            "r_multiple": pos.r_multiple,
            "entry_time_ms": pos.entry_time_ms,
            "exit_time_ms": now_ms,
            "duration_ms": duration_ms,
            "exit_reason": reason.value,
            "atr_at_entry": pos.atr_at_entry,
            "tp1_filled": pos.tp.tp1_filled,
            "tp2_filled": pos.tp.tp2_filled,
            "tp3_filled": pos.tp.tp3_filled,
            "scale_in_count": pos.scale_in_count,
            "breakeven_set": pos.breakeven_set,
            "time_extension_used": pos.time_extension_used,
            "leverage": pos.leverage,
            "highest_pnl": pos.highest_pnl,
        }
        self._closed_trades.append(record)

        trade_logger.info(
            "TRADE_CLOSED\tsymbol=%s\tdir=%s\tentry=%.4f\texit=%.4f\t"
            "pnl=%.4f\tr=%.2f\treason=%s\tduration_ms=%d\tscale_ins=%d",
            pos.symbol, pos.direction.value, pos.avg_entry_price,
            pos.last_price, pos.unrealized_pnl, pos.r_multiple,
            reason.value, duration_ms, pos.scale_in_count,
        )

        # Invoke the trade-close callback for metrics, dimensional, go-live
        if self._on_trade_close_callback is not None:
            try:
                import asyncio
                result = self._on_trade_close_callback(record)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception as e:
                logger.error("Trade close callback failed: %s", e)

    # ======================================================================
    # Emergency close
    # ======================================================================

    async def close_all_positions(self, reason: str = "EMERGENCY") -> int:
        """Close all positions immediately. Returns count of positions closed."""
        closed = 0
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            try:
                await self._execute_exit(
                    pos, pos.remaining_quantity, ExitReason.EMERGENCY, pos.last_price
                )
                closed += 1
            except Exception as e:
                logger.error("Emergency close failed for %s: %s", symbol, e)
        return closed

    async def close_losing_positions(self, pct: float = 0.5) -> int:
        """Close *pct* of losing positions. Returns count of closures."""
        closed = 0
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            if pos.unrealized_pnl < 0:
                close_qty = round(pos.remaining_quantity * pct, 8)
                if close_qty > 0:
                    await self._execute_partial_exit(
                        pos, close_qty, ExitReason.EMERGENCY, pos.last_price
                    )
                    closed += 1
        return closed

    # ======================================================================
    # State serialization
    # ======================================================================

    def get_positions_state(self) -> List[dict]:
        """Return all positions as serializable dicts."""
        result = []
        for sym, pos in self.positions.items():
            result.append({
                "symbol": pos.symbol,
                "direction": pos.direction.value,
                "entry_price": pos.entry_price,
                "avg_entry_price": pos.avg_entry_price,
                "remaining_quantity": pos.remaining_quantity,
                "total_quantity": pos.total_quantity,
                "atr_at_entry": pos.atr_at_entry,
                "hard_stop_price": pos.hard_stop_price,
                "trailing_active": pos.trailing.active,
                "trailing_stop": pos.trailing.current_stop,
                "tp1_filled": pos.tp.tp1_filled,
                "tp2_filled": pos.tp.tp2_filled,
                "tp3_filled": pos.tp.tp3_filled,
                "breakeven_set": pos.breakeven_set,
                "unrealized_pnl": pos.unrealized_pnl,
                "r_multiple": pos.r_multiple,
                "entry_time_ms": pos.entry_time_ms,
                "entry_age_hours": pos.entry_age_hours,
                "leverage": pos.leverage,
                "scale_in_count": pos.scale_in_count,
                "last_price": pos.last_price,
            })
        return result

    def get_closed_trades(self) -> List[dict]:
        """Return closed trade records."""
        return list(self._closed_trades)
