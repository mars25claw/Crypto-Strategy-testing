"""STRAT-005 Core Grid Trading Strategy Logic.

Handles:
- Range identification (ATR-based + S/R validation)
- Grid parameter calculation (geometric / arithmetic)
- Grid deployment (buy below, sell above)
- Fill-triggered order placement (core mechanism)
- Cycle profit tracking
- Breakout detection and exit logic
- Trend/volatility/spread filters
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.indicators import atr as calc_atr, adx as calc_adx, ema as calc_ema, IndicatorBuffer

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")

# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class GridType(str, Enum):
    GEOMETRIC = "geometric"
    ARITHMETIC = "arithmetic"


class GridSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class BreakoutDirection(str, Enum):
    NONE = "none"
    UPSIDE = "upside"
    DOWNSIDE = "downside"


class DownsideOption(str, Enum):
    HOLD_WIDEN = "A"       # Hold inventory, widen grid downward
    LIQUIDATE = "B"        # Liquidate all, redeploy


@dataclass
class GridLevel:
    """Represents a single grid level."""
    index: int
    price: float
    side: Optional[GridSide] = None
    order_id: Optional[str] = None
    quantity: float = 0.0
    filled: bool = False
    fill_price: float = 0.0
    fill_time_ms: int = 0
    paired_order_id: Optional[str] = None  # The opposite-side order spawned by this fill
    paired_level_index: Optional[int] = None
    active: bool = False  # Has a live order on exchange

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "price": self.price,
            "side": self.side.value if self.side else None,
            "order_id": self.order_id,
            "quantity": self.quantity,
            "filled": self.filled,
            "fill_price": self.fill_price,
            "fill_time_ms": self.fill_time_ms,
            "paired_order_id": self.paired_order_id,
            "paired_level_index": self.paired_level_index,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GridLevel":
        return cls(
            index=d["index"],
            price=d["price"],
            side=GridSide(d["side"]) if d.get("side") else None,
            order_id=d.get("order_id"),
            quantity=d.get("quantity", 0.0),
            filled=d.get("filled", False),
            fill_price=d.get("fill_price", 0.0),
            fill_time_ms=d.get("fill_time_ms", 0),
            paired_order_id=d.get("paired_order_id"),
            paired_level_index=d.get("paired_level_index"),
            active=d.get("active", False),
        )


@dataclass
class GridCycle:
    """A completed buy-sell cycle."""
    cycle_id: str
    symbol: str
    buy_level: int
    sell_level: int
    buy_price: float
    sell_price: float
    quantity: float
    gross_profit: float
    fees: float
    net_profit: float
    buy_time_ms: int
    sell_time_ms: int
    duration_ms: int

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "symbol": self.symbol,
            "buy_level": self.buy_level,
            "sell_level": self.sell_level,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "quantity": self.quantity,
            "gross_profit": self.gross_profit,
            "fees": self.fees,
            "net_profit": self.net_profit,
            "buy_time_ms": self.buy_time_ms,
            "sell_time_ms": self.sell_time_ms,
            "duration_ms": self.duration_ms,
        }


@dataclass
class GridParameters:
    """Computed grid parameters for a single instrument."""
    symbol: str
    upper_boundary: float
    lower_boundary: float
    num_levels: int
    grid_type: GridType
    grid_spacing_pct: float  # For geometric: percentage per step
    grid_spacing_abs: float  # For arithmetic: dollar step
    center_price: float
    range_width_pct: float
    levels: List[GridLevel] = field(default_factory=list)
    created_at_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "upper_boundary": self.upper_boundary,
            "lower_boundary": self.lower_boundary,
            "num_levels": self.num_levels,
            "grid_type": self.grid_type.value,
            "grid_spacing_pct": self.grid_spacing_pct,
            "grid_spacing_abs": self.grid_spacing_abs,
            "center_price": self.center_price,
            "range_width_pct": self.range_width_pct,
            "levels": [l.to_dict() for l in self.levels],
            "created_at_ms": self.created_at_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GridParameters":
        params = cls(
            symbol=d["symbol"],
            upper_boundary=d["upper_boundary"],
            lower_boundary=d["lower_boundary"],
            num_levels=d["num_levels"],
            grid_type=GridType(d["grid_type"]),
            grid_spacing_pct=d["grid_spacing_pct"],
            grid_spacing_abs=d["grid_spacing_abs"],
            center_price=d["center_price"],
            range_width_pct=d["range_width_pct"],
            created_at_ms=d.get("created_at_ms", 0),
        )
        params.levels = [GridLevel.from_dict(l) for l in d.get("levels", [])]
        return params


@dataclass
class InstrumentState:
    """Full state for a single instrument's grid."""
    symbol: str
    grid_params: Optional[GridParameters] = None
    active: bool = False
    halted: bool = False
    halt_reason: str = ""
    halt_until_ms: int = 0

    # Inventory tracking
    inventory_qty: float = 0.0
    inventory_avg_cost: float = 0.0
    inventory_levels_filled: int = 0
    unrealized_pnl: float = 0.0

    # Cycle tracking
    total_cycles: int = 0
    total_cycle_profit: float = 0.0
    total_fees: float = 0.0

    # Breakout tracking
    breakout_direction: BreakoutDirection = BreakoutDirection.NONE
    consecutive_closes_above: int = 0
    consecutive_closes_below: int = 0
    last_4h_close_above: bool = False
    last_4h_close_below: bool = False

    # Consecutive level tracking (circuit breaker)
    consecutive_buy_fills: int = 0
    consecutive_sell_fills: int = 0

    # Capital allocation
    allocated_capital: float = 0.0
    realized_profit: float = 0.0

    # Recent prices
    current_price: float = 0.0
    mark_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0

    # Volume tracking (for circuit breakers)
    recent_volumes_15m: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "grid_params": self.grid_params.to_dict() if self.grid_params else None,
            "active": self.active,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "halt_until_ms": self.halt_until_ms,
            "inventory_qty": self.inventory_qty,
            "inventory_avg_cost": self.inventory_avg_cost,
            "inventory_levels_filled": self.inventory_levels_filled,
            "unrealized_pnl": self.unrealized_pnl,
            "total_cycles": self.total_cycles,
            "total_cycle_profit": self.total_cycle_profit,
            "total_fees": self.total_fees,
            "breakout_direction": self.breakout_direction.value,
            "consecutive_closes_above": self.consecutive_closes_above,
            "consecutive_closes_below": self.consecutive_closes_below,
            "consecutive_buy_fills": self.consecutive_buy_fills,
            "consecutive_sell_fills": self.consecutive_sell_fills,
            "allocated_capital": self.allocated_capital,
            "realized_profit": self.realized_profit,
            "current_price": self.current_price,
            "mark_price": self.mark_price,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InstrumentState":
        state = cls(symbol=d["symbol"])
        if d.get("grid_params"):
            state.grid_params = GridParameters.from_dict(d["grid_params"])
        state.active = d.get("active", False)
        state.halted = d.get("halted", False)
        state.halt_reason = d.get("halt_reason", "")
        state.halt_until_ms = d.get("halt_until_ms", 0)
        state.inventory_qty = d.get("inventory_qty", 0.0)
        state.inventory_avg_cost = d.get("inventory_avg_cost", 0.0)
        state.inventory_levels_filled = d.get("inventory_levels_filled", 0)
        state.unrealized_pnl = d.get("unrealized_pnl", 0.0)
        state.total_cycles = d.get("total_cycles", 0)
        state.total_cycle_profit = d.get("total_cycle_profit", 0.0)
        state.total_fees = d.get("total_fees", 0.0)
        state.breakout_direction = BreakoutDirection(d.get("breakout_direction", "none"))
        state.consecutive_closes_above = d.get("consecutive_closes_above", 0)
        state.consecutive_closes_below = d.get("consecutive_closes_below", 0)
        state.consecutive_buy_fills = d.get("consecutive_buy_fills", 0)
        state.consecutive_sell_fills = d.get("consecutive_sell_fills", 0)
        state.allocated_capital = d.get("allocated_capital", 0.0)
        state.realized_profit = d.get("realized_profit", 0.0)
        state.current_price = d.get("current_price", 0.0)
        state.mark_price = d.get("mark_price", 0.0)
        return state


# ---------------------------------------------------------------------------
# GridStrategy
# ---------------------------------------------------------------------------

class GridStrategy:
    """Core grid trading strategy logic.

    This class is pure computation -- no I/O. The GridManager handles
    order placement and WebSocket interactions.

    Parameters
    ----------
    config : dict
        Strategy-specific parameters from config.yaml strategy_params.
    exchange_info : ExchangeInfo
        For rounding prices/quantities.
    """

    def __init__(self, config: dict, exchange_info: Any = None) -> None:
        self._cfg = config
        self._exchange_info = exchange_info

        # Grid config
        self.grid_type = GridType(config.get("grid_type", "geometric"))
        self.num_levels = config.get("num_levels", 20)
        self.atr_multiplier = config.get("atr_multiplier", 2.0)
        self.atr_period = config.get("atr_period", 14)
        self.initial_mode = config.get("initial_mode", "neutral")

        # Range validation
        self.range_min_pct = config.get("range_min_pct", 5.0)
        self.range_max_pct = config.get("range_max_pct", 25.0)
        self.min_spacing_fee_mult = config.get("min_spacing_fee_multiple", 4.0)

        # Maker fee as decimal (0.02% = 0.0002)
        self.maker_fee = 0.0002
        self.taker_fee = 0.0004

        # Filters
        self.adx_max = config.get("adx_max", 25)
        self.ema_div_max_pct = config.get("ema_divergence_max_pct", 2.0)
        self.vol_min_pct = config.get("volatility_min_pct", 1.5)
        self.vol_max_pct = config.get("volatility_max_pct", 5.0)
        self.spread_max_pct = config.get("spread_max_pct", 0.05)

        # Exposure
        self.max_inventory_pct = config.get("max_inventory_pct", 60.0)
        self.unrealized_halt_pct = config.get("unrealized_loss_halt_pct", 3.0)
        self.hard_stop_pct = config.get("hard_stop_loss_pct", 5.0)
        self.profit_extraction_pct = config.get("profit_extraction_pct", 5.0)

        # Circuit breakers
        self.consec_levels_halt = config.get("consecutive_levels_halt", 8)
        self.flash_crash_pct = config.get("flash_crash_pct", 5.0)
        self.flash_crash_min = config.get("flash_crash_minutes", 5)
        self.volume_spike_mult = config.get("volume_spike_multiplier", 5.0)

        # Breakout
        self.breakout_candles = config.get("breakout_candles_confirm", 2)
        self.downside_option = DownsideOption(config.get("downside_option", "A"))
        self.hard_stop_atr_mult = config.get("hard_stop_atr_multiple", 1.5)

        # Refresh
        self.refresh_threshold_pct = config.get("refresh_boundary_threshold_pct", 5.0)

        # Indicator buffers per symbol per timeframe
        self._buffers: Dict[str, Dict[str, IndicatorBuffer]] = {}

        # Cycle counter for unique IDs
        self._cycle_counter = 0

    # ======================================================================
    #  Indicator buffer management
    # ======================================================================

    def get_buffer(self, symbol: str, tf: str) -> IndicatorBuffer:
        """Get or create an indicator buffer for symbol/timeframe."""
        if symbol not in self._buffers:
            self._buffers[symbol] = {}
        if tf not in self._buffers[symbol]:
            self._buffers[symbol][tf] = IndicatorBuffer(max_size=500)
        return self._buffers[symbol][tf]

    def process_kline(self, symbol: str, tf: str, kline: dict) -> None:
        """Process a kline update into the indicator buffer."""
        buf = self.get_buffer(symbol, tf)
        buf.add_candle({
            "timestamp": kline.get("t", kline.get("T", 0)),
            "open": float(kline.get("o", 0)),
            "high": float(kline.get("h", 0)),
            "low": float(kline.get("l", 0)),
            "close": float(kline.get("c", 0)),
            "volume": float(kline.get("v", kline.get("V", 0))),
        })

    def has_warmup_data(self, symbol: str) -> bool:
        """Check if we have sufficient data for all timeframes (200 candles each)."""
        required = {"1m": 200, "15m": 200, "4h": 200, "1d": 200}
        for tf, min_count in required.items():
            buf = self.get_buffer(symbol, tf)
            if len(buf) < min_count:
                return False
        return True

    # ======================================================================
    #  Section 3.1: Range Identification
    # ======================================================================

    def calculate_range(self, symbol: str, current_price: float) -> Optional[Tuple[float, float, float]]:
        """Calculate the grid range boundaries.

        Returns (upper, lower, range_width_pct) or None if range is not viable.

        Steps:
        1. ATR-based range: center = current_price, +/- 2.0 * daily ATR
        2. S/R validation: clamp with 30-day high/low + buffers
        3. Viability check: 5-25% range width
        """
        buf_1d = self.get_buffer(symbol, "1d")
        if len(buf_1d) < self.atr_period + 2:
            logger.warning("[%s] Insufficient daily data for ATR (%d candles)", symbol, len(buf_1d))
            return None

        highs = buf_1d.get_highs()
        lows = buf_1d.get_lows()
        closes = buf_1d.get_closes()

        # Step 1: ATR-based boundaries
        atr_values = calc_atr(highs, lows, closes, self.atr_period)
        daily_atr = atr_values[-1]
        if np.isnan(daily_atr) or daily_atr <= 0:
            logger.warning("[%s] Invalid ATR value: %s", symbol, daily_atr)
            return None

        atr_upper = current_price + (self.atr_multiplier * daily_atr)
        atr_lower = current_price - (self.atr_multiplier * daily_atr)

        # Step 2: S/R validation using 30-day high/low
        lookback = min(30, len(highs))
        high_30d = float(np.max(highs[-lookback:]))
        low_30d = float(np.min(lows[-lookback:]))

        upper = min(atr_upper, high_30d * 1.02)
        lower = max(atr_lower, low_30d * 0.98)

        # Sanity: upper must be above lower, both positive
        if upper <= lower or lower <= 0:
            logger.warning("[%s] Invalid range: upper=%.4f lower=%.4f", symbol, upper, lower)
            return None

        # Step 3: Viability check
        mid_price = (upper + lower) / 2.0
        range_width_pct = ((upper - lower) / mid_price) * 100.0

        if range_width_pct < self.range_min_pct:
            logger.info("[%s] Range too narrow: %.2f%% < %.2f%%", symbol, range_width_pct, self.range_min_pct)
            return None

        if range_width_pct > self.range_max_pct:
            logger.info("[%s] Range too wide: %.2f%% > %.2f%%", symbol, range_width_pct, self.range_max_pct)
            return None

        # Round boundaries
        if self._exchange_info:
            upper = self._exchange_info.round_price(symbol, upper)
            lower = self._exchange_info.round_price(symbol, lower)

        logger.info(
            "[%s] Range calculated: lower=%.4f upper=%.4f width=%.2f%% ATR=%.4f",
            symbol, lower, upper, range_width_pct, daily_atr,
        )
        return upper, lower, range_width_pct

    # ======================================================================
    #  Section 3.2: Grid Parameter Calculation
    # ======================================================================

    def calculate_grid_parameters(
        self,
        symbol: str,
        current_price: float,
        upper: float,
        lower: float,
        range_width_pct: float,
        allocated_capital: float,
        num_levels: Optional[int] = None,
    ) -> Optional[GridParameters]:
        """Calculate grid levels and spacing.

        Returns GridParameters or None if spacing doesn't meet fee viability.
        """
        n_levels = num_levels or self.num_levels
        n_levels = max(10, min(50, n_levels))

        if self.grid_type == GridType.GEOMETRIC:
            # Geometric spacing: each level is separated by equal percentage
            # Grid_Percentage = ((Upper/Lower)^(1/N) - 1) * 100
            ratio = upper / lower
            grid_pct = (ratio ** (1.0 / n_levels) - 1.0) * 100.0
            grid_abs = 0.0

            # Verify spacing vs fees (Section 3.2 Step 6)
            round_trip_fee_pct = self.maker_fee * 2 * 100  # 0.04%
            min_spacing = self.min_spacing_fee_mult * round_trip_fee_pct

            while grid_pct < min_spacing and n_levels > 10:
                n_levels -= 1
                grid_pct = ((upper / lower) ** (1.0 / n_levels) - 1.0) * 100.0
                logger.info(
                    "[%s] Reducing levels to %d for fee viability (spacing=%.4f%% min=%.4f%%)",
                    symbol, n_levels, grid_pct, min_spacing,
                )

            if grid_pct < min_spacing:
                logger.warning(
                    "[%s] Grid spacing %.4f%% still below minimum %.4f%% at %d levels",
                    symbol, grid_pct, min_spacing, n_levels,
                )
                return None

            # Generate levels
            levels = []
            for i in range(n_levels + 1):
                price = lower * (1.0 + grid_pct / 100.0) ** i
                if self._exchange_info:
                    price = self._exchange_info.round_price(symbol, price)
                levels.append(GridLevel(index=i, price=price))

        else:
            # Arithmetic spacing
            grid_abs = (upper - lower) / n_levels
            grid_pct = (grid_abs / lower) * 100.0  # approximate for fee check

            round_trip_fee_pct = self.maker_fee * 2 * 100
            min_spacing = self.min_spacing_fee_mult * round_trip_fee_pct

            while grid_pct < min_spacing and n_levels > 10:
                n_levels -= 1
                grid_abs = (upper - lower) / n_levels
                grid_pct = (grid_abs / lower) * 100.0

            if grid_pct < min_spacing:
                logger.warning("[%s] Arithmetic spacing insufficient", symbol)
                return None

            levels = []
            for i in range(n_levels + 1):
                price = lower + i * grid_abs
                if self._exchange_info:
                    price = self._exchange_info.round_price(symbol, price)
                levels.append(GridLevel(index=i, price=price))

        # Calculate quantity per level: Total_Capital / Num_Levels / Level_Price
        for level in levels:
            qty = allocated_capital / n_levels / level.price if level.price > 0 else 0
            if self._exchange_info:
                qty = self._exchange_info.round_quantity(symbol, qty)
            level.quantity = qty

            # Check min notional ($5)
            if self._exchange_info and not self._exchange_info.check_notional(symbol, qty, level.price):
                logger.warning(
                    "[%s] Level %d below min notional: qty=%.8f price=%.4f",
                    symbol, level.index, qty, level.price,
                )

        params = GridParameters(
            symbol=symbol,
            upper_boundary=upper,
            lower_boundary=lower,
            num_levels=n_levels,
            grid_type=self.grid_type,
            grid_spacing_pct=grid_pct,
            grid_spacing_abs=grid_abs,
            center_price=current_price,
            range_width_pct=range_width_pct,
            levels=levels,
            created_at_ms=int(time.time() * 1000),
        )

        logger.info(
            "[%s] Grid params: type=%s levels=%d spacing=%.4f%% upper=%.4f lower=%.4f",
            symbol, self.grid_type.value, n_levels, grid_pct, upper, lower,
        )
        return params

    # ======================================================================
    #  Section 3.3: Grid Deployment — Determine order sides
    # ======================================================================

    def assign_order_sides(self, params: GridParameters, current_price: float) -> None:
        """Assign BUY/SELL sides to grid levels based on current price.

        Levels BELOW current price get BUY orders.
        Levels ABOVE current price: no orders initially (neutral mode),
        or SELL if long-biased mode.
        """
        for level in params.levels:
            if level.price < current_price:
                level.side = GridSide.BUY
            elif level.price > current_price:
                if self.initial_mode == "long_biased":
                    level.side = GridSide.SELL
                else:
                    level.side = None  # No sell orders until buys fill (neutral)
            else:
                level.side = None  # At current price -- skip

    # ======================================================================
    #  Section 3.4: Fill-Triggered Order Placement
    # ======================================================================

    def on_buy_fill(
        self,
        state: InstrumentState,
        filled_level_index: int,
        fill_price: float,
        fill_qty: float,
        fill_time_ms: int,
    ) -> Optional[Tuple[int, float, float, GridSide]]:
        """Process a BUY fill. Returns (target_level_index, price, qty, SELL) for new order.

        Section 3.4: BUY fill at Level_N -> SELL LIMIT at Level_(N+1).
        """
        if state.grid_params is None:
            return None

        levels = state.grid_params.levels
        if filled_level_index < 0 or filled_level_index >= len(levels):
            return None

        # Mark the level as filled
        level = levels[filled_level_index]
        level.filled = True
        level.fill_price = fill_price
        level.fill_time_ms = fill_time_ms
        level.active = False

        # Update inventory
        old_total = state.inventory_qty * state.inventory_avg_cost
        state.inventory_qty += fill_qty
        if state.inventory_qty > 0:
            state.inventory_avg_cost = (old_total + fill_qty * fill_price) / state.inventory_qty
        state.inventory_levels_filled += 1

        # Track consecutive fills for circuit breaker
        state.consecutive_buy_fills += 1
        state.consecutive_sell_fills = 0

        # Determine the sell target: next level up
        sell_index = filled_level_index + 1
        if sell_index >= len(levels):
            logger.warning("[%s] Buy fill at top level %d, no sell target", state.symbol, filled_level_index)
            return None

        sell_level = levels[sell_index]
        sell_price = sell_level.price
        sell_qty = fill_qty  # Same quantity

        # Link the levels
        level.paired_level_index = sell_index

        trade_logger.info(
            "GRID_BUY_FILL\t%s\tlevel=%d\tprice=%.4f\tqty=%.8f\tsell_target_level=%d\tsell_price=%.4f",
            state.symbol, filled_level_index, fill_price, fill_qty, sell_index, sell_price,
        )

        return sell_index, sell_price, sell_qty, GridSide.SELL

    def on_sell_fill(
        self,
        state: InstrumentState,
        filled_level_index: int,
        fill_price: float,
        fill_qty: float,
        fill_time_ms: int,
    ) -> Optional[Tuple[int, float, float, GridSide, GridCycle]]:
        """Process a SELL fill. Returns (target_level_index, price, qty, BUY, cycle) for new order.

        Section 3.4: SELL fill at Level_N -> BUY LIMIT at Level_(N-1).
        Also calculates cycle profit.
        """
        if state.grid_params is None:
            return None

        levels = state.grid_params.levels
        if filled_level_index < 0 or filled_level_index >= len(levels):
            return None

        level = levels[filled_level_index]
        level.filled = True
        level.fill_price = fill_price
        level.fill_time_ms = fill_time_ms
        level.active = False

        # Update inventory
        state.inventory_qty -= fill_qty
        state.inventory_qty = max(0.0, state.inventory_qty)
        state.inventory_levels_filled = max(0, state.inventory_levels_filled - 1)

        # Track consecutive fills
        state.consecutive_sell_fills += 1
        state.consecutive_buy_fills = 0

        # Calculate cycle profit: find the corresponding buy
        buy_index = filled_level_index - 1
        cycle = None
        if 0 <= buy_index < len(levels):
            buy_level = levels[buy_index]
            if buy_level.fill_price > 0:
                gross = (fill_price - buy_level.fill_price) * fill_qty
                fees = (fill_price * fill_qty * self.maker_fee) + (buy_level.fill_price * fill_qty * self.maker_fee)
                net = gross - fees

                self._cycle_counter += 1
                cycle = GridCycle(
                    cycle_id=f"{state.symbol}-{self._cycle_counter}",
                    symbol=state.symbol,
                    buy_level=buy_index,
                    sell_level=filled_level_index,
                    buy_price=buy_level.fill_price,
                    sell_price=fill_price,
                    quantity=fill_qty,
                    gross_profit=gross,
                    fees=fees,
                    net_profit=net,
                    buy_time_ms=buy_level.fill_time_ms,
                    sell_time_ms=fill_time_ms,
                    duration_ms=fill_time_ms - buy_level.fill_time_ms,
                )

                state.total_cycles += 1
                state.total_cycle_profit += net
                state.total_fees += fees
                state.realized_profit += net

                trade_logger.info(
                    "GRID_CYCLE\t%s\tbuy_lvl=%d\tsell_lvl=%d\tbuy=%.4f\tsell=%.4f\t"
                    "qty=%.8f\tgross=%.6f\tfees=%.6f\tnet=%.6f\tcycles=%d",
                    state.symbol, buy_index, filled_level_index,
                    buy_level.fill_price, fill_price, fill_qty,
                    gross, fees, net, state.total_cycles,
                )

        # Determine buy target: previous level down
        buy_target_index = filled_level_index - 1
        if buy_target_index < 0:
            logger.warning("[%s] Sell fill at bottom level %d, no buy target", state.symbol, filled_level_index)
            return None

        buy_target = levels[buy_target_index]
        buy_price = buy_target.price
        buy_qty = fill_qty

        level.paired_level_index = buy_target_index

        trade_logger.info(
            "GRID_SELL_FILL\t%s\tlevel=%d\tprice=%.4f\tqty=%.8f\tbuy_target_level=%d\tbuy_price=%.4f",
            state.symbol, filled_level_index, fill_price, fill_qty, buy_target_index, buy_price,
        )

        return buy_target_index, buy_price, buy_qty, GridSide.BUY, cycle

    # ======================================================================
    #  Section 7: Filters & Disqualifiers
    # ======================================================================

    def check_trend_filter(self, symbol: str) -> Tuple[bool, str]:
        """Section 7.1: ADX < 25 required; EMA 20 vs 50 divergence < 2%.

        Returns (allowed, reason).
        """
        buf_4h = self.get_buffer(symbol, "4h")
        if len(buf_4h) < 60:
            return True, ""  # Not enough data, allow

        highs = buf_4h.get_highs()
        lows = buf_4h.get_lows()
        closes = buf_4h.get_closes()

        # ADX check
        adx_values, _, _ = calc_adx(highs, lows, closes, 14)
        adx_val = adx_values[-1]
        if not np.isnan(adx_val) and adx_val > self.adx_max:
            return False, f"ADX {adx_val:.1f} > {self.adx_max} (trending market)"

        # EMA divergence check
        ema20 = calc_ema(closes, 20)
        ema50 = calc_ema(closes, 50)
        if not np.isnan(ema20[-1]) and not np.isnan(ema50[-1]):
            mid = (ema20[-1] + ema50[-1]) / 2.0
            if mid > 0:
                divergence = abs(ema20[-1] - ema50[-1]) / mid * 100.0
                if divergence > self.ema_div_max_pct:
                    return False, f"EMA divergence {divergence:.2f}% > {self.ema_div_max_pct}%"

        return True, ""

    def check_volatility_filter(self, symbol: str) -> Tuple[bool, str]:
        """Section 7.2: Daily ATR as % of price must be between 1.5% and 5%."""
        buf_1d = self.get_buffer(symbol, "1d")
        if len(buf_1d) < self.atr_period + 2:
            return True, ""

        highs = buf_1d.get_highs()
        lows = buf_1d.get_lows()
        closes = buf_1d.get_closes()

        atr_values = calc_atr(highs, lows, closes, self.atr_period)
        atr_val = atr_values[-1]
        current_price = closes[-1]

        if np.isnan(atr_val) or current_price <= 0:
            return True, ""

        atr_pct = (atr_val / current_price) * 100.0

        if atr_pct < self.vol_min_pct:
            return False, f"Volatility {atr_pct:.2f}% < {self.vol_min_pct}% (insufficient movement)"
        if atr_pct > self.vol_max_pct:
            return False, f"Volatility {atr_pct:.2f}% > {self.vol_max_pct}% (breakout risk)"

        return True, ""

    def check_spread_filter(self, bid: float, ask: float) -> Tuple[bool, str]:
        """Section 7.3: Bid-ask spread < 0.05%."""
        if bid <= 0 or ask <= 0:
            return True, ""
        mid = (bid + ask) / 2.0
        spread_pct = ((ask - bid) / mid) * 100.0
        if spread_pct > self.spread_max_pct:
            return False, f"Spread {spread_pct:.4f}% > {self.spread_max_pct}%"
        return True, ""

    def check_all_filters(self, symbol: str, bid: float, ask: float) -> Tuple[bool, str]:
        """Run all pre-deployment filters. Returns (allowed, reason)."""
        ok, reason = self.check_trend_filter(symbol)
        if not ok:
            return False, reason

        ok, reason = self.check_volatility_filter(symbol)
        if not ok:
            return False, reason

        ok, reason = self.check_spread_filter(bid, ask)
        if not ok:
            return False, reason

        return True, ""

    # ======================================================================
    #  Section 4: Exit / Breakout Logic
    # ======================================================================

    def check_breakout(self, state: InstrumentState) -> BreakoutDirection:
        """Check for range breakout on 4h candle closes.

        Section 4.1/4.2: 2 consecutive 4h closes above/below grid boundary.
        """
        if state.grid_params is None:
            return BreakoutDirection.NONE

        buf_4h = self.get_buffer(state.symbol, "4h")
        if len(buf_4h) < 3:
            return BreakoutDirection.NONE

        closes = buf_4h.get_closes()
        upper = state.grid_params.upper_boundary
        lower = state.grid_params.lower_boundary

        # Check upside breakout
        if closes[-1] > upper and closes[-2] > upper:
            state.consecutive_closes_above = 2
            logger.warning("[%s] UPSIDE BREAKOUT: 2 consecutive 4h closes above %.4f", state.symbol, upper)
            return BreakoutDirection.UPSIDE
        elif closes[-1] > upper:
            state.consecutive_closes_above = 1
        else:
            state.consecutive_closes_above = 0

        # Check downside breakout
        if closes[-1] < lower and closes[-2] < lower:
            state.consecutive_closes_below = 2
            logger.warning("[%s] DOWNSIDE BREAKOUT: 2 consecutive 4h closes below %.4f", state.symbol, lower)
            return BreakoutDirection.DOWNSIDE
        elif closes[-1] < lower:
            state.consecutive_closes_below = 1
        else:
            state.consecutive_closes_below = 0

        return BreakoutDirection.NONE

    def should_refresh_grid(
        self, state: InstrumentState, current_price: float
    ) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
        """Section 4.5: Check if grid needs refresh (new boundaries differ >5%).

        Returns (should_refresh, new_range_or_none).
        """
        if state.grid_params is None:
            return True, None

        new_range = self.calculate_range(state.symbol, current_price)
        if new_range is None:
            return False, None

        new_upper, new_lower, new_width = new_range
        old_upper = state.grid_params.upper_boundary
        old_lower = state.grid_params.lower_boundary

        upper_change = abs(new_upper - old_upper) / old_upper * 100.0 if old_upper > 0 else 100.0
        lower_change = abs(new_lower - old_lower) / old_lower * 100.0 if old_lower > 0 else 100.0

        if upper_change > self.refresh_threshold_pct or lower_change > self.refresh_threshold_pct:
            logger.info(
                "[%s] Grid refresh needed: upper_change=%.2f%% lower_change=%.2f%%",
                state.symbol, upper_change, lower_change,
            )
            return True, new_range

        return False, None

    def should_extract_profit(self, state: InstrumentState) -> bool:
        """Section 4.4: Extract profit when realized profit >= 5% of allocated capital."""
        if state.allocated_capital <= 0:
            return False
        return state.realized_profit >= (state.allocated_capital * self.profit_extraction_pct / 100.0)

    # ======================================================================
    #  Section 5.5: Circuit Breaker Checks
    # ======================================================================

    def check_consecutive_levels(self, state: InstrumentState) -> Tuple[bool, str]:
        """Section 5.5: Halt if 8 consecutive levels crossed in one direction."""
        if state.consecutive_buy_fills >= self.consec_levels_halt:
            return True, f"Consecutive buy fills: {state.consecutive_buy_fills} >= {self.consec_levels_halt}"
        if state.consecutive_sell_fills >= self.consec_levels_halt:
            return True, f"Consecutive sell fills: {state.consecutive_sell_fills} >= {self.consec_levels_halt}"
        return False, ""

    def check_flash_crash(self, symbol: str) -> Tuple[bool, str]:
        """Section 5.5: Price drop > 5% in 5 minutes."""
        buf_1m = self.get_buffer(symbol, "1m")
        if len(buf_1m) < self.flash_crash_min + 1:
            return False, ""

        closes = buf_1m.get_closes()
        window = closes[-(self.flash_crash_min + 1):]
        high_in_window = float(np.max(window[:-1]))
        current = float(window[-1])

        if high_in_window <= 0:
            return False, ""

        drop_pct = ((high_in_window - current) / high_in_window) * 100.0
        if drop_pct >= self.flash_crash_pct:
            return True, f"Flash crash: {drop_pct:.2f}% drop in {self.flash_crash_min}m"
        return False, ""

    def check_volume_spike(self, state: InstrumentState) -> Tuple[bool, str]:
        """Section 5.5: 15m volume exceeds 5x the 20-period average."""
        buf_15m = self.get_buffer(state.symbol, "15m")
        if len(buf_15m) < 21:
            return False, ""

        volumes = buf_15m.get_volumes()
        avg_vol = float(np.mean(volumes[-21:-1]))
        current_vol = float(volumes[-1])

        if avg_vol <= 0:
            return False, ""

        ratio = current_vol / avg_vol
        if ratio >= self.volume_spike_mult:
            return True, f"Volume spike: {ratio:.1f}x average (threshold {self.volume_spike_mult}x)"
        return False, ""

    def check_inventory_exposure(
        self, state: InstrumentState
    ) -> Tuple[bool, str]:
        """Section 5.2: Max 60% of buy levels filled without corresponding sells."""
        if state.grid_params is None:
            return False, ""

        total_buy_levels = sum(1 for l in state.grid_params.levels if l.side == GridSide.BUY or l.filled)
        if total_buy_levels == 0:
            return False, ""

        fill_pct = (state.inventory_levels_filled / total_buy_levels) * 100.0
        if fill_pct > self.max_inventory_pct:
            return True, f"Inventory {fill_pct:.1f}% > {self.max_inventory_pct}% of levels"
        return False, ""

    # ======================================================================
    #  Section 7.7: STRAT-001 Conflict Check
    # ======================================================================

    def check_strat001_conflict(
        self, symbol: str, cross_strategy_positions: List[dict]
    ) -> Tuple[bool, str]:
        """Section 7.7: If STRAT-001 has active LONG, grid selling conflicts."""
        for pos in cross_strategy_positions:
            if (pos.get("strategy_id", "").startswith("STRAT-001")
                    and pos.get("symbol") == symbol
                    and pos.get("direction", "").upper() == "LONG"):
                return True, f"STRAT-001 has active LONG on {symbol}"
        return False, ""

    # ======================================================================
    #  Section 5.3 / 5.4: Drawdown Checks
    # ======================================================================

    def check_unrealized_loss_halt(
        self, state: InstrumentState, total_equity: float
    ) -> Tuple[bool, str]:
        """Section 5.3: Unrealized loss > 3% of equity -> cancel buys."""
        if total_equity <= 0 or state.unrealized_pnl >= 0:
            return False, ""
        loss_pct = abs(state.unrealized_pnl) / total_equity * 100.0
        if loss_pct >= self.unrealized_halt_pct:
            return True, f"Unrealized loss {loss_pct:.2f}% >= {self.unrealized_halt_pct}%"
        return False, ""

    def check_hard_stop(
        self, state: InstrumentState, total_equity: float
    ) -> Tuple[bool, str]:
        """Section 5.4: Unrealized loss > 5% of equity -> liquidate all."""
        if total_equity <= 0 or state.unrealized_pnl >= 0:
            return False, ""
        loss_pct = abs(state.unrealized_pnl) / total_equity * 100.0
        if loss_pct >= self.hard_stop_pct:
            return True, f"HARD STOP: loss {loss_pct:.2f}% >= {self.hard_stop_pct}%"
        return False, ""

    # ======================================================================
    #  Unrealized PnL computation
    # ======================================================================

    def update_unrealized_pnl(self, state: InstrumentState, mark_price: float) -> None:
        """Recalculate unrealized PnL based on current mark price."""
        if state.inventory_qty <= 0 or state.inventory_avg_cost <= 0:
            state.unrealized_pnl = 0.0
            return
        state.unrealized_pnl = (mark_price - state.inventory_avg_cost) * state.inventory_qty
        state.mark_price = mark_price

    # ======================================================================
    #  Grid-specific metrics (Section 10.2)
    # ======================================================================

    def compute_grid_metrics(
        self,
        state: InstrumentState,
        cycles: List[GridCycle],
        start_time_ms: int,
        order_latencies_ms: List[float],
    ) -> dict:
        """Compute all Section 10.2 strategy-specific metrics."""
        now_ms = int(time.time() * 1000)
        elapsed_ms = max(now_ms - start_time_ms, 1)
        elapsed_hours = elapsed_ms / 3_600_000
        elapsed_days = elapsed_ms / 86_400_000

        # Grid Cycle Count
        total_cycles = len(cycles)

        # Average Profit Per Cycle
        avg_profit = (sum(c.net_profit for c in cycles) / total_cycles) if total_cycles > 0 else 0.0

        # Grid Cycle Frequency
        cycles_per_hour = total_cycles / elapsed_hours if elapsed_hours > 0 else 0.0
        cycles_per_day = total_cycles / elapsed_days if elapsed_days > 0 else 0.0

        # Grid Utilization Rate
        utilized_levels = set()
        for c in cycles:
            utilized_levels.add(c.buy_level)
            utilized_levels.add(c.sell_level)
        total_levels = state.grid_params.num_levels if state.grid_params else 1
        utilization_rate = len(utilized_levels) / total_levels * 100.0 if total_levels > 0 else 0.0

        # Maximum Concurrent Inventory
        max_inventory = state.inventory_levels_filled  # Current max; tracked over time in manager

        # Fee Efficiency
        total_fees = sum(c.fees for c in cycles)
        theoretical_maker_fees = sum(
            c.buy_price * c.quantity * self.maker_fee + c.sell_price * c.quantity * self.maker_fee
            for c in cycles
        )
        fee_efficiency = (
            theoretical_maker_fees / total_fees * 100.0 if total_fees > 0 else 100.0
        )

        # Order Placement Latency
        avg_latency = (sum(order_latencies_ms) / len(order_latencies_ms)) if order_latencies_ms else 0.0
        max_latency = max(order_latencies_ms) if order_latencies_ms else 0.0

        # Range Containment Rate (approximation from 1m closes)
        containment = 100.0
        if state.grid_params:
            buf_1m = self.get_buffer(state.symbol, "1m")
            if len(buf_1m) > 0:
                closes = buf_1m.get_closes()
                in_range = sum(
                    1 for c in closes
                    if state.grid_params.lower_boundary <= c <= state.grid_params.upper_boundary
                )
                containment = in_range / len(closes) * 100.0

        return {
            "grid_cycle_count": total_cycles,
            "average_profit_per_cycle": round(avg_profit, 6),
            "grid_cycle_frequency": {
                "per_hour": round(cycles_per_hour, 4),
                "per_day": round(cycles_per_day, 4),
            },
            "grid_utilization_rate_pct": round(utilization_rate, 2),
            "inventory_exposure_time_pct": 0.0,  # Tracked over time in manager
            "max_concurrent_inventory": max_inventory,
            "range_containment_rate_pct": round(containment, 2),
            "breakout_count": 0,  # Tracked in manager
            "redeployment_frequency": 0,  # Tracked in manager
            "upside_breakout_pnl": 0.0,  # Tracked in manager
            "downside_breakout_pnl": 0.0,  # Tracked in manager
            "fee_efficiency_pct": round(fee_efficiency, 2),
            "order_placement_latency_ms": {
                "avg": round(avg_latency, 2),
                "max": round(max_latency, 2),
            },
            "total_cycle_profit": round(state.total_cycle_profit, 6),
            "total_fees": round(state.total_fees, 6),
            "current_inventory_qty": state.inventory_qty,
            "current_inventory_avg_cost": round(state.inventory_avg_cost, 4),
            "current_unrealized_pnl": round(state.unrealized_pnl, 6),
            "inventory_levels_filled": state.inventory_levels_filled,
        }
