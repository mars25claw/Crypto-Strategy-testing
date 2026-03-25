"""STRAT-005 Grid Manager — Order lifecycle, state management, deployment.

Handles:
- Grid deployment via batchOrders (5 per call)
- Fill-triggered order placement (< 500ms latency target)
- Grid reset procedure (cancel all, recalculate, redeploy)
- Order monitoring (every 60s, verify all orders still active)
- Profit extraction at 5% of allocated capital
- Price gap handling (batch fills)
- Binance 40 order per instrument limit
- Paper trading fill simulation
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from shared.binance_client import BinanceClient, BinanceClientError
from shared.paper_trading import PaperTradingEngine
from shared.utils import ExchangeInfo

from .strategy import (
    BreakoutDirection,
    DownsideOption,
    GridCycle,
    GridLevel,
    GridParameters,
    GridSide,
    GridStrategy,
    InstrumentState,
)

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")

# Maximum open orders per instrument (Section 11.5)
MAX_ORDERS_PER_INSTRUMENT = 40
BATCH_SIZE = 5  # Binance batch order limit


class GridManager:
    """Manages grid order lifecycle for all instruments.

    Parameters
    ----------
    strategy : GridStrategy
        Core strategy logic instance.
    binance_client : BinanceClient
        REST client for order management.
    exchange_info : ExchangeInfo
        For price/qty rounding and min notional checks.
    paper_engine : PaperTradingEngine or None
        If not None, all orders are simulated.
    config : dict
        Strategy configuration parameters.
    """

    def __init__(
        self,
        strategy: GridStrategy,
        binance_client: BinanceClient,
        exchange_info: ExchangeInfo,
        paper_engine: Optional[PaperTradingEngine],
        config: dict,
    ) -> None:
        self._strategy = strategy
        self._client = binance_client
        self._exchange_info = exchange_info
        self._paper = paper_engine
        self._cfg = config
        self._is_paper = paper_engine is not None

        # Per-instrument state
        self.instruments: Dict[str, InstrumentState] = {}

        # Completed cycles log
        self.cycles: Dict[str, List[GridCycle]] = {}

        # Order ID -> (symbol, level_index) mapping for fast lookup
        self._order_map: Dict[str, Tuple[str, int]] = {}

        # Paper trading: simulated open orders
        self._paper_orders: Dict[str, dict] = {}

        # Latency tracking for metrics
        self._order_latencies: Deque[float] = deque(maxlen=1000)

        # Tracking counters
        self._breakout_count: int = 0
        self._redeployment_count: int = 0
        self._upside_breakout_pnl: float = 0.0
        self._downside_breakout_pnl: float = 0.0
        self._max_concurrent_inventory: int = 0
        self._start_time_ms: int = int(time.time() * 1000)

        # Order cancel rate tracking (Section 8.3)
        self._orders_placed: int = 0
        self._orders_cancelled_unexpectedly: int = 0

        # Consecutive loss tracking (Section 7.6)
        self._consecutive_loss_cycles: int = 0

        self._max_orders = config.get("max_orders_per_instrument", MAX_ORDERS_PER_INSTRUMENT)

    # ======================================================================
    #  Initialization
    # ======================================================================

    def init_instrument(self, symbol: str, allocated_capital: float) -> InstrumentState:
        """Initialize tracking state for an instrument."""
        state = InstrumentState(symbol=symbol, allocated_capital=allocated_capital)
        self.instruments[symbol] = state
        self.cycles[symbol] = []
        return state

    # ======================================================================
    #  Grid Deployment (Section 3.3, Step 8)
    # ======================================================================

    async def deploy_grid(self, symbol: str) -> bool:
        """Deploy the full grid for an instrument.

        Places buy orders below current price in batches of 5.
        Returns True on success.
        """
        state = self.instruments.get(symbol)
        if not state or not state.grid_params:
            logger.error("[%s] Cannot deploy: no grid parameters", symbol)
            return False

        params = state.grid_params
        current_price = state.current_price
        if current_price <= 0:
            logger.error("[%s] Cannot deploy: no current price", symbol)
            return False

        # Assign sides based on current price
        self._strategy.assign_order_sides(params, current_price)

        # Collect orders to place (only BUY orders in neutral mode)
        orders_to_place: List[Tuple[int, GridLevel]] = []
        for level in params.levels:
            if level.side == GridSide.BUY and not level.filled and not level.active:
                if level.quantity > 0 and level.price > 0:
                    orders_to_place.append((level.index, level))
            elif level.side == GridSide.SELL and not level.filled and not level.active:
                if level.quantity > 0 and level.price > 0:
                    orders_to_place.append((level.index, level))

        if not orders_to_place:
            logger.info("[%s] No orders to deploy", symbol)
            return True

        # Enforce max orders limit
        if len(orders_to_place) > self._max_orders:
            # Prioritize orders closest to current price
            orders_to_place.sort(key=lambda x: abs(x[1].price - current_price))
            orders_to_place = orders_to_place[:self._max_orders]

        # Place in batches of 5
        total_placed = 0
        for i in range(0, len(orders_to_place), BATCH_SIZE):
            batch = orders_to_place[i:i + BATCH_SIZE]
            placed = await self._place_batch_orders(symbol, batch)
            total_placed += placed

        state.active = True
        logger.info("[%s] Grid deployed: %d/%d orders placed", symbol, total_placed, len(orders_to_place))
        return total_placed > 0

    async def _place_batch_orders(
        self, symbol: str, batch: List[Tuple[int, GridLevel]]
    ) -> int:
        """Place a batch of orders (up to 5). Returns count placed."""
        if not batch:
            return 0

        state = self.instruments[symbol]

        if self._is_paper:
            return self._place_paper_orders(symbol, batch)

        # Build batch order list
        order_dicts = []
        for level_idx, level in batch:
            order_dicts.append({
                "symbol": symbol,
                "side": level.side.value,
                "type": "LIMIT",
                "quantity": level.quantity,
                "price": level.price,
                "time_in_force": "GTX",  # Post-only / maker
            })

        try:
            t0 = time.monotonic()
            results = await self._client.place_batch_futures_orders(order_dicts)
            latency_ms = (time.monotonic() - t0) * 1000
            self._order_latencies.append(latency_ms)

            placed = 0
            for (level_idx, level), result in zip(batch, results):
                if isinstance(result, dict) and "orderId" in result:
                    order_id = str(result["orderId"])
                    level.order_id = order_id
                    level.active = True
                    self._order_map[order_id] = (symbol, level_idx)
                    self._orders_placed += 1
                    placed += 1
                else:
                    err = result.get("msg", str(result)) if isinstance(result, dict) else str(result)
                    logger.error("[%s] Batch order failed for level %d: %s", symbol, level_idx, err)

            return placed

        except BinanceClientError as e:
            logger.error("[%s] Batch order error: %s", symbol, e)
            return 0
        except Exception as e:
            logger.error("[%s] Unexpected batch order error: %s", symbol, e, exc_info=True)
            return 0

    def _place_paper_orders(self, symbol: str, batch: List[Tuple[int, GridLevel]]) -> int:
        """Simulate order placement for paper trading."""
        placed = 0
        for level_idx, level in batch:
            order_id = f"PAPER-{uuid.uuid4().hex[:12]}"
            level.order_id = order_id
            level.active = True
            self._order_map[order_id] = (symbol, level_idx)
            self._paper_orders[order_id] = {
                "orderId": order_id,
                "symbol": symbol,
                "side": level.side.value,
                "type": "LIMIT",
                "price": level.price,
                "origQty": level.quantity,
                "status": "NEW",
                "level_index": level_idx,
                "time_in_force": "GTX",
                "created_ms": int(time.time() * 1000),
            }
            self._orders_placed += 1
            placed += 1
        return placed

    # ======================================================================
    #  Fill-Triggered Order Placement (Section 3.4)
    # ======================================================================

    async def handle_fill(
        self, symbol: str, order_id: str, side: str, fill_price: float,
        fill_qty: float, fill_time_ms: int,
    ) -> Optional[GridCycle]:
        """Handle an order fill event (from USER DATA STREAM).

        Must place the counter-order within 500ms.
        Returns a GridCycle if a buy-sell cycle was completed.
        """
        t0 = time.monotonic()

        lookup = self._order_map.get(order_id)
        if not lookup:
            logger.debug("[%s] Fill for unknown order %s (not ours)", symbol, order_id)
            return None

        sym, level_idx = lookup
        if sym != symbol:
            logger.warning("Order %s symbol mismatch: expected %s got %s", order_id, sym, symbol)
            return None

        state = self.instruments.get(symbol)
        if not state or not state.grid_params:
            logger.error("[%s] Fill received but no grid state", symbol)
            return None

        cycle = None

        if side.upper() == "BUY":
            result = self._strategy.on_buy_fill(state, level_idx, fill_price, fill_qty, fill_time_ms)
            if result:
                target_idx, target_price, target_qty, target_side = result
                await self._place_counter_order(symbol, state, target_idx, target_price, target_qty, target_side)

        elif side.upper() == "SELL":
            result = self._strategy.on_sell_fill(state, level_idx, fill_price, fill_qty, fill_time_ms)
            if result:
                target_idx, target_price, target_qty, target_side, cycle = result
                await self._place_counter_order(symbol, state, target_idx, target_price, target_qty, target_side)

                # Track cycle
                if cycle:
                    self.cycles.setdefault(symbol, []).append(cycle)
                    # Consecutive loss tracking
                    if cycle.net_profit < 0:
                        self._consecutive_loss_cycles += 1
                    else:
                        self._consecutive_loss_cycles = 0

        # Remove from order map
        self._order_map.pop(order_id, None)

        # Track max inventory
        if state.inventory_levels_filled > self._max_concurrent_inventory:
            self._max_concurrent_inventory = state.inventory_levels_filled

        # Latency tracking
        latency_ms = (time.monotonic() - t0) * 1000
        self._order_latencies.append(latency_ms)
        if latency_ms > 500:
            logger.warning(
                "[%s] Fill handler latency %.1fms > 500ms target", symbol, latency_ms,
            )

        return cycle

    async def _place_counter_order(
        self, symbol: str, state: InstrumentState,
        level_idx: int, price: float, qty: float, side: GridSide,
    ) -> bool:
        """Place a single counter-order at the specified grid level."""
        if not state.grid_params or level_idx < 0 or level_idx >= len(state.grid_params.levels):
            return False

        level = state.grid_params.levels[level_idx]

        # Check inventory limit before placing buy orders
        if side == GridSide.BUY:
            halted, reason = self._strategy.check_inventory_exposure(state)
            if halted:
                logger.warning("[%s] Inventory limit hit, skipping buy at level %d: %s", symbol, level_idx, reason)
                return False

        # Round price and quantity
        if self._exchange_info:
            price = self._exchange_info.round_price(symbol, price)
            qty = self._exchange_info.round_quantity(symbol, qty)

        if qty <= 0 or price <= 0:
            return False

        if self._is_paper:
            order_id = f"PAPER-{uuid.uuid4().hex[:12]}"
            level.order_id = order_id
            level.side = side
            level.quantity = qty
            level.active = True
            level.filled = False
            self._order_map[order_id] = (symbol, level_idx)
            self._paper_orders[order_id] = {
                "orderId": order_id,
                "symbol": symbol,
                "side": side.value,
                "type": "LIMIT",
                "price": price,
                "origQty": qty,
                "status": "NEW",
                "level_index": level_idx,
                "time_in_force": "GTX",
                "created_ms": int(time.time() * 1000),
            }
            self._orders_placed += 1
            return True

        try:
            result = await self._client.place_futures_order(
                symbol=symbol,
                side=side.value,
                type="LIMIT",
                quantity=qty,
                price=price,
                post_only=True,
            )
            order_id = str(result.get("orderId", ""))
            level.order_id = order_id
            level.side = side
            level.quantity = qty
            level.active = True
            level.filled = False
            self._order_map[order_id] = (symbol, level_idx)
            self._orders_placed += 1
            return True

        except BinanceClientError as e:
            logger.error("[%s] Counter-order failed at level %d: %s", symbol, level_idx, e)
            return False

    # ======================================================================
    #  Grid Reset (Section 4.3)
    # ======================================================================

    async def reset_grid(self, symbol: str, reason: str) -> bool:
        """Full grid reset: cancel all, evaluate inventory, redeploy.

        Section 4.3: Cancel ALL -> evaluate inventory -> redeploy.
        """
        state = self.instruments.get(symbol)
        if not state:
            return False

        logger.warning("[%s] GRID RESET: %s", symbol, reason)
        trade_logger.info("GRID_RESET\t%s\treason=%s", symbol, reason)
        self._redeployment_count += 1

        # Step 1: Cancel all open orders
        await self.cancel_all_orders(symbol)

        # Step 2: Calculate current inventory value
        if state.inventory_qty > 0 and state.mark_price > 0:
            self._strategy.update_unrealized_pnl(state, state.mark_price)
            logger.info(
                "[%s] Inventory after reset: qty=%.8f avg_cost=%.4f unrealized=%.4f",
                symbol, state.inventory_qty, state.inventory_avg_cost, state.unrealized_pnl,
            )

        # Step 3: Recalculate range
        current_price = state.current_price or state.mark_price
        if current_price <= 0:
            logger.error("[%s] Cannot reset: no current price", symbol)
            return False

        range_result = self._strategy.calculate_range(symbol, current_price)
        if range_result is None:
            logger.error("[%s] Cannot calculate new range after reset", symbol)
            state.active = False
            return False

        upper, lower, width = range_result

        # Step 4: Calculate new grid parameters
        params = self._strategy.calculate_grid_parameters(
            symbol, current_price, upper, lower, width, state.allocated_capital,
        )
        if params is None:
            logger.error("[%s] Cannot calculate grid params after reset", symbol)
            state.active = False
            return False

        state.grid_params = params
        state.consecutive_buy_fills = 0
        state.consecutive_sell_fills = 0
        state.breakout_direction = BreakoutDirection.NONE

        # Step 5: Redeploy
        return await self.deploy_grid(symbol)

    # ======================================================================
    #  Cancel Orders
    # ======================================================================

    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all open orders for an instrument. Returns count cancelled."""
        state = self.instruments.get(symbol)
        if not state or not state.grid_params:
            return 0

        if self._is_paper:
            count = 0
            to_remove = []
            for oid, order in self._paper_orders.items():
                if order["symbol"] == symbol and order["status"] == "NEW":
                    order["status"] = "CANCELED"
                    to_remove.append(oid)
                    count += 1
            for oid in to_remove:
                self._order_map.pop(oid, None)
                self._paper_orders.pop(oid, None)

            for level in state.grid_params.levels:
                if level.active and not level.filled:
                    level.active = False
                    level.order_id = None

            logger.info("[%s] Paper: cancelled %d orders", symbol, count)
            return count

        try:
            await self._client.cancel_all_futures_orders(symbol)
            count = 0
            for level in state.grid_params.levels:
                if level.active and not level.filled:
                    level.active = False
                    if level.order_id:
                        self._order_map.pop(level.order_id, None)
                    level.order_id = None
                    count += 1
            logger.info("[%s] Cancelled all orders (%d tracked levels)", symbol, count)
            return count

        except BinanceClientError as e:
            logger.error("[%s] Cancel all orders failed: %s", symbol, e)
            return 0

    async def cancel_buy_orders(self, symbol: str) -> int:
        """Cancel only buy orders (used in upside breakout and unrealized loss halt)."""
        state = self.instruments.get(symbol)
        if not state or not state.grid_params:
            return 0

        count = 0
        for level in state.grid_params.levels:
            if level.active and level.side == GridSide.BUY and not level.filled:
                if self._is_paper:
                    if level.order_id and level.order_id in self._paper_orders:
                        self._paper_orders[level.order_id]["status"] = "CANCELED"
                        del self._paper_orders[level.order_id]
                    self._order_map.pop(level.order_id, None)
                    level.active = False
                    level.order_id = None
                    count += 1
                else:
                    try:
                        if level.order_id:
                            await self._client.cancel_futures_order(symbol, int(level.order_id))
                            self._order_map.pop(level.order_id, None)
                            level.active = False
                            level.order_id = None
                            count += 1
                    except BinanceClientError:
                        pass

        logger.info("[%s] Cancelled %d buy orders", symbol, count)
        return count

    # ======================================================================
    #  Breakout Handling (Section 4.1 / 4.2)
    # ======================================================================

    async def handle_upside_breakout(self, symbol: str) -> None:
        """Section 4.1: Cancel buys, let sells fill, then redeploy."""
        state = self.instruments.get(symbol)
        if not state:
            return

        state.breakout_direction = BreakoutDirection.UPSIDE
        self._breakout_count += 1

        trade_logger.info("BREAKOUT_UPSIDE\t%s\tprice=%.4f", symbol, state.current_price)
        logger.warning("[%s] Upside breakout — cancelling buy orders", symbol)

        # Step 1: Cancel all buy orders
        await self.cancel_buy_orders(symbol)

        # Steps 2-5 are handled asynchronously:
        # The sell orders will fill as price moves up.
        # After 4h or all sells fill, redeploy is triggered in the main loop.

    async def handle_downside_breakout(self, symbol: str) -> None:
        """Section 4.2: Handle downside breakout based on configured option."""
        state = self.instruments.get(symbol)
        if not state:
            return

        state.breakout_direction = BreakoutDirection.DOWNSIDE
        self._breakout_count += 1

        trade_logger.info("BREAKOUT_DOWNSIDE\t%s\tprice=%.4f\toption=%s",
                          symbol, state.current_price, self._strategy.downside_option.value)

        if self._strategy.downside_option == DownsideOption.LIQUIDATE:
            # Option B: Close all inventory at market, accept loss, redeploy
            logger.warning("[%s] Downside breakout Option B — liquidating all", symbol)
            await self._liquidate_inventory(symbol)
            await asyncio.sleep(2.0)
            await self.reset_grid(symbol, "downside_breakout_option_B")
        else:
            # Option A (DEFAULT): Hold inventory, widen grid downward
            logger.warning("[%s] Downside breakout Option A — hold + widen", symbol)
            # Cancel sell orders (if any), keep inventory
            # Set hard stop at 1.5 * ATR below lower boundary
            if state.grid_params:
                buf_1d = self._strategy.get_buffer(symbol, "1d")
                if len(buf_1d) > 15:
                    from shared.indicators import atr as calc_atr
                    atr_values = calc_atr(buf_1d.get_highs(), buf_1d.get_lows(), buf_1d.get_closes(), 14)
                    daily_atr = float(atr_values[-1]) if not __import__('numpy').isnan(atr_values[-1]) else 0
                    hard_stop = state.grid_params.lower_boundary - (self._strategy.hard_stop_atr_mult * daily_atr)
                    state.halt_reason = f"downside_breakout_hard_stop_{hard_stop:.4f}"
                    logger.info("[%s] Hard stop set at %.4f (1.5 ATR below lower)", symbol, hard_stop)

    async def _liquidate_inventory(self, symbol: str) -> None:
        """Liquidate all inventory at market."""
        state = self.instruments.get(symbol)
        if not state or state.inventory_qty <= 0:
            return

        # Cancel all grid orders first
        await self.cancel_all_orders(symbol)

        qty = state.inventory_qty
        if self._exchange_info:
            qty = self._exchange_info.round_quantity(symbol, qty)

        if qty <= 0:
            return

        if self._is_paper:
            # Simulate market sell
            pnl = (state.current_price - state.inventory_avg_cost) * qty
            fee = state.current_price * qty * self._strategy.taker_fee
            state.realized_profit += pnl - fee
            state.total_fees += fee
            state.inventory_qty = 0.0
            state.inventory_avg_cost = 0.0
            state.inventory_levels_filled = 0
            state.unrealized_pnl = 0.0

            if self._paper:
                self._paper.update_equity(pnl - fee)

            trade_logger.info(
                "LIQUIDATE\t%s\tqty=%.8f\tprice=%.4f\tpnl=%.6f\tfee=%.6f",
                symbol, qty, state.current_price, pnl, fee,
            )
        else:
            try:
                await self._client.place_futures_order(
                    symbol=symbol,
                    side="SELL",
                    type="MARKET",
                    quantity=qty,
                    reduce_only=True,
                )
                state.inventory_qty = 0.0
                state.inventory_avg_cost = 0.0
                state.inventory_levels_filled = 0
                state.unrealized_pnl = 0.0

                trade_logger.info("LIQUIDATE\t%s\tqty=%.8f\tmarket_order", symbol, qty)
            except BinanceClientError as e:
                logger.error("[%s] Liquidation failed: %s", symbol, e)

    # ======================================================================
    #  Profit Extraction (Section 4.4)
    # ======================================================================

    async def extract_profit(self, symbol: str) -> float:
        """Extract realized profit when it reaches 5% of allocated capital."""
        state = self.instruments.get(symbol)
        if not state:
            return 0.0

        extracted = state.realized_profit
        state.realized_profit = 0.0

        trade_logger.info("PROFIT_EXTRACT\t%s\tamount=%.6f", symbol, extracted)
        logger.info("[%s] Profit extracted: %.6f USDT", symbol, extracted)
        return extracted

    # ======================================================================
    #  Order Monitoring (Section 8.3)
    # ======================================================================

    async def monitor_orders(self, symbol: str) -> int:
        """Verify all grid orders are still active on exchange. Returns count missing.

        Section 8.3: Every 60 seconds, verify all orders still active.
        Replace any missing orders immediately.
        """
        state = self.instruments.get(symbol)
        if not state or not state.grid_params or not state.active:
            return 0

        if self._is_paper:
            return self._monitor_paper_orders(symbol)

        try:
            exchange_orders = await self._client.get_futures_open_orders(symbol)
        except BinanceClientError as e:
            logger.error("[%s] Failed to fetch open orders: %s", symbol, e)
            return -1

        exchange_order_ids = {str(o["orderId"]) for o in exchange_orders}
        missing = []

        for level in state.grid_params.levels:
            if level.active and level.order_id and not level.filled:
                if level.order_id not in exchange_order_ids:
                    logger.warning(
                        "[%s] Order %s missing from exchange (level %d, %s at %.4f)",
                        symbol, level.order_id, level.index,
                        level.side.value if level.side else "?", level.price,
                    )
                    level.active = False
                    self._order_map.pop(level.order_id, None)
                    level.order_id = None
                    self._orders_cancelled_unexpectedly += 1
                    missing.append((level.index, level))

        # Replace missing orders
        if missing:
            cancel_rate = self._orders_cancelled_unexpectedly / max(self._orders_placed, 1) * 100.0
            if cancel_rate > 5.0:
                logger.error(
                    "[%s] Order cancel rate %.1f%% > 5%% — investigate",
                    symbol, cancel_rate,
                )

            # Place replacement orders in batches
            for i in range(0, len(missing), BATCH_SIZE):
                batch = missing[i:i + BATCH_SIZE]
                await self._place_batch_orders(symbol, batch)

            logger.info("[%s] Replaced %d missing orders", symbol, len(missing))

        return len(missing)

    def _monitor_paper_orders(self, symbol: str) -> int:
        """Paper trading order monitoring (no-op, orders are always valid)."""
        return 0

    # ======================================================================
    #  Paper Trading: Fill Simulation (Section 9.1)
    # ======================================================================

    async def simulate_paper_fills(
        self, symbol: str, current_price: float, agg_volume: float = 0.0,
    ) -> List[str]:
        """Check paper orders for fills based on current price.

        Section 9.1: BUY fills when price trades AT OR BELOW.
                     SELL fills when price trades AT OR ABOVE.

        Returns list of filled order IDs.
        """
        if not self._is_paper:
            return []

        state = self.instruments.get(symbol)
        if not state or not state.grid_params:
            return []

        filled_ids = []
        now_ms = int(time.time() * 1000)

        # Collect fillable orders
        for oid, order in list(self._paper_orders.items()):
            if order["symbol"] != symbol or order["status"] != "NEW":
                continue

            price = order["price"]
            qty = order["origQty"]
            side = order["side"]

            filled = False

            if side == "BUY" and current_price <= price:
                filled = True
            elif side == "SELL" and current_price >= price:
                filled = True

            if filled:
                # Apply maker fee
                fee = price * qty * self._strategy.maker_fee
                if self._paper:
                    self._paper.update_equity(-fee)

                order["status"] = "FILLED"
                filled_ids.append(oid)

                # Process fill through the handler
                await self.handle_fill(
                    symbol=symbol,
                    order_id=oid,
                    side=side,
                    fill_price=price,
                    fill_qty=qty,
                    fill_time_ms=now_ms,
                )

                # Clean up paper order
                self._paper_orders.pop(oid, None)

        return filled_ids

    # ======================================================================
    #  State Serialization
    # ======================================================================

    def get_state_dict(self) -> dict:
        """Serialize the full grid manager state for persistence."""
        instruments_dict = {}
        for sym, state in self.instruments.items():
            instruments_dict[sym] = state.to_dict()

        cycles_dict = {}
        for sym, cycle_list in self.cycles.items():
            cycles_dict[sym] = [c.to_dict() for c in cycle_list[-500:]]  # Keep last 500

        return {
            "instruments": instruments_dict,
            "cycles": cycles_dict,
            "order_map": dict(self._order_map),
            "breakout_count": self._breakout_count,
            "redeployment_count": self._redeployment_count,
            "upside_breakout_pnl": self._upside_breakout_pnl,
            "downside_breakout_pnl": self._downside_breakout_pnl,
            "max_concurrent_inventory": self._max_concurrent_inventory,
            "start_time_ms": self._start_time_ms,
            "orders_placed": self._orders_placed,
            "orders_cancelled_unexpectedly": self._orders_cancelled_unexpectedly,
            "consecutive_loss_cycles": self._consecutive_loss_cycles,
        }

    def load_state_dict(self, data: dict) -> None:
        """Restore state from a persisted dict."""
        for sym, state_dict in data.get("instruments", {}).items():
            self.instruments[sym] = InstrumentState.from_dict(state_dict)

        for sym, cycle_dicts in data.get("cycles", {}).items():
            self.cycles[sym] = []
            for cd in cycle_dicts:
                self.cycles[sym].append(GridCycle(**cd))

        self._order_map = {}
        for oid, (sym, idx) in data.get("order_map", {}).items():
            self._order_map[oid] = (sym, idx)

        self._breakout_count = data.get("breakout_count", 0)
        self._redeployment_count = data.get("redeployment_count", 0)
        self._upside_breakout_pnl = data.get("upside_breakout_pnl", 0.0)
        self._downside_breakout_pnl = data.get("downside_breakout_pnl", 0.0)
        self._max_concurrent_inventory = data.get("max_concurrent_inventory", 0)
        self._start_time_ms = data.get("start_time_ms", int(time.time() * 1000))
        self._orders_placed = data.get("orders_placed", 0)
        self._orders_cancelled_unexpectedly = data.get("orders_cancelled_unexpectedly", 0)
        self._consecutive_loss_cycles = data.get("consecutive_loss_cycles", 0)

    # ======================================================================
    #  Metrics
    # ======================================================================

    def get_grid_metrics(self, symbol: str) -> dict:
        """Get Section 10.2 metrics for a specific instrument."""
        state = self.instruments.get(symbol)
        if not state:
            return {}

        cycles = self.cycles.get(symbol, [])
        return self._strategy.compute_grid_metrics(
            state, cycles, self._start_time_ms, list(self._order_latencies),
        )

    def get_all_metrics(self) -> dict:
        """Get aggregated metrics across all instruments."""
        all_metrics = {}
        for symbol in self.instruments:
            all_metrics[symbol] = self.get_grid_metrics(symbol)

        # Aggregate
        total_cycles = sum(m.get("grid_cycle_count", 0) for m in all_metrics.values())
        total_profit = sum(m.get("total_cycle_profit", 0.0) for m in all_metrics.values())
        total_fees = sum(m.get("total_fees", 0.0) for m in all_metrics.values())

        return {
            "per_instrument": all_metrics,
            "aggregate": {
                "total_cycles": total_cycles,
                "total_cycle_profit": round(total_profit, 6),
                "total_fees": round(total_fees, 6),
                "breakout_count": self._breakout_count,
                "redeployment_count": self._redeployment_count,
                "max_concurrent_inventory": self._max_concurrent_inventory,
                "orders_placed": self._orders_placed,
                "orders_cancelled_unexpectedly": self._orders_cancelled_unexpectedly,
            },
        }
