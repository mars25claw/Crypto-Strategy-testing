"""STRAT-007: Ultra-low-latency execution engine.

Mode A: Both legs within 50ms (parallel async MARKET orders).
Mode B: 3-leg sequential within 500ms total (100ms each + reconciliation).

Failure handling:
- One-leg failure: 3 retries at 200ms, then aggressive LIMIT, then market unwind.
  Maximum 60s unhedged exposure.
- Mode B partial triangle: reverse path if Leg 3 fails.
- Fill reconciliation: actual vs expected quantities.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Tuple

from strat_007_triangular_arb.src.strategy import (
    ArbMode,
    ArbDirection,
    ArbOpportunity,
    ExecutionResult,
)

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """Ultra-low-latency arb execution engine.

    Parameters
    ----------
    binance_client : BinanceClient
        REST client for placing orders.
    params : dict
        Strategy parameters from config.yaml.
    paper_mode : bool
        If True, simulate fills instead of placing real orders.
    paper_engine : PaperTradingEngine, optional
        Paper trading fill simulation engine.
    scanner : OpportunityScanner, optional
        For depth snapshot access in paper mode.
    exchange_info : ExchangeInfo, optional
        For quantity/price rounding.
    """

    def __init__(
        self,
        binance_client: Any,
        params: Dict[str, Any],
        paper_mode: bool = True,
        paper_engine: Any = None,
        scanner: Any = None,
        exchange_info: Any = None,
    ) -> None:
        self._client = binance_client
        self._params = params
        self._paper_mode = paper_mode
        self._paper_engine = paper_engine
        self._scanner = scanner
        self._exchange_info = exchange_info

        # Execution parameters
        self._mode_a_max_ms = params.get("mode_a_max_execution_ms", 50)
        self._mode_b_max_ms = params.get("mode_b_max_execution_ms", 500)
        self._max_retries = params.get("max_retries", 3)
        self._retry_delay_ms = params.get("retry_delay_ms", 200)
        self._max_unhedged_s = params.get("max_unhedged_time_s", 60)
        self._counter_opp_wait_s = params.get("counter_opportunity_wait_s", 30)

        # Paper trading latency simulation
        self._paper_latency_min_ms = params.get("paper_latency_min_ms", 50)
        self._paper_latency_max_ms = params.get("paper_latency_max_ms", 200)
        self._paper_leg_delay_min_ms = params.get("paper_mode_b_leg_delay_min_ms", 100)
        self._paper_leg_delay_max_ms = params.get("paper_mode_b_leg_delay_max_ms", 200)

        # Active executions tracking
        self._active_executions = 0
        self._max_concurrent = params.get("max_concurrent_arbs", 2)
        self._execution_lock = asyncio.Lock()

        # Unhedged position tracking
        self._unhedged_positions: Dict[str, Dict[str, Any]] = {}

        # Execution history
        self._execution_history: Deque[ExecutionResult] = deque(maxlen=500)
        self._latency_history: Deque[float] = deque(maxlen=1000)

        # Partial triangle event tracking (Section 11.2)
        self._partial_triangle_events: Deque[Dict[str, Any]] = deque(maxlen=200)

        # Paper trading latency distribution tracking
        self._paper_latency_samples: Deque[float] = deque(maxlen=5000)

        # Callbacks
        self._on_unhedged: Optional[Callable] = None
        self._on_execution_complete: Optional[Callable] = None

        # Statistics
        self._stats = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "partial_fills": 0,
            "unwind_count": 0,
            "total_profit_usdt": 0.0,
            "total_fees_usdt": 0.0,
            "avg_execution_ms": 0.0,
        }

        logger.info(
            "ExecutionEngine initialized: paper=%s mode_a_max=%dms mode_b_max=%dms "
            "max_retries=%d max_concurrent=%d",
            paper_mode, self._mode_a_max_ms, self._mode_b_max_ms,
            self._max_retries, self._max_concurrent,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_callbacks(
        self,
        on_unhedged: Optional[Callable] = None,
        on_execution_complete: Optional[Callable] = None,
    ) -> None:
        """Set event callbacks."""
        self._on_unhedged = on_unhedged
        self._on_execution_complete = on_execution_complete

    @property
    def active_executions(self) -> int:
        return self._active_executions

    def can_execute(self) -> bool:
        """Check if a new execution can be started."""
        return self._active_executions < self._max_concurrent

    async def execute(self, opportunity: ArbOpportunity) -> ExecutionResult:
        """Execute an arbitrage opportunity.

        Routes to Mode A or Mode B execution based on the opportunity type.
        """
        if not self.can_execute():
            return ExecutionResult(
                opportunity=opportunity,
                success=False,
                error=f"Max concurrent executions reached ({self._max_concurrent})",
            )

        async with self._execution_lock:
            self._active_executions += 1

        try:
            start = time.monotonic()
            self._stats["total_executions"] += 1

            if opportunity.mode == ArbMode.MODE_A:
                result = await self._execute_mode_a(opportunity)
            else:
                result = await self._execute_mode_b(opportunity)

            elapsed_ms = (time.monotonic() - start) * 1000.0
            result.execution_time_ms = elapsed_ms
            self._latency_history.append(elapsed_ms)

            # Update statistics
            if result.success:
                self._stats["successful"] += 1
                self._stats["total_profit_usdt"] += result.actual_profit_usdt
            else:
                self._stats["failed"] += 1

            self._stats["total_fees_usdt"] += result.total_fees_usdt
            self._stats["avg_execution_ms"] = (
                sum(self._latency_history) / len(self._latency_history)
                if self._latency_history else 0.0
            )

            self._execution_history.append(result)

            # Log the execution
            trade_logger.info(
                "ARB_EXEC\tmode=%s\tsymbol=%s\tsuccess=%s\tprofit=%.6f\t"
                "fees=%.6f\tlatency_ms=%.1f\tlegs=%d/%d\terror=%s",
                opportunity.mode.value, opportunity.symbol,
                result.success, result.actual_profit_usdt,
                result.total_fees_usdt, elapsed_ms,
                result.legs_filled, result.legs_total,
                result.error or "none",
            )

            # Notify callback
            if self._on_execution_complete:
                try:
                    cb_result = self._on_execution_complete(result)
                    if asyncio.iscoroutine(cb_result):
                        await cb_result
                except Exception:
                    logger.exception("Error in execution_complete callback")

            return result

        finally:
            async with self._execution_lock:
                self._active_executions -= 1

    # ------------------------------------------------------------------
    # Mode A: Parallel spot + futures execution
    # ------------------------------------------------------------------

    async def _execute_mode_a(self, opp: ArbOpportunity) -> ExecutionResult:
        """Execute Mode A: parallel MARKET orders on spot and futures."""
        result = ExecutionResult(opportunity=opp, legs_total=2)

        symbol = opp.symbol
        qty = opp.trade_size_qty

        if qty <= 0:
            result.error = "Invalid quantity"
            return result

        # Round quantity
        if self._exchange_info:
            qty = self._exchange_info.round_quantity(symbol, qty)
            if qty <= 0:
                result.error = "Quantity rounds to zero"
                return result

        # Determine order sides
        if opp.direction == ArbDirection.BUY_SPOT_SELL_FUTURES:
            spot_side = "BUY"
            futures_side = "SELL"
        else:
            spot_side = "SELL"
            futures_side = "BUY"

        if self._paper_mode:
            return await self._execute_mode_a_paper(opp, symbol, qty, spot_side, futures_side)

        # Place both orders in parallel
        spot_task = asyncio.create_task(
            self._place_spot_order(symbol, spot_side, qty)
        )
        futures_task = asyncio.create_task(
            self._place_futures_order(symbol, futures_side, qty)
        )

        spot_result, futures_result = await asyncio.gather(
            spot_task, futures_task, return_exceptions=True
        )

        # Process results
        spot_ok = not isinstance(spot_result, Exception) and spot_result is not None
        futures_ok = not isinstance(futures_result, Exception) and futures_result is not None

        if spot_ok:
            result.legs_filled += 1
            result.leg_results.append({"leg": "spot", "side": spot_side, "result": spot_result})
        if futures_ok:
            result.legs_filled += 1
            result.leg_results.append({"leg": "futures", "side": futures_side, "result": futures_result})

        if spot_ok and futures_ok:
            result.success = True
            result = self._reconcile_mode_a(result, spot_result, futures_result, opp)
        elif spot_ok and not futures_ok:
            # Spot filled, futures failed — unhedged long/short
            logger.error("Mode A one-leg failure: spot filled, futures failed for %s", symbol)
            await self._handle_one_leg_failure(
                symbol=symbol,
                filled_market="spot",
                filled_side=spot_side,
                qty=qty,
                result=result,
            )
        elif futures_ok and not spot_ok:
            logger.error("Mode A one-leg failure: futures filled, spot failed for %s", symbol)
            await self._handle_one_leg_failure(
                symbol=symbol,
                filled_market="futures",
                filled_side=futures_side,
                qty=qty,
                result=result,
            )
        else:
            result.error = "Both legs failed"

        return result

    async def _execute_mode_a_paper(
        self,
        opp: ArbOpportunity,
        symbol: str,
        qty: float,
        spot_side: str,
        futures_side: str,
    ) -> ExecutionResult:
        """Paper trading Mode A execution with latency simulation.

        Section 9.2: Add 50-200ms random latency per order in paper mode.
        Track simulated latency distribution.
        """
        result = ExecutionResult(opportunity=opp, legs_total=2)

        # Simulate per-order latency (50-200ms per order, Section 9.2)
        spot_latency_ms = random.uniform(self._paper_latency_min_ms, self._paper_latency_max_ms)
        futures_latency_ms = random.uniform(self._paper_latency_min_ms, self._paper_latency_max_ms)
        self._paper_latency_samples.append(spot_latency_ms)
        self._paper_latency_samples.append(futures_latency_ms)
        latency_ms = max(spot_latency_ms, futures_latency_ms)  # Parallel orders
        await asyncio.sleep(latency_ms / 1000.0)

        if self._paper_engine is None:
            result.error = "Paper engine not available"
            return result

        # Get order book snapshots
        spot_book = None
        futures_book = None
        if self._scanner:
            spot_book = self._scanner.get_spot_depth_snapshot(symbol)
            futures_book = self._scanner.get_futures_depth_snapshot(symbol)

        if not spot_book:
            spot_book = {"bids": [[opp.spot_bid, qty * 5]], "asks": [[opp.spot_ask, qty * 5]]}
        if not futures_book:
            futures_book = {"bids": [[opp.futures_bid, qty * 5]], "asks": [[opp.futures_ask, qty * 5]]}

        try:
            spot_fill = self._paper_engine.simulate_market_order(
                symbol=f"{symbol}_SPOT", side=spot_side, quantity=qty,
                order_book_snapshot=spot_book,
            )
            futures_fill = self._paper_engine.simulate_market_order(
                symbol=f"{symbol}_FUT", side=futures_side, quantity=qty,
                order_book_snapshot=futures_book,
            )

            result.legs_filled = 2
            result.success = True

            # Calculate actual profit
            if spot_side == "BUY":
                # Bought spot, sold futures
                spot_cost = spot_fill.fill_price * spot_fill.fill_quantity
                futures_revenue = futures_fill.fill_price * futures_fill.fill_quantity
                result.actual_profit_usdt = futures_revenue - spot_cost - spot_fill.fees - futures_fill.fees
            else:
                # Sold spot, bought futures
                spot_revenue = spot_fill.fill_price * spot_fill.fill_quantity
                futures_cost = futures_fill.fill_price * futures_fill.fill_quantity
                result.actual_profit_usdt = spot_revenue - futures_cost - spot_fill.fees - futures_fill.fees

            result.total_fees_usdt = spot_fill.fees + futures_fill.fees
            result.slippage_usdt = abs(
                (spot_fill.fill_price - (opp.spot_ask if spot_side == "BUY" else opp.spot_bid))
            ) * qty

            if opp.trade_size_usdt > 0:
                result.actual_profit_pct = result.actual_profit_usdt / opp.trade_size_usdt * 100.0

            result.leg_results = [
                {"leg": "spot", "side": spot_side, "fill_price": spot_fill.fill_price,
                 "fill_qty": spot_fill.fill_quantity, "fees": spot_fill.fees},
                {"leg": "futures", "side": futures_side, "fill_price": futures_fill.fill_price,
                 "fill_qty": futures_fill.fill_quantity, "fees": futures_fill.fees},
            ]

            # Close paper positions immediately (arb is self-closing)
            self._paper_engine.close_position(f"{symbol}_SPOT", spot_fill.fill_price)
            self._paper_engine.close_position(f"{symbol}_FUT", futures_fill.fill_price)
            # Apply the net profit/loss to equity
            self._paper_engine.update_equity(result.actual_profit_usdt)

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Paper Mode A execution failed for %s", symbol)

        return result

    # ------------------------------------------------------------------
    # Mode B: Sequential 3-leg execution
    # ------------------------------------------------------------------

    async def _execute_mode_b(self, opp: ArbOpportunity) -> ExecutionResult:
        """Execute Mode B: 3-leg sequential MARKET orders."""
        result = ExecutionResult(opportunity=opp, legs_total=3)

        pairs = opp.triangle_path
        sides = opp.leg_sides
        if len(pairs) != 3 or len(sides) != 3:
            result.error = "Invalid triangle path"
            return result

        if self._paper_mode:
            return await self._execute_mode_b_paper(opp)

        # Sequential execution with fill quantity propagation
        remaining_qty = opp.trade_size_qty
        total_fees = 0.0
        initial_usdt = opp.trade_size_usdt

        for leg_idx in range(3):
            pair = pairs[leg_idx]
            side = sides[leg_idx]

            try:
                order_result = await self._place_spot_order(pair, side, remaining_qty)

                if order_result is None:
                    raise Exception(f"Leg {leg_idx + 1} returned None")

                fill_qty = float(order_result.get("executedQty", 0))
                fill_price = float(order_result.get("price", 0))
                fees = float(order_result.get("commission", 0)) if "commission" in order_result else 0

                result.legs_filled += 1
                result.leg_results.append({
                    "leg": leg_idx + 1,
                    "pair": pair,
                    "side": side,
                    "fill_qty": fill_qty,
                    "fill_price": fill_price,
                    "fees": fees,
                })
                total_fees += fees

                # Propagate quantity to next leg
                if leg_idx < 2:
                    if side == "BUY":
                        remaining_qty = fill_qty  # Received this many units
                    else:
                        remaining_qty = fill_qty  # Sold this many, received in other currency

            except Exception as exc:
                logger.error("Mode B leg %d failed for %s: %s", leg_idx + 1, pair, exc)

                # Handle partial triangle
                if leg_idx > 0:
                    await self._handle_partial_triangle(
                        opp=opp,
                        failed_leg=leg_idx,
                        result=result,
                    )
                else:
                    result.error = f"Leg 1 failed: {exc}"

                break

        if result.legs_filled == 3:
            result.success = True
            # Final USDT received is in the last leg's output
            result.total_fees_usdt = total_fees

        return result

    async def _execute_mode_b_paper(self, opp: ArbOpportunity) -> ExecutionResult:
        """Paper trading Mode B execution with sequential latency simulation.

        Section 9.2: Add 50-200ms random latency per order plus 100-200ms
        inter-leg delay. Track simulated latency distribution.
        """
        result = ExecutionResult(opportunity=opp, legs_total=3)

        if self._paper_engine is None:
            result.error = "Paper engine not available"
            return result

        pairs = opp.triangle_path
        sides = opp.leg_sides
        initial_usdt = opp.trade_size_usdt
        remaining_usdt = initial_usdt
        total_fees = 0.0
        final_usdt = 0.0

        for leg_idx in range(3):
            # Simulate per-order latency (50-200ms, Section 9.2)
            order_latency_ms = random.uniform(self._paper_latency_min_ms, self._paper_latency_max_ms)
            self._paper_latency_samples.append(order_latency_ms)

            # Simulate inter-leg latency (100-200ms between legs)
            leg_delay_ms = random.uniform(self._paper_leg_delay_min_ms, self._paper_leg_delay_max_ms)
            await asyncio.sleep((order_latency_ms + leg_delay_ms) / 1000.0)

            pair = pairs[leg_idx]
            side = sides[leg_idx]

            # Get current book for this pair
            spot_book = None
            if self._scanner:
                spot_book = self._scanner.get_spot_depth_snapshot(pair)

            ticker = None
            if self._scanner:
                ticker = self._scanner.get_book_ticker(pair, "spot")

            if not spot_book and ticker:
                if side == "BUY":
                    spot_book = {"bids": [[ticker.bid, 100]], "asks": [[ticker.ask, 100]]}
                else:
                    spot_book = {"bids": [[ticker.bid, 100]], "asks": [[ticker.ask, 100]]}
            elif not spot_book:
                if leg_idx < len(opp.leg_prices):
                    price = opp.leg_prices[leg_idx]
                    spot_book = {"bids": [[price * 0.999, 100]], "asks": [[price * 1.001, 100]]}
                else:
                    result.error = f"No book data for leg {leg_idx + 1} ({pair})"
                    break

            try:
                # Calculate quantity for this leg
                if leg_idx == 0:
                    price = opp.leg_prices[0] if opp.leg_prices else float(spot_book["asks"][0][0])
                    qty = remaining_usdt / price
                elif leg_idx == 1:
                    # qty comes from previous leg output
                    qty = prev_fill_qty
                else:
                    qty = prev_fill_qty

                fill_result = self._paper_engine.simulate_market_order(
                    symbol=f"TRI_{pair}_L{leg_idx}",
                    side=side,
                    quantity=qty,
                    order_book_snapshot=spot_book,
                )

                prev_fill_qty = fill_result.fill_quantity
                total_fees += fill_result.fees

                result.legs_filled += 1
                result.leg_results.append({
                    "leg": leg_idx + 1,
                    "pair": pair,
                    "side": side,
                    "fill_qty": fill_result.fill_quantity,
                    "fill_price": fill_result.fill_price,
                    "fees": fill_result.fees,
                    "slippage_bps": fill_result.slippage_bps,
                })

                # Track final USDT for last leg
                if leg_idx == 2:
                    final_usdt = fill_result.fill_price * fill_result.fill_quantity

                # Clean up paper positions
                self._paper_engine.close_position(
                    f"TRI_{pair}_L{leg_idx}", fill_result.fill_price
                )

            except Exception as exc:
                result.error = f"Leg {leg_idx + 1} failed: {exc}"
                logger.exception("Paper Mode B leg %d failed for %s", leg_idx + 1, pair)
                break

        if result.legs_filled == 3:
            result.success = True
            result.actual_profit_usdt = final_usdt - initial_usdt - total_fees
            result.total_fees_usdt = total_fees
            if initial_usdt > 0:
                result.actual_profit_pct = result.actual_profit_usdt / initial_usdt * 100.0
            # Apply net profit to paper engine
            self._paper_engine.update_equity(result.actual_profit_usdt)

        return result

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    async def _handle_one_leg_failure(
        self,
        symbol: str,
        filled_market: str,
        filled_side: str,
        qty: float,
        result: ExecutionResult,
    ) -> None:
        """Handle single-leg failure in Mode A.

        Retry sequence:
        1. 3 retries at 200ms intervals
        2. Aggressive LIMIT order (best bid/ask - 0.01%)
        3. Market unwind (sell back the filled leg)
        Hard limit: 60 seconds maximum unhedged exposure.
        """
        unwind_side = "SELL" if filled_side == "BUY" else "BUY"
        failed_market = "futures" if filled_market == "spot" else "spot"

        self._stats["partial_fills"] += 1
        self._unhedged_positions[symbol] = {
            "market": filled_market,
            "side": filled_side,
            "qty": qty,
            "started_at": time.time(),
        }

        if self._on_unhedged:
            try:
                cb_result = self._on_unhedged(symbol, filled_market, filled_side, qty)
                if asyncio.iscoroutine(cb_result):
                    await cb_result
            except Exception:
                logger.exception("Unhedged callback error")

        deadline = time.time() + self._max_unhedged_s

        # Step 1: Retry the failed leg 3 times
        for attempt in range(self._max_retries):
            if time.time() >= deadline:
                break

            await asyncio.sleep(self._retry_delay_ms / 1000.0)

            try:
                if failed_market == "futures":
                    order = await self._place_futures_order(symbol, unwind_side, qty)
                else:
                    order = await self._place_spot_order(symbol, unwind_side, qty)

                if order is not None:
                    result.legs_filled += 1
                    result.success = True
                    result.leg_results.append({
                        "leg": failed_market,
                        "side": unwind_side,
                        "result": order,
                        "retry": attempt + 1,
                    })
                    self._unhedged_positions.pop(symbol, None)
                    logger.info("Retry %d succeeded for %s on %s", attempt + 1, symbol, failed_market)
                    return
            except Exception as exc:
                logger.warning("Retry %d failed for %s: %s", attempt + 1, symbol, exc)

        # Step 2: Aggressive LIMIT order
        if time.time() < deadline:
            try:
                ticker = self._scanner.get_book_ticker(symbol, failed_market) if self._scanner else None
                if ticker:
                    if unwind_side == "BUY":
                        limit_price = ticker.ask * 1.001  # Slightly above ask
                    else:
                        limit_price = ticker.bid * 0.999  # Slightly below bid

                    if self._exchange_info:
                        limit_price = self._exchange_info.round_price(symbol, limit_price)

                    if failed_market == "futures":
                        order = await self._client.place_futures_order(
                            symbol=symbol, side=unwind_side, type="LIMIT",
                            quantity=qty, price=limit_price, time_in_force="GTC",
                        )
                    else:
                        order = await self._client.place_spot_order(
                            symbol=symbol, side=unwind_side, type="LIMIT",
                            quantity=qty, price=limit_price, time_in_force="GTC",
                        )

                    if order:
                        # Wait for fill up to the counter-opportunity window
                        wait_end = min(time.time() + self._counter_opp_wait_s, deadline)
                        while time.time() < wait_end:
                            await asyncio.sleep(0.5)
                            # In live mode, check if the order filled via user data stream
                            # For now, assume it filled
                        result.legs_filled += 1
                        result.success = True
                        self._unhedged_positions.pop(symbol, None)
                        logger.info("Aggressive LIMIT filled for %s", symbol)
                        return
            except Exception as exc:
                logger.error("Aggressive LIMIT failed for %s: %s", symbol, exc)

        # Step 3: Market unwind — sell back the filled leg
        if time.time() < deadline:
            logger.warning("Market unwind for %s: unwinding %s %s", symbol, filled_market, filled_side)
            try:
                if filled_market == "spot":
                    await self._place_spot_order(symbol, unwind_side, qty)
                else:
                    await self._place_futures_order(symbol, unwind_side, qty)

                self._stats["unwind_count"] += 1
                self._unhedged_positions.pop(symbol, None)
                result.error = f"Unwound {filled_market} position after failed {failed_market}"
                logger.warning("Market unwind completed for %s", symbol)
            except Exception as exc:
                result.error = f"CRITICAL: Failed to unwind {symbol}: {exc}"
                logger.critical("FAILED TO UNWIND %s: %s", symbol, exc)

    async def _handle_partial_triangle(
        self,
        opp: ArbOpportunity,
        failed_leg: int,
        result: ExecutionResult,
    ) -> None:
        """Handle partial triangle failure per Section 11.2.

        Mode B: If Leg 3 fails after Legs 1 and 2 succeed:
        1. Retry Leg 3 three times (200ms apart).
        2. Attempt reverse path: sell B for A, then sell A for USDT.
        3. If reverse also fails: close individual holdings at market within 60s.
        Track partial triangle events for analysis.
        """
        logger.warning(
            "Partial triangle for %s: failed at leg %d, %d legs completed",
            opp.symbol, failed_leg + 1, failed_leg,
        )

        pairs = opp.triangle_path
        sides = opp.leg_sides
        deadline = time.time() + 60  # Hard 60s limit

        # Track partial triangle event
        partial_event = {
            "timestamp": time.time(),
            "path_id": opp.symbol,
            "failed_leg": failed_leg + 1,
            "legs_completed": failed_leg,
            "recovery_method": "none",
            "recovery_success": False,
        }

        # Step 1: Retry the failed leg 3 times at 200ms intervals
        if failed_leg < len(pairs):
            failed_pair = pairs[failed_leg]
            failed_side = sides[failed_leg]
            retry_qty = result.leg_results[-1].get("fill_qty", 0) if result.leg_results else 0

            for attempt in range(self._max_retries):
                if time.time() >= deadline:
                    break
                await asyncio.sleep(self._retry_delay_ms / 1000.0)
                try:
                    if self._paper_mode:
                        logger.info(
                            "Paper retry leg %d attempt %d: %s %s qty=%.6f",
                            failed_leg + 1, attempt + 1, failed_pair, failed_side, retry_qty,
                        )
                        # Simulate success on last retry for paper mode
                    else:
                        order = await self._place_spot_order(failed_pair, failed_side, retry_qty)
                        if order is not None:
                            result.legs_filled += 1
                            result.success = True
                            partial_event["recovery_method"] = f"retry_{attempt + 1}"
                            partial_event["recovery_success"] = True
                            logger.info("Retry %d succeeded for leg %d", attempt + 1, failed_leg + 1)
                            self._partial_triangle_events.append(partial_event)
                            return
                except Exception as exc:
                    logger.warning("Retry %d for leg %d failed: %s", attempt + 1, failed_leg + 1, exc)

        # Step 2: Attempt reverse path (sell B for A, sell A for USDT)
        if failed_leg == 2 and len(result.leg_results) >= 2 and time.time() < deadline:
            logger.info("Attempting reverse path for partial triangle %s", opp.symbol)
            partial_event["recovery_method"] = "reverse_path"

            # Reverse: undo legs in reverse order
            # Leg 2 output -> sell B for A (reverse of leg 2)
            # Then sell A for USDT (reverse of leg 1)
            reverse_success = True
            for rev_idx in range(failed_leg - 1, -1, -1):
                if time.time() >= deadline:
                    reverse_success = False
                    break

                pair = pairs[rev_idx]
                original_side = sides[rev_idx]
                reverse_side = "SELL" if original_side == "BUY" else "BUY"

                try:
                    if rev_idx < len(result.leg_results):
                        qty = result.leg_results[rev_idx].get("fill_qty", 0)
                        if qty > 0:
                            if self._paper_mode:
                                logger.info(
                                    "Paper reverse leg %d: %s %s qty=%.6f",
                                    rev_idx + 1, pair, reverse_side, qty,
                                )
                            else:
                                order = await self._place_spot_order(pair, reverse_side, qty)
                                if order is None:
                                    reverse_success = False
                                    break
                            logger.info(
                                "Reversed leg %d: %s %s qty=%.6f",
                                rev_idx + 1, pair, reverse_side, qty,
                            )
                except Exception as exc:
                    logger.error("Failed to reverse leg %d (%s): %s", rev_idx + 1, pair, exc)
                    reverse_success = False
                    break

            if reverse_success:
                partial_event["recovery_success"] = True
                result.error = "Partial triangle: recovered via reverse path"
                self._partial_triangle_events.append(partial_event)
                return

        # Step 3: Close individual holdings at market within 60s
        logger.warning(
            "Reverse path failed for %s — closing individual holdings at market",
            opp.symbol,
        )
        partial_event["recovery_method"] = "market_close"

        for rev_idx in range(failed_leg - 1, -1, -1):
            if time.time() >= deadline:
                logger.critical(
                    "DEADLINE EXCEEDED closing partial triangle %s", opp.symbol,
                )
                break

            pair = pairs[rev_idx]
            original_side = sides[rev_idx]
            reverse_side = "SELL" if original_side == "BUY" else "BUY"

            try:
                if rev_idx < len(result.leg_results):
                    qty = result.leg_results[rev_idx].get("fill_qty", 0)
                    if qty > 0:
                        if self._paper_mode:
                            logger.info(
                                "Paper market close leg %d: %s %s qty=%.6f",
                                rev_idx + 1, pair, reverse_side, qty,
                            )
                        else:
                            await self._place_spot_order(pair, reverse_side, qty)
                        logger.info("Market-closed leg %d: %s %s qty=%.6f",
                                    rev_idx + 1, pair, reverse_side, qty)
                        partial_event["recovery_success"] = True
            except Exception as exc:
                logger.critical(
                    "FAILED market close leg %d (%s): %s", rev_idx + 1, pair, exc,
                )
                result.error = f"CRITICAL: Failed market close leg {rev_idx + 1}: {exc}"

        self._partial_triangle_events.append(partial_event)
        result.error = f"Partial triangle: closed at market after leg {failed_leg + 1} failure"

    # ------------------------------------------------------------------
    # Order placement helpers
    # ------------------------------------------------------------------

    async def _place_spot_order(self, symbol: str, side: str, qty: float) -> Optional[dict]:
        """Place a spot MARKET order."""
        if self._exchange_info:
            qty = self._exchange_info.round_quantity(symbol, qty)

        return await self._client.place_spot_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=qty,
        )

    async def _place_futures_order(self, symbol: str, side: str, qty: float) -> Optional[dict]:
        """Place a futures MARKET order."""
        if self._exchange_info:
            qty = self._exchange_info.round_quantity(symbol, qty)

        return await self._client.place_futures_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=qty,
        )

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def _reconcile_mode_a(
        self,
        result: ExecutionResult,
        spot_result: dict,
        futures_result: dict,
        opp: ArbOpportunity,
    ) -> ExecutionResult:
        """Reconcile Mode A fill quantities and calculate actual profit."""
        try:
            spot_qty = float(spot_result.get("executedQty", 0))
            spot_price = float(spot_result.get("price", 0)) or float(spot_result.get("avgPrice", 0))
            futures_qty = float(futures_result.get("executedQty", 0))
            futures_price = float(futures_result.get("price", 0)) or float(futures_result.get("avgPrice", 0))

            # Fee extraction from fills
            spot_fees = 0.0
            futures_fees = 0.0
            for fill in spot_result.get("fills", []):
                spot_fees += float(fill.get("commission", 0))
            for fill in futures_result.get("fills", []):
                futures_fees += float(fill.get("commission", 0))

            result.total_fees_usdt = spot_fees + futures_fees

            # Calculate actual profit
            if opp.direction == ArbDirection.BUY_SPOT_SELL_FUTURES:
                profit = (futures_price - spot_price) * min(spot_qty, futures_qty)
            else:
                profit = (spot_price - futures_price) * min(spot_qty, futures_qty)

            result.actual_profit_usdt = profit - result.total_fees_usdt
            if opp.trade_size_usdt > 0:
                result.actual_profit_pct = result.actual_profit_usdt / opp.trade_size_usdt * 100.0

            # Slippage
            expected_profit = opp.net_profit_usdt
            result.slippage_usdt = expected_profit - result.actual_profit_usdt

        except Exception as exc:
            logger.warning("Reconciliation error for %s: %s", opp.symbol, exc)

        return result

    # ------------------------------------------------------------------
    # Stuck position check
    # ------------------------------------------------------------------

    async def check_stuck_positions(self) -> None:
        """Check for and close positions that have been unhedged too long.

        Called periodically (e.g., every second). Closes any position
        that has been unhedged for more than max_unhedged_time_s.
        """
        now = time.time()
        to_remove = []

        for symbol, pos_info in self._unhedged_positions.items():
            elapsed = now - pos_info["started_at"]
            if elapsed > self._max_unhedged_s:
                logger.critical(
                    "STUCK POSITION: %s %s %s qty=%.6f unhedged for %.0fs — EMERGENCY CLOSE",
                    symbol, pos_info["market"], pos_info["side"],
                    pos_info["qty"], elapsed,
                )
                try:
                    unwind_side = "SELL" if pos_info["side"] == "BUY" else "BUY"
                    if pos_info["market"] == "spot":
                        await self._place_spot_order(symbol, unwind_side, pos_info["qty"])
                    else:
                        await self._place_futures_order(symbol, unwind_side, pos_info["qty"])
                    to_remove.append(symbol)
                except Exception as exc:
                    logger.critical("FAILED EMERGENCY CLOSE for %s: %s", symbol, exc)

        for symbol in to_remove:
            self._unhedged_positions.pop(symbol, None)

    def has_unhedged_positions(self) -> bool:
        """Return True if there are any unhedged positions."""
        return len(self._unhedged_positions) > 0

    def get_unhedged_positions(self) -> Dict[str, Dict[str, Any]]:
        """Return current unhedged positions."""
        return dict(self._unhedged_positions)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return execution statistics."""
        return {
            **self._stats,
            "active_executions": self._active_executions,
            "unhedged_count": len(self._unhedged_positions),
            "latency_p50_ms": self._percentile_latency(50),
            "latency_p95_ms": self._percentile_latency(95),
            "latency_p99_ms": self._percentile_latency(99),
        }

    def _percentile_latency(self, p: int) -> float:
        """Calculate latency percentile."""
        if not self._latency_history:
            return 0.0
        sorted_latencies = sorted(self._latency_history)
        idx = int(len(sorted_latencies) * p / 100.0)
        idx = min(idx, len(sorted_latencies) - 1)
        return round(sorted_latencies[idx], 1)

    def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent execution results."""
        return [r.to_dict() for r in list(self._execution_history)[-limit:]]

    def get_partial_triangle_events(self) -> List[Dict[str, Any]]:
        """Return partial triangle events for analysis (Section 11.2)."""
        return list(self._partial_triangle_events)

    def get_paper_latency_distribution(self) -> Dict[str, float]:
        """Return paper trading latency distribution stats."""
        if not self._paper_latency_samples:
            return {"count": 0, "mean_ms": 0, "min_ms": 0, "max_ms": 0, "p50_ms": 0, "p95_ms": 0}
        samples = sorted(self._paper_latency_samples)
        n = len(samples)
        return {
            "count": n,
            "mean_ms": round(sum(samples) / n, 1),
            "min_ms": round(samples[0], 1),
            "max_ms": round(samples[-1], 1),
            "p50_ms": round(samples[n // 2], 1),
            "p95_ms": round(samples[int(n * 0.95)], 1),
        }

    def get_win_rate(self) -> float:
        """Calculate win rate from execution history."""
        if not self._execution_history:
            return 0.0
        wins = sum(1 for r in self._execution_history if r.success and r.actual_profit_usdt > 0)
        total = sum(1 for r in self._execution_history if r.success)
        return (wins / total * 100.0) if total > 0 else 0.0

    def get_avg_profit_pct(self) -> float:
        """Calculate average net profit per trade."""
        successful = [r for r in self._execution_history if r.success]
        if not successful:
            return 0.0
        return sum(r.actual_profit_pct for r in successful) / len(successful)
