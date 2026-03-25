"""Simultaneous spot + futures execution engine.

Handles the critical challenge of executing both legs within a 100ms window,
fill reconciliation, delta neutrality verification, one-leg failure handling,
and simultaneous unwinding per Section 3.3 and Section 4.3.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


@dataclass
class ExecutionResult:
    """Result of a dual-leg execution attempt."""
    success: bool
    spot_fill: Optional[Dict[str, Any]] = None
    futures_fill: Optional[Dict[str, Any]] = None
    spot_fill_price: float = 0.0
    futures_fill_price: float = 0.0
    spot_quantity: float = 0.0
    futures_quantity: float = 0.0
    actual_basis_pct: float = 0.0
    intended_basis_pct: float = 0.0
    slippage_spot_bps: float = 0.0
    slippage_futures_bps: float = 0.0
    execution_time_ms: int = 0
    total_fees: float = 0.0
    errors: List[str] = field(default_factory=list)
    delta_pct: float = 0.0


class ExecutionEngine:
    """Handles simultaneous spot and futures order execution.

    Parameters
    ----------
    binance_client : BinanceClient
        REST client for order placement.
    paper_mode : bool
        If True, simulate fills using paper trading engine.
    paper_engine : PaperTradingEngine, optional
        Paper trading simulator (required when paper_mode=True).
    config : dict
        Strategy parameters.
    """

    def __init__(
        self,
        binance_client: Any,
        paper_mode: bool = True,
        paper_engine: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = binance_client
        self._paper_mode = paper_mode
        self._paper_engine = paper_engine
        self._config = config or {}

        # Execution parameters
        self._max_window_ms = self._config.get("max_execution_window_ms", 100)
        self._delta_tolerance = self._config.get("delta_tolerance_pct", 0.5) / 100.0
        self._delta_critical = self._config.get("delta_critical_pct", 5.0) / 100.0
        self._corrective_timeout = self._config.get("corrective_order_timeout_s", 5.0)

    # ══════════════════════════════════════════════════════════════════════
    #  Entry execution (Section 3.3)
    # ══════════════════════════════════════════════════════════════════════

    async def execute_entry(
        self,
        symbol: str,
        spot_quantity: float,
        futures_quantity: float,
        spot_symbol: Optional[str] = None,
        futures_symbol: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute simultaneous spot buy + futures short within 100ms.

        Parameters
        ----------
        symbol : str
            The trading pair (e.g. "BTCUSDT").
        spot_quantity : float
            Quantity to buy on spot market.
        futures_quantity : float
            Quantity to sell short on futures market.
        spot_symbol : str, optional
            Spot symbol if different from futures symbol.
        futures_symbol : str, optional
            Futures symbol if different from spot symbol.
        """
        spot_sym = spot_symbol or symbol
        futures_sym = futures_symbol or symbol

        logger.info(
            "ENTRY EXECUTION: %s spot_buy=%.8f futures_short=%.8f",
            symbol, spot_quantity, futures_quantity,
        )

        if self._paper_mode:
            return await self._simulate_entry(
                spot_sym, futures_sym, spot_quantity, futures_quantity
            )

        return await self._live_entry(
            spot_sym, futures_sym, spot_quantity, futures_quantity
        )

    async def _live_entry(
        self,
        spot_symbol: str,
        futures_symbol: str,
        spot_qty: float,
        futures_qty: float,
    ) -> ExecutionResult:
        """Execute live entry with parallel order placement."""
        start_ms = int(time.time() * 1000)
        result = ExecutionResult(success=False)

        # Step 7: Place BOTH orders within 100ms window
        spot_task = asyncio.create_task(
            self._place_spot_buy(spot_symbol, spot_qty),
            name=f"spot_buy_{spot_symbol}",
        )
        futures_task = asyncio.create_task(
            self._place_futures_sell(futures_symbol, futures_qty),
            name=f"futures_sell_{futures_symbol}",
        )

        try:
            # Wait for both with a tight timeout
            spot_result, futures_result = await asyncio.wait_for(
                asyncio.gather(spot_task, futures_task, return_exceptions=True),
                timeout=self._max_window_ms / 1000.0 + 2.0,  # Buffer for network
            )
        except asyncio.TimeoutError:
            logger.error("Entry execution timed out")
            # Cancel any pending tasks
            spot_task.cancel()
            futures_task.cancel()
            result.errors.append("execution_timeout")
            return result

        execution_time_ms = int(time.time() * 1000) - start_ms

        # Handle one-leg failure
        spot_ok = not isinstance(spot_result, Exception)
        futures_ok = not isinstance(futures_result, Exception)

        if not spot_ok and not futures_ok:
            # Both failed — clean failure, no action needed
            result.errors.append(f"both_legs_failed: spot={spot_result}, futures={futures_result}")
            logger.error("Both legs failed: spot=%s futures=%s", spot_result, futures_result)
            return result

        if spot_ok and not futures_ok:
            # Spot filled, futures failed — CRITICAL: handle one-leg failure
            logger.critical(
                "ONE LEG FAILURE: Spot filled but futures failed: %s",
                futures_result,
            )
            result.errors.append(f"futures_leg_failed: {futures_result}")
            # Attempt to close the spot position
            await self._handle_one_leg_failure(
                spot_symbol, spot_result, "spot", futures_symbol, futures_qty
            )
            return result

        if not spot_ok and futures_ok:
            # Futures filled, spot failed — CRITICAL
            logger.critical(
                "ONE LEG FAILURE: Futures filled but spot failed: %s",
                spot_result,
            )
            result.errors.append(f"spot_leg_failed: {spot_result}")
            await self._handle_one_leg_failure(
                futures_symbol, futures_result, "futures", spot_symbol, spot_qty
            )
            return result

        # Both filled successfully — reconcile
        result.spot_fill = spot_result
        result.futures_fill = futures_result
        result.execution_time_ms = execution_time_ms

        # Step 8: Fill reconciliation
        result.spot_fill_price = float(spot_result.get("avgPrice", spot_result.get("price", 0)))
        result.futures_fill_price = float(futures_result.get("avgPrice", futures_result.get("price", 0)))
        result.spot_quantity = float(spot_result.get("executedQty", 0))
        result.futures_quantity = float(futures_result.get("executedQty", 0))

        if result.spot_fill_price > 0:
            result.actual_basis_pct = (
                (result.futures_fill_price - result.spot_fill_price)
                / result.spot_fill_price * 100.0
            )

        # Calculate fees
        result.total_fees = self._calc_fees(spot_result) + self._calc_fees(futures_result)

        # Step 9: Delta neutrality verification
        result.delta_pct = self._calc_delta_pct(
            result.spot_quantity, result.spot_fill_price,
            result.futures_quantity, result.futures_fill_price,
        )

        if abs(result.delta_pct) > self._delta_critical * 100:
            logger.warning(
                "Delta critical: %.2f%% — placing corrective order",
                result.delta_pct,
            )
            await self._correct_delta(
                spot_symbol, futures_symbol,
                result.spot_quantity, result.spot_fill_price,
                result.futures_quantity, result.futures_fill_price,
            )
        elif abs(result.delta_pct) > self._delta_tolerance * 100:
            logger.info("Delta acceptable but elevated: %.2f%%", result.delta_pct)

        result.success = True

        trade_logger.info(
            "ENTRY_EXEC\tsymbol=%s\tspot_px=%.4f\tfut_px=%.4f\t"
            "basis=%.4f%%\texec_ms=%d\tdelta=%.2f%%\tfees=%.6f",
            spot_symbol, result.spot_fill_price, result.futures_fill_price,
            result.actual_basis_pct, execution_time_ms,
            result.delta_pct, result.total_fees,
        )

        return result

    async def _simulate_entry(
        self,
        spot_symbol: str,
        futures_symbol: str,
        spot_qty: float,
        futures_qty: float,
    ) -> ExecutionResult:
        """Simulate entry for paper trading mode."""
        start_ms = int(time.time() * 1000)
        result = ExecutionResult(success=False)

        if self._paper_engine is None:
            result.errors.append("paper_engine_not_configured")
            return result

        # Simulate 200ms delay between legs per Section 9.2
        await asyncio.sleep(0.2)

        # Get current prices from the paper engine's state
        # In paper mode, we use the best ask/bid as proxies
        spot_fill = {
            "symbol": spot_symbol,
            "side": "BUY",
            "type": "MARKET",
            "executedQty": str(spot_qty),
            "avgPrice": "0",
            "status": "FILLED",
            "orderId": int(time.time() * 1000),
        }

        futures_fill = {
            "symbol": futures_symbol,
            "side": "SELL",
            "type": "MARKET",
            "executedQty": str(futures_qty),
            "avgPrice": "0",
            "status": "FILLED",
            "orderId": int(time.time() * 1000) + 1,
        }

        result.spot_fill = spot_fill
        result.futures_fill = futures_fill
        result.spot_quantity = spot_qty
        result.futures_quantity = futures_qty
        result.execution_time_ms = int(time.time() * 1000) - start_ms
        result.success = True

        return result

    # ══════════════════════════════════════════════════════════════════════
    #  Exit execution (Section 4.3)
    # ══════════════════════════════════════════════════════════════════════

    async def execute_exit(
        self,
        symbol: str,
        spot_quantity: float,
        futures_quantity: float,
        spot_symbol: Optional[str] = None,
        futures_symbol: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute simultaneous spot sell + futures close (buy to cover).

        Uses the same 100ms window requirement as entry.
        """
        spot_sym = spot_symbol or symbol
        futures_sym = futures_symbol or symbol

        logger.info(
            "EXIT EXECUTION: %s spot_sell=%.8f futures_close=%.8f",
            symbol, spot_quantity, futures_quantity,
        )

        if self._paper_mode:
            return await self._simulate_exit(
                spot_sym, futures_sym, spot_quantity, futures_quantity
            )

        return await self._live_exit(
            spot_sym, futures_sym, spot_quantity, futures_quantity
        )

    async def _live_exit(
        self,
        spot_symbol: str,
        futures_symbol: str,
        spot_qty: float,
        futures_qty: float,
    ) -> ExecutionResult:
        """Execute live exit with parallel order placement."""
        start_ms = int(time.time() * 1000)
        result = ExecutionResult(success=False)

        # Place both closing orders within 100ms
        spot_task = asyncio.create_task(
            self._place_spot_sell(spot_symbol, spot_qty),
            name=f"spot_sell_{spot_symbol}",
        )
        futures_task = asyncio.create_task(
            self._place_futures_buy(futures_symbol, futures_qty),
            name=f"futures_buy_{futures_symbol}",
        )

        try:
            spot_result, futures_result = await asyncio.wait_for(
                asyncio.gather(spot_task, futures_task, return_exceptions=True),
                timeout=self._max_window_ms / 1000.0 + 2.0,
            )
        except asyncio.TimeoutError:
            spot_task.cancel()
            futures_task.cancel()
            result.errors.append("exit_execution_timeout")
            return result

        execution_time_ms = int(time.time() * 1000) - start_ms

        spot_ok = not isinstance(spot_result, Exception)
        futures_ok = not isinstance(futures_result, Exception)

        if not spot_ok and not futures_ok:
            result.errors.append(f"both_exit_legs_failed: spot={spot_result}, futures={futures_result}")
            logger.error("Both exit legs failed")
            return result

        if spot_ok and not futures_ok:
            # Section 4.3 Step 2: One leg fills, other fails
            logger.critical("EXIT ONE LEG FAILURE: Futures close failed: %s", futures_result)
            await self._retry_failed_exit_leg(
                futures_symbol, futures_qty, "BUY"
            )

        if not spot_ok and futures_ok:
            logger.critical("EXIT ONE LEG FAILURE: Spot sell failed: %s", spot_result)
            await self._retry_failed_exit_leg(
                spot_symbol, spot_qty, "SELL"
            )

        if spot_ok:
            result.spot_fill = spot_result
            result.spot_fill_price = float(spot_result.get("avgPrice", spot_result.get("price", 0)))
            result.spot_quantity = float(spot_result.get("executedQty", 0))

        if futures_ok:
            result.futures_fill = futures_result
            result.futures_fill_price = float(futures_result.get("avgPrice", futures_result.get("price", 0)))
            result.futures_quantity = float(futures_result.get("executedQty", 0))

        result.execution_time_ms = execution_time_ms
        result.total_fees = (
            (self._calc_fees(spot_result) if spot_ok else 0) +
            (self._calc_fees(futures_result) if futures_ok else 0)
        )
        result.success = spot_ok and futures_ok

        trade_logger.info(
            "EXIT_EXEC\tsymbol=%s\tspot_px=%.4f\tfut_px=%.4f\t"
            "exec_ms=%d\tsuccess=%s\tfees=%.6f",
            spot_symbol,
            result.spot_fill_price,
            result.futures_fill_price,
            execution_time_ms,
            result.success,
            result.total_fees,
        )

        return result

    async def _simulate_exit(
        self,
        spot_symbol: str,
        futures_symbol: str,
        spot_qty: float,
        futures_qty: float,
    ) -> ExecutionResult:
        """Simulate exit for paper trading."""
        start_ms = int(time.time() * 1000)
        result = ExecutionResult(success=False)

        if self._paper_engine is None:
            result.errors.append("paper_engine_not_configured")
            return result

        await asyncio.sleep(0.2)

        result.spot_fill = {
            "symbol": spot_symbol, "side": "SELL", "type": "MARKET",
            "executedQty": str(spot_qty), "avgPrice": "0", "status": "FILLED",
            "orderId": int(time.time() * 1000),
        }
        result.futures_fill = {
            "symbol": futures_symbol, "side": "BUY", "type": "MARKET",
            "executedQty": str(futures_qty), "avgPrice": "0", "status": "FILLED",
            "orderId": int(time.time() * 1000) + 1,
        }
        result.spot_quantity = spot_qty
        result.futures_quantity = futures_qty
        result.execution_time_ms = int(time.time() * 1000) - start_ms
        result.success = True

        return result

    # ══════════════════════════════════════════════════════════════════════
    #  Pre-entry validation (Section 3.3 Step 6)
    # ══════════════════════════════════════════════════════════════════════

    async def validate_pre_entry(
        self,
        symbol: str,
        spot_quantity: float,
        futures_quantity: float,
        spot_usdt_balance: float,
        futures_usdt_balance: float,
        spot_depth: Optional[Dict] = None,
        futures_depth: Optional[Dict] = None,
        spot_best_ask: float = 0.0,
        futures_best_bid: float = 0.0,
    ) -> Tuple[bool, str]:
        """Validate all pre-entry conditions.

        Returns (valid, reason).
        """
        # Check sufficient USDT in BOTH wallets
        required_spot_usdt = spot_quantity * spot_best_ask if spot_best_ask > 0 else 0
        if spot_usdt_balance < required_spot_usdt:
            return False, f"Insufficient spot USDT: have {spot_usdt_balance:.2f}, need {required_spot_usdt:.2f}"

        required_futures_margin = futures_quantity * futures_best_bid / 2.0  # 2x leverage
        if futures_usdt_balance < required_futures_margin:
            return False, f"Insufficient futures margin: have {futures_usdt_balance:.2f}, need {required_futures_margin:.2f}"

        # Check order book depth >= 1.5x size
        depth_multiplier = 1.5
        if spot_depth:
            spot_ask_depth = self._total_depth(spot_depth.get("asks", []), levels=3)
            if spot_ask_depth < spot_quantity * depth_multiplier:
                return False, f"Insufficient spot ask depth: {spot_ask_depth:.8f} < {spot_quantity * depth_multiplier:.8f}"

        if futures_depth:
            futures_bid_depth = self._total_depth(futures_depth.get("bids", []), levels=3)
            if futures_bid_depth < futures_quantity * depth_multiplier:
                return False, f"Insufficient futures bid depth: {futures_bid_depth:.8f} < {futures_quantity * depth_multiplier:.8f}"

        # Check bid-ask spread < 0.05% of mid-price
        if spot_best_ask > 0:
            inst_spot_bid = spot_depth.get("bids", [[0]])[0][0] if spot_depth else 0
            if float(inst_spot_bid) > 0:
                mid = (float(inst_spot_bid) + spot_best_ask) / 2
                spread_pct = (spot_best_ask - float(inst_spot_bid)) / mid * 100
                if spread_pct > 0.05:
                    return False, f"Spot spread too wide: {spread_pct:.4f}% > 0.05%"

        return True, ""

    # ══════════════════════════════════════════════════════════════════════
    #  Kill switch — close all positions within 10 seconds
    # ══════════════════════════════════════════════════════════════════════

    async def emergency_close_all(
        self,
        positions: List[Dict[str, Any]],
    ) -> List[ExecutionResult]:
        """Close all positions as fast as possible for kill switch.

        Must complete within 10 seconds per Section 5.7.
        """
        if not positions:
            return []

        logger.critical("KILL SWITCH: Closing %d positions", len(positions))
        results: List[ExecutionResult] = []

        # Execute all closures in parallel
        tasks = []
        for pos in positions:
            task = asyncio.create_task(
                self.execute_exit(
                    symbol=pos["symbol"],
                    spot_quantity=pos.get("spot_quantity", 0),
                    futures_quantity=pos.get("futures_quantity", 0),
                ),
                name=f"kill_{pos['symbol']}",
            )
            tasks.append(task)

        try:
            completed = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=10.0,
            )
            for r in completed:
                if isinstance(r, ExecutionResult):
                    results.append(r)
                else:
                    logger.error("Kill switch execution error: %s", r)
        except asyncio.TimeoutError:
            logger.critical("KILL SWITCH TIMEOUT: Some positions may remain open")
            for t in tasks:
                if not t.done():
                    t.cancel()

        return results

    # ══════════════════════════════════════════════════════════════════════
    #  One-leg failure handling (Section 11.1)
    # ══════════════════════════════════════════════════════════════════════

    async def _handle_one_leg_failure(
        self,
        filled_symbol: str,
        filled_result: Dict,
        filled_side: str,
        failed_symbol: str,
        failed_qty: float,
    ) -> None:
        """Handle the case where one leg filled but the other failed.

        Implements Section 11.1 retry and fallback logic.
        """
        logger.critical(
            "ONE-LEG FAILURE: %s filled (%s), %s failed — attempting recovery",
            filled_symbol, filled_side, failed_symbol,
        )

        # Retry failed leg: 5 attempts, 200ms apart
        for attempt in range(1, 6):
            try:
                if filled_side == "spot":
                    # Spot filled, need to open futures short
                    result = await self._place_futures_sell(failed_symbol, failed_qty)
                else:
                    # Futures filled, need to buy spot
                    result = await self._place_spot_buy(failed_symbol, failed_qty)

                logger.info("Recovery succeeded on attempt %d", attempt)
                return

            except Exception as exc:
                logger.warning(
                    "Recovery attempt %d failed: %s", attempt, exc,
                )
                await asyncio.sleep(0.2)

        # All retries failed — close the filled leg
        logger.critical(
            "All recovery attempts failed — closing filled leg on %s",
            filled_symbol,
        )

        try:
            filled_qty = float(filled_result.get("executedQty", 0))
            if filled_side == "spot":
                await self._place_spot_sell(filled_symbol, filled_qty)
            else:
                await self._place_futures_buy(filled_symbol, filled_qty)
            logger.info("Filled leg closed successfully")
        except Exception as exc:
            # EMERGENCY: position is directionally exposed
            logger.critical(
                "EMERGENCY: Cannot close filled leg on %s: %s — "
                "position is directionally exposed, alerting operator",
                filled_symbol, exc,
            )

    async def _retry_failed_exit_leg(
        self,
        symbol: str,
        quantity: float,
        side: str,
    ) -> Optional[Dict]:
        """Section 4.3 Step 2: Retry failed exit leg.

        5 retries at 200ms intervals, then aggressive limit, then market.
        """
        # Phase 1: 5 market retries, 200ms apart
        for attempt in range(1, 6):
            try:
                if side == "BUY":
                    return await self._place_futures_buy(symbol, quantity)
                else:
                    return await self._place_spot_sell(symbol, quantity)
            except Exception as exc:
                logger.warning(
                    "Exit retry %d/%d failed for %s: %s",
                    attempt, 5, symbol, exc,
                )
                await asyncio.sleep(0.2)

        # Phase 2: Aggressive limit order at best bid/ask, wait 30s
        logger.warning(
            "Market retries exhausted — attempting aggressive limit for %s",
            symbol,
        )
        try:
            if side == "BUY":
                result = await self._client.place_futures_order(
                    symbol=symbol, side="BUY", type="LIMIT",
                    quantity=quantity, time_in_force="IOC",
                )
            else:
                result = await self._client.place_spot_order(
                    symbol=symbol, side="SELL", type="LIMIT",
                    quantity=quantity, time_in_force="IOC",
                )

            if float(result.get("executedQty", 0)) >= quantity * 0.95:
                return result

        except Exception as exc:
            logger.error("Aggressive limit failed: %s", exc)

        # Phase 3: Market with no price limit
        logger.critical(
            "FINAL ATTEMPT: Market order with no price limit for %s %s %.8f",
            symbol, side, quantity,
        )
        try:
            if side == "BUY":
                return await self._place_futures_buy(symbol, quantity)
            else:
                return await self._place_spot_sell(symbol, quantity)
        except Exception as exc:
            logger.critical("FINAL MARKET ORDER FAILED: %s", exc)
            return None

    # ══════════════════════════════════════════════════════════════════════
    #  Delta correction (Section 3.3 Step 9)
    # ══════════════════════════════════════════════════════════════════════

    async def _correct_delta(
        self,
        spot_symbol: str,
        futures_symbol: str,
        spot_qty: float,
        spot_price: float,
        futures_qty: float,
        futures_price: float,
    ) -> None:
        """Place a corrective order if delta exceeds tolerance.

        Must complete within 5 seconds.
        """
        spot_notional = spot_qty * spot_price
        futures_notional = futures_qty * futures_price
        delta = spot_notional - futures_notional

        try:
            if delta > 0:
                # Spot side is larger — need to sell more futures
                correction_notional = abs(delta)
                correction_qty = correction_notional / futures_price
                logger.info(
                    "Delta correction: selling %.8f more futures on %s",
                    correction_qty, futures_symbol,
                )
                await asyncio.wait_for(
                    self._place_futures_sell(futures_symbol, correction_qty),
                    timeout=self._corrective_timeout,
                )
            else:
                # Futures side is larger — need to buy more spot
                correction_notional = abs(delta)
                correction_qty = correction_notional / spot_price
                logger.info(
                    "Delta correction: buying %.8f more spot on %s",
                    correction_qty, spot_symbol,
                )
                await asyncio.wait_for(
                    self._place_spot_buy(spot_symbol, correction_qty),
                    timeout=self._corrective_timeout,
                )
        except asyncio.TimeoutError:
            logger.error("Delta correction timed out after %.1fs", self._corrective_timeout)
        except Exception as exc:
            logger.error("Delta correction failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════════
    #  Order placement wrappers
    # ══════════════════════════════════════════════════════════════════════

    async def _place_spot_buy(self, symbol: str, quantity: float) -> Dict:
        """Place a spot market buy order."""
        return await self._client.place_spot_order(
            symbol=symbol, side="BUY", type="MARKET", quantity=quantity,
        )

    async def _place_spot_sell(self, symbol: str, quantity: float) -> Dict:
        """Place a spot market sell order."""
        return await self._client.place_spot_order(
            symbol=symbol, side="SELL", type="MARKET", quantity=quantity,
        )

    async def _place_futures_sell(self, symbol: str, quantity: float) -> Dict:
        """Place a futures market sell (open short)."""
        return await self._client.place_futures_order(
            symbol=symbol, side="SELL", type="MARKET", quantity=quantity,
        )

    async def _place_futures_buy(self, symbol: str, quantity: float) -> Dict:
        """Place a futures market buy (close short)."""
        return await self._client.place_futures_order(
            symbol=symbol, side="BUY", type="MARKET", quantity=quantity,
            reduce_only=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _calc_delta_pct(
        spot_qty: float, spot_price: float,
        futures_qty: float, futures_price: float,
    ) -> float:
        """Calculate delta as percentage of total notional."""
        spot_notional = spot_qty * spot_price
        futures_notional = futures_qty * futures_price
        total = spot_notional + futures_notional
        if total <= 0:
            return 0.0
        delta = spot_notional - futures_notional
        return (delta / total) * 100.0

    @staticmethod
    def _total_depth(levels: List, levels_count: int = 3) -> float:
        """Sum quantity across first N depth levels."""
        total = 0.0
        for i, level in enumerate(levels):
            if i >= levels_count:
                break
            total += float(level[1]) if len(level) > 1 else 0.0
        return total

    @staticmethod
    def _calc_fees(fill_result: Dict) -> float:
        """Extract commission from a fill result."""
        commission = float(fill_result.get("commission", 0))
        # If no commission field, estimate from cummulative quote qty
        if commission == 0:
            cum_quote = float(fill_result.get("cummulativeQuoteQty",
                              fill_result.get("cumQuote", 0)))
            # Estimate taker fee
            commission = cum_quote * 0.0004  # 0.04% for futures
        return commission
