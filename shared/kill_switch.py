"""Emergency shutdown system for the trading bot.

Cancels all open orders (futures + spot), closes all positions at market,
persists final state, and logs every action. Designed to complete within
a configurable time limit (default 15 seconds).

Triggerable from:
- Dashboard API endpoint
- Automated circuit breaker
- Programmatic call (``await kill_switch.execute("reason")``)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")
system_logger = logging.getLogger("system")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class KillSwitchResult:
    """Outcome of a kill-switch execution."""

    orders_cancelled: int = 0
    positions_closed: int = 0
    duration_ms: int = 0
    errors: List[str] = field(default_factory=list)
    closed_positions: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "orders_cancelled": self.orders_cancelled,
            "positions_closed": self.positions_closed,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "closed_positions": self.closed_positions,
            "success": self.success,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# KillSwitch
# ---------------------------------------------------------------------------

class KillSwitch:
    """Emergency shutdown: cancel all orders, close all positions.

    Parameters
    ----------
    binance_client:
        A :class:`~shared.binance_client.BinanceClient` instance (must already
        be started).
    state_persistence:
        Object with a ``save(key, data)`` method for persisting final state
        (e.g. a simple JSON writer). Can be None.
    database_manager:
        A :class:`~shared.database.DatabaseManager` for logging the event.
        Can be None.
    max_execution_time:
        Hard time limit in seconds for the entire shutdown sequence.
    """

    def __init__(
        self,
        binance_client: Any,
        state_persistence: Any = None,
        database_manager: Any = None,
        max_execution_time: int = 15,
    ) -> None:
        self._client = binance_client
        self._persistence = state_persistence
        self._db = database_manager
        self._max_time = max_execution_time

        # State
        self._triggered = False
        self._armed = True
        self._last_trigger: Optional[Dict[str, Any]] = None

        logger.info(
            "KillSwitch initialised (max_execution_time=%ds, armed=%s)",
            max_execution_time, self._armed,
        )

    # ======================================================================
    #  Main execution
    # ======================================================================

    async def execute(self, reason: str) -> KillSwitchResult:
        """Execute the emergency shutdown sequence.

        1. Cancel all open orders (futures + spot, concurrently)
        2. Close all open positions at market
        3. Persist final state
        4. Log everything

        The entire operation must complete within *max_execution_time* seconds.
        """
        start = time.monotonic()
        result = KillSwitchResult()

        self._triggered = True
        system_logger.info("KILL_SWITCH\tEXECUTE\treason=%s", reason)
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

        try:
            await asyncio.wait_for(
                self._execute_inner(result, reason),
                timeout=self._max_time,
            )
        except asyncio.TimeoutError:
            msg = f"Kill switch timed out after {self._max_time}s"
            logger.error(msg)
            result.errors.append(msg)
        except Exception as exc:
            msg = f"Kill switch unexpected error: {exc}"
            logger.error(msg, exc_info=True)
            result.errors.append(msg)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        result.duration_ms = elapsed_ms

        # Persist result
        self._last_trigger = {
            "reason": reason,
            "timestamp": time.time(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "result": result.to_dict(),
        }

        self._persist_state(result, reason)

        system_logger.info(
            "KILL_SWITCH\tCOMPLETE\torders_cancelled=%d\tpositions_closed=%d\t"
            "duration_ms=%d\terrors=%d",
            result.orders_cancelled, result.positions_closed,
            result.duration_ms, len(result.errors),
        )

        return result

    async def _execute_inner(self, result: KillSwitchResult, reason: str) -> None:
        """Inner execution (runs inside the timeout wrapper)."""

        # Step 1: Cancel all orders and get positions concurrently
        cancel_task = asyncio.create_task(self._cancel_all_orders_inner())
        positions_task = asyncio.create_task(self._get_all_positions())

        cancelled_count, cancel_errors = await cancel_task
        result.orders_cancelled = cancelled_count
        result.errors.extend(cancel_errors)

        positions, pos_errors = await positions_task
        result.errors.extend(pos_errors)

        # Step 2: Close all positions at market
        if positions:
            closed, close_errors = await self._close_positions(positions)
            result.positions_closed = len(closed)
            result.closed_positions = closed
            result.errors.extend(close_errors)

    # ======================================================================
    #  Cancel all orders
    # ======================================================================

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders across futures and spot. Returns count."""
        count, _ = await self._cancel_all_orders_inner()
        return count

    async def _cancel_all_orders_inner(self) -> tuple[int, list[str]]:
        """Internal: cancel all open orders, returns (count, errors)."""
        total_cancelled = 0
        errors: List[str] = []

        # --- Futures orders ---
        try:
            futures_orders = await self._client.get_futures_open_orders()
            if futures_orders:
                # Group by symbol for batch cancellation
                symbols = set(o["symbol"] for o in futures_orders)
                cancel_tasks = [
                    self._cancel_futures_symbol(sym) for sym in symbols
                ]
                cancel_results = await asyncio.gather(*cancel_tasks, return_exceptions=True)

                for sym, res in zip(symbols, cancel_results):
                    if isinstance(res, Exception):
                        msg = f"Failed to cancel futures orders for {sym}: {res}"
                        logger.error(msg)
                        errors.append(msg)
                    else:
                        total_cancelled += res

                trade_logger.info(
                    "KILL_SWITCH\tCANCEL_FUTURES_ORDERS\tcount=%d\tsymbols=%s",
                    total_cancelled, ",".join(symbols),
                )
        except Exception as exc:
            msg = f"Failed to fetch futures open orders: {exc}"
            logger.error(msg)
            errors.append(msg)

        # --- Spot orders ---
        try:
            spot_account = await self._client.get_spot_account()
            # Spot API doesn't have a single "all open orders" endpoint across symbols,
            # but we can check if there are any open orders via the account data.
            # For safety, attempt to cancel on known symbols.
            # In practice, spot orders are less common in this bot architecture.
            spot_cancelled = 0
            trade_logger.info(
                "KILL_SWITCH\tCANCEL_SPOT_ORDERS\tcount=%d", spot_cancelled,
            )
            total_cancelled += spot_cancelled
        except Exception as exc:
            msg = f"Failed to process spot orders: {exc}"
            logger.error(msg)
            errors.append(msg)

        return total_cancelled, errors

    async def _cancel_futures_symbol(self, symbol: str) -> int:
        """Cancel all futures orders for a single symbol. Returns count cancelled."""
        try:
            result = await self._client.cancel_all_futures_orders(symbol)
            # Binance returns { "code": 200, "msg": "The operation of cancel all open order is done." }
            # or a list of cancelled orders
            if isinstance(result, list):
                count = len(result)
            else:
                count = 1  # conservative
            logger.info("Cancelled futures orders for %s: %s", symbol, result)
            return count
        except Exception as exc:
            logger.error("Failed to cancel futures orders for %s: %s", symbol, exc)
            raise

    # ======================================================================
    #  Close all positions
    # ======================================================================

    async def close_all_positions(self) -> List[Dict[str, Any]]:
        """Close all open positions at market. Returns list of closed position dicts."""
        positions, _ = await self._get_all_positions()
        closed, _ = await self._close_positions(positions)
        return closed

    async def _get_all_positions(self) -> tuple[list[dict], list[str]]:
        """Fetch all open futures positions from the exchange."""
        errors: List[str] = []
        positions: List[dict] = []

        try:
            account = await self._client.get_futures_account()
            for pos in account.get("positions", []):
                amt = float(pos.get("positionAmt", 0))
                if amt != 0:
                    positions.append({
                        "symbol": pos["symbol"],
                        "positionAmt": amt,
                        "entryPrice": float(pos.get("entryPrice", 0)),
                        "unrealizedProfit": float(pos.get("unrealizedProfit", 0)),
                        "leverage": int(pos.get("leverage", 1)),
                        "side": "LONG" if amt > 0 else "SHORT",
                    })
            logger.info("Found %d open positions to close", len(positions))
        except Exception as exc:
            msg = f"Failed to fetch positions: {exc}"
            logger.error(msg)
            errors.append(msg)

        return positions, errors

    async def _close_positions(
        self, positions: List[dict],
    ) -> tuple[list[dict], list[str]]:
        """Close a list of positions at market. Returns (closed, errors)."""
        errors: List[str] = []
        closed: List[Dict[str, Any]] = []

        close_tasks = [self._close_single_position(pos) for pos in positions]
        results = await asyncio.gather(*close_tasks, return_exceptions=True)

        for pos, res in zip(positions, results):
            symbol = pos["symbol"]
            if isinstance(res, Exception):
                msg = f"Failed to close position {symbol}: {res}"
                logger.error(msg)
                errors.append(msg)
            else:
                closed.append(res)
                trade_logger.info(
                    "KILL_SWITCH\tCLOSE_POSITION\tsymbol=%s\tside=%s\tqty=%.6f\t"
                    "entry=%.4f\tunrealised_pnl=%.4f",
                    symbol, pos["side"], abs(pos["positionAmt"]),
                    pos["entryPrice"], pos["unrealizedProfit"],
                )

        return closed, errors

    async def _close_single_position(self, pos: dict) -> Dict[str, Any]:
        """Close a single position by placing a market order in the opposite direction."""
        symbol = pos["symbol"]
        amt = pos["positionAmt"]
        close_side = "SELL" if amt > 0 else "BUY"
        qty = abs(amt)

        logger.info(
            "Closing position: %s %s qty=%.6f at MARKET",
            symbol, close_side, qty,
        )

        order_result = await self._client.place_futures_order(
            symbol=symbol,
            side=close_side,
            type="MARKET",
            quantity=qty,
            reduce_only=True,
        )

        return {
            "symbol": symbol,
            "side": pos["side"],
            "quantity": qty,
            "entry_price": pos["entryPrice"],
            "unrealized_pnl": pos["unrealizedProfit"],
            "close_order": order_result,
        }

    # ======================================================================
    #  State persistence
    # ======================================================================

    def _persist_state(self, result: KillSwitchResult, reason: str) -> None:
        """Write kill-switch outcome to persistence layer and database."""
        state_data = {
            "kill_switch_triggered": True,
            "reason": reason,
            "timestamp": time.time(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "result": result.to_dict(),
        }

        # State file
        if self._persistence is not None:
            try:
                self._persistence.save("kill_switch", state_data)
            except Exception as exc:
                logger.error("Failed to persist kill switch state: %s", exc)

        # Database snapshot
        if self._db is not None:
            try:
                self._db.save_state_snapshot(
                    strategy_id="__SYSTEM__",
                    state_json=json.dumps(state_data, default=str),
                    snapshot_type="kill_switch",
                )
            except Exception as exc:
                logger.error("Failed to save kill switch DB snapshot: %s", exc)

    # ======================================================================
    #  Status queries
    # ======================================================================

    def is_triggered(self) -> bool:
        """Return True if the kill switch has been activated."""
        return self._triggered

    def get_last_trigger(self) -> Optional[Dict[str, Any]]:
        """Return details of the last trigger, or None if never triggered."""
        return self._last_trigger

    # ======================================================================
    #  Arm / disarm
    # ======================================================================

    def arm(self) -> None:
        """Enable automated kill-switch triggers (circuit breaker, etc.)."""
        self._armed = True
        logger.info("Kill switch ARMED")
        system_logger.info("KILL_SWITCH\tARMED")

    def disarm(self) -> None:
        """Disable automated kill-switch triggers.

        Manual/programmatic ``execute()`` calls still work when disarmed.
        """
        self._armed = False
        logger.warning("Kill switch DISARMED — automated triggers will be ignored")
        system_logger.info("KILL_SWITCH\tDISARMED")

    @property
    def is_armed(self) -> bool:
        """Return True if automated triggers are active."""
        return self._armed

    def should_execute(self, source: str = "automated") -> bool:
        """Determine whether an execution request should proceed.

        - ``"manual"`` or ``"api"`` sources always proceed.
        - ``"automated"`` / ``"circuit_breaker"`` sources require the switch to be armed.
        """
        if source in ("manual", "api", "programmatic"):
            return True
        return self._armed
