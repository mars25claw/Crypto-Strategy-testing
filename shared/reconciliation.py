"""Startup reconciliation protocol (STRAT-001 Section 8.2).

Compares persisted local state against live Binance account state on
startup, resolving orphan positions, stale orders, and missed fills.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum wall-clock time for the full reconciliation pass.
RECONCILIATION_TIMEOUT_S = 60.0


class ReconciliationTimeout(Exception):
    """Raised when reconciliation exceeds the allowed timeout."""


@dataclass
class ReconciliationResult:
    """Summary of reconciliation outcomes."""

    orphan_positions: List[Dict[str, Any]] = field(default_factory=list)
    """Positions found on Binance but missing from local state."""

    closed_positions: List[Dict[str, Any]] = field(default_factory=list)
    """Positions in local state that no longer exist on Binance."""

    cancelled_orders: List[Dict[str, Any]] = field(default_factory=list)
    """Open orders on Binance not tracked locally — cancelled during reconciliation."""

    detected_fills: List[Dict[str, Any]] = field(default_factory=list)
    """Orders in local state that were filled on Binance during downtime."""

    discrepancies: List[Dict[str, Any]] = field(default_factory=list)
    """Any mismatch that does not fit the above categories."""

    duration_ms: int = 0
    """Total reconciliation wall-clock time in milliseconds."""

    @property
    def has_issues(self) -> bool:
        return bool(
            self.orphan_positions
            or self.closed_positions
            or self.cancelled_orders
            or self.detected_fills
            or self.discrepancies
        )


class StartupReconciler:
    """Reconcile persisted local state against live Binance account data.

    Parameters
    ----------
    binance_client : BinanceClient
        An already-started :class:`shared.binance_client.BinanceClient`.
    state_persistence : StatePersistence
        The :class:`shared.state_persistence.StatePersistence` instance
        (already loaded or about to be loaded).
    database_manager : DatabaseManager
        The :class:`shared.database.DatabaseManager` instance for
        recording reconciliation artifacts.
    logger : logging.Logger
        Logger used for structured reconciliation output.
    orphan_stop_atr_multiplier : float
        ATR multiplier for the tight stop placed on orphan positions
        (default 1.0 per spec).
    check_spot : bool
        Whether to also fetch spot account balances.
    """

    def __init__(
        self,
        binance_client: Any,
        state_persistence: Any,
        database_manager: Any,
        logger: logging.Logger,
        orphan_stop_atr_multiplier: float = 1.0,
        check_spot: bool = False,
    ) -> None:
        self._client = binance_client
        self._state = state_persistence
        self._db = database_manager
        self._log = logger
        self._orphan_stop_atr_mult = orphan_stop_atr_multiplier
        self._check_spot = check_spot

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def reconcile(self) -> ReconciliationResult:
        """Run the full startup reconciliation protocol.

        Steps
        -----
        1. Load last persisted state from disk.
        2. Query Binance for live account state.
        3. Compare and resolve discrepancies.
        4. Return :class:`ReconciliationResult`.

        Raises
        ------
        ReconciliationTimeout
            If the entire process exceeds ``RECONCILIATION_TIMEOUT_S``.
        """
        start_ms = _now_ms()
        self._log.info("=== Startup reconciliation BEGIN ===")

        try:
            result = await asyncio.wait_for(
                self._do_reconcile(start_ms),
                timeout=RECONCILIATION_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            elapsed = _now_ms() - start_ms
            self._log.error(
                "Reconciliation timed out after %d ms (limit=%ds)",
                elapsed,
                RECONCILIATION_TIMEOUT_S,
            )
            raise ReconciliationTimeout(
                f"Reconciliation exceeded {RECONCILIATION_TIMEOUT_S}s timeout"
            )

        result.duration_ms = _now_ms() - start_ms
        self._log.info(
            "=== Startup reconciliation END — %d ms  orphans=%d  closed=%d  "
            "cancelled=%d  fills=%d  discrepancies=%d ===",
            result.duration_ms,
            len(result.orphan_positions),
            len(result.closed_positions),
            len(result.cancelled_orders),
            len(result.detected_fills),
            len(result.discrepancies),
        )
        return result

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    async def _do_reconcile(self, start_ms: int) -> ReconciliationResult:
        result = ReconciliationResult()

        # ── Step 1: Load persisted state ────────────────────────────────
        local_state = self._state.load()
        local_positions: List[Dict[str, Any]] = local_state.get("positions", [])
        local_orders: List[Dict[str, Any]] = local_state.get("orders", [])
        self._log.info(
            "Local state: %d positions, %d orders  (last_save=%d)",
            len(local_positions),
            len(local_orders),
            local_state.get("last_save_timestamp_ms", 0),
        )

        # ── Step 2: Query Binance ───────────────────────────────────────
        (
            account_data,
            open_orders_binance,
            spot_account,
        ) = await self._fetch_binance_state()

        # Extract non-zero positions from the futures account response
        binance_positions = [
            p for p in account_data.get("positions", [])
            if float(p.get("positionAmt", 0)) != 0
        ]

        self._log.info(
            "Binance state: %d open positions, %d open orders",
            len(binance_positions),
            len(open_orders_binance),
        )

        # Build lookup dicts keyed by symbol
        binance_pos_by_symbol: Dict[str, Dict] = {
            p["symbol"]: p for p in binance_positions
        }
        local_pos_by_symbol: Dict[str, Dict] = {
            p["symbol"]: p for p in local_positions if p.get("symbol")
        }

        # ── Step 3a: Detect orphan positions ────────────────────────────
        # On Binance but NOT in local state
        for symbol, bpos in binance_pos_by_symbol.items():
            if symbol not in local_pos_by_symbol:
                self._log.warning(
                    "ORPHAN position detected: %s  amt=%s  entryPrice=%s",
                    symbol,
                    bpos.get("positionAmt"),
                    bpos.get("entryPrice"),
                )
                orphan_record = {
                    "symbol": symbol,
                    "position_amt": float(bpos.get("positionAmt", 0)),
                    "entry_price": float(bpos.get("entryPrice", 0)),
                    "unrealized_pnl": float(bpos.get("unrealizedProfit", 0)),
                    "mark_price": float(bpos.get("markPrice", 0)),
                    "action": "set_tight_stop",
                    "atr_multiplier": self._orphan_stop_atr_mult,
                }
                result.orphan_positions.append(orphan_record)

                # Place a tight stop-market order to protect the orphan
                await self._place_orphan_stop(bpos)

        # ── Step 3b: Detect closed positions ────────────────────────────
        # In local state but NOT on Binance
        for symbol, lpos in local_pos_by_symbol.items():
            if symbol not in binance_pos_by_symbol:
                self._log.warning(
                    "Position CLOSED during downtime: %s  local_qty=%s",
                    symbol,
                    lpos.get("quantity", lpos.get("position_amt")),
                )
                closed_record = {
                    "symbol": symbol,
                    "local_entry_price": lpos.get("entry_price", 0),
                    "local_quantity": lpos.get(
                        "quantity", lpos.get("position_amt", 0)
                    ),
                    "last_known_price": lpos.get("mark_price", lpos.get("last_price", 0)),
                    "estimated_pnl": self._estimate_pnl(lpos),
                    "action": "mark_closed",
                }
                result.closed_positions.append(closed_record)

        # ── Step 3c: Reconcile open orders ──────────────────────────────
        local_order_ids: set[int] = {
            int(o["order_id"]) for o in local_orders if o.get("order_id")
        }
        binance_order_ids: set[int] = {
            int(o["orderId"]) for o in open_orders_binance
        }
        binance_order_by_id: Dict[int, Dict] = {
            int(o["orderId"]): o for o in open_orders_binance
        }

        # Orders on Binance not tracked locally — cancel them
        unknown_order_ids = binance_order_ids - local_order_ids
        for oid in unknown_order_ids:
            border = binance_order_by_id[oid]
            self._log.warning(
                "Unknown open order on Binance — cancelling: orderId=%d  "
                "symbol=%s  side=%s  type=%s  qty=%s",
                oid,
                border.get("symbol"),
                border.get("side"),
                border.get("type"),
                border.get("origQty"),
            )
            try:
                await self._client.cancel_futures_order(
                    symbol=border["symbol"],
                    order_id=oid,
                )
                result.cancelled_orders.append({
                    "order_id": oid,
                    "symbol": border.get("symbol"),
                    "side": border.get("side"),
                    "type": border.get("type"),
                    "orig_qty": border.get("origQty"),
                    "action": "cancelled",
                })
            except Exception as exc:
                self._log.error("Failed to cancel unknown order %d: %s", oid, exc)
                result.discrepancies.append({
                    "type": "cancel_failed",
                    "order_id": oid,
                    "symbol": border.get("symbol"),
                    "error": str(exc),
                })

        # ── Step 3d: Check local orders not on Binance ──────────────────
        # These might have been filled or cancelled while we were down.
        missing_order_ids = local_order_ids - binance_order_ids
        # Collect symbols that need trade lookup
        symbols_to_check: set[str] = set()
        local_order_by_id: Dict[int, Dict] = {
            int(o["order_id"]): o for o in local_orders if o.get("order_id")
        }
        for oid in missing_order_ids:
            lorder = local_order_by_id.get(oid, {})
            sym = lorder.get("symbol")
            if sym:
                symbols_to_check.add(sym)

        # Fetch recent trades for affected symbols
        recent_trades_by_symbol: Dict[str, List[Dict]] = {}
        for sym in symbols_to_check:
            try:
                trades = await self._client.get_futures_user_trades(
                    symbol=sym, limit=100
                )
                recent_trades_by_symbol[sym] = trades
            except Exception as exc:
                self._log.error(
                    "Failed to fetch userTrades for %s: %s", sym, exc
                )

        trade_order_ids: set[int] = set()
        for trades in recent_trades_by_symbol.values():
            for t in trades:
                trade_order_ids.add(int(t.get("orderId", 0)))

        for oid in missing_order_ids:
            lorder = local_order_by_id.get(oid, {})
            if oid in trade_order_ids:
                # Order was filled during downtime
                sym = lorder.get("symbol", "")
                fill_trades = [
                    t
                    for t in recent_trades_by_symbol.get(sym, [])
                    if int(t.get("orderId", 0)) == oid
                ]
                total_qty = sum(float(t.get("qty", 0)) for t in fill_trades)
                avg_price = (
                    sum(float(t.get("price", 0)) * float(t.get("qty", 0)) for t in fill_trades)
                    / total_qty
                    if total_qty > 0
                    else 0
                )
                self._log.info(
                    "Fill detected during downtime: orderId=%d  symbol=%s  "
                    "qty=%.6f  avgPrice=%.4f",
                    oid,
                    sym,
                    total_qty,
                    avg_price,
                )
                result.detected_fills.append({
                    "order_id": oid,
                    "symbol": sym,
                    "filled_qty": total_qty,
                    "avg_price": avg_price,
                    "trade_count": len(fill_trades),
                    "action": "mark_filled",
                })
            else:
                # Order is gone and no matching trades — mark cancelled
                self._log.info(
                    "Local order missing from Binance with no fill — "
                    "marking cancelled: orderId=%d  symbol=%s",
                    oid,
                    lorder.get("symbol"),
                )
                result.detected_fills.append({
                    "order_id": oid,
                    "symbol": lorder.get("symbol", ""),
                    "filled_qty": 0,
                    "avg_price": 0,
                    "trade_count": 0,
                    "action": "mark_cancelled",
                })

        # ── Step 3e: Position size mismatches ───────────────────────────
        for symbol in local_pos_by_symbol.keys() & binance_pos_by_symbol.keys():
            lpos = local_pos_by_symbol[symbol]
            bpos = binance_pos_by_symbol[symbol]
            local_qty = float(lpos.get("quantity", lpos.get("position_amt", 0)))
            binance_qty = float(bpos.get("positionAmt", 0))
            if abs(local_qty - binance_qty) > 1e-8:
                self._log.warning(
                    "Position SIZE mismatch: %s  local=%.8f  binance=%.8f",
                    symbol,
                    local_qty,
                    binance_qty,
                )
                result.discrepancies.append({
                    "type": "position_size_mismatch",
                    "symbol": symbol,
                    "local_qty": local_qty,
                    "binance_qty": binance_qty,
                    "delta": binance_qty - local_qty,
                })

        return result

    # ------------------------------------------------------------------
    # Binance data fetching
    # ------------------------------------------------------------------

    async def _fetch_binance_state(
        self,
    ) -> tuple[Dict[str, Any], List[Dict], Optional[Dict]]:
        """Fetch account, open orders, and optionally spot balances concurrently."""
        tasks = [
            self._client.get_futures_account(),
            self._client.get_futures_open_orders(),
        ]
        if self._check_spot:
            tasks.append(self._client.get_spot_account())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        account_data = results[0]
        open_orders = results[1]
        spot_account = results[2] if self._check_spot and len(results) > 2 else None

        # Raise if critical endpoints failed
        if isinstance(account_data, BaseException):
            self._log.error("Failed to fetch futures account: %s", account_data)
            raise account_data
        if isinstance(open_orders, BaseException):
            self._log.error("Failed to fetch open orders: %s", open_orders)
            raise open_orders
        if isinstance(spot_account, BaseException):
            self._log.warning("Spot account fetch failed (non-fatal): %s", spot_account)
            spot_account = None

        return account_data, open_orders, spot_account

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _place_orphan_stop(self, binance_position: Dict[str, Any]) -> None:
        """Place a tight STOP_MARKET on an orphan position to limit risk.

        Uses ``self._orphan_stop_atr_mult`` * recent ATR as the stop
        distance.  If ATR is unavailable, defaults to 1 % of mark price.
        """
        symbol = binance_position["symbol"]
        pos_amt = float(binance_position.get("positionAmt", 0))
        mark_price = float(binance_position.get("markPrice", 0))

        if pos_amt == 0 or mark_price == 0:
            return

        # Determine side: if long, stop is SELL; if short, stop is BUY
        side = "SELL" if pos_amt > 0 else "BUY"
        qty = abs(pos_amt)

        # Try to get ATR from persisted indicator state
        indicators = self._state.get_state("indicators", {})
        symbol_indicators = indicators.get(symbol, {})
        atr_value = symbol_indicators.get("atr")

        if atr_value and float(atr_value) > 0:
            stop_distance = float(atr_value) * self._orphan_stop_atr_mult
        else:
            # Fallback: 1% of mark price
            stop_distance = mark_price * 0.01
            self._log.warning(
                "No ATR available for %s — using 1%% fallback for orphan stop",
                symbol,
            )

        if pos_amt > 0:
            stop_price = mark_price - stop_distance
        else:
            stop_price = mark_price + stop_distance

        stop_price = round(stop_price, 4)

        try:
            resp = await self._client.place_futures_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                quantity=qty,
                stop_price=stop_price,
                reduce_only=True,
            )
            self._log.info(
                "Orphan stop placed: %s %s qty=%.6f stopPrice=%.4f  orderId=%s",
                symbol,
                side,
                qty,
                stop_price,
                resp.get("orderId"),
            )
        except Exception as exc:
            self._log.error(
                "Failed to place orphan stop for %s: %s", symbol, exc
            )

    @staticmethod
    def _estimate_pnl(local_position: Dict[str, Any]) -> float:
        """Best-effort PnL estimate from local state fields."""
        entry = float(local_position.get("entry_price", 0))
        last = float(
            local_position.get("mark_price", local_position.get("last_price", 0))
        )
        qty = float(
            local_position.get("quantity", local_position.get("position_amt", 0))
        )
        if entry == 0 or last == 0:
            return 0.0
        return (last - entry) * qty


def _now_ms() -> int:
    return int(time.time() * 1000)
