"""STRAT-001 Scale-In Logic — Section 6.3.

Rules:
  - Only if position is currently profitable
  - Pullback to EMA(20) 4h within an established trend
  - Max 2 additions (total 3 entries): 100% / 50% / 25%
  - Recalculate avg entry + adjust stop after each
  - Re-verify all exposure limits after each addition
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from shared.binance_client import BinanceClient, BinanceClientError
from shared.alerting import AlertLevel

from . import STRATEGY_ID
from .strategy import SignalDirection, _is_valid
from .exit_manager import ExitManager, ManagedPosition

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


@dataclass
class ScaleInOpportunity:
    """A detected scale-in opportunity."""
    symbol: str
    direction: SignalDirection
    current_price: float
    ema_20_4h: float
    original_size: float     # initial entry size
    proposed_qty: float      # quantity to add
    scale_in_number: int     # 1 or 2


class ScalingManager:
    """Manages scale-in logic for STRAT-001 positions.

    Parameters
    ----------
    config : dict
        ``strategy_params`` from config.yaml.
    client : BinanceClient
        Shared REST client.
    exit_manager : ExitManager
        For accessing and modifying open positions.
    risk_manager : object
        For re-verifying exposure limits.
    alerter : object
        Shared alerting system.
    paper_engine : object | None
        Paper trading engine if applicable.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        client: BinanceClient,
        exit_manager: ExitManager,
        risk_manager: Any,
        alerter: Any,
        paper_engine: Any = None,
    ) -> None:
        self.cfg = config
        self.client = client
        self.exit_mgr = exit_manager
        self.risk_mgr = risk_manager
        self.alerter = alerter
        self.paper = paper_engine

        # Track original entry sizes: symbol -> initial_qty
        self._original_sizes: Dict[str, float] = {}

    def register_entry(self, symbol: str, quantity: float) -> None:
        """Record the original entry size for scale-in calculations."""
        self._original_sizes[symbol] = quantity

    def unregister(self, symbol: str) -> None:
        """Remove tracking when position is fully closed."""
        self._original_sizes.pop(symbol, None)

    # ======================================================================
    # Scale-in evaluation
    # ======================================================================

    def evaluate_scale_in(
        self,
        symbol: str,
        current_price: float,
        ema_20_4h: float,
    ) -> Optional[ScaleInOpportunity]:
        """Check if a scale-in is appropriate for *symbol*.

        Returns a :class:`ScaleInOpportunity` if conditions are met, else None.
        """
        pos = self.exit_mgr.positions.get(symbol)
        if pos is None:
            return None

        max_scale_ins = self.cfg.get("max_scale_ins", 2)

        # Max 2 additions
        if pos.scale_in_count >= max_scale_ins:
            return None

        # Must be profitable
        if pos.unrealized_pnl <= 0:
            return None

        # Price must have pulled back to EMA(20) on 4h
        if not _is_valid(ema_20_4h):
            return None

        pullback_valid = False
        if pos.direction == SignalDirection.LONG:
            # Price should be near or at EMA20 from above
            tolerance = pos.atr_at_entry * 0.3  # within 0.3 ATR of EMA20
            if current_price <= ema_20_4h + tolerance and current_price >= ema_20_4h - tolerance:
                pullback_valid = True
        else:
            tolerance = pos.atr_at_entry * 0.3
            if current_price >= ema_20_4h - tolerance and current_price <= ema_20_4h + tolerance:
                pullback_valid = True

        if not pullback_valid:
            return None

        # Calculate addition size
        original_size = self._original_sizes.get(symbol, pos.total_quantity)
        scale_in_number = pos.scale_in_count + 1

        if scale_in_number == 1:
            pct = self.cfg.get("scale_in_1_pct", 50) / 100.0
        elif scale_in_number == 2:
            pct = self.cfg.get("scale_in_2_pct", 25) / 100.0
        else:
            return None

        proposed_qty = round(original_size * pct, 8)
        if proposed_qty <= 0:
            return None

        return ScaleInOpportunity(
            symbol=symbol,
            direction=pos.direction,
            current_price=current_price,
            ema_20_4h=ema_20_4h,
            original_size=original_size,
            proposed_qty=proposed_qty,
            scale_in_number=scale_in_number,
        )

    # ======================================================================
    # Execute scale-in
    # ======================================================================

    async def execute_scale_in(self, opp: ScaleInOpportunity) -> bool:
        """Execute the scale-in order and adjust the position.

        Returns True if successful.
        """
        symbol = opp.symbol
        pos = self.exit_mgr.positions.get(symbol)
        if pos is None:
            logger.warning("Scale-in for %s: position no longer exists", symbol)
            return False

        # Re-verify with risk manager
        allowed, reason = self.risk_mgr.check_entry_allowed(
            strategy_id=STRATEGY_ID,
            symbol=symbol,
            direction=pos.direction.value,
            size_usdt=opp.proposed_qty * opp.current_price,
            leverage=pos.leverage,
        )
        if not allowed:
            logger.info("Scale-in for %s rejected by risk manager: %s", symbol, reason)
            return False

        # Place the order
        side = "BUY" if pos.direction == SignalDirection.LONG else "SELL"

        try:
            if self.paper:
                fill_price = opp.current_price
                fill_qty = opp.proposed_qty
                logger.info(
                    "PAPER SCALE-IN %s: %s qty=%.6f price=%.4f",
                    symbol, side, fill_qty, fill_price,
                )
            else:
                order = await self.client.place_futures_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=opp.proposed_qty,
                )
                fill_price = float(order.get("avgPrice", opp.current_price))
                fill_qty = float(order.get("executedQty", opp.proposed_qty))
        except BinanceClientError as e:
            logger.error("Scale-in order for %s failed: %s", symbol, e)
            return False

        # Update position state
        old_total = pos.total_quantity
        old_avg = pos.avg_entry_price

        pos.total_quantity += fill_qty
        pos.remaining_quantity += fill_qty
        pos.avg_entry_price = (
            (old_avg * old_total + fill_price * fill_qty) / pos.total_quantity
        )
        pos.scale_in_count += 1

        # Adjust hard stop to maintain risk constraint on TOTAL position
        hard_stop_mult = self.cfg.get("hard_stop_atr_mult", 2.0)
        if pos.direction == SignalDirection.LONG:
            new_stop = pos.avg_entry_price - hard_stop_mult * pos.atr_at_entry
            # Stop must not be worse than current
            pos.hard_stop_price = max(pos.hard_stop_price, new_stop)
        else:
            new_stop = pos.avg_entry_price + hard_stop_mult * pos.atr_at_entry
            pos.hard_stop_price = min(pos.hard_stop_price, new_stop)

        # Update stop order on exchange
        if not self.paper and pos.hard_stop_order_id:
            stop_side = "SELL" if pos.direction == SignalDirection.LONG else "BUY"
            try:
                await self.client.cancel_futures_order(symbol, pos.hard_stop_order_id)
                new_order = await self.client.place_futures_order(
                    symbol=symbol,
                    side=stop_side,
                    type="STOP_MARKET",
                    quantity=pos.remaining_quantity,
                    stop_price=pos.hard_stop_price,
                    reduce_only=True,
                )
                pos.hard_stop_order_id = new_order.get("orderId")
            except BinanceClientError as e:
                logger.error("%s stop adjustment after scale-in failed: %s", symbol, e)

        # Update trailing stop if active
        if pos.trailing.active and pos.trailing.stop_order_id:
            try:
                if not self.paper:
                    await self.client.cancel_futures_order(symbol, pos.trailing.stop_order_id)
                    stop_side = "SELL" if pos.direction == SignalDirection.LONG else "BUY"
                    new_order = await self.client.place_futures_order(
                        symbol=symbol,
                        side=stop_side,
                        type="STOP_MARKET",
                        quantity=pos.remaining_quantity,
                        stop_price=pos.trailing.current_stop,
                        reduce_only=True,
                    )
                    pos.trailing.stop_order_id = new_order.get("orderId")
            except BinanceClientError as e:
                logger.error("%s trailing stop update after scale-in failed: %s", symbol, e)

        # Report to risk manager
        self.risk_mgr.record_position_change(
            strategy_id=STRATEGY_ID,
            symbol=symbol,
            direction=pos.direction.value,
            size_usdt=pos.remaining_quantity * pos.last_price,
            is_open=True,
        )

        trade_logger.info(
            "SCALE_IN\tsymbol=%s\tdir=%s\tqty=%.6f\tprice=%.4f\t"
            "avg_entry=%.4f\ttotal_qty=%.6f\tnew_stop=%.4f\tscale_in_num=%d",
            symbol, pos.direction.value, fill_qty, fill_price,
            pos.avg_entry_price, pos.total_quantity,
            pos.hard_stop_price, pos.scale_in_count,
        )

        if self.alerter:
            await self.alerter.send(
                f"Scale-in #{pos.scale_in_count} for {symbol} {pos.direction.value}: "
                f"+{fill_qty:.6f} @ {fill_price:.4f}, avg={pos.avg_entry_price:.4f}",
                level=AlertLevel.INFO,
                strategy_id=STRATEGY_ID,
            )

        return True
