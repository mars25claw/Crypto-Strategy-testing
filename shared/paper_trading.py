"""
Paper Trading Fill Simulation Engine — STRAT-001 Section 9.

Simulates realistic order fills with slippage, partial fills, queue position
modeling, and fee tracking for backtesting and paper trading.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FillResult:
    """Result of a simulated order fill."""
    fill_price: float
    fill_quantity: float
    fees: float
    slippage_bps: float
    is_maker: bool
    timestamp_ms: int


@dataclass
class VirtualPosition:
    """Tracks a virtual open position."""
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    entry_price: float
    unrealized_pnl: float = 0.0

    def update_unrealized_pnl(self, mark_price: float) -> None:
        if self.side == "BUY":
            self.unrealized_pnl = (mark_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - mark_price) * self.quantity


class PaperTradingEngine:
    """
    Paper trading fill simulation engine.

    Simulates market, limit, stop-market, and trailing-stop-market orders
    with realistic slippage, partial fills, queue-position modeling, and fees.

    Args:
        starting_equity: Initial equity in USDT.
        maker_fee_pct: Maker fee as a percentage (e.g. 0.02 = 0.02%).
        taker_fee_pct: Taker fee as a percentage (e.g. 0.04 = 0.04%).
    """

    def __init__(
        self,
        starting_equity: float = 10000.0,
        maker_fee_pct: float = 0.02,
        taker_fee_pct: float = 0.04,
    ) -> None:
        self._equity = starting_equity
        self._starting_equity = starting_equity
        self._maker_fee_pct = maker_fee_pct / 100.0  # Convert from pct to decimal
        self._taker_fee_pct = taker_fee_pct / 100.0
        self._total_fees_paid: float = 0.0

        self._positions: Dict[str, VirtualPosition] = {}
        self._equity_curve: List[Tuple[int, float]] = [
            (self._now_ms(), starting_equity)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_book_snapshot: dict,
    ) -> FillResult:
        """
        Simulate a MARKET order by walking through order-book depth levels.

        Slippage model:
            slippage = order_size / (available_depth_at_best * 5)
            fill_price = best_price * (1 + slippage) for buys
            fill_price = best_price * (1 - slippage) for sells

        Always charged taker fee.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").
            side: "BUY" or "SELL".
            quantity: Order size in base asset.
            order_book_snapshot: Dict with "bids" and "asks", each a list of
                [price, quantity] sorted best-first.

        Returns:
            FillResult with the simulated execution details.
        """
        side = side.upper()
        levels = (
            order_book_snapshot.get("asks", [])
            if side == "BUY"
            else order_book_snapshot.get("bids", [])
        )

        if not levels:
            raise ValueError(f"No {'asks' if side == 'BUY' else 'bids'} in order book snapshot")

        fill_price, fill_qty = self._walk_book(levels, quantity, side)

        best_price = float(levels[0][0])
        slippage_bps = abs(fill_price - best_price) / best_price * 10_000 if best_price else 0.0

        notional = fill_price * fill_qty
        fees = self.apply_fees(notional, is_maker=False)
        self._total_fees_paid += fees

        self._update_position(symbol, side, fill_qty, fill_price)
        self.update_equity(-fees)
        self._record_equity()

        return FillResult(
            fill_price=fill_price,
            fill_quantity=fill_qty,
            fees=fees,
            slippage_bps=round(slippage_bps, 4),
            is_maker=False,
            timestamp_ms=self._now_ms(),
        )

    def simulate_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        current_price: float,
        order_book_snapshot: dict,
    ) -> Optional[FillResult]:
        """
        Simulate a LIMIT order.

        Fill rules:
        - Fills ONLY if the current price trades THROUGH the limit price
          (not just touches it).
        - Partial fills based on volume at that level using queue-position
          modeling: fill_proportion = our_size / total_depth_at_price.
        - If the order rests on the book (does not cross spread) -> maker fee.
        - If the order crosses the spread immediately -> taker fee.

        Args:
            symbol: Trading pair.
            side: "BUY" or "SELL".
            quantity: Order size in base asset.
            price: Limit price.
            current_price: Current market price (last trade).
            order_book_snapshot: Dict with "bids" and "asks".

        Returns:
            FillResult if the order fills (fully or partially), None otherwise.
        """
        side = side.upper()
        crosses_spread = self._crosses_spread(side, price, order_book_snapshot)

        # Check if price trades through the limit
        if side == "BUY":
            if current_price >= price:
                return None  # Price hasn't traded through
        else:  # SELL
            if current_price <= price:
                return None  # Price hasn't traded through

        # Determine fill quantity via queue-position modeling
        depth_at_price = self._depth_at_price(side, price, order_book_snapshot)
        total_depth = depth_at_price + quantity  # Our order joins the queue

        if total_depth > 0:
            fill_proportion = quantity / total_depth
        else:
            fill_proportion = 1.0

        # Volume available at the price level determines partial fill
        fill_qty = min(quantity, quantity * fill_proportion + depth_at_price * fill_proportion)
        fill_qty = min(fill_qty, quantity)
        fill_qty = max(fill_qty, quantity * 0.1)  # At least 10% fills in sim

        is_maker = not crosses_spread
        notional = price * fill_qty
        fees = self.apply_fees(notional, is_maker=is_maker)
        self._total_fees_paid += fees

        slippage_bps = 0.0  # Limit orders fill at limit price

        self._update_position(symbol, side, fill_qty, price)
        self.update_equity(-fees)
        self._record_equity()

        return FillResult(
            fill_price=price,
            fill_quantity=round(fill_qty, 8),
            fees=fees,
            slippage_bps=slippage_bps,
            is_maker=is_maker,
            timestamp_ms=self._now_ms(),
        )

    def simulate_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        mark_price: float,
        order_book_snapshot: dict,
    ) -> Optional[FillResult]:
        """
        Simulate a STOP_MARKET order.

        Triggers when the mark price hits the stop level, then fills as a
        MARKET order with slippage simulation.

        Args:
            symbol: Trading pair.
            side: "BUY" or "SELL".
            quantity: Order size in base asset.
            stop_price: Stop trigger price.
            mark_price: Current mark price.
            order_book_snapshot: Dict with "bids" and "asks".

        Returns:
            FillResult if stop is triggered, None otherwise.
        """
        side = side.upper()
        triggered = False

        if side == "BUY" and mark_price >= stop_price:
            triggered = True
        elif side == "SELL" and mark_price <= stop_price:
            triggered = True

        if not triggered:
            return None

        return self.simulate_market_order(symbol, side, quantity, order_book_snapshot)

    def check_trailing_stop(
        self,
        side: str,
        callback_rate: float,
        highest_price: float,
        lowest_price: float,
        current_price: float,
    ) -> bool:
        """
        Check if a trailing stop has been triggered.

        For a SELL trailing stop (long position protection):
            Triggered when current_price retraces from highest_price by
            callback_rate or more.

        For a BUY trailing stop (short position protection):
            Triggered when current_price retraces from lowest_price by
            callback_rate or more.

        Args:
            side: "BUY" or "SELL" — the side of the stop order.
            callback_rate: Callback rate as a decimal (e.g. 0.01 = 1%).
            highest_price: Highest price observed since the stop was placed.
            lowest_price: Lowest price observed since the stop was placed.
            current_price: Current market price.

        Returns:
            True if the trailing stop is triggered.
        """
        side = side.upper()

        if side == "SELL":
            # Protecting a long: triggers when price drops callback_rate from high
            if highest_price <= 0:
                return False
            retrace = (highest_price - current_price) / highest_price
            return retrace >= callback_rate
        else:
            # Protecting a short: triggers when price rises callback_rate from low
            if lowest_price <= 0:
                return False
            retrace = (current_price - lowest_price) / lowest_price
            return retrace >= callback_rate

    def apply_fees(self, notional: float, is_maker: bool) -> float:
        """
        Calculate fees for a given notional value.

        Args:
            notional: Trade notional in USDT.
            is_maker: True for maker fee, False for taker fee.

        Returns:
            Fee amount in USDT.
        """
        rate = self._maker_fee_pct if is_maker else self._taker_fee_pct
        return round(notional * rate, 8)

    def get_equity(self) -> float:
        """Return current equity including unrealized PnL."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return self._equity + unrealized

    def update_equity(self, delta: float) -> None:
        """
        Adjust equity by delta (positive = gain, negative = loss/fee).

        Args:
            delta: Amount to add to equity.
        """
        self._equity += delta

    def get_equity_curve(self) -> List[Tuple[int, float]]:
        """Return equity curve as list of (timestamp_ms, equity)."""
        return list(self._equity_curve)

    def get_positions(self) -> Dict[str, VirtualPosition]:
        """Return all virtual positions."""
        return dict(self._positions)

    def get_position(self, symbol: str) -> Optional[VirtualPosition]:
        """Return a specific virtual position or None."""
        return self._positions.get(symbol)

    def update_position_pnl(self, symbol: str, mark_price: float) -> None:
        """Update unrealized PnL for a position given current mark price."""
        pos = self._positions.get(symbol)
        if pos:
            pos.update_unrealized_pnl(mark_price)

    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """
        Close a virtual position and realize PnL.

        Args:
            symbol: The trading pair.
            exit_price: The price at which to close.

        Returns:
            Realized PnL or None if no position exists.
        """
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return None

        if pos.side == "BUY":
            realized_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            realized_pnl = (pos.entry_price - exit_price) * pos.quantity

        self.update_equity(realized_pnl)
        self._record_equity()
        return realized_pnl

    @property
    def total_fees_paid(self) -> float:
        """Total fees paid across all simulated fills."""
        return self._total_fees_paid

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _walk_book(
        self,
        levels: list,
        quantity: float,
        side: str,
    ) -> Tuple[float, float]:
        """
        Walk through order book depth levels to simulate market order fill.

        Uses slippage model: slippage = order_size / (depth_at_best × 5).

        Returns (weighted_avg_fill_price, filled_quantity).
        """
        if not levels:
            raise ValueError("Empty order book levels")

        best_price = float(levels[0][0])
        best_depth = float(levels[0][1])

        # Calculate slippage factor
        if best_depth > 0:
            slippage_factor = quantity / (best_depth * 5.0)
        else:
            slippage_factor = 0.01  # Default 1% if no depth info

        slippage_factor = min(slippage_factor, 0.05)  # Cap at 5%

        remaining = quantity
        total_cost = 0.0
        filled = 0.0

        for level_price_raw, level_qty_raw in levels:
            if remaining <= 0:
                break

            level_price = float(level_price_raw)
            level_qty = float(level_qty_raw)
            take = min(remaining, level_qty)

            total_cost += take * level_price
            filled += take
            remaining -= take

        if filled <= 0:
            # Not enough depth — fill at best with max slippage
            fill_price = best_price * (1.0 + slippage_factor if side == "BUY" else 1.0 - slippage_factor)
            return fill_price, quantity

        vwap = total_cost / filled

        # Apply slippage on top of VWAP
        if side == "BUY":
            fill_price = vwap * (1.0 + slippage_factor)
        else:
            fill_price = vwap * (1.0 - slippage_factor)

        return round(fill_price, 8), round(filled, 8)

    def _crosses_spread(
        self, side: str, price: float, order_book_snapshot: dict
    ) -> bool:
        """Check if a limit order crosses the spread (acts as taker)."""
        if side == "BUY":
            asks = order_book_snapshot.get("asks", [])
            if asks:
                best_ask = float(asks[0][0])
                return price >= best_ask
        else:
            bids = order_book_snapshot.get("bids", [])
            if bids:
                best_bid = float(bids[0][0])
                return price <= best_bid
        return False

    def _depth_at_price(
        self, side: str, price: float, order_book_snapshot: dict
    ) -> float:
        """Get total depth at a specific price level from the order book."""
        if side == "BUY":
            levels = order_book_snapshot.get("bids", [])
        else:
            levels = order_book_snapshot.get("asks", [])

        total = 0.0
        tolerance = price * 1e-8  # Floating-point tolerance
        for level_price_raw, level_qty_raw in levels:
            if abs(float(level_price_raw) - price) <= tolerance:
                total += float(level_qty_raw)
        return total

    def _update_position(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> None:
        """Update or create a virtual position after a fill."""
        existing = self._positions.get(symbol)

        if existing is None:
            self._positions[symbol] = VirtualPosition(
                symbol=symbol, side=side, quantity=quantity, entry_price=price
            )
            return

        if existing.side == side:
            # Adding to position — weighted average entry
            total_qty = existing.quantity + quantity
            existing.entry_price = (
                (existing.entry_price * existing.quantity + price * quantity) / total_qty
            )
            existing.quantity = total_qty
        else:
            # Reducing or flipping position
            if quantity >= existing.quantity:
                # Close existing, possibly open opposite
                if existing.side == "BUY":
                    realized = (price - existing.entry_price) * existing.quantity
                else:
                    realized = (existing.entry_price - price) * existing.quantity
                self.update_equity(realized)

                remaining = quantity - existing.quantity
                if remaining > 0:
                    self._positions[symbol] = VirtualPosition(
                        symbol=symbol, side=side, quantity=remaining, entry_price=price
                    )
                else:
                    del self._positions[symbol]
            else:
                # Partial close
                if existing.side == "BUY":
                    realized = (price - existing.entry_price) * quantity
                else:
                    realized = (existing.entry_price - price) * quantity
                self.update_equity(realized)
                existing.quantity -= quantity

    def _record_equity(self) -> None:
        """Snapshot current equity to the equity curve."""
        self._equity_curve.append((self._now_ms(), self.get_equity()))

    @staticmethod
    def _now_ms() -> int:
        """Current time in milliseconds."""
        return int(time.time() * 1000)
