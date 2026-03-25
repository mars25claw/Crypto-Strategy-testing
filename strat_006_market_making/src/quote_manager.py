"""Quote placement and management for STRAT-006 Market Making.

Handles:
- Placing and updating bid/ask quotes at multiple layers
- Cancel-replace when prices change > 0.01%
- Layered quoting: L1 (40%), L2 (35%), L3 (25%)
- Inventory-adjusted sizing
- Fill handling with round-trip tracking
- Paper trading fill simulation with queue-position modeling
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ActiveQuote:
    """Tracks a single active quote (order) on the exchange."""
    order_id: int = 0
    symbol: str = ""
    side: str = ""           # "BUY" or "SELL"
    price: float = 0.0
    quantity: float = 0.0
    layer: int = 1           # 1, 2, or 3
    placed_at: float = 0.0   # Timestamp
    is_paper: bool = False
    client_order_id: str = ""


@dataclass
class FillRecord:
    """Record of a single fill."""
    symbol: str
    side: str                # "BUY" or "SELL"
    price: float
    quantity: float
    mid_price_at_fill: float
    timestamp: float
    order_id: int = 0
    fee: float = 0.0
    layer: int = 1
    is_maker: bool = True
    # For adverse selection tracking — mid price 1s after fill
    mid_price_1s_after: Optional[float] = None

    @property
    def spread_captured(self) -> float:
        """Spread captured = fill_price - mid_price (positive for profitable fills)."""
        if self.side == "SELL":
            return self.price - self.mid_price_at_fill
        else:
            return self.mid_price_at_fill - self.price

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "price": self.price,
            "quantity": self.quantity,
            "mid_price_at_fill": self.mid_price_at_fill,
            "timestamp": self.timestamp,
            "order_id": self.order_id,
            "fee": self.fee,
            "layer": self.layer,
            "is_maker": self.is_maker,
            "spread_captured": self.spread_captured,
            "mid_price_1s_after": self.mid_price_1s_after,
        }


@dataclass
class RoundTrip:
    """A completed round-trip (buy + sell)."""
    symbol: str
    buy_price: float
    sell_price: float
    quantity: float
    gross_profit: float
    net_profit: float
    buy_fee: float
    sell_fee: float
    buy_timestamp: float
    sell_timestamp: float
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "quantity": self.quantity,
            "gross_profit": self.gross_profit,
            "net_profit": self.net_profit,
            "total_fees": self.buy_fee + self.sell_fee,
            "duration_s": self.duration_s,
        }


# ---------------------------------------------------------------------------
# QuoteManager
# ---------------------------------------------------------------------------

class QuoteManager:
    """Manages quote placement, updates, and fill handling.

    Parameters
    ----------
    params : dict
        Strategy parameters from config.yaml.
    binance_client : object
        BinanceClient instance for order operations.
    exchange_info : object
        ExchangeInfo instance for precision.
    paper_mode : bool
        If True, use paper trading simulation.
    maker_fee_pct : float
        Maker fee as decimal (e.g. 0.0002 for 0.02%).
    """

    def __init__(
        self,
        params: dict,
        binance_client: Any = None,
        exchange_info: Any = None,
        paper_mode: bool = True,
        maker_fee_pct: float = 0.0002,
    ) -> None:
        self._params = params
        self._client = binance_client
        self._exchange_info = exchange_info
        self._paper_mode = paper_mode
        self._maker_fee_pct = maker_fee_pct

        # Layer sizing
        self._layer1_pct = params.get("layer1_pct", 0.40)
        self._layer2_pct = params.get("layer2_pct", 0.35)
        self._layer3_pct = params.get("layer3_pct", 0.25)
        self._quote_capital_fraction = params.get("quote_capital_fraction", 0.2)

        # Active quotes: symbol -> {layer_side_key -> ActiveQuote}
        # key format: "BUY_1", "SELL_2", etc.
        self._active_quotes: Dict[str, Dict[str, ActiveQuote]] = {}

        # Fill history (last 1000)
        self._fills: Deque[FillRecord] = deque(maxlen=1000)
        self._round_trips: Deque[RoundTrip] = deque(maxlen=500)

        # Pending buy fills awaiting matching sell (for round-trip tracking)
        # symbol -> list of (price, qty, fee, timestamp)
        self._pending_buys: Dict[str, List[Tuple[float, float, float, float]]] = {}

        # Fill counters per symbol
        self._fill_count: Dict[str, int] = {}

        # Same-side fill tracking for protection
        # symbol -> deque of (side, timestamp)
        self._recent_fill_sides: Dict[str, Deque[Tuple[str, float]]] = {}

        # Paper trading order ID counter
        self._paper_order_id = 100000

        # Orders placed / cancelled counters (for API efficiency tracking)
        self._orders_placed = 0
        self._orders_cancelled = 0
        self._orders_filled = 0

        logger.info(
            "QuoteManager initialized: paper=%s, layers=%.0f%%/%.0f%%/%.0f%%, "
            "capital_fraction=%.1f%%",
            paper_mode,
            self._layer1_pct * 100, self._layer2_pct * 100, self._layer3_pct * 100,
            self._quote_capital_fraction * 100,
        )

    # ------------------------------------------------------------------
    # Quote sizing
    # ------------------------------------------------------------------

    def calculate_base_size(self, symbol: str, allocated_capital: float,
                            current_price: float) -> float:
        """Calculate base quote size.

        Base_Quote_Size = (Allocated_Capital * 0.2) / Current_Price
        """
        if current_price <= 0:
            return 0.0
        base = (allocated_capital * self._quote_capital_fraction) / current_price
        return base

    def calculate_layer_sizes(
        self,
        base_size: float,
        inventory_qty: float,
        max_inventory_notional: float,
        mid_price: float,
    ) -> Dict[str, Dict[int, float]]:
        """Calculate bid and ask sizes for each layer, adjusted for inventory.

        When LONG inventory:
            Bid_Size = Base * (1 - inventory_pct * 2)  [min 0.1x]
            Ask_Size = Base * (1 + inventory_pct * 2)  [max 2x]
        When SHORT: mirror.

        Returns dict: {"BUY": {1: qty, 2: qty, 3: qty}, "SELL": {1: ..., ...}}
        """
        if base_size <= 0 or mid_price <= 0:
            return {"BUY": {1: 0, 2: 0, 3: 0}, "SELL": {1: 0, 2: 0, 3: 0}}

        inv_notional = abs(inventory_qty) * mid_price
        inv_pct = inv_notional / max_inventory_notional if max_inventory_notional > 0 else 0.0

        if inventory_qty > 0:
            # Long inventory: reduce bids, increase asks
            bid_mult = max(0.1, 1.0 - inv_pct * 2.0)
            ask_mult = min(2.0, 1.0 + inv_pct * 2.0)
        elif inventory_qty < 0:
            # Short inventory: increase bids, reduce asks
            bid_mult = min(2.0, 1.0 + inv_pct * 2.0)
            ask_mult = max(0.1, 1.0 - inv_pct * 2.0)
        else:
            bid_mult = 1.0
            ask_mult = 1.0

        bid_base = base_size * bid_mult
        ask_base = base_size * ask_mult

        sizes = {
            "BUY": {
                1: bid_base * self._layer1_pct,
                2: bid_base * self._layer2_pct,
                3: bid_base * self._layer3_pct,
            },
            "SELL": {
                1: ask_base * self._layer1_pct,
                2: ask_base * self._layer2_pct,
                3: ask_base * self._layer3_pct,
            },
        }

        return sizes

    # ------------------------------------------------------------------
    # Quote placement
    # ------------------------------------------------------------------

    async def place_quotes(
        self,
        symbol: str,
        quotes: Any,  # QuotePrices from strategy
        sizes: Dict[str, Dict[int, float]],
        tick_size: float,
        step_size: float,
        min_notional: float,
    ) -> List[ActiveQuote]:
        """Place or update all layers of bid and ask quotes.

        Performs cancel-replace if prices differ from active quotes by > 0.01%.
        """
        placed: List[ActiveQuote] = []
        if symbol not in self._active_quotes:
            self._active_quotes[symbol] = {}

        # Build desired quotes
        desired = {
            "BUY_1": (quotes.bid_l1, sizes["BUY"][1]),
            "BUY_2": (quotes.bid_l2, sizes["BUY"][2]),
            "BUY_3": (quotes.bid_l3, sizes["BUY"][3]),
            "SELL_1": (quotes.ask_l1, sizes["SELL"][1]),
            "SELL_2": (quotes.ask_l2, sizes["SELL"][2]),
            "SELL_3": (quotes.ask_l3, sizes["SELL"][3]),
        }

        mid = (quotes.bid_l1 + quotes.ask_l1) / 2.0 if quotes.bid_l1 > 0 and quotes.ask_l1 > 0 else 1.0

        for key, (price, qty) in desired.items():
            side, layer_str = key.split("_")
            layer = int(layer_str)

            if price <= 0 or qty <= 0:
                continue

            # Round quantity to step size
            if step_size > 0:
                qty = math.floor(qty / step_size) * step_size
            if qty <= 0:
                continue

            # Check min notional
            if price * qty < min_notional:
                # Try to increase to minimum
                if min_notional / price > 0:
                    qty = math.ceil(min_notional / price / step_size) * step_size if step_size > 0 else min_notional / price
                else:
                    continue

            # Check if existing quote needs update
            existing = self._active_quotes[symbol].get(key)
            if existing is not None:
                price_change = abs(existing.price - price) / mid
                if price_change <= self._params.get("quote_update_threshold", 0.0001):
                    continue  # No meaningful change, keep existing
                # Cancel existing before placing new
                await self._cancel_quote(symbol, key, existing)

            # Place new quote
            quote = await self._place_single_quote(
                symbol, side, price, qty, layer, tick_size,
            )
            if quote is not None:
                self._active_quotes[symbol][key] = quote
                placed.append(quote)

        return placed

    async def cancel_all_quotes(self, symbol: str) -> int:
        """Cancel all active quotes for a symbol. Returns count cancelled."""
        quotes = self._active_quotes.get(symbol, {})
        cancelled = 0
        keys_to_remove = list(quotes.keys())

        for key in keys_to_remove:
            quote = quotes[key]
            try:
                await self._cancel_quote(symbol, key, quote)
                cancelled += 1
            except Exception as e:
                logger.error("Failed to cancel quote %s/%s: %s", symbol, key, e)

        self._active_quotes[symbol] = {}
        return cancelled

    async def cancel_all_instruments(self) -> int:
        """Cancel all quotes on all instruments."""
        total = 0
        for symbol in list(self._active_quotes.keys()):
            total += await self.cancel_all_quotes(symbol)
        return total

    async def cancel_side(self, symbol: str, side: str) -> int:
        """Cancel all quotes on one side (BUY or SELL) for a symbol."""
        quotes = self._active_quotes.get(symbol, {})
        cancelled = 0
        keys_to_remove = [k for k in quotes if k.startswith(side)]

        for key in keys_to_remove:
            quote = quotes[key]
            try:
                await self._cancel_quote(symbol, key, quote)
                cancelled += 1
            except Exception as e:
                logger.error("Failed to cancel %s quote %s/%s: %s", side, symbol, key, e)
            quotes.pop(key, None)

        return cancelled

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def handle_fill(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        mid_price: float,
        order_id: int = 0,
        fee: float = 0.0,
        is_maker: bool = True,
        layer: int = 1,
    ) -> FillRecord:
        """Process a fill event.

        - Records the fill
        - Tracks round-trips
        - Logs to trade log
        """
        fill = FillRecord(
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            mid_price_at_fill=mid_price,
            timestamp=time.time(),
            order_id=order_id,
            fee=fee if fee > 0 else abs(price * quantity * self._maker_fee_pct),
            layer=layer,
            is_maker=is_maker,
        )

        self._fills.append(fill)
        self._fill_count[symbol] = self._fill_count.get(symbol, 0) + 1
        self._orders_filled += 1

        # Track same-side fills
        if symbol not in self._recent_fill_sides:
            self._recent_fill_sides[symbol] = deque(maxlen=20)
        self._recent_fill_sides[symbol].append((side, time.time()))

        # Round-trip tracking
        if side == "BUY":
            self._pending_buys.setdefault(symbol, []).append(
                (price, quantity, fill.fee, fill.timestamp)
            )
        elif side == "SELL" and symbol in self._pending_buys:
            # Match against oldest pending buy (FIFO)
            self._match_round_trip(symbol, price, quantity, fill.fee, fill.timestamp)

        # Remove filled quote from active
        self._remove_filled_quote(symbol, side, order_id)

        trade_logger.info(
            "FILL\tsymbol=%s\tside=%s\tprice=%.8f\tqty=%.8f\tmid=%.8f\t"
            "spread_captured=%.8f\tfee=%.8f\tlayer=%d",
            symbol, side, price, quantity, mid_price,
            fill.spread_captured, fill.fee, layer,
        )

        return fill

    def _match_round_trip(self, symbol: str, sell_price: float,
                          sell_qty: float, sell_fee: float,
                          sell_timestamp: float) -> None:
        """Match a sell fill against pending buys to create round-trips."""
        pending = self._pending_buys.get(symbol, [])
        remaining_sell = sell_qty

        while remaining_sell > 0 and pending:
            buy_price, buy_qty, buy_fee, buy_ts = pending[0]

            match_qty = min(remaining_sell, buy_qty)
            gross = (sell_price - buy_price) * match_qty
            net = gross - buy_fee * (match_qty / buy_qty) - sell_fee * (match_qty / sell_qty)

            rt = RoundTrip(
                symbol=symbol,
                buy_price=buy_price,
                sell_price=sell_price,
                quantity=match_qty,
                gross_profit=gross,
                net_profit=net,
                buy_fee=buy_fee * (match_qty / buy_qty),
                sell_fee=sell_fee * (match_qty / sell_qty),
                buy_timestamp=buy_ts,
                sell_timestamp=sell_timestamp,
                duration_s=sell_timestamp - buy_ts,
            )
            self._round_trips.append(rt)

            trade_logger.info(
                "ROUND_TRIP\tsymbol=%s\tbuy=%.8f\tsell=%.8f\tqty=%.8f\t"
                "gross=%.8f\tnet=%.8f\tduration=%.1fs",
                symbol, buy_price, sell_price, match_qty, gross, net,
                rt.duration_s,
            )

            remaining_sell -= match_qty
            if match_qty >= buy_qty:
                pending.pop(0)
            else:
                pending[0] = (buy_price, buy_qty - match_qty, buy_fee * (1 - match_qty / buy_qty), buy_ts)

    def _remove_filled_quote(self, symbol: str, side: str, order_id: int) -> None:
        """Remove a filled quote from active quotes."""
        quotes = self._active_quotes.get(symbol, {})
        for key, quote in list(quotes.items()):
            if quote.side == side and (quote.order_id == order_id or order_id == 0):
                del quotes[key]
                break

    # ------------------------------------------------------------------
    # Same-side fill protection
    # ------------------------------------------------------------------

    def check_same_side_fills(self, symbol: str) -> Optional[str]:
        """Check if 3+ fills on same side in 10 seconds.

        Returns the accumulating side to withdraw, or None.
        """
        recent = self._recent_fill_sides.get(symbol)
        if not recent:
            return None

        now = time.time()
        window = self._params.get("same_side_fill_window_seconds", 10)
        threshold = self._params.get("same_side_fill_count", 3)

        recent_in_window = [(s, t) for s, t in recent if now - t < window]
        if len(recent_in_window) < threshold:
            return None

        buy_count = sum(1 for s, _ in recent_in_window if s == "BUY")
        sell_count = sum(1 for s, _ in recent_in_window if s == "SELL")

        if buy_count >= threshold:
            return "BUY"  # Withdraw buy side
        if sell_count >= threshold:
            return "SELL"  # Withdraw sell side
        return None

    # ------------------------------------------------------------------
    # Paper trading fill simulation
    # ------------------------------------------------------------------

    def check_paper_fills(
        self,
        symbol: str,
        trade_price: float,
        trade_quantity: float,
        depth_at_bid: float,
        depth_at_ask: float,
    ) -> List[FillRecord]:
        """Check if any paper quotes would have been filled by a trade.

        Simulates queue position: only fill proportional to
        our_size / total_depth_at_price.
        """
        fills = []
        quotes = self._active_quotes.get(symbol, {})

        for key, quote in list(quotes.items()):
            if not quote.is_paper:
                continue

            filled = False
            fill_qty = 0.0

            if quote.side == "BUY" and trade_price <= quote.price:
                # Bid fill: trade must hit our price
                total_depth = depth_at_bid + quote.quantity
                if total_depth > 0:
                    our_fraction = quote.quantity / total_depth
                    fill_qty = min(quote.quantity, trade_quantity * our_fraction)
                else:
                    fill_qty = min(quote.quantity, trade_quantity)
                fill_qty = max(fill_qty, quote.quantity * 0.1)  # Min 10% in sim
                fill_qty = min(fill_qty, quote.quantity)
                filled = True

            elif quote.side == "SELL" and trade_price >= quote.price:
                # Ask fill: trade must lift our price
                total_depth = depth_at_ask + quote.quantity
                if total_depth > 0:
                    our_fraction = quote.quantity / total_depth
                    fill_qty = min(quote.quantity, trade_quantity * our_fraction)
                else:
                    fill_qty = min(quote.quantity, trade_quantity)
                fill_qty = max(fill_qty, quote.quantity * 0.1)
                fill_qty = min(fill_qty, quote.quantity)
                filled = True

            if filled and fill_qty > 0:
                mid = (quote.price + trade_price) / 2.0  # Approximate mid at fill
                fill = self.handle_fill(
                    symbol=symbol,
                    side=quote.side,
                    price=quote.price,
                    quantity=fill_qty,
                    mid_price=mid,
                    order_id=quote.order_id,
                    fee=abs(quote.price * fill_qty * self._maker_fee_pct),
                    is_maker=True,
                    layer=quote.layer,
                )
                fills.append(fill)

                # Handle partial fills
                remaining = quote.quantity - fill_qty
                if remaining > 0:
                    quote.quantity = remaining
                else:
                    del quotes[key]

        return fills

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _place_single_quote(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        layer: int,
        tick_size: float,
    ) -> Optional[ActiveQuote]:
        """Place a single quote order (live or paper)."""
        if self._paper_mode:
            self._paper_order_id += 1
            quote = ActiveQuote(
                order_id=self._paper_order_id,
                symbol=symbol,
                side=side,
                price=price,
                quantity=quantity,
                layer=layer,
                placed_at=time.time(),
                is_paper=True,
                client_order_id=f"MM_{symbol}_{side}_{layer}_{self._paper_order_id}",
            )
            self._orders_placed += 1
            logger.debug(
                "Paper quote placed: %s %s L%d price=%.8f qty=%.8f",
                symbol, side, layer, price, quantity,
            )
            return quote

        # Live mode
        if self._client is None:
            logger.error("Cannot place quote: no client")
            return None

        try:
            result = await self._client.place_futures_order(
                symbol=symbol,
                side=side,
                type="LIMIT",
                quantity=quantity,
                price=price,
                post_only=True,  # GTX: maker only
            )
            order_id = result.get("orderId", 0)
            quote = ActiveQuote(
                order_id=order_id,
                symbol=symbol,
                side=side,
                price=price,
                quantity=quantity,
                layer=layer,
                placed_at=time.time(),
                is_paper=False,
                client_order_id=result.get("clientOrderId", ""),
            )
            self._orders_placed += 1
            logger.info(
                "Quote placed: %s %s L%d price=%.8f qty=%.8f orderId=%d",
                symbol, side, layer, price, quantity, order_id,
            )
            return quote

        except Exception as e:
            logger.error(
                "Failed to place quote %s %s L%d: %s",
                symbol, side, layer, e,
            )
            return None

    async def _cancel_quote(self, symbol: str, key: str,
                            quote: ActiveQuote) -> None:
        """Cancel a single quote order."""
        if self._paper_mode:
            self._orders_cancelled += 1
            logger.debug("Paper quote cancelled: %s/%s orderId=%d", symbol, key, quote.order_id)
            return

        if self._client is None:
            return

        try:
            await self._client.cancel_futures_order(symbol, quote.order_id)
            self._orders_cancelled += 1
            logger.debug("Quote cancelled: %s/%s orderId=%d", symbol, key, quote.order_id)
        except Exception as e:
            logger.warning("Failed to cancel quote %s/%s: %s", symbol, key, e)

    # ------------------------------------------------------------------
    # Status and metrics
    # ------------------------------------------------------------------

    def get_active_quotes(self, symbol: Optional[str] = None) -> Dict[str, Dict[str, dict]]:
        """Return active quotes as serializable dicts."""
        result = {}
        targets = {symbol: self._active_quotes.get(symbol, {})} if symbol else self._active_quotes
        for sym, quotes in targets.items():
            result[sym] = {
                key: {
                    "order_id": q.order_id,
                    "side": q.side,
                    "price": q.price,
                    "quantity": q.quantity,
                    "layer": q.layer,
                    "placed_at": q.placed_at,
                    "age_s": round(time.time() - q.placed_at, 1),
                }
                for key, q in quotes.items()
            }
        return result

    def get_active_quote_count(self, symbol: Optional[str] = None) -> int:
        """Return count of active quotes."""
        if symbol:
            return len(self._active_quotes.get(symbol, {}))
        return sum(len(q) for q in self._active_quotes.values())

    def get_recent_fills(self, limit: int = 50) -> List[dict]:
        """Return recent fills as dicts."""
        fills = list(self._fills)[-limit:]
        return [f.to_dict() for f in fills]

    def get_recent_round_trips(self, limit: int = 50) -> List[dict]:
        """Return recent round trips as dicts."""
        rts = list(self._round_trips)[-limit:]
        return [rt.to_dict() for rt in rts]

    def get_fill_rate(self, symbol: Optional[str] = None) -> float:
        """Return fill rate: fills / orders placed."""
        if self._orders_placed == 0:
            return 0.0
        return self._orders_filled / self._orders_placed

    def get_quote_staleness(self, symbol: str) -> Optional[float]:
        """Return age of oldest active quote in seconds."""
        quotes = self._active_quotes.get(symbol, {})
        if not quotes:
            return None
        now = time.time()
        return max(now - q.placed_at for q in quotes.values())

    def get_metrics(self) -> Dict[str, Any]:
        """Return quote management metrics."""
        fills = list(self._fills)
        rts = list(self._round_trips)

        total_spread_captured = sum(f.spread_captured for f in fills) if fills else 0
        avg_spread_captured = total_spread_captured / len(fills) if fills else 0

        total_rt_pnl = sum(rt.net_profit for rt in rts) if rts else 0
        avg_rt_pnl = total_rt_pnl / len(rts) if rts else 0
        win_rts = sum(1 for rt in rts if rt.net_profit > 0) if rts else 0

        return {
            "orders_placed": self._orders_placed,
            "orders_cancelled": self._orders_cancelled,
            "orders_filled": self._orders_filled,
            "fill_rate": self.get_fill_rate(),
            "order_to_fill_ratio": self._orders_placed / max(1, self._orders_filled),
            "total_fills": len(fills),
            "avg_spread_captured": avg_spread_captured,
            "total_spread_captured": total_spread_captured,
            "round_trips": len(rts),
            "round_trip_pnl": total_rt_pnl,
            "avg_round_trip_pnl": avg_rt_pnl,
            "round_trip_win_rate": win_rts / max(1, len(rts)),
            "active_quotes": self.get_active_quote_count(),
        }

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return state for persistence."""
        return {
            "active_quotes": {
                sym: {
                    key: {
                        "order_id": q.order_id,
                        "side": q.side,
                        "price": q.price,
                        "quantity": q.quantity,
                        "layer": q.layer,
                        "placed_at": q.placed_at,
                    }
                    for key, q in quotes.items()
                }
                for sym, quotes in self._active_quotes.items()
            },
            "fills": [f.to_dict() for f in list(self._fills)[-100:]],
            "round_trips": [rt.to_dict() for rt in list(self._round_trips)[-50:]],
            "counters": {
                "orders_placed": self._orders_placed,
                "orders_cancelled": self._orders_cancelled,
                "orders_filled": self._orders_filled,
            },
        }
