"""Section 10.2 Strategy-Specific Metrics for STRAT-006 Market Making.

Computes all market-making-specific performance metrics:
- Spread Captured Per Fill
- Inventory Turnover
- Quote Uptime
- Fill Rate
- Adverse Selection Rate
- Realized vs Quoted Spread
- Inventory PnL vs Spread PnL decomposition
- Order-to-Fill Ratio
- API Efficiency (rate limit weight per $ profit)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QuoteUptimeRecord:
    """Tracks whether both bid and ask were active at a sample point."""
    timestamp: float
    bid_active: bool
    ask_active: bool
    symbol: str


@dataclass
class SpreadSample:
    """A sample of quoted vs realized spread."""
    timestamp: float
    symbol: str
    quoted_spread_pct: float  # Theoretical spread from model
    realized_spread_pct: float  # Actual fill spread captured


class StrategyMetrics:
    """Computes and maintains all Section 10.2 strategy-specific metrics.

    Parameters
    ----------
    params : dict
        Strategy parameters from config.yaml.
    """

    def __init__(self, params: dict) -> None:
        self._params = params

        # --- Spread Captured Per Fill ---
        # Tracks (spread_captured, timestamp) for each fill
        self._spread_per_fill: Deque[Tuple[float, float]] = deque(maxlen=10000)

        # --- Inventory Turnover ---
        # Total volume traded (buy + sell notional)
        self._total_volume_traded: float = 0.0
        # Rolling average inventory samples: (abs_inventory_notional, timestamp)
        self._inventory_samples: Deque[Tuple[float, float]] = deque(maxlen=8640)  # ~24h at 10s

        # --- Quote Uptime ---
        # Sample every 10 seconds: (bid_active, ask_active)
        self._uptime_records: Dict[str, Deque[QuoteUptimeRecord]] = {}  # per symbol

        # --- Fill Rate ---
        self._total_quotes_placed: int = 0
        self._total_fills: int = 0

        # --- Adverse Selection Rate ---
        # Tracked externally by AdverseSelectionTracker; we aggregate here
        self._adverse_fill_count: int = 0
        self._total_fill_count_for_adverse: int = 0

        # --- Realized vs Quoted Spread ---
        self._spread_samples: Deque[SpreadSample] = deque(maxlen=5000)

        # --- Inventory PnL vs Spread PnL Decomposition ---
        self._cumulative_spread_pnl: float = 0.0
        self._cumulative_inventory_pnl: float = 0.0
        self._last_inventory_mark: Dict[str, Tuple[float, float]] = {}  # symbol -> (qty, mark_price)

        # --- Order-to-Fill Ratio ---
        # (tracked via _total_quotes_placed and _total_fills)

        # --- API Efficiency ---
        self._total_api_weight_used: int = 0
        self._total_profit: float = 0.0

        # Tracking start
        self._start_time: float = time.time()

        logger.info("StrategyMetrics initialized")

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_fill(
        self,
        symbol: str,
        side: str,
        fill_price: float,
        mid_price_at_fill: float,
        quantity: float,
        fee: float,
        quoted_spread_pct: float = 0.0,
    ) -> None:
        """Record a fill for metrics calculation.

        Parameters
        ----------
        symbol : str
        side : str ("BUY" or "SELL")
        fill_price : float
        mid_price_at_fill : float
        quantity : float
        fee : float
        quoted_spread_pct : float
            The theoretical quoted spread at time of fill.
        """
        now = time.time()

        # Spread captured per fill
        if side == "SELL":
            spread_captured = fill_price - mid_price_at_fill
        else:
            spread_captured = mid_price_at_fill - fill_price
        self._spread_per_fill.append((spread_captured, now))

        # Volume for turnover
        notional = fill_price * quantity
        self._total_volume_traded += notional
        self._total_fills += 1

        # Spread PnL
        spread_income = spread_captured * quantity - fee
        self._cumulative_spread_pnl += spread_income
        self._total_profit += spread_income

        # Realized vs quoted spread
        if mid_price_at_fill > 0:
            realized_spread_pct = spread_captured / mid_price_at_fill * 100.0
            self._spread_samples.append(SpreadSample(
                timestamp=now,
                symbol=symbol,
                quoted_spread_pct=quoted_spread_pct,
                realized_spread_pct=realized_spread_pct,
            ))

    def record_adverse_result(self, was_adverse: bool) -> None:
        """Record an adverse selection check result."""
        self._total_fill_count_for_adverse += 1
        if was_adverse:
            self._adverse_fill_count += 1

    def record_quote_placed(self, count: int = 1) -> None:
        """Record that quotes were placed."""
        self._total_quotes_placed += count

    def record_api_weight(self, weight: int) -> None:
        """Record API rate limit weight used."""
        self._total_api_weight_used += weight

    def sample_quote_uptime(self, symbol: str, bid_active: bool, ask_active: bool) -> None:
        """Sample quote uptime (call every 10 seconds).

        Records whether both bid and ask quotes are active.
        """
        if symbol not in self._uptime_records:
            self._uptime_records[symbol] = deque(maxlen=8640)

        self._uptime_records[symbol].append(QuoteUptimeRecord(
            timestamp=time.time(),
            bid_active=bid_active,
            ask_active=ask_active,
            symbol=symbol,
        ))

    def sample_inventory(self, symbol: str, inventory_qty: float, mid_price: float) -> None:
        """Sample current inventory level for turnover calculation."""
        notional = abs(inventory_qty * mid_price)
        self._inventory_samples.append((notional, time.time()))

    def update_inventory_pnl(self, symbol: str, inventory_qty: float, mid_price: float) -> None:
        """Update inventory mark-to-market PnL.

        Call periodically to track inventory PnL separate from spread PnL.
        """
        if symbol in self._last_inventory_mark:
            last_qty, last_price = self._last_inventory_mark[symbol]
            if last_price > 0 and abs(last_qty) > 1e-12:
                # PnL from price movement on held inventory
                price_change = mid_price - last_price
                inventory_pnl = last_qty * price_change
                self._cumulative_inventory_pnl += inventory_pnl

        self._last_inventory_mark[symbol] = (inventory_qty, mid_price)

    # ------------------------------------------------------------------
    # Metric calculations
    # ------------------------------------------------------------------

    def get_spread_captured_per_fill(self) -> float:
        """Average spread captured per fill (in price units)."""
        if not self._spread_per_fill:
            return 0.0
        return sum(s for s, _ in self._spread_per_fill) / len(self._spread_per_fill)

    def get_inventory_turnover(self) -> float:
        """Inventory turnover: total volume traded / average inventory.

        Higher = better (inventory cycles more frequently).
        """
        if not self._inventory_samples:
            return 0.0

        avg_inventory = sum(n for n, _ in self._inventory_samples) / len(self._inventory_samples)
        if avg_inventory <= 0:
            return 0.0

        return self._total_volume_traded / avg_inventory

    def get_quote_uptime(self, symbol: Optional[str] = None) -> float:
        """Percentage of time both bid and ask quotes are active.

        Returns 0-100 percentage.
        """
        if symbol:
            records = self._uptime_records.get(symbol)
            if not records:
                return 0.0
            both_active = sum(1 for r in records if r.bid_active and r.ask_active)
            return (both_active / len(records)) * 100.0

        # Average across all symbols
        total = 0
        active = 0
        for sym, records in self._uptime_records.items():
            total += len(records)
            active += sum(1 for r in records if r.bid_active and r.ask_active)

        if total == 0:
            return 0.0
        return (active / total) * 100.0

    def get_fill_rate(self) -> float:
        """Fill rate: fills / total quotes placed (0-1)."""
        if self._total_quotes_placed == 0:
            return 0.0
        return self._total_fills / self._total_quotes_placed

    def get_adverse_selection_rate(self) -> float:
        """Percentage of fills that were immediately unprofitable at 1s (0-1)."""
        if self._total_fill_count_for_adverse == 0:
            return 0.0
        return self._adverse_fill_count / self._total_fill_count_for_adverse

    def get_realized_vs_quoted_spread(self) -> Dict[str, float]:
        """Compare actual fill spread vs theoretical quoted spread.

        Returns dict with avg_realized_pct, avg_quoted_pct, ratio.
        """
        if not self._spread_samples:
            return {"avg_realized_pct": 0.0, "avg_quoted_pct": 0.0, "ratio": 0.0}

        avg_realized = sum(s.realized_spread_pct for s in self._spread_samples) / len(self._spread_samples)
        avg_quoted = sum(s.quoted_spread_pct for s in self._spread_samples) / len(self._spread_samples)
        ratio = avg_realized / avg_quoted if avg_quoted > 0 else 0.0

        return {
            "avg_realized_pct": avg_realized,
            "avg_quoted_pct": avg_quoted,
            "ratio": ratio,
        }

    def get_pnl_decomposition(self) -> Dict[str, float]:
        """Decompose total PnL into spread income vs inventory mark-to-market.

        Returns dict with spread_pnl, inventory_pnl, total_pnl, spread_pct.
        """
        total = self._cumulative_spread_pnl + self._cumulative_inventory_pnl
        spread_pct = (self._cumulative_spread_pnl / total * 100.0) if total != 0 else 0.0

        return {
            "spread_pnl": self._cumulative_spread_pnl,
            "inventory_pnl": self._cumulative_inventory_pnl,
            "total_pnl": total,
            "spread_pct_of_total": spread_pct,
        }

    def get_order_to_fill_ratio(self) -> float:
        """Orders placed per fill (lower = more efficient)."""
        if self._total_fills == 0:
            return 0.0
        return self._total_quotes_placed / self._total_fills

    def get_api_efficiency(self) -> float:
        """Rate limit weight consumed per dollar of profit.

        Lower = better. Returns weight per $1 profit.
        """
        if self._total_profit <= 0:
            return float("inf") if self._total_api_weight_used > 0 else 0.0
        return self._total_api_weight_used / self._total_profit

    # ------------------------------------------------------------------
    # Aggregated metrics
    # ------------------------------------------------------------------

    def get_all_metrics(self) -> Dict[str, Any]:
        """Return all strategy-specific metrics as a dict."""
        spread_vs_quoted = self.get_realized_vs_quoted_spread()
        pnl_decomp = self.get_pnl_decomposition()
        runtime_hours = (time.time() - self._start_time) / 3600.0

        return {
            "spread_captured_per_fill": self.get_spread_captured_per_fill(),
            "inventory_turnover": self.get_inventory_turnover(),
            "quote_uptime_pct": self.get_quote_uptime(),
            "fill_rate": self.get_fill_rate(),
            "adverse_selection_rate": self.get_adverse_selection_rate(),
            "realized_vs_quoted_spread": spread_vs_quoted,
            "pnl_decomposition": pnl_decomp,
            "order_to_fill_ratio": self.get_order_to_fill_ratio(),
            "api_efficiency_weight_per_dollar": self.get_api_efficiency(),
            "total_volume_traded": self._total_volume_traded,
            "total_fills": self._total_fills,
            "total_quotes_placed": self._total_quotes_placed,
            "total_api_weight": self._total_api_weight_used,
            "total_profit": self._total_profit,
            "runtime_hours": runtime_hours,
            "per_symbol_uptime": {
                sym: self.get_quote_uptime(sym)
                for sym in self._uptime_records
            },
        }

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return state for persistence."""
        return {
            "total_volume_traded": self._total_volume_traded,
            "total_quotes_placed": self._total_quotes_placed,
            "total_fills": self._total_fills,
            "adverse_fill_count": self._adverse_fill_count,
            "total_fill_count_for_adverse": self._total_fill_count_for_adverse,
            "cumulative_spread_pnl": self._cumulative_spread_pnl,
            "cumulative_inventory_pnl": self._cumulative_inventory_pnl,
            "total_api_weight_used": self._total_api_weight_used,
            "total_profit": self._total_profit,
            "start_time": self._start_time,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        if not state:
            return
        self._total_volume_traded = state.get("total_volume_traded", 0.0)
        self._total_quotes_placed = state.get("total_quotes_placed", 0)
        self._total_fills = state.get("total_fills", 0)
        self._adverse_fill_count = state.get("adverse_fill_count", 0)
        self._total_fill_count_for_adverse = state.get("total_fill_count_for_adverse", 0)
        self._cumulative_spread_pnl = state.get("cumulative_spread_pnl", 0.0)
        self._cumulative_inventory_pnl = state.get("cumulative_inventory_pnl", 0.0)
        self._total_api_weight_used = state.get("total_api_weight_used", 0)
        self._total_profit = state.get("total_profit", 0.0)
        self._start_time = state.get("start_time", self._start_time)
        logger.info(
            "StrategyMetrics state restored: %d fills, %.4f profit",
            self._total_fills, self._total_profit,
        )
