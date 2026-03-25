"""STRAT-007: Continuous opportunity scanner.

Processes bookTicker and depth WebSocket messages in real-time to detect
arbitrage opportunities across Mode A (spot-futures) and Mode B (triangular).

Features:
- Mode A: 7 assets x spot+futures bookTickers, continuous comparison
- Mode B: 10 triangle paths, 3 bookTickers each
- Opportunity caching with age tracking
- Minimum viable profit >= $0.50 net per trade
- Price anomaly detection: skip if >2% deviation, wait for 3 confirmations
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Deque, List, Optional, Tuple

from strat_007_triangular_arb.src.strategy import (
    ArbMode,
    ArbOpportunity,
    TriangularArbStrategy,
    TrianglePath,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Book ticker cache
# ---------------------------------------------------------------------------

@dataclass
class BookTickerData:
    """Cached bookTicker data for a single symbol."""

    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: float = 0.0
    ask_qty: float = 0.0
    updated_at: float = 0.0

    def is_valid(self) -> bool:
        return self.bid > 0 and self.ask > 0

    def age_ms(self) -> float:
        return (time.time() - self.updated_at) * 1000.0


@dataclass
class DepthData:
    """Cached top-of-book depth data for a single symbol."""

    symbol: str
    bids: List[Tuple[float, float]] = field(default_factory=list)  # (price, qty)
    asks: List[Tuple[float, float]] = field(default_factory=list)
    updated_at: float = 0.0

    def best_bid_depth(self) -> float:
        """Total quantity available at top bid levels."""
        return sum(qty for _, qty in self.bids[:5])

    def best_ask_depth(self) -> float:
        """Total quantity available at top ask levels."""
        return sum(qty for _, qty in self.asks[:5])

    def to_orderbook_snapshot(self) -> Dict[str, List]:
        """Convert to format compatible with PaperTradingEngine."""
        return {
            "bids": [[p, q] for p, q in self.bids],
            "asks": [[p, q] for p, q in self.asks],
        }


# ---------------------------------------------------------------------------
# Opportunity cache
# ---------------------------------------------------------------------------

@dataclass
class CachedOpportunity:
    """Opportunity with age tracking."""

    opportunity: ArbOpportunity
    created_at: float = field(default_factory=time.time)

    def age_s(self) -> float:
        return time.time() - self.created_at


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class OpportunityScanner:
    """Continuous scanner for arbitrage opportunities.

    Parameters
    ----------
    strategy : TriangularArbStrategy
        Strategy engine for evaluation logic.
    params : dict
        Strategy parameters.
    on_opportunity : callable
        Async callback invoked when a viable opportunity is detected.
        Signature: async callback(opportunity: ArbOpportunity) -> None
    """

    def __init__(
        self,
        strategy: TriangularArbStrategy,
        params: Dict[str, Any],
        on_opportunity: Optional[Callable[[ArbOpportunity], Coroutine]] = None,
    ) -> None:
        self._strategy = strategy
        self._params = params
        self._on_opportunity = on_opportunity

        # Book ticker caches
        self._spot_tickers: Dict[str, BookTickerData] = {}
        self._futures_tickers: Dict[str, BookTickerData] = {}
        self._tri_tickers: Dict[str, BookTickerData] = {}  # For triangular pairs

        # Depth caches
        self._spot_depth: Dict[str, DepthData] = {}
        self._futures_depth: Dict[str, DepthData] = {}

        # Opportunity cache (recent opportunities with age)
        self._opportunity_cache: Deque[CachedOpportunity] = deque(maxlen=1000)
        self._last_taken_opportunities: Deque[ArbOpportunity] = deque(maxlen=100)
        self._last_skipped_opportunities: Deque[ArbOpportunity] = deque(maxlen=100)

        # Configuration
        self._max_age_s = params.get("opportunity_max_age_s", 1.0)
        self._mode_a_enabled = params.get("mode_a_enabled", True)
        self._mode_b_enabled = params.get("mode_b_enabled", True)
        self._mode_a_assets = params.get("mode_a_assets", [])

        # Balance / equity (updated externally)
        self._available_balance: float = 0.0
        self._equity: float = 0.0

        # Volume tracking for anomaly detection (symbol -> recent agg volume)
        self._spot_volumes: Dict[str, float] = {}
        self._futures_volumes: Dict[str, float] = {}

        # Scanning state
        self._scanning = False
        self._scan_count = 0
        self._opportunities_per_hour: Deque[float] = deque(maxlen=3600)

        # Statistics
        self._stats = {
            "mode_a_scans": 0,
            "mode_b_scans": 0,
            "mode_a_detected": 0,
            "mode_b_detected": 0,
            "stale_discarded": 0,
            "total_detected": 0,
            "total_taken": 0,
            "total_skipped": 0,
        }

        logger.info(
            "OpportunityScanner initialized: mode_a=%s mode_b=%s assets=%d triangles=%d",
            self._mode_a_enabled, self._mode_b_enabled,
            len(self._mode_a_assets), len(self._strategy.triangle_paths),
        )

    # ------------------------------------------------------------------
    # External state updates
    # ------------------------------------------------------------------

    def update_balance(self, available_balance: float, equity: float) -> None:
        """Update available balance and equity for size calculations."""
        self._available_balance = available_balance
        self._equity = equity

    # ------------------------------------------------------------------
    # WebSocket message handlers
    # ------------------------------------------------------------------

    async def handle_spot_book_ticker(self, data: Dict[str, Any]) -> None:
        """Process a spot bookTicker message."""
        symbol = data.get("s", "")
        if not symbol:
            return

        ticker = self._spot_tickers.get(symbol)
        if ticker is None:
            ticker = BookTickerData(symbol=symbol)
            self._spot_tickers[symbol] = ticker

        ticker.bid = float(data.get("b", 0))
        ticker.ask = float(data.get("a", 0))
        ticker.bid_qty = float(data.get("B", 0))
        ticker.ask_qty = float(data.get("A", 0))
        ticker.updated_at = time.time()

        # Also update tri tickers (triangular pairs share spot book tickers)
        tri_ticker = self._tri_tickers.get(symbol)
        if tri_ticker is None:
            tri_ticker = BookTickerData(symbol=symbol)
            self._tri_tickers[symbol] = tri_ticker
        tri_ticker.bid = ticker.bid
        tri_ticker.ask = ticker.ask
        tri_ticker.bid_qty = ticker.bid_qty
        tri_ticker.ask_qty = ticker.ask_qty
        tri_ticker.updated_at = ticker.updated_at

        # Trigger scan for this symbol
        if self._scanning and self._mode_a_enabled:
            await self._scan_mode_a(symbol)

        if self._scanning and self._mode_b_enabled:
            await self._scan_mode_b_for_pair(symbol)

    async def handle_futures_book_ticker(self, data: Dict[str, Any]) -> None:
        """Process a futures bookTicker message."""
        symbol = data.get("s", "")
        if not symbol:
            return

        ticker = self._futures_tickers.get(symbol)
        if ticker is None:
            ticker = BookTickerData(symbol=symbol)
            self._futures_tickers[symbol] = ticker

        ticker.bid = float(data.get("b", 0))
        ticker.ask = float(data.get("a", 0))
        ticker.bid_qty = float(data.get("B", 0))
        ticker.ask_qty = float(data.get("A", 0))
        ticker.updated_at = time.time()

        # Trigger Mode A scan
        if self._scanning and self._mode_a_enabled:
            await self._scan_mode_a(symbol)

    async def handle_spot_depth(self, data: Dict[str, Any]) -> None:
        """Process a spot depth (top 5) message."""
        # The stream format uses lowercase 's' for symbol in some cases
        symbol = data.get("s", "")
        if not symbol:
            return

        depth = self._spot_depth.get(symbol)
        if depth is None:
            depth = DepthData(symbol=symbol)
            self._spot_depth[symbol] = depth

        depth.bids = [
            (float(p), float(q)) for p, q in data.get("bids", data.get("b", []))
        ]
        depth.asks = [
            (float(p), float(q)) for p, q in data.get("asks", data.get("a", []))
        ]
        depth.updated_at = time.time()

    async def handle_futures_depth(self, data: Dict[str, Any]) -> None:
        """Process a futures depth (top 5) message."""
        symbol = data.get("s", "")
        if not symbol:
            return

        depth = self._futures_depth.get(symbol)
        if depth is None:
            depth = DepthData(symbol=symbol)
            self._futures_depth[symbol] = depth

        depth.bids = [
            (float(p), float(q)) for p, q in data.get("bids", data.get("b", []))
        ]
        depth.asks = [
            (float(p), float(q)) for p, q in data.get("asks", data.get("a", []))
        ]
        depth.updated_at = time.time()

    # ------------------------------------------------------------------
    # Volume tracking (for anomaly detection)
    # ------------------------------------------------------------------

    async def handle_spot_agg_trade(self, data: Dict[str, Any]) -> None:
        """Process a spot aggTrade message to track volume."""
        symbol = data.get("s", "")
        qty = float(data.get("q", 0))
        if symbol:
            self._spot_volumes[symbol] = self._spot_volumes.get(symbol, 0) + qty

    async def handle_futures_agg_trade(self, data: Dict[str, Any]) -> None:
        """Process a futures aggTrade message to track volume."""
        symbol = data.get("s", "")
        qty = float(data.get("q", 0))
        if symbol:
            self._futures_volumes[symbol] = self._futures_volumes.get(symbol, 0) + qty

    def _consume_volume(self, symbol: str, market: str) -> float:
        """Return and reset accumulated volume for anomaly detection."""
        if market == "spot":
            vol = self._spot_volumes.pop(symbol, 0.0)
        else:
            vol = self._futures_volumes.pop(symbol, 0.0)
        return vol

    # ------------------------------------------------------------------
    # Scanning logic
    # ------------------------------------------------------------------

    def start_scanning(self) -> None:
        """Enable opportunity scanning."""
        self._scanning = True
        logger.info("Opportunity scanning STARTED")

    def stop_scanning(self) -> None:
        """Disable opportunity scanning."""
        self._scanning = False
        logger.info("Opportunity scanning STOPPED")

    async def _scan_mode_a(self, symbol: str) -> None:
        """Scan for Mode A opportunity on a specific symbol."""
        if symbol not in self._mode_a_assets:
            return

        spot = self._spot_tickers.get(symbol)
        futures = self._futures_tickers.get(symbol)

        if not spot or not futures or not spot.is_valid() or not futures.is_valid():
            return

        if self._equity <= 0 or self._available_balance <= 0:
            return

        self._stats["mode_a_scans"] += 1

        # Get depth quantities
        spot_depth = self._spot_depth.get(symbol)
        futures_depth = self._futures_depth.get(symbol)

        spot_depth_qty = spot_depth.best_ask_depth() if spot_depth else spot.ask_qty
        futures_depth_qty = futures_depth.best_bid_depth() if futures_depth else futures.bid_qty

        # Consume accumulated volume for anomaly detection
        spot_vol = self._consume_volume(symbol, "spot")
        futures_vol = self._consume_volume(symbol, "futures")

        opp = self._strategy.evaluate_mode_a(
            symbol=symbol,
            spot_bid=spot.bid,
            spot_ask=spot.ask,
            futures_bid=futures.bid,
            futures_ask=futures.ask,
            spot_depth_qty=max(spot_depth_qty, spot.ask_qty),
            futures_depth_qty=max(futures_depth_qty, futures.bid_qty),
            available_balance=self._available_balance,
            equity=self._equity,
            spot_volume=spot_vol,
            futures_volume=futures_vol,
        )

        if opp is not None:
            self._stats["mode_a_detected"] += 1
            await self._emit_opportunity(opp)

    async def _scan_mode_b_for_pair(self, pair: str) -> None:
        """Scan all triangle paths that include the given pair."""
        for path in self._strategy.triangle_paths:
            if pair in path.pairs:
                await self._scan_mode_b(path)

    async def _scan_mode_b(self, path: TrianglePath) -> None:
        """Scan for Mode B opportunity on a specific triangle path."""
        if self._equity <= 0 or self._available_balance <= 0:
            return

        self._stats["mode_b_scans"] += 1

        # Build book ticker dict for this path
        book_tickers: Dict[str, Dict[str, float]] = {}
        for pair in path.pairs:
            ticker = self._tri_tickers.get(pair) or self._spot_tickers.get(pair)
            if ticker is None or not ticker.is_valid():
                return
            book_tickers[pair] = {
                "bid": ticker.bid,
                "ask": ticker.ask,
                "bid_qty": ticker.bid_qty,
                "ask_qty": ticker.ask_qty,
            }

        opp = self._strategy.evaluate_mode_b(
            path=path,
            book_tickers=book_tickers,
            available_balance=self._available_balance,
            equity=self._equity,
        )

        if opp is not None:
            self._stats["mode_b_detected"] += 1
            await self._emit_opportunity(opp)

    async def _emit_opportunity(self, opp: ArbOpportunity) -> None:
        """Process a detected opportunity: cache, age-check, and notify."""
        self._stats["total_detected"] += 1
        self._opportunities_per_hour.append(time.time())

        # Age check
        opp.age_ms = (time.time() - opp.detected_at) * 1000.0
        if opp.is_stale(self._max_age_s):
            self._stats["stale_discarded"] += 1
            opp.skip_reason = "stale"
            opp.taken = False
            self._last_skipped_opportunities.append(opp)
            return

        # Cache the opportunity
        cached = CachedOpportunity(opportunity=opp)
        self._opportunity_cache.append(cached)

        # Notify listener
        if self._on_opportunity is not None:
            try:
                await self._on_opportunity(opp)
            except Exception:
                logger.exception("Error in opportunity callback for %s", opp.symbol)

    # ------------------------------------------------------------------
    # Depth snapshot access (for paper trading fill simulation)
    # ------------------------------------------------------------------

    def get_spot_depth_snapshot(self, symbol: str) -> Optional[Dict[str, List]]:
        """Return spot order book snapshot for paper trading."""
        depth = self._spot_depth.get(symbol)
        if depth is None:
            return None
        return depth.to_orderbook_snapshot()

    def get_futures_depth_snapshot(self, symbol: str) -> Optional[Dict[str, List]]:
        """Return futures order book snapshot for paper trading."""
        depth = self._futures_depth.get(symbol)
        if depth is None:
            return None
        return depth.to_orderbook_snapshot()

    def get_book_ticker(self, symbol: str, market: str = "spot") -> Optional[BookTickerData]:
        """Get cached book ticker for a symbol."""
        if market == "spot":
            return self._spot_tickers.get(symbol) or self._tri_tickers.get(symbol)
        elif market == "futures":
            return self._futures_tickers.get(symbol)
        return None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return scanner statistics."""
        now = time.time()
        # Count opportunities in last hour
        recent = sum(1 for t in self._opportunities_per_hour if now - t < 3600)

        return {
            **self._stats,
            "opportunities_per_hour": recent,
            "scanning": self._scanning,
            "spot_tickers_count": len(self._spot_tickers),
            "futures_tickers_count": len(self._futures_tickers),
            "tri_tickers_count": len(self._tri_tickers),
            "cache_size": len(self._opportunity_cache),
        }

    def get_recent_opportunities(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent opportunities (taken + skipped) for the dashboard."""
        taken = [o.to_dict() for o in list(self._last_taken_opportunities)[-limit:]]
        skipped = [o.to_dict() for o in list(self._last_skipped_opportunities)[-limit:]]
        combined = taken + skipped
        combined.sort(key=lambda x: x.get("detected_at", 0), reverse=True)
        return combined[:limit]

    def record_taken(self, opp: ArbOpportunity) -> None:
        """Record that an opportunity was taken."""
        opp.taken = True
        self._stats["total_taken"] += 1
        self._last_taken_opportunities.append(opp)
        self._strategy.record_taken()

    def record_skipped(self, opp: ArbOpportunity, reason: str) -> None:
        """Record that an opportunity was skipped."""
        opp.taken = False
        opp.skip_reason = reason
        self._stats["total_skipped"] += 1
        self._last_skipped_opportunities.append(opp)
        self._strategy.record_skipped()

    # ------------------------------------------------------------------
    # Stream name helpers
    # ------------------------------------------------------------------

    def get_required_spot_streams(self) -> List[str]:
        """Return list of required spot WS stream names."""
        streams = []
        if self._mode_a_enabled:
            for asset in self._mode_a_assets:
                sym = asset.lower()
                streams.append(f"{sym}@bookTicker")
                streams.append(f"{sym}@depth5@100ms")

        if self._mode_b_enabled:
            for path in self._strategy.triangle_paths:
                for pair in path.pairs:
                    s = f"{pair.lower()}@bookTicker"
                    if s not in streams:
                        streams.append(s)
        return streams

    def get_required_futures_streams(self) -> List[str]:
        """Return list of required futures WS stream names."""
        streams = []
        if self._mode_a_enabled:
            for asset in self._mode_a_assets:
                sym = asset.lower()
                streams.append(f"{sym}@bookTicker")
                streams.append(f"{sym}@depth5@100ms")
        return streams
