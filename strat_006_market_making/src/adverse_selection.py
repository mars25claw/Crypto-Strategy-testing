"""Adverse selection detection for STRAT-006 Market Making.

Tracks mid-price 1 second after each fill to determine whether the market
maker is being adversely selected (i.e., informed traders are systematically
picking off stale quotes).

Key thresholds:
- >60% adverse over last 100 fills -> widen spreads by 50%
- >70% adverse -> halt instrument
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AdverseCheckPending:
    """A fill awaiting its 1-second-later mid-price check."""
    symbol: str
    side: str           # "BUY" or "SELL"
    fill_price: float
    mid_at_fill: float
    fill_timestamp: float
    check_at: float     # When to check (fill_timestamp + 1s)
    fill_index: int     # Index into the fills deque for updating


@dataclass
class AdverseResult:
    """Result of an adverse selection check for a single fill."""
    symbol: str
    side: str
    fill_price: float
    mid_at_fill: float
    mid_1s_after: float
    was_adverse: bool
    price_move_pct: float  # How much mid moved against fill direction
    timestamp: float


class AdverseSelectionTracker:
    """Tracks adverse selection rate across fills.

    After each fill, schedules a check of the mid-price 1 second later.
    If the mid-price moved against the fill direction, the fill was
    adversely selected.

    Parameters
    ----------
    params : dict
        Strategy parameters from config.yaml.
    on_widen : callable, optional
        Async callback(symbol) called when widen threshold is hit.
    on_halt : callable, optional
        Async callback(symbol) called when halt threshold is hit.
    """

    def __init__(
        self,
        params: dict,
        on_widen: Optional[Callable] = None,
        on_halt: Optional[Callable] = None,
    ) -> None:
        self._params = params
        self._on_widen = on_widen
        self._on_halt = on_halt

        self._fill_window = params.get("adverse_fill_window", 100)
        self._widen_threshold = params.get("adverse_widen_threshold", 0.60)
        self._halt_threshold = params.get("adverse_halt_threshold", 0.70)
        self._check_delay_s = params.get("adverse_check_delay_ms", 1000) / 1000.0

        # Per-symbol adverse selection results
        # symbol -> deque of (was_adverse: bool, timestamp)
        self._results: Dict[str, Deque[Tuple[bool, float]]] = {}

        # Pending checks awaiting mid-price measurement
        self._pending: List[AdverseCheckPending] = []

        # Per-symbol state
        self._is_widened: Dict[str, bool] = {}
        self._is_halted: Dict[str, bool] = {}
        self._widen_spread_factor: Dict[str, float] = {}

        # Counters
        self._total_checks = 0
        self._total_adverse = 0

        # Background task
        self._task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            "AdverseSelectionTracker initialized: window=%d fills, "
            "widen_at=%.0f%%, halt_at=%.0f%%, delay=%.1fs",
            self._fill_window, self._widen_threshold * 100,
            self._halt_threshold * 100, self._check_delay_s,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background check loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._check_loop(), name="adverse_selection_checker"
        )
        logger.info("AdverseSelectionTracker started")

    async def stop(self) -> None:
        """Stop the background check loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("AdverseSelectionTracker stopped")

    # ------------------------------------------------------------------
    # Fill registration
    # ------------------------------------------------------------------

    def register_fill(
        self,
        symbol: str,
        side: str,
        fill_price: float,
        mid_at_fill: float,
    ) -> None:
        """Register a fill for adverse selection tracking.

        A check will be performed ~1 second after the fill to measure
        whether the mid-price moved against the fill direction.
        """
        now = time.time()
        pending = AdverseCheckPending(
            symbol=symbol,
            side=side,
            fill_price=fill_price,
            mid_at_fill=mid_at_fill,
            fill_timestamp=now,
            check_at=now + self._check_delay_s,
            fill_index=-1,
        )
        self._pending.append(pending)

    # ------------------------------------------------------------------
    # Mid-price provider callback
    # ------------------------------------------------------------------

    def _check_pending_fills(self, get_mid_price: Callable[[str], float]) -> List[AdverseResult]:
        """Check all pending fills whose delay has elapsed.

        Parameters
        ----------
        get_mid_price : callable
            Function that returns current mid-price for a symbol.

        Returns
        -------
        List of AdverseResult for newly checked fills.
        """
        now = time.time()
        completed: List[AdverseResult] = []
        remaining: List[AdverseCheckPending] = []

        for pending in self._pending:
            if now >= pending.check_at:
                mid_now = get_mid_price(pending.symbol)
                if mid_now <= 0:
                    # Cannot check, discard
                    continue

                # Determine if adverse
                if pending.side == "BUY":
                    # Bought — adverse if mid moved DOWN after fill
                    price_move = (mid_now - pending.mid_at_fill) / pending.mid_at_fill
                    was_adverse = mid_now < pending.mid_at_fill
                else:
                    # Sold — adverse if mid moved UP after fill
                    price_move = (pending.mid_at_fill - mid_now) / pending.mid_at_fill
                    was_adverse = mid_now > pending.mid_at_fill

                result = AdverseResult(
                    symbol=pending.symbol,
                    side=pending.side,
                    fill_price=pending.fill_price,
                    mid_at_fill=pending.mid_at_fill,
                    mid_1s_after=mid_now,
                    was_adverse=was_adverse,
                    price_move_pct=price_move * 100,
                    timestamp=now,
                )
                completed.append(result)

                # Record result
                if pending.symbol not in self._results:
                    self._results[pending.symbol] = deque(maxlen=self._fill_window)
                self._results[pending.symbol].append((was_adverse, now))

                self._total_checks += 1
                if was_adverse:
                    self._total_adverse += 1

                logger.debug(
                    "Adverse check: %s %s fill=%.8f mid_at=%.8f mid_1s=%.8f "
                    "adverse=%s move=%.4f%%",
                    pending.symbol, pending.side, pending.fill_price,
                    pending.mid_at_fill, mid_now, was_adverse, result.price_move_pct,
                )
            else:
                remaining.append(pending)

        self._pending = remaining
        return completed

    # ------------------------------------------------------------------
    # Threshold evaluation
    # ------------------------------------------------------------------

    def evaluate_thresholds(self, symbol: str) -> Tuple[str, float]:
        """Evaluate adverse selection thresholds for a symbol.

        Returns
        -------
        (action, rate):
            action = "normal", "widen", or "halt"
            rate = current adverse selection rate (0.0 to 1.0)
        """
        results = self._results.get(symbol)
        if not results or len(results) < 20:
            return "normal", 0.0

        # Calculate rate over the window
        adverse_count = sum(1 for was_adverse, _ in results if was_adverse)
        rate = adverse_count / len(results)

        if rate >= self._halt_threshold:
            if not self._is_halted.get(symbol, False):
                self._is_halted[symbol] = True
                logger.warning(
                    "ADVERSE HALT: %s rate=%.1f%% (>%.0f%%) over %d fills",
                    symbol, rate * 100, self._halt_threshold * 100, len(results),
                )
            return "halt", rate

        if rate >= self._widen_threshold:
            if not self._is_widened.get(symbol, False):
                self._is_widened[symbol] = True
                self._widen_spread_factor[symbol] = 1.5  # 50% wider
                logger.warning(
                    "ADVERSE WIDEN: %s rate=%.1f%% (>%.0f%%) over %d fills — widening 50%%",
                    symbol, rate * 100, self._widen_threshold * 100, len(results),
                )
            return "widen", rate

        # If we were widened/halted but rate improved, reset
        if self._is_widened.get(symbol, False):
            self._is_widened[symbol] = False
            self._widen_spread_factor[symbol] = 1.0
            logger.info("ADVERSE CLEARED: %s rate=%.1f%% — resuming normal spreads", symbol, rate * 100)
        if self._is_halted.get(symbol, False):
            self._is_halted[symbol] = False
            logger.info("ADVERSE UNHALT: %s rate=%.1f%% — resuming quoting", symbol, rate * 100)

        return "normal", rate

    def get_spread_factor(self, symbol: str) -> float:
        """Return the spread widening factor for a symbol.

        1.0 = normal, 1.5 = widen 50%.
        """
        return self._widen_spread_factor.get(symbol, 1.0)

    def is_halted(self, symbol: str) -> bool:
        """Return True if the symbol is halted due to adverse selection."""
        return self._is_halted.get(symbol, False)

    # ------------------------------------------------------------------
    # Background check loop
    # ------------------------------------------------------------------

    async def _check_loop(self) -> None:
        """Periodically process pending adverse selection checks."""
        # This loop is minimal — actual check logic is driven by
        # process_pending() calls from the main bot loop, which
        # provides the mid-price getter.
        try:
            while self._running:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    def process_pending(self, get_mid_price: Callable[[str], float]) -> List[AdverseResult]:
        """Process all pending checks using the provided mid-price getter.

        This should be called from the main loop every ~500ms.
        """
        results = self._check_pending_fills(get_mid_price)

        # Evaluate thresholds and trigger callbacks
        symbols_to_check = set(r.symbol for r in results)
        for symbol in symbols_to_check:
            action, rate = self.evaluate_thresholds(symbol)
            if action == "widen" and self._on_widen:
                asyncio.ensure_future(self._on_widen(symbol))
            elif action == "halt" and self._on_halt:
                asyncio.ensure_future(self._on_halt(symbol))

        return results

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_rate(self, symbol: str) -> float:
        """Return current adverse selection rate for a symbol."""
        results = self._results.get(symbol)
        if not results:
            return 0.0
        adverse = sum(1 for was_adverse, _ in results if was_adverse)
        return adverse / len(results)

    def get_metrics(self) -> Dict[str, Any]:
        """Return adverse selection metrics for all symbols."""
        metrics = {
            "total_checks": self._total_checks,
            "total_adverse": self._total_adverse,
            "overall_rate": self._total_adverse / max(1, self._total_checks),
            "pending_checks": len(self._pending),
            "per_symbol": {},
        }

        for symbol, results in self._results.items():
            adverse = sum(1 for was_adverse, _ in results if was_adverse)
            metrics["per_symbol"][symbol] = {
                "checks": len(results),
                "adverse": adverse,
                "rate": adverse / max(1, len(results)),
                "is_widened": self._is_widened.get(symbol, False),
                "is_halted": self._is_halted.get(symbol, False),
                "spread_factor": self._widen_spread_factor.get(symbol, 1.0),
            }

        return metrics

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return state for persistence."""
        return {
            "results": {
                sym: [(a, t) for a, t in results]
                for sym, results in self._results.items()
            },
            "is_widened": dict(self._is_widened),
            "is_halted": dict(self._is_halted),
            "widen_factors": dict(self._widen_spread_factor),
            "total_checks": self._total_checks,
            "total_adverse": self._total_adverse,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        if not state:
            return
        for sym, results in state.get("results", {}).items():
            self._results[sym] = deque(
                [(a, t) for a, t in results],
                maxlen=self._fill_window,
            )
        self._is_widened = state.get("is_widened", {})
        self._is_halted = state.get("is_halted", {})
        self._widen_spread_factor = state.get("widen_factors", {})
        self._total_checks = state.get("total_checks", 0)
        self._total_adverse = state.get("total_adverse", 0)
        logger.info("Adverse selection state restored: %d total checks", self._total_checks)
