"""STRAT-007: Dual-mode arbitrage strategy logic.

Mode A — Spot-Futures Arbitrage:
    Continuously compares spot bid/ask vs futures bid/ask for 7 assets.
    Calculates net profit after fees, executes when threshold is met.

Mode B — Triangular Arbitrage:
    Monitors 10 triangle paths, computes cross-rate vs direct rate,
    executes 3-leg sequential trades when gross profit > 3x taker fees.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ArbMode(str, Enum):
    MODE_A = "spot_futures"
    MODE_B = "triangular"


class ArbDirection(str, Enum):
    BUY_SPOT_SELL_FUTURES = "buy_spot_sell_futures"   # Futures premium
    BUY_FUTURES_SELL_SPOT = "buy_futures_sell_spot"   # Futures discount


@dataclass
class ArbOpportunity:
    """Represents a detected arbitrage opportunity."""

    mode: ArbMode
    symbol: str = ""                    # Primary symbol (Mode A) or triangle ID (Mode B)
    direction: ArbDirection = ArbDirection.BUY_SPOT_SELL_FUTURES
    gross_profit_pct: float = 0.0
    net_profit_pct: float = 0.0
    net_profit_usdt: float = 0.0
    trade_size_usdt: float = 0.0
    trade_size_qty: float = 0.0
    detected_at: float = field(default_factory=time.time)
    age_ms: float = 0.0

    # Mode A specifics
    spot_bid: float = 0.0
    spot_ask: float = 0.0
    futures_bid: float = 0.0
    futures_ask: float = 0.0
    spot_depth_qty: float = 0.0
    futures_depth_qty: float = 0.0

    # Mode B specifics
    triangle_path: List[str] = field(default_factory=list)
    leg_prices: List[float] = field(default_factory=list)
    leg_quantities: List[float] = field(default_factory=list)
    leg_sides: List[str] = field(default_factory=list)
    final_usdt: float = 0.0
    starting_usdt: float = 0.0

    # Execution status
    taken: bool = False
    skip_reason: str = ""

    def is_stale(self, max_age_s: float = 1.0) -> bool:
        """Check if opportunity is too old to execute."""
        return (time.time() - self.detected_at) > max_age_s

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "symbol": self.symbol,
            "direction": self.direction.value if self.direction else "",
            "gross_profit_pct": round(self.gross_profit_pct, 6),
            "net_profit_pct": round(self.net_profit_pct, 6),
            "net_profit_usdt": round(self.net_profit_usdt, 4),
            "trade_size_usdt": round(self.trade_size_usdt, 2),
            "detected_at": self.detected_at,
            "age_ms": round(self.age_ms, 1),
            "taken": self.taken,
            "skip_reason": self.skip_reason,
        }


@dataclass
class ExecutionResult:
    """Result of an arb execution."""

    opportunity: ArbOpportunity
    success: bool = False
    actual_profit_usdt: float = 0.0
    actual_profit_pct: float = 0.0
    total_fees_usdt: float = 0.0
    execution_time_ms: float = 0.0
    legs_filled: int = 0
    legs_total: int = 0
    leg_results: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    slippage_usdt: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mode": self.opportunity.mode.value,
            "symbol": self.opportunity.symbol,
            "success": self.success,
            "actual_profit_usdt": round(self.actual_profit_usdt, 6),
            "actual_profit_pct": round(self.actual_profit_pct, 6),
            "total_fees_usdt": round(self.total_fees_usdt, 6),
            "execution_time_ms": round(self.execution_time_ms, 1),
            "legs_filled": self.legs_filled,
            "legs_total": self.legs_total,
            "slippage_usdt": round(self.slippage_usdt, 6),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Triangle path definition
# ---------------------------------------------------------------------------

@dataclass
class TrianglePath:
    """Defines a triangular arbitrage path."""

    path_id: str
    pairs: List[str]      # 3 pairs, e.g. ["BTCUSDT", "ETHBTC", "ETHUSDT"]

    # Derived from pair analysis:
    # Leg 1: Buy Asset_A with USDT (pair[0] = A_USDT, side=BUY)
    # Leg 2: Sell Asset_A for Asset_B (pair[1] = B_A, side depends on pair direction)
    # Leg 3: Sell Asset_B for USDT (pair[2] = B_USDT, side=SELL)
    leg_sides: List[str] = field(default_factory=lambda: ["BUY", "SELL", "SELL"])


# ---------------------------------------------------------------------------
# Strategy engine
# ---------------------------------------------------------------------------

class TriangularArbStrategy:
    """Core strategy logic for both Mode A and Mode B arbitrage.

    Parameters
    ----------
    params : dict
        Strategy parameters from config.yaml strategy_params section.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        self._params = params

        # Mode A thresholds
        self._mode_a_threshold_pct = params.get("mode_a_threshold_default_pct", 0.08)
        self._mode_a_min_threshold_pct = params.get("mode_a_threshold_min_pct", 0.05)
        self._mode_a_conservative_pct = params.get("mode_a_threshold_conservative_pct", 0.10)
        self._mode_a_max_per_leg_pct = params.get("mode_a_max_per_leg_pct", 2.0)

        # Mode B thresholds
        self._mode_b_min_gross_pct = params.get("mode_b_min_gross_profit_pct", 0.35)
        self._mode_b_max_per_leg_pct = params.get("mode_b_max_per_leg_pct", 1.0)

        # Fee rates (as decimals, e.g. 0.001 = 0.10%)
        self._spot_taker_fee = params.get("spot_taker_fee_pct", 0.10) / 100.0
        self._futures_taker_fee = params.get("futures_taker_fee_pct", 0.04) / 100.0
        self._bnb_discount_pct = params.get("bnb_fee_discount_pct", 25.0) / 100.0
        self._has_bnb_discount = False

        # Filters
        self._depth_multiplier = params.get("depth_multiplier", 1.5)
        self._min_profit_usdt = params.get("min_profit_usdt", 0.50)
        self._max_age_s = params.get("opportunity_max_age_s", 1.0)
        self._anomaly_deviation_pct = params.get("anomaly_deviation_pct", 2.0)
        self._anomaly_confirmations = params.get("anomaly_confirmations", 3)

        # Volatility
        self._high_vol_threshold_mult = params.get("high_volatility_threshold_mult", 2.0)
        self._high_volatility_active = False

        # Triangle paths
        self._triangle_paths: List[TrianglePath] = []
        self._init_triangle_paths()

        # Recent prices for anomaly detection: symbol -> list of recent prices
        self._recent_prices: Dict[str, List[float]] = {}
        self._anomaly_counters: Dict[str, int] = {}

        # Statistics
        self._opportunities_detected = 0
        self._opportunities_taken = 0
        self._opportunities_skipped = 0

        logger.info(
            "TriangularArbStrategy initialized: mode_a_threshold=%.2f%% "
            "mode_b_min_gross=%.2f%% spot_fee=%.4f futures_fee=%.4f",
            self._mode_a_threshold_pct, self._mode_b_min_gross_pct,
            self._spot_taker_fee, self._futures_taker_fee,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_triangle_paths(self) -> None:
        """Initialize triangle paths from config."""
        raw_triangles = self._params.get("mode_b_triangles", [])
        for i, pairs in enumerate(raw_triangles):
            if len(pairs) != 3:
                logger.warning("Triangle path %d has %d pairs, expected 3 — skipping", i, len(pairs))
                continue
            path = TrianglePath(
                path_id=f"TRI_{i:02d}",
                pairs=pairs,
                leg_sides=["BUY", "SELL", "SELL"],
            )
            self._triangle_paths.append(path)
        logger.info("Initialized %d triangle paths", len(self._triangle_paths))

    # ------------------------------------------------------------------
    # Fee management
    # ------------------------------------------------------------------

    def update_bnb_discount(self, has_bnb: bool) -> None:
        """Update whether BNB fee discount applies."""
        old = self._has_bnb_discount
        self._has_bnb_discount = has_bnb
        if old != has_bnb:
            logger.info("BNB fee discount %s", "enabled" if has_bnb else "disabled")

    def get_spot_taker_fee(self) -> float:
        """Return effective spot taker fee rate (decimal)."""
        base = self._spot_taker_fee
        if self._has_bnb_discount:
            return base * (1.0 - self._bnb_discount_pct)
        return base

    def get_futures_taker_fee(self) -> float:
        """Return effective futures taker fee rate (decimal)."""
        base = self._futures_taker_fee
        if self._has_bnb_discount:
            return base * (1.0 - self._bnb_discount_pct)
        return base

    def recalibrate_thresholds(
        self,
        new_spot_fee: float = None,
        new_futures_fee: float = None,
        bnb_balance: float = 0.0,
    ) -> None:
        """Recalibrate thresholds based on current fee tier.

        Called on startup, weekly (Monday 00:00 UTC), and mid-session
        when a fee tier change is detected.

        Parameters
        ----------
        new_spot_fee : float, optional
            New spot taker fee as percentage (e.g. 0.10 for 0.10%).
            If None, keeps current value.
        new_futures_fee : float, optional
            New futures taker fee as percentage (e.g. 0.04 for 0.04%).
            If None, keeps current value.
        bnb_balance : float
            Total BNB balance. If > 0, apply BNB discount (0.025% = 25%
            off standard fee) per Section 7.3.
        """
        if new_spot_fee is not None:
            self._spot_taker_fee = new_spot_fee / 100.0
        if new_futures_fee is not None:
            self._futures_taker_fee = new_futures_fee / 100.0

        # Account for BNB balance discount (Section 7.3)
        # BNB discount gives 25% off, effective rate 0.075% instead of 0.10%
        has_bnb = bnb_balance > 0
        self._has_bnb_discount = has_bnb
        if has_bnb:
            logger.info(
                "BNB balance=%.4f — applying fee discount (%.1f%% off)",
                bnb_balance, self._bnb_discount_pct * 100.0,
            )

        # Compute effective fees (after BNB discount if applicable)
        eff_spot = self.get_spot_taker_fee()
        eff_futures = self.get_futures_taker_fee()

        # Mode A: threshold = spot_fee + futures_fee + 0.03% min net profit
        total_fee_pct = (eff_spot + eff_futures) * 100.0
        min_viable = total_fee_pct + 0.03  # 3bps minimum net profit
        self._mode_a_threshold_pct = max(self._mode_a_min_threshold_pct, min_viable)

        # Mode B: gross must exceed 3x taker fees + 5bps buffer
        single_taker = max(eff_spot, 0.001) * 100.0  # in pct
        self._mode_b_min_gross_pct = single_taker * 3.0 + 0.05

        logger.info(
            "Thresholds recalibrated: mode_a=%.3f%% mode_b_gross=%.3f%% "
            "eff_spot_fee=%.4f eff_futures_fee=%.4f bnb_discount=%s",
            self._mode_a_threshold_pct, self._mode_b_min_gross_pct,
            eff_spot, eff_futures, has_bnb,
        )

    def check_fee_tier_change(
        self, current_spot_fee_pct: float, current_futures_fee_pct: float,
    ) -> bool:
        """Detect mid-session fee tier change and recalibrate if needed.

        Returns True if a change was detected and thresholds were updated.
        """
        current_spot = current_spot_fee_pct / 100.0
        current_futures = current_futures_fee_pct / 100.0
        eps = 1e-6

        changed = (
            abs(current_spot - self._spot_taker_fee) > eps
            or abs(current_futures - self._futures_taker_fee) > eps
        )
        if changed:
            logger.warning(
                "Fee tier change detected mid-session: "
                "spot %.4f→%.4f futures %.4f→%.4f — recalibrating immediately",
                self._spot_taker_fee, current_spot,
                self._futures_taker_fee, current_futures,
            )
            self.recalibrate_thresholds(current_spot_fee_pct, current_futures_fee_pct)
            return True
        return False

    # ------------------------------------------------------------------
    # Volatility filter
    # ------------------------------------------------------------------

    def set_high_volatility(self, active: bool) -> None:
        """Toggle high volatility mode (doubles thresholds)."""
        if self._high_volatility_active != active:
            self._high_volatility_active = active
            logger.info("High volatility mode %s", "ACTIVE" if active else "inactive")

    def _effective_mode_a_threshold(self) -> float:
        """Return current Mode A threshold, accounting for volatility."""
        t = self._mode_a_threshold_pct
        if self._high_volatility_active:
            t *= self._high_vol_threshold_mult
        return t

    def _effective_mode_b_gross_threshold(self) -> float:
        """Return current Mode B gross threshold, accounting for volatility."""
        t = self._mode_b_min_gross_pct
        if self._high_volatility_active:
            t *= self._high_vol_threshold_mult
        return t

    # ------------------------------------------------------------------
    # Price anomaly detection
    # ------------------------------------------------------------------

    def check_price_anomaly(
        self, symbol: str, price: float, volume: float = 0.0,
    ) -> bool:
        """Return True if the price appears anomalous (>2% deviation).

        Per Section 11.3: if bookTicker reports a price >2% from recent
        average WITHOUT corresponding aggTrade volume, treat as erroneous
        data.  Wait for 3 consecutive bookTicker updates confirming the
        new price level before accepting it.

        Parameters
        ----------
        symbol : str
            Instrument identifier (e.g. ``BTCUSDT_spot``).
        price : float
            Latest mid-price.
        volume : float
            Corresponding recent aggTrade / tick volume. When 0 the
            deviation is treated as lacking volume confirmation.
        """
        if symbol not in self._recent_prices:
            self._recent_prices[symbol] = []
            self._anomaly_counters[symbol] = 0

        history = self._recent_prices[symbol]

        if len(history) < 5:
            history.append(price)
            return False

        # Calculate recent average
        avg = sum(history[-20:]) / len(history[-20:])
        deviation_pct = abs(price - avg) / avg * 100.0

        if deviation_pct > self._anomaly_deviation_pct:
            # Section 11.3: >2% deviation WITHOUT volume → erroneous
            has_volume = volume > 0
            if not has_volume:
                self._anomaly_counters[symbol] += 1
                if self._anomaly_counters[symbol] < self._anomaly_confirmations:
                    logger.warning(
                        "erroneous price data detected for %s: price=%.4f "
                        "avg=%.4f deviation=%.2f%% volume=%.4f "
                        "confirmations=%d/%d — DO NOT arb",
                        symbol, price, avg, deviation_pct, volume,
                        self._anomaly_counters[symbol],
                        self._anomaly_confirmations,
                    )
                    return True  # Anomalous — don't trade
                else:
                    # 3 consecutive ticks confirmed the new level
                    self._anomaly_counters[symbol] = 0
                    logger.info(
                        "Price move confirmed for %s after %d consecutive "
                        "ticks: %.4f → %.4f (no volume but confirmed)",
                        symbol, self._anomaly_confirmations, avg, price,
                    )
            else:
                # Large move WITH volume — genuine market move
                self._anomaly_counters[symbol] = 0
                logger.info(
                    "Large price move WITH volume for %s: "
                    "price=%.4f avg=%.4f deviation=%.2f%% volume=%.4f — accepted",
                    symbol, price, avg, deviation_pct, volume,
                )
        else:
            self._anomaly_counters[symbol] = 0

        history.append(price)
        # Keep last 50 prices
        if len(history) > 50:
            self._recent_prices[symbol] = history[-50:]

        return False

    # ------------------------------------------------------------------
    # Mode A: Spot-Futures opportunity evaluation
    # ------------------------------------------------------------------

    def evaluate_mode_a(
        self,
        symbol: str,
        spot_bid: float,
        spot_ask: float,
        futures_bid: float,
        futures_ask: float,
        spot_depth_qty: float,
        futures_depth_qty: float,
        available_balance: float,
        equity: float,
        spot_volume: float = 0.0,
        futures_volume: float = 0.0,
    ) -> Optional[ArbOpportunity]:
        """Evaluate a potential spot-futures arbitrage opportunity.

        Checks both directions:
        1. Buy spot, sell futures (futures premium)
        2. Buy futures, sell spot (futures discount)

        Returns an ArbOpportunity if viable, None otherwise.
        """
        if spot_bid <= 0 or spot_ask <= 0 or futures_bid <= 0 or futures_ask <= 0:
            return None

        spot_fee = self.get_spot_taker_fee()
        futures_fee = self.get_futures_taker_fee()
        threshold = self._effective_mode_a_threshold()

        best_opp: Optional[ArbOpportunity] = None

        # Direction 1: Buy spot, sell futures (futures premium is excessive)
        gross_1 = (futures_bid - spot_ask) / spot_ask * 100.0
        net_1 = gross_1 - (spot_fee + futures_fee) * 100.0

        # Direction 2: Buy futures, sell spot (futures discount is excessive)
        gross_2 = (spot_bid - futures_ask) / futures_ask * 100.0
        net_2 = gross_2 - (spot_fee + futures_fee) * 100.0

        # Pick the better direction
        if net_1 >= threshold and net_1 >= net_2:
            direction = ArbDirection.BUY_SPOT_SELL_FUTURES
            gross_pct = gross_1
            net_pct = net_1
            buy_price = spot_ask
            sell_price = futures_bid
        elif net_2 >= threshold:
            direction = ArbDirection.BUY_FUTURES_SELL_SPOT
            gross_pct = gross_2
            net_pct = net_2
            buy_price = futures_ask
            sell_price = spot_bid
        else:
            return None

        # Anomaly check — pass volume so we can distinguish genuine moves
        mid_spot = (spot_bid + spot_ask) / 2.0
        mid_futures = (futures_bid + futures_ask) / 2.0
        if self.check_price_anomaly(f"{symbol}_spot", mid_spot, volume=spot_volume):
            return None
        if self.check_price_anomaly(f"{symbol}_futures", mid_futures, volume=futures_volume):
            return None

        # Calculate trade size
        max_trade_usdt = equity * (self._mode_a_max_per_leg_pct / 100.0)
        max_by_balance = available_balance

        # Depth check: required depth >= order_size × 1.5
        max_by_depth = min(spot_depth_qty, futures_depth_qty) / self._depth_multiplier
        max_by_depth_usdt = max_by_depth * buy_price

        trade_size_usdt = min(max_trade_usdt, max_by_balance, max_by_depth_usdt)

        # Minimum viable size check
        if net_pct > 0:
            min_size = self._min_profit_usdt / (net_pct / 100.0)
            if trade_size_usdt < min_size:
                return None
        else:
            return None

        trade_size_qty = trade_size_usdt / buy_price
        net_profit_usdt = trade_size_usdt * (net_pct / 100.0)

        if net_profit_usdt < self._min_profit_usdt:
            return None

        opp = ArbOpportunity(
            mode=ArbMode.MODE_A,
            symbol=symbol,
            direction=direction,
            gross_profit_pct=gross_pct,
            net_profit_pct=net_pct,
            net_profit_usdt=net_profit_usdt,
            trade_size_usdt=trade_size_usdt,
            trade_size_qty=trade_size_qty,
            detected_at=time.time(),
            spot_bid=spot_bid,
            spot_ask=spot_ask,
            futures_bid=futures_bid,
            futures_ask=futures_ask,
            spot_depth_qty=spot_depth_qty,
            futures_depth_qty=futures_depth_qty,
        )

        self._opportunities_detected += 1
        return opp

    # ------------------------------------------------------------------
    # Mode B: Triangular arbitrage evaluation
    # ------------------------------------------------------------------

    def evaluate_mode_b(
        self,
        path: TrianglePath,
        book_tickers: Dict[str, Dict[str, float]],
        available_balance: float,
        equity: float,
    ) -> Optional[ArbOpportunity]:
        """Evaluate a triangular arbitrage opportunity.

        Parameters
        ----------
        path : TrianglePath
            The triangle to evaluate.
        book_tickers : dict
            Mapping of pair -> {"bid": float, "ask": float, "bid_qty": float, "ask_qty": float}.
        available_balance : float
            USDT available for trading.
        equity : float
            Total equity.

        Returns
        -------
        ArbOpportunity if viable, None otherwise.
        """
        pairs = path.pairs
        if len(pairs) != 3:
            return None

        # Ensure all tickers are present
        for pair in pairs:
            if pair not in book_tickers:
                return None
            t = book_tickers[pair]
            if t.get("bid", 0) <= 0 or t.get("ask", 0) <= 0:
                return None

        t0 = book_tickers[pairs[0]]  # e.g., BTCUSDT
        t1 = book_tickers[pairs[1]]  # e.g., ETHBTC
        t2 = book_tickers[pairs[2]]  # e.g., ETHUSDT

        spot_fee = self.get_spot_taker_fee()
        gross_threshold = self._effective_mode_b_gross_threshold()

        # Forward path: USDT → Asset_A → Asset_B → USDT
        # Leg 1: Buy Asset_A with USDT at pair[0] ask
        a_usdt_ask = t0["ask"]
        # Leg 2: Sell Asset_A for Asset_B at pair[1] bid
        b_a_bid = t1["bid"]
        # Leg 3: Sell Asset_B for USDT at pair[2] bid
        b_usdt_bid = t2["bid"]

        # Starting with 1 USDT:
        a_qty = 1.0 / a_usdt_ask                   # Units of A
        a_qty_after_fee = a_qty * (1.0 - spot_fee)  # After fee
        b_qty = a_qty_after_fee * b_a_bid            # Units of B
        b_qty_after_fee = b_qty * (1.0 - spot_fee)  # After fee
        final_usdt = b_qty_after_fee * b_usdt_bid    # USDT received
        final_usdt_after_fee = final_usdt * (1.0 - spot_fee)  # After fee

        gross_profit_pct = (final_usdt - 1.0) * 100.0
        net_profit_pct = (final_usdt_after_fee - 1.0) * 100.0

        # Also check reverse path: USDT → Asset_B → Asset_A → USDT
        b_usdt_ask = t2["ask"]
        b_a_ask = t1["ask"]
        a_usdt_bid = t0["bid"]

        b_qty_rev = 1.0 / b_usdt_ask
        b_qty_rev_af = b_qty_rev * (1.0 - spot_fee)
        a_qty_rev = b_qty_rev_af / b_a_ask  # Buy A with B (sell B for A)
        a_qty_rev_af = a_qty_rev * (1.0 - spot_fee)
        final_usdt_rev = a_qty_rev_af * a_usdt_bid
        final_usdt_rev_af = final_usdt_rev * (1.0 - spot_fee)

        gross_profit_rev_pct = (final_usdt_rev - 1.0) * 100.0
        net_profit_rev_pct = (final_usdt_rev_af - 1.0) * 100.0

        # Pick better direction
        if net_profit_pct >= net_profit_rev_pct and gross_profit_pct >= gross_threshold / 100.0 * 100.0:
            use_forward = True
            best_gross = gross_profit_pct
            best_net = net_profit_pct
            best_final = final_usdt_after_fee
            leg_prices = [a_usdt_ask, b_a_bid, b_usdt_bid]
            leg_sides = ["BUY", "SELL", "SELL"]
        elif net_profit_rev_pct > net_profit_pct and gross_profit_rev_pct >= gross_threshold / 100.0 * 100.0:
            use_forward = False
            best_gross = gross_profit_rev_pct
            best_net = net_profit_rev_pct
            best_final = final_usdt_rev_af
            leg_prices = [b_usdt_ask, b_a_ask, a_usdt_bid]
            leg_sides = ["BUY", "BUY", "SELL"]
        else:
            return None

        if best_net < self._mode_a_min_threshold_pct:
            return None

        # Check gross threshold
        if best_gross < gross_threshold:
            return None

        # Anomaly check on all three pairs
        for pair in pairs:
            mid = (book_tickers[pair]["bid"] + book_tickers[pair]["ask"]) / 2.0
            if self.check_price_anomaly(pair, mid):
                return None

        # Size calculation
        max_trade_usdt = equity * (self._mode_b_max_per_leg_pct / 100.0)
        max_by_balance = available_balance / 3.0  # Split across 3 legs

        # Depth check across all 3 legs
        min_depth_usdt = float("inf")
        for i, pair in enumerate(pairs):
            t = book_tickers[pair]
            if leg_sides[i] == "BUY":
                depth_qty = t.get("ask_qty", 0)
                price = t["ask"]
            else:
                depth_qty = t.get("bid_qty", 0)
                price = t["bid"]
            depth_usdt = depth_qty * price / self._depth_multiplier
            min_depth_usdt = min(min_depth_usdt, depth_usdt)

        trade_size_usdt = min(max_trade_usdt, max_by_balance, min_depth_usdt)

        # Min viable size
        if best_net > 0:
            min_size = self._min_profit_usdt / (best_net / 100.0)
            if trade_size_usdt < min_size:
                return None
        else:
            return None

        net_profit_usdt = trade_size_usdt * (best_net / 100.0)
        if net_profit_usdt < self._min_profit_usdt:
            return None

        # Calculate leg quantities
        leg_quantities = []
        remaining_usdt = trade_size_usdt
        for i, pair in enumerate(pairs):
            price = leg_prices[i]
            if i == 0:
                qty = remaining_usdt / price
            elif i == 1:
                # Quantity depends on output of leg 1
                qty = leg_quantities[0] if use_forward else remaining_usdt / price
            else:
                qty = 0  # Will be calculated from leg 2 output
            leg_quantities.append(qty)

        opp = ArbOpportunity(
            mode=ArbMode.MODE_B,
            symbol=path.path_id,
            direction=ArbDirection.BUY_SPOT_SELL_FUTURES,  # Not applicable for tri
            gross_profit_pct=best_gross,
            net_profit_pct=best_net,
            net_profit_usdt=net_profit_usdt,
            trade_size_usdt=trade_size_usdt,
            trade_size_qty=trade_size_usdt / leg_prices[0],
            detected_at=time.time(),
            triangle_path=pairs,
            leg_prices=leg_prices,
            leg_quantities=leg_quantities,
            leg_sides=leg_sides,
            final_usdt=best_final * trade_size_usdt,
            starting_usdt=trade_size_usdt,
        )

        self._opportunities_detected += 1
        return opp

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def triangle_paths(self) -> List[TrianglePath]:
        return self._triangle_paths

    @property
    def mode_a_threshold_pct(self) -> float:
        return self._effective_mode_a_threshold()

    @property
    def mode_b_gross_threshold_pct(self) -> float:
        return self._effective_mode_b_gross_threshold()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "opportunities_detected": self._opportunities_detected,
            "opportunities_taken": self._opportunities_taken,
            "opportunities_skipped": self._opportunities_skipped,
            "mode_a_threshold_pct": self._effective_mode_a_threshold(),
            "mode_b_gross_threshold_pct": self._effective_mode_b_gross_threshold(),
            "spot_taker_fee": self.get_spot_taker_fee(),
            "futures_taker_fee": self.get_futures_taker_fee(),
            "bnb_discount_active": self._has_bnb_discount,
            "high_volatility_active": self._high_volatility_active,
        }

    def record_taken(self) -> None:
        self._opportunities_taken += 1

    def record_skipped(self) -> None:
        self._opportunities_skipped += 1
