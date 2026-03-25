"""Core Funding Rate Arbitrage strategy logic.

Implements entry/exit signal generation, instrument selection and ranking,
basis spread monitoring, funding rate trend analysis, and position
lifecycle management per Sections 3-7 of the build instructions.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FundingRateRecord:
    """Single historical funding rate entry."""
    symbol: str
    rate: float           # e.g. 0.0005 = 0.05%
    timestamp_ms: int
    predicted: bool = False


@dataclass
class BasisRecord:
    """Basis spread snapshot."""
    symbol: str
    basis_pct: float
    futures_price: float
    spot_price: float
    timestamp_ms: int


@dataclass
class InstrumentState:
    """Per-instrument runtime state."""
    symbol: str
    predicted_funding_rate: float = 0.0
    current_funding_rate: float = 0.0
    next_funding_time_ms: int = 0
    mark_price: float = 0.0
    index_price: float = 0.0
    spot_best_bid: float = 0.0
    spot_best_ask: float = 0.0
    futures_best_bid: float = 0.0
    futures_best_ask: float = 0.0
    spot_volume_24h: float = 0.0
    futures_volume_24h: float = 0.0

    # Historical data
    funding_history: Deque[FundingRateRecord] = field(
        default_factory=lambda: deque(maxlen=500)
    )
    basis_history: Deque[BasisRecord] = field(
        default_factory=lambda: deque(maxlen=1440)  # 24h at 1-min resolution
    )

    # Liquidation cascade tracking
    liquidation_events: Deque[float] = field(
        default_factory=lambda: deque(maxlen=200)
    )

    # Depth snapshots
    spot_depth: Optional[Dict] = None
    futures_depth: Optional[Dict] = None

    def current_basis_pct(self) -> float:
        """Current basis = (futures - spot) / spot * 100."""
        if self.index_price <= 0:
            return 0.0
        return (self.mark_price - self.index_price) / self.index_price * 100.0

    def avg_funding_rate(self, periods: int) -> float:
        """Average of the last N actual (non-predicted) funding rates."""
        actual = [r.rate for r in self.funding_history if not r.predicted]
        if not actual:
            return 0.0
        recent = actual[-periods:]
        return sum(recent) / len(recent)

    def avg_basis_pct(self, minutes: int) -> float:
        """Average basis over the last N minutes."""
        if not self.basis_history:
            return 0.0
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - minutes * 60_000
        vals = [b.basis_pct for b in self.basis_history if b.timestamp_ms >= cutoff]
        return sum(vals) / len(vals) if vals else 0.0

    def annualized_yield(self) -> float:
        """Annualized yield based on predicted funding rate."""
        return self.predicted_funding_rate * 3 * 365 * 100.0


@dataclass
class ArbPosition:
    """Tracks a live arbitrage position (both legs)."""
    position_id: str
    symbol: str
    # Spot leg
    spot_quantity: float = 0.0
    spot_entry_price: float = 0.0
    spot_notional: float = 0.0
    # Futures leg
    futures_quantity: float = 0.0
    futures_entry_price: float = 0.0
    futures_notional: float = 0.0
    # Basis captured
    entry_basis_pct: float = 0.0
    intended_basis_pct: float = 0.0
    # Timing
    entry_time_ms: int = 0
    last_funding_check_ms: int = 0
    # Accumulated funding
    cumulative_funding_income: float = 0.0
    funding_periods_collected: int = 0
    # Exit tracking
    negative_funding_streak: int = 0
    low_rate_streak: int = 0
    daily_review_fail_streak: int = 0
    # Delta
    current_delta_pct: float = 0.0

    @property
    def holding_days(self) -> float:
        elapsed_ms = int(time.time() * 1000) - self.entry_time_ms
        return elapsed_ms / 86_400_000

    @property
    def annualized_yield(self) -> float:
        if self.spot_notional <= 0 or self.holding_days <= 0:
            return 0.0
        return (self.cumulative_funding_income / self.spot_notional) * (365.0 / self.holding_days) * 100.0


@dataclass
class EntrySignal:
    """Signal generated when entry conditions are met."""
    symbol: str
    predicted_rate: float
    avg_24h_rate: float
    avg_7d_rate: float
    basis_pct: float
    annualized_yield: float
    allocation_pct: float
    score: float
    timestamp_ms: int


@dataclass
class ExitSignal:
    """Signal to close a position."""
    position_id: str
    symbol: str
    reason: str
    urgency: str = "normal"  # "normal", "urgent", "emergency"
    partial_pct: float = 100.0  # percentage of position to close


# ---------------------------------------------------------------------------
# FundingArbStrategy
# ---------------------------------------------------------------------------

class FundingArbStrategy:
    """Core funding rate arbitrage strategy engine.

    Parameters
    ----------
    config : dict
        Strategy parameters from config.yaml['strategy_params'].
    instruments : list[str]
        Universe of tradeable instrument symbols.
    """

    def __init__(self, config: Dict[str, Any], instruments: List[str]) -> None:
        self._config = config
        self._instruments_list = instruments

        # Per-instrument state
        self.instruments: Dict[str, InstrumentState] = {
            sym: InstrumentState(symbol=sym) for sym in instruments
        }

        # Active arbitrage positions: position_id -> ArbPosition
        self.positions: Dict[str, ArbPosition] = {}

        # Circuit breaker state
        self._circuit_breaker_active = False
        self._circuit_breaker_until: float = 0.0
        self._circuit_breaker_reason: str = ""

        # Rebalancing state
        self._last_rebalance_ms: int = 0
        self._rotations_this_week: int = 0
        self._week_start_ms: int = 0

        # Consecutive loss tracking
        self._consecutive_losses: int = 0

        # BTC 7-day price tracking for market regime
        self._btc_price_7d_ago: float = 0.0
        self._btc_current_price: float = 0.0

        # Thresholds
        self._entry_threshold = self._get_threshold()
        self._total_round_trip_cost = self._calc_round_trip_cost()

        logger.info(
            "FundingArbStrategy initialized: mode=%s threshold=%.4f%% "
            "instruments=%s round_trip_cost=%.4f%%",
            config.get("entry_mode", "standard"),
            self._entry_threshold * 100,
            instruments,
            self._total_round_trip_cost * 100,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Configuration helpers
    # ══════════════════════════════════════════════════════════════════════

    def _get_threshold(self) -> float:
        """Return the entry threshold as a decimal based on mode."""
        mode = self._config.get("entry_mode", "standard")
        thresholds = {
            "aggressive": self._config.get("threshold_aggressive", 0.18) / 100.0,
            "standard": self._config.get("threshold_standard", 0.05) / 100.0,
            "conservative": self._config.get("threshold_conservative", 0.03) / 100.0,
        }
        return thresholds.get(mode, thresholds["standard"])

    def _calc_round_trip_cost(self) -> float:
        """Total round-trip cost as a decimal."""
        spot_fee = self._config.get("spot_taker_fee_pct", 0.10) / 100.0
        futures_fee = self._config.get("futures_taker_fee_pct", 0.04) / 100.0
        slippage = self._config.get("estimated_slippage_per_leg_pct", 0.02) / 100.0
        return (spot_fee * 2) + (futures_fee * 2) + (slippage * 4)

    def update_entry_mode(self, mode: str) -> None:
        """Hot-update the entry mode (from dashboard)."""
        if mode in ("aggressive", "standard", "conservative"):
            self._config["entry_mode"] = mode
            self._entry_threshold = self._get_threshold()
            logger.info("Entry mode updated to %s, threshold=%.4f%%",
                        mode, self._entry_threshold * 100)

    # ══════════════════════════════════════════════════════════════════════
    #  Data ingestion callbacks (called from WebSocket handlers)
    # ══════════════════════════════════════════════════════════════════════

    async def on_mark_price(self, data: Dict[str, Any]) -> None:
        """Process markPrice stream update (1s interval)."""
        symbol = data.get("s", "")
        inst = self.instruments.get(symbol)
        if inst is None:
            return

        inst.mark_price = float(data.get("p", 0))
        inst.index_price = float(data.get("i", 0))
        inst.predicted_funding_rate = float(data.get("r", 0))
        inst.next_funding_time_ms = int(data.get("T", 0))

        # Record basis
        if inst.index_price > 0:
            basis_pct = inst.current_basis_pct()
            inst.basis_history.append(BasisRecord(
                symbol=symbol,
                basis_pct=basis_pct,
                futures_price=inst.mark_price,
                spot_price=inst.index_price,
                timestamp_ms=int(time.time() * 1000),
            ))

        # Update BTC price for regime detection
        if symbol == "BTCUSDT":
            self._btc_current_price = inst.mark_price

    async def on_spot_book_ticker(self, data: Dict[str, Any]) -> None:
        """Process spot bookTicker update."""
        symbol = data.get("s", "")
        inst = self.instruments.get(symbol)
        if inst is None:
            return
        inst.spot_best_bid = float(data.get("b", 0))
        inst.spot_best_ask = float(data.get("a", 0))

    async def on_futures_book_ticker(self, data: Dict[str, Any]) -> None:
        """Process futures bookTicker update."""
        symbol = data.get("s", "")
        inst = self.instruments.get(symbol)
        if inst is None:
            return
        inst.futures_best_bid = float(data.get("b", 0))
        inst.futures_best_ask = float(data.get("a", 0))

    async def on_force_order(self, data: Dict[str, Any]) -> None:
        """Process forceOrder (liquidation) stream event."""
        order = data.get("o", data)
        symbol = order.get("s", "")
        inst = self.instruments.get(symbol)
        if inst is None:
            return
        inst.liquidation_events.append(time.time())

    async def on_depth_update(self, symbol: str, market: str, data: Dict) -> None:
        """Store depth snapshot for pre-entry checks."""
        inst = self.instruments.get(symbol)
        if inst is None:
            return
        if market == "spot":
            inst.spot_depth = data
        else:
            inst.futures_depth = data

    def ingest_funding_rate(self, symbol: str, rate: float, timestamp_ms: int,
                            predicted: bool = False) -> None:
        """Add a funding rate record (from REST warm-up or settlement check)."""
        inst = self.instruments.get(symbol)
        if inst is None:
            return
        inst.funding_history.append(FundingRateRecord(
            symbol=symbol, rate=rate, timestamp_ms=timestamp_ms, predicted=predicted
        ))

    # ══════════════════════════════════════════════════════════════════════
    #  Entry signal generation (Section 3)
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_entries(self, equity: float) -> List[EntrySignal]:
        """Evaluate all instruments for potential entry.

        Returns a ranked list of EntrySignals for instruments that pass all
        entry conditions, with capital allocation percentages.
        """
        if self._circuit_breaker_active:
            if time.time() < self._circuit_breaker_until:
                logger.debug("Circuit breaker active: %s", self._circuit_breaker_reason)
                return []
            else:
                self._circuit_breaker_active = False
                logger.info("Circuit breaker expired, re-evaluating entries")

        # Don't enter within 5 minutes of funding settlement
        if self._near_funding_settlement():
            logger.debug("Within 5 minutes of funding settlement, skipping entry")
            return []

        candidates: List[Tuple[float, str]] = []

        for symbol, inst in self.instruments.items():
            # Skip if already holding this instrument
            if self._has_position(symbol):
                continue

            # Step 1: Funding rate threshold
            if inst.predicted_funding_rate <= self._entry_threshold:
                continue

            # Step 3: 24h average > 50% of threshold
            avg_24h = inst.avg_funding_rate(3)  # last 3 periods = 24h
            if avg_24h <= self._entry_threshold * 0.5:
                continue

            # Step 3: 7-day average must be positive
            avg_7d = inst.avg_funding_rate(21)  # last 21 periods = 7 days
            if avg_7d <= 0:
                continue

            # Section 7.3: Funding rate sustainability
            # At least 3 of last 6 periods above threshold
            min_periods = self._config.get("funding_sustainability_min_above", 3)
            check_periods = self._config.get("funding_sustainability_periods", 6)
            if not self._check_funding_sustainability(inst, check_periods, min_periods):
                continue

            # Section 7.3: Not a single spike (current > 3x 7-day avg)
            if avg_7d > 0 and inst.predicted_funding_rate > avg_7d * 3:
                # Check if there's sustained elevation
                if not self._check_funding_sustainability(inst, 6, 3):
                    logger.info(
                        "%s: Skipping — spike detected (predicted=%.4f%%, 7d_avg=%.4f%%)",
                        symbol, inst.predicted_funding_rate * 100, avg_7d * 100,
                    )
                    continue

            # Step 4-5: Basis spread check
            if not self._check_basis_conditions(inst):
                continue

            # Section 7.1: Minimum liquidity
            if not self._check_liquidity(inst):
                continue

            # Section 7.2: Spread filter
            if not self._check_spreads(inst):
                continue

            # Section 7.5: Fee threshold
            if not self._check_fee_viability(inst):
                continue

            # Section 7.4: Market regime filter
            size_multiplier = self._market_regime_multiplier()
            if size_multiplier <= 0:
                continue

            annualized = inst.annualized_yield()
            candidates.append((annualized, symbol))

        if not candidates:
            return []

        # Step 1 of Section 6.2: Rank by annualized yield (descending)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Limit to max 5 instruments
        candidates = candidates[:5]

        # Step 2: Proportional allocation with caps
        allocation_caps = self._config.get("allocation_caps", [20.0, 15.0, 10.0, 5.0, 5.0])
        max_strategy_pct = 40.0  # Max capital for this strategy
        regime_mult = self._market_regime_multiplier()

        signals: List[EntrySignal] = []
        total_allocated = self._current_allocation_pct(equity)

        for rank, (ann_yield, symbol) in enumerate(candidates):
            inst = self.instruments[symbol]
            cap_pct = allocation_caps[rank] if rank < len(allocation_caps) else 5.0
            cap_pct *= regime_mult

            # Don't exceed strategy max
            remaining = max_strategy_pct - total_allocated
            alloc = min(cap_pct, remaining)
            if alloc <= 0:
                break

            signals.append(EntrySignal(
                symbol=symbol,
                predicted_rate=inst.predicted_funding_rate,
                avg_24h_rate=inst.avg_funding_rate(3),
                avg_7d_rate=inst.avg_funding_rate(21),
                basis_pct=inst.current_basis_pct(),
                annualized_yield=ann_yield,
                allocation_pct=alloc,
                score=ann_yield,
                timestamp_ms=int(time.time() * 1000),
            ))
            total_allocated += alloc

        return signals

    # ══════════════════════════════════════════════════════════════════════
    #  Exit signal generation (Section 4)
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_exits(self) -> List[ExitSignal]:
        """Evaluate all open positions for potential exit conditions.

        Returns a list of ExitSignals for positions that should be closed.
        """
        signals: List[ExitSignal] = []

        for pos_id, pos in self.positions.items():
            inst = self.instruments.get(pos.symbol)
            if inst is None:
                signals.append(ExitSignal(
                    position_id=pos_id, symbol=pos.symbol,
                    reason="instrument_removed", urgency="urgent",
                ))
                continue

            # Section 4.6: Maximum holding period (30 days)
            max_days = self._config.get("max_holding_days", 30)
            if pos.holding_days >= max_days:
                signals.append(ExitSignal(
                    position_id=pos_id, symbol=pos.symbol,
                    reason=f"max_holding_period_{max_days}d",
                ))
                continue

            # Section 4.1: Funding rate reversal — negative for N periods
            neg_threshold = self._config.get("negative_funding_threshold", -0.005) / 100.0
            neg_periods = self._config.get("negative_periods_for_exit", 2)
            if pos.negative_funding_streak >= neg_periods:
                signals.append(ExitSignal(
                    position_id=pos_id, symbol=pos.symbol,
                    reason=f"negative_funding_{pos.negative_funding_streak}_periods",
                    urgency="urgent",
                ))
                continue

            # Section 4.1 mode-specific: rate below minimum for N periods
            mode = self._config.get("entry_mode", "standard")
            if mode == "standard":
                low_threshold = self._config.get("low_rate_threshold", 0.01) / 100.0
                low_periods = self._config.get("low_rate_periods_for_exit", 3)
                if pos.low_rate_streak >= low_periods:
                    signals.append(ExitSignal(
                        position_id=pos_id, symbol=pos.symbol,
                        reason=f"below_minimum_rate_{pos.low_rate_streak}_periods",
                    ))
                    continue
            elif mode == "conservative":
                if inst.predicted_funding_rate < 0.0002:  # 0.02%
                    if pos.low_rate_streak >= 2:
                        signals.append(ExitSignal(
                            position_id=pos_id, symbol=pos.symbol,
                            reason="conservative_low_rate_exit",
                        ))
                        continue

            # Section 4.2: Basis inversion
            basis_inversion = self._config.get("basis_inversion_exit", -0.05) / 100.0
            current_basis = inst.current_basis_pct() / 100.0
            if current_basis < basis_inversion:
                signals.append(ExitSignal(
                    position_id=pos_id, symbol=pos.symbol,
                    reason=f"basis_inversion_{current_basis*100:.3f}%",
                    urgency="emergency",
                ))
                continue

            # Section 4.2: Gradual exit on near-zero basis
            min_basis = self._config.get("min_basis_pct", 0.01) / 100.0
            avg_4h_basis = inst.avg_basis_pct(240) / 100.0
            if avg_4h_basis < min_basis and avg_4h_basis > basis_inversion:
                # Check if it's been below for 4+ hours
                signals.append(ExitSignal(
                    position_id=pos_id, symbol=pos.symbol,
                    reason="basis_near_zero_gradual",
                    partial_pct=25.0,
                ))
                continue

            # Section 4.4: Profit target with rate decline
            profit_target = self._config.get("profit_target_pct", 5.0) / 100.0
            if pos.spot_notional > 0:
                funding_yield = pos.cumulative_funding_income / pos.spot_notional
                if funding_yield >= profit_target and inst.predicted_funding_rate < self._entry_threshold:
                    signals.append(ExitSignal(
                        position_id=pos_id, symbol=pos.symbol,
                        reason=f"profit_target_reached_{funding_yield*100:.2f}%",
                    ))
                    continue

            # Section 4.5: Time-based review (24h)
            min_annualized = self._config.get("daily_review_min_annualized_yield", 8.0)
            max_fails = self._config.get("daily_review_consecutive_fails", 3)
            if pos.daily_review_fail_streak >= max_fails:
                signals.append(ExitSignal(
                    position_id=pos_id, symbol=pos.symbol,
                    reason=f"daily_review_failed_{max_fails}_consecutive",
                ))
                continue

            # Section 7.1: Liquidity dropped while in position
            if not self._check_liquidity(inst):
                signals.append(ExitSignal(
                    position_id=pos_id, symbol=pos.symbol,
                    reason="liquidity_below_minimum",
                    partial_pct=25.0,  # Gradual unwind
                ))

        return signals

    # ══════════════════════════════════════════════════════════════════════
    #  Pre-funding tactical exit (Section 4.7)
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_tactical_exit(self, symbol: str) -> Optional[ExitSignal]:
        """Check if a tactical pre-funding exit is warranted.

        Called ~10 minutes before funding settlement.
        Returns an ExitSignal for the futures leg only if the predicted
        negative funding payment exceeds round-trip cost.
        """
        inst = self.instruments.get(symbol)
        pos = self._get_position_for_symbol(symbol)
        if inst is None or pos is None:
            return None

        tactical_threshold = self._config.get("tactical_exit_negative_threshold", -0.1) / 100.0
        if inst.predicted_funding_rate >= tactical_threshold:
            return None

        # Cost of closing + reopening futures leg
        futures_fee = self._config.get("futures_taker_fee_pct", 0.04) / 100.0
        round_trip_cost = pos.futures_notional * futures_fee * 2

        # Expected negative payment
        negative_payment = abs(inst.predicted_funding_rate) * pos.futures_notional

        if negative_payment > round_trip_cost:
            logger.info(
                "%s: Tactical pre-funding exit — negative_payment=%.4f > cost=%.4f",
                symbol, negative_payment, round_trip_cost,
            )
            return ExitSignal(
                position_id=pos.position_id,
                symbol=symbol,
                reason="tactical_prefunding_exit",
                urgency="normal",
            )

        return None

    # ══════════════════════════════════════════════════════════════════════
    #  Rebalancing (Section 6.3)
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_rebalancing(self, equity: float) -> Tuple[List[ExitSignal], List[EntrySignal]]:
        """Weekly rebalancing evaluation.

        Returns (positions_to_close, instruments_to_open).
        """
        max_rotations = self._config.get("max_rotations_per_week", 2)
        if self._rotations_this_week >= max_rotations:
            logger.debug("Max rotations this week reached (%d)", max_rotations)
            return [], []

        # Get current holdings ranked by yield
        current_yields: Dict[str, float] = {}
        for pos_id, pos in self.positions.items():
            inst = self.instruments.get(pos.symbol)
            if inst:
                current_yields[pos.symbol] = inst.annualized_yield()

        # Get candidates not currently held
        entry_signals = self.evaluate_entries(equity)
        if not entry_signals:
            return [], []

        exits: List[ExitSignal] = []
        entries: List[EntrySignal] = []

        # Find instruments with declining yield that could be replaced
        for symbol, current_yield in sorted(current_yields.items(), key=lambda x: x[0]):
            if self._rotations_this_week >= max_rotations:
                break

            inst = self.instruments.get(symbol)
            if inst is None:
                continue

            # Only rotate if current yield is below threshold
            if inst.predicted_funding_rate > self._entry_threshold:
                continue

            # Find a better replacement
            for signal in entry_signals:
                if signal.symbol == symbol:
                    continue
                if signal.annualized_yield > current_yield * 1.5:  # 50% better
                    pos = self._get_position_for_symbol(symbol)
                    if pos:
                        exits.append(ExitSignal(
                            position_id=pos.position_id,
                            symbol=symbol,
                            reason=f"rebalance_rotation_to_{signal.symbol}",
                        ))
                        entries.append(signal)
                        self._rotations_this_week += 1
                    break

        return exits, entries

    def reset_weekly_rotation_count(self) -> None:
        """Reset rotation counter (called Monday 00:00 UTC)."""
        self._rotations_this_week = 0
        self._week_start_ms = int(time.time() * 1000)
        logger.info("Weekly rotation count reset")

    # ══════════════════════════════════════════════════════════════════════
    #  Circuit breakers (Section 5.7)
    # ══════════════════════════════════════════════════════════════════════

    def check_circuit_breakers(self) -> Optional[ExitSignal]:
        """Check all circuit breaker conditions.

        Returns an emergency ExitSignal affecting ALL positions if triggered.
        """
        for symbol, inst in self.instruments.items():
            if not self._has_position(symbol):
                continue

            # Basis flash inversion
            max_inversion = self._config.get("basis_flash_inversion_pct", 0.5) / 100.0
            window_min = self._config.get("basis_flash_window_minutes", 5)
            if self._detect_basis_flash(inst, max_inversion, window_min):
                cooldown_h = self._config.get("basis_flash_cooldown_hours", 4)
                self._activate_circuit_breaker(
                    f"basis_flash_inversion_{symbol}",
                    cooldown_h * 3600,
                )
                return ExitSignal(
                    position_id="ALL", symbol=symbol,
                    reason=f"circuit_breaker_basis_flash_{symbol}",
                    urgency="emergency",
                )

            # Funding rate shock
            shock_threshold = self._config.get("funding_shock_threshold", -0.2) / 100.0
            if inst.predicted_funding_rate < shock_threshold:
                cooldown_h = self._config.get("funding_shock_cooldown_hours", 24)
                self._activate_circuit_breaker(
                    f"funding_shock_{symbol}",
                    cooldown_h * 3600,
                )
                return ExitSignal(
                    position_id="ALL", symbol=symbol,
                    reason=f"circuit_breaker_funding_shock_{symbol}",
                    urgency="emergency",
                )

            # Liquidation cascade detection
            cascade_count = self._config.get("liquidation_cascade_count", 50)
            cascade_window = self._config.get("liquidation_cascade_window_s", 60)
            if self._detect_liquidation_cascade(inst, cascade_count, cascade_window):
                return ExitSignal(
                    position_id="ALL", symbol=symbol,
                    reason=f"circuit_breaker_liquidation_cascade_{symbol}",
                    urgency="emergency",
                )

            # Exchange anomaly: spot/futures divergence > 2% for > 60s
            if inst.index_price > 0 and inst.mark_price > 0:
                divergence = abs(inst.mark_price - inst.index_price) / inst.index_price
                if divergence > 0.02:
                    logger.warning(
                        "%s: Exchange anomaly — divergence %.2f%%",
                        symbol, divergence * 100,
                    )

        return None

    def _activate_circuit_breaker(self, reason: str, duration_s: float) -> None:
        """Activate circuit breaker with cooldown period."""
        self._circuit_breaker_active = True
        self._circuit_breaker_until = time.time() + duration_s
        self._circuit_breaker_reason = reason
        logger.critical(
            "CIRCUIT BREAKER activated: %s — cooldown %.0f seconds",
            reason, duration_s,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Funding rate update (called after each settlement)
    # ══════════════════════════════════════════════════════════════════════

    def update_position_funding(self, symbol: str, actual_rate: float,
                                payment: float) -> None:
        """Update position tracking after a funding settlement.

        Called by FundingTracker after each 8-hour settlement.
        """
        pos = self._get_position_for_symbol(symbol)
        if pos is None:
            return

        pos.cumulative_funding_income += payment
        pos.funding_periods_collected += 1
        pos.last_funding_check_ms = int(time.time() * 1000)

        # Track negative/low-rate streaks for exit logic
        neg_threshold = self._config.get("negative_funding_threshold", -0.005) / 100.0
        low_threshold = self._config.get("low_rate_threshold", 0.01) / 100.0

        if actual_rate < neg_threshold:
            pos.negative_funding_streak += 1
        else:
            pos.negative_funding_streak = 0

        if actual_rate < low_threshold:
            pos.low_rate_streak += 1
        else:
            pos.low_rate_streak = 0

        trade_logger.info(
            "FUNDING\tsymbol=%s\trate=%.6f\tpayment=%.6f\t"
            "cumulative=%.6f\tperiods=%d\tneg_streak=%d\tlow_streak=%d",
            symbol, actual_rate, payment,
            pos.cumulative_funding_income, pos.funding_periods_collected,
            pos.negative_funding_streak, pos.low_rate_streak,
        )

    def daily_review(self, symbol: str) -> None:
        """Section 4.5: Daily position review."""
        pos = self._get_position_for_symbol(symbol)
        inst = self.instruments.get(symbol)
        if pos is None or inst is None:
            return

        min_annualized = self._config.get("daily_review_min_annualized_yield", 8.0)
        failed = False

        # Is 24h avg funding rate above minimum threshold?
        avg_24h = inst.avg_funding_rate(3)
        if avg_24h < self._entry_threshold:
            failed = True

        # Is basis still positive?
        if inst.current_basis_pct() <= 0:
            failed = True

        # Is annualized yield still attractive?
        if pos.annualized_yield < min_annualized:
            failed = True

        if failed:
            pos.daily_review_fail_streak += 1
            logger.info(
                "%s: Daily review FAILED (streak=%d) — avg24h=%.4f%% basis=%.4f%% yield=%.2f%%",
                symbol, pos.daily_review_fail_streak,
                avg_24h * 100, inst.current_basis_pct(), pos.annualized_yield,
            )
        else:
            pos.daily_review_fail_streak = 0
            logger.info(
                "%s: Daily review PASSED — avg24h=%.4f%% basis=%.4f%% yield=%.2f%%",
                symbol, avg_24h * 100, inst.current_basis_pct(), pos.annualized_yield,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  Position management
    # ══════════════════════════════════════════════════════════════════════

    def create_position(self, signal: EntrySignal, spot_fill: Dict,
                        futures_fill: Dict) -> ArbPosition:
        """Create an ArbPosition from executed fills."""
        pos_id = f"FA-{signal.symbol}-{uuid.uuid4().hex[:8]}"
        spot_price = float(spot_fill.get("avgPrice", spot_fill.get("price", 0)))
        spot_qty = float(spot_fill.get("executedQty", spot_fill.get("origQty", 0)))
        futures_price = float(futures_fill.get("avgPrice", futures_fill.get("price", 0)))
        futures_qty = float(futures_fill.get("executedQty", futures_fill.get("origQty", 0)))

        actual_basis = (futures_price - spot_price) / spot_price if spot_price > 0 else 0

        pos = ArbPosition(
            position_id=pos_id,
            symbol=signal.symbol,
            spot_quantity=spot_qty,
            spot_entry_price=spot_price,
            spot_notional=spot_qty * spot_price,
            futures_quantity=futures_qty,
            futures_entry_price=futures_price,
            futures_notional=futures_qty * futures_price,
            entry_basis_pct=actual_basis * 100,
            intended_basis_pct=signal.basis_pct,
            entry_time_ms=int(time.time() * 1000),
        )

        self.positions[pos_id] = pos
        self._consecutive_losses = 0  # Reset on new entry

        trade_logger.info(
            "OPEN\tpos_id=%s\tsymbol=%s\tspot_qty=%.8f\tspot_px=%.4f\t"
            "fut_qty=%.8f\tfut_px=%.4f\tbasis=%.4f%%\tintended_basis=%.4f%%",
            pos_id, signal.symbol, spot_qty, spot_price,
            futures_qty, futures_price, actual_basis * 100, signal.basis_pct,
        )

        # Flag if basis capture is significantly worse than intended
        if actual_basis < (signal.basis_pct / 100.0) - 0.0005:
            logger.warning(
                "%s: Basis capture worse than intended: actual=%.4f%% intended=%.4f%%",
                signal.symbol, actual_basis * 100, signal.basis_pct,
            )

        return pos

    def close_position(self, position_id: str, spot_exit_price: float,
                       futures_exit_price: float, fees: float) -> Optional[Dict]:
        """Close a position and record the result."""
        pos = self.positions.pop(position_id, None)
        if pos is None:
            logger.warning("Position %s not found for closing", position_id)
            return None

        # Calculate PnL
        spot_pnl = (spot_exit_price - pos.spot_entry_price) * pos.spot_quantity
        futures_pnl = (pos.futures_entry_price - futures_exit_price) * pos.futures_quantity
        total_pnl = spot_pnl + futures_pnl + pos.cumulative_funding_income - fees

        result = {
            "position_id": position_id,
            "symbol": pos.symbol,
            "spot_pnl": spot_pnl,
            "futures_pnl": futures_pnl,
            "funding_income": pos.cumulative_funding_income,
            "fees": fees,
            "total_pnl": total_pnl,
            "holding_days": pos.holding_days,
            "funding_periods": pos.funding_periods_collected,
            "entry_basis_pct": pos.entry_basis_pct,
            "annualized_yield": pos.annualized_yield,
        }

        is_win = total_pnl > 0
        if not is_win:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        trade_logger.info(
            "CLOSE\tpos_id=%s\tsymbol=%s\ttotal_pnl=%.4f\tfunding=%.4f\t"
            "fees=%.4f\thold_days=%.2f\tconsec_loss=%d",
            position_id, pos.symbol, total_pnl, pos.cumulative_funding_income,
            fees, pos.holding_days, self._consecutive_losses,
        )

        return result

    # ══════════════════════════════════════════════════════════════════════
    #  Orphan detection (Section 8.2)
    # ══════════════════════════════════════════════════════════════════════

    def detect_orphans(
        self,
        spot_balances: Dict[str, float],
        futures_positions: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Detect orphaned legs: spot without futures or vice versa.

        Parameters
        ----------
        spot_balances : dict
            symbol -> quantity of base asset held in spot.
        futures_positions : dict
            symbol -> signed position amount (negative = short).

        Returns
        -------
        list of dicts describing each orphan with recommended action.
        """
        orphans: List[Dict[str, Any]] = []
        all_symbols = set(spot_balances.keys()) | set(futures_positions.keys())

        for symbol in all_symbols:
            spot_qty = spot_balances.get(symbol, 0.0)
            futures_qty = futures_positions.get(symbol, 0.0)

            has_spot = abs(spot_qty) > 1e-8
            has_futures = abs(futures_qty) > 1e-8

            if has_spot and not has_futures:
                orphans.append({
                    "symbol": symbol,
                    "type": "spot_without_futures",
                    "spot_qty": spot_qty,
                    "action": "open_matching_futures_short",
                })
                logger.critical(
                    "ORPHAN DETECTED: %s has spot (%.8f) but no futures short",
                    symbol, spot_qty,
                )
            elif has_futures and not has_spot:
                orphans.append({
                    "symbol": symbol,
                    "type": "futures_without_spot",
                    "futures_qty": futures_qty,
                    "action": "buy_matching_spot",
                })
                logger.critical(
                    "ORPHAN DETECTED: %s has futures (%.8f) but no spot",
                    symbol, futures_qty,
                )

        return orphans

    # ══════════════════════════════════════════════════════════════════════
    #  State serialization
    # ══════════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """Serialize strategy state for persistence."""
        positions_data = []
        for pos_id, pos in self.positions.items():
            positions_data.append({
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "spot_quantity": pos.spot_quantity,
                "spot_entry_price": pos.spot_entry_price,
                "spot_notional": pos.spot_notional,
                "futures_quantity": pos.futures_quantity,
                "futures_entry_price": pos.futures_entry_price,
                "futures_notional": pos.futures_notional,
                "entry_basis_pct": pos.entry_basis_pct,
                "entry_time_ms": pos.entry_time_ms,
                "cumulative_funding_income": pos.cumulative_funding_income,
                "funding_periods_collected": pos.funding_periods_collected,
                "negative_funding_streak": pos.negative_funding_streak,
                "low_rate_streak": pos.low_rate_streak,
                "daily_review_fail_streak": pos.daily_review_fail_streak,
            })

        return {
            "positions": positions_data,
            "circuit_breaker_active": self._circuit_breaker_active,
            "circuit_breaker_until": self._circuit_breaker_until,
            "consecutive_losses": self._consecutive_losses,
            "rotations_this_week": self._rotations_this_week,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore strategy state from persistence."""
        self._circuit_breaker_active = state.get("circuit_breaker_active", False)
        self._circuit_breaker_until = state.get("circuit_breaker_until", 0)
        self._consecutive_losses = state.get("consecutive_losses", 0)
        self._rotations_this_week = state.get("rotations_this_week", 0)

        for p in state.get("positions", []):
            pos = ArbPosition(
                position_id=p["position_id"],
                symbol=p["symbol"],
                spot_quantity=p.get("spot_quantity", 0),
                spot_entry_price=p.get("spot_entry_price", 0),
                spot_notional=p.get("spot_notional", 0),
                futures_quantity=p.get("futures_quantity", 0),
                futures_entry_price=p.get("futures_entry_price", 0),
                futures_notional=p.get("futures_notional", 0),
                entry_basis_pct=p.get("entry_basis_pct", 0),
                entry_time_ms=p.get("entry_time_ms", 0),
                cumulative_funding_income=p.get("cumulative_funding_income", 0),
                funding_periods_collected=p.get("funding_periods_collected", 0),
                negative_funding_streak=p.get("negative_funding_streak", 0),
                low_rate_streak=p.get("low_rate_streak", 0),
                daily_review_fail_streak=p.get("daily_review_fail_streak", 0),
            )
            self.positions[pos.position_id] = pos

        logger.info("Strategy state restored: %d positions", len(self.positions))

    # ══════════════════════════════════════════════════════════════════════
    #  Metrics (Section 10.2)
    # ══════════════════════════════════════════════════════════════════════

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Return all Section 10.2 strategy-specific metrics."""
        total_funding = sum(p.cumulative_funding_income for p in self.positions.values())
        total_notional = sum(p.spot_notional for p in self.positions.values())

        # Per-instrument breakdown
        per_instrument: Dict[str, Dict] = {}
        for pos in self.positions.values():
            inst = self.instruments.get(pos.symbol)
            per_instrument[pos.symbol] = {
                "funding_income": pos.cumulative_funding_income,
                "holding_days": pos.holding_days,
                "annualized_yield": pos.annualized_yield,
                "funding_periods": pos.funding_periods_collected,
                "predicted_rate": inst.predicted_funding_rate if inst else 0,
                "current_basis": inst.current_basis_pct() if inst else 0,
                "delta_pct": pos.current_delta_pct,
            }

        # Funding rate regime classification
        regime_breakdown = {"low": 0, "medium": 0, "high": 0}
        for inst in self.instruments.values():
            rate = inst.predicted_funding_rate
            if rate < 0.0003:
                regime_breakdown["low"] += 1
            elif rate < 0.001:
                regime_breakdown["medium"] += 1
            else:
                regime_breakdown["high"] += 1

        return {
            "cumulative_funding_income": total_funding,
            "total_notional_deployed": total_notional,
            "active_positions": len(self.positions),
            "per_instrument": per_instrument,
            "consecutive_losses": self._consecutive_losses,
            "circuit_breaker_active": self._circuit_breaker_active,
            "circuit_breaker_reason": self._circuit_breaker_reason if self._circuit_breaker_active else "",
            "rotations_this_week": self._rotations_this_week,
            "funding_regime": regime_breakdown,
            "entry_mode": self._config.get("entry_mode", "standard"),
            "entry_threshold_pct": self._entry_threshold * 100,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  Private helpers
    # ══════════════════════════════════════════════════════════════════════

    def _has_position(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self.positions.values())

    def _get_position_for_symbol(self, symbol: str) -> Optional[ArbPosition]:
        for p in self.positions.values():
            if p.symbol == symbol:
                return p
        return None

    def _current_allocation_pct(self, equity: float) -> float:
        """Current total allocation as % of equity."""
        if equity <= 0:
            return 0.0
        total = sum(p.spot_notional for p in self.positions.values())
        return (total / equity) * 100.0

    def _near_funding_settlement(self) -> bool:
        """Return True if within 5 minutes of any funding timestamp."""
        now = datetime.now(timezone.utc)
        minutes_into_8h = (now.hour * 60 + now.minute) % (8 * 60)
        # Settlement at 0 minutes into each 8h block
        # Within 5 minutes before settlement
        if minutes_into_8h >= (8 * 60 - 5):
            return True
        # Within 5 minutes after settlement
        if minutes_into_8h <= 5:
            return True
        return False

    def _check_basis_conditions(self, inst: InstrumentState) -> bool:
        """Section 3.2: Basis spread checks."""
        min_basis = self._config.get("min_basis_pct", 0.01) / 100.0
        current_basis = inst.current_basis_pct() / 100.0

        # Must be positive and above minimum
        if current_basis <= min_basis:
            return False

        # 1h avg must be positive
        avg_1h = inst.avg_basis_pct(60) / 100.0
        if avg_1h <= 0:
            return False

        # 1h avg not declining (compare to previous hour)
        avg_prev_1h = inst.avg_basis_pct(120) / 100.0
        # Allow stable within 0.01%
        if avg_1h < avg_prev_1h - 0.0001:
            # Declining — check if it's significant
            if avg_prev_1h > 0 and (avg_prev_1h - avg_1h) / avg_prev_1h > 0.1:
                return False

        # Basis contraction > 50% in 4 hours
        max_contraction = self._config.get("basis_contraction_max_pct", 50.0) / 100.0
        avg_4h = inst.avg_basis_pct(240) / 100.0
        if avg_4h > 0 and current_basis < avg_4h * (1 - max_contraction):
            logger.info(
                "%s: Basis contracted >%.0f%% in 4h (current=%.4f%% avg4h=%.4f%%)",
                inst.symbol, max_contraction * 100, current_basis * 100, avg_4h * 100,
            )
            return False

        return True

    def _check_funding_sustainability(self, inst: InstrumentState,
                                       periods: int, min_above: int) -> bool:
        """Section 7.3: Check that funding has been sustainably elevated."""
        actual = [r.rate for r in inst.funding_history if not r.predicted]
        if len(actual) < periods:
            return False
        recent = actual[-periods:]
        above = sum(1 for r in recent if r > self._entry_threshold)
        return above >= min_above

    def _check_liquidity(self, inst: InstrumentState) -> bool:
        """Section 7.1: Minimum liquidity filter."""
        min_spot = self._config.get("min_spot_volume_24h", 50_000_000)
        min_futures = self._config.get("min_futures_volume_24h", 200_000_000)
        if inst.spot_volume_24h < min_spot:
            return False
        if inst.futures_volume_24h < min_futures:
            return False
        return True

    def _check_spreads(self, inst: InstrumentState) -> bool:
        """Section 7.2: Spread filter."""
        max_spot_spread = self._config.get("max_spot_spread_pct", 0.05) / 100.0
        max_fut_spread = self._config.get("max_futures_spread_pct", 0.03) / 100.0

        if inst.spot_best_ask > 0 and inst.spot_best_bid > 0:
            mid = (inst.spot_best_ask + inst.spot_best_bid) / 2
            if mid > 0:
                spread = (inst.spot_best_ask - inst.spot_best_bid) / mid
                if spread > max_spot_spread:
                    return False

        if inst.futures_best_ask > 0 and inst.futures_best_bid > 0:
            mid = (inst.futures_best_ask + inst.futures_best_bid) / 2
            if mid > 0:
                spread = (inst.futures_best_ask - inst.futures_best_bid) / mid
                if spread > max_fut_spread:
                    return False

        return True

    def _check_fee_viability(self, inst: InstrumentState) -> bool:
        """Section 7.5: Fee threshold check."""
        min_multiple = self._config.get("min_fee_multiple", 3.0)
        # Expected holding = at least 3 periods for aggressive, more for others
        min_periods = 3
        expected_income = inst.predicted_funding_rate * min_periods
        return expected_income >= self._total_round_trip_cost * min_multiple

    def _market_regime_multiplier(self) -> float:
        """Section 7.4: Market regime position size multiplier."""
        if self._btc_price_7d_ago <= 0 or self._btc_current_price <= 0:
            return 1.0

        change_7d = (self._btc_current_price - self._btc_price_7d_ago) / self._btc_price_7d_ago

        fear_threshold = self._config.get("extreme_fear_btc_drop_7d_pct", 15.0) / 100.0
        greed_threshold = self._config.get("extreme_greed_btc_rise_7d_pct", 30.0) / 100.0
        greed_reduction = self._config.get("extreme_greed_size_reduction", 0.5)

        if change_7d < -fear_threshold:
            logger.info("Market regime: EXTREME FEAR (BTC -%.1f%% in 7d), pausing entries",
                        abs(change_7d) * 100)
            return 0.0
        elif change_7d > greed_threshold:
            logger.info("Market regime: EXTREME GREED (BTC +%.1f%% in 7d), reducing size %.0f%%",
                        change_7d * 100, greed_reduction * 100)
            return greed_reduction

        return 1.0

    def _detect_basis_flash(self, inst: InstrumentState,
                            max_inversion: float, window_min: int) -> bool:
        """Detect basis flash inversion within time window."""
        if not inst.basis_history:
            return False

        now_ms = int(time.time() * 1000)
        cutoff = now_ms - window_min * 60_000
        recent = [b for b in inst.basis_history if b.timestamp_ms >= cutoff]
        if not recent:
            return False

        min_basis = min(b.basis_pct for b in recent)
        max_basis = max(b.basis_pct for b in recent)

        # Check if basis went from positive to inverted by > threshold
        if min_basis < -max_inversion * 100 and max_basis > 0:
            return True
        return False

    def _detect_liquidation_cascade(self, inst: InstrumentState,
                                     count: int, window_s: int) -> bool:
        """Detect if liquidation events exceed threshold."""
        now = time.time()
        cutoff = now - window_s
        recent = sum(1 for t in inst.liquidation_events if t >= cutoff)
        return recent >= count
