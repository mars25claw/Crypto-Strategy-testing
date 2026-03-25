"""Core Signal-Enhanced DCA strategy logic.

Implements:
- Signal calculation: RSI(14), Fear&Greed, Price/200-SMA, SOPR (optional)
- Geometric mean of signals clamped [0.25, 3.0]
- Scheduled DCA execution (daily/weekly/bi-weekly/monthly)
- Crash-buy detection (3 levels with cooldowns)
- Value averaging mode (alternative)
- Paper trading simulation
- All Section 10.2 metrics
- All Section 11 edge cases
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.binance_client import BinanceClient, BinanceClientError
from shared.config_loader import BotConfig
from shared.external_data import FearGreedClient, GlassnodeClient
from shared.indicators import IndicatorBuffer, rsi as calc_rsi, sma as calc_sma
from shared.log_manager import log_trade, log_performance
from shared.paper_trading import PaperTradingEngine
from shared.rate_limiter import ApiType

from src.budget_manager import BudgetManager
from src.risk_manager import DCARiskManager

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")
perf_logger = logging.getLogger("performance")

# ---------------------------------------------------------------------------
# Signal result
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    """Holds individual signal components and the final multiplier."""

    rsi_value: float = 50.0
    rsi_multiplier: float = 1.0
    fg_value: int = 50
    fg_multiplier: float = 1.0
    sma_ratio: float = 1.0        # price / 200-SMA
    sma_multiplier: float = 1.0
    sopr_value: Optional[float] = None
    sopr_multiplier: Optional[float] = None
    final_multiplier: float = 1.0
    active_signals: int = 3
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "rsi_value": round(self.rsi_value, 2),
            "rsi_multiplier": round(self.rsi_multiplier, 4),
            "fg_value": self.fg_value,
            "fg_multiplier": round(self.fg_multiplier, 4),
            "sma_ratio": round(self.sma_ratio, 4),
            "sma_multiplier": round(self.sma_multiplier, 4),
            "final_multiplier": round(self.final_multiplier, 4),
            "active_signals": self.active_signals,
            "timestamp": self.timestamp,
        }
        if self.sopr_value is not None:
            d["sopr_value"] = round(self.sopr_value, 4)
            d["sopr_multiplier"] = round(self.sopr_multiplier, 4) if self.sopr_multiplier else None
        return d


@dataclass
class CrashLevel:
    """Configuration for a single crash-buy level."""

    level: int
    drop_pct: float          # e.g. 10.0 for 10%
    lookback_days: int       # e.g. 7
    amount_multiplier: float # e.g. 1.0x base
    cooldown_days: int       # e.g. 7


@dataclass
class PurchaseRecord:
    """A record of a completed DCA or crash-buy purchase."""

    symbol: str
    timestamp: float
    amount_usdt: float
    quantity: float
    price: float
    multiplier: float
    rsi: float
    fear_greed: int
    sma_ratio: float
    sopr: Optional[float]
    is_crash_buy: bool = False
    crash_level: int = 0
    is_paper: bool = False

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "timestamp_utc": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "amount_usdt": round(self.amount_usdt, 2),
            "quantity": self.quantity,
            "price": self.price,
            "multiplier": round(self.multiplier, 4),
            "rsi": round(self.rsi, 2),
            "fear_greed": self.fear_greed,
            "sma_ratio": round(self.sma_ratio, 4),
            "sopr": round(self.sopr, 4) if self.sopr is not None else None,
            "is_crash_buy": self.is_crash_buy,
            "crash_level": self.crash_level,
            "is_paper": self.is_paper,
        }


# ---------------------------------------------------------------------------
# Main Strategy
# ---------------------------------------------------------------------------

class SignalDCAStrategy:
    """Signal-Enhanced DCA strategy engine.

    Parameters
    ----------
    config : BotConfig
        Full bot configuration.
    binance_client : BinanceClient
        Shared Binance REST client.
    budget_manager : BudgetManager
        Monthly budget tracker.
    risk_manager : DCARiskManager
        DCA risk management layer.
    fear_greed_client : FearGreedClient
        Fear & Greed index client.
    paper_engine : PaperTradingEngine | None
        Paper trading engine (None for live mode).
    glassnode_client : GlassnodeClient | None
        Optional on-chain data client.
    """

    def __init__(
        self,
        config: BotConfig,
        binance_client: BinanceClient,
        budget_manager: BudgetManager,
        risk_manager: DCARiskManager,
        fear_greed_client: FearGreedClient,
        paper_engine: Optional[PaperTradingEngine] = None,
        glassnode_client: Optional[GlassnodeClient] = None,
    ) -> None:
        self._config = config
        self._client = binance_client
        self._budget = budget_manager
        self._risk = risk_manager
        self._fg_client = fear_greed_client
        self._paper = paper_engine
        self._glassnode = glassnode_client
        self._params = config.strategy_params

        self._instruments = config.instruments
        self._is_paper = config.mode == "paper"

        # Indicator buffers (daily candles per instrument)
        self._buffers: Dict[str, IndicatorBuffer] = {
            sym: IndicatorBuffer(max_size=500) for sym in self._instruments
        }

        # Hourly buffers for crash detection
        self._hourly_buffers: Dict[str, IndicatorBuffer] = {
            sym: IndicatorBuffer(max_size=2160) for sym in self._instruments  # 90 days
        }

        # Current signals per instrument
        self._signals: Dict[str, SignalResult] = {}

        # Current Fear & Greed
        self._current_fg: dict = {"value": 50, "stale": True}

        # Current SOPR
        self._current_sopr: Optional[float] = None

        # Fear & Greed degraded mode flag
        self._fg_degraded_mode: bool = False

        # Diversification skip tracking: {symbol: True} when over-concentrated
        self._diversification_skip: Dict[str, bool] = {}

        # Crash-buy cooldowns: {symbol: {level: last_trigger_timestamp}}
        self._crash_cooldowns: Dict[str, Dict[int, float]] = defaultdict(dict)

        # Purchase history
        self._purchase_history: List[PurchaseRecord] = []

        # Vanilla DCA tracking (for comparison metric)
        self._vanilla_invested: Dict[str, float] = defaultdict(float)
        self._vanilla_holdings: Dict[str, float] = defaultdict(float)

        # Value averaging state
        self._va_start_value: Dict[str, float] = {}
        self._va_week_number: int = 0
        self._va_start_ts: float = 0.0

        # Schedule tracking
        self._last_dca_execution: float = 0.0
        self._next_dca_time: float = 0.0

        # Current prices
        self._current_prices: Dict[str, float] = {}

        # Book ticker cache: {symbol: {"bid": float, "ask": float}}
        self._book_tickers: Dict[str, Dict[str, float]] = {}

        # Crash levels
        self._crash_levels = self._build_crash_levels()

    # ------------------------------------------------------------------
    # Signal calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi_multiplier(rsi_val: float) -> float:
        """Map RSI(14) value to signal multiplier component."""
        if rsi_val < 25:
            return 1.5
        elif rsi_val < 35:
            return 1.25
        elif rsi_val < 50:
            return 1.0
        elif rsi_val < 65:
            return 0.85
        elif rsi_val < 75:
            return 0.7
        else:
            return 0.5

    @staticmethod
    def _fg_multiplier(fg_value: int) -> float:
        """Map Fear & Greed Index to signal multiplier component."""
        if fg_value <= 10:
            return 1.75
        elif fg_value <= 25:
            return 1.4
        elif fg_value <= 45:
            return 1.1
        elif fg_value <= 55:
            return 1.0
        elif fg_value <= 75:
            return 0.75
        elif fg_value <= 90:
            return 0.5
        else:
            return 0.25

    @staticmethod
    def _sma_multiplier(price: float, sma_200: float) -> float:
        """Map Price/200-SMA ratio to signal multiplier component."""
        if sma_200 <= 0:
            return 1.0
        ratio = price / sma_200
        if ratio < 0.80:
            return 1.5
        elif ratio < 0.90:
            return 1.25
        elif ratio < 1.0:
            return 1.1
        elif ratio < 1.20:
            return 0.9
        elif ratio < 1.40:
            return 0.75
        else:
            return 0.5

    @staticmethod
    def _sopr_multiplier(sopr: float) -> float:
        """Map SOPR to signal multiplier component."""
        if sopr < 0.95:
            return 1.3
        elif sopr < 1.0:
            return 1.1
        elif sopr < 1.05:
            return 1.0
        else:
            return 0.85

    def calculate_signals(self, symbol: str) -> SignalResult:
        """Calculate the composite signal for an instrument.

        Uses geometric mean of all available signal components,
        clamped between min_multiplier and max_multiplier.
        """
        result = SignalResult(timestamp=time.time())

        buf = self._buffers.get(symbol)
        if buf is None or len(buf) < 15:
            logger.warning("Insufficient data for %s (%d candles), using neutral signal", symbol, len(buf) if buf else 0)
            self._signals[symbol] = result
            return result

        closes = buf.get_closes()

        # Signal A: RSI(14)
        rsi_arr = calc_rsi(closes, self._params.get("rsi_period", 14))
        current_rsi = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0
        result.rsi_value = current_rsi
        result.rsi_multiplier = self._rsi_multiplier(current_rsi)

        # Check if Fear & Greed is in degraded mode
        fg_degraded = getattr(self, "_fg_degraded_mode", False) or self._current_fg.get("stale", True)

        if fg_degraded:
            # Degraded mode: RSI-only signal -- use only RSI component as multiplier
            logger.debug(
                "degraded mode: RSI-only signal for %s (RSI=%.1f, mult=%.4f)",
                symbol, current_rsi, result.rsi_multiplier,
            )
            components = [result.rsi_multiplier]
            n_signals = 1

            # Still record F&G value for display, but do NOT use it in the geometric mean
            fg = self._current_fg.get("value", 50)
            result.fg_value = fg
            result.fg_multiplier = 1.0  # neutral, not used
        else:
            # Signal B: Fear & Greed
            fg = self._current_fg.get("value", 50)
            result.fg_value = fg
            result.fg_multiplier = self._fg_multiplier(fg)
            components = [result.rsi_multiplier, result.fg_multiplier]
            n_signals = 2

        # Signal C: Price vs 200-day SMA
        sma_period = self._params.get("sma_period", 200)
        sma_arr = calc_sma(closes, sma_period)
        current_sma = float(sma_arr[-1]) if len(sma_arr) >= sma_period and not np.isnan(sma_arr[-1]) else 0.0
        current_price = float(closes[-1])
        if current_sma > 0:
            result.sma_ratio = current_price / current_sma
        else:
            result.sma_ratio = 1.0
        result.sma_multiplier = self._sma_multiplier(current_price, current_sma)

        if not fg_degraded:
            # In normal mode, include SMA in geometric mean
            components.append(result.sma_multiplier)
            n_signals = 3
        # In degraded mode, only RSI is used (no SMA either, per spec: "Use only RSI component")

        # Signal D: SOPR (optional, only when Glassnode API key available)
        if self._params.get("sopr_enabled", False) and self._current_sopr is not None and not fg_degraded:
            sopr_mult = self._sopr_multiplier(self._current_sopr)
            result.sopr_value = self._current_sopr
            result.sopr_multiplier = sopr_mult
            components.append(sopr_mult)
            n_signals = 4

        result.active_signals = n_signals

        # Geometric mean
        product = 1.0
        for c in components:
            product *= c
        geo_mean = product ** (1.0 / n_signals)

        # Clamp
        min_mult = self._params.get("min_multiplier", 0.25)
        max_mult = self._params.get("max_multiplier", 3.0)
        result.final_multiplier = max(min_mult, min(max_mult, geo_mean))

        self._signals[symbol] = result
        return result

    # ------------------------------------------------------------------
    # Signal freshness
    # ------------------------------------------------------------------

    def _check_signal_freshness(self, symbol: str) -> Tuple[bool, str]:
        """Check that all signal data is fresh enough for trading."""
        warnings = []

        buf = self._buffers.get(symbol)
        if buf and len(buf) > 0:
            last_ts = buf.get_timestamps()[-1]
            age_hours = (time.time() * 1000 - last_ts) / 3_600_000
            rsi_max = self._params.get("rsi_max_age_hours", 4.0)
            if age_hours > rsi_max:
                warnings.append(f"RSI data stale: {age_hours:.1f}h old (max {rsi_max}h)")

        fg = self._current_fg
        if fg.get("stale", True):
            warnings.append("Fear & Greed data is stale")

        if warnings:
            return False, "; ".join(warnings)
        return True, ""

    # ------------------------------------------------------------------
    # Scheduled DCA execution
    # ------------------------------------------------------------------

    def compute_next_dca_time(self, after: Optional[float] = None) -> float:
        """Compute the next DCA execution timestamp (UTC epoch seconds)."""
        now = datetime.now(timezone.utc)
        if after is not None:
            now = datetime.fromtimestamp(after, tz=timezone.utc)

        schedule = self._params.get("dca_schedule", "weekly")
        hour = self._params.get("dca_hour_utc", 10)
        minute = self._params.get("dca_minute_utc", 0)

        if schedule == "daily":
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target.timestamp()

        elif schedule == "weekly":
            dow = self._params.get("dca_day_of_week", 0)  # 0=Monday
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            days_ahead = dow - now.weekday()
            if days_ahead < 0 or (days_ahead == 0 and target <= now):
                days_ahead += 7
            target += timedelta(days=days_ahead)
            return target.timestamp()

        elif schedule == "bi-weekly":
            dow = self._params.get("dca_day_of_week", 0)
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            days_ahead = dow - now.weekday()
            if days_ahead < 0 or (days_ahead == 0 and target <= now):
                days_ahead += 14
            elif days_ahead > 0:
                pass
            target += timedelta(days=days_ahead)
            return target.timestamp()

        elif schedule == "monthly":
            dom = self._params.get("dca_day_of_month", 1)
            target = now.replace(day=dom, hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                # Move to next month
                if now.month == 12:
                    target = target.replace(year=now.year + 1, month=1)
                else:
                    target = target.replace(month=now.month + 1)
            return target.timestamp()

        # Default weekly
        return self.compute_next_dca_time(after)

    def get_intervals_per_month(self) -> int:
        """Return approximate DCA intervals per month based on schedule."""
        schedule = self._params.get("dca_schedule", "weekly")
        if schedule == "daily":
            return 30
        elif schedule == "weekly":
            return 4
        elif schedule == "bi-weekly":
            return 2
        elif schedule == "monthly":
            return 1
        return 4

    def _is_double_execution(self) -> bool:
        """Check if executing now would be a double-execution."""
        if self._last_dca_execution <= 0:
            return False
        window = self._params.get("double_execution_window_hours", 6.0) * 3600
        return (time.time() - self._last_dca_execution) < window

    def _is_too_late(self) -> bool:
        """Check if we're more than 1 hour past scheduled time."""
        if self._next_dca_time <= 0:
            return False
        max_delay = self._params.get("max_schedule_delay_minutes", 60) * 60
        return (time.time() - self._next_dca_time) > max_delay

    async def execute_scheduled_dca(self) -> List[PurchaseRecord]:
        """Execute the scheduled DCA purchase for all instruments.

        Returns list of completed purchases.
        """
        purchases = []

        # Schedule checks
        if self._is_double_execution():
            logger.info("Skipping DCA — would be double execution (last: %.0f)", self._last_dca_execution)
            return purchases

        if self._is_too_late():
            logger.warning("Skipping DCA — more than 1 hour past scheduled time")
            self._next_dca_time = self.compute_next_dca_time()
            return purchases

        mode = self._params.get("mode", "signal_enhanced")

        for symbol in self._instruments:
            try:
                if mode == "value_averaging":
                    purchase = await self._execute_value_averaging(symbol)
                else:
                    purchase = await self._execute_signal_dca(symbol)

                if purchase is not None:
                    purchases.append(purchase)
            except Exception as exc:
                logger.exception("DCA execution failed for %s: %s", symbol, exc)
                self._risk.record_order_failure()

        if purchases:
            self._last_dca_execution = time.time()
            # Enforce diversification after purchases for next cycle
            self._check_diversification_enforcement()

        self._next_dca_time = self.compute_next_dca_time()
        return purchases

    def _check_diversification_enforcement(self) -> None:
        """After each purchase, check if any single asset exceeds 60% of DCA
        portfolio value. If so, mark it for skip on next purchase and redirect
        to under-allocated assets.
        """
        max_pct = self._params.get("max_single_asset_pct", 60.0) / 100.0
        holdings = self._risk.get_holdings()
        portfolio_value = self._risk.get_portfolio_value()

        if portfolio_value <= 0:
            self._diversification_skip.clear()
            return

        old_skip = dict(self._diversification_skip)
        self._diversification_skip.clear()

        for sym, h in holdings.items():
            pct = h.current_value / portfolio_value
            if pct > max_pct:
                self._diversification_skip[sym] = True
                if sym not in old_skip:
                    logger.warning(
                        "Diversification enforcement: %s at %.1f%% of portfolio (max %.0f%%), "
                        "will skip on next purchase and redirect to under-allocated assets",
                        sym, pct * 100, max_pct * 100,
                    )

    async def _execute_signal_dca(self, symbol: str) -> Optional[PurchaseRecord]:
        """Execute a signal-enhanced DCA purchase for a single instrument."""
        # Diversification enforcement: skip if over-concentrated
        if self._diversification_skip.get(symbol, False):
            logger.info(
                "Diversification enforcement: skipping %s (over 60%% of portfolio), "
                "redirecting to under-allocated assets",
                symbol,
            )
            return None

        # Calculate signals
        signal = self.calculate_signals(symbol)

        # Check freshness
        fresh, warning = self._check_signal_freshness(symbol)
        if not fresh:
            logger.warning("Signal freshness issue for %s: %s — using last known values", symbol, warning)

        # Calculate amount
        base_amount = self._budget.get_base_amount(symbol)
        if base_amount <= 0:
            return None

        purchase_amount = base_amount * signal.final_multiplier

        # Budget check
        approved_amount, budget_reason = self._budget.check_dca_purchase(symbol, purchase_amount)
        if approved_amount <= 0:
            logger.info("DCA purchase skipped for %s: %s", symbol, budget_reason)
            return None
        if budget_reason:
            logger.info("DCA budget adjustment for %s: %s", symbol, budget_reason)

        # Risk check
        allowed, risk_reason = self._risk.check_purchase_allowed(symbol, approved_amount)
        if not allowed:
            logger.info("DCA purchase blocked for %s: %s", symbol, risk_reason)
            return None

        # Exchange health check
        ticker = self._book_tickers.get(symbol, {})
        bid = ticker.get("bid", 0)
        ask = ticker.get("ask", 0)
        if bid > 0 and ask > 0:
            spread_ok, spread_pct = self._risk.check_spread(bid, ask)
            if not spread_ok:
                logger.warning("Spread too wide for %s (%.4f%%), delaying", symbol, spread_pct)
                return None

        # Execute
        return await self._place_buy_order(
            symbol=symbol,
            amount_usdt=approved_amount,
            signal=signal,
            is_crash_buy=False,
            crash_level=0,
        )

    async def _execute_value_averaging(self, symbol: str) -> Optional[PurchaseRecord]:
        """Execute value averaging purchase for a single instrument."""
        weekly_target = self._params.get("va_weekly_target", 100.0)
        max_purchase = weekly_target * self._params.get("va_max_purchase_multiplier", 3.0)

        # Initialize VA tracking
        if symbol not in self._va_start_value:
            holding = self._risk.get_holdings().get(symbol)
            self._va_start_value[symbol] = holding.current_value if holding else 0.0
            if self._va_start_ts == 0:
                self._va_start_ts = time.time()

        # Calculate target value
        if self._va_start_ts > 0:
            weeks_elapsed = (time.time() - self._va_start_ts) / (7 * 86400)
            self._va_week_number = int(weeks_elapsed) + 1

        target_value = self._va_start_value.get(symbol, 0.0) + (self._va_week_number * weekly_target)

        # Current value
        holding = self._risk.get_holdings().get(symbol)
        current_value = holding.current_value if holding else 0.0

        required = target_value - current_value
        if required <= 0:
            logger.info("VA: %s already above target ($%.2f >= $%.2f), skipping", symbol, current_value, target_value)
            return None

        purchase_amount = min(required, max_purchase)
        purchase_amount = max(purchase_amount, 0.0)

        if purchase_amount < self._budget.get_min_purchase():
            return None

        # Budget check
        approved_amount, budget_reason = self._budget.check_dca_purchase(symbol, purchase_amount)
        if approved_amount <= 0:
            logger.info("VA purchase skipped for %s: %s", symbol, budget_reason)
            return None

        # Risk check
        allowed, risk_reason = self._risk.check_purchase_allowed(symbol, approved_amount)
        if not allowed:
            return None

        signal = self.calculate_signals(symbol)

        return await self._place_buy_order(
            symbol=symbol,
            amount_usdt=approved_amount,
            signal=signal,
            is_crash_buy=False,
            crash_level=0,
        )

    # ------------------------------------------------------------------
    # Crash-buy detection and execution
    # ------------------------------------------------------------------

    def _build_crash_levels(self) -> List[CrashLevel]:
        """Build crash level configurations from params."""
        return [
            CrashLevel(
                level=1,
                drop_pct=self._params.get("crash_level_1_drop_pct", 10.0),
                lookback_days=self._params.get("crash_level_1_lookback_days", 7),
                amount_multiplier=self._params.get("crash_level_1_amount_multiplier", 1.0),
                cooldown_days=self._params.get("crash_level_1_cooldown_days", 7),
            ),
            CrashLevel(
                level=2,
                drop_pct=self._params.get("crash_level_2_drop_pct", 20.0),
                lookback_days=self._params.get("crash_level_2_lookback_days", 30),
                amount_multiplier=self._params.get("crash_level_2_amount_multiplier", 2.0),
                cooldown_days=self._params.get("crash_level_2_cooldown_days", 14),
            ),
            CrashLevel(
                level=3,
                drop_pct=self._params.get("crash_level_3_drop_pct", 30.0),
                lookback_days=self._params.get("crash_level_3_lookback_days", 90),
                amount_multiplier=self._params.get("crash_level_3_amount_multiplier", 3.0),
                cooldown_days=self._params.get("crash_level_3_cooldown_days", 30),
            ),
        ]

    def _check_crash_cooldown(self, symbol: str, level: int, cooldown_days: int) -> bool:
        """Return True if crash-buy is still in cooldown."""
        last = self._crash_cooldowns.get(symbol, {}).get(level, 0.0)
        if last <= 0:
            return False
        return (time.time() - last) < (cooldown_days * 86400)

    def detect_crash_buys(self) -> Dict[str, int]:
        """Detect crash-buy triggers for all instruments.

        Returns {symbol: highest_triggered_level} for instruments that trigger.
        Only the highest level triggers if multiple fire simultaneously.
        """
        if not self._params.get("crash_buy_enabled", True):
            return {}

        triggered = {}

        for symbol in self._instruments:
            buf = self._hourly_buffers.get(symbol)
            if buf is None or len(buf) < 24:
                continue

            current_price = self._current_prices.get(symbol, 0.0)
            if current_price <= 0:
                continue

            highs = buf.get_highs()
            timestamps = buf.get_timestamps()
            now_ms = int(time.time() * 1000)

            highest_level = 0

            for cl in self._crash_levels:
                # Calculate lookback window high
                lookback_ms = cl.lookback_days * 24 * 3600 * 1000
                mask = (now_ms - timestamps) <= lookback_ms
                if not np.any(mask):
                    continue

                period_high = float(np.max(highs[mask]))
                if period_high <= 0:
                    continue

                drop_pct = ((period_high - current_price) / period_high) * 100.0

                if drop_pct >= cl.drop_pct:
                    # Check cooldown
                    if not self._check_crash_cooldown(symbol, cl.level, cl.cooldown_days):
                        highest_level = cl.level

            if highest_level > 0:
                triggered[symbol] = highest_level

        return triggered

    async def execute_crash_buys(self, triggers: Dict[str, int]) -> List[PurchaseRecord]:
        """Execute crash-buy orders for triggered instruments."""
        purchases = []

        for symbol, level in triggers.items():
            try:
                cl = self._crash_levels[level - 1]
                base_amount = self._budget.get_base_amount(symbol)
                amount = base_amount * cl.amount_multiplier * self.get_intervals_per_month()
                # Crash-buy amount is level multiplier * weekly base amount
                amount = base_amount * cl.amount_multiplier

                # Budget check (crash reserve)
                approved, reason = self._budget.check_crash_purchase(symbol, amount)
                if approved <= 0:
                    logger.info("Crash-buy L%d skipped for %s: %s", level, symbol, reason)
                    continue

                # Risk check
                allowed, risk_reason = self._risk.check_purchase_allowed(symbol, approved)
                if not allowed:
                    logger.info("Crash-buy L%d blocked for %s: %s", level, symbol, risk_reason)
                    continue

                signal = self.calculate_signals(symbol)

                purchase = await self._place_buy_order(
                    symbol=symbol,
                    amount_usdt=approved,
                    signal=signal,
                    is_crash_buy=True,
                    crash_level=level,
                )

                if purchase is not None:
                    purchases.append(purchase)
                    # Set cooldown
                    self._crash_cooldowns[symbol][level] = time.time()
                    logger.info(
                        "Crash-buy L%d executed for %s: $%.2f",
                        level, symbol, approved,
                    )

            except Exception as exc:
                logger.exception("Crash-buy L%d failed for %s: %s", level, symbol, exc)
                self._risk.record_order_failure()

        return purchases

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    async def _place_buy_order(
        self,
        symbol: str,
        amount_usdt: float,
        signal: SignalResult,
        is_crash_buy: bool,
        crash_level: int,
    ) -> Optional[PurchaseRecord]:
        """Place a MARKET BUY order on spot (or simulate in paper mode).

        Returns a PurchaseRecord on success, or None on failure.
        """
        current_price = self._current_prices.get(symbol, 0.0)
        if current_price <= 0:
            logger.error("No price available for %s, cannot execute purchase", symbol)
            return None

        quantity = amount_usdt / current_price

        if self._is_paper:
            return self._simulate_paper_buy(
                symbol, amount_usdt, quantity, current_price, signal,
                is_crash_buy, crash_level,
            )

        # Live execution
        try:
            # Use quoteOrderQty for MARKET buy (spend exact USDT amount)
            result = await self._client.place_spot_order(
                symbol=symbol,
                side="BUY",
                type="MARKET",
                quantity=quantity,
            )

            # Parse fill
            fills = result.get("fills", [])
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                avg_price = total_cost / total_qty if total_qty > 0 else current_price
            else:
                total_qty = float(result.get("executedQty", quantity))
                avg_price = float(result.get("price", current_price))
                total_cost = total_qty * avg_price

            # Record
            self._risk.record_purchase(symbol, total_qty, total_cost, avg_price)
            self._risk.record_order_success()

            if is_crash_buy:
                self._budget.record_crash_purchase(symbol, total_cost)
            else:
                self._budget.record_dca_purchase(symbol, total_cost)

            # Track vanilla DCA for comparison
            base = self._budget.get_base_amount(symbol)
            self._vanilla_invested[symbol] += base
            self._vanilla_holdings[symbol] += base / avg_price

            record = PurchaseRecord(
                symbol=symbol,
                timestamp=time.time(),
                amount_usdt=total_cost,
                quantity=total_qty,
                price=avg_price,
                multiplier=signal.final_multiplier,
                rsi=signal.rsi_value,
                fear_greed=signal.fg_value,
                sma_ratio=signal.sma_ratio,
                sopr=signal.sopr_value,
                is_crash_buy=is_crash_buy,
                crash_level=crash_level,
                is_paper=False,
            )
            self._purchase_history.append(record)

            log_trade(
                action="CRASH_BUY" if is_crash_buy else "DCA_BUY",
                symbol=symbol,
                side="BUY",
                quantity=total_qty,
                price=avg_price,
                amount_usdt=total_cost,
                multiplier=signal.final_multiplier,
                crash_level=crash_level,
            )

            return record

        except BinanceClientError as exc:
            logger.error("Spot order failed for %s: %s", symbol, exc)
            self._risk.record_order_failure()
            return None

    def _simulate_paper_buy(
        self,
        symbol: str,
        amount_usdt: float,
        quantity: float,
        price: float,
        signal: SignalResult,
        is_crash_buy: bool,
        crash_level: int,
    ) -> PurchaseRecord:
        """Simulate a paper trading purchase."""
        # Apply slippage: fill at ask + 0.01%
        ticker = self._book_tickers.get(symbol, {})
        ask = ticker.get("ask", price)
        slippage = 0.0001  # 0.01%
        fill_price = ask * (1 + slippage) if ask > 0 else price * (1 + slippage)

        # Apply spot taker fee (0.10%)
        fee_rate = 0.001
        effective_amount = amount_usdt * (1 - fee_rate)
        fill_qty = effective_amount / fill_price

        # Record
        self._risk.record_purchase(symbol, fill_qty, amount_usdt, fill_price)

        if is_crash_buy:
            self._budget.record_crash_purchase(symbol, amount_usdt)
        else:
            self._budget.record_dca_purchase(symbol, amount_usdt)

        # Track vanilla DCA for comparison
        base = self._budget.get_base_amount(symbol)
        self._vanilla_invested[symbol] += base
        self._vanilla_holdings[symbol] += base / fill_price

        record = PurchaseRecord(
            symbol=symbol,
            timestamp=time.time(),
            amount_usdt=amount_usdt,
            quantity=fill_qty,
            price=fill_price,
            multiplier=signal.final_multiplier,
            rsi=signal.rsi_value,
            fear_greed=signal.fg_value,
            sma_ratio=signal.sma_ratio,
            sopr=signal.sopr_value,
            is_crash_buy=is_crash_buy,
            crash_level=crash_level,
            is_paper=True,
        )
        self._purchase_history.append(record)

        log_trade(
            action="PAPER_CRASH_BUY" if is_crash_buy else "PAPER_DCA_BUY",
            symbol=symbol,
            side="BUY",
            quantity=fill_qty,
            price=fill_price,
            amount_usdt=amount_usdt,
            multiplier=signal.final_multiplier,
            crash_level=crash_level,
        )

        return record

    # ------------------------------------------------------------------
    # Take-profit rebalancing
    # ------------------------------------------------------------------

    async def execute_take_profit(self) -> Optional[Dict[str, Any]]:
        """Check and execute take-profit rebalancing if triggered."""
        rebal = self._risk.check_take_profit()
        if rebal is None:
            return None

        results = {}
        for symbol, sell_info in rebal.get("sells", {}).items():
            qty = sell_info["quantity"]
            if qty <= 0:
                continue

            if self._is_paper:
                price = self._current_prices.get(symbol, 0)
                if price > 0:
                    amount = qty * price
                    self._risk.record_sale(symbol, qty, amount, price)
                    results[symbol] = {"quantity": qty, "amount": round(amount, 2), "paper": True}
            else:
                try:
                    result = await self._client.place_spot_order(
                        symbol=symbol,
                        side="SELL",
                        type="MARKET",
                        quantity=qty,
                    )
                    fills = result.get("fills", [])
                    total_qty = sum(float(f["qty"]) for f in fills)
                    total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                    avg_price = total_cost / total_qty if total_qty > 0 else 0

                    self._risk.record_sale(symbol, total_qty, total_cost, avg_price)
                    results[symbol] = {"quantity": total_qty, "amount": round(total_cost, 2)}

                    log_trade(
                        action="TAKE_PROFIT_SELL",
                        symbol=symbol,
                        side="SELL",
                        quantity=total_qty,
                        price=avg_price,
                    )
                except Exception as exc:
                    logger.error("Take-profit sell failed for %s: %s", symbol, exc)

        if results:
            self._risk.mark_rebalance_done()

        return results if results else None

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update_daily_candle(self, symbol: str, candle: dict) -> None:
        """Ingest a daily candle into the indicator buffer."""
        buf = self._buffers.get(symbol)
        if buf:
            buf.add_candle(candle)
            self._current_prices[symbol] = float(candle.get("close", 0))

    def update_hourly_candle(self, symbol: str, candle: dict) -> None:
        """Ingest an hourly candle into the crash-detection buffer."""
        buf = self._hourly_buffers.get(symbol)
        if buf:
            buf.add_candle(candle)

    def update_price(self, symbol: str, price: float) -> None:
        """Update latest price for an instrument."""
        self._current_prices[symbol] = price
        self._risk.update_prices({symbol: price})

    def update_book_ticker(self, symbol: str, bid: float, ask: float) -> None:
        """Update bid/ask from bookTicker stream."""
        self._book_tickers[symbol] = {"bid": bid, "ask": ask}
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            self._current_prices[symbol] = mid

    async def update_fear_greed(self) -> dict:
        """Fetch and cache the latest Fear & Greed index.

        If the API is unavailable or returns stale data (not today's),
        fall back to RSI-only signal with warning (degraded mode).
        """
        try:
            fg_data = await self._fg_client.get_current()
        except Exception as exc:
            logger.warning("Fear & Greed API error: %s", exc)
            fg_data = {"value": 50, "stale": True, "timestamp": 0}

        # Check staleness: timestamp must be from today (UTC)
        fg_ts = fg_data.get("timestamp", 0)
        is_stale = fg_data.get("stale", False)

        if fg_ts > 0 and not is_stale:
            from datetime import date
            fg_date = datetime.fromtimestamp(fg_ts, tz=timezone.utc).date()
            today = datetime.now(timezone.utc).date()
            if fg_date != today:
                is_stale = True
                logger.warning(
                    "Fear & Greed data is from %s, not today (%s) -- marking stale",
                    fg_date.isoformat(), today.isoformat(),
                )

        if is_stale or fg_ts == 0:
            fg_data["stale"] = True
            self._fg_degraded_mode = True
            logger.warning(
                "degraded mode: RSI-only signal -- Fear & Greed API unavailable or stale"
            )
        else:
            self._fg_degraded_mode = False

        self._current_fg = fg_data
        return self._current_fg

    async def update_sopr(self) -> Optional[float]:
        """Fetch and cache the latest SOPR value (if Glassnode configured)."""
        if self._glassnode is None:
            return None
        try:
            data = await self._glassnode.get_sopr("BTC")
            if data and isinstance(data, list) and len(data) > 0:
                latest = data[-1]
                if isinstance(latest, dict):
                    self._current_sopr = float(latest.get("v", 1.0))
                elif isinstance(latest, (int, float)):
                    self._current_sopr = float(latest)
                return self._current_sopr
        except Exception as exc:
            logger.warning("SOPR fetch failed: %s", exc)
        return None

    async def warm_up_data(self) -> None:
        """Fetch historical daily candles (200 for SMA) and Fear & Greed on startup."""
        logger.info("Warming up historical data for %d instruments...", len(self._instruments))

        for symbol in self._instruments:
            try:
                # 200 daily candles for 200-SMA
                klines = await self._client.get_spot_klines(
                    symbol=symbol, interval="1d", limit=250,
                )
                for k in klines:
                    self._buffers[symbol].add_candle({
                        "timestamp": k[0],
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    })
                if klines:
                    self._current_prices[symbol] = float(klines[-1][4])

                logger.info("Loaded %d daily candles for %s", len(klines), symbol)

                # Hourly candles for crash detection (at least 7 days = 168 candles)
                hourly_klines = await self._client.get_spot_klines(
                    symbol=symbol, interval="1h", limit=500,
                )
                for k in hourly_klines:
                    self._hourly_buffers[symbol].add_candle({
                        "timestamp": k[0],
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    })
                logger.info("Loaded %d hourly candles for %s", len(hourly_klines), symbol)

            except Exception as exc:
                logger.error("Failed to warm up data for %s: %s", symbol, exc)

        # Fear & Greed
        try:
            await self.update_fear_greed()
            logger.info("Fear & Greed loaded: %s", self._current_fg.get("value"))
        except Exception as exc:
            logger.warning("Failed to fetch Fear & Greed: %s", exc)

        # SOPR
        if self._params.get("sopr_enabled", False):
            await self.update_sopr()

        # Calculate initial signals
        for symbol in self._instruments:
            self.calculate_signals(symbol)

    # ------------------------------------------------------------------
    # Performance metrics (Section 10.2)
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Compute all Section 10.1 + 10.2 performance metrics."""
        holdings = self._risk.get_holdings()
        total_invested = self._risk.get_total_invested()
        portfolio_value = self._risk.get_portfolio_value()
        total_return = 0.0
        if total_invested > 0:
            total_return = ((portfolio_value - total_invested) / total_invested) * 100.0

        # Per-instrument cost basis
        cost_basis = {}
        for sym, h in holdings.items():
            cost_basis[sym] = {
                "avg_cost": round(h.avg_cost_basis, 4),
                "current_price": round(h.current_price, 4),
                "vs_cost": round(((h.current_price / h.avg_cost_basis - 1) * 100), 2)
                if h.avg_cost_basis > 0 else 0,
            }

        # Signal multiplier distribution
        multipliers = [p.multiplier for p in self._purchase_history]
        mult_dist = {}
        if multipliers:
            mult_arr = np.array(multipliers)
            mult_dist = {
                "mean": round(float(np.mean(mult_arr)), 4),
                "median": round(float(np.median(mult_arr)), 4),
                "min": round(float(np.min(mult_arr)), 4),
                "max": round(float(np.max(mult_arr)), 4),
                "std": round(float(np.std(mult_arr)), 4),
                "count": len(multipliers),
            }

        # Crash-buy performance
        crash_purchases = [p for p in self._purchase_history if p.is_crash_buy]
        crash_perf = {"count": len(crash_purchases), "total_invested": 0.0, "current_value": 0.0}
        for cp in crash_purchases:
            crash_perf["total_invested"] += cp.amount_usdt
            if cp.symbol in holdings and holdings[cp.symbol].current_price > 0:
                crash_perf["current_value"] += cp.quantity * holdings[cp.symbol].current_price

        if crash_perf["total_invested"] > 0:
            crash_perf["return_pct"] = round(
                ((crash_perf["current_value"] - crash_perf["total_invested"]) / crash_perf["total_invested"]) * 100, 2
            )
        else:
            crash_perf["return_pct"] = 0.0

        # Signal-enhanced vs vanilla DCA comparison
        vanilla_value = 0.0
        vanilla_invested_total = 0.0
        for sym in self._instruments:
            vanilla_invested_total += self._vanilla_invested.get(sym, 0.0)
            qty = self._vanilla_holdings.get(sym, 0.0)
            price = self._current_prices.get(sym, 0.0)
            vanilla_value += qty * price

        vanilla_return = 0.0
        if vanilla_invested_total > 0:
            vanilla_return = ((vanilla_value - vanilla_invested_total) / vanilla_invested_total) * 100.0

        cost_basis_improvement = 0.0
        if vanilla_return != 0:
            cost_basis_improvement = total_return - vanilla_return

        # Monthly spend variance
        budget_state = self._budget.to_state()
        history = budget_state.get("history", [])
        monthly_spends = [h.get("total_spent", 0) for h in history]
        spend_variance = round(float(np.std(monthly_spends)), 2) if len(monthly_spends) > 1 else 0.0

        # Accumulation rate: units per dollar over time
        total_units = sum(h.quantity for h in holdings.values())
        accum_rate = total_units / total_invested if total_invested > 0 else 0.0

        # Asset allocation
        allocation = {}
        if portfolio_value > 0:
            for sym, h in holdings.items():
                allocation[sym] = round((h.current_value / portfolio_value) * 100, 2)

        return {
            # 10.1 Standard DCA metrics
            "total_invested": round(total_invested, 2),
            "portfolio_value": round(portfolio_value, 2),
            "total_return_pct": round(total_return, 2),
            "cost_basis": cost_basis,
            "total_purchases": len(self._purchase_history),

            # 10.2 Strategy-specific metrics
            "multiplier_distribution": mult_dist,
            "crash_buy_performance": crash_perf,
            "vanilla_dca_return_pct": round(vanilla_return, 2),
            "signal_vs_vanilla_improvement": round(cost_basis_improvement, 2),
            "monthly_spend_variance": spend_variance,
            "accumulation_rate": round(accum_rate, 8),

            # 10.3 Portfolio analytics
            "asset_allocation": allocation,
            "holdings": {s: h.to_dict() for s, h in holdings.items()},

            # Signal info
            "current_signals": {s: sig.to_dict() for s, sig in self._signals.items()},
            "fear_greed": self._current_fg,
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def to_state(self) -> dict:
        """Serialize full strategy state for persistence."""
        return {
            "signals": {s: sig.to_dict() for s, sig in self._signals.items()},
            "crash_cooldowns": {
                sym: {str(lvl): ts for lvl, ts in levels.items()}
                for sym, levels in self._crash_cooldowns.items()
            },
            "purchase_history": [p.to_dict() for p in self._purchase_history[-1000:]],
            "vanilla_invested": dict(self._vanilla_invested),
            "vanilla_holdings": dict(self._vanilla_holdings),
            "va_start_value": dict(self._va_start_value),
            "va_week_number": self._va_week_number,
            "va_start_ts": self._va_start_ts,
            "last_dca_execution": self._last_dca_execution,
            "next_dca_time": self._next_dca_time,
            "current_prices": dict(self._current_prices),
            "current_fg": self._current_fg,
            "current_sopr": self._current_sopr,
            "fg_degraded_mode": self._fg_degraded_mode,
            "diversification_skip": dict(self._diversification_skip),
        }

    def load_state(self, state: dict) -> None:
        """Restore strategy state from persisted data."""
        # Crash cooldowns
        for sym, levels in state.get("crash_cooldowns", {}).items():
            for lvl_str, ts in levels.items():
                self._crash_cooldowns[sym][int(lvl_str)] = ts

        # Purchase history
        for pd in state.get("purchase_history", []):
            self._purchase_history.append(PurchaseRecord(
                symbol=pd["symbol"],
                timestamp=pd["timestamp"],
                amount_usdt=pd["amount_usdt"],
                quantity=pd["quantity"],
                price=pd["price"],
                multiplier=pd["multiplier"],
                rsi=pd["rsi"],
                fear_greed=pd["fear_greed"],
                sma_ratio=pd["sma_ratio"],
                sopr=pd.get("sopr"),
                is_crash_buy=pd.get("is_crash_buy", False),
                crash_level=pd.get("crash_level", 0),
                is_paper=pd.get("is_paper", False),
            ))

        self._vanilla_invested = defaultdict(float, state.get("vanilla_invested", {}))
        self._vanilla_holdings = defaultdict(float, state.get("vanilla_holdings", {}))
        self._va_start_value = state.get("va_start_value", {})
        self._va_week_number = state.get("va_week_number", 0)
        self._va_start_ts = state.get("va_start_ts", 0.0)
        self._last_dca_execution = state.get("last_dca_execution", 0.0)
        self._next_dca_time = state.get("next_dca_time", 0.0)
        self._current_prices = state.get("current_prices", {})
        self._current_fg = state.get("current_fg", {"value": 50, "stale": True})
        self._current_sopr = state.get("current_sopr")
        self._fg_degraded_mode = state.get("fg_degraded_mode", False)
        self._diversification_skip = state.get("diversification_skip", {})

        logger.info(
            "Strategy state restored: %d purchases, last_dca=%.0f, next_dca=%.0f",
            len(self._purchase_history), self._last_dca_execution, self._next_dca_time,
        )

    # ------------------------------------------------------------------
    # Dashboard data
    # ------------------------------------------------------------------

    def get_next_dca_countdown(self) -> dict:
        """Return countdown info for the next DCA execution."""
        if self._next_dca_time <= 0:
            self._next_dca_time = self.compute_next_dca_time()

        remaining = max(0, self._next_dca_time - time.time())
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        seconds = int(remaining % 60)

        return {
            "next_dca_utc": datetime.fromtimestamp(self._next_dca_time, tz=timezone.utc).isoformat(),
            "remaining_seconds": int(remaining),
            "countdown": f"{hours}h {minutes}m {seconds}s",
        }

    def get_crash_cooldown_status(self) -> dict:
        """Return crash-buy cooldown status for all instruments."""
        status = {}
        now = time.time()
        for symbol in self._instruments:
            symbol_cooldowns = {}
            for cl in self._crash_levels:
                last = self._crash_cooldowns.get(symbol, {}).get(cl.level, 0.0)
                if last <= 0:
                    symbol_cooldowns[f"level_{cl.level}"] = {"active": False, "ready": True}
                else:
                    elapsed = now - last
                    cooldown_sec = cl.cooldown_days * 86400
                    remaining = max(0, cooldown_sec - elapsed)
                    symbol_cooldowns[f"level_{cl.level}"] = {
                        "active": remaining > 0,
                        "ready": remaining <= 0,
                        "remaining_hours": round(remaining / 3600, 1),
                        "last_triggered": datetime.fromtimestamp(last, tz=timezone.utc).isoformat(),
                    }
            status[symbol] = symbol_cooldowns
        return status

    def get_recent_purchases(self, limit: int = 20) -> List[dict]:
        """Return the most recent purchases as dicts."""
        return [p.to_dict() for p in self._purchase_history[-limit:]]
