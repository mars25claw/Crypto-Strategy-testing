"""Avellaneda-Stoikov market making model implementation.

Provides optimal reservation price and spread calculation based on:
- Realized volatility (sigma) from 1m candle log-returns
- Order book depth parameter (kappa)
- Risk aversion (gamma) adaptive to inventory
- Session time remaining (T - t) in 8-hour funding windows

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a
limit order book".
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelParameters:
    """Current calibrated model parameters for one instrument."""
    sigma: float = 0.0           # Hourly realized volatility
    sigma_annual: float = 0.0    # Annualized volatility
    kappa: float = 1.0           # Order book depth parameter
    gamma: float = 0.1           # Risk aversion
    time_remaining: float = 1.0  # T - t in fractional hours
    mid_price: float = 0.0
    reservation_price: float = 0.0
    optimal_spread: float = 0.0
    bid_quote: float = 0.0
    ask_quote: float = 0.0
    last_calibration: float = 0.0
    market_spread: float = 0.0   # Current market best_ask - best_bid
    market_spread_pct: float = 0.0


@dataclass
class QuotePrices:
    """Calculated quote prices for an instrument at multiple layers."""
    bid_l1: float = 0.0
    ask_l1: float = 0.0
    bid_l2: float = 0.0
    ask_l2: float = 0.0
    bid_l3: float = 0.0
    ask_l3: float = 0.0
    reservation_price: float = 0.0
    half_spread: float = 0.0
    spread_pct: float = 0.0
    timestamp: float = 0.0


@dataclass
class InstrumentData:
    """All real-time data for a single instrument."""
    symbol: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid_price: float = 0.0
    mark_price: float = 0.0
    funding_rate: float = 0.0
    tick_size: float = 0.0001
    step_size: float = 0.01
    min_notional: float = 5.0

    # 1m candle close prices (rolling 60 minutes)
    candle_closes: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    candle_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=120))

    # 5m candle data for short-term trend
    candle_5m_closes: Deque[float] = field(default_factory=lambda: deque(maxlen=48))

    # 1h candle data for EMA trend filter
    candle_1h_closes: Deque[float] = field(default_factory=lambda: deque(maxlen=48))

    # Order book depth tracking (top 5 levels USDT value)
    depth_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1800))
    depth_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=1800))
    current_depth_usdt: float = 0.0

    # Trade tracking for average trade size
    trade_sizes: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    # Model output
    params: ModelParameters = field(default_factory=ModelParameters)
    quotes: QuotePrices = field(default_factory=QuotePrices)

    # Inventory
    inventory_qty: float = 0.0       # Positive = long, negative = short
    inventory_avg_cost: float = 0.0
    inventory_notional: float = 0.0  # abs(qty * mid_price)

    # State flags
    is_active: bool = True
    halt_reason: str = ""
    withdraw_until: float = 0.0      # Timestamp when to resume quoting

    # Volume filter
    volume_24h: float = 0.0

    # Liquidation tracking
    liquidation_events: Deque[float] = field(default_factory=lambda: deque(maxlen=600))


# ---------------------------------------------------------------------------
# Session time calculation
# ---------------------------------------------------------------------------

SESSION_RESETS_HOURS = [0, 8, 16]  # UTC hours for 8h sessions


def get_session_time_remaining() -> float:
    """Return time remaining until next 8-hour session boundary in fractional hours.

    Sessions: 00:00-08:00, 08:00-16:00, 16:00-00:00 UTC.
    Returns a value between 0.0 and 8.0.
    """
    now = datetime.now(timezone.utc)
    current_hour = now.hour + now.minute / 60.0 + now.second / 3600.0

    # Find the next session boundary
    for boundary in SESSION_RESETS_HOURS:
        if boundary > current_hour:
            return boundary - current_hour

    # Wrap around to next day's 00:00
    return 24.0 - current_hour

    return max(remaining, 0.01)  # Never exactly zero


def get_next_session_reset() -> datetime:
    """Return the datetime of the next 8-hour session reset."""
    now = datetime.now(timezone.utc)
    current_hour = now.hour

    for h in SESSION_RESETS_HOURS:
        if h > current_hour:
            return now.replace(hour=h, minute=0, second=0, microsecond=0)

    # Next day 00:00
    from datetime import timedelta
    tomorrow = now + timedelta(days=1)
    return tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov Strategy
# ---------------------------------------------------------------------------

class AvellanedaStoikovStrategy:
    """Avellaneda-Stoikov optimal market making model.

    Parameters
    ----------
    params : dict
        Strategy parameters from config.yaml (strategy_params section).
    exchange_info : object
        ExchangeInfo instance for tick/step size lookups.
    """

    def __init__(self, params: dict, exchange_info: Any = None) -> None:
        self._params = params
        self._exchange_info = exchange_info

        # Model parameters from config
        self._base_gamma: float = params.get("base_gamma", 0.1)
        self._gamma_relaxed: float = params.get("gamma_relaxed", 0.05)
        self._gamma_aggressive: float = params.get("gamma_aggressive", 0.3)
        self._gamma_very_aggressive: float = params.get("gamma_very_aggressive", 0.5)
        self._vol_window: int = params.get("vol_window_minutes", 60)
        self._depth_window: int = params.get("depth_window_minutes", 30)
        self._min_spread_pct: float = params.get("min_spread_pct", 0.045) / 100.0
        self._max_spread_mult: float = params.get("max_spread_multiplier", 5.0)
        self._quote_update_threshold: float = params.get("quote_update_threshold", 0.0001)
        self._layer2_mult: float = params.get("layer2_multiplier", 1.0)
        self._layer3_mult: float = params.get("layer3_multiplier", 1.5)

        # Per-instrument data
        self._instruments: Dict[str, InstrumentData] = {}

        logger.info(
            "AvellanedaStoikov initialized: base_gamma=%.3f, vol_window=%dm, "
            "min_spread=%.4f%%, max_spread_mult=%.1f",
            self._base_gamma, self._vol_window,
            self._min_spread_pct * 100, self._max_spread_mult,
        )

    # ------------------------------------------------------------------
    # Instrument management
    # ------------------------------------------------------------------

    def add_instrument(self, symbol: str, tick_size: float = 0.0001,
                       step_size: float = 0.01, min_notional: float = 5.0) -> None:
        """Register an instrument for quoting."""
        self._instruments[symbol] = InstrumentData(
            symbol=symbol,
            tick_size=tick_size,
            step_size=step_size,
            min_notional=min_notional,
        )
        logger.info(
            "Instrument added: %s tick=%.6f step=%.6f min_notional=%.2f",
            symbol, tick_size, step_size, min_notional,
        )

    def get_instrument(self, symbol: str) -> Optional[InstrumentData]:
        """Return instrument data or None."""
        return self._instruments.get(symbol)

    def get_all_instruments(self) -> Dict[str, InstrumentData]:
        """Return all instruments."""
        return dict(self._instruments)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update_book_ticker(self, symbol: str, best_bid: float, best_ask: float) -> None:
        """Update best bid/ask from bookTicker stream."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return
        inst.best_bid = best_bid
        inst.best_ask = best_ask
        inst.mid_price = (best_bid + best_ask) / 2.0
        inst.params.mid_price = inst.mid_price
        if inst.mid_price > 0:
            inst.params.market_spread = best_ask - best_bid
            inst.params.market_spread_pct = (best_ask - best_bid) / inst.mid_price

    def update_depth(self, symbol: str, bids: List, asks: List) -> None:
        """Update order book depth from depth20 stream.

        Calculates total USDT value of top 5 levels on both sides.
        """
        inst = self._instruments.get(symbol)
        if inst is None:
            return

        # Sum USDT value of top 5 levels on each side
        bid_depth = sum(float(b[0]) * float(b[1]) for b in bids[:5]) if bids else 0.0
        ask_depth = sum(float(a[0]) * float(a[1]) for a in asks[:5]) if asks else 0.0
        total_depth = (bid_depth + ask_depth) / 2.0

        now = time.time()
        inst.current_depth_usdt = total_depth
        inst.depth_history.append(total_depth)
        inst.depth_timestamps.append(now)

    def update_candle_1m(self, symbol: str, close_price: float, timestamp: float) -> None:
        """Ingest a 1-minute candle close for volatility calculation."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return
        inst.candle_closes.append(close_price)
        inst.candle_timestamps.append(timestamp)

    def update_candle_5m(self, symbol: str, close_price: float) -> None:
        """Ingest a 5-minute candle close."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return
        inst.candle_5m_closes.append(close_price)

    def update_candle_1h(self, symbol: str, close_price: float) -> None:
        """Ingest a 1-hour candle close for trend detection."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return
        inst.candle_1h_closes.append(close_price)

    def update_mark_price(self, symbol: str, mark_price: float,
                          funding_rate: float) -> None:
        """Update mark price and funding rate from markPrice stream."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return
        inst.mark_price = mark_price
        inst.funding_rate = funding_rate

    def update_trade(self, symbol: str, price: float, quantity: float) -> None:
        """Ingest an aggregated trade for volume/size tracking."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return
        inst.trade_sizes.append(quantity * price)  # USDT notional

    def update_liquidation(self, symbol: str, timestamp: float) -> None:
        """Record a liquidation event from forceOrder stream."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return
        inst.liquidation_events.append(timestamp)

    def update_volume_24h(self, symbol: str, volume: float) -> None:
        """Update 24h volume for viability filtering."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return
        inst.volume_24h = volume

    def update_inventory(self, symbol: str, qty: float, avg_cost: float) -> None:
        """Update the current inventory for an instrument."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return
        inst.inventory_qty = qty
        inst.inventory_avg_cost = avg_cost
        if inst.mid_price > 0:
            inst.inventory_notional = abs(qty) * inst.mid_price

    # ------------------------------------------------------------------
    # Model calibration
    # ------------------------------------------------------------------

    def calibrate(self, symbol: str, max_inventory_notional: float,
                  total_equity: float) -> Optional[ModelParameters]:
        """Recalibrate all model parameters for an instrument.

        Called on startup and every 15 minutes thereafter.

        Parameters
        ----------
        symbol : str
            Instrument to calibrate.
        max_inventory_notional : float
            Maximum allowed inventory notional for this instrument.
        total_equity : float
            Total account equity for inventory ratio calculation.

        Returns
        -------
        ModelParameters or None if calibration fails.
        """
        inst = self._instruments.get(symbol)
        if inst is None:
            return None

        if inst.mid_price <= 0:
            logger.warning("Cannot calibrate %s: no mid-price", symbol)
            return None

        # Step 1: Volatility (sigma)
        sigma = self._calculate_volatility(inst)
        if sigma is None or sigma <= 0:
            logger.warning("Cannot calibrate %s: insufficient candle data for volatility", symbol)
            return None

        # Step 2: Order book depth (kappa)
        kappa = self._calculate_kappa(inst)

        # Step 3: Risk aversion (gamma)
        gamma = self._calculate_gamma(inst, max_inventory_notional)

        # Step 4: Session time remaining
        time_remaining = get_session_time_remaining()
        time_remaining = max(time_remaining, 0.01)  # Prevent division by zero

        # Store parameters
        inst.params.sigma = sigma
        inst.params.sigma_annual = sigma * math.sqrt(8760)
        inst.params.kappa = kappa
        inst.params.gamma = gamma
        inst.params.time_remaining = time_remaining
        inst.params.last_calibration = time.time()

        logger.info(
            "Calibrated %s: sigma=%.6f (annual=%.4f), kappa=%.2f, gamma=%.3f, "
            "T-t=%.2fh, inventory=%.4f",
            symbol, sigma, inst.params.sigma_annual, kappa, gamma,
            time_remaining, inst.inventory_qty,
        )

        return inst.params

    def _calculate_volatility(self, inst: InstrumentData) -> Optional[float]:
        """Calculate realized volatility from 1m candle log-returns.

        sigma = stdev(ln(Close_i / Close_(i-1))) over last 60 candles * sqrt(60)
        This gives hourly volatility.
        """
        closes = list(inst.candle_closes)
        if len(closes) < 10:  # Need at least 10 candles
            return None

        # Use last vol_window candles
        window = closes[-self._vol_window:] if len(closes) >= self._vol_window else closes

        # Compute log returns
        log_returns = []
        for i in range(1, len(window)):
            if window[i - 1] > 0 and window[i] > 0:
                log_returns.append(math.log(window[i] / window[i - 1]))

        if len(log_returns) < 5:
            return None

        # Standard deviation of 1-minute log returns, scaled to hourly
        sigma_1m = float(np.std(log_returns, ddof=1))
        sigma_hourly = sigma_1m * math.sqrt(60)

        return sigma_hourly

    def _calculate_kappa(self, inst: InstrumentData) -> float:
        """Calculate order book depth parameter kappa.

        kappa = Average_Depth / (Mid_Price * 0.001)
        Higher kappa = deeper book = tighter spread acceptable.
        """
        if not inst.depth_history:
            return 1.0  # Default

        # Average depth over available history (up to 30 minutes)
        # depth_history is sampled roughly every 100ms from depth20 stream
        depth_values = list(inst.depth_history)
        if not depth_values:
            return 1.0

        avg_depth = sum(depth_values) / len(depth_values)

        if inst.mid_price <= 0:
            return 1.0

        kappa = avg_depth / (inst.mid_price * 0.001)
        kappa = max(kappa, 0.1)  # Floor to prevent division issues
        return kappa

    def _calculate_gamma(self, inst: InstrumentData,
                         max_inventory_notional: float) -> float:
        """Calculate adaptive risk aversion parameter gamma.

        Gamma increases with inventory to encourage mean reversion:
        - < 25% of max: relaxed (0.05) — accumulate spread
        - 25-50%: base (0.1)
        - > 50%: aggressive (0.3)
        - > 75%: very aggressive (0.5)
        """
        if max_inventory_notional <= 0:
            return self._base_gamma

        inv_pct = abs(inst.inventory_notional) / max_inventory_notional

        if inv_pct < 0.25:
            return self._gamma_relaxed
        elif inv_pct < 0.50:
            return self._base_gamma
        elif inv_pct < 0.75:
            return self._gamma_aggressive
        else:
            return self._gamma_very_aggressive

    # ------------------------------------------------------------------
    # Quote calculation
    # ------------------------------------------------------------------

    def calculate_quotes(self, symbol: str, max_inventory_notional: float,
                         total_equity: float) -> Optional[QuotePrices]:
        """Calculate optimal bid/ask quotes using Avellaneda-Stoikov model.

        Returns QuotePrices with Level 1/2/3 prices, or None if quoting
        is not possible.
        """
        inst = self._instruments.get(symbol)
        if inst is None or inst.mid_price <= 0:
            return None

        p = inst.params
        if p.sigma <= 0 or p.last_calibration == 0:
            return None

        # Update gamma dynamically on every quote calculation
        p.gamma = self._calculate_gamma(inst, max_inventory_notional)

        # Update time remaining
        p.time_remaining = max(get_session_time_remaining(), 0.01)

        mid = inst.mid_price
        q = inst.inventory_qty
        gamma = p.gamma
        sigma = p.sigma
        sigma_sq = sigma * sigma
        t_remain = p.time_remaining
        kappa = p.kappa

        # Step 1: Reservation Price
        # r = s - q * gamma * sigma^2 * (T - t)
        reservation_price = mid - q * gamma * sigma_sq * t_remain

        # Step 2: Optimal Spread
        # delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)
        if kappa > 0:
            optimal_spread = gamma * sigma_sq * t_remain + (2.0 / gamma) * math.log(1.0 + gamma / kappa)
        else:
            optimal_spread = gamma * sigma_sq * t_remain + (2.0 / gamma) * math.log(2.0)

        half_spread = optimal_spread / 2.0

        # Step 3: Raw bid/ask
        bid_raw = reservation_price - half_spread
        ask_raw = reservation_price + half_spread
        spread_raw = ask_raw - bid_raw

        # Step 5: Minimum spread enforcement
        # min_spread = 2 * maker_fee + 0.005% = 0.045%
        min_spread_abs = mid * self._min_spread_pct
        if spread_raw < min_spread_abs:
            # Widen symmetrically
            deficit = (min_spread_abs - spread_raw) / 2.0
            bid_raw -= deficit
            ask_raw += deficit
            spread_raw = min_spread_abs

        # Step 6: Maximum spread enforcement
        # max_spread = 5 * current market spread
        if p.market_spread > 0:
            max_spread_abs = self._max_spread_mult * p.market_spread
            if spread_raw > max_spread_abs and max_spread_abs > min_spread_abs:
                # Narrow to max
                excess = (spread_raw - max_spread_abs) / 2.0
                bid_raw += excess
                ask_raw -= excess
                spread_raw = max_spread_abs

        # Step 4: Round to tick size
        tick = inst.tick_size
        if tick > 0:
            bid_l1 = math.floor(bid_raw / tick) * tick   # Round bid DOWN
            ask_l1 = math.ceil(ask_raw / tick) * tick     # Round ask UP
        else:
            bid_l1 = bid_raw
            ask_l1 = ask_raw

        # Ensure bid < ask after rounding
        if bid_l1 >= ask_l1:
            ask_l1 = bid_l1 + tick

        # Layer 2 and 3 prices
        # Level 2: delta from reservation (1x half_spread wider)
        hs2 = half_spread * (1.0 + self._layer2_mult)
        bid_l2 = math.floor((reservation_price - hs2) / tick) * tick if tick > 0 else reservation_price - hs2
        ask_l2 = math.ceil((reservation_price + hs2) / tick) * tick if tick > 0 else reservation_price + hs2

        # Level 3: 1.5x delta from reservation
        hs3 = half_spread * (1.0 + self._layer3_mult)
        bid_l3 = math.floor((reservation_price - hs3) / tick) * tick if tick > 0 else reservation_price - hs3
        ask_l3 = math.ceil((reservation_price + hs3) / tick) * tick if tick > 0 else reservation_price + hs3

        spread_pct = (ask_l1 - bid_l1) / mid if mid > 0 else 0.0

        quotes = QuotePrices(
            bid_l1=bid_l1,
            ask_l1=ask_l1,
            bid_l2=bid_l2,
            ask_l2=ask_l2,
            bid_l3=bid_l3,
            ask_l3=ask_l3,
            reservation_price=reservation_price,
            half_spread=half_spread,
            spread_pct=spread_pct,
            timestamp=time.time(),
        )

        # Store in instrument data
        inst.quotes = quotes
        inst.params.reservation_price = reservation_price
        inst.params.optimal_spread = optimal_spread
        inst.params.bid_quote = bid_l1
        inst.params.ask_quote = ask_l1

        return quotes

    def quotes_need_update(self, symbol: str, current_bid: float,
                           current_ask: float) -> bool:
        """Check if new quotes differ from current by more than threshold.

        Only update quotes when > 0.01% change to minimize API calls.
        """
        inst = self._instruments.get(symbol)
        if inst is None or inst.mid_price <= 0:
            return True

        q = inst.quotes
        if q.bid_l1 <= 0 or q.ask_l1 <= 0:
            return True

        if current_bid <= 0 or current_ask <= 0:
            return True

        threshold = self._quote_update_threshold  # 0.0001 = 0.01%

        bid_change = abs(q.bid_l1 - current_bid) / inst.mid_price
        ask_change = abs(q.ask_l1 - current_ask) / inst.mid_price

        return bid_change > threshold or ask_change > threshold

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def check_spread_viability(self, symbol: str) -> Tuple[bool, str]:
        """Check if the instrument's spread is wide enough for market making.

        Market spread must be > 0.03% for retail MM to be viable.
        """
        inst = self._instruments.get(symbol)
        if inst is None:
            return False, "Unknown instrument"

        min_viable = self._params.get("min_viable_market_spread", 0.03) / 100.0
        if inst.params.market_spread_pct < min_viable:
            return False, (
                f"Market spread {inst.params.market_spread_pct * 100:.4f}% "
                f"< minimum viable {min_viable * 100:.3f}%"
            )
        return True, ""

    def check_volume_filter(self, symbol: str) -> Tuple[bool, str]:
        """Check if 24h volume is in the $30M-$200M target range."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return False, "Unknown instrument"

        min_vol = self._params.get("min_24h_volume", 30_000_000)
        max_vol = self._params.get("max_24h_volume", 200_000_000)

        if inst.volume_24h < min_vol:
            return False, f"24h volume ${inst.volume_24h / 1e6:.1f}M < ${min_vol / 1e6:.0f}M minimum"
        if inst.volume_24h > max_vol:
            return False, f"24h volume ${inst.volume_24h / 1e6:.1f}M > ${max_vol / 1e6:.0f}M maximum"
        return True, ""

    def check_trend_filter(self, symbol: str) -> Tuple[bool, str, float]:
        """Check 1h EMA trend and ADX for the instrument.

        Returns (ok, reason, widen_factor):
        - ok=True, widen_factor=0 -> normal quoting
        - ok=True, widen_factor>0 -> widen spreads by this factor
        - ok=False -> halt market making
        """
        inst = self._instruments.get(symbol)
        if inst is None:
            return False, "Unknown instrument", 0.0

        closes = list(inst.candle_1h_closes)
        if len(closes) < 2:
            return True, "", 0.0

        # EMA slope: approximate via last 2 values of simple moving average
        ema_period = self._params.get("ema_period", 20)
        if len(closes) >= ema_period:
            # Calculate EMA
            ema = self._ema(closes, ema_period)
            if len(ema) >= 2 and ema[-2] > 0:
                slope = (ema[-1] - ema[-2]) / ema[-2]
                threshold = self._params.get("ema_trend_threshold", 0.001)
                widen_pct = self._params.get("ema_trend_widen_pct", 0.25)

                if abs(slope) > threshold:
                    return True, f"Trending (slope={slope:.5f}), widening spreads", widen_pct
        elif len(closes) >= 3:
            # Not enough for full EMA, use simple slope
            if closes[-2] > 0:
                slope = (closes[-1] - closes[-2]) / closes[-2]
                threshold = self._params.get("ema_trend_threshold", 0.001)
                if abs(slope) > threshold:
                    widen_pct = self._params.get("ema_trend_widen_pct", 0.25)
                    return True, f"Short trend detected (slope={slope:.5f})", widen_pct

        # ADX check (simplified: use directional movement from 1h candles)
        adx = self._simple_adx(closes)
        adx_threshold = self._params.get("adx_halt_threshold", 30)
        if adx > adx_threshold:
            return False, f"ADX {adx:.1f} > {adx_threshold} — halt market making", 0.0

        return True, "", 0.0

    def check_volatility_spike(self, symbol: str) -> bool:
        """Check if current 1-minute volatility exceeds 3x the hourly average.

        Returns True if a volatility spike is detected -> withdraw quotes.
        """
        inst = self._instruments.get(symbol)
        if inst is None:
            return False

        closes = list(inst.candle_closes)
        if len(closes) < 15:
            return False

        # Hourly average volatility (using all data)
        all_returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                all_returns.append(abs(math.log(closes[i] / closes[i - 1])))

        if not all_returns:
            return False

        hourly_avg = sum(all_returns) / len(all_returns)

        # Last 1-minute return
        if closes[-1] > 0 and closes[-2] > 0:
            last_return = abs(math.log(closes[-1] / closes[-2]))
        else:
            return False

        multiplier = self._params.get("vol_spike_multiplier", 3.0)
        return last_return > multiplier * hourly_avg

    def check_large_trade(self, symbol: str, trade_notional: float) -> bool:
        """Check if a trade is 5x the average trade size."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return False

        if not inst.trade_sizes:
            return False

        avg_size = sum(inst.trade_sizes) / len(inst.trade_sizes)
        multiplier = self._params.get("large_trade_multiplier", 5.0)
        return trade_notional > multiplier * avg_size

    def check_liquidation_cascade(self, symbol: str) -> bool:
        """Check if liquidation events exceed 10 per minute."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return False

        now = time.time()
        threshold = self._params.get("liquidation_rate_threshold", 10)
        recent = sum(1 for t in inst.liquidation_events if now - t < 60)
        return recent > threshold

    def check_depth_withdrawal(self, symbol: str) -> bool:
        """Check if order book depth dropped >50% in 5 seconds.

        Sudden liquidity withdrawal -> withdraw quotes immediately.
        """
        inst = self._instruments.get(symbol)
        if inst is None:
            return False

        if len(inst.depth_history) < 50:  # Need some history
            return False

        window_s = self._params.get("depth_drop_window_seconds", 5)
        drop_pct = self._params.get("depth_drop_pct", 0.50)

        now = time.time()
        recent_depths = []
        older_depths = []
        for i, (ts, depth) in enumerate(zip(inst.depth_timestamps, inst.depth_history)):
            if now - ts < window_s:
                recent_depths.append(depth)
            elif now - ts < window_s * 3:
                older_depths.append(depth)

        if not recent_depths or not older_depths:
            return False

        avg_recent = sum(recent_depths) / len(recent_depths)
        avg_older = sum(older_depths) / len(older_depths)

        if avg_older <= 0:
            return False

        drop = (avg_older - avg_recent) / avg_older
        return drop > drop_pct

    # ------------------------------------------------------------------
    # Carry evaluation (session end)
    # ------------------------------------------------------------------

    def evaluate_carry(self, symbol: str) -> Tuple[bool, str]:
        """Evaluate if holding through funding is favorable.

        Returns (hold, reason).
        If long and funding rate is negative (shorts pay longs) -> favorable.
        If short and funding rate is positive (longs pay shorts) -> favorable.
        """
        inst = self._instruments.get(symbol)
        if inst is None:
            return False, "Unknown instrument"

        if abs(inst.inventory_qty) < 1e-10:
            return True, "No inventory"

        threshold = self._params.get("carry_favorable_threshold", 0.0)

        if inst.inventory_qty > 0:
            # We are long
            if inst.funding_rate < -threshold:
                return True, f"Favorable carry: long with funding={inst.funding_rate:.6f}"
            else:
                return False, f"Unfavorable carry: long with funding={inst.funding_rate:.6f}"
        else:
            # We are short
            if inst.funding_rate > threshold:
                return True, f"Favorable carry: short with funding={inst.funding_rate:.6f}"
            else:
                return False, f"Unfavorable carry: short with funding={inst.funding_rate:.6f}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(data: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if not data or period <= 0:
            return []
        alpha = 2.0 / (period + 1)
        ema_values = [data[0]]
        for i in range(1, len(data)):
            ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[-1])
        return ema_values

    @staticmethod
    def _simple_adx(closes: List[float], period: int = 14) -> float:
        """Simplified ADX calculation from close prices.

        Returns 0-100. Higher values = stronger trend.
        """
        if len(closes) < period + 2:
            return 0.0

        # Use absolute returns as proxy for directional movement
        returns = [abs(closes[i] - closes[i - 1]) / closes[i - 1]
                   for i in range(1, len(closes)) if closes[i - 1] > 0]

        if len(returns) < period:
            return 0.0

        # Smooth absolute returns
        avg_move = sum(returns[-period:]) / period

        # Calculate directional consistency
        positive_moves = sum(1 for i in range(1, len(closes))
                            if closes[i] > closes[i - 1])
        negative_moves = len(closes) - 1 - positive_moves

        if len(closes) <= 1:
            return 0.0

        # ADX proxy: directional consistency * magnitude
        consistency = abs(positive_moves - negative_moves) / (len(closes) - 1)
        adx_proxy = consistency * avg_move * 10000  # Scale to 0-100 range

        return min(adx_proxy, 100.0)

    def get_model_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return serializable model state for persistence."""
        inst = self._instruments.get(symbol)
        if inst is None:
            return None

        return {
            "symbol": symbol,
            "mid_price": inst.mid_price,
            "best_bid": inst.best_bid,
            "best_ask": inst.best_ask,
            "mark_price": inst.mark_price,
            "funding_rate": inst.funding_rate,
            "inventory_qty": inst.inventory_qty,
            "inventory_avg_cost": inst.inventory_avg_cost,
            "inventory_notional": inst.inventory_notional,
            "sigma": inst.params.sigma,
            "sigma_annual": inst.params.sigma_annual,
            "kappa": inst.params.kappa,
            "gamma": inst.params.gamma,
            "time_remaining": inst.params.time_remaining,
            "reservation_price": inst.params.reservation_price,
            "optimal_spread": inst.params.optimal_spread,
            "bid_quote": inst.params.bid_quote,
            "ask_quote": inst.params.ask_quote,
            "market_spread_pct": inst.params.market_spread_pct,
            "quotes": {
                "bid_l1": inst.quotes.bid_l1,
                "ask_l1": inst.quotes.ask_l1,
                "bid_l2": inst.quotes.bid_l2,
                "ask_l2": inst.quotes.ask_l2,
                "bid_l3": inst.quotes.bid_l3,
                "ask_l3": inst.quotes.ask_l3,
                "spread_pct": inst.quotes.spread_pct,
            },
            "is_active": inst.is_active,
            "halt_reason": inst.halt_reason,
            "volume_24h": inst.volume_24h,
            "last_calibration": inst.params.last_calibration,
        }
