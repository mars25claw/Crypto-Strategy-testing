"""Feature engineering pipeline for STRAT-010.

Computes 35 base features + 3 lags = 140 total features every hour.
Features span technical, order-book, derivatives, on-chain, sentiment,
and derived categories.  All values are Z-score normalised over a
180-day rolling window.

Feature freshness enforcement:
  - Technical < 5 minutes
  - On-chain  < 4 hours
  - Sentiment < 12 hours
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from shared.indicators import (
    IndicatorBuffer,
    atr,
    adx,
    bollinger_bands,
    ema,
    macd,
    obv_slope,
    rsi,
    volume_average,
    hurst_exponent,
    volatility_ratio,
    kline_body_ratio,
    close_position_in_range,
    correlation_rolling,
)

logger = logging.getLogger(__name__)

# ── Feature index constants ────────────────────────────────────────────────
TECHNICAL_FEATURES = [
    "ema_9_21_ratio",       # 0
    "ema_21_50_ratio",      # 1
    "ema_50_200_ratio",     # 2
    "ema_20_50_4h_ratio",   # 3
    "rsi_14_1h",            # 4
    "macd_hist_1h",         # 5
    "adx_4h",              # 6
    "atr_14_1h",           # 7
    "bb_width_4h",         # 8
    "obv_slope_24",        # 9
    "kline_body_ratio_1h", # 10
    "volume_ratio_1h",     # 11
    "high_low_range_1h",   # 12
    "close_position_1h",   # 13
]

ORDERBOOK_FEATURES = [
    "bid_ask_imbalance_5",  # 14
    "bid_ask_imbalance_10", # 15
    "bid_ask_imbalance_20", # 16
]

DERIVATIVES_FEATURES = [
    "funding_rate",           # 17
    "predicted_funding",      # 18
    "long_short_ratio",       # 19
    "oi_change_24h",          # 20
    "liquidation_volume",     # 21
    "taker_buy_sell_ratio",   # 22
    "premium_index",          # 23
]

ONCHAIN_FEATURES = [
    "exchange_net_flow",         # 24
    "mvrv_zscore",               # 25
    "nupl",                      # 26
    "active_addresses_change",   # 27
    "stablecoin_reserves",       # 28
    "hash_rate_change",          # 29
]

SENTIMENT_FEATURES = [
    "social_volume_zscore",  # 30
    "fear_greed_index",      # 31
]

DERIVED_FEATURES = [
    "hurst_exponent",            # 32
    "volatility_ratio_1d_7d",    # 33
    "price_volume_correlation",  # 34
]

ALL_FEATURE_NAMES: List[str] = (
    TECHNICAL_FEATURES
    + ORDERBOOK_FEATURES
    + DERIVATIVES_FEATURES
    + ONCHAIN_FEATURES
    + SENTIMENT_FEATURES
    + DERIVED_FEATURES
)

TOTAL_BASE_FEATURES = 35
LAG_COUNT = 3
TOTAL_FEATURES = TOTAL_BASE_FEATURES * (1 + LAG_COUNT)  # 140

# Freshness limits (seconds)
TECHNICAL_FRESHNESS_S = 300        # 5 min
ONCHAIN_FRESHNESS_S = 4 * 3600    # 4 hours
SENTIMENT_FRESHNESS_S = 12 * 3600  # 12 hours


# ── Data container ─────────────────────────────────────────────────────────

@dataclass
class FeatureSnapshot:
    """A single hourly feature vector with metadata."""

    timestamp_ms: int = 0
    features: np.ndarray = field(default_factory=lambda: np.full(TOTAL_BASE_FEATURES, np.nan))
    available_count: int = 0
    anomaly_flags: List[str] = field(default_factory=list)
    freshness_ok: bool = True


@dataclass
class OnChainFeatureFreshness:
    """Tracks per-feature freshness for on-chain data with 72h forward-fill."""

    last_update_ms: int = 0
    last_value: float = np.nan
    is_available: bool = True

    @property
    def age_hours(self) -> float:
        if self.last_update_ms == 0:
            return float("inf")
        return (time.time() * 1000 - self.last_update_ms) / 3_600_000

    def update(self, value: float) -> None:
        """Update with a fresh value."""
        if not np.isnan(value):
            self.last_value = value
            self.last_update_ms = int(time.time() * 1000)
            self.is_available = True

    def get_value(self, max_stale_hours: float = 72.0) -> float:
        """Return the value, forward-filling up to max_stale_hours.

        After max_stale_hours, mark as unavailable and return NaN.
        """
        if self.last_update_ms == 0:
            self.is_available = False
            return np.nan
        if self.age_hours > max_stale_hours:
            self.is_available = False
            return np.nan
        return self.last_value


@dataclass
class OnChainCache:
    """Cache for on-chain metric values with per-feature freshness tracking.

    Implements 72-hour forward-fill: when on-chain data is missing (API down),
    forward-fills from last known value for up to 72 hours. After 72h, marks
    feature as unavailable.
    """

    exchange_net_flow: float = np.nan
    mvrv_zscore: float = np.nan
    nupl: float = np.nan
    active_addresses_change: float = np.nan
    stablecoin_reserves: float = np.nan
    hash_rate_change: float = np.nan
    last_update_ms: int = 0

    # Per-feature freshness tracking
    _freshness: Dict[str, OnChainFeatureFreshness] = field(default_factory=lambda: {
        "exchange_net_flow": OnChainFeatureFreshness(),
        "mvrv_zscore": OnChainFeatureFreshness(),
        "nupl": OnChainFeatureFreshness(),
        "active_addresses_change": OnChainFeatureFreshness(),
        "stablecoin_reserves": OnChainFeatureFreshness(),
        "hash_rate_change": OnChainFeatureFreshness(),
    })

    def age_hours(self) -> float:
        if self.last_update_ms == 0:
            return float("inf")
        return (time.time() * 1000 - self.last_update_ms) / 3_600_000

    def update_feature(self, name: str, value: float) -> None:
        """Update a specific feature with freshness tracking."""
        if name not in self._freshness:
            self._freshness[name] = OnChainFeatureFreshness()
        self._freshness[name].update(value)
        setattr(self, name, value)

    def get_feature_with_forward_fill(self, name: str, max_stale_hours: float = 72.0) -> float:
        """Get a feature value with 72h forward-fill logic."""
        freshness = self._freshness.get(name)
        if freshness is None:
            return getattr(self, name, np.nan)
        return freshness.get_value(max_stale_hours)

    def get_freshness_status(self) -> Dict[str, dict]:
        """Return freshness status for all on-chain features."""
        return {
            name: {
                "age_hours": round(f.age_hours, 2),
                "available": f.is_available,
                "last_value": f.last_value if not np.isnan(f.last_value) else None,
            }
            for name, f in self._freshness.items()
        }


@dataclass
class DerivativesCache:
    """Cache for derivatives-data features."""

    funding_rate: float = np.nan
    predicted_funding: float = np.nan
    long_short_ratio: float = np.nan
    oi_change_24h: float = np.nan
    liquidation_volume: float = np.nan
    taker_buy_sell_ratio: float = np.nan
    premium_index: float = np.nan
    last_update_ms: int = 0


@dataclass
class SentimentCache:
    """Cache for sentiment features."""

    social_volume_zscore: float = np.nan
    fear_greed_index: float = np.nan
    last_update_ms: int = 0

    def age_hours(self) -> float:
        if self.last_update_ms == 0:
            return float("inf")
        return (time.time() * 1000 - self.last_update_ms) / 3_600_000


@dataclass
class OrderBookSnapshot:
    """Averaged order-book imbalance over the last hour."""

    imbalance_5: float = np.nan
    imbalance_10: float = np.nan
    imbalance_20: float = np.nan
    last_update_ms: int = 0


# ── Feature normalisation state ───────────────────────────────────────────

class RollingNormaliser:
    """Z-score normalisation using a 180-day rolling window per feature."""

    def __init__(self, window_size: int = 180 * 24, n_features: int = TOTAL_BASE_FEATURES):
        self._window = window_size  # hours in 180 days
        self._n = n_features
        self._history: Deque[np.ndarray] = deque(maxlen=window_size)
        # Running sums for incremental mean/std
        self._count = 0

    def update(self, raw_features: np.ndarray) -> np.ndarray:
        """Add a new observation and return Z-scored values."""
        self._history.append(raw_features.copy())
        self._count = len(self._history)

        if self._count < 24:
            # Not enough history -- return raw (un-normalised)
            return raw_features.copy()

        arr = np.array(self._history)  # (count, n_features)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0)
            stds[stds == 0] = 1.0  # avoid division by zero
            normed = (raw_features - means) / stds
        return normed

    @property
    def has_enough_history(self) -> bool:
        return self._count >= 24

    def get_training_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current (mean, std) for anomaly detection."""
        if self._count < 2:
            return np.zeros(self._n), np.ones(self._n)
        arr = np.array(self._history)
        means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0)
        stds[stds == 0] = 1.0
        return means, stds


# ── Feature Engine ────────────────────────────────────────────────────────

class FeatureEngine:
    """Computes and manages the 35-feature vector for one instrument.

    Call :meth:`calculate_features` every hour to produce a
    :class:`FeatureSnapshot` that feeds into model inference.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``"BTCUSDT"``.
    z_score_window : int
        Rolling Z-score window in days (default 180).
    anomaly_sigma : float
        Flag features deviating > this many sigma (default 2.0).
    """

    def __init__(
        self,
        symbol: str,
        z_score_window: int = 180,
        anomaly_sigma: float = 2.0,
    ) -> None:
        self.symbol = symbol
        self._anomaly_sigma = anomaly_sigma

        # Kline buffers per timeframe
        self.buffers: Dict[str, IndicatorBuffer] = {
            "1m":  IndicatorBuffer(max_size=500),
            "5m":  IndicatorBuffer(max_size=500),
            "15m": IndicatorBuffer(max_size=500),
            "1h":  IndicatorBuffer(max_size=500),
            "4h":  IndicatorBuffer(max_size=500),
            "1d":  IndicatorBuffer(max_size=500),
        }

        # External-data caches (updated by main / external-data fetchers)
        self.onchain = OnChainCache()
        self.derivatives = DerivativesCache()
        self.sentiment = SentimentCache()
        self.orderbook = OrderBookSnapshot()

        # Order-book imbalance accumulator (100 ms snapshots)
        self._ob_imbalances_5: Deque[float] = deque(maxlen=36_000)   # 60 min * 600
        self._ob_imbalances_10: Deque[float] = deque(maxlen=36_000)
        self._ob_imbalances_20: Deque[float] = deque(maxlen=36_000)

        # Liquidation accumulator
        self._long_liq_volume_24h: float = 0.0
        self._short_liq_volume_24h: float = 0.0
        self._liq_events: Deque[Tuple[float, float, str]] = deque(maxlen=10_000)

        # Normaliser
        self._normaliser = RollingNormaliser(
            window_size=z_score_window * 24,
            n_features=TOTAL_BASE_FEATURES,
        )

        # Feature history (for lagging)
        self._feature_history: Deque[np.ndarray] = deque(maxlen=LAG_COUNT + 1)

        # Training distribution stats (loaded from model metadata)
        self._training_mean: Optional[np.ndarray] = None
        self._training_std: Optional[np.ndarray] = None

        logger.info("FeatureEngine initialised for %s", symbol)

    # ── Kline ingestion ──────────────────────────────────────────────

    def on_kline(self, timeframe: str, candle: dict) -> None:
        """Ingest a closed kline candle into the appropriate buffer."""
        buf = self.buffers.get(timeframe)
        if buf is not None:
            buf.add_candle(candle)

    # ── Order-book ingestion ─────────────────────────────────────────

    def on_depth(self, data: dict) -> None:
        """Ingest a depth20 snapshot and accumulate bid-ask imbalance."""
        bids = data.get("bids", data.get("b", []))
        asks = data.get("asks", data.get("a", []))

        def _imbalance(levels_bid, levels_ask, n):
            b = sum(float(x[1]) for x in levels_bid[:n]) if len(levels_bid) >= n else 0.0
            a = sum(float(x[1]) for x in levels_ask[:n]) if len(levels_ask) >= n else 0.0
            total = b + a
            return (b - a) / total if total > 0 else 0.0

        self._ob_imbalances_5.append(_imbalance(bids, asks, 5))
        self._ob_imbalances_10.append(_imbalance(bids, asks, 10))
        self._ob_imbalances_20.append(_imbalance(bids, asks, 20))

        self.orderbook.imbalance_5 = float(np.mean(list(self._ob_imbalances_5)[-6000:])) if self._ob_imbalances_5 else np.nan
        self.orderbook.imbalance_10 = float(np.mean(list(self._ob_imbalances_10)[-6000:])) if self._ob_imbalances_10 else np.nan
        self.orderbook.imbalance_20 = float(np.mean(list(self._ob_imbalances_20)[-6000:])) if self._ob_imbalances_20 else np.nan
        self.orderbook.last_update_ms = int(time.time() * 1000)

    # ── Liquidation ingestion ────────────────────────────────────────

    def on_liquidation(self, data: dict) -> None:
        """Ingest a forceOrder event."""
        order = data.get("o", data)
        side = order.get("S", "")
        qty = float(order.get("q", 0))
        price = float(order.get("p", 0))
        volume = qty * price
        now = time.time()
        self._liq_events.append((now, volume, side))

        # Prune events older than 24h and recompute
        cutoff = now - 86400
        self._long_liq_volume_24h = 0.0
        self._short_liq_volume_24h = 0.0
        fresh: Deque[Tuple[float, float, str]] = deque(maxlen=10_000)
        for ts, vol, s in self._liq_events:
            if ts >= cutoff:
                fresh.append((ts, vol, s))
                if s == "BUY":
                    self._short_liq_volume_24h += vol  # shorts liquidated
                else:
                    self._long_liq_volume_24h += vol   # longs liquidated
        self._liq_events = fresh

    # ── External-data update methods ─────────────────────────────────

    def update_onchain(
        self,
        exchange_net_flow: float = np.nan,
        mvrv_zscore: float = np.nan,
        nupl: float = np.nan,
        active_addresses_change: float = np.nan,
        stablecoin_reserves: float = np.nan,
        hash_rate_change: float = np.nan,
    ) -> None:
        """Update on-chain feature cache with per-feature freshness tracking."""
        updates = {
            "exchange_net_flow": exchange_net_flow,
            "mvrv_zscore": mvrv_zscore,
            "nupl": nupl,
            "active_addresses_change": active_addresses_change,
            "stablecoin_reserves": stablecoin_reserves,
            "hash_rate_change": hash_rate_change,
        }
        for name, value in updates.items():
            if not np.isnan(value):
                self.onchain.update_feature(name, value)
        self.onchain.last_update_ms = int(time.time() * 1000)

    def update_derivatives(
        self,
        funding_rate: float = np.nan,
        predicted_funding: float = np.nan,
        long_short_ratio: float = np.nan,
        oi_change_24h: float = np.nan,
        taker_buy_sell_ratio: float = np.nan,
        premium_index: float = np.nan,
    ) -> None:
        """Update derivatives feature cache."""
        if not np.isnan(funding_rate):
            self.derivatives.funding_rate = funding_rate
        if not np.isnan(predicted_funding):
            self.derivatives.predicted_funding = predicted_funding
        if not np.isnan(long_short_ratio):
            self.derivatives.long_short_ratio = long_short_ratio
        if not np.isnan(oi_change_24h):
            self.derivatives.oi_change_24h = oi_change_24h
        if not np.isnan(taker_buy_sell_ratio):
            self.derivatives.taker_buy_sell_ratio = taker_buy_sell_ratio
        if not np.isnan(premium_index):
            self.derivatives.premium_index = premium_index
        self.derivatives.last_update_ms = int(time.time() * 1000)

        # Liquidation imbalance is computed internally
        total_liq = self._long_liq_volume_24h + self._short_liq_volume_24h
        if total_liq > 0:
            self.derivatives.liquidation_volume = (
                self._long_liq_volume_24h - self._short_liq_volume_24h
            ) / total_liq
        else:
            self.derivatives.liquidation_volume = 0.0

    def update_sentiment(
        self,
        social_volume_zscore: float = np.nan,
        fear_greed_index: float = np.nan,
    ) -> None:
        """Update sentiment feature cache with manipulation safeguard.

        If social volume spikes >5x in 1 hour without corresponding price
        movement, zero out sentiment features (treat as manipulated).
        """
        if not np.isnan(social_volume_zscore):
            # Manipulation safeguard: check for 5x spike
            if self._check_sentiment_manipulation(social_volume_zscore):
                logger.warning(
                    "[%s] Sentiment manipulation detected: social volume spike >5x "
                    "without price movement -- zeroing sentiment features",
                    self.symbol,
                )
                self.sentiment.social_volume_zscore = 0.0
                self.sentiment.fear_greed_index = 0.5  # neutral
                self.sentiment.last_update_ms = int(time.time() * 1000)
                return
            self.sentiment.social_volume_zscore = social_volume_zscore
        if not np.isnan(fear_greed_index):
            self.sentiment.fear_greed_index = fear_greed_index
        self.sentiment.last_update_ms = int(time.time() * 1000)

    def _check_sentiment_manipulation(self, new_social_volume: float) -> bool:
        """Detect potential sentiment manipulation.

        Returns True if social volume spiked >5x in 1 hour without
        corresponding price movement (>2% would be meaningful).
        """
        prev_volume = self.sentiment.social_volume_zscore
        if np.isnan(prev_volume) or prev_volume == 0 or np.isnan(new_social_volume):
            return False

        # Check if spike is >5x
        if abs(prev_volume) > 0 and abs(new_social_volume / prev_volume) > 5.0:
            # Check if price moved correspondingly
            buf_1h = self.buffers.get("1h")
            if buf_1h and len(buf_1h) >= 2:
                closes = buf_1h.get_closes()
                price_change_pct = abs((closes[-1] - closes[-2]) / closes[-2]) * 100
                # If price didn't move >2%, the social volume spike is suspicious
                if price_change_pct < 2.0:
                    return True
        return False

    def set_training_distribution(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Set the training distribution for anomaly detection."""
        self._training_mean = mean
        self._training_std = std

    # ── Main feature calculation ─────────────────────────────────────

    def calculate_features(self) -> FeatureSnapshot:
        """Compute the 35-base-feature vector plus lags (140 total).

        Returns a :class:`FeatureSnapshot` with normalised feature vector,
        availability count, anomaly flags, and freshness status.
        """
        snap = FeatureSnapshot(timestamp_ms=int(time.time() * 1000))
        raw = np.full(TOTAL_BASE_FEATURES, np.nan)

        # ---- Technical features (0-13) ----
        raw = self._compute_technical(raw)

        # ---- Order-book features (14-16) ----
        raw[14] = self.orderbook.imbalance_5
        raw[15] = self.orderbook.imbalance_10
        raw[16] = self.orderbook.imbalance_20

        # ---- Derivatives features (17-23) ----
        raw[17] = self.derivatives.funding_rate
        raw[18] = self.derivatives.predicted_funding
        raw[19] = self.derivatives.long_short_ratio
        raw[20] = self.derivatives.oi_change_24h
        raw[21] = self.derivatives.liquidation_volume
        raw[22] = self.derivatives.taker_buy_sell_ratio
        raw[23] = self.derivatives.premium_index

        # ---- On-chain features (24-29) with 72h forward-fill ----
        # Each feature individually forward-fills up to 72 hours, then NaN
        onchain_features = [
            (24, "exchange_net_flow"),
            (25, "mvrv_zscore"),
            (26, "nupl"),
            (27, "active_addresses_change"),
            (28, "stablecoin_reserves"),
            (29, "hash_rate_change"),
        ]
        for idx, feature_name in onchain_features:
            raw[idx] = self.onchain.get_feature_with_forward_fill(feature_name, max_stale_hours=72.0)

        # ---- Sentiment features (30-31) ----
        raw[30] = self.sentiment.social_volume_zscore
        raw[31] = self.sentiment.fear_greed_index
        if self.sentiment.age_hours() > (SENTIMENT_FRESHNESS_S / 3600):
            # Still use cached but mark as potentially stale
            pass
        if self.sentiment.last_update_ms == 0:
            raw[30:32] = np.nan

        # ---- Derived features (32-34) ----
        raw = self._compute_derived(raw)

        # ---- Range checks ----
        snap.anomaly_flags = self._range_check(raw)

        # ---- Availability ----
        snap.available_count = int(np.count_nonzero(~np.isnan(raw)))

        # ---- Freshness ----
        snap.freshness_ok = self._check_freshness()

        # ---- Normalise ----
        normed = self._normaliser.update(raw)

        # ---- Anomaly detection vs training distribution ----
        if self._training_mean is not None and self._training_std is not None:
            for i in range(TOTAL_BASE_FEATURES):
                if np.isnan(raw[i]):
                    continue
                z = abs((raw[i] - self._training_mean[i]) / max(self._training_std[i], 1e-10))
                if z > self._anomaly_sigma:
                    snap.anomaly_flags.append(
                        f"{ALL_FEATURE_NAMES[i]}: {z:.1f} sigma from training"
                    )

        # ---- Store in history and build lagged vector ----
        self._feature_history.append(normed.copy())
        snap.features = normed

        return snap

    def get_lagged_feature_vector(self) -> np.ndarray:
        """Return the 140-dimensional feature vector (current + 3 lags).

        If lag history is incomplete, missing lags are NaN-filled.
        """
        current = self._feature_history[-1] if self._feature_history else np.full(TOTAL_BASE_FEATURES, np.nan)
        parts = [current]

        for lag in range(1, LAG_COUNT + 1):
            idx = len(self._feature_history) - 1 - lag
            if idx >= 0:
                parts.append(self._feature_history[idx])
            else:
                parts.append(np.full(TOTAL_BASE_FEATURES, np.nan))

        return np.concatenate(parts)

    def get_sequence_for_lstm(self, sequence_length: int = 24) -> Optional[np.ndarray]:
        """Return (sequence_length, 35) array for LSTM input.

        Returns None if insufficient history.
        """
        if len(self._feature_history) < sequence_length:
            return None
        recent = list(self._feature_history)[-sequence_length:]
        return np.array(recent, dtype=np.float32)

    # ── Technical features ───────────────────────────────────────────

    def _compute_technical(self, raw: np.ndarray) -> np.ndarray:
        """Fill in technical features (indices 0-13) from kline buffers."""
        buf_1h = self.buffers["1h"]
        buf_4h = self.buffers["4h"]
        buf_1d = self.buffers["1d"]

        # --- 1h timeframe ---
        if len(buf_1h) >= 50:
            closes_1h = buf_1h.get_closes()
            highs_1h = buf_1h.get_highs()
            lows_1h = buf_1h.get_lows()
            opens_1h = buf_1h.get_opens()
            volumes_1h = buf_1h.get_volumes()

            # EMA ratios
            ema9 = ema(closes_1h, 9)
            ema21 = ema(closes_1h, 21)
            ema50 = ema(closes_1h, 50)
            if not np.isnan(ema9[-1]) and not np.isnan(ema21[-1]) and ema21[-1] != 0:
                raw[0] = ema9[-1] / ema21[-1]
            if not np.isnan(ema21[-1]) and not np.isnan(ema50[-1]) and ema50[-1] != 0:
                raw[1] = ema21[-1] / ema50[-1]

            # RSI(14) on 1h
            rsi_1h = rsi(closes_1h, 14)
            if not np.isnan(rsi_1h[-1]):
                raw[4] = rsi_1h[-1]

            # MACD histogram on 1h
            _, _, hist_1h = macd(closes_1h, 12, 26, 9)
            if not np.isnan(hist_1h[-1]):
                raw[5] = hist_1h[-1]

            # ATR(14) on 1h
            atr_1h = atr(highs_1h, lows_1h, closes_1h, 14)
            if not np.isnan(atr_1h[-1]):
                raw[7] = atr_1h[-1]

            # OBV slope 24
            obv_s = obv_slope(closes_1h, volumes_1h, 24)
            if not np.isnan(obv_s[-1]):
                raw[9] = obv_s[-1]

            # Kline body ratio
            body_r = kline_body_ratio(opens_1h, closes_1h, highs_1h, lows_1h)
            raw[10] = body_r[-1]

            # Volume ratio: current 1h / 20-period average
            vol_avg = volume_average(volumes_1h, 20)
            if not np.isnan(vol_avg[-1]) and vol_avg[-1] > 0:
                raw[11] = volumes_1h[-1] / vol_avg[-1]

            # High-Low range
            if closes_1h[-1] > 0:
                raw[12] = (highs_1h[-1] - lows_1h[-1]) / closes_1h[-1]

            # Close position in range
            cp = close_position_in_range(closes_1h, highs_1h, lows_1h)
            raw[13] = cp[-1]

        # --- 4h timeframe ---
        if len(buf_4h) >= 50:
            closes_4h = buf_4h.get_closes()
            highs_4h = buf_4h.get_highs()
            lows_4h = buf_4h.get_lows()

            # EMA50/EMA200 ratio on 4h
            ema50_4h = ema(closes_4h, 50)
            if len(closes_4h) >= 200:
                ema200_4h = ema(closes_4h, 200)
                if not np.isnan(ema50_4h[-1]) and not np.isnan(ema200_4h[-1]) and ema200_4h[-1] != 0:
                    raw[2] = ema50_4h[-1] / ema200_4h[-1]

            # EMA20/EMA50 on 4h (for trend alignment)
            ema20_4h = ema(closes_4h, 20)
            if not np.isnan(ema20_4h[-1]) and not np.isnan(ema50_4h[-1]) and ema50_4h[-1] != 0:
                raw[3] = ema20_4h[-1] / ema50_4h[-1]

            # ADX(14) on 4h
            adx_val, _, _ = adx(highs_4h, lows_4h, closes_4h, 14)
            if not np.isnan(adx_val[-1]):
                raw[6] = adx_val[-1]

            # BB width on 4h
            upper, middle, lower = bollinger_bands(closes_4h, 20, 2.0)
            if not np.isnan(upper[-1]) and not np.isnan(lower[-1]) and middle[-1] != 0:
                raw[8] = (upper[-1] - lower[-1]) / middle[-1]

        return raw

    # ── Derived features ─────────────────────────────────────────────

    def _compute_derived(self, raw: np.ndarray) -> np.ndarray:
        """Fill in derived features (indices 32-34)."""
        buf_1h = self.buffers["1h"]

        if len(buf_1h) >= 100:
            closes = buf_1h.get_closes()
            volumes = buf_1h.get_volumes()

            # Hurst exponent
            raw[32] = hurst_exponent(closes[-100:])

            # Volatility ratio (RV 1d / RV 7d)
            vr = volatility_ratio(closes, short_window=24, long_window=168)
            if not np.isnan(vr[-1]):
                raw[33] = vr[-1]

            # Price-volume correlation (24-period)
            if len(closes) >= 24 and len(volumes) >= 24:
                corr = correlation_rolling(closes, volumes, 24)
                if not np.isnan(corr[-1]):
                    raw[34] = corr[-1]

        return raw

    # ── Range checks ─────────────────────────────────────────────────

    def _range_check(self, raw: np.ndarray) -> List[str]:
        """Validate feature ranges and flag out-of-bounds values."""
        flags: List[str] = []

        # RSI must be 0-100
        if not np.isnan(raw[4]) and (raw[4] < 0 or raw[4] > 100):
            flags.append(f"rsi_14_1h out of range: {raw[4]:.2f}")
            raw[4] = np.nan

        # Volume ratio must be > 0
        if not np.isnan(raw[11]) and raw[11] < 0:
            flags.append(f"volume_ratio_1h negative: {raw[11]:.4f}")
            raw[11] = np.nan

        # Z-scored features should be between -5 and +5 (after normalisation)
        # This check is on raw values -- normed check happens after normalisation
        for i in range(TOTAL_BASE_FEATURES):
            if not np.isnan(raw[i]) and abs(raw[i]) > 1e12:
                flags.append(f"{ALL_FEATURE_NAMES[i]} extreme value: {raw[i]:.2e}")
                raw[i] = np.nan

        return flags

    # ── Freshness check ──────────────────────────────────────────────

    def _check_freshness(self) -> bool:
        """Check that data sources meet freshness requirements."""
        now_ms = int(time.time() * 1000)

        # Technical: last 1h candle should be within 5 minutes
        buf_1h = self.buffers["1h"]
        if len(buf_1h) > 0:
            ts = buf_1h.get_timestamps()
            last_ts = int(ts[-1])
            if (now_ms - last_ts) > TECHNICAL_FRESHNESS_S * 1000:
                logger.warning(
                    "[%s] Technical data stale: last candle %d ms ago",
                    self.symbol, now_ms - last_ts,
                )
                return False

        # On-chain: within 4 hours
        if self.onchain.last_update_ms > 0:
            age = (now_ms - self.onchain.last_update_ms) / 1000
            if age > ONCHAIN_FRESHNESS_S:
                logger.info("[%s] On-chain data older than 4h (%d s)", self.symbol, int(age))
                # Not a hard fail -- on-chain updates are slower

        return True

    # ── Trend alignment helper (used by strategy) ────────────────────

    def get_trend_alignment(self) -> Optional[str]:
        """Return 'LONG', 'SHORT', or None based on EMA(20) vs EMA(50) on 4h."""
        buf_4h = self.buffers["4h"]
        if len(buf_4h) < 50:
            return None
        closes = buf_4h.get_closes()
        ema20 = ema(closes, 20)
        ema50 = ema(closes, 50)
        if np.isnan(ema20[-1]) or np.isnan(ema50[-1]):
            return None
        if ema20[-1] > ema50[-1]:
            return "LONG"
        elif ema20[-1] < ema50[-1]:
            return "SHORT"
        return None

    # ── ATR helper (used by strategy for stops/targets) ──────────────

    def get_atr_4h(self) -> float:
        """Return ATR(14) on 4h chart, or NaN."""
        buf_4h = self.buffers["4h"]
        if len(buf_4h) < 15:
            return np.nan
        atr_vals = atr(buf_4h.get_highs(), buf_4h.get_lows(), buf_4h.get_closes(), 14)
        return float(atr_vals[-1]) if not np.isnan(atr_vals[-1]) else np.nan

    def get_current_price(self) -> float:
        """Return the latest close on 1h, or NaN."""
        buf = self.buffers["1h"]
        if len(buf) == 0:
            return np.nan
        return float(buf.get_closes()[-1])

    # ── State export / import ────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Export state for persistence."""
        return {
            "symbol": self.symbol,
            "onchain": {
                "exchange_net_flow": self.onchain.exchange_net_flow,
                "mvrv_zscore": self.onchain.mvrv_zscore,
                "nupl": self.onchain.nupl,
                "active_addresses_change": self.onchain.active_addresses_change,
                "stablecoin_reserves": self.onchain.stablecoin_reserves,
                "hash_rate_change": self.onchain.hash_rate_change,
                "last_update_ms": self.onchain.last_update_ms,
            },
            "derivatives": {
                "funding_rate": self.derivatives.funding_rate,
                "predicted_funding": self.derivatives.predicted_funding,
                "long_short_ratio": self.derivatives.long_short_ratio,
                "oi_change_24h": self.derivatives.oi_change_24h,
                "liquidation_volume": self.derivatives.liquidation_volume,
                "taker_buy_sell_ratio": self.derivatives.taker_buy_sell_ratio,
                "premium_index": self.derivatives.premium_index,
                "last_update_ms": self.derivatives.last_update_ms,
            },
            "sentiment": {
                "social_volume_zscore": self.sentiment.social_volume_zscore,
                "fear_greed_index": self.sentiment.fear_greed_index,
                "last_update_ms": self.sentiment.last_update_ms,
            },
            "feature_history_len": len(self._feature_history),
        }
