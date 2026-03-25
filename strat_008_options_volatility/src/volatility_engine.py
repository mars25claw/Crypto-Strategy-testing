"""Volatility calculation engine for STRAT-008.

Computes realized volatility (RV), implied volatility (IV — from Deribit or
synthetic estimation), IV/RV ratio, IV term structure analysis, volatility
surface interpolation, and dynamic synthetic IV calibration.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

# Minutes per window for RV calculation
_RV_WINDOWS = {
    "1d": 1440,       # 24 hours
    "7d": 1440 * 7,   # 7 days
    "30d": 1440 * 30,  # 30 days
}

# Annualization factor: sqrt(minutes per year)
# There are 525,600 minutes/year; using sqrt(1440) for 1-min returns -> annualized
_ANNUALIZATION = math.sqrt(1440)  # ~37.95 -> multiply stdev of 1min returns


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VolRegime:
    """Snapshot of the current volatility regime for a single asset."""
    symbol: str
    timestamp: float = 0.0

    # Realized volatility (annualized %)
    rv_1d: float = 0.0
    rv_7d: float = 0.0
    rv_30d: float = 0.0

    # Implied volatility (annualized %)
    iv: float = 0.0
    iv_source: str = "none"  # "deribit", "synthetic", "none"

    # IV/RV ratio (IV / RV_7d)
    iv_rv_ratio: float = 0.0

    # Classification
    regime: str = "unknown"  # "favorable", "strong", "neutral", "unfavorable", "no_data"

    # IV history for spike detection (last 4h of IV readings)
    iv_history: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "rv_1d": round(self.rv_1d, 2),
            "rv_7d": round(self.rv_7d, 2),
            "rv_30d": round(self.rv_30d, 2),
            "iv": round(self.iv, 2),
            "iv_source": self.iv_source,
            "iv_rv_ratio": round(self.iv_rv_ratio, 3),
            "regime": self.regime,
        }


@dataclass
class IVTermStructure:
    """IV across multiple expirations for term-structure analysis."""
    symbol: str
    timestamp: float = 0.0
    # dte -> atm_iv mapping
    points: Dict[int, float] = field(default_factory=dict)
    # Contango (normal) or backwardation (inverted)
    shape: str = "unknown"  # "contango", "backwardation", "flat", "unknown"


@dataclass
class VolSurface:
    """Volatility surface built from Deribit options chain data.

    Stores IV indexed by (strike, days_to_expiry) with bilinear interpolation
    support for any (strike, expiry) pair.
    """
    symbol: str
    timestamp: float = 0.0
    # Sorted unique arrays for grid axes
    strikes: np.ndarray = field(default_factory=lambda: np.array([]))
    expiry_days: np.ndarray = field(default_factory=lambda: np.array([]))
    # 2-D IV grid: shape (len(strikes), len(expiry_days))
    iv_grid: np.ndarray = field(default_factory=lambda: np.array([[]]))
    # Interpolator (built lazily)
    _interpolator: Optional[Any] = field(default=None, repr=False)
    # Anomalies detected during build
    skew_anomalies: List[dict] = field(default_factory=list)
    term_structure_anomalies: List[dict] = field(default_factory=list)

    def is_valid(self) -> bool:
        return (
            len(self.strikes) >= 2
            and len(self.expiry_days) >= 2
            and self.iv_grid.size > 0
        )

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "num_strikes": len(self.strikes),
            "num_expiries": len(self.expiry_days),
            "strike_range": (
                [float(self.strikes[0]), float(self.strikes[-1])]
                if len(self.strikes) > 0 else []
            ),
            "expiry_range": (
                [float(self.expiry_days[0]), float(self.expiry_days[-1])]
                if len(self.expiry_days) > 0 else []
            ),
            "skew_anomalies": len(self.skew_anomalies),
            "term_anomalies": len(self.term_structure_anomalies),
        }


@dataclass
class SyntheticCalibration:
    """Rolling calibration state for synthetic IV factor."""
    # Deque of (timestamp, actual_iv, synthetic_iv) observations
    observations: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=2000)
    )
    # Current calibrated factor
    calibrated_factor: float = 1.75
    # Last recalculation timestamp
    last_recalc: float = 0.0
    # Rolling window in seconds (30 days)
    window_seconds: float = 30 * 86400

    def to_dict(self) -> dict:
        return {
            "calibrated_factor": round(self.calibrated_factor, 4),
            "num_observations": len(self.observations),
            "last_recalc": self.last_recalc,
        }


# ---------------------------------------------------------------------------
# VolatilityEngine
# ---------------------------------------------------------------------------

class VolatilityEngine:
    """Computes and maintains volatility metrics for BTC and ETH.

    Parameters
    ----------
    config : dict
        Strategy parameters from config.yaml.
    """

    def __init__(self, config: dict) -> None:
        self._config = config

        # IV/RV thresholds
        self._iv_rv_favorable = config.get("iv_rv_favorable", 1.3)
        self._iv_rv_strong = config.get("iv_rv_strong", 1.5)
        self._iv_rv_neutral_low = config.get("iv_rv_neutral_low", 1.1)
        self._iv_rv_no_sell = config.get("iv_rv_no_sell", 1.0)

        # Synthetic IV factor
        self._synthetic_factor = config.get("iv_synthetic_factor", 1.75)

        # Circuit breaker thresholds
        self._iv_spike_pp = config.get("iv_spike_threshold_pp", 20.0)
        self._iv_spike_window_h = config.get("iv_spike_window_hours", 4)
        self._flash_crash_pct = config.get("flash_crash_pct", 8.0)
        self._flash_crash_window_min = config.get("flash_crash_window_minutes", 60)

        # Per-symbol data buffers: symbol -> deque of (timestamp_ms, close_price)
        self._minute_returns: Dict[str, Deque[Tuple[int, float]]] = {}
        # Max 30 days of 1-minute data
        self._max_1m_candles = 1440 * 30 + 100

        # Current regimes
        self._regimes: Dict[str, VolRegime] = {}

        # IV term structure (Deribit mode)
        self._term_structures: Dict[str, IVTermStructure] = {}

        # Hourly price snapshots for flash crash detection
        self._hourly_prices: Dict[str, Deque[Tuple[float, float]]] = {}

        # Deribit ATM IV cache
        self._deribit_atm_iv: Dict[str, float] = {}

        # Premium/basis data for synthetic IV
        self._futures_premium: Dict[str, float] = {}  # symbol -> annual basis %

        # Volatility surface (Deribit mode): symbol -> VolSurface
        self._vol_surfaces: Dict[str, VolSurface] = {}

        # Synthetic IV calibration: symbol -> SyntheticCalibration
        self._calibrations: Dict[str, SyntheticCalibration] = {}
        self._default_synthetic_factor = self._synthetic_factor

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_1m_candle(self, symbol: str, timestamp_ms: int, close: float) -> None:
        """Add a 1-minute candle close for RV calculation."""
        if symbol not in self._minute_returns:
            self._minute_returns[symbol] = deque(maxlen=self._max_1m_candles)

        self._minute_returns[symbol].append((timestamp_ms, close))

    def add_1m_candles_bulk(
        self, symbol: str, candles: List[Tuple[int, float]]
    ) -> None:
        """Bulk-add historical 1-minute candle closes (for warm-up)."""
        if symbol not in self._minute_returns:
            self._minute_returns[symbol] = deque(maxlen=self._max_1m_candles)

        buf = self._minute_returns[symbol]
        for ts, close in candles:
            buf.append((ts, close))

        logger.info(
            "Bulk loaded %d 1m candles for %s (total buffer: %d)",
            len(candles), symbol, len(buf),
        )

    def update_futures_premium(
        self, symbol: str, spot_price: float, futures_price: float
    ) -> None:
        """Update the futures basis for synthetic IV estimation.

        Annualized basis = |futures/spot - 1| * (365/days_to_funding) * 100
        For perpetual futures, we estimate using 8h funding periods.
        """
        if spot_price <= 0:
            return

        basis_pct = abs(futures_price / spot_price - 1.0)
        # Perpetual funding is every 8 hours = 1/3 day
        # Annualize: basis * 365 * 3 (3 fundings per day)
        annual_basis_pct = basis_pct * 365.0 * 3.0 * 100.0
        self._futures_premium[symbol] = annual_basis_pct

    def update_deribit_atm_iv(self, symbol: str, atm_iv: float) -> None:
        """Update ATM implied volatility from Deribit (in % e.g. 65.0)."""
        self._deribit_atm_iv[symbol] = atm_iv
        logger.debug("Deribit ATM IV for %s: %.1f%%", symbol, atm_iv)

    def update_hourly_price(self, symbol: str, price: float) -> None:
        """Record an hourly price for flash crash detection."""
        if symbol not in self._hourly_prices:
            self._hourly_prices[symbol] = deque(maxlen=24)
        self._hourly_prices[symbol].append((time.time(), price))

    # ------------------------------------------------------------------
    # Realized Volatility
    # ------------------------------------------------------------------

    def calculate_rv(self, symbol: str, window: str = "7d") -> float:
        """Calculate annualized realized volatility for a given window.

        RV = stdev(1-min log returns) * sqrt(1440) * 100

        Parameters
        ----------
        symbol : str
            Trading pair symbol.
        window : str
            One of "1d", "7d", "30d".

        Returns
        -------
        float
            Annualized RV in percentage (e.g. 65.0 for 65%).
        """
        buf = self._minute_returns.get(symbol)
        if buf is None or len(buf) < 2:
            return 0.0

        n_minutes = _RV_WINDOWS.get(window, 1440 * 7)
        # Take the last n_minutes + 1 closes (need n+1 for n returns)
        data = list(buf)
        data = data[-(n_minutes + 1):]

        if len(data) < 10:
            return 0.0

        closes = np.array([c for _, c in data], dtype=np.float64)
        # Filter out zeros/negatives
        valid = closes > 0
        closes = closes[valid]

        if len(closes) < 10:
            return 0.0

        # Log returns
        log_returns = np.diff(np.log(closes))

        # Remove any NaN/inf
        log_returns = log_returns[np.isfinite(log_returns)]
        if len(log_returns) < 5:
            return 0.0

        stdev = float(np.std(log_returns, ddof=1))
        rv = stdev * _ANNUALIZATION * 100.0
        return rv

    # ------------------------------------------------------------------
    # Implied Volatility
    # ------------------------------------------------------------------

    def get_iv(self, symbol: str, deribit_available: bool = False) -> Tuple[float, str]:
        """Get implied volatility from best available source.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. "BTCUSDT").
        deribit_available : bool
            Whether Deribit connection is active.

        Returns
        -------
        (iv, source)
            IV in percentage, and the source string.
        """
        # Prefer Deribit ATM IV
        if deribit_available:
            # Map BTCUSDT -> BTC for Deribit
            base_asset = symbol.replace("USDT", "")
            deribit_iv = self._deribit_atm_iv.get(base_asset, 0.0)
            if deribit_iv > 0:
                # Record calibration observation when both sources available
                raw_synthetic_iv = self._get_raw_synthetic_iv(symbol)
                if raw_synthetic_iv > 0:
                    self.calibrate_synthetic_factor(symbol, deribit_iv, raw_synthetic_iv)
                return deribit_iv, "deribit"

        # Fallback: calibrated synthetic IV from futures basis
        cal = self._calibrations.get(symbol)
        if cal and len(cal.observations) >= 10:
            iv = self.get_calibrated_synthetic_iv(symbol)
            if iv > 0:
                return iv, "synthetic_calibrated"

        return self._estimate_synthetic_iv(symbol)

    def _get_raw_synthetic_iv(self, symbol: str) -> float:
        """Get raw synthetic IV before applying calibration factor.

        Used internally for calibration comparisons.
        """
        annual_basis = self._futures_premium.get(symbol, 0.0)
        if annual_basis <= 0:
            return 0.0
        T = 8.0 / 8760.0
        sqrt_T = math.sqrt(T) if T > 0 else 1.0
        # Use factor=1.0 to get the raw value
        raw_iv = annual_basis / sqrt_T
        return max(10.0, min(300.0, raw_iv))

    def _estimate_synthetic_iv(self, symbol: str) -> Tuple[float, str]:
        """Estimate IV from futures basis (Synthetic Mode).

        IV_synthetic ~ |basis_annual| / sqrt(T) * adjustment_factor
        Where T = time to next funding in years (8h = 8/8760)

        This is a ROUGH estimate flagged as "estimated IV" in logs.
        """
        annual_basis = self._futures_premium.get(symbol, 0.0)
        if annual_basis <= 0:
            return 0.0, "none"

        # T = 8 hours in years
        T = 8.0 / 8760.0
        sqrt_T = math.sqrt(T) if T > 0 else 1.0

        iv_estimate = annual_basis / sqrt_T * self._synthetic_factor

        # Clamp to reasonable range (10%-300%)
        iv_estimate = max(10.0, min(300.0, iv_estimate))

        logger.debug(
            "Synthetic IV for %s: %.1f%% (basis=%.2f%%, factor=%.2f) [ESTIMATED]",
            symbol, iv_estimate, annual_basis, self._synthetic_factor,
        )
        return iv_estimate, "synthetic"

    # ------------------------------------------------------------------
    # Volatility Surface Interpolation (Deribit mode)
    # ------------------------------------------------------------------

    def build_vol_surface(self, options_chain: list) -> VolSurface:
        """Build a volatility surface from Deribit options chain data.

        Constructs a 2-D grid of IV indexed by (strike, days_to_expiry) using
        bilinear interpolation between known IV points. Detects skew and term
        structure anomalies for vol-arb signals.

        Parameters
        ----------
        options_chain : list
            List of option dicts/objects with at minimum:
            strike (float), dte (float), mark_iv (float),
            option_type (str), underlying (str).

        Returns
        -------
        VolSurface
        """
        if not options_chain:
            return VolSurface(symbol="", timestamp=time.time())

        # Determine the symbol from the first item
        first = options_chain[0]
        if hasattr(first, "underlying"):
            symbol = first.underlying
        elif isinstance(first, dict):
            symbol = first.get("underlying", "")
        else:
            symbol = ""

        # Collect valid IV points: (strike, dte, iv)
        points: List[Tuple[float, float, float]] = []
        for opt in options_chain:
            if hasattr(opt, "strike"):
                strike = opt.strike
                dte_val = opt.dte if hasattr(opt, "dte") else 0.0
                iv_val = opt.mark_iv if hasattr(opt, "mark_iv") else 0.0
            elif isinstance(opt, dict):
                strike = opt.get("strike", 0)
                dte_val = opt.get("dte", 0)
                iv_val = opt.get("mark_iv", 0)
            else:
                continue

            if strike > 0 and dte_val > 0 and iv_val > 0:
                points.append((strike, dte_val, iv_val))

        if len(points) < 4:
            logger.warning(
                "Insufficient data for vol surface (%d points)", len(points),
            )
            return VolSurface(symbol=symbol, timestamp=time.time())

        # Build unique sorted axes
        strikes_set = sorted(set(p[0] for p in points))
        expiry_set = sorted(set(p[1] for p in points))

        if len(strikes_set) < 2 or len(expiry_set) < 2:
            logger.warning("Need at least 2 strikes and 2 expiries for surface")
            return VolSurface(symbol=symbol, timestamp=time.time())

        strikes_arr = np.array(strikes_set, dtype=np.float64)
        expiry_arr = np.array(expiry_set, dtype=np.float64)

        # Build IV grid with NaN for missing points
        iv_grid = np.full(
            (len(strikes_set), len(expiry_set)), np.nan, dtype=np.float64,
        )
        strike_idx = {s: i for i, s in enumerate(strikes_set)}
        expiry_idx = {e: i for i, e in enumerate(expiry_set)}

        for strike, dte_val, iv_val in points:
            si = strike_idx[strike]
            ei = expiry_idx[dte_val]
            # Average if multiple readings for same (strike, dte)
            if np.isnan(iv_grid[si, ei]):
                iv_grid[si, ei] = iv_val
            else:
                iv_grid[si, ei] = (iv_grid[si, ei] + iv_val) / 2.0

        # Fill NaN with nearest-neighbor interpolation for grid completeness
        from scipy.interpolate import NearestNDInterpolator
        known_mask = ~np.isnan(iv_grid)
        if not known_mask.all():
            known_indices = np.argwhere(known_mask)
            known_values = iv_grid[known_mask]
            if len(known_values) >= 1:
                nn_interp = NearestNDInterpolator(known_indices, known_values)
                nan_indices = np.argwhere(~known_mask)
                if len(nan_indices) > 0:
                    iv_grid[~known_mask] = nn_interp(nan_indices)

        # Build bilinear interpolator
        try:
            interpolator = RegularGridInterpolator(
                (strikes_arr, expiry_arr),
                iv_grid,
                method="linear",
                bounds_error=False,
                fill_value=None,  # Extrapolate with nearest
            )
        except Exception as exc:
            logger.warning("Failed to build vol surface interpolator: %s", exc)
            interpolator = None

        # Detect anomalies
        skew_anomalies = self._detect_skew_anomalies(
            strikes_arr, expiry_arr, iv_grid,
        )
        term_anomalies = self._detect_term_structure_anomalies(
            strikes_arr, expiry_arr, iv_grid,
        )

        surface = VolSurface(
            symbol=symbol,
            timestamp=time.time(),
            strikes=strikes_arr,
            expiry_days=expiry_arr,
            iv_grid=iv_grid,
            _interpolator=interpolator,
            skew_anomalies=skew_anomalies,
            term_structure_anomalies=term_anomalies,
        )

        self._vol_surfaces[symbol] = surface

        logger.info(
            "Vol surface built for %s: %d strikes x %d expiries, "
            "%d skew anomalies, %d term anomalies",
            symbol, len(strikes_arr), len(expiry_arr),
            len(skew_anomalies), len(term_anomalies),
        )

        return surface

    def interpolate_iv(self, symbol: str, strike: float, expiry_days: float) -> float:
        """Interpolate IV for any (strike, expiry_days) pair using bilinear interpolation.

        Parameters
        ----------
        symbol : str
            Base asset (e.g. "BTC").
        strike : float
            Option strike price.
        expiry_days : float
            Days to expiration.

        Returns
        -------
        float
            Interpolated IV in percentage, or 0.0 if surface unavailable.
        """
        surface = self._vol_surfaces.get(symbol)
        if not surface or not surface.is_valid() or surface._interpolator is None:
            return 0.0

        try:
            point = np.array([[strike, expiry_days]])
            result = surface._interpolator(point)
            iv = float(result[0])
            # Clamp to reasonable range
            return max(5.0, min(500.0, iv))
        except Exception as exc:
            logger.debug("IV interpolation failed for %s K=%.0f DTE=%.1f: %s",
                         symbol, strike, expiry_days, exc)
            return 0.0

    def get_vol_surface(self, symbol: str) -> Optional[VolSurface]:
        """Return the latest vol surface for a symbol."""
        return self._vol_surfaces.get(symbol)

    def _detect_skew_anomalies(
        self,
        strikes: np.ndarray,
        expiry_days: np.ndarray,
        iv_grid: np.ndarray,
    ) -> List[dict]:
        """Detect skew anomalies: non-monotonic smile or extreme skew.

        Normal crypto skew: puts have higher IV than calls (negative skew).
        Anomaly: reversed skew or extreme deviations from the smile pattern.
        """
        anomalies: List[dict] = []

        for j, dte in enumerate(expiry_days):
            ivs = iv_grid[:, j]
            valid = ~np.isnan(ivs)
            if valid.sum() < 3:
                continue

            valid_strikes = strikes[valid]
            valid_ivs = ivs[valid]

            # Find approximate ATM index (middle of range)
            mid_strike = (valid_strikes[0] + valid_strikes[-1]) / 2.0
            atm_idx = int(np.argmin(np.abs(valid_strikes - mid_strike)))

            # Check for unusual skew reversal
            if atm_idx > 0 and atm_idx < len(valid_ivs) - 1:
                left_avg = np.mean(valid_ivs[:atm_idx])
                right_avg = np.mean(valid_ivs[atm_idx + 1:])
                atm_iv = valid_ivs[atm_idx]

                # Normally puts (left/lower strikes) have higher IV
                # Anomaly: if calls have significantly higher IV than puts
                if right_avg > left_avg * 1.15 and right_avg > atm_iv * 1.10:
                    anomalies.append({
                        "type": "reversed_skew",
                        "dte": float(dte),
                        "put_avg_iv": round(float(left_avg), 2),
                        "call_avg_iv": round(float(right_avg), 2),
                        "atm_iv": round(float(atm_iv), 2),
                        "signal": "buy_puts_sell_calls",
                    })

                # Extreme skew: put IV > 2x ATM IV
                if left_avg > atm_iv * 2.0:
                    anomalies.append({
                        "type": "extreme_put_skew",
                        "dte": float(dte),
                        "put_avg_iv": round(float(left_avg), 2),
                        "atm_iv": round(float(atm_iv), 2),
                        "signal": "sell_otm_puts",
                    })

        return anomalies

    def _detect_term_structure_anomalies(
        self,
        strikes: np.ndarray,
        expiry_days: np.ndarray,
        iv_grid: np.ndarray,
    ) -> List[dict]:
        """Detect term structure anomalies across expirations.

        Normal: longer-dated options have higher IV (contango).
        Anomaly: near-term IV significantly higher than longer-dated (backwardation).
        """
        anomalies: List[dict] = []

        if len(expiry_days) < 2:
            return anomalies

        # Compare ATM IV across expirations
        mid_idx = len(strikes) // 2
        atm_ivs = iv_grid[mid_idx, :]
        valid = ~np.isnan(atm_ivs)

        if valid.sum() < 2:
            return anomalies

        valid_dtes = expiry_days[valid]
        valid_atm = atm_ivs[valid]

        # Check for inversions
        for i in range(len(valid_atm) - 1):
            near_iv = valid_atm[i]
            far_iv = valid_atm[i + 1]
            near_dte = valid_dtes[i]
            far_dte = valid_dtes[i + 1]

            # Strong inversion: near-term IV > far-term by 15%+ is an anomaly
            if near_iv > far_iv * 1.15:
                anomalies.append({
                    "type": "term_inversion",
                    "near_dte": float(near_dte),
                    "far_dte": float(far_dte),
                    "near_iv": round(float(near_iv), 2),
                    "far_iv": round(float(far_iv), 2),
                    "spread_pct": round(
                        (near_iv - far_iv) / far_iv * 100, 1
                    ),
                    "signal": "sell_near_buy_far",
                })

        return anomalies

    # ------------------------------------------------------------------
    # Synthetic IV Calibration
    # ------------------------------------------------------------------

    def calibrate_synthetic_factor(
        self, symbol: str, actual_iv: float, synthetic_iv: float,
    ) -> None:
        """Record an observation for dynamic calibration of the synthetic IV factor.

        Compares actual IV (from Deribit ATM) to raw synthetic IV (before factor
        application) to derive the optimal calibration factor over a rolling
        30-day window.

        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g. "BTCUSDT" or "BTC").
        actual_iv : float
            Actual ATM IV from Deribit (percentage).
        synthetic_iv : float
            Raw synthetic IV before factor application (percentage).
        """
        if actual_iv <= 0 or synthetic_iv <= 0:
            return

        if symbol not in self._calibrations:
            self._calibrations[symbol] = SyntheticCalibration(
                calibrated_factor=self._default_synthetic_factor,
            )

        cal = self._calibrations[symbol]
        cal.observations.append((time.time(), actual_iv, synthetic_iv))

        # Recalculate factor every hour
        if time.time() - cal.last_recalc < 3600:
            return

        self._recalculate_calibration(symbol)

    def _recalculate_calibration(self, symbol: str) -> None:
        """Recalculate the calibration factor from rolling 30-day observations."""
        cal = self._calibrations.get(symbol)
        if not cal:
            return

        cutoff = time.time() - cal.window_seconds
        recent = [(t, a, s) for t, a, s in cal.observations if t >= cutoff]

        if len(recent) < 10:
            logger.debug(
                "Insufficient calibration data for %s (%d obs)", symbol, len(recent),
            )
            return

        # Optimal factor = median(actual_iv / synthetic_iv)
        ratios = []
        for _, actual, synthetic in recent:
            if synthetic > 0:
                ratios.append(actual / synthetic)

        if not ratios:
            return

        new_factor = float(np.median(ratios))
        # Clamp to reasonable range
        new_factor = max(0.5, min(5.0, new_factor))

        old_factor = cal.calibrated_factor
        cal.calibrated_factor = new_factor
        cal.last_recalc = time.time()

        # Update the engine's synthetic factor for this symbol
        self._synthetic_factor = new_factor

        logger.info(
            "Synthetic IV calibration for %s: factor %.4f -> %.4f "
            "(%d observations over 30d)",
            symbol, old_factor, new_factor, len(recent),
        )

    def get_calibrated_synthetic_iv(self, symbol: str) -> float:
        """Get the currently calibrated synthetic IV for a symbol.

        Returns
        -------
        float
            Calibrated synthetic IV in percentage, or 0.0 if unavailable.
        """
        # Use the calibrated factor for the specific symbol if available
        cal = self._calibrations.get(symbol)
        if cal:
            old_factor = self._synthetic_factor
            self._synthetic_factor = cal.calibrated_factor
            iv, source = self._estimate_synthetic_iv(symbol)
            self._synthetic_factor = old_factor
            return iv

        iv, source = self._estimate_synthetic_iv(symbol)
        return iv

    def get_calibration_status(self, symbol: str) -> dict:
        """Return calibration status for monitoring."""
        cal = self._calibrations.get(symbol)
        if not cal:
            return {
                "calibrated": False,
                "factor": self._default_synthetic_factor,
                "observations": 0,
            }
        return {
            "calibrated": True,
            **cal.to_dict(),
        }

    # ------------------------------------------------------------------
    # Full regime assessment (run hourly)
    # ------------------------------------------------------------------

    def assess_regime(
        self, symbol: str, deribit_available: bool = False
    ) -> VolRegime:
        """Perform full volatility regime assessment for a symbol.

        This is the hourly assessment from Section 3.1:
        1. Calculate RV (1d, 7d, 30d)
        2. Get IV (Deribit or synthetic)
        3. Compute IV/RV ratio
        4. Classify regime

        Returns
        -------
        VolRegime
            Complete volatility regime snapshot.
        """
        regime = VolRegime(symbol=symbol, timestamp=time.time())

        # Step 1: Realized Volatility
        regime.rv_1d = self.calculate_rv(symbol, "1d")
        regime.rv_7d = self.calculate_rv(symbol, "7d")
        regime.rv_30d = self.calculate_rv(symbol, "30d")

        # Step 2: Implied Volatility
        iv_val, iv_source = self.get_iv(symbol, deribit_available)
        regime.iv = iv_val
        regime.iv_source = iv_source

        # Step 3: IV/RV Ratio
        if regime.rv_7d > 0 and regime.iv > 0:
            regime.iv_rv_ratio = regime.iv / regime.rv_7d
        else:
            regime.iv_rv_ratio = 0.0

        # Step 4: Classify
        ratio = regime.iv_rv_ratio
        if ratio <= 0 or regime.iv <= 0:
            regime.regime = "no_data"
        elif ratio > self._iv_rv_strong:
            regime.regime = "strong"
        elif ratio > self._iv_rv_favorable:
            regime.regime = "favorable"
        elif ratio >= self._iv_rv_neutral_low:
            regime.regime = "neutral"
        elif ratio >= self._iv_rv_no_sell:
            regime.regime = "unfavorable"
        else:
            regime.regime = "no_sell"

        # Record IV history for spike detection
        regime.iv_history = list(self._get_iv_history(symbol))
        self._record_iv(symbol, regime.iv)

        # Store
        self._regimes[symbol] = regime

        logger.info(
            "Vol regime %s: RV(1d=%.1f 7d=%.1f 30d=%.1f) IV=%.1f(%s) "
            "ratio=%.2f regime=%s",
            symbol, regime.rv_1d, regime.rv_7d, regime.rv_30d,
            regime.iv, regime.iv_source, regime.iv_rv_ratio, regime.regime,
        )

        return regime

    def get_regime(self, symbol: str) -> Optional[VolRegime]:
        """Return the last assessed regime for a symbol."""
        return self._regimes.get(symbol)

    # ------------------------------------------------------------------
    # IV history and spike detection
    # ------------------------------------------------------------------

    # Per-symbol IV readings: symbol -> deque of (timestamp, iv)
    _iv_readings: Dict[str, Deque[Tuple[float, float]]] = {}

    def _record_iv(self, symbol: str, iv: float) -> None:
        """Record an IV reading for spike detection."""
        if symbol not in self._iv_readings:
            self._iv_readings[symbol] = deque(maxlen=500)
        self._iv_readings[symbol].append((time.time(), iv))

    def _get_iv_history(self, symbol: str) -> List[Tuple[float, float]]:
        """Get IV readings within the spike detection window."""
        readings = self._iv_readings.get(symbol, deque())
        window_s = self._iv_spike_window_h * 3600
        cutoff = time.time() - window_s
        return [(t, iv) for t, iv in readings if t >= cutoff]

    def check_iv_spike(self, symbol: str) -> Tuple[bool, float]:
        """Check if IV has spiked by more than threshold in the window.

        Returns
        -------
        (spike_detected, spike_magnitude_pp)
        """
        history = self._get_iv_history(symbol)
        if len(history) < 2:
            return False, 0.0

        oldest_iv = history[0][1]
        newest_iv = history[-1][1]
        spike = newest_iv - oldest_iv

        if spike > self._iv_spike_pp:
            logger.warning(
                "IV SPIKE detected for %s: %.1f -> %.1f (%.1f pp in %dh)",
                symbol, oldest_iv, newest_iv, spike, self._iv_spike_window_h,
            )
            return True, spike

        return False, spike

    def check_flash_crash(self, symbol: str) -> Tuple[bool, float]:
        """Check if price has dropped more than threshold in 1 hour.

        Returns
        -------
        (crash_detected, drop_pct)
        """
        prices = self._hourly_prices.get(symbol, deque())
        if len(prices) < 2:
            return False, 0.0

        cutoff = time.time() - self._flash_crash_window_min * 60
        window_prices = [(t, p) for t, p in prices if t >= cutoff]

        if len(window_prices) < 2:
            return False, 0.0

        highest = max(p for _, p in window_prices)
        current = window_prices[-1][1]

        if highest <= 0:
            return False, 0.0

        drop_pct = (highest - current) / highest * 100.0

        if drop_pct >= self._flash_crash_pct:
            logger.warning(
                "FLASH CRASH detected for %s: %.2f%% drop in %d min "
                "(high=%.2f current=%.2f)",
                symbol, drop_pct, self._flash_crash_window_min,
                highest, current,
            )
            return True, drop_pct

        return False, drop_pct

    # ------------------------------------------------------------------
    # IV Term Structure (Deribit mode)
    # ------------------------------------------------------------------

    def update_term_structure(
        self, symbol: str, points: Dict[int, float]
    ) -> IVTermStructure:
        """Update the IV term structure from Deribit data.

        Parameters
        ----------
        symbol : str
            Base asset (e.g. "BTC").
        points : dict
            Mapping of days-to-expiry -> ATM IV (percentage).
        """
        ts = IVTermStructure(
            symbol=symbol,
            timestamp=time.time(),
            points=dict(points),
        )

        if len(points) >= 2:
            sorted_pts = sorted(points.items())
            near_iv = sorted_pts[0][1]
            far_iv = sorted_pts[-1][1]

            if far_iv > near_iv * 1.05:
                ts.shape = "contango"
            elif near_iv > far_iv * 1.05:
                ts.shape = "backwardation"
            else:
                ts.shape = "flat"

        self._term_structures[symbol] = ts
        return ts

    def get_term_structure(self, symbol: str) -> Optional[IVTermStructure]:
        """Return the latest IV term structure."""
        return self._term_structures.get(symbol)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_all_regimes(self) -> Dict[str, dict]:
        """Return all current regimes as serializable dicts."""
        return {sym: r.to_dict() for sym, r in self._regimes.items()}

    def check_rv_exceeds_entry_iv(
        self, symbol: str, entry_iv: float
    ) -> bool:
        """Check if current rolling RV exceeds the IV at entry.

        If RV > IV_at_entry, the trade is losing its edge (Section 4.4).
        """
        rv_rolling = self.calculate_rv(symbol, "7d")
        return rv_rolling > entry_iv

    def get_data_status(self) -> Dict[str, dict]:
        """Return buffer sizes for monitoring."""
        status = {}
        for sym, buf in self._minute_returns.items():
            status[sym] = {
                "1m_candles": len(buf),
                "enough_1d": len(buf) >= 1440,
                "enough_7d": len(buf) >= 1440 * 7,
                "enough_30d": len(buf) >= 1440 * 30,
            }
        # Add vol surface and calibration info
        for sym, surface in self._vol_surfaces.items():
            status.setdefault(sym, {})["vol_surface"] = surface.to_dict()
        for sym, cal in self._calibrations.items():
            status.setdefault(sym, {})["calibration"] = cal.to_dict()
        return status
