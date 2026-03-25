"""Full Engle-Granger cointegration pipeline for pair qualification.

Runs daily at 00:00 UTC on 180-day daily closes for all candidate pairs.
Implements: OLS regression, ADF test, half-life calculation, Hurst exponent
via rescaled-range analysis, and correlation stability checks.

Must complete within 5 minutes for the full pair universe.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PairParameters:
    """Stores all cointegration test results and locked parameters for a pair."""

    asset_a: str
    asset_b: str
    hedge_ratio: float           # beta from OLS
    intercept: float             # alpha from OLS
    spread_mean: float           # mu of the spread
    spread_std: float            # sigma of the spread
    half_life_days: float
    hurst_exponent: float
    adf_p_value: float
    adf_statistic: float
    correlation_stability: float  # pct of days with corr > threshold
    recent_correlation: float    # last 30-day correlation
    qualification_time: float    # timestamp of qualification
    is_marginal: bool = False    # Hurst 0.4-0.5 => reduced size
    is_preferred: bool = False   # ADF p < 0.01
    rank_score: float = 0.0     # composite ranking score
    broken_until: float = 0.0   # blacklisted until this timestamp
    consecutive_stops: int = 0
    post_stop_entry_z: float = 2.0  # default, raised to 2.5 after stop


@dataclass
class CointegrationResult:
    """Result of a full cointegration screening run."""

    qualified_pairs: List[PairParameters]
    rejected_pairs: List[Dict[str, Any]]
    run_duration_s: float
    total_candidates: int
    timestamp: float


# ---------------------------------------------------------------------------
# Cointegration Engine
# ---------------------------------------------------------------------------

class CointegrationEngine:
    """Runs the full Engle-Granger cointegration pipeline.

    Parameters
    ----------
    config : dict
        Strategy parameters from config.yaml strategy_params section.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._cfg = config

        # Thresholds
        self._lookback_days = config.get("lookback_days", 180)
        self._adf_p_threshold = config.get("adf_p_threshold", 0.05)
        self._adf_p_preferred = config.get("adf_p_preferred", 0.01)
        self._hl_min = config.get("half_life_min", 3)
        self._hl_max = config.get("half_life_max", 30)
        self._hurst_reject = config.get("hurst_reject", 0.5)
        self._hurst_ideal = config.get("hurst_ideal", 0.4)
        self._corr_min = config.get("correlation_min", 0.6)
        self._corr_pct_thresh = config.get("correlation_pct_threshold", 80)
        self._corr_recent_reject = config.get("correlation_recent_reject", 0.4)
        self._max_qualified = config.get("max_qualified_pairs", 15)
        self._max_active = config.get("max_active_pairs", 10)
        self._broken_pair_days = config.get("broken_pair_days", 30)

        # State
        self._qualified_pairs: List[PairParameters] = []
        self._broken_pairs: Dict[Tuple[str, str], float] = {}  # (a,b) -> broken_until ts
        self._last_run: float = 0.0

    # ======================================================================
    #  Public API
    # ======================================================================

    @property
    def qualified_pairs(self) -> List[PairParameters]:
        return list(self._qualified_pairs)

    @property
    def active_pairs(self) -> List[PairParameters]:
        """Top max_active pairs by rank score."""
        return self._qualified_pairs[:self._max_active]

    @property
    def reserve_pairs(self) -> List[PairParameters]:
        """Reserve pairs beyond the active set."""
        return self._qualified_pairs[self._max_active:]

    def get_pair_params(self, asset_a: str, asset_b: str) -> Optional[PairParameters]:
        """Retrieve locked parameters for a specific pair."""
        for p in self._qualified_pairs:
            if (p.asset_a == asset_a and p.asset_b == asset_b) or \
               (p.asset_a == asset_b and p.asset_b == asset_a):
                return p
        return None

    def mark_pair_broken(self, asset_a: str, asset_b: str) -> None:
        """Mark a pair as broken for 30 days (cointegration breakdown)."""
        key = (min(asset_a, asset_b), max(asset_a, asset_b))
        broken_until = time.time() + self._broken_pair_days * 86400
        self._broken_pairs[key] = broken_until
        logger.warning(
            "Pair %s/%s marked as broken until %.0f",
            asset_a, asset_b, broken_until,
        )
        # Remove from qualified
        self._qualified_pairs = [
            p for p in self._qualified_pairs
            if not ((p.asset_a == asset_a and p.asset_b == asset_b) or
                    (p.asset_a == asset_b and p.asset_b == asset_a))
        ]

    def is_pair_broken(self, asset_a: str, asset_b: str) -> bool:
        """Check if a pair is blacklisted."""
        key = (min(asset_a, asset_b), max(asset_a, asset_b))
        until = self._broken_pairs.get(key, 0.0)
        if until > time.time():
            return True
        # Expired, clean up
        self._broken_pairs.pop(key, None)
        return False

    def run_full_screening(
        self,
        daily_closes: Dict[str, List[float]],
        symbols: List[str],
    ) -> CointegrationResult:
        """Run the complete cointegration screening pipeline.

        Parameters
        ----------
        daily_closes : dict
            symbol -> list of daily close prices (180 days, chronological).
        symbols : list
            List of symbol names to screen.

        Returns
        -------
        CointegrationResult
        """
        start = time.time()
        qualified: List[PairParameters] = []
        rejected: List[Dict[str, Any]] = []

        # Generate all pair combinations
        candidates = list(combinations(symbols, 2))
        logger.info(
            "Cointegration screening: %d candidates from %d assets",
            len(candidates), len(symbols),
        )

        for asset_a, asset_b in candidates:
            # Skip broken pairs
            if self.is_pair_broken(asset_a, asset_b):
                rejected.append({
                    "pair": f"{asset_a}/{asset_b}",
                    "reason": "blacklisted (broken)",
                })
                continue

            prices_a = daily_closes.get(asset_a)
            prices_b = daily_closes.get(asset_b)

            if prices_a is None or prices_b is None:
                rejected.append({
                    "pair": f"{asset_a}/{asset_b}",
                    "reason": "missing price data",
                })
                continue

            # Align lengths
            min_len = min(len(prices_a), len(prices_b))
            if min_len < 60:  # Need at least 60 days
                rejected.append({
                    "pair": f"{asset_a}/{asset_b}",
                    "reason": f"insufficient data ({min_len} days)",
                })
                continue

            pa = np.array(prices_a[-min_len:], dtype=np.float64)
            pb = np.array(prices_b[-min_len:], dtype=np.float64)

            # Filter out zero/negative prices
            valid = (pa > 0) & (pb > 0)
            if not np.all(valid):
                pa = pa[valid]
                pb = pb[valid]
                if len(pa) < 60:
                    rejected.append({
                        "pair": f"{asset_a}/{asset_b}",
                        "reason": "too many invalid prices",
                    })
                    continue

            result = self._test_pair(asset_a, asset_b, pa, pb)
            if result is not None:
                qualified.append(result)
            else:
                rejected.append({
                    "pair": f"{asset_a}/{asset_b}",
                    "reason": "failed qualification tests",
                })

        # Rank qualified pairs by composite score and limit
        for p in qualified:
            p.rank_score = self._compute_rank_score(p)
        qualified.sort(key=lambda p: p.rank_score, reverse=True)
        qualified = qualified[:self._max_qualified]

        # Preserve broken pair info and stop counts from previous qualified
        old_map = {
            (p.asset_a, p.asset_b): p for p in self._qualified_pairs
        }
        for p in qualified:
            old = old_map.get((p.asset_a, p.asset_b))
            if old:
                p.consecutive_stops = old.consecutive_stops
                p.post_stop_entry_z = old.post_stop_entry_z

        self._qualified_pairs = qualified
        self._last_run = time.time()

        duration = time.time() - start
        logger.info(
            "Cointegration screening complete: %d qualified, %d rejected in %.1fs",
            len(qualified), len(rejected), duration,
        )

        return CointegrationResult(
            qualified_pairs=qualified,
            rejected_pairs=rejected,
            run_duration_s=duration,
            total_candidates=len(candidates),
            timestamp=time.time(),
        )

    # ======================================================================
    #  Single pair test pipeline
    # ======================================================================

    def _test_pair(
        self,
        asset_a: str,
        asset_b: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> Optional[PairParameters]:
        """Run the full test pipeline on a single pair.

        Returns PairParameters if qualified, None otherwise.
        """
        try:
            # Step 1: Log prices
            log_a = np.log(prices_a)
            log_b = np.log(prices_b)

            # Step 2: OLS regression: ln(Price_A) = alpha + beta * ln(Price_B) + eps
            beta, alpha, residuals = self._ols_regression(log_a, log_b)

            # Step 3: ADF test on residuals
            adf_stat, adf_p = self._adf_test(residuals)
            if adf_p >= self._adf_p_threshold:
                logger.debug(
                    "Pair %s/%s rejected: ADF p=%.4f >= %.4f",
                    asset_a, asset_b, adf_p, self._adf_p_threshold,
                )
                return None

            # Step 4: Half-life
            half_life = self._calculate_half_life(residuals)
            if half_life is None or half_life < self._hl_min or half_life > self._hl_max:
                logger.debug(
                    "Pair %s/%s rejected: half_life=%s (need %d-%d)",
                    asset_a, asset_b, half_life, self._hl_min, self._hl_max,
                )
                return None

            # Step 5: Hurst exponent
            hurst = self._hurst_exponent(residuals)
            if hurst >= self._hurst_reject:
                logger.debug(
                    "Pair %s/%s rejected: Hurst=%.4f >= %.4f",
                    asset_a, asset_b, hurst, self._hurst_reject,
                )
                return None

            # Step 6: Correlation stability
            corr_stability, recent_corr = self._correlation_stability(
                prices_a, prices_b,
            )
            if recent_corr < self._corr_recent_reject:
                logger.debug(
                    "Pair %s/%s rejected: recent corr=%.4f < %.4f",
                    asset_a, asset_b, recent_corr, self._corr_recent_reject,
                )
                return None
            if corr_stability < self._corr_pct_thresh:
                logger.debug(
                    "Pair %s/%s rejected: corr stability=%.1f%% < %d%%",
                    asset_a, asset_b, corr_stability, self._corr_pct_thresh,
                )
                return None

            # Spread statistics
            spread_mean = float(np.mean(residuals))
            spread_std = float(np.std(residuals))
            if spread_std <= 0:
                return None

            is_marginal = self._hurst_ideal <= hurst < self._hurst_reject
            is_preferred = adf_p < self._adf_p_preferred

            params = PairParameters(
                asset_a=asset_a,
                asset_b=asset_b,
                hedge_ratio=beta,
                intercept=alpha,
                spread_mean=spread_mean,
                spread_std=spread_std,
                half_life_days=half_life,
                hurst_exponent=hurst,
                adf_p_value=adf_p,
                adf_statistic=adf_stat,
                correlation_stability=corr_stability,
                recent_correlation=recent_corr,
                qualification_time=time.time(),
                is_marginal=is_marginal,
                is_preferred=is_preferred,
            )

            logger.info(
                "Pair QUALIFIED: %s/%s beta=%.4f HL=%.1fd Hurst=%.3f ADF_p=%.4f corr_stab=%.1f%%",
                asset_a, asset_b, beta, half_life, hurst, adf_p, corr_stability,
            )
            return params

        except Exception:
            logger.exception("Error testing pair %s/%s", asset_a, asset_b)
            return None

    # ======================================================================
    #  Step 2: OLS Regression
    # ======================================================================

    @staticmethod
    def _ols_regression(
        log_a: np.ndarray, log_b: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """Run OLS: ln(Price_A) = alpha + beta * ln(Price_B) + eps.

        Returns (beta, alpha, residuals).
        """
        X = add_constant(log_b)
        model = OLS(log_a, X).fit()
        alpha = float(model.params[0])
        beta = float(model.params[1])
        residuals = log_a - alpha - beta * log_b
        return beta, alpha, residuals

    # ======================================================================
    #  Step 3: ADF Test
    # ======================================================================

    @staticmethod
    def _adf_test(residuals: np.ndarray) -> Tuple[float, float]:
        """Run Augmented Dickey-Fuller test on residuals.

        Returns (adf_statistic, p_value).
        """
        result = adfuller(residuals, maxlag=None, autolag="AIC")
        return float(result[0]), float(result[1])

    # ======================================================================
    #  Step 4: Half-Life Calculation
    # ======================================================================

    @staticmethod
    def _calculate_half_life(residuals: np.ndarray) -> Optional[float]:
        """Fit AR(1) to spread and compute half-life.

        delta_spread_t = theta * (spread_{t-1} - mu) + eps
        half_life = -ln(2) / ln(1 + theta)

        Returns half-life in days, or None if invalid.
        """
        spread = residuals
        mu = np.mean(spread)
        spread_lag = spread[:-1] - mu
        delta_spread = np.diff(spread)

        if len(spread_lag) < 10 or np.std(spread_lag) == 0:
            return None

        # OLS: delta_spread = theta * spread_lag
        X = add_constant(spread_lag)
        model = OLS(delta_spread, X).fit()
        theta = float(model.params[1])

        if theta >= 0:
            # Not mean-reverting
            return None

        # half_life = -ln(2) / ln(1 + theta)
        arg = 1.0 + theta
        if arg <= 0:
            return None

        half_life = -math.log(2) / math.log(arg)
        return half_life

    # ======================================================================
    #  Step 5: Hurst Exponent (Rescaled Range Analysis)
    # ======================================================================

    @staticmethod
    def _hurst_exponent(series: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S (rescaled range) analysis.

        H < 0.5: mean-reverting
        H = 0.5: random walk
        H > 0.5: trending

        Returns the Hurst exponent.
        """
        n = len(series)
        if n < 20:
            return 0.5  # Not enough data

        # Generate range of sub-series lengths
        max_k = n // 2
        min_k = 10
        sizes = []
        k = min_k
        while k <= max_k:
            sizes.append(k)
            k = int(k * 1.5)
        if not sizes:
            return 0.5

        rs_values = []
        size_values = []

        for size in sizes:
            rs_list = []
            num_subseries = n // size

            for i in range(num_subseries):
                subseries = series[i * size:(i + 1) * size]
                mean_sub = np.mean(subseries)
                deviations = subseries - mean_sub
                cumulative = np.cumsum(deviations)
                r = float(np.max(cumulative) - np.min(cumulative))
                s = float(np.std(subseries, ddof=1))

                if s > 0 and r > 0:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append(np.mean(rs_list))
                size_values.append(size)

        if len(rs_values) < 3:
            return 0.5

        log_sizes = np.log(np.array(size_values, dtype=np.float64))
        log_rs = np.log(np.array(rs_values, dtype=np.float64))

        # Linear regression: log(R/S) = H * log(n) + c
        slope, _, _, _, _ = scipy_stats.linregress(log_sizes, log_rs)
        return float(np.clip(slope, 0.0, 1.0))

    # ======================================================================
    #  Step 6: Correlation Stability
    # ======================================================================

    def _correlation_stability(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> Tuple[float, float]:
        """Calculate 30-day rolling correlation stability.

        Returns (pct_above_threshold, recent_30d_correlation).
        """
        # Daily returns
        ret_a = np.diff(np.log(prices_a))
        ret_b = np.diff(np.log(prices_b))
        n = len(ret_a)

        if n < 31:
            return 0.0, 0.0

        window = 30
        above_threshold = 0
        total_windows = 0

        for i in range(window, n + 1):
            ra = ret_a[i - window:i]
            rb = ret_b[i - window:i]
            if np.std(ra) == 0 or np.std(rb) == 0:
                continue
            corr = float(np.corrcoef(ra, rb)[0, 1])
            if np.isnan(corr):
                continue
            total_windows += 1
            if corr > self._corr_min:
                above_threshold += 1

        stability_pct = (above_threshold / total_windows * 100.0) if total_windows > 0 else 0.0

        # Recent 30-day correlation
        recent_a = ret_a[-window:]
        recent_b = ret_b[-window:]
        if np.std(recent_a) == 0 or np.std(recent_b) == 0:
            recent_corr = 0.0
        else:
            recent_corr = float(np.corrcoef(recent_a, recent_b)[0, 1])
            if np.isnan(recent_corr):
                recent_corr = 0.0

        return stability_pct, recent_corr

    # ======================================================================
    #  Ranking
    # ======================================================================

    @staticmethod
    def _compute_rank_score(params: PairParameters) -> float:
        """Composite ranking score (higher is better).

        Factors: ADF strength, half-life proximity to 10d, Hurst < 0.5,
        correlation stability, spread std.
        """
        score = 0.0

        # ADF: lower p-value is better (0 to 30 points)
        score += max(0, (0.05 - params.adf_p_value) / 0.05) * 30.0

        # Half-life: closer to 10 days is ideal (0 to 25 points)
        hl_score = 1.0 - abs(params.half_life_days - 10.0) / 20.0
        score += max(0, hl_score) * 25.0

        # Hurst: lower is better (0 to 25 points)
        score += max(0, (0.5 - params.hurst_exponent) / 0.5) * 25.0

        # Correlation stability (0 to 10 points)
        score += min(params.correlation_stability / 100.0, 1.0) * 10.0

        # Preferred bonus
        if params.is_preferred:
            score += 10.0

        # Marginal penalty
        if params.is_marginal:
            score -= 5.0

        return round(score, 4)

    # ======================================================================
    #  Quick retest for a single pair (post stop-loss)
    # ======================================================================

    def retest_pair(
        self,
        asset_a: str,
        asset_b: str,
        daily_closes_a: List[float],
        daily_closes_b: List[float],
    ) -> Optional[PairParameters]:
        """Quick retest of a single pair (e.g. after stop-loss).

        Returns updated PairParameters if still qualified, None otherwise.
        """
        pa = np.array(daily_closes_a, dtype=np.float64)
        pb = np.array(daily_closes_b, dtype=np.float64)
        min_len = min(len(pa), len(pb))
        if min_len < 60:
            return None
        pa = pa[-min_len:]
        pb = pb[-min_len:]
        valid = (pa > 0) & (pb > 0)
        pa = pa[valid]
        pb = pb[valid]
        if len(pa) < 60:
            return None
        return self._test_pair(asset_a, asset_b, pa, pb)

    # ======================================================================
    #  Serialization for state persistence
    # ======================================================================

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize engine state for persistence."""
        return {
            "qualified_pairs": [
                {
                    "asset_a": p.asset_a,
                    "asset_b": p.asset_b,
                    "hedge_ratio": p.hedge_ratio,
                    "intercept": p.intercept,
                    "spread_mean": p.spread_mean,
                    "spread_std": p.spread_std,
                    "half_life_days": p.half_life_days,
                    "hurst_exponent": p.hurst_exponent,
                    "adf_p_value": p.adf_p_value,
                    "adf_statistic": p.adf_statistic,
                    "correlation_stability": p.correlation_stability,
                    "recent_correlation": p.recent_correlation,
                    "qualification_time": p.qualification_time,
                    "is_marginal": p.is_marginal,
                    "is_preferred": p.is_preferred,
                    "rank_score": p.rank_score,
                    "broken_until": p.broken_until,
                    "consecutive_stops": p.consecutive_stops,
                    "post_stop_entry_z": p.post_stop_entry_z,
                }
                for p in self._qualified_pairs
            ],
            "broken_pairs": {
                f"{a}|{b}": ts for (a, b), ts in self._broken_pairs.items()
            },
            "last_run": self._last_run,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore engine state from persistence."""
        self._last_run = state.get("last_run", 0.0)

        # Restore broken pairs
        for key_str, ts in state.get("broken_pairs", {}).items():
            parts = key_str.split("|")
            if len(parts) == 2 and ts > time.time():
                self._broken_pairs[(parts[0], parts[1])] = ts

        # Restore qualified pairs
        self._qualified_pairs = []
        for pd in state.get("qualified_pairs", []):
            try:
                params = PairParameters(
                    asset_a=pd["asset_a"],
                    asset_b=pd["asset_b"],
                    hedge_ratio=pd["hedge_ratio"],
                    intercept=pd["intercept"],
                    spread_mean=pd["spread_mean"],
                    spread_std=pd["spread_std"],
                    half_life_days=pd["half_life_days"],
                    hurst_exponent=pd["hurst_exponent"],
                    adf_p_value=pd["adf_p_value"],
                    adf_statistic=pd["adf_statistic"],
                    correlation_stability=pd["correlation_stability"],
                    recent_correlation=pd["recent_correlation"],
                    qualification_time=pd["qualification_time"],
                    is_marginal=pd.get("is_marginal", False),
                    is_preferred=pd.get("is_preferred", False),
                    rank_score=pd.get("rank_score", 0.0),
                    broken_until=pd.get("broken_until", 0.0),
                    consecutive_stops=pd.get("consecutive_stops", 0),
                    post_stop_entry_z=pd.get("post_stop_entry_z", 2.0),
                )
                self._qualified_pairs.append(params)
            except (KeyError, TypeError) as exc:
                logger.warning("Skipping malformed pair state: %s", exc)
