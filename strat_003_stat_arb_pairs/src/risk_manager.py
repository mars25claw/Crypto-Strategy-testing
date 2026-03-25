"""STRAT-003 Pairs-Specific Risk Manager.

Extends the shared RiskManager with pairs-specific rules:
- 25% max equity, 5% per pair, 10 max pairs
- Hedge-ratio-based sizing with risk cap
- Dollar stop 1.5%, per-pair max 2%
- Leverage 3x per leg (prefer 2x)
- Drawdown halts: 2.5% daily, 5% weekly, 8% monthly
- Exposure: LONG 15%, SHORT 15%, NET 5%
- Single asset across all pairs 10%
- Correlation-aware sizing (overlap reduce 30%, net directional >3% reject)
- Circuit breakers: Z +/-5.0 close all 24h, 3+ pairs stopped 48h, BTC >8% 4h halt
- TWAP exit for illiquid legs
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.cointegration import PairParameters

logger = logging.getLogger(__name__)


class PairsRiskManager:
    """Pairs-specific risk enforcement layer.

    Sits between the strategy engine and the shared RiskManager to enforce
    pairs-specific constraints before delegating to the shared layer.

    Parameters
    ----------
    config : dict
        Strategy parameters from config.yaml strategy_params.
    shared_risk_manager : object
        The shared.risk_manager.RiskManager instance.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        shared_risk_manager: Any,
    ) -> None:
        self._cfg = config
        self._shared = shared_risk_manager

        # Limits
        self._max_equity_pct = 25.0
        self._max_per_pair_pct = 5.0
        self._max_active_pairs = config.get("max_active_pairs", 10)
        self._max_leverage = config.get("max_leverage", 3)
        self._preferred_leverage = config.get("preferred_leverage", 2)
        self._max_loss_per_pair_pct = 2.0
        self._dollar_stop_pct = config.get("dollar_stop_pct", 1.5)
        self._max_long_pct = 15.0
        self._max_short_pct = 15.0
        self._max_net_pct = 5.0
        self._max_single_asset_pct = 10.0
        self._overlap_reduction = 0.30
        self._max_directional_per_asset_pct = 3.0

        # Tracking
        self._active_pairs: Dict[str, Dict[str, Any]] = {}  # pair_id -> info
        self._asset_exposure: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"long": 0.0, "short": 0.0}
        )
        self._pair_pnl: Dict[str, float] = {}  # pair_id -> running PnL

    # ======================================================================
    #  Entry gate
    # ======================================================================

    def check_pair_entry(
        self,
        params: PairParameters,
        direction: str,
        equity: float,
    ) -> Tuple[bool, str]:
        """Check if a new pair trade is allowed.

        Returns (allowed, reason).
        """
        if equity <= 0:
            return False, "Equity not available"

        # 1. Max concurrent pairs
        if len(self._active_pairs) >= self._max_active_pairs:
            return False, f"Max active pairs reached ({self._max_active_pairs})"

        # 2. Strategy total exposure
        total_exposure = sum(
            abs(info.get("size_a", 0)) + abs(info.get("size_b", 0))
            for info in self._active_pairs.values()
        )
        max_total = equity * self._max_equity_pct / 100.0
        per_pair = equity * self._max_per_pair_pct / 100.0
        if total_exposure + per_pair > max_total:
            return False, f"Total exposure would exceed {self._max_equity_pct}% of equity"

        # 3. Single asset exposure check
        asset_a, asset_b = params.asset_a, params.asset_b
        price_a_approx = per_pair / 2.0  # approximate half per leg
        price_b_approx = per_pair / 2.0

        for asset, add_size in [(asset_a, price_a_approx), (asset_b, price_b_approx)]:
            current = self._get_total_asset_exposure(asset)
            max_asset = equity * self._max_single_asset_pct / 100.0
            if current + add_size > max_asset:
                return False, (
                    f"Asset {asset} exposure {current + add_size:.0f} would exceed "
                    f"{self._max_single_asset_pct}% of equity ({max_asset:.0f})"
                )

        # 4. Directional exposure check
        long_exp, short_exp = self._get_directional_exposure()
        add_long = per_pair / 2.0
        add_short = per_pair / 2.0

        max_long = equity * self._max_long_pct / 100.0
        max_short = equity * self._max_short_pct / 100.0
        max_net = equity * self._max_net_pct / 100.0

        if long_exp + add_long > max_long:
            return False, f"Long exposure would exceed {self._max_long_pct}%"
        if short_exp + add_short > max_short:
            return False, f"Short exposure would exceed {self._max_short_pct}%"
        net = abs((long_exp + add_long) - (short_exp + add_short))
        if net > max_net:
            return False, f"Net directional exposure {net:.0f} would exceed {self._max_net_pct}%"

        # 5. Correlation-aware sizing
        size_reduction, reject_reason = self._check_correlation_overlap(
            params, direction, equity,
        )
        if reject_reason:
            return False, reject_reason

        # 6. Drawdown check (delegate to shared)
        halted, level, dd_pct = self._shared.check_drawdown()
        if halted:
            return False, f"Drawdown halt: {level} at {dd_pct:.2f}%"

        # 7. Consecutive loss check from shared
        consec = self._shared.get_consecutive_losses("STRAT-003")
        if consec >= 5:
            return False, f"Trading halted: {consec} consecutive losses (manual review required)"

        return True, ""

    # ======================================================================
    #  Position tracking
    # ======================================================================

    def record_pair_open(
        self,
        pair_id: str,
        asset_a: str,
        asset_b: str,
        direction: str,
        size_a: float,
        size_b: float,
        side_a: str,
        side_b: str,
    ) -> None:
        """Record a new pair position opening."""
        self._active_pairs[pair_id] = {
            "asset_a": asset_a,
            "asset_b": asset_b,
            "direction": direction,
            "size_a": size_a,
            "size_b": size_b,
            "side_a": side_a,
            "side_b": side_b,
            "opened_at": time.time(),
        }

        # Update directional asset exposure
        if side_a == "BUY":
            self._asset_exposure[asset_a]["long"] += size_a
        else:
            self._asset_exposure[asset_a]["short"] += size_a
        if side_b == "BUY":
            self._asset_exposure[asset_b]["long"] += size_b
        else:
            self._asset_exposure[asset_b]["short"] += size_b

        # Report to shared risk manager
        self._shared.record_position_change(
            "STRAT-003", asset_a, side_a, size_a, is_open=True,
        )
        self._shared.record_position_change(
            "STRAT-003", asset_b, side_b, size_b, is_open=True,
        )

        logger.info(
            "Pair %s opened: %s(%s %.0f) + %s(%s %.0f)",
            pair_id, asset_a, side_a, size_a, asset_b, side_b, size_b,
        )

    def record_pair_close(
        self,
        pair_id: str,
        pnl: float,
    ) -> None:
        """Record a pair position closing."""
        info = self._active_pairs.pop(pair_id, None)
        if info is None:
            return

        asset_a, asset_b = info["asset_a"], info["asset_b"]
        side_a, side_b = info["side_a"], info["side_b"]

        # Update exposure tracking
        if side_a == "BUY":
            self._asset_exposure[asset_a]["long"] -= info["size_a"]
        else:
            self._asset_exposure[asset_a]["short"] -= info["size_a"]
        if side_b == "BUY":
            self._asset_exposure[asset_b]["long"] -= info["size_b"]
        else:
            self._asset_exposure[asset_b]["short"] -= info["size_b"]

        # Report to shared
        self._shared.record_position_change(
            "STRAT-003", asset_a, side_a, info["size_a"], is_open=False,
        )
        self._shared.record_position_change(
            "STRAT-003", asset_b, side_b, info["size_b"], is_open=False,
        )
        self._shared.record_trade_result(
            "STRAT-003", pnl, is_win=(pnl > 0),
        )

    # ======================================================================
    #  Sizing
    # ======================================================================

    def get_size_multiplier(self, params: PairParameters) -> float:
        """Return position size multiplier accounting for all reductions.

        Base is 1.0. Reduced for:
        - Marginal Hurst: 0.7x
        - Correlation overlap: 0.7x
        - Consecutive losses >= 3: 0.75x
        """
        mult = 1.0

        if params.is_marginal:
            mult *= 0.7

        # Check correlation overlap
        reduction, _ = self._check_correlation_overlap(
            params, "LONG_SPREAD", 1.0,  # direction doesn't matter for sizing
        )
        if reduction > 0:
            mult *= (1.0 - reduction)

        # Consecutive losses
        consec = self._shared.get_consecutive_losses("STRAT-003")
        if consec >= 3:
            mult *= 0.75

        return mult

    def calculate_leverage(self, params: PairParameters) -> int:
        """Determine leverage for a pair trade. Prefer 2x, max 3x."""
        if params.is_marginal:
            return min(self._preferred_leverage, 2)
        return self._preferred_leverage

    # ======================================================================
    #  Correlation-aware sizing (Section 5.5)
    # ======================================================================

    def _check_correlation_overlap(
        self,
        params: PairParameters,
        direction: str,
        equity: float,
    ) -> Tuple[float, Optional[str]]:
        """Check directional overlap with existing pairs.

        Returns (size_reduction_fraction, rejection_reason_or_None).
        """
        reduction = 0.0
        asset_a, asset_b = params.asset_a, params.asset_b

        # Determine which asset is long and which is short
        if direction == "LONG_SPREAD":
            long_asset, short_asset = asset_a, asset_b
        else:
            long_asset, short_asset = asset_b, asset_a

        # Check if long asset is already long in another pair
        for pair_id, info in self._active_pairs.items():
            if info.get("side_a") == "BUY" and info.get("asset_a") == long_asset:
                reduction = max(reduction, self._overlap_reduction)
            if info.get("side_b") == "BUY" and info.get("asset_b") == long_asset:
                reduction = max(reduction, self._overlap_reduction)

        # Check net directional > 3% in any single asset
        if equity > 0:
            for asset in [asset_a, asset_b]:
                exp = self._asset_exposure.get(asset, {"long": 0.0, "short": 0.0})
                net = abs(exp["long"] - exp["short"])
                net_pct = net / equity * 100.0
                if net_pct > self._max_directional_per_asset_pct:
                    return 0.0, (
                        f"Net directional exposure for {asset} is {net_pct:.1f}% "
                        f"(max {self._max_directional_per_asset_pct}%)"
                    )

        return reduction, None

    # ======================================================================
    #  Exposure queries
    # ======================================================================

    def _get_total_asset_exposure(self, asset: str) -> float:
        """Total absolute exposure for a single asset across all pairs."""
        exp = self._asset_exposure.get(asset, {"long": 0.0, "short": 0.0})
        return exp["long"] + exp["short"]

    def _get_directional_exposure(self) -> Tuple[float, float]:
        """Return (total_long_exposure, total_short_exposure) across all pairs."""
        total_long = 0.0
        total_short = 0.0
        for info in self._active_pairs.values():
            if info.get("side_a") == "BUY":
                total_long += info.get("size_a", 0)
            else:
                total_short += info.get("size_a", 0)
            if info.get("side_b") == "BUY":
                total_long += info.get("size_b", 0)
            else:
                total_short += info.get("size_b", 0)
        return total_long, total_short

    def get_exposure_summary(self) -> Dict[str, Any]:
        """Return a summary of current exposure for the dashboard."""
        long_exp, short_exp = self._get_directional_exposure()
        equity = self._shared.get_current_equity()

        per_asset: Dict[str, Dict[str, float]] = {}
        for asset, exp in self._asset_exposure.items():
            if exp["long"] > 0 or exp["short"] > 0:
                per_asset[asset] = {
                    "long": round(exp["long"], 2),
                    "short": round(exp["short"], 2),
                    "net": round(exp["long"] - exp["short"], 2),
                    "total": round(exp["long"] + exp["short"], 2),
                }

        return {
            "active_pairs": len(self._active_pairs),
            "max_pairs": self._max_active_pairs,
            "total_long": round(long_exp, 2),
            "total_short": round(short_exp, 2),
            "net_directional": round(long_exp - short_exp, 2),
            "total_exposure": round(long_exp + short_exp, 2),
            "total_exposure_pct": round((long_exp + short_exp) / equity * 100, 2) if equity > 0 else 0,
            "long_pct": round(long_exp / equity * 100, 2) if equity > 0 else 0,
            "short_pct": round(short_exp / equity * 100, 2) if equity > 0 else 0,
            "net_pct": round(abs(long_exp - short_exp) / equity * 100, 2) if equity > 0 else 0,
            "per_asset": per_asset,
            "drawdown": self._shared.check_drawdown(),
            "consecutive_losses": self._shared.get_consecutive_losses("STRAT-003"),
        }
