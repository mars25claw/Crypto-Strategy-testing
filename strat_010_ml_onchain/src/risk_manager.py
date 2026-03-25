"""STRAT-010-specific risk manager.

Extends the shared RiskManager with ML-specific controls:
- Confidence-based sizing (HIGH=100%, MODERATE=60%)
- Rolling 50-trade accuracy tracking with sizing adjustments
- Model freshness enforcement
- Rolling Sharpe monitoring
- Circuit breakers for inference failures and data-source outages
- Whipsaw protection (4h re-entry delay after stop-loss)
- Consecutive loss handling (4->50%, 6->halt+retrain)
- Cross-strategy conflict resolution
- Feature drift detection
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from shared.config_loader import RiskConfig

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


class TradeOutcome:
    """Record of a completed trade for rolling performance tracking."""

    def __init__(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        pnl_pct: float,
        predicted_p: float,
        actual_direction_correct: bool,
        timestamp_ms: int,
    ):
        self.symbol = symbol
        self.direction = direction
        self.pnl = pnl
        self.pnl_pct = pnl_pct
        self.predicted_p = predicted_p
        self.actual_direction_correct = actual_direction_correct
        self.timestamp_ms = timestamp_ms


class StrategyRiskManager:
    """STRAT-010-specific risk management layer.

    Parameters
    ----------
    config : RiskConfig
        Base risk configuration from shared.
    strategy_params : dict
        Strategy-specific parameters from config.yaml.
    """

    def __init__(self, config: RiskConfig, strategy_params: dict) -> None:
        self._config = config
        self._params = strategy_params

        # Drawdown limits
        self._daily_dd_limit = config.daily_drawdown_pct
        self._weekly_dd_limit = config.weekly_drawdown_pct
        self._monthly_dd_limit = config.monthly_drawdown_pct

        # Rolling trade outcomes
        self._trade_outcomes: Deque[TradeOutcome] = deque(
            maxlen=strategy_params.get("rolling_accuracy_window", 50)
        )

        # Consecutive losses
        self._consecutive_losses: int = 0
        self._consec_reduce = strategy_params.get("consec_loss_reduce_count", 4)
        self._consec_halt = strategy_params.get("consec_loss_halt_count", 6)

        # Whipsaw tracking: symbol -> last_stop_loss_timestamp
        self._stop_loss_times: Dict[str, float] = {}
        self._whipsaw_delay = strategy_params.get("whipsaw_delay_hours", 4) * 3600

        # Rolling Sharpe
        self._sharpe_window = strategy_params.get("rolling_sharpe_window", 30)
        self._sharpe_reduce_threshold = strategy_params.get("rolling_sharpe_reduce_threshold", 0.5)
        self._sharpe_recover_threshold = strategy_params.get("rolling_sharpe_recover_threshold", 1.0)
        self._sharpe_reduced = False

        # Circuit breakers
        self._max_inference_failures = strategy_params.get("max_inference_failures", 3)
        self._data_source_stale_hours = strategy_params.get("data_source_stale_hours", 24)
        self._data_source_halt_count = strategy_params.get("data_source_halt_count", 3)
        self._inference_failure_count: int = 0
        self._stale_data_sources: Dict[str, float] = {}  # source -> stale_since timestamp

        # Model freshness
        self._model_freshness_mult: float = 1.0

        # Sizing thresholds
        self._accuracy_full = strategy_params.get("accuracy_full_threshold", 0.60)
        self._accuracy_reduce = strategy_params.get("accuracy_reduce_threshold", 0.55)
        self._accuracy_halt = strategy_params.get("accuracy_halt_threshold", 0.50)

        # Confidence multipliers
        self._high_conf_mult = strategy_params.get("high_confidence_size_mult", 1.0)
        self._mod_conf_mult = strategy_params.get("moderate_confidence_size_mult", 0.60)

        # Halted state
        self._halted = False
        self._halt_reason = ""

        logger.info(
            "StrategyRiskManager initialised: max_capital=%.1f%%, risk_per_trade=%.1f%%",
            config.max_capital_pct, config.risk_per_trade_pct,
        )

    # ══════════════════════════════════════════════════════════════════
    #  Position sizing
    # ══════════════════════════════════════════════════════════════════

    def calculate_position_size(
        self,
        equity: float,
        stop_distance_pct: float,
        confidence: str,
        atr_pct: float,
    ) -> Tuple[float, int, str]:
        """Calculate position size with all adjustments.

        Returns (size_usdt, leverage, reason_if_zero).
        """
        if self._halted:
            return 0.0, 0, f"Strategy halted: {self._halt_reason}"

        if equity <= 0:
            return 0.0, 0, "Equity not initialised"

        if stop_distance_pct <= 0:
            return 0.0, 0, "Invalid stop distance"

        # Base position size: equity * risk_per_trade / stop_distance
        risk_per_trade = self._config.risk_per_trade_pct / 100.0
        base_size = (equity * risk_per_trade) / stop_distance_pct

        # Cap at max per trade
        max_per_trade = equity * (self._config.max_per_trade_pct / 100.0)
        base_size = min(base_size, max_per_trade)

        # Confidence adjustment
        conf_mult = self._get_confidence_multiplier(confidence)
        size = base_size * conf_mult

        # Performance adjustment (rolling accuracy)
        perf_mult = self._get_performance_multiplier()
        size *= perf_mult

        # Model freshness adjustment
        size *= self._model_freshness_mult

        # Rolling Sharpe adjustment
        if self._sharpe_reduced:
            size *= 0.5

        # Consecutive loss adjustment
        consec_mult = self._get_consecutive_loss_multiplier()
        size *= consec_mult

        # Data source confidence reduction
        ds_mult = self._get_data_source_multiplier()
        size *= ds_mult

        # Leverage
        leverage = self._config.preferred_leverage
        high_vol_atr = self._params.get("high_volatility_atr_pct", 3.0)
        if atr_pct > high_vol_atr:
            leverage = self._params.get("high_volatility_leverage", 2)
        leverage = min(leverage, self._config.max_leverage)

        reason = "" if size > 0 else "Size reduced to zero by adjustments"
        return round(size, 2), leverage, reason

    def _get_confidence_multiplier(self, confidence: str) -> float:
        if confidence == "HIGH":
            return self._high_conf_mult
        elif confidence == "MODERATE":
            return self._mod_conf_mult
        return 0.0  # NONE confidence = no trade

    def _get_performance_multiplier(self) -> float:
        """Adjust sizing based on rolling 50-trade accuracy."""
        if len(self._trade_outcomes) < 10:
            return 1.0  # Not enough data

        correct = sum(1 for t in self._trade_outcomes if t.actual_direction_correct)
        accuracy = correct / len(self._trade_outcomes)

        if accuracy >= self._accuracy_full:
            return 1.0
        elif accuracy >= self._accuracy_reduce:
            return 0.5
        elif accuracy >= self._accuracy_halt:
            return 0.5
        else:
            self.halt(f"Rolling accuracy {accuracy:.2%} below halt threshold {self._accuracy_halt:.2%}")
            return 0.0

    def _get_consecutive_loss_multiplier(self) -> float:
        if self._consecutive_losses >= self._consec_halt:
            self.halt(
                f"{self._consecutive_losses} consecutive losses >= halt threshold "
                f"({self._consec_halt})"
            )
            return 0.0
        elif self._consecutive_losses >= self._consec_reduce:
            return 0.5
        return 1.0

    def _get_data_source_multiplier(self) -> float:
        """Reduce confidence if data sources are stale."""
        now = time.time()
        stale_count = 0
        for source, stale_since in self._stale_data_sources.items():
            if (now - stale_since) > self._data_source_stale_hours * 3600:
                stale_count += 1

        if stale_count >= self._data_source_halt_count:
            self.halt(f"{stale_count} data sources down > {self._data_source_stale_hours}h")
            return 0.0
        elif stale_count > 0:
            return 0.75  # 25% confidence reduction per spec
        return 1.0

    # ══════════════════════════════════════════════════════════════════
    #  Trade recording
    # ══════════════════════════════════════════════════════════════════

    def record_trade(self, outcome: TradeOutcome) -> None:
        """Record a completed trade outcome."""
        self._trade_outcomes.append(outcome)

        if outcome.pnl > 0:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        trade_logger.info(
            "STRAT-010 TRADE\tsymbol=%s\tdir=%s\tpnl=%.4f\tcorrect=%s\tconsec_loss=%d",
            outcome.symbol, outcome.direction, outcome.pnl,
            outcome.actual_direction_correct, self._consecutive_losses,
        )

    def record_stop_loss(self, symbol: str) -> None:
        """Record a stop-loss event for whipsaw protection."""
        self._stop_loss_times[symbol] = time.time()
        logger.info("Stop loss recorded for %s -- %dh re-entry delay", symbol, self._whipsaw_delay // 3600)

    # ══════════════════════════════════════════════════════════════════
    #  Filters
    # ══════════════════════════════════════════════════════════════════

    def check_whipsaw(self, symbol: str) -> bool:
        """Return True if re-entry is blocked due to recent stop-loss."""
        last_stop = self._stop_loss_times.get(symbol, 0)
        if last_stop == 0:
            return False
        elapsed = time.time() - last_stop
        if elapsed < self._whipsaw_delay:
            logger.info(
                "Whipsaw block: %s stopped %.1fh ago (need %.1fh)",
                symbol, elapsed / 3600, self._whipsaw_delay / 3600,
            )
            return True
        return False

    def check_cross_strategy_conflict(
        self,
        symbol: str,
        direction: str,
        confidence: str,
        other_positions: List[Dict[str, Any]],
    ) -> bool:
        """Return True if entry is allowed given other strategies' positions.

        If other strategies have opposite-direction positions,
        HIGH confidence is required to override.
        """
        for pos in other_positions:
            if pos.get("symbol") != symbol:
                continue
            other_dir = pos.get("direction", "")
            if other_dir and other_dir != direction:
                # Opposite direction -- need HIGH confidence
                if confidence != "HIGH":
                    logger.info(
                        "Cross-strategy conflict: %s has %s on %s, "
                        "our %s %s needs HIGH confidence",
                        pos.get("strategy_id", "unknown"),
                        other_dir, symbol, direction, confidence,
                    )
                    return False
        return True

    # ══════════════════════════════════════════════════════════════════
    #  Circuit breakers
    # ══════════════════════════════════════════════════════════════════

    def record_inference_failure(self) -> None:
        """Record a model inference failure."""
        self._inference_failure_count += 1
        if self._inference_failure_count >= self._max_inference_failures:
            self.halt(
                f"{self._inference_failure_count} consecutive inference failures "
                f"(limit={self._max_inference_failures})"
            )

    def record_inference_success(self) -> None:
        """Reset the inference failure counter."""
        self._inference_failure_count = 0

    def mark_data_source_stale(self, source: str) -> None:
        """Mark an external data source as stale."""
        if source not in self._stale_data_sources:
            self._stale_data_sources[source] = time.time()
            logger.warning("Data source marked stale: %s", source)

    def mark_data_source_healthy(self, source: str) -> None:
        """Mark a data source as healthy (no longer stale)."""
        if source in self._stale_data_sources:
            del self._stale_data_sources[source]
            logger.info("Data source healthy again: %s", source)

    def set_model_freshness(self, multiplier: float) -> None:
        """Set the model freshness multiplier (from ModelManager)."""
        old = self._model_freshness_mult
        self._model_freshness_mult = multiplier
        if multiplier != old:
            logger.info("Model freshness multiplier changed: %.2f -> %.2f", old, multiplier)
        if multiplier == 0.0:
            self.halt("Model too stale (>60 days)")

    # ══════════════════════════════════════════════════════════════════
    #  Rolling Sharpe monitoring
    # ══════════════════════════════════════════════════════════════════

    def update_rolling_sharpe(self) -> float:
        """Compute rolling Sharpe from recent trades and apply sizing rule."""
        if len(self._trade_outcomes) < self._sharpe_window:
            return 0.0

        recent = list(self._trade_outcomes)[-self._sharpe_window:]
        returns = [t.pnl_pct for t in recent]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            sharpe = 0.0
        else:
            sharpe = float(mean_ret / std_ret * np.sqrt(365 * 24))

        if sharpe < self._sharpe_reduce_threshold and not self._sharpe_reduced:
            self._sharpe_reduced = True
            logger.warning(
                "Rolling Sharpe %.2f < %.2f -- reducing position sizes by 50%%",
                sharpe, self._sharpe_reduce_threshold,
            )
        elif sharpe > self._sharpe_recover_threshold and self._sharpe_reduced:
            self._sharpe_reduced = False
            logger.info(
                "Rolling Sharpe %.2f > %.2f -- restoring full position sizes",
                sharpe, self._sharpe_recover_threshold,
            )

        return sharpe

    # ══════════════════════════════════════════════════════════════════
    #  Halt / Resume
    # ══════════════════════════════════════════════════════════════════

    def halt(self, reason: str) -> None:
        """Halt the strategy (no new entries)."""
        if not self._halted:
            self._halted = True
            self._halt_reason = reason
            logger.critical("STRAT-010 HALTED: %s", reason)

    def resume(self) -> None:
        """Resume trading after manual investigation."""
        if self._halted:
            logger.info("STRAT-010 RESUMED (was halted: %s)", self._halt_reason)
            self._halted = False
            self._halt_reason = ""
            self._consecutive_losses = 0
            self._inference_failure_count = 0

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    # ══════════════════════════════════════════════════════════════════
    #  Status / Metrics
    # ══════════════════════════════════════════════════════════════════

    def get_rolling_accuracy(self) -> float:
        """Return rolling accuracy over the tracking window."""
        if not self._trade_outcomes:
            return 0.0
        correct = sum(1 for t in self._trade_outcomes if t.actual_direction_correct)
        return correct / len(self._trade_outcomes)

    def get_status(self) -> Dict[str, Any]:
        """Return risk manager status for dashboard."""
        return {
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "consecutive_losses": self._consecutive_losses,
            "rolling_accuracy": round(self.get_rolling_accuracy(), 4),
            "rolling_sharpe": round(self.update_rolling_sharpe(), 4),
            "sharpe_reduced": self._sharpe_reduced,
            "model_freshness_mult": self._model_freshness_mult,
            "inference_failures": self._inference_failure_count,
            "stale_data_sources": list(self._stale_data_sources.keys()),
            "trade_count": len(self._trade_outcomes),
            "whipsaw_blocks": {
                sym: round((time.time() - ts) / 3600, 1)
                for sym, ts in self._stop_loss_times.items()
                if (time.time() - ts) < self._whipsaw_delay
            },
        }

    def get_confidence_vs_accuracy(self) -> Dict[str, Dict[str, float]]:
        """Return accuracy broken down by confidence level for dashboard."""
        buckets: Dict[str, List[bool]] = {"HIGH": [], "MODERATE": []}
        for t in self._trade_outcomes:
            if t.predicted_p > 0.75 or t.predicted_p < 0.25:
                buckets["HIGH"].append(t.actual_direction_correct)
            else:
                buckets["MODERATE"].append(t.actual_direction_correct)

        result: Dict[str, Dict[str, float]] = {}
        for conf, outcomes in buckets.items():
            if outcomes:
                result[conf] = {
                    "accuracy": round(sum(outcomes) / len(outcomes), 4),
                    "count": len(outcomes),
                }
            else:
                result[conf] = {"accuracy": 0.0, "count": 0}
        return result
