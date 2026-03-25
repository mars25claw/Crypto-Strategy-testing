"""Comprehensive performance metrics for STRAT-010 ML & On-Chain.

Implements:
- Section 10.2: Model-specific metrics (Accuracy, AUC-ROC, Precision/Recall,
  Brier Score, Feature Importance, Prediction Distribution, Confidence vs
  Accuracy, Data Source Attribution, Model Staleness, Ensemble Agreement,
  Exchange Flow Accuracy, MVRV Accuracy)
- Section 10.3: On-chain signal analysis and dimensional breakdowns
- Concept Drift Monitoring
- Go-Live Criteria (Section 9.4)
- Dashboard integration data
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concept Drift Monitor
# ---------------------------------------------------------------------------

class ConceptDriftMonitor:
    """Monitor feature distribution over time for concept drift.

    If distribution of any feature shifts >2 sigma from training distribution
    for >24 hours, flag for retraining. Track mean, std, min, max per feature
    over rolling 7-day window. Compare vs training statistics.
    """

    def __init__(
        self,
        n_features: int = 35,
        window_hours: int = 168,  # 7 days
        drift_sigma: float = 2.0,
        drift_duration_hours: float = 24.0,
    ) -> None:
        self._n_features = n_features
        self._window_hours = window_hours
        self._drift_sigma = drift_sigma
        self._drift_duration_hours = drift_duration_hours

        # Rolling feature history (7-day window of hourly observations)
        self._feature_history: Deque[Tuple[float, np.ndarray]] = deque(maxlen=window_hours)

        # Training distribution statistics
        self._training_mean: Optional[np.ndarray] = None
        self._training_std: Optional[np.ndarray] = None

        # Drift tracking: feature_index -> first_drift_timestamp
        self._drift_start: Dict[int, float] = {}

        # Feature names for reporting
        self._feature_names: List[str] = [f"feature_{i}" for i in range(n_features)]

    def set_training_distribution(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Set the training distribution for comparison."""
        self._training_mean = mean.copy()
        self._training_std = std.copy()
        if feature_names:
            self._feature_names = feature_names

    def update(self, features: np.ndarray) -> None:
        """Add a new hourly feature observation."""
        self._feature_history.append((time.time(), features.copy()))

    def check_concept_drift(self) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Check for concept drift.

        Returns
        -------
        (drifted, drifted_features, details)
            drifted: True if any feature has drifted >2 sigma for >24 hours
            drifted_features: list of drifted feature names
            details: per-feature drift information
        """
        if self._training_mean is None or self._training_std is None:
            return False, [], {"error": "Training distribution not set"}

        if len(self._feature_history) < 24:
            return False, [], {"error": "Insufficient data (need 24h minimum)"}

        now = time.time()

        # Compute rolling statistics
        recent = [f for _, f in self._feature_history]
        arr = np.array(recent)  # (n_observations, n_features)
        rolling_mean = np.nanmean(arr, axis=0)
        rolling_std = np.nanstd(arr, axis=0)
        rolling_min = np.nanmin(arr, axis=0)
        rolling_max = np.nanmax(arr, axis=0)

        drifted_features: List[str] = []
        details: Dict[str, Any] = {
            "features": {},
            "total_checked": self._n_features,
        }

        for i in range(min(self._n_features, len(rolling_mean))):
            if np.isnan(rolling_mean[i]) or np.isnan(self._training_mean[i]):
                continue

            train_std = max(self._training_std[i], 1e-10)
            z_score = abs((rolling_mean[i] - self._training_mean[i]) / train_std)

            feature_info = {
                "rolling_mean": round(float(rolling_mean[i]), 4),
                "rolling_std": round(float(rolling_std[i]), 4),
                "rolling_min": round(float(rolling_min[i]), 4),
                "rolling_max": round(float(rolling_max[i]), 4),
                "training_mean": round(float(self._training_mean[i]), 4),
                "training_std": round(float(self._training_std[i]), 4),
                "z_score_from_training": round(float(z_score), 2),
                "drifted": False,
                "drift_duration_hours": 0.0,
            }

            if z_score > self._drift_sigma:
                # Feature is drifting
                if i not in self._drift_start:
                    self._drift_start[i] = now
                drift_hours = (now - self._drift_start[i]) / 3600

                feature_info["drifted"] = True
                feature_info["drift_duration_hours"] = round(drift_hours, 1)

                if drift_hours >= self._drift_duration_hours:
                    name = self._feature_names[i] if i < len(self._feature_names) else f"feature_{i}"
                    drifted_features.append(name)
                    logger.warning(
                        "Concept drift detected: %s shifted %.1f sigma from training "
                        "for %.1f hours",
                        name, z_score, drift_hours,
                    )
            else:
                # No longer drifting
                if i in self._drift_start:
                    del self._drift_start[i]

            name = self._feature_names[i] if i < len(self._feature_names) else f"feature_{i}"
            details["features"][name] = feature_info

        is_drifted = len(drifted_features) > 0
        details["drifted"] = is_drifted
        details["drifted_count"] = len(drifted_features)
        details["drifted_features"] = drifted_features

        return is_drifted, drifted_features, details

    def to_state(self) -> dict:
        return {
            "drift_start": {str(k): v for k, v in self._drift_start.items()},
            "history_len": len(self._feature_history),
        }

    def load_state(self, state: dict) -> None:
        self._drift_start = {int(k): v for k, v in state.get("drift_start", {}).items()}


# ---------------------------------------------------------------------------
# Section 10.2 Model-Specific Metrics
# ---------------------------------------------------------------------------

class MLStrategyMetrics:
    """Comprehensive metrics tracker for Section 10.2 and 10.3.

    Tracks:
    - Model Accuracy, AUC-ROC, Precision/Recall
    - Brier Score
    - Feature Importance Rankings
    - Prediction Distribution
    - Confidence vs Accuracy
    - Data Source Attribution
    - Model Staleness
    - Ensemble Agreement
    - Exchange Flow Accuracy
    - MVRV Accuracy
    """

    def __init__(self) -> None:
        # Trade outcomes for accuracy tracking
        self._predictions: List[Dict[str, Any]] = []

        # On-chain signal tracking
        self._exchange_flow_signals: List[Dict[str, Any]] = []
        self._mvrv_signals: List[Dict[str, Any]] = []

        # Feature source attribution tracking
        self._with_onchain_trades: List[Dict[str, float]] = []
        self._without_onchain_trades: List[Dict[str, float]] = []
        self._with_sentiment_trades: List[Dict[str, float]] = []
        self._without_sentiment_trades: List[Dict[str, float]] = []

    def record_prediction(
        self,
        symbol: str,
        p_final: float,
        p_xgboost: float,
        p_lstm: float,
        signal: str,
        confidence: str,
        actual_correct: Optional[bool] = None,
        pnl: Optional[float] = None,
        had_onchain: bool = True,
        had_sentiment: bool = True,
        exchange_net_flow: float = 0.0,
        mvrv_zscore: float = 0.0,
    ) -> None:
        """Record a prediction with its outcome for metrics."""
        record = {
            "timestamp": time.time(),
            "symbol": symbol,
            "p_final": p_final,
            "p_xgboost": p_xgboost,
            "p_lstm": p_lstm,
            "signal": signal,
            "confidence": confidence,
            "actual_correct": actual_correct,
            "pnl": pnl,
            "had_onchain": had_onchain,
            "had_sentiment": had_sentiment,
        }
        self._predictions.append(record)

        # Track on-chain signal accuracy
        if not np.isnan(exchange_net_flow) and exchange_net_flow != 0:
            self._exchange_flow_signals.append({
                "timestamp": time.time(),
                "value": exchange_net_flow,
                "signal": "SHORT" if exchange_net_flow > 0 else "LONG",
                "actual_correct": actual_correct,
                "pnl": pnl,
            })

        if not np.isnan(mvrv_zscore):
            self._mvrv_signals.append({
                "timestamp": time.time(),
                "value": mvrv_zscore,
                "signal": "LONG" if mvrv_zscore < 0 else "SHORT",
                "actual_correct": actual_correct,
                "pnl": pnl,
            })

    def update_outcome(self, timestamp: float, actual_correct: bool, pnl: float) -> None:
        """Update a prediction with its actual outcome (4h later)."""
        for pred in reversed(self._predictions):
            if abs(pred["timestamp"] - timestamp) < 3600:
                pred["actual_correct"] = actual_correct
                pred["pnl"] = pnl

                # Data source attribution
                if pred.get("had_onchain"):
                    self._with_onchain_trades.append({"pnl": pnl})
                else:
                    self._without_onchain_trades.append({"pnl": pnl})
                if pred.get("had_sentiment"):
                    self._with_sentiment_trades.append({"pnl": pnl})
                else:
                    self._without_sentiment_trades.append({"pnl": pnl})
                break

    def get_full_metrics(
        self,
        model_status: Dict[str, Any],
        risk_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute all Section 10.2 and 10.3 metrics."""
        completed = [p for p in self._predictions if p.get("actual_correct") is not None]

        # Model Accuracy
        accuracy = 0.0
        if completed:
            accuracy = sum(1 for p in completed if p["actual_correct"]) / len(completed)

        # AUC-ROC (simplified: use p_final as score)
        auc_roc = model_status.get("xgb_meta", {}).get("auc_roc", 0.0)

        # Precision/Recall for LONG and SHORT
        precision_recall = self._compute_precision_recall(completed)

        # Brier Score
        brier = self._compute_brier_score(completed)

        # Feature Importance Rankings
        feature_importance = model_status.get("feature_importance", {})

        # Prediction Distribution
        pred_dist = self._compute_prediction_distribution()

        # Confidence vs Accuracy
        conf_accuracy = self._compute_confidence_vs_accuracy(completed)

        # Data Source Attribution
        source_attribution = self._compute_source_attribution()

        # Model Staleness
        staleness = {
            "freshness_multiplier": model_status.get("freshness_multiplier", 1.0),
            "xgb_age_days": model_status.get("xgb_meta", {}).get("age_days", 0),
            "lstm_age_days": model_status.get("lstm_meta", {}).get("age_days", 0),
            "should_halt": model_status.get("should_halt_staleness", False),
        }

        # Ensemble Agreement
        ensemble = {
            "agreement_rate": model_status.get("ensemble_agreement_rate", 0.0),
        }

        # On-chain Signal Analysis (Section 10.3)
        exchange_flow_accuracy = self._compute_signal_accuracy(self._exchange_flow_signals)
        mvrv_accuracy = self._compute_signal_accuracy(self._mvrv_signals)

        return {
            # Section 10.2
            "model_accuracy": round(accuracy, 4),
            "total_predictions": len(completed),
            "auc_roc": round(auc_roc, 4),
            "precision_recall": precision_recall,
            "brier_score": round(brier, 4),
            "feature_importance_top10": feature_importance,
            "prediction_distribution": pred_dist,
            "confidence_vs_accuracy": conf_accuracy,
            "data_source_attribution": source_attribution,
            "model_staleness": staleness,
            "ensemble_agreement": ensemble,

            # Section 10.3
            "exchange_flow_accuracy": exchange_flow_accuracy,
            "mvrv_accuracy": mvrv_accuracy,
        }

    def _compute_precision_recall(self, completed: List[dict]) -> Dict[str, Any]:
        """Precision/Recall for LONG and SHORT separately."""
        result = {}
        for direction in ["LONG", "SHORT"]:
            directed = [p for p in completed if p["signal"] == direction]
            if not directed:
                result[direction] = {"precision": 0.0, "recall": 0.0, "count": 0}
                continue
            tp = sum(1 for p in directed if p["actual_correct"])
            fp = len(directed) - tp
            # Recall: of all actual moves in this direction, how many did we predict?
            all_correct = [p for p in completed if p["actual_correct"]]
            fn = sum(1 for p in all_correct if p["signal"] != direction)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            result[direction] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "count": len(directed),
            }
        return result

    def _compute_brier_score(self, completed: List[dict]) -> float:
        """Brier Score: calibration of probability estimates."""
        if not completed:
            return 0.0
        scores = []
        for p in completed:
            actual = 1.0 if p["actual_correct"] else 0.0
            predicted = p["p_final"]
            scores.append((predicted - actual) ** 2)
        return float(np.mean(scores))

    def _compute_prediction_distribution(self) -> Dict[str, int]:
        """Histogram of P_final values."""
        if not self._predictions:
            return {}
        buckets = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}
        for p in self._predictions:
            idx = min(int(p["p_final"] * 10), 9)
            key = list(buckets.keys())[idx]
            buckets[key] += 1
        return buckets

    def _compute_confidence_vs_accuracy(self, completed: List[dict]) -> Dict[str, Any]:
        """Accuracy broken down by confidence level."""
        result = {}
        for conf in ["HIGH", "MODERATE"]:
            group = [p for p in completed if p["confidence"] == conf]
            if group:
                acc = sum(1 for p in group if p["actual_correct"]) / len(group)
                result[conf] = {
                    "accuracy": round(acc, 4),
                    "count": len(group),
                    "avg_pnl": round(
                        float(np.mean([p.get("pnl", 0) or 0 for p in group])), 4
                    ),
                }
            else:
                result[conf] = {"accuracy": 0.0, "count": 0, "avg_pnl": 0.0}
        return result

    def _compute_source_attribution(self) -> Dict[str, Any]:
        """Performance with vs without on-chain and sentiment features."""
        def _avg_pnl(trades):
            if not trades:
                return 0.0
            return round(float(np.mean([t["pnl"] for t in trades])), 4)

        return {
            "with_onchain": {
                "trades": len(self._with_onchain_trades),
                "avg_pnl": _avg_pnl(self._with_onchain_trades),
            },
            "without_onchain": {
                "trades": len(self._without_onchain_trades),
                "avg_pnl": _avg_pnl(self._without_onchain_trades),
            },
            "with_sentiment": {
                "trades": len(self._with_sentiment_trades),
                "avg_pnl": _avg_pnl(self._with_sentiment_trades),
            },
            "without_sentiment": {
                "trades": len(self._without_sentiment_trades),
                "avg_pnl": _avg_pnl(self._without_sentiment_trades),
            },
        }

    @staticmethod
    def _compute_signal_accuracy(signals: List[dict]) -> Dict[str, Any]:
        """Compute accuracy for a specific on-chain signal."""
        completed = [s for s in signals if s.get("actual_correct") is not None]
        if not completed:
            return {"total": 0, "accuracy": 0.0}
        correct = sum(1 for s in completed if s["actual_correct"])
        return {
            "total": len(completed),
            "accuracy": round(correct / len(completed), 4),
            "avg_pnl": round(
                float(np.mean([s.get("pnl", 0) or 0 for s in completed])), 4
            ),
        }

    def to_state(self) -> dict:
        return {
            "predictions_count": len(self._predictions),
            "exchange_flow_signals_count": len(self._exchange_flow_signals),
            "mvrv_signals_count": len(self._mvrv_signals),
        }


# ---------------------------------------------------------------------------
# Go-Live Criteria (Section 9.4)
# ---------------------------------------------------------------------------

class MLGoLiveCriteria:
    """Evaluate go-live criteria for STRAT-010.

    Section 9.4 requirements:
    - 90-day paper trading
    - Minimum 80 completed trades
    - Win rate > 50%
    - Sharpe ratio > 0.7
    - AUC-ROC > 0.58
    - Walk-forward positive in 8/12 months
    """

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._trades: List[Dict[str, Any]] = []
        self._peak_equity: float = 0.0
        self._max_drawdown_pct: float = 0.0

    def start(self) -> None:
        self._start_time = time.time()

    def record_trade(self, pnl: float, pnl_pct: float, correct: bool) -> None:
        self._trades.append({
            "timestamp": time.time(),
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "correct": correct,
        })

    def update_equity(self, equity: float) -> None:
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            dd = ((self._peak_equity - equity) / self._peak_equity) * 100.0
            if dd > self._max_drawdown_pct:
                self._max_drawdown_pct = dd

    def evaluate(
        self,
        model_auc_roc: float = 0.0,
        model_accuracy: float = 0.0,
        walk_forward_positive: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate all go-live criteria."""
        elapsed_days = (time.time() - self._start_time) / 86400 if self._start_time > 0 else 0
        n_trades = len(self._trades)
        win_rate = 0.0
        sharpe = 0.0
        net_pnl = 0.0
        profit_factor = 0.0

        if self._trades:
            wins = sum(1 for t in self._trades if t["pnl"] > 0)
            win_rate = wins / n_trades if n_trades > 0 else 0.0
            net_pnl = sum(t["pnl"] for t in self._trades)

            # Sharpe
            returns = [t["pnl_pct"] for t in self._trades]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(365 * 24))

            # Profit factor
            gross_profit = sum(t["pnl"] for t in self._trades if t["pnl"] > 0)
            gross_loss = abs(sum(t["pnl"] for t in self._trades if t["pnl"] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        criteria = {
            "paper_days_90": {
                "required": 90,
                "actual": round(elapsed_days, 1),
                "pass": elapsed_days >= 90,
            },
            "min_80_trades": {
                "required": 80,
                "actual": n_trades,
                "pass": n_trades >= 80,
            },
            "net_positive_pnl": {
                "required": True,
                "actual": round(net_pnl, 2),
                "pass": net_pnl > 0,
            },
            "win_rate_above_50pct": {
                "required": 0.50,
                "actual": round(win_rate, 4),
                "pass": win_rate > 0.50,
            },
            "profit_factor_above_1_2": {
                "required": 1.2,
                "actual": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
                "pass": profit_factor > 1.2,
            },
            "sharpe_above_0_7": {
                "required": 0.7,
                "actual": round(sharpe, 4),
                "pass": sharpe > 0.7,
            },
            "max_drawdown_under_10pct": {
                "required": 10.0,
                "actual": round(self._max_drawdown_pct, 2),
                "pass": self._max_drawdown_pct < 10.0,
            },
            "model_accuracy_above_55pct": {
                "required": 0.55,
                "actual": round(model_accuracy, 4),
                "pass": model_accuracy > 0.55,
            },
            "auc_roc_above_0_58": {
                "required": 0.58,
                "actual": round(model_auc_roc, 4),
                "pass": model_auc_roc > 0.58,
            },
            "walk_forward_8_of_12": {
                "required": 8,
                "actual": walk_forward_positive,
                "pass": walk_forward_positive >= 8,
            },
        }

        all_pass = all(c["pass"] for c in criteria.values())

        return {
            "criteria": criteria,
            "overall_pass": all_pass,
            "ready_for_live": all_pass,
            "evaluation_time": datetime.now(timezone.utc).isoformat(),
        }

    def to_state(self) -> dict:
        return {
            "start_time": self._start_time,
            "trades_count": len(self._trades),
            "peak_equity": self._peak_equity,
            "max_drawdown_pct": self._max_drawdown_pct,
        }

    def load_state(self, state: dict) -> None:
        self._start_time = state.get("start_time", 0.0)
        self._peak_equity = state.get("peak_equity", 0.0)
        self._max_drawdown_pct = state.get("max_drawdown_pct", 0.0)
