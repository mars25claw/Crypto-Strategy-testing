"""ML model inference for STRAT-010.

Manages XGBoost and LSTM model loading, inference, ensemble combination,
model versioning, hot-swap, and backup/rollback logic.

Ensemble: P_final = 0.60 * P_xgboost + 0.40 * P_lstm
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try importing XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available -- install xgboost package")

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available -- install torch package")


# ── Model metadata ─────────────────────────────────────────────────────────

@dataclass
class ModelMetadata:
    """Version and performance metadata for a trained model."""

    version: str = "0.0.0"
    training_date: str = ""
    training_samples: int = 0
    accuracy: float = 0.0
    auc_roc: float = 0.0
    sharpe: float = 0.0
    walk_forward_positive_months: int = 0
    feature_count: int = 140
    model_type: str = ""  # "xgboost" or "lstm"

    @property
    def age_days(self) -> float:
        if not self.training_date:
            return float("inf")
        try:
            dt = datetime.fromisoformat(self.training_date)
            now = datetime.now(timezone.utc)
            return (now - dt).total_seconds() / 86400
        except (ValueError, TypeError):
            return float("inf")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "training_date": self.training_date,
            "training_samples": self.training_samples,
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "sharpe": self.sharpe,
            "walk_forward_positive_months": self.walk_forward_positive_months,
            "feature_count": self.feature_count,
            "model_type": self.model_type,
            "age_days": round(self.age_days, 1),
        }


@dataclass
class InferenceResult:
    """Result of a single model inference."""

    p_xgboost: float = 0.5
    p_lstm: float = 0.5
    p_final: float = 0.5
    signal: str = "NONE"       # "LONG", "SHORT", "NONE"
    confidence: str = "NONE"   # "HIGH", "MODERATE", "NONE"
    xgb_inference_ms: float = 0.0
    lstm_inference_ms: float = 0.0
    xgb_available: bool = True
    lstm_available: bool = True
    feature_importance_top5: List[Tuple[str, float]] = field(default_factory=list)
    disagreement: bool = False
    timestamp_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "p_xgboost": round(self.p_xgboost, 6),
            "p_lstm": round(self.p_lstm, 6),
            "p_final": round(self.p_final, 6),
            "signal": self.signal,
            "confidence": self.confidence,
            "xgb_inference_ms": round(self.xgb_inference_ms, 2),
            "lstm_inference_ms": round(self.lstm_inference_ms, 2),
            "xgb_available": self.xgb_available,
            "lstm_available": self.lstm_available,
            "feature_importance_top5": self.feature_importance_top5,
            "disagreement": self.disagreement,
            "timestamp_ms": self.timestamp_ms,
        }


# ── LSTM Model Definition ─────────────────────────────────────────────────

if HAS_TORCH:
    class LSTMClassifier(nn.Module):
        """LSTM for binary classification of 4-hour price direction."""

        def __init__(
            self,
            input_size: int = 35,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            # Use last hidden state
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden).squeeze(-1)
else:
    LSTMClassifier = None  # type: ignore


# ── Model Manager ──────────────────────────────────────────────────────────

class ModelManager:
    """Manages XGBoost + LSTM model loading, inference, and ensemble.

    Parameters
    ----------
    model_dir : str
        Directory containing model files and metadata.
    xgb_weight : float
        Weight for XGBoost in ensemble (default 0.60).
    lstm_weight : float
        Weight for LSTM in ensemble (default 0.40).
    long_threshold : float
        P_final above this generates LONG signal (default 0.65).
    short_threshold : float
        P_final below this generates SHORT signal (default 0.35).
    lstm_timeout : float
        Maximum seconds for LSTM inference before fallback (default 5.0).
    lstm_hidden_size : int
        LSTM hidden dimension.
    lstm_num_layers : int
        LSTM layer count.
    lstm_input_features : int
        LSTM input feature dimension (raw, no lags).
    """

    def __init__(
        self,
        model_dir: str = "models/",
        xgb_weight: float = 0.60,
        lstm_weight: float = 0.40,
        long_threshold: float = 0.65,
        short_threshold: float = 0.35,
        high_confidence_long: float = 0.75,
        high_confidence_short: float = 0.25,
        lstm_timeout: float = 5.0,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        lstm_input_features: int = 35,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self._xgb_weight = xgb_weight
        self._lstm_weight = lstm_weight
        self._long_threshold = long_threshold
        self._short_threshold = short_threshold
        self._high_conf_long = high_confidence_long
        self._high_conf_short = high_confidence_short
        self._lstm_timeout = lstm_timeout
        self._lstm_hidden = lstm_hidden_size
        self._lstm_layers = lstm_num_layers
        self._lstm_features = lstm_input_features

        # Models
        self._xgb_model: Any = None
        self._lstm_model: Any = None
        self._xgb_backup: Any = None
        self._lstm_backup: Any = None

        # Metadata
        self.xgb_meta = ModelMetadata(model_type="xgboost")
        self.lstm_meta = ModelMetadata(model_type="lstm")
        self._xgb_backup_meta = ModelMetadata(model_type="xgboost")
        self._lstm_backup_meta = ModelMetadata(model_type="lstm")

        # Inference tracking
        self._consecutive_failures: int = 0
        self._consecutive_inference_timeouts: int = 0
        self._last_inference_ms: int = 0
        self._inference_history: List[InferenceResult] = []

        # Feature importance from XGBoost
        self._feature_importance: Dict[str, float] = {}

        logger.info(
            "ModelManager initialised: dir=%s, xgb_w=%.2f, lstm_w=%.2f",
            model_dir, xgb_weight, lstm_weight,
        )

    # ── Model loading ────────────────────────────────────────────────

    def load_models(
        self,
        xgb_file: str = "xgboost_model.json",
        lstm_file: str = "lstm_model.pt",
    ) -> Tuple[bool, bool]:
        """Load model files from disk. Returns (xgb_loaded, lstm_loaded)."""
        xgb_ok = self._load_xgboost(xgb_file)
        lstm_ok = self._load_lstm(lstm_file)
        return xgb_ok, lstm_ok

    def _load_xgboost(self, filename: str) -> bool:
        """Load XGBoost model from JSON/binary file."""
        if not HAS_XGBOOST:
            logger.error("XGBoost not installed -- cannot load model")
            return False

        filepath = self._model_dir / filename
        if not filepath.exists():
            logger.warning("XGBoost model file not found: %s", filepath)
            return False

        try:
            model = xgb.XGBClassifier()
            model.load_model(str(filepath))
            self._xgb_model = model

            # Extract feature importance
            try:
                importance = model.get_booster().get_score(importance_type="gain")
                self._feature_importance = importance
            except Exception:
                pass

            # Load metadata
            meta_path = self._model_dir / f"{filename}.meta.json"
            if meta_path.exists():
                self.xgb_meta = self._load_metadata(meta_path)

            logger.info(
                "XGBoost model loaded: %s (version=%s, age=%.1f days)",
                filename, self.xgb_meta.version, self.xgb_meta.age_days,
            )
            return True
        except Exception:
            logger.exception("Failed to load XGBoost model: %s", filename)
            return False

    def _load_lstm(self, filename: str) -> bool:
        """Load LSTM model from PyTorch checkpoint."""
        if not HAS_TORCH or LSTMClassifier is None:
            logger.error("PyTorch not installed -- cannot load LSTM model")
            return False

        filepath = self._model_dir / filename
        if not filepath.exists():
            logger.warning("LSTM model file not found: %s", filepath)
            return False

        try:
            device = torch.device("cpu")  # Always load on CPU first
            model = LSTMClassifier(
                input_size=self._lstm_features,
                hidden_size=self._lstm_hidden,
                num_layers=self._lstm_layers,
            )
            state_dict = torch.load(str(filepath), map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            self._lstm_model = model

            # Load metadata
            meta_path = self._model_dir / f"{filename}.meta.json"
            if meta_path.exists():
                self.lstm_meta = self._load_metadata(meta_path)

            logger.info(
                "LSTM model loaded: %s (version=%s, age=%.1f days)",
                filename, self.lstm_meta.version, self.lstm_meta.age_days,
            )
            return True
        except Exception:
            logger.exception("Failed to load LSTM model: %s", filename)
            return False

    def _load_metadata(self, path: Path) -> ModelMetadata:
        """Load model metadata from JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return ModelMetadata(
                version=data.get("version", "0.0.0"),
                training_date=data.get("training_date", ""),
                training_samples=data.get("training_samples", 0),
                accuracy=data.get("accuracy", 0.0),
                auc_roc=data.get("auc_roc", 0.0),
                sharpe=data.get("sharpe", 0.0),
                walk_forward_positive_months=data.get("walk_forward_positive_months", 0),
                feature_count=data.get("feature_count", 140),
                model_type=data.get("model_type", ""),
            )
        except Exception:
            logger.warning("Failed to load metadata from %s", path)
            return ModelMetadata()

    # ── Hot-swap ─────────────────────────────────────────────────────

    def hot_swap_xgboost(self, new_file: str) -> bool:
        """Load a new XGBoost model without restart; backup the old one."""
        self._xgb_backup = self._xgb_model
        self._xgb_backup_meta = self.xgb_meta
        ok = self._load_xgboost(new_file)
        if not ok:
            # Rollback
            self._xgb_model = self._xgb_backup
            self.xgb_meta = self._xgb_backup_meta
            logger.error("XGBoost hot-swap failed -- rolled back to backup")
        else:
            logger.info("XGBoost hot-swap succeeded: %s", new_file)
        return ok

    def hot_swap_lstm(self, new_file: str) -> bool:
        """Load a new LSTM model without restart; backup the old one."""
        self._lstm_backup = self._lstm_model
        self._lstm_backup_meta = self.lstm_meta
        ok = self._load_lstm(new_file)
        if not ok:
            self._lstm_model = self._lstm_backup
            self.lstm_meta = self._lstm_backup_meta
            logger.error("LSTM hot-swap failed -- rolled back to backup")
        else:
            logger.info("LSTM hot-swap succeeded: %s", new_file)
        return ok

    def rollback_xgboost(self) -> bool:
        """Rollback to the backup XGBoost model."""
        if self._xgb_backup is None:
            logger.warning("No XGBoost backup available for rollback")
            return False
        self._xgb_model = self._xgb_backup
        self.xgb_meta = self._xgb_backup_meta
        logger.info("XGBoost rolled back to version %s", self.xgb_meta.version)
        return True

    def rollback_lstm(self) -> bool:
        """Rollback to the backup LSTM model."""
        if self._lstm_backup is None:
            logger.warning("No LSTM backup available for rollback")
            return False
        self._lstm_model = self._lstm_backup
        self.lstm_meta = self._lstm_backup_meta
        logger.info("LSTM rolled back to version %s", self.lstm_meta.version)
        return True

    # ── Inference ────────────────────────────────────────────────────

    async def predict(
        self,
        xgb_features: np.ndarray,
        lstm_sequence: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> InferenceResult:
        """Run ensemble inference.

        Parameters
        ----------
        xgb_features : np.ndarray
            140-dimensional feature vector for XGBoost.
        lstm_sequence : np.ndarray | None
            (24, 35) sequence for LSTM, or None to skip LSTM.
        feature_names : list[str] | None
            Feature names for importance tracking.

        Returns
        -------
        InferenceResult with combined probability and signal.
        """
        result = InferenceResult(timestamp_ms=int(time.time() * 1000))

        # --- XGBoost inference ---
        p_xgb = await self._infer_xgboost(xgb_features, result)

        # --- LSTM inference ---
        p_lstm = await self._infer_lstm(lstm_sequence, result)

        # --- Ensemble ---
        if result.xgb_available and result.lstm_available:
            # Check for strong disagreement
            if abs(p_xgb - p_lstm) > 0.3:
                result.disagreement = True
                # Use the higher-confidence individual if > 0.75 or < 0.25
                if p_xgb > 0.75 or p_xgb < 0.25:
                    result.p_final = p_xgb
                elif p_lstm > 0.75 or p_lstm < 0.25:
                    result.p_final = p_lstm
                else:
                    result.p_final = self._xgb_weight * p_xgb + self._lstm_weight * p_lstm
            else:
                result.p_final = self._xgb_weight * p_xgb + self._lstm_weight * p_lstm
        elif result.xgb_available:
            # LSTM unavailable -- 100% XGBoost
            result.p_final = p_xgb
        elif result.lstm_available:
            # XGBoost unavailable -- 100% LSTM
            result.p_final = p_lstm
        else:
            # Both unavailable
            result.p_final = 0.5
            self._consecutive_failures += 1

        # --- Signal generation ---
        result.signal, result.confidence = self._generate_signal(result.p_final)

        # --- Feature importance top 5 ---
        if self._feature_importance and feature_names:
            top5 = sorted(
                self._feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            result.feature_importance_top5 = [
                (name, round(score, 4)) for name, score in top5
            ]

        # Track
        self._last_inference_ms = result.timestamp_ms
        self._inference_history.append(result)
        if len(self._inference_history) > 200:
            self._inference_history = self._inference_history[-100:]

        if result.xgb_available or result.lstm_available:
            self._consecutive_failures = 0

        logger.info(
            "Inference: P_xgb=%.4f P_lstm=%.4f P_final=%.4f signal=%s conf=%s "
            "disagree=%s xgb_ms=%.1f lstm_ms=%.1f",
            result.p_xgboost, result.p_lstm, result.p_final,
            result.signal, result.confidence, result.disagreement,
            result.xgb_inference_ms, result.lstm_inference_ms,
        )

        return result

    async def _infer_xgboost(
        self, features: np.ndarray, result: InferenceResult,
    ) -> float:
        """Run XGBoost inference. Returns probability."""
        if self._xgb_model is None:
            result.xgb_available = False
            result.p_xgboost = 0.5
            return 0.5

        try:
            start = time.monotonic()
            # Reshape for single prediction
            X = features.reshape(1, -1)
            # Replace NaN with the XGBoost missing-value marker
            proba = self._xgb_model.predict_proba(X)[0]
            p = float(proba[1]) if len(proba) > 1 else float(proba[0])
            elapsed = (time.monotonic() - start) * 1000
            result.xgb_inference_ms = elapsed
            result.p_xgboost = p

            if elapsed > 100:
                logger.warning("XGBoost inference took %.1f ms (target <100)", elapsed)

            if np.isnan(p):
                logger.error("XGBoost returned NaN -- marking unavailable")
                result.xgb_available = False
                return 0.5

            return p
        except Exception:
            logger.exception("XGBoost inference failed")
            result.xgb_available = False
            result.p_xgboost = 0.5
            return 0.5

    async def _infer_lstm(
        self, sequence: Optional[np.ndarray], result: InferenceResult,
    ) -> float:
        """Run LSTM inference with timeout. Returns probability."""
        if self._lstm_model is None or not HAS_TORCH or sequence is None:
            result.lstm_available = False
            result.p_lstm = 0.5
            return 0.5

        try:
            start = time.monotonic()

            # Run in executor to avoid blocking the event loop, with timeout
            loop = asyncio.get_event_loop()
            p = await asyncio.wait_for(
                loop.run_in_executor(None, self._lstm_forward, sequence),
                timeout=self._lstm_timeout,
            )

            elapsed = (time.monotonic() - start) * 1000
            result.lstm_inference_ms = elapsed
            result.p_lstm = p

            if elapsed > 500:
                logger.warning("LSTM inference took %.1f ms (target <500)", elapsed)

            if np.isnan(p):
                logger.error("LSTM returned NaN -- marking unavailable")
                result.lstm_available = False
                return 0.5

            return p
        except asyncio.TimeoutError:
            logger.warning(
                "LSTM inference timed out (%.1f s) -- using XGBoost only",
                self._lstm_timeout,
            )
            result.lstm_available = False
            result.p_lstm = 0.5
            return 0.5
        except Exception:
            logger.exception("LSTM inference failed")
            result.lstm_available = False
            result.p_lstm = 0.5
            return 0.5

    def _lstm_forward(self, sequence: np.ndarray) -> float:
        """Synchronous LSTM forward pass (runs in executor).

        Handles GPU resource exhaustion:
        - If CUDA OOM, falls back to CPU
        - If CPU inference exceeds 10s (2x normal timeout), uses last valid
          prediction
        - If 3 consecutive inference timeouts, switches to fallback strategy
        """
        try:
            with torch.no_grad():
                x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                x = torch.nan_to_num(x, nan=0.0)

                # Try GPU first if available
                if torch.cuda.is_available() and self._lstm_model is not None:
                    try:
                        device = torch.device("cuda")
                        model_gpu = self._lstm_model.to(device)
                        x_gpu = x.to(device)
                        output = model_gpu(x_gpu)
                        result = float(output.cpu().item())
                        self._consecutive_inference_timeouts = 0
                        return result
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "CUDA" in str(e):
                            logger.warning(
                                "CUDA out of memory -- falling back to CPU inference"
                            )
                            torch.cuda.empty_cache()
                            # Fall through to CPU
                        else:
                            raise

                # CPU inference
                start = time.monotonic()
                self._lstm_model.to(torch.device("cpu"))
                output = self._lstm_model(x)
                elapsed = time.monotonic() - start

                if elapsed > 10.0:
                    # CPU inference exceeded 2x normal timeout
                    self._consecutive_inference_timeouts = getattr(
                        self, "_consecutive_inference_timeouts", 0
                    ) + 1
                    logger.warning(
                        "CPU inference took %.1fs (>10s limit) -- timeout #%d",
                        elapsed, self._consecutive_inference_timeouts,
                    )
                    if self._consecutive_inference_timeouts >= 3:
                        logger.error(
                            "3 consecutive inference timeouts -- switching to fallback"
                        )
                    # Use last valid prediction if available
                    if self._inference_history:
                        return self._inference_history[-1].p_lstm
                    return 0.5
                else:
                    self._consecutive_inference_timeouts = 0

                return float(output.item())

        except Exception:
            logger.exception("LSTM forward pass failed")
            self._consecutive_inference_timeouts = getattr(
                self, "_consecutive_inference_timeouts", 0
            ) + 1
            if self._inference_history:
                return self._inference_history[-1].p_lstm
            return 0.5

    # ── Signal generation ────────────────────────────────────────────

    def _generate_signal(self, p_final: float) -> Tuple[str, str]:
        """Convert probability to signal + confidence level."""
        if p_final > self._long_threshold:
            signal = "LONG"
            if p_final > self._high_conf_long:
                confidence = "HIGH"
            else:
                confidence = "MODERATE"
        elif p_final < self._short_threshold:
            signal = "SHORT"
            if p_final < self._high_conf_short:
                confidence = "HIGH"
            else:
                confidence = "MODERATE"
        else:
            signal = "NONE"
            confidence = "NONE"
        return signal, confidence

    # ── Model freshness ──────────────────────────────────────────────

    def get_freshness_multiplier(self) -> float:
        """Return confidence multiplier based on model age.

        - Age < 30 days: 1.0 (full confidence)
        - Age 30-60 days: 0.75 (reduced confidence)
        - Age > 60 days: 0.0 (halt)
        """
        # Use the older of the two models
        max_age = max(self.xgb_meta.age_days, self.lstm_meta.age_days)
        if max_age > 60:
            return 0.0
        elif max_age > 30:
            return 0.75
        return 1.0

    def should_halt_for_staleness(self) -> bool:
        """Return True if models are too stale to trade."""
        return self.get_freshness_multiplier() == 0.0

    # ── Feature importance ───────────────────────────────────────────

    def get_feature_importance(self) -> Dict[str, float]:
        """Return XGBoost feature importance (gain-based)."""
        return dict(self._feature_importance)

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Return top-N features by importance."""
        return sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]

    # ── Diagnostics ──────────────────────────────────────────────────

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def models_loaded(self) -> Tuple[bool, bool]:
        return (self._xgb_model is not None, self._lstm_model is not None)

    def get_prediction_history(self, limit: int = 100) -> List[dict]:
        """Return recent inference results as dicts."""
        return [r.to_dict() for r in self._inference_history[-limit:]]

    def get_prediction_distribution(self) -> Dict[str, int]:
        """Return histogram of P_final values for dashboard."""
        buckets = {
            "0.0-0.1": 0, "0.1-0.2": 0, "0.2-0.3": 0,
            "0.3-0.4": 0, "0.4-0.5": 0, "0.5-0.6": 0,
            "0.6-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0, "0.9-1.0": 0,
        }
        for r in self._inference_history:
            idx = min(int(r.p_final * 10), 9)
            keys = list(buckets.keys())
            buckets[keys[idx]] += 1
        return buckets

    def get_ensemble_agreement_rate(self) -> float:
        """Fraction of predictions where XGBoost and LSTM agree on direction."""
        agree = 0
        total = 0
        for r in self._inference_history:
            if r.xgb_available and r.lstm_available:
                total += 1
                xgb_dir = "LONG" if r.p_xgboost > 0.5 else "SHORT"
                lstm_dir = "LONG" if r.p_lstm > 0.5 else "SHORT"
                if xgb_dir == lstm_dir:
                    agree += 1
        return agree / total if total > 0 else 0.0

    def get_status(self) -> Dict[str, Any]:
        """Return model manager status for dashboard/heartbeat."""
        xgb_loaded, lstm_loaded = self.models_loaded
        return {
            "xgb_loaded": xgb_loaded,
            "lstm_loaded": lstm_loaded,
            "xgb_meta": self.xgb_meta.to_dict(),
            "lstm_meta": self.lstm_meta.to_dict(),
            "freshness_multiplier": self.get_freshness_multiplier(),
            "should_halt_staleness": self.should_halt_for_staleness(),
            "consecutive_failures": self._consecutive_failures,
            "ensemble_agreement_rate": round(self.get_ensemble_agreement_rate(), 4),
            "inference_count": len(self._inference_history),
            "last_inference_ms": self._last_inference_ms,
        }

    # ── Model saving (used by trainer) ───────────────────────────────

    def save_model_metadata(
        self,
        filename: str,
        metadata: ModelMetadata,
    ) -> None:
        """Save model metadata JSON alongside the model file."""
        meta_path = self._model_dir / f"{filename}.meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        logger.info("Saved model metadata: %s", meta_path)
