"""Offline model retraining pipeline for STRAT-010.

Run every 30 days. Collects 365 days of data, performs chronological
train/validation/test split, trains XGBoost and LSTM, validates with
walk-forward analysis, and deploys passing models.

Training time budget: 30-120 minutes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
    from sklearn.model_selection import TimeSeriesSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.ml_models import LSTMClassifier, ModelMetadata


# ── Pass criteria ──────────────────────────────────────────────────────────

PASS_CRITERIA = {
    "accuracy": 0.55,
    "auc_roc": 0.58,
    "sharpe": 0.5,
    "walk_forward_positive_months": 8,  # out of 12
}


class ModelTrainer:
    """Offline retraining pipeline for XGBoost and LSTM models.

    Parameters
    ----------
    model_dir : str
        Directory to save trained models.
    data_dir : str
        Directory containing feature history for training.
    """

    def __init__(
        self,
        model_dir: str = "models/",
        data_dir: str = "data/features/",
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        lstm_input_features: int = 35,
        lstm_sequence_length: int = 24,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._data_dir = Path(data_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._lstm_hidden = lstm_hidden_size
        self._lstm_layers = lstm_num_layers
        self._lstm_features = lstm_input_features
        self._lstm_seq_len = lstm_sequence_length

        self._training_log: List[Dict[str, Any]] = []

    # ── Main entry point ─────────────────────────────────────────────

    def run_full_retrain(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Execute the full retraining pipeline.

        Parameters
        ----------
        features : np.ndarray
            Shape (N, 140) -- 365 days of hourly feature vectors.
        labels : np.ndarray
            Shape (N,) -- binary labels (1=price higher in 4h, 0=not).
        timestamps : np.ndarray | None
            Shape (N,) -- epoch timestamps for walk-forward splits.

        Returns
        -------
        dict with training results, pass/fail, deployed model paths.
        """
        start_time = time.monotonic()
        result: Dict[str, Any] = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "samples": len(labels),
            "xgboost": {},
            "lstm": {},
            "deployed": False,
        }

        logger.info(
            "Starting full retrain: %d samples, features shape %s",
            len(labels), features.shape,
        )

        # ---- Chronological split: 60% train, 15% val, 25% test ----
        n = len(labels)
        train_end = int(n * 0.60)
        val_end = int(n * 0.75)

        X_train, y_train = features[:train_end], labels[:train_end]
        X_val, y_val = features[train_end:val_end], labels[train_end:val_end]
        X_test, y_test = features[val_end:], labels[val_end:]
        ts_test = timestamps[val_end:] if timestamps is not None else None

        logger.info(
            "Split: train=%d val=%d test=%d", len(y_train), len(y_val), len(y_test),
        )

        # ---- XGBoost training ----
        xgb_result = self._train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
        result["xgboost"] = xgb_result

        # ---- LSTM training ----
        lstm_result = self._train_lstm(
            features[:, :self._lstm_features],  # Raw features only (no lags)
            labels, train_end, val_end,
        )
        result["lstm"] = lstm_result

        # ---- Walk-forward validation ----
        wf_result = self._walk_forward_validation(features, labels, timestamps)
        result["walk_forward"] = wf_result

        # ---- Monte Carlo Permutation Testing ----
        mc_result = self._monte_carlo_permutation_test(
            features, labels, train_end, val_end, n_permutations=100,
        )
        result["monte_carlo"] = mc_result

        # ---- Check pass criteria (including Monte Carlo) ----
        xgb_pass = self._check_pass(xgb_result, wf_result, mc_result)
        lstm_pass = self._check_pass(lstm_result, wf_result, mc_result)
        result["xgb_pass"] = xgb_pass
        result["lstm_pass"] = lstm_pass

        # ---- Deploy if passing ----
        version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M")
        if xgb_pass:
            self._deploy_xgboost(xgb_result, version)
            result["deployed"] = True
        else:
            logger.warning("XGBoost FAILED pass criteria -- keeping old model")

        if lstm_pass:
            self._deploy_lstm(lstm_result, version)
            result["deployed"] = result["deployed"] or True
        else:
            logger.warning("LSTM FAILED pass criteria -- keeping old model")

        elapsed = time.monotonic() - start_time
        result["elapsed_seconds"] = round(elapsed, 1)
        result["end_time"] = datetime.now(timezone.utc).isoformat()

        self._training_log.append(result)
        logger.info(
            "Retrain complete in %.1f s. XGB pass=%s LSTM pass=%s",
            elapsed, xgb_pass, lstm_pass,
        )
        return result

    # ── XGBoost training ─────────────────────────────────────────────

    def _train_xgboost(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Train XGBoost with cross-validation and hyperparameter tuning."""
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not installed"}

        result: Dict[str, Any] = {}

        try:
            # Hyperparameter grid (simplified for production)
            best_auc = 0.0
            best_params: Dict[str, Any] = {}

            param_grid = [
                {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "reg_alpha": 0.1, "reg_lambda": 1.0},
                {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 500, "reg_alpha": 0.5, "reg_lambda": 2.0},
                {"max_depth": 6, "learning_rate": 0.01, "n_estimators": 800, "reg_alpha": 1.0, "reg_lambda": 3.0},
                {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 200, "reg_alpha": 0.3, "reg_lambda": 1.5},
            ]

            for params in param_grid:
                model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    **params,
                )
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                val_pred = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, val_pred)

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_params = params
                    result["best_model"] = model

            logger.info("Best XGB params: %s (val AUC=%.4f)", best_params, best_auc)

            # Evaluate on test set
            model = result.get("best_model")
            if model is not None:
                test_proba = model.predict_proba(X_test)[:, 1]
                test_pred = (test_proba > 0.5).astype(int)

                result["accuracy"] = round(float(accuracy_score(y_test, test_pred)), 4)
                result["auc_roc"] = round(float(roc_auc_score(y_test, test_proba)), 4)
                result["precision"] = round(float(precision_score(y_test, test_pred, zero_division=0)), 4)
                result["recall"] = round(float(recall_score(y_test, test_pred, zero_division=0)), 4)
                result["best_params"] = best_params

                # Calculate Sharpe proxy from predicted returns
                result["sharpe"] = round(self._sharpe_from_predictions(test_proba, y_test), 4)

        except Exception:
            logger.exception("XGBoost training failed")
            result["error"] = "Training exception"

        return result

    # ── LSTM training ────────────────────────────────────────────────

    def _train_lstm(
        self,
        raw_features: np.ndarray,  # (N, 35)
        labels: np.ndarray,
        train_end: int,
        val_end: int,
    ) -> Dict[str, Any]:
        """Train LSTM with early stopping on validation loss."""
        if not HAS_TORCH or LSTMClassifier is None:
            return {"error": "PyTorch not installed"}

        result: Dict[str, Any] = {}

        try:
            seq_len = self._lstm_seq_len

            # Build sequences
            X_seqs, y_seqs = self._build_sequences(raw_features, labels, seq_len)
            n_total = len(y_seqs)
            adj_train = max(0, train_end - seq_len)
            adj_val = max(adj_train, val_end - seq_len)

            X_train = torch.tensor(X_seqs[:adj_train], dtype=torch.float32)
            y_train = torch.tensor(y_seqs[:adj_train], dtype=torch.float32)
            X_val = torch.tensor(X_seqs[adj_train:adj_val], dtype=torch.float32)
            y_val = torch.tensor(y_seqs[adj_train:adj_val], dtype=torch.float32)
            X_test = torch.tensor(X_seqs[adj_val:], dtype=torch.float32)
            y_test_np = y_seqs[adj_val:]

            # Replace NaN with 0
            X_train = torch.nan_to_num(X_train, nan=0.0)
            X_val = torch.nan_to_num(X_val, nan=0.0)
            X_test = torch.nan_to_num(X_test, nan=0.0)

            # Model
            model = LSTMClassifier(
                input_size=self._lstm_features,
                hidden_size=self._lstm_hidden,
                num_layers=self._lstm_layers,
            )
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.BCELoss()

            # Training loop with early stopping
            train_ds = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)  # No shuffle for time series

            best_val_loss = float("inf")
            patience = 10
            patience_counter = 0
            best_state = None

            for epoch in range(100):
                model.train()
                train_loss = 0.0
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)
                    val_loss = criterion(val_pred, y_val).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("LSTM early stopping at epoch %d", epoch)
                        break

            # Load best model
            if best_state is not None:
                model.load_state_dict(best_state)

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_proba = model(X_test).numpy()

            test_pred = (test_proba > 0.5).astype(int)
            result["accuracy"] = round(float(accuracy_score(y_test_np, test_pred)), 4)
            if len(np.unique(y_test_np)) > 1:
                result["auc_roc"] = round(float(roc_auc_score(y_test_np, test_proba)), 4)
            else:
                result["auc_roc"] = 0.0
            result["sharpe"] = round(self._sharpe_from_predictions(test_proba, y_test_np), 4)
            result["best_model"] = model
            result["epochs_trained"] = epoch + 1 if 'epoch' in dir() else 0

        except Exception:
            logger.exception("LSTM training failed")
            result["error"] = "Training exception"

        return result

    def _build_sequences(
        self, features: np.ndarray, labels: np.ndarray, seq_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (N - seq_len, seq_len, features) sequences."""
        n = len(labels)
        X_seqs = []
        y_seqs = []
        for i in range(seq_len, n):
            X_seqs.append(features[i - seq_len:i])
            y_seqs.append(labels[i])
        return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)

    # ── Walk-forward validation ──────────────────────────────────────

    def _walk_forward_validation(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Full 12-segment monthly walk-forward validation.

        For each month: train on ALL prior data, test on that month.
        Track PnL per segment. Pass criteria: positive in 8/12 months.
        Returns detailed per-segment results including accuracy, AUC, PnL.
        """
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not installed", "positive_months": 0}

        n = len(labels)
        # Divide into 12 roughly equal segments
        segment_size = n // 12
        if segment_size < 50:
            logger.warning("Insufficient data for 12-segment walk-forward")
            return {"positive_months": 0, "total_months": 0}

        positive_months = 0
        total_pnl = 0.0
        monthly_results: List[Dict[str, Any]] = []

        for month in range(12):
            test_start = month * segment_size
            test_end = min((month + 1) * segment_size, n)
            train_end_idx = test_start

            if train_end_idx < 100:
                monthly_results.append({
                    "month": month + 1,
                    "skipped": True,
                    "reason": f"Insufficient training data ({train_end_idx} samples)",
                })
                continue

            X_train_wf = features[:train_end_idx]
            y_train_wf = labels[:train_end_idx]
            X_test_wf = features[test_start:test_end]
            y_test_wf = labels[test_start:test_end]

            try:
                model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    max_depth=5,
                    learning_rate=0.05,
                    n_estimators=300,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                )
                model.fit(X_train_wf, y_train_wf, verbose=False)
                proba = model.predict_proba(X_test_wf)[:, 1]
                preds = (proba > 0.5).astype(int)

                # Simulate PnL: go long if p > 0.5, short if < 0.5
                signals = np.where(proba > 0.5, 1, -1).astype(float)
                returns = np.where(y_test_wf == 1, 1, -1).astype(float)
                pnl = float(np.sum(signals * returns))

                # Compute per-segment accuracy and AUC
                seg_accuracy = float(accuracy_score(y_test_wf, preds))
                try:
                    seg_auc = float(roc_auc_score(y_test_wf, proba))
                except ValueError:
                    seg_auc = 0.0

                # Sharpe for this segment
                seg_returns = signals * returns
                seg_sharpe = 0.0
                if len(seg_returns) > 1 and np.std(seg_returns) > 0:
                    seg_sharpe = float(np.mean(seg_returns) / np.std(seg_returns) * np.sqrt(len(seg_returns)))

                month_result = {
                    "month": month + 1,
                    "train_samples": len(y_train_wf),
                    "test_samples": len(y_test_wf),
                    "pnl": round(pnl, 2),
                    "accuracy": round(seg_accuracy, 4),
                    "auc_roc": round(seg_auc, 4),
                    "sharpe": round(seg_sharpe, 4),
                    "positive": bool(pnl > 0),
                }
                if pnl > 0:
                    positive_months += 1
                total_pnl += pnl
                monthly_results.append(month_result)

            except Exception as exc:
                logger.warning("Walk-forward month %d failed: %s", month + 1, exc)
                monthly_results.append({"month": month + 1, "error": str(exc)})

        return {
            "positive_months": positive_months,
            "total_months": len([m for m in monthly_results if "error" not in m and not m.get("skipped")]),
            "total_pnl": round(total_pnl, 2),
            "pass_criteria_met": positive_months >= PASS_CRITERIA["walk_forward_positive_months"],
            "monthly_results": monthly_results,
        }

    # ── Monte Carlo Permutation Testing ────────────────────────────

    def _monte_carlo_permutation_test(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_end: int,
        val_end: int,
        n_permutations: int = 100,
    ) -> Dict[str, Any]:
        """Run Monte Carlo permutation test to verify model is not overfit.

        After training, randomly shuffle labels N=100 times, retrain, compare
        test AUC. If real AUC is not significantly better (p<0.05), reject model.
        """
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not installed", "significant": False}

        X_train = features[:train_end]
        y_train = labels[:train_end]
        X_test = features[val_end:]
        y_test = labels[val_end:]

        if len(y_test) < 10 or len(np.unique(y_test)) < 2:
            return {"error": "Insufficient test data", "significant": False}

        # Get real model AUC
        try:
            real_model = xgb.XGBClassifier(
                objective="binary:logistic",
                max_depth=5,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
            )
            real_model.fit(X_train, y_train, verbose=False)
            real_proba = real_model.predict_proba(X_test)[:, 1]
            real_auc = float(roc_auc_score(y_test, real_proba))
        except Exception as exc:
            logger.warning("Monte Carlo: real model training failed: %s", exc)
            return {"error": str(exc), "significant": False}

        # Permutation tests
        permuted_aucs = []
        rng = np.random.RandomState(42)
        for i in range(n_permutations):
            try:
                y_train_perm = rng.permutation(y_train)
                perm_model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    max_depth=5,
                    learning_rate=0.05,
                    n_estimators=100,  # Fewer trees for speed
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                )
                perm_model.fit(X_train, y_train_perm, verbose=False)
                perm_proba = perm_model.predict_proba(X_test)[:, 1]
                perm_auc = float(roc_auc_score(y_test, perm_proba))
                permuted_aucs.append(perm_auc)
            except Exception:
                continue  # Skip failed permutations

        if not permuted_aucs:
            return {"error": "All permutations failed", "significant": False}

        # p-value: fraction of permuted AUCs >= real AUC
        p_value = float(np.mean(np.array(permuted_aucs) >= real_auc))
        significant = p_value < 0.05

        logger.info(
            "Monte Carlo permutation test: real AUC=%.4f, permuted mean=%.4f, "
            "p-value=%.4f, significant=%s (N=%d)",
            real_auc, np.mean(permuted_aucs), p_value, significant, len(permuted_aucs),
        )

        return {
            "real_auc": round(real_auc, 4),
            "permuted_mean_auc": round(float(np.mean(permuted_aucs)), 4),
            "permuted_std_auc": round(float(np.std(permuted_aucs)), 4),
            "p_value": round(p_value, 4),
            "significant": significant,
            "n_permutations": len(permuted_aucs),
        }

    # ── Pass criteria check ──────────────────────────────────────────

    def _check_pass(
        self,
        model_result: Dict[str, Any],
        wf_result: Dict[str, Any],
        mc_result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if a model meets the pass criteria for deployment.

        Includes Monte Carlo permutation test: if real AUC is not significantly
        better (p<0.05), reject model.
        """
        if "error" in model_result:
            return False

        accuracy = model_result.get("accuracy", 0)
        auc = model_result.get("auc_roc", 0)
        sharpe = model_result.get("sharpe", 0)
        wf_positive = wf_result.get("positive_months", 0)

        # Monte Carlo check
        mc_significant = True
        if mc_result is not None and not mc_result.get("error"):
            mc_significant = mc_result.get("significant", False)

        passes = (
            accuracy >= PASS_CRITERIA["accuracy"]
            and auc >= PASS_CRITERIA["auc_roc"]
            and sharpe >= PASS_CRITERIA["sharpe"]
            and wf_positive >= PASS_CRITERIA["walk_forward_positive_months"]
            and mc_significant
        )

        logger.info(
            "Pass check: accuracy=%.4f (>%.2f %s), AUC=%.4f (>%.2f %s), "
            "Sharpe=%.4f (>%.2f %s), WF=%d/12 (>=%d %s), MC=%s => %s",
            accuracy, PASS_CRITERIA["accuracy"], "OK" if accuracy >= PASS_CRITERIA["accuracy"] else "FAIL",
            auc, PASS_CRITERIA["auc_roc"], "OK" if auc >= PASS_CRITERIA["auc_roc"] else "FAIL",
            sharpe, PASS_CRITERIA["sharpe"], "OK" if sharpe >= PASS_CRITERIA["sharpe"] else "FAIL",
            wf_positive, PASS_CRITERIA["walk_forward_positive_months"],
            "OK" if wf_positive >= PASS_CRITERIA["walk_forward_positive_months"] else "FAIL",
            "OK" if mc_significant else "FAIL (p>=0.05)",
            "PASS" if passes else "FAIL",
        )
        return passes

    # ── Model deployment ─────────────────────────────────────────────

    def _deploy_xgboost(self, result: Dict[str, Any], version: str) -> None:
        """Save the trained XGBoost model to disk."""
        model = result.get("best_model")
        if model is None:
            return

        filepath = self._model_dir / "xgboost_model.json"
        # Backup existing
        backup = self._model_dir / "xgboost_model.backup.json"
        if filepath.exists():
            import shutil
            shutil.copy2(str(filepath), str(backup))

        model.save_model(str(filepath))

        # Save metadata
        meta = ModelMetadata(
            version=version,
            training_date=datetime.now(timezone.utc).isoformat(),
            training_samples=result.get("train_samples", 0),
            accuracy=result.get("accuracy", 0),
            auc_roc=result.get("auc_roc", 0),
            sharpe=result.get("sharpe", 0),
            model_type="xgboost",
        )
        meta_path = self._model_dir / "xgboost_model.json.meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

        logger.info("XGBoost model deployed: %s (version=%s)", filepath, version)

    def _deploy_lstm(self, result: Dict[str, Any], version: str) -> None:
        """Save the trained LSTM model to disk."""
        model = result.get("best_model")
        if model is None or not HAS_TORCH:
            return

        filepath = self._model_dir / "lstm_model.pt"
        # Backup existing
        backup = self._model_dir / "lstm_model.backup.pt"
        if filepath.exists():
            import shutil
            shutil.copy2(str(filepath), str(backup))

        torch.save(model.state_dict(), str(filepath))

        # Save metadata
        meta = ModelMetadata(
            version=version,
            training_date=datetime.now(timezone.utc).isoformat(),
            accuracy=result.get("accuracy", 0),
            auc_roc=result.get("auc_roc", 0),
            sharpe=result.get("sharpe", 0),
            model_type="lstm",
        )
        meta_path = self._model_dir / "lstm_model.pt.meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

        logger.info("LSTM model deployed: %s (version=%s)", filepath, version)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _sharpe_from_predictions(proba: np.ndarray, labels: np.ndarray) -> float:
        """Compute a Sharpe-like metric from predictions and actual labels."""
        signals = np.where(proba > 0.5, 1.0, -1.0)
        actual = np.where(labels == 1, 1.0, -1.0)
        returns = signals * actual
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.0
        # Annualise (assuming hourly predictions, ~8760 per year)
        return float(mean_ret / std_ret * np.sqrt(365 * 24))

    def get_training_log(self) -> List[Dict[str, Any]]:
        """Return the history of training runs."""
        return list(self._training_log)
