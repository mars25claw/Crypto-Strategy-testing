"""Dashboard for STRAT-010 — ML & On-Chain Quantitative Models.

Extends the shared DashboardBase with ML-specific panels:
- Model accuracy gauge
- Prediction distribution histogram
- Feature importance bar chart
- Confidence vs outcome scatter
- On-chain metrics panel
- Data source health indicators
- Prediction timeline on price chart
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from shared.dashboard_base import DashboardBase

logger = logging.getLogger(__name__)


class MLDashboard(DashboardBase):
    """STRAT-010 ML dashboard with model-specific data providers.

    Parameters
    ----------
    host : str
        Bind address.
    port : int
        Bind port.
    template_dir : str | None
        Path to dashboard HTML templates.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8089,
        template_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            strategy_id="STRAT-010",
            strategy_name="ML & On-Chain Quantitative Models",
            host=host,
            port=port,
            template_dir=template_dir,
        )

        # ML-specific data providers
        self._model_status_fn: Optional[Callable] = None
        self._feature_importance_fn: Optional[Callable] = None
        self._prediction_dist_fn: Optional[Callable] = None
        self._prediction_history_fn: Optional[Callable] = None
        self._risk_status_fn: Optional[Callable] = None
        self._onchain_status_fn: Optional[Callable] = None
        self._data_source_health_fn: Optional[Callable] = None
        self._confidence_accuracy_fn: Optional[Callable] = None
        self._ensemble_agreement_fn: Optional[Callable] = None
        self._feature_engines_status_fn: Optional[Callable] = None
        self._fallback_status_fn: Optional[Callable] = None
        self._strategy_metrics_fn: Optional[Callable] = None
        self._go_live_fn: Optional[Callable] = None
        self._concept_drift_fn: Optional[Callable] = None

        # Register additional API endpoints
        self._register_ml_endpoints()

    def set_ml_providers(
        self,
        model_status_fn: Optional[Callable] = None,
        feature_importance_fn: Optional[Callable] = None,
        prediction_dist_fn: Optional[Callable] = None,
        prediction_history_fn: Optional[Callable] = None,
        risk_status_fn: Optional[Callable] = None,
        onchain_status_fn: Optional[Callable] = None,
        data_source_health_fn: Optional[Callable] = None,
        confidence_accuracy_fn: Optional[Callable] = None,
        ensemble_agreement_fn: Optional[Callable] = None,
        feature_engines_status_fn: Optional[Callable] = None,
        fallback_status_fn: Optional[Callable] = None,
    ) -> None:
        """Register ML-specific data provider callbacks."""
        if model_status_fn:
            self._model_status_fn = model_status_fn
        if feature_importance_fn:
            self._feature_importance_fn = feature_importance_fn
        if prediction_dist_fn:
            self._prediction_dist_fn = prediction_dist_fn
        if prediction_history_fn:
            self._prediction_history_fn = prediction_history_fn
        if risk_status_fn:
            self._risk_status_fn = risk_status_fn
        if onchain_status_fn:
            self._onchain_status_fn = onchain_status_fn
        if data_source_health_fn:
            self._data_source_health_fn = data_source_health_fn
        if confidence_accuracy_fn:
            self._confidence_accuracy_fn = confidence_accuracy_fn
        if ensemble_agreement_fn:
            self._ensemble_agreement_fn = ensemble_agreement_fn
        if feature_engines_status_fn:
            self._feature_engines_status_fn = feature_engines_status_fn
        if fallback_status_fn:
            self._fallback_status_fn = fallback_status_fn

    def set_metrics_providers(
        self,
        strategy_metrics_fn: Optional[Callable] = None,
        go_live_fn: Optional[Callable] = None,
        concept_drift_fn: Optional[Callable] = None,
    ) -> None:
        """Register metrics-specific data provider callbacks."""
        if strategy_metrics_fn:
            self._strategy_metrics_fn = strategy_metrics_fn
        if go_live_fn:
            self._go_live_fn = go_live_fn
        if concept_drift_fn:
            self._concept_drift_fn = concept_drift_fn

    def _register_ml_endpoints(self) -> None:
        """Add ML-specific REST endpoints to the FastAPI app."""
        from fastapi.responses import JSONResponse

        @self.app.get("/api/model")
        async def api_model():
            data = await self._call_provider(self._model_status_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/features/importance")
        async def api_feature_importance():
            data = await self._call_provider(self._feature_importance_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/predictions/distribution")
        async def api_prediction_dist():
            data = await self._call_provider(self._prediction_dist_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/predictions/history")
        async def api_prediction_history():
            data = await self._call_provider(self._prediction_history_fn)
            return JSONResponse(data or [])

        @self.app.get("/api/risk")
        async def api_risk():
            data = await self._call_provider(self._risk_status_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/onchain")
        async def api_onchain():
            data = await self._call_provider(self._onchain_status_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/datasources")
        async def api_datasources():
            data = await self._call_provider(self._data_source_health_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/confidence")
        async def api_confidence():
            data = await self._call_provider(self._confidence_accuracy_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/ensemble")
        async def api_ensemble():
            data = await self._call_provider(self._ensemble_agreement_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/fallback")
        async def api_fallback():
            data = await self._call_provider(self._fallback_status_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/strategy-metrics")
        async def api_strategy_metrics():
            data = await self._call_provider(self._strategy_metrics_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/go-live")
        async def api_go_live():
            data = await self._call_provider(self._go_live_fn)
            return JSONResponse(data or {})

        @self.app.get("/api/concept-drift")
        async def api_concept_drift():
            data = await self._call_provider(self._concept_drift_fn)
            return JSONResponse(data or {})

    async def _build_ws_payload(self) -> Dict[str, Any]:
        """Override to include ML-specific data in WebSocket pushes."""
        base = await super()._build_ws_payload()

        # Append ML data
        model = await self._call_provider(self._model_status_fn)
        risk = await self._call_provider(self._risk_status_fn)

        base["model"] = model or {}
        base["risk"] = risk or {}
        base["strategy_type"] = "ml_onchain"

        return base
