"""Base dashboard server using FastAPI + Jinja2 + WebSocket.

Provides a real-time trading dashboard with:
- WebSocket push for live metrics, positions, and alerts
- REST API for strategy status, positions, trades, metrics, equity, alerts, config
- Health check endpoint for monitoring
- Kill switch endpoint for emergency shutdown
- Hot-reload config updates via POST /api/config

Usage in a strategy bot::

    from shared.dashboard_base import DashboardBase

    dashboard = DashboardBase("my_strat", "My Strategy", port=8080)
    dashboard.set_data_providers(
        positions_fn=lambda: bot.get_positions(),
        trades_fn=lambda limit: bot.get_recent_trades(limit),
        metrics_fn=lambda: bot.get_metrics(),
        equity_fn=lambda: bot.get_equity_curve(),
        alerts_fn=lambda: bot.get_alerts(),
        config_fn=lambda: bot.get_config(),
        kill_fn=lambda reason: bot.kill_switch.execute(reason),
    )
    await dashboard.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil
import uvicorn
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SENSITIVE_KEYS = frozenset({
    "api_key", "api_secret", "secret", "password", "token",
    "private_key", "webhook_secret", "telegram_token",
})

_WS_PUSH_INTERVAL = 2.0  # seconds


def _mask_sensitive(data: Any, depth: int = 0) -> Any:
    """Recursively mask sensitive fields in config data."""
    if depth > 10:
        return data
    if isinstance(data, dict):
        masked = {}
        for k, v in data.items():
            if k.lower() in _SENSITIVE_KEYS and isinstance(v, str) and len(v) > 4:
                masked[k] = v[:4] + "****"
            else:
                masked[k] = _mask_sensitive(v, depth + 1)
        return masked
    if isinstance(data, list):
        return [_mask_sensitive(item, depth + 1) for item in data]
    return data


def _safe_json(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if hasattr(obj, "__dict__"):
        return {k: _safe_json(v) for k, v in obj.__dict__.items()
                if not k.startswith("_")}
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# DashboardBase
# ---------------------------------------------------------------------------

class DashboardBase:
    """Base real-time trading dashboard.

    Parameters
    ----------
    strategy_id : str
        Short identifier, e.g. ``"funding_arb"``.
    strategy_name : str
        Human-readable name shown in the UI header.
    host : str
        Bind address. Defaults to ``"0.0.0.0"``.
    port : int
        Bind port. Defaults to ``8080``.
    template_dir : str | None
        Override the templates directory. Defaults to the ``templates/``
        folder next to this module.
    """

    def __init__(
        self,
        strategy_id: str,
        strategy_name: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        template_dir: Optional[str] = None,
    ) -> None:
        self.strategy_id = strategy_id
        self.strategy_name = strategy_name
        self.host = host
        self.port = port
        self._start_time = time.monotonic()
        self._start_dt = datetime.now(timezone.utc)
        self._server: Optional[uvicorn.Server] = None
        self._server_task: Optional[asyncio.Task] = None
        self._ws_clients: List[WebSocket] = []
        self._ws_task: Optional[asyncio.Task] = None

        # Data provider callbacks — set via set_data_providers()
        self._positions_fn: Optional[Callable] = None
        self._trades_fn: Optional[Callable] = None
        self._metrics_fn: Optional[Callable] = None
        self._equity_fn: Optional[Callable] = None
        self._alerts_fn: Optional[Callable] = None
        self._config_fn: Optional[Callable] = None
        self._kill_fn: Optional[Callable] = None
        self._config_update_fn: Optional[Callable] = None

        # Build FastAPI app
        if template_dir is None:
            template_dir = str(Path(__file__).parent / "templates")
        self._template_dir = template_dir
        self.app = self._create_app()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data_providers(
        self,
        positions_fn: Optional[Callable] = None,
        trades_fn: Optional[Callable] = None,
        metrics_fn: Optional[Callable] = None,
        equity_fn: Optional[Callable] = None,
        alerts_fn: Optional[Callable] = None,
        config_fn: Optional[Callable] = None,
        kill_fn: Optional[Callable] = None,
        config_update_fn: Optional[Callable] = None,
    ) -> None:
        """Register callbacks the dashboard uses to fetch live data.

        All callbacks are optional; endpoints return empty data when a
        callback is not registered.

        Parameters
        ----------
        positions_fn : callable
            ``() -> list[dict]``  Open positions.
        trades_fn : callable
            ``(limit: int) -> list[dict]``  Recent trades.
        metrics_fn : callable
            ``() -> dict``  Performance metrics.
        equity_fn : callable
            ``() -> list[dict]``  Equity curve points.
        alerts_fn : callable
            ``() -> list[dict]``  Recent alerts / notifications.
        config_fn : callable
            ``() -> dict``  Current configuration.
        kill_fn : callable
            ``(reason: str) -> Any``  Trigger emergency shutdown.
        config_update_fn : callable
            ``(params: dict) -> dict``  Hot-reload strategy params.
        """
        if positions_fn is not None:
            self._positions_fn = positions_fn
        if trades_fn is not None:
            self._trades_fn = trades_fn
        if metrics_fn is not None:
            self._metrics_fn = metrics_fn
        if equity_fn is not None:
            self._equity_fn = equity_fn
        if alerts_fn is not None:
            self._alerts_fn = alerts_fn
        if config_fn is not None:
            self._config_fn = config_fn
        if kill_fn is not None:
            self._kill_fn = kill_fn
        if config_update_fn is not None:
            self._config_update_fn = config_update_fn

    async def start(self) -> None:
        """Start the dashboard server in the background."""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(
            self._server.serve(), name=f"dashboard-{self.strategy_id}"
        )
        self._ws_task = asyncio.create_task(
            self._ws_broadcast_loop(), name=f"ws-broadcast-{self.strategy_id}"
        )
        logger.info(
            "Dashboard started at http://%s:%d for %s",
            self.host, self.port, self.strategy_name,
        )

    def stop(self) -> None:
        """Signal the server to shut down."""
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
        if self._server is not None:
            self._server.should_exit = True
        logger.info("Dashboard stop requested for %s", self.strategy_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def _uptime_str(self) -> str:
        secs = int(self._uptime_seconds())
        days, rem = divmod(secs, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        return " ".join(parts)

    def _memory_mb(self) -> float:
        try:
            proc = psutil.Process(os.getpid())
            return round(proc.memory_info().rss / (1024 * 1024), 1)
        except Exception:
            return 0.0

    async def _call_provider(self, fn: Optional[Callable], *args: Any) -> Any:
        """Call a data-provider callback, handling sync and async."""
        if fn is None:
            return None
        try:
            result = fn(*args)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception:
            logger.exception("Dashboard data-provider error")
            return None

    # ------------------------------------------------------------------
    # WebSocket broadcast
    # ------------------------------------------------------------------

    async def _ws_broadcast_loop(self) -> None:
        """Push updates to all connected WebSocket clients every N seconds."""
        while True:
            try:
                await asyncio.sleep(_WS_PUSH_INTERVAL)
                if not self._ws_clients:
                    continue

                payload = await self._build_ws_payload()
                message = json.dumps(_safe_json(payload))

                disconnected: List[WebSocket] = []
                for ws in self._ws_clients:
                    try:
                        await ws.send_text(message)
                    except Exception:
                        disconnected.append(ws)

                for ws in disconnected:
                    self._ws_clients.remove(ws)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("WebSocket broadcast error")
                await asyncio.sleep(5)

    async def _build_ws_payload(self) -> Dict[str, Any]:
        """Assemble the data packet pushed over WebSocket."""
        positions = await self._call_provider(self._positions_fn) or []
        metrics = await self._call_provider(self._metrics_fn) or {}
        alerts = await self._call_provider(self._alerts_fn) or []

        return {
            "type": "update",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime": self._uptime_str(),
            "memory_mb": self._memory_mb(),
            "positions": positions,
            "metrics": metrics,
            "alerts": alerts[-20:] if isinstance(alerts, list) else alerts,
            "positions_count": len(positions) if isinstance(positions, list) else 0,
        }

    # ------------------------------------------------------------------
    # App factory
    # ------------------------------------------------------------------

    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title=f"{self.strategy_name} Dashboard",
            docs_url=None,
            redoc_url=None,
        )
        templates = Jinja2Templates(directory=self._template_dir)

        # ---- Template pages ----

        @app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            positions = await self._call_provider(self._positions_fn) or []
            metrics = await self._call_provider(self._metrics_fn) or {}
            equity = await self._call_provider(self._equity_fn) or []
            alerts = await self._call_provider(self._alerts_fn) or []
            trades = await self._call_provider(self._trades_fn, 20) or []
            config = await self._call_provider(self._config_fn) or {}

            mode = "live"
            if isinstance(config, dict):
                if config.get("paper_trading") or config.get("testnet"):
                    mode = "paper"

            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "strategy_id": self.strategy_id,
                "strategy_name": self.strategy_name,
                "uptime": self._uptime_str(),
                "mode": mode,
                "positions": _safe_json(positions),
                "metrics": _safe_json(metrics),
                "equity": _safe_json(equity),
                "alerts": _safe_json(alerts[-30:]) if isinstance(alerts, list) else [],
                "trades": _safe_json(trades),
                "memory_mb": self._memory_mb(),
                "ws_url": f"ws://{self.host}:{self.port}/ws",
            })

        # ---- WebSocket ----

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self._ws_clients.append(ws)
            logger.info("WebSocket client connected (%d total)", len(self._ws_clients))
            try:
                while True:
                    # Keep alive — read pings / client messages
                    data = await ws.receive_text()
                    if data == "ping":
                        await ws.send_text(json.dumps({"type": "pong"}))
            except WebSocketDisconnect:
                pass
            finally:
                if ws in self._ws_clients:
                    self._ws_clients.remove(ws)
                logger.info(
                    "WebSocket client disconnected (%d remaining)",
                    len(self._ws_clients),
                )

        # ---- Health check ----

        @app.get("/health")
        async def health():
            return JSONResponse({
                "strategy_id": self.strategy_id,
                "status": "running",
                "uptime": self._uptime_seconds(),
                "uptime_str": self._uptime_str(),
                "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                "memory_mb": self._memory_mb(),
                "ws_clients": len(self._ws_clients),
            })

        # ---- Kill switch ----

        @app.post("/kill")
        async def kill(request: Request):
            body = await request.json() if request.headers.get("content-type") == "application/json" else {}
            reason = body.get("reason", "Dashboard kill switch triggered")
            if self._kill_fn is None:
                return JSONResponse(
                    {"error": "Kill switch not configured"},
                    status_code=503,
                )
            try:
                result = await self._call_provider(self._kill_fn, reason)
                logger.critical("KILL SWITCH activated via dashboard: %s", reason)
                return JSONResponse({
                    "status": "killed",
                    "reason": reason,
                    "result": _safe_json(result),
                })
            except Exception as exc:
                logger.exception("Kill switch failed")
                return JSONResponse(
                    {"error": str(exc), "traceback": traceback.format_exc()},
                    status_code=500,
                )

        # ---- REST API ----

        @app.get("/api/status")
        async def api_status():
            positions = await self._call_provider(self._positions_fn) or []
            metrics = await self._call_provider(self._metrics_fn) or {}
            return JSONResponse(_safe_json({
                "strategy_id": self.strategy_id,
                "strategy_name": self.strategy_name,
                "status": "running",
                "uptime": self._uptime_str(),
                "memory_mb": self._memory_mb(),
                "positions_count": len(positions) if isinstance(positions, list) else 0,
                "positions": positions,
                "metrics_summary": {
                    "total_pnl": metrics.get("total_pnl", 0),
                    "win_rate": metrics.get("win_rate", 0),
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                    "max_drawdown": metrics.get("max_drawdown", 0),
                    "total_trades": metrics.get("total_trades", 0),
                },
            }))

        @app.get("/api/positions")
        async def api_positions():
            positions = await self._call_provider(self._positions_fn) or []
            return JSONResponse(_safe_json(positions))

        @app.get("/api/trades")
        async def api_trades(limit: int = Query(50, ge=1, le=1000)):
            trades = await self._call_provider(self._trades_fn, limit) or []
            return JSONResponse(_safe_json(trades))

        @app.get("/api/metrics")
        async def api_metrics():
            metrics = await self._call_provider(self._metrics_fn) or {}
            return JSONResponse(_safe_json(metrics))

        @app.get("/api/equity")
        async def api_equity():
            equity = await self._call_provider(self._equity_fn) or []
            return JSONResponse(_safe_json(equity))

        @app.get("/api/alerts")
        async def api_alerts():
            alerts = await self._call_provider(self._alerts_fn) or []
            return JSONResponse(_safe_json(alerts))

        @app.get("/api/config")
        async def api_config():
            config = await self._call_provider(self._config_fn) or {}
            return JSONResponse(_safe_json(_mask_sensitive(config)))

        @app.post("/api/config")
        async def api_config_update(request: Request):
            if self._config_update_fn is None:
                return JSONResponse(
                    {"error": "Config update not supported"},
                    status_code=501,
                )
            try:
                body = await request.json()
                params = body.get("strategy_params", body)
                result = await self._call_provider(self._config_update_fn, params)
                logger.info("Config updated via dashboard: %s", list(params.keys()))
                return JSONResponse(_safe_json({
                    "status": "updated",
                    "applied_params": list(params.keys()),
                    "result": result,
                    "note": "Changes take effect on next trade cycle.",
                }))
            except Exception as exc:
                logger.exception("Config update failed")
                return JSONResponse(
                    {"error": str(exc)},
                    status_code=400,
                )

        return app
