"""
Aggregator service for the Crypto Strategy Lab.

Polls all 10 strategy dashboards, computes a unified leaderboard,
and streams updates over WebSocket.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aggregator")

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
STRATEGIES: list[dict[str, Any]] = [
    {"id": "STRAT-001", "host": "strat-001-trend-following", "port": 8081},
    {"id": "STRAT-002", "host": "strat-002-funding-arb", "port": 8082},
    {"id": "STRAT-003", "host": "strat-003-stat-arb-pairs", "port": 8083},
    {"id": "STRAT-004", "host": "strat-004-mean-reversion", "port": 8084},
    {"id": "STRAT-005", "host": "strat-005-grid-trading", "port": 8085},
    {"id": "STRAT-006", "host": "strat-006-market-making", "port": 8086},
    {"id": "STRAT-007", "host": "strat-007-triangular-arb", "port": 8087},
    {"id": "STRAT-008", "host": "strat-008-options-vol", "port": 8088},
    {"id": "STRAT-009", "host": "strat-009-signal-dca", "port": 8089},
    {"id": "STRAT-010", "host": "strat-010-ml-onchain", "port": 8090},
]

STARTING_EQUITY_PER_STRATEGY = 800.0
TOTAL_STARTING_EQUITY = STARTING_EQUITY_PER_STRATEGY * len(STRATEGIES)
POLL_INTERVAL_S = 0.5
CRASH_THRESHOLD = 3  # consecutive misses before flagging as crashed

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class StrategyMetrics(BaseModel):
    strategy_id: str = ""
    strategy_name: str = ""
    equity: float = STARTING_EQUITY_PER_STRATEGY
    pnl: float = 0.0
    pnl_pct: float = 0.0
    trade_count: int = 0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    status: str = "offline"
    rank: int = 0
    consecutive_misses: int = 0
    last_updated: float = 0.0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
# Keyed by strategy_id
strategy_state: dict[str, StrategyMetrics] = {}
ws_clients: set[WebSocket] = set()
_poll_task: asyncio.Task | None = None
_http_client: httpx.AsyncClient | None = None


def _init_state() -> None:
    """Populate initial offline state for every strategy."""
    for s in STRATEGIES:
        sid = s["id"]
        strategy_state[sid] = StrategyMetrics(
            strategy_id=sid,
            strategy_name=sid,
            status="offline",
        )


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

async def _fetch_metrics(client: httpx.AsyncClient, strat: dict[str, Any]) -> StrategyMetrics | None:
    """Fetch metrics from a single strategy. Returns None on failure."""
    url = f"http://{strat['host']}:{strat['port']}/api/metrics"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        return StrategyMetrics(
            strategy_id=data.get("strategy_id", strat["id"]),
            strategy_name=data.get("strategy_name", strat["id"]),
            equity=float(data.get("equity", STARTING_EQUITY_PER_STRATEGY)),
            pnl=float(data.get("pnl", 0.0)),
            pnl_pct=float(data.get("pnl_pct", 0.0)),
            trade_count=int(data.get("trade_count", 0)),
            sharpe_ratio=float(data.get("sharpe_ratio", 0.0)),
            win_rate=float(data.get("win_rate", 0.0)),
            max_drawdown=float(data.get("max_drawdown", 0.0)),
            status=data.get("status", "running"),
            consecutive_misses=0,
            last_updated=time.time(),
        )
    except Exception:
        return None


async def _poll_once(client: httpx.AsyncClient) -> None:
    """Poll every strategy once, update state, rank, and broadcast."""
    tasks = [_fetch_metrics(client, s) for s in STRATEGIES]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for strat, result in zip(STRATEGIES, results):
        sid = strat["id"]
        prev = strategy_state.get(sid, StrategyMetrics(strategy_id=sid))

        if isinstance(result, StrategyMetrics):
            strategy_state[sid] = result
        else:
            # Poll failed — increment miss counter, keep last-known values
            misses = prev.consecutive_misses + 1
            status = "crashed" if misses >= CRASH_THRESHOLD else "offline"
            strategy_state[sid] = prev.model_copy(
                update={
                    "status": status,
                    "consecutive_misses": misses,
                }
            )

    # Rank strategies
    _rank_strategies()

    # Broadcast to WebSocket clients
    await _broadcast_leaderboard()


def _rank_strategies() -> None:
    """Sort and assign ranks. Online first, then by pnl_pct desc, sharpe desc, trade_count desc."""
    entries = list(strategy_state.values())

    def _sort_key(m: StrategyMetrics) -> tuple:
        is_online = 0 if m.status in ("running", "online") else 1
        return (is_online, -m.pnl_pct, -m.sharpe_ratio, -m.trade_count)

    entries.sort(key=_sort_key)
    for rank, entry in enumerate(entries, start=1):
        strategy_state[entry.strategy_id] = entry.model_copy(update={"rank": rank})


def _build_leaderboard() -> list[dict[str, Any]]:
    entries = sorted(strategy_state.values(), key=lambda m: m.rank)
    return [m.model_dump() for m in entries]


def _build_summary() -> dict[str, Any]:
    metrics = list(strategy_state.values())
    total_equity = sum(m.equity for m in metrics)
    total_pnl = sum(m.pnl for m in metrics)
    total_pnl_pct = ((total_equity - TOTAL_STARTING_EQUITY) / TOTAL_STARTING_EQUITY) * 100.0 if TOTAL_STARTING_EQUITY else 0.0

    online = [m for m in metrics if m.status in ("running", "online")]
    offline = [m for m in metrics if m.status not in ("running", "online")]

    best = max(metrics, key=lambda m: m.pnl_pct) if metrics else None
    worst = min(metrics, key=lambda m: m.pnl_pct) if metrics else None

    return {
        "total_equity": round(total_equity, 2),
        "starting_equity": TOTAL_STARTING_EQUITY,
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 4),
        "best_strategy": best.model_dump() if best else None,
        "worst_strategy": worst.model_dump() if worst else None,
        "online_count": len(online),
        "offline_count": len(offline),
        "strategies": _build_leaderboard(),
    }


# ---------------------------------------------------------------------------
# WebSocket broadcasting
# ---------------------------------------------------------------------------

async def _broadcast_leaderboard() -> None:
    if not ws_clients:
        return
    payload = json.dumps(_build_leaderboard())
    stale: list[WebSocket] = []
    for ws in ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Background polling loop
# ---------------------------------------------------------------------------

async def _polling_loop() -> None:
    global _http_client

    timeout = httpx.Timeout(connect=2.0, read=3.0, write=3.0, pool=5.0)
    limits = httpx.Limits(max_connections=30, max_keepalive_connections=15)

    while True:
        try:
            if _http_client is None or _http_client.is_closed:
                _http_client = httpx.AsyncClient(timeout=timeout, limits=limits)

            await _poll_once(_http_client)
        except Exception:
            logger.exception("Error in polling loop")
            # Re-create client on unexpected errors
            try:
                if _http_client and not _http_client.is_closed:
                    await _http_client.aclose()
            except Exception:
                pass
            _http_client = None

        await asyncio.sleep(POLL_INTERVAL_S)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _poll_task
    _init_state()
    logger.info("Starting aggregator polling loop (interval=%.1fs)", POLL_INTERVAL_S)
    _poll_task = asyncio.create_task(_polling_loop())
    yield
    # Shutdown
    logger.info("Shutting down aggregator")
    if _poll_task:
        _poll_task.cancel()
        try:
            await _poll_task
        except asyncio.CancelledError:
            pass
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Crypto Strategy Lab Aggregator",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    online = sum(1 for m in strategy_state.values() if m.status in ("running", "online"))
    return {
        "status": "healthy",
        "service": "aggregator",
        "strategies_online": online,
        "strategies_total": len(STRATEGIES),
    }


@app.get("/api/leaderboard")
async def leaderboard():
    return _build_leaderboard()


@app.get("/api/summary")
async def summary():
    return _build_summary()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    logger.info("WebSocket client connected (total=%d)", len(ws_clients))
    try:
        # Send current state immediately
        await ws.send_text(json.dumps(_build_leaderboard()))
        # Keep connection alive — wait for client disconnect
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.debug("WebSocket connection closed")
    finally:
        ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8099,
        log_level="info",
    )
