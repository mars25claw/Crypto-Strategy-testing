"""
Master Web Dashboard for Crypto Strategy Lab.

Serves a single-page dashboard that aggregates all 10 strategies
via the aggregator service.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dashboard")

AGGREGATOR_URL = os.getenv("AGGREGATOR_URL", "http://aggregator:8099")
AGGREGATOR_WS_URL = os.getenv("AGGREGATOR_WS_URL", "ws://aggregator:8099/ws")

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# WebSocket clients connected to this dashboard
ws_clients: set[WebSocket] = set()
_proxy_task: asyncio.Task | None = None


async def _fetch_json(session: aiohttp.ClientSession, path: str) -> dict | list | None:
    try:
        async with session.get(f"{AGGREGATOR_URL}{path}", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        logger.debug("Failed to fetch %s", path)
    return None


async def _ws_proxy_loop() -> None:
    """Connect to aggregator WebSocket and broadcast to all dashboard clients."""
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(AGGREGATOR_WS_URL) as ws:
                    logger.info("Connected to aggregator WebSocket")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            stale = []
                            for client in ws_clients:
                                try:
                                    await client.send_text(msg.data)
                                except Exception:
                                    stale.append(client)
                            for c in stale:
                                ws_clients.discard(c)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
        except Exception:
            logger.warning("Aggregator WebSocket disconnected, retrying in 3s")
        await asyncio.sleep(3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _proxy_task
    _proxy_task = asyncio.create_task(_ws_proxy_loop())
    logger.info("Dashboard started — proxying from %s", AGGREGATOR_WS_URL)
    yield
    if _proxy_task:
        _proxy_task.cancel()
        try:
            await _proxy_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Crypto Strategy Lab Dashboard", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/data")
async def api_data():
    async with aiohttp.ClientSession() as session:
        leaderboard, summary = await asyncio.gather(
            _fetch_json(session, "/api/leaderboard"),
            _fetch_json(session, "/api/summary"),
        )
    return {
        "leaderboard": leaderboard or [],
        "summary": summary or {},
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    logger.info("Dashboard WS client connected (total=%d)", len(ws_clients))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ws_clients.discard(ws)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=9000, log_level="info")
