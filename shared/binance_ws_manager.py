"""Async WebSocket manager for Binance spot and futures streams.

Handles combined stream subscriptions, auto-reconnection with exponential backoff,
listen key lifecycle management, stream staleness detection, and ping-pong health
monitoring. Each strategy registers callbacks for specific streams, and the manager
routes incoming messages accordingly.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidStatusCode,
)

logger = logging.getLogger(__name__)
system_logger = logging.getLogger("system")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SPOT_WS_URL = "wss://stream.binance.com:9443/stream"
DEFAULT_FUTURES_WS_URL = "wss://fstream.binance.com/stream"
SPOT_USER_WS_BASE = "wss://stream.binance.com:9443/ws/"
FUTURES_USER_WS_BASE = "wss://fstream.binance.com/ws/"

PING_INTERVAL_S = 30
PONG_TIMEOUT_S = 10
LISTEN_KEY_KEEPALIVE_S = 25 * 60  # 25 minutes
STALENESS_WARN_S = 10
STALENESS_HALT_S = 60
RECONNECT_BACKOFF_BASE_S = 1
RECONNECT_BACKOFF_MAX_S = 30
SUBSCRIBE_ID_START = 1


class ConnectionType(Enum):
    SPOT = "spot"
    FUTURES = "futures"
    SPOT_USER = "spot_user"
    FUTURES_USER = "futures_user"


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _StrategyRegistration:
    """Tracks a single strategy's subscriptions on a connection."""
    strategy_id: str
    # stream_name -> list of async callbacks
    callbacks: Dict[str, List[Callable[..., Coroutine]]] = field(default_factory=dict)


@dataclass
class _ConnectionState:
    """Runtime state for a single WebSocket connection."""
    conn_type: ConnectionType
    url: str
    ws: Optional[Any] = None
    connected: bool = False
    healthy: bool = False
    last_message_time: float = 0.0
    last_pong_time: float = 0.0
    reconnect_count: int = 0
    subscribe_id: int = SUBSCRIBE_ID_START
    # stream_name -> last message timestamp
    stream_last_msg: Dict[str, float] = field(default_factory=dict)
    # strategy_id -> _StrategyRegistration
    strategies: Dict[str, _StrategyRegistration] = field(default_factory=dict)
    _task: Optional[asyncio.Task] = None
    _ping_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# WebSocketManager
# ---------------------------------------------------------------------------

class WebSocketManager:
    """Manages Binance WebSocket connections for spot and futures markets.

    Parameters
    ----------
    spot_ws_url : str
        Combined-stream endpoint for spot market data.
    futures_ws_url : str
        Combined-stream endpoint for futures market data.
    binance_client : object
        Reference to a BinanceClient (or compatible) instance that exposes:
          - async create_spot_listen_key() -> str
          - async keepalive_spot_listen_key(key: str)
          - async create_futures_listen_key() -> str
          - async keepalive_futures_listen_key(key: str)
    """

    def __init__(
        self,
        spot_ws_url: str = DEFAULT_SPOT_WS_URL,
        futures_ws_url: str = DEFAULT_FUTURES_WS_URL,
        binance_client: Optional[Any] = None,
    ):
        self._spot_url = spot_ws_url
        self._futures_url = futures_ws_url
        self._client = binance_client

        # Connection states keyed by ConnectionType
        self._connections: Dict[ConnectionType, _ConnectionState] = {}
        self._init_connection(ConnectionType.SPOT, self._spot_url)
        self._init_connection(ConnectionType.FUTURES, self._futures_url)

        # Listen keys
        self._spot_listen_key: Optional[str] = None
        self._futures_listen_key: Optional[str] = None

        # Background tasks
        self._listen_key_task: Optional[asyncio.Task] = None
        self._staleness_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks invoked on reconnection so strategies can re-validate state.
        # Signature: async callback(conn_type: ConnectionType)
        self._reconnect_callbacks: Dict[str, Callable[..., Coroutine]] = {}

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_connection(self, conn_type: ConnectionType, url: str) -> _ConnectionState:
        state = _ConnectionState(conn_type=conn_type, url=url)
        self._connections[conn_type] = state
        return state

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all WebSocket connections and background tasks."""
        if self._running:
            logger.warning("WebSocketManager already running")
            return

        self._running = True
        system_logger.info("WebSocketManager starting")

        # Start listen-key management if a client is available
        if self._client is not None:
            self._listen_key_task = asyncio.create_task(
                self._listen_key_loop(), name="ws-listen-key-loop"
            )

        # Start staleness monitor
        self._staleness_task = asyncio.create_task(
            self._staleness_monitor(), name="ws-staleness-monitor"
        )

        # Start market-data connections that already have subscriptions
        for conn_type, state in self._connections.items():
            if conn_type in (ConnectionType.SPOT, ConnectionType.FUTURES):
                if self._all_streams(state):
                    self._ensure_connection_task(state)

        system_logger.info("WebSocketManager started")

    async def stop(self) -> None:
        """Gracefully shut down all connections and background tasks."""
        self._running = False
        system_logger.info("WebSocketManager stopping")

        tasks_to_cancel: List[asyncio.Task] = []

        for state in self._connections.values():
            if state._task and not state._task.done():
                state._task.cancel()
                tasks_to_cancel.append(state._task)
            if state._ping_task and not state._ping_task.done():
                state._ping_task.cancel()
                tasks_to_cancel.append(state._ping_task)

        if self._listen_key_task and not self._listen_key_task.done():
            self._listen_key_task.cancel()
            tasks_to_cancel.append(self._listen_key_task)

        if self._staleness_task and not self._staleness_task.done():
            self._staleness_task.cancel()
            tasks_to_cancel.append(self._staleness_task)

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        # Close remaining sockets
        for state in self._connections.values():
            await self._close_ws(state)

        system_logger.info("WebSocketManager stopped")

    # ------------------------------------------------------------------
    # Strategy registration
    # ------------------------------------------------------------------

    def register_strategy(
        self,
        strategy_id: str,
        subscriptions: List[Tuple[str, Callable[..., Coroutine]]],
        conn_type: ConnectionType = ConnectionType.SPOT,
        on_reconnect: Optional[Callable[..., Coroutine]] = None,
    ) -> None:
        """Register a strategy's stream subscriptions.

        Parameters
        ----------
        strategy_id : str
            Unique identifier for the strategy.
        subscriptions : list of (stream_name, async_callback)
            Each tuple maps a Binance stream name (e.g. ``"btcusdt@kline_1m"``)
            to an async callable that receives the parsed message dict.
        conn_type : ConnectionType
            Which connection to subscribe on (SPOT, FUTURES, SPOT_USER, FUTURES_USER).
        on_reconnect : async callable, optional
            Called after a reconnection so the strategy can re-validate its state.
        """
        state = self._connections.get(conn_type)
        if state is None:
            # User data streams (SPOT_USER, FUTURES_USER) are not yet
            # initialised — log and skip so paper-mode strategies can start.
            logger.info(
                "Strategy %s: %s connection not initialised (paper mode — skipping)",
                strategy_id, conn_type.value,
            )
            return

        reg = state.strategies.get(strategy_id)
        if reg is None:
            reg = _StrategyRegistration(strategy_id=strategy_id)
            state.strategies[strategy_id] = reg

        for stream_name, callback in subscriptions:
            stream_lower = stream_name.lower()
            reg.callbacks.setdefault(stream_lower, []).append(callback)
            # Initialise staleness tracking
            state.stream_last_msg.setdefault(stream_lower, 0.0)

        if on_reconnect is not None:
            self._reconnect_callbacks[strategy_id] = on_reconnect

        logger.info(
            "Strategy %s registered %d streams on %s",
            strategy_id, len(subscriptions), conn_type.value,
        )

        # If we're already running, ensure the connection task is alive and
        # send a SUBSCRIBE for the new streams.
        if self._running:
            self._ensure_connection_task(state)
            # Schedule subscribe in background so this stays sync-friendly
            asyncio.ensure_future(self._subscribe_streams(state, [s.lower() for s, _ in subscriptions]))

    def register_user_data_stream(
        self,
        strategy_id: str,
        conn_type: ConnectionType = ConnectionType.FUTURES,
        callbacks: Optional[Dict[str, Callable[..., Coroutine]]] = None,
    ) -> None:
        """Register user data stream callbacks (e.g. ORDER_TRADE_UPDATE).

        In paper trading mode this is a no-op since no real orders are placed.
        """
        logger.info(
            "Strategy %s: user data stream registration for %s (paper mode — no-op)",
            strategy_id, conn_type.value,
        )

    def unregister_strategy(
        self,
        strategy_id: str,
        conn_type: Optional[ConnectionType] = None,
    ) -> None:
        """Remove a strategy's subscriptions.

        Parameters
        ----------
        strategy_id : str
            Strategy to remove.
        conn_type : ConnectionType, optional
            If provided, only unregister from that connection.  Otherwise
            unregister from all connections.
        """
        targets = (
            [self._connections[conn_type]] if conn_type else list(self._connections.values())
        )
        for state in targets:
            if strategy_id in state.strategies:
                removed_streams = list(state.strategies[strategy_id].callbacks.keys())
                del state.strategies[strategy_id]
                logger.info(
                    "Strategy %s unregistered from %s (streams: %s)",
                    strategy_id, state.conn_type.value, removed_streams,
                )
                # Unsubscribe streams no longer needed by any strategy
                still_needed = self._all_streams(state)
                to_unsub = [s for s in removed_streams if s not in still_needed]
                if to_unsub and self._running:
                    asyncio.ensure_future(self._unsubscribe_streams(state, to_unsub))

        self._reconnect_callbacks.pop(strategy_id, None)

    def get_registered_streams(self, conn_type: Optional[ConnectionType] = None) -> List[str]:
        """Return list of unique stream names currently subscribed."""
        streams: Set[str] = set()
        targets = (
            [self._connections[conn_type]] if conn_type else list(self._connections.values())
        )
        for state in targets:
            streams.update(self._all_streams(state))
        return sorted(streams)

    # ------------------------------------------------------------------
    # Staleness
    # ------------------------------------------------------------------

    def is_stream_stale(self, stream_name: str) -> bool:
        """Return True if *stream_name* has not received data in >10 seconds."""
        now = time.time()
        for state in self._connections.values():
            ts = state.stream_last_msg.get(stream_name.lower(), 0.0)
            if ts > 0 and (now - ts) <= STALENESS_WARN_S:
                return False
        return True

    def get_stream_staleness(self) -> Dict[str, float]:
        """Return dict of stream_name -> seconds since last message.

        Streams that have never received a message show ``float('inf')``.
        """
        now = time.time()
        result: Dict[str, float] = {}
        for state in self._connections.values():
            for stream, ts in state.stream_last_msg.items():
                if ts == 0.0:
                    result[stream] = float("inf")
                else:
                    result[stream] = round(now - ts, 2)
        return result

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_health(self) -> Dict[str, Any]:
        """Return aggregated health information across all connections."""
        info: Dict[str, Any] = {
            "running": self._running,
            "connections": {},
        }
        for conn_type, state in self._connections.items():
            staleness = {
                s: round(time.time() - ts, 2) if ts > 0 else float("inf")
                for s, ts in state.stream_last_msg.items()
            }
            stale_streams = [
                s for s, age in staleness.items() if age > STALENESS_WARN_S
            ]
            info["connections"][conn_type.value] = {
                "connected": state.connected,
                "healthy": state.healthy,
                "last_message_time": state.last_message_time,
                "streams_active": [
                    s for s, age in staleness.items() if age <= STALENESS_WARN_S
                ],
                "streams_stale": stale_streams,
                "reconnect_count": state.reconnect_count,
            }
        return info

    # ------------------------------------------------------------------
    # Internal: connection lifecycle
    # ------------------------------------------------------------------

    def _ensure_connection_task(self, state: _ConnectionState) -> None:
        if state._task is None or state._task.done():
            state._task = asyncio.create_task(
                self._connection_loop(state),
                name=f"ws-{state.conn_type.value}",
            )

    async def _connection_loop(self, state: _ConnectionState) -> None:
        """Main loop: connect, read messages, reconnect on failure."""
        backoff = RECONNECT_BACKOFF_BASE_S

        while self._running:
            try:
                await self._connect(state)
                backoff = RECONNECT_BACKOFF_BASE_S  # reset on successful connect

                # Subscribe all currently registered streams
                all_streams = list(self._all_streams(state))
                if all_streams:
                    await self._subscribe_streams(state, all_streams)

                # Notify strategies of (re)connection
                await self._fire_reconnect_callbacks(state)

                # Start ping task
                state._ping_task = asyncio.create_task(
                    self._ping_loop(state),
                    name=f"ws-ping-{state.conn_type.value}",
                )

                # Read loop
                await self._read_loop(state)

            except asyncio.CancelledError:
                logger.debug("Connection loop cancelled for %s", state.conn_type.value)
                break

            except Exception as exc:
                logger.error(
                    "Connection error on %s: %s", state.conn_type.value, exc, exc_info=True,
                )

            finally:
                await self._close_ws(state)
                if state._ping_task and not state._ping_task.done():
                    state._ping_task.cancel()
                    try:
                        await state._ping_task
                    except asyncio.CancelledError:
                        pass

            if not self._running:
                break

            state.reconnect_count += 1
            logger.warning(
                "Reconnecting %s in %.1fs (attempt #%d)",
                state.conn_type.value, backoff, state.reconnect_count,
            )
            system_logger.info(
                "ws_reconnect conn=%s backoff=%.1f attempt=%d",
                state.conn_type.value, backoff, state.reconnect_count,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)

    async def _connect(self, state: _ConnectionState) -> None:
        """Open a WebSocket connection."""
        url = state.url
        logger.info("Connecting to %s (%s)", state.conn_type.value, url)

        state.ws = await websockets.connect(
            url,
            ping_interval=None,  # we handle ping ourselves
            ping_timeout=None,
            close_timeout=5,
            max_size=10 * 1024 * 1024,  # 10 MB
        )
        state.connected = True
        state.healthy = True
        state.last_pong_time = time.time()
        logger.info("Connected to %s", state.conn_type.value)
        system_logger.info("ws_connected conn=%s", state.conn_type.value)

    async def _close_ws(self, state: _ConnectionState) -> None:
        state.connected = False
        state.healthy = False
        if state.ws is not None:
            try:
                await state.ws.close()
            except Exception:
                pass
            state.ws = None

    # ------------------------------------------------------------------
    # Internal: read loop and message routing
    # ------------------------------------------------------------------

    async def _read_loop(self, state: _ConnectionState) -> None:
        """Read messages until the connection drops."""
        assert state.ws is not None
        async for raw in state.ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON message on %s: %s", state.conn_type.value, raw[:200])
                continue

            now = time.time()
            state.last_message_time = now

            # Handle pong frames delivered as messages (some libs)
            # Also handle subscription ack {"result": null, "id": ...}
            if "result" in msg and "id" in msg:
                # Subscription acknowledgement
                continue

            # Combined-stream format: {"stream": "...", "data": {...}}
            stream_name = msg.get("stream")
            data = msg.get("data", msg)

            if stream_name:
                stream_lower = stream_name.lower()
                state.stream_last_msg[stream_lower] = now
                await self._route_message(state, stream_lower, data)
            else:
                # User-data streams send top-level events without "stream" wrapper
                event_type = msg.get("e", "")
                if event_type:
                    state.stream_last_msg[event_type] = now
                    await self._route_message(state, event_type, msg)
                else:
                    logger.debug(
                        "Unroutable message on %s: %s",
                        state.conn_type.value, str(msg)[:200],
                    )

    async def _route_message(
        self, state: _ConnectionState, stream_name: str, data: Any
    ) -> None:
        """Dispatch *data* to every callback registered for *stream_name*."""
        for reg in state.strategies.values():
            callbacks = reg.callbacks.get(stream_name, [])
            for cb in callbacks:
                try:
                    await cb(data)
                except Exception:
                    logger.exception(
                        "Callback error for strategy=%s stream=%s",
                        reg.strategy_id, stream_name,
                    )

    # ------------------------------------------------------------------
    # Internal: subscribe / unsubscribe
    # ------------------------------------------------------------------

    def _all_streams(self, state: _ConnectionState) -> Set[str]:
        """Return the union of all stream names across strategies."""
        streams: Set[str] = set()
        for reg in state.strategies.values():
            streams.update(reg.callbacks.keys())
        return streams

    async def _subscribe_streams(
        self, state: _ConnectionState, streams: List[str]
    ) -> None:
        if not streams or state.ws is None:
            return
        state.subscribe_id += 1
        payload = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": state.subscribe_id,
        }
        try:
            await state.ws.send(json.dumps(payload))
            logger.info(
                "SUBSCRIBE sent on %s: %s (id=%d)",
                state.conn_type.value, streams, state.subscribe_id,
            )
        except Exception:
            logger.exception("Failed to send SUBSCRIBE on %s", state.conn_type.value)

    async def _unsubscribe_streams(
        self, state: _ConnectionState, streams: List[str]
    ) -> None:
        if not streams or state.ws is None:
            return
        state.subscribe_id += 1
        payload = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": state.subscribe_id,
        }
        try:
            await state.ws.send(json.dumps(payload))
            logger.info(
                "UNSUBSCRIBE sent on %s: %s (id=%d)",
                state.conn_type.value, streams, state.subscribe_id,
            )
        except Exception:
            logger.exception("Failed to send UNSUBSCRIBE on %s", state.conn_type.value)

    # ------------------------------------------------------------------
    # Internal: ping / pong
    # ------------------------------------------------------------------

    async def _ping_loop(self, state: _ConnectionState) -> None:
        """Send periodic pings and monitor pong responses."""
        try:
            while self._running and state.connected and state.ws is not None:
                await asyncio.sleep(PING_INTERVAL_S)
                if state.ws is None or not state.connected:
                    break

                try:
                    pong_waiter = await state.ws.ping()
                    try:
                        await asyncio.wait_for(pong_waiter, timeout=PONG_TIMEOUT_S)
                        state.last_pong_time = time.time()
                        state.healthy = True
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Pong timeout on %s — marking unhealthy, triggering reconnect",
                            state.conn_type.value,
                        )
                        state.healthy = False
                        # Force-close so the read loop exits and reconnect fires
                        await self._close_ws(state)
                        return
                except ConnectionClosed:
                    logger.warning("Connection closed during ping on %s", state.conn_type.value)
                    return

        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Internal: listen key management
    # ------------------------------------------------------------------

    async def _listen_key_loop(self) -> None:
        """Create listen keys on startup and keepalive every 25 minutes."""
        try:
            # Initial creation
            await self._create_listen_keys()

            while self._running:
                await asyncio.sleep(LISTEN_KEY_KEEPALIVE_S)
                if not self._running:
                    break
                await self._keepalive_listen_keys()

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Listen key loop error")

    async def _create_listen_keys(self) -> None:
        """Create spot and futures listen keys and start user-data connections."""
        if self._client is None:
            return

        # Spot listen key
        try:
            self._spot_listen_key = await self._client.create_spot_listen_key()
            url = f"{SPOT_USER_WS_BASE}{self._spot_listen_key}"
            state = self._init_connection(ConnectionType.SPOT_USER, url)
            if self._running:
                self._ensure_connection_task(state)
            logger.info("Spot listen key created")
            system_logger.info("listen_key_created market=spot")
        except Exception:
            logger.exception("Failed to create spot listen key")

        # Futures listen key
        try:
            self._futures_listen_key = await self._client.create_futures_listen_key()
            url = f"{FUTURES_USER_WS_BASE}{self._futures_listen_key}"
            state = self._init_connection(ConnectionType.FUTURES_USER, url)
            if self._running:
                self._ensure_connection_task(state)
            logger.info("Futures listen key created")
            system_logger.info("listen_key_created market=futures")
        except Exception:
            logger.exception("Failed to create futures listen key")

    async def _keepalive_listen_keys(self) -> None:
        """Send keepalive for both listen keys; recreate if expired."""
        if self._client is None:
            return

        # Spot
        if self._spot_listen_key:
            try:
                await self._client.keepalive_spot_listen_key(self._spot_listen_key)
                logger.debug("Spot listen key keepalive OK")
            except Exception:
                logger.warning("Spot listen key keepalive failed — recreating")
                try:
                    self._spot_listen_key = await self._client.create_spot_listen_key()
                    url = f"{SPOT_USER_WS_BASE}{self._spot_listen_key}"
                    state = self._connections.get(ConnectionType.SPOT_USER)
                    if state is not None:
                        state.url = url
                        # Force reconnect with new key
                        await self._close_ws(state)
                    system_logger.info("listen_key_recreated market=spot")
                except Exception:
                    logger.exception("Failed to recreate spot listen key")

        # Futures
        if self._futures_listen_key:
            try:
                await self._client.keepalive_futures_listen_key(self._futures_listen_key)
                logger.debug("Futures listen key keepalive OK")
            except Exception:
                logger.warning("Futures listen key keepalive failed — recreating")
                try:
                    self._futures_listen_key = await self._client.create_futures_listen_key()
                    url = f"{FUTURES_USER_WS_BASE}{self._futures_listen_key}"
                    state = self._connections.get(ConnectionType.FUTURES_USER)
                    if state is not None:
                        state.url = url
                        await self._close_ws(state)
                    system_logger.info("listen_key_recreated market=futures")
                except Exception:
                    logger.exception("Failed to recreate futures listen key")

    # ------------------------------------------------------------------
    # Internal: reconnect notifications
    # ------------------------------------------------------------------

    async def _fire_reconnect_callbacks(self, state: _ConnectionState) -> None:
        """Notify strategies that a (re)connection occurred."""
        strategy_ids = set(state.strategies.keys())
        for sid in strategy_ids:
            cb = self._reconnect_callbacks.get(sid)
            if cb is not None:
                try:
                    await cb(state.conn_type)
                except Exception:
                    logger.exception(
                        "Reconnect callback error for strategy=%s conn=%s",
                        sid, state.conn_type.value,
                    )

    # ------------------------------------------------------------------
    # Internal: staleness monitor
    # ------------------------------------------------------------------

    async def _staleness_monitor(self) -> None:
        """Periodically log warnings about stale streams."""
        try:
            while self._running:
                await asyncio.sleep(5)
                if not self._running:
                    break

                now = time.time()
                for state in self._connections.values():
                    for stream, ts in state.stream_last_msg.items():
                        if ts == 0.0:
                            continue
                        age = now - ts
                        if age > STALENESS_HALT_S:
                            logger.warning(
                                "HALT-LEVEL staleness on %s/%s: %.0fs without data",
                                state.conn_type.value, stream, age,
                            )
                            system_logger.info(
                                "stream_stale_halt conn=%s stream=%s age=%.0f",
                                state.conn_type.value, stream, age,
                            )
                        elif age > STALENESS_WARN_S:
                            logger.info(
                                "Stream stale on %s/%s: %.0fs without data",
                                state.conn_type.value, stream, age,
                            )
        except asyncio.CancelledError:
            pass
