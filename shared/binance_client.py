"""Async Binance REST client for spot and futures APIs.

Uses httpx with connection pooling.  Every request passes through
:class:`shared.rate_limiter.RateLimiter` and is signed with HMAC-SHA256
via :func:`shared.utils.sign_request`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from shared.rate_limiter import RateLimiter, Priority, ApiType
from shared.utils import TimeSync, ExchangeInfo, sign_request

logger = logging.getLogger(__name__)

# ── Binance error codes worth special-casing ────────────────────────────────
ERR_INSUFFICIENT_MARGIN = -2019
ERR_TOO_MANY_ORDERS = -1015
ERR_TIMESTAMP = -4028  # recvWindow / timestamp ahead
ERR_TIMESTAMP_ALT = -1021  # Timestamp for this request is outside of the recvWindow

# ── Default endpoint weights (conservative estimates) ────────────────────────
_WEIGHT: Dict[str, int] = {
    # Futures
    "/fapi/v1/order": 1,
    "/fapi/v1/batchOrders": 5,
    "/fapi/v1/allOpenOrders": 1,
    "/fapi/v1/openOrders": 1,
    "/fapi/v2/account": 5,
    "/fapi/v1/userTrades": 5,
    "/fapi/v1/klines": 5,
    "/fapi/v1/premiumIndex": 1,
    "/fapi/v1/fundingRate": 1,
    "/fapi/v1/income": 30,
    "/fapi/v1/ticker/24hr": 1,
    "/fapi/v1/exchangeInfo": 1,
    "/fapi/v1/time": 1,
    "/fapi/v1/depth": 5,
    "/fapi/v1/listenKey": 1,
    # Spot
    "/api/v3/order": 1,
    "/api/v3/account": 10,
    "/api/v3/klines": 5,
    "/api/v3/exchangeInfo": 10,
    "/api/v3/depth": 5,
    # System
    "/sapi/v1/system/status": 1,
}


def _weight_for(path: str) -> int:
    return _WEIGHT.get(path, 1)


class BinanceClientError(Exception):
    """Raised on non-retryable Binance API errors."""

    def __init__(self, code: int, msg: str, response: Optional[dict] = None):
        self.code = code
        self.msg = msg
        self.response = response
        super().__init__(f"Binance API error {code}: {msg}")


class BinanceClient:
    """Async Binance REST client for spot and futures.

    Parameters
    ----------
    api_key : str
        Binance API key.
    api_secret : str
        Binance API secret.
    time_sync : TimeSync
        Shared :class:`TimeSync` instance for server-time adjustments.
    exchange_info : ExchangeInfo
        Shared :class:`ExchangeInfo` instance.
    rate_limiter : RateLimiter
        Shared :class:`RateLimiter` instance.
    spot_base_url : str
        Base URL for spot REST.
    futures_base_url : str
        Base URL for futures REST.
    recv_window : int
        ``recvWindow`` sent with every signed request (ms).
    max_retries : int
        Maximum retry attempts for transient errors.
    retry_delay : float
        Base delay between retries (seconds).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        time_sync: TimeSync,
        exchange_info: ExchangeInfo,
        rate_limiter: RateLimiter,
        spot_base_url: str = "https://api.binance.com",
        futures_base_url: str = "https://fapi.binance.com",
        recv_window: int = 5000,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._time_sync = time_sync
        self._exchange_info = exchange_info
        self._rate_limiter = rate_limiter

        self._spot_base = spot_base_url.rstrip("/")
        self._futures_base = futures_base_url.rstrip("/")
        self._recv_window = recv_window
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._client: Optional[httpx.AsyncClient] = None

    # ── lifecycle ───────────────────────────────────────────────────────

    async def start(self) -> None:
        """Create the httpx client with connection pooling."""
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
            ),
            timeout=httpx.Timeout(10.0, connect=5.0),
            headers={
                "X-MBX-APIKEY": self._api_key,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        logger.info("BinanceClient started")

    async def close(self) -> None:
        """Gracefully shut down the httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("BinanceClient closed")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # ── low-level request plumbing ──────────────────────────────────────

    def _base_url(self, api_type: str) -> str:
        return self._futures_base if api_type == ApiType.FUTURES else self._spot_base

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add timestamp, recvWindow, and HMAC signature to *params*."""
        params["timestamp"] = self._time_sync.get_timestamp()
        params["recvWindow"] = self._recv_window
        qs = urlencode(params)
        params["signature"] = sign_request(qs, self._api_secret)
        return params

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = True,
        api_type: str = ApiType.FUTURES,
        priority: Priority = Priority.NORMAL,
        weight: Optional[int] = None,
        reduce_size_on_margin_error: bool = False,
    ) -> Any:
        """Execute a REST request with rate limiting, signing, and retries."""
        assert self._client is not None, "BinanceClient not started — call start() first"

        if weight is None:
            weight = _weight_for(path)

        url = self._base_url(api_type) + path
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            # ── rate limit ──────────────────────────────────────────────
            await self._rate_limiter.acquire(
                weight=weight,
                priority=priority,
                api_type=api_type,
                endpoint=path,
            )

            # ── sign ────────────────────────────────────────────────────
            req_params = dict(params) if params else {}
            if signed:
                req_params = self._sign(req_params)

            try:
                if method == "GET":
                    resp = await self._client.get(url, params=req_params)
                elif method == "POST":
                    resp = await self._client.post(url, data=req_params)
                elif method == "PUT":
                    resp = await self._client.put(url, data=req_params)
                elif method == "DELETE":
                    resp = await self._client.delete(url, params=req_params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning("Timeout %s %s (attempt %d/%d)", method, path, attempt, self._max_retries)
                await asyncio.sleep(self._retry_delay * attempt)
                continue
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning("HTTP error %s %s: %s (attempt %d/%d)", method, path, exc, attempt, self._max_retries)
                await asyncio.sleep(self._retry_delay * attempt)
                continue

            # ── update rate-limit headers if present ────────────────────
            used_weight_hdr = resp.headers.get("X-MBX-USED-WEIGHT-1M") or resp.headers.get("X-MBX-USED-WEIGHT-1m")
            if used_weight_hdr:
                try:
                    used = int(used_weight_hdr)
                    if used > self._rate_limiter.get_usage().get(api_type, {}).get("used", 0):
                        logger.debug("Server reports used weight %d for %s", used, api_type)
                except ValueError:
                    pass

            # ── parse response ──────────────────────────────────────────
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 60))
                logger.error("429 rate limited on %s, backing off %.1fs", path, retry_after)
                await asyncio.sleep(retry_after)
                continue

            if resp.status_code >= 500:
                logger.warning("Server error %d on %s (attempt %d/%d)", resp.status_code, path, attempt, self._max_retries)
                await asyncio.sleep(self._retry_delay * attempt)
                continue

            try:
                data = resp.json()
            except Exception:
                data = resp.text

            # ── Binance application-level errors ────────────────────────
            if isinstance(data, dict) and "code" in data and data["code"] < 0:
                code = data["code"]
                msg = data.get("msg", "")

                # Insufficient margin — flag for caller, retry once
                if code == ERR_INSUFFICIENT_MARGIN and reduce_size_on_margin_error and attempt == 1:
                    logger.warning("Insufficient margin on %s — caller should reduce size", path)
                    raise BinanceClientError(code, msg, data)

                # Too many orders — wait and retry
                if code == ERR_TOO_MANY_ORDERS:
                    logger.warning("Too many orders, waiting 5s before retry (attempt %d)", attempt)
                    await asyncio.sleep(5.0)
                    continue

                # Timestamp error — resync time and retry
                if code in (ERR_TIMESTAMP, ERR_TIMESTAMP_ALT):
                    logger.warning("Timestamp error, resyncing time (attempt %d)", attempt)
                    await self.sync_time()
                    await asyncio.sleep(0.2)
                    continue

                # Non-retryable
                logger.error("Binance error %d: %s  path=%s", code, msg, path)
                raise BinanceClientError(code, msg, data)

            return data

        # Exhausted retries
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Request to {path} failed after {self._max_retries} retries")

    # ── time sync ───────────────────────────────────────────────────────

    async def sync_time(self) -> int:
        """Sync local clock offset with Binance server time."""
        data = await self._request("GET", "/fapi/v1/time", signed=False, api_type=ApiType.FUTURES)
        server_time = data["serverTime"]
        self._time_sync.update_offset(server_time)
        logger.info("Time synced: offset=%dms", self._time_sync.offset_ms)
        return server_time

    # ════════════════════════════════════════════════════════════════════
    #  FUTURES ORDER MANAGEMENT
    # ════════════════════════════════════════════════════════════════════

    async def place_futures_order(
        self,
        symbol: str,
        side: str,
        type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        callback_rate: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: Optional[str] = None,
        post_only: bool = False,
    ) -> dict:
        """Place a single futures order.

        Supports: MARKET, LIMIT, STOP_MARKET, TAKE_PROFIT_MARKET,
        TRAILING_STOP_MARKET.
        """
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": type.upper(),
            "quantity": str(quantity),
        }

        if price is not None:
            params["price"] = str(price)

        if stop_price is not None:
            params["stopPrice"] = str(stop_price)

        if callback_rate is not None:
            params["callbackRate"] = str(callback_rate)

        if reduce_only:
            params["reduceOnly"] = "true"

        if time_in_force:
            params["timeInForce"] = time_in_force
        elif type.upper() == "LIMIT":
            params["timeInForce"] = "GTX" if post_only else "GTC"

        if post_only and type.upper() == "LIMIT":
            params["timeInForce"] = "GTX"

        logger.info(
            "Futures order: %s %s %s qty=%s price=%s stop=%s",
            symbol, side, type, quantity, price, stop_price,
        )

        return await self._request(
            "POST", "/fapi/v1/order",
            params=params,
            api_type=ApiType.FUTURES,
            priority=Priority.HIGH,
            reduce_size_on_margin_error=True,
        )

    async def place_batch_futures_orders(self, orders: List[Dict[str, Any]]) -> List[dict]:
        """Place up to 5 futures orders in a single request.

        Each element in *orders* is a dict with the same keys as
        :meth:`place_futures_order` parameters (symbol, side, type, etc.).
        """
        if not orders:
            return []
        if len(orders) > 5:
            raise ValueError("Binance batch limit is 5 orders per request")

        batch_list: List[Dict[str, Any]] = []
        for o in orders:
            entry: Dict[str, Any] = {
                "symbol": o["symbol"],
                "side": o["side"].upper(),
                "type": o["type"].upper(),
                "quantity": str(o["quantity"]),
            }
            if "price" in o and o["price"] is not None:
                entry["price"] = str(o["price"])
            if "stop_price" in o and o["stop_price"] is not None:
                entry["stopPrice"] = str(o["stop_price"])
            if "callback_rate" in o and o["callback_rate"] is not None:
                entry["callbackRate"] = str(o["callback_rate"])
            if o.get("reduce_only"):
                entry["reduceOnly"] = "true"
            tif = o.get("time_in_force")
            if tif:
                entry["timeInForce"] = tif
            elif entry["type"] == "LIMIT":
                entry["timeInForce"] = "GTC"
            batch_list.append(entry)

        params = {"batchOrders": json.dumps(batch_list)}

        logger.info("Batch futures orders: %d orders", len(batch_list))
        return await self._request(
            "POST", "/fapi/v1/batchOrders",
            params=params,
            api_type=ApiType.FUTURES,
            priority=Priority.HIGH,
            weight=5,
        )

    async def cancel_futures_order(self, symbol: str, order_id: int) -> dict:
        """Cancel a single futures order by orderId."""
        return await self._request(
            "DELETE", "/fapi/v1/order",
            params={"symbol": symbol, "orderId": order_id},
            api_type=ApiType.FUTURES,
            priority=Priority.HIGH,
        )

    async def cancel_all_futures_orders(self, symbol: str) -> dict:
        """Cancel ALL open futures orders for a symbol."""
        logger.warning("Cancelling ALL futures orders for %s", symbol)
        return await self._request(
            "DELETE", "/fapi/v1/allOpenOrders",
            params={"symbol": symbol},
            api_type=ApiType.FUTURES,
            priority=Priority.CRITICAL,
        )

    # ════════════════════════════════════════════════════════════════════
    #  SPOT ORDER MANAGEMENT
    # ════════════════════════════════════════════════════════════════════

    async def place_spot_order(
        self,
        symbol: str,
        side: str,
        type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: Optional[str] = None,
    ) -> dict:
        """Place a single spot order."""
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": type.upper(),
            "quantity": str(quantity),
        }
        if price is not None:
            params["price"] = str(price)
        if time_in_force:
            params["timeInForce"] = time_in_force
        elif type.upper() == "LIMIT":
            params["timeInForce"] = "GTC"

        logger.info("Spot order: %s %s %s qty=%s price=%s", symbol, side, type, quantity, price)
        return await self._request(
            "POST", "/api/v3/order",
            params=params,
            api_type=ApiType.SPOT,
            priority=Priority.HIGH,
        )

    async def cancel_spot_order(self, symbol: str, order_id: int) -> dict:
        """Cancel a single spot order."""
        return await self._request(
            "DELETE", "/api/v3/order",
            params={"symbol": symbol, "orderId": order_id},
            api_type=ApiType.SPOT,
            priority=Priority.HIGH,
        )

    # ════════════════════════════════════════════════════════════════════
    #  ACCOUNT & POSITION DATA
    # ════════════════════════════════════════════════════════════════════

    async def get_futures_account(self) -> dict:
        """GET /fapi/v2/account — full account info (balances + positions)."""
        return await self._request("GET", "/fapi/v2/account", api_type=ApiType.FUTURES, weight=5)

    async def get_spot_account(self) -> dict:
        """GET /api/v3/account — spot balances."""
        return await self._request("GET", "/api/v3/account", api_type=ApiType.SPOT, weight=10)

    async def get_futures_open_orders(self, symbol: Optional[str] = None) -> list:
        """GET /fapi/v1/openOrders."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/fapi/v1/openOrders", params=params, api_type=ApiType.FUTURES)

    async def get_futures_user_trades(
        self, symbol: str, limit: int = 50, start_time: Optional[int] = None
    ) -> list:
        """GET /fapi/v1/userTrades."""
        params: Dict[str, Any] = {"symbol": symbol, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        return await self._request(
            "GET", "/fapi/v1/userTrades",
            params=params,
            api_type=ApiType.FUTURES,
            weight=5,
        )

    # ════════════════════════════════════════════════════════════════════
    #  MARKET DATA
    # ════════════════════════════════════════════════════════════════════

    async def get_futures_klines(
        self, symbol: str, interval: str, limit: int = 500, start_time: Optional[int] = None
    ) -> list:
        """GET /fapi/v1/klines."""
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        return await self._request(
            "GET", "/fapi/v1/klines",
            params=params,
            signed=False,
            api_type=ApiType.FUTURES,
            weight=5,
        )

    async def get_spot_klines(
        self, symbol: str, interval: str, limit: int = 500, start_time: Optional[int] = None
    ) -> list:
        """GET /api/v3/klines."""
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        return await self._request(
            "GET", "/api/v3/klines",
            params=params,
            signed=False,
            api_type=ApiType.SPOT,
            weight=5,
        )

    async def get_premium_index(self, symbol: Optional[str] = None) -> Any:
        """GET /fapi/v1/premiumIndex — mark price & funding rate."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request(
            "GET", "/fapi/v1/premiumIndex",
            params=params,
            signed=False,
            api_type=ApiType.FUTURES,
        )

    async def get_funding_rate_history(
        self,
        symbol: str,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list:
        """GET /fapi/v1/fundingRate."""
        params: Dict[str, Any] = {"symbol": symbol, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return await self._request(
            "GET", "/fapi/v1/fundingRate",
            params=params,
            signed=False,
            api_type=ApiType.FUTURES,
        )

    async def get_income_history(
        self,
        symbol: Optional[str] = None,
        income_type: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list:
        """GET /fapi/v1/income — PnL, funding, commission, etc."""
        params: Dict[str, Any] = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        if income_type:
            params["incomeType"] = income_type
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return await self._request(
            "GET", "/fapi/v1/income",
            params=params,
            api_type=ApiType.FUTURES,
            weight=30,
        )

    async def get_ticker_24hr(self, symbol: Optional[str] = None) -> Any:
        """GET /fapi/v1/ticker/24hr."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        w = 1 if symbol else 40
        return await self._request(
            "GET", "/fapi/v1/ticker/24hr",
            params=params,
            signed=False,
            api_type=ApiType.FUTURES,
            weight=w,
        )

    async def get_futures_depth(self, symbol: str, limit: int = 20) -> dict:
        """GET /fapi/v1/depth — order book snapshot."""
        params: Dict[str, Any] = {"symbol": symbol, "limit": limit}
        w = 5 if limit <= 50 else 10 if limit <= 100 else 20
        return await self._request(
            "GET", "/fapi/v1/depth",
            params=params,
            signed=False,
            api_type=ApiType.FUTURES,
            weight=w,
        )

    async def get_spot_depth(self, symbol: str, limit: int = 20) -> dict:
        """GET /api/v3/depth — order book snapshot."""
        params: Dict[str, Any] = {"symbol": symbol, "limit": limit}
        w = 5 if limit <= 50 else 10 if limit <= 100 else 25
        return await self._request(
            "GET", "/api/v3/depth",
            params=params,
            signed=False,
            api_type=ApiType.SPOT,
            weight=w,
        )

    # ════════════════════════════════════════════════════════════════════
    #  EXCHANGE INFO & SYSTEM
    # ════════════════════════════════════════════════════════════════════

    async def get_futures_exchange_info(self) -> dict:
        """GET /fapi/v1/exchangeInfo."""
        return await self._request(
            "GET", "/fapi/v1/exchangeInfo",
            signed=False,
            api_type=ApiType.FUTURES,
        )

    async def get_spot_exchange_info(self) -> dict:
        """GET /api/v3/exchangeInfo."""
        return await self._request(
            "GET", "/api/v3/exchangeInfo",
            signed=False,
            api_type=ApiType.SPOT,
            weight=10,
        )

    async def get_system_status(self) -> dict:
        """GET /sapi/v1/system/status — overall system status."""
        return await self._request(
            "GET", "/sapi/v1/system/status",
            signed=False,
            api_type=ApiType.SPOT,
        )

    # ════════════════════════════════════════════════════════════════════
    #  LISTEN KEY (User Data Stream)
    # ════════════════════════════════════════════════════════════════════

    async def create_futures_listen_key(self) -> str:
        """POST /fapi/v1/listenKey — create a new listen key."""
        data = await self._request(
            "POST", "/fapi/v1/listenKey",
            signed=False,
            api_type=ApiType.FUTURES,
        )
        key = data["listenKey"]
        logger.info("Created futures listen key: %s...%s", key[:8], key[-4:])
        return key

    async def keepalive_futures_listen_key(self) -> dict:
        """PUT /fapi/v1/listenKey — keepalive (call every 30 min)."""
        return await self._request(
            "PUT", "/fapi/v1/listenKey",
            signed=False,
            api_type=ApiType.FUTURES,
        )

    async def delete_futures_listen_key(self) -> dict:
        """DELETE /fapi/v1/listenKey — close user data stream."""
        return await self._request(
            "DELETE", "/fapi/v1/listenKey",
            signed=False,
            api_type=ApiType.FUTURES,
        )

    async def create_spot_listen_key(self) -> str:
        """POST /api/v3/userDataStream."""
        data = await self._request(
            "POST", "/api/v3/userDataStream",
            signed=False,
            api_type=ApiType.SPOT,
        )
        key = data["listenKey"]
        logger.info("Created spot listen key: %s...%s", key[:8], key[-4:])
        return key

    async def keepalive_spot_listen_key(self, listen_key: str) -> dict:
        """PUT /api/v3/userDataStream."""
        return await self._request(
            "PUT", "/api/v3/userDataStream",
            params={"listenKey": listen_key},
            signed=False,
            api_type=ApiType.SPOT,
        )

    async def delete_spot_listen_key(self, listen_key: str) -> dict:
        """DELETE /api/v3/userDataStream."""
        return await self._request(
            "DELETE", "/api/v3/userDataStream",
            params={"listenKey": listen_key},
            signed=False,
            api_type=ApiType.SPOT,
        )

    # ════════════════════════════════════════════════════════════════════
    #  CONVENIENCE HELPERS
    # ════════════════════════════════════════════════════════════════════

    async def load_exchange_info(self, symbols: Optional[List[str]] = None) -> None:
        """Fetch futures exchange info and populate :attr:`_exchange_info`."""
        data = await self.get_futures_exchange_info()
        for sym_info in data.get("symbols", []):
            sym = sym_info["symbol"]
            if symbols and sym not in symbols:
                continue
            self._exchange_info.update_from_filters(
                sym,
                sym_info.get("filters", []),
                base_precision=sym_info.get("baseAssetPrecision", 8),
                quote_precision=sym_info.get("quotePrecision", 8),
            )
        logger.info("Loaded exchange info for %d symbols", len(self._exchange_info._symbols))

    async def ensure_time_synced(self) -> None:
        """Sync time if interval has elapsed."""
        if self._time_sync.needs_sync():
            await self.sync_time()
