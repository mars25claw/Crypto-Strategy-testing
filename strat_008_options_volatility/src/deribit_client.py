"""Optional Deribit REST + WebSocket client for STRAT-008.

Provides options chain fetching, order management, Greeks streaming,
and connection lifecycle management independent of Binance.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)
system_logger = logging.getLogger("system")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DERIBIT_REST_URL = "https://www.deribit.com/api/v2"
DERIBIT_WS_URL = "wss://www.deribit.com/ws/api/v2"
DERIBIT_TESTNET_REST_URL = "https://test.deribit.com/api/v2"
DERIBIT_TESTNET_WS_URL = "wss://test.deribit.com/ws/api/v2"

WS_HEARTBEAT_INTERVAL = 15
OPTIONS_POLL_INTERVAL = 5.0
ORDERBOOK_POLL_INTERVAL = 1.0
RECONNECT_BACKOFF_MAX = 30
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DeribitOption:
    """Represents a single option instrument on Deribit."""
    instrument_name: str
    underlying: str       # "BTC" or "ETH"
    strike: float
    expiration_ms: int
    option_type: str      # "call" or "put"
    # Market data
    mark_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    mark_iv: float = 0.0
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    open_interest: float = 0.0

    @property
    def dte(self) -> float:
        """Days to expiration."""
        remaining_ms = self.expiration_ms - int(time.time() * 1000)
        return max(0.0, remaining_ms / (1000 * 86400))

    @property
    def mid_price(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.mark_price

    @property
    def spread_pct(self) -> float:
        mid = self.mid_price
        if mid <= 0:
            return float("inf")
        return abs(self.ask - self.bid) / mid * 100.0

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument_name,
            "underlying": self.underlying,
            "strike": self.strike,
            "expiration_ms": self.expiration_ms,
            "type": self.option_type,
            "mark_price": self.mark_price,
            "bid": self.bid,
            "ask": self.ask,
            "mark_iv": self.mark_iv,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "dte": round(self.dte, 2),
            "open_interest": self.open_interest,
            "spread_pct": round(self.spread_pct, 2),
        }


@dataclass
class DeribitPosition:
    """An open Deribit option position."""
    instrument_name: str
    direction: str  # "buy" or "sell"
    size: float
    average_price: float
    mark_price: float = 0.0
    unrealized_pnl: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0


# ---------------------------------------------------------------------------
# DeribitClient
# ---------------------------------------------------------------------------

class DeribitClient:
    """Async Deribit client for options trading.

    Manages its own connection lifecycle independently from Binance.

    Parameters
    ----------
    api_key : str
        Deribit API key.
    api_secret : str
        Deribit API secret.
    testnet : bool
        Use testnet endpoints.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet

        self._rest_url = DERIBIT_TESTNET_REST_URL if testnet else DERIBIT_REST_URL
        self._ws_url = DERIBIT_TESTNET_WS_URL if testnet else DERIBIT_WS_URL

        self._client: Optional[httpx.AsyncClient] = None
        self._ws: Optional[Any] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._running = False
        self._authenticated = False
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: float = 0.0

        # Options chain cache: underlying -> list[DeribitOption]
        self._options_cache: Dict[str, List[DeribitOption]] = {}
        self._cache_timestamp: Dict[str, float] = {}

        # Position cache
        self._positions: Dict[str, DeribitPosition] = {}

        # WS subscription callbacks
        self._ws_callbacks: Dict[str, List[Callable]] = {}
        self._ws_id_counter = 0
        self._ws_pending: Dict[int, asyncio.Future] = {}

        # Index prices
        self._index_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize HTTP client and authenticate."""
        if self._client is not None:
            return

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._running = True

        if self._api_key and self._api_secret:
            await self._authenticate()

        logger.info(
            "DeribitClient started (testnet=%s, authenticated=%s)",
            self._testnet, self._authenticated,
        )

    async def stop(self) -> None:
        """Shut down connections."""
        self._running = False

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._client:
            await self._client.aclose()
            self._client = None

        logger.info("DeribitClient stopped")

    @property
    def is_connected(self) -> bool:
        return self._running and self._client is not None

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated and time.time() < self._token_expiry

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def _authenticate(self) -> bool:
        """Authenticate with Deribit using client_credentials."""
        try:
            result = await self._public_request("public/auth", {
                "grant_type": "client_credentials",
                "client_id": self._api_key,
                "client_secret": self._api_secret,
            })

            if result and "access_token" in result:
                self._access_token = result["access_token"]
                self._refresh_token = result.get("refresh_token")
                expires_in = result.get("expires_in", 900)
                self._token_expiry = time.time() + expires_in - 60
                self._authenticated = True
                logger.info("Deribit authentication successful")
                return True

        except Exception as exc:
            logger.error("Deribit authentication failed: %s", exc)

        self._authenticated = False
        return False

    async def _ensure_auth(self) -> None:
        """Refresh authentication if token is expiring."""
        if not self._authenticated:
            return

        if time.time() >= self._token_expiry:
            if self._refresh_token:
                try:
                    result = await self._public_request("public/auth", {
                        "grant_type": "refresh_token",
                        "refresh_token": self._refresh_token,
                    })
                    if result and "access_token" in result:
                        self._access_token = result["access_token"]
                        self._refresh_token = result.get("refresh_token")
                        expires_in = result.get("expires_in", 900)
                        self._token_expiry = time.time() + expires_in - 60
                        return
                except Exception:
                    pass

            await self._authenticate()

    # ------------------------------------------------------------------
    # REST: Public endpoints
    # ------------------------------------------------------------------

    async def _public_request(
        self, method: str, params: Optional[dict] = None
    ) -> Optional[dict]:
        """Make a public Deribit API request."""
        assert self._client is not None, "DeribitClient not started"

        url = f"{self._rest_url}/{method}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await self._client.get(url, params=params or {})
                data = resp.json()

                if "result" in data:
                    return data["result"]

                if "error" in data:
                    logger.error("Deribit error: %s", data["error"])
                    return None

                return data

            except httpx.TimeoutException:
                logger.warning(
                    "Deribit timeout on %s (attempt %d/%d)",
                    method, attempt, MAX_RETRIES,
                )
                await asyncio.sleep(1.0 * attempt)
            except Exception as exc:
                logger.error("Deribit request error: %s", exc)
                await asyncio.sleep(1.0 * attempt)

        return None

    async def _private_request(
        self, method: str, params: Optional[dict] = None
    ) -> Optional[dict]:
        """Make an authenticated Deribit API request."""
        assert self._client is not None
        await self._ensure_auth()

        if not self._access_token:
            logger.error("No Deribit access token for private request")
            return None

        url = f"{self._rest_url}/{method}"
        headers = {"Authorization": f"Bearer {self._access_token}"}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await self._client.get(
                    url, params=params or {}, headers=headers
                )
                data = resp.json()

                if "result" in data:
                    return data["result"]

                if "error" in data:
                    err = data["error"]
                    if err.get("code") == 13009:  # unauthorized
                        await self._authenticate()
                        headers = {"Authorization": f"Bearer {self._access_token}"}
                        continue
                    logger.error("Deribit error: %s", err)
                    return None

                return data

            except httpx.TimeoutException:
                logger.warning(
                    "Deribit timeout on %s (attempt %d/%d)",
                    method, attempt, MAX_RETRIES,
                )
                await asyncio.sleep(1.0 * attempt)
            except Exception as exc:
                logger.error("Deribit request error: %s", exc)
                await asyncio.sleep(1.0 * attempt)

        return None

    # ------------------------------------------------------------------
    # Options chain
    # ------------------------------------------------------------------

    async def get_options_chain(
        self, underlying: str, currency: str = "BTC"
    ) -> List[DeribitOption]:
        """Fetch the full options chain for an underlying.

        Parameters
        ----------
        underlying : str
            "BTC" or "ETH".
        currency : str
            Currency for Deribit API (usually same as underlying).

        Returns
        -------
        List of DeribitOption.
        """
        result = await self._public_request(
            "public/get_instruments",
            {"currency": currency, "kind": "option", "expired": "false"},
        )

        if not result:
            return self._options_cache.get(underlying, [])

        options: List[DeribitOption] = []
        for inst in result:
            name = inst["instrument_name"]
            # Parse: BTC-28MAR25-100000-C
            parts = name.split("-")
            if len(parts) < 4:
                continue

            opt = DeribitOption(
                instrument_name=name,
                underlying=parts[0],
                strike=float(inst.get("strike", 0)),
                expiration_ms=inst.get("expiration_timestamp", 0),
                option_type="call" if parts[-1] == "C" else "put",
            )
            options.append(opt)

        # Fetch ticker data for all options (in batches)
        await self._enrich_options_data(options)

        self._options_cache[underlying] = options
        self._cache_timestamp[underlying] = time.time()

        logger.info(
            "Fetched %d options for %s", len(options), underlying,
        )
        return options

    async def _enrich_options_data(self, options: List[DeribitOption]) -> None:
        """Fetch mark price, IV, and Greeks for options via book_summary."""
        if not options:
            return

        underlying = options[0].underlying
        result = await self._public_request(
            "public/get_book_summary_by_currency",
            {"currency": underlying, "kind": "option"},
        )

        if not result:
            return

        # Build lookup by instrument name
        summary_map: Dict[str, dict] = {}
        for item in result:
            summary_map[item.get("instrument_name", "")] = item

        for opt in options:
            summary = summary_map.get(opt.instrument_name)
            if not summary:
                continue

            opt.mark_price = summary.get("mark_price", 0.0)
            opt.bid = summary.get("bid_price", 0.0) or 0.0
            opt.ask = summary.get("ask_price", 0.0) or 0.0
            opt.mark_iv = summary.get("mark_iv", 0.0) or 0.0
            opt.open_interest = summary.get("open_interest", 0.0) or 0.0

            greeks = summary.get("greeks", {})
            if greeks:
                opt.delta = greeks.get("delta", 0.0)
                opt.gamma = greeks.get("gamma", 0.0)
                opt.theta = greeks.get("theta", 0.0)
                opt.vega = greeks.get("vega", 0.0)

    async def get_atm_iv(self, underlying: str) -> float:
        """Get the ATM implied volatility for the nearest expiration.

        Finds the option with strike closest to current index price
        with the shortest DTE (>= 3 days).
        """
        # Get index price
        index = await self._public_request(
            "public/get_index_price",
            {"index_name": f"{underlying.lower()}_usd"},
        )
        if not index:
            return 0.0

        index_price = index.get("index_price", 0.0)
        if index_price <= 0:
            return 0.0

        self._index_prices[underlying] = index_price

        # Get options chain
        chain = await self.get_options_chain(underlying, underlying)
        if not chain:
            return 0.0

        # Filter: DTE >= 3, calls, closest to ATM for nearest expiry
        candidates = [
            o for o in chain
            if o.option_type == "call"
            and o.dte >= 3
            and o.mark_iv > 0
        ]

        if not candidates:
            return 0.0

        # Sort by DTE then by distance from ATM
        candidates.sort(key=lambda o: (o.dte, abs(o.strike - index_price)))

        # Pick closest ATM from nearest expiry
        nearest_dte = candidates[0].dte
        near_options = [o for o in candidates if abs(o.dte - nearest_dte) < 1]
        if near_options:
            atm = min(near_options, key=lambda o: abs(o.strike - index_price))
            return atm.mark_iv

        return candidates[0].mark_iv

    def get_index_price(self, underlying: str) -> float:
        """Return the cached index price for an underlying."""
        return self._index_prices.get(underlying, 0.0)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def place_order(
        self,
        instrument_name: str,
        direction: str,  # "buy" or "sell"
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Optional[dict]:
        """Place an order on Deribit.

        Parameters
        ----------
        instrument_name : str
            e.g. "BTC-28MAR25-100000-C"
        direction : str
            "buy" or "sell"
        amount : float
            Number of contracts.
        order_type : str
            "market" or "limit"
        price : float, optional
            Limit price (required for limit orders).
        reduce_only : bool
            If True, only reduce existing position.

        Returns
        -------
        Order response dict or None on failure.
        """
        method = f"private/{direction}"
        params = {
            "instrument_name": instrument_name,
            "amount": amount,
            "type": order_type,
        }

        if price is not None and order_type == "limit":
            params["price"] = price

        if reduce_only:
            params["reduce_only"] = True

        result = await self._private_request(method, params)

        if result:
            order = result.get("order", {})
            logger.info(
                "Deribit order placed: %s %s %.4f %s @ %s -> %s",
                direction, instrument_name, amount, order_type,
                price or "market", order.get("order_id", "?"),
            )

        return result

    async def cancel_order(self, order_id: str) -> Optional[dict]:
        """Cancel an order by its ID."""
        return await self._private_request(
            "private/cancel", {"order_id": order_id}
        )

    async def cancel_all_by_instrument(self, instrument_name: str) -> Optional[dict]:
        """Cancel all orders for a given instrument."""
        return await self._private_request(
            "private/cancel_all_by_instrument",
            {"instrument_name": instrument_name},
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(
        self, currency: str = "BTC", kind: str = "option"
    ) -> List[DeribitPosition]:
        """Fetch all open positions."""
        result = await self._private_request(
            "private/get_positions",
            {"currency": currency, "kind": kind},
        )

        if not result:
            return []

        positions: List[DeribitPosition] = []
        for p in result:
            pos = DeribitPosition(
                instrument_name=p.get("instrument_name", ""),
                direction=p.get("direction", "buy"),
                size=abs(p.get("size", 0.0)),
                average_price=p.get("average_price", 0.0),
                mark_price=p.get("mark_price", 0.0),
                unrealized_pnl=p.get("floating_profit_loss", 0.0),
                delta=p.get("delta", 0.0),
                gamma=p.get("gamma", 0.0),
                theta=p.get("theta", 0.0),
                vega=p.get("vega", 0.0),
            )
            positions.append(pos)
            self._positions[pos.instrument_name] = pos

        return positions

    async def get_account_summary(self, currency: str = "BTC") -> Optional[dict]:
        """Get account summary including equity and margins."""
        return await self._private_request(
            "private/get_account_summary",
            {"currency": currency, "extended": "true"},
        )

    # ------------------------------------------------------------------
    # Greeks streaming via polling
    # ------------------------------------------------------------------

    async def start_greeks_polling(
        self,
        underlyings: List[str],
        callback: Callable[[str, Dict[str, float]], Coroutine],
        interval: float = 5.0,
    ) -> None:
        """Start background task that polls Greeks for open positions.

        The callback receives (instrument_name, greeks_dict) for each position.
        """
        self._ws_task = asyncio.create_task(
            self._greeks_poll_loop(underlyings, callback, interval),
            name="deribit-greeks-poll",
        )

    async def _greeks_poll_loop(
        self,
        underlyings: List[str],
        callback: Callable,
        interval: float,
    ) -> None:
        """Poll Deribit for position Greeks updates."""
        try:
            while self._running:
                await asyncio.sleep(interval)

                for currency in underlyings:
                    positions = await self.get_positions(currency, "option")
                    for pos in positions:
                        greeks = {
                            "delta": pos.delta,
                            "gamma": pos.gamma,
                            "theta": pos.theta,
                            "vega": pos.vega,
                        }
                        try:
                            result = callback(pos.instrument_name, greeks)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception:
                            logger.exception(
                                "Greeks callback error for %s",
                                pos.instrument_name,
                            )

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Greeks polling loop error")

    # ------------------------------------------------------------------
    # Connection health
    # ------------------------------------------------------------------

    def get_health(self) -> dict:
        """Return connection health info."""
        return {
            "connected": self.is_connected,
            "authenticated": self.is_authenticated,
            "testnet": self._testnet,
            "cached_options": {
                k: len(v) for k, v in self._options_cache.items()
            },
            "positions": len(self._positions),
            "index_prices": dict(self._index_prices),
        }

    # ------------------------------------------------------------------
    # Convenience: find specific options
    # ------------------------------------------------------------------

    def find_atm_options(
        self,
        underlying: str,
        index_price: float,
        min_dte: float = 3.0,
        max_dte: float = 30.0,
    ) -> Tuple[Optional[DeribitOption], Optional[DeribitOption]]:
        """Find the ATM call and put for the nearest valid expiration.

        Returns (atm_call, atm_put) or (None, None) if not found.
        """
        chain = self._options_cache.get(underlying, [])
        if not chain:
            return None, None

        valid = [
            o for o in chain
            if min_dte <= o.dte <= max_dte
            and o.mark_iv > 0
        ]

        if not valid:
            return None, None

        # Group by expiration
        expirations: Dict[int, List[DeribitOption]] = {}
        for o in valid:
            expirations.setdefault(o.expiration_ms, []).append(o)

        # Nearest expiration
        nearest_exp = min(expirations.keys())
        opts = expirations[nearest_exp]

        # Find ATM strike (closest to index)
        strikes = set(o.strike for o in opts)
        atm_strike = min(strikes, key=lambda s: abs(s - index_price))

        atm_call = next(
            (o for o in opts if o.strike == atm_strike and o.option_type == "call"),
            None,
        )
        atm_put = next(
            (o for o in opts if o.strike == atm_strike and o.option_type == "put"),
            None,
        )

        return atm_call, atm_put

    def find_otm_strangle(
        self,
        underlying: str,
        index_price: float,
        call_otm_pct: float = 0.05,
        put_otm_pct: float = 0.05,
        min_dte: float = 5.0,
        max_dte: float = 14.0,
    ) -> Tuple[Optional[DeribitOption], Optional[DeribitOption]]:
        """Find OTM call and put for a strangle.

        Returns (otm_call, otm_put) or (None, None).
        """
        chain = self._options_cache.get(underlying, [])
        if not chain:
            return None, None

        target_call_strike = index_price * (1 + call_otm_pct)
        target_put_strike = index_price * (1 - put_otm_pct)

        valid = [o for o in chain if min_dte <= o.dte <= max_dte and o.mark_iv > 0]

        calls = [o for o in valid if o.option_type == "call" and o.strike >= index_price]
        puts = [o for o in valid if o.option_type == "put" and o.strike <= index_price]

        otm_call = min(
            calls,
            key=lambda o: abs(o.strike - target_call_strike),
            default=None,
        )
        otm_put = min(
            puts,
            key=lambda o: abs(o.strike - target_put_strike),
            default=None,
        )

        return otm_call, otm_put
