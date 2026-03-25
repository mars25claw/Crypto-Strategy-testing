"""External data source clients with caching and graceful degradation.

Provides unified access to Fear & Greed Index, Glassnode on-chain metrics,
and social sentiment data. All clients use async HTTP, retry with backoff,
per-metric caching, and return stale data on failure instead of raising.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class CachedValue:
    """A cached metric value with freshness tracking."""
    value: Any = None
    timestamp: float = 0.0
    stale: bool = False

    @property
    def age_seconds(self) -> float:
        if self.timestamp == 0:
            return float("inf")
        return time.time() - self.timestamp


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    json_body: Optional[dict] = None,
    max_retries: int = 3,
    backoff_base: float = 1.0,
) -> Optional[httpx.Response]:
    """Execute an HTTP request with exponential backoff retries."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.request(
                method, url, params=params, headers=headers, json=json_body, timeout=15.0
            )
            resp.raise_for_status()
            return resp
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as exc:
            if attempt == max_retries:
                logger.warning("Request failed after %d attempts: %s %s — %s", max_retries, method, url, exc)
                return None
            wait = backoff_base * (2 ** (attempt - 1))
            logger.debug("Retry %d/%d for %s %s in %.1fs", attempt, max_retries, method, url, wait)
            await asyncio.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDataClient:
    """Common caching / freshness infrastructure for all data clients."""

    def __init__(self) -> None:
        self._cache: Dict[str, CachedValue] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient()
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # -- cache helpers -------------------------------------------------------

    def _get_cached(self, key: str, max_age_seconds: float) -> Optional[CachedValue]:
        """Return cached value if fresh enough, else None."""
        cached = self._cache.get(key)
        if cached and cached.age_seconds < max_age_seconds:
            return cached
        return None

    def _set_cache(self, key: str, value: Any) -> CachedValue:
        entry = CachedValue(value=value, timestamp=time.time(), stale=False)
        self._cache[key] = entry
        return entry

    def _get_stale(self, key: str) -> Optional[Any]:
        """Return last known value marked as stale, or None if never fetched."""
        cached = self._cache.get(key)
        if cached and cached.value is not None:
            cached.stale = True
            return cached.value
        return None

    # -- public freshness API ------------------------------------------------

    def is_fresh(self, metric: str, max_age_hours: float = 1.0) -> bool:
        """Check whether a cached metric is younger than *max_age_hours*."""
        cached = self._cache.get(metric)
        if cached is None or cached.timestamp == 0:
            return False
        return cached.age_seconds < max_age_hours * 3600

    def get_all_statuses(self) -> Dict[str, dict]:
        """Return {metric: {value, timestamp, fresh, stale}} for every cached metric."""
        now = time.time()
        result: Dict[str, dict] = {}
        for key, cv in self._cache.items():
            result[key] = {
                "value": cv.value,
                "timestamp": cv.timestamp,
                "fresh": cv.age_seconds < 3600,  # default 1-hour freshness
                "stale": cv.stale,
            }
        return result


# ---------------------------------------------------------------------------
# Fear & Greed Index
# ---------------------------------------------------------------------------

class FearGreedClient(BaseDataClient):
    """Fetch the Crypto Fear & Greed Index from alternative.me."""

    BASE_URL = "https://api.alternative.me/fng/"
    CACHE_SECONDS = 3600  # 1 hour

    async def get_current(self) -> dict:
        """Return current Fear & Greed value.

        Returns:
            dict with keys: value (int 0-100), value_classification (str),
            timestamp (int), stale (bool).
        """
        cache_key = "fear_greed_current"
        cached = self._get_cached(cache_key, self.CACHE_SECONDS)
        if cached:
            return cached.value

        client = await self._ensure_client()
        resp = await _request_with_retry(client, "GET", self.BASE_URL, params={"limit": "1"})

        if resp is not None:
            try:
                data = resp.json()["data"][0]
                result = {
                    "value": int(data["value"]),
                    "value_classification": data["value_classification"],
                    "timestamp": int(data["timestamp"]),
                    "stale": False,
                }
                self._set_cache(cache_key, result)
                return result
            except (KeyError, IndexError, ValueError) as exc:
                logger.warning("Failed to parse Fear & Greed response: %s", exc)

        # Graceful degradation
        stale = self._get_stale(cache_key)
        if stale:
            stale["stale"] = True
            return stale
        return {"value": 50, "value_classification": "Neutral", "timestamp": 0, "stale": True}

    async def get_historical(self, days: int = 30) -> list:
        """Return list of Fear & Greed entries for the last *days* days."""
        cache_key = f"fear_greed_hist_{days}"
        cached = self._get_cached(cache_key, self.CACHE_SECONDS)
        if cached:
            return cached.value

        client = await self._ensure_client()
        resp = await _request_with_retry(client, "GET", self.BASE_URL, params={"limit": str(days)})

        if resp is not None:
            try:
                entries = [
                    {
                        "value": int(d["value"]),
                        "value_classification": d["value_classification"],
                        "timestamp": int(d["timestamp"]),
                    }
                    for d in resp.json()["data"]
                ]
                self._set_cache(cache_key, entries)
                return entries
            except (KeyError, ValueError) as exc:
                logger.warning("Failed to parse historical Fear & Greed: %s", exc)

        stale = self._get_stale(cache_key)
        return stale if stale else []


# ---------------------------------------------------------------------------
# Glassnode on-chain metrics
# ---------------------------------------------------------------------------

class GlassnodeClient(BaseDataClient):
    """Fetch on-chain metrics from the Glassnode API."""

    BASE_URL = "https://api.glassnode.com/v1/metrics"
    CACHE_SECONDS = 4 * 3600  # 4 hours — on-chain data updates slowly

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self._api_key = api_key

    async def _fetch_metric(self, path: str, asset: str, since: Optional[int] = None) -> Optional[Any]:
        """Low-level fetcher shared by all Glassnode methods."""
        cache_key = f"glassnode:{path}:{asset}"
        cached = self._get_cached(cache_key, self.CACHE_SECONDS)
        if cached:
            return cached.value

        client = await self._ensure_client()
        params: dict = {"a": asset, "api_key": self._api_key}
        if since is not None:
            params["s"] = str(since)

        url = f"{self.BASE_URL}/{path}"
        resp = await _request_with_retry(client, "GET", url, params=params)

        if resp is not None:
            try:
                data = resp.json()
                self._set_cache(cache_key, data)
                return data
            except Exception as exc:
                logger.warning("Glassnode parse error for %s: %s", path, exc)

        stale = self._get_stale(cache_key)
        return stale

    # -- public methods ------------------------------------------------------

    async def get_exchange_net_flow(self, asset: str = "BTC", since: Optional[int] = None) -> Optional[Any]:
        """Net flow of *asset* into/out of exchanges."""
        return await self._fetch_metric("transactions/transfers_volume_exchanges_net", asset, since)

    async def get_mvrv_zscore(self, asset: str = "BTC") -> Optional[Any]:
        """Market Value to Realized Value Z-Score."""
        return await self._fetch_metric("market/mvrv_z_score", asset)

    async def get_nupl(self, asset: str = "BTC") -> Optional[Any]:
        """Net Unrealized Profit/Loss."""
        return await self._fetch_metric("indicators/net_unrealized_profit_loss", asset)

    async def get_sopr(self, asset: str = "BTC") -> Optional[Any]:
        """Spent Output Profit Ratio."""
        return await self._fetch_metric("indicators/sopr", asset)

    async def get_active_addresses(self, asset: str = "BTC") -> Optional[Any]:
        """Active addresses count."""
        return await self._fetch_metric("addresses/active_count", asset)

    async def get_hash_rate(self, asset: str = "BTC") -> Optional[Any]:
        """Mean hash rate."""
        return await self._fetch_metric("mining/hash_rate_mean", asset)

    async def get_stablecoin_reserves(self) -> Optional[Any]:
        """Stablecoin supply on exchanges (uses USDT as proxy)."""
        return await self._fetch_metric("distribution/exchange_net_position_change", "USDT")


# ---------------------------------------------------------------------------
# Social sentiment
# ---------------------------------------------------------------------------

class SentimentClient(BaseDataClient):
    """Aggregate social sentiment from LunarCrush and CryptoCompare."""

    LUNARCRUSH_BASE = "https://lunarcrush.com/api4/public"
    CRYPTOCOMPARE_BASE = "https://min-api.cryptocompare.com/data"
    CACHE_SECONDS = 3600  # 1 hour

    def __init__(self, lunarcrush_api_key: str = "", cryptocompare_api_key: str = "") -> None:
        super().__init__()
        self._lunarcrush_key = lunarcrush_api_key
        self._cryptocompare_key = cryptocompare_api_key

    async def get_social_volume(self, asset: str = "BTC") -> float:
        """Return social volume score for *asset* (higher = more social chatter).

        Tries LunarCrush first, falls back to CryptoCompare social stats.
        """
        cache_key = f"social_volume:{asset}"
        cached = self._get_cached(cache_key, self.CACHE_SECONDS)
        if cached:
            return cached.value

        client = await self._ensure_client()

        # Try LunarCrush
        if self._lunarcrush_key:
            headers = {"Authorization": f"Bearer {self._lunarcrush_key}"}
            resp = await _request_with_retry(
                client, "GET", f"{self.LUNARCRUSH_BASE}/coins/{asset.lower()}/v1",
                headers=headers,
            )
            if resp is not None:
                try:
                    data = resp.json().get("data", {})
                    volume = float(data.get("social_volume", data.get("social_mentions", 0)))
                    self._set_cache(cache_key, volume)
                    return volume
                except (KeyError, TypeError, ValueError) as exc:
                    logger.debug("LunarCrush social volume parse error: %s", exc)

        # Fallback: CryptoCompare
        if self._cryptocompare_key:
            params = {"coinId": asset.upper()}
            headers = {"Authorization": f"Apikey {self._cryptocompare_key}"}
            resp = await _request_with_retry(
                client, "GET", f"{self.CRYPTOCOMPARE_BASE}/social/coin/latest",
                params=params, headers=headers,
            )
            if resp is not None:
                try:
                    cc_data = resp.json().get("Data", {})
                    reddit = cc_data.get("Reddit", {})
                    twitter = cc_data.get("Twitter", {})
                    volume = float(reddit.get("posts_per_day", 0)) + float(twitter.get("statuses", 0))
                    self._set_cache(cache_key, volume)
                    return volume
                except (KeyError, TypeError, ValueError) as exc:
                    logger.debug("CryptoCompare social volume parse error: %s", exc)

        stale = self._get_stale(cache_key)
        return stale if stale is not None else 0.0

    async def get_sentiment_score(self, asset: str = "BTC") -> float:
        """Return sentiment score for *asset* (0-100 scale, 50 = neutral).

        Tries LunarCrush first, falls back to CryptoCompare.
        """
        cache_key = f"sentiment_score:{asset}"
        cached = self._get_cached(cache_key, self.CACHE_SECONDS)
        if cached:
            return cached.value

        client = await self._ensure_client()

        # Try LunarCrush
        if self._lunarcrush_key:
            headers = {"Authorization": f"Bearer {self._lunarcrush_key}"}
            resp = await _request_with_retry(
                client, "GET", f"{self.LUNARCRUSH_BASE}/coins/{asset.lower()}/v1",
                headers=headers,
            )
            if resp is not None:
                try:
                    data = resp.json().get("data", {})
                    # LunarCrush sentiment is 0-100
                    score = float(data.get("sentiment", data.get("galaxy_score", 50)))
                    self._set_cache(cache_key, score)
                    return score
                except (KeyError, TypeError, ValueError) as exc:
                    logger.debug("LunarCrush sentiment parse error: %s", exc)

        # Fallback: CryptoCompare
        if self._cryptocompare_key:
            params = {"coinId": asset.upper()}
            headers = {"Authorization": f"Apikey {self._cryptocompare_key}"}
            resp = await _request_with_retry(
                client, "GET", f"{self.CRYPTOCOMPARE_BASE}/social/coin/latest",
                params=params, headers=headers,
            )
            if resp is not None:
                try:
                    cc_data = resp.json().get("Data", {})
                    # Normalize CryptoCompare data to 0-100
                    general = cc_data.get("General", {})
                    points = float(general.get("Points", 50))
                    # Clamp to 0-100
                    score = max(0.0, min(100.0, points))
                    self._set_cache(cache_key, score)
                    return score
                except (KeyError, TypeError, ValueError) as exc:
                    logger.debug("CryptoCompare sentiment parse error: %s", exc)

        stale = self._get_stale(cache_key)
        return stale if stale is not None else 50.0
