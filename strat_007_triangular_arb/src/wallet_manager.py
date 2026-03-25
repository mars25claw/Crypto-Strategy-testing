"""STRAT-007: Wallet manager for spot/futures balance allocation.

Maintains 50/50 split between spot and futures wallets.
Rebalances every 4 hours if deviation exceeds 10%.
Tracks BNB balance for fee discount eligibility.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
system_logger = logging.getLogger("system")


@dataclass
class WalletState:
    """Snapshot of wallet balances."""

    spot_usdt: float = 0.0
    futures_usdt: float = 0.0
    spot_bnb: float = 0.0
    futures_bnb: float = 0.0
    total_usdt: float = 0.0
    spot_pct: float = 50.0
    futures_pct: float = 50.0
    deviation_pct: float = 0.0
    last_updated: float = 0.0

    def to_dict(self) -> dict:
        return {
            "spot_usdt": round(self.spot_usdt, 4),
            "futures_usdt": round(self.futures_usdt, 4),
            "spot_bnb": round(self.spot_bnb, 4),
            "futures_bnb": round(self.futures_bnb, 4),
            "total_usdt": round(self.total_usdt, 4),
            "spot_pct": round(self.spot_pct, 2),
            "futures_pct": round(self.futures_pct, 2),
            "deviation_pct": round(self.deviation_pct, 2),
            "last_updated": self.last_updated,
        }


class WalletManager:
    """Manages spot/futures wallet allocation and rebalancing.

    Parameters
    ----------
    binance_client : BinanceClient
        REST client for account queries and internal transfers.
    params : dict
        Strategy parameters.
    paper_mode : bool
        Paper trading mode — simulate balances.
    starting_equity : float
        Starting equity for paper mode.
    """

    def __init__(
        self,
        binance_client: Any,
        params: Dict[str, Any],
        paper_mode: bool = True,
        starting_equity: float = 10000.0,
    ) -> None:
        self._client = binance_client
        self._params = params
        self._paper_mode = paper_mode

        # Target allocation
        self._spot_target_pct = params.get("spot_wallet_target_pct", 50.0)
        self._futures_target_pct = params.get("futures_wallet_target_pct", 50.0)
        self._deviation_threshold = params.get("wallet_deviation_threshold_pct", 10.0)
        self._rebalance_interval_s = params.get("wallet_rebalance_interval_h", 4) * 3600
        self._min_bnb_balance = params.get("min_bnb_balance", 0.5)

        # State
        self._state = WalletState()
        self._last_rebalance_time: float = 0.0
        self._rebalance_count: int = 0

        # Paper mode balances
        if paper_mode:
            half = starting_equity / 2.0
            self._state.spot_usdt = half
            self._state.futures_usdt = half
            self._state.total_usdt = starting_equity
            self._state.spot_pct = 50.0
            self._state.futures_pct = 50.0
            self._state.spot_bnb = 1.0  # Assume some BNB in paper mode
            self._state.last_updated = time.time()

        logger.info(
            "WalletManager initialized: paper=%s target=%.0f/%.0f "
            "deviation_threshold=%.1f%% rebalance_interval=%dh",
            paper_mode, self._spot_target_pct, self._futures_target_pct,
            self._deviation_threshold, params.get("wallet_rebalance_interval_h", 4),
        )

    # ------------------------------------------------------------------
    # Balance queries
    # ------------------------------------------------------------------

    async def refresh_balances(self) -> WalletState:
        """Fetch current balances from the exchange (or return paper state)."""
        if self._paper_mode:
            self._update_percentages()
            return self._state

        try:
            # Fetch spot balance
            spot_account = await self._client.get_spot_account()
            spot_usdt = 0.0
            spot_bnb = 0.0
            for bal in spot_account.get("balances", []):
                asset = bal.get("asset", "")
                free = float(bal.get("free", 0))
                if asset == "USDT":
                    spot_usdt = free
                elif asset == "BNB":
                    spot_bnb = free

            # Fetch futures balance
            futures_account = await self._client.get_futures_account()
            futures_usdt = 0.0
            futures_bnb = 0.0
            for asset_info in futures_account.get("assets", []):
                a = asset_info.get("asset", "")
                avail = float(asset_info.get("availableBalance", 0))
                if a == "USDT":
                    futures_usdt = avail
                elif a == "BNB":
                    futures_bnb = avail

            self._state.spot_usdt = spot_usdt
            self._state.futures_usdt = futures_usdt
            self._state.spot_bnb = spot_bnb
            self._state.futures_bnb = futures_bnb
            self._state.total_usdt = spot_usdt + futures_usdt
            self._state.last_updated = time.time()

            self._update_percentages()

            logger.debug(
                "Balances refreshed: spot=%.2f futures=%.2f total=%.2f BNB=%.4f",
                spot_usdt, futures_usdt, self._state.total_usdt,
                spot_bnb + futures_bnb,
            )

        except Exception as exc:
            logger.error("Failed to refresh balances: %s", exc)

        return self._state

    def _update_percentages(self) -> None:
        """Recalculate allocation percentages."""
        total = self._state.spot_usdt + self._state.futures_usdt
        if total > 0:
            self._state.spot_pct = (self._state.spot_usdt / total) * 100.0
            self._state.futures_pct = (self._state.futures_usdt / total) * 100.0
            self._state.deviation_pct = abs(self._state.spot_pct - self._spot_target_pct)
        self._state.total_usdt = total

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def needs_rebalance(self) -> bool:
        """Check if wallets need rebalancing."""
        if time.time() - self._last_rebalance_time < self._rebalance_interval_s:
            return False
        return self._state.deviation_pct > self._deviation_threshold

    async def rebalance(self) -> bool:
        """Rebalance spot and futures wallets to target allocation.

        Returns True if rebalance was performed.
        """
        if not self.needs_rebalance():
            return False

        total = self._state.total_usdt
        if total <= 0:
            return False

        target_spot = total * (self._spot_target_pct / 100.0)
        target_futures = total * (self._futures_target_pct / 100.0)

        diff_spot = target_spot - self._state.spot_usdt
        # positive = need to move from futures to spot
        # negative = need to move from spot to futures

        transfer_amount = abs(diff_spot)
        if transfer_amount < 1.0:  # Minimum transfer $1
            return False

        if self._paper_mode:
            # Simulate transfer
            if diff_spot > 0:
                # Transfer from futures to spot
                self._state.futures_usdt -= transfer_amount
                self._state.spot_usdt += transfer_amount
            else:
                # Transfer from spot to futures
                self._state.spot_usdt -= transfer_amount
                self._state.futures_usdt += transfer_amount

            self._update_percentages()
            self._last_rebalance_time = time.time()
            self._rebalance_count += 1

            logger.info(
                "Paper rebalance: transferred %.2f USDT %s → %s. "
                "New split: spot=%.1f%% futures=%.1f%%",
                transfer_amount,
                "futures" if diff_spot > 0 else "spot",
                "spot" if diff_spot > 0 else "futures",
                self._state.spot_pct, self._state.futures_pct,
            )
            return True

        # Live mode: use Binance internal transfer API
        try:
            if diff_spot > 0:
                # Transfer USDT from futures to spot
                # type=2: UMFUTURE -> MAIN
                params = {
                    "asset": "USDT",
                    "amount": str(round(transfer_amount, 8)),
                    "type": 2,  # UMFUTURE to MAIN
                }
                await self._client._request(
                    "POST", "/sapi/v1/futures/transfer",
                    params=params, api_type="spot",
                )
            else:
                # Transfer USDT from spot to futures
                # type=1: MAIN -> UMFUTURE
                params = {
                    "asset": "USDT",
                    "amount": str(round(transfer_amount, 8)),
                    "type": 1,  # MAIN to UMFUTURE
                }
                await self._client._request(
                    "POST", "/sapi/v1/futures/transfer",
                    params=params, api_type="spot",
                )

            self._last_rebalance_time = time.time()
            self._rebalance_count += 1

            logger.info(
                "Wallet rebalanced: transferred %.2f USDT %s → %s",
                transfer_amount,
                "futures" if diff_spot > 0 else "spot",
                "spot" if diff_spot > 0 else "futures",
            )
            system_logger.info(
                "WALLET_REBALANCE\tamount=%.2f\tdirection=%s",
                transfer_amount, "futures_to_spot" if diff_spot > 0 else "spot_to_futures",
            )

            # Refresh balances after transfer
            await self.refresh_balances()
            return True

        except Exception as exc:
            logger.error("Wallet rebalance failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # BNB discount check
    # ------------------------------------------------------------------

    def has_bnb_for_discount(self) -> bool:
        """Check if there's enough BNB balance for fee discount."""
        total_bnb = self._state.spot_bnb + self._state.futures_bnb
        return total_bnb >= self._min_bnb_balance

    # ------------------------------------------------------------------
    # Available balance for trading
    # ------------------------------------------------------------------

    def get_available_balance(self) -> float:
        """Return the total USDT available across both wallets."""
        return self._state.spot_usdt + self._state.futures_usdt

    def get_spot_available(self) -> float:
        """Return available spot USDT."""
        return self._state.spot_usdt

    def get_futures_available(self) -> float:
        """Return available futures USDT."""
        return self._state.futures_usdt

    # ------------------------------------------------------------------
    # Paper mode balance updates
    # ------------------------------------------------------------------

    def update_paper_balance(self, delta: float, market: str = "spot") -> None:
        """Update paper trading balance after a trade.

        Parameters
        ----------
        delta : float
            Positive = gained USDT, negative = spent USDT.
        market : str
            "spot" or "futures".
        """
        if not self._paper_mode:
            return

        if market == "spot":
            self._state.spot_usdt += delta
        else:
            self._state.futures_usdt += delta

        self._update_percentages()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> WalletState:
        """Return current wallet state."""
        return self._state

    def get_stats(self) -> Dict[str, Any]:
        """Return wallet statistics."""
        return {
            **self._state.to_dict(),
            "rebalance_count": self._rebalance_count,
            "last_rebalance_time": self._last_rebalance_time,
            "needs_rebalance": self.needs_rebalance(),
            "bnb_discount_eligible": self.has_bnb_for_discount(),
        }
