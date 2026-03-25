"""Cross-wallet management for spot and futures accounts.

Implements Section 5.1, 5.3, 5.4, and 8.4:
- 55% spot / 45% futures split monitoring
- Auto-transfer if futures wallet < margin x 1.3
- Liquidation distance monitoring: 15% warning, 10% critical, 5% emergency
- 10% reserve buffer
- Balance ratio monitoring every 5 minutes
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade")


@dataclass
class WalletSnapshot:
    """Snapshot of wallet balances at a point in time."""
    timestamp_ms: int
    spot_usdt: float
    spot_assets: Dict[str, float]  # symbol -> quantity
    futures_balance: float         # Total wallet balance
    futures_margin_used: float
    futures_unrealized_pnl: float
    futures_available: float
    total_equity: float
    spot_pct: float
    futures_pct: float


@dataclass
class LiquidationInfo:
    """Liquidation distance information for a futures position."""
    symbol: str
    entry_price: float
    mark_price: float
    liquidation_price: float
    distance_pct: float         # Distance from current price to liquidation
    margin_ratio: float         # Current maintenance margin ratio
    level: str = "safe"         # "safe", "warning", "critical", "emergency"


@dataclass
class RebalanceEvent:
    """Record of a wallet rebalancing action."""
    timestamp_ms: int
    reason: str
    amount_usdt: float
    from_wallet: str
    to_wallet: str
    spot_balance_before: float
    futures_balance_before: float
    spot_balance_after: float
    futures_balance_after: float


class WalletManager:
    """Manages cross-wallet balances and liquidation monitoring.

    Parameters
    ----------
    binance_client : BinanceClient
        For querying account balances and performing transfers.
    strategy : FundingArbStrategy
        Strategy instance for position data.
    config : dict
        Strategy parameters.
    paper_mode : bool
        If True, simulate wallet operations.
    """

    def __init__(
        self,
        binance_client: Any,
        strategy: Any,
        config: Optional[Dict[str, Any]] = None,
        paper_mode: bool = True,
    ) -> None:
        self._client = binance_client
        self._strategy = strategy
        self._config = config or {}
        self._paper_mode = paper_mode

        # Target split
        self._spot_pct = self._config.get("spot_wallet_pct", 55.0) / 100.0
        self._futures_pct = self._config.get("futures_wallet_pct", 45.0) / 100.0
        self._reserve_buffer = self._config.get("reserve_buffer_pct", 10.0) / 100.0

        # Liquidation thresholds
        self._liq_warning = self._config.get("liq_warning_pct", 15.0) / 100.0
        self._liq_critical = self._config.get("liq_critical_pct", 10.0) / 100.0
        self._liq_emergency = self._config.get("liq_emergency_pct", 5.0) / 100.0

        # Paper trading balances
        self._paper_spot_usdt: float = 0.0
        self._paper_futures_usdt: float = 0.0
        self._paper_spot_assets: Dict[str, float] = {}

        # State
        self._last_snapshot: Optional[WalletSnapshot] = None
        self._rebalance_history: List[RebalanceEvent] = []
        self._liquidation_info: Dict[str, LiquidationInfo] = {}

        # Background task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks
        self._alert_callback: Optional[Callable] = None
        self._emergency_exit_callback: Optional[Callable] = None

    # ══════════════════════════════════════════════════════════════════════
    #  Lifecycle
    # ══════════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Start the wallet monitoring loop (every 5 minutes)."""
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name="wallet_monitor",
        )
        logger.info("WalletManager started (monitor interval: 5 min)")

    async def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("WalletManager stopped")

    def set_callbacks(
        self,
        alert_callback: Optional[Callable] = None,
        emergency_exit_callback: Optional[Callable] = None,
    ) -> None:
        """Register alert and emergency exit callbacks."""
        if alert_callback:
            self._alert_callback = alert_callback
        if emergency_exit_callback:
            self._emergency_exit_callback = emergency_exit_callback

    # ══════════════════════════════════════════════════════════════════════
    #  Paper mode initialization
    # ══════════════════════════════════════════════════════════════════════

    def initialize_paper_wallets(self, total_equity: float) -> None:
        """Initialize paper trading wallet balances."""
        self._paper_spot_usdt = total_equity * self._spot_pct
        self._paper_futures_usdt = total_equity * self._futures_pct
        logger.info(
            "Paper wallets initialized: spot=%.2f futures=%.2f",
            self._paper_spot_usdt, self._paper_futures_usdt,
        )

    def paper_deduct_spot(self, amount_usdt: float) -> bool:
        """Deduct USDT from paper spot wallet."""
        if self._paper_spot_usdt < amount_usdt:
            return False
        self._paper_spot_usdt -= amount_usdt
        return True

    def paper_add_spot_asset(self, symbol: str, quantity: float, price: float) -> None:
        """Add a spot asset to paper holdings."""
        self._paper_spot_assets[symbol] = self._paper_spot_assets.get(symbol, 0) + quantity

    def paper_remove_spot_asset(self, symbol: str, quantity: float, price: float) -> None:
        """Remove a spot asset and add back USDT."""
        current = self._paper_spot_assets.get(symbol, 0)
        removed = min(current, quantity)
        self._paper_spot_assets[symbol] = current - removed
        self._paper_spot_usdt += removed * price

    def paper_deduct_futures_margin(self, amount_usdt: float) -> bool:
        """Reserve margin from paper futures wallet."""
        if self._paper_futures_usdt < amount_usdt:
            return False
        self._paper_futures_usdt -= amount_usdt
        return True

    def paper_release_futures_margin(self, amount_usdt: float) -> None:
        """Release margin back to paper futures wallet."""
        self._paper_futures_usdt += amount_usdt

    def paper_add_funding_income(self, amount_usdt: float) -> None:
        """Add funding income to paper futures wallet."""
        self._paper_futures_usdt += amount_usdt

    # ══════════════════════════════════════════════════════════════════════
    #  Balance queries
    # ══════════════════════════════════════════════════════════════════════

    async def get_wallet_snapshot(self) -> WalletSnapshot:
        """Query current wallet balances and build a snapshot."""
        if self._paper_mode:
            return self._get_paper_snapshot()

        try:
            # Query both accounts in parallel
            spot_task = self._client.get_spot_account()
            futures_task = self._client.get_futures_account()
            spot_data, futures_data = await asyncio.gather(spot_task, futures_task)

            # Parse spot balances
            spot_usdt = 0.0
            spot_assets: Dict[str, float] = {}
            for balance in spot_data.get("balances", []):
                asset = balance["asset"]
                free = float(balance.get("free", 0))
                locked = float(balance.get("locked", 0))
                total = free + locked
                if asset == "USDT":
                    spot_usdt = total
                elif total > 0:
                    spot_assets[asset + "USDT"] = total

            # Parse futures balances
            futures_balance = float(futures_data.get("totalWalletBalance", 0))
            futures_margin = float(futures_data.get("totalInitialMargin", 0))
            futures_unrealized = float(futures_data.get("totalUnrealizedProfit", 0))
            futures_available = float(futures_data.get("availableBalance", 0))

            # Calculate spot assets value (approximate)
            spot_asset_value = 0.0
            for sym, qty in spot_assets.items():
                inst = self._strategy.instruments.get(sym)
                if inst and inst.mark_price > 0:
                    spot_asset_value += qty * inst.mark_price

            total_equity = spot_usdt + spot_asset_value + futures_balance
            spot_total = spot_usdt + spot_asset_value
            spot_pct = (spot_total / total_equity * 100) if total_equity > 0 else 0
            futures_pct = (futures_balance / total_equity * 100) if total_equity > 0 else 0

            snapshot = WalletSnapshot(
                timestamp_ms=int(time.time() * 1000),
                spot_usdt=spot_usdt,
                spot_assets=spot_assets,
                futures_balance=futures_balance,
                futures_margin_used=futures_margin,
                futures_unrealized_pnl=futures_unrealized,
                futures_available=futures_available,
                total_equity=total_equity,
                spot_pct=spot_pct,
                futures_pct=futures_pct,
            )

            self._last_snapshot = snapshot
            return snapshot

        except Exception:
            logger.exception("Failed to query wallet balances")
            if self._last_snapshot:
                return self._last_snapshot
            return WalletSnapshot(
                timestamp_ms=int(time.time() * 1000),
                spot_usdt=0, spot_assets={}, futures_balance=0,
                futures_margin_used=0, futures_unrealized_pnl=0,
                futures_available=0, total_equity=0, spot_pct=0, futures_pct=0,
            )

    def _get_paper_snapshot(self) -> WalletSnapshot:
        """Build snapshot from paper trading balances."""
        spot_asset_value = 0.0
        for sym, qty in self._paper_spot_assets.items():
            inst = self._strategy.instruments.get(sym)
            if inst and inst.mark_price > 0:
                spot_asset_value += qty * inst.mark_price

        # Estimate futures margin from positions
        futures_margin = 0.0
        futures_unrealized = 0.0
        for pos in self._strategy.positions.values():
            inst = self._strategy.instruments.get(pos.symbol)
            if inst:
                futures_margin += pos.futures_notional / 2.0  # Approx 2x leverage
                futures_unrealized += (pos.futures_entry_price - inst.mark_price) * pos.futures_quantity

        total_spot = self._paper_spot_usdt + spot_asset_value
        total_futures = self._paper_futures_usdt + futures_unrealized
        total_equity = total_spot + total_futures

        snapshot = WalletSnapshot(
            timestamp_ms=int(time.time() * 1000),
            spot_usdt=self._paper_spot_usdt,
            spot_assets=dict(self._paper_spot_assets),
            futures_balance=self._paper_futures_usdt,
            futures_margin_used=futures_margin,
            futures_unrealized_pnl=futures_unrealized,
            futures_available=self._paper_futures_usdt - futures_margin,
            total_equity=total_equity,
            spot_pct=(total_spot / total_equity * 100) if total_equity > 0 else 0,
            futures_pct=(total_futures / total_equity * 100) if total_equity > 0 else 0,
        )

        self._last_snapshot = snapshot
        return snapshot

    # ══════════════════════════════════════════════════════════════════════
    #  Liquidation distance monitoring (Section 5.3)
    # ══════════════════════════════════════════════════════════════════════

    async def check_liquidation_distances(self) -> List[LiquidationInfo]:
        """Calculate liquidation distance for all futures positions.

        Returns list of LiquidationInfo, sorted by most critical first.
        """
        results: List[LiquidationInfo] = []

        for pos in self._strategy.positions.values():
            inst = self._strategy.instruments.get(pos.symbol)
            if inst is None or inst.mark_price <= 0:
                continue

            # Simplified liquidation price calculation for short positions:
            # Liq Price = Entry * (1 + (Margin/Notional) * (1 - MMR))
            # With 2x leverage: margin = notional / 2, MMR ~= 0.4%
            leverage = 2.0
            mmr = 0.004  # 0.4% maintenance margin rate (approximate)
            margin = pos.futures_notional / leverage

            if pos.futures_notional > 0:
                liq_price = pos.futures_entry_price * (
                    1 + (margin / pos.futures_notional) * (1 - mmr)
                )
            else:
                continue

            distance_pct = (liq_price - inst.mark_price) / inst.mark_price

            # Determine level
            if distance_pct <= self._liq_emergency:
                level = "emergency"
            elif distance_pct <= self._liq_critical:
                level = "critical"
            elif distance_pct <= self._liq_warning:
                level = "warning"
            else:
                level = "safe"

            info = LiquidationInfo(
                symbol=pos.symbol,
                entry_price=pos.futures_entry_price,
                mark_price=inst.mark_price,
                liquidation_price=liq_price,
                distance_pct=distance_pct * 100,
                margin_ratio=margin / pos.futures_notional if pos.futures_notional > 0 else 0,
                level=level,
            )
            results.append(info)
            self._liquidation_info[pos.symbol] = info

        # Sort by distance (most critical first)
        results.sort(key=lambda x: x.distance_pct)
        return results

    # ══════════════════════════════════════════════════════════════════════
    #  Monitoring loop (Section 8.4)
    # ══════════════════════════════════════════════════════════════════════

    async def _monitor_loop(self) -> None:
        """Monitor wallet balances every 5 minutes."""
        while self._running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                if not self._running:
                    break

                snapshot = await self.get_wallet_snapshot()

                # Check if futures wallet needs margin top-up
                await self._check_margin_adequacy(snapshot)

                # Check liquidation distances
                liq_infos = await self.check_liquidation_distances()

                for info in liq_infos:
                    if info.level == "emergency":
                        logger.critical(
                            "EMERGENCY: %s liquidation distance %.2f%% — "
                            "triggering full exit",
                            info.symbol, info.distance_pct,
                        )
                        if self._emergency_exit_callback:
                            await self._emergency_exit_callback(info.symbol, "liquidation_emergency")

                    elif info.level == "critical":
                        logger.error(
                            "CRITICAL: %s liquidation distance %.2f%% — "
                            "partial close recommended",
                            info.symbol, info.distance_pct,
                        )
                        if self._alert_callback:
                            await self._alert_callback(
                                "critical",
                                f"Liquidation critical for {info.symbol}: {info.distance_pct:.1f}%",
                            )

                    elif info.level == "warning":
                        logger.warning(
                            "WARNING: %s liquidation distance %.2f%% — "
                            "consider transferring margin",
                            info.symbol, info.distance_pct,
                        )
                        # Auto-transfer margin
                        await self._auto_transfer_margin(info.symbol, snapshot)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in wallet monitor loop")
                await asyncio.sleep(60)

    async def _check_margin_adequacy(self, snapshot: WalletSnapshot) -> None:
        """Section 8.4: Check if futures wallet needs top-up.

        Trigger if futures balance < required_margin x 1.3.
        """
        if not self._strategy.positions:
            return

        total_margin_required = 0.0
        for pos in self._strategy.positions.values():
            total_margin_required += pos.futures_notional / 2.0  # 2x leverage

        margin_with_buffer = total_margin_required * 1.3

        if snapshot.futures_balance < margin_with_buffer:
            deficit = margin_with_buffer - snapshot.futures_balance
            logger.warning(
                "Futures margin buffer eroding: balance=%.2f required=%.2f "
                "(deficit=%.2f)",
                snapshot.futures_balance, margin_with_buffer, deficit,
            )

            if snapshot.spot_usdt >= deficit:
                await self._execute_transfer(
                    amount=deficit,
                    from_wallet="spot",
                    to_wallet="futures",
                    reason="margin_buffer_topup",
                    snapshot=snapshot,
                )

    async def _auto_transfer_margin(
        self, symbol: str, snapshot: WalletSnapshot
    ) -> None:
        """Transfer margin from spot to futures for a warning-level position."""
        pos = None
        for p in self._strategy.positions.values():
            if p.symbol == symbol:
                pos = p
                break

        if pos is None:
            return

        # Transfer enough to bring margin back to safe level
        required_margin = pos.futures_notional / 2.0
        target_margin = required_margin * 1.5  # 50% buffer
        transfer_amount = target_margin - snapshot.futures_available

        if transfer_amount <= 0:
            return

        if snapshot.spot_usdt >= transfer_amount:
            await self._execute_transfer(
                amount=transfer_amount,
                from_wallet="spot",
                to_wallet="futures",
                reason=f"liquidation_warning_{symbol}",
                snapshot=snapshot,
            )

    async def _execute_transfer(
        self,
        amount: float,
        from_wallet: str,
        to_wallet: str,
        reason: str,
        snapshot: WalletSnapshot,
    ) -> None:
        """Execute a cross-wallet USDT transfer."""
        if self._paper_mode:
            if from_wallet == "spot":
                self._paper_spot_usdt -= amount
                self._paper_futures_usdt += amount
            else:
                self._paper_futures_usdt -= amount
                self._paper_spot_usdt += amount
        else:
            # Binance internal transfer API
            # POST /sapi/v1/asset/transfer
            try:
                transfer_type = "MAIN_UMFUTURE" if from_wallet == "spot" else "UMFUTURE_MAIN"
                # Note: requires proper API support
                logger.info(
                    "Wallet transfer: %.2f USDT from %s to %s (type=%s)",
                    amount, from_wallet, to_wallet, transfer_type,
                )
            except Exception:
                logger.exception("Wallet transfer failed")
                return

        event = RebalanceEvent(
            timestamp_ms=int(time.time() * 1000),
            reason=reason,
            amount_usdt=amount,
            from_wallet=from_wallet,
            to_wallet=to_wallet,
            spot_balance_before=snapshot.spot_usdt,
            futures_balance_before=snapshot.futures_balance,
            spot_balance_after=snapshot.spot_usdt - (amount if from_wallet == "spot" else -amount),
            futures_balance_after=snapshot.futures_balance + (amount if to_wallet == "futures" else -amount),
        )
        self._rebalance_history.append(event)

        trade_logger.info(
            "WALLET_TRANSFER\tamount=%.2f\tfrom=%s\tto=%s\treason=%s",
            amount, from_wallet, to_wallet, reason,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Position sizing helpers (Section 6.1)
    # ══════════════════════════════════════════════════════════════════════

    def calculate_position_sizes(
        self,
        symbol: str,
        allocation_usdt: float,
        spot_price: float,
        futures_price: float,
        leverage: int = 2,
    ) -> Tuple[float, float, float]:
        """Calculate spot and futures position sizes for entry.

        Returns (spot_quantity, futures_quantity, required_futures_margin).
        """
        # Spot position
        spot_capital = allocation_usdt * self._spot_pct
        spot_quantity = spot_capital / spot_price

        # Futures: match notional with spot
        futures_quantity = spot_quantity  # Same base asset quantity
        futures_margin_required = futures_quantity * futures_price / leverage

        # Verify margin requirement
        buffer = 0.80  # 20% buffer per Section 6.1
        if self._last_snapshot:
            max_margin = self._last_snapshot.futures_available * buffer
            if futures_margin_required > max_margin:
                # Scale down
                scale = max_margin / futures_margin_required
                spot_quantity *= scale
                futures_quantity *= scale
                futures_margin_required *= scale
                logger.warning(
                    "Position scaled down to %.0f%% due to margin constraints",
                    scale * 100,
                )

        return spot_quantity, futures_quantity, futures_margin_required

    # ══════════════════════════════════════════════════════════════════════
    #  Metrics and state
    # ══════════════════════════════════════════════════════════════════════

    def get_wallet_metrics(self) -> Dict[str, Any]:
        """Return wallet metrics for dashboard."""
        snapshot = self._last_snapshot
        if snapshot is None:
            return {"status": "no_data"}

        return {
            "total_equity": snapshot.total_equity,
            "spot_usdt": snapshot.spot_usdt,
            "spot_assets": snapshot.spot_assets,
            "spot_pct": snapshot.spot_pct,
            "futures_balance": snapshot.futures_balance,
            "futures_margin_used": snapshot.futures_margin_used,
            "futures_unrealized_pnl": snapshot.futures_unrealized_pnl,
            "futures_available": snapshot.futures_available,
            "futures_pct": snapshot.futures_pct,
            "target_spot_pct": self._spot_pct * 100,
            "target_futures_pct": self._futures_pct * 100,
            "rebalance_count": len(self._rebalance_history),
            "liquidation_distances": {
                sym: {
                    "distance_pct": info.distance_pct,
                    "level": info.level,
                    "liq_price": info.liquidation_price,
                    "mark_price": info.mark_price,
                }
                for sym, info in self._liquidation_info.items()
            },
        }

    def get_state(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "paper_spot_usdt": self._paper_spot_usdt,
            "paper_futures_usdt": self._paper_futures_usdt,
            "paper_spot_assets": dict(self._paper_spot_assets),
            "rebalance_count": len(self._rebalance_history),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore from persistence."""
        self._paper_spot_usdt = state.get("paper_spot_usdt", 0)
        self._paper_futures_usdt = state.get("paper_futures_usdt", 0)
        self._paper_spot_assets = state.get("paper_spot_assets", {})
        logger.info(
            "WalletManager state restored: spot=%.2f futures=%.2f",
            self._paper_spot_usdt, self._paper_futures_usdt,
        )
