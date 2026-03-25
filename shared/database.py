"""Database manager: wraps engine/session lifecycle and provides CRUD helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from sqlalchemy import Engine, select, desc
from sqlalchemy.orm import Session, sessionmaker

from .models.base import get_engine, init_db
from .models.orders import Order, Fill, Trade, Position
from .models.performance import DailyPnL, EquityCurve, DrawdownTracker
from .models.state import StateSnapshot


class DatabaseManager:
    """Synchronous database manager for the trading bot library.

    Usage::

        db = DatabaseManager("sqlite:///my_strategy.db")
        with db.get_session() as session:
            session.add(...)
        # or use the convenience helpers:
        db.save_order({...})
    """

    def __init__(self, db_url: str = "sqlite:///trading.db") -> None:
        self._engine: Engine = get_engine(db_url)
        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False
        )
        init_db(self._engine)

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Yield a session that auto-commits on success and rolls back on error."""
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def save_order(self, order_data: dict[str, Any]) -> Order:
        """Insert or update an order. Returns the persisted Order."""
        with self.get_session() as session:
            existing = session.execute(
                select(Order).where(Order.order_id == order_data.get("order_id"))
            ).scalar_one_or_none()
            if existing is not None:
                for key, value in order_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = int(time.time() * 1000)
                session.flush()
                return existing
            order = Order(**order_data)
            session.add(order)
            session.flush()
            return order

    def get_open_orders(self, strategy_id: str) -> list[Order]:
        """Return all orders with status NEW or PARTIALLY_FILLED for a strategy."""
        with self.get_session() as session:
            stmt = (
                select(Order)
                .where(Order.strategy_id == strategy_id)
                .where(Order.status.in_(["NEW", "PARTIALLY_FILLED"]))
                .order_by(desc(Order.created_at))
            )
            return list(session.execute(stmt).scalars().all())

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def save_fill(self, fill_data: dict[str, Any]) -> Fill:
        """Insert a fill record."""
        with self.get_session() as session:
            fill = Fill(**fill_data)
            session.add(fill)
            session.flush()
            return fill

    def get_fills_for_order(self, order_id: str) -> list[Fill]:
        """Return all fills for the given exchange order_id."""
        with self.get_session() as session:
            stmt = (
                select(Fill)
                .where(Fill.order_id == order_id)
                .order_by(Fill.timestamp_ms)
            )
            return list(session.execute(stmt).scalars().all())

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def save_trade(self, trade_data: dict[str, Any]) -> Trade:
        """Insert or update a trade record. Matches on trade_id."""
        with self.get_session() as session:
            existing = session.execute(
                select(Trade).where(Trade.trade_id == trade_data.get("trade_id"))
            ).scalar_one_or_none()
            if existing is not None:
                for key, value in trade_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                session.flush()
                return existing
            trade = Trade(**trade_data)
            session.add(trade)
            session.flush()
            return trade

    def get_trades(self, strategy_id: str, limit: int = 100) -> list[Trade]:
        """Return the most recent trades for a strategy."""
        with self.get_session() as session:
            stmt = (
                select(Trade)
                .where(Trade.strategy_id == strategy_id)
                .order_by(desc(Trade.entry_time_ms))
                .limit(limit)
            )
            return list(session.execute(stmt).scalars().all())

    def get_open_trades(self, strategy_id: str) -> list[Trade]:
        """Return trades that have no exit_time_ms (still open)."""
        with self.get_session() as session:
            stmt = (
                select(Trade)
                .where(Trade.strategy_id == strategy_id)
                .where(Trade.exit_time_ms.is_(None))
                .order_by(desc(Trade.entry_time_ms))
            )
            return list(session.execute(stmt).scalars().all())

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def save_position(self, pos_data: dict[str, Any]) -> Position:
        """Insert or update a position. Matches on (strategy_id, symbol, is_open)."""
        with self.get_session() as session:
            existing = session.execute(
                select(Position)
                .where(Position.strategy_id == pos_data.get("strategy_id"))
                .where(Position.symbol == pos_data.get("symbol"))
                .where(Position.is_open == True)  # noqa: E712
            ).scalar_one_or_none()
            if existing is not None:
                for key, value in pos_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                session.flush()
                return existing
            position = Position(**pos_data)
            session.add(position)
            session.flush()
            return position

    def get_open_positions(self, strategy_id: str) -> list[Position]:
        """Return all open positions for a strategy."""
        with self.get_session() as session:
            stmt = (
                select(Position)
                .where(Position.strategy_id == strategy_id)
                .where(Position.is_open == True)  # noqa: E712
            )
            return list(session.execute(stmt).scalars().all())

    def close_position(self, position_id: int, exit_data: dict[str, Any]) -> Position:
        """Mark a position as closed and apply exit_data fields.

        Args:
            position_id: The Position.id to close.
            exit_data: Dict with fields to update (e.g. unrealized_pnl, size=0).
        """
        with self.get_session() as session:
            position = session.get(Position, position_id)
            if position is None:
                raise ValueError(f"Position {position_id} not found")
            for key, value in exit_data.items():
                if hasattr(position, key):
                    setattr(position, key, value)
            position.is_open = False
            session.flush()
            return position

    # ------------------------------------------------------------------
    # Daily PnL
    # ------------------------------------------------------------------

    def save_daily_pnl(self, data: dict[str, Any]) -> DailyPnL:
        """Insert or update a daily PnL record. Matches on (strategy_id, date)."""
        with self.get_session() as session:
            existing = session.execute(
                select(DailyPnL)
                .where(DailyPnL.strategy_id == data.get("strategy_id"))
                .where(DailyPnL.date == data.get("date"))
            ).scalar_one_or_none()
            if existing is not None:
                for key, value in data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                session.flush()
                return existing
            record = DailyPnL(**data)
            session.add(record)
            session.flush()
            return record

    def get_daily_pnl(self, strategy_id: str, days: int = 30) -> list[DailyPnL]:
        """Return the most recent N days of PnL records."""
        with self.get_session() as session:
            stmt = (
                select(DailyPnL)
                .where(DailyPnL.strategy_id == strategy_id)
                .order_by(desc(DailyPnL.date))
                .limit(days)
            )
            return list(session.execute(stmt).scalars().all())

    # ------------------------------------------------------------------
    # Equity Curve
    # ------------------------------------------------------------------

    def save_equity_point(
        self, strategy_id: str, equity: float, drawdown_pct: float
    ) -> EquityCurve:
        """Append a point to the equity curve."""
        with self.get_session() as session:
            point = EquityCurve(
                strategy_id=strategy_id,
                timestamp_ms=int(time.time() * 1000),
                equity=equity,
                drawdown_pct=drawdown_pct,
            )
            session.add(point)
            session.flush()
            return point

    # ------------------------------------------------------------------
    # State Snapshots
    # ------------------------------------------------------------------

    def save_state_snapshot(
        self, strategy_id: str, state_json: str, snapshot_type: str
    ) -> StateSnapshot:
        """Persist a strategy state snapshot."""
        with self.get_session() as session:
            snap = StateSnapshot(
                strategy_id=strategy_id,
                timestamp_ms=int(time.time() * 1000),
                state_json=state_json,
                snapshot_type=snapshot_type,
            )
            session.add(snap)
            session.flush()
            return snap

    def get_latest_state(
        self, strategy_id: str, snapshot_type: str
    ) -> Optional[StateSnapshot]:
        """Return the most recent state snapshot of the given type."""
        with self.get_session() as session:
            stmt = (
                select(StateSnapshot)
                .where(StateSnapshot.strategy_id == strategy_id)
                .where(StateSnapshot.snapshot_type == snapshot_type)
                .order_by(desc(StateSnapshot.timestamp_ms))
                .limit(1)
            )
            return session.execute(stmt).scalar_one_or_none()

    # ------------------------------------------------------------------
    # Drawdown Tracker
    # ------------------------------------------------------------------

    def save_drawdown_tracker(self, data: dict[str, Any]) -> DrawdownTracker:
        """Insert or update the drawdown tracker for a strategy (one row per strategy)."""
        with self.get_session() as session:
            existing = session.execute(
                select(DrawdownTracker).where(
                    DrawdownTracker.strategy_id == data.get("strategy_id")
                )
            ).scalar_one_or_none()
            if existing is not None:
                for key, value in data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                session.flush()
                return existing
            tracker = DrawdownTracker(**data)
            session.add(tracker)
            session.flush()
            return tracker

    def get_drawdown_tracker(self, strategy_id: str) -> Optional[DrawdownTracker]:
        """Return the drawdown tracker row for a strategy, or None."""
        with self.get_session() as session:
            return session.execute(
                select(DrawdownTracker).where(
                    DrawdownTracker.strategy_id == strategy_id
                )
            ).scalar_one_or_none()
