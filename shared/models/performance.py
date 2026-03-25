"""Performance tracking ORM models: daily PnL, equity curve, drawdown."""

from typing import Optional

from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class DailyPnL(Base):
    """Aggregated daily performance metrics for a strategy."""

    __tablename__ = "daily_pnl"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(64), index=True)
    date: Mapped[str] = mapped_column(String(10), index=True, comment="YYYY-MM-DD")
    starting_equity: Mapped[float] = mapped_column(Float)
    ending_equity: Mapped[float] = mapped_column(Float)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    total_fees: Mapped[float] = mapped_column(Float, default=0.0)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    win_count: Mapped[int] = mapped_column(Integer, default=0)
    loss_count: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return (
            f"<DailyPnL(strategy={self.strategy_id!r}, date={self.date!r}, "
            f"pnl={self.realized_pnl})>"
        )


class EquityCurve(Base):
    """Point-in-time equity and drawdown snapshot."""

    __tablename__ = "equity_curve"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp_ms: Mapped[int] = mapped_column(Integer, index=True, comment="Epoch milliseconds")
    equity: Mapped[float] = mapped_column(Float)
    drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)

    def __repr__(self) -> str:
        return (
            f"<EquityCurve(strategy={self.strategy_id!r}, "
            f"equity={self.equity}, dd={self.drawdown_pct}%)>"
        )


class DrawdownTracker(Base):
    """Tracks rolling drawdown metrics across multiple time windows."""

    __tablename__ = "drawdown_tracker"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    peak_equity: Mapped[float] = mapped_column(Float)
    current_equity: Mapped[float] = mapped_column(Float)
    drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    daily_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    weekly_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    monthly_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    daily_start_equity: Mapped[float] = mapped_column(Float)
    weekly_start_equity: Mapped[float] = mapped_column(Float)
    monthly_start_equity: Mapped[float] = mapped_column(Float)
    last_reset_daily: Mapped[str] = mapped_column(String(10), comment="YYYY-MM-DD")
    last_reset_weekly: Mapped[str] = mapped_column(String(10), comment="YYYY-MM-DD")
    last_reset_monthly: Mapped[str] = mapped_column(String(10), comment="YYYY-MM-DD")

    def __repr__(self) -> str:
        return (
            f"<DrawdownTracker(strategy={self.strategy_id!r}, "
            f"peak={self.peak_equity}, dd={self.drawdown_pct}%)>"
        )
