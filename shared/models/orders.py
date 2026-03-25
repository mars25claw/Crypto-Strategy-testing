"""Order, Fill, Trade, and Position ORM models."""

import time
from typing import Optional

from sqlalchemy import Boolean, Float, Integer, String, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Order(Base):
    """Represents a single order sent to the exchange."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, comment="Exchange order ID")
    strategy_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(4), comment="BUY or SELL")
    order_type: Mapped[str] = mapped_column(
        String(32),
        comment="MARKET, LIMIT, STOP_MARKET, TAKE_PROFIT_MARKET, TRAILING_STOP_MARKET",
    )
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    stop_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        default="NEW",
        index=True,
        comment="NEW, FILLED, PARTIALLY_FILLED, CANCELED, EXPIRED, REJECTED",
    )
    purpose: Mapped[str] = mapped_column(
        String(32),
        comment="entry, stop_loss, take_profit, trailing_stop, scale_in, grid_buy, grid_sell, hedge, exit",
    )
    created_at: Mapped[int] = mapped_column(
        Integer,
        default=lambda: int(time.time() * 1000),
        comment="Epoch milliseconds",
    )
    updated_at: Mapped[int] = mapped_column(
        Integer,
        default=lambda: int(time.time() * 1000),
        onupdate=lambda: int(time.time() * 1000),
        comment="Epoch milliseconds",
    )
    fill_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fill_quantity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    slippage_bps: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Slippage in basis points")
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="Round-trip latency ms")
    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Relationship to fills
    fills: Mapped[list["Fill"]] = relationship("Fill", back_populates="order", lazy="select")

    def __repr__(self) -> str:
        return (
            f"<Order(id={self.id}, order_id={self.order_id!r}, "
            f"symbol={self.symbol!r}, side={self.side!r}, status={self.status!r})>"
        )


class Fill(Base):
    """Individual fill (execution) for an order."""

    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("orders.order_id"), index=True
    )
    trade_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(4))
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    timestamp_ms: Mapped[int] = mapped_column(Integer, comment="Epoch milliseconds")
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    fee_asset: Mapped[str] = mapped_column(String(16), default="USDT")
    maker: Mapped[bool] = mapped_column(Boolean, default=False)

    order: Mapped["Order"] = relationship("Order", back_populates="fills", lazy="select")

    def __repr__(self) -> str:
        return (
            f"<Fill(id={self.id}, order_id={self.order_id!r}, "
            f"price={self.price}, qty={self.quantity})>"
        )


class Trade(Base):
    """A complete trade (entry to exit) or an open trade in progress."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    strategy_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(5), comment="LONG or SHORT")
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    entry_time_ms: Mapped[int] = mapped_column(Integer)
    exit_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    size: Mapped[float] = mapped_column(Float)
    realized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    total_fees: Mapped[float] = mapped_column(Float, default=0.0)
    total_slippage_bps: Mapped[float] = mapped_column(Float, default=0.0)
    exit_reason: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        comment="stop_loss, trailing_stop, take_profit, signal, time, regime_change, circuit_breaker, kill_switch, manual",
    )
    r_multiple: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    entry_indicator_values: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="JSON string of indicator values at entry"
    )
    scale_in_count: Mapped[int] = mapped_column(Integer, default=0)
    tranche_exits: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="JSON string of tranche exit records"
    )

    def __repr__(self) -> str:
        return (
            f"<Trade(id={self.id}, trade_id={self.trade_id!r}, "
            f"symbol={self.symbol!r}, direction={self.direction!r}, pnl={self.realized_pnl})>"
        )


class Position(Base):
    """Current open position state for a strategy/symbol pair."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(5), comment="LONG or SHORT")
    size: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    avg_entry_price: Mapped[float] = mapped_column(Float)
    entry_time_ms: Mapped[int] = mapped_column(Integer)
    current_stop_level: Mapped[float] = mapped_column(Float)
    stop_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    take_profit_levels: Mapped[str] = mapped_column(
        Text, default="[]", comment="JSON array of take-profit levels"
    )
    trailing_stop_state: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="JSON: {active, trail_distance, highest_price, current_stop}",
    )
    scale_in_history: Mapped[str] = mapped_column(
        Text, default="[]", comment="JSON array of scale-in records"
    )
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    leverage: Mapped[int] = mapped_column(Integer, default=1)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    def __repr__(self) -> str:
        return (
            f"<Position(id={self.id}, strategy={self.strategy_id!r}, "
            f"symbol={self.symbol!r}, dir={self.direction!r}, size={self.size}, open={self.is_open})>"
        )
