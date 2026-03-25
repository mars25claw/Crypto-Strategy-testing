"""State snapshot ORM model for persisting strategy state."""

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class StateSnapshot(Base):
    """Point-in-time snapshot of strategy state (positions, orders, indicators, or full)."""

    __tablename__ = "state_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp_ms: Mapped[int] = mapped_column(Integer, index=True, comment="Epoch milliseconds")
    state_json: Mapped[str] = mapped_column(Text, comment="Serialised JSON blob")
    snapshot_type: Mapped[str] = mapped_column(
        String(20),
        index=True,
        comment="positions, orders, indicators, full",
    )

    def __repr__(self) -> str:
        return (
            f"<StateSnapshot(strategy={self.strategy_id!r}, "
            f"type={self.snapshot_type!r}, ts={self.timestamp_ms})>"
        )
