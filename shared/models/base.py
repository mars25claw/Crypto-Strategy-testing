"""SQLAlchemy base configuration with SQLite support."""

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def get_engine(db_url: str = "sqlite:///trading.db") -> Engine:
    """Create and return a SQLAlchemy engine.

    Args:
        db_url: Database URL. Defaults to a local SQLite file.
                Examples:
                  - "sqlite:///trading.db"        (relative path)
                  - "sqlite:////tmp/trading.db"   (absolute path)
                  - "sqlite://"                   (in-memory)

    Returns:
        A configured SQLAlchemy Engine.
    """
    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(
        db_url,
        connect_args=connect_args,
        echo=False,
        pool_pre_ping=True,
    )
    return engine


def get_session(engine: Engine) -> Session:
    """Create and return a new Session bound to the given engine.

    Args:
        engine: A SQLAlchemy Engine instance.

    Returns:
        A new Session instance.
    """
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return factory()


def init_db(engine: Engine) -> None:
    """Create all tables defined by Base metadata.

    Args:
        engine: A SQLAlchemy Engine instance.
    """
    Base.metadata.create_all(bind=engine)
