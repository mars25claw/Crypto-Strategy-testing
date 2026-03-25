"""STRAT-006 Market Making Bot — Avellaneda-Stoikov based market maker."""

__version__ = "1.1.0"

from src.strategy import AvellanedaStoikovStrategy
from src.quote_manager import QuoteManager
from src.adverse_selection import AdverseSelectionTracker
from src.risk_manager import MarketMakingRiskManager
from src.dashboard import MarketMakingDashboard
from src.news_event_filter import NewsEventFilter
from src.strategy_metrics import StrategyMetrics
