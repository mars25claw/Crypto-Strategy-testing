"""News/Event filter for STRAT-006 Market Making.

Implements Section 7.6: withdraw all quotes 1 hour before known high-impact
events and resume 30 minutes after the event concludes.

Configurable event calendar supports:
- FOMC rate decisions
- CPI / PPI / NFP releases
- Major crypto-specific events (upgrades, halvings, etc.)
- Custom user-defined events

Usage:
    filter = NewsEventFilter(params)
    filter.add_event(...)
    should_withdraw, event_name, minutes_to = filter.should_withdraw_for_event()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ScheduledEvent:
    """A known upcoming event that affects market making."""
    name: str                        # e.g. "FOMC Rate Decision"
    category: str                    # "macro", "crypto", "custom"
    timestamp: float                 # UTC epoch when event occurs
    withdraw_before_min: float = 60  # Withdraw quotes N minutes before event
    resume_after_min: float = 30     # Resume N minutes after event
    instruments: List[str] = field(default_factory=list)  # Empty = all instruments
    active: bool = True

    @property
    def withdraw_start(self) -> float:
        """Timestamp when quote withdrawal should begin."""
        return self.timestamp - (self.withdraw_before_min * 60)

    @property
    def resume_at(self) -> float:
        """Timestamp when quoting can resume."""
        return self.timestamp + (self.resume_after_min * 60)


# ---------------------------------------------------------------------------
# Default event templates (recurring monthly/quarterly patterns)
# ---------------------------------------------------------------------------

# These are category templates; actual dates must be populated via
# add_event() or load_calendar().
DEFAULT_EVENT_CATEGORIES = {
    "FOMC": {
        "category": "macro",
        "withdraw_before_min": 60,
        "resume_after_min": 30,
        "description": "Federal Open Market Committee rate decision",
    },
    "CPI": {
        "category": "macro",
        "withdraw_before_min": 60,
        "resume_after_min": 30,
        "description": "Consumer Price Index release",
    },
    "PPI": {
        "category": "macro",
        "withdraw_before_min": 60,
        "resume_after_min": 30,
        "description": "Producer Price Index release",
    },
    "NFP": {
        "category": "macro",
        "withdraw_before_min": 60,
        "resume_after_min": 30,
        "description": "Non-Farm Payrolls release",
    },
    "GDP": {
        "category": "macro",
        "withdraw_before_min": 60,
        "resume_after_min": 30,
        "description": "GDP report release",
    },
    "ETH_UPGRADE": {
        "category": "crypto",
        "withdraw_before_min": 120,
        "resume_after_min": 60,
        "description": "Ethereum network upgrade",
    },
    "BTC_HALVING": {
        "category": "crypto",
        "withdraw_before_min": 120,
        "resume_after_min": 60,
        "description": "Bitcoin block reward halving",
    },
    "EXCHANGE_MAINTENANCE": {
        "category": "crypto",
        "withdraw_before_min": 30,
        "resume_after_min": 15,
        "description": "Exchange scheduled maintenance",
    },
}


class NewsEventFilter:
    """Filters market making activity around scheduled events.

    Parameters
    ----------
    params : dict
        Strategy parameters from config.yaml.
        Relevant keys:
        - event_withdraw_before_min: default minutes to withdraw before (60)
        - event_resume_after_min: default minutes to resume after (30)
        - event_calendar: list of event dicts to preload
    """

    def __init__(self, params: dict) -> None:
        self._params = params
        self._default_withdraw_min: float = params.get("event_withdraw_before_min", 60)
        self._default_resume_min: float = params.get("event_resume_after_min", 30)

        # Event storage
        self._events: List[ScheduledEvent] = []

        # Load events from config if present
        calendar = params.get("event_calendar", [])
        for evt in calendar:
            self._load_event_from_config(evt)

        logger.info(
            "NewsEventFilter initialized: %d events loaded, "
            "default withdraw=%dm, resume=%dm",
            len(self._events), self._default_withdraw_min,
            self._default_resume_min,
        )

    # ------------------------------------------------------------------
    # Event management
    # ------------------------------------------------------------------

    def add_event(
        self,
        name: str,
        timestamp: float,
        category: str = "custom",
        withdraw_before_min: Optional[float] = None,
        resume_after_min: Optional[float] = None,
        instruments: Optional[List[str]] = None,
    ) -> None:
        """Add a scheduled event to the calendar.

        Parameters
        ----------
        name : str
            Human-readable event name (e.g. "FOMC Rate Decision Jan 2026").
        timestamp : float
            UTC epoch timestamp when the event occurs.
        category : str
            Event category ("macro", "crypto", "custom").
        withdraw_before_min : float, optional
            Minutes to withdraw before event. Defaults to config value.
        resume_after_min : float, optional
            Minutes to wait after event before resuming. Defaults to config value.
        instruments : list of str, optional
            Specific instruments affected. Empty/None = all instruments.
        """
        # Look up defaults from category template
        template = DEFAULT_EVENT_CATEGORIES.get(name.split()[0], {})
        if withdraw_before_min is None:
            withdraw_before_min = template.get(
                "withdraw_before_min", self._default_withdraw_min
            )
        if resume_after_min is None:
            resume_after_min = template.get(
                "resume_after_min", self._default_resume_min
            )

        event = ScheduledEvent(
            name=name,
            category=category,
            timestamp=timestamp,
            withdraw_before_min=withdraw_before_min,
            resume_after_min=resume_after_min,
            instruments=instruments or [],
        )
        self._events.append(event)
        logger.info(
            "Event added: %s at %s (withdraw %dm before, resume %dm after)",
            name,
            datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            withdraw_before_min, resume_after_min,
        )

    def add_event_datetime(
        self,
        name: str,
        dt: datetime,
        category: str = "custom",
        withdraw_before_min: Optional[float] = None,
        resume_after_min: Optional[float] = None,
        instruments: Optional[List[str]] = None,
    ) -> None:
        """Add event using a datetime object (convenience wrapper)."""
        self.add_event(
            name=name,
            timestamp=dt.timestamp(),
            category=category,
            withdraw_before_min=withdraw_before_min,
            resume_after_min=resume_after_min,
            instruments=instruments,
        )

    def remove_past_events(self) -> int:
        """Remove events whose resume window has passed. Returns count removed."""
        now = time.time()
        before = len(self._events)
        self._events = [e for e in self._events if e.resume_at > now]
        removed = before - len(self._events)
        if removed > 0:
            logger.info("Removed %d past events", removed)
        return removed

    def get_upcoming_events(self, hours_ahead: float = 24) -> List[ScheduledEvent]:
        """Return events occurring within the next N hours."""
        now = time.time()
        cutoff = now + hours_ahead * 3600
        return [
            e for e in self._events
            if e.active and now < e.resume_at and e.timestamp < cutoff
        ]

    # ------------------------------------------------------------------
    # Core filter method
    # ------------------------------------------------------------------

    def should_withdraw_for_event(
        self, symbol: Optional[str] = None
    ) -> Tuple[bool, str, float]:
        """Check if quotes should be withdrawn due to an upcoming event.

        Parameters
        ----------
        symbol : str, optional
            If provided, only check events affecting this instrument.

        Returns
        -------
        (should_withdraw, event_name, minutes_to_event):
            should_withdraw: True if quotes should be withdrawn now.
            event_name: Name of the triggering event (empty if no withdrawal).
            minutes_to_event: Minutes until the event (negative if event has
                passed but still in resume window).
        """
        now = time.time()

        for event in self._events:
            if not event.active:
                continue

            # Check if this event applies to the given instrument
            if symbol and event.instruments and symbol not in event.instruments:
                continue

            # Check if we're in the withdrawal window
            # Withdrawal window: [event_time - withdraw_before, event_time + resume_after]
            if event.withdraw_start <= now <= event.resume_at:
                minutes_to_event = (event.timestamp - now) / 60.0
                return True, event.name, minutes_to_event

        return False, "", 0.0

    def get_next_event(self, symbol: Optional[str] = None) -> Optional[ScheduledEvent]:
        """Return the next upcoming event, optionally filtered by symbol."""
        now = time.time()
        upcoming = []
        for e in self._events:
            if not e.active:
                continue
            if symbol and e.instruments and symbol not in e.instruments:
                continue
            if e.timestamp > now:
                upcoming.append(e)

        if not upcoming:
            return None
        return min(upcoming, key=lambda e: e.timestamp)

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_event_from_config(self, evt_data: dict) -> None:
        """Load a single event from config dict."""
        name = evt_data.get("name", "Unknown Event")
        ts_str = evt_data.get("timestamp", "")
        ts_epoch = evt_data.get("timestamp_epoch", 0)

        if ts_epoch:
            timestamp = float(ts_epoch)
        elif ts_str:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                timestamp = dt.timestamp()
            except (ValueError, TypeError):
                logger.warning("Cannot parse event timestamp: %s", ts_str)
                return
        else:
            logger.warning("Event %s has no timestamp — skipping", name)
            return

        self.add_event(
            name=name,
            timestamp=timestamp,
            category=evt_data.get("category", "custom"),
            withdraw_before_min=evt_data.get("withdraw_before_min"),
            resume_after_min=evt_data.get("resume_after_min"),
            instruments=evt_data.get("instruments"),
        )

    def load_calendar(self, events: List[dict]) -> int:
        """Load multiple events from a list of dicts.

        Each dict should have: name, timestamp (ISO) or timestamp_epoch,
        and optionally: category, withdraw_before_min, resume_after_min,
        instruments.

        Returns count of events loaded.
        """
        loaded = 0
        for evt in events:
            try:
                self._load_event_from_config(evt)
                loaded += 1
            except Exception as e:
                logger.warning("Failed to load event: %s", e)
        logger.info("Loaded %d events from calendar", loaded)
        return loaded

    # ------------------------------------------------------------------
    # Metrics / state
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return event filter metrics."""
        now = time.time()
        active_events = [e for e in self._events if e.active and e.resume_at > now]
        withdrawing = []
        for e in active_events:
            if e.withdraw_start <= now <= e.resume_at:
                withdrawing.append(e.name)

        return {
            "total_events": len(self._events),
            "active_events": len(active_events),
            "currently_withdrawing": withdrawing,
            "next_event": None if not active_events else {
                "name": min(active_events, key=lambda e: e.timestamp).name,
                "timestamp": min(active_events, key=lambda e: e.timestamp).timestamp,
                "minutes_until": (min(active_events, key=lambda e: e.timestamp).timestamp - now) / 60.0,
            },
        }

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return state for persistence."""
        return {
            "events": [
                {
                    "name": e.name,
                    "category": e.category,
                    "timestamp_epoch": e.timestamp,
                    "withdraw_before_min": e.withdraw_before_min,
                    "resume_after_min": e.resume_after_min,
                    "instruments": e.instruments,
                    "active": e.active,
                }
                for e in self._events
                if e.resume_at > time.time()  # Only persist future events
            ],
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        if not state:
            return
        events = state.get("events", [])
        for evt in events:
            self._load_event_from_config(evt)
        logger.info("Event filter state restored: %d events", len(self._events))
