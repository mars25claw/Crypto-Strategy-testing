"""
Crypto Strategy Lab - Discord Dashboard Bot

Full multi-channel dashboard with per-strategy embeds, trade feed,
alert log, leaderboard, and portfolio overview. Connects to the
aggregator WebSocket for real-time updates.

Channel structure under "STRATEGY LAB" category:
  #overview       — Portfolio summary (edited every 2s)
  #leaderboard    — Ranked strategies (edited every 2s)
  #alerts         — Alert log (new messages, never edited)
  #trade-feed     — Trade executions (new messages, never edited)
  #s001-trend … #s010-ml-onchain — Per-strategy embeds (staggered ~5s each)
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import discord
from discord import app_commands
from discord.ext import tasks

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("discord_bot")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DISCORD_BOT_TOKEN: str = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_GUILD_ID: str = os.environ.get("DISCORD_GUILD_ID", "")

AGGREGATOR_WS_URL: str = os.environ.get("AGGREGATOR_WS_URL", "ws://aggregator:8099/ws")
AGGREGATOR_HTTP_URL: str = os.environ.get("AGGREGATOR_HTTP_URL", "http://aggregator:8099")

STARTING_EQUITY_PER_STRATEGY = 800.0
TOTAL_STARTING_EQUITY = STARTING_EQUITY_PER_STRATEGY * 10

# Persistent state file for message/channel IDs across restarts
STATE_FILE = Path(os.environ.get("DASHBOARD_STATE_FILE", "/data/dashboard_state.json"))

# Rate limits — Discord allows ~50 requests/s globally, but edits on one
# channel are limited to ~5/5s.  We stay well under with this scheme:
#   overview + leaderboard: every 2s each = 60 edits/min
#   10 strategy channels staggered: 1 per second rotation = 60 edits/min
# Total: ~120 edits/min = 2/s — well within limits.
OVERVIEW_INTERVAL = 2.0
LEADERBOARD_INTERVAL = 2.0
STRATEGY_ROTATION_INTERVAL = 1.0  # update one strategy channel every 1s

# Embed colours
COLOR_GREEN = 0x00C851
COLOR_YELLOW = 0xFFBB33
COLOR_RED = 0xFF4444
COLOR_GREY = 0x2F3136

# Category name
CATEGORY_NAME = "STRATEGY LAB"

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
STRATEGY_MAP: Dict[str, Dict[str, Any]] = {
    "STRAT-001": {"host": "strat-001-trend-following", "port": 8081, "name": "Trend Following", "channel": "s001-trend"},
    "STRAT-002": {"host": "strat-002-funding-arb", "port": 8082, "name": "Funding Arb", "channel": "s002-funding-arb"},
    "STRAT-003": {"host": "strat-003-stat-arb-pairs", "port": 8083, "name": "Stat Arb Pairs", "channel": "s003-stat-arb"},
    "STRAT-004": {"host": "strat-004-mean-reversion", "port": 8084, "name": "Mean Reversion", "channel": "s004-mean-rev"},
    "STRAT-005": {"host": "strat-005-grid-trading", "port": 8085, "name": "Grid Trading", "channel": "s005-grid"},
    "STRAT-006": {"host": "strat-006-market-making", "port": 8086, "name": "Market Making", "channel": "s006-mm"},
    "STRAT-007": {"host": "strat-007-triangular-arb", "port": 8087, "name": "Triangular Arb", "channel": "s007-tri-arb"},
    "STRAT-008": {"host": "strat-008-options-vol", "port": 8088, "name": "Options Vol", "channel": "s008-options"},
    "STRAT-009": {"host": "strat-009-signal-dca", "port": 8089, "name": "Signal DCA", "channel": "s009-signal-dca"},
    "STRAT-010": {"host": "strat-010-ml-onchain", "port": 8090, "name": "ML On-Chain", "channel": "s010-ml-onchain"},
}

VALID_STRATEGY_IDS: List[str] = sorted(STRATEGY_MAP.keys())

# Ordered list for rotation — matches the channel creation order
STRATEGY_ORDER: List[str] = VALID_STRATEGY_IDS

# Channel name -> purpose mapping for all fixed channels
FIXED_CHANNELS = ["overview", "leaderboard", "alerts", "trade-feed"]


# ---------------------------------------------------------------------------
# Sparkline generator
# ---------------------------------------------------------------------------

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: List[float], width: int = 12) -> str:
    """Generate an ASCII sparkline from a list of numeric values.

    Returns a string of block characters representing the trend over
    the last *width* data points.
    """
    if not values:
        return "▁" * width

    # Take the most recent `width` points
    recent = values[-width:]

    lo = min(recent)
    hi = max(recent)
    span = hi - lo

    if span == 0:
        return SPARK_CHARS[4] * len(recent)

    chars: List[str] = []
    for v in recent:
        idx = int((v - lo) / span * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])

    # Pad to width if we have fewer points
    if len(chars) < width:
        chars = ["▁"] * (width - len(chars)) + chars

    return "".join(chars)


# ---------------------------------------------------------------------------
# Embed colour helper
# ---------------------------------------------------------------------------

def pnl_color(pnl_pct: float, status: str = "running") -> int:
    """Pick embed colour based on P&L percentage and status."""
    if status in ("offline", "crashed"):
        return COLOR_GREY
    if pnl_pct > 0:
        return COLOR_GREEN
    if pnl_pct >= -0.5:
        return COLOR_YELLOW
    return COLOR_RED


def status_emoji(status: str) -> str:
    """Emoji for strategy live-status indicator."""
    if status in ("running", "online"):
        return "\U0001F7E2 LIVE"     # green circle
    if status == "crashed":
        return "\U0001F534 CRASHED"  # red circle
    return "\u26AB OFFLINE"          # black circle


def rank_emoji(pnl_pct: float, status: str, trade_count: int) -> str:
    """Row emoji for the leaderboard."""
    if status in ("offline", "crashed"):
        return "\U0001F480"  # skull
    if trade_count == 0:
        return "\u26AA"      # white circle (warming)
    if pnl_pct > 0.5:
        return "\U0001F7E2"  # green
    if pnl_pct < -0.5:
        return "\U0001F534"  # red
    return "\U0001F7E1"      # yellow


# ---------------------------------------------------------------------------
# Dashboard state persistence
# ---------------------------------------------------------------------------

def load_dashboard_state() -> Dict[str, Any]:
    """Load saved channel/message IDs from disk."""
    try:
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            logger.info("Loaded dashboard state from %s", STATE_FILE)
            return data
    except Exception as exc:
        logger.warning("Could not load dashboard state: %s", exc)
    return {}


def save_dashboard_state(state: Dict[str, Any]) -> None:
    """Persist channel/message IDs to disk."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as exc:
        logger.warning("Could not save dashboard state: %s", exc)


# ---------------------------------------------------------------------------
# Bot Client
# ---------------------------------------------------------------------------

class StrategyLabBot(discord.Client):
    """Full multi-channel dashboard bot for the Crypto Strategy Lab."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.guilds = True
        intents.guild_messages = True
        intents.message_content = True
        super().__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)

        # Discord objects
        self.target_guild: Optional[discord.Guild] = None
        self.category: Optional[discord.CategoryChannel] = None

        # Channel references keyed by channel purpose name
        # e.g. "overview", "leaderboard", "alerts", "trade-feed", "STRAT-001"
        self.channels: Dict[str, discord.TextChannel] = {}

        # Pinned/edited message objects keyed by same keys as channels
        # (only for channels that use a single edited embed)
        self.dashboard_messages: Dict[str, discord.Message] = {}

        # HTTP session for calling strategy APIs
        self.http_session: Optional[aiohttp.ClientSession] = None

        # Background tasks
        self.ws_task: Optional[asyncio.Task] = None
        self.overview_task: Optional[asyncio.Task] = None
        self.leaderboard_task: Optional[asyncio.Task] = None
        self.strategy_rotation_task: Optional[asyncio.Task] = None
        self.shutting_down: bool = False

        # Latest data from aggregator WS — list of strategy metric dicts
        self.latest_strategies: List[Dict[str, Any]] = []

        # Equity history per strategy for sparklines (ring buffer of last 720 points = ~12h at 1/min)
        self.equity_history: Dict[str, List[float]] = {sid: [] for sid in STRATEGY_ORDER}
        self._last_equity_record: float = 0.0

        # Trade deduplication — track last known trade_count per strategy
        self.last_trade_counts: Dict[str, int] = {sid: -1 for sid in STRATEGY_ORDER}

        # Alert tracking
        self.active_alerts: Set[str] = set()
        self.sent_alert_keys: Set[str] = set()

        # Cached detailed data per strategy (positions, trades) — refreshed on rotation
        self.strategy_detail_cache: Dict[str, Dict[str, Any]] = {}

        # Persisted state
        self._persisted_state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup_hook(self) -> None:
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        self._persisted_state = load_dashboard_state()
        logger.info("HTTP session created. Dashboard state loaded.")

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (id=%s)", self.user, self.user.id)

        self.target_guild = await self._resolve_guild()
        if self.target_guild is None:
            logger.error("Could not resolve guild. Shutting down.")
            await self.close()
            return

        # Sync slash commands
        self.tree.copy_global_to(guild=self.target_guild)
        await self.tree.sync(guild=self.target_guild)
        logger.info("Slash commands synced to guild %s.", self.target_guild.name)

        # Create category + all channels
        await self._setup_channels()

        # Resolve or create pinned embeds in each channel
        await self._setup_messages()

        # Persist state after setup
        self._save_state()

        # Start background loops
        self.ws_task = asyncio.create_task(self._ws_loop(), name="ws_loop")
        self.overview_task = asyncio.create_task(self._overview_loop(), name="overview_loop")
        self.leaderboard_task = asyncio.create_task(self._leaderboard_loop(), name="leaderboard_loop")
        self.strategy_rotation_task = asyncio.create_task(
            self._strategy_rotation_loop(), name="strategy_rotation_loop"
        )

        logger.info("Bot fully started — all channels and embeds ready.")

    async def close(self) -> None:
        self.shutting_down = True
        logger.info("Shutting down...")

        for t in (self.ws_task, self.overview_task, self.leaderboard_task, self.strategy_rotation_task):
            if t and not t.done():
                t.cancel()

        if self.http_session and not self.http_session.closed:
            await self.http_session.close()

        self._save_state()
        await super().close()
        logger.info("Shutdown complete.")

    # ------------------------------------------------------------------
    # Guild resolution
    # ------------------------------------------------------------------

    async def _resolve_guild(self) -> Optional[discord.Guild]:
        if DISCORD_GUILD_ID:
            try:
                guild = self.get_guild(int(DISCORD_GUILD_ID))
                if guild:
                    return guild
            except (ValueError, TypeError):
                pass
        if self.guilds:
            return self.guilds[0]
        return None

    # ------------------------------------------------------------------
    # Channel / category setup
    # ------------------------------------------------------------------

    async def _setup_channels(self) -> None:
        """Create the STRATEGY LAB category and all 14 channels if they don't exist."""
        guild = self.target_guild
        if guild is None:
            return

        # Find or create category
        self.category = None
        for cat in guild.categories:
            if cat.name == CATEGORY_NAME:
                self.category = cat
                break

        if self.category is None:
            try:
                self.category = await guild.create_category(
                    CATEGORY_NAME,
                    reason="Strategy Lab dashboard auto-setup",
                )
                logger.info("Created category: %s", CATEGORY_NAME)
            except discord.HTTPException as exc:
                logger.error("Failed to create category: %s", exc)
                return

        # Build list of all desired channel names and their purpose keys
        desired_channels: List[Tuple[str, str]] = []
        for name in FIXED_CHANNELS:
            desired_channels.append((name, name))
        for sid in STRATEGY_ORDER:
            ch_name = STRATEGY_MAP[sid]["channel"]
            desired_channels.append((ch_name, sid))

        # Index existing channels in the category
        existing = {ch.name: ch for ch in self.category.text_channels}

        # Also check persisted state for channel IDs
        persisted_channels = self._persisted_state.get("channels", {})

        for ch_name, purpose_key in desired_channels:
            ch = existing.get(ch_name)

            # Try persisted ID if not found by name
            if ch is None and purpose_key in persisted_channels:
                try:
                    ch_id = int(persisted_channels[purpose_key])
                    ch = guild.get_channel(ch_id)
                    if ch is not None and not isinstance(ch, discord.TextChannel):
                        ch = None
                except (ValueError, TypeError):
                    ch = None

            if ch is None:
                try:
                    topic = self._channel_topic(purpose_key)
                    ch = await guild.create_text_channel(
                        ch_name,
                        category=self.category,
                        topic=topic,
                        reason="Strategy Lab dashboard auto-setup",
                    )
                    logger.info("Created channel: #%s", ch_name)
                except discord.HTTPException as exc:
                    logger.error("Failed to create channel #%s: %s", ch_name, exc)
                    continue

            self.channels[purpose_key] = ch

    @staticmethod
    def _channel_topic(key: str) -> str:
        topics = {
            "overview": "Portfolio overview — live updated",
            "leaderboard": "Strategy leaderboard — ranked by P&L",
            "alerts": "Alert feed — scrollable log",
            "trade-feed": "Trade executions across all strategies",
        }
        if key in topics:
            return topics[key]
        if key in STRATEGY_MAP:
            return f"{key} — {STRATEGY_MAP[key]['name']} live dashboard"
        return "Strategy Lab"

    # ------------------------------------------------------------------
    # Message setup — find or create pinned embeds
    # ------------------------------------------------------------------

    async def _setup_messages(self) -> None:
        """For channels that use a single edited embed (overview, leaderboard,
        and each strategy channel), find the existing pinned message or create one."""

        # Channels that need a pinned embed
        embed_channels = ["overview", "leaderboard"] + STRATEGY_ORDER

        persisted_msgs = self._persisted_state.get("messages", {})

        for key in embed_channels:
            ch = self.channels.get(key)
            if ch is None:
                continue

            msg = None

            # Try persisted message ID first
            if key in persisted_msgs:
                try:
                    msg_id = int(persisted_msgs[key])
                    msg = await ch.fetch_message(msg_id)
                except (ValueError, TypeError, discord.NotFound, discord.HTTPException):
                    msg = None

            # Fallback: look for a pinned message from us
            if msg is None:
                try:
                    pins = await ch.pins()
                    for p in pins:
                        if p.author == self.user and p.embeds:
                            msg = p
                            break
                except discord.HTTPException:
                    pass

            # Create new embed if needed
            if msg is None:
                embed = self._build_waiting_embed(key)
                try:
                    msg = await ch.send(embed=embed)
                    await msg.pin(reason="Strategy Lab live dashboard")
                    logger.info("Created pinned embed in #%s (id=%s)", ch.name, msg.id)
                except discord.HTTPException as exc:
                    logger.error("Failed to create embed in #%s: %s", ch.name, exc)
                    continue

            self.dashboard_messages[key] = msg

    def _build_waiting_embed(self, key: str) -> discord.Embed:
        """Placeholder embed shown before data arrives."""
        if key == "overview":
            title = "\U0001F9EA CRYPTO STRATEGY LAB"
        elif key == "leaderboard":
            title = "\U0001F3C6 STRATEGY LEADERBOARD"
        elif key in STRATEGY_MAP:
            info = STRATEGY_MAP[key]
            title = f"\U0001F4C9 {key} \u2014 {info['name']}"
        else:
            title = "Strategy Lab"

        embed = discord.Embed(title=title, color=COLOR_GREY)
        embed.description = "\u2501" * 24 + "\n\u23F3 Waiting for aggregator data...\n" + "\u2501" * 24
        return embed

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        state: Dict[str, Any] = {
            "channels": {k: v.id for k, v in self.channels.items()},
            "messages": {k: v.id for k, v in self.dashboard_messages.items()},
        }
        save_dashboard_state(state)

    # ------------------------------------------------------------------
    # Embed builders
    # ------------------------------------------------------------------

    def _get_strategy_data(self, sid: str) -> Optional[Dict[str, Any]]:
        """Look up a strategy's metrics from the latest WS data."""
        for s in self.latest_strategies:
            if s.get("strategy_id") == sid:
                return s
        return None

    def _portfolio_totals(self) -> Tuple[float, float, float, int, int]:
        """Compute portfolio-wide totals from latest strategy data.

        Returns (total_equity, total_pnl, total_pnl_pct, active_count, total_count).
        """
        if not self.latest_strategies:
            return TOTAL_STARTING_EQUITY, 0.0, 0.0, 0, 10

        total_eq = sum(s.get("equity", STARTING_EQUITY_PER_STRATEGY) for s in self.latest_strategies)
        total_pnl = sum(s.get("pnl", 0.0) for s in self.latest_strategies)
        pnl_pct = ((total_eq - TOTAL_STARTING_EQUITY) / TOTAL_STARTING_EQUITY * 100.0) if TOTAL_STARTING_EQUITY else 0.0
        active = sum(1 for s in self.latest_strategies if s.get("status") in ("running", "online"))
        return total_eq, total_pnl, pnl_pct, active, len(self.latest_strategies)

    def _build_overview_embed(self) -> discord.Embed:
        """Build the #overview embed."""
        total_eq, total_pnl, pnl_pct, active, total = self._portfolio_totals()
        now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        pnl_sign = "+" if total_pnl >= 0 else ""
        pct_sign = "+" if pnl_pct >= 0 else ""

        # Best and worst today
        online = [s for s in self.latest_strategies if s.get("status") in ("running", "online")]
        best_line = "N/A"
        worst_line = "N/A"
        if online:
            best = max(online, key=lambda s: s.get("pnl_pct", 0.0))
            worst = min(online, key=lambda s: s.get("pnl_pct", 0.0))
            bp = best.get("pnl_pct", 0.0)
            wp = worst.get("pnl_pct", 0.0)
            best_line = f"{best.get('strategy_id', '???')} {'+' if bp >= 0 else ''}{bp:.2f}%"
            worst_line = f"{worst.get('strategy_id', '???')} {'+' if wp >= 0 else ''}{wp:.2f}%"

        desc = (
            "\u2501" * 24 + "\n"
            f"\U0001F4BC Total Equity:  ${total_eq:,.2f}\n"
            f"\U0001F4C8 Total P&L:     {pnl_sign}${abs(total_pnl):,.2f} ({pct_sign}{pnl_pct:.2f}%)\n"
            f"\U0001F4CA Active Strats: {active}/{total}\n"
            f"\U0001F3C6 Best Today:    {best_line}\n"
            f"\U0001F480 Worst Today:   {worst_line}\n"
            f"\U0001F550 Updated: {now_utc}\n"
            + "\u2501" * 24
        )

        color = COLOR_GREEN if total_pnl >= 0 else COLOR_RED
        embed = discord.Embed(
            title="\U0001F9EA CRYPTO STRATEGY LAB",
            description=desc,
            color=color,
        )
        return embed

    def _build_leaderboard_embed(self) -> discord.Embed:
        """Build the #leaderboard embed."""
        now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        # Sort by rank
        strategies = sorted(self.latest_strategies, key=lambda s: s.get("rank", 99))

        lines: List[str] = ["\u2501" * 24]
        for s in strategies:
            r = s.get("rank", 0)
            sid = s.get("strategy_id", "???")
            name = s.get("strategy_name", STRATEGY_MAP.get(sid, {}).get("name", "Unknown"))
            st = s.get("status", "offline")
            pnl_pct = s.get("pnl_pct", 0.0)
            trades = s.get("trade_count", 0)
            sharpe = s.get("sharpe_ratio", 0.0)
            emoji = rank_emoji(pnl_pct, st, trades)

            if st in ("offline", "crashed"):
                detail = "OFFLINE"
            elif trades == 0:
                detail = f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:>6.2f}%  {trades:>3}T  warming"
            else:
                detail = f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:>6.2f}%  {trades:>3}T  SR:{sharpe:.1f}"

            # Truncate name for alignment
            short_name = name[:18]
            lines.append(f"#{r:<2} {emoji} {sid}  {short_name:<18} {detail}")

        lines.append("\u2501" * 24)
        lines.append(f"Updated: {now_utc}")

        embed = discord.Embed(
            title="\U0001F3C6 STRATEGY LEADERBOARD",
            description="\n".join(lines),
            color=COLOR_GREY,
        )
        return embed

    def _build_strategy_embed(self, sid: str) -> discord.Embed:
        """Build a per-strategy channel embed with positions and recent trades."""
        info = STRATEGY_MAP.get(sid, {})
        sname = info.get("name", "Unknown")
        data = self._get_strategy_data(sid)

        if data is None:
            embed = discord.Embed(
                title=f"\U0001F4C9 {sid} \u2014 {sname}    \u26AB OFFLINE",
                description="\u2501" * 24 + "\n\u23F3 No data from aggregator.\n" + "\u2501" * 24,
                color=COLOR_GREY,
            )
            return embed

        st = data.get("status", "offline")
        equity = data.get("equity", STARTING_EQUITY_PER_STRATEGY)
        pnl_pct = data.get("pnl_pct", 0.0)
        trades = data.get("trade_count", 0)
        win_rate = data.get("win_rate", 0.0) * 100.0
        sharpe = data.get("sharpe_ratio", 0.0)
        max_dd = data.get("max_drawdown", 0.0) * 100.0
        now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        status_str = status_emoji(st)
        pct_sign = "+" if pnl_pct >= 0 else ""

        lines: List[str] = ["\u2501" * 24]
        lines.append(f"\U0001F4B0 Equity:    ${equity:,.2f}  ({pct_sign}{pnl_pct:.2f}%)")
        lines.append(f"\U0001F4CA Trades:    {trades}  |  Win Rate: {win_rate:.1f}%")
        lines.append(f"\U0001F4D0 Sharpe:    {sharpe:.1f}  |  Max DD: {max_dd:.1f}%")

        # Cached detail data (positions + trades)
        detail = self.strategy_detail_cache.get(sid, {})
        positions = detail.get("positions", [])
        recent_trades = detail.get("trades", [])
        pairs_set: Set[str] = set()

        # Pairs from positions
        for p in positions:
            sym = p.get("symbol", "")
            if sym:
                pairs_set.add(sym.replace("USDT", "").replace("/", ""))

        # If no positions, show pairs from recent trades
        if not pairs_set:
            for t in recent_trades:
                sym = t.get("symbol", "")
                if sym:
                    pairs_set.add(sym.replace("USDT", "").replace("/", ""))

        if pairs_set:
            lines.append(f"\U0001F4C8 Pairs:     {' '.join(sorted(pairs_set)[:8])}")

        # Open positions
        lines.append("")
        if positions:
            lines.append("\U0001F513 OPEN POSITIONS")
            for p in positions[:5]:
                sym = p.get("symbol", "???")
                direction = p.get("direction", "???")
                qty = p.get("remaining_quantity", p.get("total_quantity", 0))
                entry = p.get("avg_entry_price", p.get("entry_price", 0))
                upnl = p.get("unrealized_pnl", 0)
                pnl_s = f"+${upnl:.2f}" if upnl >= 0 else f"-${abs(upnl):.2f}"
                lines.append(f"  {sym}  {direction}  {qty}  Entry: ${entry:,.2f}  PnL: {pnl_s}")
        else:
            lines.append("\U0001F513 OPEN POSITIONS: None")

        # Last 5 trades
        lines.append("")
        if recent_trades:
            lines.append("\U0001F4CB LAST 5 TRADES")
            for t in recent_trades[:5]:
                sym = t.get("symbol", "???")
                direction = t.get("direction", t.get("side", "???"))
                rpnl = t.get("realized_pnl", t.get("pnl", 0))
                icon = "\u2705" if rpnl >= 0 else "\u274C"
                pnl_s = f"+${rpnl:.2f}" if rpnl >= 0 else f"-${abs(rpnl):.2f}"
                ts = t.get("close_time", t.get("timestamp", ""))
                time_str = _format_trade_time(ts)
                lines.append(f"  {icon} {sym}  {direction}  {pnl_s}  ({time_str})")
        else:
            lines.append("\U0001F4CB LAST 5 TRADES: None yet")

        # Sparkline
        eq_hist = self.equity_history.get(sid, [])
        if len(eq_hist) >= 2:
            spark = sparkline(eq_hist, 12)
            lines.append(f"\n{spark}  Equity Sparkline (12h)")

        lines.append("\u2501" * 24)
        lines.append(f"Updated: {now_utc}")

        embed = discord.Embed(
            title=f"\U0001F4C9 {sid} \u2014 {sname}    {status_str}",
            description="\n".join(lines),
            color=pnl_color(pnl_pct, st),
        )
        return embed

    # ------------------------------------------------------------------
    # Trade feed & alert message builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trade_message(sid: str, trade: Dict[str, Any]) -> str:
        """Format a single trade for the #trade-feed channel."""
        sym = trade.get("symbol", "???")
        direction = trade.get("direction", trade.get("side", "???"))
        rpnl = trade.get("realized_pnl", trade.get("pnl", 0))
        entry = trade.get("entry_price", trade.get("avg_entry_price", 0))
        exit_p = trade.get("exit_price", trade.get("close_price", 0))
        qty = trade.get("total_quantity", trade.get("quantity", 0))
        fees = trade.get("total_fees", trade.get("fees", 0))
        pnl_pct = trade.get("pnl_pct", 0)
        now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        icon = "\u2705" if rpnl >= 0 else "\u274C"
        pnl_sign = "+" if rpnl >= 0 else ""

        return (
            f"\U0001F4CA **{sid}** | {sym} {direction} CLOSED\n"
            f"{icon} P&L: {pnl_sign}${rpnl:.2f} ({pnl_sign}{pnl_pct:.3f}%)\n"
            f"Entry: ${entry:,.2f} \u2192 Exit: ${exit_p:,.2f}\n"
            f"Size: {qty} | Fees: ${fees:.2f}\n"
            f"{now_utc}"
        )

    @staticmethod
    def _build_alert_message(text: str) -> str:
        now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        return f"\U0001F6A8 **ALERT** \u2014 {text}\n{now_utc}"

    # ------------------------------------------------------------------
    # WebSocket listener
    # ------------------------------------------------------------------

    async def _ws_loop(self) -> None:
        """Connect to the aggregator WS and receive strategy metrics."""
        import websockets
        import websockets.exceptions

        backoff = 1.0
        max_backoff = 30.0

        while not self.shutting_down:
            try:
                logger.info("Connecting to aggregator WS at %s ...", AGGREGATOR_WS_URL)
                async with websockets.connect(AGGREGATOR_WS_URL) as ws:
                    logger.info("WebSocket connected.")
                    backoff = 1.0

                    async for raw in ws:
                        if self.shutting_down:
                            break
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        # data is a list of strategy metric dicts
                        if not isinstance(data, list):
                            continue

                        self.latest_strategies = data

                        # Record equity for sparklines (at most once per minute)
                        now = time.monotonic()
                        if now - self._last_equity_record >= 60.0:
                            self._record_equity()
                            self._last_equity_record = now

                        # Detect new trades and alerts
                        await self._detect_new_trades(data)
                        await self._check_alerts(data)

            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self.shutting_down:
                    return
                logger.warning("WebSocket error: %s. Reconnecting in %.1fs...", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    def _record_equity(self) -> None:
        """Snapshot current equity for sparklines."""
        for s in self.latest_strategies:
            sid = s.get("strategy_id", "")
            eq = s.get("equity", STARTING_EQUITY_PER_STRATEGY)
            if sid in self.equity_history:
                hist = self.equity_history[sid]
                hist.append(eq)
                # Keep last 720 points (~12h at 1/min)
                if len(hist) > 720:
                    self.equity_history[sid] = hist[-720:]

    # ------------------------------------------------------------------
    # New trade detection
    # ------------------------------------------------------------------

    async def _detect_new_trades(self, strategies: List[Dict[str, Any]]) -> None:
        """Compare trade counts to detect new trades, then post to #trade-feed."""
        trade_feed_ch = self.channels.get("trade-feed")
        if trade_feed_ch is None:
            return

        for s in strategies:
            sid = s.get("strategy_id", "")
            current_count = s.get("trade_count", 0)
            last_count = self.last_trade_counts.get(sid, -1)

            if last_count == -1:
                # First time — initialize without posting
                self.last_trade_counts[sid] = current_count
                continue

            if current_count > last_count:
                # New trades detected — fetch the most recent ones
                new_count = current_count - last_count
                self.last_trade_counts[sid] = current_count
                await self._fetch_and_post_trades(sid, trade_feed_ch, min(new_count, 5))

    async def _fetch_and_post_trades(
        self, sid: str, channel: discord.TextChannel, limit: int
    ) -> None:
        """Fetch recent trades from a strategy and post them to trade-feed."""
        info = STRATEGY_MAP.get(sid)
        if info is None:
            return

        url = f"http://{info['host']}:{info['port']}/api/trades?limit={limit}"
        data = await self._http_get(url)

        if data is None or not isinstance(data, list):
            # Fallback: post a generic trade notification
            try:
                s_data = self._get_strategy_data(sid)
                count = s_data.get("trade_count", 0) if s_data else 0
                msg = f"\U0001F4CA **{sid}** | New trade executed (total: {count})"
                await channel.send(msg)
            except discord.HTTPException:
                pass
            return

        # Post each new trade (most recent first)
        for trade in data[:limit]:
            try:
                msg = self._build_trade_message(sid, trade)
                await channel.send(msg)
            except discord.HTTPException as exc:
                logger.warning("Failed to post trade to #trade-feed: %s", exc)
                break  # Avoid spamming on rate limit

    # ------------------------------------------------------------------
    # Alert detection
    # ------------------------------------------------------------------

    async def _check_alerts(self, strategies: List[Dict[str, Any]]) -> None:
        """Check for alert conditions and post to #alerts + DM owner."""
        new_alerts: Set[str] = set()
        alert_messages: List[str] = []

        # Portfolio P&L check
        total_eq = sum(s.get("equity", STARTING_EQUITY_PER_STRATEGY) for s in strategies)
        total_pnl_pct = ((total_eq - TOTAL_STARTING_EQUITY) / TOTAL_STARTING_EQUITY * 100.0) if TOTAL_STARTING_EQUITY else 0.0

        if total_pnl_pct < -10.0:
            key = f"portfolio_pnl_critical:{total_pnl_pct:.2f}"
            new_alerts.add(key)
            alert_messages.append(f"Portfolio P&L critically low: {total_pnl_pct:.2f}%")

        for s in strategies:
            sid = s.get("strategy_id", "UNKNOWN")
            st = s.get("status", "offline")
            pnl_pct = s.get("pnl_pct", 0.0)
            misses = s.get("consecutive_misses", 0)

            if st in ("offline", "crashed"):
                key = f"offline:{sid}"
                new_alerts.add(key)
                detail = f"({misses} missed polls)" if misses >= 3 else ""
                alert_messages.append(f"{sid} Offline — Strategy container not responding {detail}")

            if pnl_pct < -5.0:
                key = f"pnl_critical:{sid}:{pnl_pct:.2f}"
                new_alerts.add(key)
                alert_messages.append(f"{sid} P&L dropped to {pnl_pct:.2f}%")

        # Determine truly new alerts
        to_send_keys = new_alerts - self.sent_alert_keys
        cleared = self.sent_alert_keys - new_alerts
        if cleared:
            self.sent_alert_keys -= cleared

        self.active_alerts = new_alerts

        if to_send_keys:
            self.sent_alert_keys |= to_send_keys

            # Post new alerts to #alerts channel
            alerts_ch = self.channels.get("alerts")
            if alerts_ch:
                for msg_text in alert_messages:
                    # Only post if this particular message's key is new
                    try:
                        await alerts_ch.send(self._build_alert_message(msg_text))
                    except discord.HTTPException as exc:
                        logger.warning("Failed to post alert: %s", exc)

            # DM guild owner
            asyncio.create_task(self._send_owner_alerts(to_send_keys))

    async def _send_owner_alerts(self, alert_keys: Set[str]) -> None:
        """DM alert messages to the guild owner."""
        if self.target_guild is None:
            return
        try:
            owner = self.target_guild.owner
            if owner is None:
                owner = await self.target_guild.fetch_member(self.target_guild.owner_id)
            if owner is None:
                return

            lines: List[str] = ["\u26A0\uFE0F **Crypto Strategy Lab Alert**\n"]
            for key in sorted(alert_keys):
                if key.startswith("portfolio_pnl_critical:"):
                    pct = key.split(":")[1]
                    lines.append(f"\U0001F6A8 Portfolio P&L critically low: **{pct}%**")
                elif key.startswith("offline:"):
                    sid = key.split(":")[1]
                    lines.append(f"\U0001F480 **{sid}** container is OFFLINE")
                elif key.startswith("pnl_critical:"):
                    parts = key.split(":")
                    sid, pct = parts[1], parts[2]
                    lines.append(f"\U0001F534 **{sid}** P&L dropped to **{pct}%**")

            await owner.send("\n".join(lines))
            logger.info("Sent %d alert(s) to guild owner.", len(alert_keys))
        except discord.Forbidden:
            logger.warning("Cannot DM guild owner (DMs disabled).")
        except discord.HTTPException as exc:
            logger.error("Failed to DM guild owner: %s", exc)
        except Exception as exc:
            logger.error("Unexpected error sending owner alerts: %s", exc)

    # ------------------------------------------------------------------
    # Rate-limited embed update loops
    # ------------------------------------------------------------------

    async def _overview_loop(self) -> None:
        """Edit #overview embed every 2 seconds."""
        while not self.shutting_down:
            try:
                if self.latest_strategies:
                    msg = self.dashboard_messages.get("overview")
                    if msg:
                        embed = self._build_overview_embed()
                        try:
                            await msg.edit(embed=embed)
                        except discord.HTTPException as exc:
                            logger.warning("Failed to edit overview: %s", exc)
                await asyncio.sleep(OVERVIEW_INTERVAL)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Overview loop error: %s", exc)
                await asyncio.sleep(5)

    async def _leaderboard_loop(self) -> None:
        """Edit #leaderboard embed every 2 seconds."""
        while not self.shutting_down:
            try:
                if self.latest_strategies:
                    msg = self.dashboard_messages.get("leaderboard")
                    if msg:
                        embed = self._build_leaderboard_embed()
                        try:
                            await msg.edit(embed=embed)
                        except discord.HTTPException as exc:
                            logger.warning("Failed to edit leaderboard: %s", exc)
                await asyncio.sleep(LEADERBOARD_INTERVAL)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Leaderboard loop error: %s", exc)
                await asyncio.sleep(5)

    async def _strategy_rotation_loop(self) -> None:
        """Update one strategy channel per second in rotation.

        Fetches detailed data (positions + trades) from the strategy API
        before building the embed, so each strategy gets fresh detail data
        every ~10 seconds.
        """
        idx = 0
        while not self.shutting_down:
            try:
                if self.latest_strategies:
                    sid = STRATEGY_ORDER[idx % len(STRATEGY_ORDER)]
                    idx += 1

                    # Fetch positions + trades from the strategy API
                    await self._refresh_strategy_detail(sid)

                    msg = self.dashboard_messages.get(sid)
                    if msg:
                        embed = self._build_strategy_embed(sid)
                        try:
                            await msg.edit(embed=embed)
                        except discord.HTTPException as exc:
                            logger.warning("Failed to edit %s embed: %s", sid, exc)

                await asyncio.sleep(STRATEGY_ROTATION_INTERVAL)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Strategy rotation error: %s", exc)
                await asyncio.sleep(5)

    async def _refresh_strategy_detail(self, sid: str) -> None:
        """Fetch positions and recent trades from a strategy's HTTP API."""
        info = STRATEGY_MAP.get(sid)
        if info is None:
            return

        base = f"http://{info['host']}:{info['port']}"
        detail: Dict[str, Any] = {}

        # Fetch positions and trades in parallel
        pos_url = f"{base}/api/positions"
        trades_url = f"{base}/api/trades?limit=5"

        pos_data, trades_data = await asyncio.gather(
            self._http_get(pos_url),
            self._http_get(trades_url),
            return_exceptions=True,
        )

        if isinstance(pos_data, list):
            detail["positions"] = pos_data
        elif isinstance(pos_data, Exception) or pos_data is None:
            detail["positions"] = []

        if isinstance(trades_data, list):
            detail["trades"] = trades_data
        elif isinstance(trades_data, Exception) or trades_data is None:
            detail["trades"] = []

        self.strategy_detail_cache[sid] = detail

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _http_get(self, url: str) -> Any:
        if self.http_session is None or self.http_session.closed:
            return None
        try:
            async with self.http_session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        return None

    async def _http_post(self, url: str) -> Optional[Dict[str, Any]]:
        if self.http_session is None or self.http_session.closed:
            return None
        try:
            async with self.http_session.post(url) as resp:
                try:
                    body = await resp.json()
                except Exception:
                    body = {"status": resp.status, "text": await resp.text()}
                if resp.status in (200, 201, 202, 204):
                    return body
                return body
        except Exception as exc:
            logger.warning("POST %s failed: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _format_trade_time(ts: Any) -> str:
    """Convert a timestamp (ms, s, or ISO string) to HH:MM UTC."""
    try:
        if isinstance(ts, (int, float)):
            # Assume ms if > 1e12
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M UTC")
        if isinstance(ts, str) and ts:
            return ts[:16]
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Instantiate bot and register slash commands
# ---------------------------------------------------------------------------
bot = StrategyLabBot()


def strategy_id_autocomplete() -> List[app_commands.Choice[str]]:
    return [app_commands.Choice(name=sid, value=sid) for sid in VALID_STRATEGY_IDS]


# /status — DM full report
@bot.tree.command(name="status", description="DM a full status report from the aggregator")
async def cmd_status(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    data = await bot._http_get(f"{AGGREGATOR_HTTP_URL}/api/summary")
    if data is None:
        await interaction.followup.send("Could not reach the aggregator. Try again later.", ephemeral=True)
        return

    total_eq = data.get("total_equity", 0.0)
    total_pnl = data.get("total_pnl", 0.0)
    pnl_pct = data.get("total_pnl_pct", 0.0)
    pnl_sign = "+" if total_pnl >= 0 else ""
    pct_sign = "+" if pnl_pct >= 0 else ""

    lines: List[str] = [
        "**Crypto Strategy Lab — Status Report**\n",
        f"\U0001F4CA Portfolio: ${total_eq:,.2f} | P&L: {pnl_sign}${abs(total_pnl):,.2f} ({pct_sign}{pnl_pct:.2f}%)\n",
    ]

    strategies = data.get("strategies", [])
    for s in strategies:
        sid = s.get("strategy_id", "???")
        name = s.get("strategy_name", STRATEGY_MAP.get(sid, {}).get("name", "Unknown"))
        st = s.get("status", "offline")
        if st in ("offline", "crashed"):
            lines.append(f"\U0001F480 **{sid}** {name} \u2014 OFFLINE")
        else:
            pnl = s.get("pnl_pct", 0.0)
            trades = s.get("trade_count", 0)
            sharpe = s.get("sharpe_ratio", 0.0)
            sign = "+" if pnl >= 0 else ""
            lines.append(f"**{sid}** {name} \u2014 {sign}{pnl:.2f}% | {trades} trades | Sharpe {sharpe:.1f}")

    report = "\n".join(lines)
    try:
        await interaction.user.send(report)
        await interaction.followup.send("Status report sent to your DMs.", ephemeral=True)
    except discord.Forbidden:
        await interaction.followup.send(
            "I can't DM you. Please enable DMs.\n\n" + report, ephemeral=True
        )


# /kill <strat> — kill strategy
@bot.tree.command(name="kill", description="Kill a running strategy")
@app_commands.describe(strategy_id="Strategy ID to kill (e.g. STRAT-001)")
@app_commands.choices(strategy_id=strategy_id_autocomplete())
async def cmd_kill(interaction: discord.Interaction, strategy_id: str) -> None:
    await interaction.response.defer(ephemeral=False)
    strategy_id = strategy_id.upper()

    if strategy_id not in STRATEGY_MAP:
        await interaction.followup.send(f"Unknown strategy: `{strategy_id}`.")
        return

    info = STRATEGY_MAP[strategy_id]
    url = f"http://{info['host']}:{info['port']}/api/kill"
    result = await bot._http_post(url)

    if result is not None:
        await interaction.followup.send(
            f"\u2705 Kill sent to **{strategy_id}** ({info['name']}). Response: `{json.dumps(result)}`"
        )
    else:
        await interaction.followup.send(
            f"\u274C Failed to reach **{strategy_id}**. May already be down."
        )


# /restart <strat> — restart strategy
@bot.tree.command(name="restart", description="Restart a strategy via the aggregator")
@app_commands.describe(strategy_id="Strategy ID to restart (e.g. STRAT-001)")
@app_commands.choices(strategy_id=strategy_id_autocomplete())
async def cmd_restart(interaction: discord.Interaction, strategy_id: str) -> None:
    await interaction.response.defer(ephemeral=False)
    strategy_id = strategy_id.upper()

    if strategy_id not in STRATEGY_MAP:
        await interaction.followup.send(f"Unknown strategy: `{strategy_id}`.")
        return

    url = f"{AGGREGATOR_HTTP_URL}/api/restart/{strategy_id}"
    result = await bot._http_post(url)

    if result is not None:
        await interaction.followup.send(
            f"\u2705 Restart sent for **{strategy_id}** ({STRATEGY_MAP[strategy_id]['name']}). "
            f"Response: `{json.dumps(result)}`"
        )
    else:
        await interaction.followup.send(
            f"\u274C Failed to reach aggregator for restart of **{strategy_id}**."
        )


# /leaderboard — post current leaderboard
@bot.tree.command(name="leaderboard", description="Post the current leaderboard to this channel")
async def cmd_leaderboard(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=False)

    if not bot.latest_strategies:
        fetched = await bot._http_get(f"{AGGREGATOR_HTTP_URL}/api/leaderboard")
        if isinstance(fetched, list) and fetched:
            bot.latest_strategies = fetched
        else:
            await interaction.followup.send("No strategy data available yet.")
            return

    embed = bot._build_leaderboard_embed()
    await interaction.followup.send(embed=embed)


# /strategy <strat_id> — post detailed strategy card
@bot.tree.command(name="strategy", description="Post a detailed strategy card")
@app_commands.describe(strategy_id="Strategy ID (e.g. STRAT-001)")
@app_commands.choices(strategy_id=strategy_id_autocomplete())
async def cmd_strategy(interaction: discord.Interaction, strategy_id: str) -> None:
    await interaction.response.defer(ephemeral=False)
    strategy_id = strategy_id.upper()

    if strategy_id not in STRATEGY_MAP:
        await interaction.followup.send(f"Unknown strategy: `{strategy_id}`.")
        return

    # Refresh detail data
    await bot._refresh_strategy_detail(strategy_id)

    # If no WS data yet, try to populate from aggregator
    if not bot.latest_strategies:
        fetched = await bot._http_get(f"{AGGREGATOR_HTTP_URL}/api/leaderboard")
        if isinstance(fetched, list):
            bot.latest_strategies = fetched

    embed = bot._build_strategy_embed(strategy_id)
    await interaction.followup.send(embed=embed)


# /pause <strat_id> — pause a strategy
@bot.tree.command(name="pause", description="Pause a strategy (calls /api/pause)")
@app_commands.describe(strategy_id="Strategy ID to pause (e.g. STRAT-001)")
@app_commands.choices(strategy_id=strategy_id_autocomplete())
async def cmd_pause(interaction: discord.Interaction, strategy_id: str) -> None:
    await interaction.response.defer(ephemeral=False)
    strategy_id = strategy_id.upper()

    if strategy_id not in STRATEGY_MAP:
        await interaction.followup.send(f"Unknown strategy: `{strategy_id}`.")
        return

    info = STRATEGY_MAP[strategy_id]
    url = f"http://{info['host']}:{info['port']}/api/pause"
    result = await bot._http_post(url)

    if result is not None:
        await interaction.followup.send(
            f"\u23F8\uFE0F Pause sent to **{strategy_id}** ({info['name']}). Response: `{json.dumps(result)}`"
        )
    else:
        await interaction.followup.send(
            f"\u274C Failed to reach **{strategy_id}**. May be offline."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not DISCORD_BOT_TOKEN:
        logger.error("DISCORD_BOT_TOKEN not set. Exiting.")
        sys.exit(1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler() -> None:
        logger.info("Received shutdown signal.")
        loop.create_task(bot.close())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    logger.info("Starting Crypto Strategy Lab Discord dashboard bot...")
    bot.run(DISCORD_BOT_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
