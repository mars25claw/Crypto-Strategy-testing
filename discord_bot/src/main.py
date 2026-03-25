"""
Crypto Strategy Lab - Discord Bot Service

Live dashboard and command interface for monitoring 10 crypto trading strategies.
Connects to the aggregator WebSocket for real-time updates and provides slash
commands for status, kill, restart, and leaderboard operations.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

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
DISCORD_CHANNEL_ID: str = os.environ.get("DISCORD_CHANNEL_ID", "")

AGGREGATOR_WS_URL: str = os.environ.get("AGGREGATOR_WS_URL", "ws://aggregator:8099/ws")
AGGREGATOR_HTTP_URL: str = os.environ.get("AGGREGATOR_HTTP_URL", "http://aggregator:8099")

DASHBOARD_CHANNEL_NAME = "strategy-lab"
EMBED_EDIT_COOLDOWN = 2.0  # seconds between embed edits

# Strategy ID -> (host, port) mapping
STRATEGY_MAP: Dict[str, Dict[str, Any]] = {
    "STRAT-001": {"host": "strat-001-trend-following", "port": 8081, "name": "Trend Following"},
    "STRAT-002": {"host": "strat-002-funding-arb", "port": 8082, "name": "Funding Arb"},
    "STRAT-003": {"host": "strat-003-stat-arb-pairs", "port": 8083, "name": "Stat Arb Pairs"},
    "STRAT-004": {"host": "strat-004-mean-reversion", "port": 8084, "name": "Mean Reversion"},
    "STRAT-005": {"host": "strat-005-grid-trading", "port": 8085, "name": "Grid Trading"},
    "STRAT-006": {"host": "strat-006-market-making", "port": 8086, "name": "Market Making"},
    "STRAT-007": {"host": "strat-007-triangular-arb", "port": 8087, "name": "Triangular Arb"},
    "STRAT-008": {"host": "strat-008-options-vol", "port": 8088, "name": "Options Vol"},
    "STRAT-009": {"host": "strat-009-signal-dca", "port": 8089, "name": "Signal DCA"},
    "STRAT-010": {"host": "strat-010-ml-onchain", "port": 8090, "name": "ML On-Chain"},
}

VALID_STRATEGY_IDS: List[str] = sorted(STRATEGY_MAP.keys())

# ---------------------------------------------------------------------------
# Bot Client
# ---------------------------------------------------------------------------

class StrategyLabBot(discord.Client):
    """Discord bot that displays a live dashboard and handles slash commands."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.guilds = True
        intents.guild_messages = True
        intents.message_content = True
        super().__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)

        # State
        self.dashboard_channel: Optional[discord.TextChannel] = None
        self.dashboard_message: Optional[discord.Message] = None
        self.target_guild: Optional[discord.Guild] = None
        self.latest_data: Dict[str, Any] = {}
        self.last_embed_edit: float = 0.0
        self.pending_update: bool = False
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.ws_task: Optional[asyncio.Task] = None
        self.update_task: Optional[asyncio.Task] = None
        self.shutting_down: bool = False

        # Alert tracking — keys are alert description strings; cleared when condition clears
        self.active_alerts: Set[str] = set()
        self.sent_alert_keys: Set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup_hook(self) -> None:
        """Called before the bot starts connecting to Discord."""
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        logger.info("HTTP session created.")

    async def on_ready(self) -> None:
        """Called when the bot is connected and ready."""
        logger.info("Logged in as %s (id=%s)", self.user, self.user.id)

        # Resolve guild
        self.target_guild = await self._resolve_guild()
        if self.target_guild is None:
            logger.error("Could not resolve guild. Shutting down.")
            await self.close()
            return

        # Sync slash commands to guild for instant availability
        self.tree.copy_global_to(guild=self.target_guild)
        await self.tree.sync(guild=self.target_guild)
        logger.info("Slash commands synced to guild %s.", self.target_guild.name)

        # Resolve or create dashboard channel
        self.dashboard_channel = await self._resolve_channel()
        if self.dashboard_channel is None:
            logger.error("Could not resolve or create dashboard channel. Shutting down.")
            await self.close()
            return

        # Find or create pinned dashboard message
        await self._resolve_dashboard_message()

        # Start WebSocket listener
        self.ws_task = asyncio.create_task(self._ws_loop(), name="ws_loop")

        # Start rate-limited embed updater
        self.update_task = asyncio.create_task(self._embed_update_loop(), name="embed_update_loop")

        logger.info("Bot fully started.")

    async def close(self) -> None:
        """Graceful shutdown."""
        self.shutting_down = True
        logger.info("Shutting down...")

        if self.ws_task and not self.ws_task.done():
            self.ws_task.cancel()
        if self.update_task and not self.update_task.done():
            self.update_task.cancel()
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()

        await super().close()
        logger.info("Shutdown complete.")

    # ------------------------------------------------------------------
    # Guild / Channel resolution
    # ------------------------------------------------------------------

    async def _resolve_guild(self) -> Optional[discord.Guild]:
        if DISCORD_GUILD_ID:
            try:
                guild = self.get_guild(int(DISCORD_GUILD_ID))
                if guild:
                    logger.info("Resolved guild by ID: %s", guild.name)
                    return guild
            except (ValueError, TypeError):
                pass
        # Fallback: use first guild
        if self.guilds:
            guild = self.guilds[0]
            logger.info("Using first available guild: %s", guild.name)
            return guild
        return None

    async def _resolve_channel(self) -> Optional[discord.TextChannel]:
        guild = self.target_guild
        if guild is None:
            return None

        # Try by explicit channel ID
        if DISCORD_CHANNEL_ID:
            try:
                ch = guild.get_channel(int(DISCORD_CHANNEL_ID))
                if isinstance(ch, discord.TextChannel):
                    logger.info("Resolved channel by ID: #%s", ch.name)
                    return ch
            except (ValueError, TypeError):
                pass

        # Try by name
        for ch in guild.text_channels:
            if ch.name == DASHBOARD_CHANNEL_NAME:
                logger.info("Found existing channel: #%s", ch.name)
                return ch

        # Create it
        try:
            ch = await guild.create_text_channel(
                DASHBOARD_CHANNEL_NAME,
                topic="Crypto Strategy Lab - Live Dashboard",
                reason="Strategy Lab bot auto-setup",
            )
            logger.info("Created channel: #%s", ch.name)
            return ch
        except discord.Forbidden:
            logger.error("Missing permissions to create channel.")
        except discord.HTTPException as exc:
            logger.error("Failed to create channel: %s", exc)
        return None

    async def _resolve_dashboard_message(self) -> None:
        """Find existing pinned bot message or create + pin a new one."""
        channel = self.dashboard_channel
        if channel is None:
            return

        try:
            pins = await channel.pins()
            for msg in pins:
                if msg.author == self.user and msg.embeds:
                    self.dashboard_message = msg
                    logger.info("Reusing existing pinned dashboard message (id=%s).", msg.id)
                    return
        except discord.HTTPException as exc:
            logger.warning("Could not fetch pins: %s", exc)

        # Create new embed
        embed = self._build_embed(None)
        try:
            msg = await channel.send(embed=embed)
            await msg.pin(reason="Strategy Lab live dashboard")
            self.dashboard_message = msg
            logger.info("Created and pinned new dashboard message (id=%s).", msg.id)
        except discord.HTTPException as exc:
            logger.error("Failed to create dashboard message: %s", exc)

    # ------------------------------------------------------------------
    # Embed builder
    # ------------------------------------------------------------------

    @staticmethod
    def _strategy_emoji(strategy: Dict[str, Any]) -> str:
        """Pick the status emoji for a single strategy row."""
        if strategy.get("offline", False):
            return "\U0001F480"  # skull
        trades = strategy.get("total_trades", 0)
        if trades == 0:
            return "\u26AA"  # white circle
        pnl_pct = strategy.get("pnl_pct", 0.0)
        if pnl_pct > 0.5:
            return "\U0001F7E2"  # green
        if pnl_pct < -0.5:
            return "\U0001F534"  # red
        return "\U0001F7E1"  # yellow

    @staticmethod
    def _strategy_detail(strategy: Dict[str, Any]) -> str:
        """Format the right-hand detail for one leaderboard line."""
        if strategy.get("offline", False):
            return "OFFLINE"
        trades = strategy.get("total_trades", 0)
        if trades == 0:
            return "0.00% |  0 trades | warming up"
        pnl_pct = strategy.get("pnl_pct", 0.0)
        sharpe = strategy.get("sharpe", 0.0)
        sign = "+" if pnl_pct >= 0 else ""
        return f"{sign}{pnl_pct:.2f}% | {trades} trades | Sharpe {sharpe:.1f}"

    def _build_embed(self, data: Optional[Dict[str, Any]]) -> discord.Embed:
        """Build the dashboard embed from aggregator data."""
        embed = discord.Embed(
            title="\U0001F9EA CRYPTO STRATEGY LAB \u2014 LIVE",
            color=0x2F3136,
        )

        if data is None:
            embed.description = (
                "\u2501" * 24 + "\n"
                "\u23F3 Waiting for aggregator data...\n"
                + "\u2501" * 24
            )
            return embed

        # Portfolio summary
        portfolio_total = data.get("portfolio_total", 0.0)
        portfolio_pnl = data.get("portfolio_pnl", 0.0)
        portfolio_pnl_pct = data.get("portfolio_pnl_pct", 0.0)
        pnl_sign = "+" if portfolio_pnl >= 0 else ""
        pct_sign = "+" if portfolio_pnl_pct >= 0 else ""
        now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        # Build leaderboard rows
        strategies: List[Dict[str, Any]] = data.get("strategies", [])
        # Sort: offline last, then by pnl_pct descending
        strategies_sorted = sorted(
            strategies,
            key=lambda s: (s.get("offline", False), -(s.get("pnl_pct", 0.0))),
        )

        leaderboard_lines: List[str] = []
        for rank, strat in enumerate(strategies_sorted, start=1):
            emoji = self._strategy_emoji(strat)
            strat_id = strat.get("strategy_id", "STRAT-???")
            strat_name = strat.get("name", STRATEGY_MAP.get(strat_id, {}).get("name", "Unknown"))
            detail = self._strategy_detail(strat)
            line = f"#{rank:<2} {emoji} {strat_id} {strat_name:<20} {detail}"
            leaderboard_lines.append(line)

        leaderboard = "\n".join(leaderboard_lines) if leaderboard_lines else "No strategy data yet."

        # Alerts
        alerts: List[str] = data.get("alerts", [])
        alert_text = ", ".join(alerts) if alerts else "None"

        description = (
            "\u2501" * 24 + "\n"
            f"\U0001F4CA Portfolio: ${portfolio_total:,.2f} total | "
            f"P&L: {pnl_sign}${abs(portfolio_pnl):,.2f} ({pct_sign}{portfolio_pnl_pct:.2f}%)\n"
            f"\U0001F550 Updated: {now_utc}\n\n"
            f"\U0001F3C6 LEADERBOARD\n"
            f"{leaderboard}\n\n"
            + "\u2501" * 24 + "\n"
            f"\u26A0\uFE0F ALERTS: {alert_text}"
        )
        embed.description = description
        return embed

    # ------------------------------------------------------------------
    # WebSocket listener with auto-reconnect
    # ------------------------------------------------------------------

    async def _ws_loop(self) -> None:
        """Connect to aggregator WebSocket and listen for updates. Auto-reconnects."""
        import websockets
        import websockets.exceptions

        backoff = 1.0
        max_backoff = 30.0

        while not self.shutting_down:
            try:
                logger.info("Connecting to aggregator WebSocket at %s ...", AGGREGATOR_WS_URL)
                async with websockets.connect(AGGREGATOR_WS_URL) as ws:
                    logger.info("WebSocket connected.")
                    backoff = 1.0  # reset on success

                    async for raw in ws:
                        if self.shutting_down:
                            break
                        try:
                            data = json.loads(raw)
                            self.latest_data = data
                            self.pending_update = True
                            self._check_alerts(data)
                        except json.JSONDecodeError:
                            logger.warning("Received non-JSON WebSocket message, ignoring.")

            except asyncio.CancelledError:
                logger.info("WebSocket loop cancelled.")
                return
            except Exception as exc:
                if self.shutting_down:
                    return
                logger.warning("WebSocket error: %s. Reconnecting in %.1fs...", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    # ------------------------------------------------------------------
    # Rate-limited embed updater
    # ------------------------------------------------------------------

    async def _embed_update_loop(self) -> None:
        """Polls for pending updates and edits the embed, rate-limited."""
        while not self.shutting_down:
            try:
                if self.pending_update and self.dashboard_message:
                    now = time.monotonic()
                    elapsed = now - self.last_embed_edit
                    if elapsed >= EMBED_EDIT_COOLDOWN:
                        embed = self._build_embed(self.latest_data)
                        try:
                            await self.dashboard_message.edit(embed=embed)
                            self.last_embed_edit = time.monotonic()
                            self.pending_update = False
                        except discord.HTTPException as exc:
                            logger.warning("Failed to edit dashboard embed: %s", exc)
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Unexpected error in embed update loop: %s", exc)
                await asyncio.sleep(2)

    # ------------------------------------------------------------------
    # Alert system
    # ------------------------------------------------------------------

    def _check_alerts(self, data: Dict[str, Any]) -> None:
        """Evaluate alert conditions and DM guild owner if needed."""
        new_alerts: Set[str] = set()

        # Portfolio-wide P&L check
        portfolio_pnl_pct = data.get("portfolio_pnl_pct", 0.0)
        if portfolio_pnl_pct < -10.0:
            new_alerts.add(f"portfolio_pnl_critical:{portfolio_pnl_pct:.2f}")

        strategies: List[Dict[str, Any]] = data.get("strategies", [])
        for strat in strategies:
            sid = strat.get("strategy_id", "UNKNOWN")
            if strat.get("offline", False):
                new_alerts.add(f"offline:{sid}")
            pnl_pct = strat.get("pnl_pct", 0.0)
            if pnl_pct < -5.0:
                new_alerts.add(f"pnl_critical:{sid}:{pnl_pct:.2f}")

        # Determine which alerts are new (not yet sent)
        to_send = new_alerts - self.sent_alert_keys

        # Clear sent alerts whose conditions are no longer active
        cleared = self.sent_alert_keys - new_alerts
        if cleared:
            self.sent_alert_keys -= cleared
            logger.info("Cleared alerts: %s", cleared)

        self.active_alerts = new_alerts

        if to_send:
            self.sent_alert_keys |= to_send
            asyncio.create_task(self._send_owner_alerts(to_send))

    async def _send_owner_alerts(self, alert_keys: Set[str]) -> None:
        """DM alert messages to the guild owner."""
        if self.target_guild is None:
            return
        try:
            owner = self.target_guild.owner
            if owner is None:
                # Fetch if not cached
                owner = await self.target_guild.fetch_member(self.target_guild.owner_id)
            if owner is None:
                logger.warning("Could not resolve guild owner for DM alerts.")
                return

            lines: List[str] = ["\u26A0\uFE0F **Crypto Strategy Lab Alert**\n"]
            for key in sorted(alert_keys):
                if key.startswith("portfolio_pnl_critical:"):
                    pct = key.split(":")[1]
                    lines.append(f"\U0001F6A8 Portfolio P&L is critically low: **{pct}%**")
                elif key.startswith("offline:"):
                    sid = key.split(":")[1]
                    lines.append(f"\U0001F480 **{sid}** container is OFFLINE")
                elif key.startswith("pnl_critical:"):
                    parts = key.split(":")
                    sid, pct = parts[1], parts[2]
                    lines.append(f"\U0001F534 **{sid}** P&L dropped to **{pct}%**")

            message = "\n".join(lines)
            await owner.send(message)
            logger.info("Sent %d alert(s) to guild owner.", len(alert_keys))
        except discord.Forbidden:
            logger.warning("Cannot DM guild owner (DMs disabled).")
        except discord.HTTPException as exc:
            logger.error("Failed to DM guild owner: %s", exc)
        except Exception as exc:
            logger.error("Unexpected error sending owner alerts: %s", exc)

    # ------------------------------------------------------------------
    # Helpers for HTTP calls
    # ------------------------------------------------------------------

    async def _http_get(self, url: str) -> Optional[Dict[str, Any]]:
        if self.http_session is None or self.http_session.closed:
            return None
        try:
            async with self.http_session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.warning("GET %s returned %d", url, resp.status)
        except Exception as exc:
            logger.warning("GET %s failed: %s", url, exc)
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
                logger.warning("POST %s returned %d: %s", url, resp.status, body)
                return body
        except Exception as exc:
            logger.warning("POST %s failed: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Instantiate bot and register slash commands
# ---------------------------------------------------------------------------
bot = StrategyLabBot()


def strategy_id_autocomplete() -> List[app_commands.Choice[str]]:
    return [app_commands.Choice(name=sid, value=sid) for sid in VALID_STRATEGY_IDS]


@bot.tree.command(name="status", description="DM a full status report from the aggregator")
async def cmd_status(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    url = f"{AGGREGATOR_HTTP_URL}/api/summary"
    data = await bot._http_get(url)
    if data is None:
        await interaction.followup.send("Could not reach the aggregator. Try again later.", ephemeral=True)
        return

    # Build a text report
    lines: List[str] = ["**Crypto Strategy Lab - Status Report**\n"]
    portfolio_total = data.get("portfolio_total", 0.0)
    portfolio_pnl = data.get("portfolio_pnl", 0.0)
    portfolio_pnl_pct = data.get("portfolio_pnl_pct", 0.0)
    pnl_sign = "+" if portfolio_pnl >= 0 else ""
    pct_sign = "+" if portfolio_pnl_pct >= 0 else ""
    lines.append(
        f"\U0001F4CA Portfolio: ${portfolio_total:,.2f} | "
        f"P&L: {pnl_sign}${abs(portfolio_pnl):,.2f} ({pct_sign}{portfolio_pnl_pct:.2f}%)\n"
    )

    strategies: List[Dict[str, Any]] = data.get("strategies", [])
    strategies_sorted = sorted(
        strategies,
        key=lambda s: (s.get("offline", False), -(s.get("pnl_pct", 0.0))),
    )
    for strat in strategies_sorted:
        sid = strat.get("strategy_id", "???")
        name = strat.get("name", STRATEGY_MAP.get(sid, {}).get("name", "Unknown"))
        if strat.get("offline", False):
            lines.append(f"\U0001F480 **{sid}** {name} \u2014 OFFLINE")
        else:
            pnl = strat.get("pnl_pct", 0.0)
            trades = strat.get("total_trades", 0)
            sharpe = strat.get("sharpe", 0.0)
            sign = "+" if pnl >= 0 else ""
            lines.append(f"**{sid}** {name} \u2014 {sign}{pnl:.2f}% | {trades} trades | Sharpe {sharpe:.1f}")

    report = "\n".join(lines)
    try:
        await interaction.user.send(report)
        await interaction.followup.send("Status report sent to your DMs.", ephemeral=True)
    except discord.Forbidden:
        await interaction.followup.send(
            "I can't DM you. Please enable DMs from server members.\n\n" + report,
            ephemeral=True,
        )


@bot.tree.command(name="kill", description="Kill a running strategy")
@app_commands.describe(strategy_id="Strategy ID to kill (e.g. STRAT-001)")
@app_commands.choices(strategy_id=strategy_id_autocomplete())
async def cmd_kill(interaction: discord.Interaction, strategy_id: str) -> None:
    await interaction.response.defer(ephemeral=False)
    strategy_id = strategy_id.upper()

    if strategy_id not in STRATEGY_MAP:
        await interaction.followup.send(
            f"Unknown strategy: `{strategy_id}`. Valid IDs: {', '.join(VALID_STRATEGY_IDS)}"
        )
        return

    info = STRATEGY_MAP[strategy_id]
    url = f"http://{info['host']}:{info['port']}/api/kill"
    result = await bot._http_post(url)

    if result is not None:
        await interaction.followup.send(
            f"\u2705 Kill command sent to **{strategy_id}** ({info['name']}). Response: `{json.dumps(result)}`"
        )
    else:
        await interaction.followup.send(
            f"\u274C Failed to reach **{strategy_id}** ({info['name']}). The container may already be down."
        )


@bot.tree.command(name="restart", description="Restart a strategy via the aggregator")
@app_commands.describe(strategy_id="Strategy ID to restart (e.g. STRAT-001)")
@app_commands.choices(strategy_id=strategy_id_autocomplete())
async def cmd_restart(interaction: discord.Interaction, strategy_id: str) -> None:
    await interaction.response.defer(ephemeral=False)
    strategy_id = strategy_id.upper()

    if strategy_id not in STRATEGY_MAP:
        await interaction.followup.send(
            f"Unknown strategy: `{strategy_id}`. Valid IDs: {', '.join(VALID_STRATEGY_IDS)}"
        )
        return

    url = f"{AGGREGATOR_HTTP_URL}/api/restart/{strategy_id}"
    result = await bot._http_post(url)

    if result is not None:
        await interaction.followup.send(
            f"\u2705 Restart command sent for **{strategy_id}** ({STRATEGY_MAP[strategy_id]['name']}). "
            f"Response: `{json.dumps(result)}`"
        )
    else:
        await interaction.followup.send(
            f"\u274C Failed to reach aggregator for restart of **{strategy_id}**. Try again later."
        )


@bot.tree.command(name="leaderboard", description="Post the current leaderboard to this channel")
async def cmd_leaderboard(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=False)

    data = bot.latest_data
    if not data or not data.get("strategies"):
        # Try fetching from aggregator
        fetched = await bot._http_get(f"{AGGREGATOR_HTTP_URL}/api/summary")
        if fetched:
            data = fetched
        else:
            await interaction.followup.send("No strategy data available yet. Try again later.")
            return

    portfolio_total = data.get("portfolio_total", 0.0)
    portfolio_pnl = data.get("portfolio_pnl", 0.0)
    portfolio_pnl_pct = data.get("portfolio_pnl_pct", 0.0)
    pnl_sign = "+" if portfolio_pnl >= 0 else ""
    pct_sign = "+" if portfolio_pnl_pct >= 0 else ""

    strategies: List[Dict[str, Any]] = data.get("strategies", [])
    strategies_sorted = sorted(
        strategies,
        key=lambda s: (s.get("offline", False), -(s.get("pnl_pct", 0.0))),
    )

    lines: List[str] = [
        "```",
        "\u2501" * 50,
        f"\U0001F4CA Portfolio: ${portfolio_total:,.2f} | P&L: {pnl_sign}${abs(portfolio_pnl):,.2f} ({pct_sign}{portfolio_pnl_pct:.2f}%)",
        f"\U0001F550 {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}",
        "",
        "\U0001F3C6 LEADERBOARD",
    ]

    for rank, strat in enumerate(strategies_sorted, start=1):
        emoji_code = StrategyLabBot._strategy_emoji(strat)
        sid = strat.get("strategy_id", "???")
        name = strat.get("name", STRATEGY_MAP.get(sid, {}).get("name", "Unknown"))
        detail = StrategyLabBot._strategy_detail(strat)
        lines.append(f"#{rank:<2} {emoji_code} {sid} {name:<20} {detail}")

    lines.append("\u2501" * 50)
    lines.append("```")

    await interaction.followup.send("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not DISCORD_BOT_TOKEN:
        logger.error("DISCORD_BOT_TOKEN not set. Exiting.")
        sys.exit(1)

    # Register signal handlers for graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler() -> None:
        logger.info("Received shutdown signal.")
        loop.create_task(bot.close())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    logger.info("Starting Crypto Strategy Lab Discord bot...")
    bot.run(DISCORD_BOT_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
