# Crypto Strategy Lab

A professional paper-trading laboratory running 10 independent crypto strategies with a unified aggregator dashboard and live Discord reporting.

## Quick Start

```bash
cp .env.example .env        # fill in Discord credentials
docker compose up -d         # launch everything
docker compose logs -f       # watch logs
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Docker Network: stratlab                      │
│                                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        ┌──────────┐        │
│  │ STRAT-001│ │ STRAT-002│ │ STRAT-003│  ...   │ STRAT-010│        │
│  │ Trend    │ │ Funding  │ │ Stat Arb │        │ ML/Chain │        │
│  │ :8081    │ │ :8082    │ │ :8083    │        │ :8090    │        │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘        └────┬─────┘        │
│       │             │            │                    │              │
│       └─────────────┴────────┬───┴────────────────────┘              │
│                              │                                       │
│                     ┌────────▼────────┐                              │
│                     │   AGGREGATOR    │                              │
│                     │   :8099         │                              │
│                     │  polls /metrics │                              │
│                     │  serves WS+API │                              │
│                     └────────┬────────┘                              │
│                              │ WebSocket                             │
│                     ┌────────▼────────┐                              │
│                     │  DISCORD BOT    │                              │
│                     │  live embed     │                              │
│                     │  slash commands │                              │
│                     └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Strategy Dashboards

| # | Strategy | Port | Description |
|---|----------|------|-------------|
| 001 | Trend Following & Momentum | [localhost:8081](http://localhost:8081) | EMA crossover with multi-timeframe confirmation, ATR-based stops |
| 002 | Funding Rate Arbitrage | [localhost:8082](http://localhost:8082) | Captures funding rate differentials between spot and perpetuals |
| 003 | Statistical Arbitrage (Pairs) | [localhost:8083](http://localhost:8083) | Cointegrated pair mean-reversion with z-score triggers |
| 004 | Mean Reversion | [localhost:8084](http://localhost:8084) | Bollinger Band / RSI mean-reversion on short timeframes |
| 005 | Grid Trading | [localhost:8085](http://localhost:8085) | Dynamic grid placement with volatility-adjusted spacing |
| 006 | Market Making | [localhost:8086](http://localhost:8086) | Bid/ask spread capture with inventory management |
| 007 | Triangular Arbitrage | [localhost:8087](http://localhost:8087) | Cross-exchange triangular path detection and execution |
| 008 | Options & Volatility | [localhost:8088](http://localhost:8088) | Deribit options strategies (simulation-only, no API key) |
| 009 | Signal-Enhanced DCA | [localhost:8089](http://localhost:8089) | Dollar-cost averaging with technical signal gating |
| 010 | ML & On-Chain Quant | [localhost:8090](http://localhost:8090) | Machine learning models with on-chain data features |
| — | **Aggregator** | [localhost:8099](http://localhost:8099) | Unified leaderboard, portfolio summary, WebSocket stream |

## Discord Setup

1. Create a Discord bot at [discord.com/developers](https://discord.com/developers/applications)
2. Enable **Message Content Intent**, **Server Members Intent**, and **Presence Intent**
3. Grant bot permissions: Send Messages, Embed Links, Manage Messages, Read Message History, Use Slash Commands
4. Invite bot to your server with the generated OAuth2 URL
5. Copy your bot token, guild ID, and (optionally) channel ID into `.env`

The bot will:
- Create/find a `#strategy-lab` channel
- Post and pin a live-updating embed with the leaderboard
- Respond to slash commands: `/status`, `/kill`, `/restart`, `/leaderboard`
- DM the server owner on critical alerts (strategy offline, P&L thresholds)

## Aggregator API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/leaderboard` | GET | Ranked strategy list with metrics |
| `/api/summary` | GET | Portfolio totals, best/worst strategy |
| `/api/health` | GET | Aggregator health check |
| `/ws` | WebSocket | Streams leaderboard updates every 500ms |

## Running Individual Strategies

Each strategy can run standalone:

```bash
docker network create stratlab
cd strat_001_trend_following
docker compose up -d
```

## Configuration

Each strategy has a `config.yaml` with:
- **Paper trading**: $800 starting equity per strategy ($8,000 total portfolio)
- **Risk management**: Per-trade limits, drawdown protection, circuit breakers
- **Instruments**: Top crypto pairs via Binance public WebSocket (no API key needed)

## Project Structure

```
crypto-strategy-lab/
├── docker-compose.yml          # Root orchestration (all services)
├── .env.example                # Environment template
├── shared/                     # Common library (mounted read-only)
│   ├── paper_trading.py
│   ├── dashboard_base.py
│   ├── circuit_breaker.py
│   └── ...
├── aggregator/                 # Metrics aggregator + WebSocket server
│   └── src/main.py
├── discord_bot/                # Live Discord dashboard bot
│   └── src/main.py
├── strat_001_trend_following/  # Strategy containers
├── strat_002_funding_arb/
├── ...
└── strat_010_ml_onchain/
```

## License

Private — internal use only.
