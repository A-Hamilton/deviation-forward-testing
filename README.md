# Deviation Magnet - Forward Tester

Real-time paper trading bot for the Deviation Magnet strategy on Bybit perpetuals.

## Strategy

Enters positions when price hits extreme deviation bands (3σ from 20-period SMA):
- **Long** when close ≤ lower band (oversold)
- **Short** when close ≥ upper band (overbought)
- **Exit** when profit > 0.2% or max hold time (2 hours)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python deviation_magnet_forward.py
```

## Deploy to Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app)
3. New Project → Deploy from GitHub
4. Select your repo → Done!

## Configuration

Edit `deviation_magnet_forward.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TIMEFRAME` | 5 | Candle timeframe in minutes |
| `CHECK_INTERVAL` | 30 | Seconds between checks |
| `POSITION_SIZE` | 100 | USD per trade |
| `PROFIT_TARGET_PCT` | 0.2 | Exit at this % profit |
| `MAX_HOLD_MINUTES` | 120 | Force exit after this |

## Data Persistence

Trades and state are saved to `forward_test_data/`:
- `trades.json` - All completed trades
- `state.json` - Open positions (survives restarts)
- `forward_test.log` - Full log history

## Files

```
├── deviation_magnet_forward.py  # Main bot
├── requirements.txt             # Dependencies
├── Procfile                     # Railway config
├── runtime.txt                  # Python version
└── README.md                    # This file
```

## License

MIT

