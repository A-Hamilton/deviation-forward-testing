"""
Deviation Magnet Forward Tester
===============================

Real-time paper trading bot for the Deviation Magnet strategy.
Monitors Bybit perpetuals and enters when price hits extreme deviation bands.

Usage:
    python deviation_magnet_forward.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Config:
    """Trading configuration parameters."""

    # Indicator settings (matches TradingView)
    bb_length: int = 20
    mult: float = 3.0
    dev_mult: float = 1.5

    # Timing
    timeframe: str = "5"  # 5-minute candles
    check_interval: int = 30  # seconds between checks

    # Position management
    position_size: float = 100.0  # USD per trade
    fee_pct: float = 0.0011  # 0.11% round trip
    profit_target_pct: float = 0.2  # Exit at 0.2% profit
    max_hold_minutes: int = 120  # Force exit after 2 hours

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("forward_test_data"))

    @property
    def trades_file(self) -> Path:
        return self.data_dir / "trades.json"

    @property
    def state_file(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def log_file(self) -> Path:
        return self.data_dir / "forward_test.log"


# Symbols to monitor (top liquid Bybit perpetuals)
SYMBOLS: list[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT",
    "SUIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "NEARUSDT",
    "ATOMUSDT", "LTCUSDT", "BNBUSDT", "INJUSDT", "TAOUSDT",
    "PEPEUSDT", "WIFUSDT", "FETUSDT", "RENDERUSDT", "ONDOUSDT",
    "JUPUSDT", "ENAUSDT", "STXUSDT", "IMXUSDT", "SEIUSDT",
]

# Global config instance
config = Config()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Position:
    """Represents an open trading position."""

    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    entry_time: datetime
    bar_time: str


@dataclass
class Trade:
    """Represents a completed trade."""

    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    pnl_pct: float
    pnl_usd: float
    hold_seconds: int
    exit_reason: str


@dataclass
class BandData:
    """Calculated indicator data for a symbol."""

    close: float
    high: float
    low: float
    upper3: float
    lower3: float
    basis: float
    bar_time: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════


class TradingState:
    """Manages trading state with persistence."""

    def __init__(self) -> None:
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.signals_seen: dict[str, str] = {}
        self.start_time: datetime = datetime.now()
        self.total_signals: int = 0
        self.api_errors: int = 0

    def save(self) -> None:
        """Save state to disk."""
        try:
            positions_data = {}
            for sym, pos in self.positions.items():
                positions_data[sym] = {
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time.isoformat(),
                    "bar_time": pos.bar_time,
                }

            state = {
                "positions": positions_data,
                "start_time": self.start_time.isoformat(),
                "total_signals": self.total_signals,
                "api_errors": self.api_errors,
            }

            with open(config.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load(self) -> None:
        """Load state from disk if exists."""
        if not config.state_file.exists():
            return

        try:
            with open(config.state_file, "r") as f:
                state = json.load(f)

            for sym, data in state.get("positions", {}).items():
                self.positions[sym] = Position(
                    symbol=data["symbol"],
                    direction=data["direction"],
                    entry_price=data["entry_price"],
                    entry_time=datetime.fromisoformat(data["entry_time"]),
                    bar_time=data["bar_time"],
                )

            if state.get("start_time"):
                self.start_time = datetime.fromisoformat(state["start_time"])
            self.total_signals = state.get("total_signals", 0)
            self.api_errors = state.get("api_errors", 0)

            logger.info(f"Restored {len(self.positions)} open positions")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def load_trades(self) -> None:
        """Load historical trades from disk."""
        if not config.trades_file.exists():
            return

        try:
            with open(config.trades_file, "r") as f:
                trades_data = json.load(f)

            for t in trades_data:
                self.trades.append(Trade(**t))

            logger.info(f"Loaded {len(self.trades)} historical trades")
        except Exception as e:
            logger.error(f"Failed to load trades: {e}")

    def save_trade(self, trade: Trade) -> None:
        """Append trade to trades file."""
        try:
            trades_data = []
            if config.trades_file.exists():
                with open(config.trades_file, "r") as f:
                    trades_data = json.load(f)

            trades_data.append(asdict(trade))

            with open(config.trades_file, "w") as f:
                json.dump(trades_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")

    def cleanup_signals(self) -> None:
        """Remove old signals to prevent memory leak."""
        if len(self.signals_seen) > 1000:
            # Keep only the 500 most recent
            items = list(self.signals_seen.items())
            self.signals_seen = dict(items[-500:])


# Global state
state = TradingState()


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════


def setup_logging() -> logging.Logger:
    """Configure logging with console and file handlers."""
    log = logging.getLogger("deviation_magnet")
    log.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    log.addHandler(console)

    # File handler (if data dir exists)
    if config.data_dir.exists():
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

    return log


logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════════
# API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_klines(symbol: str, limit: int = 50) -> Optional[pd.DataFrame]:
    """
    Fetch recent klines from Bybit.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        limit: Number of candles to fetch

    Returns:
        DataFrame with OHLCV data or None on error
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": config.timeframe,
        "limit": limit,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data["retCode"] != 0 or not data["result"]["list"]:
            return None

        df = pd.DataFrame(
            data["result"]["list"],
            columns=["open_time", "open", "high", "low", "close", "volume", "turnover"],
        )
        df = df.iloc[::-1].reset_index(drop=True)
        df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms")

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)

        return df

    except requests.exceptions.RequestException:
        state.api_errors += 1
        return None
    except Exception:
        state.api_errors += 1
        return None


def get_current_price(symbol: str) -> Optional[float]:
    """Get current last price from Bybit."""
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": "linear", "symbol": symbol}

    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        if data["retCode"] == 0 and data["result"]["list"]:
            return float(data["result"]["list"][0]["lastPrice"])
    except Exception:
        pass

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_bands(df: pd.DataFrame) -> Optional[BandData]:
    """
    Calculate deviation bands from OHLCV data.

    Uses SMA(20) of OHLC4 with 3σ standard deviation bands.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        BandData with calculated values or None if insufficient data
    """
    if df is None or len(df) < config.bb_length:
        return None

    df = df.copy()
    df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    df["basis"] = df["ohlc4"].rolling(config.bb_length).mean()
    df["stdev"] = df["ohlc4"].rolling(config.bb_length).std(ddof=0)  # Population std
    df["dev"] = config.mult * df["stdev"]
    df["upper3"] = df["basis"] + df["dev"] * config.dev_mult
    df["lower3"] = df["basis"] - df["dev"] * config.dev_mult

    latest = df.iloc[-1]
    if pd.isna(latest["upper3"]):
        return None

    return BandData(
        close=latest["close"],
        high=latest["high"],
        low=latest["low"],
        upper3=latest["upper3"],
        lower3=latest["lower3"],
        basis=latest["basis"],
        bar_time=latest["open_time"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════


def check_entry_signal(symbol: str, data: BandData) -> Optional[str]:
    """
    Check for entry signal.

    Args:
        symbol: Trading pair
        data: Calculated band data

    Returns:
        "long", "short", or None
    """
    if symbol in state.positions:
        return None  # Already in position

    bar_key = f"{symbol}_{data.bar_time}"
    if bar_key in state.signals_seen:
        return None  # Already signaled this bar

    if data.close <= data.lower3:
        state.signals_seen[bar_key] = "long"
        state.total_signals += 1
        return "long"
    elif data.close >= data.upper3:
        state.signals_seen[bar_key] = "short"
        state.total_signals += 1
        return "short"

    return None


def check_exit_signal(symbol: str, current_price: float) -> tuple[bool, str]:
    """
    Check if position should exit.

    Args:
        symbol: Trading pair
        current_price: Current market price

    Returns:
        Tuple of (should_exit, reason)
    """
    if symbol not in state.positions:
        return False, ""

    pos = state.positions[symbol]

    # Calculate unrealized PnL
    if pos.direction == "long":
        pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
    else:
        pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100

    # Check profit target
    if pnl_pct >= config.profit_target_pct:
        return True, "profit_target"

    # Check max hold time
    hold_minutes = (datetime.now() - pos.entry_time).total_seconds() / 60
    if hold_minutes >= config.max_hold_minutes:
        return True, "max_hold"

    return False, ""


def enter_position(symbol: str, direction: str, price: float, bar_time: datetime) -> None:
    """Open a new position."""
    state.positions[symbol] = Position(
        symbol=symbol,
        direction=direction,
        entry_price=price,
        entry_time=datetime.now(),
        bar_time=str(bar_time),
    )
    logger.info(f"📥 ENTRY: {symbol} {direction.upper()} @ ${price:.4f}")
    state.save()


def exit_position(symbol: str, exit_price: float, reason: str) -> None:
    """Close a position and record the trade."""
    if symbol not in state.positions:
        return

    pos = state.positions[symbol]

    # Calculate PnL
    if pos.direction == "long":
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
    else:
        pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

    pnl_usd = (pnl_pct / 100 * config.position_size) - (config.position_size * config.fee_pct)

    # Create trade record
    trade = Trade(
        symbol=symbol,
        direction=pos.direction,
        entry_price=pos.entry_price,
        exit_price=exit_price,
        entry_time=pos.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
        exit_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pnl_pct=round(pnl_pct, 4),
        pnl_usd=round(pnl_usd, 4),
        hold_seconds=int((datetime.now() - pos.entry_time).total_seconds()),
        exit_reason=reason,
    )

    state.trades.append(trade)
    state.save_trade(trade)

    # Log result
    emoji = "✅" if pnl_usd > 0 else "❌"
    logger.info(
        f"{emoji} EXIT: {symbol} {pos.direction.upper()} @ ${exit_price:.4f} | "
        f"PnL: ${pnl_usd:.2f} ({pnl_pct:.2f}%) | Reason: {reason}"
    )

    del state.positions[symbol]
    state.save()


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════


def print_status() -> None:
    """Print current trading status."""
    runtime = datetime.now() - state.start_time

    print(f"\n{'─' * 70}")
    print(f"  ⏱  Runtime: {str(runtime).split('.')[0]} | Signals: {state.total_signals} | Errors: {state.api_errors}")
    print(f"{'─' * 70}")

    # Open positions
    print(f"  📊 Open Positions: {len(state.positions)}")
    for sym, pos in state.positions.items():
        price = get_current_price(sym)
        if price:
            if pos.direction == "long":
                pnl = (price - pos.entry_price) / pos.entry_price * 100
            else:
                pnl = (pos.entry_price - price) / pos.entry_price * 100

            hold_mins = (datetime.now() - pos.entry_time).total_seconds() / 60
            color = "\033[92m" if pnl > 0 else "\033[91m"
            print(
                f"      {sym:<12} {pos.direction.upper():<5} "
                f"${pos.entry_price:<12.4f} → ${price:<12.4f} "
                f"{color}{pnl:>+7.2f}%\033[0m  ({hold_mins:.0f}m)"
            )

    # Trade statistics
    print(f"\n  📈 Closed Trades: {len(state.trades)}")
    if state.trades:
        total_pnl = sum(t.pnl_usd for t in state.trades)
        winners = [t for t in state.trades if t.pnl_usd > 0]
        losers = [t for t in state.trades if t.pnl_usd <= 0]

        win_rate = len(winners) / len(state.trades) * 100 if state.trades else 0
        avg_win = sum(t.pnl_usd for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t.pnl_usd for t in losers) / len(losers) if losers else 0

        color = "\033[92m" if total_pnl > 0 else "\033[91m"
        print(f"      Total PnL: {color}${total_pnl:.2f}\033[0m | Win Rate: {win_rate:.0f}% ({len(winners)}W / {len(losers)}L)")
        print(f"      Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")

        # Last 5 trades
        print(f"\n      Recent trades:")
        for t in state.trades[-5:]:
            emoji = "✅" if t.pnl_usd > 0 else "❌"
            print(f"        {emoji} {t.symbol:<10} {t.direction:<5} ${t.pnl_usd:>+7.2f} ({t.exit_reason})")

    print(f"{'─' * 70}\n")


def print_header() -> None:
    """Print startup header."""
    print(f"\n{'═' * 70}")
    print(f"  🚀 DEVIATION MAGNET - 24/7 FORWARD TEST")
    print(f"{'═' * 70}")
    print(f"  Timeframe:      {config.timeframe}m candles")
    print(f"  Check Interval: {config.check_interval}s")
    print(f"  Symbols:        {len(SYMBOLS)}")
    print(f"  Position Size:  ${config.position_size}")
    print(f"  Profit Target:  {config.profit_target_pct}%")
    print(f"  Max Hold:       {config.max_hold_minutes} minutes")
    print(f"  Bands:          SMA({config.bb_length}) ± {config.mult}σ × {config.dev_mult}")
    print(f"{'═' * 70}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if state.positions:
        print(f"  Restored: {len(state.positions)} open positions")
    if state.trades:
        print(f"  Historical: {len(state.trades)} trades")
    print(f"{'═' * 70}")
    print(f"  Press Ctrl+C to stop (state will be saved)")
    print(f"{'═' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main trading loop."""
    # Initialize
    config.data_dir.mkdir(exist_ok=True)

    # Re-setup logging with file handler now that dir exists
    global logger
    logger = setup_logging()

    # Load previous state
    state.load()
    state.load_trades()

    print_header()

    iteration = 0
    last_daily = datetime.now().date()

    try:
        while True:
            iteration += 1

            # Process each symbol
            for symbol in SYMBOLS:
                try:
                    df = fetch_klines(symbol)
                    data = calculate_bands(df)

                    if data is None:
                        continue

                    current_price = data.close

                    # Check exit first
                    if symbol in state.positions:
                        should_exit, reason = check_exit_signal(symbol, current_price)
                        if should_exit:
                            exit_position(symbol, current_price, reason)

                    # Check entry
                    signal = check_entry_signal(symbol, data)
                    if signal:
                        enter_position(symbol, signal, current_price, data.bar_time)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            # Status update every 5 iterations (~2.5 min)
            if iteration % 5 == 0:
                print_status()
                state.cleanup_signals()

            # Daily summary
            if datetime.now().date() != last_daily:
                last_daily = datetime.now().date()
                today_trades = [
                    t for t in state.trades
                    if t.exit_time.startswith(str(last_daily))
                ]
                if today_trades:
                    total = sum(t.pnl_usd for t in today_trades)
                    logger.info(f"📊 Daily: {len(today_trades)} trades | PnL: ${total:.2f}")

            time.sleep(config.check_interval)

    except KeyboardInterrupt:
        logger.info("Stopping... saving state")
        state.save()
        print_status()
        print("\n✅ State saved. Run again to resume.\n")


if __name__ == "__main__":
    main()
