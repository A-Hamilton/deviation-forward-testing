"""
Deviation Magnet Forward Tester
===============================

Real-time paper trading bot for the Deviation Magnet strategy.
Refactored for reusability, modularity, and efficiency.

Usage:
    python deviation_magnet_forward.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import pandas as pd
import requests
from pybit.unified_trading import HTTP, WebSocket

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Trading configuration parameters."""

    # Indicator settings
    bb_length: int = 200
    mult: float = 3.0
    dev_mult: float = 1.5

    # Timing
    timeframe: str = field(default_factory=lambda: os.environ.get("TIMEFRAME", "1"))  # Changed to 1m
    check_interval: int = 30  # Keep for status/cleanup, but not for trading loop
    
    # Parallel processing
    max_workers: int = 10  # Increased for faster REST init

    # Position management
    base_order_size: float = field(default_factory=lambda: float(os.environ.get("BASE_ORDER_SIZE", "10.0")))
    fee_pct: float = 0.00055  # 0.055% per side (standard taker)
    profit_target_pct: float = field(default_factory=lambda: float(os.environ.get("PROFIT_TARGET", "0.1")))
    max_hold_minutes: int = field(default_factory=lambda: int(os.environ.get("MAX_HOLD_MINUTES", "120")))
    
    # DCA settings
    dca_scale: float = field(default_factory=lambda: float(os.environ.get("DCA_SCALE", "2.0")))
    max_dca_orders: int = field(default_factory=lambda: int(os.environ.get("MAX_DCA_ORDERS", "10")))

    # Memory management
    max_signals_cache: int = 2000
    signals_cleanup_keep: int = 1000
    trades_memory_cap: int = 500

    # Status display
    status_interval: int = 5

    # API retry settings
    api_retries: int = 3
    api_retry_delay: float = 1.0

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("forward_test_data"))

    @property
    def trades_file(self) -> Path:
        return self.data_dir / "trades.json"

    @property
    def trades_csv(self) -> Path:
        return self.data_dir / "trades.csv"

    @property
    def state_file(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def log_file(self) -> Path:
        return self.data_dir / "forward_test.log"


def load_symbols() -> list[str]:
    """Load symbols to monitor."""
    env_symbols = os.environ.get("TRADING_SYMBOLS")
    if env_symbols:
        return [s.strip().upper() for s in env_symbols.split(",")]
    
    return [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
        "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "POLUSDT",
        "SUIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "NEARUSDT",
        "ATOMUSDT", "LTCUSDT", "BNBUSDT", "INJUSDT", "TAOUSDT",
        "1000PEPEUSDT", "WIFUSDT", "TIAUSDT", "RENDERUSDT", "ONDOUSDT",
        "JUPUSDT", "ENAUSDT", "STXUSDT", "IMXUSDT", "SEIUSDT",
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

class FlushStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit."""
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logging(config: Config) -> logging.Logger:
    """Configure logging."""
    log = logging.getLogger("deviation_magnet")
    if log.handlers:
        return log
    
    log.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = FlushStreamHandler(sys.stdout)
    console.setFormatter(formatter)
    log.addHandler(console)

    try:
        config.data_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create file handler: {e}", flush=True)

    return log


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DCAOrder:
    """Single DCA order."""
    price: float
    size: float
    time: datetime
    bar_time: str

@dataclass
class Position:
    """Open trading position."""
    symbol: str
    direction: str
    orders: list
    entry_time: datetime
    
    # Stats
    max_runup_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    bars_held: int = 0
    last_processed_bar: Optional[str] = None

    @property
    def total_size(self) -> float:
        return sum(o["size"] for o in self.orders)

    @property
    def avg_entry_price(self) -> float:
        if not self.orders:
            return 0
        total_cost = sum(o["price"] * o["size"] for o in self.orders)
        return total_cost / self.total_size

    @property
    def num_orders(self) -> int:
        return len(self.orders)
    
    @property
    def last_bar_time(self) -> str:
        return self.orders[-1]["bar_time"] if self.orders else ""

@dataclass
class Trade:
    """Completed trade record."""
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
    position_size: float = 0.0
    num_dca_orders: int = 1

    # Stats
    max_runup_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    bars_held: int = 0

@dataclass
class BandData:
    """Indicator values."""
    close: float
    high: float
    low: float
    upper3: float
    lower3: float
    basis: float
    bar_time: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

class TradingState:
    """Manages trading state with persistence."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.signals_seen: Dict[str, str] = {}
        self.start_time: datetime = datetime.now(timezone.utc)
        self.total_signals: int = 0
        self.api_errors: int = 0
        self._lock = Lock()
        self.logger = logging.getLogger("deviation_magnet")

    def increment_errors(self) -> None:
        with self._lock:
            self.api_errors += 1

    def save(self) -> None:
        """Atomic save of state."""
        try:
            positions_data = {}
            for sym, pos in self.positions.items():
                positions_data[sym] = {
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "orders": pos.orders,
                    "entry_time": pos.entry_time.isoformat(),
                    "max_runup_pct": pos.max_runup_pct,
                    "max_drawdown_pct": pos.max_drawdown_pct,
                    "bars_held": pos.bars_held,
                    "last_processed_bar": pos.last_processed_bar,
                }

            state_data = {
                "positions": positions_data,
                "start_time": self.start_time.isoformat(),
                "total_signals": self.total_signals,
                "api_errors": self.api_errors,
            }

            temp_file = self.config.state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(state_data, f, indent=2)
            temp_file.replace(self.config.state_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def load(self) -> None:
        """Load state from disk."""
        if not self.config.state_file.exists():
            return

        try:
            with open(self.config.state_file, "r") as f:
                state_data = json.load(f)

            # LOAD POSITIONS
            for sym, data in state_data.get("positions", {}).items():
                entry_time = datetime.fromisoformat(data["entry_time"])
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                
                orders = data.get("orders", [])
                    
                self.positions[sym] = Position(
                    symbol=data["symbol"],
                    direction=data["direction"],
                    orders=orders,
                    entry_time=entry_time,
                    max_runup_pct=data.get("max_runup_pct", 0.0),
                    max_drawdown_pct=data.get("max_drawdown_pct", 0.0),
                    bars_held=data.get("bars_held", 0),
                    last_processed_bar=data.get("last_processed_bar"),
                )

            # IGNORE: start_time, total_signals, api_errors
            # We want a fresh session look
            # if state_data.get("start_time"): ...
                
            self.logger.info(f"Restored {len(self.positions)} open positions")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")

    def load_trades(self) -> None:
        """Load historical trades."""
        # For new "clean slate" logic, we can just skip loading if user wants
        # Or we can just log that we found them but start fresh.
        # But to truly ignore them, we should rename/archive the old file.
        # For now, let's keep the file but NOT load them into memory stats if a flag is set.
        # However, simpler approach requested: "ignore these".
        # Let's check an env var or just not load them if we want a fresh start.
        
        # ACTUALLY: The user asked to "ignore these". The cleanest way is to
        # archive the current trades.json if it exists on startup, or just don't load.
        # Let's modify this to load ONLY if we want persistence.
        # Since this is a forward test, maybe we just want to see NEW trades.
        
        if not self.config.trades_file.exists():
            return

        try:
            with open(self.config.trades_file, "r") as f:
                trades_data = json.load(f)

            # Cap memory usage
            recent = trades_data[-self.config.trades_memory_cap:]
            for t in recent:
                self.trades.append(Trade(**t))

            self.logger.info(f"Loaded {len(self.trades)} recent trades (total {len(trades_data)})")
        except Exception as e:
            self.logger.error(f"Failed to load trades: {e}")

    def add_trade(self, trade: Trade) -> None:
        """Add trade to memory and disk."""
        self.trades.append(trade)
        if len(self.trades) > self.config.trades_memory_cap:
            self.trades.pop(0)
        
        self.save_trade(trade)

    def save_trade(self, trade: Trade) -> None:
        """Append trade to JSON and CSV files."""
        self._save_trade_json(trade)
        self._save_trade_csv(trade)

    def _save_trade_json(self, trade: Trade) -> None:
        try:
            trades_data = []
            if self.config.trades_file.exists():
                with open(self.config.trades_file, "r") as f:
                    trades_data = json.load(f)

            trades_data.append(asdict(trade))

            temp_file = self.config.trades_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(trades_data, f, indent=2)
            temp_file.replace(self.config.trades_file)
        except Exception as e:
            self.logger.error(f"Failed to save trade JSON: {e}")

    def _save_trade_csv(self, trade: Trade) -> None:
        try:
            import csv
            file_exists = self.config.trades_csv.exists()
            
            with open(self.config.trades_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(trade).keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(asdict(trade))
                
        except Exception as e:
            self.logger.error(f"Failed to save trade CSV: {e}")

    def cleanup_signals(self) -> None:
        if len(self.signals_seen) > self.config.max_signals_cache:
            items = list(self.signals_seen.items())
            self.signals_seen = dict(items[-self.config.signals_cleanup_keep:])


class BybitClient:
    """Handles API interactions via Pybit and WebSockets."""

    def __init__(self, config: Config, state: TradingState):
        self.config = config
        self.state = state
        self.logger = logging.getLogger("deviation_magnet")
        
        # REST session for initial data
        self.session = HTTP(testnet=False)
        
        # WebSocket for real-time updates (Public Linear)
        self.ws = WebSocket(
            testnet=False,
            channel_type="linear",
        )
        self.data_buffer: Dict[str, pd.DataFrame] = {}
        self.data_lock = Lock()

    def fetch_history_rest(self, symbol: str, limit: int = 250) -> Optional[pd.DataFrame]:
        """Initial fetch of historical candles."""
        try:
            resp = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=self.config.timeframe,
                limit=limit,
            )
            
            if resp["retCode"] != 0 or not resp.get("result", {}).get("list"):
                return None

            # Pybit returns list of strings, convert to DF
            # Data format: [startTime, open, high, low, close, volume, turnover]
            # List is reversed (latest first), similar to REST logic
            raw_data = resp["result"]["list"]
            
            df = pd.DataFrame(
                raw_data,
                columns=["open_time", "open", "high", "low", "close", "volume", "turnover"],
            )
            # Pybit returns strings, convert types
            df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            
            # Sort chronological (oldest -> newest) for indicator calc
            df = df.sort_values("open_time").reset_index(drop=True)
            return df

        except Exception as e:
            self.logger.warning(f"REST fetch failed for {symbol}: {e}")
            self.state.increment_errors()
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        # Fallback to REST if needed, but we should use WS cache ideally
        try:
            resp = self.session.get_tickers(category="linear", symbol=symbol)
            if resp["retCode"] == 0 and resp["result"]["list"]:
                return float(resp["result"]["list"][0]["lastPrice"])
        except Exception:
            pass
        return None


class DeviationMagnetStrategy:
    """Core strategy logic."""

    def __init__(self, config: Config):
        self.config = config

    def calculate_bands(self, df: pd.DataFrame) -> Optional[BandData]:
        if df is None or len(df) < self.config.bb_length:
            return None

        df = df.copy()
        df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        df["basis"] = df["ohlc4"].rolling(self.config.bb_length).mean()
        df["stdev"] = df["ohlc4"].rolling(self.config.bb_length).std(ddof=0)
        df["dev"] = self.config.mult * df["stdev"]
        df["upper3"] = df["basis"] + df["dev"] * self.config.dev_mult
        df["lower3"] = df["basis"] - df["dev"] * self.config.dev_mult

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

    def check_entry(self, symbol: str, data: BandData, state: TradingState) -> Tuple[Optional[str], bool]:
        """Returns (direction, is_dca)."""
        bar_time_str = data.bar_time.isoformat() if hasattr(data.bar_time, 'isoformat') else str(data.bar_time)
        bar_key = f"{symbol}_{bar_time_str}"
        
        # 1. Check Existing Position (DCA)
        if symbol in state.positions:
            pos = state.positions[symbol]
            
            if pos.last_bar_time == bar_time_str:
                return None, False  # Already acted this bar
            
            if pos.num_orders >= self.config.max_dca_orders:
                return None, False
            
            # Get last order price
            last_order_price = pos.orders[-1]["price"]
            
            if pos.direction == "long" and data.close <= data.lower3:
                # DCA Filter: Only buy if price is lower than last entry
                if data.close < last_order_price:
                    state.total_signals += 1
                    return "long", True
            
            elif pos.direction == "short" and data.close >= data.upper3:
                # DCA Filter: Only sell if price is higher than last entry
                if data.close > last_order_price:
                    state.total_signals += 1
                    return "short", True
            
            return None, False
        
        # 2. Check New Entry
        if bar_key in state.signals_seen:
            return None, False

        if data.close <= data.lower3:
            state.signals_seen[bar_key] = "long"
            state.total_signals += 1
            return "long", False
        elif data.close >= data.upper3:
            state.signals_seen[bar_key] = "short"
            state.total_signals += 1
            return "short", False

        return None, False

    def check_exit(self, position: Position, current_price: float) -> Tuple[bool, str]:
        avg_entry = position.avg_entry_price
        
        if position.direction == "long":
            pnl_pct = (current_price - avg_entry) / avg_entry * 100
        else:
            pnl_pct = (avg_entry - current_price) / avg_entry * 100

        if pnl_pct >= self.config.profit_target_pct:
            return True, "profit_target"

        now = datetime.now(timezone.utc)
        entry = position.entry_time
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=timezone.utc)
        
        hold_minutes = (now - entry).total_seconds() / 60
        if hold_minutes >= self.config.max_hold_minutes:
            return True, "max_hold"

        return False, ""


class TradeExecutor:
    """Handles order execution, PnL math, and state updates."""

    def __init__(self, config: Config, state: TradingState):
        self.config = config
        self.state = state
        self.logger = logging.getLogger("deviation_magnet")

    def execute_entry(self, symbol: str, direction: str, price: float, bar_time: datetime, is_dca: bool):
        now = datetime.now(timezone.utc)
        bar_str = self._fmt_time(bar_time)

        if is_dca and symbol in self.state.positions:
            pos = self.state.positions[symbol]
            # Size = Current Total Exposure * Scale (Martingale Tripling with scale 2.0)
            size = pos.total_size * self.config.dca_scale
            
            # Fallback
            if size <= 0:
                size = self.config.base_order_size

            pos.orders.append({
                "price": price,
                "size": size,
                "time": now.isoformat(),
                "bar_time": bar_str
            })
            
            self.logger.info(
                f"📥 DCA #{pos.num_orders}: {symbol} {direction.upper()} @ ${price:.4f} "
                f"(+${size:.0f}, total: ${pos.total_size:.0f}) | Avg: ${pos.avg_entry_price:.4f}"
            )
        else:
            initial = {
                "price": price,
                "size": self.config.base_order_size,
                "time": now.isoformat(),
                "bar_time": bar_str
            }
            self.state.positions[symbol] = Position(
                symbol=symbol,
                direction=direction,
                orders=[initial],
                entry_time=now,
            )
            self.logger.info(f"📥 ENTRY: {symbol} {direction.upper()} @ ${price:.4f} (${self.config.base_order_size:.0f})")

        self.state.save()

    def execute_exit(self, symbol: str, pos: Position, price: float, reason: str):
        now = datetime.now(timezone.utc)
        avg = pos.avg_entry_price
        total_size = pos.total_size

        if pos.direction == "long":
            pnl_pct = (price - avg) / avg * 100
        else:
            pnl_pct = (avg - price) / avg * 100
            
        # Fees logic
        entry_fee = total_size * self.config.fee_pct
        notional_exit = total_size * (price / avg)
        exit_fee = notional_exit * self.config.fee_pct
        total_fees = entry_fee + exit_fee
        
        gross_pnl_usd = pnl_pct / 100 * total_size
        pnl_usd = gross_pnl_usd - total_fees
        
        entry_time = pos.entry_time.replace(tzinfo=timezone.utc) if pos.entry_time.tzinfo is None else pos.entry_time
        
        trade = Trade(
            symbol=symbol,
            direction=pos.direction,
            entry_price=avg,
            exit_price=price,
            entry_time=entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            exit_time=now.strftime("%Y-%m-%d %H:%M:%S"),
            pnl_pct=round(pnl_pct, 4),
            pnl_usd=round(pnl_usd, 4),
            hold_seconds=int((now - entry_time).total_seconds()),
            exit_reason=reason,
            position_size=total_size,
            num_dca_orders=pos.num_orders,
            max_runup_pct=pos.max_runup_pct,
            max_drawdown_pct=pos.max_drawdown_pct,
            bars_held=pos.bars_held
        )
        
        self.state.add_trade(trade)
        if symbol in self.state.positions:
            del self.state.positions[symbol]
        self.state.save()

        emoji = "✅" if pnl_usd > 0 else "❌"
        self.logger.info(
            f"{emoji} EXIT: {symbol} {pos.direction.upper()} @ ${price:.4f} | "
            f"PnL: ${pnl_usd:.2f} ({pnl_pct:.2f}%) | {reason}"
        )
        self.logger.info(f"   Stats: Runup: {pos.max_runup_pct:.2f}% | DD: {pos.max_drawdown_pct:.2f}% | Bars: {pos.bars_held}")

    def update_position_stats(self, symbol: str, current_price: float, data: BandData):
        if symbol not in self.state.positions:
            return
            
        pos = self.state.positions[symbol]
        avg = pos.avg_entry_price
        
        if pos.direction == "long":
            pnl_pct = (current_price - avg) / avg * 100
        else:
            pnl_pct = (avg - current_price) / avg * 100
            
        pos.max_runup_pct = max(pos.max_runup_pct, pnl_pct)
        pos.max_drawdown_pct = min(pos.max_drawdown_pct, pnl_pct)
        
        bar_str = self._fmt_time(data.bar_time)
        if pos.last_processed_bar != bar_str:
            pos.bars_held += 1
            pos.last_processed_bar = bar_str

    def _fmt_time(self, t: datetime) -> str:
        return t.isoformat() if hasattr(t, 'isoformat') else str(t)


class DataManager:
    """Manages hybrid data: REST history + WebSocket updates."""
    
    def __init__(self, client: BybitClient, symbols: List[str]):
        self.client = client
        self.symbols = symbols
        self.data: Dict[str, pd.DataFrame] = {}
        # Optimization: Separate locks per symbol could be better if contention is high,
        # but for 30 symbols, a single lock is fine IF the critical section is fast.
        self.lock = Lock()
        self.logger = logging.getLogger("deviation_magnet")
        
    def initialize(self):
        """Fetch initial history via REST in parallel."""
        self.logger.info("Fetching initial history for indicators...")
        with ThreadPoolExecutor(max_workers=self.client.config.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.client.fetch_history_rest, sym): sym 
                for sym in self.symbols
            }
            for future in as_completed(future_to_symbol):
                sym = future_to_symbol[future]
                df = future.result()
                if df is not None:
                    with self.lock:
                        self.data[sym] = df
        
        count = len(self.data)
        self.logger.info(f"Initialized history for {count}/{len(self.symbols)} symbols")

    def on_kline_update(self, message: dict):
        """Handle WS kline update efficiently."""
        if "data" not in message:
            return

        # Extract symbol from topic (e.g. "kline.1.BTCUSDT")
        topic = message.get("topic", "")
        parts = topic.split(".")
        if len(parts) < 3:
            return
        symbol = parts[2]

        for kline in message["data"]:
            # Convert timestamp once
            new_ts = pd.to_datetime(int(kline["start"]), unit="ms")
            
            with self.lock:
                if symbol not in self.data:
                    continue
                
                df = self.data[symbol]
                last_ts = df.iloc[-1]["open_time"]
                
                # OPTIMIZATION: Avoid creating full Series/Dict for update if possible
                # But treating as dict is readable and safe.
                
                if new_ts == last_ts:
                    # Update last row in-place (Fastest)
                    # We map API keys to DF columns
                    df.at[df.index[-1], "close"] = float(kline["close"])
                    df.at[df.index[-1], "high"] = float(kline["high"])
                    df.at[df.index[-1], "low"] = float(kline["low"])
                    df.at[df.index[-1], "open"] = float(kline["open"])
                    df.at[df.index[-1], "volume"] = float(kline["volume"])
                    
                elif new_ts > last_ts:
                    # Append new row
                    # Creating a 1-row DataFrame is standard for appending
                    new_row = {
                        "open_time": new_ts,
                        "open": float(kline["open"]),
                        "high": float(kline["high"]),
                        "low": float(kline["low"]),
                        "close": float(kline["close"]),
                        "volume": float(kline["volume"]),
                        "turnover": float(kline["turnover"])
                    }
                    new_df = pd.DataFrame([new_row])
                    
                    # Concat is expensive, but necessary for expanding the DF.
                    # Optimization: Limit history length to prevent unbounded growth slows concat.
                    self.data[symbol] = pd.concat([df, new_df], ignore_index=True)
                    
                    # Truncate if too large (keep 300, need 200 for BB)
                    if len(self.data[symbol]) > 300:
                        self.data[symbol] = self.data[symbol].iloc[-300:].reset_index(drop=True)

    def get_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        with self.lock:
            # Return a copy to prevent concurrent modification issues during calculation
            # Copy is O(N) but safer. For 300 rows it's very fast.
            return self.data.get(symbol).copy() if symbol in self.data else None


# ═══════════════════════════════════════════════════════════════════════════════
# BOT CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class Bot:
    """Main controller."""

    def __init__(self):
        self.config = Config()
        self.logger = setup_logging(self.config)
        self.state = TradingState(self.config)
        self.client = BybitClient(self.config, self.state)
        self.strategy = DeviationMagnetStrategy(self.config)
        self.symbols = load_symbols()
        self.data_manager = DataManager(self.client, self.symbols)

    def run(self):
        """Start the bot loop."""
        self.state.load()
        # self.state.load_trades()
        self._print_header()

        self.logger.info("🔄 Starting...")
        
        # 1. Initialize Data
        self.data_manager.initialize()
        
        # 2. Subscribe to WebSockets
        self.logger.info("🔌 Connecting to WebSockets...")
        
        def handle_ws_message(message):
            try:
                self.data_manager.on_kline_update(message)
                # Trigger processing for this symbol immediately
                topic = message.get("topic", "")
                if "kline" in topic:
                    symbol = topic.split(".")[-1]
                    self._process_symbol_event(symbol)
            except Exception as e:
                self.logger.error(f"WS Handler Error: {e}")

        # Subscribe to all symbols
        # Note: Pybit handles the connection loop
        for symbol in self.symbols:
            self.client.ws.kline_stream(
                interval=int(self.config.timeframe),
                symbol=symbol,
                callback=handle_ws_message
            )

        print("⚡ Real-time processing started. Press Ctrl+C to stop.", flush=True)

        try:
            while True:
                # Keep main thread alive and print status
                time.sleep(self.config.check_interval)
                self._print_status()
                self.state.cleanup_signals()
                
                # Daily report
                last_daily = datetime.now(timezone.utc).date()  # Simplified check
                # (Add robust daily check if needed)

        except KeyboardInterrupt:
            self.logger.info("Stopping... saving state")
            self.state.save()
            self._print_status()
            print("\n✅ State saved.\n", flush=True)
        except Exception as e:
            self.logger.error(f"💥 FATAL ERROR: {e}")
            self.logger.error(traceback.format_exc())
            self.state.save()
            raise

    def _process_symbol_event(self, symbol: str):
        """Process strategy for a single symbol upon data update."""
        df = self.data_manager.get_latest_data(symbol)
        if df is None:
            return

        data = self.strategy.calculate_bands(df)
        if not data:
            return
        
        current_price = data.close
        
        # Check Exit
        if symbol in self.state.positions:
            pos = self.state.positions[symbol]
            self._update_position_stats(symbol, current_price, data)
            
            should_exit, reason = self.strategy.check_exit(pos, current_price)
            if should_exit:
                self._execute_exit(symbol, pos, current_price, reason)
                # Prevent immediate re-entry
                bar_key = f"{symbol}_{self._fmt_time(data.bar_time)}"
                self.state.signals_seen[bar_key] = "exit"
                return

        # Check Entry
        direction, is_dca = self.strategy.check_entry(symbol, data, self.state)
        if direction:
            self._execute_entry(symbol, direction, current_price, data.bar_time, is_dca)

    # ... (rest of methods like _fetch_all_data removed as they are replaced by DataManager)
    # Keeping helper methods
    
    def _update_position_stats(self, symbol: str, current_price: float, data: BandData):
        # ... existing implementation ...
        pos = self.state.positions[symbol]
        avg = pos.avg_entry_price
        
        if pos.direction == "long":
            pnl_pct = (current_price - avg) / avg * 100
        else:
            pnl_pct = (avg - current_price) / avg * 100
            
        pos.max_runup_pct = max(pos.max_runup_pct, pnl_pct)
        pos.max_drawdown_pct = min(pos.max_drawdown_pct, pnl_pct)
        
        bar_str = self._fmt_time(data.bar_time)
        if pos.last_processed_bar != bar_str:
            pos.bars_held += 1
            pos.last_processed_bar = bar_str

    def _execute_entry(self, symbol: str, direction: str, price: float, bar_time: datetime, is_dca: bool):
        # ... existing implementation ...
        now = datetime.now(timezone.utc)
        bar_str = self._fmt_time(bar_time)

        if is_dca and symbol in self.state.positions:
            pos = self.state.positions[symbol]
            size = pos.total_size * self.config.dca_scale
            if size <= 0:
                size = self.config.base_order_size

            pos.orders.append({
                "price": price,
                "size": size,
                "time": now.isoformat(),
                "bar_time": bar_str
            })
            
            self.logger.info(
                f"📥 DCA #{pos.num_orders}: {symbol} {direction.upper()} @ ${price:.4f} "
                f"(+${size:.0f}, total: ${pos.total_size:.0f}) | Avg: ${pos.avg_entry_price:.4f}"
            )
        else:
            initial = {
                "price": price,
                "size": self.config.base_order_size,
                "time": now.isoformat(),
                "bar_time": bar_str
            }
            self.state.positions[symbol] = Position(
                symbol=symbol,
                direction=direction,
                orders=[initial],
                entry_time=now,
            )
            self.logger.info(f"📥 ENTRY: {symbol} {direction.upper()} @ ${price:.4f} (${self.config.base_order_size:.0f})")

        self.state.save()

    def _execute_exit(self, symbol: str, pos: Position, price: float, reason: str):
        # ... existing implementation ...
        now = datetime.now(timezone.utc)
        avg = pos.avg_entry_price
        total_size = pos.total_size

        if pos.direction == "long":
            pnl_pct = (price - avg) / avg * 100
        else:
            pnl_pct = (avg - price) / avg * 100
            
        entry_fee = total_size * self.config.fee_pct
        notional_exit = total_size * (price / avg)
        exit_fee = notional_exit * self.config.fee_pct
        total_fees = entry_fee + exit_fee
        
        gross_pnl_usd = pnl_pct / 100 * total_size
        pnl_usd = gross_pnl_usd - total_fees
        
        entry_time = pos.entry_time.replace(tzinfo=timezone.utc) if pos.entry_time.tzinfo is None else pos.entry_time
        
        trade = Trade(
            symbol=symbol,
            direction=pos.direction,
            entry_price=avg,
            exit_price=price,
            entry_time=entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            exit_time=now.strftime("%Y-%m-%d %H:%M:%S"),
            pnl_pct=round(pnl_pct, 4),
            pnl_usd=round(pnl_usd, 4),
            hold_seconds=int((now - entry_time).total_seconds()),
            exit_reason=reason,
            position_size=total_size,
            num_dca_orders=pos.num_orders,
            max_runup_pct=pos.max_runup_pct,
            max_drawdown_pct=pos.max_drawdown_pct,
            bars_held=pos.bars_held
        )
        
        self.state.add_trade(trade)
        del self.state.positions[symbol]
        self.state.save()

        emoji = "✅" if pnl_usd > 0 else "❌"
        self.logger.info(
            f"{emoji} EXIT: {symbol} {pos.direction.upper()} @ ${price:.4f} | "
            f"PnL: ${pnl_usd:.2f} ({pnl_pct:.2f}%) | {reason}"
        )
        self.logger.info(f"   Stats: Runup: {pos.max_runup_pct:.2f}% | DD: {pos.max_drawdown_pct:.2f}% | Bars: {pos.bars_held}")

    def _daily_report(self, date):
        date_str = str(date)
        trades = [t for t in self.state.trades if t.exit_time.startswith(date_str)]
        if trades:
            total = sum(t.pnl_usd for t in trades)
            self.logger.info(f"📊 {date_str}: {len(trades)} trades | PnL: ${total:.2f}")

    def _fmt_time(self, t: datetime) -> str:
        return t.isoformat() if hasattr(t, 'isoformat') else str(t)

    def _print_header(self):
        c = self.config
        print(f"\n{'═' * 70}")
        print(f"  🚀 DEVIATION MAGNET - 24/7 FORWARD TEST (WebSocket Mode)")
        print(f"{'═' * 70}")
        print(f"  Timeframe:      {c.timeframe}m (Real-time Stream)")
        print(f"  Symbols:        {len(self.symbols)}")
        print(f"  DCA Config:     Base ${c.base_order_size} | Scale {c.dca_scale}x (Target Exposure Doubling)")
        print(f"  Target Profit:  {c.profit_target_pct}%")
        print(f"  Outputs:        {c.trades_file.name}, {c.trades_csv.name}")
        print(f"{'─' * 70}")
        print(f"  DCA Progression Preview (Martingale):")
        current_exposure = c.base_order_size
        print(f"    Base:    ${c.base_order_size:,.0f} (Total Exposure: ${current_exposure:,.0f})")
        for i in range(min(5, c.max_dca_orders)):
            next_size = current_exposure * c.dca_scale
            current_exposure += next_size
            print(f"    DCA #{i+1}:  ${next_size:,.0f} (Total Exposure: ${current_exposure:,.0f})")
        if c.max_dca_orders > 5:
            print(f"    ... up to {c.max_dca_orders} orders")
        print(f"{'═' * 70}\n", flush=True)

    def _print_status(self):
        # Simplified status
        print(f"\n{'─' * 60}")
        print(f"  ⏱  Active Positions: {len(self.state.positions)}")
        for sym, pos in self.state.positions.items():
            # Get cached price from DataManager
            df = self.data_manager.get_latest_data(sym)
            if df is not None:
                current = df.iloc[-1]["close"]
                avg = pos.avg_entry_price
                pnl = (current - avg)/avg*100 if pos.direction == "long" else (avg - current)/avg*100
                print(f"      {sym:<10} {pos.direction:<5} {pnl:>+6.2f}% (Hold: {pos.bars_held} bars)")
        
        if self.state.trades:
            total_pnl = sum(t.pnl_usd for t in self.state.trades)
            print(f"  📈 Total PnL: ${total_pnl:.2f} ({len(self.state.trades)} trades)")
        print(f"{'─' * 60}\n", flush=True)


if __name__ == "__main__":
    try:
        bot = Bot()
        bot.run()
    except Exception as e:
        print(f"💥 Critical Failure: {e}")
        traceback.print_exc()
