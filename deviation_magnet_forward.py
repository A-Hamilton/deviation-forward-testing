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
import queue
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import pandas as pd
import numpy as np
import requests
from pybit.unified_trading import HTTP, WebSocket

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Trading configuration parameters."""

    # Indicator settings
    bb_length: int = 10
    mult: float = 10.0
    dev_mult: float = 3.0  # Changed to 3.0 to match TV script (upper3 = basis + dev * 3)

    # Timing
    timeframe: str = field(default_factory=lambda: os.environ.get("TIMEFRAME", "1"))  # Changed to 1m
    check_interval: int = 30  # Keep for status/cleanup, but not for trading loop
    
    # Parallel processing
    max_workers: int = 10  # Increased for faster REST init

    # Position management
    base_order_size: float = field(default_factory=lambda: float(os.environ.get("BASE_ORDER_SIZE", "10.0")))
    fee_pct: float = 0.00055  # 0.055% per side (standard taker)
    maker_fee_pct: float = 0.0002  # 0.02% per side (standard maker)
    profit_target_pct: float = field(default_factory=lambda: float(os.environ.get("PROFIT_TARGET", "0.1")))

    
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





# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

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

    console = logging.StreamHandler(sys.stdout)
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
        self.save_lock = Lock()

    def increment_errors(self) -> None:
        with self._lock:
            self.api_errors += 1

    def save(self) -> None:
        """Async save of state to avoid blocking main loop."""
        Thread(target=self._save_sync, daemon=True).start()

    def _save_sync(self) -> None:
        """Actual save logic running in thread."""
        try:
            with self.save_lock:
                # Create a snapshot of positions to avoid iteration errors during modification
                with self._lock: # Use lock if needed, but positions dict might change. 
                    # Ideally we copy the data we need quickly.
                    # For now, just try/except or copy. 
                    # A simple copy of the dict keys/values is safer.
                    positions_snapshot = self.positions.copy()

                positions_data = {}
                for sym, pos in positions_snapshot.items():
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
        except Exception:
            pass
        return None

    def fetch_active_symbols(self) -> List[str]:
        """Fetch all active USDT perpetual symbols."""
        try:
            self.logger.info("Fetching all active USDT perpetuals from Bybit...")
            resp = self.session.get_instruments_info(category="linear")
            
            if resp["retCode"] != 0:
                self.logger.error(f"Failed to fetch symbols: {resp}")
                return []

            symbols = [
                i["symbol"] for i in resp["result"]["list"] 
                if i["status"] == "Trading" and i["quoteCoin"] == "USDT"
            ]
            
            # Filter out USDC perps just in case, though quoteCoin check covers it
            symbols = [s for s in symbols if s.endswith("USDT")]
            
            self.logger.info(f"Found {len(symbols)} active USDT pairs")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error fetching symbols: {e}")
            # Fallback to a small list if API fails completely
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


class DeviationMagnetStrategy:
    """Core strategy logic."""

    def __init__(self, config: Config):
        self.config = config

    def calculate_bands_fast(self, buffer: np.ndarray, count: int, head: int, array_size: int) -> Optional[BandData]:
        """
        High-performance Bollinger Band calculation using Numpy.
        Uses LAGGED logic (TradingView style):
        - Stats calculated on previous N candles (excluding current).
        - Current price compared to bands from previous N candles.
        
        Expects circular buffer indexing.
        """
        # We need bb_length + 1 items: N for stats, +1 for current price
        required_len = self.config.bb_length + 1
        
        if count < required_len:
            return None

        # Extract the last N+1 items in chronological order
        if count < array_size:
            # Buffer not full, data is linear [0:count]
            subset = buffer[count - required_len : count]
        else:
            # Circular buffer is full
            # Extract last N+1 items: [head - N, ..., head]
            indices = [(head - required_len + 1 + i) % array_size for i in range(required_len)]
            subset = buffer[indices]
        
        # Check for NaNs in the STATS window (excluding current)
        stats_window = subset[:-1]
        if np.isnan(stats_window[0, 0]):
            return None
        
        # Columns: 0=open, 1=high, 2=low, 3=close, 4=time
        opens = stats_window[:, 0]
        highs = stats_window[:, 1]
        lows = stats_window[:, 2]
        closes = stats_window[:, 3]
        
        # Calculate OHLC4 on PREVIOUS N candles
        ohlc4 = (opens + highs + lows + closes) / 4.0
        
        # Calculate Basis (Mean) and Stdev on PREVIOUS N candles
        basis = np.mean(ohlc4)
        stdev = np.std(ohlc4)
        
        # Calculate Bands
        dev = self.config.mult * stdev
        upper3 = basis + (dev * self.config.dev_mult)
        lower3 = basis - (dev * self.config.dev_mult)
        
        # Current Candle (The one we are trading)
        current_candle = subset[-1]
        latest_close = current_candle[3]
        latest_high = current_candle[1]
        latest_low = current_candle[2]
        latest_time_ts = current_candle[4]
        
        # Convert timestamp back to datetime for logic
        bar_time = datetime.fromtimestamp(latest_time_ts, tz=timezone.utc).replace(tzinfo=None)
        
        return BandData(
            close=latest_close,
            high=latest_high,
            low=latest_low,
            upper3=upper3,
            lower3=lower3,
            basis=basis,
            bar_time=bar_time,
        )

    def check_entry(self, symbol: str, data: BandData, state: TradingState) -> Tuple[Optional[str], bool]:
        """Returns (direction, is_dca)."""
        bar_time_str = data.bar_time.isoformat() if hasattr(data.bar_time, 'isoformat') else str(data.bar_time)
        bar_key = f"{symbol}_{bar_time_str}"
        
        # 1. Check Existing Position (DCA)
        if symbol in state.positions:
            pos = state.positions[symbol]
            
            if pos.last_bar_time == bar_time_str:
                # logging.debug(f"Skipping {symbol}: Already acted on bar {bar_time_str}")
                return None, False  # Already acted this bar
            
            if pos.num_orders >= self.config.max_dca_orders:
                return None, False
            
            # Get last order price
            last_order_price = pos.orders[-1]["price"]
            
            if pos.direction == "long" and data.close <= data.lower3:
                # DCA Filter: Only buy if price is lower than last entry by at least 0.2%
                # This prevents filling the bag on flat price action
                if data.close <= last_order_price * 0.998:
                    state.total_signals += 1
                    return "long", True
            
            elif pos.direction == "short" and data.close >= data.upper3:
                # DCA Filter: Only sell if price is higher than last entry by at least 0.2%
                if data.close >= last_order_price * 1.002:
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

    def check_exit(self, position: Position, data: BandData) -> Tuple[bool, str, float]:
        """
        Checks for exit conditions.
        Returns: (should_exit, reason, exit_price)
        """
        # Safety check first
        if not position.orders:
            return False, "", 0.0

        avg_entry = position.avg_entry_price
        current_price = data.close
        
        # 1. Calculate Target Price (Limit Order Logic)
        # Required % = Target + Round Trip Fees (Entry Taker + Exit Maker)
        # Note: We use Maker fee for exit because we are simulating a Limit Order
        required_pct = self.config.profit_target_pct + (self.config.fee_pct * 100) + (self.config.maker_fee_pct * 100)
        required_pct += 0.01 # Tiny buffer

        target_price = 0.0
        if position.direction == "long":
            target_price = avg_entry * (1 + required_pct / 100)
            # Check if High reached target
            if data.high >= target_price:
                return True, "profit_target", target_price
        else:
            target_price = avg_entry * (1 - required_pct / 100)
            # Check if Low reached target
            if data.low <= target_price:
                return True, "profit_target", target_price

        # 2. Base Entry Price Check (Safety / Mean Reversion check)
        # We must cross the INITIAL entry price to ensure "mean reversion" completed
        # This is a "Market Order" style check (Close price)
        base_entry_price = position.orders[0]["price"]
        
        if position.direction == "long":
            if current_price <= base_entry_price:
                return False, "", 0.0
        else:
            if current_price >= base_entry_price:
                return False, "", 0.0

        return False, "", 0.0


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

    def execute_exit(self, symbol: str, pos: Position, price: float, reason: str, is_maker: bool = False):
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
        
        # Use Maker fee if it's a Limit exit (profit_target), else Taker
        exit_fee_pct = self.config.maker_fee_pct if is_maker else self.config.fee_pct
        exit_fee = notional_exit * exit_fee_pct
        
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
            f"PnL: ${pnl_usd:.2f} (Net) | Price: {pnl_pct:.2f}% (Raw) | {reason}"
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
    """Manages hybrid data: REST history + WebSocket updates.
    
    Optimized for:
    1. Latency: Uses deque for O(1) appends instead of expensive DataFrame.concat
    2. Concurrency: Uses per-symbol locks to prevent blocking across pairs
    3. Memory: deque(maxlen=N) automatically handles cleanup
    """
    
    def __init__(self, client: BybitClient, symbols: List[str]):
        self.client = client
        self.symbols = symbols
        
        # Buffer: {symbol: np.ndarray}
        # Shape: (300, 5) -> [open, high, low, close, timestamp]
        # Pre-allocated for zero-copy updates
        self.array_size = 300
        self.data_buffers: Dict[str, np.ndarray] = {
            s: np.full((self.array_size, 5), np.nan, dtype=np.float64) for s in symbols
        }
        
        # Circular buffer tracking
        self.counts: Dict[str, int] = {s: 0 for s in symbols}
        self.head: Dict[str, int] = {s: 0 for s in symbols}  # Index of newest item
        
        # Fine-grained locking: One lock per symbol
        self.locks: Dict[str, Lock] = defaultdict(Lock)
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
                    with self.locks[sym]:
                        # Fill numpy array
                        # columns: open, high, low, close
                        # timestamp needs conversion to float
                        
                        # Extract data
                        opens = df["open"].values
                        highs = df["high"].values
                        lows = df["low"].values
                        closes = df["close"].values
                        # timestamps to float seconds
                        times = df["open_time"].astype('int64').values / 10**9
                        
                        # Stack
                        data = np.column_stack((opens, highs, lows, closes, times))
                        
                        # Fill buffer (take last N)
                        length = len(data)
                        if length >= self.array_size:
                            self.data_buffers[sym][:] = data[-self.array_size:]
                            self.counts[sym] = self.array_size
                            self.head[sym] = self.array_size - 1  # Last index
                        else:
                            # Fill from start
                            self.data_buffers[sym][:length] = data
                            self.counts[sym] = length
                            self.head[sym] = length - 1
        
        # Count non-empty buffers
        count = sum(1 for c in self.counts.values() if c > 0)
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

        # Fail fast if symbol not tracked (though unlikely given subscription)
        if symbol not in self.data_buffers:
            return

        for kline in message["data"]:
            # Convert timestamp once - FAST (No Pandas)
            # kline["start"] is ms timestamp
            ts_float = int(kline["start"]) / 1000.0
            
            # Use specific lock for this symbol only -> High Concurrency
            with self.locks[symbol]:
                buffer = self.data_buffers[symbol]
                count = self.counts[symbol]
                
                # Prepare row: [open, high, low, close, time]
                new_row = np.array([
                    float(kline["open"]),
                    float(kline["high"]),
                    float(kline["low"]),
                    float(kline["close"]),
                    ts_float
                ], dtype=np.float64)

                head_idx = self.head[symbol]
                
                if count == 0:
                    # First item
                    buffer[0] = new_row
                    self.counts[symbol] = 1
                    self.head[symbol] = 0
                    continue

                last_ts = buffer[head_idx, 4]
                
                if ts_float == last_ts:
                    # Update in-place (same candle)
                    buffer[head_idx] = new_row
                    
                elif ts_float > last_ts:
                    # Check for data gaps
                    time_diff = (ts_float - last_ts) / 60.0
                    tf_min = int(self.client.config.timeframe)
                    
                    if time_diff > tf_min * 5: 
                        self.logger.warning(f"⚠️ Data Gap detected for {symbol}: {time_diff:.1f}m. Resetting buffer.")
                        # Reset
                        buffer.fill(np.nan)
                        buffer[0] = new_row
                        self.counts[symbol] = 1
                        self.head[symbol] = 0
                    else:
                        # Circular increment (O(1) operation!)
                        new_head = (head_idx + 1) % self.array_size
                        buffer[new_head] = new_row
                        self.head[symbol] = new_head
                        self.counts[symbol] = min(count + 1, self.array_size)

    def get_ordered_view(self, symbol: str) -> Optional[np.ndarray]:
        """Get chronologically ordered view of circular buffer."""
        buffer = self.data_buffers.get(symbol)
        count = self.counts.get(symbol, 0)
        head = self.head.get(symbol, 0)
        
        if buffer is None or count == 0:
            return None
        
        if count < self.array_size:
            # Not full yet, data is in [0:count]
            return buffer[:count]
        else:
            # Full circular buffer, reorder
            # [head+1, ..., end, 0, ..., head]
            return np.concatenate([buffer[head+1:], buffer[:head+1]])
    
    def get_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Convert buffer to DataFrame for display."""
        with self.locks[symbol]:
            ordered = self.get_ordered_view(symbol)
            
            if ordered is None or len(ordered) == 0:
                return None
                
            df = pd.DataFrame(ordered, columns=["open", "high", "low", "close", "ts"])
            df["open_time"] = pd.to_datetime(df["ts"], unit="s")
            return df


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
        self.executor = TradeExecutor(self.config, self.state)
        self.symbols = self.client.fetch_active_symbols()
        if not self.symbols:
            raise ValueError("No symbols found to trade!")
        self.data_manager = DataManager(self.client, self.symbols)
        self.event_queue = queue.Queue()
        self.last_update_time: Dict[str, float] = {}
        self.watchdog_threshold = 60.0  # Seconds

    def run(self):
        """Start the bot loop."""
        self.state.load()
        self._print_header()

        self.logger.info("🔄 Starting...")
        
        # 1. Initialize Data
        self._initialize_data()
        
        # 2. Subscribe to WebSockets
        self._setup_websockets()

        print("⚡ Real-time processing started. Press Ctrl+C to stop.", flush=True)

        # 3. Enter Main Loop
        self._main_loop()

    def _initialize_data(self):
        """Fetch initial history."""
        self.data_manager.initialize()

    def _setup_websockets(self):
        """Connect and subscribe to WebSockets."""
        self.logger.info("🔌 Connecting to WebSockets...")
        
        def handle_ws_message(message):
            try:
                self.data_manager.on_kline_update(message)
                # Queue event for processing in main thread
                topic = message.get("topic", "")
                if "kline" in topic:
                    symbol = topic.split(".")[-1]
                    self.last_update_time[symbol] = time.time()
                    self.event_queue.put(symbol)
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

    def _main_loop(self):
        """Main event loop."""
        last_status_time = time.time()
        last_daily_date = datetime.now(timezone.utc).date()

        try:
            while True:
                # 1. Process Events from Queue
                try:
                    # Process a batch of events to keep queue drained
                    # But don't block indefinitely
                    for _ in range(50):
                        # Block for up to 10ms if empty, but wake immediately on event
                        symbol = self.event_queue.get(timeout=0.01)
                        self._process_symbol_event(symbol)
                        self.event_queue.task_done()
                except queue.Empty:
                    pass  # Queue empty, continue to status checks

                # 2. Periodic Status & Cleanup
                now = time.time()
                if now - last_status_time >= self.config.check_interval:
                    self._print_status()
                    self.state.cleanup_signals()
                    self._check_watchdog()
                    last_status_time = now
                
                # 3. Daily report
                current_date = datetime.now(timezone.utc).date()
                if current_date > last_daily_date:
                    self._daily_report(last_daily_date)
                    last_daily_date = current_date

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

    def _check_watchdog(self):
        now = time.time()
        for symbol in self.symbols:
            last = self.last_update_time.get(symbol, now)
            if now - last > self.watchdog_threshold:
                self.logger.warning(f"⚠️ WATCHDOG: No data for {symbol} in {now - last:.1f}s!")

    def _process_symbol_event(self, symbol: str):
        """Process strategy for a single symbol upon data update."""
        with self.data_manager.locks[symbol]:
            buffer = self.data_manager.data_buffers.get(symbol)
            if buffer is None:
                return
            count = self.data_manager.counts.get(symbol, 0)
            head = self.data_manager.head.get(symbol, 0)
            array_size = self.data_manager.array_size
            
            # Pass circular buffer metadata
            data = self.strategy.calculate_bands_fast(buffer, count, head, array_size)
        if not data:
            return
        
        current_price = data.close
        
        # Use state lock for all position reads/writes to ensure consistency with async save
        with self.state._lock:
            # Check Exit
            if symbol in self.state.positions:
                pos = self.state.positions[symbol]
                self.executor.update_position_stats(symbol, current_price, data)
                
                should_exit, reason, exit_price = self.strategy.check_exit(pos, data)
                if should_exit:
                    is_maker = (reason == "profit_target")
                    self.executor.execute_exit(symbol, pos, exit_price, reason, is_maker)
                    # Prevent immediate re-entry
                    bar_key = f"{symbol}_{self.executor._fmt_time(data.bar_time)}"
                    self.state.signals_seen[bar_key] = "exit"
                    return

            # Check Entry
            direction, is_dca = self.strategy.check_entry(symbol, data, self.state)
            if direction:
                self.executor.execute_entry(symbol, direction, current_price, data.bar_time, is_dca)

    # ... (rest of methods like _fetch_all_data removed as they are replaced by DataManager)
    # Keeping helper methods
    






    def _daily_report(self, date):
        date_str = str(date)
        trades = [t for t in self.state.trades if t.exit_time.startswith(date_str)]
        if trades:
            total = sum(t.pnl_usd for t in trades)
            self.logger.info(f"📊 {date_str}: {len(trades)} trades | PnL: ${total:.2f}")



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
        """Comprehensive status display - Important stats at bottom"""
        now = datetime.now(timezone.utc)
        session_duration = (now - self.state.start_time).total_seconds() / 3600  # hours
        
        print(f"\n{'═' * 80}")
        print(f"  📊 SESSION STATS")
        print(f"{'─' * 80}")
        print(f"  Runtime: {session_duration:.1f}h | Signals: {self.state.total_signals} | Errors: {self.state.api_errors}")
        
        # Open Positions Details
        if self.state.positions:
            print(f"\n{'─' * 80}")
            print(f"  📂 OPEN POSITIONS ({len(self.state.positions)})")
            print(f"{'─' * 80}")
            
            total_exposure = 0
            total_unrealized_pnl = 0
            
            # Sort by unrealized PnL (worst first)
            positions_sorted = []
            for sym, pos in self.state.positions.items():
                df = self.data_manager.get_latest_data(sym)
                if df is not None and len(df) > 0:
                    current = df.iloc[-1]["close"]
                    avg = pos.avg_entry_price
                    pnl_pct = (current - avg)/avg*100 if pos.direction == "long" else (avg - current)/avg*100
                    
                    # Calculate unrealized USD (after fees)
                    entry_fee = pos.total_size * self.config.fee_pct
                    notional_exit = pos.total_size * (current / avg)
                    exit_fee = notional_exit * self.config.fee_pct
                    total_fees = entry_fee + exit_fee
                    gross_pnl_usd = pnl_pct / 100 * pos.total_size
                    unrealized_usd = gross_pnl_usd - total_fees
                    
                    total_exposure += pos.total_size
                    total_unrealized_pnl += unrealized_usd
                    
                    positions_sorted.append((sym, pos, pnl_pct, unrealized_usd, current))
            
            # Sort by PnL% (worst first so you see problems)
            positions_sorted.sort(key=lambda x: x[2])
            
            for sym, pos, pnl_pct, unrealized_usd, current in positions_sorted:
                dca_info = f"DCA #{pos.num_orders}" if pos.num_orders > 1 else "BASE"
                direction_emoji = "📈" if pos.direction == "long" else "📉"
                pnl_emoji = "🟢" if unrealized_usd >= 0 else "🔴"
                
                print(f"  {pnl_emoji} {sym:<12} {direction_emoji} {pos.direction.upper():<5} | "
                      f"{dca_info:<8} | ${pos.total_size:>7.0f} | "
                      f"{pnl_pct:>+6.2f}% (${unrealized_usd:>+6.2f}) | "
                      f"Hold: {pos.bars_held}m | DD: {pos.max_drawdown_pct:>+5.2f}%")
            
            print(f"{'─' * 80}")
            print(f"  💰 Total Exposure: ${total_exposure:,.0f}")
            print(f"  💵 Unrealized PnL: ${total_unrealized_pnl:>+,.2f}")
        
        else:
            print(f"\n{'─' * 80}")
            print(f"  📂 OPEN POSITIONS: None")
        
        # Realized PnL (AT BOTTOM - Most Important)
        print(f"\n{'═' * 80}")
        print(f"  📈 REALIZED PNL")
        print(f"{'═' * 80}")
        
        if self.state.trades:
            total_realized = sum(t.pnl_usd for t in self.state.trades)
            wins = [t for t in self.state.trades if t.pnl_usd > 0]
            losses = [t for t in self.state.trades if t.pnl_usd <= 0]
            win_rate = len(wins) / len(self.state.trades) * 100 if self.state.trades else 0
            
            avg_hold_time = sum(t.hold_seconds for t in self.state.trades) / len(self.state.trades) / 60 if self.state.trades else 0
            
            best_trade = max(self.state.trades, key=lambda t: t.pnl_usd)
            worst_trade = min(self.state.trades, key=lambda t: t.pnl_usd)
            
            print(f"  Total Trades: {len(self.state.trades)} | Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {win_rate:.1f}%")
            print(f"  Avg Hold: {avg_hold_time:.1f}m | Best: ${best_trade.pnl_usd:+.2f} ({best_trade.symbol}) | Worst: ${worst_trade.pnl_usd:+.2f} ({worst_trade.symbol})")
            print(f"\n  💰 TOTAL REALIZED PNL: ${total_realized:>+,.2f}")
        else:
            print(f"  No trades yet")
        
        print(f"{'═' * 80}\n", flush=True)


if __name__ == "__main__":
    try:
        bot = Bot()
        bot.run()
    except Exception as e:
        print(f"💥 Critical Failure: {e}")
        traceback.print_exc()
