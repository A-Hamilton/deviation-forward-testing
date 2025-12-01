"""
Deviation Magnet Forward Tester
===============================

Real-time paper trading bot for the Deviation Magnet strategy.
Refactored for reusability, modularity, and efficiency.

Usage:
    python deviation_magnet_forward.py
"""

from __future__ import annotations

import csv
import logging
import os
import queue
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from operator import itemgetter
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP, WebSocket

# Use orjson if available (10x faster JSON parsing)
try:
    import orjson
    def json_loads(s): return orjson.loads(s)
    def json_dumps(obj, indent=None): 
        opts = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=opts).decode()
except ImportError:
    import json as _json
    json_loads = _json.loads
    def json_dumps(obj, indent=None): return _json.dumps(obj, indent=indent)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Trading configuration parameters."""

    # Indicator settings
    bb_length: int = 20
    mult: float = 3.0  # Total: 3.0 * 1.5 = 4.5 sigma
    dev_mult: float = 1.5

    # Timing
    timeframe: str = field(default_factory=lambda: os.environ.get("TIMEFRAME", "1"))  # Changed to 1m
    check_interval: int = 30  # Keep for status/cleanup, but not for trading loop
    
    # Parallel processing
    max_workers: int = 10  # Increased for faster REST init

    # Position management
    base_order_size: float = field(default_factory=lambda: float(os.environ.get("BASE_ORDER_SIZE", "10.0")))
    fee_pct: float = 0.00055  # 0.055% per side (standard taker)
    maker_fee_pct: float = 0.0002  # 0.02% per side (standard maker)
    slippage_pct: float = 0.0005  # 0.05% estimated slippage per side
    profit_target_pct: float = field(default_factory=lambda: float(os.environ.get("PROFIT_TARGET", "0.1")))
    
    # Volatility filter
    min_volatility_pct: float = 0.2  # Minimum avg candle size (high-low as % of close)
    
    # DCA settings
    dca_scale: float = field(default_factory=lambda: float(os.environ.get("DCA_SCALE", "5.0")))
    max_dca_orders: int = field(default_factory=lambda: int(os.environ.get("MAX_DCA_ORDERS", "2")))

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

    # Pre-computed fee combinations (computed once via __post_init__)
    entry_cost_pct: float = field(init=False)
    exit_taker_cost_pct: float = field(init=False)
    exit_maker_cost_pct: float = field(init=False)
    total_fee_pct_for_tp: float = field(init=False)  # Used in check_exit

    def __post_init__(self):
        # Pre-compute fee combinations to avoid repeated arithmetic
        self.entry_cost_pct = self.fee_pct + self.slippage_pct
        self.exit_taker_cost_pct = self.fee_pct + self.slippage_pct
        self.exit_maker_cost_pct = self.maker_fee_pct + self.slippage_pct
        self.total_fee_pct_for_tp = (self.fee_pct + self.maker_fee_pct + self.slippage_pct * 2) * 100

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

class Position:
    """Open trading position with __slots__ for memory efficiency."""
    __slots__ = ('symbol', 'direction', 'orders', 'entry_time', 'max_runup_pct',
                 'max_drawdown_pct', 'bars_held', 'last_processed_bar',
                 '_cached_total_size', '_cached_avg_price')
    
    def __init__(self, symbol: str, direction: str, orders: list, entry_time: datetime,
                 max_runup_pct: float = 0.0, max_drawdown_pct: float = 0.0,
                 bars_held: int = 0, last_processed_bar: Optional[str] = None):
        self.symbol = symbol
        self.direction = direction
        self.orders = orders
        self.entry_time = entry_time
        self.max_runup_pct = max_runup_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.bars_held = bars_held
        self.last_processed_bar = last_processed_bar
        self._cached_total_size: Optional[float] = None
        self._cached_avg_price: Optional[float] = None

    @property
    def total_size(self) -> float:
        if self._cached_total_size is None:
            self._cached_total_size = sum(o["size"] for o in self.orders)
        return self._cached_total_size

    @property
    def avg_entry_price(self) -> float:
        if self._cached_avg_price is None:
            if not self.orders:
                return 0.0
            total = self.total_size
            if total <= 0:
                return 0.0
            total_cost = sum(o["price"] * o["size"] for o in self.orders)
            self._cached_avg_price = total_cost / total
        return self._cached_avg_price
    
    def invalidate_cache(self) -> None:
        """Call after modifying orders to reset cached values."""
        self._cached_total_size = None
        self._cached_avg_price = None

    @property
    def num_orders(self) -> int:
        return len(self.orders)
    
    @property
    def last_bar_time(self) -> str:
        return self.orders[-1]["bar_time"] if self.orders else ""

@dataclass(slots=True)
class Trade:
    """Completed trade record. Uses __slots__ for memory efficiency."""
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

@dataclass(slots=True)
class BandData:
    """Indicator values. Uses __slots__ for memory efficiency."""
    close: float
    high: float
    low: float
    upper3: float
    lower3: float
    basis: float
    bar_time: datetime
    avg_volatility_pct: float  # Average candle size (high-low as % of close)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

class TradingState:
    """Manages trading state with persistence."""
    
    __slots__ = ('config', 'positions', 'trades', 'signals_seen', 'start_time',
                 'total_signals', 'api_errors', '_lock', 'logger', 'save_lock')
    
    # Cache Trade field names once (avoid repeated asdict().keys() calls)
    _TRADE_FIELDS: Tuple[str, ...] = (
        'symbol', 'direction', 'entry_price', 'exit_price', 'entry_time', 'exit_time',
        'pnl_pct', 'pnl_usd', 'hold_seconds', 'exit_reason', 'position_size',
        'num_dca_orders', 'max_runup_pct', 'max_drawdown_pct', 'bars_held'
    )

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
        """Actual save logic running in background thread."""
        try:
            with self.save_lock:
                with self._lock:
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
                    f.write(json_dumps(state_data, indent=2))
                temp_file.replace(self.config.state_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def load(self) -> None:
        """Load state from disk."""
        if not self.config.state_file.exists():
            return

        try:
            with open(self.config.state_file, "r") as f:
                state_data = json_loads(f.read())

            # Restore session metadata
            if "start_time" in state_data:
                saved_start = datetime.fromisoformat(state_data["start_time"])
                if saved_start.tzinfo is None:
                    saved_start = saved_start.replace(tzinfo=timezone.utc)
                self.start_time = saved_start
            
            # Restore counters
            self.total_signals = state_data.get("total_signals", 0)
            self.api_errors = state_data.get("api_errors", 0)
            
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

            self.logger.info(f"Restored {len(self.positions)} open positions")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")

    def load_trades(self) -> None:
        """Load historical trades from disk."""
        if not self.config.trades_file.exists():
            return

        try:
            with open(self.config.trades_file, "r") as f:
                trades_data = json_loads(f.read())

            # Cap memory usage
            recent = trades_data[-self.config.trades_memory_cap:]
            skipped = 0
            for t in recent:
                try:
                    self.trades.append(Trade(**t))
                except (TypeError, KeyError) as e:
                    skipped += 1
                    self.logger.warning(f"Skipped corrupted trade record: {e}")
            
            if skipped:
                self.logger.warning(f"Skipped {skipped} corrupted trade records")
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
                    trades_data = json_loads(f.read())

            trades_data.append(asdict(trade))

            temp_file = self.config.trades_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                f.write(json_dumps(trades_data, indent=2))
            temp_file.replace(self.config.trades_file)
        except Exception as e:
            self.logger.error(f"Failed to save trade JSON: {e}")

    def _save_trade_csv(self, trade: Trade) -> None:
        try:
            file_exists = self.config.trades_csv.exists()
            
            with open(self.config.trades_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._TRADE_FIELDS)
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
    
    __slots__ = ('config', 'state', 'logger', 'session', 'ws')

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
        """Fallback to REST if needed, but we should use WS cache ideally."""
        try:
            resp = self.session.get_tickers(category="linear", symbol=symbol)
            if resp["retCode"] == 0 and resp["result"]["list"]:
                return float(resp["result"]["list"][0]["lastPrice"])
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

            # Single pass filter: Trading status, USDT quote, and ends with USDT
            symbols = [
                i["symbol"] for i in resp["result"]["list"] 
                if i["status"] == "Trading" 
                and i["quoteCoin"] == "USDT"
                and i["symbol"].endswith("USDT")
            ]
            
            self.logger.info(f"Found {len(symbols)} active USDT pairs")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error fetching symbols: {e}")
            # Fallback to a small list if API fails completely
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def check_symbol_status(self, symbol: str) -> str:
        """Check if a symbol is still trading. Returns 'Trading', 'Closed', or 'Unknown'."""
        try:
            resp = self.session.get_instruments_info(category="linear", symbol=symbol)
            if resp["retCode"] == 0 and resp["result"]["list"]:
                return resp["result"]["list"][0]["status"]
            return "Unknown"
        except Exception as e:
            self.logger.error(f"Status check failed for {symbol}: {e}")
            return "Error"


class DeviationMagnetStrategy:
    """Core strategy logic."""
    
    __slots__ = ('config',)
    
    # DCA price filter thresholds (pre-computed)
    _DCA_LONG_THRESHOLD = 0.998   # Buy only if 0.2% below last entry
    _DCA_SHORT_THRESHOLD = 1.002  # Sell only if 0.2% above last entry

    def __init__(self, config: Config):
        self.config = config

    def calculate_bands_fast(self, buffer: np.ndarray, count: int, head: int, array_size: int) -> Optional[BandData]:
        """
        High-performance Bollinger Band calculation using Numpy.
        Uses CURRENT-BAR logic (TradingView default):
        - Stats calculated on last N candles (INCLUDING current).
        - Current price compared to bands calculated from those same N candles.
        
        Expects circular buffer indexing.
        """
        # Cache config values as locals (faster attribute access)
        required_len = self.config.bb_length
        mult = self.config.mult
        dev_mult = self.config.dev_mult
        
        if count < required_len:
            return None

        # Extract the last N items in chronological order (INCLUDING current)
        if count < array_size:
            # Buffer not full, data is linear [0:count]
            subset = buffer[count - required_len : count]
        else:
            # Circular buffer is full - compute indices locally (thread-safe)
            # Extract last N items: [head - N + 1, ..., head]
            base = head - required_len + 1
            indices = (np.arange(required_len) + base) % array_size
            subset = buffer[indices]
        
        # Check for NaNs in the window
        if np.isnan(subset[0, 0]):
            return None
        
        # Extract columns once (faster than repeated indexing)
        opens = subset[:, 0]
        highs = subset[:, 1]
        lows = subset[:, 2]
        closes = subset[:, 3]
        
        # Combined OHLC4 and volatility calculation (single pass where possible)
        ohlc4 = (opens + highs + lows + closes) * 0.25  # Multiply by 0.25 faster than divide by 4
        
        # Volatility: average (high-low)/close as percentage
        avg_candle_size_pct = np.mean((highs - lows) / closes) * 100
        
        # Calculate Basis (Mean) and Stdev on last N candles (INCLUDING current)
        basis = np.mean(ohlc4)
        stdev = np.std(ohlc4, ddof=1)  # Sample stdev to match Pine Script's ta.stdev
        
        # Calculate Bands (using cached locals)
        dev = mult * stdev
        upper3 = basis + (dev * dev_mult)
        lower3 = basis - (dev * dev_mult)
        
        # Current Candle (The last one in the window)
        current_candle = subset[-1]
        latest_close = current_candle[3]
        latest_high = current_candle[1]
        latest_low = current_candle[2]
        latest_time_ts = current_candle[4]
        
        # Convert timestamp to datetime (UTC)
        bar_time = datetime.fromtimestamp(latest_time_ts, tz=timezone.utc)
        
        return BandData(
            close=latest_close,
            high=latest_high,
            low=latest_low,
            upper3=upper3,
            lower3=lower3,
            basis=basis,
            bar_time=bar_time,
            avg_volatility_pct=avg_candle_size_pct,
        )

    def check_entry(self, symbol: str, data: BandData, state: TradingState) -> Tuple[Optional[str], bool]:
        """Returns (direction, is_dca)."""
        
        # Cache frequently accessed attributes
        data_low = data.low
        data_high = data.high
        lower3 = data.lower3
        upper3 = data.upper3
        
        # 1. Check Existing Position (DCA) - Allow DCA even in low volatility to help exit losing positions
        pos = state.positions.get(symbol)
        if pos is not None:
            # Cache bar_time string only when needed
            bar_time_str = data.bar_time.isoformat()
            
            if pos.last_bar_time == bar_time_str:
                return None, False  # Already acted this bar
            
            # num_orders includes base order, so check if we've already added max DCAs
            if pos.num_orders > self.config.max_dca_orders:
                return None, False
            
            # Get last order price and direction once
            last_order_price = pos.orders[-1]["price"]
            direction = pos.direction
            
            if direction == "long" and data_low <= lower3:
                # DCA Filter: Only buy if band price is lower than last entry by at least 0.2%
                # Use lower3 (the actual entry price) not close
                if lower3 <= last_order_price * self._DCA_LONG_THRESHOLD:
                    state.total_signals += 1
                    return "long", True
            
            elif direction == "short" and data_high >= upper3:
                # DCA Filter: Only sell if band price is higher than last entry by at least 0.2%
                # Use upper3 (the actual entry price) not close
                if upper3 >= last_order_price * self._DCA_SHORT_THRESHOLD:
                    state.total_signals += 1
                    return "short", True
            
            return None, False
        
        # 2. Check New Entry
        # Volatility Filter: Only block NEW entries on dead/illiquid pairs (not DCA)
        if data.avg_volatility_pct < self.config.min_volatility_pct:
            return None, False
        
        # Cache bar_time string for new entry check
        bar_time_str = data.bar_time.isoformat()
        bar_key = f"{symbol}_{bar_time_str}"
        
        if bar_key in state.signals_seen:
            return None, False

        if data_low <= lower3:
            state.signals_seen[bar_key] = "long"
            state.total_signals += 1
            return "long", False
        
        if data_high >= upper3:
            state.signals_seen[bar_key] = "short"
            state.total_signals += 1
            return "short", False

        return None, False

    def check_exit(self, position: Position, data: BandData) -> Tuple[bool, str, float]:
        """Volatility-Based Exit Logic.
        
        TP = avg candle volatility + total fees
        This ensures we capture the typical price movement PLUS cover all trading costs.
        """
        # Safety check first
        if not position.orders:
            return False, "", 0.0

        # Cache as locals for faster access
        avg_entry = position.avg_entry_price
        total_fee_pct = self.config.total_fee_pct_for_tp  # ~0.17%
        direction = position.direction
        
        # TP = volatility + fees (capture avg movement AND cover costs)
        target_pct = data.avg_volatility_pct + total_fee_pct
        
        # Pre-compute multiplier
        pct_mult = target_pct / 100
        
        if direction == "long":
            target_price = avg_entry * (1 + pct_mult)
            # For longs: entry is at lower3 (low of candle touched it)
            # TP requires HIGH to reach target - this is valid because:
            # - Entry happens when LOW touches lower3
            # - TP happens when HIGH reaches target (different part of candle)
            # - On same bar: if high already >= target when low touched entry, it's a valid spike profit
            if data.high >= target_price:
                return True, "volatility_tp", target_price
        else:
            target_price = avg_entry * (1 - pct_mult)
            # For shorts: entry is at upper3 (high of candle touched it)  
            # TP requires LOW to reach target
            if data.low <= target_price:
                return True, "volatility_tp", target_price

        return False, "", 0.0


class TradeExecutor:
    """Handles order execution, PnL math, and state updates."""
    
    __slots__ = ('config', 'state', 'logger')

    def __init__(self, config: Config, state: TradingState):
        self.config = config
        self.state = state
        self.logger = logging.getLogger("deviation_magnet")

    def execute_entry(self, symbol: str, direction: str, price: float, bar_time: datetime, is_dca: bool):
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()  # Cache iso string
        bar_str = bar_time.isoformat()  # Inline _fmt_time

        if is_dca and symbol in self.state.positions:
            pos = self.state.positions[symbol]
            # Size = Current Total Exposure * DCA Scale (aggressive averaging down)
            size = pos.total_size * self.config.dca_scale
            
            # Fallback
            if size <= 0:
                size = self.config.base_order_size

            pos.orders.append({
                "price": price,
                "size": size,
                "time": now_iso,
                "bar_time": bar_str
            })
            pos.invalidate_cache()  # Reset cached total_size and avg_entry_price
            
            self.logger.info(
                f"📥 DCA #{pos.num_orders}: {symbol} {direction.upper()} @ ${price:.4f} "
                f"(+${size:.0f}, total: ${pos.total_size:.0f}) | Avg: ${pos.avg_entry_price:.4f}"
            )
        else:
            initial = {
                "price": price,
                "size": self.config.base_order_size,
                "time": now_iso,
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
        direction = pos.direction
        
        # Pre-compute price ratio for efficiency
        price_ratio = price / avg

        if direction == "long":
            pnl_pct = (price_ratio - 1.0) * 100
        else:
            pnl_pct = (1.0 - price_ratio) * 100
            
        # Fees + slippage logic (using pre-computed costs from Config)
        entry_fee = total_size * self.config.entry_cost_pct
        
        notional_exit = total_size * price_ratio
        
        # Exit: maker/taker fee + slippage
        exit_cost_pct = self.config.exit_maker_cost_pct if is_maker else self.config.exit_taker_cost_pct
        exit_fee = notional_exit * exit_cost_pct
        
        total_fees = entry_fee + exit_fee
        
        gross_pnl_usd = pnl_pct * 0.01 * total_size  # * 0.01 faster than / 100
        pnl_usd = gross_pnl_usd - total_fees
        
        entry_time = pos.entry_time.replace(tzinfo=timezone.utc) if pos.entry_time.tzinfo is None else pos.entry_time
        
        # Cache strftime format string
        time_fmt = "%Y-%m-%d %H:%M:%S"
        
        trade = Trade(
            symbol=symbol,
            direction=direction,
            entry_price=avg,
            exit_price=price,
            entry_time=entry_time.strftime(time_fmt),
            exit_time=now.strftime(time_fmt),
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
            f"{emoji} EXIT: {symbol} {direction.upper()} @ ${price:.4f} | "
            f"PnL: ${pnl_usd:.2f} (Net) | Price: {pnl_pct:.2f}% (Raw) | {reason}"
        )
        self.logger.info(f"   Stats: Runup: {pos.max_runup_pct:.2f}% | DD: {pos.max_drawdown_pct:.2f}% | Bars: {pos.bars_held}")

    def update_position_stats(self, symbol: str, current_price: float, data: BandData):
        pos = self.state.positions.get(symbol)
        if pos is None:
            return
        
        avg = pos.avg_entry_price
        if avg <= 0:
            return  # Safety: can't compute PnL without valid avg price
        
        direction = pos.direction  # Cache
        
        if direction == "long":
            pnl_pct = (current_price - avg) / avg * 100
        else:
            pnl_pct = (avg - current_price) / avg * 100
            
        pos.max_runup_pct = max(pos.max_runup_pct, pnl_pct)
        pos.max_drawdown_pct = min(pos.max_drawdown_pct, pnl_pct)
        
        bar_str = data.bar_time.isoformat()  # Inline _fmt_time
        if pos.last_processed_bar != bar_str:
            pos.bars_held += 1
            pos.last_processed_bar = bar_str


class DataManager:
    """Manages hybrid data: REST history + WebSocket updates.
    
    Optimized for:
    1. Latency: O(1) circular buffer updates
    2. Concurrency: Per-symbol locks to prevent blocking across pairs
    3. Memory: Fixed-size numpy arrays with automatic wraparound
    """
    
    __slots__ = ('client', 'symbols', 'array_size', 'data_buffers', 'counts',
                 'head', 'locks', 'logger', '_tf_minutes', '_gap_threshold')
    
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
        
        # Cache timeframe as int for gap detection
        self._tf_minutes = int(client.config.timeframe)
        self._gap_threshold = self._tf_minutes * 5 * 60.0  # Pre-compute gap threshold in seconds
        
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
        # Early exit if no data
        if not (kline_data := message.get("data")):
            return

        # Extract symbol from topic (e.g. "kline.1.BTCUSDT")
        topic = message.get("topic", "")
        if not topic.startswith("kline."):
            return
        symbol = topic.rsplit(".", 1)[-1]

        # Fail fast if symbol not tracked
        buffers = self.data_buffers
        if symbol not in buffers:
            return
        
        # Cache dict references for this symbol (avoid repeated lookups)
        buffer = buffers[symbol]
        symbol_lock = self.locks[symbol]
        array_size = self.array_size
        gap_threshold = self._gap_threshold

        for kline in kline_data:
            # Convert timestamp once (use multiplication, faster than division)
            ts_float = int(kline["start"]) * 0.001
            
            with symbol_lock:
                count = self.counts[symbol]
                
                # Create row data (inside lock for thread safety)
                row = np.array([
                    float(kline["open"]),
                    float(kline["high"]),
                    float(kline["low"]),
                    float(kline["close"]),
                    ts_float
                ], dtype=np.float64)

                head_idx = self.head[symbol]
                
                if count == 0:
                    buffer[0] = row
                    self.counts[symbol] = 1
                    self.head[symbol] = 0
                    continue

                last_ts = buffer[head_idx, 4]
                
                if ts_float == last_ts:
                    buffer[head_idx] = row
                    
                elif ts_float > last_ts:
                    time_diff = ts_float - last_ts
                    
                    if time_diff > gap_threshold:
                        self.logger.warning(f"⚠️ Data Gap detected for {symbol}: {time_diff/60:.1f}m. Resetting buffer.")
                        buffer.fill(np.nan)
                        buffer[0] = row
                        self.counts[symbol] = 1
                        self.head[symbol] = 0
                    else:
                        new_head = (head_idx + 1) % array_size
                        buffer[new_head] = row
                        self.head[symbol] = new_head
                        self.counts[symbol] = min(count + 1, array_size)

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
    
    __slots__ = ('config', 'logger', 'state', 'client', 'strategy', 'executor',
                 'symbols', 'data_manager', 'event_queue', 'last_update_time',
                 'watchdog_threshold', 'reconnect_counts', '_timeframe_int')
    
    # Class-level constant for valid trading statuses
    _VALID_STATUSES = frozenset({"Trading", "PreLaunch"})
    
    # Pre-computed separator strings (class-level to avoid recreation)
    _SEP_DOUBLE = '═' * 80
    _SEP_SINGLE = '─' * 80

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
        self.event_queue: queue.Queue[str] = queue.Queue()
        self.last_update_time: Dict[str, float] = {}
        self.watchdog_threshold = 60.0
        self.reconnect_counts: Dict[str, int] = {}
        self._timeframe_int = int(self.config.timeframe)  # Cache int conversion

    def run(self):
        """Start the bot loop."""
        self.state.load()
        self.state.load_trades()
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

    def _handle_ws_message(self, message):
        try:
            self.data_manager.on_kline_update(message)
            # Queue event for processing in main thread
            topic = message.get("topic", "")
            # Fast path: check for kline prefix
            if topic.startswith("kline."):
                # Topic format: "kline.1.BTCUSDT" - symbol is after second dot
                symbol = topic.rsplit(".", 1)[-1]
                self.last_update_time[symbol] = time.time()
                self.event_queue.put(symbol)
        except Exception as e:
            self.logger.error(f"WS Handler Error: {e}")

    def _setup_websockets(self):
        """Connect and subscribe to WebSockets."""
        self.logger.info("🔌 Connecting to WebSockets...")
        
        # Subscribe to all symbols using cached timeframe int
        tf_int = self._timeframe_int
        for symbol in self.symbols:
            self.client.ws.kline_stream(
                interval=tf_int,
                symbol=symbol,
                callback=self._handle_ws_message
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
        # Use list() copy to avoid modifying list while iterating
        for symbol in list(self.symbols):
            last = self.last_update_time.get(symbol, now)
            diff = now - last
            if diff > self.watchdog_threshold:
                self.logger.warning(f"⚠️ WATCHDOG: No data for {symbol} in {diff:.1f}s! Checking status...")
                
                # 1. Check if Delisted / Not Trading
                status = self.client.check_symbol_status(symbol)
                
                if status not in self._VALID_STATUSES:
                    self.logger.warning(f"🚨 CRITICAL: {symbol} status is '{status}'. Potential DELISTING!")
                    
                    # Emergency Exit if position exists
                    with self.state._lock:
                        if symbol in self.state.positions:
                            self.logger.warning(f"🚨 Force closing position for {symbol} due to invalid status.")
                            pos = self.state.positions[symbol]
                            # Try to get a price, any price
                            price = self.client.get_current_price(symbol) or pos.avg_entry_price
                            self.executor.execute_exit(symbol, pos, price, "delisted_emergency")
                    
                    # Stop tracking
                    if symbol in self.symbols:
                        self.symbols.remove(symbol)
                    continue

                # 2. Attempt to re-subscribe
                self.reconnect_counts[symbol] = self.reconnect_counts.get(symbol, 0) + 1
                reconnect_count = self.reconnect_counts[symbol]
                
                if reconnect_count >= 5:
                    self.logger.warning(f"⚠️ {symbol} has reconnected {reconnect_count} times - possible connectivity issue")
                
                self.logger.info(f"🔄 Symbol {symbol} is {status}. Re-subscribing to WS... (attempt #{reconnect_count})")
                try:
                    self.client.ws.kline_stream(
                        interval=self._timeframe_int,
                        symbol=symbol,
                        callback=self._handle_ws_message
                    )
                    # Reset timer to give it a chance to recover without spamming logs
                    self.last_update_time[symbol] = now
                except Exception as e:
                    self.logger.error(f"Failed to re-subscribe {symbol}: {e}")

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
            # Check Exit - use .get() to avoid double lookup
            pos = self.state.positions.get(symbol)
            if pos is not None:
                self.executor.update_position_stats(symbol, current_price, data)
                
                should_exit, reason, exit_price = self.strategy.check_exit(pos, data)
                if should_exit:
                    # volatility_tp uses limit orders = maker fees
                    is_maker = reason in ("profit_target", "volatility_tp")
                    self.executor.execute_exit(symbol, pos, exit_price, reason, is_maker)
                    # Prevent immediate re-entry
                    bar_key = f"{symbol}_{data.bar_time.isoformat()}"
                    self.state.signals_seen[bar_key] = "exit"
                    return

            # Check Entry
            direction, is_dca = self.strategy.check_entry(symbol, data, self.state)
            if direction:
                # Use band price for entry, not current close (which may have moved)
                # Long entries touch lower band, short entries touch upper band
                entry_price = data.lower3 if direction == "long" else data.upper3
                self.executor.execute_entry(symbol, direction, entry_price, data.bar_time, is_dca)

    def _daily_report(self, date):
        """Log daily trade summary."""
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
        
        # Use class-level separator strings
        sep_double = self._SEP_DOUBLE
        sep_single = self._SEP_SINGLE
        
        print(f"\n{sep_double}")
        print(f"  📊 SESSION STATS")
        print(sep_single)
        print(f"  Runtime: {session_duration:.1f}h | Signals: {self.state.total_signals} | Errors: {self.state.api_errors}")
        
        # Open Positions Details
        if self.state.positions:
            print(f"\n{sep_single}")
            print(f"  📂 OPEN POSITIONS ({len(self.state.positions)})")
            print(sep_single)
            
            total_exposure = 0
            total_unrealized_pnl = 0
            
            # Sort by unrealized PnL (worst first)
            positions_sorted = []
            for sym, pos in self.state.positions.items():
                df = self.data_manager.get_latest_data(sym)
                if df is not None and len(df) > 0:
                    current = df.iloc[-1]["close"]
                    avg = pos.avg_entry_price
                    price_ratio = current / avg
                    pnl_pct = (price_ratio - 1.0) * 100 if pos.direction == "long" else (1.0 - price_ratio) * 100
                    
                    # Calculate unrealized USD (after fees + slippage, using pre-computed costs)
                    total_size = pos.total_size
                    entry_fee = total_size * self.config.entry_cost_pct
                    notional_exit = total_size * price_ratio
                    exit_fee = notional_exit * self.config.exit_taker_cost_pct
                    total_fees = entry_fee + exit_fee
                    gross_pnl_usd = pnl_pct * 0.01 * total_size
                    unrealized_usd = gross_pnl_usd - total_fees
                    
                    total_exposure += total_size
                    total_unrealized_pnl += unrealized_usd
                    
                    positions_sorted.append((sym, pos, pnl_pct, unrealized_usd, current))
            
            # Sort by PnL% (worst first so you see problems)
            positions_sorted.sort(key=itemgetter(2))
            
            for sym, pos, pnl_pct, unrealized_usd, current in positions_sorted:
                dca_info = f"DCA #{pos.num_orders}" if pos.num_orders > 1 else "BASE"
                direction_emoji = "📈" if pos.direction == "long" else "📉"
                pnl_emoji = "🟢" if unrealized_usd >= 0 else "🔴"
                
                print(f"  {pnl_emoji} {sym:<12} {direction_emoji} {pos.direction.upper():<5} | "
                      f"{dca_info:<8} | ${pos.total_size:>7.0f} | "
                      f"{pnl_pct:>+6.2f}% (${unrealized_usd:>+6.2f}) | "
                      f"Hold: {pos.bars_held}m | DD: {pos.max_drawdown_pct:>+5.2f}%")
            
            print(sep_single)
            print(f"  💰 Total Exposure: ${total_exposure:,.0f}")
            print(f"  💵 Unrealized PnL: ${total_unrealized_pnl:>+,.2f}")
        
        else:
            print(f"\n{sep_single}")
            print(f"  📂 OPEN POSITIONS: None")
        
        # Realized PnL (AT BOTTOM - Most Important)
        print(f"\n{sep_double}")
        print(f"  📈 REALIZED PNL")
        print(sep_double)
        
        if self.state.trades:
            # Single-pass computation of all trade statistics
            total_realized = 0.0
            win_count = 0
            total_hold_seconds = 0
            best_trade = self.state.trades[0]
            worst_trade = self.state.trades[0]
            
            for t in self.state.trades:
                total_realized += t.pnl_usd
                total_hold_seconds += t.hold_seconds
                if t.pnl_usd > 0:
                    win_count += 1
                if t.pnl_usd > best_trade.pnl_usd:
                    best_trade = t
                if t.pnl_usd < worst_trade.pnl_usd:
                    worst_trade = t
            
            trade_count = len(self.state.trades)
            loss_count = trade_count - win_count
            win_rate = win_count / trade_count * 100
            avg_hold_time = total_hold_seconds / trade_count / 60
            
            print(f"  Total Trades: {trade_count} | Wins: {win_count} | Losses: {loss_count} | Win Rate: {win_rate:.1f}%")
            print(f"  Avg Hold: {avg_hold_time:.1f}m | Best: ${best_trade.pnl_usd:+.2f} ({best_trade.symbol}) | Worst: ${worst_trade.pnl_usd:+.2f} ({worst_trade.symbol})")
            print(f"\n  💰 TOTAL REALIZED PNL: ${total_realized:>+,.2f}")
        else:
            print(f"  No trades yet")
        
        print(f"{sep_double}\n", flush=True)


if __name__ == "__main__":
    try:
        bot = Bot()
        bot.run()
    except Exception as e:
        print(f"💥 Critical Failure: {e}")
        traceback.print_exc()
