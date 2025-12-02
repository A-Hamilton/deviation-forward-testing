"""
Deviation Magnet Forward Tester
===============================

Real-time paper trading bot for the Deviation Magnet strategy.
Simplified logic: Multiple positions per direction, opposite signal = exit all.

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
from pathlib import Path
from threading import RLock, Lock, Thread
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

# Version tracking
__version__ = "2.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Trading configuration parameters."""

    # Indicator settings
    bb_length: int = 20
    mult: float = 3.0  # Lower for more trades - will be tuned by optimizer
    dev_mult: float = 1.5

    # Timing
    timeframe: str = field(default_factory=lambda: os.environ.get("TIMEFRAME", "1"))
    check_interval: int = 30
    
    # Parallel processing
    max_workers: int = 10

    # Position management - SIMPLIFIED
    base_order_size: float = field(default_factory=lambda: float(os.environ.get("BASE_ORDER_SIZE", "10.0")))
    max_positions_per_direction: int = 100  # Max longs OR shorts per symbol
    
    # Fee structure
    fee_pct: float = 0.00055  # 0.055% taker
    maker_fee_pct: float = 0.0002  # 0.02% maker
    slippage_pct: float = 0.0005  # 0.05%
    
    # Volatility filter
    min_volatility_pct: float = 0.2

    # Memory management
    max_signals_cache: int = 2000
    signals_cleanup_keep: int = 1000
    trades_memory_cap: int = 500

    # Status display
    status_interval: int = 5

    # API retry settings
    api_retries: int = 3
    api_retry_delay: float = 1.0
    
    # Queue monitoring
    max_queue_size: int = 10000

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("forward_test_data"))

    # Pre-computed fee combinations
    entry_cost_pct: float = field(init=False)
    exit_taker_cost_pct: float = field(init=False)
    exit_maker_cost_pct: float = field(init=False)
    total_fee_pct_for_tp: float = field(init=False)

    def __post_init__(self):
        self.entry_cost_pct = self.fee_pct + self.slippage_pct
        self.exit_taker_cost_pct = self.fee_pct + self.slippage_pct
        self.exit_maker_cost_pct = self.maker_fee_pct + self.slippage_pct
        # TP uses maker for exit (limit order)
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

@dataclass
class SinglePosition:
    """A single position entry (no DCA, just one order)."""
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    size: float
    entry_time: datetime
    bar_time: str
    max_runup_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    bars_held: int = 0
    last_processed_bar: Optional[str] = None


@dataclass(slots=True)
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
    num_positions: int = 1  # How many positions closed at once

    # Stats
    max_runup_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    bars_held: int = 0


@dataclass(slots=True)
class BandData:
    """Indicator values."""
    close: float
    high: float
    low: float
    upper3: float
    lower3: float
    basis: float
    bar_time: datetime
    avg_volatility_pct: float


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION MANAGER - NEW SIMPLIFIED LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class PositionManager:
    """
    Manages multiple positions per symbol.
    
    Logic:
    - Can have multiple positions in SAME direction (up to max_positions_per_direction)
    - Opposite signal closes ALL positions for that symbol
    - Each position tracked individually for TP
    """
    
    def __init__(self, config: Config):
        self.config = config
        # positions[symbol] = list of SinglePosition objects
        self.positions: Dict[str, List[SinglePosition]] = defaultdict(list)
        self._lock = RLock()
    
    def get_direction(self, symbol: str) -> Optional[str]:
        """Get current direction for symbol (None if no positions)."""
        with self._lock:
            positions = self.positions.get(symbol, [])
            if not positions:
                return None
            return positions[0].direction
    
    def get_position_count(self, symbol: str) -> int:
        """Get number of open positions for symbol."""
        with self._lock:
            return len(self.positions.get(symbol, []))
    
    def can_add_position(self, symbol: str, direction: str) -> bool:
        """Check if we can add a position in this direction."""
        with self._lock:
            positions = self.positions.get(symbol, [])
            if not positions:
                return True
            # Can only add if same direction and under limit
            if positions[0].direction != direction:
                return False  # Would need to close first
            return len(positions) < self.config.max_positions_per_direction
    
    def add_position(self, pos: SinglePosition) -> None:
        """Add a new position."""
        with self._lock:
            self.positions[pos.symbol].append(pos)
    
    def get_positions(self, symbol: str) -> List[SinglePosition]:
        """Get all positions for a symbol."""
        with self._lock:
            return list(self.positions.get(symbol, []))
    
    def remove_position(self, symbol: str, pos: SinglePosition) -> None:
        """Remove a specific position (for TP exit)."""
        with self._lock:
            if symbol in self.positions:
                try:
                    self.positions[symbol].remove(pos)
                    if not self.positions[symbol]:
                        del self.positions[symbol]
                except ValueError:
                    pass
    
    def close_all_positions(self, symbol: str) -> List[SinglePosition]:
        """Close all positions for a symbol, return the closed positions."""
        with self._lock:
            positions = self.positions.pop(symbol, [])
            return positions
    
    def get_all_symbols_with_positions(self) -> List[str]:
        """Get all symbols that have open positions."""
        with self._lock:
            return list(self.positions.keys())
    
    def get_total_exposure(self, symbol: str) -> float:
        """Get total exposure for a symbol."""
        with self._lock:
            return sum(p.size for p in self.positions.get(symbol, []))
    
    def to_dict(self) -> dict:
        """Serialize for saving."""
        with self._lock:
            result = {}
            for symbol, positions in self.positions.items():
                result[symbol] = [
                    {
                        "symbol": p.symbol,
                        "direction": p.direction,
                        "entry_price": p.entry_price,
                        "size": p.size,
                        "entry_time": p.entry_time.isoformat(),
                        "bar_time": p.bar_time,
                        "max_runup_pct": p.max_runup_pct,
                        "max_drawdown_pct": p.max_drawdown_pct,
                        "bars_held": p.bars_held,
                        "last_processed_bar": p.last_processed_bar,
                    }
                    for p in positions
                ]
            return result
    
    def from_dict(self, data: dict) -> None:
        """Deserialize from saved state."""
        with self._lock:
            self.positions.clear()
            for symbol, positions_data in data.items():
                for p in positions_data:
                    entry_time = datetime.fromisoformat(p["entry_time"])
                    if entry_time.tzinfo is None:
                        entry_time = entry_time.replace(tzinfo=timezone.utc)
                    
                    self.positions[symbol].append(SinglePosition(
                        symbol=p["symbol"],
                        direction=p["direction"],
                        entry_price=p["entry_price"],
                        size=p["size"],
                        entry_time=entry_time,
                        bar_time=p["bar_time"],
                        max_runup_pct=p.get("max_runup_pct", 0.0),
                        max_drawdown_pct=p.get("max_drawdown_pct", 0.0),
                        bars_held=p.get("bars_held", 0),
                        last_processed_bar=p.get("last_processed_bar"),
                    ))


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING STATE
# ═══════════════════════════════════════════════════════════════════════════════

class TradingState:
    """Manages trading state with persistence."""
    
    _TRADE_FIELDS: Tuple[str, ...] = (
        'symbol', 'direction', 'entry_price', 'exit_price', 'entry_time', 'exit_time',
        'pnl_pct', 'pnl_usd', 'hold_seconds', 'exit_reason', 'position_size',
        'num_positions', 'max_runup_pct', 'max_drawdown_pct', 'bars_held'
    )

    def __init__(self, config: Config) -> None:
        self.config = config
        self.position_manager = PositionManager(config)
        self.trades: List[Trade] = []
        self.signals_seen: Dict[str, str] = {}
        self.start_time: datetime = datetime.now(timezone.utc)
        self.total_signals: int = 0
        self.api_errors: int = 0
        self._lock = RLock()
        self.logger = logging.getLogger("deviation_magnet")
        self.save_lock = Lock()

    def increment_errors(self) -> None:
        with self._lock:
            self.api_errors += 1

    def save(self) -> None:
        """Async save."""
        Thread(target=self._save_sync, daemon=True).start()

    def _save_sync(self) -> None:
        try:
            with self.save_lock:
                state_data = {
                    "positions": self.position_manager.to_dict(),
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

            if "start_time" in state_data:
                saved_start = datetime.fromisoformat(state_data["start_time"])
                if saved_start.tzinfo is None:
                    saved_start = saved_start.replace(tzinfo=timezone.utc)
                self.start_time = saved_start
            
            self.total_signals = state_data.get("total_signals", 0)
            self.api_errors = state_data.get("api_errors", 0)
            
            if "positions" in state_data:
                self.position_manager.from_dict(state_data["positions"])

            total_positions = sum(
                len(positions) 
                for positions in self.position_manager.positions.values()
            )
            self.logger.info(f"Restored {total_positions} open positions")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")

    def load_trades(self) -> None:
        """Load historical trades from disk."""
        if not self.config.trades_file.exists():
            return

        try:
            with open(self.config.trades_file, "r") as f:
                trades_data = json_loads(f.read())

            recent = trades_data[-self.config.trades_memory_cap:]
            skipped = 0
            for t in recent:
                try:
                    # Handle old format compatibility
                    if 'num_dca_orders' in t:
                        t['num_positions'] = t.pop('num_dca_orders')
                    self.trades.append(Trade(**t))
                except (TypeError, KeyError) as e:
                    skipped += 1
            
            if skipped:
                self.logger.warning(f"Skipped {skipped} corrupted trade records")
            self.logger.info(f"Loaded {len(self.trades)} recent trades")
        except Exception as e:
            self.logger.error(f"Failed to load trades: {e}")

    def add_trade(self, trade: Trade) -> None:
        """Add trade to memory and disk."""
        self.trades.append(trade)
        if len(self.trades) > self.config.trades_memory_cap:
            self.trades.pop(0)
        
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


# ═══════════════════════════════════════════════════════════════════════════════
# BYBIT CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class BybitClient:
    """Handles API interactions."""
    
    def __init__(self, config: Config, state: TradingState):
        self.config = config
        self.state = state
        self.logger = logging.getLogger("deviation_magnet")
        self.session = HTTP(testnet=False)
        self.ws = WebSocket(testnet=False, channel_type="linear")

    def fetch_history_rest(self, symbol: str, limit: int = 250) -> Optional[pd.DataFrame]:
        """Fetch historical candles."""
        try:
            resp = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=self.config.timeframe,
                limit=limit,
            )
            
            if resp["retCode"] != 0 or not resp.get("result", {}).get("list"):
                return None

            raw_data = resp["result"]["list"]
            df = pd.DataFrame(
                raw_data,
                columns=["open_time", "open", "high", "low", "close", "volume", "turnover"],
            )
            df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            
            df = df.sort_values("open_time").reset_index(drop=True)
            return df

        except Exception as e:
            self.logger.warning(f"REST fetch failed for {symbol}: {e}")
            self.state.increment_errors()
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price via REST."""
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
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def check_symbol_status(self, symbol: str) -> str:
        """Check if a symbol is still trading."""
        try:
            resp = self.session.get_instruments_info(category="linear", symbol=symbol)
            if resp["retCode"] == 0 and resp["result"]["list"]:
                return resp["result"]["list"][0]["status"]
            return "Unknown"
        except Exception as e:
            self.logger.error(f"Status check failed for {symbol}: {e}")
            return "Error"


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

class DeviationMagnetStrategy:
    """Core strategy logic - simplified."""
    
    def __init__(self, config: Config):
        self.config = config

    def calculate_bands_fast(self, buffer: np.ndarray, count: int, head: int, array_size: int) -> Optional[BandData]:
        """Calculate Bollinger Bands."""
        required_len = self.config.bb_length
        mult = self.config.mult
        dev_mult = self.config.dev_mult
        
        if count < required_len:
            return None

        if count < array_size:
            subset = buffer[count - required_len : count]
        else:
            base = head - required_len + 1
            indices = (np.arange(required_len) + base) % array_size
            subset = buffer[indices]
        
        if np.isnan(subset[0, 0]):
            return None
        
        opens = subset[:, 0]
        highs = subset[:, 1]
        lows = subset[:, 2]
        closes = subset[:, 3]
        
        ohlc4 = (opens + highs + lows + closes) * 0.25
        avg_candle_size_pct = np.mean((highs - lows) / closes) * 100
        
        basis = np.mean(ohlc4)
        stdev = np.std(ohlc4, ddof=1)
        
        dev = mult * stdev
        upper3 = basis + (dev * dev_mult)
        lower3 = basis - (dev * dev_mult)
        
        current_candle = subset[-1]
        latest_close = current_candle[3]
        latest_high = current_candle[1]
        latest_low = current_candle[2]
        latest_time_ts = current_candle[4]
        
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

    def check_signal(self, symbol: str, data: BandData, state: TradingState) -> Optional[str]:
        """
        Check for entry signal. Returns direction or None.
        Does NOT check current positions - that's handled by caller.
        """
        # Volatility filter
        if data.avg_volatility_pct < self.config.min_volatility_pct:
            return None
        
        bar_time_str = data.bar_time.isoformat()
        bar_key = f"{symbol}_{bar_time_str}"
        
        # Only one signal per bar per symbol
        if bar_key in state.signals_seen:
            return None

        if data.low <= data.lower3:
            state.signals_seen[bar_key] = "long"
            state.total_signals += 1
            return "long"
        
        if data.high >= data.upper3:
            state.signals_seen[bar_key] = "short"
            state.total_signals += 1
            return "short"

        return None

    def check_tp(self, pos: SinglePosition, data: BandData) -> Tuple[bool, float]:
        """
        Check if position should exit via TP.
        Returns (should_exit, target_price).
        """
        avg_entry = pos.entry_price
        total_fee_pct = self.config.total_fee_pct_for_tp
        direction = pos.direction
        
        # TP = volatility + fees
        target_pct = data.avg_volatility_pct + total_fee_pct
        pct_mult = target_pct / 100
        
        if direction == "long":
            target_price = avg_entry * (1 + pct_mult)
            if data.high >= target_price:
                return True, target_price
        else:
            target_price = avg_entry * (1 - pct_mult)
            if data.low <= target_price:
                return True, target_price

        return False, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class TradeExecutor:
    """Handles trade execution and PnL calculation."""
    
    def __init__(self, config: Config, state: TradingState):
        self.config = config
        self.state = state
        self.logger = logging.getLogger("deviation_magnet")

    def execute_entry(self, symbol: str, direction: str, price: float, bar_time: datetime) -> None:
        """Open a new position."""
        now = datetime.now(timezone.utc)
        bar_str = bar_time.isoformat()

        pos = SinglePosition(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            size=self.config.base_order_size,
            entry_time=now,
            bar_time=bar_str,
        )
        
        self.state.position_manager.add_position(pos)
        
        count = self.state.position_manager.get_position_count(symbol)
        self.logger.info(f"📥 ENTRY #{count}: {symbol} {direction.upper()} @ ${price:.4f} (${self.config.base_order_size:.0f})")
        self.state.save()

    def execute_tp_exit(self, symbol: str, pos: SinglePosition, price: float) -> None:
        """Exit a single position via TP (maker fees)."""
        self._execute_single_exit(symbol, pos, price, "volatility_tp", is_maker=True)

    def execute_opposite_signal_exit(self, symbol: str, positions: List[SinglePosition], exit_price: float) -> None:
        """
        Exit ALL positions for a symbol due to opposite signal (taker fees).
        Records as individual trades for accurate PnL tracking.
        """
        for pos in positions:
            self._execute_single_exit(symbol, pos, exit_price, "opposite_signal", is_maker=False)

    def _execute_single_exit(self, symbol: str, pos: SinglePosition, price: float, reason: str, is_maker: bool) -> None:
        """Execute exit for a single position with proper fee calculation."""
        now = datetime.now(timezone.utc)
        entry_price = pos.entry_price
        size = pos.size
        direction = pos.direction
        
        price_ratio = price / entry_price

        if direction == "long":
            pnl_pct = (price_ratio - 1.0) * 100
        else:
            pnl_pct = (1.0 - price_ratio) * 100
        
        # Fee calculation
        entry_fee = size * self.config.entry_cost_pct
        notional_exit = size * price_ratio
        exit_cost_pct = self.config.exit_maker_cost_pct if is_maker else self.config.exit_taker_cost_pct
        exit_fee = notional_exit * exit_cost_pct
        total_fees = entry_fee + exit_fee
        
        gross_pnl_usd = pnl_pct * 0.01 * size
        pnl_usd = gross_pnl_usd - total_fees
        
        entry_time = pos.entry_time
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        
        time_fmt = "%Y-%m-%d %H:%M:%S"
        
        trade = Trade(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=price,
            entry_time=entry_time.strftime(time_fmt),
            exit_time=now.strftime(time_fmt),
            pnl_pct=round(pnl_pct, 4),
            pnl_usd=round(pnl_usd, 4),
            hold_seconds=int((now - entry_time).total_seconds()),
            exit_reason=reason,
            position_size=size,
            num_positions=1,
            max_runup_pct=pos.max_runup_pct,
            max_drawdown_pct=pos.max_drawdown_pct,
            bars_held=pos.bars_held
        )
        
        self.state.add_trade(trade)
        
        # Remove from position manager
        self.state.position_manager.remove_position(symbol, pos)
        
        emoji = "✅" if pnl_usd > 0 else "❌"
        self.logger.info(
            f"{emoji} EXIT: {symbol} {direction.upper()} @ ${price:.4f} | "
            f"PnL: ${pnl_usd:.2f} | {pnl_pct:.2f}% | {reason}"
        )

    def update_position_stats(self, pos: SinglePosition, current_price: float, data: BandData) -> None:
        """Update position tracking stats."""
        entry_price = pos.entry_price
        if entry_price <= 0:
            return
        
        if pos.direction == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
            
        pos.max_runup_pct = max(pos.max_runup_pct, pnl_pct)
        pos.max_drawdown_pct = min(pos.max_drawdown_pct, pnl_pct)
        
        bar_str = data.bar_time.isoformat()
        if pos.last_processed_bar != bar_str:
            pos.bars_held += 1
            pos.last_processed_bar = bar_str


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class DataManager:
    """Manages data buffers."""
    
    def __init__(self, client: BybitClient, symbols: List[str]):
        self.client = client
        self.symbols = symbols
        self.array_size = 300
        self.data_buffers: Dict[str, np.ndarray] = {
            s: np.full((self.array_size, 5), np.nan, dtype=np.float64) for s in symbols
        }
        self.counts: Dict[str, int] = {s: 0 for s in symbols}
        self.head: Dict[str, int] = {s: 0 for s in symbols}
        self.locks: Dict[str, Lock] = defaultdict(Lock)
        self.logger = logging.getLogger("deviation_magnet")
        self._tf_minutes = int(client.config.timeframe)
        self._gap_threshold = self._tf_minutes * 5 * 60.0
        
    def initialize(self):
        """Fetch initial history via REST."""
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
                        opens = df["open"].values
                        highs = df["high"].values
                        lows = df["low"].values
                        closes = df["close"].values
                        times = df["open_time"].astype('int64').values / 10**9
                        
                        data = np.column_stack((opens, highs, lows, closes, times))
                        
                        length = len(data)
                        if length >= self.array_size:
                            self.data_buffers[sym][:] = data[-self.array_size:]
                            self.counts[sym] = self.array_size
                            self.head[sym] = self.array_size - 1
                        else:
                            self.data_buffers[sym][:length] = data
                            self.counts[sym] = length
                            self.head[sym] = length - 1
        
        count = sum(1 for c in self.counts.values() if c > 0)
        self.logger.info(f"Initialized history for {count}/{len(self.symbols)} symbols")

    def on_kline_update(self, message: dict):
        """Handle WS kline update."""
        if not (kline_data := message.get("data")):
            return

        topic = message.get("topic", "")
        if not topic.startswith("kline."):
            return
        symbol = topic.rsplit(".", 1)[-1]

        if symbol not in self.data_buffers:
            return
        
        buffer = self.data_buffers[symbol]
        symbol_lock = self.locks[symbol]
        array_size = self.array_size
        gap_threshold = self._gap_threshold

        for kline in kline_data:
            ts_float = int(kline["start"]) * 0.001
            
            with symbol_lock:
                count = self.counts[symbol]
                
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
                        self.logger.warning(f"⚠️ Data Gap for {symbol}: {time_diff/60:.1f}m. Resetting.")
                        buffer.fill(np.nan)
                        buffer[0] = row
                        self.counts[symbol] = 1
                        self.head[symbol] = 0
                    else:
                        new_head = (head_idx + 1) % array_size
                        buffer[new_head] = row
                        self.head[symbol] = new_head
                        self.counts[symbol] = min(count + 1, array_size)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest close price."""
        buffer = self.data_buffers.get(symbol)
        count = self.counts.get(symbol, 0)
        if buffer is not None and count > 0:
            head = self.head.get(symbol, 0)
            return float(buffer[head, 3])
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# BOT CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class Bot:
    """Main controller."""
    
    _VALID_STATUSES = frozenset({"Trading", "PreLaunch"})
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
        self._timeframe_int = int(self.config.timeframe)
        self._queue_overflow_warned = False

    def run(self):
        """Start the bot."""
        self.state.load()
        self.state.load_trades()
        self._print_header()

        self.logger.info("🔄 Starting...")
        self.data_manager.initialize()
        self._setup_websockets()

        print("⚡ Real-time processing started. Press Ctrl+C to stop.", flush=True)
        self._main_loop()

    def _handle_ws_message(self, message):
        try:
            self.data_manager.on_kline_update(message)
            topic = message.get("topic", "")
            if topic.startswith("kline."):
                symbol = topic.rsplit(".", 1)[-1]
                self.last_update_time[symbol] = time.time()
                self.event_queue.put(symbol)
        except Exception as e:
            self.logger.error(f"WS Handler Error: {e}")

    def _setup_websockets(self):
        """Connect to WebSockets."""
        self.logger.info("🔌 Connecting to WebSockets...")
        for symbol in self.symbols:
            self.client.ws.kline_stream(
                interval=self._timeframe_int,
                symbol=symbol,
                callback=self._handle_ws_message
            )

    def _main_loop(self):
        """Main event loop."""
        last_status_time = time.time()
        last_daily_date = datetime.now(timezone.utc).date()

        try:
            while True:
                try:
                    for _ in range(50):
                        symbol = self.event_queue.get(timeout=0.01)
                        try:
                            self._process_symbol_event(symbol)
                        except Exception as e:
                            self.logger.error(f"Error processing {symbol}: {e}")
                        self.event_queue.task_done()
                except queue.Empty:
                    pass

                now = time.time()
                if now - last_status_time >= self.config.check_interval:
                    self._print_status()
                    self.state.cleanup_signals()
                    self._check_watchdog()
                    self._check_queue_health()
                    last_status_time = now
                
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

    def _check_queue_health(self):
        """Monitor queue health."""
        queue_size = self.event_queue.qsize()
        if queue_size > self.config.max_queue_size:
            if not self._queue_overflow_warned:
                self.logger.warning(f"⚠️ Queue backpressure: {queue_size} items")
                self._queue_overflow_warned = True
        elif self._queue_overflow_warned and queue_size < self.config.max_queue_size // 2:
            self.logger.info(f"✅ Queue resolved: {queue_size} items")
            self._queue_overflow_warned = False

    def _check_watchdog(self):
        """Check for stale symbols."""
        now = time.time()
        for symbol in list(self.symbols):
            last = self.last_update_time.get(symbol, now)
            diff = now - last
            if diff > self.watchdog_threshold:
                self.logger.warning(f"⚠️ WATCHDOG: No data for {symbol} in {diff:.1f}s!")
                
                status = self.client.check_symbol_status(symbol)
                
                if status not in self._VALID_STATUSES:
                    self.logger.warning(f"🚨 {symbol} status: '{status}'. Closing positions.")
                    
                    positions = self.state.position_manager.close_all_positions(symbol)
                    if positions:
                        price = self.client.get_current_price(symbol) or positions[0].entry_price
                        for pos in positions:
                            self.executor._execute_single_exit(symbol, pos, price, "delisted_emergency", is_maker=False)
                    
                    if symbol in self.symbols:
                        self.symbols.remove(symbol)
                    continue

                self.reconnect_counts[symbol] = self.reconnect_counts.get(symbol, 0) + 1
                
                try:
                    self.client.ws.kline_stream(
                        interval=self._timeframe_int,
                        symbol=symbol,
                        callback=self._handle_ws_message
                    )
                    self.last_update_time[symbol] = now
                except Exception as e:
                    self.logger.error(f"Failed to re-subscribe {symbol}: {e}")

    def _process_symbol_event(self, symbol: str):
        """Process strategy for a symbol."""
        with self.data_manager.locks[symbol]:
            buffer = self.data_manager.data_buffers.get(symbol)
            if buffer is None:
                return
            count = self.data_manager.counts.get(symbol, 0)
            head = self.data_manager.head.get(symbol, 0)
            array_size = self.data_manager.array_size
            
            data = self.strategy.calculate_bands_fast(buffer, count, head, array_size)
        
        if not data:
            return
        
        current_price = data.close
        pm = self.state.position_manager
        
        # 1. Update stats for all positions
        for pos in pm.get_positions(symbol):
            self.executor.update_position_stats(pos, current_price, data)
        
        # 2. Check TP for each position
        for pos in pm.get_positions(symbol):
            should_tp, tp_price = self.strategy.check_tp(pos, data)
            if should_tp:
                self.executor.execute_tp_exit(symbol, pos, tp_price)
        
        # 3. Check for new signal
        signal = self.strategy.check_signal(symbol, data, self.state)
        if not signal:
            return
        
        current_direction = pm.get_direction(symbol)
        
        # 4. Handle signal based on current positions
        if current_direction is None:
            # No positions - open new
            entry_price = data.lower3 if signal == "long" else data.upper3
            self.executor.execute_entry(symbol, signal, entry_price, data.bar_time)
        
        elif current_direction == signal:
            # Same direction - add position if under limit
            if pm.can_add_position(symbol, signal):
                entry_price = data.lower3 if signal == "long" else data.upper3
                self.executor.execute_entry(symbol, signal, entry_price, data.bar_time)
        
        else:
            # OPPOSITE SIGNAL - close all positions, then open new
            positions_to_close = pm.close_all_positions(symbol)
            if positions_to_close:
                # Exit at opposite band (stop loss price)
                exit_price = data.upper3 if signal == "long" else data.lower3
                self.executor.execute_opposite_signal_exit(symbol, positions_to_close, exit_price)
            
            # Open new position in new direction
            entry_price = data.lower3 if signal == "long" else data.upper3
            self.executor.execute_entry(symbol, signal, entry_price, data.bar_time)
        
        self.state.save()

    def _daily_report(self, date):
        """Daily summary."""
        date_str = str(date)
        trades = [t for t in self.state.trades if t.exit_time.startswith(date_str)]
        if trades:
            total = sum(t.pnl_usd for t in trades)
            wins = sum(1 for t in trades if t.pnl_usd > 0)
            self.logger.info(f"📊 {date_str}: {len(trades)} trades | Wins: {wins} | PnL: ${total:.2f}")

    def _print_header(self):
        c = self.config
        print(f"\n{'═' * 70}")
        print(f"  🚀 DEVIATION MAGNET v{__version__} - MULTI-POSITION MODE")
        print(f"{'═' * 70}")
        print(f"  Timeframe:      {c.timeframe}m")
        print(f"  Symbols:        {len(self.symbols)}")
        print(f"  BB Settings:    Length={c.bb_length}, Mult={c.mult} (Sigma={c.mult * c.dev_mult:.2f})")
        print(f"  Position Size:  ${c.base_order_size}")
        print(f"  Max Positions:  {c.max_positions_per_direction} per direction per symbol")
        print(f"{'─' * 70}")
        print(f"  Logic:")
        print(f"    - Same direction signal = Add position (up to max)")
        print(f"    - Opposite signal = Close ALL + Open new (stop loss)")
        print(f"    - Volatility TP still active per position")
        print(f"{'═' * 70}\n", flush=True)

    def _print_status(self):
        """Status display."""
        now = datetime.now(timezone.utc)
        session_duration = (now - self.state.start_time).total_seconds() / 3600
        
        sep_double = self._SEP_DOUBLE
        sep_single = self._SEP_SINGLE
        
        print(f"\n{sep_double}")
        print(f"  📊 SESSION STATS")
        print(sep_single)
        print(f"  Runtime: {session_duration:.1f}h | Signals: {self.state.total_signals} | Errors: {self.state.api_errors}")
        
        pm = self.state.position_manager
        all_symbols = pm.get_all_symbols_with_positions()
        
        if all_symbols:
            total_positions = sum(pm.get_position_count(s) for s in all_symbols)
            print(f"\n{sep_single}")
            print(f"  📂 OPEN POSITIONS ({total_positions} across {len(all_symbols)} symbols)")
            print(sep_single)
            
            total_exposure = 0
            total_unrealized = 0
            
            symbol_data = []
            for sym in all_symbols:
                positions = pm.get_positions(sym)
                if not positions:
                    continue
                
                direction = positions[0].direction
                count = len(positions)
                exposure = sum(p.size for p in positions)
                avg_entry = sum(p.entry_price * p.size for p in positions) / exposure
                
                current = self.data_manager.get_latest_price(sym)
                if current:
                    if direction == "long":
                        pnl_pct = (current - avg_entry) / avg_entry * 100
                    else:
                        pnl_pct = (avg_entry - current) / avg_entry * 100
                    
                    # Estimate unrealized PnL
                    price_ratio = current / avg_entry
                    entry_fee = exposure * self.config.entry_cost_pct
                    exit_fee = exposure * price_ratio * self.config.exit_taker_cost_pct
                    gross_pnl = pnl_pct * 0.01 * exposure
                    unrealized = gross_pnl - entry_fee - exit_fee
                    
                    total_exposure += exposure
                    total_unrealized += unrealized
                    
                    symbol_data.append((sym, direction, count, exposure, pnl_pct, unrealized))
            
            # Sort by PnL
            symbol_data.sort(key=lambda x: x[4])
            
            for sym, direction, count, exposure, pnl_pct, unrealized in symbol_data:
                emoji = "🟢" if unrealized >= 0 else "🔴"
                dir_emoji = "📈" if direction == "long" else "📉"
                print(f"  {emoji} {sym:<12} {dir_emoji} {direction.upper():<5} | "
                      f"{count:>3} pos | ${exposure:>7.0f} | "
                      f"{pnl_pct:>+6.2f}% (${unrealized:>+7.2f})")
            
            print(sep_single)
            print(f"  💰 Total Exposure: ${total_exposure:,.0f}")
            print(f"  💵 Unrealized PnL: ${total_unrealized:>+,.2f}")
        else:
            print(f"\n{sep_single}")
            print(f"  📂 OPEN POSITIONS: None")
        
        # Realized PnL
        print(f"\n{sep_double}")
        print(f"  📈 REALIZED PNL")
        print(sep_double)
        
        if self.state.trades:
            total_realized = 0.0
            win_count = 0
            tp_exits = 0
            opposite_exits = 0
            
            for t in self.state.trades:
                total_realized += t.pnl_usd
                if t.pnl_usd > 0:
                    win_count += 1
                if t.exit_reason == "volatility_tp":
                    tp_exits += 1
                elif t.exit_reason == "opposite_signal":
                    opposite_exits += 1
            
            trade_count = len(self.state.trades)
            win_rate = win_count / trade_count * 100 if trade_count > 0 else 0
            
            print(f"  Total Trades: {trade_count} | Wins: {win_count} | Win Rate: {win_rate:.1f}%")
            print(f"  Exit Types: TP={tp_exits} | Opposite Signal={opposite_exits} | Other={trade_count - tp_exits - opposite_exits}")
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
