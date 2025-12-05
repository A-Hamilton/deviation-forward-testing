"""
Deviation Magnet Live Production Bot
====================================

Live trading bot for the Deviation Magnet strategy on Bybit.
Risk Management:
- 1x Leverage
- Max 100 Positions (50 unique symbols × DCA)
- Fixed $10 Position Size
- Limit Orders for All Order Types (Entry, Exit, TP)
- Predictive Entry/Exit at Bollinger Bands

Usage:
    python deviation_magnet_live.py
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock, Lock, Thread
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
from pybit.unified_trading import HTTP, WebSocket
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix for Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Use orjson if available
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

__version__ = "1.10.0-LIVE"


def _safe_error(e: Exception) -> str:
    """Convert exception to ASCII-safe string for Windows console."""
    return str(e).encode('ascii', 'replace').decode('ascii')


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Trading configuration parameters."""

    # API Credentials
    api_key: str = field(default_factory=lambda: os.environ.get("BYBIT_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.environ.get("BYBIT_API_SECRET", ""))
    
    # Strategy Settings (Optimized from 527 symbols × 78 days backtest)
    # Winner: bb=7, mult=3.0, vol=0.20% -> 76% ROI, 91.5% win rate, 151 max positions
    bb_length: int = 7
    mult: float = 3.0
    dev_mult: float = 1.5
    timeframe: str = "1"
    
    # Risk Management ($20 account: 10 × $10 ÷ 5x = $20 margin)
    leverage: int = 5
    position_size_usd: float = 10.0
    max_positions_global: int = 10  # Total positions (allows ~2 DCAs per symbol)
    max_unique_symbols: int = 5     # Max different symbols at once
    
    # Fee structure (for calculation only, real fees from exchange)
    fee_pct: float = 0.00055
    maker_fee_pct: float = 0.0002
    slippage_pct: float = 0.0005
    base_profit_pct: float = 0.1
    
    # Volatility filter (0.20% filters noise while maintaining 76% ROI)
    min_volatility_pct: float = 0.20
    
    # Entry order buffer (% beyond band to improve fill rate)
    entry_buffer_pct: float = 0.05  # 0.05% buffer for limit entries
    
    # Predictive entry settings
    band_proximity_pct: float = 0.3  # Place order when within 0.3% of band
    band_exit_pct: float = 0.5  # Cancel order when price moves 0.5% away from band
    
    # Timing constants
    ws_health_timeout_sec: float = 30.0  # Warn if no WS data for this long
    stale_data_timeout_sec: float = 120.0  # Skip symbol if no update for this long
    signal_cleanup_threshold: int = 5000  # Cleanup signals_seen when exceeding this

    # System
    max_workers: int = 10
    max_queue_size: int = 10000
    data_dir: Path = field(default_factory=lambda: Path("live_data"))
    
    # Calculated fields
    total_fee_pct_for_tp: float = field(init=False)

    def __post_init__(self):
        if not self.api_key or not self.api_secret: raise ValueError("API_KEY and API_SECRET must be set")
        assert self.bb_length >= 2, "bb_length must be >= 2"
        assert self.mult > 0, "mult must be positive"
        assert self.dev_mult > 0, "dev_mult must be positive"
        assert self.position_size_usd > 0, "position_size_usd must be positive"
        assert self.max_positions_global >= 1, "max_positions_global must be >= 1"
        assert self.min_volatility_pct >= 0, "min_volatility_pct cannot be negative"
        self.total_fee_pct_for_tp = self.base_profit_pct + (self.fee_pct + self.maker_fee_pct + self.slippage_pct * 2) * 100

    @property
    def state_file(self) -> Path: return self.data_dir / "live_state.json"
    @property
    def log_file(self) -> Path: return self.data_dir / "live_bot.log"


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(config: Config) -> logging.Logger:
    log = logging.getLogger("deviation_magnet_live")
    if log.handlers: return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    log.addHandler(console)
    try:
        config.data_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(config.log_file)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except Exception as e:
        print(f"Warning: Could not create file handler: {e}", flush=True)
    return log


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SinglePosition:
    """Live position tracking with thread-safe updates."""
    symbol: str
    direction: str
    entry_price: float
    size: float
    entry_time: datetime
    entry_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    tp_price: Optional[float] = None
    close_order_id: Optional[str] = None
    is_closing: bool = False
    is_entry_filled: bool = False
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)
    
    def update_fill(self, price: float, size: float) -> None:
        with self._lock: self.entry_price, self.size, self.is_entry_filled = price, size, True
    
    def update_tp(self, order_id: str, price: float) -> None:
        with self._lock: self.tp_order_id, self.tp_price = order_id, price
    
    def get_entry_info(self) -> Tuple[float, float, str]:
        with self._lock: return self.entry_price, self.size, self.direction
    
    def start_closing(self, close_order_id: str) -> None:
        with self._lock: self.is_closing, self.close_order_id = True, close_order_id
    
    def is_position_closing(self) -> bool:
        with self._lock: return self.is_closing
    
    def has_entry_filled(self) -> bool:
        with self._lock: return self.is_entry_filled


@dataclass(slots=True)
class BandData:
    close: float
    upper3: float
    lower3: float
    bar_time: datetime
    avg_volatility_pct: float
    current_price: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class PositionManager:
    """
    Manages live positions with DCA support.
    Enforces:
    - max_positions_global: Total positions across all symbols (100)
    - max_unique_symbols: Max different symbols at once (50)
    """
    __slots__ = ('config', 'positions', '_lock', '_total_count')
    
    def __init__(self, config: Config):
        self.config = config
        self.positions: Dict[str, List[SinglePosition]] = {}  # symbol -> list of positions (DCA)
        self._lock = RLock()
        self._total_count: int = 0
    
    def get_total_count(self) -> int:
        with self._lock: return self._total_count
    
    def get_symbol_position_count(self, symbol: str) -> int:
        with self._lock: return len(self.positions.get(symbol, ()))
    
    def can_open_new_symbol(self) -> bool:
        with self._lock: return self._total_count < self.config.max_positions_global and len(self.positions) < self.config.max_unique_symbols
    
    def can_add_to_symbol(self, symbol: str) -> bool:
        with self._lock: return symbol in self.positions and self._total_count < self.config.max_positions_global
    
    def get_position(self, symbol: str) -> Optional[SinglePosition]:
        with self._lock:
            p = self.positions.get(symbol, [])
            return p[-1] if p else None
    
    def get_all_positions(self, symbol: str) -> List[SinglePosition]:
        with self._lock: return list(self.positions.get(symbol, []))
    
    def get_symbol_direction(self, symbol: str) -> Optional[str]:
        with self._lock:
            p = self.positions.get(symbol, [])
            return p[0].direction if p else None
    
    def add_position(self, pos: SinglePosition) -> None:
        with self._lock:
            if self._total_count >= self.config.max_positions_global: raise RuntimeError("Global position limit!")
            if pos.symbol not in self.positions:
                if len(self.positions) >= self.config.max_unique_symbols: raise RuntimeError("Unique symbol limit!")
                self.positions[pos.symbol] = []
            self.positions[pos.symbol].append(pos)
            self._total_count += 1
    
    def remove_position(self, symbol: str, order_id: Optional[str] = None) -> None:
        with self._lock:
            if symbol not in self.positions: return
            if order_id is None:
                self._total_count -= len(self.positions[symbol])
                del self.positions[symbol]
            else:
                old_len = len(self.positions[symbol])
                self.positions[symbol] = [p for p in self.positions[symbol] if p.entry_order_id != order_id and p.tp_order_id != order_id and p.close_order_id != order_id]
                self._total_count -= old_len - len(self.positions[symbol])
                if not self.positions[symbol]: del self.positions[symbol]
    
    def remove_if_unfilled(self, symbol: str, order_id: str) -> bool:
        """Atomically remove position only if entry hasn't filled. Returns True if removed."""
        with self._lock:
            if symbol not in self.positions:
                return False
            
            for i, pos in enumerate(self.positions[symbol]):
                if pos.entry_order_id == order_id and not pos.is_entry_filled:
                    self.positions[symbol].pop(i)
                    self._total_count -= 1
                    if not self.positions[symbol]:
                        del self.positions[symbol]
                    return True
            return False

    def to_dict(self) -> dict:
        with self._lock:
            return {sym: [{"symbol": p.symbol, "direction": p.direction, "entry_price": p.entry_price, "size": p.size,
                          "entry_time": p.entry_time.isoformat(), "entry_order_id": p.entry_order_id, "tp_order_id": p.tp_order_id,
                          "tp_price": p.tp_price, "close_order_id": p.close_order_id, "is_closing": p.is_closing, "is_entry_filled": p.is_entry_filled}
                         for p in positions] for sym, positions in self.positions.items()}

    def from_dict(self, data: dict) -> None:
        with self._lock:
            self.positions.clear()
            self._total_count = 0
            for sym, plist in data.items():
                plist = [plist] if isinstance(plist, dict) else plist  # Handle old format
                self.positions[sym] = []
                for d in plist:
                    et = datetime.fromisoformat(d["entry_time"])
                    self.positions[sym].append(SinglePosition(
                        symbol=d["symbol"], direction=d["direction"], entry_price=d["entry_price"], size=d["size"],
                        entry_time=et if et.tzinfo else et.replace(tzinfo=timezone.utc), entry_order_id=d.get("entry_order_id"),
                        tp_order_id=d.get("tp_order_id"), tp_price=d.get("tp_price"), close_order_id=d.get("close_order_id"),
                        is_closing=d.get("is_closing", False), is_entry_filled=d.get("is_entry_filled", False)))
                self._total_count += len(self.positions[sym])


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING STATE
# ═══════════════════════════════════════════════════════════════════════════════

class TradingState:
    """Manages persistent state with optimized memory layout."""
    __slots__ = ('config', 'position_manager', 'signals_seen',
                 'logger', 'save_lock', '_save_queue', '_save_worker_thread')
    
    def __init__(self, config: Config):
        self.config = config
        self.position_manager = PositionManager(config)
        self.signals_seen: Set[str] = set()
        self.logger = logging.getLogger("deviation_magnet_live")
        self.save_lock = Lock()
        
        # Debounced save worker (prevents thread explosion)
        self._save_queue: queue.Queue = queue.Queue()
        self._save_worker_thread = Thread(target=self._save_worker, daemon=True)
        self._save_worker_thread.start()
        
    def save(self) -> None:
        """Queue a save request (non-blocking, debounced)."""
        try:
            self._save_queue.put_nowait(True)
        except queue.Full:
            pass  # Already queued

    def _save_worker(self) -> None:
        """Background worker that debounces save requests."""
        while True:
            try:
                self._save_queue.get()
                while not self._save_queue.empty():  # Drain pending
                    try: self._save_queue.get_nowait()
                    except queue.Empty: break
                self._save_sync()
            except Exception as e:
                self.logger.error(f"Save worker error: {e}")

    def _save_sync(self) -> None:
        try:
            with self.save_lock:
                state_data = {
                    "positions": self.position_manager.to_dict(),
                }
                temp_file = self.config.state_file.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(json_dumps(state_data, indent=2))
                temp_file.replace(self.config.state_file)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def load(self) -> None:
        if not self.config.state_file.exists():
            return
        try:
            with open(self.config.state_file, "r", encoding="utf-8") as f:
                state_data = json_loads(f.read())
            
            if "positions" in state_data:
                self.position_manager.from_dict(state_data["positions"])
                
            self.logger.info(f"Restored state. Open positions: {self.position_manager.get_total_count()}")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")

    def cleanup_signals(self) -> None:
        """Remove signals older than 5 minutes."""
        if len(self.signals_seen) <= self.config.signal_cleanup_threshold: return
        cutoff = time.time() - 300
        self.signals_seen = {k for k in self.signals_seen if (idx := k.rfind('_')) == -1 or (float(k[idx+1:]) > cutoff if k[idx+1:].replace('.','',1).isdigit() else True)}


# ═══════════════════════════════════════════════════════════════════════════════
# BYBIT CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class BybitClient:
    """Authenticated Bybit Client."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("deviation_magnet_live")
        
        self.session = HTTP(
            testnet=False,
            api_key=config.api_key,
            api_secret=config.api_secret
        )
        
        self.ws = WebSocket(
            testnet=False,
            channel_type="linear",
        )
        
        self.private_ws = WebSocket(
            testnet=False,
            channel_type="private",
            api_key=config.api_key,
            api_secret=config.api_secret
        )

    def fetch_history_rest(self, symbol: str, limit: int = 250) -> Optional[np.ndarray]:
        """Fetch kline history as numpy array (N, 5): [open, high, low, close, timestamp]."""
        try:
            resp = self.session.get_kline(category="linear", symbol=symbol, interval=self.config.timeframe, limit=limit)
            if resp["retCode"] != 0: return None
            raw = resp["result"]["list"]
            # raw is [[ts, o, h, l, c, vol, turnover], ...] in DESC order
            arr = np.array([[float(r[1]), float(r[2]), float(r[3]), float(r[4]), int(r[0]) * 0.001] for r in reversed(raw)])
            return arr
        except Exception as e:
            self.logger.error(f"History fetch failed for {symbol}: {_safe_error(e)}")
            return None

    def fetch_active_symbols(self) -> List[str]:
        """Fetch only fully active USDT perpetuals with pagination (exclude PreLaunch/Delivering)."""
        symbols = []
        cursor = ""
        
        while True:
            try:
                resp = self.session.get_instruments_info(
                    category="linear",
                    limit=1000,  # Max limit per request
                    cursor=cursor if cursor else None
                )
                if resp["retCode"] != 0: 
                    break
                
                # Process this page of symbols
                for i in resp["result"]["list"]:
                    status = i.get("status", "")
                    quote = i.get("quoteCoin", "")
                    sym = i.get("symbol", "")
                    
                    # Must be Trading status, USDT quote, and not in any special state
                    if status == "Trading" and quote == "USDT" and sym.endswith("USDT"):
                        # Check if symbol is being delisted (deliveryTime set)
                        delivery_time = i.get("deliveryTime", "0")
                        if delivery_time == "0" or delivery_time == "":
                            symbols.append(sym)
                
                # Check for next page
                cursor = resp["result"].get("nextPageCursor", "")
                if not cursor:
                    break
                    
            except Exception as e:
                self.logger.error(f"Symbol fetch failed: {_safe_error(e)}")
                break
        
        self.logger.info(f"Found {len(symbols)} active USDT pairs (filtered delisting)")
        return symbols

    def set_leverage(self, symbol: str, leverage: int = 1) -> None:
        try:
            self.session.set_leverage(category="linear", symbol=symbol, buyLeverage=str(leverage), sellLeverage=str(leverage))
        except Exception:
            pass

    def place_limit_order(self, symbol: str, side: str, qty: float, price: float, reduce_only: bool = False, 
                          qty_decimals: int = 0, price_decimals: int = 2, time_in_force: str = "PostOnly",
                          take_profit: Optional[float] = None) -> Optional[str]:
        try:
            qty_str = str(int(qty)) if qty_decimals == 0 else f"{qty:.{qty_decimals}f}"
            order_params = {
                "category": "linear", "symbol": symbol, "side": side.capitalize(), "orderType": "Limit",
                "qty": qty_str, "price": f"{price:.{price_decimals}f}", "timeInForce": time_in_force, "reduceOnly": reduce_only
            }
            # Add built-in TP if specified (Bybit handles partial fills automatically)
            if take_profit is not None:
                tp_str = f"{take_profit:.{price_decimals}f}"
                order_params.update({
                    "takeProfit": tp_str,
                    "tpTriggerBy": "LastPrice",
                    "tpslMode": "Partial",       # Auto-adjusts qty on partial fills
                    "tpOrderType": "Limit",
                    "tpLimitPrice": tp_str
                })
            resp = self.session.place_order(**order_params)
            if resp["retCode"] == 0:
                return resp["result"]["orderId"]
            self.logger.error(f"Limit order failed: {resp}")
        except Exception as e:
            self.logger.error(f"Place limit exception: {_safe_error(e)}")
        return None

    def amend_order(self, symbol: str, order_id: str, price: float, price_decimals: int = 8, 
                    take_profit: Optional[float] = None) -> bool:
        try:
            params = {"category": "linear", "symbol": symbol, "orderId": order_id, "price": f"{price:.{price_decimals}f}"}
            if take_profit is not None:
                tp_str = f"{take_profit:.{price_decimals}f}"
                params.update({"takeProfit": tp_str, "tpLimitPrice": tp_str})
            return self.session.amend_order(**params)["retCode"] == 0
        except Exception as e:
            self.logger.error(f"Amend failed: {_safe_error(e)}")
            return False

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            return self.session.cancel_order(category="linear", symbol=symbol, orderId=order_id)["retCode"] == 0
        except Exception as e:
            self.logger.error(f"Cancel failed: {_safe_error(e)}")
            return False

    def get_instrument_info(self, symbol: str) -> Optional[dict]:
        try:
            resp = self.session.get_instruments_info(category="linear", symbol=symbol)
            if resp["retCode"] == 0 and resp["result"]["list"]:
                return resp["result"]["list"][0]
        except Exception:
            pass
        return None

    def fetch_positions(self) -> List[dict]:
        try:
            resp = self.session.get_positions(category="linear", settleCoin="USDT")
            return [p for p in resp["result"]["list"] if float(p["size"]) > 0] if resp["retCode"] == 0 else []
        except Exception as e:
            self.logger.error(f"Fetch positions failed: {_safe_error(e)}")
            return []

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        try:
            kw = {"category": "linear", "openOnly": 1, **({"symbol": symbol} if symbol else {"settleCoin": "USDT"})}
            resp = self.session.get_open_orders(**kw)
            return resp["result"]["list"] if resp["retCode"] == 0 else []
        except Exception as e:
            self.logger.error(f"Fetch orders failed: {_safe_error(e)}")
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

class DeviationMagnetStrategy:
    __slots__ = ('_indices_template', '_mult', '_dev_mult', '_bb_length', '_fee_pct_decimal')
    
    def __init__(self, config: Config):
        # Pre-allocate index array for band calculation
        self._indices_template = np.arange(config.bb_length)
        # Cache frequently used values
        self._mult = config.mult
        self._dev_mult = config.dev_mult
        self._bb_length = config.bb_length
        self._fee_pct_decimal = config.total_fee_pct_for_tp / 100.0

    def calculate_bands(self, buffer: np.ndarray, count: int, head: int, array_size: int) -> Optional[BandData]:
        bb_length = self._bb_length
        # Need bb_length+1 bars: bb_length closed bars + 1 current forming bar
        if count < bb_length + 1:
            return None
        
        # Get the last bb_length CLOSED bars (exclude current forming bar at head)
        if count < array_size:
            # Buffer not wrapped yet - closed bars are [count-bb_length-1 : count-1]
            start_idx = count - bb_length - 1
            subset = buffer[start_idx : count - 1]
            current_bar = buffer[count - 1]  # Current forming bar
        else:
            # Circular buffer - get bb_length bars ending at head-1 (last closed bar)
            last_closed = (head - 1) % array_size
            base = last_closed - bb_length + 1
            indices = (self._indices_template + base) % array_size
            subset = buffer[indices]
            current_bar = buffer[head]  # Current forming bar
        
        # Early exit if data invalid
        if np.isnan(subset[0, 0]):
            return None
        
        # Vectorized OHLC4 calculation: (open + high + low + close) / 4
        ohlc4 = (subset[:, 0] + subset[:, 1] + subset[:, 2] + subset[:, 3]) * 0.25
        
        basis = ohlc4.mean()
        stdev = ohlc4.std(ddof=1)
        dev = self._mult * stdev
        dev_scaled = dev * self._dev_mult
        
        # Get volatility - vectorized
        highs = subset[:, 1]
        lows = subset[:, 2]
        closes = subset[:, 3]
        avg_vol = ((highs - lows) / closes).mean() * 100
        
        # Use last CLOSED bar for high/low/close
        last_bar = subset[-1]
        
        return BandData(
            close=last_bar[3],
            upper3=basis + dev_scaled,
            lower3=basis - dev_scaled,
            bar_time=datetime.fromtimestamp(current_bar[4], tz=timezone.utc),
            avg_volatility_pct=avg_vol
        )

    def calculate_tp_price(self, entry_price: float, direction: str, volatility_pct: float) -> float:
        """
        Calculate TP price using volatility-adjusted formula.
        
        Formula: TP = entry * (1 ± target_pct)
        Where target_pct = base_fee_pct + (volatility / 2)
        """
        # volatility_pct is already in % (e.g., 0.5 means 0.5%)
        # Convert to decimal and divide by 2: (0.5 / 100) / 2 = 0.0025
        target_pct = self._fee_pct_decimal + (volatility_pct * 0.005)
        
        if direction == "long":
            return entry_price * (1.0 + target_pct)
        return entry_price * (1.0 - target_pct)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class TradeExecutor:
    """Executes trades on Bybit."""
    __slots__ = ('state', 'client', 'logger', 'instrument_cache', '_strategy')
    
    def __init__(self, config: Config, state: TradingState, client: BybitClient):
        self.state = state
        self.client = client
        self.logger = logging.getLogger("deviation_magnet_live")
        self.instrument_cache: Dict[str, Tuple[int, int, float]] = {}
        self._strategy = DeviationMagnetStrategy(config)  # Cache instance

    def _get_qty_precision(self, symbol: str) -> Tuple[int, int, float]:
        """Returns (qty_decimals, price_decimals, min_qty)."""
        if symbol not in self.instrument_cache:
            info = self.client.get_instrument_info(symbol)
            if info:
                qty_step = info["lotSizeFilter"]["qtyStep"]
                price_tick = info["priceFilter"]["tickSize"]
                min_qty = float(info["lotSizeFilter"].get("minOrderQty", "0"))
                
                # Calculate decimals
                qty_decimals = 0
                if "." in qty_step:
                    qty_decimals = len(qty_step.split(".")[1].rstrip('0'))
                
                price_decimals = 0
                if "." in price_tick:
                    price_decimals = len(price_tick.split(".")[1].rstrip('0'))
                    
                self.instrument_cache[symbol] = (qty_decimals, price_decimals, min_qty)
            else:
                # Cache fallback to prevent repeated failed API calls
                self.instrument_cache[symbol] = (3, 2, 0.0)
        return self.instrument_cache[symbol]

    def ensure_tp_exists(self, pos: SinglePosition, data: BandData) -> None:
        """Ensure TP order exists for a filled position. TP is set once at entry."""
        if pos.tp_order_id:
            return
        entry_price, _, _ = pos.get_entry_info()
        if entry_price > 0:
            self._place_initial_tp(pos, data)

    def _place_initial_tp(self, pos: SinglePosition, data: BandData) -> None:
        entry_price, size, direction = pos.get_entry_info()
        tp_price = self._strategy.calculate_tp_price(entry_price, direction, data.avg_volatility_pct)
        qty_dec, price_dec, _ = self._get_qty_precision(pos.symbol)
        tp_price = round(tp_price, price_dec)
        
        # Re-round qty to ensure it matches instrument's qtyStep
        size = round(size, qty_dec)
        if size <= 0:
            self.logger.error(f"[TP] {pos.symbol} size rounded to 0, skipping TP")
            return
        
        side = "Sell" if direction == "long" else "Buy"
        
        # Use GTC for TP orders (PostOnly can be rejected if price crosses)
        oid = self.client.place_limit_order(pos.symbol, side, size, tp_price, reduce_only=True, 
                                             qty_decimals=qty_dec, price_decimals=price_dec,
                                             time_in_force="GTC")
        if oid:
            pos.update_tp(oid, tp_price)
            self.logger.info(f"[TP PLACED] {pos.symbol} @ {tp_price}")
            self.state.save()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class DataManager:
    """Manages market data with optimized circular buffers."""
    __slots__ = ('client', 'symbols', '_symbol_set', 'array_size', 'data_buffers', 'counts', 'head', 'locks', 'logger')
    
    def __init__(self, client: BybitClient, symbols: List[str]):
        self.client = client
        self.symbols = symbols
        self._symbol_set = frozenset(symbols)  # O(1) lookup
        self.array_size = 300
        self.data_buffers: Dict[str, np.ndarray] = {
            s: np.full((self.array_size, 5), np.nan, dtype=np.float64) for s in symbols
        }
        self.counts: Dict[str, int] = dict.fromkeys(symbols, 0)
        self.head: Dict[str, int] = dict.fromkeys(symbols, 0)
        self.locks: Dict[str, Lock] = {s: Lock() for s in symbols}
        self.logger = logging.getLogger("deviation_magnet_live")
        
    def initialize(self) -> None:
        self.logger.info("Fetching history...")
        failed = []
        with ThreadPoolExecutor(max_workers=self.client.config.max_workers) as executor:
            futures = {executor.submit(self.client.fetch_history_rest, s): s for s in self.symbols}
            for f in as_completed(futures):
                sym = futures[f]
                try:
                    arr = f.result()
                    if arr is not None:
                        with self.locks[sym]:
                            length = len(arr)
                            if length >= self.array_size:
                                self.data_buffers[sym][:] = arr[-self.array_size:]
                                self.counts[sym] = self.array_size
                                self.head[sym] = self.array_size - 1
                            else:
                                self.data_buffers[sym][:length] = arr
                                self.counts[sym] = length
                                self.head[sym] = length - 1
                    else:
                        failed.append(sym)
                except Exception as e:
                    self.logger.error(f"History init failed for {sym}: {_safe_error(e)}")
                    failed.append(sym)
        if failed:
            self.logger.warning(f"Failed to fetch history for {len(failed)} symbols")

    def on_kline_update(self, message: dict) -> None:
        data = message.get("data")
        if not data:
            return
        
        topic = message.get("topic", "")
        # Extract symbol from topic - last part after final dot
        dot_idx = topic.rfind(".")
        if dot_idx == -1:
            return
        symbol = topic[dot_idx + 1:]
        
        if symbol not in self._symbol_set:
            return
        
        buffer = self.data_buffers[symbol]
        lock = self.locks[symbol]
        array_size = self.array_size
        
        with lock:
            head = self.head[symbol]
            count = self.counts[symbol]
            
            for k in data:
                # Inline float conversion - faster than np.array()
                o = float(k["open"])
                h = float(k["high"])
                lo = float(k["low"])
                c = float(k["close"])
                ts = int(k["start"]) * 0.001
                
                if count == 0:
                    buffer[0, 0] = o
                    buffer[0, 1] = h
                    buffer[0, 2] = lo
                    buffer[0, 3] = c
                    buffer[0, 4] = ts
                    count = 1
                    head = 0
                    continue
                
                last_ts = buffer[head, 4]
                if ts == last_ts:
                    # Update current bar in place
                    buffer[head, 0] = o
                    buffer[head, 1] = h
                    buffer[head, 2] = lo
                    buffer[head, 3] = c
                elif ts > last_ts:
                    # New bar - advance head
                    head = (head + 1) % array_size
                    buffer[head, 0] = o
                    buffer[head, 1] = h
                    buffer[head, 2] = lo
                    buffer[head, 3] = c
                    buffer[head, 4] = ts
                    if count < array_size:
                        count += 1
            
            self.head[symbol] = head
            self.counts[symbol] = count


# ═══════════════════════════════════════════════════════════════════════════════
# BOT CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class Bot:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logging(self.config)
        self.state = TradingState(self.config)
        self.client = BybitClient(self.config)
        self.executor = TradeExecutor(self.config, self.state, self.client)
        
        self.symbols = self.client.fetch_active_symbols()
        if not self.symbols: raise ValueError("No symbols found")
        
        self.data_manager = DataManager(self.client, self.symbols)
        self.event_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.last_update = {}
        self.pending_entries: Dict[str, dict] = {}  # symbol -> {bar_time, direction, order_id, limit_price, last_amend_time, is_exit}
        self._pending_entries_lock = Lock()  # Thread safety for pending_entries
        self.current_prices: Dict[str, float] = {}  # symbol -> real-time price from current bar
        self._reversal_entry_queue: queue.Queue = queue.Queue()  # Queue reversal entries after exit fill
        self._amend_cooldown_sec: float = 2.0  # Minimum seconds between order amendments
        self._last_ws_heartbeat: float = time.time()
        self._last_ws_warning: float = 0.0  # Throttle WS health warnings
        self._last_order_verify: float = 0.0
        self._order_verify_interval: float = 60.0  # Verify pending orders every 60s
        self._symbols_with_data: Set[str] = set()  # Track symbols that have received WS data
        
        # Pre-compute config values used in hot paths
        self._exit_pct: float = self.config.band_exit_pct / 100
        self._proximity_pct: float = self.config.band_proximity_pct / 100
        self._buffer_mult: float = 1 + (self.config.entry_buffer_pct / 100)
        
        # Cache strategy reference for hot path (avoid property overhead)
        self._strategy: DeviationMagnetStrategy = self.executor._strategy

    def run(self) -> None:
        self.state.load()
        self.logger.info(">>> STARTING LIVE BOT")
        self.logger.info(f"Config: bb={self.config.bb_length}, mult={self.config.mult}, vol_filter={self.config.min_volatility_pct}%")
        self.logger.info(f"Limits: {self.config.max_unique_symbols} symbols × DCA = {self.config.max_positions_global} max positions")
        self.logger.info(f"Risk: {self.config.leverage}x Lev, ${self.config.position_size_usd} per position")
        
        self._reconcile_state()
        
        self.data_manager.initialize()
        self._setup_websockets()
        
        self.logger.info(f"Monitoring {len(self.symbols)} symbols. Waiting for market data...")
        self._main_loop()

    def _reconcile_state(self) -> None:
        """Sync local state with exchange state."""
        self.logger.info("... Reconciling state with exchange...")
        on_chain_positions = self.client.fetch_positions()
        orders_by_sym = {}
        for o in self.client.fetch_open_orders():
            orders_by_sym.setdefault(o["symbol"], []).append(o)
        
        pm = self.state.position_manager
        on_chain_syms = {p["symbol"] for p in on_chain_positions}
        for sym in list(pm.positions.keys()):
            if sym not in on_chain_syms:
                self.logger.warning(f"[WARN] Position {sym} missing on exchange. Removing.")
                pm.remove_position(sym)
        
        for p in on_chain_positions:
            sym, size, direction = p["symbol"], float(p["size"]), "long" if p["side"] == "Buy" else "short"
            entry_price, local_pos = float(p["avgPrice"]), pm.get_position(sym)
            tp_order_id = tp_price = None
            for o in orders_by_sym.get(sym, []):
                if o["orderType"] == "Limit" and str(o.get("reduceOnly", "false")).lower() == "true":
                    tp_order_id, tp_price = o["orderId"], float(o["price"])
                    break
            if not local_pos:
                self.logger.info(f"[FOUND] Existing position {sym} {direction} {size}")
                pm.add_position(SinglePosition(symbol=sym, direction=direction, entry_price=entry_price, size=size,
                                              entry_time=datetime.now(timezone.utc), tp_order_id=tp_order_id, tp_price=tp_price, is_entry_filled=True))
            else:
                local_pos.update_fill(entry_price, size)
                if tp_order_id: local_pos.update_tp(tp_order_id, tp_price)
        self.state.save()
        self.logger.info("[OK] Reconciliation complete.")

    def _setup_websockets(self) -> None:
        # Public Data - batch subscribe to avoid rate limits
        # Pybit accepts list of symbols for batch subscription
        # Bybit allows up to 500 topics per connection, 50 is safe
        batch_size = 50
        for i in range(0, len(self.symbols), batch_size):
            batch = self.symbols[i:i + batch_size]
            try:
                self.client.ws.kline_stream(
                    interval=int(self.config.timeframe),
                    symbol=batch,
                    callback=self._on_kline
                )
                time.sleep(0.05)  # Small delay between batches
            except Exception as e:
                self.logger.error(f"WS subscription failed for batch: {_safe_error(e)}")
        
        # Private Data (Orders)
        self.client.private_ws.order_stream(callback=self._on_order_update)

    def _on_kline(self, msg) -> None:
        try:
            self.data_manager.on_kline_update(msg)
            topic = msg.get("topic", "")
            if not topic.startswith("kline."):
                return
            
            # Fast symbol extraction
            dot_idx = topic.rfind(".")
            symbol = topic[dot_idx + 1:] if dot_idx != -1 else ""
            if not symbol:
                return
            
            now = time.time()
            self.last_update[symbol] = now
            self._last_ws_heartbeat = now
            self._symbols_with_data.add(symbol)
            
            # Extract real-time price from current forming bar
            data = msg.get("data")
            if data:
                close_price = data[0].get("close")
                if close_price:
                    self.current_prices[symbol] = float(close_price)
            
            # Non-blocking queue put
            try:
                self.event_queue.put_nowait(symbol)
            except queue.Full:
                pass
        except Exception as e:
            self.logger.error(f"WS Error: {_safe_error(e)}")

    def _place_tp_after_fill(self, pos: SinglePosition) -> None:
        """Place TP order after entry fill using current market data."""
        # Guard against duplicate TP placement or pending cancel
        if pos.tp_order_id:
            return
        try:
            with self.data_manager.locks[pos.symbol]:
                buffer = self.data_manager.data_buffers.get(pos.symbol)
                count = self.data_manager.counts.get(pos.symbol, 0)
                head = self.data_manager.head.get(pos.symbol, 0)
                data = self._strategy.calculate_bands(buffer, count, head, self.data_manager.array_size)
            
            if data:
                self.executor._place_initial_tp(pos, data)
        except Exception as e:
            self.logger.error(f"Failed to place TP after fill: {_safe_error(e)}")

    def _on_order_update(self, msg) -> None:
        """Handle order fills (including partial fills) to update position state."""
        try:
            for order in msg.get("data", []):
                symbol = order["symbol"]
                status = order["orderStatus"]
                oid = order["orderId"]
                
                # Find the specific position with this order ID
                all_positions = self.state.position_manager.get_all_positions(symbol)
                if not all_positions:
                    continue
                
                pos = None
                for p in all_positions:
                    if p.entry_order_id == oid or p.tp_order_id == oid or p.close_order_id == oid:
                        pos = p
                        break
                
                if not pos:
                    continue
                
                # Entry Fill or Partial Fill - Bybit manages TP automatically with tpslMode=Partial
                if oid == pos.entry_order_id and status in ["Filled", "PartiallyFilled"]:
                    filled_qty = float(order.get("cumExecQty", 0))
                    avg_price = float(order.get("avgPrice", 0))
                    
                    if filled_qty > 0 and avg_price > 0:
                        pos.update_fill(avg_price, filled_qty)
                        self.logger.info(f"[{'FILLED' if status == 'Filled' else 'PARTIAL'}] ENTRY: {symbol} @ {avg_price} (size: {filled_qty})")
                        
                        # Clear pending entry tracking since we got a fill
                        with self._pending_entries_lock:
                            self.pending_entries.pop(symbol, None)
                    
                    self.state.save()
                
                # Entry order cancelled or rejected (PostOnly rejected, insufficient margin, manual cancel)
                elif oid == pos.entry_order_id and status in ["Cancelled", "Rejected"]:
                    # Use atomic removal to prevent race condition
                    if self.state.position_manager.remove_if_unfilled(symbol, oid):
                        self.logger.info(f"[{status.upper()}] Entry order {symbol} - removing position")
                        with self._pending_entries_lock:
                            self.pending_entries.pop(symbol, None)
                        self.state.save()
                
                # TP Fill - position closed via take profit
                elif oid == pos.tp_order_id and status == "Filled":
                    self.logger.info(f"[FILLED] TP: {symbol}")
                    self.state.position_manager.remove_position(symbol, oid)  # Remove specific position
                    self.state.save()
                
                # Close Order Fill - position closed via signal/manual close
                elif oid == pos.close_order_id and status == "Filled":
                    # When closing all DCA positions with one order, multiple positions share the same close_order_id
                    # Remove ALL positions for this symbol (they all share the close order)
                    count_before = self.state.position_manager.get_symbol_position_count(symbol)
                    self.logger.info(f"[CLOSED] {symbol} ({count_before} positions closed via close order)")
                    self.state.position_manager.remove_position(symbol)  # Remove all for symbol
                    
                    # Check if this was a reversal exit - place entry in new direction
                    with self._pending_entries_lock:
                        pending_info = self.pending_entries.pop(symbol, None)
                    
                    if pending_info and pending_info.get("is_exit") and pending_info.get("direction"):
                        # Queue the reversal entry to be placed
                        new_dir = pending_info["direction"]
                        limit_price = pending_info["limit_price"]
                        bar_time = pending_info["bar_time"]
                        self.logger.info(f"[REVERSAL] {symbol} placing {new_dir} entry @ {limit_price}")
                        try:
                            self._reversal_entry_queue.put_nowait((symbol, new_dir, limit_price, bar_time))
                        except queue.Full:
                            self.logger.error(f"[REVERSAL] {symbol} entry queue full")
                    
                    self.state.save()
                
                # Close Order Cancelled/Rejected - need to handle (PostOnly rejected, etc.)
                elif oid == pos.close_order_id and status in ["Cancelled", "Rejected"]:
                    self.logger.warning(f"[{status.upper()}] Close order {symbol} - will retry")
                    # Clear is_closing flag so we can retry
                    for p in self.state.position_manager.get_all_positions(symbol):
                        if p.close_order_id == oid:
                            with p._lock:
                                p.is_closing = False
                                p.close_order_id = None
                    # Clean up pending entry tracking so it can be retried
                    with self._pending_entries_lock:
                        self.pending_entries.pop(symbol, None)
                    self.state.save()
                    
        except Exception as e:
            self.logger.error(f"Order Update Error: {_safe_error(e)}")

    def _main_loop(self) -> None:
        cleanup_counter = 0
        while True:
            try:
                # Process any pending reversal entries (from WS callbacks)
                self._process_reversal_queue()
                
                symbol = self.event_queue.get(timeout=1.0)
                self._process_symbol(symbol)
                self.event_queue.task_done()
                
                # Periodic cleanup (less frequent)
                cleanup_counter += 1
                if cleanup_counter >= 5000:
                    self.state.cleanup_signals()
                    cleanup_counter = 0
            except queue.Empty:
                # Idle time - perform maintenance tasks
                self._periodic_maintenance()
            except KeyboardInterrupt:
                self.logger.info("Stopping...")
                self._shutdown()
                break
            except Exception as e:
                self.logger.error(f"Loop Error: {_safe_error(e)}")
    
    def _shutdown(self) -> None:
        """Graceful shutdown - close WebSocket connections and save state."""
        try:
            self.logger.info("Closing WebSocket connections...")
            try:
                self.client.ws.exit()
            except Exception:
                pass
            try:
                self.client.private_ws.exit()
            except Exception:
                pass
            self.state._save_sync()  # Force immediate save
            self.logger.info("Shutdown complete.")
        except Exception as e:
            self.logger.error(f"Shutdown error: {_safe_error(e)}")
    
    def _process_reversal_queue(self) -> None:
        """Process reversal entries from WS callbacks (avoids deadlock)."""
        # Process reversal entries (after exit fills)
        reversal_processed = 0
        while reversal_processed < 5:
            try:
                symbol, direction, limit_price, bar_time = self._reversal_entry_queue.get_nowait()
                # Only place if we can open new symbol and no pending order exists
                pm = self.state.position_manager
                with self._pending_entries_lock:
                    has_pending = symbol in self.pending_entries
                
                if not has_pending and pm.can_open_new_symbol():
                    # Get current band data for the entry
                    with self.data_manager.locks[symbol]:
                        buffer = self.data_manager.data_buffers.get(symbol)
                        count = self.data_manager.counts.get(symbol, 0)
                        head = self.data_manager.head.get(symbol, 0)
                        data = self._strategy.calculate_bands(buffer, count, head, self.data_manager.array_size)
                    
                    if data:
                        # Recalculate entry price from current bands (may have moved since exit was placed)
                        if direction == "long":
                            fresh_limit_price = data.lower3 * self._buffer_mult
                        else:
                            fresh_limit_price = data.upper3 / self._buffer_mult
                        
                        data.bar_time = bar_time  # Use original bar time
                        self._place_predictive_order(symbol, direction, fresh_limit_price, data)
                reversal_processed += 1
            except queue.Empty:
                break

    def _periodic_maintenance(self) -> None:
        """Perform periodic maintenance tasks during idle time."""
        now = time.time()
        
        # Process any pending reversal entries
        self._process_reversal_queue()
        
        # Check WS health (warn if no data for configured timeout, throttle warnings)
        ws_silence = now - self._last_ws_heartbeat
        if ws_silence > self.config.ws_health_timeout_sec and self._symbols_with_data and (now - self._last_ws_warning) > self.config.ws_health_timeout_sec:
            self.logger.warning(f"[WS HEALTH] No data received for {ws_silence:.0f}s")
            self._last_ws_warning = now
        
        # Verify pending orders still exist (every 60s)
        if self.pending_entries and (now - self._last_order_verify) > self._order_verify_interval:
            self._verify_pending_orders()
            self._last_order_verify = now
    
    def _verify_pending_orders(self) -> None:
        """Verify pending entry and exit orders still exist on exchange."""
        if not self.pending_entries:
            return
        
        try:
            open_orders = self.client.fetch_open_orders()
            open_order_ids = {o["orderId"] for o in open_orders}
            
            # Check each pending entry/exit (iterate over copy to allow modification)
            symbols_to_remove = []
            with self._pending_entries_lock:
                pending_copy = list(self.pending_entries.items())
            
            for symbol, info in pending_copy:
                order_id = info.get("order_id")
                is_exit = info.get("is_exit", False)
                
                if order_id and order_id not in open_order_ids:
                    # Order no longer exists - was filled or cancelled
                    if is_exit:
                        self.logger.warning(f"[VERIFY] {symbol} exit order {order_id} missing - cleaning up")
                    else:
                        # Use atomic removal to prevent race condition
                        if self.state.position_manager.remove_if_unfilled(symbol, order_id):
                            self.logger.warning(f"[VERIFY] {symbol} entry order {order_id} missing - cleaning up")
                    symbols_to_remove.append(symbol)
            
            if symbols_to_remove:
                with self._pending_entries_lock:
                    for symbol in symbols_to_remove:
                        self.pending_entries.pop(symbol, None)
                self.state.save()
                
        except Exception as e:
            self.logger.error(f"Order verification failed: {_safe_error(e)}")

    def _process_symbol(self, symbol: str) -> None:
        now = time.time()
        
        # Check for stale data (no update in configured timeout)
        last_update = self.last_update.get(symbol, 0)
        if last_update > 0 and (now - last_update) > self.config.stale_data_timeout_sec:
            # Data is stale, skip signal generation
            return
        
        with self.data_manager.locks[symbol]:
            buffer = self.data_manager.data_buffers.get(symbol)
            count = self.data_manager.counts.get(symbol, 0)
            head = self.data_manager.head.get(symbol, 0)
            data = self._strategy.calculate_bands(buffer, count, head, self.data_manager.array_size)
        
        if not data: return
        
        # Add real-time price from current forming bar
        data.current_price = self.current_prices.get(symbol, data.close)
        
        pm = self.state.position_manager
        pos = pm.get_position(symbol)
        
        # Get pending order info (single lock acquisition)
        with self._pending_entries_lock:
            pending_info = self.pending_entries.get(symbol)
        
        # Check for unfilled entry orders that should be cancelled (new bar started)
        if pending_info:
            pending_bar_time = pending_info["bar_time"]
            if data.bar_time > pending_bar_time:
                # New bar started, cancel unfilled entry
                if pos and pos.entry_order_id and not pos.has_entry_filled():
                    self.logger.info(f"[CANCEL] {symbol} entry not filled by bar close")
                    self.client.cancel_order(symbol, pos.entry_order_id)
                with self._pending_entries_lock:
                    self.pending_entries.pop(symbol, None)
                return  # Skip further processing this cycle
        
        # Get current direction for this symbol (if any positions exist)
        current_direction = pm.get_symbol_direction(symbol)
        
        has_pending = pending_info is not None
        is_pending_exit = has_pending and pending_info.get("is_exit", False)
        
        # If we have a pending exit, manage it
        if is_pending_exit:
            self._manage_predictive_order(symbol, data)
            return
        
        # 1. Manage Existing Positions
        if current_direction:
            all_positions = pm.get_all_positions(symbol)
            
            # Skip if any position is being closed (wait for close to complete)
            if any(p.is_position_closing() for p in all_positions):
                return
            
            # TP is now set atomically via Bybit's built-in takeProfit parameter
            # No need to check/place TP here - Bybit manages it
            
            # Check for predictive entry/exit opportunity
            if has_pending:
                self._manage_predictive_order(symbol, data)
            else:
                # Check for predictive entry - will detect opposite direction for exit
                # or same direction for DCA (if allowed)
                self._check_predictive_entry(symbol, data, current_direction)
                
        # 2. Look for New Entry on NEW symbol
        elif pm.can_open_new_symbol():
            if has_pending:
                self._manage_predictive_order(symbol, data)
            else:
                # Look for new predictive entry opportunity (no existing position)
                self._check_predictive_entry(symbol, data, None)

    def _check_predictive_entry(self, symbol: str, data: BandData, current_direction: Optional[str]) -> None:
        """Check if price is approaching a band and place predictive limit order.
        
        If we have an open position and detect opposite direction approaching,
        this becomes an EXIT order (close existing + open new direction).
        
        Args:
            symbol: Trading symbol
            data: Current band data
            current_direction: Direction of existing position (None if no position)
        """
        if data.avg_volatility_pct < self.config.min_volatility_pct:
            return
        
        # Use real-time price from current forming bar
        current_price = data.current_price
        if current_price <= 0:
            return
        
        predicted_direction = None
        entry_price = None
        
        # Check proximity to lower band (potential long)
        if data.lower3 > 0:
            dist_to_lower = (current_price - data.lower3) / data.lower3
            if 0 < dist_to_lower <= self._proximity_pct:
                predicted_direction = "long"
                entry_price = data.lower3 * self._buffer_mult
                self.logger.info(f"[PROXIMITY] {symbol} price {current_price:.6f} within {dist_to_lower*100:.2f}% of lower3 {data.lower3:.6f}")
        
        # Check proximity to upper band (potential short)
        if predicted_direction is None and data.upper3 > 0:
            dist_to_upper = (data.upper3 - current_price) / data.upper3
            if 0 < dist_to_upper <= self._proximity_pct:
                predicted_direction = "short"
                entry_price = data.upper3 / self._buffer_mult
                self.logger.info(f"[PROXIMITY] {symbol} price {current_price:.6f} within {dist_to_upper*100:.2f}% of upper3 {data.upper3:.6f}")
        
        if not predicted_direction:
            return
        
        # Check if this is an EXIT (opposite direction) or regular ENTRY
        is_exit = current_direction is not None and predicted_direction != current_direction
        is_dca = current_direction is not None and predicted_direction == current_direction
        
        if is_exit:
            # This is an EXIT + REVERSAL
            # First close all existing positions, then place new entry
            self._place_predictive_exit_and_entry(symbol, predicted_direction, entry_price, data)
        elif is_dca:
            # DCA - same direction, check if allowed
            if self.state.position_manager.can_add_to_symbol(symbol):
                # Mark signal as seen to prevent multiple DCA attempts on same bar
                bar_key = f"{symbol}_{int(data.bar_time.timestamp())}"
                if bar_key not in self.state.signals_seen:
                    self.state.signals_seen.add(bar_key)
                    self._place_predictive_order(symbol, predicted_direction, entry_price, data)
        elif current_direction is None:
            # New position on new symbol - mark signal as seen
            bar_key = f"{symbol}_{int(data.bar_time.timestamp())}"
            if bar_key not in self.state.signals_seen:
                self.state.signals_seen.add(bar_key)
                self._place_predictive_order(symbol, predicted_direction, entry_price, data)

    def _place_predictive_exit_and_entry(self, symbol: str, new_direction: str, limit_price: float, data: BandData) -> None:
        """Close existing positions and place entry in new direction - all with one limit order.
        
        Since we're reversing, the limit order will:
        1. Close existing position (reduceOnly portion)
        2. Open new position in opposite direction
        
        But Bybit doesn't support this in one order, so we:
        1. Cancel all TP orders
        2. Place close order (reduceOnly) at band price
        3. Place new entry order at same band price
        """
        pm = self.state.position_manager
        all_positions = pm.get_all_positions(symbol)
        if not all_positions:
            return
        
        old_direction = all_positions[0].direction
        qty_dec, price_dec, min_qty = self.executor._get_qty_precision(symbol)
        limit_price = round(limit_price, price_dec)
        
        # Calculate total size to close
        total_close_size = 0.0
        for pos in all_positions:
            if pos.has_entry_filled() and not pos.is_position_closing():
                _, size, _ = pos.get_entry_info()
                total_close_size += size
                # Cancel TP orders
                if pos.tp_order_id:
                    self.client.cancel_order(symbol, pos.tp_order_id)
        
        if total_close_size <= 0:
            return
        
        total_close_size = round(total_close_size, qty_dec)
        close_side = "Sell" if old_direction == "long" else "Buy"
        
        # 1. Place CLOSE order (reduceOnly) at band price
        self.logger.info(f"[PREDICT EXIT] {symbol} {close_side} {total_close_size} @ {limit_price} (closing {old_direction})")
        close_order_id = self.client.place_limit_order(
            symbol, close_side, total_close_size, limit_price,
            reduce_only=True, qty_decimals=qty_dec, price_decimals=price_dec
        )
        
        if close_order_id:
            # Mark all positions as closing
            for pos in all_positions:
                if not pos.is_position_closing():
                    pos.start_closing(close_order_id)
            
            # Track this as a pending exit+entry
            with self._pending_entries_lock:
                self.pending_entries[symbol] = {
                    "bar_time": data.bar_time,
                    "direction": new_direction,
                    "order_id": close_order_id,
                    "limit_price": limit_price,
                    "last_amend_time": 0.0,
                    "is_exit": True,
                    "old_direction": old_direction
                }
            self.state.save()

    def _place_predictive_order(self, symbol: str, direction: str, limit_price: float, data: BandData) -> None:
        """Place a predictive limit order at the band."""
        qty_dec, price_dec, min_qty = self.executor._get_qty_precision(symbol)
        raw_qty = self.config.position_size_usd / limit_price
        qty = round(raw_qty, qty_dec)
        
        if qty <= 0 or qty < min_qty:
            return
        
        self.client.set_leverage(symbol, self.config.leverage)
        
        side = "Buy" if direction == "long" else "Sell"
        limit_price = round(limit_price, price_dec)
        
        # Calculate TP at entry time (based on limit price)
        tp_price = self._strategy.calculate_tp_price(limit_price, direction, data.avg_volatility_pct)
        tp_price = round(tp_price, price_dec)
        
        self.logger.info(f"[PREDICT] {symbol} {side} {qty} @ {limit_price} TP:{tp_price} (price approaching band)")
        
        order_id = self.client.place_limit_order(symbol, side, qty, limit_price,
                                                  reduce_only=False, qty_decimals=qty_dec,
                                                  price_decimals=price_dec, take_profit=tp_price)
        if order_id:
            pos = SinglePosition(
                symbol=symbol,
                direction=direction,
                entry_price=limit_price,
                size=qty,
                entry_time=datetime.now(timezone.utc),
                entry_order_id=order_id,
                tp_price=tp_price  # Track TP price (Bybit manages the actual order)
            )
            self.state.position_manager.add_position(pos)
            with self._pending_entries_lock:
                self.pending_entries[symbol] = {
                    "bar_time": data.bar_time,
                    "direction": direction,
                    "order_id": order_id,
                    "limit_price": limit_price,
                    "last_amend_time": 0.0,
                    "is_exit": False
                }
            self.state.save()

    def _manage_predictive_order(self, symbol: str, data: BandData) -> None:
        """Manage existing predictive order - cancel if price moves away, update price continuously."""
        with self._pending_entries_lock:
            pending_info = self.pending_entries.get(symbol)
            if not pending_info:
                return
            # Extract all needed values in single lock acquisition
            direction = pending_info["direction"]
            is_exit = pending_info.get("is_exit", False)
            order_id = pending_info["order_id"]
            old_direction = pending_info.get("old_direction")
            price_dec = pending_info.get("price_dec")
            old_price = pending_info.get("limit_price", 0)
            last_amend = pending_info.get("last_amend_time", 0)
        
        # For exit orders, check if positions are still being closed
        if is_exit:
            all_positions = self.state.position_manager.get_all_positions(symbol)
            if not any(p.is_position_closing() for p in all_positions):
                self.logger.info(f"[EXIT COMPLETE] {symbol} - closed, ready for new entry")
                with self._pending_entries_lock:
                    self.pending_entries.pop(symbol, None)
                return
        else:
            pos = self.state.position_manager.get_position(symbol)
            if not pos or pos.has_entry_filled():
                with self._pending_entries_lock:
                    self.pending_entries.pop(symbol, None)
                return
        
        current_price = data.current_price
        if current_price <= 0:
            return
        
        # Cache precision if not already cached
        if price_dec is None:
            _, price_dec, _ = self.executor._get_qty_precision(symbol)
        
        # Update bar_time and cache price_dec in single lock
        with self._pending_entries_lock:
            if symbol in self.pending_entries:
                self.pending_entries[symbol]["bar_time"] = data.bar_time
                self.pending_entries[symbol]["price_dec"] = price_dec
        
        should_cancel = False
        new_limit_price = None
        
        # Determine which band to track based on direction
        if is_exit:
            if old_direction == "long":
                if data.upper3 <= 0:
                    should_cancel = True
                else:
                    dist_from_upper = (data.upper3 - current_price) / data.upper3
                    if dist_from_upper > self._exit_pct:
                        should_cancel = True
                        self.logger.info(f"[EXIT RANGE] {symbol} exit - price moved away from upper band")
                    else:
                        new_limit_price = data.upper3 / self._buffer_mult
            else:
                if data.lower3 <= 0:
                    should_cancel = True
                else:
                    dist_from_lower = (current_price - data.lower3) / data.lower3
                    if dist_from_lower > self._exit_pct:
                        should_cancel = True
                        self.logger.info(f"[EXIT RANGE] {symbol} exit - price moved away from lower band")
                    else:
                        new_limit_price = data.lower3 * self._buffer_mult
        else:
            if direction == "long":
                if data.lower3 <= 0:
                    should_cancel = True
                else:
                    dist_from_lower = (current_price - data.lower3) / data.lower3
                    if dist_from_lower > self._exit_pct:
                        should_cancel = True
                        self.logger.info(f"[EXIT RANGE] {symbol} long - price {current_price:.6f} is {dist_from_lower*100:.2f}% above lower band")
                    else:
                        new_limit_price = data.lower3 * self._buffer_mult
            else:
                if data.upper3 <= 0:
                    should_cancel = True
                else:
                    dist_from_upper = (data.upper3 - current_price) / data.upper3
                    if dist_from_upper > self._exit_pct:
                        should_cancel = True
                        self.logger.info(f"[EXIT RANGE] {symbol} short - price {current_price:.6f} is {dist_from_upper*100:.2f}% below upper band")
                    else:
                        new_limit_price = data.upper3 / self._buffer_mult
        
        if should_cancel:
            self.logger.info(f"[CANCEL PREDICT] {symbol} {'exit' if is_exit else 'entry'} - price moved away from band")
            self.client.cancel_order(symbol, order_id)
            with self._pending_entries_lock:
                self.pending_entries.pop(symbol, None)
            
            if is_exit:
                for pos in self.state.position_manager.get_all_positions(symbol):
                    if pos.close_order_id == order_id:
                        with pos._lock:
                            pos.is_closing = False
                            pos.close_order_id = None
                self.state.save()
            return
        
        # Update limit price to track current band (with rate limiting)
        if new_limit_price:
            new_limit_price = round(new_limit_price, price_dec)
            
            # Update if price changed (with cooldown to avoid rate limits)
            if new_limit_price != old_price:
                now = time.time()
                if now - last_amend >= self._amend_cooldown_sec:
                    # For entry orders, recalculate TP based on new entry price
                    new_tp = None
                    if not is_exit:
                        new_tp = self._strategy.calculate_tp_price(new_limit_price, direction, data.avg_volatility_pct)
                        new_tp = round(new_tp, price_dec)
                    
                    if self.client.amend_order(symbol, order_id, new_limit_price, price_dec, take_profit=new_tp):
                        with self._pending_entries_lock:
                            if symbol in self.pending_entries:
                                self.pending_entries[symbol]["limit_price"] = new_limit_price
                                self.pending_entries[symbol]["last_amend_time"] = now
                        self.logger.debug(f"[UPDATE PREDICT] {symbol} @ {new_limit_price} TP:{new_tp} (was {old_price})")
                    else:
                        self.logger.warning(f"[AMEND FAILED] {symbol} - will retry or clean up")

if __name__ == "__main__":
    Bot().run()
