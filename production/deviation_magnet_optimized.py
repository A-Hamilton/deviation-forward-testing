"""
Deviation Magnet Bot - Optimized Production Version
===================================================

Performance Optimizations:
- Reduced dict lookups via local caching
- Optimized hot paths (check_signals, band calculation)
- Minimized object allocations
- Streamlined control flow
- Efficient data structure usage
- Batched operations where possible
"""

import asyncio
import logging
import logging.handlers
import os
import sys
import time
import hmac
import hashlib
import json
import uuid
import math
import queue
import itertools
from enum import Enum
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import websockets
from pybit.unified_trading import HTTP, WebSocket
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# Constants & Configuration
# =============================================================================

class Side(str, Enum):
    BUY = "Buy"
    SELL = "Sell"

class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"

class Category(str, Enum):
    LINEAR = "linear"

@dataclass
class Config:
    api_key: str = field(default_factory=lambda: os.environ.get("BYBIT_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.environ.get("BYBIT_API_SECRET", ""))

    bb_length: int = 7
    mult: float = 3.0
    dev_mult: float = 1.5
    min_volatility_pct: float = 0.20
    timeframe: str = "1"

    position_size_usd: float = 10.0
    max_total_orders: int = 2
    amend_threshold_pct: float = 0.0005

    data_dir: Path = field(default_factory=lambda: Path("refined_data"))
    log_file_name: str = "refined_bot.log"

    def __post_init__(self):
        if not self.api_key or not self.api_secret:
            raise ValueError("API_KEY and API_SECRET must be set in .env file")
        self.data_dir.mkdir(exist_ok=True)

@dataclass
class BandData:
    close: float
    upper3: float
    lower3: float
    avg_volatility_pct: float
    current_price: float
    bar_time: datetime

# =============================================================================
# Utils & Math
# =============================================================================

class LoggerSetup:
    @staticmethod
    def setup_async(config: Config) -> Tuple[logging.Logger, logging.handlers.QueueListener]:
        log_queue = queue.Queue()
        queue_handler = logging.handlers.QueueHandler(log_queue)

        log = logging.getLogger("OptimizedBot")
        log.setLevel(logging.INFO)
        log.handlers = []
        log.addHandler(queue_handler)

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        fh = logging.FileHandler(config.data_dir / config.log_file_name, encoding='utf-8')
        fh.setFormatter(formatter)

        listener = logging.handlers.QueueListener(log_queue, ch, fh)
        listener.start()
        return log, listener

    @staticmethod
    def safe_str(s) -> str:
        try:
            return str(s).encode('ascii', 'replace').decode('ascii')
        except Exception:
            return "Unprintable"

class TechnicalAnalysis:
    @staticmethod
    def calculate_bands(history: deque, bb_length: int, mult: float, dev_mult: float, current_price: float, forming_bar: Optional[List] = None) -> Optional[BandData]:
        hist_len = len(history)
        if hist_len < bb_length - 1:  # Need at least bb_length-1 closed bars if using forming bar
            return None

        # If we have a forming bar, use last (bb_length-1) closed bars + forming bar
        # Otherwise use last bb_length closed bars
        if forming_bar and hist_len >= bb_length - 1:
            start_idx = hist_len - (bb_length - 1)
            window = [history[i] for i in range(start_idx, hist_len)]
            window.append(forming_bar)  # Add the forming bar as the most recent
        else:
            if hist_len < bb_length:
                return None
            start_idx = hist_len - bb_length
            window = [history[i] for i in range(start_idx, hist_len)]

        ohlc4_sum = 0.0
        ohlc4_values = []
        vol_pct_sum = 0.0

        # Optimized: single pass calculation
        for bar in window:
            val = (bar[1] + bar[2] + bar[3] + bar[4]) * 0.25
            ohlc4_values.append(val)
            ohlc4_sum += val
            if bar[4] > 0:
                vol_pct_sum += (bar[2] - bar[3]) / bar[4]

        basis = ohlc4_sum / bb_length
        avg_vol_pct = (vol_pct_sum / bb_length) * 100.0

        # Optimized: inline variance calculation
        variance = sum((x - basis) * (x - basis) for x in ohlc4_values) / (bb_length - 1)
        dev = mult * math.sqrt(variance) * dev_mult

        return BandData(
            close=window[-1][4],
            upper3=basis + dev,
            lower3=basis - dev,
            avg_volatility_pct=avg_vol_pct,
            current_price=current_price if current_price > 0 else window[-1][4],
            bar_time=datetime.fromtimestamp(window[-1][0] / 1000, tz=timezone.utc)
        )

    @staticmethod
    def calculate_tp_price(entry_price: float, direction: Direction, vol_pct: float) -> float:
        target_pct = 0.002 + (vol_pct * 0.005)
        return entry_price * (1.0 + target_pct) if direction == Direction.LONG else entry_price * (1.0 - target_pct)

# =============================================================================
# Network Layer
# =============================================================================

class BybitConnector:
    __slots__ = ('config', 'queue', 'url', 'ws', 'log', 'connected', 'conn_lock', 'ping_task', 'pending_requests', '_tasks')

    def __init__(self, config: Config, callback_queue: asyncio.Queue):
        self.config = config
        self.queue = callback_queue
        self.url = "wss://stream.bybit.com/v5/trade"
        self.ws = None
        self.log = logging.getLogger("OptimizedBot")
        self.connected = asyncio.Event()
        self.conn_lock = asyncio.Lock()
        self.ping_task = None
        self.pending_requests: Dict[str, Tuple[str, str]] = {}
        self._tasks = []

    async def connect(self):
        async with self.conn_lock:
            if self.connected.is_set() and self.ws:
                return

            self.log.info("Connecting to Trade WebSocket...")
            self.ws = await websockets.connect(self.url, ping_interval=None)
            await self._authenticate()
            self.connected.set()
            self.log.info("Trade WS Connected & Authenticated")

            self._tasks.append(asyncio.create_task(self._read_loop()))
            if self.ping_task:
                self.ping_task.cancel()
            self.ping_task = asyncio.create_task(self._keep_alive())
            self._tasks.append(self.ping_task)

    async def close(self):
        self.connected.clear()
        if self.ws:
            await self.ws.close()
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

    async def _authenticate(self):
        expires = int((time.time() + 10) * 1000)
        signature = hmac.new(
            self.config.api_secret.encode("utf-8"),
            f"GET/realtime{expires}".encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        await self.ws.send(json.dumps({"op": "auth", "args": [self.config.api_key, expires, signature]}))

    async def _keep_alive(self):
        while True:
            try:
                await asyncio.sleep(20)
                if self.ws and self.connected.is_set():
                    await self.ws.send('{"op":"ping"}')
            except asyncio.CancelledError:
                break
            except Exception:
                self.connected.clear()
                break

    async def _read_loop(self):
        try:
            async for msg in self.ws:
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue

                req_id = data.get("reqId")
                ret_code = data.get("retCode", 0)

                if ret_code != 0:
                    if not (ret_code == 110001 and "order.cancel" in data.get("op", "")):
                        self.log.warning(f"Trade WS Error: {data}")

                    if req_id in self.pending_requests:
                        symbol, order_link_id = self.pending_requests.pop(req_id)
                        if ret_code != 110001:
                            self.log.info(f"Rollback {symbol} {order_link_id}")

                        await self.queue.put({
                            "type": "error_rollback",
                            "symbol": symbol,
                            "order_link_id": order_link_id,
                            "retMsg": data.get("retMsg", "Unknown Error"),
                            "retCode": ret_code
                        })
                elif req_id in self.pending_requests:
                    self.pending_requests.pop(req_id, None)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log.error(f"Trade WS Read Loop Error: {e}")
        finally:
            self.connected.clear()

    async def _send_payload(self, op: str, kwargs: dict) -> str:
        if not self.ws or not self.connected.is_set():
            await self.connect()

        req_id = str(uuid.uuid4())
        symbol = kwargs.get("symbol")
        order_link_id = kwargs.get("orderLinkId")

        if symbol:
            self.pending_requests[req_id] = (symbol, order_link_id if order_link_id else "NO_LINK_ID")

        payload = {
            "op": op,
            "reqId": req_id,
            "header": {
                "X-BAPI-TIMESTAMP": str(int(time.time() * 1000)),
                "X-BAPI-RECV-WINDOW": "20000"
            },
            "args": [kwargs]
        }

        await self.ws.send(json.dumps(payload))
        return req_id

    async def place_order(self, **kwargs): return await self._send_payload("order.create", kwargs)
    async def amend_order(self, **kwargs): return await self._send_payload("order.amend", kwargs)
    async def cancel_order(self, **kwargs): return await self._send_payload("order.cancel", kwargs)

# =============================================================================
# Main Bot Logic
# =============================================================================

class DeviationBot:
    __slots__ = ('config', 'log', 'log_listener', 'http', 'queue', 'trade_ws', 'ws_public', 'ws_private',
                 'history', 'forming_bars', 'current_prices', 'instrument_info', 'positions', 'open_orders',
                 'active_orders', 'blacklist', 'pending_flips', 'pending_entries', 'recently_filled',
                 'last_dca_time', 'last_dca_bar', 'FLIP_COOLDOWN', 'AMEND_COOLDOWN',
                 'FILL_COOLDOWN', 'DCA_COOLDOWN', 'loop', '_position_size_decimal')

    def __init__(self):
        self.config = Config()
        self.log, self.log_listener = LoggerSetup.setup_async(self.config)

        self.http = HTTP(testnet=False, api_key=self.config.api_key, api_secret=self.config.api_secret, recv_window=20000)

        self.queue = asyncio.Queue()
        self.trade_ws = BybitConnector(self.config, self.queue)
        self.ws_public = None
        self.ws_private = None

        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
        self.forming_bars: Dict[str, List] = {}  # Real-time forming bars updated each tick
        self.current_prices: Dict[str, float] = {}
        self.instrument_info: Dict[str, Tuple[int, int, float]] = {}

        self.positions: Dict[str, dict] = {}
        self.open_orders: Dict[str, list] = {}
        self.active_orders: Set[str] = set()
        self.blacklist: Set[str] = set()

        self.pending_flips: Dict[str, float] = {}
        self.pending_entries: Dict[str, dict] = {}
        self.recently_filled: Dict[str, float] = {}
        self.last_dca_time: Dict[str, float] = {}
        self.last_dca_bar: Dict[str, int] = {}  # Track by closed bar timestamp (int)

        self.FLIP_COOLDOWN = 60.0
        self.AMEND_COOLDOWN = 2.0
        self.FILL_COOLDOWN = 5.0
        self.DCA_COOLDOWN = 10.0

        self.loop = None
        self._position_size_decimal = Decimal(str(self.config.position_size_usd))

    async def _http_call(self, func, *args, **kwargs):
        return await self.loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def get_instrument_info(self, symbol: str) -> Tuple[int, int, float]:
        # Optimized: early return
        cached = self.instrument_info.get(symbol)
        if cached:
            return cached

        try:
            resp = await self._http_call(self.http.get_instruments_info, category=Category.LINEAR.value, symbol=symbol)
            info = resp["result"]["list"][0]

            qty_dec = max(0, -Decimal(str(info["lotSizeFilter"]["qtyStep"])).as_tuple().exponent)
            price_dec = max(0, -Decimal(str(info["priceFilter"]["tickSize"])).as_tuple().exponent)
            min_qty = float(info["lotSizeFilter"]["minOrderQty"])

            result = (qty_dec, price_dec, min_qty)
            self.instrument_info[symbol] = result
            return result
        except Exception:
            return 3, 2, 0.1

    def _calculate_safe_qty(self, price: float, qty_dec: int, min_qty: float) -> str:
        # Optimized: reuse position_size_decimal
        raw_qty = self._position_size_decimal / Decimal(str(price))
        step_fmt = f"1.{'0' * qty_dec}"
        qty = raw_qty.quantize(Decimal(step_fmt), rounding=ROUND_DOWN)

        if float(qty) < min_qty:
            qty = Decimal(str(min_qty))

        return str(qty)

    def _get_cached_bands(self, symbol: str) -> Optional[BandData]:
        # Real-time band calculation using forming bar
        history = self.history.get(symbol)
        forming_bar = self.forming_bars.get(symbol)
        current_price = self.current_prices.get(symbol, 0)

        # Need at least bb_length-1 closed bars if we have a forming bar, otherwise bb_length
        min_history = self.config.bb_length - 1 if forming_bar else self.config.bb_length
        if not history or len(history) < min_history:
            return None

        # Calculate bands with forming bar for real-time updates
        bands = TechnicalAnalysis.calculate_bands(
            history,
            self.config.bb_length,
            self.config.mult,
            self.config.dev_mult,
            current_price,
            forming_bar  # Include forming bar for real-time calculation
        )

        return bands

    async def place_entry(self, symbol: str, direction: Direction, price: float, vol_pct: float, is_dca: bool = False):
        qty_dec, price_dec, min_qty = await self.get_instrument_info(symbol)
        qty_str = self._calculate_safe_qty(price, qty_dec, min_qty)

        if float(qty_str) < min_qty:
            return

        side = Side.BUY.value if direction == Direction.LONG else Side.SELL.value
        price_fmt = f"{price:.{price_dec}f}"
        link_id = uuid.uuid4().hex

        self.log.info(f"[{symbol}] Placing {direction.value.upper()} {'DCA' if is_dca else 'ENTRY'}: {qty_str} @ {price_fmt}")

        tp_price = TechnicalAnalysis.calculate_tp_price(price, direction, vol_pct)
        tp_fmt = f"{tp_price:.{price_dec}f}"

        order_params = {
            "category": Category.LINEAR.value,
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": qty_str,
            "price": price_fmt,
            "timeInForce": "PostOnly",
            "tpslMode": "Partial",
            "orderLinkId": link_id,
            "takeProfit": tp_fmt,
            "tpTriggerBy": "LastPrice",
            "tpOrderType": "Limit",
            "tpLimitPrice": tp_fmt,
            "tpQty": qty_str
        }

        try:
            await self.trade_ws.place_order(**order_params)

            self.pending_entries[symbol] = {
                "order_id": None,
                "order_link_id": link_id,
                "direction": direction.value,
                "limit_price": price,
                "last_amend_time": time.time(),
                "price_dec": price_dec,
                "qty_dec": qty_dec,
                "vol_pct": vol_pct,
                "is_dca": is_dca
            }

            if symbol not in self.open_orders:
                self.open_orders[symbol] = []

            self.open_orders[symbol].append({
                "orderId": "",
                "orderLinkId": link_id,
                "side": side,
                "orderType": "Limit",
                "qty": qty_str,
                "price": price_fmt,
                "takeProfit": tp_fmt,
                "orderStatus": "New",
                "reduceOnly": False
            })
        except Exception as e:
            self.log.error(f"[{symbol}] Entry Exception: {LoggerSetup.safe_str(e)}")

    async def amend_entry(self, symbol: str, order_id: str, new_price: float):
        _, price_dec, _ = await self.get_instrument_info(symbol)
        price_fmt = f"{new_price:.{price_dec}f}"

        self.log.info(f"[{symbol}] Amending order {order_id} -> {price_fmt}")
        await self.trade_ws.amend_order(category=Category.LINEAR.value, symbol=symbol, orderId=order_id, price=price_fmt)

    async def cancel_pending_order(self, symbol: str, order_id: str):
        pending = self.pending_entries.get(symbol)
        link_id = pending.get("order_link_id") if pending else None

        if not order_id and not link_id:
            return

        cancel_params = {"category": Category.LINEAR.value, "symbol": symbol}
        if order_id:
            cancel_params["orderId"] = order_id
        else:
            cancel_params["orderLinkId"] = link_id

        try:
            await self.trade_ws.cancel_order(**cancel_params)
        except Exception as e:
            self.log.warning(f"[{symbol}] Cancel order failed: {LoggerSetup.safe_str(e)}")
        finally:
            self.pending_entries.pop(symbol, None)

    async def execute_flip(self, symbol: str, band_price: float, current_pos: dict, target_dir: Direction, vol_pct: float):
        qty_dec, price_dec, min_qty = await self.get_instrument_info(symbol)

        pos_size = float(current_pos['size'])
        entry_qty_str = self._calculate_safe_qty(band_price, qty_dec, min_qty)
        entry_qty = float(entry_qty_str)

        # Check if existing TP is close to current price (might hit before flip fills)
        # If so, use entry-only size instead of flip size to avoid massive entry after TP hits
        existing_tp_raw = current_pos.get('takeProfit', '0')
        existing_tp = float(existing_tp_raw) if existing_tp_raw and existing_tp_raw != '' else 0.0
        current_price = self.current_prices.get(symbol, band_price)
        tp_distance_pct = abs(current_price - existing_tp) / current_price if existing_tp > 0 and current_price > 0 else 999.0

        # If TP is within 0.1% (very close), risk of TP hitting first is high
        # Place normal entry instead of flip to avoid oversized trade
        if tp_distance_pct < 0.001:  # 0.1% proximity
            self.log.info(f"[{symbol}] TP too close ({tp_distance_pct*100:.3f}%) - placing normal entry instead of flip")
            await self.place_entry(symbol, target_dir, band_price, vol_pct, is_dca=False)
            return

        total_qty = pos_size + entry_qty

        step_fmt = f"1.{'0' * qty_dec}"
        total_qty_fmt = str(Decimal(str(total_qty)).quantize(Decimal(step_fmt), rounding=ROUND_DOWN))
        entry_qty_fmt = str(Decimal(str(entry_qty)).quantize(Decimal(step_fmt), rounding=ROUND_DOWN))

        if float(total_qty_fmt) <= 0:
            return

        flip_side = Side.SELL.value if target_dir == Direction.SHORT else Side.BUY.value
        tp_price = TechnicalAnalysis.calculate_tp_price(band_price, target_dir, vol_pct)
        price_fmt = f"{band_price:.{price_dec}f}"
        tp_fmt = f"{tp_price:.{price_dec}f}"

        link_id = uuid.uuid4().hex

        self.log.info(f"[{symbol}] FLIP: {flip_side} {total_qty_fmt} @ {price_fmt} (TP: {entry_qty_fmt} @ {tp_fmt})")

        order_params = {
            "category": Category.LINEAR.value,
            "symbol": symbol,
            "side": flip_side,
            "orderType": "Limit",
            "qty": total_qty_fmt,
            "price": price_fmt,
            "timeInForce": "GTC",
            "reduceOnly": False,
            "tpslMode": "Partial",
            "orderLinkId": link_id,
            "takeProfit": tp_fmt,
            "tpTriggerBy": "LastPrice",
            "tpOrderType": "Limit",
            "tpLimitPrice": tp_fmt,
            "tpQty": entry_qty_fmt
        }

        try:
            await self.trade_ws.place_order(**order_params)
            self.pending_flips[symbol] = time.time()

            # NOTE: We do NOT cancel existing orders here!
            # Existing TP will be cancelled automatically when flip order FILLS
            # This is handled in the order fill handler (main_loop)
            # This prevents leaving position without TP if flip signal is lost

        except Exception as e:
            self.log.error(f"[{symbol}] Flip Place Failed: {LoggerSetup.safe_str(e)}")
            return

        self.pending_entries[symbol] = {
            "order_link_id": link_id,
            "order_id": None,
            "direction": target_dir.value,
            "limit_price": band_price,
            "last_amend_time": time.time(),
            "is_dca": False,
            "is_flip": True,
            "vol_pct": vol_pct,
            "qty_dec": qty_dec,
            "price_dec": price_dec,
            "cancel_old_orders_on_fill": True  # Flag to cancel old orders when flip fills
        }

    async def _process_existing_position(self, symbol: str, pos: dict, bands: BandData, long_proximity: bool, short_proximity: bool):
        # Optimized: reduce branching
        pos_side = pos["side"]
        pos_dir = Direction.LONG if pos_side == Side.BUY.value else Direction.SHORT
        target_dir = Direction.LONG if long_proximity else Direction.SHORT

        is_flip_signal = (long_proximity or short_proximity) and (pos_dir != target_dir)
        pending = self.pending_entries.get(symbol)

        if is_flip_signal:
            if pending:
                await self._process_pending_order(symbol, bands, pending, long_proximity, short_proximity)
            elif time.time() - self.pending_flips.get(symbol, 0) > self.FLIP_COOLDOWN:
                limit_price = bands.lower3 if long_proximity else bands.upper3
                await self.execute_flip(symbol, limit_price, pos, target_dir, bands.avg_volatility_pct)
        else:
            if pending and pending.get("is_flip"):
                self.log.info(f"[{symbol}] Flip signal lost - cancelling Reversal Order")
                await self.cancel_pending_order(symbol, pending.get("order_id"))

            if pending and not pending.get("is_flip"):
                await self._process_pending_order(symbol, bands, pending, long_proximity, short_proximity)
            elif (long_proximity or short_proximity) and (pos_dir == target_dir):
                # Use closed bar timestamp instead of bands.bar_time (which uses forming bar)
                history = self.history.get(symbol)
                current_bar_time = history[-1][0] if history and len(history) > 0 else 0

                if current_bar_time != self.last_dca_bar.get(symbol, 0):
                    now = time.time()
                    if (now - self.last_dca_time.get(symbol, 0) > self.DCA_COOLDOWN) and (now - self.recently_filled.get(symbol, 0) > self.DCA_COOLDOWN):
                        limit_price = bands.lower3 if long_proximity else bands.upper3
                        await self.place_entry(symbol, target_dir, limit_price, bands.avg_volatility_pct, is_dca=True)
                        self.last_dca_time[symbol] = now
                        self.last_dca_bar[symbol] = current_bar_time

    async def _process_pending_order(self, symbol: str, bands: BandData, pending: dict, long_proximity: bool, short_proximity: bool):
        # Optimized: extract all dict values at once
        order_id = pending.get("order_id")
        direction_str = pending.get("direction")
        current_limit = pending.get("limit_price", 0)
        last_amend = pending.get("last_amend_time", 0)
        is_flip = pending.get("is_flip", False)

        direction = Direction(direction_str)
        new_limit = bands.lower3 if direction == Direction.LONG else bands.upper3
        in_proximity = long_proximity if direction == Direction.LONG else short_proximity

        if not in_proximity:
            self.log.info(f"[{symbol}] Price out of {direction.value} proximity - cancelling order")
            await self.cancel_pending_order(symbol, order_id)
            return

        # For real-time bands: amend if band price changed
        if current_limit > 0 and abs(new_limit - current_limit) > 0.00001:  # Changed at all
            now = time.time()
            if now - last_amend >= self.AMEND_COOLDOWN:
                await self.amend_entry(symbol, order_id, new_limit)
                pending["limit_price"] = new_limit
                pending["last_amend_time"] = now

    async def _process_new_signal(self, symbol: str, bands: BandData, long_proximity: bool):
        if bands.avg_volatility_pct < self.config.min_volatility_pct:
            return

        # Optimized: combine set operations
        active_syms = set(self.positions.keys()) | set(self.pending_entries.keys())
        if symbol not in active_syms and len(active_syms) >= self.config.max_total_orders:
            return

        if time.time() - self.recently_filled.get(symbol, 0) < self.FILL_COOLDOWN:
            return

        target_dir = Direction.LONG if long_proximity else Direction.SHORT
        limit_price = bands.lower3 if long_proximity else bands.upper3

        await self.place_entry(symbol, target_dir, limit_price, bands.avg_volatility_pct)

        if symbol in self.pending_entries:
            # Use closed bar timestamp instead of bands.bar_time
            history = self.history.get(symbol)
            current_bar_time = history[-1][0] if history and len(history) > 0 else 0
            self.last_dca_bar[symbol] = current_bar_time

    async def check_signals(self, symbol: str):
        # Optimized: early returns
        if symbol in self.blacklist or symbol in self.active_orders:
            return

        self.active_orders.add(symbol)

        try:
            bands = self._get_cached_bands(symbol)
            if not bands:
                return

            # Optimized: cache calculations
            current_price = bands.current_price
            vol_range = current_price * bands.avg_volatility_pct * 0.01

            long_proximity = current_price <= (bands.lower3 + vol_range)
            short_proximity = current_price >= (bands.upper3 - vol_range)

            pos = self.positions.get(symbol)

            if pos and float(pos["size"]) > 0:
                await self._process_existing_position(symbol, pos, bands, long_proximity, short_proximity)
                return

            pending = self.pending_entries.get(symbol)
            if pending:
                await self._process_pending_order(symbol, bands, pending, long_proximity, short_proximity)
                return

            if long_proximity or short_proximity:
                await self._process_new_signal(symbol, bands, long_proximity)
        finally:
            self.active_orders.discard(symbol)

    async def main_loop(self):
        self.log.info("Starting Main Loop...")
        while True:
            try:
                msg = await self.queue.get()
                topic = msg.get("topic", "")
                mtype = msg.get("type", "")

                if mtype == "reconcile_trigger":
                    await self.reconcile_state()
                    continue

                if mtype == "error_rollback":
                    sym = msg["symbol"]
                    link_id = msg["order_link_id"]
                    ret_code = msg["retCode"]

                    if ret_code != 110001:
                        self.log.warning(f"[{sym}] Rollback {link_id}: {msg.get('retMsg')} ({ret_code})")

                    if ret_code == 30228:
                        self.blacklist.add(sym)

                    if sym in self.open_orders:
                        self.open_orders[sym] = [x for x in self.open_orders[sym] if x.get("orderLinkId") != link_id]

                    pending = self.pending_entries.get(sym)
                    if pending and pending.get("order_link_id") == link_id:
                        self.pending_entries.pop(sym, None)
                    continue

                if mtype == "kline":
                    sym = msg["symbol"]
                    k = msg["data"]
                    row = [int(k["start"]), float(k["open"]), float(k["high"]), float(k["low"]), float(k["close"])]

                    if k["confirm"]:
                        # Bar closed - add to history and clear forming bar
                        hist = self.history[sym]
                        if not hist or hist[-1][0] != row[0]:
                            hist.append(row)
                        self.forming_bars.pop(sym, None)  # Clear forming bar on close
                    else:
                        # Bar still forming - update forming bar for real-time bands
                        self.forming_bars[sym] = row

                    self.current_prices[sym] = row[4]
                    await self.check_signals(sym)

                elif mtype == "trade":
                    # Note: Trade stream not subscribed, but keeping handler for future use
                    sym = msg["symbol"]
                    self.current_prices[sym] = msg["price"]
                    await self.check_signals(sym)

                elif topic == "position":
                    for p in msg["data"]:
                        sym = p["symbol"]
                        size_raw = p.get("size", "0")
                        size = float(size_raw) if size_raw and size_raw != '' else 0.0

                        if size > 0:
                            self.positions[sym] = p
                            await self.check_signals(sym)
                        else:
                            was_open = sym in self.positions
                            self.positions.pop(sym, None)

                            if was_open:
                                self.log.info(f"[{sym}] Position closed - Cancelling all orders (Safety)")
                                await self._http_call(self.http.cancel_all_orders, category=Category.LINEAR.value, symbol=sym, settleCoin="USDT")
                                self.open_orders[sym] = []
                                self.pending_entries.pop(sym, None)

                elif topic == "order":
                    for o in msg["data"]:
                        sym = o["symbol"]
                        oid = o.get("orderId", "")
                        lid = o.get("orderLinkId", "")
                        status = o["orderStatus"]

                        if status in ("Filled", "Cancelled", "Rejected"):
                            if sym in self.open_orders:
                                self.open_orders[sym] = [x for x in self.open_orders[sym] if x.get("orderId") != oid and x.get("orderLinkId") != lid]

                            pending = self.pending_entries.get(sym)
                            is_match = pending and (pending.get("order_id") == oid or (lid and pending.get("order_link_id") == lid))

                            if is_match:
                                if status == "Filled":
                                    self.log.info(f"[{sym}] Entry FILLED")
                                    self.recently_filled[sym] = time.time()

                                    # Update flip cooldown if this was a flip order
                                    if pending.get("is_flip"):
                                        self.pending_flips[sym] = time.time()

                                        # Cancel existing orders only when flip FILLS (not on placement)
                                        # This prevents leaving position without TP if flip signal is lost before fill
                                        if pending.get("cancel_old_orders_on_fill"):
                                            self.log.info(f"[{sym}] FLIP FILLED: Cancelling old orders")
                                            try:
                                                await self._http_call(
                                                    self.http.cancel_all_orders,
                                                    category=Category.LINEAR.value,
                                                    symbol=sym,
                                                    settleCoin="USDT"
                                                )
                                                self.open_orders[sym] = []
                                            except Exception as e:
                                                self.log.warning(f"[{sym}] Failed to cancel old orders after flip fill: {LoggerSetup.safe_str(e)}")

                                self.pending_entries.pop(sym, None)

                        elif status in ("New", "PartiallyFilled"):
                            if sym not in self.open_orders:
                                self.open_orders[sym] = []

                            found = False
                            for ex in self.open_orders[sym]:
                                if (oid and ex.get("orderId") == oid) or (lid and ex.get("orderLinkId") == lid):
                                    ex.update(o)
                                    found = True
                                    break

                            if not found:
                                self.open_orders[sym].append(o)

                            pending = self.pending_entries.get(sym)
                            if pending and lid and pending.get("order_link_id") == lid:
                                pending["order_id"] = oid
                                pending["status"] = status

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback
                self.log.error(f"Loop Error: {LoggerSetup.safe_str(e)}")
                self.log.error(f"Traceback: {traceback.format_exc()}")

    def ws_handler_generic(self, msg):
        topic = msg.get("topic", "")
        packet = None

        if "kline" in topic:
            packet = {"type": "kline", "symbol": topic.split(".")[-1], "data": msg["data"][0]}
        elif topic == "position":
            packet = {"type": "position", "topic": "position", "data": msg["data"]}
        elif topic == "order":
            packet = {"type": "order", "topic": "order", "data": msg["data"]}

        if packet and self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.queue.put_nowait, packet)

    async def _fetch_history_async(self, symbol):
        try:
            resp = await self._http_call(self.http.get_kline, category=Category.LINEAR.value, symbol=symbol, interval=self.config.timeframe, limit=200)
            if resp["retCode"] == 0:
                rows = [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in reversed(resp["result"]["list"])]
                return symbol, rows
        except Exception:
            pass
        return symbol, []

    async def initialize(self):
        self.log.info("Connecting to Bybit...")
        await self.trade_ws.connect()

        resp = await self._http_call(self.http.get_instruments_info, category=Category.LINEAR.value, limit=1000)
        symbols = []

        for i in resp["result"]["list"]:
            if i["status"] == "Trading" and i["quoteCoin"] == "USDT" and i["symbol"].endswith("USDT"):
                s = i["symbol"]
                symbols.append(s)

                self.instrument_info[s] = (
                    max(0, -Decimal(str(i["lotSizeFilter"]["qtyStep"])).as_tuple().exponent),
                    max(0, -Decimal(str(i["priceFilter"]["tickSize"])).as_tuple().exponent),
                    float(i["lotSizeFilter"]["minOrderQty"])
                )

        self.log.info(f"Found {len(symbols)} active pairs.")

        # Optimized: batch fetching with smaller batches for faster startup
        tasks = [self._fetch_history_async(s) for s in symbols]
        batch_size = 20

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            for res in results:
                if isinstance(res, tuple) and len(res) == 2:
                    self.history[res[0]].extend(res[1])
            await asyncio.sleep(0.1)

        await self.reconcile_state()
        return symbols

    async def reconcile_state(self):
        try:
            p_resp = await self._http_call(self.http.get_positions, category=Category.LINEAR.value, settleCoin="USDT", limit=100)
            self.positions = {p["symbol"]: p for p in p_resp["result"]["list"] if float(p["size"]) > 0}

            o_resp = await self._http_call(self.http.get_open_orders, category=Category.LINEAR.value, settleCoin="USDT", openOnly=0, limit=50)
            api_orders = defaultdict(list)
            for o in o_resp["result"]["list"]:
                api_orders[o["symbol"]].append(o)

            for symbol in set(list(self.open_orders.keys()) + list(api_orders.keys())):
                preserved = [o for o in self.open_orders.get(symbol, []) if o.get("orderLinkId") and not o.get("orderId")]
                self.open_orders[symbol] = preserved + api_orders[symbol]

                if not self.pending_entries.get(symbol):
                    for o in api_orders[symbol]:
                        if o["orderStatus"] == "New" and o["orderType"] == "Limit" and not o.get("reduceOnly", False):
                            inst = self.instrument_info.get(symbol, (3, 2, 0.1))
                            self.pending_entries[symbol] = {
                                "order_id": o["orderId"],
                                "order_link_id": o.get("orderLinkId"),
                                "direction": Direction.LONG.value if o["side"] == Side.BUY.value else Direction.SHORT.value,
                                "limit_price": float(o["price"]),
                                "last_amend_time": time.time(),
                                "is_dca": False,
                                "vol_pct": 0.5,
                                "qty_dec": inst[0],
                                "price_dec": inst[1]
                            }
                            break

            # Optimized: memory cleanup
            now = time.time()
            stale_threshold = 3600.0
            active_symbols = set(self.positions.keys()) | set(self.pending_entries.keys())

            for cooldown_dict in (self.recently_filled, self.last_dca_time, self.pending_flips):
                stale_keys = [k for k, v in cooldown_dict.items() if k not in active_symbols and (now - v) > stale_threshold]
                for k in stale_keys:
                    cooldown_dict.pop(k, None)

            stale_bar_keys = [k for k in self.last_dca_bar.keys() if k not in active_symbols]
            for k in stale_bar_keys:
                self.last_dca_bar.pop(k, None)

            # Cleanup forming_bars for inactive symbols (prevent memory leak)
            stale_forming = [k for k in self.forming_bars.keys() if k not in active_symbols]
            for k in stale_forming:
                self.forming_bars.pop(k, None)

            self.log.info(f"State Reconciled: {len(self.positions)} Pos, {len(self.open_orders)} Symbols with Orders")
        except Exception as e:
            self.log.error(f"Reconcile Error: {LoggerSetup.safe_str(e)}")

    async def periodic_reconcile(self):
        while True:
            try:
                await asyncio.sleep(60)
                await self.queue.put({"type": "reconcile_trigger"})
            except asyncio.CancelledError:
                break

    def run(self):
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        symbols = self.loop.run_until_complete(self.initialize())

        self.ws_public = WebSocket(testnet=False, channel_type="linear")
        self.ws_private = WebSocket(testnet=False, channel_type="private", api_key=self.config.api_key, api_secret=self.config.api_secret)

        self.ws_private.position_stream(callback=self.ws_handler_generic)
        self.ws_private.order_stream(callback=self.ws_handler_generic)

        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            self.ws_public.kline_stream(interval=int(self.config.timeframe), symbol=symbols[i:i+batch_size], callback=self.ws_handler_generic)
            time.sleep(0.1)

        tasks = [
            self.loop.create_task(self.main_loop()),
            self.loop.create_task(self.periodic_reconcile())
        ]

        try:
            self.log.info("Loop Running...")
            self.loop.run_forever()
        except KeyboardInterrupt:
            self.log.info("Stopping...")
            for t in tasks:
                t.cancel()
            if self.trade_ws:
                self.loop.run_until_complete(self.trade_ws.close())
            self.log_listener.stop()
        finally:
            self.loop.close()

if __name__ == "__main__":
    DeviationBot().run()
