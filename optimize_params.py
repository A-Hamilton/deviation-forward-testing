"""
Parameter Optimizer for Deviation Magnet Strategy
=================================================

Fast offline backtester that sweeps through parameter combinations.

Usage:
    python optimize_params.py
    python optimize_params.py --bb_start 5 --bb_end 100 --bb_step 5
    python optimize_params.py --top 20

Output:
    - Console: Top performers ranked by PnL
    - CSV: optimization_results.csv with full results
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing

import numpy as np
from pybit.unified_trading import HTTP


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptConfig:
    """Optimizer configuration."""
    
    # Parameter grid
    bb_length_start: int = 5
    bb_length_end: int = 200
    bb_length_step: int = 5  # 40 values: 5, 10, 15, ... 200
    
    mult_start: float = 1.0
    mult_end: float = 50.0
    mult_step: float = 0.5  # 99 values: 1.0, 1.5, ... 50.0
    
    # Fixed strategy params (same as forward tester)
    dev_mult: float = 1.5
    base_order_size: float = 10.0
    max_positions_total: int = 100  # Max positions TOTAL across all symbols
    
    # Fee structure (same as forward tester)
    fee_pct: float = 0.00055  # 0.055% taker
    maker_fee_pct: float = 0.0002  # 0.02% maker
    slippage_pct: float = 0.0005  # 0.05%
    
    # Data settings
    timeframe: str = "1"
    candles_per_symbol: int = 5000  # Will paginate to get this many (1000 per request)
    max_workers: int = 10
    tp_lookahead: int = 50  # Bars to look ahead for TP per position
    min_trades: int = 5  # Minimum trades for ranking
    target_win_rate: float = 0.0  # 0 = no filter (multi-position mode accepts losses)
    
    # Parallelization
    num_processes: int = 0  # 0 = auto (cpu_count)
    
    # Volatility filter
    min_volatility_pct: float = 0.2
    
    # Output
    output_file: Path = Path("optimization_results.csv")
    top_n: int = 10


@dataclass
class BacktestResult:
    """Results from a single parameter combination."""
    bb_length: int
    mult: float
    total_sigma: float  # mult * dev_mult
    
    # Performance metrics
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    
    # Exit breakdown
    tp_exits: int  # Trades exited via TP
    opposite_exits: int  # Trades exited via opposite signal (stop loss)
    tp_rate: float  # tp_exits / total_trades
    
    total_pnl_usd: float
    avg_pnl_per_trade: float
    best_trade: float
    worst_trade: float
    
    # Risk metrics
    sharpe_ratio: float
    profit_factor: float  # gross_profit / gross_loss
    max_drawdown_pct: float
    
    # Timing
    avg_hold_bars: float
    
    # Symbol coverage
    symbols_traded: int
    
    # Multi-position stats
    max_concurrent_positions: int = 0
    avg_positions_per_signal: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

class DataFetcher:
    """Fetches historical data from Bybit."""
    
    def __init__(self, config: OptConfig):
        self.config = config
        self.session = HTTP(testnet=False)
        
    def fetch_active_symbols(self) -> List[str]:
        """Fetch all active USDT perpetual symbols."""
        print("📡 Fetching active USDT perpetuals from Bybit...")
        
        try:
            resp = self.session.get_instruments_info(category="linear")
            
            if resp["retCode"] != 0:
                print(f"❌ Failed to fetch symbols: {resp}")
                return []

            symbols = [
                i["symbol"] for i in resp["result"]["list"]
                if i["status"] == "Trading"
                and i["quoteCoin"] == "USDT"
                and i["symbol"].endswith("USDT")
            ]
            
            print(f"✅ Found {len(symbols)} active pairs")
            return symbols
            
        except Exception as e:
            print(f"❌ Error fetching symbols: {e}")
            return []

    def fetch_history(self, symbol: str) -> Optional[np.ndarray]:
        """Fetch candle history for a symbol with pagination."""
        try:
            all_candles = []
            end_time = None  # Start from most recent
            target_candles = self.config.candles_per_symbol
            
            while len(all_candles) < target_candles:
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": self.config.timeframe,
                    "limit": 1000,  # Bybit max per request
                }
                if end_time:
                    params["end"] = end_time
                
                resp = self.session.get_kline(**params)
                
                if resp["retCode"] != 0 or not resp.get("result", {}).get("list"):
                    break
                
                batch = resp["result"]["list"]
                if not batch:
                    break
                
                all_candles.extend(batch)
                
                # Get timestamp of oldest candle for next request
                # Bybit returns newest first, so last item is oldest
                oldest_ts = int(batch[-1][0])
                end_time = oldest_ts - 1  # Go back 1ms before oldest
                
                # Avoid infinite loop if we get same data
                if len(batch) < 1000:
                    break
            
            if not all_candles:
                return None
            
            # Trim to target and convert to numpy
            all_candles = all_candles[:target_candles]
            
            arr = np.array([
                [float(r[1]), float(r[2]), float(r[3]), float(r[4])]
                for r in all_candles
            ], dtype=np.float64)
            
            # Reverse to chronological order (Bybit returns newest first)
            return arr[::-1]

        except Exception:
            return None

    def fetch_all_history(self, symbols: List[str]) -> Dict[str, np.ndarray]:
        """Fetch history for all symbols in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print(f"📥 Fetching {self.config.candles_per_symbol} candles for {len(symbols)} symbols...")
        
        data = {}
        failed = 0
        
        with ThreadPoolExecutor(max_workers=20) as executor:  # More threads for I/O
            future_to_symbol = {
                executor.submit(self.fetch_history, sym): sym
                for sym in symbols
            }
            
            for i, future in enumerate(as_completed(future_to_symbol)):
                sym = future_to_symbol[future]
                result = future.result()
                
                if result is not None and len(result) >= 50:
                    data[sym] = result
                else:
                    failed += 1
                
                if (i + 1) % 100 == 0:
                    print(f"   Fetched {i + 1}/{len(symbols)}...")
        
        print(f"✅ Loaded data for {len(data)} symbols ({failed} failed/insufficient)")
        return data

# ═══════════════════════════════════════════════════════════════════════════════
# FAST BACKTESTER (Vectorized)
# ═══════════════════════════════════════════════════════════════════════════════

def fast_backtest_symbol_multipos(data: np.ndarray, bb_length: int, mult: float, dev_mult: float,
                                   tp_lookahead: int, total_fee_pct: float, min_vol: float,
                                   base_size: float, entry_cost: float, exit_maker_cost: float,
                                   exit_taker_cost: float, max_positions: int) -> Tuple[float, int, int, int, int, int, int]:
    """
    Multi-position backtest for a single symbol.
    Returns (total_pnl, total_trades, tp_exits, opposite_exits, wins, max_concurrent, total_positions_opened).
    
    Logic:
    - Same direction signal = add position (up to max_positions)
    - Opposite direction signal = close ALL positions at opposite band, then open new
    - Each position can exit independently via TP
    """
    n = len(data)
    if n < bb_length + 10:
        return 0.0, 0, 0, 0, 0, 0, 0
    
    # Pre-extract columns
    opens = data[:, 0]
    highs = data[:, 1]
    lows = data[:, 2]
    closes = data[:, 3]
    
    # Pre-compute OHLC4 for all bars
    ohlc4 = (opens + highs + lows + closes) * 0.25
    
    # Pre-compute volatility for all bars
    vol_raw = (highs - lows) / closes
    
    # Pre-compute rolling stats for ALL bars (vectorized)
    cumsum = np.concatenate([[0], np.cumsum(ohlc4)])
    rolling_sum = cumsum[bb_length:] - cumsum[:-bb_length]
    rolling_mean = rolling_sum / bb_length
    
    cumsum_sq = np.concatenate([[0], np.cumsum(ohlc4 ** 2)])
    rolling_sum_sq = cumsum_sq[bb_length:] - cumsum_sq[:-bb_length]
    rolling_var = (rolling_sum_sq / bb_length) - (rolling_mean ** 2)
    rolling_var = rolling_var * bb_length / (bb_length - 1)
    rolling_std = np.sqrt(np.maximum(rolling_var, 0))
    
    vol_cumsum = np.concatenate([[0], np.cumsum(vol_raw)])
    vol_rolling_sum = vol_cumsum[bb_length:] - vol_cumsum[:-bb_length]
    rolling_vol = (vol_rolling_sum / bb_length) * 100
    
    # Position tracking: list of (entry_price, entry_idx, tp_price, direction)
    # direction: 1=long, -1=short
    open_positions = []
    current_direction = 0  # 0=none, 1=long, -1=short
    
    total_pnl = 0.0
    total_trades = 0
    tp_exits = 0
    opposite_exits = 0
    wins = 0
    max_concurrent = 0
    total_positions_opened = 0
    
    last_signal_bar = -1  # Track to avoid multiple signals on same bar
    
    for idx in range(len(rolling_mean)):
        orig_idx = idx + bb_length - 1
        if orig_idx >= n:
            break
        
        basis = rolling_mean[idx]
        stdev = rolling_std[idx]
        avg_vol = rolling_vol[idx]
        
        dev = mult * stdev * dev_mult
        upper3 = basis + dev
        lower3 = basis - dev
        
        high = highs[orig_idx]
        low = lows[orig_idx]
        close = closes[orig_idx]
        
        # 1. Check TP for all open positions on this bar
        positions_to_remove = []
        for i, (entry_price, entry_idx, tp_price, direction) in enumerate(open_positions):
            # Can only exit starting from bar AFTER entry
            if orig_idx <= entry_idx:
                continue
            
            tp_hit = False
            if direction == 1 and high >= tp_price:
                tp_hit = True
            elif direction == -1 and low <= tp_price:
                tp_hit = True
            
            if tp_hit:
                # Calculate PnL for this position (maker exit)
                price_ratio = tp_price / entry_price
                if direction == 1:
                    pnl_pct = (price_ratio - 1.0) * 100
                else:
                    pnl_pct = (1.0 - price_ratio) * 100
                
                entry_fee = base_size * entry_cost
                exit_fee = base_size * price_ratio * exit_maker_cost
                gross_pnl = pnl_pct * 0.01 * base_size
                net_pnl = gross_pnl - entry_fee - exit_fee
                
                total_pnl += net_pnl
                total_trades += 1
                tp_exits += 1
                if net_pnl > 0:
                    wins += 1
                
                positions_to_remove.append(i)
        
        # Remove TP'd positions (in reverse order to preserve indices)
        for i in reversed(positions_to_remove):
            open_positions.pop(i)
        
        # Update current direction after TP exits
        if not open_positions:
            current_direction = 0
        
        # 2. Check for new signal (only one signal per bar)
        if orig_idx == last_signal_bar:
            continue
        
        # Volatility check for new signals only
        if avg_vol < min_vol:
            continue
        
        signal = 0
        if low <= lower3:
            signal = 1  # long
        elif high >= upper3:
            signal = -1  # short
        
        if signal == 0:
            continue
        
        last_signal_bar = orig_idx
        
        # 3. Handle signal based on current positions
        if current_direction == 0:
            # No positions - open new
            entry_price = lower3 if signal == 1 else upper3
            target_pct = (avg_vol + total_fee_pct) / 100
            if signal == 1:
                tp_price = entry_price * (1 + target_pct)
            else:
                tp_price = entry_price * (1 - target_pct)
            
            open_positions.append((entry_price, orig_idx, tp_price, signal))
            current_direction = signal
            total_positions_opened += 1
            max_concurrent = max(max_concurrent, len(open_positions))
        
        elif current_direction == signal:
            # Same direction - add position if under limit
            if len(open_positions) < max_positions:
                entry_price = lower3 if signal == 1 else upper3
                target_pct = (avg_vol + total_fee_pct) / 100
                if signal == 1:
                    tp_price = entry_price * (1 + target_pct)
                else:
                    tp_price = entry_price * (1 - target_pct)
                
                open_positions.append((entry_price, orig_idx, tp_price, signal))
                total_positions_opened += 1
                max_concurrent = max(max_concurrent, len(open_positions))
        
        else:
            # OPPOSITE SIGNAL - close all at opposite band, then open new
            exit_price = upper3 if signal == 1 else lower3  # Exit at opposite band
            
            for (entry_price, entry_idx, tp_price, direction) in open_positions:
                price_ratio = exit_price / entry_price
                if direction == 1:
                    pnl_pct = (price_ratio - 1.0) * 100
                else:
                    pnl_pct = (1.0 - price_ratio) * 100
                
                entry_fee = base_size * entry_cost
                exit_fee = base_size * price_ratio * exit_taker_cost
                gross_pnl = pnl_pct * 0.01 * base_size
                net_pnl = gross_pnl - entry_fee - exit_fee
                
                total_pnl += net_pnl
                total_trades += 1
                opposite_exits += 1
                if net_pnl > 0:
                    wins += 1
            
            open_positions.clear()
            
            # Open new position in new direction
            entry_price = lower3 if signal == 1 else upper3
            target_pct = (avg_vol + total_fee_pct) / 100
            if signal == 1:
                tp_price = entry_price * (1 + target_pct)
            else:
                tp_price = entry_price * (1 - target_pct)
            
            open_positions.append((entry_price, orig_idx, tp_price, signal))
            current_direction = signal
            total_positions_opened += 1
            max_concurrent = max(max_concurrent, len(open_positions))
    
    # Close any remaining positions at last close (taker)
    if open_positions:
        last_close = closes[-1]
        for (entry_price, entry_idx, tp_price, direction) in open_positions:
            price_ratio = last_close / entry_price
            if direction == 1:
                pnl_pct = (price_ratio - 1.0) * 100
            else:
                pnl_pct = (1.0 - price_ratio) * 100
            
            entry_fee = base_size * entry_cost
            exit_fee = base_size * price_ratio * exit_taker_cost
            gross_pnl = pnl_pct * 0.01 * base_size
            net_pnl = gross_pnl - entry_fee - exit_fee
            
            total_pnl += net_pnl
            total_trades += 1
            opposite_exits += 1  # Count as opposite exit (forced close)
            if net_pnl > 0:
                wins += 1
    
    return total_pnl, total_trades, tp_exits, opposite_exits, wins, max_concurrent, total_positions_opened


def backtest_params_fast(args: Tuple) -> Tuple[int, float, float, int, int, int, int, int, int, int]:
    """Worker function for parallel processing. 
    Returns (bb_length, mult, pnl, trades, tp_exits, opposite_exits, wins, symbols, max_concurrent, total_positions)."""
    bb_length, mult, all_data_list, config_dict = args
    
    dev_mult = config_dict["dev_mult"]
    tp_lookahead = config_dict["tp_lookahead"]
    total_fee_pct = config_dict["total_fee_pct"]
    min_vol = config_dict["min_volatility_pct"]
    base_size = config_dict["base_order_size"]
    entry_cost = config_dict["entry_cost"]
    exit_maker = config_dict["exit_maker_cost"]
    exit_taker = config_dict["exit_taker_cost"]
    max_positions = config_dict["max_positions"]
    
    total_pnl = 0.0
    total_trades = 0
    total_tp_exits = 0
    total_opposite_exits = 0
    total_wins = 0
    symbols_traded = 0
    max_concurrent = 0
    total_positions_opened = 0
    
    for data in all_data_list:
        if len(data) < bb_length + 10:
            continue
        
        pnl, trades, tp_exits, opposite_exits, wins, sym_max_concurrent, sym_positions = fast_backtest_symbol_multipos(
            data, bb_length, mult, dev_mult, tp_lookahead, total_fee_pct,
            min_vol, base_size, entry_cost, exit_maker, exit_taker, max_positions
        )
        
        total_pnl += pnl
        total_trades += trades
        total_tp_exits += tp_exits
        total_opposite_exits += opposite_exits
        total_wins += wins
        total_positions_opened += sym_positions
        max_concurrent = max(max_concurrent, sym_max_concurrent)
        if trades > 0:
            symbols_traded += 1
    
    return bb_length, mult, total_pnl, total_trades, total_tp_exits, total_opposite_exits, total_wins, symbols_traded, max_concurrent, total_positions_opened


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class ParameterOptimizer:
    """Main optimizer that sweeps through parameter grid."""
    
    def __init__(self, config: OptConfig):
        self.config = config
        self.fetcher = DataFetcher(config)
        
    def generate_param_grid(self) -> List[Tuple[int, float]]:
        """Generate all parameter combinations."""
        bb_lengths = list(range(
            self.config.bb_length_start,
            self.config.bb_length_end + 1,
            self.config.bb_length_step
        ))
        
        mults = []
        m = self.config.mult_start
        while m <= self.config.mult_end + 0.001:
            mults.append(round(m, 1))
            m += self.config.mult_step
        
        grid = [(bb, mult) for bb in bb_lengths for mult in mults]
        return grid

    def run(self) -> List[BacktestResult]:
        """Run the full optimization sweep with parallel processing."""
        print("\n" + "═" * 70)
        print("  🔬 DEVIATION MAGNET PARAMETER OPTIMIZER (FAST MODE)")
        print("═" * 70)
        
        # 1. Fetch symbols and data
        symbols = self.fetcher.fetch_active_symbols()
        if not symbols:
            print("❌ No symbols found. Exiting.")
            return []
        
        all_data = self.fetcher.fetch_all_history(symbols)
        if not all_data:
            print("❌ No data fetched. Exiting.")
            return []
        
        # 2. Generate parameter grid
        param_grid = self.generate_param_grid()
        total_combos = len(param_grid)
        
        print(f"\n📊 Parameter Grid:")
        print(f"   bb_length: {self.config.bb_length_start} to {self.config.bb_length_end} (step {self.config.bb_length_step})")
        print(f"   mult: {self.config.mult_start} to {self.config.mult_end} (step {self.config.mult_step})")
        print(f"   Total combinations: {total_combos:,}")
        print(f"   Symbols with data: {len(all_data)}")
        
        # 3. Prepare data for parallel processing
        # Convert dict to list (dicts aren't easily picklable for multiprocessing)
        all_data_list = list(all_data.values())
        
        # Create config dict for workers
        # total_fee_pct matches forward tester: (fee + maker + slippage*2) * 100
        total_fee_pct = (self.config.fee_pct + self.config.maker_fee_pct + self.config.slippage_pct * 2) * 100
        config_dict = {
            "dev_mult": self.config.dev_mult,
            "tp_lookahead": self.config.tp_lookahead,
            "total_fee_pct": total_fee_pct,
            "min_volatility_pct": self.config.min_volatility_pct,
            "base_order_size": self.config.base_order_size,
            "entry_cost": self.config.fee_pct + self.config.slippage_pct,
            "exit_maker_cost": self.config.maker_fee_pct + self.config.slippage_pct,
            "exit_taker_cost": self.config.fee_pct + self.config.slippage_pct,
            "max_positions": self.config.max_positions_total,
        }
        
        # 4. Run backtests in parallel
        print(f"\n⏳ Running {total_combos:,} backtests...")
        num_workers = self.config.num_processes or multiprocessing.cpu_count()
        print(f"   Using {num_workers} parallel workers")
        
        start_time = time.time()
        
        # Prepare args for all param combos
        work_items = [
            (bb_length, mult, all_data_list, config_dict)
            for bb_length, mult in param_grid
        ]
        
        raw_results = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(backtest_params_fast, item): i for i, item in enumerate(work_items)}
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                raw_results.append(result)
                completed += 1
                
                # Progress
                if completed % 50 == 0 or completed == total_combos:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_combos - completed) / rate if rate > 0 else 0
                    print(f"   Progress: {completed:,}/{total_combos:,} ({completed / total_combos * 100:.1f}%) | "
                          f"Rate: {rate:.1f}/s | ETA: {remaining:.0f}s")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed {total_combos:,} backtests in {elapsed:.1f}s ({total_combos / elapsed:.1f}/s)")
        
        # 5. Convert raw results to BacktestResult objects
        results = []
        filtered_count = 0
        
        for bb_length, mult, total_pnl, total_trades, tp_exits, opposite_exits, wins, symbols_traded, max_concurrent, total_positions in raw_results:
            if total_trades < self.config.min_trades:
                continue
            
            losses = total_trades - wins
            win_rate = wins / total_trades * 100 if total_trades > 0 else 0
            
            # Filter by target win rate if set
            if win_rate < self.config.target_win_rate:
                filtered_count += 1
                continue
            
            # Calculate profit factor properly
            # Need to track gross profit and loss separately for true profit factor
            # For now use wins/losses as proxy
            if losses == 0:
                profit_factor = float('inf')
            elif wins == 0:
                profit_factor = 0.0
            else:
                # Approximate: assume avg win = avg_pnl when winning, avg loss = avg_pnl when losing
                profit_factor = wins / losses if losses > 0 else float('inf')
            
            avg_positions = total_positions / symbols_traded if symbols_traded > 0 else 0
            
            results.append(BacktestResult(
                bb_length=bb_length,
                mult=mult,
                total_sigma=mult * self.config.dev_mult,
                total_trades=total_trades,
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                tp_exits=tp_exits,
                opposite_exits=opposite_exits,
                tp_rate=tp_exits / total_trades * 100 if total_trades > 0 else 0,
                total_pnl_usd=total_pnl,
                avg_pnl_per_trade=total_pnl / total_trades if total_trades > 0 else 0,
                best_trade=0.0,  # Not tracked in fast mode
                worst_trade=0.0,  # Not tracked in fast mode
                sharpe_ratio=0.0,  # Not tracked in fast mode
                profit_factor=profit_factor,
                max_drawdown_pct=0.0,  # Not tracked in fast mode
                avg_hold_bars=0.0,  # Not tracked in fast mode
                symbols_traded=symbols_traded,
                max_concurrent_positions=max_concurrent,
                avg_positions_per_signal=avg_positions,
            ))
        
        print(f"\n📊 Results Summary:")
        print(f"   Total combinations tested: {total_combos:,}")
        print(f"   Below min trades ({self.config.min_trades}): {total_combos - len(results) - filtered_count:,}")
        if self.config.target_win_rate > 0:
            print(f"   Below {self.config.target_win_rate}% win rate: {filtered_count:,}")
        print(f"   Valid results: {len(results):,}")
        
        # 6. Sort by total PnL (most profitable first)
        results.sort(key=lambda r: r.total_pnl_usd, reverse=True)
        
        return results

    def save_results(self, results: List[BacktestResult]):
        """Save results to CSV."""
        if not results:
            return
        
        fieldnames = [
            "rank", "bb_length", "mult", "total_sigma",
            "total_trades", "tp_exits", "opposite_exits", "tp_rate",
            "wins", "losses", "win_rate",
            "total_pnl_usd", "avg_pnl_per_trade",
            "profit_factor", "symbols_traded",
            "max_concurrent_positions", "avg_positions_per_signal"
        ]
        
        with open(self.config.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for rank, r in enumerate(results, 1):
                writer.writerow({
                    "rank": rank,
                    "bb_length": r.bb_length,
                    "mult": r.mult,
                    "total_sigma": round(r.total_sigma, 2),
                    "total_trades": r.total_trades,
                    "tp_exits": r.tp_exits,
                    "opposite_exits": r.opposite_exits,
                    "tp_rate": round(r.tp_rate, 2),
                    "wins": r.wins,
                    "losses": r.losses,
                    "win_rate": round(r.win_rate, 2),
                    "total_pnl_usd": round(r.total_pnl_usd, 2),
                    "avg_pnl_per_trade": round(r.avg_pnl_per_trade, 4),
                    "profit_factor": round(r.profit_factor, 4) if r.profit_factor != float('inf') else "inf",
                    "symbols_traded": r.symbols_traded,
                    "max_concurrent_positions": r.max_concurrent_positions,
                    "avg_positions_per_signal": round(r.avg_positions_per_signal, 1),
                })
        
        print(f"\n💾 Full results saved to: {self.config.output_file}")

    def print_leaderboard(self, results: List[BacktestResult]):
        """Print top performers to console."""
        if not results:
            print("\n❌ No parameter combinations met the criteria!")
            print("   Try: Lowering --win_rate, adjusting parameter ranges, or using more candles")
            return
        
        top_n = min(self.config.top_n, len(results))
        
        print("\n" + "═" * 130)
        print(f"  🏆 TOP {top_n} PARAMETER COMBINATIONS (Multi-Position Mode, Ranked by Total PnL)")
        print("═" * 130)
        print(f"{'Rank':<5} {'BB Len':<8} {'Mult':<6} {'Sigma':<7} {'Trades':<8} {'TP%':<7} {'Opp%':<7} {'Win%':<7} "
              f"{'PnL $':<10} {'$/Trade':<9} {'MaxPos':<8} {'Symbols':<8}")
        print("─" * 130)
        
        for rank, r in enumerate(results[:top_n], 1):
            opp_rate = r.opposite_exits / r.total_trades * 100 if r.total_trades > 0 else 0
            print(f"{rank:<5} {r.bb_length:<8} {r.mult:<6.1f} {r.total_sigma:<7.1f} {r.total_trades:<8} "
                  f"{r.tp_rate:<7.1f} {opp_rate:<7.1f} {r.win_rate:<7.1f} {r.total_pnl_usd:<10.2f} {r.avg_pnl_per_trade:<9.4f} "
                  f"{r.max_concurrent_positions:<8} {r.symbols_traded:<8}")
        
        print("═" * 130)
        
        # Recommendations
        if results:
            best = results[0]
            print(f"\n📌 BEST PNL: bb_length={best.bb_length}, mult={best.mult} (Sigma: {best.total_sigma:.1f})")
            print(f"   {best.total_trades} trades | {best.win_rate:.1f}% win rate | {best.tp_rate:.1f}% TP | ${best.total_pnl_usd:.2f} PnL")
            
            # Show highest win rate
            best_wr = max(results, key=lambda r: (r.win_rate, r.total_pnl_usd))
            if best_wr != best:
                print(f"\n🎯 BEST WIN RATE: bb_length={best_wr.bb_length}, mult={best_wr.mult}")
                print(f"   {best_wr.total_trades} trades | {best_wr.win_rate:.1f}% win rate | ${best_wr.total_pnl_usd:.2f} PnL")
            
            # Show most trades
            most_trades = max(results, key=lambda r: r.total_trades)
            if most_trades != best and most_trades != best_wr:
                print(f"\n📈 MOST TRADES: bb_length={most_trades.bb_length}, mult={most_trades.mult}")
                print(f"   {most_trades.total_trades} trades | {most_trades.win_rate:.1f}% win rate | ${most_trades.total_pnl_usd:.2f} PnL")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize Deviation Magnet parameters")
    
    # BB Length range
    parser.add_argument("--bb_start", type=int, default=5, help="BB Length start (default: 5)")
    parser.add_argument("--bb_end", type=int, default=200, help="BB Length end (default: 200)")
    parser.add_argument("--bb_step", type=int, default=5, help="BB Length step (default: 5)")
    
    # Mult range
    parser.add_argument("--mult_start", type=float, default=1.0, help="Mult start (default: 1.0)")
    parser.add_argument("--mult_end", type=float, default=50.0, help="Mult end (default: 50.0)")
    parser.add_argument("--mult_step", type=float, default=0.5, help="Mult step (default: 0.5)")
    
    # Data settings
    parser.add_argument("--candles", type=int, default=5000, help="Candles per symbol (default: 5000, paginates)")
    parser.add_argument("--lookahead", type=int, default=50, help="TP lookahead bars (default: 50)")
    parser.add_argument("--min_trades", type=int, default=5, help="Minimum trades for ranking (default: 5)")
    parser.add_argument("--win_rate", type=float, default=0.0, help="Min win rate %% filter (default: 0, no filter)")
    parser.add_argument("--max_positions", type=int, default=100, help="Max positions per direction (default: 100)")
    
    # Output
    parser.add_argument("--top", type=int, default=10, help="Show top N results (default: 10)")
    parser.add_argument("--output", type=str, default="optimization_results.csv", help="Output CSV file")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = OptConfig(
        bb_length_start=args.bb_start,
        bb_length_end=args.bb_end,
        bb_length_step=args.bb_step,
        mult_start=args.mult_start,
        mult_end=args.mult_end,
        mult_step=args.mult_step,
        candles_per_symbol=args.candles,
        tp_lookahead=args.lookahead,
        min_trades=args.min_trades,
        target_win_rate=args.win_rate,
        max_positions_total=args.max_positions,
        top_n=args.top,
        output_file=Path(args.output),
    )
    
    optimizer = ParameterOptimizer(config)
    results = optimizer.run()
    
    optimizer.save_results(results)
    optimizer.print_leaderboard(results)
    
    print("\n✅ Optimization complete!")
    print(f"   Run the forward tester with the recommended parameters:")
    if results:
        best = results[0]
        print(f"   python deviation_magnet_forward.py  # After editing bb_length={best.bb_length}, mult={best.mult}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
