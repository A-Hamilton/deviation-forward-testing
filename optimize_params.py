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
    
    # Parameter grid - REDUCED DEFAULT RANGE for speed
    bb_length_start: int = 5
    bb_length_end: int = 100
    bb_length_step: int = 5  # 20 values: 5, 10, 15, ... 100
    
    mult_start: float = 2.0
    mult_end: float = 8.0
    mult_step: float = 0.5  # 13 values: 2.0, 2.5, ... 8.0
    
    # Fixed strategy params (same as forward tester)
    dev_mult: float = 1.5
    base_order_size: float = 10.0
    dca_scale: float = 5.0
    max_dca_orders: int = 2
    
    # Fee structure (same as forward tester)
    fee_pct: float = 0.00055  # 0.055% taker
    maker_fee_pct: float = 0.0002  # 0.02% maker
    slippage_pct: float = 0.0005  # 0.05%
    
    # Data settings
    timeframe: str = "1"
    candles_per_symbol: int = 5000  # Will paginate to get this many (1000 per request)
    max_workers: int = 10
    tp_lookahead: int = 50  # Bars to look ahead for TP (increased for realism)
    min_trades: int = 5  # Minimum trades for ranking
    target_win_rate: float = 100.0  # Target win rate % (early stop if below)
    
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
    tp_hits: int  # Trades that hit TP within lookahead
    no_tp_exits: int  # Trades that didn't hit TP (timeout)
    tp_rate: float  # tp_hits / total_trades
    
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

def fast_backtest_symbol(data: np.ndarray, bb_length: int, mult: float, dev_mult: float,
                         tp_lookahead: int, total_fee_pct: float, min_vol: float,
                         base_size: float, entry_cost: float, exit_maker_cost: float,
                         exit_taker_cost: float, early_stop: bool = True) -> Tuple[float, int, int, int, bool]:
    """
    Ultra-fast backtest for a single symbol. 
    Returns (total_pnl, total_trades, tp_hits, wins, should_stop).
    
    Key features:
    - TP must hit on NEXT bar or later (no same-bar to avoid look-ahead bias)
    - TP = volatility + ALL fees (matches forward tester exactly)
    - Early stopping: Returns immediately if any trade loses (for 100% win rate search)
    - 50-bar lookahead for TP validation
    """
    n = len(data)
    if n < bb_length + tp_lookahead + 1:
        return 0.0, 0, 0, 0, False
    
    # Pre-extract columns
    opens = data[:, 0]
    highs = data[:, 1]
    lows = data[:, 2]
    closes = data[:, 3]
    
    # Pre-compute OHLC4 for all bars
    ohlc4 = (opens + highs + lows + closes) * 0.25
    
    # Pre-compute volatility for all bars
    vol_raw = (highs - lows) / closes
    
    # Pre-compute rolling stats for ALL bars (vectorized - much faster)
    # Using cumsum trick for rolling mean
    cumsum = np.concatenate([[0], np.cumsum(ohlc4)])
    rolling_sum = cumsum[bb_length:] - cumsum[:-bb_length]
    rolling_mean = rolling_sum / bb_length
    
    # Rolling std using cumsum trick: std = sqrt(E[x^2] - E[x]^2)
    cumsum_sq = np.concatenate([[0], np.cumsum(ohlc4 ** 2)])
    rolling_sum_sq = cumsum_sq[bb_length:] - cumsum_sq[:-bb_length]
    rolling_var = (rolling_sum_sq / bb_length) - (rolling_mean ** 2)
    # Bessel correction for sample std (ddof=1): multiply by n/(n-1)
    rolling_var = rolling_var * bb_length / (bb_length - 1)
    rolling_std = np.sqrt(np.maximum(rolling_var, 0))  # Clamp negative due to float precision
    
    # Rolling volatility mean
    vol_cumsum = np.concatenate([[0], np.cumsum(vol_raw)])
    vol_rolling_sum = vol_cumsum[bb_length:] - vol_cumsum[:-bb_length]
    rolling_vol = (vol_rolling_sum / bb_length) * 100
    
    total_pnl = 0.0
    total_trades = 0
    tp_hits = 0
    wins = 0
    
    # Start from bb_length (first valid index for rolling stats)
    idx = bb_length - 1  # Offset for pre-computed arrays
    
    while idx < len(rolling_mean) - tp_lookahead - 1:
        # Use pre-computed values (idx in rolling array = idx + bb_length - 1 in original)
        orig_idx = idx + bb_length - 1
        
        basis = rolling_mean[idx]
        stdev = rolling_std[idx]
        avg_vol = rolling_vol[idx]
        
        # Volatility check
        if avg_vol < min_vol:
            idx += 1
            continue
        
        dev = mult * stdev * dev_mult
        upper3 = basis + dev
        lower3 = basis - dev
        
        high = highs[orig_idx]
        low = lows[orig_idx]
        
        # Check entry (use integers for direction: 1=long, -1=short)
        entry_price = 0.0
        direction = 0
        
        if low <= lower3:
            entry_price = lower3
            direction = 1  # long
        elif high >= upper3:
            entry_price = upper3
            direction = -1  # short
        
        if direction == 0:
            idx += 1
            continue
        
        # Calculate TP target
        # TP = avg_volatility + total_fees (to cover costs AND capture movement)
        target_pct = (avg_vol + total_fee_pct) / 100
        
        if direction == 1:  # long
            target_price = entry_price * (1 + target_pct)
        else:  # short
            target_price = entry_price * (1 - target_pct)
        
        # Check lookahead window for TP (starting from NEXT bar - no same-bar TP)
        tp_hit = False
        exit_idx = orig_idx + tp_lookahead
        
        for check_idx in range(orig_idx + 1, min(orig_idx + tp_lookahead + 1, n)):
            if direction == 1 and highs[check_idx] >= target_price:
                tp_hit = True
                exit_idx = check_idx
                break
            elif direction == -1 and lows[check_idx] <= target_price:
                tp_hit = True
                exit_idx = check_idx
                break
        
        # Calculate PnL
        if tp_hit:
            exit_price = target_price
            tp_hits += 1
            exit_cost = exit_maker_cost
        else:
            exit_price = closes[min(exit_idx, n - 1)]
            exit_cost = exit_taker_cost
        
        price_ratio = exit_price / entry_price
        if direction == 1:  # long
            pnl_pct = (price_ratio - 1.0) * 100
        else:  # short
            pnl_pct = (1.0 - price_ratio) * 100
        
        # Fees (deduct from PnL - NOT already in TP target)
        entry_fee = base_size * entry_cost
        exit_fee = base_size * price_ratio * exit_cost
        gross_pnl = pnl_pct * 0.01 * base_size
        net_pnl = gross_pnl - entry_fee - exit_fee
        
        total_pnl += net_pnl
        total_trades += 1
        
        if net_pnl > 0:
            wins += 1
        elif early_stop:
            return total_pnl, total_trades, tp_hits, wins, True
        
        # Skip ahead past this trade (convert back to rolling array index)
        idx = exit_idx - bb_length + 2
    
    return total_pnl, total_trades, tp_hits, wins, False


def backtest_params_fast(args: Tuple) -> Tuple[int, float, float, int, int, int, int, bool]:
    """Worker function for parallel processing. 
    Returns (bb_length, mult, pnl, trades, tp_hits, wins, symbols, early_stopped)."""
    bb_length, mult, all_data_list, config_dict = args
    
    dev_mult = config_dict["dev_mult"]
    tp_lookahead = config_dict["tp_lookahead"]
    total_fee_pct = config_dict["total_fee_pct"]
    min_vol = config_dict["min_volatility_pct"]
    base_size = config_dict["base_order_size"]
    entry_cost = config_dict["entry_cost"]
    exit_maker = config_dict["exit_maker_cost"]
    exit_taker = config_dict["exit_taker_cost"]
    early_stop = config_dict.get("early_stop", True)
    
    total_pnl = 0.0
    total_trades = 0
    total_tp_hits = 0
    total_wins = 0
    symbols_traded = 0
    
    for data in all_data_list:
        if len(data) < bb_length + tp_lookahead + 1:
            continue
        
        pnl, trades, tp_hits, wins, should_stop = fast_backtest_symbol(
            data, bb_length, mult, dev_mult, tp_lookahead, total_fee_pct,
            min_vol, base_size, entry_cost, exit_maker, exit_taker, early_stop
        )
        
        total_pnl += pnl
        total_trades += trades
        total_tp_hits += tp_hits
        total_wins += wins
        if trades > 0:
            symbols_traded += 1
        
        # Early stop if any symbol had a losing trade
        if should_stop and early_stop:
            return bb_length, mult, total_pnl, total_trades, total_tp_hits, total_wins, symbols_traded, True
    
    return bb_length, mult, total_pnl, total_trades, total_tp_hits, total_wins, symbols_traded, False


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
            "early_stop": self.config.target_win_rate >= 100.0,  # Enable early stopping for 100% win rate search
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
        early_stopped_count = 0
        perfect_count = 0
        
        for bb_length, mult, total_pnl, total_trades, tp_hits, wins, symbols_traded, early_stopped in raw_results:
            if total_trades == 0:
                continue
            
            if early_stopped:
                early_stopped_count += 1
                continue  # Skip combos that had losing trades (didn't achieve 100% win rate)
            
            losses = total_trades - wins
            no_tp_exits = total_trades - tp_hits
            win_rate = wins / total_trades * 100 if total_trades > 0 else 0
            
            # Only include 100% win rate combos
            if win_rate < self.config.target_win_rate:
                continue
            
            perfect_count += 1
            
            results.append(BacktestResult(
                bb_length=bb_length,
                mult=mult,
                total_sigma=mult * self.config.dev_mult,
                total_trades=total_trades,
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                tp_hits=tp_hits,
                no_tp_exits=no_tp_exits,
                tp_rate=tp_hits / total_trades * 100 if total_trades > 0 else 0,
                total_pnl_usd=total_pnl,
                avg_pnl_per_trade=total_pnl / total_trades if total_trades > 0 else 0,
                best_trade=0.0,  # Not tracked in fast mode
                worst_trade=0.0,  # Not tracked in fast mode
                sharpe_ratio=0.0,  # Not tracked in fast mode
                profit_factor=float('inf') if losses == 0 else wins / losses,
                max_drawdown_pct=0.0,  # Not tracked in fast mode
                avg_hold_bars=0.0,  # Not tracked in fast mode
                symbols_traded=symbols_traded,
            ))
        
        print(f"\n📊 Results Summary:")
        print(f"   Early stopped (had losses): {early_stopped_count:,}")
        print(f"   100% win rate combos: {perfect_count:,}")
        
        # 6. Sort by total PnL (most profitable first)
        results.sort(key=lambda r: r.total_pnl_usd, reverse=True)
        
        return results

    def save_results(self, results: List[BacktestResult]):
        """Save results to CSV."""
        if not results:
            return
        
        fieldnames = [
            "rank", "bb_length", "mult", "total_sigma",
            "total_trades", "tp_hits", "no_tp_exits", "tp_rate",
            "wins", "losses", "win_rate",
            "total_pnl_usd", "avg_pnl_per_trade", "best_trade", "worst_trade",
            "sharpe_ratio", "profit_factor", "max_drawdown_pct",
            "avg_hold_bars", "symbols_traded"
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
                    "tp_hits": r.tp_hits,
                    "no_tp_exits": r.no_tp_exits,
                    "tp_rate": round(r.tp_rate, 2),
                    "wins": r.wins,
                    "losses": r.losses,
                    "win_rate": round(r.win_rate, 2),
                    "total_pnl_usd": round(r.total_pnl_usd, 2),
                    "avg_pnl_per_trade": round(r.avg_pnl_per_trade, 4),
                    "best_trade": round(r.best_trade, 4),
                    "worst_trade": round(r.worst_trade, 4),
                    "sharpe_ratio": round(r.sharpe_ratio, 4),
                    "profit_factor": round(r.profit_factor, 4) if r.profit_factor != float('inf') else "inf",
                    "max_drawdown_pct": round(r.max_drawdown_pct, 2),
                    "avg_hold_bars": round(r.avg_hold_bars, 1),
                    "symbols_traded": r.symbols_traded,
                })
        
        print(f"\n💾 Full results saved to: {self.config.output_file}")

    def print_leaderboard(self, results: List[BacktestResult]):
        """Print top performers to console."""
        if not results:
            print("\n❌ No parameter combinations achieved 100% win rate!")
            print("   Try: Increasing mult range, increasing bb_length, or using more candles")
            return
        
        top_n = min(self.config.top_n, len(results))
        
        print("\n" + "═" * 115)
        print(f"  🏆 TOP {top_n} PARAMETER COMBINATIONS (100% Win Rate, Ranked by Total PnL)")
        print("═" * 115)
        print(f"{'Rank':<5} {'BB Len':<8} {'Mult':<6} {'Sigma':<7} {'Trades':<8} {'TP%':<7} {'Win%':<7} "
              f"{'PnL $':<10} {'$/Trade':<9} {'PF':<8} {'Symbols':<8}")
        print("─" * 115)
        
        for rank, r in enumerate(results[:top_n], 1):
            pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float('inf') else "∞"
            print(f"{rank:<5} {r.bb_length:<8} {r.mult:<6.1f} {r.total_sigma:<7.1f} {r.total_trades:<8} "
                  f"{r.tp_rate:<7.1f} {r.win_rate:<7.1f} {r.total_pnl_usd:<10.2f} {r.avg_pnl_per_trade:<9.4f} "
                  f"{pf_str:<8} {r.symbols_traded:<8}")
        
        print("═" * 115)
        
        # Recommendations
        if results:
            best = results[0]
            print(f"\n📌 RECOMMENDED: bb_length={best.bb_length}, mult={best.mult} (Total Sigma: {best.total_sigma:.1f})")
            print(f"   Expected: {best.total_trades} trades, {best.tp_rate:.1f}% TP rate, 100% win rate, ${best.total_pnl_usd:.2f} PnL")
            
            # Also show highest trade count with 100% win rate
            most_trades = max(results, key=lambda r: r.total_trades)
            if most_trades != best:
                print(f"\n🎯 MOST TRADES (100% WR): bb_length={most_trades.bb_length}, mult={most_trades.mult}")
                print(f"   Expected: {most_trades.total_trades} trades, ${most_trades.total_pnl_usd:.2f} PnL")


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
    parser.add_argument("--mult_end", type=float, default=10.0, help="Mult end (default: 10.0)")
    parser.add_argument("--mult_step", type=float, default=0.5, help="Mult step (default: 0.5)")
    
    # Data settings
    parser.add_argument("--candles", type=int, default=5000, help="Candles per symbol (default: 5000, paginates)")
    parser.add_argument("--lookahead", type=int, default=50, help="TP lookahead bars (default: 50)")
    parser.add_argument("--min_trades", type=int, default=5, help="Minimum trades for ranking (default: 5)")
    parser.add_argument("--win_rate", type=float, default=100.0, help="Target win rate %% (default: 100)")
    
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
