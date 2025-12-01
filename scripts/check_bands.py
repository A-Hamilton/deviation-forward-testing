"""Quick script to check band calculations for BTCUSDT"""
import numpy as np
from pybit.unified_trading import HTTP

# Fetch latest data
client = HTTP(testnet=False)
klines = client.get_kline(
    category="linear",
    symbol="BTCUSDT",
    interval="1",
    limit=20  # Need 20 candles INCLUDING current
)

if klines["retCode"] != 0:
    print(f"Error: {klines}")
    exit(1)

# Extract OHLC data (reversed to chronological order)
candles = klines["result"]["list"][::-1]

print(f"Fetched {len(candles)} candles for BTCUSDT\n")

# Take last 20 candles (INCLUDING current)
recent = candles[-20:]

# Calculate OHLC4 on all 20 candles
ohlc4_values = []

print("Stats Window (Last 20 candles INCLUDING current):")
for i, c in enumerate(recent):
    o = float(c[1])
    h = float(c[2])
    l = float(c[3])
    close = float(c[4])
    ohlc4 = (o + h + l + close) / 4.0
    ohlc4_values.append(ohlc4)
    print(f"  {i+1}. OHLC4 = {ohlc4:.2f}")

# Calculate stats
ohlc4_array = np.array(ohlc4_values)
basis = np.mean(ohlc4_array)
stdev = np.std(ohlc4_array, ddof=1)  # Sample stdev to match Pine Script

# Calculate bands (mult=3, dev_mult=1.5)
mult = 3.0
dev_mult = 1.5
dev = mult * stdev
upper3 = basis + (dev * dev_mult)
lower3 = basis - (dev * dev_mult)

# Current candle
current = recent[-1]
current_close = float(current[4])
current_high = float(current[2])
current_low = float(current[3])

print(f"\n{'='*60}")
print(f"CALCULATION RESULTS:")
print(f"{'='*60}")
print(f"Basis (Mean): {basis:.2f}")
print(f"Stdev:        {stdev:.4f}")
print(f"Dev:          {dev:.4f}")
print(f"\nUpper Band:   {upper3:.2f}")
print(f"Lower Band:   {lower3:.2f}")
print(f"\nCurrent Close: {current_close:.2f}")
print(f"Current High:  {current_high:.2f}")
print(f"Current Low:   {current_low:.2f}")
print(f"\n{'='*60}")
print(f"SIGNAL CHECK:")
print(f"{'='*60}")
print(f"Short Signal: high ({current_high:.2f}) >= upper ({upper3:.2f})? {current_high >= upper3}")
print(f"Long Signal:  low ({current_low:.2f}) <= lower ({lower3:.2f})? {current_low <= lower3}")
