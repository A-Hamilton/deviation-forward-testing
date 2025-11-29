import pandas as pd
import numpy as np
from collections import deque
from deviation_magnet_forward import DeviationMagnetStrategy, Config, BandData

def test_bands():
    config = Config(bb_length=20, mult=2.0, dev_mult=1.5)
    strategy = DeviationMagnetStrategy(config)
    
    # Generate random data
    np.random.seed(42)
    closes = np.random.normal(100, 1, 100)
    highs = closes + 0.5
    lows = closes - 0.5
    opens = closes
    
    # Create DataFrame for "old" logic (simulated)
    df = pd.DataFrame({
        "close": closes,
        "high": highs,
        "low": lows,
        "open": opens,
        "open_time": pd.date_range("2023-01-01", periods=100, freq="1min")
    })
    
    # Create deque for "new" logic
    buffer = deque(maxlen=300)
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        buffer.append(row)
        
    # Calculate using new fast method
    result_fast = strategy.calculate_bands_fast(buffer)
    
    # Calculate using pandas manually to verify
    df["ohlc4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    df["basis"] = df["ohlc4"].rolling(20).mean()
    df["stdev"] = df["ohlc4"].rolling(20).std(ddof=0)
    df["dev"] = 2.0 * df["stdev"]
    df["upper3"] = df["basis"] + df["dev"] * 1.5
    df["lower3"] = df["basis"] - df["dev"] * 1.5
    
    latest_pd = df.iloc[-1]
    
    print(f"Fast Basis: {result_fast.basis:.6f}")
    print(f"Pandas Basis: {latest_pd['basis']:.6f}")
    
    assert np.isclose(result_fast.basis, latest_pd["basis"]), "Basis mismatch"
    assert np.isclose(result_fast.upper3, latest_pd["upper3"]), "Upper3 mismatch"
    assert np.isclose(result_fast.lower3, latest_pd["lower3"]), "Lower3 mismatch"
    
    print("Verification Passed: Numpy implementation matches Pandas logic.")

if __name__ == "__main__":
    test_bands()
