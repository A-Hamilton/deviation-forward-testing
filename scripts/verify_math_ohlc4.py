
import numpy as np
import pandas as pd

def test_ohlc4_settings(length=10, mult=5.0, dev_mult=1.5):
    total_mult = mult * dev_mult
    print(f"Testing OHLC4 Logic: Length={length}, Total Multiplier={total_mult}")
    
    # Create baseline data (flat)
    # Open, High, Low, Close
    data = np.zeros((100, 4)) + 100.0
    
    # Inject a massive candle at the end
    # Open=100, High=100, Low=10, Close=10
    # This is a huge crash.
    # OHLC4 = (100+100+10+10)/4 = 55
    
    # Previous 9 candles: OHLC4 = 100
    # Current candle: OHLC4 = 55
    
    # Array of OHLC4s: [100, 100, ..., 100, 55]
    ohlc4s = np.array([100.0] * 9 + [55.0])
    
    mean = np.mean(ohlc4s)
    std = np.std(ohlc4s)
    
    print(f"  OHLC4s: {ohlc4s}")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std: {std:.4f}")
    
    # Lower Band = Mean - (7.5 * Std)
    lower = mean - (total_mult * std)
    print(f"  Lower Band: {lower:.4f}")
    
    # Check Signal: Close <= Lower
    close = 10.0
    print(f"  Close: {close}")
    
    if close <= lower:
        print(f"  ✅ SIGNAL FOUND! Close {close} <= Lower {lower}")
        return True
    else:
        print(f"  ❌ NO SIGNAL. Close {close} > Lower {lower}")
        
        # Calculate max possible deviation
        # The 'Close' is 10. The OHLC4 is 55.
        # The 'Close' is much lower than the OHLC4.
        # This 'extra' deviation helps us!
        
        return False

if __name__ == "__main__":
    test_ohlc4_settings()
