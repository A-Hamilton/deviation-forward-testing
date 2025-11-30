
import numpy as np
import pandas as pd

def test_settings(length=10, mult=5.0, dev_mult=1.5):
    total_mult = mult * dev_mult
    print(f"Testing Length={length}, Total Multiplier={total_mult}")
    
    # Create random data
    data = np.random.normal(100, 1, 1000)
    
    # Inject massive outliers to try and force a signal
    # We need the price to be > mean + (total_mult * stdev)
    # But the price ITSELF is part of the mean and stdev calculation.
    
    # Let's try a brute force approach with a sliding window
    signals = 0
    max_z = 0
    
    for i in range(length, len(data)):
        window = data[i-length+1 : i+1] # Window includes current point i
        
        # Inject an outlier at the current point i
        # Try a range of values to see if ANY can trigger it
        original_val = window[-1]
        
        # Test values from 100 to 10000
        for shock in [105, 110, 120, 150, 200, 500, 1000, 10000]:
            window[-1] = shock
            
            mean = np.mean(window)
            std = np.std(window) # ddof=0 matches numpy default in bot
            
            upper = mean + (total_mult * std)
            
            # Z-score of the current point relative to the window it is IN
            if std > 0:
                z_score = (shock - mean) / std
                max_z = max(max_z, z_score)
            
            if shock > upper:
                print(f"  SIGNAL FOUND! Value={shock}, Mean={mean:.2f}, Std={std:.2f}, Upper={upper:.2f}")
                print(f"  Window: {window}")
                return True
                
        # Restore
        window[-1] = original_val

    print(f"  No signals found. Max theoretical Z-score achieved: {max_z:.4f}")
    return False

def test_lagged_settings(length=10, mult=5.0, dev_mult=1.5):
    total_mult = mult * dev_mult
    print(f"Testing LAGGED Length={length}, Total Multiplier={total_mult}")
    
    data = np.random.normal(100, 1, 1000)
    
    for i in range(length, len(data)):
        # Window EXCLUDES current point i
        window = data[i-length : i] 
        current_val = data[i]
        
        # Inject outlier
        for shock in [105, 110, 120, 150]:
            current_val = shock
            
            mean = np.mean(window)
            std = np.std(window)
            
            upper = mean + (total_mult * std)
            
            if current_val > upper:
                print(f"  SIGNAL FOUND! Value={current_val}, Mean={mean:.2f}, Std={std:.2f}, Upper={upper:.2f}")
                return True
    
    print("  No signals found.")
    return False

if __name__ == "__main__":
    print("--- Verifying User Settings (Standard) ---")
    test_settings(length=10, mult=5.0, dev_mult=1.5)
    
    print("\n--- Verifying User Settings (Lagged) ---")
    test_lagged_settings(length=10, mult=5.0, dev_mult=1.5)
