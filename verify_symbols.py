from deviation_magnet_forward import Config, TradingState, BybitClient
import logging

# Setup dummy logging
logging.basicConfig(level=logging.INFO)

def test_fetch_symbols():
    config = Config()
    state = TradingState(config)
    client = BybitClient(config, state)
    
    print("Fetching symbols...")
    symbols = client.fetch_active_symbols()
    
    print(f"Total Symbols Fetched: {len(symbols)}")
    
    if len(symbols) > 400:
        print("Success: Fetched > 400 symbols (likely all USDT perps)")
    else:
        print(f"❌ Warning: Only fetched {len(symbols)} symbols. Check API or filters.")
        
    # Check common pairs
    for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "PEPEUSDT"]:
        if s in symbols:
            print(f"  - {s} found")
        else:
            print(f"  - {s} MISSING")

if __name__ == "__main__":
    test_fetch_symbols()
