# Deviation Magnet Live Bot

This is the production version of the Deviation Magnet strategy.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install pybit pandas numpy python-dotenv
    ```

2.  **Configure API Keys**:
    *   Copy `.env.example` to `.env`.
    *   Edit `.env` and add your Bybit API Key and Secret.
    *   **Permissions**: The API key needs "Contract - Orders" and "Contract - Position" permissions.

3.  **Run**:
    ```bash
    python deviation_magnet_live.py
    ```

## Features

*   **Strict Risk Management**:
    *   1x Leverage (Hardcoded).
    *   Max 1 Position Global (Hardcoded).
    *   $10 Fixed Position Size.
*   **Execution**:
    *   Market Entry.
    *   Limit Order Take Profit (Maker Fees).
    *   Dynamic TP updates based on volatility.
    *   Private WebSocket for real-time order updates.
*   **Resilience**:
    *   State persistence (`live_state.json`).
    *   Startup reconciliation (syncs with exchange).

## Files

*   `deviation_magnet_live.py`: Main bot script.
*   `live_state.json`: Stores current position state (auto-generated).
*   `live_bot.log`: Log file.
