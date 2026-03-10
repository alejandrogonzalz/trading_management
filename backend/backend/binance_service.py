import os
from binance.client import Client
from binance.enums import *
from fastapi import HTTPException
import time
from typing import Optional
from .config import settings

# Initialize Binance Client with sanitized keys
binance_client = Client(
    settings.BINANCE_API_KEY.strip().replace('"', '').replace("'", ""),
    settings.BINANCE_API_SECRET.strip().replace('"', '').replace("'", "")
)

def sync_binance_time():
    """
    Synchronizes the local client timestamp with Binance server time.
    Crucial for preventing 400 Bad Request / Timestamp errors.
    """
    try:
        server_time = binance_client.get_server_time()['serverTime']
        local_time = int(time.time() * 1000)
        binance_client.timestamp_offset = server_time - local_time
        print(f"Synced time. Offset: {binance_client.timestamp_offset}ms")
    except Exception as e:
        print(f"Error syncing time: {e}")

# Initial sync
sync_binance_time()

def get_24hr_tickers():
    try:
        return binance_client.get_ticker()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def get_symbols():
    try:
        info = binance_client.get_exchange_info()
        symbols = [s['symbol'] for s in info['symbols'] if s['symbol'].endswith('USDC') and s['status'] == 'TRADING']
        return sorted(symbols)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def get_balances():
    try:
        sync_binance_time() # Sync immediately before request
        info = binance_client.get_account(recvWindow=60000)
        balances = [b for b in info['balances'] if b['asset'] == 'USDC' or float(b['free']) > 0]
        return balances
    except Exception as e:
        print(f"Balance error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

def get_open_orders():
    try:
        sync_binance_time()
        orders = binance_client.get_open_orders(recvWindow=60000)
        usdc_orders = [o for o in orders if 'USDC' in o['symbol']]
        return usdc_orders
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def create_smart_trade(symbol: str, quantity: float, buy_price: Optional[float], take_profit_price: float, stop_loss_price: float):
    try:
        sync_binance_time()
        entry_order = binance_client.create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET if not buy_price else ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC if buy_price else None,
            quantity=quantity,
            price=buy_price if buy_price else None,
            recvWindow=60000
        )
        
        oco_order = None
        if take_profit_price > 0 or stop_loss_price > 0:
            oco_params = {
                "symbol": symbol,
                "side": SIDE_SELL,
                "quantity": quantity,
                "recvWindow": 60000
            }
            if take_profit_price > 0: oco_params["price"] = take_profit_price
            if stop_loss_price > 0:
                oco_params["stopPrice"] = stop_loss_price
                oco_params["stopLimitPrice"] = stop_loss_price
                oco_params["stopLimitTimeInForce"] = TIME_IN_FORCE_GTC
            
            oco_order = binance_client.create_oco_order(**oco_params)
        
        return {"entry": entry_order, "exit_strategy": oco_order}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def get_trade_history(symbol: Optional[str] = None):
    try:
        sync_binance_time()
        params = {"recvWindow": 60000}
        if symbol:
            params["symbol"] = symbol
            orders = binance_client.get_all_orders(**params)
        else:
            # Fallback to BTCUSDC history if no symbol provided
            orders = binance_client.get_all_orders(symbol="BTCUSDC", **params)
        
        history = [o for o in orders if o['status'] in ['FILLED', 'CANCELED', 'REJECTED']]
        return sorted(history, key=lambda x: x['updateTime'], reverse=True)[:50]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def cancel_order(symbol: str, order_id: int):
    try:
        sync_binance_time()
        return binance_client.cancel_order(symbol=symbol, orderId=order_id, recvWindow=60000)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def market_close_position(symbol: str):
    try:
        sync_binance_time()
        binance_client.cancel_all_open_orders(symbol=symbol, recvWindow=60000)
        balance_info = binance_client.get_account(recvWindow=60000)
        asset_to_sell = symbol.replace('USDC', '')
        available_quantity = 0
        for balance in balance_info['balances']:
            if balance['asset'] == asset_to_sell:
                available_quantity = float(balance['free'])
                break
        
        if available_quantity <= 0:
            raise HTTPException(status_code=400, detail=f"No available {asset_to_sell} to sell.")

        return binance_client.create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=available_quantity,
            recvWindow=60000
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
