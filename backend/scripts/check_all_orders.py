import os
import sys
import json
from typing import Dict, Any

# Setup path to import app
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.futures_service import futures_service

def check_all():
    symbol = "FORTHUSDT"
    print(f"Checking all open orders for {symbol}...")
    
    try:
        # 1. SDK method for open orders
        res = futures_service.lead_client.current_all_open_orders(symbol=symbol.upper())
        sdk_orders = res.data()
        print(f"\nSDK reported {len(sdk_orders)} open orders.")
        for o in sdk_orders:
            # SDK object inspection
            cid = getattr(o, 'client_order_id', 'N/A')
            oid = getattr(o, 'order_id', 'N/A')
            type = getattr(o, 'type', 'N/A')
            print(f" - ID: {oid} | ClientID: {cid} | Type: {type}")

        # 2. Standard client method for open orders
        std_orders = futures_service.std_client.futures_get_open_orders(symbol=symbol.upper())
        print(f"\nSTD Client reported {len(std_orders)} open orders.")
        for o in std_orders:
            cid = o.get('clientOrderId', 'N/A')
            oid = o.get('orderId', 'N/A')
            type = o.get('type', 'N/A')
            print(f" - ID: {oid} | ClientID: {cid} | Type: {type}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_all()
