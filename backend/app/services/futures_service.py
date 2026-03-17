import os
import time
import math
from typing import Optional, List, Dict, Any
from fastapi import HTTPException

# Specialized Binance SDKs for Lead/Futures
from binance_sdk_copy_trading.rest_api import CopyTradingRestAPI
from binance_sdk_derivatives_trading_usds_futures.rest_api import DerivativesTradingUsdsFuturesRestAPI
from binance_common.configuration import ConfigurationRestAPI

from app.core.config import settings
from app.services import audit_service
from app.db.database import db
from app.services.binance_service import binance_client

import binance_common.utils as binance_utils

# Isolated collection for Lead/Futures trades
lead_trades_collection = db.lead_trades

class FuturesService:
    _time_offset = 0

    def __init__(self):
        self.api_key = settings.LEAD_API_KEY
        self.api_secret = settings.LEAD_API_SECRET
        
        if self.api_key and self.api_secret:
            # Copy Trading Configuration (uses standard API)
            self.copy_config = ConfigurationRestAPI(
                api_key=self.api_key,
                api_secret=self.api_secret,
                base_path="https://api.binance.com"
            )
            
            # Futures Configuration (uses fapi)
            self.futures_config = ConfigurationRestAPI(
                api_key=self.api_key,
                api_secret=self.api_secret,
                base_path="https://fapi.binance.com"
            )
            
            self.sync_time()
            
            self.copy_client = CopyTradingRestAPI(self.copy_config)
            self.lead_client = DerivativesTradingUsdsFuturesRestAPI(self.futures_config)
        else:
            self.copy_client = None
            self.lead_client = None

    def sync_time(self):
        """Syncs local time offset with Binance server time via monkeypatching."""
        try:
            res = binance_client.get_server_time()
            server_time = res['serverTime']
            local_time = int(time.time() * 1000)
            FuturesService._time_offset = server_time - local_time
            
            # Monkeypatch the SDK's internal timestamp generator
            def patched_get_timestamp():
                return int(time.time() * 1000) + FuturesService._time_offset
            
            binance_utils.get_timestamp = patched_get_timestamp
            print(f"Lead Sync: Server Time Offset = {FuturesService._time_offset}ms")
        except Exception as e:
            print(f"Lead Sync Time Error: {e}")

    def _ensure_client(self):
        if not self.lead_client or not self.copy_client:
            raise HTTPException(
                status_code=400, 
                detail="Lead Trading API keys not configured in backend .env"
            )

    def get_balances(self):
        """Retrieves USDS-M Futures wallet balances."""
        self._ensure_client()
        try:
            # v2 returns account info including balances
            res = self.lead_client.account_information_v2()
            return res.data()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Futures Balance Error: {str(e)}")

    def get_lead_history(self):
        """Retrieves closed Lead/Futures trades from MongoDB."""
        cursor = lead_trades_collection.find({"status": "CLOSED"}).sort("close_time", -1)
        trades = []
        for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            trades.append(doc)
        return trades

    def get_lead_status(self):
        """Returns the user's status. Silently handles missing keys."""
        if not self.copy_client:
            return {"status": "RESTRICTED", "message": "API Keys Not Configured"}
        try:
            res = self.copy_client.get_futures_lead_trader_status()
            return res.data()
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def get_tradable_symbols(self):
        """Returns whitelisted symbols. Returns empty list if keys missing."""
        if not self.copy_client:
            return []
        try:
            res = self.copy_client.get_futures_lead_trading_symbol_whitelist()
            return res.data()
        except Exception as e:
            print(f"Lead Whitelist Fetch Error: {e}")
            return []

    def set_leverage(self, symbol: str, leverage: int):
        """Sets leverage for a specific symbol."""
        self._ensure_client()
        try:
            params = {
                "symbol": symbol.upper(),
                "leverage": leverage
            }
            res = self.lead_client.change_initial_leverage(**params)
            return res.data()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Leverage Error: {str(e)}")

    def create_lead_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None):
        """
        Places a USDS-M Futures order. 
        If account is a Lead account, followers will copy this automatically.
        """
        self._ensure_client()
        try:
            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": quantity
            }
            if price:
                params["price"] = price
                params["timeInForce"] = "GTC"

            # Execute via specialized Lead Order Client
            res = self.lead_client.new_order(**params)
            response = res.data()
            
            # Log to audit
            audit_service.log_api_call("POST", "lead/order", params, response)
            
            return response
        except Exception as e:
            audit_service.log_api_call("POST", "lead/order-FAILED", {"symbol": symbol}, str(e), 400)
            raise HTTPException(status_code=400, detail=f"Order Failed: {str(e)}")

    def create_smart_lead_order(self, symbol: str, side: str, quantity: float, tp_price: float, sl_price: float, leverage: int = 10):
        """
        Executes a Lead Entry order and initializes background management for TP/SL.
        Stored in an isolated 'lead_trades' collection.
        """
        self._ensure_client()
        try:
            # 1. Set Leverage first
            self.set_leverage(symbol, leverage)
            
            # 2. Execute Entry (Market for immediate follow)
            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": "MARKET",
                "quantity": quantity
            }
            
            res = self.lead_client.new_order(**params)
            entry_res = res.data()
            audit_service.log_api_call("POST", "lead/smart-entry", params, entry_res)
            
            # 3. Store Metadata for management
            trade_id = f"LEAD_{int(time.time())}"
            metadata = {
                "_id": trade_id,
                "symbol": symbol.upper(),
                "side": side.upper(),
                "entry_price": float(entry_res.get('avgPrice', 0)),
                "quantity": quantity,
                "tp": tp_price,
                "sl": sl_price,
                "leverage": leverage,
                "status": "ACTIVE",
                "timestamp": time.time(),
                "entry_order_id": entry_res.get('orderId')
            }
            lead_trades_collection.insert_one(metadata)
            
            return {
                "trade_id": trade_id,
                "entry": entry_res,
                "status": "ACTIVE"
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Lead Smart Order Failed: {str(e)}")

    def get_active_positions(self, symbol: str = None):
        """Retrieves currently open Futures positions."""
        self._ensure_client()
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol.upper()
            
            res = self.lead_client.position_information_v2(**params)
            raw_positions = res.data()
            # Filter for non-zero positions. SDK uses snake_case attributes.
            return [p for p in raw_positions if float(p.position_amt or 0) != 0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Position Error: {str(e)}")

    def get_open_orders(self, symbol: str = None):
        """Retrieves pending Lead orders and merges with virtual Smart Trades."""
        self._ensure_client()
        try:
            # 1. Fetch real orders from Binance
            params = {"recv_window": 60000}
            if symbol:
                params["symbol"] = symbol.upper()
            res = self.lead_client.current_all_open_orders(**params)
            raw_orders = res.data()
            
            # 2. Fetch virtual setups from MongoDB
            query = {"status": "ACTIVE"}
            if symbol:
                query["symbol"] = symbol.upper()
            
            virtual_setups = list(lead_trades_collection.find(query))
            
            # 3. Format virtual setups to match the frontend 'SmartTradeCard' expectations
            formatted_setups = []
            for vs in virtual_setups:
                formatted_setups.append({
                    "orderId": vs["_id"],
                    "symbol": vs["symbol"],
                    "side": vs["side"],
                    "type": "POSITION", # Virtual tag
                    "origQty": vs["quantity"],
                    "price": vs["entry_price"],
                    "clientOrderId": vs["_id"],
                    "smart_meta": {
                        "entry_price": vs["entry_price"],
                        "tp": vs["tp"],
                        "sl": vs["sl"],
                        "side": vs["side"],
                        "quantity": vs["quantity"],
                        "leverage": vs["leverage"]
                    }
                })

            return raw_orders + formatted_setups
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Open Orders Error: {str(e)}")

    def cancel_order(self, symbol: str, order_id: int):
        """Cancels a pending Lead order."""
        self._ensure_client()
        try:
            res = self.lead_client.cancel_order(symbol=symbol.upper(), orderId=order_id)
            return res.data()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cancel Failed: {str(e)}")

    def get_binance_trade_history(self, symbol: str = None):
        """Retrieves raw execution history from USDS-M Futures."""
        self._ensure_client()
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol.upper()
            res = self.lead_client.get_account_trades(**params)
            return res.data()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"History Error: {str(e)}")

    def close_position(self, symbol: str, quantity: float = None):
        """
        Closes an active Lead position by placing an opposite Market order.
        """
        self._ensure_client()
        try:
            # 1. Get position details to find the current side
            positions = self.get_active_positions(symbol)
            if not positions:
                return {"status": "NO_POSITION_FOUND"}
            
            # Find the exact match for the symbol
            pos = next((p for p in positions if p['symbol'] == symbol.upper()), None)
            if not pos:
                return {"status": "NO_MATCHING_POSITION"}
                
            amt = float(pos.get('positionAmt', 0))
            if amt == 0:
                return {"status": "EMPTY_POSITION"}

            # 2. Determine exit side
            exit_side = "SELL" if amt > 0 else "BUY"
            exit_qty = abs(amt) if not quantity else quantity
            
            # 3. Execute Market Close
            params = {
                "symbol": symbol.upper(),
                "side": exit_side,
                "type": "MARKET",
                "quantity": exit_qty
            }
            
            res = self.lead_client.new_order(**params)
            response = res.data()
            audit_service.log_api_call("POST", "lead/close-position", params, response)
            
            # 4. Mark associated MongoDB trades as CLOSED
            # We look for all ACTIVE trades for this symbol in lead_trades_collection
            lead_trades_collection.update_many(
                {"symbol": symbol.upper(), "status": "ACTIVE"},
                {"$set": {
                    "status": "CLOSED",
                    "exit_price": float(response.get('avgPrice', 0)),
                    "close_time": time.time()
                }}
            )
            
            return response
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Close Failed: {str(e)}")

    def reconcile_lead_trades(self):
        """
        Background worker to sync lead positions.
        """
        # Note: Futures TP/SL management logic will be built here
        pass

# Singleton instance
futures_service = FuturesService()
