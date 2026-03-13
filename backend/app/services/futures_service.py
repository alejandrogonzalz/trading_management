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

class FuturesService:
    def __init__(self):
        self.api_key = settings.LEAD_API_KEY
        self.api_secret = settings.LEAD_API_SECRET
        
        if self.api_key and self.api_secret:
            self.config = ConfigurationRestAPI(
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            # Management & Whitelist
            self.copy_client = CopyTradingRestAPI(self.config)
            # Order Execution (This is what triggers copy-trading for followers)
            self.lead_client = DerivativesTradingUsdsFuturesRestAPI(self.config)
        else:
            self.copy_client = None
            self.lead_client = None

    def _ensure_client(self):
        if not self.lead_client or not self.copy_client:
            raise HTTPException(
                status_code=400, 
                detail="Lead Trading API keys not configured in backend .env"
            )

    def get_lead_status(self):
        """Returns the user's status in the Lead Trading ecosystem."""
        self._ensure_client()
        try:
            return self.copy_client.get_futures_lead_trader_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Binance Error: {str(e)}")

    def get_tradable_symbols(self):
        """Returns symbols whitelisted for Lead Trading."""
        self._ensure_client()
        try:
            return self.copy_client.get_futures_lead_trading_symbol_whitelist()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Binance Error: {str(e)}")

    def set_leverage(self, symbol: str, leverage: int):
        """Sets leverage for a specific symbol."""
        self._ensure_client()
        try:
            params = {
                "symbol": symbol.upper(),
                "leverage": leverage
            }
            return self.lead_client.change_initial_leverage(**params)
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
            response = self.lead_client.new_order(**params)
            
            # Log to audit
            audit_service.log_api_call("POST", "lead/order", params, response)
            
            return response
        except Exception as e:
            audit_service.log_api_call("POST", "lead/order-FAILED", {"symbol": symbol}, str(e), 400)
            raise HTTPException(status_code=400, detail=f"Order Failed: {str(e)}")

    def get_active_positions(self, symbol: str = None):
        """Retrieves currently open Futures positions."""
        self._ensure_client()
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol.upper()
            
            raw_positions = self.lead_client.position_information_v2(**params)
            # Filter for non-zero positions
            return [p for p in raw_positions if float(p.get('positionAmt', 0)) != 0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Position Error: {str(e)}")

    def get_open_orders(self, symbol: str = None):
        """Retrieves pending Lead orders."""
        self._ensure_client()
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol.upper()
            return self.lead_client.current_all_open_orders(**params)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Open Orders Error: {str(e)}")

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
            
            pos = positions[0]
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
            
            response = self.lead_client.new_order(**params)
            audit_service.log_api_call("POST", "lead/close-position", params, response)
            
            return response
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Close Failed: {str(e)}")

# Singleton instance
futures_service = FuturesService()
