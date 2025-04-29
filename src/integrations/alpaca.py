"""
Alpaca API client for paper trading.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import logging
import pandas as pd
import numpy as np
import re

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Load environment variables
load_dotenv()

# Custom JSON encoder to handle non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)  # Fallback to string representation

# Helper function for JSON serialization with our custom encoder
def safe_json_dumps(obj, **kwargs):
    return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)

class AlpacaClient:
    """Client for interacting with Alpaca API for paper trading."""
    
    def __init__(self, paper=True):
        """
        Initialize the Alpaca client.
        
        Args:
            paper: Whether to use paper trading (True) or live trading (False).
        """
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            init(autoreset=True)
            
            missing_keys = []
            if not self.api_key:
                missing_keys.append("ALPACA_API_KEY")
            if not self.api_secret:
                missing_keys.append("ALPACA_API_SECRET")
                
            print(f"\n{Fore.RED}ERROR: Missing Alpaca API credentials: {', '.join(missing_keys)}{Style.RESET_ALL}")
            print(f"Please add these to your .env file with the format:")
            print(f"{Fore.YELLOW}ALPACA_API_KEY=your_api_key_here")
            print(f"ALPACA_API_SECRET=your_api_secret_here{Style.RESET_ALL}")
            print(f"You can obtain these keys from your Alpaca account dashboard at https://app.alpaca.markets/paper/dashboard/overview\n")
            
            raise ValueError(f"Alpaca API credentials not found. Make sure {', '.join(missing_keys)} are set in your .env file.")
        
        # Set the appropriate base URL based on paper trading mode
        if paper:
            self.base_url = 'https://paper-api.alpaca.markets'
        else:
            self.base_url = 'https://api.alpaca.markets'
            
        # Initialize the Alpaca API client
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=self.base_url,
            api_version='v2'
        )
        
        print(f"Alpaca client initialized for {'paper' if paper else 'live'} trading.")
    
    def get_account(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict containing account information.
        """
        try:
            account = self.api.get_account()
            return {
                'id': account.id,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'daytrade_count': account.daytrade_count,
                'last_equity': float(account.last_equity),
                'status': account.status
            }
        except APIError as e:
            print(f"Error getting account: {e}")
            raise
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List of positions.
        """
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': p.symbol,
                    'qty': int(p.qty),
                    'avg_entry_price': float(p.avg_entry_price),
                    'market_value': float(p.market_value),
                    'current_price': float(p.current_price),
                    'unrealized_pl': float(p.unrealized_pl),
                    'unrealized_plpc': float(p.unrealized_plpc),
                    'side': 'long' if int(p.qty) > 0 else 'short'
                }
                for p in positions
            ]
        except APIError as e:
            print(f"Error getting positions: {e}")
            return []
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: The ticker symbol.
            
        Returns:
            Position information or None if not found.
        """
        try:
            position = self.api.get_position(symbol)
            return {
                'symbol': position.symbol,
                'qty': int(position.qty),
                'avg_entry_price': float(position.avg_entry_price),
                'market_value': float(position.market_value),
                'current_price': float(position.current_price),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'side': 'long' if int(position.qty) > 0 else 'short'
            }
        except APIError as e:
            if "position does not exist" in str(e).lower():
                return None
            print(f"Error getting position for {symbol}: {e}")
            raise
    
    def get_orders(self, status: str = 'open') -> List[Dict[str, Any]]:
        """
        Get orders with the specified status.
        
        Args:
            status: Order status ('open', 'closed', or 'all').
            
        Returns:
            List of orders.
        """
        try:
            if status == 'all':
                orders = self.api.list_orders()
            else:
                orders = self.api.list_orders(status=status)
                
            return [
                {
                    'id': o.id,
                    'client_order_id': o.client_order_id,
                    'symbol': o.symbol,
                    'side': o.side,
                    'qty': float(o.qty),
                    'filled_qty': float(o.filled_qty),
                    'type': o.type,
                    'time_in_force': o.time_in_force,
                    'limit_price': float(o.limit_price) if o.limit_price else None,
                    'stop_price': float(o.stop_price) if o.stop_price else None,
                    'status': o.status,
                    'created_at': o.created_at,
                    'updated_at': o.updated_at,
                    'submitted_at': o.submitted_at,
                    'filled_at': o.filled_at
                }
                for o in orders
            ]
        except APIError as e:
            print(f"Error getting {status} orders: {e}")
            return []
    
    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        type: str = 'market',
        time_in_force: str = 'day',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit an order to Alpaca.
        
        Args:
            symbol: The ticker symbol.
            qty: Quantity of shares.
            side: 'buy' or 'sell'.
            type: Order type ('market', 'limit', 'stop', 'stop_limit').
            time_in_force: Time in force ('day', 'gtc', 'opg', 'cls', 'ioc', 'fok').
            limit_price: Limit price for limit and stop-limit orders.
            stop_price: Stop price for stop and stop-limit orders.
            client_order_id: Custom client order ID.
            
        Returns:
            Order information.
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id
            )
            
            return {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'qty': float(order.qty),
                'filled_qty': float(order.filled_qty),
                'type': order.type,
                'time_in_force': order.time_in_force,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'status': order.status,
                'created_at': order.created_at
            }
        except APIError as e:
            print(f"Error submitting order for {symbol}: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by ID.
        
        Args:
            order_id: The order ID to cancel.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.api.cancel_order(order_id)
            return True
        except APIError as e:
            print(f"Error canceling order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders canceled.
        """
        try:
            canceled = self.api.cancel_all_orders()
            return len(canceled)
        except APIError as e:
            print(f"Error canceling all orders: {e}")
            return 0
    
    def get_clock(self) -> Dict[str, Any]:
        """
        Get market clock information.
        
        Returns:
            Market clock information.
        """
        try:
            clock = self.api.get_clock()
            return {
                'timestamp': clock.timestamp,
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close
            }
        except APIError as e:
            error_str = str(e).lower()
            if "forbidden" in error_str:
                raise ValueError(
                    "Authentication failed with Alpaca API. Your API keys may be invalid or missing. "
                    "Please check your ALPACA_API_KEY and ALPACA_API_SECRET in your .env file."
                )
            elif "unauthorized" in error_str:
                raise ValueError(
                    "Unauthorized access to Alpaca API. Your API keys don't have proper permissions. "
                    "Please verify your Alpaca account has the correct access level."
                )
            else:
                print(f"Error getting clock: {e}")
                raise
    
    def get_portfolio_history(
        self,
        period: str = '1W',
        timeframe: str = '1D',
        date_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get portfolio history.
        
        Args:
            period: Time period ('1D', '1W', '1M', '3M', '1A', etc.).
            timeframe: Time resolution ('1D', '1H', '15Min', etc.).
            date_end: End date for the data (defaults to current time).
            
        Returns:
            Portfolio history data.
        """
        try:
            history = self.api.get_portfolio_history(
                period=period,
                timeframe=timeframe,
                date_end=date_end.isoformat() if date_end else None
            )
            
            return {
                'timestamp': history.timestamp,
                'equity': history.equity,
                'profit_loss': history.profit_loss,
                'profit_loss_pct': history.profit_loss_pct,
                'base_value': history.base_value,
                'timeframe': history.timeframe
            }
        except APIError as e:
            print(f"Error getting portfolio history: {e}")
            raise
    
    def get_market_data(self, symbol: str, timeframe: str = '1D', limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: The ticker symbol.
            timeframe: Bar timeframe ('1D', '1H', '15Min', etc.).
            limit: Maximum number of bars to return.
            
        Returns:
            List of bars with price data.
        """
        try:
            bars = self.api.get_bars(symbol, timeframe, limit=limit).df
            return bars.reset_index().to_dict('records')
        except APIError as e:
            print(f"Error getting market data for {symbol}: {e}")
            return []
    
    def get_latest_trade(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest trade for a symbol.
        
        Args:
            symbol: The ticker symbol.
            
        Returns:
            Latest trade information.
        """
        try:
            trade = self.api.get_latest_trade(symbol)
            return {
                'price': float(trade.price),
                'size': float(trade.size),
                'timestamp': trade.timestamp
            }
        except APIError as e:
            print(f"Error getting latest trade for {symbol}: {e}")
            raise

    def execute_trading_decisions(self, decisions: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Execute trading decisions from the AI Hedge Fund.
        
        Args:
            decisions: Dict mapping ticker symbols to trading decisions.
            
        Returns:
            Dict of execution results for each ticker.
        """
        execution_results = {}
        
        for ticker, decision in decisions.items():
            action = decision.get("action", "hold").lower()
            quantity = int(decision.get("quantity", 0))
            
            if quantity <= 0 or action == "hold":
                execution_results[ticker] = {
                    "status": "skipped",
                    "message": f"No action taken for {ticker} due to zero quantity or hold action",
                    "order": None
                }
                continue
            
            try:
                # Map AI Hedge Fund actions to Alpaca API actions
                side = None
                if action == "buy":
                    side = "buy"
                elif action == "sell":
                    side = "sell"
                elif action == "short":
                    side = "sell"  # Short is a sell in Alpaca
                elif action == "cover":
                    side = "buy"   # Cover shorts is a buy in Alpaca
                
                if side:
                    order = self.submit_order(
                        symbol=ticker,
                        qty=quantity,
                        side=side,
                        type="market",
                        time_in_force="day"
                    )
                    
                    execution_results[ticker] = {
                        "status": "executed",
                        "message": f"{side.capitalize()} order for {quantity} shares of {ticker} submitted successfully",
                        "order": order
                    }
                else:
                    execution_results[ticker] = {
                        "status": "error",
                        "message": f"Unknown action '{action}' for {ticker}",
                        "order": None
                    }
            except Exception as e:
                execution_results[ticker] = {
                    "status": "error",
                    "message": f"Error executing {action} for {ticker}: {str(e)}",
                    "order": None
                }
        
        return execution_results

    def get_realized_pnl(self, timeframe: str = '1D') -> float:
        """
        Get realized profit and loss for a specific timeframe.
        NOTE: This currently returns the total portfolio P/L change for the period,
              not strictly realized P&L, due to API limitations/complexity.

        Args:
            timeframe: Time period ('1D', '1W', '1M', '3M', '1A', etc.).

        Returns:
            Portfolio P&L change value for the period.
        """
        try:
            # Map common timeframe strings to Alpaca period values if needed
            # Alpaca uses: 1D, 1W, 1M, 3M, 1A, All
            period_map = {
                '1D': '1D',
                '1W': '1W',
                '1M': '1M',
                '3M': '3M',
                '1A': '1A',
            }
            alpaca_period = period_map.get(timeframe.upper(), '1D') # Default to 1D

            # Determine date_end (usually None, meaning 'now')
            # Determine timeframe (resolution, e.g., '1Min', '1H', '1D')
            # For simple daily/weekly P&L, timeframe='1D' is likely sufficient
            alpaca_timeframe = '1D'

            history = self.api.get_portfolio_history(
                period=alpaca_period,
                timeframe=alpaca_timeframe, 
                # date_end=None, # Let Alpaca default to now
            )
            return float(history.profit_loss[-1]) if history.profit_loss else 0.0
        except APIError as e:
            # Log the specific error for debugging
            logging.error(f"API Error getting portfolio history for P&L ({timeframe}): {e}")
            return 0.0
        except Exception as e:
            # Catch other potential errors (e.g., processing the response)
            logging.error(f"Error processing portfolio history for P&L ({timeframe}): {e}")
            return 0.0

    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Get current portfolio status, formatted for application use.

        Returns:
            Dictionary with formatted portfolio status.
        """
        account = self.get_account() # Use existing method
        positions = self.get_positions() # Use existing method

        # Check if account fetch failed
        if account is None:
             # Handle appropriately, maybe raise an error or return a default dict
             # For now, let's return a dict indicating failure
             return {
                 'error': 'Failed to fetch account details',
                 'cash': 0, 'equity': 0, 'buying_power': 0, 'positions': {},
                 'margin_requirement': 0, 'margin_used': 0
             }

        # Safely access account details with defaults
        cash = account.get('cash', 0.0)
        equity = account.get('equity', 0.0)
        initial_margin = account.get('initial_margin', 0.0)


        return {
            'cash': cash,
            'equity': equity,
            'buying_power': account.get('buying_power', 0.0),
            'positions': {
                p['symbol']: {
                    'qty': p.get('qty', 0),
                    'avg_entry_price': p.get('avg_entry_price', 0.0),
                    'market_value': p.get('market_value', 0.0),
                    'current_price': p.get('current_price', 0.0),
                    'unrealized_pl': p.get('unrealized_pl', 0.0),
                    'unrealized_plpc': p.get('unrealized_plpc', 0.0),
                    'side': p.get('side', 'unknown')
                }
                for p in positions # Assumes get_positions returns a list of dicts
            },
            # Calculate margin requirement percentage safely
            'margin_requirement': initial_margin / equity if equity > 0 else 0,
            'margin_used': initial_margin
        }

# Create a singleton instance
_alpaca_client = None

def get_alpaca_client(paper=True) -> AlpacaClient:
    """
    Factory function to get an Alpaca client instance.
    
    Args:
        paper: Whether to use paper trading (True) or live trading (False).
        
    Returns:
        An instance of AlpacaClient.
    """
    _alpaca_client = AlpacaClient(paper=paper)
    return _alpaca_client

class AlpacaTrader(AlpacaClient):
    """Extended Alpaca client for stock trading operations."""
    
    def __init__(self, paper=True):
        """
        Initialize the AlpacaTrader.
        
        Args:
            paper: Whether to use paper trading (True) or live trading (False).
        """
        super().__init__(paper=paper)
    
    def buy(self, symbol: str, qty: int, order_type: str = 'market', 
            time_in_force: str = 'day', limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Buy a stock.
        
        Args:
            symbol: Ticker symbol to buy
            qty: Quantity to buy
            order_type: Order type ('market', 'limit', etc.)
            time_in_force: Time in force specification
            limit_price: Price limit for limit orders
            
        Returns:
            Order details dictionary
        """
        return self.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price
        )
    
    def sell(self, symbol: str, qty: int, order_type: str = 'market', 
             time_in_force: str = 'day', limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Sell a stock.
        
        Args:
            symbol: Ticker symbol to sell
            qty: Quantity to sell
            order_type: Order type ('market', 'limit', etc.)
            time_in_force: Time in force specification
            limit_price: Price limit for limit orders
            
        Returns:
            Order details dictionary
        """
        return self.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price
        )
    
    def short(self, symbol: str, qty: int, order_type: str = 'market', 
              time_in_force: str = 'day', limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Short a stock.
        
        Args:
            symbol: Ticker symbol to short
            qty: Quantity to short
            order_type: Order type ('market', 'limit', etc.)
            time_in_force: Time in force specification
            limit_price: Price limit for limit orders
            
        Returns:
            Order details dictionary
        """
        return self.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price
        )
    
    def cover(self, symbol: str, qty: int, order_type: str = 'market', 
              time_in_force: str = 'day', limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Cover a short position.
        
        Args:
            symbol: Ticker symbol to cover
            qty: Quantity to cover
            order_type: Order type ('market', 'limit', etc.)
            time_in_force: Time in force specification
            limit_price: Price limit for limit orders
            
        Returns:
            Order details dictionary
        """
        return self.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price
        )


class AlpacaOptionsTrader(AlpacaTrader):
    """Extended Alpaca client for options trading operations."""
    
    def __init__(self, paper=True):
        """
        Initialize the AlpacaOptionsTrader.
        
        Args:
            paper: Whether to use paper trading (True) or live trading (False).
        """
        super().__init__(paper=paper)
        # Set up options-specific configuration
        logging.info("Options trading client initialized.")
        
    def _convert_polygon_ticker_to_occ(self, polygon_ticker: str) -> str:
        """
        Convert a Polygon.io options ticker (e.g., O:AAPL230616C00150000)
        to the OCC format (e.g., AAPL230616C00150000).

        Args:
            polygon_ticker: The Polygon.io formatted options ticker.

        Returns:
            The OCC formatted options ticker.

        Raises:
            ValueError: If the input ticker format is invalid.
        """
        if not isinstance(polygon_ticker, str):
            raise ValueError(f"Invalid ticker type: Expected string, got {type(polygon_ticker)}")

        if polygon_ticker.startswith("O:"):
            occ_ticker = polygon_ticker[2:]
        else:
            # Assume it might already be OCC format, but validate basic structure
            occ_ticker = polygon_ticker
        
        # Basic validation for OCC format: Root + YYMMDD + C/P + Strike
        # Example: AAPL230616C00150000 (Variable length root)
        match = re.match(r"^[A-Z]{1,5}(\d{6})([CP])(\d{8})$", occ_ticker)
        if not match:
             # Allow slightly longer root symbols (up to 6 chars are technically possible but rare)
             match = re.match(r"^[A-Z]{1,6}(\d{6})([CP])(\d{8})$", occ_ticker)
             if not match:
                raise ValueError(f"Invalid options ticker format: '{polygon_ticker}' does not match expected Polygon (O:...) or OCC format.")

        # Further validation could be added (e.g., check date validity)
        # For now, basic structural check is done.
        return occ_ticker

    def get_option_position(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a specific options contract.
        
        Args:
            option_symbol: The option symbol (e.g., AAPL230616C00150000)
            
        Returns:
            Position information or None if not found.
        """
        try:
            position = self.api.get_position(option_symbol)
            return {
                'symbol': position.symbol,
                'qty': int(position.qty),
                'avg_entry_price': float(position.avg_entry_price),
                'market_value': float(position.market_value),
                'current_price': float(position.current_price),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'side': 'long' if int(position.qty) > 0 else 'short'
            }
        except Exception as e:
            if "position does not exist" in str(e).lower():
                return None
            logging.error(f"Error getting option position for {option_symbol}: {e}")
            return None
    
    def buy_option(
        self, 
        option_symbol: str, 
        quantity: int, 
        order_type: str = 'market', 
        time_in_force: str = 'day',
        limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Buy an options contract.
        
        Args:
            option_symbol: The full OCC option symbol
            quantity: Number of contracts to buy
            order_type: Order type ('market', 'limit', etc.)
            time_in_force: Time in force specification
            limit_price: Price limit for limit orders
            
        Returns:
            Order details dictionary
        """
        return self.submit_order(
            symbol=option_symbol,
            qty=quantity,
            side='buy',
            type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price
        )
    
    def sell_option(
        self, 
        option_symbol: str, 
        quantity: int, 
        order_type: str = 'market', 
        time_in_force: str = 'day',
        limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Sell an options contract (either to open a short position or close a long position).
        
        Args:
            option_symbol: The full OCC option symbol
            quantity: Number of contracts to sell
            order_type: Order type ('market', 'limit', etc.)
            time_in_force: Time in force specification
            limit_price: Price limit for limit orders
            
        Returns:
            Order details dictionary
        """
        return self.submit_order(
            symbol=option_symbol,
            qty=quantity,
            side='sell',
            type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price
        )
    
    def close_option_position(
        self, 
        option_symbol: str, 
        order_type: str = 'market', 
        time_in_force: str = 'day',
        limit_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Close an existing options position entirely.
        
        Args:
            option_symbol: The full OCC option symbol
            order_type: Order type ('market', 'limit', etc.)
            time_in_force: Time in force specification
            limit_price: Price limit for limit orders
            
        Returns:
            Order details dictionary
        """
        # Get current position
        position = self.get_option_position(option_symbol)
        if not position:
            raise ValueError(f"No position exists for {option_symbol}")
        
        # Determine side for closing
        qty = abs(int(position['qty']))
        side = 'sell' if position['side'] == 'long' else 'buy'
        
        return self.submit_order(
            symbol=option_symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price
        )
    
    def execute_option_decision(
        self, 
        option_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an options trading decision.
        
        Args:
            option_decision: The options contract decision
            
        Returns:
            Dict with execution result
        """
        is_multi_leg = 'legs' in option_decision and isinstance(option_decision['legs'], list)
        
        if is_multi_leg:
            # This is a multi-leg decision (e.g., a spread)
            underlying_ticker = option_decision.get('underlying_ticker', '')
            if not underlying_ticker:
                return {
                    'status': 'error',
                    'message': 'Multi-leg option decision missing underlying_ticker',
                    'order': None
                }
                
            leg_orders = []
            leg_results = []
            overall_status = 'executed' # Assume success unless a leg fails
            combined_message = f"Multi-leg {option_decision.get('strategy','spread')} for {underlying_ticker}: "
            
            submitted_leg_orders = [] # Keep track of successfully submitted orders
            error_occurred = False

            # Use the overall quantity decided by the sizer for each leg
            spread_quantity = int(option_decision.get('quantity', 0))

            if spread_quantity <= 0:
                 return {
                    'status': 'skipped',
                    'message': f"Multi-leg spread skipped due to zero quantity (Original decision: {underlying_ticker})",
                    'order': None
                }

            for leg in option_decision['legs']:
                leg_ticker = leg.get('ticker', '')
                leg_action = leg.get('action', '').lower()
                leg_limit_price = leg.get('limit_price')
                
                if not leg_ticker or leg_action not in ['buy', 'sell']:
                    logger.error(f"Invalid leg in multi-leg decision: {leg}")
                    overall_status = 'error'
                    leg_results.append({'status': 'error', 'message': 'Invalid leg data'})
                    error_occurred = True
                    continue # Skip this leg
                    
                try:
                    occ_ticker = self._convert_polygon_ticker_to_occ(leg_ticker)
                    order_func = self.buy_option if leg_action == 'buy' else self.sell_option
                    leg_order = order_func(
                        option_symbol=occ_ticker,
                        quantity=spread_quantity, # Use same quantity for each leg
                        order_type='limit' if leg_limit_price else 'market',
                        limit_price=leg_limit_price
                    )
                    leg_orders.append(leg_order) # Store individual order details
                    leg_results.append({'status': 'executed', 'message': f'{leg_action.capitalize()} {spread_quantity} {occ_ticker}', 'order': leg_order})
                    # Add successfully submitted order for potential cancellation
                    submitted_leg_orders.append(leg_order)
                except Exception as e:
                    logger.error(f"Error executing leg {leg_ticker} in multi-leg spread: {e}")
                    overall_status = 'error'
                    leg_results.append({'status': 'error', 'message': str(e)})
                    error_occurred = True
                    break # Stop processing legs if one fails
           
            # If any leg failed, cancel any submitted orders to maintain the spread integrity
            if error_occurred and submitted_leg_orders:
                combined_message += "Error occurred, cancelling any submitted legs. "
                for order in submitted_leg_orders:
                    try:
                        self.cancel_order(order['id'])
                        combined_message += f"Cancelled {order['id']}. "
                    except Exception as cancel_err:
                        combined_message += f"Failed to cancel {order['id']}: {cancel_err}. "
            
            # Compile result message
            for i, result in enumerate(leg_results):
                combined_message += f"Leg {i+1}: {result['message']}. "
                
            # Return combined result
            return {
                'status': overall_status,
                'message': combined_message,
                'order': leg_orders if leg_orders else None # Return all leg orders
            }
        else:
            # Single leg option
            polygon_ticker = option_decision.get('ticker', '')
            action = option_decision.get('action', '').lower()
            quantity = int(option_decision.get('quantity', 0))
            limit_price = option_decision.get('limit_price')
            
            if not polygon_ticker:
                return {
                    'status': 'error',
                    'message': 'Missing option ticker in single-leg decision',
                    'order': None
                }
                
            if action not in ['buy', 'sell', 'close']:
                return {
                    'status': 'error',
                    'message': f'Invalid action "{action}" for option order',
                    'order': None
                }
                
            if quantity <= 0:
                return {
                    'status': 'skipped',
                    'message': f'Option order for {polygon_ticker} skipped due to zero quantity',
                    'order': None
                }
                
            # Convert Polygon ticker to OCC format
            try:
                occ_ticker = self._convert_polygon_ticker_to_occ(polygon_ticker)
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Error converting option ticker {polygon_ticker} to OCC format: {e}',
                    'order': None
                }
                
            # Execute the order
            try:
                if action == 'buy':
                    order = self.buy_option(
                        option_symbol=occ_ticker,
                        quantity=quantity,
                        order_type='limit' if limit_price else 'market',
                        limit_price=limit_price
                    )
                    return {
                        'status': 'executed',
                        'message': f'Buy order for {quantity} {occ_ticker} submitted successfully',
                        'order': order
                    }
                elif action == 'sell' or action == 'close':
                    order = self.sell_option(
                        option_symbol=occ_ticker,
                        quantity=quantity,
                        order_type='limit' if limit_price else 'market',
                        limit_price=limit_price
                    )
                    return {
                        'status': 'executed',
                        'message': f'Sell order for {quantity} {occ_ticker} submitted successfully',
                        'order': order
                    }
            except Exception as e:
                logger.error(f"Error executing option order for {occ_ticker}: {e}")
                return {
                    'status': 'error',
                    'message': f'Error executing option order: {e}',
                    'order': None
                }
    
    def execute_options_decisions(
        self, 
        options_decisions: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Execute multiple options trading decisions.
        
        Args:
            options_decisions: Dict mapping unique identifiers (like underlying ticker or spread ID)
                               to single or multi-leg options decisions.
            
        Returns:
            Dict of execution results for each identifier.
        """
        execution_results = {}
        
        for identifier, decision in options_decisions.items():
            # Determine if this is a multi-leg strategy
            is_multi_leg = 'legs' in decision and isinstance(decision['legs'], list)
            
            if is_multi_leg:
                # For multi-leg strategy, we don't need to check 'action' field
                # Just make sure it has legs and underlying_ticker
                if not decision['legs']:
                    # Empty legs list
                    execution_results[identifier] = {
                        'status': 'skipped',
                        'message': f'Multi-leg decision has empty legs list for identifier {identifier}',
                        'order': None
                    }
                    continue
                elif not decision.get('underlying_ticker'):
                    # Missing underlying ticker
                    execution_results[identifier] = {
                        'status': 'skipped',
                        'message': f'Multi-leg decision missing underlying_ticker for identifier {identifier}',
                        'order': None
                    }
                    continue
                    
                # If we got here, we have a valid multi-leg strategy with legs and underlying ticker
                strategy = decision.get('strategy', 'spread').lower()
                if strategy == 'none':
                    execution_results[identifier] = {
                        'status': 'skipped',
                        'message': decision.get('reasoning', 'Strategy is none'),
                        'order': None
                    }
                    continue
                    
                # Execute the multi-leg strategy
                result = self.execute_option_decision(decision)
                execution_results[identifier] = result
                continue
            
            # For single-leg options
            action = decision.get('action', 'none').lower()
            if action == 'none' or decision.get('strategy', '').lower() == 'none':
                execution_results[identifier] = {
                    'status': 'skipped',
                    'message': decision.get('reasoning', 'No action required'),
                    'order': None
                }
                continue
                
            # Check for essential keys for single-leg
            contract_ticker = decision.get('ticker', '')
            if not contract_ticker:
                execution_results[identifier] = {
                    'status': 'skipped',
                    'message': f'Missing contract ticker for identifier {identifier}',
                    'order': None
                }
                continue
                
            # Execute the single-leg decision
            result = self.execute_option_decision(decision)
            execution_results[identifier] = result
            
        return execution_results

    def get_option_chain(self, underlying_symbol: str) -> List[Dict[str, Any]]:
        """
        Get options chain data for a given underlying symbol.
        
        Args:
            underlying_symbol: The underlying stock symbol (e.g., AAPL)
            
        Returns:
            List of available option contracts
        """
        try:
            # Alpaca's API for options data is still evolving
            # This is a simplified implementation
            contracts = self.api.get_option_chain(underlying_symbol)
            return [
                {
                    'symbol': c.symbol,
                    'underlying_symbol': underlying_symbol,
                    'strike_price': float(c.strike_price),
                    'expiration_date': c.expiration_date,
                    'option_type': c.option_type,
                    'bid': float(c.bid),
                    'ask': float(c.ask),
                    'last_price': float(c.last_price) if hasattr(c, 'last_price') else None,
                    'volume': int(c.volume) if hasattr(c, 'volume') else 0,
                    'open_interest': int(c.open_interest) if hasattr(c, 'open_interest') else 0
                }
                for c in contracts
            ]
        except Exception as e:
            logging.error(f"Error getting option chain for {underlying_symbol}: {e}")
            return []