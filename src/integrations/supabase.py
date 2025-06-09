"""
Supabase integration for storing trading data.

This module provides functions to connect to Supabase and store:
- Portfolio Strategy decisions
- Options Analysis results
- Execution Summary
- Position Management actions

Note: This implementation uses direct HTTP requests instead of the supabase-py
client to avoid dependency conflicts with alpaca-trade-api.
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import pandas as pd  # For Timestamp type checking
import traceback

logger = logging.getLogger(__name__)

# Define a custom JSON encoder that handles Timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle pandas Timestamp objects
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        # Handle other non-serializable types
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)  # Convert any other non-serializable objects to strings

def json_serialize(obj):
    """Serialize object to JSON string with custom encoder."""
    return json.dumps(obj, cls=CustomJSONEncoder)

class DirectSupabaseClient:
    """Direct HTTP client for interacting with Supabase REST API."""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize the Supabase client.
        
        Args:
            url: Supabase project URL
            key: Supabase API key
        """
        # Use parameters or environment variables
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and API key are required. Set SUPABASE_URL and SUPABASE_KEY environment variables or pass them directly.")
        
        # Ensure URL has no trailing slash
        self.url = self.url.rstrip('/')
        
        # REST endpoint
        self.rest_url = f"{self.url}/rest/v1"
        
        # Default headers
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        logger.info("Direct Supabase client initialized")
    
    def execute_query(self, table: str, method: str = "GET", data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
        """
        Execute a query against Supabase REST API.
        
        Args:
            table: Table name
            method: HTTP method (GET, POST, PUT, DELETE)
            data: Data to send in the request body
            params: Query parameters
            
        Returns:
            Dict with response data
        """
        url = f"{self.rest_url}/{table}"
        
        try:
            # Serialize data with custom encoder to handle Timestamp objects if data is provided
            if data:
                # Ensure we're working with a copy to avoid modifying the original
                data_copy = data.copy() if isinstance(data, dict) else data
                
                # Log data being sent for debugging
                logger.debug(f"Sending data to {table}: {json_serialize(data_copy)[:500]}...")

            if method == "GET":
                response = requests.get(url, headers=self.headers, params=params)
            elif method == "POST":
                # Directly use json parameter which handles serialization properly
                response = requests.post(
                    url, 
                    headers=self.headers, 
                    json=data_copy if data else None,
                    params=params
                )
            elif method == "PUT":
                # Directly use json parameter which handles serialization properly
                response = requests.put(
                    url, 
                    headers=self.headers, 
                    json=data_copy if data else None,
                    params=params
                )
            elif method == "DELETE":
                response = requests.delete(url, headers=self.headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # If there's an error, get detailed information
            if response.status_code >= 400:
                error_msg = f"{response.status_code} {response.reason} for url: {url}"
                try:
                    error_details = response.json()
                    error_msg += f"\nDetails: {json.dumps(error_details)}"
                except:
                    error_msg += f"\nResponse text: {response.text[:500]}"
                
                # Log the request data on error
                if method in ["POST", "PUT"] and data:
                    error_msg += f"\nRequest data: {json_serialize(data_copy)[:500]}"
                
                logger.error(f"Error executing query: {error_msg}")
                return {"status": "error", "message": error_msg}
            
            # Raise for any other HTTP errors
            response.raise_for_status()
            
            if response.status_code in (200, 201, 204):
                if response.content:
                    return response.json()
                return {"status": "success"}
            else:
                return {"status": "error", "message": f"Unexpected status code: {response.status_code}"}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error executing query: {str(e)}")
            # Log the traceback for debugging
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to Supabase.
        
        Returns:
            Dict with connection test results
        """
        try:
            # Simple health check - try to get schema info
            response = requests.get(f"{self.rest_url}/", headers=self.headers, timeout=10)
            if response.status_code == 200:
                return {"status": "success", "message": "Connection successful"}
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.ConnectionError as e:
            return {"status": "error", "message": f"Connection failed: {str(e)}"}
        except requests.exceptions.Timeout as e:
            return {"status": "error", "message": f"Connection timeout: {str(e)}"}
        except Exception as e:
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}
    
    def insert(self, table: str, data: Dict) -> Dict:
        """
        Insert data into a table.
        
        Args:
            table: Table name
            data: Data to insert
            
        Returns:
            Dict with response data
        """
        # Sanitize the data before insertion
        sanitized_data = self._sanitize_data_for_insertion(data, table)
        return self.execute_query(table, method="POST", data=sanitized_data)
    
    def _sanitize_data_for_insertion(self, data: Dict, table: str) -> Dict:
        """
        Sanitize data before insertion to ensure it conforms to table schema.
        
        Args:
            data: Data to sanitize
            table: Table name for context-specific sanitization
            
        Returns:
            Dict with sanitized data
        """
        if not data:
            return {}
            
        # Create a copy to avoid modifying the original
        sanitized = {}
        
        # Apply table-specific sanitization
        if table == "execution_summary":
            # Handle specific fields for execution_summary
            # Ensure raw_data is serialized if it's not already a string
            if "raw_data" in data and not isinstance(data["raw_data"], str):
                sanitized["raw_data"] = json_serialize(data["raw_data"])
            else:
                sanitized["raw_data"] = data.get("raw_data", "{}")
                
            # Convert quantity to integer for execution_summary table
            if "quantity" in data:
                try:
                    # Convert float to integer
                    sanitized["quantity"] = int(float(data["quantity"]))
                except (ValueError, TypeError):
                    sanitized["quantity"] = 0
                    logger.warning(f"Could not convert quantity value to integer: {data.get('quantity')}")
                
            # Copy all other fields
            for key, value in data.items():
                if key not in ["raw_data", "quantity"]:
                    # Convert any timestamps in the data
                    if isinstance(value, (pd.Timestamp, datetime)):
                        sanitized[key] = value.isoformat()
                    else:
                        sanitized[key] = value
        
        elif table == "options_execution_summary":
            # Handle specific fields for options_execution_summary
            # Ensure raw_data is serialized if it's not already a string
            if "raw_data" in data and not isinstance(data["raw_data"], str):
                sanitized["raw_data"] = json_serialize(data["raw_data"])
            else:
                sanitized["raw_data"] = data.get("raw_data", "{}")
                
            # Convert quantity to integer for options_execution_summary table
            if "quantity" in data:
                try:
                    # Convert float to integer
                    sanitized["quantity"] = int(float(data["quantity"]))
                except (ValueError, TypeError):
                    sanitized["quantity"] = 0
                    logger.warning(f"Could not convert quantity value to integer: {data.get('quantity')}")
                
            # Copy all other fields
            for key, value in data.items():
                if key not in ["raw_data", "quantity"]:
                    # Convert any timestamps in the data
                    if isinstance(value, (pd.Timestamp, datetime)):
                        sanitized[key] = value.isoformat()
                    else:
                        sanitized[key] = value
        else:
            # Default sanitization for other tables
            for key, value in data.items():
                # Ensure JSONB fields are serialized
                if key in ["raw_data", "details", "cycle_data"] and not isinstance(value, str):
                    sanitized[key] = json_serialize(value)
                # Convert timestamps
                elif isinstance(value, (pd.Timestamp, datetime)):
                    sanitized[key] = value.isoformat()
                else:
                    sanitized[key] = value
        
        return sanitized
    
    def select(self, table: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Select data from a table.
        
        Args:
            table: Table name
            params: Query parameters
            
        Returns:
            List of dicts with response data
        """
        result = self.execute_query(table, method="GET", params=params)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and result.get("status") == "error":
            logger.error(f"Error selecting data: {result.get('message')}")
            return []
        return result
    
    def store_trading_cycle(self, cycle_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a complete trading cycle result in Supabase.
        
        Args:
            cycle_result: The complete cycle result from TradingManager.run_trading_cycle
            
        Returns:
            Dict with status and stored IDs
        """
        timestamp = datetime.now().isoformat()
        cycle_id = None
        stored_ids = {
            "cycle": None,
            "portfolio_strategy": [],
            "options_analysis": [],
            "execution_summary": [],
            "options_execution": []
        }
        
        try:
            # 1. Store cycle metadata
            cycle_data = {
                "timestamp": timestamp,
                "tickers": cycle_result.get("tickers", []),
                "portfolio_value_before": cycle_result.get("initial_portfolio", {}).get("portfolio_value"),
                "portfolio_value_after": cycle_result.get("final_portfolio", {}).get("portfolio_value"),
                "cash_before": cycle_result.get("initial_portfolio", {}).get("cash"),
                "cash_after": cycle_result.get("final_portfolio", {}).get("cash"),
                "cycle_data": cycle_result  # This will be serialized in the insert method
            }
            
            response = self.insert("trading_cycles", cycle_data)
            if isinstance(response, list) and len(response) > 0:
                cycle_id = response[0].get("id")
                stored_ids["cycle"] = cycle_id
                logger.info(f"Stored trading cycle with ID: {cycle_id}")
            else:
                logger.error(f"Failed to store trading cycle: {response}")
                return {"status": "error", "message": "Failed to store trading cycle", "stored_ids": stored_ids}
            
            # 2. Store Portfolio Strategy decisions
            stock_decisions = cycle_result.get("stock_decisions", {})
            for ticker, decision in stock_decisions.items():
                strategy_data = {
                    "cycle_id": cycle_id,
                    "timestamp": timestamp,
                    "ticker": ticker,
                    "action": decision.get("action", ""),
                    "confidence": decision.get("confidence", 0),
                    "reasoning": decision.get("reasoning", ""),
                    "raw_data": decision
                }
                
                response = self.insert("portfolio_strategy", strategy_data)
                if isinstance(response, list) and len(response) > 0:
                    stored_ids["portfolio_strategy"].append(response[0].get("id"))
            
            # 3. Store Options Analysis results
            options_decisions = cycle_result.get("options_decisions", {})
            for key, decision in options_decisions.items():
                # Determine the underlying ticker
                underlying = decision.get("underlying_ticker", key)
                
                option_data = {
                    "cycle_id": cycle_id,
                    "timestamp": timestamp,
                    "underlying_ticker": underlying,
                    "action": decision.get("action", ""),
                    "option_ticker": decision.get("ticker", ""),
                    "details": {
                        "option_type": decision.get("option_type", ""),
                        "strike_price": decision.get("strike_price", 0),
                        "expiration_date": decision.get("expiration_date", "")
                    },
                    "strategy": decision.get("strategy", ""),
                    "confidence": decision.get("confidence", 0),
                    "limit_price": decision.get("limit_price", 0),
                    "reasoning": decision.get("reasoning", ""),
                    "raw_data": decision
                }
                
                response = self.insert("options_analysis", option_data)
                if isinstance(response, list) and len(response) > 0:
                    stored_ids["options_analysis"].append(response[0].get("id"))
            
            # 4. Store Execution Summary
            execution_results = cycle_result.get("execution_results", {})
            for ticker, result in execution_results.items():
                order = result.get("order", {})
                
                # Ensure order fields are properly normalized
                order_side = ""
                order_qty = 0 
                order_status = ""
                order_id = ""
                
                if order:
                    if isinstance(order, dict):
                        order_side = str(order.get("side", ""))
                        # Convert order quantity to integer
                        try:
                            order_qty = int(float(order.get("qty", 0)))
                        except (ValueError, TypeError):
                            order_qty = 0
                        order_status = str(order.get("status", ""))
                        order_id = str(order.get("id", ""))
                
                execution_data = {
                    "cycle_id": cycle_id,
                    "timestamp": timestamp,
                    "ticker": ticker,
                    "status": str(result.get("status", "")),
                    "action": order_side,
                    "quantity": order_qty,  # This is now an integer
                    "order_status": order_status,
                    "order_id": order_id,
                    "message": str(result.get("message", "")),
                    "raw_data": result
                }
                
                response = self.insert("execution_summary", execution_data)
                if isinstance(response, list) and len(response) > 0:
                    stored_ids["execution_summary"].append(response[0].get("id"))
                elif isinstance(response, dict) and response.get("status") == "error":
                    logger.error(f"Error storing execution summary for {ticker}: {response.get('message')}")
            
            # 5. Store Options Execution Summary if available
            options_execution_results = cycle_result.get("options_execution_results", {})
            for option_ticker, result in options_execution_results.items():
                order = result.get("order", {})
                
                # Normalize order fields
                order_side = ""
                order_qty = 0
                order_type = ""
                order_status = ""
                order_id = ""
                
                if order:
                    if isinstance(order, dict):
                        order_side = str(order.get("side", ""))
                        # Convert order quantity to integer
                        try:
                            order_qty = int(float(order.get("qty", 0)))
                        except (ValueError, TypeError):
                            order_qty = 0
                        order_type = str(order.get("type", ""))
                        order_status = str(order.get("status", ""))
                        order_id = str(order.get("id", ""))
                
                option_execution_data = {
                    "cycle_id": cycle_id,
                    "timestamp": timestamp,
                    "option_ticker": option_ticker,
                    "status": str(result.get("status", "")),
                    "action": order_side,
                    "quantity": order_qty,  # This is now an integer
                    "type": order_type,
                    "order_status": order_status,
                    "order_id": order_id,
                    "message": str(result.get("message", "")),
                    "raw_data": result
                }
                
                response = self.insert("options_execution_summary", option_execution_data)
                if isinstance(response, list) and len(response) > 0:
                    stored_ids["options_execution"].append(response[0].get("id"))
                elif isinstance(response, dict) and response.get("status") == "error":
                    logger.error(f"Error storing options execution summary for {option_ticker}: {response.get('message')}")
            
            return {"status": "success", "cycle_id": cycle_id, "stored_ids": stored_ids}
            
        except Exception as e:
            logger.error(f"Error storing trading cycle: {str(e)}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e), "stored_ids": stored_ids}
    
    def store_position_management(self, management_result: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Store position management actions in Supabase.
        
        Args:
            management_result: The result from PositionManagementAgent.execute_management_actions
            
        Returns:
            Dict with status and stored IDs
        """
        timestamp = datetime.now().isoformat()
        stored_ids = []
        
        try:
            for ticker, actions in management_result.items():
                if not isinstance(actions, list):
                    # Handle non-list entries (like error messages)
                    continue
                    
                for action_result in actions:
                    action_type = action_result.get("action", "")
                    status = action_result.get("status", "")
                    message = action_result.get("message", "")
                    details = action_result.get("details", {})
                    
                    # Extract quantity if available
                    quantity = None
                    if details and "quantity" in details:
                        try:
                            # Convert quantity to integer if it's a float
                            quantity = int(float(details["quantity"])) if details["quantity"] is not None else None
                        except (ValueError, TypeError):
                            quantity = None
                    
                    # Extract order ID if available
                    order_id = ""
                    order = action_result.get("order", {})
                    if order and isinstance(order, dict):
                        order_id = order.get("id", "")
                    
                    position_data = {
                        "timestamp": timestamp,
                        "ticker": ticker,
                        "action_type": action_type,
                        "status": status,
                        "quantity": quantity,
                        "order_id": order_id,
                        "message": message,
                        "details": details,
                        "raw_data": action_result
                    }
                    
                    response = self.insert("position_management", position_data)
                    if isinstance(response, list) and len(response) > 0:
                        stored_ids.append(response[0].get("id"))
            
            return {"status": "success", "stored_ids": stored_ids}
            
        except Exception as e:
            logger.error(f"Error storing position management actions: {str(e)}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e), "stored_ids": stored_ids}

# Function to get a singleton instance of DirectSupabaseClient
_supabase_client = None

def get_supabase_client(url: Optional[str] = None, key: Optional[str] = None) -> DirectSupabaseClient:
    """
    Get a singleton instance of DirectSupabaseClient.
    
    Args:
        url: Supabase project URL
        key: Supabase API key
        
    Returns:
        DirectSupabaseClient instance
    """
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = DirectSupabaseClient(url, key)
    return _supabase_client 