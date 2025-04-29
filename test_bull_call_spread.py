#!/usr/bin/env python
"""
Test script for executing a bull call spread to verify our implementation works.
This script will:
1. Create a simple bull call spread for a stock
2. Execute it using our modified AlpacaOptionsTrader implementation
3. Print out the results
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the project root to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.integrations.alpaca import AlpacaOptionsTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

def create_bull_call_spread(symbol, quantity=1):
    """
    Create a bull call spread options decision using hard-coded values.
    
    Args:
        symbol: The underlying stock symbol
        quantity: Number of spreads to trade
        
    Returns:
        Dictionary representing a bull call spread decision
    """
    # Use known working option symbols for the underlying
    # For AAPL, we'll use May 16, 2025 expiration
    # AAPL is currently around $211, so let's use strikes around that
    
    # Use a real option symbol - from an existing position
    long_call = "AAPL250516C00210000"  # $210 strike call
    short_call = "AAPL250516C00220000" # $220 strike call
    
    # Set reasonable limit prices
    long_price = 12.50  # For $210 strike
    short_price = 7.25   # For $220 strike
    
    net_debit = round(long_price - short_price, 2)
    
    # Create the spread decision
    return {
        'underlying_ticker': symbol,
        'strategy': 'bull_call_spread',
        'confidence': 75.0,
        'reasoning': 'Test bull call spread execution',
        'legs': [
            {
                'ticker': f"O:{long_call}",  # Adding O: prefix
                'action': 'buy',
                'option_type': 'call',
                'strike_price': 210.0,  # Adding strike price for clarity
                'limit_price': long_price
            },
            {
                'ticker': f"O:{short_call}",  # Adding O: prefix
                'action': 'sell',
                'option_type': 'call',
                'strike_price': 220.0,  # Adding strike price for clarity
                'limit_price': short_price
            }
        ],
        'net_limit_price': net_debit,
        'action': 'open_spread',
        'quantity': quantity
    }

def test_bull_call_spread():
    """Execute a test bull call spread trade."""
    load_dotenv()  # Load environment variables
    
    # Initialize the trader with paper trading
    trader = AlpacaOptionsTrader(paper=True)
    logger.info("Initialized AlpacaOptionsTrader for paper trading")
    
    # Choose a liquid stock with options to test
    symbol = 'AAPL'
    
    try:
        # Create the bull call spread with 1 contract using hard-coded values
        spread_decision = create_bull_call_spread(symbol=symbol, quantity=1)
        
        logger.info(f"Testing bull call spread for {symbol}:")
        logger.info(f"Long call: {spread_decision['legs'][0]['ticker']} @ {spread_decision['legs'][0]['limit_price']}")
        logger.info(f"Short call: {spread_decision['legs'][1]['ticker']} @ {spread_decision['legs'][1]['limit_price']}")
        logger.info(f"Net debit: {spread_decision['net_limit_price']}")
        
        # Execute the spread
        logger.info("Executing bull call spread...")
        result = trader.execute_option_decision(spread_decision)
        
        # Print the result
        logger.info(f"Execution result status: {result['status']}")
        logger.info(f"Result message: {result['message']}")
        
        if result['status'] == 'executed':
            logger.info("Bull call spread executed successfully!")
            logger.info(f"Order ID: {result['order']['id']}")
            
            # Print details of the individual legs
            for i, leg in enumerate(result['order']['legs']):
                logger.info(f"Leg {i+1}: {leg['symbol']} - {leg['side']} - {leg['status']}")
        else:
            logger.error("Failed to execute bull call spread")
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing bull call spread: {e}")
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    logger.info("Starting bull call spread test")
    result = test_bull_call_spread()
    logger.info("Test completed")
    
    # Exit with appropriate code
    sys.exit(0 if result.get('status') == 'executed' else 1) 