#!/usr/bin/env python3
"""
Trading Manager for AI Hedge Fund with Alpaca integration.

This script integrates the AI Hedge Fund analysis with Alpaca trading.
It handles the full trading lifecycle including:
- Retrieving current portfolio state from Alpaca
- Running AI analysis on selected tickers
- Executing trading decisions through Alpaca
- Managing positions (scaling, stops, targets)
- Tracking and reporting performance
"""

import argparse
import json
import os
import sys
import time
import logging
import traceback
from datetime import datetime, timedelta
import pytz
from colorama import Fore, Style, init
import questionary
from dotenv import load_dotenv
import schedule
import pandas as pd # Import pandas for Timestamp type check
from pathlib import Path
# Import RichHandler for integrated logging
from rich.logging import RichHandler

# Add the current directory to the path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for .env file and necessary environment variables
def check_env_setup():
    """
    Check if the .env file exists and has the necessary environment variables.
    Provides guidance if the file or variables are missing.
    """
    env_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    
    # Check if .env file exists
    if not os.path.exists(env_file_path):
        print(f"\n{Fore.RED}ERROR: No .env file found at {env_file_path}{Style.RESET_ALL}")
        print(f"Please create a .env file with your Alpaca API credentials:")
        print(f"{Fore.YELLOW}ALPACA_API_KEY=your_api_key_here")
        print(f"ALPACA_API_SECRET=your_api_secret_here{Style.RESET_ALL}")
        print(f"You can obtain these keys from your Alpaca account dashboard at https://app.alpaca.markets/paper/dashboard/overview\n")
        return False
    
    # Load .env file explicitly
    load_dotenv(env_file_path)
    
    # Check for necessary environment variables
    missing_vars = []
    if not os.getenv('ALPACA_API_KEY'):
        missing_vars.append('ALPACA_API_KEY')
    if not os.getenv('ALPACA_API_SECRET'):
        missing_vars.append('ALPACA_API_SECRET')
    
    if missing_vars:
        print(f"\n{Fore.RED}ERROR: Missing environment variables in .env file: {', '.join(missing_vars)}{Style.RESET_ALL}")
        print(f"Please update your .env file with the missing variables:")
        for var in missing_vars:
            print(f"{Fore.YELLOW}{var}=your_{var.lower()}_here{Style.RESET_ALL}")
        print(f"You can obtain these keys from your Alpaca account dashboard at https://app.alpaca.markets/paper/dashboard/overview\n")
        return False
    
    return True

# Now import the modules
from src.integrations.alpaca import get_alpaca_client
from src.portfolio.manager import PortfolioManager
from src.main import run_hedge_fund
from src.utils.display import print_trading_output, print_execution_summary, print_portfolio_status, print_options_analysis_summary
from src.utils.analysts import ANALYST_ORDER
from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from src.utils.ollama import ensure_ollama_and_model
# Import the global progress instance AND the global console object
from src.utils.progress import progress, console
# Import cache control functions
from src.data.cache import close_cache, get_cache, clear_cache

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

# Configure logging
# Use RichHandler for console output to integrate with the progress display
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # RichHandler handles formatting, message only needed
    datefmt="[%X]",         # Optional: RichHandler date format
    handlers=[
        logging.FileHandler("trading_manager.log"), # Keep file logging
        # Use the imported global console object here
        RichHandler(console=console, show_path=False, rich_tracebacks=True, markup=True) 
    ]
)

logger = logging.getLogger("trading_manager")

# Helper function for JSON serialization of datetime/timestamp objects
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError (f"Type {type(obj)} not serializable")

class TradingManager:
    """
    Manages the integration between AI Hedge Fund and Alpaca trading.
    """
    
    def __init__(
        self,
        tickers: list[str],
        model_name: str,
        model_provider: str,
        selected_analysts: list[str],
        trading_frequency: str = "daily",
        market_hours_only: bool = True,
        paper: bool = True,
        position_management_interval: int = 15,  # minutes
        max_position_size_pct: float = 0.20,
        stop_loss_pct: float = 0.05,
        profit_target_pct: float = 0.15,
        cash_reserve_pct: float = 0.10,
        enable_shorts: bool = True,
        enable_options: bool = True,  # Add options toggle
        show_reasoning: bool = False,
    ):
        """
        Initialize the TradingManager.
        
        Args:
            tickers: List of ticker symbols to trade.
            model_name: Name of the LLM model to use.
            model_provider: Provider of the LLM model.
            selected_analysts: List of selected analyst IDs.
            trading_frequency: How often to run the trading logic ("daily", "hourly").
            market_hours_only: Whether to trade only during market hours.
            paper: Whether to use paper trading (True) or live trading (False).
            position_management_interval: How often to check positions in minutes.
            max_position_size_pct: Maximum position size as a percentage of portfolio.
            stop_loss_pct: Default stop loss percentage.
            profit_target_pct: Default profit target percentage.
            cash_reserve_pct: Minimum cash reserve as a percentage of portfolio.
            enable_shorts: Whether to allow short positions.
            enable_options: Whether to enable options trading.
            show_reasoning: Whether to show reasoning from the AI models.
        """
        self.tickers = tickers
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts
        self.trading_frequency = trading_frequency
        self.market_hours_only = market_hours_only
        self.paper = paper
        self.position_management_interval = position_management_interval
        self.show_reasoning = show_reasoning
        
        # Initialize Alpaca client
        try:
            self.alpaca = get_alpaca_client(paper=paper)
            # Verify API credentials work by making a simple API call
            self._verify_alpaca_credentials()
        except Exception as e:
            error_msg = str(e).lower()
            if "forbidden" in error_msg or "unauthorized" in error_msg:
                print(f"\n{Fore.RED}ERROR: Authentication failed with Alpaca API. Please verify your ALPACA_API_KEY and ALPACA_API_SECRET in your .env file.{Style.RESET_ALL}")
                print(f"You can get these keys from your Alpaca account dashboard.")
                print(f"Add them to your .env file with the format:")
                print(f"{Fore.YELLOW}ALPACA_API_KEY=your_api_key_here")
                print(f"ALPACA_API_SECRET=your_api_secret_here{Style.RESET_ALL}\n")
                raise ValueError("Alpaca API authentication failed. Check your API keys.")
            else:
                print(f"\n{Fore.RED}ERROR initializing Alpaca client: {str(e)}{Style.RESET_ALL}")
                raise
        
        # Initialize Portfolio Manager with configuration
        portfolio_config = {
            'max_position_size_pct': max_position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'profit_target_pct': profit_target_pct,
            'cash_reserve_pct': cash_reserve_pct,
            'market_hours_only': market_hours_only,
            'enable_shorts': enable_shorts,
            'enable_options': enable_options,
            'max_options_position_size_pct': 0.05,  # 5% max for any single option
            'max_options_allocation_pct': 0.15,     # 15% max for all options combined
        }
        self.portfolio_manager = PortfolioManager(config=portfolio_config)
        
        # Set up state tracking
        self.last_trading_run = None
        self.running = False
        self.next_scheduled_run = None
        
        # Store access to utils module for display functions
        from src.utils import display as display_module
        self.utils = type('Utils', (), {'display': display_module})()
        
        logger.info("Trading Manager initialized with tickers: %s", tickers)
        
    def _verify_alpaca_credentials(self):
        """
        Verify that Alpaca API credentials are valid by making a test API call.
        Raises an exception if authentication fails.
        """
        try:
            # Try to get the clock as a simple API call to verify credentials
            self.alpaca.get_clock()
            logger.info("Alpaca API credentials verified successfully.")
        except Exception as e:
            logger.error("Failed to verify Alpaca API credentials: %s", str(e))
            raise
        
    def start(self):
        """
        Start the trading manager with appropriate scheduling.
        """
        # Set up scheduling based on trading frequency
        self.running = True
        
        if self.trading_frequency == "daily":
            # Schedule daily trading at market open
            schedule.every().monday.at("09:35").do(self.run_trading_cycle)
            schedule.every().tuesday.at("09:35").do(self.run_trading_cycle)
            schedule.every().wednesday.at("09:35").do(self.run_trading_cycle)
            schedule.every().thursday.at("09:35").do(self.run_trading_cycle)
            schedule.every().friday.at("09:35").do(self.run_trading_cycle)
            next_run = schedule.next_run()
            self.next_scheduled_run = next_run
            
            # Also schedule position management more frequently
            for minute in range(0, 60, self.position_management_interval):
                schedule.every().hour.at(f":{minute:02d}").do(self.run_position_management)
            
        elif self.trading_frequency == "hourly":
            # Schedule hourly trading during market hours
            for hour in range(9, 16):  # 9 AM to 3 PM
                schedule.every().monday.at(f"{hour:02d}:31").do(self.run_trading_cycle)
                schedule.every().tuesday.at(f"{hour:02d}:31").do(self.run_trading_cycle)
                schedule.every().wednesday.at(f"{hour:02d}:31").do(self.run_trading_cycle)
                schedule.every().thursday.at(f"{hour:02d}:31").do(self.run_trading_cycle)
                schedule.every().friday.at(f"{hour:02d}:31").do(self.run_trading_cycle)
            next_run = schedule.next_run()
            self.next_scheduled_run = next_run
            
            # Also schedule position management more frequently

            for minute in range(0, 60, self.position_management_interval):
                schedule.every().hour.at(f":{minute:02d}").do(self.run_position_management)
        
        # Run trading cycle immediately if requested
        if questionary.confirm(
            "Do you want to run an initial trading cycle now?",
            default=True
        ).ask():
            self.run_trading_cycle()
        
        logger.info("Trading Manager started. Next scheduled run: %s", self.next_scheduled_run)
        
        # Run the scheduler loop
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Trading Manager stopped by user.")
            self.running = False
            
    def stop(self):
        """
        Stop the trading manager.
        """
        self.running = False
        logger.info("Trading Manager stopped.")
        
    def run_trading_cycle(self):
        """
        Run a complete trading cycle:
        1. Get current portfolio state
        2. Run AI analysis for stock decisions
        3. Run Options analysis (if enabled)
        4. Execute combined decisions (stock & options) via PortfolioManager
        5. Log results
        
        Returns:
            Dict with results of the trading cycle.
        """
        logger.info("Starting trading cycle for tickers: %s", self.tickers)
        
        # Check if market is open (if configured to trade only during market hours)
        if self.market_hours_only:
            try:
                clock = self.alpaca.get_clock()
                if not clock['is_open']:
                    logger.info("Market is closed. Skipping trading cycle.")
                    return {
                        "status": "skipped",
                        "reason": "Market is closed",
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                error_msg = str(e).lower()
                if "forbidden" in error_msg:
                    logger.error("Authentication error with Alpaca API. Please check your API keys in .env file.")
                    print(f"{Fore.RED}ERROR: Authentication failed with Alpaca API. Please verify your ALPACA_API_KEY and ALPACA_API_SECRET in your .env file.{Style.RESET_ALL}")
                    self.running = False
                    return {
                        "status": "error",
                        "reason": "Alpaca API authentication failed. Check your API keys.",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    logger.error("Error checking market status: %s", str(e))
        
        try:
            # Update last run time
            self.last_trading_run = datetime.now()
            
            # Get current portfolio state from Alpaca
            try:
                initial_portfolio_state = self.portfolio_manager.get_portfolio_state()
                logger.info("Initial portfolio state retrieved. Cash: $%.2f, Portfolio Value: $%.2f", 
                            initial_portfolio_state['cash'], initial_portfolio_state['portfolio_value'])
            except Exception as e:
                error_msg = str(e).lower()
                if "forbidden" in error_msg or "unauthorized" in error_msg:
                    logger.error("Authentication error with Alpaca API. Please check your API keys in .env file.")
                    print(f"{Fore.RED}ERROR: Authentication failed with Alpaca API. Please verify your ALPACA_API_KEY and ALPACA_API_SECRET in your .env file.{Style.RESET_ALL}")
                    self.running = False
                    return {
                        "status": "error",
                        "reason": "Alpaca API authentication failed. Check your API keys.",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    logger.error("Error retrieving portfolio state: %s", str(e))
                    raise
            
            # Run AI analysis for stock decisions
            start_time = time.time()
            ai_portfolio_context = initial_portfolio_state # Use initial state for analysis context
            # Run AI analysis (graph execution + final synthesis via PortfolioManager method)
            ai_result = run_hedge_fund(
                tickers=self.tickers,
                start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
                portfolio=ai_portfolio_context,
                portfolio_manager=self.portfolio_manager,
                show_reasoning=self.show_reasoning,
                selected_analysts=self.selected_analysts,
                model_name=self.model_name,
                model_provider=self.model_provider,
            )
            analysis_time = time.time() - start_time
            logger.info("Stock AI analysis completed in %.2f seconds", analysis_time)
            stock_decisions = ai_result.get("decisions", {}) # Get quantity-less stock decisions
            print_trading_output(ai_result) # Print stock analysis summary
            
            # Run Options analysis (if enabled)
            options_decisions = {} # Initialize as empty
            if self.portfolio_manager.config.get('enable_options', True):
                logger.info("Running options analysis based on stock decisions...")
                try:
                    from src.agents.options_analysis import options_analysis_agent
                    # Construct state, passing the *result* of stock analysis
                    state_for_options = {
                        "data": {
                            "tickers": self.tickers,
                            "analyst_signals": ai_result.get("analyst_signals", {}),
                            "decisions": stock_decisions, # Pass the generated stock decisions
                        },
                        "metadata": {
                            "model_name": self.model_name,
                            "model_provider": self.model_provider,
                            "show_reasoning": self.show_reasoning
                        },
                        "messages": []
                    }
                    options_result = options_analysis_agent(state_for_options)
                    options_decisions = options_result.get("data", {}).get("options_decisions", {})
                    print_options_analysis_summary(options_decisions) # Print the new table
                    # Store options decisions in ai_result for logging later
                    ai_result["options_decisions"] = options_decisions
                except Exception as e:
                    logger.error(f"Error in options analysis: {e}")
                    traceback.print_exc()
            else:
                logger.info("Options trading is disabled. Skipping options analysis.")

            # Execute combined decisions (stock & options) via PortfolioManager
            stock_execution_results = {}
            option_execution_results = {}
            combined_execution_results = {} # For overall summary

            # We need to iterate through all involved tickers (from stock and options decisions)
            all_tickers = set(stock_decisions.keys())

            # Build a map of underlying tickers to option decisions for multi-leg strategies
            underlying_to_option_map = {}
            for option_key, option_decision in options_decisions.items():
                # Check if this is a multi-leg strategy
                if 'legs' in option_decision and isinstance(option_decision['legs'], list):
                    # Use the underlying ticker as key
                    underlying = option_decision.get('underlying_ticker')
                    if underlying:
                        underlying_to_option_map[underlying] = option_decision
                        # Add the underlying to the all_tickers set
                        all_tickers.add(underlying)
                else:
                    # For single-leg options, use the option ticker
                    all_tickers.add(option_key)

            # For normal options (not multi-leg), just add them directly
            for option_key in options_decisions.keys():
                if option_key not in underlying_to_option_map:
                    all_tickers.add(option_key)

            if not all_tickers:
                 logger.info("No stock or option decisions generated. Nothing to execute.")
            else:
                 logger.info(f"Processing combined execution for tickers: {list(all_tickers)}")
                 for ticker in all_tickers:
                     # Get the specific stock/option decision for this ticker (or None)
                     current_stock_decision = stock_decisions.get(ticker)
                     
                     # First check if this ticker is an underlying for a multi-leg option strategy
                     current_option_decision = underlying_to_option_map.get(ticker)
                     
                     # If not found in multi-leg map, check regular options decisions
                     if not current_option_decision:
                         current_option_decision = options_decisions.get(ticker)

                     # Prepare decisions with the ticker key added *inside* the dictionary
                     stock_decision_for_exec = None
                     if current_stock_decision:
                         stock_decision_for_exec = current_stock_decision.copy()
                         stock_decision_for_exec['ticker'] = ticker # Add ticker key

                     option_decision_for_exec = None
                     if current_option_decision:
                         option_decision_for_exec = current_option_decision.copy()
                         # No need to add ticker for options - they already have it

                     if not stock_decision_for_exec and not option_decision_for_exec:
                          continue # Should not happen if ticker came from the sets
                          
                     logger.info(f"Executing combined decision for ticker: {ticker}")
                     stock_exec, option_exec = self.portfolio_manager.execute_combined_decision(
                         stock_decision=stock_decision_for_exec,
                         option_decision=option_decision_for_exec
                     )
                     
                     # Store results keyed by ticker for summary
                     if stock_exec:
                          stock_execution_results[ticker] = stock_exec
                          combined_execution_results[ticker] = stock_exec # Use stock result if available
                     
                     # --- MODIFIED: Check if option_exec is not None before processing ---
                     if option_exec:
                          # For multi-leg options, use the underlying ticker as the key
                          # Check if 'order' exists and is a list (indicating multi-leg)
                          order_info = option_exec.get('order')
                          if isinstance(order_info, list):
                              # This is a multi-leg option
                              option_execution_results[ticker] = option_exec
                          elif isinstance(order_info, dict):
                              # For single-leg options, use the option ticker
                              option_key = order_info.get('symbol', ticker) # Safely get symbol
                              option_execution_results[option_key] = option_exec
                          # else: Handle cases where order_info is None or unexpected type? Log maybe?

                          # If only option trade, use its result for combined summary
                          if ticker not in combined_execution_results:
                              combined_execution_results[ticker] = option_exec
                     # --- END MODIFICATION ---

                 # Print Execution Summary Table using combined results
                 print_execution_summary(combined_execution_results)
                 # --- ADDED: Print the dedicated Options Execution Summary --- 
                 if option_execution_results:
                     # Use the dedicated function for options execution results
                     from src.utils.display import print_options_execution_summary # Ensure import
                     print_options_execution_summary(option_execution_results)
                     
                 logger.info("Combined trading decisions executed.")
                 logger.info(f"Stock Executions: {json.dumps(stock_execution_results, default=json_serial)}")
                 logger.info(f"Option Executions: {json.dumps(option_execution_results, default=json_serial)}")

            # Wait for orders to potentially fill
            sleep_duration = 30
            logger.info(f"Waiting {sleep_duration} seconds for orders to fill...")
            time.sleep(sleep_duration)
            
            # Force refresh portfolio cache
            logger.info("Forcing portfolio cache refresh...")
            self.portfolio_manager.update_portfolio_cache(force=True)
            
            # Get final portfolio state
            final_portfolio_state = self.portfolio_manager.get_portfolio_state()
            logger.info("Final portfolio state retrieved.")
            print_portfolio_status(final_portfolio_state)
            
            # Store results
            results = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "portfolio_before": initial_portfolio_state, # Use initial state
                "ai_analysis": ai_result, # Includes stock decisions & maybe options decisions
                "stock_execution_results": stock_execution_results,
                "option_execution_results": option_execution_results,
                "portfolio_after": final_portfolio_state
            }
            
            self._save_results(results)
            return results
            
        except Exception as e:
            logger.exception("Error in trading cycle: %s", str(e))
            return {
                "status": "error",
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_position_management(self):
        """
        Run position management cycle:
        1. Check and update stops/targets
        2. Scale in/out as needed
        
        Returns:
            Dict with results of the position management cycle.
        """
        logger.info("Starting position management cycle")
        
        # Check if market is open (if configured to trade only during market hours)
        if self.market_hours_only:
            clock = self.alpaca.get_clock()
            if not clock['is_open']:
                logger.info("Market is closed. Skipping position management.")
                return {
                    "status": "skipped",
                    "reason": "Market is closed",
                    "timestamp": datetime.now().isoformat()
                }
        
        try:
            # Manage stops and targets
            stop_actions = self.portfolio_manager.manage_stops_and_targets()
            
            # Manage positions (scaling in/out)
            position_actions = self.portfolio_manager.manage_positions()
            
            # Manage options positions if enabled
            options_management_actions = {}
            if self.portfolio_manager.config.get('enable_options', True):
                try:
                    options_management_actions = self.portfolio_manager.manage_options_positions()
                    
                    # Log results if any actions were taken
                    if options_management_actions:
                        logger.info("Options position management actions: %s", 
                                    json.dumps(options_management_actions, default=json_serial))
                except Exception as e:
                    logger.error(f"Error in options position management: {e}")
                    traceback.print_exc()
            
            # Log results
            if stop_actions or position_actions or options_management_actions:
                logger.info("Position management actions taken: %s", 
                            json.dumps({
                                "stops": stop_actions, 
                                "positions": position_actions,
                                "options": options_management_actions
                            }, default=json_serial))
            else:
                logger.info("No position management actions needed")
            
            return {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "stop_actions": stop_actions,
                "position_actions": position_actions,
                "options_management_actions": options_management_actions
            }
            
        except Exception as e:
            logger.exception("Error in position management cycle: %s", str(e))
            return {
                "status": "error",
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _save_results(self, results):
        """
        Save trading results to file.
        
        Args:
            results: Dict with trading cycle results.
        """
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/trading_{timestamp}.json"
        
        # Save results
        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=json_serial)
        
        logger.info("Results saved to %s", filename)


def select_model():
    """Helper function to select LLM model - HARDCODED."""
    model_name = "meta-llama/llama-4-maverick-17b-128e-instruct" # Hardcoded model
    model_provider = ModelProvider.GROQ.value # Hardcoded provider
    
    print(f"\nUsing hardcoded {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
    return model_name, model_provider


def select_analysts():
    """Helper function to select analysts - HARDCODED to select all."""
    # Select all analysts defined in ANALYST_ORDER
    all_analyst_ids = [value for _, value in ANALYST_ORDER]
    
    print(f"\nUsing hardcoded analysts (all): {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in all_analyst_ids)}\n")
    return all_analyst_ids


def main():
    """
    Main execution function. Parses arguments, sets up manager, and runs.
    """
    parser = argparse.ArgumentParser(description="AI Hedge Fund Trading Manager")
    
    parser.add_argument(
        "--tickers", 
        type=str, 
        required=True,
        help="Comma-separated list of stock ticker symbols (e.g., AAPL,MSFT,GOOGL)"
    )
    parser.add_argument(
        "--trading-frequency", 
        type=str, 
        default="daily",
        choices=["daily", "hourly"],
        help="Trading frequency (daily or hourly)"
    )
    parser.add_argument(
        "--market-hours-only", 
        action="store_true", 
        help="Trade only during market hours"
    )
    parser.add_argument(
        "--paper", 
        action="store_true", 
        default=True,
        help="Use paper trading (default: True)"
    )
    parser.add_argument(
        "--live", 
        action="store_true", 
        help="Use live trading (overrides --paper)"
    )
    parser.add_argument(
        "--management-interval", 
        type=int, 
        default=15,
        help="Position management interval in minutes (default: 15)"
    )
    parser.add_argument(
        "--max-position-size", 
        type=float, 
        default=0.20,
        help="Maximum position size as percentage of portfolio (default: 0.20)"
    )
    parser.add_argument(
        "--stop-loss", 
        type=float, 
        default=0.05,
        help="Stop loss percentage (default: 0.05)"
    )
    parser.add_argument(
        "--profit-target", 
        type=float, 
        default=0.15,
        help="Profit target percentage (default: 0.15)"
    )
    parser.add_argument(
        "--cash-reserve", 
        type=float, 
        default=0.10,
        help="Cash reserve percentage (default: 0.10)"
    )
    parser.add_argument(
        "--enable-shorts", 
        action="store_true",
        default=True,
        help="Enable short selling (default: True)"
    )
    parser.add_argument(
        "--disable-shorts", 
        action="store_true", 
        help="Disable short selling"
    )
    parser.add_argument(
        "--enable-options",
        action="store_true",
        help="Enable options trading"
    )
    parser.add_argument(
        "--max-options-allocation",
        type=float,
        default=0.15,
        help="Maximum portfolio allocation to options (default: 0.15)"
    )
    parser.add_argument(
        "--show-reasoning", 
        action="store_true", 
        help="Show reasoning from the AI models"
    )
    
    args = parser.parse_args()

    # Initialize colorama
    init(autoreset=True)

    # Check for proper environment setup
    if not check_env_setup():
        print(f"{Fore.RED}Aborting due to missing environment configuration.{Style.RESET_ALL}")
        sys.exit(1)
    
    # Parse tickers from comma-separated string
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    
    # Select LLM model
    model_name, model_provider = select_model()

    # Ensure Ollama model is available if needed
    if model_provider == ModelProvider.OLLAMA.value:
        if not ensure_ollama_and_model(model_name):
            sys.exit(1)
    
    # Select analysts
    selected_analysts = select_analysts()
    
    # Determine if we're using paper or live trading
    paper = not args.live
    
    if not paper:
        # Confirm live trading mode
        confirm = questionary.confirm(
            f"{Fore.RED}WARNING: You are about to use LIVE trading with real money. Are you sure?{Style.RESET_ALL}",
            default=False
        ).ask()
        
        if not confirm:
            print("Switching to paper trading mode.")
            paper = True
    
    # Determine short selling setting (enabled by default, disabled if flag is set)
    enable_shorts = args.enable_shorts and not args.disable_shorts

    # ---- Handle cache clearing ----
    try:
        # logger.info("Attempting to close any existing cache connection...")
        # close_cache() # Removed this line

        logger.info("Attempting to clear cache data...")
        if clear_cache(): # This ensures cache is initialized and clears tables
            logger.info("Cache data cleared successfully.")
        else:
            logger.warning("Failed to clear cache data. The application will continue but may use stale data.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while managing the cache: {e}")
        print(f"\n{Fore.RED}ERROR: An unexpected error occurred while trying to clear the cache: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Continuing, but may use stale data.{Style.RESET_ALL}")
    # ---- End cache management logic ----
    
    # Create and start the trading manager
    manager = TradingManager(
        tickers=tickers,
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        trading_frequency=args.trading_frequency,
        market_hours_only=args.market_hours_only,
        paper=paper,
        position_management_interval=args.management_interval,
        max_position_size_pct=args.max_position_size,
        stop_loss_pct=args.stop_loss,
        profit_target_pct=args.profit_target,
        cash_reserve_pct=args.cash_reserve,
        enable_shorts=enable_shorts,
        enable_options=args.enable_options,
        show_reasoning=args.show_reasoning,
    )
    
    manager.start()


if __name__ == "__main__":
    main()