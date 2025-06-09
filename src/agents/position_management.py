"""
Position Management AI Agent

This agent analyzes current positions and makes decisions about:
1. Scaling in/out of positions
2. Setting and adjusting stop losses
3. Taking profits
4. Position adjustments based on underlying asset performance
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import numpy as np
import pandas as pd
import re

from src.utils.llm import call_llm
from src.utils.progress import progress
from src.data.cache import get_cache

class PositionAction(BaseModel):
    """Model for position management actions."""
    ticker: str = Field(description="The ticker symbol")
    action: str = Field(description="Action to take: scale_in, scale_out, stop_loss, take_profit, adjust_stop, adjust_target, hold")
    reason: str = Field(description="Reasoning for the action")
    confidence: float = Field(description="Confidence in the action (0-100)")
    quantity: Optional[int] = Field(None, description="Quantity to trade (for scale in/out). Must be an integer, not a decimal.")
    price_target: Optional[float] = Field(None, description="Price target (for adjustments)")

class PositionManagementResult(BaseModel):
    """Model for the position management results."""
    actions: List[PositionAction] = Field(description="List of position actions to take")

class PositionManagementAgent:
    def __init__(self, config: Dict[str, Any], alpaca_client, polygon_client, portfolio_cache: Dict[str, Any]):
        """
        Initialize the PositionManagementAgent.
        
        Args:
            config: Configuration dictionary for management parameters
            alpaca_client: Client for interacting with Alpaca
            polygon_client: Client for interacting with Polygon
            portfolio_cache: Access to cached portfolio data
        """
        self.config = config
        self.alpaca = alpaca_client
        self.polygon = polygon_client
        self.portfolio_cache = portfolio_cache
        self.logger = logging.getLogger(__name__)
        self.cache = get_cache()  # Get cache instance for position tracking
        self.logger.info("PositionManagementAgent initialized.")
        
        # Enhanced configuration with new parameters
        self.scaling_config = {
            'profit_scale_out_levels': config.get('profit_scale_out_levels', [
                {'pnl_pct': 10, 'qty_pct': 25},   # Take 25% at 10% profit
                {'pnl_pct': 20, 'qty_pct': 33},   # Take 33% at 20% profit
                {'pnl_pct': 30, 'qty_pct': 50},   # Take 50% at 30% profit
                {'pnl_pct': 50, 'qty_pct': 100},  # Take all at 50% profit
            ]),
            'loss_scale_in_threshold': config.get('loss_scale_in_threshold', -10),  # Scale in at -10%
            'loss_scale_out_threshold': config.get('loss_scale_out_threshold', -15), # Reduce at -15%
            'trailing_stop_activation': config.get('trailing_stop_activation', 15),  # Activate at 15% profit
            'trailing_stop_distance': config.get('trailing_stop_distance', 5),       # Trail by 5%
            'momentum_scale_threshold': config.get('momentum_scale_threshold', 5),   # Scale on 5% momentum
            'max_scale_count': config.get('max_scale_count', 3),                    # Max 3 scale-ins
            'min_scale_interval': config.get('min_scale_interval', 4),              # Hours between scales
        }
        
    def manage_positions(self, model_name: str, model_provider: str) -> Dict[str, Any]:
        """
        Analyze current positions and generate management actions with enhanced logic.
        
        Args:
            model_name: LLM model name
            model_provider: LLM provider
            
        Returns:
            Dictionary with management actions
        """
        self.logger.info("Starting enhanced position management analysis")
        
        # Get current portfolio state
        portfolio_state = None
        options_portfolio_state = None
        try:
            # Import here to avoid circular imports
            from src.portfolio.manager import PortfolioManager
            
            # Use the existing PortfolioManager instance if possible, or create one
            portfolio_manager = None
            
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                 portfolio_manager = self.portfolio_manager
                 self.logger.info("Using existing portfolio manager reference")
            else:
                 self.logger.info("Creating new portfolio manager instance")
                 portfolio_manager = PortfolioManager(config=self.config)
                 # Store a reference to portfolio_manager for later use
                 self.portfolio_manager = portfolio_manager
                 
            portfolio_state = portfolio_manager.get_portfolio_state()
            options_portfolio_state = portfolio_manager.get_options_portfolio_state()
            
            # Log position count to help with debugging
            stock_count = len(portfolio_state.get('positions', {}))
            options_count = len(options_portfolio_state.get('positions', {}))
            self.logger.info(f"Found {options_count} option positions out of {stock_count} total positions")
        except Exception as e:
            self.logger.error(f"Error getting portfolio state: {e}", exc_info=True)
            return {"error": f"Failed to get portfolio state: {str(e)}"}
            
        # Extract current positions
        stock_positions = portfolio_state.get('positions', {})
        option_positions = options_portfolio_state.get('positions', {})
        
        # Combine stock and option positions for analysis
        all_positions_to_analyze = {}
        
        # First add all stock positions
        all_positions_to_analyze.update(stock_positions)
        
        # Then add all option positions, ensuring correct merging
        all_positions_to_analyze.update(option_positions)
        
        # Log the count of positions we found to aid debugging
        self.logger.info(f"Analyzing {len(all_positions_to_analyze)} positions: {len(stock_positions)} stocks and {len(option_positions)} options")
        
        if not all_positions_to_analyze:
            self.logger.info("No positions to manage")
            return {"status": "no_positions"}
            
        # Process all positions
        management_actions = {}
        
        # Analyze each position (stock or option)
        for ticker, position in all_positions_to_analyze.items():
            self.logger.info(f"Analyzing position for {ticker}")
            progress.update_status("position_management_agent", ticker, "Analyzing position")
            
            # Get historical data and performance metrics
            try:
                # Determine if it's an option using a more comprehensive check
                # This pattern matches standard OCC format as well as some variations
                is_option = (
                    ticker.startswith('O:') or 
                    bool(re.match(r"^[A-Z]{1,6}(\d{6})([CP])(\d{8})$", ticker)) or
                    bool(re.match(r"^[A-Z]{1,6}_(\d{6})_?([CP])_?(\d{8})$", ticker)) or
                    'option_type' in position or
                    'strike_price' in position
                )
                
                if is_option:
                    self.logger.info(f"Position {ticker} detected as an option contract. Side: {position.get('side', 'unknown')}")
                    # For options, ensure the side is set correctly (some brokers use sign of qty instead)
                    qty = position.get('qty', 0)
                    if qty < 0 and position.get('side', 'long') == 'long':
                        self.logger.warning(f"Option position {ticker} has negative qty but side is 'long', correcting side to 'short'")
                        position['side'] = 'short'
                    elif qty > 0 and position.get('side', 'long') == 'short':
                        self.logger.warning(f"Option position {ticker} has positive qty but side is 'short', correcting side to 'long'")
                        position['side'] = 'long'
                    
                position_metrics = self._get_position_metrics(ticker, position)
                
                # Get related option positions (only relevant if analyzing a stock)
                related_options = {}
                if not is_option:
                    for option_ticker, option_position in option_positions.items():
                        # Ensure option_position has 'underlying' key
                        if option_position.get('underlying') == ticker:
                            related_options[option_ticker] = option_position
                        
                # Generate management decisions using LLM
                position_actions_result = self._analyze_position(
                    ticker=ticker,
                    position=position,
                    position_metrics=position_metrics,
                    related_options=related_options, # Pass empty dict if analyzing an option
                    model_name=model_name,
                    model_provider=model_provider
                )
                
                # Store the actions if analysis was successful
                if position_actions_result and position_actions_result.actions:
                    # Filter out invalid actions (e.g., None)
                    valid_actions = [action.model_dump() for action in position_actions_result.actions if action]
                    if valid_actions:
                        management_actions[ticker] = {
                            "actions": valid_actions
                        }
                    else:
                        self.logger.warning(f"Analysis for {ticker} resulted in no valid actions.")
                        management_actions[ticker] = {"actions": []} # Ensure key exists
                elif position_actions_result:
                     self.logger.info(f"Analysis for {ticker} resulted in no actions.")
                     management_actions[ticker] = {"actions": []} # Ensure key exists
                else:
                     self.logger.error(f"Analysis for {ticker} failed or returned None.")
                     management_actions[ticker] = {"error": "Analysis failed"}
                    
            except Exception as e:
                self.logger.error(f"Error analyzing position for {ticker}: {e}", exc_info=True)
                management_actions[ticker] = {"error": str(e)}
                
            progress.update_status("position_management_agent", ticker, "Done")
            
        # Create consolidated result
        result = {
            "timestamp": datetime.now().isoformat(),
            "management_actions": management_actions,
            # Pass the fetched states for execution consistency
            "_portfolio_state": portfolio_state, 
            "_options_portfolio_state": options_portfolio_state,
            # Add summary counts for better debugging
            "_summary": {
                "total_positions_analyzed": len(all_positions_to_analyze),
                "stock_positions": len(stock_positions),
                "option_positions": len(option_positions),
                "actions_generated": sum(1 for ticker_actions in management_actions.values() 
                                       if ticker_actions.get("actions") and len(ticker_actions.get("actions", [])) > 0),
                "errors": sum(1 for ticker_actions in management_actions.values() if "error" in ticker_actions)
            } 
        }
        
        # Log summary information
        self.logger.info(f"Position management analysis completed. Generated actions for {result['_summary']['actions_generated']} positions.")
        if result['_summary']['errors'] > 0:
            self.logger.warning(f"Position management encountered {result['_summary']['errors']} errors during analysis.")
        
        return result
    
    def _sync_position_with_db(self, ticker: str, position: Dict[str, Any]) -> int:
        """
        Sync a position with the database, creating or updating as needed.
        
        Args:
            ticker: Position ticker
            position: Position data from portfolio
            
        Returns:
            Position ID in the database
        """
        # Check if we have an active position in DB
        db_position = self.cache.get_active_position(ticker)
        
        position_type = 'option' if self._is_option_symbol(ticker) else 'stock'
        side = position.get('side', 'long')
        entry_price = float(position.get('avg_entry_price', 0))
        current_price = float(position.get('current_price', 0))
        quantity = abs(int(position.get('qty', 0)))
        
        if db_position:
            # Update existing position
            position_id = db_position['id']
            self.cache.update_position_price(position_id, current_price)
            
            # Check if quantity changed (partial fill or scale)
            if db_position['quantity'] != quantity:
                self.logger.info(f"Position quantity changed for {ticker}: {db_position['quantity']} -> {quantity}")
                # Record the change as an action
                self.cache.record_position_action(
                    position_id=position_id,
                    action='quantity_change',
                    ticker=ticker,
                    quantity=quantity - db_position['quantity'],
                    price=current_price,
                    reason='Position quantity changed (fill or manual adjustment)'
                )
        else:
            # Create new position tracking
            self.logger.info(f"Creating new position tracking for {ticker}")
            
            # Calculate initial stops based on config
            stop_loss_pct = self.config.get('stop_loss_pct', 0.25) if position_type == 'option' else self.config.get('stock_stop_loss_pct', 0.05)
            profit_target_pct = self.config.get('profit_target_pct', 0.50) if position_type == 'option' else self.config.get('stock_profit_target_pct', 0.15)
            
            if side == 'long':
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + profit_target_pct)
            else:  # short
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - profit_target_pct)
            
            position_id = self.cache.create_position_tracking(
                ticker=ticker,
                position_type=position_type,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                additional_data={
                    'underlying': position.get('underlying'),
                    'option_type': position.get('option_type'),
                    'strike_price': position.get('strike_price'),
                    'expiration_date': position.get('expiration_date'),
                }
            )
            
            # Create profit-taking levels for new positions
            if side == 'long' and self.scaling_config['profit_scale_out_levels']:
                profit_levels = []
                for level in self.scaling_config['profit_scale_out_levels']:
                    target_price = entry_price * (1 + level['pnl_pct'] / 100)
                    profit_levels.append({
                        'pnl_percent': level['pnl_pct'],
                        'target_price': target_price,
                        'quantity_percent': level['qty_pct']
                    })
                self.cache.create_profit_levels(ticker, position_id, profit_levels)
                self.logger.info(f"Created {len(profit_levels)} profit-taking levels for {ticker}")
        
        return position_id
        
    def _get_position_metrics(self, ticker: str, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate enhanced performance metrics for a position including DB tracking.
        
        Args:
            ticker: The ticker symbol (stock or option)
            position: Position information
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Sync with database and get position ID
        position_id = self._sync_position_with_db(ticker, position)
        metrics['position_id'] = position_id
        
        # Get DB position for enhanced tracking
        db_position = self.cache.get_active_position(ticker)
        if db_position:
            metrics['db_position'] = db_position
            metrics['scale_count'] = db_position.get('scale_count', 0)
            metrics['highest_price'] = db_position.get('highest_price')
            metrics['lowest_price'] = db_position.get('lowest_price')
            metrics['trailing_stop_active'] = db_position.get('trailing_stop_active', False)
            metrics['last_scale_price'] = db_position.get('last_scale_price')
            
            # Get profit levels
            profit_levels = self.cache.get_profit_levels(position_id)
            metrics['profit_levels'] = profit_levels
            metrics['next_profit_level'] = profit_levels[0] if profit_levels else None
        
        # Extract basic position data
        side = position.get('side', 'long')
        entry_price = float(position.get('avg_entry_price', 0))
        current_price = float(position.get('current_price', 0))
        quantity = abs(int(position.get('qty', 0)))
        market_value = abs(float(position.get('market_value', 0)))
        
        # Calculate P&L metrics (based on the specific asset)
        if entry_price > 0 and current_price > 0:
            if side == 'long':
                profit_loss_pct = ((current_price / entry_price) - 1) * 100
            else:  # short
                profit_loss_pct = ((entry_price / current_price) - 1) * 100
        else:
            profit_loss_pct = 0
            
        metrics['profit_loss_pct'] = profit_loss_pct
        
        # Determine if it's an option and get underlying symbol
        is_option = ticker.startswith('O:') or bool(re.match(r"^[A-Z]{1,6}(\d{6})([CP])(\d{8})$", ticker))
        underlying_symbol = ticker
        if is_option:
            try:
                # Attempt to parse OCC format or rely on position data if available
                if 'underlying' in position:
                    underlying_symbol = position['underlying']
                else:
                    # Basic OCC parsing logic (may need refinement)
                    match = re.match(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$", ticker)
                    if match:
                        underlying_symbol = match.group(1)
                    else:
                        self.logger.warning(f"Could not extract underlying from option symbol {ticker}. Cannot fetch technicals.")
                        underlying_symbol = None # Indicate failure
            except Exception as e:
                self.logger.error(f"Error parsing underlying symbol from {ticker}: {e}")
                underlying_symbol = None
            
        # Get historical data for technical analysis using the underlying symbol
        metrics['technical'] = {}
        metrics['trends'] = {}
        metrics['volatility'] = 0
        
        if underlying_symbol:
            try:
                # Get daily bars for the past 30 days for the UNDERLYING
                bars = self.alpaca.get_market_data(underlying_symbol, timeframe='1D', limit=30)
                
                if bars and len(bars) > 0:
                    # Convert to DataFrame for analysis
                    df = pd.DataFrame(bars)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                    
                    # --- Calculate technical indicators based on UNDERLYING --- 
                    
                    # Moving Averages
                    df['ma_5'] = df['close'].rolling(window=5).mean()
                    df['ma_10'] = df['close'].rolling(window=10).mean()
                    df['ma_20'] = df['close'].rolling(window=20).mean()
                    
                    # RSI (14-day)
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    
                    # MACD
                    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
                    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
                    df['macd'] = df['ema_12'] - df['ema_26']
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    
                    # ATR (14-day)
                    high_low = df['high'] - df['low']
                    high_close = abs(df['high'] - df['close'].shift())
                    low_close = abs(df['low'] - df['close'].shift())
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    df['atr'] = tr.rolling(window=14).mean()
                    
                    # Bollinger Bands
                    df['bb_middle'] = df['close'].rolling(window=20).mean()
                    df['bb_std'] = df['close'].rolling(window=20).std()
                    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
                    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
                    
                    # Get the latest values for metrics
                    latest = df.iloc[-1] if not df.empty else {}
                    
                    # Store metrics
                    metrics['technical'] = {
                        'ma_5': float(latest.get('ma_5', 0)),
                        'ma_10': float(latest.get('ma_10', 0)),
                        'ma_20': float(latest.get('ma_20', 0)),
                        'rsi': float(latest.get('rsi', 50)),
                        'macd': float(latest.get('macd', 0)),
                        'macd_signal': float(latest.get('macd_signal', 0)),
                        'atr': float(latest.get('atr', 0)), # Underlying ATR
                        'bb_upper': float(latest.get('bb_upper', 0)),
                        'bb_middle': float(latest.get('bb_middle', 0)),
                        'bb_lower': float(latest.get('bb_lower', 0))
                    }
                    
                    # Add trend information based on UNDERLYING
                    trends = {}
                    underlying_current_price = df['close'].iloc[-1] # Use latest close of underlying
                    
                    # Price trend over different timeframes
                    if len(df) >= 5:
                        trends['price_5d'] = 'up' if underlying_current_price > df['close'].iloc[-5] else 'down'
                    if len(df) >= 10:
                        trends['price_10d'] = 'up' if underlying_current_price > df['close'].iloc[-10] else 'down'
                    
                    # Moving average trends
                    trends['ma_cross'] = 'bullish' if metrics['technical']['ma_5'] > metrics['technical']['ma_10'] else 'bearish'
                    
                    # RSI trend
                    rsi = metrics['technical']['rsi']
                    trends['rsi'] = 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
                    
                    # MACD trend
                    trends['macd'] = 'bullish' if metrics['technical']['macd'] > metrics['technical']['macd_signal'] else 'bearish'
                    
                    # Bollinger Band position
                    bb_upper = metrics['technical']['bb_upper']
                    bb_lower = metrics['technical']['bb_lower']
                    trends['bb'] = 'above_upper' if underlying_current_price > bb_upper else 'below_lower' if underlying_current_price < bb_lower else 'within_bands'
                    
                    metrics['trends'] = trends
                    
                    # Calculate volatility (standard deviation of returns) of UNDERLYING
                    df['returns'] = df['close'].pct_change()
                    metrics['volatility'] = float(df['returns'].std() * 100)  # as percentage
                    
                else:
                    self.logger.warning(f"No historical data available for underlying {underlying_symbol} (from {ticker})")
                    
            except Exception as e:
                self.logger.error(f"Error calculating technical metrics for underlying {underlying_symbol} (from {ticker}): {e}")
            
        # Add stop loss and take profit levels based on the ASSET (stock or option)
        stop_loss_pct = self.config.get('stop_loss_pct', 0.25) # Default higher for options
        profit_target_pct = self.config.get('profit_target_pct', 0.50) # Default higher for options
        if not is_option:
            stop_loss_pct = self.config.get('stock_stop_loss_pct', 0.05)
            profit_target_pct = self.config.get('stock_profit_target_pct', 0.15)
        
        if entry_price > 0: # Avoid division by zero
            if side == 'long':
                metrics['stop_price'] = entry_price * (1 - stop_loss_pct)
                metrics['target_price'] = entry_price * (1 + profit_target_pct)
            else:  # short
                metrics['stop_price'] = entry_price * (1 + stop_loss_pct)
                metrics['target_price'] = entry_price * (1 - profit_target_pct)
        else:
            metrics['stop_price'] = 0
            metrics['target_price'] = 0
            
        # Additional metrics for trailing stop if configured
        metrics['effective_stop'] = metrics.get('stop_price', 0) # Default to initial stop
        
        # Enhanced trailing stop logic
        if db_position and db_position.get('trailing_stop_active'):
            trailing_distance = db_position.get('trailing_stop_distance', self.scaling_config['trailing_stop_distance'])
            if side == 'long' and metrics.get('highest_price'):
                metrics['effective_stop'] = metrics['highest_price'] * (1 - trailing_distance / 100)
            elif side == 'short' and metrics.get('lowest_price'):
                metrics['effective_stop'] = metrics['lowest_price'] * (1 + trailing_distance / 100)
        
        # Check if trailing stop should be activated
        elif profit_loss_pct >= self.scaling_config['trailing_stop_activation']:
            metrics['should_activate_trailing'] = True
            
        # Scaling analysis
        metrics['scaling_analysis'] = self._analyze_scaling_opportunity(
            ticker, position, metrics, profit_loss_pct
        )
            
        return metrics
    
    def _analyze_scaling_opportunity(self, ticker: str, position: Dict[str, Any], 
                                   metrics: Dict[str, Any], pnl_pct: float) -> Dict[str, Any]:
        """
        Analyze if position should be scaled based on performance and momentum.
        
        Returns:
            Dictionary with scaling recommendations
        """
        analysis = {
            'should_scale_in': False,
            'should_scale_out': False,
            'scale_reason': None,
            'scale_size': 0
        }
        
        side = position.get('side', 'long')
        scale_count = metrics.get('scale_count', 0)
        last_scale_price = metrics.get('last_scale_price')
        current_price = float(position.get('current_price', 0))
        
        # Check scale-in opportunities
        if scale_count < self.scaling_config['max_scale_count']:
            # Loss-based scale-in
            if pnl_pct <= self.scaling_config['loss_scale_in_threshold']:
                # Check if enough price movement since last scale
                if not last_scale_price or abs((current_price - last_scale_price) / last_scale_price) > 0.05:
                    analysis['should_scale_in'] = True
                    analysis['scale_reason'] = f"Position down {pnl_pct:.1f}%, opportunity to average down"
                    analysis['scale_size'] = 0.5  # Scale in with 50% of current position
            
            # Momentum-based scale-in (for winning positions)
            elif pnl_pct > self.scaling_config['momentum_scale_threshold']:
                # Check technical momentum
                technical = metrics.get('technical', {})
                trends = metrics.get('trends', {})
                
                momentum_signals = 0
                if trends.get('ma_cross') == 'bullish': momentum_signals += 1
                if trends.get('macd') == 'bullish': momentum_signals += 1
                if technical.get('rsi', 50) > 50 and technical.get('rsi', 50) < 70: momentum_signals += 1
                
                if momentum_signals >= 2:
                    analysis['should_scale_in'] = True
                    analysis['scale_reason'] = f"Strong momentum with {momentum_signals} bullish signals"
                    analysis['scale_size'] = 0.3  # Scale in with 30% for momentum
        
        # Check scale-out opportunities (profit-taking)
        if metrics.get('next_profit_level'):
            next_level = metrics['next_profit_level']
            if pnl_pct >= next_level['target_pnl_percent']:
                analysis['should_scale_out'] = True
                analysis['scale_reason'] = f"Reached profit level {next_level['level_number']} at {pnl_pct:.1f}%"
                analysis['scale_size'] = next_level['quantity_percent'] / 100
        
        # Emergency scale-out for excessive losses
        elif pnl_pct <= self.scaling_config['loss_scale_out_threshold']:
            analysis['should_scale_out'] = True
            analysis['scale_reason'] = f"Risk reduction due to {pnl_pct:.1f}% loss"
            analysis['scale_size'] = 0.5  # Reduce position by 50%
        
        return analysis
        
    def _analyze_position(
        self,
        ticker: str,
        position: Dict[str, Any],
        position_metrics: Dict[str, Any],
        related_options: Dict[str, Dict[str, Any]],
        model_name: str,
        model_provider: str
    ) -> Optional[PositionManagementResult]:
        """
        Analyze position using LLM for management decisions.
        
        Args:
            ticker: The ticker symbol
            position: Position information
            position_metrics: Calculated metrics for the position
            related_options: Related options positions
            model_name: LLM model name
            model_provider: LLM provider
            
        Returns:
            PositionManagementResult with actions to take
        """
        # Create prompt for position analysis
        template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert portfolio manager specializing in dynamic position management and systematic profit optimization.

Your task is to analyze a current position and determine optimal actions to maximize profits while minimizing losses through intelligent position scaling and risk management.

Consider the following actions:
1. scale_in: Add to the position based on momentum or to improve cost basis
2. scale_out: Partially reduce the position to lock in profits or reduce risk
3. stop_loss: Exit the position completely to prevent further losses
4. take_profit: Exit the position completely to realize gains
5. adjust_stop: Modify the stop loss level (including trailing stop activation)
6. adjust_target: Modify the profit target based on current price action
7. hold: Make no changes to the current position

ENHANCED ANALYSIS FRAMEWORK:
1. Scaling Analysis:
   - Review the scaling_analysis in metrics for automatic recommendations
   - Consider scale count and last scale price to avoid over-scaling
   - For losses: Consider averaging down if fundamentals remain strong
   - For profits: Follow systematic profit-taking levels
   
2. Trailing Stop Management:
   - Activate trailing stops after 15% profit (configurable)
   - Adjust stops to lock in gains while allowing upside
   - Consider volatility when setting trailing distances
   
3. Profit Taking Strategy:
   - Check profit_levels for systematic exit points
   - Take partial profits at predetermined levels (10%, 20%, 30%, 50%)
   - Balance between letting winners run and securing gains
   
4. Risk Management:
   - Monitor effective stop levels (may differ from initial stops)
   - Consider position size relative to portfolio
   - Account for correlation with other positions

5. Technical Confirmation:
   - Use technical indicators to confirm scaling decisions
   - Look for momentum signals when adding to winners
   - Identify reversal signals for exit timing

Return a PositionManagementResult object with prioritized actions.
Each action should include:
- ticker: The ticker symbol
- action: One of the actions listed above
- reason: Detailed explanation incorporating the enhanced analysis
- confidence: Confidence level (0-100)
- quantity: Number of shares/contracts (for scale_in or scale_out) - MUST BE A WHOLE INTEGER
- price_target: Target price (for adjust_stop or adjust_target)

IMPORTANT: Consider the scaling_analysis recommendations but use your judgment to confirm or override based on the full context.
"""
            ),
            (
                "human",
                """Here is the current position to analyze for ticker {ticker}:

Position Information:
```json
{position}
```

Position Metrics:
```json
{position_metrics}
```

Related Options Positions:
```json
{related_options}
```

Risk Management Configuration:
```json
{risk_config}
```

Analyze this position and recommend management actions.
Return a PositionManagementResult in this JSON format:
{{"actions": [
  {{
    "ticker": "{ticker}",
    "action": "scale_in|scale_out|stop_loss|take_profit|adjust_stop|adjust_target|hold",
    "reason": "Detailed explanation",
    "confidence": float between 0 and 100,
    "quantity": integer or null (must be whole numbers like 1, 2, 3 etc., not decimals like 0.3),
    "price_target": float or null
  }}
]}}

You can recommend multiple actions if needed, but ensure they are consistent with each other.
"""
            )
        ])
        
        # Extract risk management config for the prompt
        risk_config = {
            'stop_loss_pct': self.config.get('stop_loss_pct', 0.05),
            'profit_target_pct': self.config.get('profit_target_pct', 0.15),
            'scaling_threshold_pct': self.config.get('scaling_threshold_pct', 0.07),
            'scaling_size_pct': self.config.get('scaling_size_pct', 0.30),
            'enable_trailing_stops': self.config.get('enable_trailing_stops', False),
            'trailing_stop_atr_multiplier': self.config.get('trailing_stop_atr_multiplier', 2.5),
        }
        
        prompt = template.invoke({
            "ticker": ticker,
            "position": json.dumps(position, indent=2),
            "position_metrics": json.dumps(position_metrics, indent=2),
            "related_options": json.dumps(related_options, indent=2),
            "risk_config": json.dumps(risk_config, indent=2)
        })
        
        # Define default result
        def create_default_result():
            return PositionManagementResult(actions=[
                PositionAction(
                    ticker=ticker,
                    action="hold",
                    reason="Default hold action due to analysis error or lack of data",
                    confidence=0.0,
                    quantity=None,
                    price_target=None
                )
            ])
        
        # Get LLM analysis
        try:
            result = call_llm(
                prompt=prompt,
                model_name=model_name,
                model_provider=model_provider,
                pydantic_model=PositionManagementResult,
                agent_name="position_management_agent",
                default_factory=create_default_result,
            )
            
            # Check if LLM call returned None (error scenario)
            if result is None:
                self.logger.error(f"LLM call for position management of {ticker} returned None.")
                return create_default_result()
            
            # Validate result - ensure quantities are integers and price_target is provided when needed
            if result.actions:
                for action in result.actions:
                    # Validate price_target for adjust actions
                    if action.action in ['adjust_stop', 'adjust_target'] and action.price_target is None:
                        self.logger.warning(f"LLM did not provide price_target for {action.action} on {ticker}")
                        # Try to calculate a reasonable price_target based on current price and position
                        current_price = float(position.get('current_price', 0))
                        entry_price = float(position.get('avg_entry_price', 0))
                        if current_price > 0:
                            if action.action == 'adjust_stop':
                                # Set stop at 5% below current for long, 5% above for short
                                if position.get('side', 'long') == 'long':
                                    action.price_target = current_price * 0.95
                                else:
                                    action.price_target = current_price * 1.05
                                self.logger.info(f"Set default stop price to {action.price_target:.2f} for {ticker}")
                            elif action.action == 'adjust_target':
                                # Set target based on profit level or default percentage
                                pnl_pct = position.get('unrealized_pl_percent', 0)
                                if position.get('side', 'long') == 'long':
                                    # If already profitable, set target 10% above current
                                    if pnl_pct > 10:
                                        action.price_target = current_price * 1.10
                                    else:
                                        # Otherwise use entry price + 15% (stocks) or 30% (options)
                                        target_pct = 0.30 if self._is_option_symbol(ticker) else 0.15
                                        action.price_target = entry_price * (1 + target_pct) if entry_price > 0 else current_price * 1.15
                                else:
                                    # Short position
                                    if pnl_pct > 10:
                                        action.price_target = current_price * 0.90
                                    else:
                                        target_pct = 0.30 if self._is_option_symbol(ticker) else 0.15
                                        action.price_target = entry_price * (1 - target_pct) if entry_price > 0 else current_price * 0.85
                                self.logger.info(f"Set default target price to {action.price_target:.2f} for {ticker}")
                    
                    # If quantity is a float, convert it to an integer
                    if action.quantity is not None and isinstance(action.quantity, float):
                        position_qty = abs(int(position.get('qty', 0)))
                        original_qty = action.quantity
                        # If it's a fraction (< 1), interpret it as a percentage of the position
                        if action.quantity < 1:
                            # Calculate actual quantity based on percentage
                            action.quantity = max(1, int(position_qty * action.quantity))
                            self.logger.info(f"Converted fractional quantity {original_qty} to integer {action.quantity} for {ticker}")
                        else:
                            # Just round to the nearest integer
                            action.quantity = int(action.quantity)
                            self.logger.info(f"Rounded quantity {original_qty} to integer {action.quantity} for {ticker}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Exception during LLM call for position management of {ticker}: {e}", exc_info=True)
            return create_default_result()
        
    def _validate_action(self, ticker: str, action: str, position: Dict[str, Any], is_option: bool) -> Tuple[bool, str]:
        """
        Validate if an action is appropriate for the given position.
        
        Args:
            ticker: Position ticker
            action: The proposed action (scale_in, scale_out, stop_loss, take_profit)
            position: The position data dictionary
            is_option: Whether this is an option position
            
        Returns:
            Tuple of (is_valid, reason_message)
        """
        # Skip validation for hold/adjust_stop which don't execute trades
        if action in ['hold', 'adjust_stop', 'adjust_target']:
            return True, "Non-trading action"
            
        # Always validate these basic things regardless of position type
        if not position:
            return False, f"Position data missing for {ticker}"
            
        qty = position.get('qty', 0)
        if qty == 0:
            return False, f"Position {ticker} has zero quantity"
        
        side = position.get('side', 'unknown')
        if side == 'unknown':
            return False, f"Position {ticker} has unknown side"
        
        # For scaling out actions, check we have sufficient quantity
        if action in ['scale_out', 'stop_loss', 'take_profit']:
            abs_qty = abs(qty)
            if abs_qty <= 0:
                return False, f"Cannot {action} on {ticker} - insufficient quantity available ({abs_qty})"
            
        # For take_profit, verify we have a profit
        if action == 'take_profit':
            pnl_pct = position.get('unrealized_pl_percent', 0)
            if pnl_pct <= 0:
                return False, f"Cannot take profit on {ticker} with PnL {pnl_pct}%"
                
        # For stop_loss, verify we have a loss
        if action == 'stop_loss':
            pnl_pct = position.get('unrealized_pl_percent', 0)
            if pnl_pct >= 0:
                return False, f"Cannot stop loss on {ticker} with PnL {pnl_pct}% (not a loss)"
                
        # Option-specific validations
        if is_option:
            # No specific option limitations other than using correct buy/sell mechanics
            # which are handled in _execute_trade_action
            pass
            
        return True, "Action is valid"

    def execute_management_actions(self, management_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the recommended management actions.
        
        Args:
            management_result: Result from manage_positions, which includes portfolio states.
            
        Returns:
            Dictionary with execution results
        """
        execution_results = {}
        
        # Extract management actions and portfolio states
        management_actions = management_result.get('management_actions', {})
        portfolio_state = management_result.get('_portfolio_state')
        options_portfolio_state = management_result.get('_options_portfolio_state')
        
        # Validate that portfolio states were passed
        if not portfolio_state or not options_portfolio_state:
            self.logger.error("Portfolio states not provided with management actions")
            return {
                'status': 'ERROR',
                'message': 'Portfolio states missing, cannot execute management actions safely',
                'orders': []
            }
        
        # Create a consolidated positions map for fast lookup
        all_positions = {}
        
        # First add stock positions
        stock_positions = portfolio_state.get('positions', {})
        all_positions.update(stock_positions)
        
        # Then add option positions, ensuring we don't overwrite any stock positions with same symbol
        options_positions = options_portfolio_state.get('positions', {})
        all_positions.update(options_positions)
        
        # Also create position lookup maps for fuzzy matching if needed
        position_lookup_maps = self._create_position_lookup_maps(all_positions)
        
        # Execute each action for each ticker
        for ticker, ticker_data in management_actions.items():
            execution_results[ticker] = []
            
            # Skip if the ticker has an error
            if 'error' in ticker_data:
                execution_results[ticker].append({
                    'action': 'unknown',
                    'status': 'ERROR',
                    'message': f"Analysis error: {ticker_data['error']}",
                    'order': None
                })
                continue
                
            # Get actions from the ticker data
            actions_list = ticker_data.get('actions', [])
            if not actions_list:
                execution_results[ticker].append({
                    'action': 'none',
                    'status': 'INFO',
                    'message': 'No management actions recommended by analysis.',
                    'order': None
                })
                continue
            
            # Find the position (first directly, then using advanced lookup)
            position = all_positions.get(ticker)
            is_option = False
            
            if not position:
                # Try to find using our lookup method
                position_info = self._find_position(ticker, position_lookup_maps)
                if position_info:
                    actual_ticker, position = position_info
                    self.logger.info(f"Found position for {ticker} using lookup as {actual_ticker}")
                    ticker = actual_ticker  # Use the actual ticker from now on
                else:
                    self.logger.warning(f"Position for {ticker} not found during execution")
                    execution_results[ticker].append({
                        'action': 'unknown',
                        'status': 'ERROR',
                        'message': f"Position for {ticker} not found in portfolio",
                        'order': None
                    })
                    continue
            
            # Determine if it's an option using our comprehensive check
            is_option = self._is_option_symbol(ticker)
            
            # Log position details for debugging
            side = position.get('side', 'unknown')
            qty = position.get('qty', 0)
            position_type = "option" if is_option else "stock"
            
            # Process each action for this ticker
            for action_item in actions_list:
                # Handle case where action_item might be a string or dict
                if isinstance(action_item, str):
                    # If it's just a string, create a simple action dict
                    action = action_item
                    action_dict = {
                        'action': action,
                        'reason': f"No specific reason provided for {action}",
                        'confidence': 75,  # Default confidence
                        'quantity': None,  # Use default quantity logic
                        'price_target': None  # No specific price target
                    }
                elif isinstance(action_item, dict):
                    # It's a dictionary with action data
                    action = action_item.get('action', 'unknown')
                    action_dict = action_item
                else:
                    # Unknown format, skip
                    self.logger.warning(f"Unknown action format for {ticker}: {action_item}")
                    continue
                
                # Log position details for each action for better debugging
                self.logger.info(f"Position details - Ticker: {ticker}, Type: {position_type}, Side: {side}, Qty: {qty}, Action: {action}")
                
                # Skip actions with low confidence
                confidence = float(action_dict.get('confidence', 75))
                if confidence < 70 and action not in ['hold', 'adjust_stop']:
                    self.logger.info(f"Skipping action '{action}' for {ticker} due to low confidence ({confidence:.1f})")
                    execution_results[ticker].append({
                        'action': action,
                        'status': 'SKIPPED',
                        'message': f"Low confidence ({confidence:.1f})",
                        'order': None
                    })
                    continue
                
                # Process specific action types
                if action == 'hold':
                    # Just record the hold decision
                    execution_results[ticker].append({
                        'action': 'hold',
                        'status': 'SUCCESS',
                        'message': action_dict.get('reason', 'No specific reason provided for hold'),
                        'order': None
                    })
                    
                elif action == 'adjust_stop':
                    # Handle stop adjustment (update stops in the system)
                    price_target = action_dict.get('price_target')
                    if price_target:
                        # Update stop in database
                        db_position = self.cache.get_active_position(ticker)
                        if db_position:
                            position_id = db_position['id']
                            
                            # Check if we should activate trailing stop
                            pnl_pct = position.get('unrealized_pl_percent', 0)
                            trailing_active = pnl_pct >= self.scaling_config['trailing_stop_activation']
                            
                            self.cache.update_position_stops(
                                position_id=position_id,
                                stop_loss=price_target,
                                trailing_stop_active=trailing_active,
                                trailing_stop_distance=self.scaling_config['trailing_stop_distance'] if trailing_active else None
                            )
                            
                            # Record the action
                            action_id = self.cache.record_position_action(
                                position_id=position_id,
                                action='adjust_stop',
                                ticker=ticker,
                                price=price_target,
                                reason=action_dict.get('reason', f"Stop adjusted to {price_target}"),
                                confidence=confidence
                            )
                            
                            self.cache.update_action_result(action_id, True, 'success')
                            
                            execution_results[ticker].append({
                                'action': 'adjust_stop',
                                'status': 'SUCCESS',
                                'message': f"Stop adjusted to {price_target}" + (" with trailing stop" if trailing_active else ""),
                                'details': {
                                    'ticker': ticker,
                                    'action': 'adjust_stop',
                                    'price_target': price_target,
                                    'trailing_active': trailing_active
                                }
                            })
                        else:
                            execution_results[ticker].append({
                                'action': 'adjust_stop',
                                'status': 'ERROR',
                                'message': f"No active position found in database for {ticker}",
                                'details': None
                            })
                    else:
                        execution_results[ticker].append({
                            'action': 'adjust_stop',
                            'status': 'ERROR',
                            'message': f"No price target specified for adjust_stop on {ticker}",
                            'details': None
                        })
                
                elif action == 'adjust_target':
                    # Handle target adjustment (update take profit targets in the system)
                    price_target = action_dict.get('price_target')
                    if not price_target:
                        # Calculate a reasonable default target based on current position
                        self.logger.warning(f"No price target provided for adjust_target on {ticker}, calculating default")
                        current_price = float(position.get('current_price', 0))
                        entry_price = float(position.get('avg_entry_price', 0))
                        pnl_pct = position.get('unrealized_pl_percent', 0)
                        
                        if current_price > 0 and entry_price > 0:
                            # For options, use more aggressive targets
                            if is_option:
                                target_multiplier = 1.30 if position.get('side', 'long') == 'long' else 0.70
                            else:
                                target_multiplier = 1.15 if position.get('side', 'long') == 'long' else 0.85
                            
                            # If already profitable, set target higher than current
                            if pnl_pct > 10:
                                price_target = current_price * 1.10 if position.get('side', 'long') == 'long' else current_price * 0.90
                            else:
                                price_target = entry_price * target_multiplier
                            
                            self.logger.info(f"Calculated default target price {price_target:.2f} for {ticker}")
                    
                    if price_target:
                        # Execute the adjustment using the existing helper method
                        result = self._execute_adjustment_action(
                            ticker=ticker,
                            action='adjust_target',
                            price_target=price_target,
                            reason=action_dict.get('reason', f"adjust_target set to {price_target} for {ticker}")
                        )
                        execution_results[ticker].append({
                            'action': 'adjust_target',
                            'status': result.get('status', 'ERROR'),
                            'message': result.get('message', f"Target price adjustment for {ticker}"),
                            'details': result.get('details')
                        })
                    else:
                        execution_results[ticker].append({
                            'action': 'adjust_target',
                            'status': 'ERROR',
                            'message': f"Unable to determine target price for {ticker} (no current/entry price)",
                            'details': None
                        })
                        
                elif action in ['scale_in', 'scale_out', 'stop_loss', 'take_profit']:
                    # Validate the action is appropriate for this position
                    is_valid, reason = self._validate_action(ticker, action, position, is_option)
                    
                    if not is_valid:
                        self.logger.warning(f"Invalid action '{action}' for {ticker}: {reason}")
                        execution_results[ticker].append({
                            'action': action,
                            'status': 'SKIPPED',
                            'message': f"Action '{action}' not valid for this position: {reason}",
                            'order': None
                        })
                        continue
                        
                    # Extract action details
                    reason = action_dict.get('reason', f"No specific reason provided for {action}")
                    quantity = action_dict.get('quantity')  # Can be None, _execute_trade_action handles defaults
                    
                    # Execute the trade action
                    result = self._execute_trade_action(
                        ticker=ticker,
                        action=action,
                        quantity=quantity,
                        reason=reason,
                        confidence=confidence,
                        portfolio_state=portfolio_state,
                        options_portfolio_state=options_portfolio_state
                    )
                    
                    # Extract and normalize status for display
                    status_value = result.get('status', '').upper()
                    if status_value == 'EXECUTED' or status_value == 'SUCCESS':
                        display_status = 'EXECUTED'
                    elif status_value == 'SKIPPED':
                        display_status = 'SKIPPED'
                    elif status_value == 'ERROR':
                        display_status = 'FAILED'
                    else:
                        display_status = 'PENDING'
                        
                    # Extract quantity for display if available
                    display_qty = "-"
                    if result.get('order') and 'qty' in result['order']:
                        display_qty = str(result['order']['qty'])
                    elif 'quantity' in result:
                        display_qty = str(result['quantity'])
                    
                    execution_results[ticker].append({
                        'action': action,
                        'status': display_status,
                        'quantity': display_qty,
                        'result': result
                    })
                else:
                    # Unknown action type
                    self.logger.warning(f"Unknown action type: {action} for {ticker}")
                    execution_results[ticker].append({
                        'action': 'unknown',
                        'status': 'ERROR',
                        'message': f"Unknown action type: {action}",
                        'order': None
                    })
        
        return execution_results
        
    def _create_position_lookup_maps(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create lookup maps for case-insensitive and format-insensitive position matching.
        
        Args:
            positions: Dictionary of positions keyed by ticker
            
        Returns:
            Dictionary with lookup maps
        """
        # Two main maps:
        # 1. Lowercase map for case-insensitive lookups
        # 2. Normalized map for option format variations
        lowercase_map = {}
        normalized_map = {}
        
        for ticker, position in positions.items():
            # Case-insensitive lookup
            lowercase_map[ticker.lower()] = (ticker, position)
            
            # For options: handle O: prefix and different formats
            is_option = ticker.startswith('O:') or bool(re.match(r"^[A-Z]{1,6}(\d{6})([CP])(\d{8})$", ticker))
            
            if is_option:
                # With or without O: prefix
                if ticker.startswith('O:'):
                    normalized_map[ticker[2:].upper()] = (ticker, position)
                else:
                    normalized_map[ticker.upper()] = (ticker, position)
                    normalized_map[f"O:{ticker}".upper()] = (ticker, position)
                
                # Try to parse OCC format for additional normalized forms
                try:
                    # Match standard OCC format
                    match = re.match(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$", ticker)
                    if match:
                        root, date, cp, strike = match.groups()
                        
                        # Create various format variations
                        alt_formats = [
                            f"{root}_{date}{cp}{strike}".upper(),
                            f"{root}{date}_{cp}_{strike}".upper(),
                            f"{root}_{date}_{cp}_{strike}".upper(),
                        ]
                        
                        for alt_format in alt_formats:
                            normalized_map[alt_format] = (ticker, position)
                    
                    # Handle O: prefix variations
                    if ticker.startswith('O:'):
                        match = re.match(r"^O:([A-Z]{1,6})(\d{6})([CP])(\d{8})$", ticker)
                        if match:
                            root, date, cp, strike = match.groups()
                            normalized_map[f"{root}{date}{cp}{strike}".upper()] = (ticker, position)
                except Exception as e:
                    self.logger.debug(f"Error creating normalized forms for {ticker}: {e}")
        
        return {
            "lowercase": lowercase_map,
            "normalized": normalized_map,
            "original": positions
        }
    
    def _find_position(self, ticker: str, lookup_maps: Dict[str, Any]) -> Optional[Tuple[str, Dict]]:
        """
        Find a position using robust lookup methods.
        
        Args:
            ticker: The ticker to find
            lookup_maps: Lookup maps created by _create_position_lookup_maps
            
        Returns:
            Tuple of (actual_ticker, position_data) if found, None otherwise
        """
        # 1. Direct lookup in original positions
        if ticker in lookup_maps["original"]:
            return (ticker, lookup_maps["original"][ticker])
        
        # 2. Case-insensitive lookup
        if ticker.lower() in lookup_maps["lowercase"]:
            return lookup_maps["lowercase"][ticker.lower()]
        
        # 3. For options: normalized format lookup (handles O: prefix and format variations)
        is_option = ticker.startswith('O:') or bool(re.match(r"^[A-Z]{1,6}(\d{6})([CP])(\d{8})$", ticker))
        
        if is_option:
            # Try with normalized forms
            # Remove O: prefix if present
            normalized_ticker = ticker[2:].upper() if ticker.startswith('O:') else ticker.upper()
            
            if normalized_ticker in lookup_maps["normalized"]:
                return lookup_maps["normalized"][normalized_ticker]
            
            # Try to parse OCC format for additional lookups
            try:
                # First extract the ticker pattern regardless of O: prefix
                pattern = ticker[2:] if ticker.startswith('O:') else ticker
                
                match = re.match(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$", pattern)
                if match:
                    root, date, cp, strike = match.groups()
                    
                    # Try various format combinations
                    alt_formats = [
                        f"{root}{date}{cp}{strike}".upper(),
                        f"{root}_{date}{cp}{strike}".upper(),
                        f"{root}{date}_{cp}_{strike}".upper(),
                    ]
                    
                    for alt_format in alt_formats:
                        if alt_format in lookup_maps["normalized"]:
                            return lookup_maps["normalized"][alt_format]
            except Exception as e:
                self.logger.debug(f"Error in normalized lookup for {ticker}: {e}")
        
        # 4. Last resort: partial matching for option tickers
        if is_option:
            # For options, try to find any position with matching components
            try:
                pattern = ticker[2:] if ticker.startswith('O:') else ticker
                match = re.match(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$", pattern)
                
                if match:
                    root, date, cp, strike = match.groups()
                    
                    # Look for positions with the same underlying, expiry, and option type
                    base_pattern = f"{root}{date}{cp}"
                    
                    for key, value in lookup_maps["original"].items():
                        if base_pattern.upper() in key.upper():
                            return (key, value)
            except Exception as e:
                self.logger.debug(f"Error in partial option matching for {ticker}: {e}")
        
        # Not found with any method
        return None
        
    def _is_option_symbol(self, ticker: str) -> bool:
        """
        Determine if a ticker represents an option contract using comprehensive pattern matching.
        
        Args:
            ticker: The ticker symbol to check
            
        Returns:
            True if it's an option symbol, False otherwise
        """
        # Common option prefixes
        if ticker.startswith(('O:', 'OPTION:')):
            return True
            
        # Standard OCC format: AAPL230616C00170000
        if re.match(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$", ticker):
            return True
            
        # Format with underscores: AAPL_230616_C_170.00
        if re.match(r"^[A-Z]{1,6}_\d{6}_[CP]_\d+\.\d+$", ticker):
            return True
            
        # Format with spaces: AAPL 230616C170
        if re.match(r"^[A-Z]{1,6} \d{6}[CP]\d+(\.\d+)?$", ticker):
            return True
            
        # Format with dashes: AAPL-230616-C-170
        if re.match(r"^[A-Z]{1,6}-\d{6}-[CP]-\d+(\.\d+)?$", ticker):
            return True
            
        # Format with slashes: AAPL/230616/C/170
        if re.match(r"^[A-Z]{1,6}/\d{6}/[CP]/\d+(\.\d+)?$", ticker):
            return True
            
        return False
        
    def _execute_trade_action(self, ticker: str, action: str, quantity: Optional[int], 
                              reason: str, confidence: float, 
                              portfolio_state: Dict[str, Any], 
                              options_portfolio_state: Dict[str, Any]
                             ) -> Dict[str, Any]:
        """
        Execute a trade action (scale_in, scale_out, stop_loss, take_profit).
        Uses the provided portfolio state for consistency.
        
        Args:
            ticker: The ticker symbol (stock or option) - should be the ACTUAL ticker as found in the portfolio
            action: Action to take
            quantity: Quantity to trade (can be None for full positions)
            reason: Reason for the action
            confidence: Confidence in the action
            portfolio_state: The portfolio state used for analysis.
            options_portfolio_state: The options portfolio state used for analysis.
            
        Returns:
            Dictionary with execution result
        """
        # Determine if it's an option symbol using our comprehensive check
        is_option = self._is_option_symbol(ticker)
        
        # Get positions from the PASSED-IN state
        if is_option:
            positions = options_portfolio_state.get('positions', {})
        else:
            positions = portfolio_state.get('positions', {})
        
        # Since we're now using the exact ticker from the portfolio (passed from execute_management_actions),
        # we should find the position directly without additional matching
        position = positions.get(ticker)
        
        # If not found directly (which should be rare now), use more robust matching as fallback
        if not position:
            self.logger.warning(f"Position for {ticker} not found directly in expected portfolio state. Trying advanced lookup.")
            
            # Create a combined dictionary and use our robust lookup method
            all_positions = {}
            all_positions.update(portfolio_state.get('positions', {}))
            all_positions.update(options_portfolio_state.get('positions', {}))
            
            lookup_maps = self._create_position_lookup_maps(all_positions)
            position_info = self._find_position(ticker, lookup_maps)
            
            if position_info:
                actual_ticker, position = position_info
                self.logger.info(f"Found position for {ticker} using advanced lookup: {actual_ticker}")
            else:
                # If closing and position not found, maybe it was already closed
                if action in ['stop_loss', 'take_profit']:
                    self.logger.warning(f"Position for {ticker} not found during execution, potentially already closed.")
                    return {
                        'status': 'SKIPPED',
                        'message': f"Position for {ticker} not found during execution, potentially already closed.",
                        'order': None
                    }
                else:
                    # For scale_in/scale_out, if analysis recommended it but position disappeared,
                    # it's an error/inconsistency.
                    self.logger.error(f"Position for {ticker} not found during execution attempt for action '{action}'. Analysis state might be inconsistent.")
                    return {
                        'status': 'ERROR',
                        'message': f"Position for {ticker} not found during execution (action: {action}). Inconsistent state?",
                        'order': None
                    }
        
        # Extract position details
        side = position.get('side', 'long')
        position_qty = abs(int(position.get('qty', 0)))
        
        # Determine trade direction and quantity
        trade_action = None
        trade_qty = 0
        
        # Validate position quantity first
        if position_qty <= 0:
            return {
                'status': 'ERROR',
                'message': f"Invalid position quantity ({position_qty}) for {ticker}",
                'order': None
            }
        
        # === STOCK-SPECIFIC ACTIONS ===
        if not is_option:
            # For stocks: use buy/sell for long and short/cover for short
            if action == 'scale_in':
                trade_action = 'buy' if side == 'long' else 'short'
                trade_qty = quantity if quantity is not None else max(1, int(position_qty * self.config.get('scaling_size_pct', 0.30)))
            elif action == 'scale_out':
                trade_action = 'sell' if side == 'long' else 'cover'
                trade_qty = quantity if quantity is not None else max(1, int(position_qty * self.config.get('scaling_size_pct', 0.30)))
                trade_qty = min(trade_qty, position_qty) # Ensure not scaling out more than held
                
                # Additional safety check
                if trade_qty > position_qty:
                    return {
                        'status': 'ERROR',
                        'message': f"Cannot scale out {trade_qty} {ticker} shares - only {position_qty} available",
                        'order': None
                    }
                    
            elif action in ['stop_loss', 'take_profit']:
                trade_action = 'sell' if side == 'long' else 'cover'
                trade_qty = position_qty  # Full position
        # === OPTION-SPECIFIC ACTIONS ===
        else:
            # For options trading, we need to map actions to either 'buy' or 'sell'
            # The mapping depends on both the position side (long/short) and the action type
            
            if action == 'scale_in':
                # Adding to existing position:
                # - For LONG positions: "buy" more
                # - For SHORT positions: "sell" more
                trade_action = 'buy' if side == 'long' else 'sell'
                trade_qty = quantity if quantity is not None else max(1, int(position_qty * self.config.get('scaling_size_pct', 0.30)))
            elif action in ['scale_out', 'stop_loss', 'take_profit']:
                # Reducing or closing position:
                # - For LONG positions: "sell" to close
                # - For SHORT positions: "buy" to close
                trade_action = 'sell' if side == 'long' else 'buy'
                # For complete exit (stop_loss, take_profit), use full position quantity
                if action in ['stop_loss', 'take_profit']:
                    trade_qty = position_qty  # Full position
                else:  # scale_out
                    trade_qty = quantity if quantity is not None else max(1, int(position_qty * self.config.get('scaling_size_pct', 0.30)))
                    trade_qty = min(trade_qty, position_qty)  # Ensure not scaling out more than held
                    
                    # Additional safety check for options
                    if trade_qty > position_qty:
                        return {
                            'status': 'ERROR',
                            'message': f"Cannot scale out {trade_qty} {ticker} contracts - only {position_qty} available",
                            'order': None
                        }
        
        if not trade_action or trade_qty <= 0:
            return {
                'status': 'SKIPPED',
                'message': f"Invalid trade action ({trade_action}) or quantity ({trade_qty}) determined for {ticker}",
                'order': None
            }
        
        # Create decision dictionary for execution
        decision = {
            'action': trade_action,
            'quantity': trade_qty,
            'confidence': confidence,
            'reason': reason
        }
        
        self.logger.info(f"Executing {trade_action} for {trade_qty} {ticker} {'contracts' if is_option else 'shares'}")
        
        # Get position ID for database tracking
        db_position = self.cache.get_active_position(ticker)
        position_id = db_position['id'] if db_position else None
        
        # Record the intended action
        action_id = None
        if position_id:
            action_id = self.cache.record_position_action(
                position_id=position_id,
                action=action,
                ticker=ticker,
                quantity=trade_qty,
                price=float(position.get('current_price', 0)),
                reason=reason,
                confidence=confidence,
                result='pending'
            )
        
        # --- Execute the trade --- 
        try:
            if is_option:
                from src.integrations import AlpacaOptionsTrader
                options_trader = AlpacaOptionsTrader(paper=True) # Assuming paper trading
                
                # Format the option decision for the options trader
                # For options, side doesn't matter in the way we construct the order -
                # the 'action' (buy/sell) is what determines if we're opening or closing
                option_decision_formatted = {
                    'action': trade_action,  # Must be 'buy' or 'sell' only for options
                    'quantity': trade_qty,
                    'confidence': confidence,
                    'reason': reason,
                    'ticker': ticker,
                    'underlying_ticker': position.get('underlying', ''), 
                    'option_type': position.get('option_type', ''),
                    'strike_price': position.get('strike_price', 0),
                    'expiration_date': position.get('expiration_date', '')
                }
                
                self.logger.info(f"Executing option order: {option_decision_formatted}")
                options_to_execute = {ticker: option_decision_formatted}
                
                # Execute the option order through the options trader
                try:
                    result = options_trader.execute_options_decisions(options_to_execute)
                    # Get the result for this specific ticker, and standardize the status values
                    ticker_result = result.get(ticker, {})
                    
                    # Standardize status values
                    status = ticker_result.get('status', '').upper()
                    if status in ['EXECUTED', 'SUCCESS']:
                        status = 'EXECUTED'
                    elif status == 'SKIPPED':
                        status = 'SKIPPED'
                    elif status in ['ERROR', 'FAILED']:
                        status = 'ERROR'
                    else:
                        status = 'PENDING'
                        
                    # Update database records
                    if action_id and status == 'EXECUTED':
                        self.cache.update_action_result(action_id, True, 'success', 
                                                      ticker_result.get('order', {}).get('id'))
                        
                        # Update position tracking
                        if action in ['stop_loss', 'take_profit'] and position_id:
                            close_price = float(position.get('current_price', 0))
                            self.cache.close_position(position_id, action, close_price)
                        elif action == 'scale_in' and position_id:
                            # Update scale count and last scale price
                            current_price = float(position.get('current_price', 0))
                            with self.cache.lock, self.cache.conn:
                                self.cache.conn.execute(
                                    """UPDATE position_tracking 
                                    SET scale_count = scale_count + 1, 
                                        last_scale_price = ?
                                    WHERE id = ?""",
                                    (current_price, position_id)
                                )
                        elif action == 'scale_out' and position_id:
                            # Check if this was a profit level trigger
                            current_price = float(position.get('current_price', 0))
                            pnl_pct = position.get('unrealized_pl_percent', 0)
                            profit_levels = self.cache.get_profit_levels(position_id)
                            for level in profit_levels:
                                if abs(pnl_pct - level['target_pnl_percent']) < 2:  # Within 2% tolerance
                                    self.cache.trigger_profit_level(level['id'], current_price)
                                    break
                    elif action_id:
                        self.cache.update_action_result(action_id, False, status.lower())
                    
                    return {
                        'status': status,
                        'message': ticker_result.get('message', 'Option order executed'),
                        'order': ticker_result.get('order'),
                        'quantity': trade_qty
                    }
                except Exception as option_e:
                    self.logger.error(f"Error executing option order for {ticker}: {option_e}", exc_info=True)
                    # Try to provide helpful error information
                    error_msg = str(option_e)
                    if "Invalid action" in error_msg:
                        error_msg += f". Note: Options require 'buy' or 'sell' actions only. Attempted: '{trade_action}'"
                    
                    return {
                        'status': 'ERROR',
                        'message': error_msg,
                        'order': None
                    }
            else:
                # Use the Alpaca client directly (assuming it's available via self.alpaca)
                decisions_to_execute = {ticker: decision}
                result = self.alpaca.execute_trading_decisions(decisions_to_execute)
                # Get the result for this specific ticker, and standardize the status values
                ticker_result = result.get(ticker, {})
                
                # Standardize status values
                status = ticker_result.get('status', '').upper()
                if status in ['EXECUTED', 'SUCCESS']:
                    status = 'EXECUTED'
                elif status == 'SKIPPED':
                    status = 'SKIPPED'
                elif status in ['ERROR', 'FAILED']:
                    status = 'ERROR'
                else:
                    status = 'PENDING'
                    
                # Update database records
                if action_id and status == 'EXECUTED':
                    self.cache.update_action_result(action_id, True, 'success', 
                                                  ticker_result.get('order', {}).get('id'))
                    
                    # Update position tracking
                    if action in ['stop_loss', 'take_profit'] and position_id:
                        close_price = float(position.get('current_price', 0))
                        self.cache.close_position(position_id, action, close_price)
                    elif action == 'scale_in' and position_id:
                        # Update scale count and last scale price
                        current_price = float(position.get('current_price', 0))
                        with self.cache.lock, self.cache.conn:
                            self.cache.conn.execute(
                                """UPDATE position_tracking 
                                SET scale_count = scale_count + 1, 
                                    last_scale_price = ?
                                WHERE id = ?""",
                                (current_price, position_id)
                            )
                    elif action == 'scale_out' and position_id:
                        # Check if this was a profit level trigger
                        current_price = float(position.get('current_price', 0))
                        pnl_pct = position.get('unrealized_pl_percent', 0)
                        profit_levels = self.cache.get_profit_levels(position_id)
                        for level in profit_levels:
                            if abs(pnl_pct - level['target_pnl_percent']) < 2:  # Within 2% tolerance
                                self.cache.trigger_profit_level(level['id'], current_price)
                                break
                elif action_id:
                    self.cache.update_action_result(action_id, False, status.lower())
                
                return {
                    'status': status,
                    'message': ticker_result.get('message', 'Stock order executed'),
                    'order': ticker_result.get('order'),
                    'quantity': trade_qty
                }
        except Exception as e:
            self.logger.error(f"Error executing {trade_action} for {ticker}: {e}", exc_info=True)
            return {
                'status': 'ERROR',
                'message': f"Exception during execution: {str(e)}",
                'order': None
            }
            
    def _execute_adjustment_action(self, ticker: str, action: str, price_target: Optional[float], reason: str) -> Dict[str, Any]:
        """
        Execute an adjustment action (adjust_stop, adjust_target).
        
        Args:
            ticker: The ticker symbol
            action: Action to take
            price_target: Target price
            reason: Reason for the action
            
        Returns:
            Dictionary with execution result
        """
        if not price_target:
            return {
                'status': 'ERROR',
                'message': f"No price target provided for {action}",
                'details': None
            }
            
        # Update database tracking
        db_position = self.cache.get_active_position(ticker)
        if db_position:
            position_id = db_position['id']
            
            if action == 'adjust_target':
                self.cache.update_position_stops(
                    position_id=position_id,
                    take_profit=price_target
                )
            
            # Record the action
            action_id = self.cache.record_position_action(
                position_id=position_id,
                action=action,
                ticker=ticker,
                price=price_target,
                reason=reason,
                result='success'
            )
            
            self.cache.update_action_result(action_id, True, 'success')
        
        return {
            'status': 'SUCCESS',
            'message': f"{action} set to {price_target} for {ticker}: {reason}",
            'details': {
                'ticker': ticker,
                'action': action,
                'price_target': price_target
            }
        }
    
    def get_position_summary(self, ticker: str = None) -> Dict[str, Any]:
        """
        Get enhanced position summary including history and performance metrics.
        
        Args:
            ticker: Optional ticker to filter by
            
        Returns:
            Dictionary with position summaries
        """
        positions = self.cache.get_position_history(ticker, days=30)
        
        summary = {
            'active_positions': [],
            'closed_positions': [],
            'total_pnl': 0,
            'win_rate': 0,
            'average_win': 0,
            'average_loss': 0,
            'best_trade': None,
            'worst_trade': None
        }
        
        wins = []
        losses = []
        
        for pos in positions:
            if pos['status'] == 'active':
                summary['active_positions'].append({
                    'ticker': pos['ticker'],
                    'entry_price': pos['entry_price'],
                    'current_price': pos['current_price'],
                    'quantity': pos['quantity'],
                    'pnl_percent': ((pos['current_price'] / pos['entry_price']) - 1) * 100 if pos['side'] == 'long' else ((pos['entry_price'] / pos['current_price']) - 1) * 100,
                    'scale_count': pos['scale_count'],
                    'trailing_stop_active': pos['trailing_stop_active']
                })
            else:
                summary['closed_positions'].append({
                    'ticker': pos['ticker'],
                    'pnl': pos['pnl'],
                    'pnl_percent': pos['pnl_percent'],
                    'close_reason': pos['close_reason'],
                    'close_timestamp': pos['close_timestamp']
                })
                
                summary['total_pnl'] += pos['pnl'] or 0
                
                if pos['pnl'] and pos['pnl'] > 0:
                    wins.append(pos)
                elif pos['pnl'] and pos['pnl'] < 0:
                    losses.append(pos)
        
        # Calculate statistics
        if wins or losses:
            summary['win_rate'] = len(wins) / (len(wins) + len(losses)) * 100
            
        if wins:
            summary['average_win'] = sum(w['pnl'] for w in wins) / len(wins)
            summary['best_trade'] = max(wins, key=lambda x: x['pnl'])
            
        if losses:
            summary['average_loss'] = sum(l['pnl'] for l in losses) / len(losses)
            summary['worst_trade'] = min(losses, key=lambda x: x['pnl'])
        
        return summary 