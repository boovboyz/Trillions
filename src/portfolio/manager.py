"""
Portfolio manager for Alpaca-based trading integration.
Handles portfolio tracking, position sizing, risk management, 
and execution of trading decisions.
"""

import json
import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_core.prompts import ChatPromptTemplate

from src.integrations.alpaca import get_alpaca_client, safe_json_dumps
from src.utils.llm import call_llm
import logging
from datetime import datetime
import re
from src.agents.portfolio_sizer import PortfolioSizerAgent  # Import the new agent
from src.integrations.polygon import PolygonClient # Assuming this exists

logger = logging.getLogger(__name__) # Add logger instance

# --- Define Pydantic Models ---
class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")

class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions (action, confidence, reasoning - NO quantity)")
# --- End Pydantic Models ---


class PortfolioManager:
    """Manages a portfolio through Alpaca, handling position sizing and risk management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the portfolio manager.
        
        Args:
            config: Optional configuration dictionary with risk parameters.
        """
        self.alpaca = get_alpaca_client(paper=True)
        # Assuming Polygon client is needed by sizer and can be instantiated here
        # This might need API keys from config or environment
        self.polygon = PolygonClient() 
        
        # Default configuration
        self.config = {
            'max_position_size_pct': 0.33,              # Max position size as percentage of portfolio
            'max_single_order_size_pct': 0.10,          # Max single order as percentage of portfolio
            'stop_loss_pct': 0.05,                      # Default stop loss percentage
            'profit_target_pct': 0.15,                  # Default profit target percentage
            'position_scaling': True,                   # Whether to scale positions
            'scaling_threshold_pct': 0.07,              # When to scale in/out (profit/loss percentage)
            'scaling_size_pct': 0.30,                   # How much to scale by (% of current position)
            'max_drawdown_pct': 0.15,                   # Maximum allowed drawdown before reducing risk
            'cash_reserve_pct': 0.10,                   # Minimum cash reserve as percentage of portfolio
            'max_concentration_pct': 0.30,              # Maximum sector/industry concentration
            'volatility_adjustment': True,              # Whether to adjust position size based on volatility
            'max_trades_per_day': 10,                   # Maximum number of trades per day
            'market_hours_only': True,                  # Only trade during market hours
            'enable_shorts': True,                      # Whether to allow short positions
            'max_short_position_size_pct': 0.20,        # Max short position size (% of portfolio)
            'short_stop_loss_pct': 0.06,                # Stop loss for short positions (higher to be cautious)
            'short_profit_target_pct': 0.12,            # Profit target for short positions
            'enable_trailing_stops': True,              # NEW: Enable ATR trailing stops
            'trailing_stop_atr_multiplier': 2.5,       # NEW: ATR multiplier for trailing stops
            'atr_lookback_period': 14,                  # NEW: Lookback period for ATR calculation
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
        
        self._last_update = None  # Timestamp of last update
        self.update_interval = 30  # Update portfolio status every 30 seconds
        
        # Cache for market data
        self.market_data_cache = {}
        
        # Cache for portfolio data
        self.portfolio_cache = {
            'account': None,
            'positions': None,
            'orders': None,
            'last_update': None
        }
        
        # Instantiate the Portfolio Sizer Agent
        # It needs config, alpaca client, polygon client, and portfolio cache
        self.portfolio_sizer_agent = PortfolioSizerAgent(
            config=self.config,
            alpaca_client=self.alpaca,
            polygon_client=self.polygon,
            portfolio_cache=self.portfolio_cache # Pass the cache reference
        )
        logger.info("PortfolioManager initialized with PortfolioSizerAgent.")

    def update_portfolio_cache(self, force: bool = False) -> None:
        """
        Update the portfolio cache if the last update was older than the update interval.
        
        Args:
            force: Force update regardless of last update time.
        """
        now = datetime.now()
        
        if (not self.portfolio_cache['last_update'] or
            force or
            (now - self.portfolio_cache['last_update']).total_seconds() > self.update_interval):
            
            # Add a small delay to allow Alpaca's systems to update
            if force:
                time.sleep(0.5)  # 500ms additional delay when forcing an update
            
            # Clear the cache before fetching fresh data
            self.market_data_cache = {}  # Reset market data cache
            
            # Fetch fresh data from Alpaca
            self.portfolio_cache['account'] = self.alpaca.get_account()
            self.portfolio_cache['positions'] = self.alpaca.get_positions()
            
            # Get all orders to ensure we capture recent orders
            self.portfolio_cache['orders'] = self.alpaca.get_orders(status='all')
            
            self.portfolio_cache['last_update'] = now
            
    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current state of the portfolio.
        
        Returns:
            Dict containing portfolio state including positions, cash, and performance metrics.
        """
        self.update_portfolio_cache()
        
        account = self.portfolio_cache['account']
        positions = self.portfolio_cache['positions']
        
        # Calculate total exposure (sum of absolute position values)
        total_exposure = sum(abs(float(p['market_value'])) for p in positions)
        
        # Calculate portfolio metrics
        portfolio_value = float(account['portfolio_value'])
        cash_value = float(account['cash'])
        position_value = portfolio_value - cash_value
        cash_ratio = cash_value / portfolio_value if portfolio_value > 0 else 0
        
        # Get daily and weekly P&L data
        daily_pnl = self._calculate_daily_pnl()
        weekly_pnl = self._calculate_weekly_pnl()
        
        # Get positions with additional risk metrics
        enhanced_positions_list = self._enhance_positions(positions)

        # Convert enhanced positions list to a dict keyed by symbol
        enhanced_positions_dict = {p['symbol']: p for p in enhanced_positions_list}

        # Calculate portfolio concentration
        concentration = self._calculate_concentration(enhanced_positions_list) # Pass list here
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': portfolio_value,
            'cash': cash_value,
            'position_value': position_value,
            'cash_ratio': cash_ratio,
            'buying_power': float(account['buying_power']),
            'margin_used': portfolio_value - cash_value - float(account['equity']),
            'day_trades_remaining': 3 - account['daytrade_count'],
            'total_exposure': total_exposure,
            'exposure_ratio': total_exposure / portfolio_value if portfolio_value > 0 else 0,
            'positions': enhanced_positions_dict, # Return the dictionary here
            'concentration': concentration,
            'daily_pnl': daily_pnl,
            'weekly_pnl': weekly_pnl,
            'orders': self._get_recent_orders(),
        }
    
    def _calculate_daily_pnl(self) -> Dict[str, Any]:
        """
        Calculate the daily P&L metrics.
        
        Returns:
            Dict with daily P&L metrics.
        """
        try:
            # Get the portfolio history for today
            history = self.alpaca.get_portfolio_history(period='1D', timeframe='1H')
            
            # Calculate daily metrics
            daily_change = history['equity'][-1] - history['equity'][0] if history['equity'] else 0
            daily_pct = (daily_change / history['equity'][0]) * 100 if history['equity'] and history['equity'][0] > 0 else 0
            
            # Calculate realized P&L separately
            realized_pnl = self.alpaca.get_realized_pnl(timeframe='1D')
            
            return {
                'change': daily_change,
                'change_pct': daily_pct,
                'realized': realized_pnl,
                'unrealized': daily_change - realized_pnl
            }
        except Exception as e:
            print(f"Error calculating daily P&L: {e}")
            return {
                'change': 0,
                'change_pct': 0,
                'realized': 0,
                'unrealized': 0
            }
    
    def _calculate_weekly_pnl(self) -> Dict[str, Any]:
        """
        Calculate the weekly P&L metrics.
        
        Returns:
            Dict with weekly P&L metrics.
        """
        try:
            # Get the portfolio history for the week
            history = self.alpaca.get_portfolio_history(period='1W', timeframe='1D')
            
            # Calculate weekly metrics
            weekly_change = history['equity'][-1] - history['equity'][0] if history['equity'] else 0
            weekly_pct = (weekly_change / history['equity'][0]) * 100 if history['equity'] and history['equity'][0] > 0 else 0
            
            # Calculate realized P&L separately
            realized_pnl = self.alpaca.get_realized_pnl(timeframe='1W')
            
            return {
                'change': weekly_change,
                'change_pct': weekly_pct,
                'realized': realized_pnl,
                'unrealized': weekly_change - realized_pnl
            }
        except Exception as e:
            print(f"Error calculating weekly P&L: {e}")
            return {
                'change': 0,
                'change_pct': 0,
                'realized': 0,
                'unrealized': 0
            }
    
    def _enhance_positions(self, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance position data with additional risk metrics.
        
        Args:
            positions: List of positions from Alpaca API.
            
        Returns:
            List of enhanced positions with additional risk metrics.
        """
        enhanced = []
        
        for position in positions:
            symbol = position['symbol']

            # --- Check if it's an options symbol --- 
            # Basic OCC format check: Root(1-6 chars) + YYMMDD + C/P + Strike(8 digits)
            is_option = bool(re.match(r"^[A-Z]{1,6}(\d{6})([CP])(\d{8})$", symbol))
            # ----------------------------------------

            entry_price = float(position['avg_entry_price'])
            current_price = float(position['current_price'])
            quantity = int(position['qty'])
            side = position['side']
            
            # Calculate risk metrics
            profit_loss_pct = ((current_price / entry_price) - 1) * 100 if side == 'long' else ((entry_price / current_price) - 1) * 100
            
            # Calculate stop loss and take profit levels using position-specific parameters
            if side == 'long':
                stop_price = entry_price * (1 - self.config['stop_loss_pct'])
                target_price = entry_price * (1 + self.config['profit_target_pct'])
            else:  # short position
                stop_loss_pct = self.config.get('short_stop_loss_pct', self.config['stop_loss_pct'])
                profit_target_pct = self.config.get('short_profit_target_pct', self.config['profit_target_pct'])
                stop_price = entry_price * (1 + stop_loss_pct)
                target_price = entry_price * (1 - profit_target_pct)
            
            # Calculate distance to stop and target as percentage
            distance_to_stop_pct = abs((current_price - stop_price) / current_price) * 100
            distance_to_target_pct = abs((target_price - current_price) / current_price) * 100
            
            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_price)
            reward = abs(target_price - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Get position weight in portfolio
            account = self.portfolio_cache['account']
            portfolio_value = float(account['portfolio_value'])
            position_value = float(position['market_value'])
            weight = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            # --- Skip stock-specific data for options --- 
            volatility = 0.0
            market_data = {}
            if not is_option:
                volatility = self._calculate_volatility(symbol)
                market_data = self._get_market_data(symbol)
            else:
                 # Option-specific data might be fetched here if needed in the future
                 # For now, use defaults or data already in 'position'
                 market_data = {
                     'price': current_price, # Use price from position data
                     'volume': 0,
                     'high': current_price,
                     'low': current_price,
                     'open': current_price,
                     'close': current_price,
                     'vwap': current_price,
                     'timestamp': datetime.now().isoformat()
                 }
            # -------------------------------------------
            
            # Determine if scaling is appropriate
            scale_in = profit_loss_pct < -self.config['scaling_threshold_pct'] * 100 if side == 'long' else profit_loss_pct > self.config['scaling_threshold_pct'] * 100
            scale_out = profit_loss_pct > self.config['scaling_threshold_pct'] * 100 if side == 'long' else profit_loss_pct < -self.config['scaling_threshold_pct'] * 100
            
            enhanced_position = {
                **position,
                'profit_loss_pct': profit_loss_pct,
                'stop_price': stop_price,
                'target_price': target_price,
                'distance_to_stop_pct': distance_to_stop_pct,
                'distance_to_target_pct': distance_to_target_pct,
                'risk_reward_ratio': risk_reward_ratio,
                'weight': weight,
                'volatility': volatility,
                'scale_in': scale_in,
                'scale_out': scale_out,
                'market_data': market_data
            }
            
            enhanced.append(enhanced_position)
        
        return enhanced
    
    def _calculate_volatility(self, symbol: str, lookback_days: int = 20) -> float:
        """
        Calculate historical volatility for a symbol.
        
        Args:
            symbol: Ticker symbol.
            lookback_days: Number of days to use for volatility calculation.
            
        Returns:
            Annualized volatility as a percentage.
        """
        try:
            # Get daily bar data
            bars = self.alpaca.get_market_data(symbol, timeframe='1D', limit=lookback_days + 1) # Need +1 for pct_change
            if not bars or len(bars) <= 1:
                return 0.0
            
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', utc=True)
            df = df.set_index('timestamp')
            
            # Calculate daily returns
            df['return'] = df['close'].pct_change()
            
            # Calculate standard deviation of returns (ignoring first NaN)
            daily_std = df['return'].iloc[1:].std()
            
            # Annualize volatility (approx 252 trading days per year)
            annualized_vol = daily_std * math.sqrt(252)
            
            return annualized_vol * 100  # Convert to percentage
        except Exception as e:
            print(f"Error calculating volatility for {symbol}: {e}")
            return 0.0

    def _calculate_atr(self, symbol: str, lookback_period: Optional[int] = None) -> float:
        """
        Calculate the Average True Range (ATR) for a symbol.

        Args:
            symbol: Ticker symbol.
            lookback_period: Number of periods (days) for ATR. Defaults to config['atr_lookback_period'].

        Returns:
            The current ATR value, or 0.0 if calculation fails.
        """
        if lookback_period is None:
            lookback_period = self.config.get('atr_lookback_period', 14)
        
        try:
            # Need lookback_period + 1 bars to calculate the first TR
            # Need additional bars for the initial SMA calculation if using Wilder's smoothing
            # Fetching more bars simplifies things (e.g., 2*lookback_period)
            bars_df = self.alpaca.get_market_data_df(symbol, timeframe='1D', limit=lookback_period * 2)
            if bars_df is None or len(bars_df) < lookback_period + 1:
                print(f"Insufficient data to calculate ATR({lookback_period}) for {symbol}")
                return 0.0

            # Calculate True Range (TR)
            high_low = bars_df['high'] - bars_df['low']
            high_close = np.abs(bars_df['high'] - bars_df['close'].shift())
            low_close = np.abs(bars_df['low'] - bars_df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # Calculate ATR using simple moving average for this example
            # For Wilder's smoothing (common): atr = tr.ewm(alpha=1/lookback_period, adjust=False).mean()
            atr = tr.rolling(window=lookback_period).mean()
            
            current_atr = atr.iloc[-1]
            return current_atr if pd.notna(current_atr) else 0.0
        
        except Exception as e:
            print(f"Error calculating ATR for {symbol}: {e}")
            return 0.0

    def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a symbol, with caching.
        
        Args:
            symbol: Ticker symbol.
            
        Returns:
            Dict with market data.
        """
        # Check if we have recent data in cache
        now = datetime.now()
        if (symbol in self.market_data_cache and 
            (now - self.market_data_cache[symbol]['timestamp']).total_seconds() < 300):  # Cache for 5 minutes
            return self.market_data_cache[symbol]['data']
        
        try:
            # Get the latest trade
            latest_trade = self.alpaca.get_latest_trade(symbol)
            
            # Get volume and other metrics from daily bar
            bars = self.alpaca.get_market_data(symbol, timeframe='1D', limit=20)
            
            if bars:
                daily_bar = bars[-1]
            else:
                daily_bar = {}
                
            # Create market data object
            market_data = {
                'price': latest_trade['price'],
                'volume': daily_bar.get('volume', 0),
                'high': daily_bar.get('high', latest_trade['price']),
                'low': daily_bar.get('low', latest_trade['price']),
                'open': daily_bar.get('open', latest_trade['price']),
               'close': daily_bar.get('close', latest_trade['price']),
                'vwap': daily_bar.get('vwap', latest_trade['price']),
                'timestamp': now.isoformat()
            }
            
            # Cache the data
            self.market_data_cache[symbol] = {
                'data': market_data,
                'timestamp': now
            }
            
            return market_data
        except Exception as e:
            print(f"Error getting market data for {symbol}: {e}")
            return {
                'price': 0,
                'volume': 0,
                'high': 0,
                'low': 0,
                'open': 0,
                'close': 0,
                'vwap': 0,
                'timestamp': now.isoformat()
            }
    
    def _calculate_concentration(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate the concentration of the portfolio by sector, industry, etc.
        
        Args:
            positions: List of enhanced positions.
            
        Returns:
            Dict with concentration metrics.
        """
        # In a real implementation, this would categorize positions by sector/industry
        # For this implementation, we'll just calculate concentration by symbol
        
        total_value = sum(abs(float(p['market_value'])) for p in positions)
        
        if total_value == 0:
            return {
                'by_symbol': {},
                'max_symbol': None,
                'max_concentration': 0
            }
        
        # Calculate concentration by symbol
        concentration = {}
        for position in positions:
            symbol = position['symbol']
            value = abs(float(position['market_value']))
            concentration[symbol] = (value / total_value) * 100
        
        # Find the maximum concentration
        max_symbol = max(concentration.items(), key=lambda x: x[1], default=(None, 0))
        
        return {
            'by_symbol': concentration,
            'max_symbol': max_symbol[0],
            'max_concentration': max_symbol[1]
        }
    
    def _get_recent_orders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent orders, including pending and filled.
        
        Args:
            limit: Maximum number of orders to return.
            
        Returns:
            List of recent orders.
        """
        all_orders = self.portfolio_cache['orders']
        
        if not all_orders:
            return [] # Return empty list if no orders

        # Sort by updated_at in descending order
        # Use the timestamp object directly as the key, assuming it's comparable
        try:
            sorted_orders = sorted(
                all_orders,
                key=lambda o: o['updated_at'],
                reverse=True
            )
        except TypeError as e:
            # Fallback: If direct comparison fails (e.g., mixed types or None),
            # try converting to string and hope for alphanumeric sort (less ideal)
            print(f"Warning: Direct sorting of order timestamps failed ({e}). Falling back to string sort.")
            try:
                sorted_orders = sorted(
                    all_orders,
                    key=lambda o: str(o.get('updated_at', '')) if o else '', # Handle None orders/timestamps
                    reverse=True
                )
            except Exception as final_e:
                 print(f"Error: Final fallback sorting failed: {final_e}")
                 return all_orders[:limit] # Return unsorted slice on complete failure
        except Exception as e:
             print(f"Error sorting orders: {e}")
             return all_orders[:limit] # Return unsorted slice on other errors

        return sorted_orders[:limit]
    
    def execute_decision(self, decisions: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Execute trading decisions with risk management.
        Uses _generate_ai_trading_decisions if needed (or assumes decisions are already processed)

        Args:
            decisions: Dict mapping ticker symbols to trading decisions.

        Returns:
            Dict with execution results.
        """
        # NOTE: This function might now directly receive the structured decisions
        # from an AI synthesis step, or it might need to call a generation step.
        # For now, we assume 'decisions' are ready for risk management and sizing.

        # Update portfolio state
        self.update_portfolio_cache(force=True)
        portfolio_state = self.get_portfolio_state()

        # Check if market is open if configured to trade only during market hours
        if self.config['market_hours_only']:
            clock = self.alpaca.get_clock()
            if not clock['is_open']:
                return {
                    symbol: {
                        'status': 'skipped',
                        'message': 'Market is closed',
                        'order': None
                    }
                    for symbol in decisions.keys()
                }

        # Apply risk management filters
        filtered_decisions = self._apply_risk_management(decisions, portfolio_state)

        # Calculate position sizes
        sized_decisions = self._calculate_position_sizes(filtered_decisions, portfolio_state)

        # Execute the orders
        execution_results = self.alpaca.execute_trading_decisions(sized_decisions)

        return execution_results

    # --- Add AI Decision Generation Method ---
    def _generate_ai_trading_decisions(
        self,
        tickers: list[str],
        analyst_signals: dict, # Assuming format {agent_name: {ticker: {signal:..., confidence:...}}}
        portfolio_state: dict,
        model_name: str,
        model_provider: str,
    ) -> PortfolioManagerOutput:
        """Generates final trading decisions by synthesizing analyst signals and portfolio context using an LLM."""

        # Extract signals, excluding the risk agent's output which is now handled separately
        signals_by_ticker = {}
        risk_context = {} # Initialize risk context
        for agent, signals in analyst_signals.items():
            if agent == "risk_management_agent":
                 # Extract the risk context generated by the risk agent
                 risk_context = signals # Assuming the entire signal dict is the context
                 continue # Don't include risk agent in ticker_signals

            for ticker, signal_data in signals.items():
                 if ticker not in signals_by_ticker:
                     signals_by_ticker[ticker] = {}
                 signals_by_ticker[ticker][agent] = {
                     "signal": signal_data.get("signal", "hold"),
                     "confidence": signal_data.get("confidence", 0)
                 }

        # Get current prices (can be redundant if risk_context provides it, but useful for clarity)
        current_prices = {}
        for ticker in tickers:
             # Prioritize price from risk context if available
             ticker_risk_context = risk_context.get(ticker, {})
             price = ticker_risk_context.get("current_price")
             if price is None or price <= 0:
                 # Fallback to portfolio state or market data
                 position_data = portfolio_state.get('positions', {}).get(ticker)
                 if position_data:
                     price = position_data.get('current_price', 0)
                 if price is None or price <= 0:
                     try:
                         market_data = self._get_market_data(ticker)
                         price = market_data.get('price', 0)
                     except Exception:
                         price = 0 # Final fallback
             current_prices[ticker] = price


        # Create the prompt template
        template = ChatPromptTemplate.from_messages(
            [
                (
                  "system",
                  """You are a sophisticated portfolio manager synthesizing analyst signals into final trading decisions.
                  Your goal is to maximize returns while adhering to strict risk management rules provided in the `risk_context`.

                  Decision Process:
                  1. Analyze the signals from various analysts (`signals_by_ticker`). Note their confidence levels.
                  2. CRITICAL: Evaluate the `risk_context` provided for each ticker and the overall portfolio. This context overrides general rules.
                     - `max_potential_buy_value`: The maximum dollar amount you can allocate to a NEW BUY order for this ticker, considering cash reserves and position limits.
                     - `max_potential_short_value`: The maximum dollar amount you can allocate to a NEW SHORT order for this ticker, considering margin and short limits.
                     - `portfolio_context`: Flags indicating overall portfolio health (e.g., `max_drawdown_reached`, `max_trades_reached`, `cash_below_reserve`, `day_trades_remaining`).
                  3. Evaluate the current portfolio state (`portfolio_positions`, `portfolio_cash`).
                  4. Synthesize all information to determine the optimal action (buy, sell, short, cover, hold) for each ticker.
                  5. Propose a *quantity* reflecting conviction, scaled by confidence, BUT STRICTLY LIMITED BY:
                     - The calculated `max_potential_buy_value` or `max_potential_short_value` from `risk_context`. Convert value to shares using `current_prices`.
                     - Existing position sizes for sell/cover actions (cannot sell/cover more than held).
                     - Overall portfolio constraints from `portfolio_context` (e.g., NO new buy/short orders if `max_drawdown_reached` or `cash_below_reserve` is true).
                     - Available day trades (`day_trades_remaining`) if the action constitutes a day trade (opening and closing same day). Prioritize covering shorts even if it uses the last day trade.
                  6. Provide clear reasoning, referencing specific signals and risk context constraints.

                  Trading Rules Summary (Refer to Risk Context First!):
                  - Buys: Require available cash (check `cash_below_reserve`=false), respect `max_potential_buy_value`. Blocked if `max_drawdown_reached`.
                  - Sells: Require existing long position. Quantity <= held shares.
                  - Shorts: Require shorting enabled (`enable_shorts`=true), available margin (implied by `max_potential_short_value` > 0), strong bearish signals. Blocked if `max_drawdown_reached`.
                  - Covers: Require existing short position. Quantity <= held shares. Generally allowed even if day trade limits are hit.
                  - Hold: Default action if no strong signal or if blocked by risk context.

                  Available Actions: "buy", "sell", "short", "cover", "hold"

                  Inputs Provided in Human Message:
                  - `signals_by_ticker`: Dictionary of ticker -> {{agent -> {{signal, confidence}}}}.
                  - `risk_context`: Dictionary of ticker -> {{risk parameters and limits}}. **THIS IS YOUR PRIMARY GUIDE FOR SIZING AND ALLOWED ACTIONS.**
                  - `current_prices`: Dictionary of ticker -> current price (for converting value limits to shares).
                  - `portfolio_cash`: Current cash available.
                  - `portfolio_positions`: Dictionary of ticker -> position details (e.g., quantity, side).
                  """
                ),
                (
                  "human",
                  """Synthesize analyst signals and make trading decisions adhering strictly to the provided risk context.

                  Analyst Signals by Ticker:
                  ```json
                  {signals_by_ticker}
                  ```

                  Risk Context (Max Values, Config, Portfolio Flags):
                  ```json
                  {risk_context_json}
                  ```

                  Current Prices:
                  ```json
                  {current_prices_json}
                  ```

                  Portfolio Cash: {portfolio_cash:.2f}
                  Current Positions:
                  ```json
                  {portfolio_positions_json}
                  ```

                  Output strictly in JSON with the following structure:
                  {{\n                    "decisions": {{\n                      "TICKER1": {{\n                        "action": "buy/sell/short/cover/hold",\n                        "confidence": float between 0 and 100,\n                        "reasoning": "string (MUST mention risk context constraints if they influenced the decision)"\n                      }},\n                      "TICKER2": {{\n                        ...\n                      }},\n                      ...\n                    }}\n                  }}\n                  """
                ),
            ]
        )

        # Safely dump data for the prompt
        try:
            # Use safe_json_dumps for potentially complex nested structures
            signals_json = safe_json_dumps(signals_by_ticker, indent=2)
            risk_context_json = safe_json_dumps(risk_context, indent=2)
            current_prices_json = safe_json_dumps(current_prices, indent=2)
            # Simplify portfolio positions for the prompt - focus on qty and side
            simple_positions = {
                 ticker: {
                     "quantity": pos.get("qty"),
                     "side": pos.get("side")
                 } for ticker, pos in portfolio_state.get('positions', {}).items()
            }
            portfolio_positions_json = safe_json_dumps(simple_positions, indent=2)

        except Exception as e:
            logging.error(f"Error serializing data for AI prompt: {e}")
            # Provide empty structures on error to avoid breaking the prompt format
            signals_json = "{}"
            risk_context_json = "{}"
            current_prices_json = "{}"
            portfolio_positions_json = "{}"


        # Generate the prompt
        prompt = template.invoke(
            {
                "signals_by_ticker": signals_json,
                "risk_context_json": risk_context_json,
                "current_prices_json": current_prices_json,
                "portfolio_cash": portfolio_state.get('cash', 0),
                "portfolio_positions_json": portfolio_positions_json,
                # Removed outdated variables like max_shares, margin_requirement, total_margin_used
                # as they are now encapsulated within risk_context
            }
        )

        # Create default factory for PortfolioManagerOutput
        def create_default_portfolio_output():
            return PortfolioManagerOutput(decisions={ticker: PortfolioDecision(action="hold", confidence=0.0, reasoning="Error in portfolio management AI or risk constraints prevent action, defaulting to hold") for ticker in tickers}) # Updated default reason

        # Call the LLM
        # Note: Assumes 'call_llm' utility exists and handles Pydantic parsing/retries
        return call_llm(prompt=prompt, model_name=model_name, model_provider=model_provider, pydantic_model=PortfolioManagerOutput, agent_name="portfolio_management_synthesis", default_factory=create_default_portfolio_output) # Renamed agent_name for clarity
    # --- End AI Decision Generation Method ---

    def _apply_risk_management(
        self, 
        decisions: Dict[str, Dict], # Decisions now only have action, confidence, reasoning
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """
        Apply risk management filters to trading decisions (action only).
        This runs BEFORE sizing. It can change the action to 'hold'.
        
        Args:
            decisions: Dict mapping ticker symbols to trading decisions (NO quantity).
            portfolio_state: Current portfolio state.
            
        Returns:
            Dict with filtered trading decisions (action may be changed to 'hold').
        """
        filtered_decisions = {}
        
        # Get portfolio value and cash
        portfolio_value = portfolio_state['portfolio_value']
        cash = portfolio_state['cash']
        positions = portfolio_state.get('positions', {}) # Use .get for safety
        
        # Check overall portfolio risk
        max_drawdown_reached = self._check_drawdown_threshold(portfolio_state)
        day_trades_available = portfolio_state['day_trades_remaining'] > 0
        max_trades_reached = len(self._get_today_trades()) >= self.config['max_trades_per_day']
        cash_reserve_pct = self.config['cash_reserve_pct']
        
        # Apply overall portfolio risk rules
        for symbol, decision in decisions.items():
            current_decision = decision.copy() # Work on a copy
            action = current_decision.get('action', 'hold').lower()
            # confidence = float(current_decision.get('confidence', 0)) # Confidence not used for filtering action

            # Skip if already hold
            if action == 'hold':
                filtered_decisions[symbol] = current_decision
                continue
                
            # Check if we've reached maximum drawdown threshold
            if max_drawdown_reached and action in ['buy', 'short']:
                self.logger.warning(f"Risk Filter: Setting {symbol} action to 'hold' due to max drawdown.")
                current_decision['original_action'] = action
                current_decision['action'] = 'hold'
                current_decision['skip_reason'] = 'Max drawdown threshold reached'
                filtered_decisions[symbol] = current_decision
                continue
                
            # Check if we've reached maximum trades per day
            if max_trades_reached and action in ['buy', 'short']:
                self.logger.warning(f"Risk Filter: Setting {symbol} action to 'hold' due to max trades per day.")
                current_decision['original_action'] = action
                current_decision['action'] = 'hold'
                current_decision['skip_reason'] = 'Maximum trades per day reached'
                filtered_decisions[symbol] = current_decision
                continue
                
            # Check if we have day trades available (for same-day round trips)
            position_exists = symbol in positions
            is_day_trade = False
            if position_exists:
                today_orders = self._get_today_trades()
                symbol_today_orders = [
                    order for order in today_orders 
                    if order.get('symbol') == symbol and order.get('status') == 'filled'
                ]
                if symbol_today_orders and (
                    (action == 'sell' and positions[symbol]['side'] == 'long') or
                    (action == 'cover' and positions[symbol]['side'] == 'short')
                ):
                    is_day_trade = True
            
            if is_day_trade and not day_trades_available and action != 'cover':
                self.logger.warning(f"Risk Filter: Setting {symbol} {action} to 'hold' due to no day trades available.")
                current_decision['original_action'] = action
                current_decision['action'] = 'hold'
                current_decision['skip_reason'] = 'No day trades available'
                filtered_decisions[symbol] = current_decision
                continue
            
            if is_day_trade and not day_trades_available and action == 'cover':
                self.logger.warning(f"Risk Filter: Allowing {symbol} {action} despite it being a day trade with no day trades available (prioritizing cover)")
            
            # Check if we have enough cash (minus reserve) - ONLY for BUY actions
            if action == 'buy':
                available_cash = cash - (portfolio_value * cash_reserve_pct)
                if available_cash <= 0:
                    self.logger.warning(f"Risk Filter: Setting {symbol} action to 'hold' due to insufficient cash (below reserve). Available: {available_cash:.2f}")
                    current_decision['original_action'] = action
                    current_decision['action'] = 'hold'
                    current_decision['skip_reason'] = 'Insufficient cash (below reserve)'
                    filtered_decisions[symbol] = current_decision
                    continue
            
            # Check short selling configuration
            if action == 'short' and not self.config.get('enable_shorts', False):
                 self.logger.warning(f"Risk Filter: Setting {symbol} action to 'hold' because short selling is disabled.")
                 current_decision['original_action'] = action
                 current_decision['action'] = 'hold'
                 current_decision['skip_reason'] = 'Short selling disabled'
                 filtered_decisions[symbol] = current_decision
                 continue
            
            # If we passed all filters, keep the decision (potentially modified action)
            filtered_decisions[symbol] = current_decision
            
        return filtered_decisions
    
    def _check_drawdown_threshold(self, portfolio_state: Dict[str, Any]) -> bool:
        """
        Check if portfolio drawdown has exceeded the configured threshold.
        
        Args:
            portfolio_state: Current portfolio state.
            
        Returns:
            True if drawdown threshold is exceeded, False otherwise.
        """
        # Get weekly P&L data
        weekly_pnl = portfolio_state['weekly_pnl']
        
        # Check if weekly drawdown exceeds threshold
        if weekly_pnl['change_pct'] < -self.config['max_drawdown_pct'] * 100:
            return True
            
        return False
    
    def _get_today_trades(self) -> List[Dict[str, Any]]:
        """
        Get trades from today.
        
        Returns:
            List of today's trades.
        """
        all_orders = self.portfolio_cache['orders']
        
        # Filter orders from today
        today = datetime.now().date()
        today_orders = [
            order for order in all_orders
            if (order['filled_at'] and
                datetime.fromisoformat(order['filled_at'].replace('Z', '+00:00')).date() == today)
        ]
        
        return today_orders
    
    def manage_positions(self) -> Dict[str, Dict]:
        """
        Manage existing positions (adjust stops, scale in/out, etc.).
        
        Returns:
            Dict with management actions taken.
        """
        # Update portfolio state
        self.update_portfolio_cache(force=True)
        portfolio_state = self.get_portfolio_state()
        
        management_actions = {}
        
        # Only proceed if we have positions
        # portfolio_state['positions'] is now a dict {symbol: position_dict}
        if not portfolio_state.get('positions'):
            return management_actions
            
        # Check scaling conditions for each position dictionary (iterate over values)
        for position in portfolio_state['positions'].values(): 
            symbol = position['symbol']
            side = position['side']
            scale_in = position.get('scale_in', False)
            scale_out = position.get('scale_out', False)
            
            # Skip if neither scaling condition is met
            if not scale_in and not scale_out:
                continue
                
            # Calculate scaling size (percentage of current position)
            qty = int(position['qty'])
            scaling_size = max(1, int(abs(qty) * self.config['scaling_size_pct']))
            
            if scale_in:
                # We're losing money, consider adding to position if conditions are right
                
                # Verify risk management
                if self._check_drawdown_threshold(portfolio_state):
                    # Skip scaling in if we're in drawdown
                    management_actions[symbol] = {
                        'action': 'skip_scale_in',
                        'reason': 'Drawdown threshold reached'
                    }
                    continue
                    
                # Check position already at max size
                position_pct = float(position['weight'])
                if position_pct >= self.config['max_position_size_pct'] * 100:
                    management_actions[symbol] = {
                        'action': 'skip_scale_in',
                        'reason': 'Position already at maximum size'
                    }
                    continue
                
                # Create scaling decision
                if side == 'long':
                    action = 'buy'
                else:  # short
                    action = 'short'
                    
                decision = {
                    'action': action,
                    'quantity': scaling_size,
                    'confidence': 80,  # High confidence for scaling decisions
                    'reason': 'Scale in'
                }
                
                # Size the decision properly
                sized_decisions = self._calculate_position_sizes(
                    {symbol: decision}, 
                    portfolio_state
                )
                
                # Execute if quantity > 0
                if sized_decisions[symbol]['quantity'] > 0:
                    result = self.alpaca.execute_trading_decisions(sized_decisions)
                    management_actions[symbol] = {
                        'action': 'scale_in',
                        'result': result[symbol]
                    }
                    
            elif scale_out:
                # We're profitable, consider reducing position
                
                # Calculate scaling size
                scaling_size = max(1, int(abs(qty) * self.config['scaling_size_pct']))
                
                # Create scaling decision
                if side == 'long':
                    action = 'sell'
                else:  # short
                    action = 'cover'
                    
                decision = {
                    'action': action,
                    'quantity': scaling_size,
                    'confidence': 80,  # High confidence for scaling decisions
                    'reason': 'Scale out'
                }
                
                # Size the decision properly
                sized_decisions = self._calculate_position_sizes(
                    {symbol: decision}, 
                    portfolio_state
                )
                
                # Execute if quantity > 0
                if sized_decisions[symbol]['quantity'] > 0:
                    result = self.alpaca.execute_trading_decisions(sized_decisions)
                    management_actions[symbol] = {
                        'action': 'scale_out',
                        'result': result[symbol]
                    }
                    
        return management_actions
    
    def manage_stops_and_targets(self) -> Dict[str, Dict]:
        """
        Manage stop losses and take profit targets for positions.
        Uses ATR trailing stops if enabled in config.
        
        Returns:
            Dict with management actions taken.
        """
        # Update portfolio state
        self.update_portfolio_cache(force=True)
        portfolio_state = self.get_portfolio_state()
        
        management_actions = {}
        
        # Only proceed if we have positions
        # portfolio_state['positions'] is now a dict {symbol: position_dict}
        if not portfolio_state.get('positions'):
            return management_actions
            
        # Check stop and target conditions for each position dictionary (iterate over values)
        for position in portfolio_state['positions'].values(): 
            symbol = position['symbol']
            side = position['side']
            current_price = float(position['current_price'])
            initial_stop_price = position.get('stop_price', 0) # The stop set at entry
            target_price = position.get('target_price', 0)
            profit_loss_pct = position.get('profit_loss_pct', 0.0)
            qty = abs(int(position['qty']))
            
            stop_price_to_use = initial_stop_price
            stop_reason = 'Initial stop loss triggered'
            is_trailing_stop = False

            # --- ATR Trailing Stop Logic ---
            if self.config.get('enable_trailing_stops', False) and profit_loss_pct > 0:
                atr = self._calculate_atr(symbol)
                atr_multiplier = self.config.get('trailing_stop_atr_multiplier', 2.5)
                
                if atr > 0:
                    trailing_stop_level = 0
                    if side == 'long':
                        # Trail below the current price
                        trailing_stop_level = current_price - (atr * atr_multiplier)
                        # Stop only moves up, never down
                        stop_price_to_use = max(initial_stop_price, trailing_stop_level)
                    else: # side == 'short'
                        # Trail above the current price
                        trailing_stop_level = current_price + (atr * atr_multiplier)
                        # Stop only moves down, never up
                        stop_price_to_use = min(initial_stop_price, trailing_stop_level)
                    
                    # Check if the effective stop is the trailing one
                    if stop_price_to_use == trailing_stop_level and stop_price_to_use != initial_stop_price:
                         is_trailing_stop = True
                         stop_reason = f'ATR trailing stop triggered ({atr_multiplier}x ATR={atr:.2f})'
            # --- End ATR Trailing Stop Logic ---
            
            # Check if stop hit using the determined stop price
            stop_hit = (side == 'long' and current_price <= stop_price_to_use) or \
                       (side == 'short' and current_price >= stop_price_to_use)
            
            # Check if target hit (original logic)
            target_hit = (side == 'long' and current_price >= target_price) or \
                         (side == 'short' and current_price <= target_price)
            
            if stop_hit:
                # Create stop loss order to exit position
                action = 'sell' if side == 'long' else 'cover'
                    
                decision = {
                    'action': action,
                    'quantity': qty,  # Exit full position
                    'confidence': 100,  # Maximum confidence for stop loss
                    'reason': stop_reason # Use dynamic reason
                }
                
                # Execute decision
                # Ensure we size correctly even for stop/target exits
                sized_decisions = self._calculate_position_sizes({symbol: decision}, portfolio_state)
                if sized_decisions[symbol]['quantity'] > 0:
                    result = self.alpaca.execute_trading_decisions(sized_decisions)
                    management_actions[symbol] = {
                        'action': 'stop_loss' + ('_trailing' if is_trailing_stop else ''),
                        'result': result.get(symbol, {'status': 'error', 'error': 'Execution result not found'}) # Add safety
                    }
                else:
                    logger.warning(f"Stop loss for {symbol} resulted in zero quantity after sizing.")

            elif target_hit:
                # Create take profit order to exit position or scale out
                action = 'sell' if side == 'long' else 'cover'
                
                # Determine quantity (full exit or scale out)
                exit_quantity = qty
                if self.config['position_scaling']:
                    # Scale out with percentage of position
                    exit_quantity = max(1, int(qty * self.config['scaling_size_pct']))
                
                decision = {
                    'action': action,
                    'quantity': exit_quantity,
                    'confidence': 90,  # High confidence for take profit
                    'reason': 'Take profit triggered'
                }
                
                # Execute decision
                sized_decisions = self._calculate_position_sizes({symbol: decision}, portfolio_state)
                if sized_decisions[symbol]['quantity'] > 0:
                    result = self.alpaca.execute_trading_decisions(sized_decisions)
                    management_actions[symbol] = {
                        'action': 'take_profit',
                        'result': result.get(symbol, {'status': 'error', 'error': 'Execution result not found'}) # Add safety
                    }
                else:
                    logger.warning(f"Take profit for {symbol} resulted in zero quantity after sizing.")
                    
        return management_actions
                
    # --- Start: Options Portfolio Methods ---
    def get_options_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current state of the options portfolio.

        Returns:
            Dict containing options portfolio state including positions and metrics.
        """
        self.update_portfolio_cache(force=True)

        # Filter options positions (typically start with 'O:')
        all_positions = self.portfolio_cache['positions']
        options_positions = [p for p in all_positions if p['symbol'].startswith('O:')]

        # Calculate options portfolio metrics
        account = self.portfolio_cache['account']
        portfolio_value = float(account['portfolio_value'])

        # Calculate options exposure
        options_market_value = sum(abs(float(p['market_value'])) for p in options_positions)
        options_exposure_ratio = options_market_value / portfolio_value if portfolio_value > 0 else 0

        # Enhance options positions with additional metrics
        enhanced_options_positions = self._enhance_options_positions(options_positions)

        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': portfolio_value,
            'options_market_value': options_market_value,
            'options_exposure_ratio': options_exposure_ratio,
            'positions': enhanced_options_positions,
            'buying_power': float(account['buying_power']),
        }

    def _enhance_options_positions(self, positions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Enhance options positions with additional metrics.

        Args:
            positions: List of options positions from Alpaca API.

        Returns:
            Dict of enhanced positions keyed by option symbol.
        """
        enhanced = {}

        for position in positions:
            symbol = position['symbol']
            entry_price = float(position['avg_entry_price'])
            current_price = float(position['current_price'])
            quantity = int(position['qty'])
            side = position['side']

            # Extract option details from symbol (O:AAPL230616C00150000)
            parts = symbol.split(':')
            if len(parts) != 2:
                continue

            option_parts = parts[1]
            try:
                underlying = ''.join(c for c in option_parts if c.isalpha())
                exp_date_str = option_parts[len(underlying):len(underlying) + 6]
                exp_date = datetime.strptime(exp_date_str, '%y%m%d')
                option_type = 'call' if 'C' in option_parts[len(underlying) + 6:] else 'put'
                strike_str = option_parts[option_parts.find('C' if option_type == 'call' else 'P') + 1:]
                strike_price = float(strike_str) / 1000  # Assuming format like 00150000 = $150.00
            except Exception as e:
                logging.error(f"Error parsing option symbol {symbol}: {e}")
                continue

            # Calculate days to expiration
            now = datetime.now()
            days_to_expiration = (exp_date - now).days + (exp_date - now).seconds / 86400.0

            # Calculate risk metrics
            profit_loss_pct = ((current_price / entry_price) - 1) * 100 if side == 'long' else ((entry_price / current_price) - 1) * 100

            # Get Greeks from Polygon (if available)
            greeks = {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'implied_volatility': 0.0
            }

            try:
                # Import here to avoid circular imports
                from src.integrations.polygon import PolygonClient
                polygon_client = PolygonClient()
                contract_details = polygon_client.get_option_contract_details(symbol)
                greeks = {
                    'delta': contract_details.delta,
                    'gamma': contract_details.gamma,
                    'theta': contract_details.theta,
                    'vega': contract_details.vega,
                    'implied_volatility': contract_details.implied_volatility
                }
            except Exception as e:
                logging.warning(f"Could not fetch Greeks for {symbol}: {e}")

            enhanced[symbol] = {
                **position,
                'underlying': underlying,
                'expiration_date': exp_date.strftime('%Y-%m-%d'),
                'option_type': option_type,
                'strike_price': strike_price,
                'days_to_expiration': days_to_expiration,
                'profit_loss_pct': profit_loss_pct,
                'greeks': greeks
            }

        return enhanced

    def execute_options_decision(self, options_decisions: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Execute options trading decisions.

        Args:
            options_decisions: Dict mapping ticker symbols to options decisions.

        Returns:
            Dict with execution results.
        """
        # Update portfolio state
        self.update_portfolio_cache(force=True)
        portfolio_state = self.get_portfolio_state()
        options_portfolio_state = self.get_options_portfolio_state()

        # Check if market is open if configured to trade only during market hours
        if self.config['market_hours_only']:
            clock = self.alpaca.get_clock()
            if not clock['is_open']:
                return {
                    symbol: {
                        'status': 'skipped',
                        'message': 'Market is closed',
                        'order': None
                    }
                    for symbol in options_decisions.keys()
                }

        # Apply risk management filters
        filtered_decisions = self._apply_options_risk_management(options_decisions, portfolio_state, options_portfolio_state)

        # Calculate position sizes
        sized_decisions = self._calculate_options_position_sizes(filtered_decisions, portfolio_state, options_portfolio_state)

        # Execute the orders
        from src.integrations import AlpacaOptionsTrader
        options_trader = AlpacaOptionsTrader(paper=True)  # Use paper trading by default
        execution_results = options_trader.execute_options_decisions(sized_decisions)

        return execution_results

    def _apply_options_risk_management(
        self,
        decisions: Dict[str, Dict], # Decisions only have action, confidence, reasoning etc.
        portfolio_state: Dict[str, Any],
        options_portfolio_state: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """
        Apply risk management filters to options trading decisions (action only).
        This runs BEFORE sizing. It can change the action to 'none'.

        Args:
            decisions: Dict mapping ticker symbols to options decisions (NO quantity).
            portfolio_state: Current portfolio state.
            options_portfolio_state: Current options portfolio state.

        Returns:
            Dict with filtered options trading decisions (action may be changed to 'none').
        """
        filtered_decisions = {}

        # Get portfolio value and cash
        portfolio_value = portfolio_state['portfolio_value']
        cash = portfolio_state['cash']

        # Get options-specific metrics
        options_market_value = options_portfolio_state['options_market_value']
        # options_exposure_ratio = options_portfolio_state['options_exposure_ratio'] # Not directly used here

        # Check overall portfolio risk
        max_drawdown_reached = self._check_drawdown_threshold(portfolio_state)
        day_trades_available = portfolio_state['day_trades_remaining'] > 0
        max_trades_reached = len(self._get_today_trades()) >= self.config['max_trades_per_day']
        cash_reserve_pct = self.config['cash_reserve_pct']

        # Options-specific risk limits from config
        max_options_allocation_pct = self.config.get('max_options_allocation_pct', 0.15)
        max_single_option_allocation_pct = self.config.get('max_options_position_size_pct', 0.05)

        max_options_allocation_value = portfolio_value * max_options_allocation_pct
        max_single_option_allocation_value = portfolio_value * max_single_option_allocation_pct

        for ticker, decision in decisions.items():
            current_decision = decision.copy()
            
            # Check if this is a multi-leg strategy (like bull call spread, bear put spread)
            is_multi_leg = 'legs' in current_decision and isinstance(current_decision['legs'], list)
            
            if is_multi_leg:
                # For multi-leg strategies (like bear put spread)
                
                # Skip if legs are empty
                if not current_decision['legs']:
                    current_decision['action'] = 'none'
                    current_decision['skip_reason'] = 'Multi-leg decision has empty legs list'
                    filtered_decisions[ticker] = current_decision
                    continue
                
                # Use underlying ticker for context if available
                log_context_ticker = current_decision.get('underlying_ticker', ticker)
                
                # Set action to 'open_spread' if not set
                if 'action' not in current_decision or current_decision.get('action', '').lower() == 'none':
                    current_decision['action'] = 'open_spread'
                
                # Consider this an opening trade for risk filtering purposes
                effective_action = 'open_spread'
            else:
                # For single leg option
                action = current_decision.get('action', 'none').lower()
                
                # Skip if already no action
                if action == 'none':
                    filtered_decisions[ticker] = current_decision
                    continue
                
                # Make sure we have a valid ticker
                log_context_ticker = current_decision.get('ticker', '')
                if not log_context_ticker:
                    self.logger.warning(f"Risk Filter: Setting action to 'none' due to missing option ticker in single-leg decision.")
                    current_decision['action'] = 'none'
                    current_decision['skip_reason'] = 'Invalid option ticker'
                    filtered_decisions[ticker] = current_decision
                    continue
                    
                # Determine effective action type (opening vs closing)
                if action == 'close':
                    effective_action = 'close'
                else:  # 'buy' or 'sell'
                    effective_action = 'open'

            # === Apply Portfolio-Level Risk Checks ===

            # Check max drawdown threshold (restrict opening new risk)
            if max_drawdown_reached and effective_action in ['open', 'open_spread']:
                self.logger.warning(f"Risk Filter: Setting {log_context_ticker} action to 'none' due to max drawdown.")
                current_decision['original_action'] = current_decision.get('action', 'none')
                current_decision['action'] = 'none'
                current_decision['skip_reason'] = 'Max drawdown threshold reached'
                filtered_decisions[ticker] = current_decision
                continue

            # Check max trades per day (restrict opening new risk)
            if max_trades_reached and effective_action in ['open', 'open_spread']:
                self.logger.warning(f"Risk Filter: Setting {log_context_ticker} action to 'none' due to max trades per day.")
                current_decision['original_action'] = current_decision.get('action', 'none')
                current_decision['action'] = 'none'
                current_decision['skip_reason'] = 'Maximum trades per day reached'
                filtered_decisions[ticker] = current_decision
                continue

            # Check day trading restrictions (currently only for closing single legs)
            # Multi-leg day trade check would be more complex
            if effective_action == 'close': 
                position_exists = log_context_ticker in options_portfolio_state.get('positions', {})
                is_day_trade = False
                if position_exists:
                    today_orders = self._get_today_trades()
                    option_today_orders = [
                        order for order in today_orders
                        if order.get('symbol') == log_context_ticker and order.get('status') == 'filled'
                    ]
                    if option_today_orders:
                        is_day_trade = True

                if is_day_trade and not day_trades_available:
                    self.logger.warning(f"Risk Filter: Setting {log_context_ticker} {current_decision.get('action', 'none')} to 'none' due to no day trades available.")
                    current_decision['original_action'] = current_decision.get('action', 'none')
                    current_decision['action'] = 'none'
                    current_decision['skip_reason'] = 'No day trades available'
                    filtered_decisions[ticker] = current_decision
                    continue

            # Check options allocation limits (only for OPENING actions)
            if effective_action in ['open', 'open_spread'] and options_market_value >= max_options_allocation_value:
                self.logger.warning(f"Risk Filter: Setting {log_context_ticker} action to 'none' due to max options allocation reached ({options_market_value:.2f} >= {max_options_allocation_value:.2f}).")
                current_decision['original_action'] = current_decision.get('action', 'none')
                current_decision['action'] = 'none'
                current_decision['skip_reason'] = 'Maximum options allocation reached'
                filtered_decisions[ticker] = current_decision
                continue

            # If we passed all filters, keep the decision 
            filtered_decisions[ticker] = current_decision

        return filtered_decisions

    def manage_options_positions(self) -> Dict[str, Dict]:
        """
        Manage existing options positions (check for expiring contracts, adjust for Greeks, etc.).

        Returns:
            Dict with management actions taken.
        """
        # Update portfolio state
        self.update_portfolio_cache(force=True)
        options_portfolio_state = self.get_options_portfolio_state()

        management_actions = {}

        # Only proceed if we have options positions
        if not options_portfolio_state.get('positions'): # Check key exists
            return management_actions

        # Check each position for management actions
        for symbol, position in options_portfolio_state['positions'].items():
            # Check for near-expiry contracts
            days_to_expiration = position.get('days_to_expiration', 0)

            # Close positions with <= 1 day to expiration to avoid assignment/exercise
            if days_to_expiration <= 1:
                action = 'close'
                decision = {
                    'ticker': symbol,
                    'action': action,
                    'quantity': abs(int(position['qty'])),
                    'underlying_ticker': position.get('underlying', ''),
                    'option_type': position.get('option_type', ''),
                    'strike_price': position.get('strike_price', 0),
                    'expiration_date': position.get('expiration_date', ''),
                    'strategy': 'expiration_management',
                    'confidence': 100,
                    'reasoning': f"Automatically closing position with {days_to_expiration:.1f} days to expiration"
                }

                # Execute the decision
                from src.integrations import AlpacaOptionsTrader
                options_trader = AlpacaOptionsTrader(paper=True)
                result = options_trader.execute_option_decision(decision)

                management_actions[symbol] = {
                    'action': 'close_expiring',
                    'result': result
                }
                continue # Don't check other conditions if closing for expiration

            # Check for stop loss / take profit conditions
            profit_loss_pct = position.get('profit_loss_pct', 0)
            side = position.get('side', '')

            # Use configured thresholds if available, otherwise use defaults
            if side == 'long':
                stop_loss_threshold = -self.config.get('options_stop_loss_pct', 0.25) * 100
                take_profit_threshold = self.config.get('options_take_profit_pct', 0.50) * 100
            elif side == 'short':
                stop_loss_threshold = self.config.get('options_stop_loss_pct', 0.25) * 100 # Loss is gain for short
                take_profit_threshold = -self.config.get('options_take_profit_pct', 0.50) * 100 # Profit is loss for short
            else:
                continue # Skip if side is unknown

            # Check stop loss
            if (side == 'long' and profit_loss_pct <= stop_loss_threshold) or \
               (side == 'short' and profit_loss_pct >= stop_loss_threshold):
                action = 'close'
                decision = {
                    'ticker': symbol,
                    'action': action,
                    'quantity': abs(int(position['qty'])),
                    'underlying_ticker': position.get('underlying', ''),
                    'option_type': position.get('option_type', ''),
                    'strike_price': position.get('strike_price', 0),
                    'expiration_date': position.get('expiration_date', ''),
                    'strategy': 'stop_loss',
                    'confidence': 100,
                    'reasoning': f"Stop loss triggered at {profit_loss_pct:.1f}% (Threshold: {stop_loss_threshold:.1f}%)"
                }

                # Execute the decision
                from src.integrations import AlpacaOptionsTrader
                options_trader = AlpacaOptionsTrader(paper=True)
                result = options_trader.execute_option_decision(decision)

                management_actions[symbol] = {
                    'action': 'stop_loss',
                    'result': result
                }
                continue # Exit after stop loss

            # Check take profit
            if (side == 'long' and profit_loss_pct >= take_profit_threshold) or \
               (side == 'short' and profit_loss_pct <= take_profit_threshold):
                action = 'close'
                decision = {
                    'ticker': symbol,
                    'action': action,
                    'quantity': abs(int(position['qty'])),
                    'underlying_ticker': position.get('underlying', ''),
                    'option_type': position.get('option_type', ''),
                    'strike_price': position.get('strike_price', 0),
                    'expiration_date': position.get('expiration_date', ''),
                    'strategy': 'take_profit',
                    'confidence': 100,
                    'reasoning': f"Take profit triggered at {profit_loss_pct:.1f}% (Threshold: {take_profit_threshold:.1f}%)"
                }

                # Execute the decision
                from src.integrations import AlpacaOptionsTrader
                options_trader = AlpacaOptionsTrader(paper=True)
                result = options_trader.execute_option_decision(decision)

                management_actions[symbol] = {
                    'action': 'take_profit',
                    'result': result
                }

        return management_actions
    # --- End: Options Portfolio Methods ---

    # --- AI Integration Method ---
    def get_ai_portfolio_format(self) -> Dict[str, Any]:
        """
        Get portfolio data in a format compatible with AI Hedge Fund, including options.
        
        Returns:
            Dict with portfolio information formatted for AI Hedge Fund.
        """
        # Update portfolio state
        self.update_portfolio_cache(force=True)
        
        account = self.portfolio_cache['account']
        positions = self.portfolio_cache['positions']
        
        # Create position mapping for AI Hedge Fund format
        ai_positions = {}
        options_positions = {}
        
        for position in positions:
            symbol = position['symbol']
            qty = int(position['qty'])
            side = position['side']
            
            # Check if this is an options position
            if symbol.startswith('O:'):
                # Extract underlying ticker
                try:
                    parts = symbol.split(':')
                    if len(parts) != 2:
                        continue
                        
                    option_parts = parts[1]
                    underlying = ''.join(c for c in option_parts if c.isalpha())
                    exp_date_str = option_parts[len(underlying):len(underlying) + 6]
                    exp_date = datetime.strptime(exp_date_str, '%y%m%d')
                    option_type = 'call' if 'C' in option_parts[len(underlying) + 6:] else 'put'
                    strike_str = option_parts[option_parts.find('C' if option_type == 'call' else 'P') + 1:]
                    strike_price = float(strike_str) / 1000  # Assuming format like 00150000 = $150.00
                    
                    if underlying not in options_positions:
                        options_positions[underlying] = []
                        
                    options_positions[underlying].append({
                        "symbol": symbol,
                        "option_type": option_type,
                        "strike_price": strike_price,
                        "expiration_date": exp_date.strftime('%Y-%m-%d'),
                        "quantity": qty,
                        "side": side,
                        "avg_entry_price": float(position['avg_entry_price']),
                        "current_price": float(position['current_price']),
                        "market_value": float(position['market_value'])
                    })
                except Exception as e:
                    logging.error(f"Error parsing option symbol {symbol}: {e}")
                    continue
            else:
                # For stock positions
                if side == 'long':
                    # For long positions
                    ai_positions[symbol] = {
                        "long": qty,
                        "short": 0,
                        "long_cost_basis": float(position['avg_entry_price']),
                        "short_cost_basis": 0.0,
                        "short_margin_used": 0.0
                    }
                else:  # short
                    # For short positions
                    ai_positions[symbol] = {
                        "long": 0,
                        "short": abs(qty),  # Make sure it's positive
                        "long_cost_basis": 0.0,
                        "short_cost_basis": float(position['avg_entry_price']),
                        "short_margin_used": float(position['market_value'])  # Approximate
                    }
        
        # Create the portfolio dict in AI Hedge Fund format
        ai_portfolio = {
            "cash": float(account['cash']),
            "margin_requirement": 0.0,  # Set to 0 or a value based on your brokerage
            "margin_used": float(account['initial_margin']),
            "positions": ai_positions,
            "options_positions": options_positions,
            "realized_gains": {
                symbol: {
                    "long": 0.0,  # We don't have this data readily available
                    "short": 0.0  # We don't have this data readily available
                } for symbol in ai_positions.keys()
            }
        }
        
        return ai_portfolio

    # --- Execute Combined Decision (New Method) ---
    def execute_combined_decision(
        self,
        stock_decision: Optional[Dict[str, Any]], # Action, confidence, reasoning (NO quantity)
        option_decision: Optional[Dict[str, Any]] # Action, confidence, reasoning etc. (NO quantity)
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Executes a potentially combined stock and option decision after risk filtering and sizing.

        Args:
            stock_decision: The stock decision dictionary (or None).
            option_decision: The option decision dictionary (or None).

        Returns:
            A tuple containing the execution results for the stock and option orders (or None).
            Result format: { 'status': '...', 'message': '...', 'order': {...} or None }
        """
        stock_execution_result = None
        option_execution_result = None

        # --- Basic Validation & State Update ---
        if not stock_decision and not option_decision:
            logger.info("execute_combined_decision: No decisions provided.")
            return None, None

        self.update_portfolio_cache(force=True)
        portfolio_state = self.get_portfolio_state()
        options_portfolio_state = self.get_options_portfolio_state()

        # --- Market Hours Check ---
        if self.config['market_hours_only']:
            clock = self.alpaca.get_clock()
            if not clock['is_open']:
                logger.warning("Market is closed. Skipping combined execution.")
                skip_result = {'status': 'skipped', 'message': 'Market is closed', 'order': None}
                if stock_decision: stock_execution_result = skip_result
                if option_decision: option_execution_result = skip_result
                return stock_execution_result, option_execution_result

        # --- Apply Risk Filters (Action Filtering) ---
        filtered_stock_decision = None
        if stock_decision:
            # Wrap in dict for filter method compatibility
            temp_stock_decisions = {stock_decision['ticker']: stock_decision}
            filtered_result = self._apply_risk_management(temp_stock_decisions, portfolio_state)
            filtered_stock_decision = filtered_result.get(stock_decision['ticker'])

        filtered_option_decision = None
        if option_decision:
            # Check if this is a multi-leg decision or a single-leg decision
            is_multi_leg = 'legs' in option_decision and isinstance(option_decision['legs'], list)
            
            if is_multi_leg:
                # For multi-leg, use underlying_ticker as the key
                option_ticker_key = option_decision.get('underlying_ticker')
                if option_ticker_key:
                    temp_option_decisions = {option_ticker_key: option_decision}
                    filtered_result = self._apply_options_risk_management(temp_option_decisions, portfolio_state, options_portfolio_state)
                    filtered_option_decision = filtered_result.get(option_ticker_key)
                else:
                    logger.warning(f"Multi-leg option decision received without an 'underlying_ticker' key: {option_decision}. Skipping options risk management for it.")
                    filtered_option_decision = option_decision
            else:
                # For single-leg, use ticker as the key
                option_ticker_key = option_decision.get('ticker')
                if option_ticker_key:
                    temp_option_decisions = {option_ticker_key: option_decision}
                    filtered_result = self._apply_options_risk_management(temp_option_decisions, portfolio_state, options_portfolio_state)
                    filtered_option_decision = filtered_result.get(option_ticker_key)
                else:
                    logger.warning(f"Option decision received without a 'ticker' key: {option_decision}. Skipping options risk management for it.")
                    filtered_option_decision = option_decision

        # Check if actions were changed to hold/none
        final_stock_action = filtered_stock_decision.get('action', 'hold') if filtered_stock_decision else 'hold'
        final_option_action = filtered_option_decision.get('action', 'none') if filtered_option_decision else 'none'

        if final_stock_action == 'hold' and final_option_action == 'none':
             logger.info("Both stock and option actions are hold/none after risk filtering. Nothing to size or execute.")
             if filtered_stock_decision:
                 stock_execution_result = {'status': 'skipped', 'message': f"Action '{filtered_stock_decision.get('original_action', 'hold')}' filtered to hold. Reason: {filtered_stock_decision.get('skip_reason', 'N/A')}", 'order': None}
             if filtered_option_decision:
                 option_execution_result = {'status': 'skipped', 'message': f"Action '{filtered_option_decision.get('original_action', 'none')}' filtered to none. Reason: {filtered_option_decision.get('skip_reason', 'N/A')}", 'order': None}
             return stock_execution_result, option_execution_result

        # --- Call Portfolio Sizer Agent ---
        logger.info("Calling PortfolioSizerAgent...")
        try:
            # Pass the decisions that passed risk filtering (or None if they didn't exist initially)
            sized_stock_decision, sized_option_decision = self.portfolio_sizer_agent.calculate_combined_sizes(
                filtered_stock_decision if final_stock_action != 'hold' else None,
                filtered_option_decision if final_option_action != 'none' else None,
                portfolio_state,
                options_portfolio_state
            )
        except Exception as e:
            logger.error(f"Error during portfolio sizing: {e}", exc_info=True)
            error_result = {'status': 'error', 'message': f'Sizing agent failed: {e}', 'order': None}
            if stock_decision: stock_execution_result = error_result
            if option_decision: option_execution_result = error_result
            return stock_execution_result, option_execution_result

        # --- Execute Orders ---
        # Execute Stock Order if sized quantity > 0
        if sized_stock_decision and sized_stock_decision.get('quantity', 0) > 0:
            logger.info(f"Executing sized stock decision: {sized_stock_decision}")
            # Alpaca execution expects a dict keyed by symbol
            stock_orders_to_execute = {sized_stock_decision['ticker']: sized_stock_decision}
            try:
                 results = self.alpaca.execute_trading_decisions(stock_orders_to_execute)
                 stock_execution_result = results.get(sized_stock_decision['ticker'])
                 logger.info(f"Stock Execution Result: {stock_execution_result}")
            except Exception as e:
                 logger.error(f"Error executing stock trade for {sized_stock_decision['ticker']}: {e}", exc_info=True)
                 stock_execution_result = {'status': 'error', 'message': f'Stock execution failed: {e}', 'order': None}
        elif stock_decision: # If there was an initial decision but quantity is 0
            reason = sized_stock_decision.get('skip_reason', 'Quantity sized to zero') if sized_stock_decision else 'Filtered to hold'
            stock_execution_result = {'status': 'skipped', 'message': reason, 'order': None}
            logger.info(f"Skipping stock execution for {stock_decision['ticker']}. Reason: {reason}")

        # Execute Option Order if sized quantity > 0
        if sized_option_decision and sized_option_decision.get('quantity', 0) > 0:
            logger.info(f"Executing sized option decision: {sized_option_decision}")
            from src.integrations import AlpacaOptionsTrader
            options_trader = AlpacaOptionsTrader(paper=True)
            
            # Determine the key for the option decision
            option_key = None
            if 'legs' in sized_option_decision and isinstance(sized_option_decision['legs'], list):
                # Multi-leg strategy, use underlying_ticker as key
                option_key = sized_option_decision.get('underlying_ticker')
            else:
                # Single-leg, use ticker as key
                option_key = sized_option_decision.get('ticker')
                
            if not option_key:
                logger.error(f"Missing key (ticker/underlying_ticker) for option decision: {sized_option_decision}")
                option_execution_result = {'status': 'error', 'message': 'Missing ticker identifier', 'order': None}
            else:
                option_orders_to_execute = {option_key: sized_option_decision}
                try:
                    results = options_trader.execute_options_decisions(option_orders_to_execute)
                    option_execution_result = results.get(option_key)
                    logger.info(f"Option Execution Result: {option_execution_result}")
                except Exception as e:
                    logger.error(f"Error executing option trade for {option_key}: {e}", exc_info=True)
                    option_execution_result = {'status': 'error', 'message': f'Option execution failed: {e}', 'order': None}
        elif option_decision: # If there was an initial decision but quantity is 0
            reason = sized_option_decision.get('skip_reason', 'Quantity sized to zero') if sized_option_decision else 'Filtered to none'
            option_execution_result = {'status': 'skipped', 'message': reason, 'order': None}
            # Use the underlying ticker from stock_decision for logging context, as option_decision might lack 'ticker'
            log_ticker = stock_decision.get('ticker', option_decision.get('underlying_ticker', 'UNKNOWN'))
            logger.info(f"Skipping option execution for {log_ticker}. Reason: {reason}")

        return stock_execution_result, option_execution_result

    # --- Old Execution Methods (Refactor/Remove) ---
    def execute_decision(self, decisions: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        DEPRECATED: Use execute_combined_decision. Executes stock trading decisions individually.
        """
        logger.warning("execute_decision is deprecated. Use execute_combined_decision for holistic sizing.")
        results = {}
        for ticker, decision in decisions.items():
             # Add the ticker back into the decision dict before passing it
             decision_with_ticker = decision.copy()
             decision_with_ticker['ticker'] = ticker # Add the stock ticker key
             # Assume no associated option decision when called this way
             stock_result, _ = self.execute_combined_decision(stock_decision=decision_with_ticker, option_decision=None)
             results[ticker] = stock_result if stock_result else {'status': 'error', 'message': 'Execution failed', 'order': None}
        return results

    def execute_options_decision(self, options_decisions: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        DEPRECATED: Use execute_combined_decision. Executes option trading decisions individually.
        """
        logger.warning("execute_options_decision is deprecated. Use execute_combined_decision for holistic sizing.")
        results = {}
        # The key in options_decisions should be the option ticker itself
        for option_ticker, decision in options_decisions.items(): 
             # Add the ticker back into the decision dict before passing it
             decision_with_ticker = decision.copy()
             decision_with_ticker['ticker'] = option_ticker # Add the option ticker key
             # Assume no associated stock decision when called this way
             _, option_result = self.execute_combined_decision(stock_decision=None, option_decision=decision_with_ticker)
             results[option_ticker] = option_result if option_result else {'status': 'error', 'message': 'Execution failed', 'order': None}
        return results