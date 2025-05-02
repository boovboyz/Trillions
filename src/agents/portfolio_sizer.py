"""
Robust portfolio sizer that implements modern risk management principles
and properly handles both stock and options sizing.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import math

class RobustPortfolioSizer:
    def __init__(self, config: Dict[str, Any], alpaca_client, polygon_client, portfolio_cache: Dict[str, Any]):
        """
        Initialize the RobustPortfolioSizer.

        Args:
            config: Configuration dictionary for sizing parameters
            alpaca_client: Client for interacting with Alpaca
            polygon_client: Client for interacting with Polygon
            portfolio_cache: Access to cached portfolio data
        """
        self.config = config
        self.alpaca = alpaca_client
        self.polygon = polygon_client
        self.portfolio_cache = portfolio_cache
        self.logger = logging.getLogger(__name__)
        self.logger.info("RobustPortfolioSizer initialized.")

    def calculate_position_size(self, ticker: str, action: str, current_price: float, 
                               confidence: float, portfolio_value: float, 
                               is_option: bool = False, risk_multiplier: float = 1.0) -> int:
        """
        Calculate position size based on Kelly Criterion and risk management principles.
        
        Args:
            ticker: The ticker symbol
            action: The action (buy, sell, short, cover)
            current_price: Current price of the asset
            confidence: Confidence level (0-100)
            portfolio_value: Total portfolio value
            is_option: Whether this is an options contract
            risk_multiplier: Adjust risk for special scenarios
            
        Returns:
            Quantity to trade
        """
        if current_price <= 0:
            self.logger.warning(f"Cannot size {action} for {ticker} with non-positive price.")
            return 0
            
        # For closing actions (sell, cover), we'll determine quantity differently
        if action in ['sell', 'cover', 'close']:
            return self._calculate_closing_size(ticker, action, is_option)
            
        # Get base risk percentage from config
        base_risk_percent = self.config.get('risk_percent_per_trade', 1.0)  # Default 1%
        
        # Adjust risk based on confidence
        confidence_factor = max(0.2, min(1.0, confidence / 100))
        
        # Risk percentage of portfolio for this trade (confidence-adjusted)
        risk_percent = base_risk_percent * confidence_factor * risk_multiplier
        
        # For options, adjust risk further down (options are inherently leveraged)
        if is_option:
            options_risk_factor = self.config.get('options_risk_factor', 0.5)  # Default 50% of stock risk
            risk_percent *= options_risk_factor
            
        # Calculate dollar amount to risk
        risk_amount = portfolio_value * (risk_percent / 100)
        
        # Adjust for volatility
        volatility_factor = self._calculate_volatility_factor(ticker)
        risk_amount *= volatility_factor
        
        # Calculate quantity
        if is_option:
            # For options, consider contract multiplier (typically 100 shares per contract)
            contract_value = current_price * 100
            quantity = math.floor(risk_amount / contract_value)
        else:
            quantity = math.floor(risk_amount / current_price)
            
        self.logger.info(f"Base size calculated for {ticker} ({action}): {quantity} units at ${current_price} " +
                         f"(Risk: {risk_percent:.2f}%, Amount: ${risk_amount:.2f}, Vol factor: {volatility_factor:.2f})")
        
        return max(0, quantity)

    def _calculate_volatility_factor(self, ticker: str) -> float:
        """
        Calculate volatility adjustment factor.
        
        Args:
            ticker: The ticker symbol
            
        Returns:
            Volatility adjustment factor (lower for higher volatility)
        """
        try:
            # Get ATR or historical volatility
            volatility = self._get_asset_volatility(ticker)
            
            # Define volatility thresholds
            base_volatility = self.config.get('base_volatility', 20.0)  # Expected "normal" volatility
            max_reduction = self.config.get('max_volatility_reduction', 0.7)  # Maximum reduction factor
            
            # Calculate adjustment factor (inverse relationship)
            if volatility <= base_volatility:
                return 1.0
            else:
                # Reduce position size as volatility increases
                reduction_factor = min(max_reduction, base_volatility / volatility)
                return max(1.0 - max_reduction, reduction_factor)
        except Exception as e:
            self.logger.warning(f"Error calculating volatility factor for {ticker}: {e}")
            return 0.8  # Default to 80% if can't calculate volatility
    
    def _get_asset_volatility(self, ticker: str) -> float:
        """
        Get asset volatility (ATR or standard deviation).
        
        Args:
            ticker: The ticker symbol
            
        Returns:
            Volatility value (percentage)
        """
        # Placeholder implementation - replace with actual volatility calculation
        # Could use standard deviation of returns or ATR normalized by price
        return 20.0  # Default 20% volatility if calculation unavailable

    def _calculate_closing_size(self, ticker: str, action: str, is_option: bool = False) -> int:
        """
        Calculate size for closing positions (sell, cover).
        
        Args:
            ticker: The ticker symbol
            action: The action (sell, cover, close)
            is_option: Whether this is an options contract
            
        Returns:
            Quantity to trade
        """
        # Get current positions
        positions = self.portfolio_cache.get('positions', [])
        
        # Find position for this ticker
        position = None
        for pos in positions:
            if pos.get('symbol') == ticker:
                position = pos
                break
                
        if not position:
            self.logger.warning(f"No existing position found for {ticker}")
            return 0
            
        # Get quantity
        qty = abs(int(position.get('qty', 0)))
        
        # For sell/cover, use the full position
        return qty

    def apply_constraints(self, ticker: str, action: str, target_quantity: int, 
                         current_price: float, portfolio_state: Dict[str, Any], 
                         is_option: bool = False, strike_price: Optional[float] = None) -> int:
        """
        Apply portfolio constraints to the target quantity.
        
        Args:
            ticker: The ticker symbol
            action: The action (buy, sell, short, cover)
            target_quantity: Initial target quantity
            current_price: Current price of the asset
            portfolio_state: Current portfolio state
            is_option: Whether this is an options contract
            strike_price: Strike price (for options only)
            
        Returns:
            Final quantity after constraints
        """
        if target_quantity <= 0:
            return 0
            
        # Extract portfolio data
        portfolio_value = portfolio_state.get('portfolio_value', 0)
        cash = portfolio_state.get('cash', 0)
        buying_power = portfolio_state.get('buying_power', 0)
        positions = portfolio_state.get('positions', {})
        
        # Get position limits from config
        max_position_pct = self.config.get('max_position_size_pct', 0.15)  # Default 15%
        if is_option:
            max_position_pct = self.config.get('max_options_position_size_pct', 0.05)  # Default 5% for options
            
        # Cash reserve (% of portfolio to keep as cash)
        cash_reserve_pct = self.config.get('cash_reserve_pct', 0.10)  # Default 10%
        cash_reserve = portfolio_value * cash_reserve_pct
        available_cash = max(0, cash - cash_reserve)
        
        # Calculate maximum position value
        max_position_value = portfolio_value * max_position_pct
        
        # Initialize variables for constraints
        final_quantity = target_quantity
        
        # Apply constraints based on action
        if action == 'buy':
            # Check position limit
            position_value = final_quantity * current_price
            if position_value > max_position_value:
                final_quantity = math.floor(max_position_value / current_price)
                self.logger.info(f"Reducing {ticker} quantity from {target_quantity} to {final_quantity} due to position limit")
                
            # Check cash constraint
            if final_quantity * current_price > available_cash:
                final_quantity = math.floor(available_cash / current_price)
                self.logger.info(f"Reducing {ticker} quantity from {target_quantity} to {final_quantity} due to cash constraint")
                
        elif action == 'short':
            # Check position limit (for shorts)
            position_value = final_quantity * current_price
            max_short_pct = self.config.get('max_short_position_size_pct', max_position_pct)
            max_short_value = portfolio_value * max_short_pct
            
            if position_value > max_short_value:
                final_quantity = math.floor(max_short_value / current_price)
                self.logger.info(f"Reducing {ticker} short quantity from {target_quantity} to {final_quantity} due to short position limit")
                
            # Check margin requirement
            margin_requirement = current_price * final_quantity * 1.5  # Typical 150% requirement
            if margin_requirement > buying_power:
                final_quantity = math.floor(buying_power / (current_price * 1.5))
                self.logger.info(f"Reducing {ticker} short quantity from {target_quantity} to {final_quantity} due to buying power constraint")
                
        elif action == 'buy' and is_option:
            # For options 'buy', apply options-specific constraints
            contract_value = current_price * 100
            position_value = final_quantity * contract_value
            
            # Check position limit
            if position_value > max_position_value:
                final_quantity = math.floor(max_position_value / contract_value)
                self.logger.info(f"Reducing {ticker} option quantity from {target_quantity} to {final_quantity} due to position limit")
                
            # Check cash constraint
            if final_quantity * contract_value > available_cash:
                final_quantity = math.floor(available_cash / contract_value)
                self.logger.info(f"Reducing {ticker} option quantity from {target_quantity} to {final_quantity} due to cash constraint")
                
        elif action == 'sell' and is_option and option_type == 'put' and strike_price:
            # For cash-secured puts, ensure enough cash to secure the puts
            cash_required = final_quantity * strike_price * 100
            if cash_required > available_cash:
                final_quantity = math.floor(available_cash / (strike_price * 100))
                self.logger.info(f"Reducing {ticker} CSP quantity from {target_quantity} to {final_quantity} due to cash required for securing puts")
        
        # Ensure minimum quantity
        final_quantity = max(0, final_quantity)
        
        # Apply round lot sizing if enabled (stocks only, not options)
        if not is_option and self.config.get('use_round_lots', False):
            lot_size = self.config.get('lot_size', 100)
            final_quantity = (final_quantity // lot_size) * lot_size
            
        return final_quantity

    def size_stock_position(self, ticker: str, action: str, confidence: float, 
                          portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate and apply constraints for a stock position.
        
        Args:
            ticker: The ticker symbol
            action: The action (buy, sell, short, cover)
            confidence: Confidence level (0-100)
            portfolio_state: Current portfolio state
            
        Returns:
            Dictionary with action, quantity, and skip reason if applicable
        """
        result = {
            'ticker': ticker,
            'action': action,
            'quantity': 0,
            'skip_reason': None
        }
        
        if action == 'hold':
            result['skip_reason'] = "Action is hold"
            return result
            
        # Get current price
        try:
            current_price = self._get_stock_price(ticker)
            if current_price is None or current_price <= 0:
                result['skip_reason'] = f"Could not get valid price for {ticker}"
                return result
        except Exception as e:
            self.logger.error(f"Error getting price for {ticker}: {e}")
            result['skip_reason'] = f"Error getting price: {str(e)}"
            return result
            
        # Calculate initial target quantity
        portfolio_value = portfolio_state.get('portfolio_value', 0)
        target_quantity = self.calculate_position_size(
            ticker=ticker,
            action=action,
            current_price=current_price,
            confidence=confidence,
            portfolio_value=portfolio_value,
            is_option=False
        )
        
        if target_quantity <= 0:
            result['skip_reason'] = "Sized to zero"
            return result
            
        # Apply constraints
        final_quantity = self.apply_constraints(
            ticker=ticker,
            action=action,
            target_quantity=target_quantity,
            current_price=current_price,
            portfolio_state=portfolio_state,
            is_option=False
        )
        
        result['quantity'] = final_quantity
        if final_quantity <= 0:
            result['skip_reason'] = "Constrained to zero"
            
        return result

    def size_option_position(self, option_decision: Dict[str, Any], 
                           portfolio_state: Dict[str, Any],
                           options_portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate and apply constraints for an option position.

        Args:
            option_decision: Option decision dictionary
            portfolio_state: Current portfolio state
            options_portfolio_state: Current options portfolio state

        Returns:
            Dictionary with action, quantity, and skip reason if applicable
        """
        ticker = option_decision.get('ticker')
        action = option_decision.get('action', 'none')
        confidence = option_decision.get('confidence', 50.0)
        option_type = option_decision.get('option_type')
        strike_price = option_decision.get('strike_price')
        
        result = {
            'ticker': ticker,
            'action': action,
            'quantity': 0,
            'skip_reason': None
        }
        
        if action in ['none', 'hold']:
            result['skip_reason'] = f"Action is {action}"
            return result
            
        # Get current price
        try:
            if 'limit_price' in option_decision and option_decision['limit_price'] > 0:
                current_price = option_decision['limit_price']
            else:
                # Try to get from option positions or Polygon
                options_positions = options_portfolio_state.get('positions', {})
                if ticker in options_positions:
                    current_price = options_positions[ticker].get('current_price', 0)
                else:
                    # Try Polygon API
                    try:
                        contract_details = self.polygon.get_option_contract_details(ticker)
                        current_price = contract_details.last_price or contract_details.ask
                    except Exception as e:
                        self.logger.error(f"Could not get price from Polygon for {ticker}: {e}")
                        current_price = 0
            
            if current_price is None or current_price <= 0:
                result['skip_reason'] = f"Could not get valid price for {ticker}"
                return result
        except Exception as e:
            self.logger.error(f"Error getting price for option {ticker}: {e}")
            result['skip_reason'] = f"Error getting price: {str(e)}"
            return result
            
        # Calculate initial target quantity
        portfolio_value = portfolio_state.get('portfolio_value', 0)
        
        # For closing positions, use full position size
        if action == 'close':
            options_positions = options_portfolio_state.get('positions', {})
            position = options_positions.get(ticker)
            
            if not position:
                result['skip_reason'] = f"No existing position found for {ticker}"
                return result
                
            result['quantity'] = abs(int(position.get('qty', 0)))
            return result
            
        # For opening positions, calculate size
        target_quantity = self.calculate_position_size(
            ticker=ticker,
            action=action,
            current_price=current_price,
            confidence=confidence,
            portfolio_value=portfolio_value,
            is_option=True
        )
        
        if target_quantity <= 0:
            result['skip_reason'] = "Sized to zero"
            return result
            
        # Apply constraints
        final_quantity = self.apply_constraints(
            ticker=ticker,
            action=action,
            target_quantity=target_quantity,
            current_price=current_price,
            portfolio_state=portfolio_state,
            is_option=True,
            strike_price=strike_price
        )
        
        result['quantity'] = final_quantity
        if final_quantity <= 0:
            result['skip_reason'] = "Constrained to zero"
            
        return result

    def _get_stock_price(self, symbol: str) -> Optional[float]:
        """Helper to get current stock price."""
        try:
            latest_trade = self.alpaca.get_latest_trade(symbol)
            price = latest_trade['price']
            if price <= 0:
                 self.logger.warning(f"Received non-positive price {price} for {symbol}")
                 return None
            return float(price)
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
            
    def calculate_batch_sizes(
        self,
        stock_decisions: Dict[str, Dict[str, Any]],
        option_decisions: Dict[str, Dict[str, Any]],
        portfolio_state: Dict[str, Any],
        options_portfolio_state: Dict[str, Any]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Calculates sizes for a batch of stock and option decisions.
        
        Args:
            stock_decisions: Dict mapping tickers to stock decision dicts.
            option_decisions: Dict mapping option tickers to option decision dicts.
            portfolio_state: Current state of the stock portfolio.
            options_portfolio_state: Current state of the options portfolio.
            
        Returns:
            Tuple of (sized_stock_decisions, sized_option_decisions).
        """
        sized_stock_decisions = {}
        sized_option_decisions = {}
        
        # Calculate initial sizes and constraints for each decision individually
        for ticker, decision in stock_decisions.items():
            sized_stock_decisions[ticker] = self.size_stock_position(
                ticker=ticker,
                action=decision.get('action', 'hold'),
                confidence=decision.get('confidence', 50.0),
                portfolio_state=portfolio_state
            )
            
        for ticker, decision in option_decisions.items():
            sized_option_decisions[ticker] = self.size_option_position(
                option_decision=decision,
                portfolio_state=portfolio_state,
                options_portfolio_state=options_portfolio_state
            )
            
        # --- Simple Batch Constraint (Optional Refinement) ---
        # TODO: Implement more sophisticated batch constraint logic if needed.
        # For now, relying on individual constraints applied within size_*_position.
        # A more robust implementation might calculate total resource needs
        # and scale down proportionally if constraints are violated by the batch.
        
        self.logger.info(f"Completed batch sizing for {len(stock_decisions)} stock and {len(option_decisions)} option decisions.")
        
        return sized_stock_decisions, sized_option_decisions