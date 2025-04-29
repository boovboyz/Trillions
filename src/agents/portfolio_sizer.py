"""
Agent responsible for calculating position sizes for both stock and options trades,
considering portfolio state, risk parameters, and the combined nature of the trades.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import math

# Assuming these integrations exist and provide necessary methods
# We might need to adjust these based on actual implementations
from src.integrations.alpaca import AlpacaTrader # Assuming a class for Alpaca interaction
from src.integrations.polygon import PolygonClient # Assuming a class for Polygon interaction
# Need access to portfolio state, potentially manager or just its data methods
# from src.portfolio.manager import PortfolioManager

# Placeholder for data structures if not importing manager directly
PortfolioState = Dict[str, Any]
OptionsPortfolioState = Dict[str, Any]
PortfolioCache = Dict[str, Any] # Cache containing account, positions, orders

class PortfolioSizerAgent:
    # Added alpaca_client, polygon_client, portfolio_cache to __init__
    def __init__(self, config: Dict[str, Any], alpaca_client: AlpacaTrader, polygon_client: PolygonClient, portfolio_cache: PortfolioCache):
        """
        Initializes the PortfolioSizerAgent.

        Args:
            config: Configuration dictionary for sizing parameters
                    (e.g., max allocation, risk per trade, volatility adjustment).
            alpaca_client: Client for interacting with Alpaca (fetching prices, positions).
            polygon_client: Client for interacting with Polygon (fetching option prices).
            portfolio_cache: Access to cached portfolio data (account, positions, orders).
        """
        # Removed portfolio_manager dependency, assuming direct access to data/clients
        self.config = config
        self.alpaca = alpaca_client # Renamed for consistency with original code
        self.polygon = polygon_client
        self.portfolio_cache = portfolio_cache
        self.logger = logging.getLogger(__name__)
        self.logger.info("PortfolioSizerAgent initialized.")

    # Added specific state types to signature
    def calculate_combined_sizes(
        self,
        stock_decision: Dict[str, Any], # Expected: ticker, action, confidence, reasoning
        option_decision: Optional[Dict[str, Any]], # Expected: ticker, action, confidence, etc.
        portfolio_state: PortfolioState,
        options_portfolio_state: OptionsPortfolioState
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Calculates the appropriate position sizes for a stock decision and an optional
        accompanying option decision, considering current portfolio state and risk.

        Args:
            stock_decision: Dictionary containing the stock trade details (NO quantity).
            option_decision: Optional dictionary containing the option trade details (NO quantity).
            portfolio_state: Current state of the stock portfolio (cash, equity, positions, buying_power).
            options_portfolio_state: Current state of the options portfolio (positions).

        Returns:
            A tuple containing:
            - The sized stock decision (including quantity, skip_reason if skipped).
            - The sized option decision (including quantity, skip_reason if skipped), or None.
        """
        self.logger.info(f"Calculating combined sizes for stock: {stock_decision.get('ticker')} and option: {option_decision.get('ticker') if option_decision else 'None'}")

        sized_stock_decision = stock_decision.copy()
        sized_option_decision = option_decision.copy() if option_decision else None

        # Initialize quantity and potential skip reason
        sized_stock_decision['quantity'] = 0
        sized_stock_decision['skip_reason'] = None
        if sized_option_decision:
            sized_option_decision['quantity'] = 0
            sized_option_decision['skip_reason'] = None

        # --- Get Portfolio State ---
        portfolio_value = portfolio_state.get('portfolio_value', 0)
        cash = portfolio_state.get('cash', 0)
        buying_power = portfolio_state.get('buying_power', 0)
        # Attempt to get options-specific buying power, fall back to regular BP if not found
        options_buying_power = portfolio_state.get('options_buying_power', buying_power)
        stock_positions = portfolio_state.get('positions', {}) # {symbol: {qty: ..., side: ...}}
        options_positions = options_portfolio_state.get('positions', {}) # {option_ticker: {qty: ..., side: ...}}

        if portfolio_value <= 0:
            self.logger.warning("Portfolio value is zero or negative. Cannot size positions.")
            sized_stock_decision['skip_reason'] = "Zero or negative portfolio value"
            if sized_option_decision:
                sized_option_decision['skip_reason'] = "Zero or negative portfolio value"
            return sized_stock_decision, sized_option_decision

        # --- Get Pending Order Quantities (Stock Only - mirroring original logic) ---
        pending_quantities = self._get_pending_stock_quantities()

        # --- TODO: Implement Combined Sizing Strategy ---
        # This is where the core new logic goes. How much capital to allocate overall?
        # How to split between stock and option based on confidence, risk profile etc.?
        # For now, we will size them somewhat independently based on original logic,
        # but using a calculated target size instead of a 'requested_quantity'.

        # --- Stock Sizing ---
        stock_ticker = sized_stock_decision.get('ticker')
        stock_action = sized_stock_decision.get('action', 'hold').lower()
        stock_confidence = sized_stock_decision.get('confidence', 50.0) # Default confidence

        if stock_action != 'hold':
            current_stock_price = self._get_stock_price(stock_ticker)
            if current_stock_price is None:
                sized_stock_decision['skip_reason'] = f"Could not get current price for {stock_ticker}"
            else:
                target_stock_quantity = self._calculate_target_stock_quantity(
                    stock_ticker, stock_action, stock_confidence, current_stock_price, portfolio_value
                )
                final_stock_quantity = self._apply_stock_constraints(
                    stock_ticker, stock_action, target_stock_quantity, current_stock_price,
                    portfolio_value, cash, buying_power, stock_positions, pending_quantities
                )
                sized_stock_decision['quantity'] = final_stock_quantity
                # Add sizing info like in original? (Optional)
                # sized_stock_decision['sizing_info'] = {...}

        # --- Option Sizing ---
        if sized_option_decision:
            # Check if this is a multi-leg strategy
            is_multi_leg = 'legs' in sized_option_decision and isinstance(sized_option_decision['legs'], list)
            
            if is_multi_leg:
                # Handle multi-leg strategy (like bear put spread)
                final_option_quantity = self._calculate_multi_leg_option_quantity(
                    sized_option_decision, portfolio_value, cash, buying_power, cash_for_csp_check=cash
                )
                sized_option_decision['quantity'] = final_option_quantity
            else:
                # Single-leg options
                option_ticker = sized_option_decision.get('ticker')
                option_action = sized_option_decision.get('action', 'none').lower()
                option_confidence = sized_option_decision.get('confidence', 50.0) # Default confidence

                if option_action not in ['none', 'close']: # Size for opening trades (buy/sell)
                    current_option_price = self._get_option_price(option_ticker, sized_option_decision.get('limit_price'), options_positions)
                    if current_option_price is None:
                        sized_option_decision['skip_reason'] = f"Could not determine current price for {option_ticker}"
                        sized_option_decision['quantity'] = 0 # Ensure quantity is 0 if price fails
                    else:
                         target_option_quantity = self._calculate_target_option_quantity(
                             option_ticker, option_action, option_confidence, current_option_price, portfolio_value
                         )
                         final_option_quantity = self._apply_option_constraints(
                             option_ticker, option_action, target_option_quantity, current_option_price,
                             portfolio_value, cash, options_positions,
                             strike_price=sized_option_decision.get('strike_price'), 
                             option_type=sized_option_decision.get('option_type')
                         )
                         sized_option_decision['quantity'] = final_option_quantity

                elif option_action == 'close':
                    # Closing logic depends purely on existing position
                     existing_position = options_positions.get(option_ticker, None)
                     existing_contracts = abs(int(existing_position['qty'])) if existing_position else 0
                     sized_option_decision['quantity'] = existing_contracts # Close full position
                     self.logger.info(f"Sizing CLOSE action for {option_ticker} to existing quantity: {existing_contracts}")

        # --- Covered Call Share Purchase Check --- 
        # Check if the final option decision is a covered call sell and if we need to buy shares
        if sized_option_decision and \
           sized_option_decision.get('strategy') == 'covered_call' and \
           sized_option_decision.get('action') == 'sell' and \
           sized_option_decision.get('quantity', 0) > 0:

            option_quantity = sized_option_decision['quantity']
            required_shares = 100 * option_quantity
            underlying_ticker = sized_option_decision.get('underlying_ticker')
            
            if not underlying_ticker:
                self.logger.error("Covered call decision missing underlying_ticker. Cannot check/buy shares.")
            else:
                current_holdings = 0
                stock_position = stock_positions.get(underlying_ticker)
                if stock_position and stock_position.get('side') == 'long':
                    current_holdings = int(stock_position.get('qty', 0))
                
                shares_to_buy = required_shares - current_holdings

                if shares_to_buy > 0:
                    self.logger.info(f"Covered call strategy for {underlying_ticker} requires {required_shares} shares, holding {current_holdings}. Will attempt to buy {shares_to_buy}.")
                    
                    # Create a new stock buy decision (or potentially modify existing)
                    # Get current price for the stock
                    current_stock_price = self._get_stock_price(underlying_ticker)
                    if current_stock_price:
                        # Define the new buy decision
                        # Note: We don't use target_quantity here, just the required amount
                        buy_decision_for_cc = {
                            'ticker': underlying_ticker,
                            'action': 'buy',
                            'quantity': shares_to_buy, # This is the target quantity now
                            'confidence': sized_option_decision.get('confidence', 100.0), # Use option confidence
                            'reasoning': f'Auto-buying {shares_to_buy} shares for covered call strategy.'
                        }
                        
                        # IMPORTANT: Re-apply constraints to this buy decision
                        # We pass shares_to_buy as the target_quantity to check if it fits constraints
                        final_buy_quantity = self._apply_stock_constraints(
                            underlying_ticker, 'buy', shares_to_buy, current_stock_price,
                            portfolio_value, cash, buying_power, stock_positions, pending_quantities
                        )
                        
                        if final_buy_quantity == shares_to_buy:
                            # If constraints allow buying the required shares, override the original stock decision
                            self.logger.info(f"Constraints allow buying {shares_to_buy} shares for {underlying_ticker} covered call. Overriding initial stock decision.")
                            sized_stock_decision = buy_decision_for_cc # Use the constrained decision
                            sized_stock_decision['quantity'] = final_buy_quantity # Ensure quantity is set correctly
                        else:
                            # If we cannot buy the required shares, we MUST skip the covered call
                            self.logger.warning(f"Cannot execute covered call for {underlying_ticker}: Constraints limit needed stock buy from {shares_to_buy} to {final_buy_quantity}. Skipping option trade.")
                            sized_option_decision['quantity'] = 0
                            sized_option_decision['skip_reason'] = f'Cannot buy required {shares_to_buy} shares due to constraints (limited to {final_buy_quantity})'
                            # Reset stock decision to original (or hold if none existed)
                            # This part is tricky - what was the original? Let's just set it to hold if it wasn't a buy already.
                            if sized_stock_decision is None or sized_stock_decision.get('action') != 'buy':
                                 sized_stock_decision = {'ticker': underlying_ticker, 'action': 'hold', 'quantity': 0, 'reasoning': 'Option skipped, reverting stock decision'}
                    else:
                        self.logger.error(f"Cannot buy shares for {underlying_ticker} covered call: Could not get stock price. Skipping option trade.")
                        sized_option_decision['quantity'] = 0
                        sized_option_decision['skip_reason'] = 'Failed to get stock price for share purchase'

        # --- End Covered Call Check --- 

        self.logger.info(f"Final Sized Stock Decision: {sized_stock_decision}")
        if sized_option_decision:
             self.logger.info(f"Final Sized Option Decision: {sized_option_decision}")

        return sized_stock_decision, sized_option_decision


    def _get_stock_price(self, symbol: str) -> Optional[float]:
        """Helper to get current stock price."""
        try:
            # Assuming alpaca client has a method like get_latest_trade
            latest_trade = self.alpaca.get_latest_trade(symbol)
            price = latest_trade['price']
            if price <= 0:
                 self.logger.warning(f"Received non-positive price {price} for {symbol}")
                 return None
            return float(price)
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def _get_option_price(self, option_ticker: str, limit_price: Optional[float], options_positions: Dict) -> Optional[float]:
        """Helper to get current option price."""
        current_price = 0
        # Priority: Existing position data -> Limit price -> Polygon lookup
        if option_ticker in options_positions and 'current_price' in options_positions[option_ticker]:
             current_price = float(options_positions[option_ticker]['current_price'])
        elif limit_price and limit_price > 0:
             current_price = limit_price
        else:
             try:
                 # Assuming polygon client has a method like get_option_contract_details
                 contract_details = self.polygon.get_option_contract_details(option_ticker)
                 # Use ask for buys, bid for sells? Or just last/mid? Using last/ask for now.
                 current_price = contract_details.last_price or contract_details.ask
             except Exception as e:
                 self.logger.error(f"Could not get current price for {option_ticker} via Polygon: {e}")

        if current_price <= 0:
             self.logger.warning(f"Could not determine positive price for {option_ticker}")
             return None
        return float(current_price)


    def _calculate_target_stock_quantity(self, symbol: str, action: str, confidence: float, current_price: float, portfolio_value: float) -> int:
         """Calculate the initial target quantity based on risk and confidence."""
         if current_price <= 0: return 0

         risk_fraction = self.config.get('risk_fraction_per_trade', 0.01) # Base risk fraction
         confidence_factor = max(0.1, confidence / 100.0) # Scale confidence (0 to 1), minimum 0.1

         # Adjust risk based on confidence
         adjusted_risk_fraction = risk_fraction * confidence_factor

         target_value = portfolio_value * adjusted_risk_fraction
         target_quantity = int(target_value / current_price)

         self.logger.debug(f"Target stock qty for {symbol} ({action}): {target_quantity} (Value: ${target_value:.2f}, RiskFrac: {adjusted_risk_fraction:.4f}, Conf: {confidence:.1f})")
         return max(0, target_quantity)

    def _calculate_target_option_quantity(self, option_ticker: str, action: str, confidence: float, current_price: float, portfolio_value: float) -> int:
         """Calculate the initial target quantity for options."""
         if current_price <= 0: return 0

         # Use a potentially different risk fraction for options?
         risk_fraction = self.config.get('risk_fraction_per_option_trade', self.config.get('risk_fraction_per_trade', 0.01))
         confidence_factor = max(0.1, confidence / 100.0)
         adjusted_risk_fraction = risk_fraction * confidence_factor

         target_value = portfolio_value * adjusted_risk_fraction
         # Price is per share, contract involves 100 shares
         target_quantity = int(target_value / (current_price * 100))

         self.logger.debug(f"Target option qty for {option_ticker} ({action}): {target_quantity} (Value: ${target_value:.2f}, RiskFrac: {adjusted_risk_fraction:.4f}, Conf: {confidence:.1f})")
         return max(0, target_quantity)


    def _apply_stock_constraints(self, symbol: str, action: str, target_quantity: int, current_price: float, portfolio_value: float, cash: float, buying_power: float, positions: Dict, pending_quantities: Dict) -> int:
        """Apply portfolio constraints to the target stock quantity."""
        # Target quantity is relevant for buy/short actions
        # For sell/cover, we primarily care about existing position size

        # Price check is still relevant for sell/cover if needed elsewhere, but not for basic sizing logic below
        if current_price <= 0 and action in ['buy', 'short']:
            self.logger.warning(f"Cannot size {action} for {symbol} with non-positive price.")
            return 0

        # Get pending quantities for the current symbol
        symbol_pending_buy = pending_quantities.get(symbol, {}).get('buy', 0)
        symbol_pending_sell = pending_quantities.get(symbol, {}).get('sell', 0)

        # Calculate position limits (needed for buy/short)
        max_pos_pct = self.config.get('max_position_size_pct', 0.10)
        max_order_pct = self.config.get('max_single_order_size_pct', 0.05)
        max_short_pos_pct = self.config.get('max_short_position_size_pct', max_pos_pct)

        max_position_value = portfolio_value * max_pos_pct
        max_order_value = portfolio_value * max_order_pct
        max_short_position_value = portfolio_value * max_short_pos_pct

        # --- Volatility Adjustment (from original - applies mainly to buy/short limits) ---
        if self.config.get('volatility_adjustment', False) and action in ['buy', 'short']:
            volatility_factor = 1.0
            try:
                # Placeholder - needs implementation
                volatility = 25 
                volatility_threshold = self.config.get('volatility_threshold', 20.0)
                volatility_scale = self.config.get('volatility_scale_factor', 0.05)
                min_vol_factor = self.config.get('min_volatility_factor', 0.2)

                if volatility > volatility_threshold:
                    volatility_factor = 1.0 - (volatility_scale * (volatility - volatility_threshold) / 1.0)
                    volatility_factor = max(min_vol_factor, volatility_factor)
                    self.logger.debug(f"Applying volatility factor {volatility_factor:.2f} for {symbol} (vol={volatility:.1f})")
                    max_position_value *= volatility_factor
                    max_order_value *= volatility_factor
                    max_short_position_value *= volatility_factor
            except Exception as e:
                self.logger.warning(f"Could not calculate volatility for {symbol}: {e}")
        # --- End Volatility Adjustment ---

        max_position_shares = int(max_position_value / current_price) if current_price > 0 else 0
        max_order_shares = int(max_order_value / current_price) if current_price > 0 else 0
        max_short_position_shares = int(max_short_position_value / current_price) if current_price > 0 else 0

        final_quantity = 0
        existing_position = positions.get(symbol, None)

        if action == 'buy':
            if target_quantity <= 0: return 0 # No target size means no buy
            existing_shares = int(existing_position['qty']) if existing_position and existing_position.get('side') == 'long' else 0
            effective_max_position_shares = max_position_shares - symbol_pending_buy
            shares_to_target = effective_max_position_shares - existing_shares
            max_shares_by_cash = int(cash / current_price) if current_price > 0 else 0
            allowed_new_shares = min(shares_to_target, max_order_shares, max_shares_by_cash)
            allowed_new_shares = max(0, allowed_new_shares)
            final_quantity = min(target_quantity, allowed_new_shares)

        elif action == 'sell':
            existing_shares = int(existing_position['qty']) if existing_position and existing_position.get('side') == 'long' else 0
            if existing_shares <= 0:
                self.logger.warning(f"Attempted to SELL {symbol} but no existing long position found.")
                return 0 # Cannot sell if not holding long
            effective_existing_shares = existing_shares - symbol_pending_sell
            effective_existing_shares = max(0, effective_existing_shares)
            # SELL action implies closing the existing position. Quantity is the amount available.
            final_quantity = effective_existing_shares
            # Optionally, could consider confidence here for partial closes, but default is full close.
            # Example partial close: final_quantity = min(target_quantity, effective_existing_shares)
            # IF target_quantity is calculated appropriately for a close action (e.g., % of existing based on confidence)

        elif action == 'short':
            if target_quantity <= 0: return 0 # No target size means no short
            existing_shares = abs(int(existing_position['qty'])) if existing_position and existing_position.get('side') == 'short' else 0
            effective_max_short_shares = max_short_position_shares - symbol_pending_sell
            shares_to_target = effective_max_short_shares - existing_shares
            max_shares_by_bp = int((buying_power / 2) / current_price) if current_price > 0 else 0
            allowed_new_shares = min(shares_to_target, max_order_shares, max_shares_by_bp)
            allowed_new_shares = max(0, allowed_new_shares)
            final_quantity = min(target_quantity, allowed_new_shares)

        elif action == 'cover':
            existing_shares = abs(int(existing_position['qty'])) if existing_position and existing_position.get('side') == 'short' else 0
            if existing_shares <= 0:
                self.logger.warning(f"Attempted to COVER {symbol} but no existing short position found.")
                return 0 # Cannot cover if not holding short
            effective_existing_shares = existing_shares - symbol_pending_buy # Covering reduces short, affected by pending buys/covers
            effective_existing_shares = max(0, effective_existing_shares)
            # COVER action implies closing the existing short position. Quantity is the amount available.
            final_quantity = effective_existing_shares 
            # Optionally, could consider confidence here for partial covers.

        # Ensure final quantity is an integer
        final_quantity = int(final_quantity)
        self.logger.debug(f"Constrained stock qty for {symbol} ({action}): {final_quantity} (Target: {target_quantity if action in ['buy', 'short'] else 'N/A - Closing'}) (Existing: {existing_shares if existing_position else 0})")
        return final_quantity

    def _apply_option_constraints(self, option_ticker: str, action: str, target_quantity: int, current_price: float, portfolio_value: float, cash: float, options_positions: Dict, strike_price: Optional[float] = None, option_type: Optional[str] = None) -> int:
        """Apply portfolio constraints to the target option quantity."""
        if target_quantity <= 0 or current_price <= 0:
            return 0

        contract_value = current_price * 100

        # Calculate limits
        max_pos_pct = self.config.get('max_options_position_size_pct', 0.05) # Max value in one option ticker
        max_order_pct = self.config.get('max_single_order_size_pct', 0.10) # Max value in one order (shared with stocks?)
        # max_total_options_pct = self.config.get('max_options_allocation_pct', 0.15) # Overall max in options

        max_position_value = portfolio_value * max_pos_pct
        max_order_value = portfolio_value * max_order_pct
        # max_total_options_value = portfolio_value * max_total_options_pct

        # TODO: Check total options allocation if needed (requires summing current options market value)

        max_position_contracts = int(max_position_value / contract_value) if contract_value > 0 else 0
        max_order_contracts = int(max_order_value / contract_value) if contract_value > 0 else 0

        final_quantity = 0
        existing_position = options_positions.get(option_ticker, None)

        if action == 'buy':
            existing_contracts = int(existing_position['qty']) if existing_position and existing_position.get('side') == 'long' else 0
            # Note: Original options sizing didn't check pending orders. Adding this might be complex.
            contracts_to_target = max_position_contracts - existing_contracts
            max_contracts_by_cash = int(cash / contract_value) if contract_value > 0 else 0
            allowed_new_contracts = min(contracts_to_target, max_order_contracts, max_contracts_by_cash)
            allowed_new_contracts = max(0, allowed_new_contracts)
            final_quantity = min(target_quantity, allowed_new_contracts)

        elif action == 'sell': # Sell to open
             existing_contracts = abs(int(existing_position['qty'])) if existing_position and existing_position.get('side') == 'short' else 0
             # Use short position limits if defined, otherwise standard position limits
             max_short_pos_pct = self.config.get('max_short_position_size_pct', max_pos_pct)
             # Consider if short option limit should be different/combined with stock short limits
             max_short_position_value = portfolio_value * max_short_pos_pct * self.config.get('max_options_allocation_pct', 0.15) # Similar combination as original
             max_short_position_contracts = int(max_short_position_value / contract_value) if contract_value > 0 else 0

             # --- Add CSP Check for Selling Puts --- 
             max_contracts_by_cash = float('inf') # Default to no cash limit
             if option_type == 'put' and strike_price and strike_price > 0:
                 required_cash_per_contract = strike_price * 100
                 if required_cash_per_contract > 0:
                     max_contracts_by_cash = int(cash / required_cash_per_contract)
                     self.logger.debug(f"CSP Check for selling put {option_ticker}: Strike={strike_price}, CashReqPerCont=${required_cash_per_contract:.2f}, AvailCash=${cash:.2f}, MaxContractsByCash={max_contracts_by_cash}")
                 else:
                      self.logger.warning(f"CSP Check for selling put {option_ticker}: Invalid required cash per contract ({required_cash_per_contract}). Skipping cash constraint.")
             elif option_type == 'put':
                  self.logger.warning(f"CSP Check for selling put {option_ticker}: Strike price missing or invalid. Skipping cash constraint.")
             # --- End CSP Check ---
             
             # Apply constraints (Position Limit, Order Limit, Cash Limit for Puts)
             allowed_new_contracts = min(contracts_to_target, max_order_contracts, max_contracts_by_cash)
             allowed_new_contracts = max(0, allowed_new_contracts)
             final_quantity = min(target_quantity, allowed_new_contracts)

        # 'close' action is handled directly in the main method

        self.logger.debug(f"Constrained option qty for {option_ticker} ({action}): {final_quantity} (Target: {target_quantity})")
        return final_quantity


    def _get_pending_stock_quantities(self) -> Dict[str, Dict[str, float]]:
        """Extract pending stock order quantities from the portfolio cache."""
        pending_quantities = {}
        try:
            all_orders = self.portfolio_cache.get('orders', [])
            if not isinstance(all_orders, list):
                 self.logger.warning(f"Expected 'orders' in cache to be a list, got {type(all_orders)}. Returning empty pending quantities.")
                 return {}

            pending_statuses = {'new', 'pending_new', 'accepted', 'partially_filled', 'held', 'calculated', 'pending_cancel', 'pending_replace'}
            pending_orders = [o for o in all_orders if isinstance(o, dict) and o.get("status") in pending_statuses]

            for order in pending_orders:
                 symbol = order.get('symbol')
                 # Filter out options symbols if necessary (e.g., check for 'O:')
                 if not symbol or symbol.startswith('O:'): continue

                 side = order.get('side')
                 try:
                      qty = abs(float(order.get('qty', 0)))
                      filled_qty = abs(float(order.get('filled_qty', 0)))
                      pending_qty = qty - filled_qty
                 except (ValueError, TypeError):
                      self.logger.warning(f"Could not parse quantity for pending order: {order}")
                      continue

                 if pending_qty <= 0: continue

                 if symbol not in pending_quantities:
                      pending_quantities[symbol] = {'buy': 0.0, 'sell': 0.0}

                 if side in ['buy', 'cover']:
                      pending_quantities[symbol]['buy'] += pending_qty
                 elif side in ['sell', 'short']:
                      pending_quantities[symbol]['sell'] += pending_qty
        except Exception as e:
             self.logger.error(f"Error processing pending orders from cache: {e}", exc_info=True)

        self.logger.debug(f"Pending stock quantities: {pending_quantities}")
        return pending_quantities

    # --- Placeholder for volatility calculation if needed ---
    # def _calculate_volatility(self, symbol: str) -> float:
    #     # Implement or call appropriate method from clients/utils
    #     self.logger.warning(f"Volatility calculation not implemented in sizer for {symbol}. Returning default.")
    #     return 25.0 # Placeholder default


    def _calculate_target_stock_quantity(self, symbol: str, action: str, confidence: float, current_price: float, portfolio_value: float) -> int:
         """Calculate the initial target quantity based on risk and confidence."""
         if current_price <= 0: return 0

         risk_fraction = self.config.get('risk_fraction_per_trade', 0.01) # Base risk fraction
         confidence_factor = max(0.1, confidence / 100.0) # Scale confidence (0 to 1), minimum 0.1

         # Adjust risk based on confidence
         adjusted_risk_fraction = risk_fraction * confidence_factor

         target_value = portfolio_value * adjusted_risk_fraction
         target_quantity = int(target_value / current_price)

         self.logger.debug(f"Target stock qty for {symbol} ({action}): {target_quantity} (Value: ${target_value:.2f}, RiskFrac: {adjusted_risk_fraction:.4f}, Conf: {confidence:.1f})")
         return max(0, target_quantity)

    def _calculate_target_option_quantity(self, option_ticker: str, action: str, confidence: float, current_price: float, portfolio_value: float) -> int:
         """Calculate the initial target quantity for options."""
         if current_price <= 0: return 0

         # Use a potentially different risk fraction for options?
         risk_fraction = self.config.get('risk_fraction_per_option_trade', self.config.get('risk_fraction_per_trade', 0.01))
         confidence_factor = max(0.1, confidence / 100.0)
         adjusted_risk_fraction = risk_fraction * confidence_factor

         target_value = portfolio_value * adjusted_risk_fraction
         # Price is per share, contract involves 100 shares
         target_quantity = int(target_value / (current_price * 100))

         self.logger.debug(f"Target option qty for {option_ticker} ({action}): {target_quantity} (Value: ${target_value:.2f}, RiskFrac: {adjusted_risk_fraction:.4f}, Conf: {confidence:.1f})")
         return max(0, target_quantity)


    def _apply_stock_constraints(self, symbol: str, action: str, target_quantity: int, current_price: float, portfolio_value: float, cash: float, buying_power: float, positions: Dict, pending_quantities: Dict) -> int:
        """Apply portfolio constraints to the target stock quantity."""
        # Target quantity is relevant for buy/short actions
        # For sell/cover, we primarily care about existing position size

        # Price check is still relevant for sell/cover if needed elsewhere, but not for basic sizing logic below
        if current_price <= 0 and action in ['buy', 'short']:
            self.logger.warning(f"Cannot size {action} for {symbol} with non-positive price.")
            return 0

        # Get pending quantities for the current symbol
        symbol_pending_buy = pending_quantities.get(symbol, {}).get('buy', 0)
        symbol_pending_sell = pending_quantities.get(symbol, {}).get('sell', 0)

        # Calculate position limits (needed for buy/short)
        max_pos_pct = self.config.get('max_position_size_pct', 0.10)
        max_order_pct = self.config.get('max_single_order_size_pct', 0.05)
        max_short_pos_pct = self.config.get('max_short_position_size_pct', max_pos_pct)

        max_position_value = portfolio_value * max_pos_pct
        max_order_value = portfolio_value * max_order_pct
        max_short_position_value = portfolio_value * max_short_pos_pct

        # --- Volatility Adjustment (from original - applies mainly to buy/short limits) ---
        if self.config.get('volatility_adjustment', False) and action in ['buy', 'short']:
            volatility_factor = 1.0
            try:
                # Placeholder - needs implementation
                volatility = 25 
                volatility_threshold = self.config.get('volatility_threshold', 20.0)
                volatility_scale = self.config.get('volatility_scale_factor', 0.05)
                min_vol_factor = self.config.get('min_volatility_factor', 0.2)

                if volatility > volatility_threshold:
                    volatility_factor = 1.0 - (volatility_scale * (volatility - volatility_threshold) / 1.0)
                    volatility_factor = max(min_vol_factor, volatility_factor)
                    self.logger.debug(f"Applying volatility factor {volatility_factor:.2f} for {symbol} (vol={volatility:.1f})")
                    max_position_value *= volatility_factor
                    max_order_value *= volatility_factor
                    max_short_position_value *= volatility_factor
            except Exception as e:
                self.logger.warning(f"Could not calculate volatility for {symbol}: {e}")
        # --- End Volatility Adjustment ---

        max_position_shares = int(max_position_value / current_price) if current_price > 0 else 0
        max_order_shares = int(max_order_value / current_price) if current_price > 0 else 0
        max_short_position_shares = int(max_short_position_value / current_price) if current_price > 0 else 0

        final_quantity = 0
        existing_position = positions.get(symbol, None)

        if action == 'buy':
            if target_quantity <= 0: return 0 # No target size means no buy
            existing_shares = int(existing_position['qty']) if existing_position and existing_position.get('side') == 'long' else 0
            effective_max_position_shares = max_position_shares - symbol_pending_buy
            shares_to_target = effective_max_position_shares - existing_shares
            max_shares_by_cash = int(cash / current_price) if current_price > 0 else 0
            allowed_new_shares = min(shares_to_target, max_order_shares, max_shares_by_cash)
            allowed_new_shares = max(0, allowed_new_shares)
            final_quantity = min(target_quantity, allowed_new_shares)

        elif action == 'sell':
            existing_shares = int(existing_position['qty']) if existing_position and existing_position.get('side') == 'long' else 0
            if existing_shares <= 0:
                self.logger.warning(f"Attempted to SELL {symbol} but no existing long position found.")
                return 0 # Cannot sell if not holding long
            effective_existing_shares = existing_shares - symbol_pending_sell
            effective_existing_shares = max(0, effective_existing_shares)
            # SELL action implies closing the existing position. Quantity is the amount available.
            final_quantity = effective_existing_shares
            # Optionally, could consider confidence here for partial closes, but default is full close.
            # Example partial close: final_quantity = min(target_quantity, effective_existing_shares)
            # IF target_quantity is calculated appropriately for a close action (e.g., % of existing based on confidence)

        elif action == 'short':
            if target_quantity <= 0: return 0 # No target size means no short
            existing_shares = abs(int(existing_position['qty'])) if existing_position and existing_position.get('side') == 'short' else 0
            effective_max_short_shares = max_short_position_shares - symbol_pending_sell
            shares_to_target = effective_max_short_shares - existing_shares
            max_shares_by_bp = int((buying_power / 2) / current_price) if current_price > 0 else 0
            allowed_new_shares = min(shares_to_target, max_order_shares, max_shares_by_bp)
            allowed_new_shares = max(0, allowed_new_shares)
            final_quantity = min(target_quantity, allowed_new_shares)

        elif action == 'cover':
            existing_shares = abs(int(existing_position['qty'])) if existing_position and existing_position.get('side') == 'short' else 0
            if existing_shares <= 0:
                self.logger.warning(f"Attempted to COVER {symbol} but no existing short position found.")
                return 0 # Cannot cover if not holding short
            effective_existing_shares = existing_shares - symbol_pending_buy # Covering reduces short, affected by pending buys/covers
            effective_existing_shares = max(0, effective_existing_shares)
            # COVER action implies closing the existing short position. Quantity is the amount available.
            final_quantity = effective_existing_shares 
            # Optionally, could consider confidence here for partial covers.

        # Ensure final quantity is an integer
        final_quantity = int(final_quantity)
        self.logger.debug(f"Constrained stock qty for {symbol} ({action}): {final_quantity} (Target: {target_quantity if action in ['buy', 'short'] else 'N/A - Closing'}) (Existing: {existing_shares if existing_position else 0})")
        return final_quantity

    def _apply_option_constraints(self, option_ticker: str, action: str, target_quantity: int, current_price: float, portfolio_value: float, cash: float, options_positions: Dict, strike_price: Optional[float] = None, option_type: Optional[str] = None) -> int:
        """Apply portfolio constraints to the target option quantity."""
        if target_quantity <= 0 or current_price <= 0:
            return 0

        contract_value = current_price * 100

        # Calculate limits
        max_pos_pct = self.config.get('max_options_position_size_pct', 0.05) # Max value in one option ticker
        max_order_pct = self.config.get('max_single_order_size_pct', 0.10) # Max value in one order (shared with stocks?)
        # max_total_options_pct = self.config.get('max_options_allocation_pct', 0.15) # Overall max in options

        max_position_value = portfolio_value * max_pos_pct
        max_order_value = portfolio_value * max_order_pct
        # max_total_options_value = portfolio_value * max_total_options_pct

        # TODO: Check total options allocation if needed (requires summing current options market value)

        max_position_contracts = int(max_position_value / contract_value) if contract_value > 0 else 0
        max_order_contracts = int(max_order_value / contract_value) if contract_value > 0 else 0

        final_quantity = 0
        existing_position = options_positions.get(option_ticker, None)

        if action == 'buy':
            existing_contracts = int(existing_position['qty']) if existing_position and existing_position.get('side') == 'long' else 0
            # Note: Original options sizing didn't check pending orders. Adding this might be complex.
            contracts_to_target = max_position_contracts - existing_contracts
            max_contracts_by_cash = int(cash / contract_value) if contract_value > 0 else 0
            allowed_new_contracts = min(contracts_to_target, max_order_contracts, max_contracts_by_cash)
            allowed_new_contracts = max(0, allowed_new_contracts)
            final_quantity = min(target_quantity, allowed_new_contracts)

        elif action == 'sell': # Sell to open
             existing_contracts = abs(int(existing_position['qty'])) if existing_position and existing_position.get('side') == 'short' else 0
             # Use short position limits if defined, otherwise standard position limits
             max_short_pos_pct = self.config.get('max_short_position_size_pct', max_pos_pct)
             # Consider if short option limit should be different/combined with stock short limits
             max_short_position_value = portfolio_value * max_short_pos_pct * self.config.get('max_options_allocation_pct', 0.15) # Similar combination as original
             max_short_position_contracts = int(max_short_position_value / contract_value) if contract_value > 0 else 0

             # --- Add CSP Check for Selling Puts --- 
             max_contracts_by_cash = float('inf') # Default to no cash limit
             if option_type == 'put' and strike_price and strike_price > 0:
                 required_cash_per_contract = strike_price * 100
                 if required_cash_per_contract > 0:
                     max_contracts_by_cash = int(cash / required_cash_per_contract)
                     self.logger.debug(f"CSP Check for selling put {option_ticker}: Strike={strike_price}, CashReqPerCont=${required_cash_per_contract:.2f}, AvailCash=${cash:.2f}, MaxContractsByCash={max_contracts_by_cash}")
                 else:
                      self.logger.warning(f"CSP Check for selling put {option_ticker}: Invalid required cash per contract ({required_cash_per_contract}). Skipping cash constraint.")
             elif option_type == 'put':
                  self.logger.warning(f"CSP Check for selling put {option_ticker}: Strike price missing or invalid. Skipping cash constraint.")
             # --- End CSP Check ---
             
             # Apply constraints (Position Limit, Order Limit, Cash Limit for Puts)
             allowed_new_contracts = min(contracts_to_target, max_order_contracts, max_contracts_by_cash)
             allowed_new_contracts = max(0, allowed_new_contracts)
             final_quantity = min(target_quantity, allowed_new_contracts)

        # 'close' action is handled directly in the main method

        self.logger.debug(f"Constrained option qty for {option_ticker} ({action}): {final_quantity} (Target: {target_quantity})")
        return final_quantity

    # --- ADD SPREAD SIZING LOGIC ---
    def _calculate_target_spread_quantity(self, spread_decision: Dict[str, Any], portfolio_value: float) -> int:
        """Calculate the initial target quantity for an option spread based on its net cost/risk."""
        if not spread_decision.get('legs') or len(spread_decision['legs']) != 2:
            self.logger.error("Cannot calculate spread quantity: Invalid or missing legs.")
            return 0

        # Estimate net debit/credit (assuming limit prices are populated)
        leg1_price = spread_decision['legs'][0].get('limit_price', 0)
        leg2_price = spread_decision['legs'][1].get('limit_price', 0)
        leg1_action = spread_decision['legs'][0].get('action', '')
        # leg2_action = spread_decision['legs'][1].get('action', '') # Not strictly needed if we know spread type

        # Determine net cost (risk for debit spreads, max profit for credit spreads - use net debit for risk calc for now)
        # Assumes leg 0 is buy, leg 1 is sell for bear put / bull call
        net_debit = 0
        if leg1_action == 'buy':
            net_debit = leg1_price - leg2_price
        else: # Assuming leg 0 is sell
             net_debit = leg2_price - leg1_price # Should be negative (credit)

        if net_debit <= 0:
            # For credit spreads, risk is difference in strikes minus credit.
            # For simplicity now, we'll skip sizing purely based on credit received.
            # Or, we could use a small assumed risk value or margin requirement.
            # Let's prevent trading credit spreads if net_debit calc is zero or negative for now.
            self.logger.warning(f"Calculated net debit <= 0 ({net_debit:.2f}) for spread {spread_decision.get('underlying_ticker')} {spread_decision.get('strategy')}. Cannot size based on debit risk. Skipping.")
            return 0

        spread_risk_per_contract = net_debit * 100
        if spread_risk_per_contract <= 0:
             self.logger.warning(f"Spread risk per contract is zero or negative for {spread_decision.get('underlying_ticker')}. Skipping.")
             return 0

        # Use a potentially different risk fraction for spreads?
        risk_fraction = self.config.get('risk_fraction_per_spread_trade', self.config.get('risk_fraction_per_trade', 0.01))
        confidence = spread_decision.get('confidence', 50.0)
        confidence_factor = max(0.1, confidence / 100.0)
        adjusted_risk_fraction = risk_fraction * confidence_factor

        target_risk_value = portfolio_value * adjusted_risk_fraction
        target_quantity = int(target_risk_value / spread_risk_per_contract)

        self.logger.debug(f"Target spread qty for {spread_decision.get('underlying_ticker')} {spread_decision.get('strategy')}: {target_quantity} (RiskValue: ${target_risk_value:.2f}, SpreadRisk: ${spread_risk_per_contract:.2f}, RiskFrac: {adjusted_risk_fraction:.4f}, Conf: {confidence:.1f})")
        return max(0, target_quantity)

    def _apply_spread_constraints(self, spread_decision: Dict[str, Any], target_quantity: int, portfolio_value: float, cash: float, options_positions: Dict) -> int:
        """Apply portfolio constraints to the target spread quantity."""
        if target_quantity <= 0:
            return 0

        # --- Calculate Spread Cost/Risk (Similar to target calc) ---
        leg1_price = spread_decision['legs'][0].get('limit_price', 0)
        leg2_price = spread_decision['legs'][1].get('limit_price', 0)
        leg1_action = spread_decision['legs'][0].get('action', '')
        net_debit = 0
        if leg1_action == 'buy': net_debit = leg1_price - leg2_price
        else: net_debit = leg2_price - leg1_price

        if net_debit <= 0:
             # Cannot apply cash constraint properly for credit spreads without margin info
             # For now, assume if target_quantity > 0, it passed the initial debit check
             self.logger.warning(f"Net debit is zero/negative ({net_debit:.2f}) during constraint check for {spread_decision.get('underlying_ticker')}. Cash constraint might be inaccurate.")
             # Use a nominal cost for constraint checks if debit is non-positive? Or rely on margin checks later.
             spread_cost_per_contract = 1 # Nominal cost to avoid division by zero, assumes margin handles risk
        else:
             spread_cost_per_contract = net_debit * 100

        if spread_cost_per_contract <= 0:
             self.logger.error(f"Spread cost per contract is zero or negative during constraint check. Cannot size.")
             return 0
        # --- End Spread Cost Calc ---

        # Calculate limits based on overall spread portfolio value/risk
        # These percentages might need tuning specifically for spreads
        max_pos_pct = self.config.get('max_options_position_size_pct', 0.05) # Limit per underlying/strategy?
        max_order_pct = self.config.get('max_single_order_size_pct', 0.10)
        max_total_options_pct = self.config.get('max_options_allocation_pct', 0.15)

        max_position_value_allowed = portfolio_value * max_pos_pct # Max value allowed in this specific spread
        max_order_value_allowed = portfolio_value * max_order_pct # Max value per single transaction
        max_total_options_value_allowed = portfolio_value * max_total_options_pct

        # --- Check Total Options Allocation --- TODO: Needs current options market value sum
        # current_total_options_value = sum(pos.get('market_value', 0) for pos in options_positions.values())
        # available_options_allocation = max(0, max_total_options_value_allowed - current_total_options_value)
        # max_contracts_by_total_alloc = int(available_options_allocation / spread_cost_per_contract) if spread_cost_per_contract > 0 else 0
        max_contracts_by_total_alloc = float('inf') # Placeholder: Ignore total allocation limit for now
        # --- End Total Options Allocation Check ---

        max_contracts_by_position_limit = int(max_position_value_allowed / spread_cost_per_contract) if spread_cost_per_contract > 0 else 0
        max_contracts_by_order_limit = int(max_order_value_allowed / spread_cost_per_contract) if spread_cost_per_contract > 0 else 0
        max_contracts_by_cash = int(cash / spread_cost_per_contract) if spread_cost_per_contract > 0 else 0

        # --- Check Existing Position --- TODO: Needs better way to track spread positions
        # Simple check: if any leg exists, assume no increase for now.
        # A better approach would track spreads as units.
        leg1_ticker = spread_decision['legs'][0].get('ticker')
        leg2_ticker = spread_decision['legs'][1].get('ticker')
        existing_contracts = 0
        if (leg1_ticker in options_positions) or (leg2_ticker in options_positions):
            # This is inaccurate - doesn't represent the spread qty.
            # For simplicity, prevent adding to existing legs for now.
            self.logger.warning(f"Existing legs found for spread {spread_decision.get('underlying_ticker')}. Preventing increase. Manual closing required.")
            return 0 # Prevent increasing position if legs exist
            # existing_leg1 = options_positions.get(leg1_ticker, {})
            # existing_leg2 = options_positions.get(leg2_ticker, {})
            # existing_contracts = min(abs(int(existing_leg1.get('qty', 0))), abs(int(existing_leg2.get('qty', 0))))

        contracts_to_target = max_contracts_by_position_limit - existing_contracts

        # Apply constraints
        # For debit spreads, cash is the primary constraint.
        # For credit spreads, margin is the constraint (not checked here).
        allowed_new_contracts = min(
            contracts_to_target,
            max_contracts_by_order_limit,
            max_contracts_by_cash if net_debit > 0 else float('inf'), # Only check cash for debit spreads
            max_contracts_by_total_alloc
        )
        allowed_new_contracts = max(0, allowed_new_contracts)
        final_quantity = min(target_quantity, allowed_new_contracts)

        self.logger.debug(f"Constrained spread qty for {spread_decision.get('underlying_ticker')} {spread_decision.get('strategy')}: {final_quantity} (Target: {target_quantity}) (Constraints: Pos={max_contracts_by_position_limit}, Ord={max_contracts_by_order_limit}, Cash={max_contracts_by_cash if net_debit > 0 else 'N/A'})")
        return final_quantity 

    def _calculate_multi_leg_option_quantity(self, spread_decision: Dict[str, Any], portfolio_value: float, cash: float, buying_power: float, cash_for_csp_check: float) -> int:
        """
        Calculate the quantity for a multi-leg options strategy (like spreads).
        Relies on internal config for risk limits, not input max_position_value.
        Includes an optional check to simulate potential CSP margin requirements on sell put legs (for paper trading issues).
        
        Args:
            spread_decision: The spread decision dictionary including legs
            portfolio_value: Current portfolio value
            cash: Available cash (used for general sizing/reserve check)
            buying_power: Available buying power (potentially used for other constraints)
            cash_for_csp_check: Available cash to use for the simulated CSP check
            
        Returns:
            Quantity of spreads to trade
        """
        if not spread_decision or not spread_decision.get('legs') or portfolio_value <= 0:
            return 0
            
        # Use the net debit price if available
        net_debit = spread_decision.get('net_limit_price')
        
        # If net_debit is not provided, calculate it from the legs
        if net_debit is None:
            net_debit = 0
            for leg in spread_decision['legs']:
                leg_action = leg.get('action', '').lower()
                leg_price = leg.get('limit_price', 0)
                
                if leg_action == 'buy':
                    net_debit += leg_price
                elif leg_action == 'sell':
                    net_debit -= leg_price
        
        # For safety, ensure net_debit is positive for sizing debit spreads
        # Credit spreads (net_debit <= 0) sizing relies on margin, not handled here.
        # We assume only debit spreads are sized by this function for now.
        if net_debit <= 0:
            self.logger.warning(f"Net debit <= 0 ({net_debit:.2f}) for spread {spread_decision.get('underlying_ticker')} {spread_decision.get('strategy')}. Cannot size debit spread based on risk. Returning 0.")
            return 0
            
        # Calculate risk per contract (100 shares per contract) for debit spread
        spread_risk_per_contract = net_debit * 100
        if spread_risk_per_contract <= 0:
             self.logger.error(f"Spread risk per contract is zero or negative ({spread_risk_per_contract:.2f}). Cannot size.")
             return 0
        
        # --- Calculate Risk/Allocation Limits --- 
        # Use a risk-based approach based on config and confidence
        risk_fraction = self.config.get('risk_fraction_per_spread', 0.01)
        confidence = spread_decision.get('confidence', 50.0)
        confidence_factor = max(0.1, confidence / 100.0)
        adjusted_risk_fraction = risk_fraction * confidence_factor
        
        target_risk_value = portfolio_value * adjusted_risk_fraction
        
        # Apply overall position size limit based on config
        max_allocation_pct = self.config.get('max_options_position_size_pct', 0.05)
        max_allowed_value_per_pos = portfolio_value * max_allocation_pct
        
        # Apply cash constraint (minus reserve)
        cash_reserve_pct = self.config.get('cash_reserve_pct', 0.1)
        available_cash = cash * (1 - cash_reserve_pct)
        
        # Determine the maximum value we can allocate based on target risk, position limits, and available cash
        usable_value = min(target_risk_value, max_allowed_value_per_pos, available_cash)
        
        # Calculate quantity based on usable value and spread risk
        quantity = int(usable_value / spread_risk_per_contract) if spread_risk_per_contract > 0 else 0
        initial_calculated_quantity = quantity # Store initial calculation
        
        # --- Check for Cash-Secured Put requirements for put spreads ---
        strategy = spread_decision.get('strategy', '').lower()
        if 'put' in strategy and quantity > 0:
            # Find the short put leg
            short_put_leg = None
            for leg in spread_decision.get('legs', []):
                if leg.get('action') == 'sell' and leg.get('option_type') == 'put':
                    short_put_leg = leg
                    break
                    
            if short_put_leg:
                short_put_strike = short_put_leg.get('strike_price', 0)
                if short_put_strike > 0:
                    # Calculate the buying power required for cash-secured put
                    csp_requirement = short_put_strike * 100 * quantity
                    
                    # For bear put spreads, we need to account for collateral on the short put
                    if csp_requirement > cash_for_csp_check:
                        self.logger.warning(
                            f"Cash-secured put requirement for {spread_decision.get('underlying_ticker')} bear put spread exceeds available cash: " +
                            f"Required: ${csp_requirement:,.2f}, Available: ${cash_for_csp_check:,.2f}. Reducing quantity."
                        )
                        
                        # Calculate max quantity based on available cash
                        max_qty_by_csp = int(cash_for_csp_check / (short_put_strike * 100)) if short_put_strike > 0 else 0
                        
                        # Reduce quantity to match cash available for CSP requirement
                        quantity = min(quantity, max_qty_by_csp)
                        quantity = max(0, quantity)
                        
                        # Additional safety factor for paper trading
                        safety_factor = 0.8  # Use 80% of calculated max to allow for market fluctuations
                        quantity = int(quantity * safety_factor)
                        
                        self.logger.warning(f"Reduced quantity to {quantity} based on cash-secured put requirements")
        
        # Log the calculation details (using initial quantity before optional check)
        self.logger.info(
            f"Sizing multi-leg {spread_decision.get('strategy')} for {spread_decision.get('underlying_ticker')}: "
            f"SpreadRisk=${spread_risk_per_contract:.2f}, TargetRiskVal=${target_risk_value:.2f}, MaxPosVal=${max_allowed_value_per_pos:.2f}, "
            f"AvailCash=${available_cash:.2f} -> UsableVal=${usable_value:.2f} -> InitialQty={initial_calculated_quantity} -> FinalQty={quantity}"
        )

        # Ensure we have at least 1 contract if we're trading at all and quantity was > 0 before integer conversion AND optional reduction
        # (Check usable_value against risk to avoid sizing up due to rounding)
        if usable_value >= spread_risk_per_contract and initial_calculated_quantity == 0 and quantity == 0:
            quantity = 1 # Ensure at least 1 contract if calculation was just below 1
            self.logger.info(f"Adjusting quantity to 1 as initial calculation was slightly below threshold but usable value permits.")
        elif quantity < 0:
             quantity = 0 # Should not happen, but safety check
        
        return quantity 