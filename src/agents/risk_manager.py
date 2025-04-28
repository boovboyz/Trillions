from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
# REMOVE: from tools.api import get_prices, prices_to_df
import json
# Import or get access to the PortfolioManager instance
# This depends on your application structure. Example:
# from src.portfolio.manager import portfolio_manager_instance as pm
# Or assume it's passed via state:
# pm = state["portfolio_manager"]
# Attempt to import safe_json_dumps, fallback to standard json
try:
    from src.integrations.alpaca import safe_json_dumps
except ImportError:
    safe_json_dumps = lambda obj, **kwargs: json.dumps(obj, default=str, **kwargs)


##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Provides risk context based on the PortfolioManager's state."""
    # Assume PortfolioManager instance 'pm' is accessible
    # This needs to be set up in your agent graph initialization
    if "portfolio_manager" not in state:
        # Try accessing through data if not top-level
        if "data" not in state or "portfolio_manager" not in state["data"]:
             raise ValueError("PortfolioManager instance not found in state or state['data'].")
        pm = state["data"]["portfolio_manager"]
    else:
        pm = state["portfolio_manager"]


    # --- Get Comprehensive State from PortfolioManager ---
    progress.update_status("risk_management_agent", "all", "Fetching portfolio state")
    # Force update to ensure latest data
    pm.update_portfolio_cache(force=True)
    portfolio_state = pm.get_portfolio_state()
    progress.update_status("risk_management_agent", "all", "Analyzing portfolio state")

    tickers = state["data"]["tickers"]
    data = state["data"] # Keep data reference for output
    risk_analysis = {}

    portfolio_value = portfolio_state.get('portfolio_value', 0)
    cash = portfolio_state.get('cash', 0)
    cash_reserve_pct = pm.config.get('cash_reserve_pct', 0)
    max_pos_pct = pm.config.get('max_position_size_pct', 0.20) # Use manager's config
    max_short_pos_pct = pm.config.get('max_short_position_size_pct', max_pos_pct) # Use manager's config
    enable_shorts = pm.config.get('enable_shorts', False)

    # --- Extract Overall Portfolio Risk Flags ---
    max_drawdown_reached = pm._check_drawdown_threshold(portfolio_state)
    today_trades = pm._get_today_trades() # Get trades for count
    max_trades_per_day = pm.config.get('max_trades_per_day', 10)
    max_trades_reached = len(today_trades) >= max_trades_per_day
    cash_below_reserve = cash < (portfolio_value * cash_reserve_pct)
    day_trades_remaining = portfolio_state.get('day_trades_remaining', 0)


    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Extracting risk context")

        # --- Get Ticker-Specific Data from PortfolioManager State ---
        current_price = 0
        # Access positions via the enhanced dictionary format from get_portfolio_state
        position_data = portfolio_state.get('positions', {}).get(ticker)
        if position_data:
            current_price = float(position_data.get('current_price', 0))
            current_position_value = abs(float(position_data.get('market_value', 0)))
            position_side = position_data.get('side')
        else:
             # If no position, try getting price from market data cache via PortfolioManager method
             try:
                 market_data = pm._get_market_data(ticker) # Use manager's method
                 current_price = float(market_data.get('price', 0))
             except Exception as e:
                 print(f"Risk Agent: Error getting market data for {ticker}: {e}")
                 current_price = 0 # Fallback
             current_position_value = 0
             position_side = None

        if current_price <= 0:
             progress.update_status("risk_management_agent", ticker, "Warning: Could not get current price")
             # Skip or handle tickers with no price? For now, set limits to 0.
             max_potential_buy_value = 0
             max_potential_short_value = 0
        else:
            # --- Calculate POTENTIAL Max Size based on Manager Config (NOT final trade size) ---
            # This provides a general guideline for the decision-making LLM
            max_value_limit = portfolio_value * max_pos_pct
            max_short_value_limit = portfolio_value * max_short_pos_pct

            # Potential new buy value allowed
            current_long_value = current_position_value if position_side == 'long' else 0
            potential_buy_allowance = max(0, max_value_limit - current_long_value)
            # Consider available cash after reserve
            effective_cash = max(0, cash - (portfolio_value * cash_reserve_pct))
            max_potential_buy_value = min(potential_buy_allowance, effective_cash)

            # Potential new short value allowed (if enabled)
            if enable_shorts:
                 current_short_value = current_position_value if position_side == 'short' else 0
                 potential_short_allowance = max(0, max_short_value_limit - current_short_value)
                 # Shorting also depends on margin, which is complex. Provide a basic value limit for now.
                 # A more accurate check would involve buying_power and margin requirements.
                 # Let's use buying_power as a rough upper bound along with the % limit.
                 buying_power = float(portfolio_state.get('buying_power', 0))
                 max_potential_short_value = min(potential_short_allowance, buying_power)
            else:
                 max_potential_short_value = 0


        risk_analysis[ticker] = {
            # Provide context useful for the synthesis LLM
            "current_price": float(current_price),
            "max_potential_buy_value": float(max_potential_buy_value), # Max $ value for a new buy order
            "max_potential_short_value": float(max_potential_short_value), # Max $ value for a new short order
            "manager_config": { # Expose relevant manager settings
                 "max_pos_pct": max_pos_pct,
                 "max_short_pos_pct": max_short_pos_pct,
                 "cash_reserve_pct": cash_reserve_pct,
                 "enable_shorts": enable_shorts,
            },
            "portfolio_context": { # Overall portfolio constraints
                "max_drawdown_reached": max_drawdown_reached,
                "max_trades_reached": max_trades_reached,
                "cash_below_reserve": cash_below_reserve,
                "day_trades_remaining": day_trades_remaining,
            }
            # Removed the old 'remaining_position_limit' which was less clear
        }
        progress.update_status("risk_management_agent", ticker, "Done")

    # --- Format Output ---
    content_json = safe_json_dumps(risk_analysis, indent=2)


    message = HumanMessage(
        content=content_json,
        name="risk_management_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    # Add the context to the analyst_signals (or maybe a new key like 'risk_context'?)
    # Ensure analyst_signals exists
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis
    # Or potentially: state["data"]["risk_context"] = risk_analysis

    return {
        # messages might not be needed if this agent doesn't directly talk to LLM
        "messages": state["messages"], # Keep structure for now
        "data": state["data"], # Return the modified data
    }
