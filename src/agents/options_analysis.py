# src/agents/options_analysis.py
"""
Options Analysis Agent

Analyzes stock signals and converts them to options strategies,
then finds optimal contracts from the options chain.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import Literal
import numpy as np

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.integrations.polygon import PolygonClient, OptionContract, OptionsChain
from langchain_core.prompts import ChatPromptTemplate

# Initialize logger
logger = logging.getLogger(__name__)


class OptionsStrategy(BaseModel):
    """Model for an options trading strategy."""
    strategy_type: Literal[
        "long_call", "long_put", "covered_call", "cash_secured_put", 
        "bull_call_spread", "bear_put_spread", "iron_condor", "calendar_spread",
        "none"  # No suitable options strategy
    ]
    confidence: float = Field(description="Confidence in the strategy, between 0 and 100")
    reasoning: str = Field(description="Reasoning for the strategy selection")
    max_days_to_expiration: int = Field(description="Maximum days to expiration to consider")
    ideal_delta: float = Field(description="Ideal delta value to target")
    price_target: Optional[float] = Field(description="Price target for the underlying asset")
    stop_loss_percentage: float = Field(description="Stop loss percentage for risk management")
    take_profit_percentage: float = Field(description="Take profit percentage for risk management")
    max_position_size_pct: float = Field(description="Maximum position size as percentage of portfolio")


class OptionsContractDecision(BaseModel):
    """Model for a decision on a specific options contract."""
    ticker: str = Field(description="The option contract ticker")
    underlying_ticker: str = Field(description="The underlying stock ticker")
    action: Literal["buy", "sell", "close"] = Field(description="Action to take on the contract")
    option_type: Literal["call", "put"] = Field(description="Type of option (call or put)")
    strike_price: float = Field(description="Strike price of the option")
    expiration_date: str = Field(description="Expiration date of the option (YYYY-MM-DD)")
    strategy: str = Field(description="Strategy this contract is part of")
    confidence: float = Field(description="Confidence in the decision (0-100)")
    reasoning: str = Field(description="Reasoning for choosing this contract")
    limit_price: Optional[float] = Field(description="Limit price for the order")
    stop_loss: Optional[float] = Field(description="Stop loss price")
    take_profit: Optional[float] = Field(description="Take profit price")
    greeks: Dict[str, float] = Field(description="Option Greeks")


# Define the structure for a single leg within a multi-leg decision
class OptionLeg(BaseModel):
    ticker: str = Field(description="The option contract ticker for this leg")
    action: Literal["buy", "sell"] = Field(description="Action for this leg")
    option_type: Literal["call", "put"] = Field(description="Option type for this leg")
    strike_price: float = Field(description="Strike price for this leg")
    limit_price: Optional[float] = Field(description="Estimated limit price for this leg order")


# Define the structure for the overall multi-leg decision
class MultiLegOptionsDecision(BaseModel):
    underlying_ticker: str = Field(description="The underlying stock ticker")
    strategy: str = Field(description="The multi-leg strategy being employed (e.g., bear_put_spread)")
    confidence: float = Field(description="Confidence in the overall spread decision (0-100)")
    reasoning: str = Field(description="Reasoning for choosing this specific spread")
    legs: List[OptionLeg] = Field(description="List containing details for each leg of the spread")
    net_limit_price: Optional[float] = Field(description="Net limit price for the spread order (debit>0, credit<0), if applicable/calculable")
    stop_loss_on_underlying: Optional[float] = Field(description="Stop loss condition based on underlying price movement")
    take_profit_on_underlying: Optional[float] = Field(description="Take profit condition based on underlying price movement")


def options_analysis_agent(state: AgentState):
    """
    Analyze stock signals and determine optimal options strategies and contracts.
    
    This agent:
    1. Takes the stock trading decision and analyst signals
    2. Determines the best options strategy
    3. Fetches relevant options data from Polygon
    4. Analyzes contracts based on Greeks, liquidity, etc.
    5. Selects optimal contracts for the strategy
    6. Returns the final options trading decision
    """
    data = state["data"]
    # Extract relevant data from state
    tickers = data["tickers"]
    analyst_signals = data.get("analyst_signals", {})
    trading_decisions = data.get("decisions", {})
    
    # Initialize options decisions dictionary
    options_decisions = {}
    
    # Process each ticker with a stock decision
    for ticker in tickers:
        if ticker not in trading_decisions:
            continue
            
        progress.update_status("options_analysis_agent", ticker, "Analyzing stock signals")
        
        # Get stock decision
        stock_decision = trading_decisions[ticker]
        
        # Skip if no action or hold
        if stock_decision.get("action", "hold").lower() == "hold": 
            options_decisions[ticker] = {
                "action": "none",
                "reasoning": "Stock action is 'hold' - no options strategy applied"
            }
            continue
            
        # Prepare ticker-specific signals for options analysis
        ticker_signals = {
            agent: signals[ticker] for agent, signals in analyst_signals.items()
            if ticker in signals
        }
        
        # Determine options strategy using AI
        progress.update_status("options_analysis_agent", ticker, "Determining options strategy")
        options_strategy = determine_options_strategy(
            ticker=ticker,
            stock_decision=stock_decision,
            analyst_signals=ticker_signals,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        # Skip if no suitable options strategy
        if options_strategy.strategy_type == "none":
            options_decisions[ticker] = {
                "action": "none",
                "reasoning": options_strategy.reasoning
            }
            continue
            
        # Get options chain data from Polygon
        progress.update_status("options_analysis_agent", ticker, "Fetching options data")
        try:
            polygon_client = PolygonClient()
            filtered_contracts = polygon_client.filter_options_by_strategy(
                ticker=ticker,
                strategy=options_strategy.strategy_type,
                price_target=options_strategy.price_target,
                max_days_to_expiration=options_strategy.max_days_to_expiration,
                min_delta=options_strategy.ideal_delta * 0.5,  # Allow range around ideal delta
                min_liquidity_score=30,
                max_contracts=10
            )
            
            if not filtered_contracts:
                options_decisions[ticker] = {
                    "action": "none",
                    "reasoning": f"No suitable contracts found for {options_strategy.strategy_type} strategy"
                }
                continue
                
        except Exception as e:
            logger.error(f"Error fetching options data for {ticker}: {e}")
            options_decisions[ticker] = {
                "action": "none",
                "reasoning": f"Error fetching options data: {str(e)}"
            }
            continue
            
        # Analyze contracts and select optimal one
        progress.update_status("options_analysis_agent", ticker, "Selecting optimal contract")
        logger.info(f"Selecting optimal contract for strategy: {options_strategy.strategy_type}")
        
        contract_decision = select_optimal_contract(
            ticker=ticker,
            stock_decision=stock_decision,
            options_strategy=options_strategy,
            filtered_contracts_or_pairs=filtered_contracts,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        # Add to decisions
        if contract_decision:
            options_decisions[ticker] = contract_decision
        else:
            options_decisions[ticker] = {
                "action": "none",
                "reasoning": "Could not determine optimal contract"
            }
            
        progress.update_status("options_analysis_agent", ticker, "Done")
        
    # Create message
    message = HumanMessage(
        content=json.dumps(options_decisions),
        name="options_analysis_agent",
    )
    
    # Show reasoning if requested
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(options_decisions, "Options Analysis Agent")
        
    # Add options decisions to state
    state["data"]["options_decisions"] = options_decisions
    
    return {
        "messages": [message],
        "data": data,
    }


def determine_options_strategy(
    ticker: str,
    stock_decision: Dict[str, Any],
    analyst_signals: Dict[str, Dict[str, Any]],
    model_name: str,
    model_provider: str,
) -> OptionsStrategy:
    """
    Use LLM to determine the best options strategy based on stock decision and analyst signals.
    
    Args:
        ticker: Stock ticker
        stock_decision: Stock trading decision
        analyst_signals: Analyst signals for the ticker
        model_name: LLM model name
        model_provider: LLM provider
        
    Returns:
        OptionsStrategy object with the determined strategy
    """
    # Create prompt for generating options strategy
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an options trading strategist with deep expertise in derivatives analysis.
            
            Your task is to analyze a stock trading decision and corresponding analyst signals to determine the MOST APPROPRIATE options strategy.
            
            Consider the following factors:
            1. The stock trading decision (buy, sell, short, cover)
            2. The confidence level of the decision
            3. The collective signals from various analysts
            4. The volatility implications
            5. Risk/reward balance
            6. Time horizon implications
            
            If the signals are contradictory or unclear, or if options simply aren't appropriate, return "none" as the strategy_type.
            
            Available strategy types:
            - "long_call": Buy call options (bullish outlook, defined risk, leverage)
            - "long_put": Buy put options (bearish outlook, defined risk, leverage)
            - "covered_call": Sell call against owned stock (income strategy, reduced upside)
            - "cash_secured_put": Sell put secured by cash (income strategy, potential acquisition)
            - "bull_call_spread": Buy lower strike call, sell higher strike call (bullish, defined risk/reward)
            - "bear_put_spread": Buy higher strike put, sell lower strike put (bearish, defined risk/reward)
            - "iron_condor": Sell call spread and put spread (neutral, income strategy, range-bound)
            - "calendar_spread": Sell near-term, buy longer-term option (neutral/volatility play)
            - "none": No options strategy appropriate
            
            IMPORTANT: If the confidence level is below 60%, default to "none" as the options strategy, as the certainty isn't high enough to warrant options exposure.
            
            Return a detailed options strategy recommendation with the following fields:
            - strategy_type: One of the defined strategy types
            - confidence: 0-100 confidence in the strategy (never higher than stock confidence)
            - reasoning: Detailed explanation for the strategy selection
            - max_days_to_expiration: Recommended maximum days to expiration
            - ideal_delta: Target delta value to look for (0-1)
            - price_target: Price target for the underlying (null if none)
            - stop_loss_percentage: Recommended stop loss percentage
            - take_profit_percentage: Recommended take profit percentage
            - max_position_size_pct: Maximum position size as percentage of portfolio value
            """
        ),
        (
            "human",
            """Here is the stock trading decision and analyst signals for ticker {ticker}:
            
            Stock Decision:
            ```json
            {stock_decision}
            ```
            
            Analyst Signals:
            ```json
            {analyst_signals}
            ```
            
            Determine the optimal options strategy for this scenario and return a strategy recommendation in this exact JSON format:
            {{
                "strategy_type": "long_call|long_put|covered_call|cash_secured_put|bull_call_spread|bear_put_spread|iron_condor|calendar_spread|none",
                "confidence": float between 0 and 100,
                "reasoning": "string with detailed explanation",
                "max_days_to_expiration": integer (recommended DTE),
                "ideal_delta": float between 0 and 1,
                "price_target": float or null,
                "stop_loss_percentage": float,
                "take_profit_percentage": float,
                "max_position_size_pct": float between 0 and 1
            }}
            """
        )
    ])
    
    prompt = template.invoke({
        "ticker": ticker,
        "stock_decision": json.dumps(stock_decision, indent=2),
        "analyst_signals": json.dumps(analyst_signals, indent=2)
    })
    
    # Define default strategy
    def create_default_options_strategy():
        return OptionsStrategy(
            strategy_type="none",
            confidence=0.0,
            reasoning="Error in options strategy analysis, defaulting to no options strategy",
            max_days_to_expiration=30,
            ideal_delta=0.5,
            price_target=None,
            stop_loss_percentage=0.25,
            take_profit_percentage=0.5,
            max_position_size_pct=0.05
        )
    
    # Use LLM to determine strategy
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=OptionsStrategy,
        agent_name="options_strategy_agent",
        default_factory=create_default_options_strategy,
    )


def select_optimal_contract(
    ticker: str,
    stock_decision: Dict[str, Any],
    options_strategy: OptionsStrategy,
    filtered_contracts_or_pairs: Union[List[OptionContract], List[Tuple[OptionContract, OptionContract]]],
    model_name: str,
    model_provider: str,
) -> Optional[Dict[str, Any]]:
    """
    Select the optimal options contract or spread from filtered candidates.
    
    Args:
        ticker: Stock ticker
        stock_decision: Stock trading decision
        options_strategy: Options strategy determination
        filtered_contracts_or_pairs: List of filtered option contracts or pairs
        model_name: LLM model name
        model_provider: LLM provider
        
    Returns:
        Dictionary representing the final decision (either single leg or multi-leg)
    """
    if not filtered_contracts_or_pairs:
        return None
        
    # Determine if we are dealing with single contracts or pairs
    is_spread_strategy = options_strategy.strategy_type in ["bull_call_spread", "bear_put_spread", "iron_condor"]
    is_pair_list = isinstance(filtered_contracts_or_pairs[0], tuple) if filtered_contracts_or_pairs else False

    if is_spread_strategy and not is_pair_list:
        logger.error(f"Strategy {options_strategy.strategy_type} requires pairs, but received single contracts.")
        return None
    elif not is_spread_strategy and is_pair_list:
        logger.error(f"Strategy {options_strategy.strategy_type} expects single contracts, but received pairs.")
        return None

    # Prepare contract data for LLM based on single or pair structure
    contracts_data_for_llm = []
    if is_pair_list:
        for long_leg, short_leg in filtered_contracts_or_pairs:
            contracts_data_for_llm.append({
                "long_leg": long_leg.model_dump(exclude_none=True),
                "short_leg": short_leg.model_dump(exclude_none=True),
                "net_debit_credit_estimate": abs(long_leg.last_price - short_leg.last_price) # Simple estimate
            })
    else: # Single contracts
        for contract in filtered_contracts_or_pairs:
            contracts_data_for_llm.append(contract.model_dump(exclude_none=True))
    
    # --- Prompt Setup --- 
    # Adjust system prompt based on single vs spread
    system_prompt = """You are an options trading expert specializing in contract selection.

For SPREAD strategies (like Bull Call Spread, Bear Put Spread):
- Analyze the provided pairs (long leg, short leg).
- Evaluate the net debit/credit, max profit/loss, and breakeven point for each pair.
- Consider the spread width relative to volatility and target price.
- Select the pair that offers the best risk/reward profile aligned with the strategy.
- The final JSON should represent the chosen SPREAD, including details for BOTH legs.
"""

    if is_spread_strategy:
        candidate_data_header = "Candidate Spreads (Pairs)"
        selection_type = "spread pair"
        # Define the desired MULTI-LEG output format more clearly
        output_format_instruction = ("""Select the best pair based on risk/reward and strategy alignment.
Return ONLY the JSON object representing the chosen spread, adhering strictly to this format:

```json
{{
    "underlying_ticker": "{ticker}",
    "strategy": "{strategy_type}",
    "confidence": <float between 0 and 100>,
    "reasoning": "Detailed explanation for choosing this spread.",
    "legs": [
        {{
            "ticker": "<O:TICKER...>",
            "action": "buy",
            "option_type": "call" or "put",
            "strike_price": <float>,
            "limit_price": <float> // Estimated limit price for this leg
        }},
        {{
            "ticker": "<O:TICKER...>",
            "action": "sell",
            "option_type": "call" or "put",
            "strike_price": <float>,
            "limit_price": <float> // Estimated limit price for this leg
        }}
    ],
    "net_limit_price": <float or null>, // Estimated net debit (positive) or credit (negative)
    "stop_loss_on_underlying": <float or null>,
    "take_profit_on_underlying": <float or null>
}}
```

IMPORTANT: Ensure the output is a single, valid JSON object starting with `{{` and ending with `}}`, matching the structure above exactly. Do not include any text before or after the JSON object. Use the actual calculated or estimated values where indicated by `<...>`. Ensure `legs` is a list containing exactly two dictionaries, one for the buy leg and one for the sell leg. Verify correct brace placement.
""")
        pydantic_model_for_llm = MultiLegOptionsDecision # Use a new Pydantic model for spreads

    else: # Single leg
        candidate_data_header = "Filtered Contracts"
        selection_type = "contract"
        # Define the SINGLE-LEG output format (existing format) more clearly
        output_format_instruction = ("""Select the single best contract based on risk/reward and strategy alignment.
Return ONLY the JSON object representing the chosen contract, adhering strictly to this format:

```json
{{
    "ticker": "<O:TICKER...>",
    "underlying_ticker": "{ticker}",
    "action": "buy" or "sell" or "close",
    "option_type": "call" or "put",
    "strike_price": <float>,
    "expiration_date": "YYYY-MM-DD",
    "strategy": "{strategy_type}",
    "confidence": <float between 0 and 100>,
    "reasoning": "Detailed explanation for choosing this contract.",
    "limit_price": <float>,
    "stop_loss": <float or null>,
    "take_profit": <float or null>,
    "greeks": {{
        "delta": <float>,
        "gamma": <float>,
        "theta": <float>,
        "vega": <float>
    }}
}}
```

IMPORTANT: Ensure the output is a single, valid JSON object starting with `{{` and ending with `}}`, matching the structure above exactly. Do not include any text before or after the JSON object. Use the actual values where indicated by `<...>` Verify correct brace placement.
""")
        pydantic_model_for_llm = OptionsContractDecision # Use existing model for single leg

    # Create the full prompt
    # Combine system prompt and human prompt parts into the template
    # Note: Langchain templates handle parameter formatting
    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt), # General instructions + specific format rules based on strategy type
        ("human",
         """Here is the stock trading decision, options strategy, and candidate {candidate_data_header} for ticker {ticker}:

Stock Decision:
```json
{stock_decision}
```

Options Strategy:
```json
{options_strategy}
```

{candidate_data_header}:
```json
{contracts_data}
```

{output_format_instruction}
""")
    ])
    
    prompt = template.invoke({
        "ticker": ticker,
        "stock_decision": json.dumps(stock_decision, indent=2),
        "options_strategy": json.dumps(options_strategy.model_dump(), indent=2),
        "candidate_data_header": candidate_data_header,
        "contracts_data": json.dumps(contracts_data_for_llm, indent=2, default=str),
        "selection_type": selection_type,
        "strategy_type": options_strategy.strategy_type,
        "output_format_instruction": output_format_instruction
    })
    
    # Define default contract decision factory
    def create_default_decision():
        logger.error(f"LLM call failed in select_optimal_contract for {ticker}. Returning None.")
        return None
    
    # Use LLM to select the optimal contract/spread
    llm_result = call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=pydantic_model_for_llm, # Use the dynamically chosen model
        agent_name="options_contract_selection_agent",
        default_factory=create_default_decision,
    )

    # Return the result directly as a dictionary if valid
    if llm_result:
        # Convert to dictionary
        result_dict = llm_result.model_dump()
        
        # For multi-leg strategies, set the 'action' field to 'open_spread'
        # This ensures filtering doesn't skip it due to missing 'action'
        if is_spread_strategy:
            result_dict['action'] = 'open_spread'
            
        return result_dict
    else:
        return None