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
        "long_call", "long_put", 
        "none"  # Keep 'none' for cases where no strategy is appropriate
    ]
    confidence: float = Field(description="Confidence in the strategy, between 0 and 100")
    reasoning: str = Field(description="Reasoning for the strategy selection")
    max_days_to_expiration: Optional[int] = Field(
        description="Maximum days to expiration to consider",
        default=30  # Default value used when None is provided
    )
    ideal_delta: Optional[float] = Field(
        description="Ideal delta value to target",
        default=0.5  # Default value used when None is provided
    )
    price_target: Optional[float] = Field(
        description="Price target for the underlying asset",
        default=None
    )
    stop_loss_percentage: Optional[float] = Field(
        description="Stop loss percentage for risk management",
        default=0.15  # Default value used when None is provided
    )
    take_profit_percentage: Optional[float] = Field(
        description="Take profit percentage for risk management",
        default=0.25  # Default value used when None is provided
    )
    max_position_size_pct: Optional[float] = Field(
        description="Maximum position size as percentage of portfolio",
        default=0.05  # Default value used when None is provided
    )
    
    def __init__(self, **data):
        # If strategy_type is "none", set default values for required fields
        if data.get("strategy_type") == "none":
            # Set default values for numeric fields if they're None
            if data.get("max_days_to_expiration") is None:
                data["max_days_to_expiration"] = 30
            if data.get("ideal_delta") is None:
                data["ideal_delta"] = 0.5
            if data.get("stop_loss_percentage") is None:
                data["stop_loss_percentage"] = 0.15
            if data.get("take_profit_percentage") is None:
                data["take_profit_percentage"] = 0.25
            if data.get("max_position_size_pct") is None:
                data["max_position_size_pct"] = 0.05
        super().__init__(**data)


class OptionsContractDecision(BaseModel):
    """Model for a decision on a specific options contract."""
    ticker: str = Field(description="The option contract ticker")
    underlying_ticker: str = Field(description="The underlying stock ticker")
    action: Literal["buy", "sell", "close", "hold"] = Field(description="Action to take on the contract")
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


# Define the model for a single leg within a multi-leg strategy
class OptionLeg(BaseModel):
    ticker: str = Field(description="The option contract ticker for this leg")
    action: Literal["buy", "sell"] = Field(description="Action for this leg")
    option_type: Literal["call", "put"] = Field(description="Type of option for this leg")
    strike_price: float = Field(description="Strike price of this leg")
    limit_price: Optional[float] = Field(description="Estimated limit price for this leg")


class OptionsAnalysis(BaseModel):
    """Model for the overall options analysis output for a ticker."""


def options_analysis_agent(state: AgentState):
    """
    Analyze stock signals and determine optimal options strategies and contracts.
    
    This agent:
    1. Takes the stock trading decision and analyst signals
    2. Determines the best options strategy
    3. Fetches relevant options data from Polygon
    4. Analyzes contracts based on Greeks, liquidity, etc.
    5. Analyzes current option positions
    6. Selects optimal contracts for the strategy
    7. Returns the final options trading decision
    """
    data = state["data"]
    # Extract relevant data from state
    tickers = data["tickers"]
    analyst_signals = data.get("analyst_signals", {})
    trading_decisions = data.get("decisions", {})
    
    # Initialize options decisions dictionary
    options_decisions = {}
    
    # Get current portfolio state for position analysis
    try:
        from src.portfolio.manager import PortfolioManager
        portfolio_manager = PortfolioManager(config={})
        current_portfolio = portfolio_manager.get_portfolio_state()
        options_portfolio = portfolio_manager.get_options_portfolio_state()
    except Exception as e:
        logger.error(f"Error getting portfolio state: {e}")
        current_portfolio = {"positions": {}}
        options_portfolio = {"positions": {}}
    
    # Process each ticker with a stock decision
    for ticker in tickers:
        if ticker not in trading_decisions:
            continue
            
        progress.update_status("options_analysis_agent", ticker, "Analyzing stock signals")
        
        # Get stock decision
        stock_decision = trading_decisions[ticker]
        
        # Prepare ticker-specific signals for options analysis
        ticker_signals = {
            agent: signals[ticker] for agent, signals in analyst_signals.items()
            if ticker in signals
        }
        
        # Get current positions for this underlying
        current_stock_position = current_portfolio.get("positions", {}).get(ticker)
        # Find options positions related to this underlying
        current_option_positions = {}
        for option_symbol, position in options_portfolio.get("positions", {}).items():
            if position.get("underlying") == ticker:
                current_option_positions[option_symbol] = position
        
        # Determine options strategy using AI, passing current positions
        progress.update_status("options_analysis_agent", ticker, "Determining options strategy")
        options_strategy = determine_options_strategy(
            ticker=ticker,
            analyst_signals=ticker_signals,
            current_stock_position=current_stock_position,
            current_option_positions=current_option_positions,
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
                min_liquidity_score=30
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
            options_strategy=options_strategy,
            filtered_contracts=filtered_contracts,
            current_option_positions=current_option_positions,
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
    analyst_signals: Dict[str, Dict[str, Any]],
    current_stock_position: Optional[Dict[str, Any]],
    current_option_positions: Dict[str, Dict[str, Any]],
    model_name: str,
    model_provider: str,
) -> OptionsStrategy:
    """
    Use LLM to determine the best options strategy based on analyst signals,
    and current positions.
    
    Args:
        ticker: Stock ticker
        analyst_signals: Analyst signals for the ticker
        current_stock_position: Current stock position (if exists)
        current_option_positions: Current option positions for this underlying
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
            
            Your task is to analyze analyst signals and current positions to determine the MOST APPROPRIATE options strategy.
            
            Consider the following factors:
            1. The collective signals from various analysts
            2. Current stock and options positions: Evaluate how current positions align with, contradict, or could be hedged by a new options strategy. Consider if a new strategy would create undue concentration, complement existing holdings, or if existing options could be rolled or adjusted.
            3. The volatility implications
            4. Risk/reward balance
            5. Time horizon implications
            
            If the signals are contradictory or unclear, or if options simply aren't appropriate for the current market conditions or portfolio state, return "none" as the strategy_type.
            
            Available strategy types:
            - "long_call": Buy call options (bullish outlook, defined risk, leverage)
            - "long_put": Buy put options (bearish outlook, defined risk, leverage)
            - "none": No options strategy appropriate
            
            Return a detailed options strategy recommendation with the following fields:
            - strategy_type: One of the defined strategy types
            - confidence: 0-100 confidence in the strategy
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
            """IMPORTANT: PROVIDE ALL OUTPUT IN ENGLISH ONLY. DO NOT USE ANY NON-ENGLISH CHARACTERS OR TEXT.

Here is the analyst signals and current positions for ticker {ticker}:

Analyst Signals:
```json
{analyst_signals}
```

Current Stock Position:
```json
{current_stock_position}
```

Current Option Positions:
```json
{current_option_positions}
```

Determine the optimal options strategy for this scenario and return a strategy recommendation in this exact JSON format:
{{
    "strategy_type": "long_call|long_put|none",
    "confidence": float between 0 and 100,
    "reasoning": "string with detailed explanation",
    "max_days_to_expiration": integer (recommended DTE),
    "ideal_delta": float between 0 and 1,
    "price_target": float or null,
    "stop_loss_percentage": float,
    "take_profit_percentage": float,
    "max_position_size_pct": float between 0 and 1
}}

IMPORTANT: If you select "none" as the strategy_type because no options strategy is appropriate, you must still include values for all fields. For numeric fields, use:
- max_days_to_expiration: 30
- ideal_delta: 0.5
- stop_loss_percentage: 0.15
- take_profit_percentage: 0.25
- max_position_size_pct: 0.05

Do not return null values for these fields when strategy_type is "none".
""")
    ])
    
    prompt = template.invoke({
        "ticker": ticker,
        "analyst_signals": json.dumps(analyst_signals, indent=2),
        "current_stock_position": json.dumps(current_stock_position, indent=2) if current_stock_position else "null",
        "current_option_positions": json.dumps(current_option_positions, indent=2)
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
            stop_loss_percentage=0.15,
            take_profit_percentage=0.25,
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
    options_strategy: OptionsStrategy,
    filtered_contracts: List[OptionContract],
    current_option_positions: Dict[str, Dict[str, Any]],
    model_name: str,
    model_provider: str,
) -> Optional[Dict[str, Any]]:
    """
    Select the optimal options contract from filtered candidates.
    
    Args:
        ticker: Stock ticker
        options_strategy: Options strategy determination
        filtered_contracts: List of filtered option contracts
        current_option_positions: Current option positions for this underlying
        model_name: LLM model name
        model_provider: LLM provider
        
    Returns:
        Dictionary representing the final decision
    """
    if not filtered_contracts:
        return None
    
    # Sort contracts by liquidity score (highest first) to prioritize the most liquid contracts
    sorted_contracts = sorted(filtered_contracts, key=lambda c: getattr(c, 'liquidity_score', 0), reverse=True)
    
    # Limit the number of contracts sent to LLM to prevent overloading it
    # We still get the benefit of all contracts being considered in filtering,
    # but only send the top ones to the LLM for final selection
    llm_max_contracts = 100
    contracts_for_llm = sorted_contracts[:llm_max_contracts]
    logger.info(f"Limiting LLM input to top {llm_max_contracts} contracts out of {len(filtered_contracts)} total filtered contracts for {ticker}")
    
    # Prepare contract data for LLM
    contracts_data_for_llm = []
    for contract in contracts_for_llm:
        contracts_data_for_llm.append(contract.model_dump(exclude_none=True))
    
    # System prompt for contract selection
    system_prompt = """You are an options trading expert specializing in contract selection. YOU MUST WRITE ALL OUTPUT IN ENGLISH ONLY. 

Analyze the provided option contracts to select the one that best aligns with the determined strategy.
Consider the following factors:
1. Delta alignment with the ideal_delta from the strategy
2. Balance between time decay (theta) and time to expiration
3. Implied volatility relative to historical volatility
4. Liquidity and bid-ask spread
5. Risk-reward profile given the target price
6. Current option positions (avoid duplicate strategies, consider closing or adjusting existing positions)
7. Overall portfolio construction and how this specific contract selection fits within the broader strategy suggested by analyst signals and existing holdings.

For existing positions, consider:
- If they align with the current strategy, recommend holding or adjusting
- If they contradict the current strategy, consider closing positions
- Look for opportunities to average down cost basis or take profits on winning positions

CRITICAL INSTRUCTIONS:
1. You MUST provide comprehensive reasoning of consistent length for ALL contracts, regardless of ticker. Do not abbreviate your analysis for any ticker. Your reasoning must be at least 150 words for each contract and cover all the key factors listed above. Consistency in detail across all tickers is essential.
2. ALL OUTPUT MUST BE IN ENGLISH ONLY. Do not include any non-English characters or symbols.
3. Use only actual data from the provided contracts. DO NOT HALLUCINATE or make up values that don't exist in the data.
4. All numeric values must be valid numbers without any text mixed in. For example, use "0.5" not "0.5%" or "also0.5".
5. Ensure proper JSON formatting with no errors or duplicate fields.
"""

    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         """IMPORTANT: PROVIDE ALL OUTPUT IN ENGLISH ONLY. DO NOT USE ANY NON-ENGLISH CHARACTERS OR TEXT.

Here is the options strategy, current positions, and candidate contracts for ticker {ticker}:

Options Strategy:
```json
{options_strategy}
```

Current Option Positions:
```json
{current_option_positions}
```

Filtered Contracts:
```json
{contracts_data}
```

Select the single best contract based on risk/reward and strategy alignment.

Here is an example of the exact JSON format you must return:
```json
{{
    "ticker": "O:XYZ241231C00100000",
    "underlying_ticker": "XYZ",
    "action": "buy",
    "option_type": "call",
    "strike_price": 100.0,
    "expiration_date": "2024-12-31",
    "strategy": "long_call",
    "confidence": 85.5,
    "reasoning": "This contract was selected due to its optimal delta of 0.6, which closely aligns with our ideal delta target of 0.65 for this strategy. The expiration date of 2024-12-31 provides a 45-day timeframe that balances time decay concerns with sufficient duration for the expected price movement. The strike price is positioned strategically at 100.0, just 5% above the current price, offering an attractive risk-reward profile given our bullish outlook and price target of 115.0. The contract's implied volatility of 32% is reasonably valued compared to the 30-day historical volatility of 28%, suggesting fair pricing. Liquidity metrics are strong with a tight bid-ask spread of only $0.15 and daily volume exceeding 500 contracts. This supports ease of entry and exit. The theta value of -0.02 represents manageable time decay relative to the delta, indicating good time value preservation. The contract's vega of 0.1 provides moderate sensitivity to volatility changes, which aligns with our expectation of stable to slightly increasing volatility. Overall, this contract provides an optimal balance of directional exposure, reasonable premium cost, and favorable Greek metrics for our current market thesis and portfolio positioning.",
    "limit_price": 1.25,
    "stop_loss": 0.80,
    "take_profit": 2.50,
    "greeks": {{
        "delta": 0.6,
        "gamma": 0.05,
        "theta": -0.02,
        "vega": 0.1
    }}
}}
```

Now, based on the data provided for {ticker}, select the best contract.
Return ONLY the JSON object representing the chosen contract, adhering strictly to this format:

```json
{{
    "ticker": "<O:TICKER...>",
    "underlying_ticker": "{ticker}",
    "action": "buy" or "sell" or "close" or "hold",
    "option_type": "call" or "put",
    "strike_price": <float, e.g., 150.0 or 45.50>,  // MUST be a numerical value only. DO NOT include any other characters, words, or symbols.
    "expiration_date": "YYYY-MM-DD",
    "strategy": "{strategy_type}",
    "confidence": <float between 0 and 100>,
    "reasoning": "Detailed explanation for choosing this contract that MUST be at least 150 words long. You MUST include analysis of: 1) why this specific strike price and expiration date is optimal, 2) discussion of the contract's delta and how it aligns with the strategy, 3) evaluation of the contract's liquidity and implied volatility, 4) risk-reward assessment, and 5) how this contract fits with the overall market outlook. Your reasoning MUST be comprehensive and consistent across all tickers.",
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

IMPORTANT: Ensure the output is a single, valid JSON object starting with `{{` and ending with `}}`, matching the structure above exactly. Do not include any text before or after the JSON object.

CRITICAL FORMATTING REQUIREMENTS:
1. ALL numeric values must be plain numbers without units or text (e.g., use 0.5 not "0.5%" or "approximately 0.5")
2. Ensure expiration_date is in YYYY-MM-DD format exactly
3. Use only ASCII characters in ALL fields, especially in the reasoning field
4. Do not include any duplicate fields in the JSON
5. Use proper JSON syntax with quotes around all string values
""")
    ])
    
    prompt = template.invoke({
        "ticker": ticker,
        "options_strategy": json.dumps(options_strategy.model_dump(), indent=2),
        "current_option_positions": json.dumps(current_option_positions, indent=2),
        "contracts_data": json.dumps(contracts_data_for_llm, indent=2, default=str),
        "strategy_type": options_strategy.strategy_type
    })
    
    # Define default contract decision factory
    def create_default_decision():
        logger.error(f"LLM call failed in select_optimal_contract for {ticker}. Returning None.")
        return None
    
    # Use LLM to select the optimal contract
    llm_result = call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=OptionsContractDecision,
        agent_name="options_contract_selection_agent",
        default_factory=create_default_decision,
    )

    # Return the result directly as a dictionary if valid
    if llm_result:
        try:
            # Convert to dictionary
            result_dict = llm_result.model_dump()
            
            # Sanitize the reasoning to ensure it's valid English text
            if "reasoning" in result_dict:
                # Remove any non-ASCII characters that could cause issues
                import re
                reasoning = result_dict["reasoning"]
                # Replace non-ASCII characters with spaces
                sanitized_reasoning = re.sub(r'[^\x00-\x7F]+', ' ', reasoning)
                # Fix any double spaces created in the process
                sanitized_reasoning = re.sub(r'\s+', ' ', sanitized_reasoning).strip()
                result_dict["reasoning"] = sanitized_reasoning
            
            # Validate numeric fields are actually numeric
            numeric_fields = ["confidence", "strike_price", "limit_price", "stop_loss", "take_profit"]
            for field in numeric_fields:
                if field in result_dict and result_dict[field] is not None:
                    try:
                        # Ensure it's a valid float
                        result_dict[field] = float(result_dict[field])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid numeric value for {field}, defaulting to None")
                        result_dict[field] = None
            
            # Validate greeks are actually numeric
            if "greeks" in result_dict and isinstance(result_dict["greeks"], dict):
                for greek, value in list(result_dict["greeks"].items()):
                    if value is not None:
                        try:
                            result_dict["greeks"][greek] = float(value)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid numeric value for greek {greek}, defaulting to 0.0")
                            result_dict["greeks"][greek] = 0.0
            
            # Validate that reasoning meets minimum length requirements
            min_words = 100  # Aim for minimum 100 words
            reasoning = result_dict.get("reasoning", "")
            word_count = len(reasoning.split())
            
            if word_count < min_words:
                logger.warning(f"Short reasoning detected for {ticker} ({word_count} words, minimum {min_words} expected). This may result in inconsistent analysis detail.")
                
                # Append note about truncated reasoning for transparency
                if reasoning:
                    result_dict["reasoning"] = f"{reasoning}\n\nNote: This analysis appears to be shorter than expected ({word_count} words). Future updates will provide more comprehensive analysis."
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Error processing LLM result for {ticker}: {e}")
            # Return simplified valid contract with minimal information
            return {
                "ticker": llm_result.ticker if hasattr(llm_result, "ticker") else f"O:{ticker}",
                "underlying_ticker": ticker,
                "action": "buy",
                "option_type": options_strategy.strategy_type.replace("long_", ""),
                "strike_price": getattr(filtered_contracts[0], "strike_price", 0.0) if filtered_contracts else 0.0,
                "expiration_date": getattr(filtered_contracts[0], "expiration_date", datetime.now()).strftime("%Y-%m-%d") if filtered_contracts else datetime.now().strftime("%Y-%m-%d"),
                "strategy": options_strategy.strategy_type,
                "confidence": 50.0,
                "reasoning": "Error processing the detailed analysis. Using a simplified recommendation based on the strategy type and available contracts. The original analysis contained errors or invalid formatting.",
                "limit_price": getattr(filtered_contracts[0], "last_price", 0.0) if filtered_contracts else 0.0,
                "stop_loss": None,
                "take_profit": None,
                "greeks": {
                    "delta": 0.0,
                    "gamma": 0.0,
                    "theta": 0.0,
                    "vega": 0.0
                }
            }
    else:
        return None
