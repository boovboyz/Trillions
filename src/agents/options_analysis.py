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
    quantity: int = Field(description="Number of contracts to trade")
    option_type: Literal["call", "put"] = Field(description="Type of option (call or put)")
    strike_price: float = Field(description="Strike price of the option")
    expiration_date: str = Field(description="Expiration date of the option (YYYY-MM-DD)")
    strategy: str = Field(description="Strategy this contract is part of")
    confidence: float = Field(description="Confidence in the decision (0-100)")
    reasoning: str = Field(description="Reasoning for choosing this contract")
    limit_price: Optional[float] = Field(description="Limit price for the order")
    stop_loss: Optional[float] = Field(description="Stop loss price")
    take_profit: Optional[float] = Field(description="Take profit price")
    max_position_value: float = Field(description="Maximum position value in dollars")
    greeks: Dict[str, float] = Field(description="Option Greeks")


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
        if stock_decision.get("action", "hold").lower() == "hold" or stock_decision.get("quantity", 0) <= 0:
            options_decisions[ticker] = {
                "action": "none",
                "reasoning": "No stock action or holding position - no options strategy applied"
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
        contract_decision = select_optimal_contract(
            ticker=ticker,
            stock_decision=stock_decision,
            options_strategy=options_strategy,
            filtered_contracts=filtered_contracts,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        # Add to decisions
        if contract_decision:
            options_decisions[ticker] = contract_decision.model_dump()
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
    filtered_contracts: List[OptionContract],
    model_name: str,
    model_provider: str,
) -> Optional[OptionsContractDecision]:
    """
    Select the optimal options contract from filtered candidates.
    
    Args:
        ticker: Stock ticker
        stock_decision: Stock trading decision
        options_strategy: Options strategy determination
        filtered_contracts: List of filtered option contracts
        model_name: LLM model name
        model_provider: LLM provider
        
    Returns:
        OptionsContractDecision with the selected contract or None if no suitable contract
    """
    if not filtered_contracts:
        return None
        
    # Prepare contract data for LLM
    contracts_data = []
    for contract in filtered_contracts:
        contracts_data.append({
            "ticker": contract.ticker,
            "strike_price": contract.strike_price,
            "expiration_date": contract.expiration_date.strftime("%Y-%m-%d"),
            "option_type": contract.option_type,
            "last_price": contract.last_price,
            "bid": contract.bid,
            "ask": contract.ask,
            "volume": contract.volume,
            "open_interest": contract.open_interest,
            "implied_volatility": contract.implied_volatility,
            "delta": contract.delta,
            "gamma": contract.gamma,
            "theta": contract.theta,
            "vega": contract.vega,
            "time_to_expiration": contract.time_to_expiration,
            "liquidity_score": contract.liquidity_score,
            "intrinsic_value": contract.intrinsic_value,
            "extrinsic_value": contract.extrinsic_value,
            "in_the_money": contract.in_the_money
        })
    
    # Create prompt for selecting optimal contract
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an options trading expert specializing in contract selection.
            
            Your task is to analyze a list of option contracts and select the optimal one that
            best fits the given strategy and stock decision. Focus on these key factors:
            
            1. Liquidity (tight bid-ask spread, high volume/open interest)
            2. Greeks alignment with strategy goals
            3. Risk/reward profile
            4. Time decay considerations
            5. Implied Volatility (IV): Analyze the current IV level relative to the stock's historical IV. Consider whether IV is high (potentially favoring premium selling strategies like covered calls, cash-secured puts, or credit spreads) or low (potentially favoring premium buying strategies like long calls/puts or debit spreads). Assess if the IV justifies the option's price (extrinsic value).
            
            For each strategy:
            - long_call: Favor strong delta, manageable theta, reasonable IV (consider buying when IV is relatively low)
            - long_put: Favor strong delta, manageable theta, reasonable IV (consider buying when IV is relatively low)
            - covered_call: Focus on optimal premium/risk balance, theta decay (often sold when IV is relatively high)
            - cash_secured_put: Focus on optimal premium/risk balance, theta decay (often sold when IV is relatively high)
            - bull_call_spread: Balance between long and short strikes, net delta, consider net premium vs IV level
            - bear_put_spread: Balance between long and short strikes, net delta, consider net premium vs IV level
            - iron_condor: Width of strikes, probability of profit, max loss (often sold when IV is high)
            - calendar_spread: IV differential, theta differential (often benefits from IV changes)
            
            For each contract, calculate:
            - Expected value based on probability models
            - Risk/reward ratio
            - Theoretical edge
            
            Return a detailed contract recommendation with the following fields:
            - ticker: The option contract ticker
            - underlying_ticker: The underlying stock ticker
            - action: "buy" or "sell" the contract
            - quantity: Always set this to 1. The final quantity will be determined by risk management later.
            - option_type: "call" or "put"
            - strike_price: Strike price of the contract
            - expiration_date: Expiration date (YYYY-MM-DD format)
            - strategy: Strategy this contract is part of
            - confidence: Confidence in this selection (0-100)
            - reasoning: Detailed explanation for selecting this contract
            - limit_price: Recommended limit price
            - stop_loss: Recommended stop loss price
            - take_profit: Recommended take profit price
            - max_position_value: Maximum dollar value to allocate
            - greeks: Current Greeks values (delta, gamma, theta, vega)
            """
        ),
        (
            "human",
            """Here is the stock trading decision, options strategy, and filtered contracts for ticker {ticker}:
            
            Stock Decision:
            ```json
            {stock_decision}
            ```
            
            Options Strategy:
            ```json
            {options_strategy}
            ```
            
            Filtered Contracts:
            ```json
            {contracts_data}
            ```
            
            Select the optimal contract and return a contract decision in this exact JSON format:
            {{
                "ticker": "string",
                "underlying_ticker": "{ticker}",
                "action": "buy|sell|close",
                "quantity": 1,
                "option_type": "call|put",
                "strike_price": float,
                "expiration_date": "YYYY-MM-DD",
                "strategy": "string",
                "confidence": float between 0 and 100,
                "reasoning": "string with detailed explanation",
                "limit_price": float,
                "stop_loss": float or null,
                "take_profit": float or null,
                "max_position_value": float,
                "greeks": {{
                    "delta": float,
                    "gamma": float,
                    "theta": float,
                    "vega": float
                }}
            }}
            """
        )
    ])
    
    prompt = template.invoke({
        "ticker": ticker,
        "stock_decision": json.dumps(stock_decision, indent=2),
        "options_strategy": json.dumps(options_strategy.model_dump(), indent=2),
        "contracts_data": json.dumps(contracts_data, indent=2, default=str)
    })
    
    # Define default contract decision
    def create_default_contract_decision():
        return OptionsContractDecision(
            ticker="",
            underlying_ticker=ticker,
            action="buy" if options_strategy.strategy_type in ["long_call", "long_put"] else "sell",
            quantity=0,
            option_type="call" if options_strategy.strategy_type in ["long_call", "covered_call"] else "put",
            strike_price=0.0,
            expiration_date=datetime.now().strftime("%Y-%m-%d"),
            strategy=options_strategy.strategy_type,
            confidence=0.0,
            reasoning="Error in contract selection analysis, no valid contract selected",
            limit_price=None,
            stop_loss=None,
            take_profit=None,
            max_position_value=0.0,
            greeks={"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        )
    
    # Use LLM to select the optimal contract
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=OptionsContractDecision,
        agent_name="options_contract_selection_agent",
        default_factory=create_default_contract_decision,
    )