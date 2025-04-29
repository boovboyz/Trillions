# src/integrations/polygon.py
"""
Polygon API integration for options data.
Provides functions to fetch options chains and analyze contract metrics.
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import requests
import math # Add math import
import statistics # Add statistics import
from pydantic import BaseModel, Field, validator
from enum import Enum

from src.utils.rate_limiter import RateLimiter
from src.data.cache import get_cache

logger = logging.getLogger(__name__)

# Rate limiter for Polygon API
# Increased max_calls slightly as snapshot endpoint might be heavier
_polygon_rate_limiter = RateLimiter(max_calls=4, period=1.0)


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class OptionContract(BaseModel):
    """Model for an options contract with complete data."""
    ticker: str  # The option ticker (e.g., O:AAPL230616C00150000)
    underlying_ticker: str  # The underlying stock ticker
    expiration_date: datetime
    strike_price: float
    option_type: OptionType
    # Fields populated later by _get_option_last_price_and_greeks or fallback
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0 # Typically from snapshot/aggs, default 0 initially
    open_interest: int = 0 # Typically from snapshot/aggs, default 0 initially
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    # Calculated fields
    time_to_expiration: float = 0.0 # Calculated after basic info
    is_weekly: bool = False
    is_monthly: bool = False
    is_quarterly: bool = False
    intrinsic_value: float = 0.0 # Calculated after fetching price/details
    extrinsic_value: float = 0.0 # Calculated after fetching price/details
    in_the_money: bool = False   # Calculated after fetching price/details
    liquidity_score: float = 0.0 # Calculated after fetching price/details

    @classmethod
    def from_snapshot_result(cls, result: Dict[str, Any], underlying_ticker: str) -> Optional['OptionContract']:
        """Create an OptionContract from a single item in the chain snapshot results."""
        details = result.get('details', {})
        greeks = result.get('greeks', {}) # Greeks might be missing
        day_data = result.get('day', {}) # Day aggregates
        last_quote = result.get('last_quote', {}) # Quote data

        ticker = details.get('ticker')
        expiration_str = details.get('expiration_date')
        strike_price = details.get('strike_price')
        option_type_str = details.get('contract_type')

        if not all([ticker, expiration_str, strike_price is not None, option_type_str]):
            logger.warning(f"Skipping snapshot result due to missing basic details: {details}")
            return None

        try:
            expiration_date = datetime.strptime(expiration_str, '%Y-%m-%d')
            option_type = OptionType(option_type_str)
            strike_price = float(strike_price)

            # Calculate time to expiration
            now = datetime.now()
            time_to_expiration = max(0, (expiration_date - now).days + (expiration_date - now).seconds / 86400.0)

            # Check if weekly, monthly, or quarterly (simplification)
            days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            exp_day = expiration_date.day
            exp_month = expiration_date.month
            is_leap = expiration_date.year % 4 == 0 and (expiration_date.year % 100 != 0 or expiration_date.year % 400 == 0)
            is_monthly = expiration_date.weekday() == 4 and 15 <= exp_day <= 21
            is_quarterly = is_monthly and exp_month in [3, 6, 9, 12]
            is_weekly = not is_monthly

            # Extract values safely, providing defaults
            last_price = float(day_data.get('close', 0.0)) # Use day close as 'last_price'
            bid = float(last_quote.get('bid', 0.0))
            ask = float(last_quote.get('ask', 0.0))
            volume = int(day_data.get('volume', 0))
            open_interest = int(result.get('open_interest', 0)) # Directly on result
            implied_volatility = float(result.get('implied_volatility', 0.0)) # Directly on result

            # Greeks might be missing or null
            delta = float(greeks.get('delta', 0.0)) if greeks else 0.0
            gamma = float(greeks.get('gamma', 0.0)) if greeks else 0.0
            theta = float(greeks.get('theta', 0.0)) if greeks else 0.0
            vega = float(greeks.get('vega', 0.0)) if greeks else 0.0
            # Rho is often missing/zero in snapshots
            rho = float(greeks.get('rho', 0.0)) if greeks else 0.0


            return cls(
                ticker=ticker,
                underlying_ticker=underlying_ticker,
                expiration_date=expiration_date,
                strike_price=strike_price,
                option_type=option_type,
                last_price=last_price,
                bid=bid,
                ask=ask,
                volume=volume,
                open_interest=open_interest,
                implied_volatility=implied_volatility,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                time_to_expiration=time_to_expiration,
                is_weekly=is_weekly,
                is_monthly=is_monthly,
                is_quarterly=is_quarterly
                # Other calculated fields (intrinsic, extrinsic, ITM, liquidity) are calculated later
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing snapshot result for {ticker}: {e}. Data: {result}", exc_info=True)
            return None


# Note: OptionsChain model is no longer used by the primary logic,
# but kept for potential future use or reference.
class OptionsChain(BaseModel):
    """Model for an options chain response."""
    underlying_ticker: str
    expiration_dates: List[datetime]
    strike_prices: List[float]
    calls: List[OptionContract]
    puts: List[OptionContract]
    as_of: datetime
    iv_rank: float = 50.0 # Placeholder
    iv_percentile: float = 50.0 # Placeholder


class PolygonClient:
    """Client for interacting with Polygon API for options data using Option Chain Snapshot."""
    
    def __init__(self):
        """Initialize the Polygon client."""
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            logger.warning("Polygon API key not found. Options data fetching might fail.")
        
        self.base_url = 'https://api.polygon.io'
        self.cache = get_cache()
        
    def _get_option_chain_snapshot(
        self,
        underlying_ticker: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetches the entire option chain snapshot for a given underlying ticker,
        handling pagination.

        Args:
            underlying_ticker: The underlying stock ticker.
            params: Optional dictionary of query parameters for the snapshot API
                    (e.g., expiration_date, contract_type, strike_price with modifiers).

        Returns:
            A list of all contract snapshot results dictionaries fetched from the API.
        """
        all_results = []
        url = f"{self.base_url}/v3/snapshot/options/{underlying_ticker}"
        
        current_params = {'apiKey': self.api_key, 'limit': 250}
        if params:
            current_params.update(params)

        page_count = 0
        max_pages = 20

        while url and page_count < max_pages:
            _polygon_rate_limiter.acquire()
            page_count += 1
            logger.debug(f"Fetching snapshot page {page_count} for {underlying_ticker} from {url}")
            
            try:
                # Pass appropriate params based on page number
                request_params = current_params if page_count == 1 else {'apiKey': self.api_key, 'limit': 250}
                response = requests.get(url, params=request_params)
                response.raise_for_status()
                data = response.json()

                results = data.get('results')
                if isinstance(results, list):
                    all_results.extend(results)
                else:
                    logger.warning(f"Snapshot page {page_count} for {underlying_ticker} had no 'results' list.")

                # Check for next page
                next_url_from_data = data.get('next_url')
                if next_url_from_data:
                    # Add API key to the next_url if not present
                    if 'apiKey=' not in next_url_from_data:
                         url = f"{next_url_from_data}&apiKey={self.api_key}"
                    else:
                         url = next_url_from_data # Use the url directly if key is already there
                    current_params = {} # Clear params for next_url fetching
                else:
                    logger.debug(f"No more snapshot pages for {underlying_ticker}.")
                    url = None # Set url to None to exit loop

            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP Error fetching snapshot page {page_count} for {underlying_ticker}: {e}")
                url = None # Stop pagination on error
            except json.JSONDecodeError as e:
                 logger.error(f"JSON Decode Error fetching snapshot page {page_count} for {underlying_ticker}: {e}")
                 url = None # Stop pagination on error
            except Exception as e:
                 logger.error(f"Unexpected error fetching snapshot page {page_count} for {underlying_ticker}: {e}", exc_info=True)
                 url = None # Stop pagination on error
        
        if page_count >= max_pages:
             logger.warning(f"Reached max page limit ({max_pages}) fetching snapshot for {underlying_ticker}. Data might be incomplete.")

        logger.info(f"Fetched a total of {len(all_results)} contracts in snapshot for {underlying_ticker}.")
        return all_results
    
    def _get_underlying_price(self, ticker: str) -> float:
        """Get the current price of the underlying stock from the cache."""
        try:
            cached_prices = self.cache.get_prices(ticker)
            if cached_prices:
                latest_price_data = cached_prices[-1]
                price = latest_price_data.get('close')
                if price is not None:
                    logger.debug(f"Using cached price for {ticker}: {price}")
                    return float(price)
                else:
                    logger.warning(f"Cached price data for {ticker} is missing 'close' field: {latest_price_data}")
            else:
                 logger.warning(f"Price for {ticker} not found in cache.")

        except Exception as e:
            logger.error(f"Error retrieving price for {ticker} from cache: {e}")

        logger.warning(f"Could not determine underlying price for {ticker} from cache. Returning 0.")
        return 0.0

    def get_historical_volatility(self, ticker: str, window: int = 30) -> float:
        """
        Calculate historical volatility for a ticker using Polygon's aggs endpoint.
        """
        _polygon_rate_limiter.acquire()
        
        end = datetime.now()
        # Fetch slightly more data to increase chance of getting 'window' valid trading days
        start = end - timedelta(days=window * 2.5) # Adjusted multiplier
        
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        # Request more bars initially, then slice
        params = {'apiKey': self.api_key, 'limit': window * 2, 'sort': 'asc'} # Get oldest first to sort easily
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = data.get('results', [])
            if not results:
                 logger.warning(f"No historical agg results found for {ticker}")
                 return 0.0
                 
            # Ensure results are sorted ascending by time, take the most recent 'window+1' days for 'window' returns
            # Already sorted asc by API param
            results = results[-(window + 1):]
            if len(results) < window + 1:
                 logger.warning(f"Insufficient historical data points for {ticker} (need {window+1}, got {len(results)}) for volatility calc")
                 return 0.0

            # Calculate daily log returns from closing prices
            prices = [r['c'] for r in results]
            # Filter out potential zero prices before log calculation
            valid_prices = [(prices[i], prices[i-1]) for i in range(1, len(prices)) if prices[i] > 0 and prices[i-1] > 0]
            if len(valid_prices) < window * 0.8: # Require a good portion of the window
                logger.warning(f"Insufficient valid price pairs ({len(valid_prices)}) for log returns for {ticker}")
                return 0.0

            log_returns = [math.log(p_i / p_im1) for p_i, p_im1 in valid_prices]

            if len(log_returns) < 2: # Need at least 2 returns to calculate stdev
                logger.warning(f"Could not calculate enough log returns for {ticker} ({len(log_returns)} found)")
                return 0.0

            # Calculate the standard deviation of log returns
            std_dev = statistics.stdev(log_returns)

            # Annualize (using ~252 trading days)
            annualized_vol = std_dev * math.sqrt(252)
            logger.info(f"Calculated historical volatility for {ticker}: {annualized_vol:.4f}")
            return annualized_vol

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error calculating historical volatility for {ticker}: {e}")
            return 0.0
        except statistics.StatisticsError as e:
            logger.error(f"Statistics error calculating volatility for {ticker}: {e} (LogReturns: {log_returns}) ")
            return 0.0
        except Exception as e:
             logger.error(f"Unexpected error calculating volatility for {ticker}: {e}", exc_info=True)
             return 0.0

    def filter_options_by_strategy(
        self,
        ticker: str,
        strategy: str,
        price_target: Optional[float] = None,
        max_days_to_expiration: int = 60,
        min_delta: float = 0.1,
        min_liquidity_score: float = 30,
        max_contracts: int = 10 # Limit final selection size
    ) -> Union[List[OptionContract], List[Tuple[OptionContract, OptionContract]]]:
        """
        Filter options chain by a given strategy using the Option Chain Snapshot.
        1. Fetches full chain snapshot.
        2. Converts raw snapshot results to OptionContract objects.
        3. Filters by expiration date (DTE).
        4. Filters by plausible strike price range.
        5. Calculates liquidity score and other derived metrics.
        6. Applies strategy-specific filters (delta, etc.).
        7. Sorts and returns the final candidates.

        Args:
            ticker: Stock ticker
            strategy: Strategy name (e.g., 'long_call', 'covered_call')
            price_target: Optional target price for the underlying
            max_days_to_expiration: Maximum days to expiration
            min_delta: Minimum absolute delta value to consider
            min_liquidity_score: Minimum liquidity score to consider
            max_contracts: Maximum number of final contracts to return

        Returns:
            List of filtered OptionContract objects suitable for the strategy.
        """
        # Get current stock price and historical volatility (once per ticker)
        current_price = self._get_underlying_price(ticker)
        # hist_vol = self.get_historical_volatility(ticker) # Volatility calc currently unused later

        if current_price <= 0:
            logger.error(f"Could not get current price for {ticker}. Cannot filter options.")
            return []

        # --- Step 1: Fetch Option Chain Snapshot ---
        logger.info(f"Fetching options chain snapshot for {ticker}...")
        # Define query params for snapshot if needed (e.g., filter by type early)
        # snapshot_params = {'contract_type': 'call'} # Example
        snapshot_params = {} # Fetch all initially
        
        # Add expiration date range filtering directly to the snapshot API call
        now = datetime.now()
        soonest_valid_exp_str = (now + timedelta(days=1)).strftime('%Y-%m-%d')
        latest_valid_exp_str = (now + timedelta(days=max_days_to_expiration)).strftime('%Y-%m-%d')
        snapshot_params['expiration_date.gte'] = soonest_valid_exp_str
        snapshot_params['expiration_date.lte'] = latest_valid_exp_str
        # Optionally add contract_type filter if strategy always uses one type
        # if strategy.lower() in ['long_call', 'bull_call_spread', 'covered_call']:
        #     snapshot_params['contract_type'] = 'call'
        # elif strategy.lower() in ['long_put', 'bear_put_spread', 'cash_secured_put']:
        #      snapshot_params['contract_type'] = 'put'
             
        snapshot_results = self._get_option_chain_snapshot(ticker, params=snapshot_params)

        if not snapshot_results:
             logger.warning(f"Options chain snapshot for {ticker} returned no contracts (or failed).")
             return []

        # --- Step 2: Convert raw results to OptionContract objects ---
        initial_contracts = [
            OptionContract.from_snapshot_result(res, ticker)
            for res in snapshot_results
        ]
        # Filter out None values from conversion errors
        initial_contracts = [c for c in initial_contracts if c is not None]

        if not initial_contracts:
             logger.warning(f"No contracts could be parsed from snapshot for {ticker}.")
             return []
        
        logger.info(f"Parsed {len(initial_contracts)} contracts from snapshot for {ticker}. Applying filters...")

        # --- Step 3: Filter by DTE (already partly done by API query) ---
        # Redundant check, but ensures consistency if API filter failed
        now_dt = datetime.now() # Use consistent timestamp
        soonest_valid_exp = now_dt + timedelta(days=1)
        latest_valid_exp = now_dt + timedelta(days=max_days_to_expiration)
        contracts_in_dte = [
            c for c in initial_contracts
                if soonest_valid_exp <= c.expiration_date <= latest_valid_exp
        ]
        count_after_dte = len(contracts_in_dte)
        logger.debug(f"{count_after_dte} contracts remain after DTE filter for {ticker}.")

        if not contracts_in_dte:
            logger.warning(f"No contracts found for {ticker} within {max_days_to_expiration} DTE after parsing.")
            return []

        # --- Step 4: Pre-filter by plausible strike price range ---
        min_strike_call = current_price * 0.70
        max_strike_call = current_price * 1.50
        min_strike_put = current_price * 0.50
        max_strike_put = current_price * 1.30

        contracts_in_strike_range = []
        for contract in contracts_in_dte:
            plausible = False
            if contract.option_type == OptionType.CALL:
                if min_strike_call <= contract.strike_price <= max_strike_call:
                    plausible = True
            elif contract.option_type == OptionType.PUT:
                if min_strike_put <= contract.strike_price <= max_strike_put:
                     plausible = True
            else: # Should not happen
                 plausible = True

            if plausible:
                 contracts_in_strike_range.append(contract)

        count_after_strike = len(contracts_in_strike_range)
        logger.info(f"Reduced contracts from {count_after_dte} to {count_after_strike} after strike price pre-filter for {ticker}.")

        if not contracts_in_strike_range:
             logger.warning(f"No contracts remaining for {ticker} after strike price pre-filter.")
             return []
             
        # --- Step 5: Calculate liquidity, derived values, and apply final filters ---
        detailed_contracts = []
        for contract in contracts_in_strike_range:
            # Calculate liquidity score (requires bid/ask from snapshot)
            midpoint = (contract.bid + contract.ask) / 2 if contract.bid > 0 and contract.ask > 0 else 0
            spread_pct = (contract.ask - contract.bid) / midpoint if midpoint > 0 else 1.0
            spread_score = max(0, 100 * (1 - spread_pct * 5)) # Penalize wider spreads more
            # Use volume/OI directly from snapshot data
            volume_score = min(100, (contract.volume / 100) ** 0.5 * 100)
            oi_score = min(100, (contract.open_interest / 1000) ** 0.5 * 100)
            contract.liquidity_score = 0.5 * spread_score + 0.3 * volume_score + 0.2 * oi_score

            # Calculate other derived values
            if current_price > 0:
                if contract.option_type == OptionType.CALL:
                    contract.intrinsic_value = max(0, current_price - contract.strike_price)
                    contract.in_the_money = current_price > contract.strike_price
                else: # PUT
                    contract.intrinsic_value = max(0, contract.strike_price - current_price)
                    contract.in_the_money = current_price < contract.strike_price
                contract.extrinsic_value = max(0, contract.last_price - contract.intrinsic_value) if contract.last_price > 0 else 0

            # Check if contract has basic price info (needed for most strategies)
            if contract.last_price > 0 or (contract.bid > 0 and contract.ask > 0):
                 detailed_contracts.append(contract)
            else:
                 logger.debug(f"Skipping contract {contract.ticker} due to missing price/quote data.")

        logger.info(f"Calculated details for {len(detailed_contracts)} contracts for {ticker}. Applying strategy filters...")
        if not detailed_contracts:
            return []

        # --- Step 6: Apply strategy-specific filters ---
        candidates = []
        # Filter based on contract type required by strategy FIRST
        if strategy.lower() in ['long_call', 'bull_call_spread', 'covered_call']:
            strategy_contracts = [c for c in detailed_contracts if c.option_type == OptionType.CALL]
        elif strategy.lower() in ['long_put', 'bear_put_spread', 'cash_secured_put']:
             strategy_contracts = [c for c in detailed_contracts if c.option_type == OptionType.PUT]
        else: # Strategies using both (e.g., iron condor) - keep all for now
             strategy_contracts = detailed_contracts

        if not strategy_contracts:
            logger.warning(f"No contracts of required type found for strategy {strategy} for {ticker}.")
            return []

        # Apply common filters (delta, liquidity) and strategy-specific logic/sorting
        if strategy.lower() == 'long_call':
            calls = [c for c in strategy_contracts if abs(c.delta) >= min_delta and c.liquidity_score >= min_liquidity_score]
            if price_target and price_target > current_price:
                target_strike = current_price + (price_target - current_price) * 0.67
                calls = sorted(calls, key=lambda c: abs(c.strike_price - target_strike))
            else:
                calls = sorted(calls, key=lambda c: abs(c.delta / c.theta) if c.theta and c.theta != 0 else -1, reverse=True)
            candidates = calls

        elif strategy.lower() == 'long_put':
            puts = [p for p in strategy_contracts if abs(p.delta) >= min_delta and p.liquidity_score >= min_liquidity_score]
            if price_target and price_target < current_price:
                target_strike = current_price - (current_price - price_target) * 0.67
                puts = sorted(puts, key=lambda p: abs(p.strike_price - target_strike))
            else:
                puts = sorted(puts, key=lambda p: abs(p.delta / p.theta) if p.theta and p.theta != 0 else -1, reverse=True)
            candidates = puts

        elif strategy.lower() == 'covered_call':
            otm_calls = [
                c for c in strategy_contracts
                if c.strike_price > current_price * 1.01 # Slightly OTM
                and c.liquidity_score >= min_liquidity_score * 0.8 # Relax liquidity slightly
            ]
            valid_calls = []
            for call in otm_calls:
                if call.bid > 0 and call.time_to_expiration > 0:
                     days = max(1, call.time_to_expiration)
                     annual_return = (call.bid / call.strike_price) * (365 / days)
                     call.annualized_return = annual_return * 100 # Add field dynamically
                     valid_calls.append(call)
            candidates = sorted(valid_calls, key=lambda c: getattr(c, 'annualized_return', 0), reverse=True)

        elif strategy.lower() == 'cash_secured_put':
            otm_puts = [
                p for p in strategy_contracts
                if p.strike_price < current_price * 0.99 # Slightly OTM
                and p.liquidity_score >= min_liquidity_score * 0.8 # Relax liquidity slightly
            ]
            valid_puts = []
            for put in otm_puts:
                 if put.bid > 0 and put.time_to_expiration > 0:
                     days = max(1, put.time_to_expiration)
                     annual_return = (put.bid / put.strike_price) * (365 / days)
                     put.annualized_return = annual_return * 100 # Add field dynamically
                     valid_puts.append(put)
            candidates = sorted(valid_puts, key=lambda p: getattr(p, 'annualized_return', 0), reverse=True)
        
        # --- ADD MORE STRATEGIES HERE ---
        elif strategy.lower() == 'bull_call_spread':
             # Find pairs: Long call (lower strike), Short call (higher strike)
             # Typically filter for ITM/ATM long leg, OTM short leg
             # Example: Find long candidates first
             long_candidates = [
                 c for c in strategy_contracts 
                 if c.strike_price <= current_price * 1.02 # Near/slightly OTM
                 and abs(c.delta) >= 0.4 # Higher delta for long leg
                 and c.liquidity_score >= min_liquidity_score
             ]
             long_candidates = sorted(long_candidates, key=lambda c: c.strike_price) # Lower strike better

             spread_pairs = []
             for long_call in long_candidates:
                 # Find suitable short leg (higher strike, same expiration)
                 short_candidates = [
                     s for s in strategy_contracts
                     if s.expiration_date == long_call.expiration_date
                     and s.strike_price > long_call.strike_price
                     and s.strike_price <= long_call.strike_price * 1.15 # Reasonable spread width
                     and abs(s.delta) < 0.5 # Lower delta for short leg
                     and s.liquidity_score >= min_liquidity_score * 0.7 # Slightly lower liquidity ok
                 ]
                 # Choose the best short leg (e.g., closest strike)
                 if short_candidates:
                     best_short = min(short_candidates, key=lambda s: s.strike_price - long_call.strike_price)
                     # Represent pair as a tuple or dedicated object if needed
                     # Append the pair (long_call, best_short)
                     spread_pairs.append((long_call, best_short))
             
             # Sort the found pairs. Example: Sort by liquidity of the long leg, then proximity of long leg strike to ATM
             candidates = sorted(spread_pairs, key=lambda pair: (pair[0].liquidity_score, abs(pair[0].strike_price - current_price)), reverse=True)
             # Note: The return type is now List[Tuple[OptionContract, OptionContract]] for spreads
             # The caller needs to be adapted to handle this structure.

        # --- Bear Put Spread (similar logic to Bull Call Spread but with puts) ---
        elif strategy.lower() == 'bear_put_spread':
            long_candidates = [
                p for p in strategy_contracts
                if p.strike_price >= current_price * 0.98 # Near/slightly OTM Puts
                and abs(p.delta) >= 0.4
                and p.liquidity_score >= min_liquidity_score
            ]
            long_candidates = sorted(long_candidates, key=lambda p: p.strike_price, reverse=True) # Higher strike

            spread_pairs = [] # Change variable name for clarity
            for long_put in long_candidates:
                short_candidates = [
                    s for s in strategy_contracts
                    if s.expiration_date == long_put.expiration_date
                    and s.strike_price < long_put.strike_price # Lower strike short put
                    and s.strike_price >= long_put.strike_price * 0.85 # Reasonable width
                    and abs(s.delta) < 0.5
                    and s.liquidity_score >= min_liquidity_score * 0.7
                ]
                if short_candidates:
                    # Find the short put with the closest strike price below the long put
                    best_short = max(short_candidates, key=lambda s: s.strike_price) # Highest strike below long_put
                    # Append the pair (long_put, best_short)
                    spread_pairs.append((long_put, best_short)) 

            # Sort the found pairs. Example: Sort by liquidity of the long leg, then proximity of long leg strike to ATM
            candidates = sorted(spread_pairs, key=lambda pair: (pair[0].liquidity_score, abs(pair[0].strike_price - current_price)), reverse=True)
            # Note: The return type is now List[Tuple[OptionContract, OptionContract]] for spreads
            # The caller needs to be adapted to handle this structure.

        else:
            # This is the default case if strategy doesn't match above
            logger.warning(f"Strategy '{strategy}' filtering logic not fully implemented or matched. Returning candidates based on type.")
            candidates = sorted(strategy_contracts, key=lambda c: (
                 -c.liquidity_score, # Prioritize liquidity
                 abs(c.strike_price - current_price) # Then proximity to ATM
             ))

        # --- Step 7: Return final candidates, limited by max_contracts ---
        final_count = len(candidates)
        logger.info(f"Found {final_count} final candidates for {ticker} strategy {strategy}. Limiting to {max_contracts}.")
        # Add logging for top candidates if needed
        # for i, c in enumerate(candidates[:max_contracts]):
        #     logger.debug(f"  Candidate {i+1}: {c.ticker} (Liq: {c.liquidity_score:.1f}, Delta: {c.delta:.2f})")
            
        return candidates[:max_contracts]