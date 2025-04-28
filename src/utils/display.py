from colorama import Fore, Style
from tabulate import tabulate
from .analysts import ANALYST_ORDER
import os
import json
from datetime import datetime # Added for timestamp parsing
from typing import Dict, List, Optional, Any


def sort_agent_signals(signals):
    """Sort agent signals in a consistent order."""
    # Create order mapping from ANALYST_ORDER
    analyst_order = {display: idx for idx, (display, _) in enumerate(ANALYST_ORDER)}
    analyst_order["Risk Management"] = len(ANALYST_ORDER)  # Add Risk Management at the end

    return sorted(signals, key=lambda x: analyst_order.get(x[0], 999))


def print_trading_output(result: dict) -> None:
    """
    Print formatted trading results with colored tables for multiple tickers.

    Args:
        result (dict): Dictionary containing decisions and analyst signals for multiple tickers
    """
    decisions = result.get("decisions", {})
    if not decisions:
        print(f"{Fore.RED}No trading decisions available{Style.RESET_ALL}")
        return

    # Get the list of requested tickers (those that have analyst signals)
    requested_tickers = set()
    for agent_signals in result.get("analyst_signals", {}).values():
        for ticker in agent_signals:
            requested_tickers.add(ticker)
    
    # Only process tickers that were requested for analysis 
    # (have analyst signals or explicitly requested)
    for ticker, decision in decisions.items():
        # Skip tickers that don't have analyst signals 
        # (likely in portfolio but not requested for analysis)
        if ticker not in requested_tickers and not result.get("analyst_signals", {}):
            continue
            
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Analysis for {Fore.CYAN}{ticker}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{Style.BRIGHT}{'=' * 50}{Style.RESET_ALL}")

        # Prepare analyst signals table for this ticker
        table_data = []
        for agent, signals in result.get("analyst_signals", {}).items():
            if ticker not in signals:
                continue
                
            # Skip Risk Management agent in the signals section
            if agent == "risk_management_agent":
                continue

            signal = signals[ticker]
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            signal_type = signal.get("signal", "").upper()
            confidence = signal.get("confidence", 0)

            signal_color = {
                "BULLISH": Fore.GREEN,
                "BEARISH": Fore.RED,
                "NEUTRAL": Fore.YELLOW,
            }.get(signal_type, Fore.WHITE)
            
            # Get reasoning if available
            reasoning_str = ""
            if "reasoning" in signal and signal["reasoning"]:
                reasoning = signal["reasoning"]
                
                # Handle different types of reasoning (string, dict, etc.)
                if isinstance(reasoning, str):
                    reasoning_str = reasoning
                elif isinstance(reasoning, dict):
                    # Convert dict to string representation
                    reasoning_str = json.dumps(reasoning, indent=2)
                else:
                    # Convert any other type to string
                    reasoning_str = str(reasoning)
                
                # Wrap long reasoning text to make it more readable
                wrapped_reasoning = ""
                current_line = ""
                # Use a fixed width of 60 characters to match the table column width
                max_line_length = 60
                for word in reasoning_str.split():
                    if len(current_line) + len(word) + 1 > max_line_length:
                        wrapped_reasoning += current_line + "\n"
                        current_line = word
                    else:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                if current_line:
                    wrapped_reasoning += current_line
                
                reasoning_str = wrapped_reasoning

            table_data.append(
                [
                    f"{Fore.CYAN}{agent_name}{Style.RESET_ALL}",
                    f"{signal_color}{signal_type}{Style.RESET_ALL}",
                    f"{Fore.WHITE}{confidence}%{Style.RESET_ALL}",
                    f"{Fore.WHITE}{reasoning_str}{Style.RESET_ALL}",
                ]
            )

        # Sort the signals according to the predefined order
        table_data = sort_agent_signals(table_data)

        print(f"\n{Fore.WHITE}{Style.BRIGHT}AGENT ANALYSIS:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]")
        print(
            tabulate(
                table_data,
                headers=[f"{Fore.WHITE}Agent", "Signal", "Confidence", "Reasoning"],
                tablefmt="grid",
            )
        )

        # Print Trading Decision Table
        action = decision.get("action", "").upper()
        action_color = {
            "BUY": Fore.GREEN,
            "SELL": Fore.RED,
            "HOLD": Fore.YELLOW,
            "COVER": Fore.GREEN,
            "SHORT": Fore.RED,
        }.get(action, Fore.WHITE)

        # Get reasoning and format it
        reasoning = decision.get("reasoning", "")
        # Wrap long reasoning text to make it more readable
        wrapped_reasoning = ""
        if reasoning:
            current_line = ""
            # Use a fixed width of 60 characters to match the table column width
            max_line_length = 60
            for word in reasoning.split():
                if len(current_line) + len(word) + 1 > max_line_length:
                    wrapped_reasoning += current_line + "\n"
                    current_line = word
                else:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
            if current_line:
                wrapped_reasoning += current_line

        decision_data = [
            ["Action", f"{action_color}{action}{Style.RESET_ALL}"],
            ["Quantity", f"{action_color}{decision.get('quantity')}{Style.RESET_ALL}"],
            [
                "Confidence",
                f"{Fore.WHITE}{decision.get('confidence'):.1f}%{Style.RESET_ALL}",
            ],
            ["Reasoning", f"{Fore.WHITE}{wrapped_reasoning}{Style.RESET_ALL}"],
        ]
        
        print(f"\n{Fore.WHITE}{Style.BRIGHT}TRADING DECISION:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]")
        print(tabulate(decision_data, tablefmt="grid"))

    # --- Consolidated AI Decisions Table --- 
    print(f"\n{Fore.WHITE}{Style.BRIGHT}AI DECISIONS:{Style.RESET_ALL}") # Renamed from PORTFOLIO SUMMARY
    ai_decision_data = []
    
    # --- Revised Portfolio Strategy Logic (Collect data first) ---
    overall_strategy_text = ""
    ticker_breakdown_data = [] # Use a list of lists for tabulate

    # Filter decisions to only include requested tickers
    filtered_decisions = {ticker: decision for ticker, decision in decisions.items() 
                          if ticker in requested_tickers or not result.get("analyst_signals", {})}

    # Attempt to find a general reasoning
    first_decision = next(iter(filtered_decisions.values()), None)
    pm_reasoning = first_decision.get("reasoning", "") if first_decision else ""
    is_general_reasoning = False
    if pm_reasoning and isinstance(pm_reasoning, str):
        # Heuristic: If PM reasoning doesn't mention a specific ticker from the filtered list, treat it as general.
        if not any(ticker in pm_reasoning for ticker in filtered_decisions.keys()):
            is_general_reasoning = True
            overall_strategy_text = pm_reasoning

    # Populate AI Decisions Table data (without Quantity) and Ticker Breakdown data
    for ticker, decision in filtered_decisions.items():
        action = decision.get("action", "-").upper() # Default to '-' if missing
        confidence = decision.get('confidence', 0.0) # Default to 0.0 if missing

        action_color = {
            "BUY": Fore.GREEN,
            "SELL": Fore.RED,
            "HOLD": Fore.YELLOW,
            "COVER": Fore.GREEN,
            "SHORT": Fore.RED,
            "-": Fore.WHITE, # Color for default action
        }.get(action, Fore.WHITE)

        confidence_str = f"{confidence:.1f}%" if confidence is not None else "0.0%"

        # Add to AI Decisions table (Ticker, Action, Confidence)
        ai_decision_data.append(
            [
                f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
                f"{action_color}{action}{Style.RESET_ALL}",
                f"{Fore.WHITE}{confidence_str}{Style.RESET_ALL}",
            ]
        )

        # Populate Ticker Breakdown data
        reason = decision.get("reasoning", f"No specific reasoning provided for {ticker}.")
        # If PM reasoning was general and identical, provide a pointer
        if is_general_reasoning and reason == pm_reasoning:
            reason = f"See overall strategy for {ticker}."
        
        # Wrap reasoning for the table
        wrapped_reasoning_bd = ""
        current_line_bd = ""
        max_line_length_bd = 50 # Adjust width for breakdown table
        for word in reason.split():
            if len(current_line_bd) + len(word) + 1 > max_line_length_bd:
                wrapped_reasoning_bd += current_line_bd + "\n"
                current_line_bd = word
            else:
                if current_line_bd:
                    current_line_bd += " " + word
                else:
                    current_line_bd = word
        if current_line_bd:
            wrapped_reasoning_bd += current_line_bd
            
        ticker_breakdown_data.append([
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{action}{Style.RESET_ALL}",
            f"{Fore.WHITE}{confidence_str}{Style.RESET_ALL}",
            f"{Fore.WHITE}{wrapped_reasoning_bd}{Style.RESET_ALL}"
        ])

    # --- Print AI Decisions Table (No Quantity) ---
    headers_ai = [f"{Fore.WHITE}Ticker", "Action", "Confidence"]
    print(
        tabulate(
            ai_decision_data,
            headers=headers_ai,
            tablefmt="grid",
        )
    )
    
    # --- Print Enhanced Portfolio Strategy (General + Table) ---
    print(f"\n{Fore.WHITE}{Style.BRIGHT}Portfolio Strategy:{Style.RESET_ALL}")
    if overall_strategy_text:
        # Wrap general strategy text
        wrapped_overall = ""
        current_line_os = ""
        max_line_length_os = 80 
        for word in overall_strategy_text.split():
            if len(current_line_os) + len(word) + 1 > max_line_length_os:
                wrapped_overall += current_line_os + "\n"
                current_line_os = word
            else:
                if current_line_os:
                    current_line_os += " " + word
                else:
                    current_line_os = word
        if current_line_os:
            wrapped_overall += current_line_os
        print(f"{Fore.CYAN}{wrapped_overall}{Style.RESET_ALL}\n") # Add newline for separation

    # Print Ticker Breakdown Table
    if ticker_breakdown_data:
        print(f"{Fore.WHITE}{Style.BRIGHT}Ticker Breakdown:{Style.RESET_ALL}")
        headers_bd = [f"{Fore.WHITE}Ticker", "Action", "Confidence", "Reasoning"]
        print(
            tabulate(
                ticker_breakdown_data,
                headers=headers_bd,
                tablefmt="grid",
            )
        )
    elif not overall_strategy_text:
        # Handle case where there's neither general strategy nor breakdown
        print(f"{Fore.YELLOW}No portfolio strategy reasoning available.{Style.RESET_ALL}")

    # Add options trading decisions if available
    if "options_decisions" in result:
        print_options_decisions(result["options_decisions"])
        
    # Add options execution summary if available  
    if "options_execution_results" in result:
        print_options_execution_summary(result["options_execution_results"])
        
    # Add options positions if available
    if "options_portfolio" in result:
        print_options_positions(result["options_portfolio"].get("positions", {}))


def print_backtest_results(table_rows: list) -> None:
    """Print the backtest results in a nicely formatted table"""
    # Clear the screen
    os.system("cls" if os.name == "nt" else "clear")

    # Split rows into ticker rows and summary rows
    ticker_rows = []
    summary_rows = []

    for row in table_rows:
        if isinstance(row[1], str) and "PORTFOLIO SUMMARY" in row[1]:
            summary_rows.append(row)
        else:
            ticker_rows.append(row)

    
    # Display latest portfolio summary
    if summary_rows:
        latest_summary = summary_rows[-1]
        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY:{Style.RESET_ALL}")

        # Extract values and remove commas before converting to float
        cash_str = latest_summary[7].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        position_str = latest_summary[6].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        total_str = latest_summary[8].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")

        print(f"Cash Balance: {Fore.CYAN}${float(cash_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Position Value: {Fore.YELLOW}${float(position_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Value: {Fore.WHITE}${float(total_str):,.2f}{Style.RESET_ALL}")
        print(f"Return: {latest_summary[9]}")
        
        # Display performance metrics if available
        if latest_summary[10] and latest_summary[11]:  # Sharpe and Sortino
            print(f"Sharpe Ratio: {latest_summary[10]} | Sortino Ratio: {latest_summary[11]}")
        if latest_summary[12]:  # Max Drawdown
            print(f"Max Drawdown: {latest_summary[12]}")

    headers = [
        f"{Fore.WHITE}Date",
        "Ticker",
        "Action",
        "Quantity",
        "Price",
        "Shares",
        "Position Value",
        "Bullish",
        "Bearish",
        "Neutral",
    ]

    print(f"\n{Fore.WHITE}{Style.BRIGHT}BACKTEST RESULTS:{Style.RESET_ALL}")
    print(tabulate(ticker_rows, headers=headers, tablefmt="grid"))


def format_backtest_row(
    date: str,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    shares_owned: float,
    position_value: float,
    bullish_count: int,
    bearish_count: int,
    neutral_count: int,
    is_summary: bool = False,
    total_value: float = None,
    return_pct: float = None,
    cash_balance: float = None,
    total_position_value: float = None,
    sharpe_ratio: float = None,
    sortino_ratio: float = None,
    max_drawdown: float = None,
) -> list[any]:
    """Format a row for the backtest results table"""
    # Color the action
    action_color = {
        "BUY": Fore.GREEN,
        "COVER": Fore.GREEN,
        "SELL": Fore.RED,
        "SHORT": Fore.RED,
        "HOLD": Fore.WHITE,
    }.get(action.upper(), Fore.WHITE)

    if is_summary:
        return_color = Fore.GREEN if return_pct >= 0 else Fore.RED
        return [
            date,
            f"{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY{Style.RESET_ALL}",
            "",  # Action
            "",  # Quantity
            "",  # Price
            "",  # Shares
            f"{Fore.YELLOW}${total_position_value:,.2f}{Style.RESET_ALL}",  # Total Position Value
            f"{Fore.CYAN}${cash_balance:,.2f}{Style.RESET_ALL}",  # Cash Balance
            f"{Fore.WHITE}${total_value:,.2f}{Style.RESET_ALL}",  # Total Value
            f"{return_color}{return_pct:+.2f}%{Style.RESET_ALL}",  # Return
            f"{Fore.YELLOW}{sharpe_ratio:.2f}{Style.RESET_ALL}" if sharpe_ratio is not None else "",  # Sharpe Ratio
            f"{Fore.YELLOW}{sortino_ratio:.2f}{Style.RESET_ALL}" if sortino_ratio is not None else "",  # Sortino Ratio
            f"{Fore.RED}{abs(max_drawdown):.2f}%{Style.RESET_ALL}" if max_drawdown is not None else "",  # Max Drawdown
        ]
    else:
        return [
            date,
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{action.upper()}{Style.RESET_ALL}",
            f"{action_color}{quantity:,.0f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{price:,.2f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{shares_owned:,.0f}{Style.RESET_ALL}",
            f"{Fore.YELLOW}{position_value:,.2f}{Style.RESET_ALL}",
            f"{Fore.GREEN}{bullish_count}{Style.RESET_ALL}",
            f"{Fore.RED}{bearish_count}{Style.RESET_ALL}",
            f"{Fore.BLUE}{neutral_count}{Style.RESET_ALL}",
        ]


# --- New Function: Print Execution Summary --- 
def print_execution_summary(execution_results: dict) -> None:
    """
    Print a summary table of executed trades.

    Args:
        execution_results (dict): Results from PortfolioManager.execute_decision.
    """
    print(f"\n{Fore.WHITE}{Style.BRIGHT}EXECUTION SUMMARY:{Style.RESET_ALL}")
    if not execution_results:
        print(f"{Fore.YELLOW}No execution results available.{Style.RESET_ALL}")
        return

    table_data = []
    for ticker, result in execution_results.items():
        status = result.get("status", "unknown").upper()
        message = result.get("message", "")
        order = result.get("order")

        if status == "SKIPPED":
            status_color = Fore.YELLOW
            action = "-"
            qty = "-"
            order_id_short = "-"
            order_status = "-"
            order_type = "-"
        elif status == "ERROR":
            status_color = Fore.RED
            action = "-"
            qty = "-"
            order_id_short = "-"
            order_status = "-"
            order_type = "-"
        elif status == "EXECUTED" and order:
            status_color = Fore.GREEN
            action = order.get("side", "-").upper()
            qty = order.get("qty", "-")
            order_id = order.get("id", "-")
            order_id_short = order_id.split('-')[0] + "..." if order_id != "-" else "-"
            order_status = order.get("status", "-")
            order_type = order.get("type", "-")
        else: # Unknown status or missing order info
             status_color = Fore.MAGENTA
             action = "-"
             qty = "-"
             order_id_short = "-"
             order_status = status # Show original status if possible
             order_type = "-"
        
        # Wrap message
        wrapped_message = ""
        current_line = ""
        max_line_length = 40
        for word in message.split():
            if len(current_line) + len(word) + 1 > max_line_length:
                wrapped_message += current_line + "\n"
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        if current_line:
            wrapped_message += current_line

        table_data.append([
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{status_color}{status}{Style.RESET_ALL}",
            f"{Fore.WHITE}{action}{Style.RESET_ALL}",
            f"{Fore.WHITE}{qty}{Style.RESET_ALL}",
            f"{Fore.WHITE}{order_status}{Style.RESET_ALL}",
            f"{Fore.WHITE}{order_id_short}{Style.RESET_ALL}",
            f"{Fore.WHITE}{wrapped_message}{Style.RESET_ALL}"
        ])

    headers = [f"{Fore.WHITE}Ticker", "Result", "Action", "Qty", "Order Status", "Order ID", "Message"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


# --- New Function: Print Portfolio Status --- 
def print_portfolio_status(portfolio_state: dict) -> None:
    """
    Print a summary of the current Alpaca portfolio status.

    Args:
        portfolio_state (dict): Results from PortfolioManager.get_portfolio_state.
    """
    print(f"\n{Fore.WHITE}{Style.BRIGHT}CURRENT PORTFOLIO STATUS (Alpaca):{Style.RESET_ALL}")
    
    positions = portfolio_state.get("positions", [])
    orders = portfolio_state.get("orders", []) # Get orders
    cash = portfolio_state.get("cash", 0.0)
    portfolio_value = portfolio_state.get("portfolio_value", 0.0)
    timestamp_str = portfolio_state.get("timestamp", datetime.now().isoformat())
    
    try:
        timestamp = datetime.fromisoformat(timestamp_str).strftime("%Y-%m-%d %H:%M:%S %Z")
    except:
        timestamp = timestamp_str # Fallback if parsing fails

    print(f"{Fore.WHITE}Timestamp: {timestamp}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Portfolio Value: {Fore.GREEN}${portfolio_value:,.2f}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Cash Balance: {Fore.CYAN}${cash:,.2f}{Style.RESET_ALL}")
    
    if not positions:
        print(f"{Fore.YELLOW}No open positions.{Style.RESET_ALL}")
        # Don't return early, still show pending orders
    else:
        position_table_data = []
        for pos in positions.values():
            symbol = pos.get("symbol", "?")
            side = pos.get("side", "?").upper()
            qty = float(pos.get("qty", 0))
            market_value = float(pos.get("market_value", 0))
            entry_price = float(pos.get("avg_entry_price", 0))
            current_price = float(pos.get("current_price", 0))
            unrealized_pl = float(pos.get("unrealized_pl", 0))
            unrealized_pl_pct = float(pos.get("unrealized_plpc", 0)) * 100 # Convert to percentage

            side_color = Fore.GREEN if side == "LONG" else Fore.RED
            pl_color = Fore.GREEN if unrealized_pl >= 0 else Fore.RED

            position_table_data.append([
                f"{Fore.CYAN}{symbol}{Style.RESET_ALL}",
                f"{side_color}{side}{Style.RESET_ALL}",
                f"{Fore.WHITE}{qty:,.0f}{Style.RESET_ALL}",
                f"{Fore.WHITE}${market_value:,.2f}{Style.RESET_ALL}",
                f"{Fore.WHITE}${entry_price:,.2f}{Style.RESET_ALL}",
                f"{Fore.WHITE}${current_price:,.2f}{Style.RESET_ALL}",
                f"{pl_color}{unrealized_pl:+,.2f}{Style.RESET_ALL}",
                f"{pl_color}{unrealized_pl_pct:+,.2f}%{Style.RESET_ALL}"
            ])

        position_headers = [f"{Fore.WHITE}Ticker", "Side", "Qty", "Market Value", "Entry Price", "Current Price", "Unrealized P/L ($)", "Unrealized P/L (%)"]
        print(tabulate(position_table_data, headers=position_headers, tablefmt="grid", floatfmt=",.2f"))

    # --- Display Pending Orders --- 
    pending_statuses = {'new', 'pending_new', 'accepted', 'partially_filled', 'held', 'calculated', 'pending_cancel', 'pending_replace'}
    pending_orders = [o for o in orders if o.get("status") in pending_statuses]

    print(f"\n{Fore.WHITE}{Style.BRIGHT}PENDING ORDERS:{Style.RESET_ALL}")
    if not pending_orders:
        print(f"{Fore.YELLOW}No pending orders.{Style.RESET_ALL}")
    else:
        order_table_data = []
        for order in pending_orders:
            symbol = order.get("symbol", "?")
            side = order.get("side", "?").upper()
            qty = float(order.get("qty", 0))
            filled_qty = float(order.get("filled_qty", 0))
            order_type = order.get("type", "?").upper()
            status = order.get("status", "?")
            limit_price = order.get("limit_price")
            stop_price = order.get("stop_price")
            order_id = order.get("id", "-")
            order_id_short = order_id.split('-')[0] + "..." if order_id != "-" else "-"
            
            side_color = Fore.GREEN if side == "BUY" else Fore.RED
            status_color = Fore.YELLOW # Pending orders are typically yellow

            price_info = f"@{limit_price}" if order_type == "LIMIT" and limit_price else (
                          f"Stop @{stop_price}" if order_type in ["STOP", "STOP_LIMIT"] and stop_price else "")

            order_table_data.append([
                f"{Fore.CYAN}{symbol}{Style.RESET_ALL}",
                f"{side_color}{side}{Style.RESET_ALL}",
                f"{Fore.WHITE}{order_type} {price_info}{Style.RESET_ALL}",
                f"{Fore.WHITE}{qty:,.0f}{Style.RESET_ALL}",
                f"{Fore.WHITE}{filled_qty:,.0f}{Style.RESET_ALL}",
                f"{status_color}{status}{Style.RESET_ALL}",
                f"{Fore.WHITE}{order_id_short}{Style.RESET_ALL}",
            ])

        order_headers = [f"{Fore.WHITE}Ticker", "Side", "Type", "Qty", "Filled", "Status", "Order ID"]
        print(tabulate(order_table_data, headers=order_headers, tablefmt="grid"))

# --- Options Trading Display Functions ---

def print_options_decisions(options_decisions: Dict[str, Dict]) -> None:
    """
    Print formatted options trading decisions. (OLD - Kept for potential compatibility, use print_options_analysis_summary for detailed table)
    
    Args:
        options_decisions: Dict containing options decisions for multiple tickers
    """
    if not options_decisions:
        # print(f"{Fore.YELLOW}No options trading decisions available{Style.RESET_ALL}") # Suppressed redundant message
        return
        
    print(f"\n{Fore.WHITE}{Style.BRIGHT}OPTIONS TRADING DECISIONS (Legacy View):{Style.RESET_ALL}")
    options_table_data = []
    
    for ticker, decision in options_decisions.items():
        action = decision.get('action', 'none').upper()
        
        if action == 'NONE':
            options_table_data.append([
                f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
                f"{Fore.YELLOW}NO ACTION{Style.RESET_ALL}",
                f"",  # Option ticker
                f"",  # Strike/Exp
                f"",  # Quantity
                f"{Fore.WHITE}{decision.get('reasoning', 'No suitable options strategy')}{Style.RESET_ALL}"
            ])
            continue
            
        option_ticker = decision.get('ticker', '')
        underlying = decision.get('underlying_ticker', ticker)
        option_type = decision.get('option_type', '').upper()
        strike = decision.get('strike_price', 0)
        expiration = decision.get('expiration_date', '')
        quantity = decision.get('quantity', 0) # Note: Quantity here is from analysis, not final execution
        # confidence = decision.get('confidence', 0) # Not shown in this legacy view
        reasoning = decision.get('reasoning', '')
        
        # Format the strike/expiration
        strike_exp = f"{strike:.2f} {option_type} {expiration}"
        
        # Determine color based on action
        action_color = {
            'BUY': Fore.GREEN,
            'SELL': Fore.RED,
            'CLOSE': Fore.YELLOW, # Assuming CLOSE might be an action
        }.get(action, Fore.WHITE)
        
        # Wrap reasoning
        wrapped_reasoning = ""
        current_line = ""
        max_line_length = 45 # Adjusted width
        for word in reasoning.split():
            if len(current_line) + len(word) + 1 > max_line_length:
                wrapped_reasoning += current_line + "\n"
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        if current_line:
            wrapped_reasoning += current_line
            
        options_table_data.append([
            f"{Fore.CYAN}{underlying}{Style.RESET_ALL}",
            f"{action_color}{action}{Style.RESET_ALL}",
            f"{Fore.WHITE}{option_ticker}{Style.RESET_ALL}",
            f"{Fore.WHITE}{strike_exp}{Style.RESET_ALL}",
            f"{action_color}{quantity}{Style.RESET_ALL}", # Show analysis quantity
            f"{Fore.WHITE}{wrapped_reasoning}{Style.RESET_ALL}"
        ])
    
    options_headers = [f"{Fore.WHITE}Underlying", "Action", "Option Ticker", "Strike/Type/Exp", "Analysis Qty", "Reasoning"]
    print(tabulate(options_table_data, headers=options_headers, tablefmt="grid"))

# --- NEW: Detailed Options Analysis Summary Table ---
def print_options_analysis_summary(options_decisions: Dict[str, Dict]) -> None:
    """
    Print a detailed summary table of the generated options analysis decisions.
    
    Args:
        options_decisions: Dict containing options analysis results for multiple tickers.
                           Expected format per ticker: 
                           {'ticker': 'O:XYZ...', 'action': 'buy', 'option_type': 'put', ...}
                           or {'action': 'none', 'reasoning': '...'}
    """
    print(f"\n{Fore.WHITE}{Style.BRIGHT}OPTIONS ANALYSIS SUMMARY:{Style.RESET_ALL}")
    if not options_decisions:
        print(f"{Fore.YELLOW}No options analysis results available.{Style.RESET_ALL}")
        return
        
    table_data = []
    for underlying_ticker, decision in options_decisions.items():
        action = decision.get('action', 'none').upper()
        
        if action == 'NONE':
            # Handle 'hold' or 'none' decision for the underlying
            reason = decision.get('reasoning', 'Stock action is hold/none, no options analysis performed.')
            wrapped_reasoning = _wrap_text(reason, 50)
            table_data.append([
                f"{Fore.CYAN}{underlying_ticker}{Style.RESET_ALL}",
                f"{Fore.YELLOW}NO ACTION{Style.RESET_ALL}", # Action
                "-", # Option Ticker
                "-", # Details
                "-", # Strategy
                "-", # Confidence
                "-", # Limit Price
                f"{Fore.WHITE}{wrapped_reasoning}{Style.RESET_ALL}" # Reasoning
            ])
            continue
            
        # Extract details for active option decisions
        option_ticker = decision.get('ticker', '-')
        option_type = decision.get('option_type', '?').upper()
        strike = decision.get('strike_price', 0)
        expiration = decision.get('expiration_date', '????-??-??')
        strategy = decision.get('strategy', '-').replace('_', ' ').title()
        confidence = decision.get('confidence', 0.0)
        limit_price = decision.get('limit_price') # Can be None
        reasoning = decision.get('reasoning', 'No reasoning provided.')
        
        # Format columns
        action_color = {'BUY': Fore.GREEN, 'SELL': Fore.RED}.get(action, Fore.WHITE)
        option_details = f"{strike:.2f} {option_type} {expiration}"
        confidence_str = f"{confidence:.1f}%" if confidence is not None else "-"
        limit_price_str = f"${limit_price:.2f}" if limit_price is not None else "-"
        wrapped_reasoning = _wrap_text(reasoning, 50) # Use helper for wrapping

        table_data.append([
            f"{Fore.CYAN}{underlying_ticker}{Style.RESET_ALL}",
            f"{action_color}{action}{Style.RESET_ALL}",
            f"{Fore.WHITE}{option_ticker}{Style.RESET_ALL}",
            f"{Fore.WHITE}{option_details}{Style.RESET_ALL}",
            f"{Fore.WHITE}{strategy}{Style.RESET_ALL}",
            f"{Fore.WHITE}{confidence_str}{Style.RESET_ALL}",
            f"{Fore.WHITE}{limit_price_str}{Style.RESET_ALL}",
            f"{Fore.WHITE}{wrapped_reasoning}{Style.RESET_ALL}"
        ])
    
    headers = [
        f"{Fore.WHITE}Underlying", 
        "Action", 
        "Option Ticker", 
        "Details", 
        "Strategy", 
        "Confidence", 
        "Limit Price", 
        "Reasoning"
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def print_options_execution_summary(execution_results: Dict[str, Dict]) -> None:
    """
    Print a summary table of executed options trades, similar to stock summary.
    
    Args:
        execution_results: Results from PortfolioManager.execute_options_decision.
    """
    print(f"\n{Fore.WHITE}{Style.BRIGHT}OPTIONS EXECUTION SUMMARY:{Style.RESET_ALL}")
    if not execution_results:
        print(f"{Fore.YELLOW}No options execution results available.{Style.RESET_ALL}")
        return
        
    table_data = []
    for ticker, result in execution_results.items(): # Ticker here is the *option* ticker
        status = result.get("status", "unknown").upper()
        message = result.get("message", "")
        order = result.get("order")
        
        status_color = {
            'SKIPPED': Fore.YELLOW,
            'ERROR': Fore.RED,
            'EXECUTED': Fore.GREEN,
        }.get(status, Fore.MAGENTA)
        
        action = "-"
        qty = "-"
        order_id_short = "-"
        order_status = "-"
        order_type = "-"
        limit_price_str = "-"

        if status == "SKIPPED":
            pass # Defaults are fine
        elif status == "ERROR":
            pass # Defaults are fine
        elif status == "EXECUTED" and order:
            action = order.get("side", "-").upper()
            qty = order.get("qty", "-")
            order_id = order.get("id", "-")
            order_id_short = order_id.split('-')[0] + "..." if order_id != "-" else "-"
            order_status = order.get("status", "-")
            order_type = order.get("type", "-").upper()
            limit_price = order.get("limit_price")
            if order_type == "LIMIT" and limit_price is not None:
                limit_price_str = f"@{limit_price}"
        else:
            order_status = status # Show original status if possible

        # Wrap message - Increased width
        wrapped_message = _wrap_text(message, 45) # Use helper and increased width
            
        table_data.append([
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}", # Option Ticker
            f"{status_color}{status}{Style.RESET_ALL}",
            f"{Fore.WHITE}{action}{Style.RESET_ALL}",
            f"{Fore.WHITE}{qty}{Style.RESET_ALL}",
            f"{Fore.WHITE}{order_type} {limit_price_str}{Style.RESET_ALL}",
            f"{Fore.WHITE}{order_status}{Style.RESET_ALL}",
            f"{Fore.WHITE}{order_id_short}{Style.RESET_ALL}",
            f"{Fore.WHITE}{wrapped_message}{Style.RESET_ALL}"
        ])
    
    # Match headers with stock execution summary where possible
    options_headers = [
        f"{Fore.WHITE}Option Ticker", 
        "Result", 
        "Action", 
        "Qty", 
        "Type", 
        "Order Status", 
        "Order ID", 
        "Message"
    ]
    print(tabulate(table_data, headers=options_headers, tablefmt="grid"))

def print_options_positions(positions: Dict[str, Dict]) -> None:
    """
    Print current options positions.
    
    Args:
        positions: Dict containing options positions
    """
    print(f"\n{Fore.WHITE}{Style.BRIGHT}CURRENT OPTIONS POSITIONS:{Style.RESET_ALL}")
    if not positions:
        print(f"{Fore.YELLOW}No options positions held.{Style.RESET_ALL}")
        return
    
    table_data = []
    # Iterate over the VALUES (position dictionaries)
    for pos in positions.values():
        symbol = pos.get("symbol", "?")
        # Get option specific details directly from pos dictionary
        underlying = pos.get('underlying', '') 
        option_type = pos.get('option_type', '').upper()
        strike = pos.get('strike_price', 0)
        expiration = pos.get('expiration_date', '')
        quantity = pos.get('qty', 0)
        entry_price = pos.get('avg_entry_price', 0)
        current_price = pos.get('current_price', 0)
        market_value = pos.get('market_value', 0)
        unrealized_pl = pos.get('unrealized_pl', 0)
        days_to_exp = pos.get('days_to_expiration', 0)
        
        # Format the description
        strike_exp = f"{strike:.2f} {option_type} {expiration}"
        
        # Determine profit/loss color
        pl_color = Fore.GREEN if unrealized_pl >= 0 else Fore.RED
        
        table_data.append([
            # Use underlying symbol for the first column as intended
            f"{Fore.CYAN}{underlying if underlying else symbol}{Style.RESET_ALL}",
            f"{Fore.WHITE}{strike_exp}{Style.RESET_ALL}",
            f"{Fore.WHITE}{quantity:,.0f}{Style.RESET_ALL}",
            f"{Fore.WHITE}${entry_price:.2f}{Style.RESET_ALL}",
            f"{Fore.WHITE}${current_price:.2f}{Style.RESET_ALL}",
            f"{Fore.WHITE}${market_value:.2f}{Style.RESET_ALL}",
            f"{pl_color}${unrealized_pl:.2f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{days_to_exp:.1f}{Style.RESET_ALL}"
        ])
    
    options_headers = [
        f"{Fore.WHITE}Underlying", 
        "Strike/Type/Exp", 
        "Quantity", 
        "Entry Price", 
        "Current Price", 
        "Market Value", 
        "Unrealized P/L", 
        "Days to Exp"
    ]
    print(tabulate(table_data, headers=options_headers, tablefmt="grid"))

# --- Helper function for text wrapping ---
def _wrap_text(text: str, width: int) -> str:
    """Wrap text to a specified width."""
    if not isinstance(text, str):
        text = str(text) # Ensure text is a string
        
    wrapped = ""
    current_line = ""
    for word in text.split():
        if len(current_line) + len(word) + 1 > width:
            wrapped += current_line + "\n"
            current_line = word
        else:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
    if current_line:
        wrapped += current_line
    return wrapped

# --- End Options Trading Display Functions ---