#!/usr/bin/env python3
"""
Supabase Data Analyzer

This script provides utilities to query and analyze trading data from Supabase.
It can generate reports and visualizations based on stored trading data.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tabulate import tabulate
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import Supabase client
from src.integrations.supabase import get_supabase_client

# Load environment variables
load_dotenv()

class SupabaseAnalyzer:
    """Class for analyzing trading data stored in Supabase."""
    
    def __init__(self):
        """Initialize the analyzer with a Supabase client."""
        self.client = get_supabase_client()
        self.style_map = {
            'BUY': {'color': 'green', 'marker': '^'},
            'SELL': {'color': 'red', 'marker': 'v'},
            'SHORT': {'color': 'purple', 'marker': 'v'},
            'COVER': {'color': 'blue', 'marker': '^'},
            'HOLD': {'color': 'gray', 'marker': 'o'},
        }
    
    def get_trading_cycles(self, limit=10):
        """
        Get trading cycles from Supabase.
        
        Args:
            limit: Maximum number of cycles to return
            
        Returns:
            DataFrame containing trading cycles
        """
        params = {"order": "timestamp.desc", "limit": str(limit)}
        response = self.client.select("trading_cycles", params=params)
        
        if not response:
            return pd.DataFrame()
        
        df = pd.DataFrame(response)
        
        # Calculate ROI
        df['profit_loss'] = df['portfolio_value_after'] - df['portfolio_value_before']
        df['roi_percent'] = ((df['portfolio_value_after'] / df['portfolio_value_before']) - 1) * 100
        
        return df
    
    def get_portfolio_strategy(self, cycle_id=None, ticker=None, action=None, limit=50):
        """
        Get portfolio strategy decisions from Supabase.
        
        Args:
            cycle_id: Optional cycle ID to filter by
            ticker: Optional ticker to filter by
            action: Optional action to filter by (buy, sell, short, cover, hold)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame containing portfolio strategy decisions
        """
        params = {"order": "timestamp.desc", "limit": str(limit)}
        
        if cycle_id:
            params["cycle_id"] = f"eq.{cycle_id}"
        
        if ticker:
            params["ticker"] = f"eq.{ticker}"
        
        if action:
            params["action"] = f"eq.{action.lower()}"
        
        response = self.client.select("portfolio_strategy", params=params)
        
        if not response:
            return pd.DataFrame()
        
        return pd.DataFrame(response)
    
    def get_options_analysis(self, cycle_id=None, underlying_ticker=None, limit=50):
        """
        Get options analysis from Supabase.
        
        Args:
            cycle_id: Optional cycle ID to filter by
            underlying_ticker: Optional underlying ticker to filter by
            limit: Maximum number of records to return
            
        Returns:
            DataFrame containing options analysis
        """
        params = {"order": "timestamp.desc", "limit": str(limit)}
        
        if cycle_id:
            params["cycle_id"] = f"eq.{cycle_id}"
        
        if underlying_ticker:
            params["underlying_ticker"] = f"eq.{underlying_ticker}"
        
        response = self.client.select("options_analysis", params=params)
        
        if not response:
            return pd.DataFrame()
        
        df = pd.DataFrame(response)
        
        # Extract details from JSONB
        if not df.empty and 'details' in df.columns:
            df['details'] = df['details'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            df['option_type'] = df['details'].apply(lambda x: x.get('option_type', '') if x else '')
            df['strike_price'] = df['details'].apply(lambda x: x.get('strike_price', 0) if x else 0)
            df['expiration_date'] = df['details'].apply(lambda x: x.get('expiration_date', '') if x else '')
        
        return df
    
    def get_execution_summary(self, cycle_id=None, ticker=None, status=None, limit=50):
        """
        Get execution summary from Supabase.
        
        Args:
            cycle_id: Optional cycle ID to filter by
            ticker: Optional ticker to filter by
            status: Optional status to filter by (executed, skipped, error)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame containing execution summary
        """
        params = {"order": "timestamp.desc", "limit": str(limit)}
        
        if cycle_id:
            params["cycle_id"] = f"eq.{cycle_id}"
        
        if ticker:
            params["ticker"] = f"eq.{ticker}"
        
        if status:
            params["status"] = f"eq.{status.lower()}"
        
        response = self.client.select("execution_summary", params=params)
        
        if not response:
            return pd.DataFrame()
        
        df = pd.DataFrame(response)
        
        # Parse raw_data for analysis
        if not df.empty and 'raw_data' in df.columns:
            df['raw_data'] = df['raw_data'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
        return df
    
    def get_position_management(self, ticker=None, action_type=None, limit=50):
        """
        Get position management actions from Supabase.
        
        Args:
            ticker: Optional ticker to filter by
            action_type: Optional action type to filter by (adjust_stop, scale_in, scale_out, exit, hold)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame containing position management actions
        """
        params = {"order": "timestamp.desc", "limit": str(limit)}
        
        if ticker:
            params["ticker"] = f"eq.{ticker}"
        
        if action_type:
            params["action_type"] = f"eq.{action_type}"
        
        response = self.client.select("position_management", params=params)
        
        if not response:
            return pd.DataFrame()
        
        df = pd.DataFrame(response)
        
        # Parse details and raw_data for analysis
        if not df.empty:
            if 'details' in df.columns:
                df['details'] = df['details'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            if 'raw_data' in df.columns:
                df['raw_data'] = df['raw_data'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
        return df
    
    def plot_portfolio_performance(self, days=30, save_path=None):
        """
        Plot portfolio performance over time.
        
        Args:
            days: Number of days to look back
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib.figure.Figure
        """
        # Get trading cycles
        cycles_df = self.get_trading_cycles(limit=1000)
        
        if cycles_df.empty:
            print("No trading cycles found")
            return None
        
        # Filter by date range
        start_date = datetime.now() - timedelta(days=days)
        cycles_df['timestamp'] = pd.to_datetime(cycles_df['timestamp'])
        cycles_df = cycles_df[cycles_df['timestamp'] >= start_date]
        
        if cycles_df.empty:
            print(f"No trading cycles found in the last {days} days")
            return None
        
        # Sort by timestamp
        cycles_df = cycles_df.sort_values('timestamp')
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cycles_df['timestamp'], cycles_df['portfolio_value_after'], 'b-', linewidth=2)
        ax.set_title(f'Portfolio Performance (Last {days} Days)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_ticker_decisions(self, ticker, days=30, save_path=None):
        """
        Plot trading decisions for a specific ticker alongside its price.
        
        Args:
            ticker: Ticker symbol
            days: Number of days to look back
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib.figure.Figure
        """
        # Get portfolio strategy decisions for the ticker
        decisions_df = self.get_portfolio_strategy(ticker=ticker, limit=1000)
        
        if decisions_df.empty:
            print(f"No decisions found for {ticker}")
            return None
        
        # Filter by date range
        start_date = datetime.now() - timedelta(days=days)
        decisions_df['timestamp'] = pd.to_datetime(decisions_df['timestamp'])
        decisions_df = decisions_df[decisions_df['timestamp'] >= start_date]
        
        if decisions_df.empty:
            print(f"No decisions found for {ticker} in the last {days} days")
            return None
        
        # Get execution data to overlay price information
        executions_df = self.get_execution_summary(ticker=ticker, limit=1000)
        executions_df['timestamp'] = pd.to_datetime(executions_df['timestamp'])
        executions_df = executions_df[executions_df['timestamp'] >= start_date]
        
        if executions_df.empty:
            print(f"No execution data found for {ticker} in the last {days} days")
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price over time if available
        if not executions_df.empty and 'raw_data' in executions_df.columns:
            # Extract price data if available
            prices = []
            timestamps = []
            
            for _, row in executions_df.iterrows():
                raw_data = row.get('raw_data', {})
                if raw_data and 'order' in raw_data:
                    order = raw_data['order']
                    if order and 'filled_avg_price' in order and order['filled_avg_price']:
                        prices.append(float(order['filled_avg_price']))
                        timestamps.append(row['timestamp'])
            
            if prices and timestamps:
                ax.plot(timestamps, prices, 'k-', alpha=0.5, label=f"{ticker} Price")
        
        # Plot decisions as markers
        for _, row in decisions_df.iterrows():
            action = row['action'].upper()
            style = self.style_map.get(action, {'color': 'black', 'marker': 'o'})
            
            # Get price point if possible
            price_point = None
            matching_exec = executions_df[executions_df['timestamp'] == row['timestamp']]
            
            if not matching_exec.empty and 'raw_data' in matching_exec.iloc[0]:
                raw_data = matching_exec.iloc[0]['raw_data']
                if raw_data and 'order' in raw_data and raw_data['order'] and 'filled_avg_price' in raw_data['order']:
                    price_point = float(raw_data['order']['filled_avg_price'])
            
            if price_point:
                ax.scatter(row['timestamp'], price_point, 
                           color=style['color'], marker=style['marker'], s=100, 
                           label=f"{action} ({row['confidence']}%)")
        
        # Clean up legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        ax.set_title(f'Trading Decisions for {ticker} (Last {days} Days)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.grid(True)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def generate_performance_report(self, days=30):
        """
        Generate a performance report for the portfolio.
        
        Args:
            days: Number of days to look back
            
        Returns:
            str: Report as a string
        """
        # Get trading cycles
        cycles_df = self.get_trading_cycles(limit=1000)
        
        if cycles_df.empty:
            return "No trading cycles found"
        
        # Filter by date range
        start_date = datetime.now() - timedelta(days=days)
        cycles_df['timestamp'] = pd.to_datetime(cycles_df['timestamp'])
        cycles_df = cycles_df[cycles_df['timestamp'] >= start_date]
        
        if cycles_df.empty:
            return f"No trading cycles found in the last {days} days"
        
        # Calculate performance metrics
        total_profit_loss = cycles_df['profit_loss'].sum()
        avg_roi = cycles_df['roi_percent'].mean()
        positive_cycles = cycles_df[cycles_df['profit_loss'] > 0]
        negative_cycles = cycles_df[cycles_df['profit_loss'] < 0]
        win_rate = len(positive_cycles) / len(cycles_df) * 100 if len(cycles_df) > 0 else 0
        
        # Get latest portfolio value
        latest_value = cycles_df.loc[cycles_df['timestamp'].idxmax(), 'portfolio_value_after'] if not cycles_df.empty else 0
        
        # Get decisions by ticker
        all_decisions = self.get_portfolio_strategy(limit=1000)
        all_decisions['timestamp'] = pd.to_datetime(all_decisions['timestamp'])
        all_decisions = all_decisions[all_decisions['timestamp'] >= start_date]
        
        ticker_summary = {}
        if not all_decisions.empty:
            for ticker, group in all_decisions.groupby('ticker'):
                buys = group[group['action'].str.lower() == 'buy']
                sells = group[group['action'].str.lower() == 'sell']
                shorts = group[group['action'].str.lower() == 'short']
                covers = group[group['action'].str.lower() == 'cover']
                holds = group[group['action'].str.lower() == 'hold']
                
                ticker_summary[ticker] = {
                    'buys': len(buys),
                    'sells': len(sells),
                    'shorts': len(shorts),
                    'covers': len(covers),
                    'holds': len(holds),
                    'total_decisions': len(group)
                }
        
        # Build report
        report = []
        report.append("# Performance Report")
        report.append(f"Period: Last {days} days ({start_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')})")
        report.append(f"Number of Trading Cycles: {len(cycles_df)}")
        report.append(f"Latest Portfolio Value: ${latest_value:,.2f}")
        report.append(f"Total Profit/Loss: ${total_profit_loss:,.2f}")
        report.append(f"Average ROI: {avg_roi:.2f}%")
        report.append(f"Win Rate: {win_rate:.2f}%")
        report.append(f"Positive Cycles: {len(positive_cycles)}")
        report.append(f"Negative Cycles: {len(negative_cycles)}")
        
        if ticker_summary:
            report.append("\n## Ticker Summary")
            ticker_data = []
            for ticker, summary in ticker_summary.items():
                ticker_data.append([
                    ticker,
                    summary['buys'],
                    summary['sells'],
                    summary['shorts'],
                    summary['covers'],
                    summary['holds'],
                    summary['total_decisions']
                ])
            
            report.append(tabulate(
                ticker_data,
                headers=["Ticker", "Buys", "Sells", "Shorts", "Covers", "Holds", "Total"],
                tablefmt="pipe"
            ))
        
        return "\n\n".join(report)

def main():
    """Main function to run the analyzer from command line."""
    parser = argparse.ArgumentParser(description='Analyze trading data from Supabase')
    
    # Main commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List trading cycles
    cycles_parser = subparsers.add_parser('cycles', help='List trading cycles')
    cycles_parser.add_argument('--limit', type=int, default=10, help='Maximum number of cycles to return')
    
    # List portfolio strategy decisions
    strategy_parser = subparsers.add_parser('strategy', help='List portfolio strategy decisions')
    strategy_parser.add_argument('--ticker', type=str, help='Filter by ticker')
    strategy_parser.add_argument('--action', type=str, help='Filter by action (buy, sell, short, cover, hold)')
    strategy_parser.add_argument('--limit', type=int, default=20, help='Maximum number of decisions to return')
    
    # List options analysis
    options_parser = subparsers.add_parser('options', help='List options analysis')
    options_parser.add_argument('--ticker', type=str, help='Filter by underlying ticker')
    options_parser.add_argument('--limit', type=int, default=20, help='Maximum number of analyses to return')
    
    # List execution summary
    execution_parser = subparsers.add_parser('execution', help='List execution summary')
    execution_parser.add_argument('--ticker', type=str, help='Filter by ticker')
    execution_parser.add_argument('--status', type=str, help='Filter by status (executed, skipped, error)')
    execution_parser.add_argument('--limit', type=int, default=20, help='Maximum number of executions to return')
    
    # List position management actions
    management_parser = subparsers.add_parser('management', help='List position management actions')
    management_parser.add_argument('--ticker', type=str, help='Filter by ticker')
    management_parser.add_argument('--action', type=str, help='Filter by action type (adjust_stop, scale_in, scale_out, exit, hold)')
    management_parser.add_argument('--limit', type=int, default=20, help='Maximum number of actions to return')
    
    # Plot portfolio performance
    plot_parser = subparsers.add_parser('plot-portfolio', help='Plot portfolio performance')
    plot_parser.add_argument('--days', type=int, default=30, help='Number of days to look back')
    plot_parser.add_argument('--save', type=str, help='Path to save the plot')
    
    # Plot ticker decisions
    plot_ticker_parser = subparsers.add_parser('plot-ticker', help='Plot trading decisions for a specific ticker')
    plot_ticker_parser.add_argument('ticker', type=str, help='Ticker symbol')
    plot_ticker_parser.add_argument('--days', type=int, default=30, help='Number of days to look back')
    plot_ticker_parser.add_argument('--save', type=str, help='Path to save the plot')
    
    # Generate performance report
    report_parser = subparsers.add_parser('report', help='Generate performance report')
    report_parser.add_argument('--days', type=int, default=30, help='Number of days to look back')
    report_parser.add_argument('--save', type=str, help='Path to save the report')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SupabaseAnalyzer()
    
    # Run command
    if args.command == 'cycles':
        df = analyzer.get_trading_cycles(limit=args.limit)
        if df.empty:
            print("No trading cycles found")
        else:
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    elif args.command == 'strategy':
        df = analyzer.get_portfolio_strategy(ticker=args.ticker, action=args.action, limit=args.limit)
        if df.empty:
            print("No portfolio strategy decisions found")
        else:
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    elif args.command == 'options':
        df = analyzer.get_options_analysis(underlying_ticker=args.ticker, limit=args.limit)
        if df.empty:
            print("No options analysis found")
        else:
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    elif args.command == 'execution':
        df = analyzer.get_execution_summary(ticker=args.ticker, status=args.status, limit=args.limit)
        if df.empty:
            print("No execution summary found")
        else:
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    elif args.command == 'management':
        df = analyzer.get_position_management(ticker=args.ticker, action_type=args.action, limit=args.limit)
        if df.empty:
            print("No position management actions found")
        else:
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    elif args.command == 'plot-portfolio':
        fig = analyzer.plot_portfolio_performance(days=args.days, save_path=args.save)
        if fig:
            plt.tight_layout()
            plt.show()
    
    elif args.command == 'plot-ticker':
        fig = analyzer.plot_ticker_decisions(ticker=args.ticker, days=args.days, save_path=args.save)
        if fig:
            plt.tight_layout()
            plt.show()
    
    elif args.command == 'report':
        report = analyzer.generate_performance_report(days=args.days)
        print(report)
        
        if args.save:
            with open(args.save, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.save}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 