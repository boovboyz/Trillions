"""
Integrations package for connecting the AI Hedge Fund to external services.
"""

from .alpaca import AlpacaTrader, AlpacaOptionsTrader, get_alpaca_client

__all__ = ['AlpacaTrader', 'AlpacaOptionsTrader', 'get_alpaca_client'] 