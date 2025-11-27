"""
FCD Data Fetching Module
========================
Data fetchers for various providers (Alpaca, yfinance, etc.)
"""

from .alpaca_fetcher import fetch_data, AlpacaDataFetcher

__all__ = ['fetch_data', 'AlpacaDataFetcher']
