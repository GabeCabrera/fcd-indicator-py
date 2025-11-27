#!/usr/bin/env python3
"""
Alpaca Historical Data Fetcher
===============================

Fetches historical OHLCV bars from Alpaca API for crypto assets.
Supports unlimited intraday history for parameter sweep testing.

Key Features:
- Crypto pairs: BTC/USD, ETH/USD, etc.
- Multiple timeframes: 1Min, 5Min, 15Min, 1Hour, 4Hour, 1Day
- Automatic pagination for large date ranges (>10,000 bars)
- Returns pandas DataFrame matching yfinance output format
- Environment-based API key configuration

Usage:
    from fcd.data.alpaca_fetcher import fetch_data
    
    # Fetch 30 days of BTC hourly data
    df = fetch_data('BTC/USD', '1h', days=30)
    
    # Fetch 365 days of ETH daily data
    df = fetch_data('ETH/USD', '1d', days=365)
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict
import pandas as pd
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AlpacaDataFetcher:
    """
    Fetches historical OHLCV data from Alpaca API.
    
    Attributes:
        api: Alpaca REST API client
        max_bars_per_request: Maximum bars per API call (Alpaca limit)
        
    Environment Variables:
        ALPACA_API_KEY: Alpaca API key ID
        ALPACA_API_SECRET: Alpaca API secret key
        ALPACA_BASE_URL: Alpaca base URL (defaults to paper trading)
    """
    
    # Alpaca timeframe mapping
    TIMEFRAME_MAP = {
        '1min': tradeapi.TimeFrame.Minute,
        '5min': tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
        '15min': tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
        '1h': tradeapi.TimeFrame.Hour,
        '4h': tradeapi.TimeFrame(4, tradeapi.TimeFrameUnit.Hour),
        '1d': tradeapi.TimeFrame.Day,
        # Aliases
        '1Hour': tradeapi.TimeFrame.Hour,
        '4Hour': tradeapi.TimeFrame(4, tradeapi.TimeFrameUnit.Hour),
        '1Day': tradeapi.TimeFrame.Day,
        '1Min': tradeapi.TimeFrame.Minute,
        '5Min': tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
        '15Min': tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
    }
    
    def __init__(self):
        """Initialize Alpaca API client from environment variables."""
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_API_SECRET')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not api_key or not secret_key:
            raise ValueError(
                "Missing Alpaca API credentials. "
                "Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
            )
        
        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url
        )
        
        # Alpaca limits responses to 10,000 bars per request
        self.max_bars_per_request = 10000
    
    def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical bars from Alpaca API.
        
        Args:
            symbol: Crypto pair (e.g., 'BTC/USD', 'ETH/USD')
            timeframe: Bar interval ('1min', '5min', '15min', '1h', '4h', '1d')
            start_date: Start datetime (inclusive)
            end_date: End datetime (inclusive)
            limit: Maximum number of bars to fetch (None = unlimited with pagination)
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
            Index is datetime
            
        Raises:
            ValueError: If timeframe is invalid or symbol is invalid
            Exception: If API request fails
        """
        # Validate timeframe
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. "
                f"Valid options: {list(self.TIMEFRAME_MAP.keys())}"
            )
        
        alpaca_timeframe = self.TIMEFRAME_MAP[timeframe]
        
        # Ensure symbol is in Alpaca format (e.g., BTC/USD not BTCUSD)
        if '/' not in symbol and len(symbol) == 6:
            # Convert BTCUSD -> BTC/USD
            symbol = f"{symbol[:3]}/{symbol[3:]}"
        
        print(f"📊 Fetching {symbol} {timeframe} bars from Alpaca...")
        print(f"   Date range: {start_date.date()} to {end_date.date()}")
        
        try:
            # Fetch bars with pagination handling
            all_bars = []
            current_start = start_date
            request_count = 0
            
            while current_start < end_date:
                request_count += 1
                
                # Format dates as YYYY-MM-DD for Alpaca API
                start_str = current_start.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                # Get bars for this chunk
                bars = self.api.get_crypto_bars(
                    symbol,
                    alpaca_timeframe,
                    start=start_str,
                    end=end_str,
                    limit=self.max_bars_per_request if limit is None else min(limit - len(all_bars), self.max_bars_per_request)
                ).df
                
                if bars is None or len(bars) == 0:
                    break
                
                all_bars.append(bars)
                
                # Check if we got less than max (means we're done)
                if len(bars) < self.max_bars_per_request:
                    break
                
                # Check if we hit the limit
                if limit is not None and sum(len(b) for b in all_bars) >= limit:
                    break
                
                # Move start to last bar's timestamp + 1 timeframe unit
                last_timestamp = bars.index[-1]
                # Make current_start timezone-aware if bars have timezone
                if last_timestamp.tzinfo is not None:
                    current_start = last_timestamp + self._get_timeframe_delta(timeframe)
                else:
                    current_start = current_start + self._get_timeframe_delta(timeframe)
                
                # Pagination status
                total_bars = sum(len(b) for b in all_bars)
                print(f"   📦 Request {request_count}: {len(bars)} bars (total: {total_bars})")
            
            # Combine all bars
            if not all_bars:
                raise ValueError(f"No data returned for {symbol} {timeframe}")
            
            df = pd.concat(all_bars)
            
            # Remove duplicates (can happen at pagination boundaries)
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by index (datetime)
            df = df.sort_index()
            
            print(f"   ✅ Retrieved {len(df)} bars ({request_count} requests)")
            
        except Exception as e:
            if "invalid symbol" in str(e).lower():
                raise ValueError(f"Invalid symbol '{symbol}'. Use format 'BTC/USD' or 'ETH/USD'")
            elif "rate limit" in str(e).lower():
                raise Exception(f"Alpaca API rate limit exceeded. Wait 1 minute and retry.")
            elif "timeout" in str(e).lower():
                raise Exception(f"Alpaca API timeout. Check network connection.")
            else:
                raise Exception(f"Alpaca API error: {e}")
        
        # Convert to standard OHLCV format (matching yfinance)
        df_standard = pd.DataFrame({
            'Open': df['open'],
            'High': df['high'],
            'Low': df['low'],
            'Close': df['close'],
            'Volume': df['volume'],
            'Adj Close': df['close']  # Crypto has no adjusted close
        })
        
        # Ensure datetime index
        df_standard.index.name = 'Date'
        
        return df_standard
    
    def _get_timeframe_delta(self, timeframe: str) -> timedelta:
        """
        Convert timeframe string to timedelta for pagination.
        
        Args:
            timeframe: Timeframe string (e.g., '1min', '1h', '1d')
            
        Returns:
            timedelta representing one bar's duration
        """
        delta_map = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
            '1Min': timedelta(minutes=1),
            '5Min': timedelta(minutes=5),
            '15Min': timedelta(minutes=15),
            '1Hour': timedelta(hours=1),
            '4Hour': timedelta(hours=4),
            '1Day': timedelta(days=1),
        }
        return delta_map.get(timeframe, timedelta(days=1))


def fetch_data(
    ticker: str,
    timeframe: str,
    days: int,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Convenience wrapper matching backtest script signature.
    
    Fetches historical crypto data from Alpaca API.
    
    Args:
        ticker: Crypto pair (e.g., 'BTC/USD', 'BTC-USD', 'BTCUSD')
        timeframe: Bar interval ('1min', '5min', '15min', '1h', '4h', '1d')
        days: Number of days of history to fetch
        end_date: End date (defaults to now)
        
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        Index is datetime named 'Date'
        
    Examples:
        >>> # Fetch 30 days of BTC hourly data
        >>> df = fetch_data('BTC/USD', '1h', days=30)
        >>> print(df.shape)
        (720, 6)  # 30 days * 24 hours
        
        >>> # Fetch 365 days of ETH daily data
        >>> df = fetch_data('ETH-USD', '1d', days=365)
        >>> print(df.columns)
        Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'], dtype='object')
    """
    # Normalize ticker format (convert BTC-USD -> BTC/USD)
    if '-' in ticker:
        ticker = ticker.replace('-', '/')
    
    # Calculate date range
    if end_date is None:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch data
    fetcher = AlpacaDataFetcher()
    df = fetcher.fetch_bars(
        symbol=ticker,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    return df


if __name__ == "__main__":
    """
    Test script: Fetch 30 days of BTC/USD 1h bars and print summary.
    
    Usage:
        python3 fcd-cloud-server/fcd/data/alpaca_fetcher.py
    """
    print("=" * 70)
    print("Alpaca Data Fetcher Test")
    print("=" * 70)
    
    # Test 1: Fetch BTC/USD hourly data
    print("\n[Test 1] Fetching 30 days of BTC/USD 1h bars...")
    try:
        df = fetch_data('BTC/USD', '1h', days=30)
        print(f"\n✅ Success!")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"\n   First 5 rows:")
        print(df.head())
        print(f"\n   Last 5 rows:")
        print(df.tail())
        print(f"\n   Summary stats:")
        print(df.describe())
    except Exception as e:
        print(f"\n❌ Failed: {e}")
    
    # Test 2: Fetch ETH/USD daily data
    print("\n" + "=" * 70)
    print("[Test 2] Fetching 90 days of ETH/USD 1d bars...")
    try:
        df = fetch_data('ETH-USD', '1d', days=90)  # Test with hyphen format
        print(f"\n✅ Success!")
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"\n   Recent prices:")
        print(df.tail())
    except Exception as e:
        print(f"\n❌ Failed: {e}")
    
    # Test 3: Test pagination with large request (>10,000 bars)
    print("\n" + "=" * 70)
    print("[Test 3] Testing pagination with 180 days of 1min bars (~259k bars)...")
    try:
        df = fetch_data('BTC/USD', '1min', days=180)
        print(f"\n✅ Success!")
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Total bars: {len(df):,}")
    except Exception as e:
        print(f"\n❌ Failed (expected if rate limited): {e}")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
