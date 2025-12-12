"""
Data fetching module for silver prices and options contracts.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class SilverDataFetcher:
    """Fetches silver price data and options contracts."""
    
    def __init__(self, ticker: str = "SLV"):
        """
        Initialize with silver ETF ticker.
        
        Args:
            ticker: Ticker symbol for silver (default: SLV for iShares Silver Trust)
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
    
    def get_historical_data(self, period: str = "2y") -> pd.DataFrame:
        """
        Fetch historical price data for silver.
        
        Args:
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            DataFrame with OHLCV data
        """
        data = self.stock.history(period=period)
        return data
    
    def get_options_expirations(self) -> List[str]:
        """Get available options expiration dates."""
        try:
            expirations = self.stock.options
            return list(expirations)
        except Exception as e:
            print(f"Error fetching options expirations: {e}")
            return []
    
    def get_options_chain(self, expiration: Optional[str] = None) -> Dict:
        """
        Get options chain for a specific expiration date.
        
        Args:
            expiration: Expiration date string (YYYY-MM-DD). If None, uses nearest expiration.
        
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        """
        try:
            if expiration is None:
                expirations = self.get_options_expirations()
                if not expirations:
                    return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
                expiration = expirations[0]
            
            opt_chain = self.stock.option_chain(expiration)
            return {
                "calls": opt_chain.calls,
                "puts": opt_chain.puts,
                "expiration": expiration
            }
        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiration": None}
    
    def get_current_price(self) -> float:
        """Get current silver price."""
        try:
            info = self.stock.info
            return info.get('regularMarketPrice', info.get('currentPrice', 0.0))
        except Exception as e:
            print(f"Error fetching current price: {e}")
            # Fallback to last close price
            data = self.get_historical_data(period="5d")
            if not data.empty:
                return data['Close'].iloc[-1]
            return 0.0

