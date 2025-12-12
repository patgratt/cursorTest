"""
Feature engineering for silver price prediction.
"""
import pandas as pd
import numpy as np
import ta


class FeatureEngineer:
    """Creates features for machine learning model."""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to price data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with additional technical indicator columns
        """
        df = df.copy()
        
        # Moving averages
        df['SMA_5'] = ta.trend.SMAIndicator(df['Close'], window=5).sma_indicator()
        df['SMA_10'] = ta.trend.SMAIndicator(df['Close'], window=10).sma_indicator()
        df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_low'] = bollinger.bollinger_lband()
        df['BB_mid'] = bollinger.bollinger_mavg()
        df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['BB_mid']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price changes
        df['Price_change'] = df['Close'].pct_change()
        df['Price_change_5d'] = df['Close'].pct_change(periods=5)
        df['Price_change_10d'] = df['Close'].pct_change(periods=10)
        
        # Volatility
        df['Volatility'] = df['Price_change'].rolling(window=20).std()
        
        # High-Low spread
        df['HL_spread'] = (df['High'] - df['Low']) / df['Close']
        
        # Day of week, month features
        df['Day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['Month'] = pd.to_datetime(df.index).month
        
        return df
    
    @staticmethod
    def create_target_variable(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
        """
        Create target variable: future price change.
        
        Args:
            df: DataFrame with price data
            forward_days: Number of days ahead to predict
        
        Returns:
            DataFrame with target column added
        """
        df = df.copy()
        df['Future_return'] = df['Close'].shift(-forward_days) / df['Close'] - 1
        df['Future_price'] = df['Close'].shift(-forward_days)
        return df
    
    @staticmethod
    def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare final feature set for model training.
        
        Args:
            df: DataFrame with all indicators
        
        Returns:
            DataFrame with selected features
        """
        feature_cols = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_12', 'EMA_26',
            'RSI',
            'MACD', 'MACD_signal', 'MACD_diff',
            'BB_high', 'BB_low', 'BB_mid', 'BB_width',
            'Volume_ratio',
            'Price_change', 'Price_change_5d', 'Price_change_10d',
            'Volatility',
            'HL_spread',
            'Day_of_week', 'Month'
        ]
        
        # Select only columns that exist
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Add Close and target columns only if they exist
        extra_cols = ['Close', 'Future_return', 'Future_price']
        extra_cols = [col for col in extra_cols if col in df.columns]
        
        return df[available_cols + extra_cols]
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove NaN values and infinite values.
        
        Args:
            df: DataFrame to clean
        
        Returns:
            Cleaned DataFrame
        """
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        return df

