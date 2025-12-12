# Silver Price Analysis & Options Recommendation System

A machine learning system that analyzes silver prices and recommends the best options contracts to buy based on price predictions.

## Features

- **Historical Data Analysis**: Fetches and analyzes 2 years of silver price data
- **Technical Indicators**: Uses 20+ technical indicators (RSI, MACD, Bollinger Bands, moving averages, etc.)
- **ML Price Prediction**: Random Forest model predicts future price movements (5-day horizon)
- **Options Recommendation**: Automatically recommends CALL or PUT options based on predictions
- **Risk Assessment**: Calculates expected profit and return for recommended contracts

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis script:
```bash
python main.py
```

The script will:
1. Fetch historical silver price data (SLV ETF)
2. Engineer features using technical indicators
3. Train a machine learning model
4. Predict future price movements
5. Fetch available options contracts
6. Recommend the best options contract to buy

## Output

The script generates:
- Console output with detailed analysis and recommendations
- `recommendation_output.json` - JSON file with all results

## Components

- **`data_fetcher.py`**: Fetches silver prices and options data from Yahoo Finance
- **`feature_engineering.py`**: Creates technical indicators and prepares features
- **`ml_model.py`**: Machine learning model for price prediction and options recommendation
- **`main.py`**: Main execution script

## Model Details

- **Algorithm**: Random Forest Regressor
- **Features**: 20+ technical indicators including:
  - Moving averages (SMA, EMA)
  - Momentum indicators (RSI, MACD)
  - Volatility indicators (Bollinger Bands)
  - Volume indicators
  - Price patterns

## Options Recommendation Logic

- **Bullish (>2% expected gain)**: Recommends CALL options
- **Bearish (>2% expected loss)**: Recommends PUT options
- **Neutral**: Suggests waiting or neutral strategies

The system selects contracts based on:
- Expected profit at predicted price
- Strike price proximity to predicted price
- Risk/reward ratio

## Disclaimer

This is a tool for analysis and educational purposes. Options trading involves significant risk. Always do your own research and consult with financial advisors before making investment decisions.
