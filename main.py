"""
Main script to run silver price analysis and options recommendations.
"""
import pandas as pd
from data_fetcher import SilverDataFetcher
from feature_engineering import FeatureEngineer
from ml_model import SilverPricePredictor, OptionsRecommender
import json
from datetime import datetime


def main():
    """Main execution function."""
    print("=" * 60)
    print("Silver Price Analysis & Options Recommendation System")
    print("=" * 60)
    print()
    
    # Initialize components
    print("1. Fetching silver price data...")
    fetcher = SilverDataFetcher(ticker="SLV")
    
    # Get historical data
    historical_data = fetcher.get_historical_data(period="2y")
    print(f"   ✓ Fetched {len(historical_data)} days of historical data")
    
    # Get current price
    current_price = fetcher.get_current_price()
    print(f"   ✓ Current silver price: ${current_price:.2f}")
    print()
    
    # Feature engineering
    print("2. Engineering features...")
    engineer = FeatureEngineer()
    data_with_features = engineer.add_technical_indicators(historical_data)
    data_with_target = engineer.create_target_variable(data_with_features, forward_days=5)
    prepared_data = engineer.prepare_features(data_with_target)
    cleaned_data = engineer.clean_data(prepared_data)
    print(f"   ✓ Created {len(cleaned_data.columns)} features")
    print()
    
    # Train model
    print("3. Training machine learning model...")
    predictor = SilverPricePredictor(model_type="random_forest")
    
    # Prepare training data
    X = cleaned_data.drop(columns=['Future_return', 'Future_price'], errors='ignore')
    y = cleaned_data['Future_return']
    
    metrics = predictor.train(X, y)
    print(f"   ✓ Model trained successfully")
    print(f"   - Training R²: {metrics['train_r2']:.4f}")
    print(f"   - Test R²: {metrics['test_r2']:.4f}")
    print(f"   - Test MAE: {metrics['test_mae']:.4f}")
    print()
    
    # Get most recent data for prediction
    print("4. Making price prediction...")
    recent_data = engineer.add_technical_indicators(historical_data.tail(100))
    recent_prepared = engineer.prepare_features(recent_data)
    recent_cleaned = engineer.clean_data(recent_prepared)
    
    if recent_cleaned.empty:
        print("   ✗ Not enough recent data for prediction")
        return
    
    # Use last row for prediction
    last_row = recent_cleaned.iloc[[-1]].copy()
    last_row['Close'] = current_price  # Update with current price
    
    predicted_return, predicted_price = predictor.predict(last_row)
    print(f"   ✓ Prediction complete")
    print(f"   - Predicted return: {predicted_return*100:.2f}%")
    print(f"   - Predicted price (5 days): ${predicted_price:.2f}")
    print()
    
    # Get options data
    print("5. Fetching options contracts...")
    options_chain = fetcher.get_options_chain()
    if options_chain.get('expiration'):
        print(f"   ✓ Fetched options for expiration: {options_chain['expiration']}")
        print(f"   - Available calls: {len(options_chain['calls'])}")
        print(f"   - Available puts: {len(options_chain['puts'])}")
    else:
        print("   ⚠ No options data available")
    print()
    
    # Generate recommendation
    print("6. Generating options recommendation...")
    recommender = OptionsRecommender(
        current_price=current_price,
        predicted_return=predicted_return,
        predicted_price=predicted_price
    )
    
    recommendation = recommender.recommend_options(options_chain)
    
    # Display results
    print("=" * 60)
    print("RECOMMENDATION RESULTS")
    print("=" * 60)
    print()
    print(f"Current Price: ${recommendation['current_price']:.2f}")
    print(f"Predicted Price (5 days): ${recommendation['predicted_price']:.2f}")
    print(f"Expected Return: {recommendation['expected_return']*100:.2f}%")
    print()
    print(f"Strategy: {recommendation['strategy']}")
    print()
    
    if recommendation['recommended_contract']:
        contract = recommendation['recommended_contract']
        print("Recommended Contract:")
        print(f"  Type: {contract['type']}")
        print(f"  Strike: ${contract['strike']:.2f}")
        print(f"  Mid Price: ${contract['mid_price']:.2f}")
        print(f"  Expected Profit: ${contract['expected_profit']:.2f}")
        print(f"  Expected Return: {contract['expected_return_pct']:.2f}%")
        print(f"  Volume: {contract['volume']}")
        print(f"  Open Interest: {contract['open_interest']}")
        print(f"  Contract Symbol: {contract['contract_symbol']}")
    else:
        print("No specific contract recommended.")
    
    print()
    print("Reasoning:")
    print(f"  {recommendation['reasoning']}")
    print()
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'current_price': current_price,
        'predicted_price': predicted_price,
        'predicted_return': predicted_return,
        'recommendation': recommendation
    }
    
    with open('recommendation_output.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("Results saved to 'recommendation_output.json'")
    print()
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

