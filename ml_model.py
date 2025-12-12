"""
Machine learning model for silver price prediction and options recommendation.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from typing import Dict, Tuple, Optional


class SilverPricePredictor:
    """ML model for predicting silver price movements."""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
    
    def _create_model(self):
        """Create the ML model based on model_type."""
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target variable (future returns)
        
        Returns:
            Dictionary with training metrics
        """
        # Remove target columns if present
        X_clean = X.drop(columns=['Future_return', 'Future_price', 'Close'], errors='ignore')
        
        # Store feature names
        self.feature_names = X_clean.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        self.is_trained = True
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[float, float]:
        """
        Predict future return and price.
        
        Args:
            X: Feature matrix (should have same features as training data)
        
        Returns:
            Tuple of (predicted_return, predicted_price)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Remove target columns if present
        X_clean = X.drop(columns=['Future_return', 'Future_price', 'Close'], errors='ignore')
        
        # Ensure same feature order
        X_clean = X_clean[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(X_clean)
        
        # Predict
        predicted_return = self.model.predict(X_scaled)[0]
        current_price = X['Close'].iloc[-1] if 'Close' in X.columns else X_clean.index[0]
        predicted_price = current_price * (1 + predicted_return)
        
        return predicted_return, predicted_price
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True


class OptionsRecommender:
    """Recommends options contracts based on price predictions."""
    
    def __init__(self, current_price: float, predicted_return: float, predicted_price: float):
        """
        Initialize the recommender.
        
        Args:
            current_price: Current silver price
            predicted_return: Predicted return (as decimal, e.g., 0.05 for 5%)
            predicted_price: Predicted future price
        """
        self.current_price = current_price
        self.predicted_return = predicted_return
        self.predicted_price = predicted_price
    
    def recommend_options(self, options_chain: Dict, min_days_to_expiry: int = 7) -> Dict:
        """
        Recommend the best options contract.
        
        Args:
            options_chain: Dictionary with 'calls' and 'puts' DataFrames
            min_days_to_expiry: Minimum days until expiration
        
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'strategy': None,
            'recommended_contract': None,
            'reasoning': None,
            'predicted_price': self.predicted_price,
            'current_price': self.current_price,
            'expected_return': self.predicted_return
        }
        
        calls = options_chain.get('calls', pd.DataFrame())
        puts = options_chain.get('puts', pd.DataFrame())
        
        if calls.empty and puts.empty:
            recommendations['reasoning'] = "No options data available"
            return recommendations
        
        # Determine strategy based on prediction
        if self.predicted_return > 0.02:  # Bullish (>2% expected gain)
            recommendations['strategy'] = 'CALL'
            best_contract = self._find_best_call(calls)
        elif self.predicted_return < -0.02:  # Bearish (>2% expected loss)
            recommendations['strategy'] = 'PUT'
            best_contract = self._find_best_put(puts)
        else:  # Neutral/sideways
            recommendations['strategy'] = 'NEUTRAL'
            # Could recommend straddle/strangle, but for simplicity, suggest waiting
            recommendations['reasoning'] = "Market expected to be neutral. Consider waiting or using neutral strategies."
            return recommendations
        
        recommendations['recommended_contract'] = best_contract
        recommendations['reasoning'] = self._generate_reasoning(best_contract)
        
        return recommendations
    
    def _find_best_call(self, calls: pd.DataFrame) -> Optional[Dict]:
        """Find the best call option based on predicted price."""
        if calls.empty:
            return None
        
        # Filter for contracts with reasonable strike prices
        # Target strikes near predicted price
        target_strike = self.predicted_price
        
        # Calculate expected profit for each call
        calls = calls.copy()
        calls['strike'] = calls['strike'].astype(float)
        calls['lastPrice'] = calls['lastPrice'].astype(float)
        calls['bid'] = calls['bid'].astype(float)
        calls['ask'] = calls['ask'].astype(float)
        
        # Use mid price
        calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
        calls.loc[calls['mid_price'] == 0, 'mid_price'] = calls['lastPrice']
        
        # Calculate intrinsic value at predicted price
        calls['intrinsic_value'] = np.maximum(self.predicted_price - calls['strike'], 0)
        calls['expected_profit'] = calls['intrinsic_value'] - calls['mid_price']
        calls['expected_return_pct'] = (calls['expected_profit'] / calls['mid_price']) * 100
        
        # Filter for contracts with positive expected profit
        profitable = calls[calls['expected_profit'] > 0].copy()
        
        if profitable.empty:
            # If no profitable contracts, find closest to ATM
            atm_calls = calls.iloc[(calls['strike'] - self.current_price).abs().argsort()[:5]]
            best = atm_calls.loc[atm_calls['expected_return_pct'].idxmax()]
        else:
            # Find contract with best risk/reward
            # Prefer contracts with strike near predicted price
            profitable['strike_distance'] = abs(profitable['strike'] - target_strike)
            profitable['score'] = profitable['expected_return_pct'] / (1 + profitable['strike_distance'] / self.current_price)
            best = profitable.loc[profitable['score'].idxmax()]
        
        return {
            'type': 'CALL',
            'strike': float(best['strike']),
            'bid': float(best['bid']),
            'ask': float(best['ask']),
            'last_price': float(best['lastPrice']),
            'mid_price': float(best['mid_price']),
            'volume': int(best.get('volume', 0)),
            'open_interest': int(best.get('openInterest', 0)),
            'expected_profit': float(best['expected_profit']),
            'expected_return_pct': float(best['expected_return_pct']),
            'contract_symbol': best.get('contractSymbol', 'N/A')
        }
    
    def _find_best_put(self, puts: pd.DataFrame) -> Optional[Dict]:
        """Find the best put option based on predicted price."""
        if puts.empty:
            return None
        
        target_strike = self.predicted_price
        
        puts = puts.copy()
        puts['strike'] = puts['strike'].astype(float)
        puts['lastPrice'] = puts['lastPrice'].astype(float)
        puts['bid'] = puts['bid'].astype(float)
        puts['ask'] = puts['ask'].astype(float)
        
        puts['mid_price'] = (puts['bid'] + puts['ask']) / 2
        puts.loc[puts['mid_price'] == 0, 'mid_price'] = puts['lastPrice']
        
        # Calculate intrinsic value at predicted price
        puts['intrinsic_value'] = np.maximum(puts['strike'] - self.predicted_price, 0)
        puts['expected_profit'] = puts['intrinsic_value'] - puts['mid_price']
        puts['expected_return_pct'] = (puts['expected_profit'] / puts['mid_price']) * 100
        
        profitable = puts[puts['expected_profit'] > 0].copy()
        
        if profitable.empty:
            atm_puts = puts.iloc[(puts['strike'] - self.current_price).abs().argsort()[:5]]
            best = atm_puts.loc[atm_puts['expected_return_pct'].idxmax()]
        else:
            profitable['strike_distance'] = abs(profitable['strike'] - target_strike)
            profitable['score'] = profitable['expected_return_pct'] / (1 + profitable['strike_distance'] / self.current_price)
            best = profitable.loc[profitable['score'].idxmax()]
        
        return {
            'type': 'PUT',
            'strike': float(best['strike']),
            'bid': float(best['bid']),
            'ask': float(best['ask']),
            'last_price': float(best['lastPrice']),
            'mid_price': float(best['mid_price']),
            'volume': int(best.get('volume', 0)),
            'open_interest': int(best.get('openInterest', 0)),
            'expected_profit': float(best['expected_profit']),
            'expected_return_pct': float(best['expected_return_pct']),
            'contract_symbol': best.get('contractSymbol', 'N/A')
        }
    
    def _generate_reasoning(self, contract: Optional[Dict]) -> str:
        """Generate human-readable reasoning for the recommendation."""
        if contract is None:
            return "No suitable contract found."
        
        direction = "bullish" if contract['type'] == 'CALL' else "bearish"
        return (
            f"Model predicts {direction} movement ({self.predicted_return*100:.2f}% expected return). "
            f"Recommended {contract['type']} option with strike ${contract['strike']:.2f}. "
            f"Expected profit: ${contract['expected_profit']:.2f} ({contract['expected_return_pct']:.2f}% return). "
            f"Current price: ${self.current_price:.2f}, Predicted price: ${self.predicted_price:.2f}."
        )

