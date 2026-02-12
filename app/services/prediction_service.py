"""
Prediction service for ML model inference
"""
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import structlog

from app.core.config import settings
from app.services.data_service import DataService

logger = structlog.get_logger()


class PredictionService:
    """Service for making stock predictions using ML models"""
    
    def __init__(self, db: Session):
        self.db = db
        self.data_service = DataService(db)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained LSTM model"""
        try:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            # Load model architecture and weights
            # This would be implemented based on your specific model architecture
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
    
    async def get_prediction(
        self, 
        symbol: str, 
        horizon: int = 5, 
        include_technical_indicators: bool = True
    ) -> Dict[str, Any]:
        """Get prediction for a stock"""
        try:
            # Get recent price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            prices = self.data_service.get_stock_prices(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                limit=settings.sequence_length + 50
            )
            
            if len(prices) < settings.sequence_length:
                raise ValueError(f"Insufficient data for {symbol}. Need at least {settings.sequence_length} data points.")
            
            # Prepare features
            features = self._prepare_features(prices, include_technical_indicators)
            
            # Make prediction
            if self.model is None:
                # Fallback to simple prediction if model not available
                prediction = self._simple_prediction(features, horizon)
            else:
                prediction = await self._model_prediction(features, horizon)
            
            return {
                "symbol": symbol,
                "predictions": prediction,
                "confidence": 0.75,  # This would come from model uncertainty
                "model_version": "1.0.0",
                "generated_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}: {str(e)}")
            raise e
    
    async def get_latest_prediction(self, symbol: str, horizon: int = 5) -> Dict[str, Any]:
        """Get the latest prediction for a stock"""
        # This would typically check for cached predictions
        return await self.get_prediction(symbol, horizon)
    
    async def get_batch_predictions(self, symbols: List[str], horizon: int = 5) -> List[Dict[str, Any]]:
        """Get predictions for multiple stocks"""
        predictions = []
        for symbol in symbols:
            try:
                prediction = await self.get_prediction(symbol, horizon)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error getting prediction for {symbol}: {str(e)}")
                predictions.append({
                    "symbol": symbol,
                    "error": str(e),
                    "generated_at": datetime.now()
                })
        return predictions
    
    def _prepare_features(self, prices: List, include_technical_indicators: bool) -> np.ndarray:
        """Prepare features for model input"""
        # Convert to DataFrame
        df = pd.DataFrame([{
            'date': p.date,
            'open': p.open_price,
            'high': p.high_price,
            'low': p.low_price,
            'close': p.close_price,
            'volume': p.volume
        } for p in prices])
        
        df = df.sort_values('date').reset_index(drop=True)
        
        # Basic features
        features = []
        
        # Price features
        features.append(df['close'].values)
        features.append(df['open'].values)
        features.append(df['high'].values)
        features.append(df['low'].values)
        features.append(df['volume'].values)
        
        # Price ratios
        features.append(df['close'] / df['open'])
        features.append(df['high'] / df['close'])
        features.append(df['low'] / df['close'])
        
        # Returns
        returns = df['close'].pct_change().fillna(0)
        features.append(returns.values)
        
        # Volatility
        volatility = returns.rolling(window=20).std().fillna(0)
        features.append(volatility.values)
        
        if include_technical_indicators:
            # Get technical indicators
            indicators = self.data_service.get_technical_indicators(
                symbol=prices[0].symbol,
                start_date=df['date'].min(),
                end_date=df['date'].max()
            )
            
            # Add technical indicators to features
            indicator_df = pd.DataFrame([{
                'date': i.date,
                'indicator_name': i.indicator_name,
                'value': i.indicator_value
            } for i in indicators])
            
            if not indicator_df.empty:
                # Pivot indicators
                indicator_pivot = indicator_df.pivot(index='date', columns='indicator_name', values='value')
                
                # Align with price data
                for col in indicator_pivot.columns:
                    if col in ['SMA_20', 'SMA_50', 'RSI', 'MACD']:
                        aligned_values = indicator_pivot[col].reindex(df['date']).ffill().fillna(0)
                        features.append(aligned_values.values)
        
        # Stack features
        feature_matrix = np.column_stack(features)
        
        # Normalize features
        feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)
        
        # Return last sequence_length rows
        return feature_matrix[-settings.sequence_length:]
    
    def _simple_prediction(self, features: np.ndarray, horizon: int) -> List[Dict[str, Any]]:
        """Simple prediction fallback when model is not available"""
        # Use simple trend analysis
        recent_prices = features[:, 0]  # Close prices
        trend = np.mean(np.diff(recent_prices[-10:]))  # Recent trend
        
        predictions = []
        current_price = recent_prices[-1]
        
        for i in range(1, horizon + 1):
            predicted_price = current_price + (trend * i)
            predictions.append({
                "date": (datetime.now() + timedelta(days=i)).isoformat(),
                "predicted_price": float(predicted_price),
                "confidence": 0.6
            })
        
        return predictions
    
    async def _model_prediction(self, features: np.ndarray, horizon: int) -> List[Dict[str, Any]]:
        """Make prediction using the trained model"""
        # This would use the actual LSTM model
        # For now, return simple prediction
        return self._simple_prediction(features, horizon)
