"""
Data processing utilities for stock data
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import ta
import yfinance as yf
from datetime import datetime, timedelta


class StockDataProcessor:
    """Data processor for stock data"""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 5):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
    
    def download_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Download stock data from Yahoo Finance"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        data.reset_index(inplace=True)
        data['symbol'] = symbol
        return data
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Price-based indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # Momentum indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd(df['Close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
        df['MACD_histogram'] = ta.trend.macd_diff(df['Close'])
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volume indicators
        df['Volume_SMA'] = ta.volume.volume_sma(df['Close'], df['Volume'], window=20)
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price ratios
        df['High_Low_ratio'] = df['High'] / df['Low']
        df['Close_Open_ratio'] = df['Close'] / df['Open']
        
        # Returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Moving averages of returns
        df['Returns_SMA_5'] = df['Returns'].rolling(window=5).mean()
        df['Returns_SMA_10'] = df['Returns'].rolling(window=10).mean()
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix"""
        # Select relevant columns
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_position',
            'Volume_ratio', 'High_Low_ratio', 'Close_Open_ratio',
            'Returns', 'Log_returns', 'Volatility',
            'Returns_SMA_5', 'Returns_SMA_10'
        ]
        
        # Create feature dataframe
        features_df = df[feature_columns].copy()
        
        # Handle missing values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        return features_df
    
    def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def prepare_training_data(self, df: pd.DataFrame, target_column: str = 'Close') -> Dict[str, Any]:
        """Prepare data for training"""
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Create features
        features_df = self.create_features(df)
        
        # Create targets (future prices)
        targets = df[target_column].shift(-self.prediction_horizon).values
        
        # Remove rows with NaN targets
        valid_indices = ~np.isnan(targets)
        features_df = features_df[valid_indices]
        targets = targets[valid_indices]
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        self.scalers['features'] = scaler
        
        # Normalize targets
        target_scaler = MinMaxScaler()
        targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        self.scalers['targets'] = target_scaler
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, targets_scaled)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': features_df.columns.tolist(),
            'scalers': self.scalers
        }
    
    def inverse_transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Inverse transform normalized targets"""
        if 'targets' in self.scalers:
            return self.scalers['targets'].inverse_transform(targets.reshape(-1, 1)).flatten()
        return targets
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance (for tree-based models)"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return pd.DataFrame()
    
    def create_prediction_data(self, df: pd.DataFrame, last_n_days: int = None) -> np.ndarray:
        """Create data for making predictions"""
        if last_n_days:
            df = df.tail(last_n_days)
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Create features
        features_df = self.create_features(df)
        
        # Normalize features
        if 'features' in self.scalers:
            features_scaled = self.scalers['features'].transform(features_df)
        else:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_df)
            self.scalers['features'] = scaler
        
        # Get last sequence
        if len(features_scaled) >= self.sequence_length:
            return features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        else:
            raise ValueError(f"Not enough data. Need at least {self.sequence_length} data points.")


class MultiStockProcessor:
    """Processor for multiple stocks"""
    
    def __init__(self, symbols: List[str], sequence_length: int = 60):
        self.symbols = symbols
        self.sequence_length = sequence_length
        self.processors = {symbol: StockDataProcessor(sequence_length) for symbol in symbols}
    
    def prepare_multi_stock_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Prepare data for multiple stocks"""
        results = {}
        
        for symbol, df in data_dict.items():
            processor = self.processors[symbol]
            results[symbol] = processor.prepare_training_data(df)
        
        return results
    
    def create_ensemble_data(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Create ensemble training data from multiple stocks"""
        all_X, all_y = [], []
        
        for symbol, df in data_dict.items():
            processor = self.processors[symbol]
            data = processor.prepare_training_data(df)
            all_X.append(data['X_train'])
            all_y.append(data['y_train'])
        
        return np.concatenate(all_X), np.concatenate(all_y)
