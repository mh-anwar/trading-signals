"""
Training pipeline for LSTM models
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import structlog
from tqdm import tqdm
import os
from datetime import datetime

from ml.models.lstm_model import StockPredictor, EnsemblePredictor
from ml.data.data_processor import StockDataProcessor, MultiStockProcessor

logger = structlog.get_logger()


class StockTrainer:
    """Training pipeline for stock prediction models"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.input_size = input_size
        
        # Initialize model
        self.model = StockPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            device=self.device
        )
        
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def prepare_data(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders"""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            loss = self.model.train_step(batch_x, batch_y)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in val_loader:
            loss = self.model.validate(batch_x, batch_y)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        patience: int = 20,
        save_path: str = None
    ) -> Dict[str, Any]:
        """Train the model"""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(X_train, y_train, X_val, y_val)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.model.scheduler.step(val_loss)
            current_lr = self.model.optimizer.param_groups[0]['lr']
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(current_lr)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    self.model.save_model(save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        return {
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'training_history': self.training_history
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        # Make predictions
        y_pred = self.model.predict(torch.FloatTensor(X_test))
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy
        y_test_direction = np.diff(y_test) > 0
        y_pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(y_test_direction == y_pred_direction)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.training_history['learning_rate'])
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # Loss difference
        loss_diff = np.array(self.training_history['train_loss']) - np.array(self.training_history['val_loss'])
        axes[1, 0].plot(loss_diff)
        axes[1, 0].set_title('Train-Validation Loss Difference')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Difference')
        axes[1, 0].grid(True)
        
        # Loss ratio
        loss_ratio = np.array(self.training_history['val_loss']) / np.array(self.training_history['train_loss'])
        axes[1, 1].plot(loss_ratio)
        axes[1, 1].set_title('Validation/Train Loss Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Ratio')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, X_test: np.ndarray, y_test: np.ndarray, 
                        y_pred: np.ndarray, save_path: str = None):
        """Plot predictions vs actual values"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Time series plot
        x_axis = range(len(y_test))
        axes[0].plot(x_axis, y_test, label='Actual', alpha=0.7)
        axes[0].plot(x_axis, y_pred, label='Predicted', alpha=0.7)
        axes[0].set_title('Predictions vs Actual Values')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Normalized Price')
        axes[0].legend()
        axes[0].grid(True)
        
        # Scatter plot
        axes[1].scatter(y_test, y_pred, alpha=0.6)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual')
        axes[1].set_ylabel('Predicted')
        axes[1].set_title('Actual vs Predicted Scatter Plot')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class MultiStockTrainer:
    """Trainer for multiple stocks"""
    
    def __init__(self, symbols: List[str], **kwargs):
        self.symbols = symbols
        self.trainers = {}
        self.kwargs = kwargs
    
    def train_individual_models(self, data_dict: Dict[str, Dict[str, Any]], 
                              epochs: int = 100) -> Dict[str, Any]:
        """Train individual models for each stock"""
        results = {}
        
        for symbol in self.symbols:
            logger.info(f"Training model for {symbol}")
            
            # Initialize trainer
            data = data_dict[symbol]
            input_size = data['X_train'].shape[2]
            
            trainer = StockTrainer(input_size=input_size, **self.kwargs)
            
            # Train model
            training_result = trainer.train(
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                epochs=epochs
            )
            
            # Evaluate model
            evaluation = trainer.evaluate(data['X_test'], data['y_test'])
            
            results[symbol] = {
                'trainer': trainer,
                'training_result': training_result,
                'evaluation': evaluation
            }
            
            logger.info(f"Completed training for {symbol}: R² = {evaluation['r2']:.4f}")
        
        return results
    
    def train_ensemble_model(self, data_dict: Dict[str, Dict[str, Any]], 
                           epochs: int = 100) -> Dict[str, Any]:
        """Train ensemble model on all stocks"""
        logger.info("Training ensemble model")
        
        # Combine data from all stocks
        all_X_train, all_y_train = [], []
        all_X_test, all_y_test = [], []
        
        for symbol, data in data_dict.items():
            all_X_train.append(data['X_train'])
            all_y_train.append(data['y_train'])
            all_X_test.append(data['X_test'])
            all_y_test.append(data['y_test'])
        
        X_train_combined = np.concatenate(all_X_train)
        y_train_combined = np.concatenate(all_y_train)
        X_test_combined = np.concatenate(all_X_test)
        y_test_combined = np.concatenate(all_y_test)
        
        # Train ensemble model
        input_size = X_train_combined.shape[2]
        trainer = StockTrainer(input_size=input_size, **self.kwargs)
        
        training_result = trainer.train(
            X_train_combined, y_train_combined,
            X_test_combined, y_test_combined,
            epochs=epochs
        )
        
        evaluation = trainer.evaluate(X_test_combined, y_test_combined)
        
        return {
            'trainer': trainer,
            'training_result': training_result,
            'evaluation': evaluation
        }
