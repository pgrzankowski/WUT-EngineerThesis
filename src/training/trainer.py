"""Unified training interface."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from pathlib import Path


class ModelTrainer:
    """Unified interface for model training and data preprocessing."""
    
    def __init__(self, test_size=0.2, val_size=0.25, random_state=42, 
                 scale_features=True, use_log_transform=True):
        """
        Initialize trainer.
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed
            scale_features: Whether to scale features using StandardScaler
            use_log_transform: Whether to apply log1p transform to target
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scale_features = scale_features
        self.use_log_transform = use_log_transform
        
        self.scaler = StandardScaler() if scale_features else None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None
    
    def prepare_data(self, X, y):
        """
        Prepare train/val/test splits and apply preprocessing.
        
        Args:
            X: Features (DataFrame or array)
            y: Target (Series or array)
        """
        # Convert to numpy if needed (but keep original for reference)
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.val_size, random_state=self.random_state
        )
        
        # Store original splits
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Scale features if requested
        if self.scale_features:
            self.X_train_scaled = self.scaler.fit_transform(X_train)
            self.X_val_scaled = self.scaler.transform(X_val)
            self.X_test_scaled = self.scaler.transform(X_test)
        else:
            # Convert to numpy arrays
            if isinstance(X_train, pd.DataFrame):
                self.X_train_scaled = X_train.values
            else:
                self.X_train_scaled = X_train
            
            if isinstance(X_val, pd.DataFrame):
                self.X_val_scaled = X_val.values
            else:
                self.X_val_scaled = X_val
            
            if isinstance(X_test, pd.DataFrame):
                self.X_test_scaled = X_test.values
            else:
                self.X_test_scaled = X_test
        
        return self
    
    def get_train_data(self, scaled=True):
        """Get training data."""
        if scaled:
            return self.X_train_scaled, self.y_train
        # Return as DataFrame if original was DataFrame
        if self.feature_names is not None and isinstance(self.X_train, pd.DataFrame):
            return self.X_train, self.y_train
        return self.X_train, self.y_train
    
    def get_val_data(self, scaled=True):
        """Get validation data."""
        if scaled:
            return self.X_val_scaled, self.y_val
        # Return as DataFrame if original was DataFrame
        if self.feature_names is not None and isinstance(self.X_val, pd.DataFrame):
            return self.X_val, self.y_val
        return self.X_val, self.y_val
    
    def get_test_data(self, scaled=True):
        """Get test data."""
        if scaled:
            return self.X_test_scaled, self.y_test
        # Return as DataFrame if original was DataFrame
        if self.feature_names is not None and isinstance(self.X_test, pd.DataFrame):
            return self.X_test, self.y_test
        return self.X_test, self.y_test
    
    def train_model(self, model, use_val_for_training=False):
        """
        Train a model.
        
        Args:
            model: BaseModel instance
            use_val_for_training: If True, combine train and val sets for training
            
        Returns:
            Trained model
        """
        if self.X_train_scaled is None:
            raise ValueError("Data must be prepared first. Call prepare_data()")
        
        # Get training data
        if use_val_for_training:
            X_train = np.vstack([self.X_train_scaled, self.X_val_scaled])
            y_train = pd.concat([self.y_train, self.y_val]) if isinstance(self.y_train, pd.Series) else np.concatenate([self.y_train, self.y_val])
        else:
            X_train = self.X_train_scaled
            y_train = self.y_train
        
        # Special handling for MLP which needs validation data
        if hasattr(model, '__class__') and 'MLP' in model.__class__.__name__:
            if not use_val_for_training:
                model.fit(X_train, y_train, X_val=self.X_val_scaled, y_val=self.y_val)
            else:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        return model
    
    def save_model(self, model, filepath):
        """Save a trained model."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
            }, f)
    
    def load_model(self, filepath):
        """Load a saved model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            self.scaler = data.get('scaler')
            self.feature_names = data.get('feature_names')
            return model

