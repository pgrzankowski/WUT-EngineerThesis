"""Classical diameter approximation model."""

import numpy as np
from .base_model import BaseModel


class ClassicalDiameterModel(BaseModel):
    """Classical diameter approximation using albedo and absolute magnitude."""
    
    def __init__(self, use_log_transform=False, **kwargs):
        """
        Initialize classical model.
        
        Note: Classical model doesn't use log transform as it's a direct formula.
        
        Args:
            use_log_transform: Ignored for classical model (always False)
            **kwargs: Additional parameters (not used)
        """
        super().__init__(use_log_transform=False)
    
    def _create_model(self, **kwargs):
        """Classical model doesn't need a sklearn-style model."""
        return self
    
    def fit(self, X, y):
        """
        Fit the model (no training needed for classical formula).
        
        Args:
            X: Features (must contain 'albedo' and 'H' columns)
            y: Target (not used, for compatibility)
        """
        # Check if required columns exist
        if hasattr(X, 'columns'):
            required_cols = ['albedo', 'H']
            missing = [col for col in required_cols if col not in X.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
        else:
            raise ValueError("X must be a DataFrame with 'albedo' and 'H' columns")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict diameter using classical approximation.
        
        Formula: p = 3.1236 - 0.5 * log10(albedo) - 0.2 * H
                 diameter = 10^p
        
        Args:
            X: Features (must contain 'albedo' and 'H' columns)
            
        Returns:
            Predicted diameters
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if hasattr(X, 'values'):
            # DataFrame
            albedo = X['albedo'].values
            H = X['H'].values
        else:
            raise ValueError("X must be a DataFrame with 'albedo' and 'H' columns")
        
        # Classical approximation formula
        p = 3.1236 - 0.5 * np.log10(albedo) - 0.2 * H
        diameter = 10 ** p
        
        return diameter
    
    def get_params(self, deep=True):
        """Get model parameters."""
        return {'use_log_transform': False}
    
    def set_params(self, **params):
        """Set model parameters (no parameters to set)."""
        return self

