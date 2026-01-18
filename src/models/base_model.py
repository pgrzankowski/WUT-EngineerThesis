"""Base model class for unified interface."""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all regression models."""
    
    def __init__(self, use_log_transform=True, **kwargs):
        """
        Initialize base model.
        
        Args:
            use_log_transform: Whether to apply log1p transform to target
            **kwargs: Additional model-specific parameters
        """
        self.use_log_transform = False
        self.model_kwargs = kwargs  # Store kwargs for model creation
        self.is_fitted = False
        self.model = None
    
    @abstractmethod
    def _create_model(self, **kwargs):
        """Create the underlying model instance."""
        pass
    
    def fit(self, X, y):
        """
        Train the model.
        
        Args:
            X: Training features (numpy array or pandas DataFrame)
            y: Training target (numpy array or pandas Series)
        """
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Apply log transform if specified
        # if self.use_log_transform:
        #     y = np.log1p(y)
        
        # Create and fit model with stored kwargs
        self.model = self._create_model(**self.model_kwargs)
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features (numpy array or pandas DataFrame)
            
        Returns:
            Predictions (inverse log transformed if use_log_transform=True)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        
        # Get predictions
        preds = self.model.predict(X)
        
        # Inverse log transform if needed
        # if self.use_log_transform:
        #     preds = np.expm1(preds)
        
        return preds
    
    def get_params(self, deep=True):
        """Get model parameters."""
        if self.model is None:
            return {'use_log_transform': self.use_log_transform}
        params = self.model.get_params(deep=deep) if hasattr(self.model, 'get_params') else {}
        params['use_log_transform'] = self.use_log_transform
        return params
    
    def set_params(self, **params):
        """Set model parameters."""
        if 'use_log_transform' in params:
            self.use_log_transform = params.pop('use_log_transform')
        
        # Update stored kwargs
        self.model_kwargs.update(params)
        
        if self.model is not None and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        
        return self
    
    def get_model_name(self):
        """Get the name of the model."""
        return self.__class__.__name__.replace('Model', '')

