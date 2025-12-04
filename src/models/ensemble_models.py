"""Ensemble model implementations."""

from sklearn.ensemble import VotingRegressor, StackingRegressor
from .base_model import BaseModel
import numpy as np


class VotingRegressorModel(BaseModel):
    """Voting ensemble of multiple regressors."""
    
    def __init__(self, estimators, use_log_transform=True, weights=None, **kwargs):
        """
        Initialize voting regressor.
        
        Args:
            estimators: List of (name, model) tuples where model is a BaseModel instance
            use_log_transform: Whether to apply log1p transform to target
            weights: Optional weights for each estimator
            **kwargs: Additional parameters for VotingRegressor
        """
        super().__init__(use_log_transform=use_log_transform)
        self.estimators = estimators
        self.weights = weights
        self.estimator_kwargs = kwargs
    
    def _create_model(self, **kwargs):
        """Create VotingRegressor with underlying sklearn models."""
        # Extract sklearn models from BaseModel wrappers
        sklearn_estimators = []
        for name, model in self.estimators:
            if hasattr(model, 'model') and model.model is not None:
                sklearn_estimators.append((name, model.model))
            else:
                # Model not fitted yet, need to create it
                if hasattr(model, '_create_model'):
                    sklearn_estimators.append((name, model._create_model()))
                else:
                    raise ValueError(f"Estimator {name} is not a valid model")
        
        return VotingRegressor(
            estimators=sklearn_estimators,
            weights=self.weights,
            **self.estimator_kwargs
        )
    
    def fit(self, X, y):
        """Fit the voting regressor."""
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Fit individual estimators first
        for name, estimator in self.estimators:
            if hasattr(estimator, 'fit'):
                if self.use_log_transform:
                    y_transformed = np.log1p(y)
                else:
                    y_transformed = y
                estimator.fit(X, y_transformed if self.use_log_transform else y)
        
        # Apply log transform if specified
        if self.use_log_transform:
            y = np.log1p(y)
        
        # Create and fit voting regressor
        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class StackingRegressorModel(BaseModel):
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, estimators, final_estimator=None, use_log_transform=True, 
                 cv=5, **kwargs):
        """
        Initialize stacking regressor.
        
        Args:
            estimators: List of (name, model) tuples where model is a BaseModel instance
            final_estimator: Meta-learner (BaseModel instance). Defaults to LinearRegression
            use_log_transform: Whether to apply log1p transform to target
            cv: Cross-validation folds for stacking
            **kwargs: Additional parameters for StackingRegressor
        """
        super().__init__(use_log_transform=use_log_transform)
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.estimator_kwargs = kwargs
    
    def _create_model(self, **kwargs):
        """Create StackingRegressor with underlying sklearn models."""
        # Extract sklearn models from BaseModel wrappers
        sklearn_estimators = []
        for name, model in self.estimators:
            if hasattr(model, 'model') and model.model is not None:
                sklearn_estimators.append((name, model.model))
            else:
                # Model not fitted yet, need to create it
                if hasattr(model, '_create_model'):
                    sklearn_estimators.append((name, model._create_model()))
                else:
                    raise ValueError(f"Estimator {name} is not a valid model")
        
        # Handle final estimator
        final_sklearn_estimator = None
        if self.final_estimator is not None:
            if hasattr(self.final_estimator, 'model') and self.final_estimator.model is not None:
                final_sklearn_estimator = self.final_estimator.model
            elif hasattr(self.final_estimator, '_create_model'):
                final_sklearn_estimator = self.final_estimator._create_model()
            else:
                raise ValueError("Final estimator is not a valid model")
        else:
            # Default to LinearRegression
            from sklearn.linear_model import LinearRegression
            final_sklearn_estimator = LinearRegression()
        
        return StackingRegressor(
            estimators=sklearn_estimators,
            final_estimator=final_sklearn_estimator,
            cv=self.cv,
            **self.estimator_kwargs
        )
    
    def fit(self, X, y):
        """Fit the stacking regressor."""
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Apply log transform if specified
        if self.use_log_transform:
            y = np.log1p(y)
        
        # Create and fit stacking regressor
        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

