from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from .base_model import BaseModel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class LinearRegressionModel(BaseModel):
    """Linear Regression model wrapper."""
    
    def _create_model(self, **kwargs):
        return LinearRegression(**kwargs)


class ElasticNetModel(BaseModel):
    """Elastic Net regression model wrapper."""
    
    def __init__(self, use_log_transform=True, alpha=0.000001, l1_ratio=0.2, **kwargs):
        """
        Initialize Elastic Net model.
        
        Args:
            use_log_transform: Whether to apply log1p transform to target
            alpha: Regularization strength (default: 0.1, lower than sklearn's 1.0)
            l1_ratio: Mixing parameter (0=ridge, 1=lasso, default: 0.5)
            **kwargs: Additional ElasticNet parameters
        """
        # Set defaults if not provided
        kwargs.setdefault('alpha', alpha)
        kwargs.setdefault('l1_ratio', l1_ratio)
        super().__init__(use_log_transform=use_log_transform, **kwargs)
    
    def _create_model(self, **kwargs):
        return ElasticNet(**kwargs)


class KNeighborsRegressorModel(BaseModel):
    """K-Nearest Neighbors regressor wrapper."""
    
    def _create_model(self, **kwargs):
        return KNeighborsRegressor(**kwargs)


class RandomForestRegressorModel(BaseModel):
    """Random Forest regressor wrapper."""
    
    def _create_model(self, **kwargs):
        return RandomForestRegressor(**kwargs)


class GradientBoostingRegressorModel(BaseModel):
    """Gradient Boosting regressor wrapper."""
    
    def _create_model(self, **kwargs):
        return GradientBoostingRegressor(**kwargs)


class XGBRegressorModel(BaseModel):
    """XGBoost regressor wrapper."""
    
    def _create_model(self, **kwargs):
        return XGBRegressor(**kwargs)


class LGBMRegressorModel(BaseModel):
    """LightGBM regressor wrapper."""
    
    def _create_model(self, **kwargs):
        return LGBMRegressor(**kwargs)


class SVRModel(BaseModel):
    """Support Vector Regression model wrapper."""
    
    def _create_model(self, **kwargs):
        return SVR(**kwargs)

