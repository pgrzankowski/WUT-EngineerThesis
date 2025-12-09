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

