"""Hyperparameter tuning framework."""

import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Dict, Any, Optional

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterTuner:
    """Unified interface for hyperparameter tuning."""
    
    def __init__(self, config_path=None, config_dict=None):
        """
        Initialize tuner.
        
        Args:
            config_path: Path to YAML config file
            config_dict: Dictionary with config (alternative to config_path)
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            self.config = {}
    
    def get_model_config(self, model_name):
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model (key in config)
            
        Returns:
            Dictionary with model configuration
        """
        if 'models' not in self.config:
            return {}
        return self.config['models'].get(model_name, {})
    
    def get_tuning_config(self):
        """Get tuning configuration."""
        return self.config.get('tuning', {})
    
    def tune_model(self, model, X, y, model_name, method=None, 
                   param_grid=None, cv=None, n_iter=None, 
                   scoring=None, n_jobs=None, random_state=None, verbose=None):
        """
        Tune hyperparameters for a model.
        
        Args:
            model: BaseModel instance
            X: Training features
            y: Training target
            model_name: Name of the model (for config lookup)
            method: Tuning method ('grid', 'randomized', 'optuna'). If None, uses config
            param_grid: Parameter grid. If None, uses config
            cv: Cross-validation folds. If None, uses config
            n_iter: Number of iterations for RandomizedSearch. If None, uses config
            scoring: Scoring metric. If None, uses config
            n_jobs: Number of parallel jobs. If None, uses config
            random_state: Random seed. If None, uses config
            verbose: Verbosity level. If None, uses config
            
        Returns:
            Best model (fitted) and search results
        """
        # Get config values
        tuning_config = self.get_tuning_config()
        model_config = self.get_model_config(model_name)
        
        method = method or tuning_config.get('method', 'randomized')
        cv = cv or tuning_config.get('cv', 5)
        n_iter = n_iter or tuning_config.get('n_iter', 50)
        scoring = scoring or tuning_config.get('scoring', 'neg_mean_squared_error')
        n_jobs = n_jobs if n_jobs is not None else tuning_config.get('n_jobs', -1)
        random_state = random_state if random_state is not None else tuning_config.get('random_state', 42)
        verbose = verbose if verbose is not None else tuning_config.get('verbose', 1)
        
        # Get parameter grid
        if param_grid is None:
            param_grid = model_config.get('param_grid', {})
        
        # Convert YAML boolean strings to Python booleans
        param_grid = self._convert_param_grid(param_grid)
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Apply log transform if model uses it
        use_log_transform = model_config.get('use_log_transform', True)
        if use_log_transform:
            y = np.log1p(y)
        
        # Get underlying sklearn model
        if hasattr(model, 'model') and model.model is not None:
            sklearn_model = model.model
        else:
            # Create model with default params
            default_params = model_config.get('default_params', {})
            model.set_params(**default_params)
            sklearn_model = model._create_model()
        
        # Perform tuning
        if method == 'grid':
            search = GridSearchCV(
                sklearn_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=True
            )
        elif method == 'randomized':
            search = RandomizedSearchCV(
                sklearn_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                return_train_score=True
            )
        elif method == 'optuna':
            if not OPTUNA_AVAILABLE:
                raise ImportError("Optuna is not installed. Install it with: pip install optuna")
            return self._tune_with_optuna(model, X, y, model_name, param_grid, cv, scoring, n_iter, random_state)
        else:
            raise ValueError(f"Unknown tuning method: {method}")
        
        # Fit search
        search.fit(X, y)
        
        # Update model with best parameters
        model.set_params(**search.best_params_)
        model.model = search.best_estimator_
        model.is_fitted = True
        
        return model, search
    
    def _tune_with_optuna(self, model, X, y, model_name, param_grid, cv, scoring, n_trials, random_state):
        """Tune using Optuna."""
        from sklearn.model_selection import cross_val_score
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        if all(isinstance(v, int) for v in param_values):
                            params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                        else:
                            params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    params[param_name] = param_values
            
            # Create model with sampled parameters
            model_copy = model.__class__(**params)
            sklearn_model = model_copy._create_model()
            
            # Cross-validation score
            scores = cross_val_score(sklearn_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials)
        
        # Update model with best parameters
        model.set_params(**study.best_params)
        model.model = model._create_model()
        model.model.fit(X, y)
        model.is_fitted = True
        
        return model, study
    
    def _convert_param_grid(self, param_grid):
        """Convert YAML param grid to Python format."""
        converted = {}
        for key, value in param_grid.items():
            if isinstance(value, list):
                converted[key] = [self._convert_value(v) for v in value]
            else:
                converted[key] = self._convert_value(value)
        return converted
    
    def _convert_value(self, value):
        """Convert YAML value to Python value."""
        if isinstance(value, str):
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
            elif value.lower() == 'null' or value.lower() == 'none':
                return None
        return value

