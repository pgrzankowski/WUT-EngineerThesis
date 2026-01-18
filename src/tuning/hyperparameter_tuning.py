"""Hyperparameter tuning framework using Optuna."""

import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Optional

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterTuner:
    """Unified interface for hyperparameter tuning using Optuna."""
    
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
    
    def tune_model(self, model, X, y, model_name, 
                   param_grid=None, cv=None, n_iter=None, 
                   scoring=None, n_jobs=None, random_state=None, verbose=None):
        """
        Tune hyperparameters for a model using Optuna.
        
        Args:
            model: BaseModel instance
            X: Training features
            y: Training target
            model_name: Name of the model (for config lookup)
            param_grid: Parameter grid. If None, uses config
            cv: Cross-validation folds. If None, uses config
            n_iter: Number of Optuna trials. If None, uses config
            scoring: Scoring metric. If None, uses config
            n_jobs: Number of parallel jobs for cross-validation. If None, uses config
            random_state: Random seed. If None, uses config
            verbose: Verbosity level. If None, uses config
            
        Returns:
            Best model (fitted) and Optuna study results
        """
        
        # Get config values
        tuning_config = self.get_tuning_config()
        model_config = self.get_model_config(model_name)
        
        cv = cv or tuning_config.get('cv', 5)
        n_iter = n_iter or tuning_config.get('n_iter', 50)
        scoring = scoring or tuning_config.get('scoring', 'neg_mean_squared_error')
        n_jobs = n_jobs if n_jobs is not None else tuning_config.get('n_jobs', 2)
        random_state = random_state if random_state is not None else tuning_config.get('random_state', 42)
        verbose = verbose if verbose is not None else tuning_config.get('verbose', 1)
        
        # Ensure n_jobs is not -1 to prevent system crashes
        if n_jobs == -1:
            n_jobs = 2
        
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
        
        return self._tune_with_optuna(
            model, X, y, model_name, param_grid, 
            cv, scoring, n_iter, random_state, n_jobs, verbose
        )
    
    def _tune_with_optuna(self, model, X, y, model_name, param_grid, 
                          cv, scoring, n_trials, random_state, n_jobs, verbose):
        """Tune using Optuna."""
        
        # Set Optuna verbosity
        if verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif verbose == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values, list):
                    if len(param_values) == 0:
                        continue
                    # Check if all values are numeric for range-based sampling
                    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in param_values):
                        if all(isinstance(v, int) for v in param_values):
                            params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                        else:
                            params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        # Categorical parameter
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    params[param_name] = param_values
            
            # Create model with sampled parameters
            model_copy = model.__class__(**params)
            sklearn_model = model_copy._create_model(**model_copy.model_kwargs)
            
            # Cross-validation score with controlled n_jobs
            scores = cross_val_score(sklearn_model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
            return scores.mean()
        
        study = optuna.create_study(
            direction='maximize', 
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=(verbose > 0))
        
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
