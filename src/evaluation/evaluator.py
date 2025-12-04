"""Model evaluation utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error
from pathlib import Path


class ModelEvaluator:
    """Standardized model evaluation."""
    
    def __init__(self, fig_dir='fig'):
        """
        Initialize evaluator.
        
        Args:
            fig_dir: Directory to save figures
        """
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self, y_true, y_pred, model_name='model'):
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model (for display)
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy if needed
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
        
        metrics = {
            'model': model_name,
            'rmse': root_mean_squared_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
        
        return metrics
    
    def plot_predictions(self, y_true, y_pred, model_name='model', 
                        save_path=None, figsize=(10, 10)):
        """
        Create prediction scatter plot.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save figure (if None, auto-generate)
            figsize: Figure size
        """
        # Convert to numpy if needed
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
        
        # Calculate metrics
        metrics = self.evaluate(y_true, y_pred, model_name)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        metrics_label = f"RMSE: {metrics['rmse']:.3f}\nMSE: {metrics['mse']:.3f}\nMAE: {metrics['mae']:.3f}\nR²: {metrics['r2']:.3f}"
        
        sns.scatterplot(x=y_true, y=y_pred, ax=ax, label=metrics_label)
        ax.set_xlabel('Prawdziwa średnica (km)')
        ax.set_ylabel('Przewidziana średnica (km)')
        ax.grid(True)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(title='Metryki', loc='upper left', frameon=True)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.fig_dir / f'{model_name.lower().replace(" ", "_")}_prediction_visual.jpg'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return metrics
    
    def plot_residuals(self, y_true, y_pred, model_name='model', 
                      save_path=None, figsize=(10, 6)):
        """
        Create residual plot.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save figure (if None, auto-generate)
            figsize: Figure size
        """
        # Convert to numpy if needed
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
        
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals vs predicted
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Przewidziana średnica (km)')
        ax1.set_ylabel('Residua')
        ax1.set_title('Residua vs Przewidziane wartości')
        ax1.grid(True, alpha=0.3)
        
        # Residuals distribution
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Residua')
        ax2.set_ylabel('Częstość')
        ax2.set_title('Rozkład residuów')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.fig_dir / f'{model_name.lower().replace(" ", "_")}_residuals.jpg'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

