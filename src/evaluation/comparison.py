"""Model comparison utilities."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelComparison:
    """Utilities for comparing multiple models."""
    
    def __init__(self, fig_dir='fig'):
        """
        Initialize comparison utility.
        
        Args:
            fig_dir: Directory to save figures
        """
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def add_result(self, model_name, metrics):
        """
        Add a model's evaluation results.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics (from ModelEvaluator.evaluate)
        """
        self.results.append(metrics)
    
    def get_comparison_table(self):
        """
        Get comparison table as DataFrame.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('rmse')  # Sort by RMSE (best first)
        return df
    
    def plot_comparison(self, save_path=None, figsize=(16, 6)):
        """
        Create comparison visualization.
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No results to compare. Add results first.")
        
        df = self.get_comparison_table()
        
        metrics = ['rmse', 'mse', 'mae', 'r2']
        metric_labels = ['RMSE', 'MSE', 'MAE', 'R²']
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        for ax, metric, label in zip(axes, metrics, metric_labels):
            bars = ax.barh(df['model'], df[metric], alpha=0.7)
            ax.set_xlabel(label)
            ax.set_title(f'Porównanie modeli - {label}')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, df[metric])):
                ax.text(val, i, f'{val:.3f}', va='center', ha='left' if metric != 'r2' else 'right')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.fig_dir / 'model_comparison.jpg'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metric_comparison(self, metric='rmse', save_path=None, figsize=(10, 6)):
        """
        Create detailed comparison for a single metric.
        
        Args:
            metric: Metric to compare ('rmse', 'mse', 'mae', 'r2')
            save_path: Path to save figure
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No results to compare. Add results first.")
        
        df = self.get_comparison_table()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(df['model'], df[metric], alpha=0.7, color='steelblue')
        ax.set_xlabel(metric.upper())
        ax.set_title(f'Porównanie modeli - {metric.upper()}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, df[metric]):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', va='center', ha='left' if metric != 'r2' else 'right')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.fig_dir / f'model_comparison_{metric}.jpg'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self, filepath='model_comparison_results.csv'):
        """
        Export comparison results to CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        df = self.get_comparison_table()
        df.to_csv(filepath, index=False)
        return df

