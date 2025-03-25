import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd

class Evaluator:
    def __init__(self, config):
        self.config = config
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute various evaluation metrics."""
        metrics = {}
        
        if 'mse' in self.config.evaluation.metrics:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        if 'mae' in self.config.evaluation.metrics:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        if 'r2' in self.config.evaluation.metrics:
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def compute_fairness_metrics(self, 
                               predictions: np.ndarray,
                               sensitive_attributes: np.ndarray) -> Dict[str, float]:
        """Compute fairness metrics."""
        metrics = {}
        
        if 'demographic_parity' in self.config.evaluation.fairness_metrics:
            # Calculate demographic parity
            unique_groups = np.unique(sensitive_attributes)
            group_predictions = [predictions[sensitive_attributes == group] for group in unique_groups]
            group_means = [np.mean(preds) for preds in group_predictions]
            metrics['demographic_parity'] = np.std(group_means)
        
        if 'equal_opportunity' in self.config.evaluation.fairness_metrics:
            # Calculate equal opportunity
            positive_predictions = predictions > 0.5
            group_opportunities = []
            for group in unique_groups:
                group_mask = sensitive_attributes == group
                group_opportunity = np.mean(positive_predictions[group_mask])
                group_opportunities.append(group_opportunity)
            metrics['equal_opportunity'] = np.max(group_opportunities) - np.min(group_opportunities)
        
        return metrics
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: str = None):
        """Plot training history."""
        plt.figure(figsize=(12, 6))
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot bias scores
        plt.subplot(1, 2, 2)
        plt.plot(history['bias_scores'], label='Bias Score')
        plt.axhline(y=self.config.model.bias_threshold, color='r', linestyle='--', label='Threshold')
        plt.title('Bias Scores')
        plt.xlabel('Epoch')
        plt.ylabel('Bias Score')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_recommendation_distribution(self, 
                                      recommendations: np.ndarray,
                                      save_path: str = None):
        """Plot distribution of recommendations."""
        plt.figure(figsize=(10, 6))
        sns.histplot(recommendations, bins=30)
        plt.title('Distribution of Recommendations')
        plt.xlabel('Recommendation Score')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_fairness_metrics(self, 
                            fairness_metrics: Dict[str, float],
                            save_path: str = None):
        """Plot fairness metrics."""
        plt.figure(figsize=(8, 6))
        metrics = list(fairness_metrics.keys())
        values = list(fairness_metrics.values())
        
        plt.bar(metrics, values)
        plt.axhline(y=self.config.evaluation.fairness_threshold, 
                   color='r', linestyle='--', label='Threshold')
        plt.title('Fairness Metrics')
        plt.xticks(rotation=45)
        plt.ylabel('Score')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def generate_report(self, 
                       metrics: Dict[str, float],
                       fairness_metrics: Dict[str, float],
                       save_path: str = None) -> str:
        """Generate a comprehensive evaluation report."""
        report = "Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Performance metrics
        report += "Performance Metrics:\n"
        report += "-" * 20 + "\n"
        for metric, value in metrics.items():
            report += f"{metric.upper()}: {value:.4f}\n"
        
        # Fairness metrics
        report += "\nFairness Metrics:\n"
        report += "-" * 20 + "\n"
        for metric, value in fairness_metrics.items():
            report += f"{metric}: {value:.4f}\n"
            if value > self.config.evaluation.fairness_threshold:
                report += f"⚠️ Warning: {metric} exceeds fairness threshold\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report 