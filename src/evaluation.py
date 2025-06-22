# src/evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import mlflow

from .config import Config
from .data_ingestion import DataIngestion
from .utils import logger, load_pickle, calculate_metrics, log_metrics

class ModelEvaluator:
    """
    Comprehensive model evaluation class for regression models.
    """
    
    def __init__(self):
        """Initialize ModelEvaluator with configuration."""
        self.config = Config()
        self.data_ingestion = DataIngestion()
        self.evaluation_results = {}
        
        logger.info("ModelEvaluator initialized")
    
    def load_model_and_data(self) -> Tuple[Any, pd.DataFrame, pd.Series]:
        """
        Load the best model and test data.
        
        Returns:
            Tuple of (model, X_test, y_test)
        """
        try:
            # Load the best model
            model = load_pickle(self.config.MODEL_ARTIFACT_PATH)
            
            # Load model metadata to get feature preprocessing info
            metadata_path = self.config.MODELS_DIR / "model_metadata.pkl"
            if metadata_path.exists():
                metadata = load_pickle(metadata_path)
                logger.info(f"Loaded model: {metadata['model_name']} ({metadata['model_type']})")
            
            # Load processed data and split
            df = self.data_ingestion.load_processed_data()
            X, y = self.data_ingestion.split_features_target(df)
            
            # For evaluation, we'll use a subset or recreate the test split
            # In production, you'd typically save the test set separately
            from sklearn.model_selection import train_test_split
            
            _, X_test, _, y_test = train_test_split(
                X, y,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE
            )
            
            # Apply the same preprocessing as during training
            scaler_path = self.config.SCALER_PATH
            if scaler_path.exists():
                scaler = load_pickle(scaler_path)
                numerical_columns = X_test.select_dtypes(include=[np.number]).columns.tolist()
                if numerical_columns:
                    X_test_scaled = X_test.copy()
                    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])
                    X_test = X_test_scaled
                    logger.info("Applied feature scaling to test data")
            
            logger.info(f"Loaded model and test data: {X_test.shape}")
            return model, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error loading model and data: {e}")
            raise
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model with comprehensive metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate basic metrics
            metrics = calculate_metrics(y_test, y_pred)
            
            # Add additional metrics
            residuals = y_test - y_pred
            metrics.update({
                'mean_residual': np.mean(residuals),
                'std_residual': np.std(residuals),
                'max_error': np.max(np.abs(residuals)),
                'prediction_range': np.ptp(y_pred),  # Peak-to-peak range
                'actual_range': np.ptp(y_test)
            })
            
            # Calculate percentage metrics
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            metrics['mape'] = mape
            
            logger.info("Model evaluation completed")
            log_metrics(metrics, "Evaluation")
            
            self.evaluation_results = {
                'metrics': metrics,
                'predictions': y_pred,
                'actuals': y_test.values,
                'residuals': residuals
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def create_evaluation_plots(self, save_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        Create comprehensive evaluation plots.
        
        Args:
            save_plots: Whether to save plots to docs directory
            
        Returns:
            Dictionary of matplotlib figures
        """
        try:
            if not self.evaluation_results:
                raise ValueError("No evaluation results found. Run evaluate_model() first.")
            
            plots = {}
            
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Predicted vs Actual scatter plot
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            
            y_pred = self.evaluation_results['predictions']
            y_actual = self.evaluation_results['actuals']
            
            ax1.scatter(y_actual, y_pred, alpha=0.6, s=50)
            ax1.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual Price', fontsize=12)
            ax1.set_ylabel('Predicted Price', fontsize=12)
            ax1.set_title('Predicted vs Actual House Prices', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add R² score to the plot
            r2_score = self.evaluation_results['metrics']['r2']
            ax1.text(0.05, 0.95, f'R² = {r2_score:.4f}', transform=ax1.transAxes, 
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plots['predicted_vs_actual'] = fig1
            
            # 2. Residuals plot
            fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(15, 6))
            
            residuals = self.evaluation_results['residuals']
            
            # Residuals vs Predicted
            ax2a.scatter(y_pred, residuals, alpha=0.6, s=50)
            ax2a.axhline(y=0, color='r', linestyle='--', lw=2)
            ax2a.set_xlabel('Predicted Price', fontsize=12)
            ax2a.set_ylabel('Residuals', fontsize=12)
            ax2a.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
            ax2a.grid(True, alpha=0.3)
            
            # Histogram of residuals
            ax2b.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2b.axvline(x=0, color='r', linestyle='--', lw=2)
            ax2b.set_xlabel('Residuals', fontsize=12)
            ax2b.set_ylabel('Frequency', fontsize=12)
            ax2b.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
            ax2b.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['residuals'] = fig2
            
            # 3. Error metrics visualization
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            metrics = self.evaluation_results['metrics']
            error_metrics = ['mae', 'mse', 'rmse']
            error_values = [metrics[metric] for metric in error_metrics]
            
            bars = ax3.bar(error_metrics, error_values, color=['coral', 'lightblue', 'lightgreen'])
            ax3.set_xlabel('Metrics', fontsize=12)
            ax3.set_ylabel('Values', fontsize=12)
            ax3.set_title('Error Metrics Comparison', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, error_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=11)
            
            plots['error_metrics'] = fig3
            
            # 4. Prediction error distribution
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            
            percentage_errors = np.abs((y_actual - y_pred) / y_actual) * 100
            
            ax4.hist(percentage_errors, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax4.axvline(x=np.mean(percentage_errors), color='r', linestyle='--', lw=2, 
                       label=f'Mean: {np.mean(percentage_errors):.2f}%')
            ax4.axvline(x=np.median(percentage_errors), color='g', linestyle='--', lw=2, 
                       label=f'Median: {np.median(percentage_errors):.2f}%')
            ax4.set_xlabel('Absolute Percentage Error (%)', fontsize=12)
            ax4.set_ylabel('Frequency', fontsize=12)
            ax4.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plots['error_distribution'] = fig4
            
            # Save plots if requested
            if save_plots:
                plots_dir = self.config.DOCS_DIR / "evaluation_plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                for plot_name, fig in plots.items():
                    plot_path = plots_dir / f"{plot_name}.png"
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved plot: {plot_path}")
            
            logger.info("Created all evaluation plots")
            return plots
            
        except Exception as e:
            logger.error(f"Error creating evaluation plots: {e}")
            raise
    
    def save_evaluation_report(self) -> None:
        """Save comprehensive evaluation report to docs."""
        try:
            if not self.evaluation_results:
                raise ValueError("No evaluation results found. Run evaluate_model() first.")
            
            metrics = self.evaluation_results['metrics']
            
            # Create evaluation report
            report = {
                'model_evaluation_summary': {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'test_samples': len(self.evaluation_results['actuals']),
                    'metrics': metrics
                },
                'performance_analysis': {
                    'r2_interpretation': self._interpret_r2_score(metrics['r2']),
                    'rmse_interpretation': self._interpret_rmse(metrics['rmse']),
                    'mape_interpretation': self._interpret_mape(metrics['mape']),
                    'overall_performance': self._get_overall_performance_rating(metrics)
                },
                'recommendations': self._generate_recommendations(metrics)
            }
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                return obj
            
            report = convert_numpy_types(report)
            
            # Save as JSON
            report_path = self.config.DOCS_DIR / "evaluation_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Create markdown summary
            self._create_markdown_summary(report)
            
            logger.info(f"Evaluation report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")
            raise
    
    def _interpret_r2_score(self, r2: float) -> str:
        """Interpret R² score."""
        if r2 >= 0.9:
            return "Excellent: Model explains ≥90% of variance"
        elif r2 >= 0.8:
            return "Good: Model explains 80-90% of variance"
        elif r2 >= 0.6:
            return "Moderate: Model explains 60-80% of variance"
        elif r2 >= 0.4:
            return "Poor: Model explains 40-60% of variance"
        else:
            return "Very Poor: Model explains <40% of variance"
    
    def _interpret_rmse(self, rmse: float) -> str:
        """Interpret RMSE value."""
        # This is context-dependent; adjust based on your price range
        if rmse < 20000:
            return "Excellent: Very low prediction error"
        elif rmse < 40000:
            return "Good: Acceptable prediction error"
        elif rmse < 60000:
            return "Moderate: Noticeable prediction error"
        else:
            return "Poor: High prediction error"
    
    def _interpret_mape(self, mape: float) -> str:
        """Interpret MAPE value."""
        if mape < 5:
            return "Excellent: Very accurate predictions"
        elif mape < 10:
            return "Good: Accurate predictions"
        elif mape < 15:
            return "Moderate: Reasonably accurate predictions"
        elif mape < 25:
            return "Poor: Less accurate predictions"
        else:
            return "Very Poor: Inaccurate predictions"
    
    def _get_overall_performance_rating(self, metrics: Dict[str, float]) -> str:
        """Get overall performance rating."""
        r2 = metrics['r2']
        mape = metrics['mape']
        
        if r2 >= 0.8 and mape < 10:
            return "Excellent"
        elif r2 >= 0.6 and mape < 15:
            return "Good"
        elif r2 >= 0.4 and mape < 25:
            return "Moderate"
        else:
            return "Needs Improvement"
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> list:
        """Generate improvement recommendations."""
        recommendations = []
        
        if metrics['r2'] < 0.6:
            recommendations.append("Consider feature engineering or more complex models")
        
        if metrics['mape'] > 15:
            recommendations.append("High prediction errors - review outliers and data quality")
        
        if abs(metrics['mean_residual']) > 1000:
            recommendations.append("Model shows bias - consider feature transformations")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory")
        
        return recommendations
    
    def _create_markdown_summary(self, report: dict) -> None:
        """Create markdown summary of evaluation."""
        markdown_content = f"""# Model Evaluation Report

## Performance Summary

**Overall Rating**: {report['performance_analysis']['overall_performance']}

### Key Metrics
- **R² Score**: {report['model_evaluation_summary']['metrics']['r2']:.4f}
- **RMSE**: ${report['model_evaluation_summary']['metrics']['rmse']:,.2f}
- **MAE**: ${report['model_evaluation_summary']['metrics']['mae']:,.2f}
- **MAPE**: {report['model_evaluation_summary']['metrics']['mape']:.2f}%

### Performance Analysis
- **R² Interpretation**: {report['performance_analysis']['r2_interpretation']}
- **RMSE Interpretation**: {report['performance_analysis']['rmse_interpretation']}
- **MAPE Interpretation**: {report['performance_analysis']['mape_interpretation']}

### Recommendations
"""
        for rec in report['recommendations']:
            markdown_content += f"- {rec}\n"
        
        markdown_content += f"""
### Test Details
- **Test Samples**: {report['model_evaluation_summary']['test_samples']}
- **Evaluation Date**: {report['model_evaluation_summary']['timestamp']}

### Plots
Evaluation plots are available in the `docs/evaluation_plots/` directory:
- `predicted_vs_actual.png`: Scatter plot of predictions vs actual values
- `residuals.png`: Residual analysis plots
- `error_metrics.png`: Comparison of error metrics
- `error_distribution.png`: Distribution of prediction errors
"""
        
        summary_path = self.config.DOCS_DIR / "evaluation_summary.md"
        with open(summary_path, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Evaluation summary saved to {summary_path}")

    def run_full_evaluation(self, log_to_mlflow: bool = True) -> Dict[str, float]:
        """
        Run complete model evaluation pipeline.
        Optionally log metrics and plots to MLflow.
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Load model and data
            model, X_test, y_test = self.load_model_and_data()
            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test)
            # Create plots
            plots = self.create_evaluation_plots(save_plots=True)
            # Save report
            self.save_evaluation_report()
            # MLflow logging
            if log_to_mlflow:
                with mlflow.start_run(run_name="Model Evaluation"):
                    mlflow.log_metric("r2", metrics["r2"])
                    mlflow.log_metric("mae", metrics["mae"])
                    mlflow.log_metric("rmse", metrics["rmse"])
                    # Log plots as artifacts
                    plots_dir = self.config.DOCS_DIR / "evaluation_plots"
                    for plot_name in ["predicted_vs_actual", "residuals", "error_metrics", "error_distribution"]:
                        plot_path = plots_dir / f"{plot_name}.png"
                        if plot_path.exists():
                            mlflow.log_artifact(str(plot_path), artifact_path="evaluation_plots")
                    # Log report files
                    report_path = self.config.DOCS_DIR / "evaluation_report.json"
                    summary_path = self.config.DOCS_DIR / "evaluation_summary.md"
                    if report_path.exists():
                        mlflow.log_artifact(str(report_path), artifact_path="docs")
                    if summary_path.exists():
                        mlflow.log_artifact(str(summary_path), artifact_path="docs")
            logger.info("Full evaluation completed successfully")
            return metrics
        except Exception as e:
            logger.error(f"Error in full evaluation: {e}")
            raise

# Convenience function
def evaluate_best_model() -> Dict[str, float]:
    """Evaluate the best saved model."""
    evaluator = ModelEvaluator()
    return evaluator.run_full_evaluation()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the best saved model and log results.")
    parser.add_argument("--no-mlflow", action="store_true", help="Do not log results to MLflow.")
    args = parser.parse_args()
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation(log_to_mlflow=not args.no_mlflow)
