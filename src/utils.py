# src/utils.py

import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
from abc import ABC, abstractmethod

# Logging setup
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Abstract Base Classes for Strategy Pattern
class ModelStrategy(ABC):
    """Abstract base class for model training strategies."""
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Any:
        """Train the model with given data."""
        pass
    
    @abstractmethod
    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the model strategy."""
        pass

# Model Factory
class ModelFactory:
    """Factory class to create different model strategies."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> ModelStrategy:
        """
        Create a model strategy based on the specified type.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional parameters for model configuration
            
        Returns:
            Model strategy instance
        """
        from .model_strategies import (
            LinearRegressionStrategy,
            RandomForestStrategy,
            XGBoostStrategy
        )
        
        strategies = {
            'linear_regression': LinearRegressionStrategy,
            'random_forest': RandomForestStrategy,
            'xgboost': XGBoostStrategy
        }
        
        if model_type not in strategies:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(strategies.keys())}")
        
        return strategies[model_type](**kwargs)

# Utility Functions
def save_pickle(obj: Any, filepath: Path) -> None:
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save the file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Object saved to {filepath}")

def load_pickle(filepath: Path) -> Any:
    """
    Load an object from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Loaded object
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Object loaded from {filepath}")
    return obj

def validate_data(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for all NaN columns
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        logger.warning(f"Columns with all NaN values: {all_nan_cols}")
    
    logger.info("Data validation passed")
    return True

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
    
    return metrics

def log_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Log metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for log messages
    """
    prefix = f"{prefix} " if prefix else ""
    for metric, value in metrics.items():
        logger.info(f"{prefix}{metric.upper()}: {value:.4f}")

def create_feature_importance_df(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Create DataFrame with feature importance if available.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        elif hasattr(model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': model.coef_
            })
            return importance_df
        else:
            logger.warning("Model doesn't have feature importance or coefficients")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error creating feature importance DataFrame: {e}")
        return pd.DataFrame()

def format_price(price: float) -> str:
    """
    Format price for display.
    
    Args:
        price: Price value
        
    Returns:
        Formatted price string
    """
    return f"${price:,.2f}"
