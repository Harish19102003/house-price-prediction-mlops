# src/model_strategies.py

import pandas as pd
import numpy as np
from typing import Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# XGBoost import - optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. XGBoost models will be disabled.")

from .utils import ModelStrategy, logger

class LinearRegressionStrategy(ModelStrategy):
    """Strategy for Linear Regression model."""
    
    def __init__(self, **kwargs):
        self.model_params = kwargs
        logger.info(f"Initialized LinearRegressionStrategy with params: {kwargs}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> LinearRegression:
        """
        Train Linear Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional training parameters
            
        Returns:
            Trained Linear Regression model
        """
        model = LinearRegression(**self.model_params)
        model.fit(X_train, y_train)
        logger.info("Linear Regression model trained successfully")
        return model
    
    def predict(self, model: LinearRegression, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using Linear Regression model.
        
        Args:
            model: Trained model
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        predictions = model.predict(X)
        logger.info(f"Made predictions for {len(X)} samples")
        return predictions
    
    def get_name(self) -> str:
        """Return model name."""
        return "Linear Regression"

class RandomForestStrategy(ModelStrategy):
    """Strategy for Random Forest model."""
    
    def __init__(self, **kwargs):
        self.model_params = kwargs
        logger.info(f"Initialized RandomForestStrategy with params: {kwargs}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> RandomForestRegressor:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional training parameters
            
        Returns:
            Trained Random Forest model
        """
        model = RandomForestRegressor(**self.model_params)
        model.fit(X_train, y_train)
        logger.info("Random Forest model trained successfully")
        return model
    
    def predict(self, model: RandomForestRegressor, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using Random Forest model.
        
        Args:
            model: Trained model
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        predictions = model.predict(X)
        logger.info(f"Made predictions for {len(X)} samples")
        return predictions
    
    def get_name(self) -> str:
        """Return model name."""
        return "Random Forest"

class XGBoostStrategy(ModelStrategy):
    """Strategy for XGBoost model."""
    
    def __init__(self, **kwargs):
        self.model_params = kwargs
        logger.info(f"Initialized XGBoostStrategy with params: {kwargs}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> xgb.XGBRegressor:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional training parameters
            
        Returns:
            Trained XGBoost model
        """
        model = xgb.XGBRegressor(**self.model_params)
        model.fit(X_train, y_train)
        logger.info("XGBoost model trained successfully")
        return model
    
    def predict(self, model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using XGBoost model.
        
        Args:
            model: Trained model
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        predictions = model.predict(X)
        logger.info(f"Made predictions for {len(X)} samples")
        return predictions
    
    def get_name(self) -> str:
        """Return model name."""
        return "XGBoost"
