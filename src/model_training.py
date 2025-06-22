# src/model_training.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# MLflow imports - optional
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Model tracking will be disabled.")

from .config import Config
from .data_ingestion import DataIngestion
from .utils import (
    logger, save_pickle, load_pickle, ModelFactory, 
    calculate_metrics, log_metrics, create_feature_importance_df
)

class ModelTrainer:
    """
    Model training class implementing Template Method pattern.
    Supports multiple algorithms via Strategy pattern.
    """
    
    def __init__(self):
        """Initialize ModelTrainer with configuration."""
        self.config = Config()
        self.data_ingestion = DataIngestion()
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.model_metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('-inf')
        
        # Create necessary directories
        self.config.create_directories()
        
        logger.info("ModelTrainer initialized")
    
    def load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load processed data and split into train/test sets.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            # Load processed data
            df = self.data_ingestion.load_processed_data()
            
            # Split features and target
            X, y = self.data_ingestion.split_features_target(df)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE,
                stratify=None  # For regression
            )
            
            logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
            logger.info(f"Target distribution - Train mean: {y_train.mean():.2f}, Test mean: {y_test.mean():.2f}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading and splitting data: {e}")
            raise
    
    def preprocess_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess features (scaling, encoding, etc.).
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of preprocessed (X_train, X_test)
        """
        try:
            # Get numerical columns for scaling
            numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
            
            if numerical_columns:
                # Fit scaler on training data
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()
                
                X_train_scaled[numerical_columns] = self.scaler.fit_transform(X_train[numerical_columns])
                X_test_scaled[numerical_columns] = self.scaler.transform(X_test[numerical_columns])
                
                # Save scaler
                save_pickle(self.scaler, self.config.SCALER_PATH)
                
                logger.info(f"Features scaled for {len(numerical_columns)} numerical columns")
                return X_train_scaled, X_test_scaled
            else:
                logger.warning("No numerical columns found for scaling")
                return X_train, X_test
                
        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            raise
    
    def train_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train a specific model using the Strategy pattern.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        try:
            # Get model configuration
            model_config = self.config.MODEL_CONFIGS.get(model_type, {})
            
            # Create model strategy
            model_strategy = ModelFactory.create_model(model_type, **model_config)
            
            # Train model
            logger.info(f"Training {model_strategy.get_name()} model...")
            model = model_strategy.train(X_train, y_train)
            
            # Store trained model and strategy
            self.trained_models[model_type] = {
                'model': model,
                'strategy': model_strategy,
                'config': model_config
            }
            
            logger.info(f"{model_strategy.get_name()} model trained successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            raise
    
    def evaluate_model(self, model_type: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model_type: Type of model to evaluate
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if model_type not in self.trained_models:
                raise ValueError(f"Model {model_type} has not been trained yet")
            
            model_info = self.trained_models[model_type]
            model = model_info['model']
            strategy = model_info['strategy']
            
            # Make predictions
            y_pred = strategy.predict(model, X_test)
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred)
            
            # Store metrics
            self.model_metrics[model_type] = metrics
            
            logger.info(f"Evaluation for {strategy.get_name()}:")
            log_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_type} model: {e}")
            raise
    
    def select_best_model(self, metric: str = 'r2') -> Tuple[str, Any]:
        """
        Select the best model based on a specific metric.
        
        Args:
            metric: Metric to use for selection (higher is better for r2, lower for others)
            
        Returns:
            Tuple of (best_model_name, best_model)
        """
        try:
            if not self.model_metrics:
                raise ValueError("No models have been evaluated yet")
            
            best_model_name = None
            best_score = float('-inf') if metric == 'r2' else float('inf')
            
            for model_name, metrics in self.model_metrics.items():
                score = metrics[metric]
                
                if metric == 'r2':
                    # Higher R² is better
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                else:
                    # Lower error is better
                    if score < best_score:
                        best_score = score
                        best_model_name = model_name
            
            self.best_model_name = best_model_name
            self.best_model = self.trained_models[best_model_name]['model']
            self.best_score = best_score
            
            logger.info(f"Best model selected: {best_model_name} with {metric.upper()}: {best_score:.4f}")
            
            return best_model_name, self.best_model
            
        except Exception as e:
            logger.error(f"Error selecting best model: {e}")
            raise
    
    def save_best_model(self) -> None:
        """Save the best model to disk."""
        try:
            if self.best_model is None:
                raise ValueError("No best model selected. Run select_best_model() first.")
            
            save_pickle(self.best_model, self.config.MODEL_ARTIFACT_PATH)
            
            # Save model metadata
            metadata = {
                'model_name': self.best_model_name,
                'model_type': type(self.best_model).__name__,
                'best_score': self.best_score,
                'metrics': self.model_metrics[self.best_model_name],
                'config': self.trained_models[self.best_model_name]['config']
            }
            
            metadata_path = self.config.MODELS_DIR / "model_metadata.pkl"
            save_pickle(metadata, metadata_path)
            
            logger.info(f"Best model ({self.best_model_name}) saved to {self.config.MODEL_ARTIFACT_PATH}")
            
        except Exception as e:
            logger.error(f"Error saving best model: {e}")
            raise
    
    def train_all_models(self) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate all configured models.
        
        Returns:
            Dictionary of all model metrics
        """
        try:
            # Load and split data
            X_train, X_test, y_train, y_test = self.load_and_split_data()
            
            # Preprocess features
            X_train_processed, X_test_processed = self.preprocess_features(X_train, X_test)
            
            # Train all configured models
            for model_type in self.config.MODEL_CONFIGS.keys():
                logger.info(f"Training {model_type} model...")
                self.train_model(model_type, X_train_processed, y_train)
                self.evaluate_model(model_type, X_test_processed, y_test)
            
            # Select and save best model
            self.select_best_model()
            self.save_best_model()
            
            return self.model_metrics
            
        except Exception as e:
            logger.error(f"Error training all models: {e}")
            raise

    def get_feature_importance(self, model_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get feature importance for a specific model or the best model.
        
        Args:
            model_type: Specific model type, uses best model if None
            
        Returns:
            DataFrame with feature importance
        """
        try:
            if model_type is None:
                if self.best_model_name is None:
                    raise ValueError("No best model selected and no model_type specified")
                model_type = self.best_model_name
            
            if model_type not in self.trained_models:
                raise ValueError(f"Model {model_type} not found in trained models")
            
            model = self.trained_models[model_type]['model']
            
            # Get feature names (assuming last preprocessing step)
            # This is a simplified approach - in production, you'd track feature names through preprocessing
            df = self.data_ingestion.load_processed_data()
            X, _ = self.data_ingestion.split_features_target(df)
            feature_names = X.columns.tolist()
            
            return create_feature_importance_df(model, feature_names)
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise

# Convenience functions
def train_all_models() -> Dict[str, Dict[str, float]]:
    """Train all models using default configuration."""
    trainer = ModelTrainer()
    return trainer.train_all_models()

def load_best_model() -> Any:
    """Load the best saved model."""
    config = Config()
    return load_pickle(config.MODEL_ARTIFACT_PATH)
