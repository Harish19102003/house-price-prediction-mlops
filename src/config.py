# src/config.py

import os
from pathlib import Path

class Config:
    """
    Configuration class containing all project settings and paths.
    """
    # Base paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    DOCS_DIR = PROJECT_ROOT / "docs"
    MLRUNS_DIR = PROJECT_ROOT / "mlruns"
    
    # Data files
    RAW_TRAIN_FILE = RAW_DATA_DIR / "train.csv"
    RAW_TEST_FILE = RAW_DATA_DIR / "test.csv"
    PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
    PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test_processed.csv"
    
    # Model files
    MODEL_ARTIFACT_PATH = MODELS_DIR / "best_model.pkl"
    SCALER_PATH = MODELS_DIR / "scaler.pkl"
    
    # ML Configuration
    TARGET_COLUMN = "SalePrice"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = "file://" + str(MLRUNS_DIR)
    EXPERIMENT_NAME = "house_price_prediction"
    
    # Model hyperparameters
    MODEL_CONFIGS = {
        'linear_regression': {
            'fit_intercept': True,
            'positive': False
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': RANDOM_STATE
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': RANDOM_STATE
        }
    }
    
    # Evaluation metrics
    METRICS = ['mae', 'mse', 'rmse', 'r2']
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.DOCS_DIR,
            cls.MLRUNS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
