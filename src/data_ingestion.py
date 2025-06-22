# src/data_ingestion.py

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from .config import Config
from .utils import logger, validate_data

class DataIngestion:
    """
    Class responsible for data ingestion and basic data operations.
    Implements Template Method pattern for consistent data loading workflow.
    """
    
    def __init__(self):
        """Initialize DataIngestion with configuration."""
        self.config = Config()
        logger.info("DataIngestion initialized")
    
    def load_raw_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load raw training data from CSV file.
        
        Args:
            file_path: Optional custom file path. Uses config default if None.
            
        Returns:
            DataFrame containing raw data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
        """
        if file_path is None:
            file_path = self.config.RAW_TRAIN_FILE
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded raw data with shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Basic validation
            validate_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def load_processed_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load processed training data from CSV file.
        
        Args:
            file_path: Optional custom file path. Uses config default if None.
            
        Returns:
            DataFrame containing processed data
        """
        if file_path is None:
            file_path = self.config.PROCESSED_TRAIN_FILE
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded processed data with shape: {df.shape}")
            
            # Validate target column exists
            if self.config.TARGET_COLUMN not in df.columns:
                raise ValueError(f"Target column '{self.config.TARGET_COLUMN}' not found in processed data")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading processed data from {file_path}: {e}")
            raise
    
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features and target variable.
        
        Args:
            df: DataFrame containing both features and target
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            # Separate features and target
            X = df.drop(columns=[self.config.TARGET_COLUMN])
            y = df[self.config.TARGET_COLUMN]
            
            logger.info(f"Split data into features: {X.shape} and target: {y.shape}")
            logger.info(f"Feature columns: {list(X.columns)}")
            
            # Validate that we have both features and target
            if X.empty:
                raise ValueError("No features found after splitting")
            if y.empty:
                raise ValueError("No target values found after splitting")
            
            return X, y
            
        except KeyError:
            logger.error(f"Target column '{self.config.TARGET_COLUMN}' not found in DataFrame")
            raise
        except Exception as e:
            logger.error(f"Error splitting features and target: {e}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing dataset information
        """
        try:
            info = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'numerical_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
            }
            
            # Add target column statistics if present
            if self.config.TARGET_COLUMN in df.columns:
                target_col = df[self.config.TARGET_COLUMN]
                info['target_stats'] = {
                    'mean': target_col.mean(),
                    'median': target_col.median(),
                    'std': target_col.std(),
                    'min': target_col.min(),
                    'max': target_col.max(),
                    'skewness': target_col.skew(),
                    'kurtosis': target_col.kurtosis()
                }
            
            logger.info("Generated comprehensive data information")
            return info
            
        except Exception as e:
            logger.error(f"Error generating data information: {e}")
            raise
    
    def save_data(self, df: pd.DataFrame, file_path: Path) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            file_path: Path where to save the file
        """
        try:
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save DataFrame
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved to {file_path} with shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise

# Convenience functions for direct usage
def load_raw_data() -> pd.DataFrame:
    """Load raw training data using default configuration."""
    ingestion = DataIngestion()
    return ingestion.load_raw_data()

def load_processed_data() -> pd.DataFrame:
    """Load processed training data using default configuration."""
    ingestion = DataIngestion()
    return ingestion.load_processed_data()

def get_features_and_target() -> Tuple[pd.DataFrame, pd.Series]:
    """Load processed data and split into features and target."""
    ingestion = DataIngestion()
    df = ingestion.load_processed_data()
    return ingestion.split_features_target(df)
