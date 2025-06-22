# src/inference.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Union, Dict, List, Any, Optional
import json
from datetime import datetime

from .config import Config
from .utils import logger, load_pickle
from .preprocessing import preprocess_data

class HousePricePredictor:
    """
    Inference pipeline for house price prediction.
    Supports loading models from MLflow Model Registry or local artifacts.
    """
    
    def __init__(self, model_source: str = "local"):
        """
        Initialize the predictor.
        
        Args:
            model_source: "local" for local pickle file, "mlflow" for MLflow Model Registry
        """
        self.config = Config()
        self.model_source = model_source
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Set up MLflow if using MLflow models
        if model_source == "mlflow":
            mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        
        self._load_model()
        logger.info(f"HousePricePredictor initialized with {model_source} model")
    
    def _load_model(self):
        """Load the trained model and related artifacts."""
        try:
            if self.model_source == "local":
                # Load local model
                model_path = self.config.MODEL_ARTIFACT_PATH
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                self.model = load_pickle(model_path)
                
                # Load scaler
                scaler_path = self.config.SCALER_PATH
                if scaler_path.exists():
                    self.scaler = load_pickle(scaler_path)
                
                # Load feature columns from processed data
                processed_data_path = self.config.PROCESSED_TRAIN_FILE
                if processed_data_path.exists():
                    df = pd.read_csv(processed_data_path)
                    self.feature_columns = [col for col in df.columns if col != self.config.TARGET_COLUMN]
                
                logger.info("Local model and artifacts loaded successfully")
                
            elif self.model_source == "mlflow":
                # Load from MLflow Model Registry
                model_name = "house_price_predictor"
                model_version = "latest"
                
                model_uri = f"models:/{model_name}/{model_version}"
                self.model = mlflow.sklearn.load_model(model_uri)
                
                logger.info(f"MLflow model loaded: {model_uri}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input data to match training format.
        
        Args:
            data: Input data as dictionary or DataFrame
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        try:
            # Convert dict to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()
            
            # Apply the same preprocessing as training
            df_processed = preprocess_data(df, target_column=self.config.TARGET_COLUMN)
            
            # Ensure all expected features are present
            if self.feature_columns:
                missing_cols = set(self.feature_columns) - set(df_processed.columns)
                for col in missing_cols:
                    df_processed[col] = 0
                
                # Reorder columns to match training
                df_processed = df_processed[self.feature_columns]
            
            # Apply scaling if available
            if self.scaler is not None:
                numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
                if numerical_columns:
                    df_processed[numerical_columns] = self.scaler.transform(df_processed[numerical_columns])
            
            logger.info(f"Input preprocessed: {df_processed.shape}")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {e}")
            raise
    
    def predict(self, data: Union[Dict, pd.DataFrame, List[Dict]]) -> Union[float, List[float]]:
        """
        Make house price predictions.
        
        Args:
            data: Input data (single dict, DataFrame, or list of dicts)
            
        Returns:
            Predicted house price(s)
        """
        try:
            # Handle batch predictions
            if isinstance(data, list):
                df = pd.DataFrame(data)
                is_batch = True
            else:
                df = data if isinstance(data, pd.DataFrame) else pd.DataFrame([data])
                is_batch = len(df) > 1
            
            # Preprocess input
            df_processed = self.preprocess_input(df)
            
            # Make predictions
            predictions = self.model.predict(df_processed)
            
            # Convert to list for consistency
            if not is_batch:
                predictions = predictions[0]
            
            logger.info(f"Predictions made: {len(predictions) if is_batch else 1} sample(s)")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_with_confidence(self, data: Union[Dict, pd.DataFrame], 
                              confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Make predictions with confidence intervals (if model supports it).
        
        Args:
            data: Input data
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with prediction and confidence info
        """
        try:
            prediction = self.predict(data)
            
            # For now, return basic prediction info
            # In a real scenario, you might use ensemble methods or model-specific confidence
            result = {
                "prediction": prediction,
                "confidence_level": confidence_level,
                "timestamp": datetime.now().isoformat(),
                "model_source": self.model_source
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction with confidence: {e}")
            raise
    
    def save_example_predictions(self, sample_data: List[Dict]) -> None:
        """
        Save example predictions to docs/ for documentation.
        
        Args:
            sample_data: List of sample input data
        """
        try:
            predictions = self.predict(sample_data)
            
            examples = []
            for i, (data, pred) in enumerate(zip(sample_data, predictions)):
                examples.append({
                    "sample_id": i + 1,
                    "input_features": data,
                    "predicted_price": float(pred),
                    "predicted_price_formatted": f"${pred:,.2f}"
                })
            
            # Save to docs
            examples_path = self.config.DOCS_DIR / "example_predictions.json"
            with open(examples_path, 'w') as f:
                json.dump(examples, f, indent=2)
            
            logger.info(f"Example predictions saved to {examples_path}")
            
        except Exception as e:
            logger.error(f"Error saving example predictions: {e}")
            raise

def create_sample_data() -> List[Dict]:
    """Create sample house data for testing predictions."""
    return [
        {
            "MSSubClass": 60,
            "MSZoning": "RL",
            "LotFrontage": 65.0,
            "LotArea": 8450,
            "Street": "Pave",
            "Alley": "NA",
            "LotShape": "Reg",
            "LandContour": "Lvl",
            "Utilities": "AllPub",
            "LotConfig": "Inside",
            "LandSlope": "Gtl",
            "Neighborhood": "CollgCr",
            "Condition1": "Norm",
            "Condition2": "Norm",
            "BldgType": "1Fam",
            "HouseStyle": "2Story",
            "OverallQual": 7,
            "OverallCond": 5,
            "YearBuilt": 2003,
            "YearRemodAdd": 2003,
            "RoofStyle": "Gable",
            "RoofMatl": "CompShg",
            "Exterior1st": "VinylSd",
            "Exterior2nd": "VinylSd",
            "MasVnrType": "BrkFace",
            "MasVnrArea": 196.0,
            "ExterQual": "Gd",
            "ExterCond": "TA",
            "Foundation": "PConc",
            "BsmtQual": "Gd",
            "BsmtCond": "TA",
            "BsmtExposure": "No",
            "BsmtFinType1": "GLQ",
            "BsmtFinSF1": 706.0,
            "BsmtFinType2": "Unf",
            "BsmtFinSF2": 0.0,
            "BsmtUnfSF": 150.0,
            "TotalBsmtSF": 856.0,
            "Heating": "GasA",
            "HeatingQC": "Ex",
            "CentralAir": "Y",
            "Electrical": "SBrkr",
            "1stFlrSF": 856,
            "2ndFlrSF": 854,
            "LowQualFinSF": 0,
            "GrLivArea": 1710,
            "BsmtFullBath": 1,
            "BsmtHalfBath": 0,
            "FullBath": 2,
            "HalfBath": 1,
            "BedroomAbvGr": 3,
            "KitchenAbvGr": 1,
            "KitchenQual": "Gd",
            "TotRmsAbvGrd": 8,
            "Functional": "Typ",
            "Fireplaces": 0,
            "FireplaceQu": "NA",
            "GarageType": "Attchd",
            "GarageYrBlt": 2003.0,
            "GarageFinish": "RFn",
            "GarageCars": 2,
            "GarageArea": 548.0,
            "GarageQual": "TA",
            "GarageCond": "TA",
            "PavedDrive": "Y",
            "WoodDeckSF": 0,
            "OpenPorchSF": 61,
            "EnclosedPorch": 0,
            "3SsnPorch": 0,
            "ScreenPorch": 0,
            "PoolArea": 0,
            "PoolQC": "NA",
            "Fence": "NA",
            "MiscFeature": "NA",
            "MiscVal": 0,
            "MoSold": 2,
            "YrSold": 2008,
            "SaleType": "WD",
            "SaleCondition": "Normal"
        }
    ]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Make house price predictions")
    parser.add_argument("--model-source", choices=["local", "mlflow"], default="local",
                       help="Source of the model (local or mlflow)")
    parser.add_argument("--input-file", type=str, help="Path to CSV file with input data")
    parser.add_argument("--sample", action="store_true", help="Run prediction on sample data")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = HousePricePredictor(model_source=args.model_source)
    
    if args.sample:
        # Use sample data
        sample_data = create_sample_data()
        predictions = predictor.predict(sample_data)
        
        print("\n=== Sample Predictions ===")
        for i, (data, pred) in enumerate(zip(sample_data, predictions)):
            print(f"Sample {i+1}: ${pred:,.2f}")
        
        # Save examples
        predictor.save_example_predictions(sample_data)
        
    elif args.input_file:
        # Load from file
        df = pd.read_csv(args.input_file)
        predictions = predictor.predict(df)
        
        print(f"\n=== Predictions for {len(predictions)} samples ===")
        for i, pred in enumerate(predictions):
            print(f"Sample {i+1}: ${pred:,.2f}")
    
    else:
        print("Please provide --input-file or use --sample for example predictions") 