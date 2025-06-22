#!/usr/bin/env python3
"""
Main orchestration script for House Price Prediction MLOps Pipeline.
Runs the complete pipeline: data → preprocessing → training → evaluation → deployment → inference
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import json
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.data_ingestion import DataIngestion
from src.preprocessing import preprocess_data
from src.model_training import train_all_models, load_best_model
from src.evaluation import evaluate_best_model
from src.deployment import ModelDeployer
from src.inference import HousePricePredictor
from src.utils import logger, save_pickle, load_pickle

class MLPipeline:
    """
    Complete ML pipeline orchestrator for house price prediction.
    """
    
    def __init__(self):
        """Initialize the pipeline with configuration."""
        self.config = Config()
        self.config.create_directories()
        self.pipeline_results = {}
        self.start_time = time.time()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.DOCS_DIR / 'pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info("🚀 ML Pipeline initialized")
    
    def run_data_ingestion(self) -> bool:
        """Run data ingestion and validation."""
        try:
            logger.info("📊 Step 1: Data Ingestion")
            
            data_ingestion = DataIngestion()
            
            # Load raw data
            train_df = data_ingestion.load_raw_data()
            logger.info(f"✅ Raw training data loaded: {train_df.shape}")
            
            # Process data
            processed_df = data_ingestion.process_data()
            logger.info(f"✅ Processed data saved: {processed_df.shape}")
            
            # Validate data
            if processed_df.empty:
                raise ValueError("Processed data is empty")
            
            if self.config.TARGET_COLUMN not in processed_df.columns:
                raise ValueError(f"Target column '{self.config.TARGET_COLUMN}' not found")
            
            self.pipeline_results['data_ingestion'] = {
                'status': 'success',
                'raw_shape': train_df.shape,
                'processed_shape': processed_df.shape,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ Data ingestion completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data ingestion failed: {e}")
            self.pipeline_results['data_ingestion'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_model_training(self) -> bool:
        """Run model training pipeline."""
        try:
            logger.info("🤖 Step 2: Model Training")
            
            # Train all models
            model_metrics = train_all_models()
            logger.info(f"✅ All models trained successfully")
            
            # Load best model for validation
            best_model = load_best_model()
            if best_model is None:
                raise ValueError("Best model not found after training")
            
            # Load model metadata
            metadata_path = self.config.MODELS_DIR / "model_metadata.pkl"
            if metadata_path.exists():
                metadata = load_pickle(metadata_path)
                logger.info(f"✅ Best model: {metadata.get('model_name', 'Unknown')}")
                logger.info(f"✅ Best R² score: {metadata.get('best_score', 0):.4f}")
            
            self.pipeline_results['model_training'] = {
                'status': 'success',
                'model_metrics': model_metrics,
                'best_model_metadata': metadata if metadata_path.exists() else None,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            self.pipeline_results['model_training'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_evaluation(self) -> bool:
        """Run model evaluation."""
        try:
            logger.info("📈 Step 3: Model Evaluation")
            
            # Run evaluation
            metrics = evaluate_best_model()
            
            # Validate metrics
            required_metrics = ['r2', 'rmse', 'mae', 'mape']
            for metric in required_metrics:
                if metric not in metrics:
                    raise ValueError(f"Required metric '{metric}' not found in evaluation results")
            
            logger.info(f"✅ Evaluation completed - R²: {metrics['r2']:.4f}, RMSE: ${metrics['rmse']:,.0f}")
            
            self.pipeline_results['evaluation'] = {
                'status': 'success',
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ Model evaluation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            self.pipeline_results['evaluation'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_deployment(self) -> bool:
        """Run model deployment to MLflow."""
        try:
            logger.info("🚀 Step 4: Model Deployment")
            
            deployer = ModelDeployer()
            model_uri = deployer.register_model()
            
            logger.info(f"✅ Model deployed to MLflow: {model_uri}")
            
            self.pipeline_results['deployment'] = {
                'status': 'success',
                'model_uri': model_uri,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ Model deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model deployment failed: {e}")
            self.pipeline_results['deployment'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_inference_test(self) -> bool:
        """Test inference with sample data."""
        try:
            logger.info("🔮 Step 5: Inference Testing")
            
            # Initialize predictor
            predictor = HousePricePredictor(model_source="local")
            
            # Create sample data
            sample_data = {
                'OverallQual': 7,
                'GrLivArea': 1500,
                'TotalBsmtSF': 1000,
                'GarageCars': 2,
                'YearBuilt': 2000,
                'FullBath': 2,
                'KitchenQual': 'Gd',
                'Neighborhood': 'NAmes'
            }
            
            # Make prediction
            prediction = predictor.predict(sample_data)
            
            # Validate prediction
            if not isinstance(prediction, (int, float)) or prediction <= 0:
                raise ValueError(f"Invalid prediction result: {prediction}")
            
            logger.info(f"✅ Sample prediction: ${prediction:,.2f}")
            
            # Test batch prediction
            batch_data = [sample_data, sample_data]
            batch_predictions = predictor.predict(batch_data)
            
            if len(batch_predictions) != 2:
                raise ValueError(f"Batch prediction returned {len(batch_predictions)} results, expected 2")
            
            logger.info(f"✅ Batch prediction test passed")
            
            self.pipeline_results['inference'] = {
                'status': 'success',
                'sample_prediction': float(prediction),
                'batch_prediction_count': len(batch_predictions),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ Inference testing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Inference testing failed: {e}")
            self.pipeline_results['inference'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def validate_outputs(self) -> bool:
        """Validate all expected outputs exist."""
        try:
            logger.info("🔍 Step 6: Output Validation")
            
            required_files = [
                self.config.PROCESSED_TRAIN_FILE,
                self.config.MODEL_ARTIFACT_PATH,
                self.config.SCALER_PATH,
                self.config.MODELS_DIR / "model_metadata.pkl",
                self.config.DOCS_DIR / "evaluation_report.json",
                self.config.DOCS_DIR / "evaluation_summary.md"
            ]
            
            required_dirs = [
                self.config.DOCS_DIR / "evaluation_plots",
                self.config.MLRUNS_DIR
            ]
            
            # Check files
            for file_path in required_files:
                if not file_path.exists():
                    raise FileNotFoundError(f"Required file not found: {file_path}")
                logger.info(f"✅ Found: {file_path}")
            
            # Check directories
            for dir_path in required_dirs:
                if not dir_path.exists():
                    raise FileNotFoundError(f"Required directory not found: {dir_path}")
                logger.info(f"✅ Found: {dir_path}")
            
            # Check evaluation plots
            plots_dir = self.config.DOCS_DIR / "evaluation_plots"
            expected_plots = ['predicted_vs_actual.png', 'residuals.png', 'error_metrics.png', 'error_distribution.png']
            for plot in expected_plots:
                plot_path = plots_dir / plot
                if not plot_path.exists():
                    raise FileNotFoundError(f"Expected plot not found: {plot_path}")
                logger.info(f"✅ Found plot: {plot}")
            
            self.pipeline_results['validation'] = {
                'status': 'success',
                'files_checked': len(required_files),
                'directories_checked': len(required_dirs),
                'plots_checked': len(expected_plots),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ Output validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Output validation failed: {e}")
            self.pipeline_results['validation'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def save_pipeline_report(self):
        """Save comprehensive pipeline report."""
        try:
            end_time = time.time()
            duration = end_time - self.start_time
            
            report = {
                'pipeline_summary': {
                    'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                    'end_time': datetime.fromtimestamp(end_time).isoformat(),
                    'duration_seconds': duration,
                    'duration_minutes': duration / 60,
                    'total_steps': len(self.pipeline_results),
                    'successful_steps': sum(1 for step in self.pipeline_results.values() if step.get('status') == 'success'),
                    'failed_steps': sum(1 for step in self.pipeline_results.values() if step.get('status') == 'failed')
                },
                'step_results': self.pipeline_results,
                'system_info': {
                    'python_version': sys.version,
                    'config': {
                        'project_root': str(self.config.PROJECT_ROOT),
                        'data_dir': str(self.config.DATA_DIR),
                        'models_dir': str(self.config.MODELS_DIR),
                        'docs_dir': str(self.config.DOCS_DIR)
                    }
                }
            }
            
            # Save report
            report_path = self.config.DOCS_DIR / "pipeline_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📋 Pipeline report saved to: {report_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("🎉 PIPELINE EXECUTION SUMMARY")
            print("="*60)
            print(f"⏱️  Total Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print(f"✅ Successful Steps: {report['pipeline_summary']['successful_steps']}")
            print(f"❌ Failed Steps: {report['pipeline_summary']['failed_steps']}")
            print(f"📊 Report Location: {report_path}")
            print("="*60)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to save pipeline report: {e}")
            return None
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete ML pipeline."""
        logger.info("🚀 Starting complete ML pipeline execution")
        
        steps = [
            ("Data Ingestion", self.run_data_ingestion),
            ("Model Training", self.run_model_training),
            ("Model Evaluation", self.run_evaluation),
            ("Model Deployment", self.run_deployment),
            ("Inference Testing", self.run_inference_test),
            ("Output Validation", self.validate_outputs)
        ]
        
        all_successful = True
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"🔄 Running: {step_name}")
            logger.info(f"{'='*50}")
            
            success = step_func()
            if not success:
                all_successful = False
                logger.error(f"❌ Pipeline failed at step: {step_name}")
                break
        
        # Save final report
        self.save_pipeline_report()
        
        if all_successful:
            logger.info("🎉 Complete pipeline executed successfully!")
        else:
            logger.error("❌ Pipeline execution failed")
        
        return all_successful

def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="House Price Prediction ML Pipeline")
    parser.add_argument("--step", choices=["data", "train", "eval", "deploy", "inference", "validate", "all"], 
                       default="all", help="Pipeline step to run")
    parser.add_argument("--skip-deployment", action="store_true", 
                       help="Skip MLflow deployment step")
    
    args = parser.parse_args()
    
    pipeline = MLPipeline()
    
    if args.step == "all":
        if args.skip_deployment:
            # Run pipeline without deployment
            pipeline.run_data_ingestion()
            pipeline.run_model_training()
            pipeline.run_evaluation()
            pipeline.run_inference_test()
            pipeline.validate_outputs()
            pipeline.save_pipeline_report()
        else:
            pipeline.run_complete_pipeline()
    elif args.step == "data":
        pipeline.run_data_ingestion()
    elif args.step == "train":
        pipeline.run_model_training()
    elif args.step == "eval":
        pipeline.run_evaluation()
    elif args.step == "deploy":
        pipeline.run_deployment()
    elif args.step == "inference":
        pipeline.run_inference_test()
    elif args.step == "validate":
        pipeline.validate_outputs()

if __name__ == "__main__":
    main() 