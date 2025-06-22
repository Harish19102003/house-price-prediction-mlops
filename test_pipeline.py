#!/usr/bin/env python3
"""
Comprehensive test script for House Price Prediction MLOps Pipeline.
Tests all modules, CLI interfaces, Docker functionality, and validates outputs.
"""

import sys
import subprocess
import time
import json
import requests
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__parent / "src")))

from src.config import Config
from src.data_ingestion import DataIngestion
from src.model_training import train_all_models, load_best_model
from src.evaluation import evaluate_best_model
from src.deployment import ModelDeployer
from src.inference import HousePricePredictor

class PipelineTester:
    """Comprehensive pipeline testing class."""
    
    def __init__(self):
        self.config = Config()
        self.test_results = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Setup test logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.DOCS_DIR / 'test_results.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def test_data_ingestion(self) -> bool:
        """Test data ingestion module."""
        try:
            self.logger.info("🧪 Testing data ingestion...")
            
            data_ingestion = DataIngestion()
            
            # Test raw data loading
            train_df = data_ingestion.load_raw_data()
            assert not train_df.empty, "Raw training data is empty"
            assert self.config.TARGET_COLUMN in train_df.columns, f"Target column {self.config.TARGET_COLUMN} not found"
            
            # Test data processing
            processed_df = data_ingestion.process_data()
            assert not processed_df.empty, "Processed data is empty"
            assert self.config.TARGET_COLUMN in processed_df.columns, f"Target column {self.config.TARGET_COLUMN} not found in processed data"
            
            # Test file saving
            assert self.config.PROCESSED_TRAIN_FILE.exists(), "Processed training file not saved"
            
            self.logger.info(f"✅ Data ingestion test passed - Raw: {train_df.shape}, Processed: {processed_df.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data ingestion test failed: {e}")
            return False
    
    def test_model_training(self) -> bool:
        """Test model training module."""
        try:
            self.logger.info("🧪 Testing model training...")
            
            # Train models
            model_metrics = train_all_models()
            
            # Validate metrics
            assert isinstance(model_metrics, dict), "Model metrics should be a dictionary"
            assert len(model_metrics) > 0, "No model metrics returned"
            
            # Check for best model
            best_model = load_best_model()
            assert best_model is not None, "Best model not found"
            
            # Check model files
            assert self.config.MODEL_ARTIFACT_PATH.exists(), "Model artifact not saved"
            assert self.config.SCALER_PATH.exists(), "Scaler not saved"
            
            # Check metadata
            metadata_path = self.config.MODELS_DIR / "model_metadata.pkl"
            assert metadata_path.exists(), "Model metadata not saved"
            
            self.logger.info(f"✅ Model training test passed - {len(model_metrics)} models trained")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model training test failed: {e}")
            return False
    
    def test_evaluation(self) -> bool:
        """Test model evaluation module."""
        try:
            self.logger.info("🧪 Testing model evaluation...")
            
            # Run evaluation
            metrics = evaluate_best_model()
            
            # Validate metrics
            required_metrics = ['r2', 'rmse', 'mae', 'mape']
            for metric in required_metrics:
                assert metric in metrics, f"Required metric {metric} not found"
                assert isinstance(metrics[metric], (int, float)), f"Metric {metric} should be numeric"
            
            # Check evaluation files
            assert (self.config.DOCS_DIR / "evaluation_report.json").exists(), "Evaluation report not saved"
            assert (self.config.DOCS_DIR / "evaluation_summary.md").exists(), "Evaluation summary not saved"
            
            # Check evaluation plots
            plots_dir = self.config.DOCS_DIR / "evaluation_plots"
            expected_plots = ['predicted_vs_actual.png', 'residuals.png', 'error_metrics.png', 'error_distribution.png']
            for plot in expected_plots:
                assert (plots_dir / plot).exists(), f"Evaluation plot {plot} not found"
            
            self.logger.info(f"✅ Model evaluation test passed - R²: {metrics['r2']:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model evaluation test failed: {e}")
            return False
    
    def test_deployment(self) -> bool:
        """Test model deployment module."""
        try:
            self.logger.info("🧪 Testing model deployment...")
            
            deployer = ModelDeployer()
            model_uri = deployer.register_model()
            
            # Validate model URI
            assert model_uri is not None, "Model URI not returned"
            assert "runs:/" in model_uri, "Invalid model URI format"
            
            # Check deployment metadata
            deployment_metadata_path = self.config.DOCS_DIR / "deployment_metadata.json"
            assert deployment_metadata_path.exists(), "Deployment metadata not saved"
            
            self.logger.info(f"✅ Model deployment test passed - URI: {model_uri}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model deployment test failed: {e}")
            return False
    
    def test_inference(self) -> bool:
        """Test inference module."""
        try:
            self.logger.info("🧪 Testing inference...")
            
            predictor = HousePricePredictor(model_source="local")
            
            # Test single prediction
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
            
            prediction = predictor.predict(sample_data)
            assert isinstance(prediction, (int, float)), "Prediction should be numeric"
            assert prediction > 0, "Prediction should be positive"
            
            # Test batch prediction
            batch_data = [sample_data, sample_data]
            batch_predictions = predictor.predict(batch_data)
            assert len(batch_predictions) == 2, "Batch prediction should return 2 results"
            
            # Test confidence prediction
            confidence_result = predictor.predict_with_confidence(sample_data)
            assert 'prediction' in confidence_result, "Confidence result should contain prediction"
            assert 'confidence_level' in confidence_result, "Confidence result should contain confidence level"
            
            self.logger.info(f"✅ Inference test passed - Sample prediction: ${prediction:,.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Inference test failed: {e}")
            return False
    
    def test_cli_interfaces(self) -> bool:
        """Test all CLI interfaces."""
        try:
            self.logger.info("🧪 Testing CLI interfaces...")
            
            # Test main.py CLI
            cli_tests = [
                ["python", "main.py", "--step", "data"],
                ["python", "main.py", "--step", "train"],
                ["python", "main.py", "--step", "eval"],
                ["python", "main.py", "--step", "inference"],
                ["python", "main.py", "--step", "validate"],
            ]
            
            for cmd in cli_tests:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    self.logger.error(f"CLI test failed for {' '.join(cmd)}: {result.stderr}")
                    return False
            
            # Test evaluation CLI
            result = subprocess.run(["python", "src/evaluation.py"], capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                self.logger.error(f"Evaluation CLI test failed: {result.stderr}")
                return False
            
            # Test inference CLI
            result = subprocess.run(["python", "src/inference.py", "--sample"], capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                self.logger.error(f"Inference CLI test failed: {result.stderr}")
                return False
            
            self.logger.info("✅ CLI interfaces test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ CLI interfaces test failed: {e}")
            return False
    
    def test_docker_functionality(self) -> bool:
        """Test Docker functionality."""
        try:
            self.logger.info("🧪 Testing Docker functionality...")
            
            # Test Docker build
            result = subprocess.run(["docker-compose", "build"], capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                self.logger.error(f"Docker build failed: {result.stderr}")
                return False
            
            # Test Docker services startup
            result = subprocess.run(["docker-compose", "up", "-d"], capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                self.logger.error(f"Docker services startup failed: {result.stderr}")
                return False
            
            # Wait for services to be ready
            time.sleep(30)
            
            # Test service health
            try:
                # Test Streamlit
                response = requests.get("http://localhost:8501", timeout=10)
                if response.status_code != 200:
                    self.logger.error(f"Streamlit health check failed: {response.status_code}")
                    return False
                
                # Test MLflow
                response = requests.get("http://localhost:5000/health", timeout=10)
                if response.status_code != 200:
                    self.logger.error(f"MLflow health check failed: {response.status_code}")
                    return False
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Service health check failed: {e}")
                return False
            
            # Clean up Docker services
            subprocess.run(["docker-compose", "down"], capture_output=True, text=True)
            
            self.logger.info("✅ Docker functionality test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Docker functionality test failed: {e}")
            # Clean up on failure
            subprocess.run(["docker-compose", "down"], capture_output=True, text=True)
            return False
    
    def test_output_validation(self) -> bool:
        """Test output validation."""
        try:
            self.logger.info("🧪 Testing output validation...")
            
            # Check required files
            required_files = [
                self.config.PROCESSED_TRAIN_FILE,
                self.config.MODEL_ARTIFACT_PATH,
                self.config.SCALER_PATH,
                self.config.MODELS_DIR / "model_metadata.pkl",
                self.config.DOCS_DIR / "evaluation_report.json",
                self.config.DOCS_DIR / "evaluation_summary.md",
                self.config.DOCS_DIR / "deployment_metadata.json"
            ]
            
            for file_path in required_files:
                if not file_path.exists():
                    self.logger.error(f"Required file not found: {file_path}")
                    return False
            
            # Check required directories
            required_dirs = [
                self.config.DOCS_DIR / "evaluation_plots",
                self.config.MLRUNS_DIR
            ]
            
            for dir_path in required_dirs:
                if not dir_path.exists():
                    self.logger.error(f"Required directory not found: {dir_path}")
                    return False
            
            # Check evaluation plots
            plots_dir = self.config.DOCS_DIR / "evaluation_plots"
            expected_plots = ['predicted_vs_actual.png', 'residuals.png', 'error_metrics.png', 'error_distribution.png']
            for plot in expected_plots:
                if not (plots_dir / plot).exists():
                    self.logger.error(f"Expected plot not found: {plot}")
                    return False
            
            # Validate JSON files
            try:
                with open(self.config.DOCS_DIR / "evaluation_report.json", 'r') as f:
                    eval_report = json.load(f)
                assert 'model_evaluation_summary' in eval_report, "Invalid evaluation report format"
                
                with open(self.config.DOCS_DIR / "deployment_metadata.json", 'r') as f:
                    deploy_metadata = json.load(f)
                assert 'model_uri' in deploy_metadata, "Invalid deployment metadata format"
                
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"JSON validation failed: {e}")
                return False
            
            self.logger.info("✅ Output validation test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Output validation test failed: {e}")
            return False
    
    def test_mlflow_integration(self) -> bool:
        """Test MLflow integration."""
        try:
            self.logger.info("🧪 Testing MLflow integration...")
            
            import mlflow
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
            
            # Test experiment creation
            experiment = mlflow.get_experiment_by_name(self.config.EXPERIMENT_NAME)
            if experiment is None:
                experiment = mlflow.create_experiment(self.config.EXPERIMENT_NAME)
            
            assert experiment is not None, "Failed to create/get MLflow experiment"
            
            # Test run creation
            with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
                mlflow.log_metric("test_metric", 1.0)
                mlflow.log_param("test_param", "test_value")
                
                # Test artifact logging
                test_artifact_path = self.config.DOCS_DIR / "test_artifact.txt"
                with open(test_artifact_path, 'w') as f:
                    f.write("test artifact")
                
                mlflow.log_artifact(str(test_artifact_path))
                
                run_id = run.info.run_id
                assert run_id is not None, "Run ID not generated"
            
            # Clean up test artifact
            test_artifact_path.unlink(missing_ok=True)
            
            self.logger.info("✅ MLflow integration test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ MLflow integration test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests."""
        self.logger.info("🚀 Starting comprehensive pipeline testing...")
        
        tests = [
            ("Data Ingestion", self.test_data_ingestion),
            ("Model Training", self.test_model_training),
            ("Model Evaluation", self.test_evaluation),
            ("Model Deployment", self.test_deployment),
            ("Inference", self.test_inference),
            ("CLI Interfaces", self.test_cli_interfaces),
            ("Docker Functionality", self.test_docker_functionality),
            ("Output Validation", self.test_output_validation),
            ("MLflow Integration", self.test_mlflow_integration)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"🧪 Running: {test_name}")
            self.logger.info(f"{'='*50}")
            
            try:
                success = test_func()
                self.test_results[test_name] = {
                    'status': 'passed' if success else 'failed',
                    'timestamp': time.time()
                }
                
                if not success:
                    all_passed = False
                    self.logger.error(f"❌ Test failed: {test_name}")
                else:
                    self.logger.info(f"✅ Test passed: {test_name}")
                    
            except Exception as e:
                self.logger.error(f"❌ Test error in {test_name}: {e}")
                self.test_results[test_name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                }
                all_passed = False
        
        # Save test results
        self.save_test_results()
        
        # Print summary
        self.print_test_summary()
        
        return all_passed
    
    def save_test_results(self):
        """Save test results to file."""
        try:
            results_path = self.config.DOCS_DIR / "test_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            
            self.logger.info(f"📋 Test results saved to: {results_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save test results: {e}")
    
    def print_test_summary(self):
        """Print test summary."""
        passed = sum(1 for result in self.test_results.values() if result['status'] == 'passed')
        failed = sum(1 for result in self.test_results.values() if result['status'] == 'failed')
        errors = sum(1 for result in self.test_results.values() if result['status'] == 'error')
        total = len(self.test_results)
        
        print("\n" + "="*60)
        print("🧪 TEST EXECUTION SUMMARY")
        print("="*60)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Errors: {errors}")
        print(f"📊 Total: {total}")
        print(f"🎯 Success Rate: {(passed/total)*100:.1f}%")
        print("="*60)
        
        if failed > 0 or errors > 0:
            print("\n❌ Failed/Error Tests:")
            for test_name, result in self.test_results.items():
                if result['status'] in ['failed', 'error']:
                    print(f"  - {test_name}: {result.get('error', 'Test failed')}")
        
        print(f"\n📋 Detailed results: {self.config.DOCS_DIR / 'test_results.json'}")
        print(f"📋 Test logs: {self.config.DOCS_DIR / 'test_results.log'}")

def main():
    """Main test execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Testing Suite")
    parser.add_argument("--test", choices=["all", "data", "train", "eval", "deploy", "inference", "cli", "docker", "validation", "mlflow"], 
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    tester = PipelineTester()
    
    if args.test == "all":
        success = tester.run_all_tests()
    elif args.test == "data":
        success = tester.test_data_ingestion()
    elif args.test == "train":
        success = tester.test_model_training()
    elif args.test == "eval":
        success = tester.test_evaluation()
    elif args.test == "deploy":
        success = tester.test_deployment()
    elif args.test == "inference":
        success = tester.test_inference()
    elif args.test == "cli":
        success = tester.test_cli_interfaces()
    elif args.test == "docker":
        success = tester.test_docker_functionality()
    elif args.test == "validation":
        success = tester.test_output_validation()
    elif args.test == "mlflow":
        success = tester.test_mlflow_integration()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 