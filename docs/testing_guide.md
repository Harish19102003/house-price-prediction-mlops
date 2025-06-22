# Testing Guide for House Price Prediction MLOps Pipeline

## 🧪 Overview

This guide covers comprehensive testing of the entire ML pipeline, including individual modules, CLI interfaces, Docker functionality, and end-to-end validation.

## 🚀 Quick Start Testing

### 1. Run Complete Pipeline Test
```bash
# Test everything locally
python test_pipeline.py

# Test with Makefile
make full-test
```

### 2. Run Individual Component Tests
```bash
# Test specific components
python test_pipeline.py --test data
python test_pipeline.py --test train
python test_pipeline.py --test eval
python test_pipeline.py --test deploy
python test_pipeline.py --test inference
python test_pipeline.py --test cli
python test_pipeline.py --test docker
python test_pipeline.py --test validation
python test_pipeline.py --test mlflow
```

## 📋 Test Components

### 1. Data Ingestion Tests
- **Raw data loading**: Validates CSV files exist and are readable
- **Data processing**: Ensures preprocessing pipeline works correctly
- **File saving**: Confirms processed data is saved to disk
- **Data validation**: Checks for required columns and data integrity

### 2. Model Training Tests
- **Multi-model training**: Tests LinearRegression, RandomForest, XGBoost
- **Model selection**: Validates best model selection logic
- **Artifact saving**: Ensures models, scalers, and metadata are saved
- **Performance metrics**: Validates training metrics are calculated

### 3. Model Evaluation Tests
- **Evaluation metrics**: Tests R², RMSE, MAE, MAPE calculations
- **Plot generation**: Validates evaluation plots are created
- **Report generation**: Ensures JSON and Markdown reports are saved
- **MLflow logging**: Tests metric logging to MLflow

### 4. Model Deployment Tests
- **MLflow registration**: Tests model registration to MLflow Model Registry
- **Metadata saving**: Validates deployment metadata is saved
- **URI generation**: Ensures proper model URI format

### 5. Inference Tests
- **Single prediction**: Tests individual house price predictions
- **Batch prediction**: Validates multiple predictions
- **Confidence intervals**: Tests prediction confidence functionality
- **Input validation**: Ensures proper input preprocessing

### 6. CLI Interface Tests
- **Main pipeline CLI**: Tests `main.py` with different steps
- **Evaluation CLI**: Tests `src/evaluation.py` execution
- **Inference CLI**: Tests `src/inference.py` with sample data
- **Error handling**: Validates proper error messages

### 7. Docker Functionality Tests
- **Image building**: Tests Docker image creation
- **Service startup**: Validates container startup
- **Health checks**: Tests service availability
- **Network connectivity**: Ensures services can communicate

### 8. Output Validation Tests
- **File existence**: Checks all required files are created
- **Directory structure**: Validates proper directory organization
- **JSON validation**: Ensures proper JSON format
- **Plot validation**: Confirms evaluation plots exist

### 9. MLflow Integration Tests
- **Experiment creation**: Tests MLflow experiment setup
- **Metric logging**: Validates metric recording
- **Artifact logging**: Tests file artifact storage
- **Run tracking**: Ensures proper run management

## 🛠️ Testing Commands

### Using Makefile
```bash
# Quick validation
make quick-validate

# Check pipeline status
make status

# Run development tests
make dev-test

# Run production tests
make prod-test

# Full system test
make full-test
```

### Using Main Pipeline
```bash
# Run complete pipeline
python main.py --step all

# Run without deployment
python main.py --step all --skip-deployment

# Run individual steps
python main.py --step data
python main.py --step train
python main.py --step eval
python main.py --step deploy
python main.py --step inference
python main.py --step validate
```

### Using Individual Modules
```bash
# Test evaluation
python src/evaluation.py

# Test inference
python src/inference.py --sample

# Test deployment
python src/deployment.py

# Test Streamlit
python run_streamlit.py
```

## 🐳 Docker Testing

### Build and Test Docker Services
```bash
# Build images
make docker-build

# Run services
make docker-run

# Test in Docker
make docker-test

# Check health
make health

# View logs
make logs

# Clean up
make docker-clean
```

### Development Environment Testing
```bash
# Start development environment
make docker-dev

# Access services:
# - Jupyter: http://localhost:8888
# - Streamlit: http://localhost:8502
# - MLflow: http://localhost:5001
```

## 📊 Expected Test Results

### Successful Test Output
```
🧪 TEST EXECUTION SUMMARY
============================================================
✅ Passed: 9
❌ Failed: 0
⚠️  Errors: 0
📊 Total: 9
🎯 Success Rate: 100.0%
============================================================

🎉 All tests passed!
```

### Required Files After Testing
```
data/
├── processed/
│   └── train_processed.csv

models/
├── best_model.pkl
├── scaler.pkl
└── model_metadata.pkl

docs/
├── evaluation_report.json
├── evaluation_summary.md
├── deployment_metadata.json
├── test_results.json
├── test_results.log
└── evaluation_plots/
    ├── predicted_vs_actual.png
    ├── residuals.png
    ├── error_metrics.png
    └── error_distribution.png

mlruns/
└── [MLflow tracking data]
```

## 🔍 Validation Checks

### File Validation
```bash
# Check required files exist
make status

# Quick validation
make quick-validate

# Detailed validation
python main.py --step validate
```

### Service Health Checks
```bash
# Check service health
make health

# Expected output:
# ✅ Streamlit: Healthy
# ✅ MLflow: Healthy
```

### Performance Validation
- **R² Score**: Should be > 0.85
- **RMSE**: Should be < $50,000
- **MAPE**: Should be < 15%
- **Training Time**: Should be < 5 minutes
- **Inference Time**: Should be < 1 second

## 🚨 Troubleshooting

### Common Test Failures

#### 1. Data Ingestion Failures
```bash
# Check data files exist
ls -la data/raw/

# Re-run data ingestion
python main.py --step data
```

#### 2. Model Training Failures
```bash
# Check dependencies
pip install -r requirements.txt

# Clear existing models
rm -rf models/*.pkl

# Re-run training
python main.py --step train
```

#### 3. Evaluation Failures
```bash
# Ensure model exists
ls -la models/best_model.pkl

# Re-run evaluation
python main.py --step eval
```

#### 4. Docker Failures
```bash
# Check Docker is running
docker --version
docker-compose --version

# Clean Docker environment
make docker-clean

# Rebuild and test
make docker-build
make docker-test
```

#### 5. MLflow Connection Issues
```bash
# Check MLflow is running
curl http://localhost:5000/health

# Restart MLflow service
docker-compose restart mlflow
```

### Debug Commands

#### Check Logs
```bash
# View all logs
make logs

# View specific service logs
make logs-streamlit
make logs-mlflow

# View test logs
cat docs/test_results.log
```

#### Check System Status
```bash
# Check pipeline status
make status

# Check Docker status
docker-compose ps

# Check disk space
df -h

# Check memory usage
free -h
```

#### Validate Configuration
```bash
# Check Python environment
python -c "import sys; print(sys.version)"

# Check dependencies
pip list | grep -E "(mlflow|streamlit|sklearn|xgboost)"

# Check file permissions
ls -la data/ models/ docs/
```

## 📈 Performance Testing

### Load Testing
```bash
# Test batch inference
python -c "
from src.inference import HousePricePredictor
import time

predictor = HousePricePredictor()
sample_data = {'OverallQual': 7, 'GrLivArea': 1500, 'TotalBsmtSF': 1000, 'GarageCars': 2, 'YearBuilt': 2000, 'FullBath': 2, 'KitchenQual': 'Gd', 'Neighborhood': 'NAmes'}

# Test 1000 predictions
start_time = time.time()
for _ in range(1000):
    predictor.predict(sample_data)
end_time = time.time()

print(f'1000 predictions in {end_time - start_time:.2f} seconds')
print(f'Average time per prediction: {(end_time - start_time)/1000*1000:.2f} ms')
"
```

### Memory Testing
```bash
# Monitor memory usage during training
python -c "
import psutil
import time
from src.model_training import train_all_models

process = psutil.Process()
print(f'Initial memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')

start_time = time.time()
train_all_models()
end_time = time.time()

print(f'Final memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
print(f'Training time: {end_time - start_time:.2f} seconds')
"
```

## 🔒 Security Testing

### Input Validation
```bash
# Test malicious inputs
python -c "
from src.inference import HousePricePredictor

predictor = HousePricePredictor()

# Test with invalid data types
try:
    predictor.predict({'OverallQual': 'invalid'})
except Exception as e:
    print(f'✅ Caught invalid input: {e}')

# Test with missing required fields
try:
    predictor.predict({'OverallQual': 7})
except Exception as e:
    print(f'✅ Caught missing fields: {e}')
"
```

### File Access Validation
```bash
# Check file permissions
ls -la models/
ls -la docs/
ls -la data/

# Ensure no world-writable files
find . -type f -perm -002 -ls
```

## 📋 Test Reports

### Generated Reports
After running tests, the following reports are generated:

1. **`docs/test_results.json`**: Detailed test results with timestamps
2. **`docs/test_results.log`**: Comprehensive test execution logs
3. **`docs/pipeline_report.json`**: Pipeline execution summary
4. **`docs/evaluation_report.json`**: Model evaluation metrics
5. **`docs/deployment_metadata.json`**: Deployment information

### Report Analysis
```bash
# View test results
cat docs/test_results.json | jq '.'

# Check success rate
cat docs/test_results.json | jq '. | to_entries | map(select(.value.status == "passed")) | length'

# View pipeline metrics
cat docs/evaluation_report.json | jq '.model_evaluation_summary.metrics'
```

## 🎯 Continuous Integration

### GitHub Actions Example
```yaml
name: Pipeline Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python test_pipeline.py
      - name: Upload test results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: docs/test_results.json
```

## 📞 Support

### Getting Help
1. **Check logs**: `cat docs/test_results.log`
2. **Validate setup**: `make status`
3. **Run quick test**: `make quick-validate`
4. **Check documentation**: `make docs`

### Common Issues
- **Port conflicts**: Change ports in `docker-compose.yml`
- **Memory issues**: Increase Docker memory limit
- **Permission errors**: Check file ownership and permissions
- **Network issues**: Verify Docker network configuration

### Reporting Bugs
When reporting test failures, include:
1. Test command executed
2. Full error output
3. System information (OS, Python version, Docker version)
4. Test results file: `docs/test_results.json`
5. Log file: `docs/test_results.log` 