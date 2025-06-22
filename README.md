# 🏠 House Price Prediction - End-to-End ML Pipeline

A production-ready machine learning pipeline for predicting house prices using MLOps best practices and clean software design.

## 🚀 Features
- EDA-driven feature engineering with comprehensive analysis
- Modular architecture with design patterns (Factory, Strategy, Template)
- MLflow for experiment tracking and model registry
- CI/CD-ready structure with comprehensive testing
- Streamlit UI with interactive visualizations
- Dockerized for deployment with multi-stage builds
- Comprehensive testing suite with validation
- AWS EC2 deployment support with CloudFormation
- Production-ready MLOps practices
- [Project Roadmap](./roadmap.md) – Complete implementation status

## 🔧 Setup

### Local Development
```bash
git clone https://github.com/Harish19102003/house-price-prediction-mlops.git
cd house-price-prediction-mlops
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Docker Setup
```bash
# Build and run with Docker Compose
docker-compose up --build

# Development environment
docker-compose --profile dev up -d
```

## 🗂️ Project Structure

```
house-price-prediction-mlops/
├── data/                          # Data directory
│   ├── raw/                       # Raw dataset (Kaggle dataset)
│   │   ├── train.csv              # Training data (1,461 samples)
│   │   └── test.csv               # Test data (1,460 samples)
│   └── processed/                 # Processed data (after EDA & preprocessing)
│       └── train_processed.csv    # Preprocessed training data
├── docs/                          # Project documentation
│   ├── deployment_aws_ec2.md      # AWS EC2 deployment guide
│   ├── docker_usage.md            # Docker usage documentation
│   ├── streamlit_usage.md         # Streamlit app guide
│   ├── testing_guide.md           # Comprehensive testing guide
│   ├── EDA_Summary.md             # Exploratory data analysis summary
│   ├── preprocessing_summary.md   # Data preprocessing summary
│   ├── evaluation_summary.md      # Model evaluation summary
│   ├── evaluation_report.json     # Detailed evaluation metrics
│   └── evaluation_plots/          # Model evaluation visualizations
│       ├── predicted_vs_actual.png
│       ├── residuals.png
│       ├── error_distribution.png
│       └── error_metrics.png
├── mlruns/                        # MLflow tracking logs and experiments
├── models/                        # Trained models and artifacts
│   ├── best_model.pkl             # Best performing model (XGBoost)
│   ├── scaler.pkl                 # Feature scaler
│   └── model_metadata.pkl         # Model metadata
├── notebooks/                     # Jupyter Notebooks for analysis
│   ├── 01_EDA_HousePrices.ipynb  # Exploratory data analysis
│   └── 02_Test_Preprocessing.ipynb # Preprocessing validation
├── src/                           # Source code
│   ├── __init__.py
│   ├── config.py                  # Configuration file (paths, hyperparameters)
│   ├── data_ingestion.py          # Data ingestion script
│   ├── preprocessing.py           # Data preprocessing and feature engineering
│   ├── model_strategies.py        # Model strategy patterns
│   ├── model_training.py          # Model training script
│   ├── evaluation.py              # Model evaluation script
│   ├── deployment.py              # Model deployment script using MLflow
│   ├── inference.py               # Inference pipeline for prediction
│   ├── streamlit_app.py           # Streamlit web application
│   └── utils.py                   # Helper functions (logging, etc.)
├── venv/                          # Python virtual environment
├── .git/                          # Git repository
├── Dockerfile                     # Multi-stage Dockerfile
├── docker-compose.yml             # Docker Compose for MLflow + App
├── docker-compose.override.yml    # Development overrides
├── .dockerignore                  # Docker build exclusions
├── aws-cloudformation.yml         # AWS CloudFormation template
├── deploy_ec2.sh                  # AWS EC2 deployment script
├── requirements.txt               # Python dependencies (185 packages)
├── .gitignore                     # Git ignore rules
├── README.md                      # Project documentation
├── roadmap.md                     # Project phases and implementation status
├── main.py                        # Main pipeline orchestration script
├── test_pipeline.py               # Comprehensive testing suite
├── test_evaluation.py             # Evaluation testing
├── test_training.py               # Training testing
├── run_streamlit.py               # Streamlit launcher script
├── reprocess_data.py              # Data reprocessing utility
├── Makefile                       # Build and deployment commands
└── setup.py                       # For packaging and installation
```

## 🧪 Run the Pipeline

### Quick Start
```bash
# Run complete pipeline
python main.py --step all

# Or use Makefile
make pipeline
```

### Individual Steps
```bash
# Data ingestion
python main.py --step data

# Model training
python main.py --step train

# Model evaluation
python main.py --step eval

# Model deployment
python main.py --step deploy

# Test inference
python main.py --step inference
```

### Testing
```bash
# Run all tests
python test_pipeline.py

# Test specific components
python test_pipeline.py --test data
python test_pipeline.py --test train
python test_pipeline.py --test eval
python test_pipeline.py --test docker
```

## 📦 Deployment

### Local Development
```bash
# Run with Docker Compose
docker-compose up --build

# Development environment
docker-compose --profile dev up -d
```

### AWS EC2 Production Deployment

For production deployment on AWS EC2, we provide multiple options:

📖 **[Complete AWS EC2 Deployment Guide](./docs/deployment_aws_ec2.md)**

#### Option 1: CloudFormation Template (Recommended)
```bash
# Deploy using CloudFormation
aws cloudformation create-stack \
  --stack-name house-price-mlops \
  --template-body file://aws-cloudformation.yml \
  --parameters ParameterKey=KeyName,ParameterValue=your-key-pair-name \
  --capabilities CAPABILITY_NAMED_IAM
```

#### Option 2: Automated Deployment Script
```bash
# On your EC2 instance
curl -sSL https://raw.githubusercontent.com/Harish19102003/house-price-prediction-mlops/main/deploy_ec2.sh | bash
```

#### Option 3: Manual Deployment
```bash
# On your EC2 instance
git clone https://github.com/Harish19102003/house-price-prediction-mlops.git
cd house-price-prediction-mlops
docker-compose up -d --build
```

#### Deployed URLs
Once deployed, access your application at:
- **Status Page**: `http://<ec2-public-ip>`
- **Streamlit App**: `http://<ec2-public-ip>:8501`
- **MLflow UI**: `http://<ec2-public-ip>:5000`
- **With Domain**: `https://your-domain.com` (after NGINX + SSL setup)

### Docker Commands
```bash
# Build images
make docker-build

# Run services
make docker-run

# Check status
make status

# View logs
make logs

# Health check
make health
```

### AWS Deployment Commands
```bash
# Show AWS deployment help
make aws-help

# Setup AWS EC2 instance
make aws-setup

# Deploy to AWS EC2
make aws-deploy

# Clean AWS artifacts
make aws-clean
```

## 📊 Model Performance

Our XGBoost model achieves excellent performance on the test set (292 samples):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.907 | **Excellent**: Model explains ≥90% of variance |
| **RMSE** | $26,712 | **Good**: Acceptable prediction error |
| **MAE** | $16,685 | **Good**: Average absolute error |
| **MAPE** | 10.05% | **Moderate**: Reasonably accurate predictions |

### Model Comparison
| Model | R² Score | RMSE | MAE | MAPE |
|-------|----------|------|-----|------|
| **XGBoost (Best)** | **0.907** | **$26,712** | **$16,685** | **10.05%** |
| Linear Regression | 0.82 | $32,450 | $24,120 | 15.2% |
| Random Forest | 0.89 | $28,340 | $18,230 | 11.8% |

### Evaluation Visualizations
- **Predicted vs Actual Plot**: Shows correlation between predictions and actual values
- **Residuals Plot**: Displays prediction errors across price range
- **Error Distribution**: Histogram of prediction errors
- **Error Metrics**: Comparative visualization of different metrics

## 🧰 Tools & Tech

- **Python 3.9+** - Core programming language
- **Pandas, NumPy, Scikit-Learn, XGBoost** - ML libraries
- **MLflow** - Experiment tracking and model registry
- **Streamlit** - Web application framework
- **Docker & Docker Compose** - Containerization
- **AWS EC2** - Cloud deployment platform
- **AWS CloudFormation** - Infrastructure as Code
- **NGINX** - Reverse proxy and load balancing
- **Let's Encrypt** - SSL certificate management

## 🔍 Testing & Validation

### Comprehensive Testing Suite
- ✅ Data ingestion and preprocessing
- ✅ Model training and evaluation
- ✅ MLflow integration and deployment
- ✅ Docker containerization
- ✅ CLI interface validation
- ✅ Performance benchmarking
- ✅ Security validation
- ✅ AWS deployment testing

### Quality Assurance
```bash
# Run full test suite
make full-test

# Quick validation
make quick-validate

# Check pipeline status
make status
```

## 📚 Documentation

- **[AWS EC2 Deployment Guide](./docs/deployment_aws_ec2.md)** - Complete production deployment
- **[Docker Usage Guide](./docs/docker_usage.md)** - Container management
- **[Streamlit App Guide](./docs/streamlit_usage.md)** - Web interface usage
- **[Testing Guide](./docs/testing_guide.md)** - Comprehensive testing
- **[EDA Summary](./docs/EDA_Summary.md)** - Exploratory data analysis
- **[Evaluation Summary](./docs/evaluation_summary.md)** - Model performance analysis
- **[Documentation Status](./docs/documentation_status.md)** - Documentation coverage and status
- **[Project Roadmap](./roadmap.md)** - Implementation status and phases

## 🚀 Quick Commands

```bash
# Development setup
make dev-setup

# Run complete pipeline
make pipeline

# Start Streamlit app
make streamlit

# Production deployment
make prod-deploy

# AWS deployment
make aws-deploy

# Full system test
make full-test
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make full-test`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

- **Documentation**: Check the `docs/` directory
- **Issues**: Create an issue on GitHub
- **Testing**: Run `make status` to check system health
- **Deployment**: Follow the [AWS EC2 guide](./docs/deployment_aws_ec2.md)
- **AWS Help**: Run `make aws-help` for deployment assistance

---

**🎉 Project Status**: ✅ **PRODUCTION READY**  
**Last Updated**: June 22, 2025  
**Model Performance**: R² = 0.907 (Excellent)
