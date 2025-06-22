# Makefile for House Price Prediction MLOps Pipeline

.PHONY: help install test clean pipeline docker-build docker-run docker-test docker-clean validate aws-deploy aws-setup aws-clean

# Default target
help:
	@echo "🏠 House Price Prediction MLOps Pipeline"
	@echo "========================================"
	@echo ""
	@echo "Available targets:"
	@echo "  help          - Show this help message"
	@echo "  install       - Install dependencies"
	@echo "  test          - Run all tests"
	@echo "  pipeline      - Run complete ML pipeline"
	@echo "  pipeline-no-deploy - Run pipeline without MLflow deployment"
	@echo "  data          - Run data ingestion only"
	@echo "  train         - Run model training only"
	@echo "  eval          - Run model evaluation only"
	@echo "  deploy        - Deploy model to MLflow only"
	@echo "  inference     - Test inference only"
	@echo "  validate      - Validate outputs only"
	@echo "  streamlit     - Run Streamlit app locally"
	@echo "  docker-build  - Build Docker images"
	@echo "  docker-run    - Run services with Docker Compose"
	@echo "  docker-dev    - Run development environment"
	@echo "  docker-test   - Test pipeline in Docker"
	@echo "  docker-clean  - Clean Docker containers and images"
	@echo "  clean         - Clean generated files"
	@echo "  status        - Check pipeline status"
	@echo ""
	@echo "AWS EC2 Deployment:"
	@echo "  aws-deploy    - Deploy to AWS EC2 (requires EC2 instance)"
	@echo "  aws-setup     - Setup AWS EC2 instance (manual steps)"
	@echo "  aws-clean     - Clean AWS deployment artifacts"
	@echo "  aws-help      - Show AWS deployment help"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Run all tests
test:
	@echo "🧪 Running tests..."
	python -m pytest test_*.py -v
	@echo "✅ Tests completed"

# Run complete pipeline
pipeline:
	@echo "🚀 Running complete ML pipeline..."
	python main.py --step all
	@echo "✅ Pipeline completed"

# Run pipeline without deployment
pipeline-no-deploy:
	@echo "🚀 Running ML pipeline (without deployment)..."
	python main.py --step all --skip-deployment
	@echo "✅ Pipeline completed"

# Individual pipeline steps
data:
	@echo "📊 Running data ingestion..."
	python main.py --step data

train:
	@echo "🤖 Running model training..."
	python main.py --step train

eval:
	@echo "📈 Running model evaluation..."
	python main.py --step eval

deploy:
	@echo "🚀 Running model deployment..."
	python main.py --step deploy

inference:
	@echo "🔮 Testing inference..."
	python main.py --step inference

validate:
	@echo "🔍 Validating outputs..."
	python main.py --step validate

# Run Streamlit app locally
streamlit:
	@echo "🌐 Starting Streamlit app..."
	python run_streamlit.py

# Docker operations
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build
	@echo "✅ Docker images built"

docker-run:
	@echo "🐳 Starting Docker services..."
	docker-compose up -d
	@echo "✅ Services started"
	@echo "🌐 Streamlit: http://localhost:8501"
	@echo "📊 MLflow: http://localhost:5000"

docker-dev:
	@echo "🐳 Starting development environment..."
	docker-compose --profile dev up -d
	@echo "✅ Development environment started"
	@echo "📓 Jupyter: http://localhost:8888"
	@echo "🌐 Streamlit: http://localhost:8502"
	@echo "📊 MLflow: http://localhost:5001"

docker-test:
	@echo "🐳 Testing pipeline in Docker..."
	docker-compose exec streamlit python main.py --step all --skip-deployment
	@echo "✅ Docker pipeline test completed"

docker-clean:
	@echo "🧹 Cleaning Docker containers and images..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Docker cleanup completed"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf .pytest_cache/
	rm -rf mlruns/
	rm -rf models/*.pkl
	rm -rf docs/evaluation_plots/*.png
	rm -rf docs/*.json
	rm -rf docs/*.md
	rm -rf docs/pipeline.log
	@echo "✅ Cleanup completed"

# Check pipeline status
status:
	@echo "📊 Pipeline Status Check"
	@echo "========================"
	@echo "Checking required files and directories..."
	@python -c "
import sys
from pathlib import Path
sys.path.append('src')
from config import Config
config = Config()

files_to_check = [
    config.PROCESSED_TRAIN_FILE,
    config.MODEL_ARTIFACT_PATH,
    config.SCALER_PATH,
    config.MODELS_DIR / 'model_metadata.pkl',
    config.DOCS_DIR / 'evaluation_report.json',
    config.DOCS_DIR / 'evaluation_summary.md'
]

dirs_to_check = [
    config.DOCS_DIR / 'evaluation_plots',
    config.MLRUNS_DIR
]

print('Files:')
for file_path in files_to_check:
    status = '✅' if file_path.exists() else '❌'
    print(f'  {status} {file_path}')

print('\nDirectories:')
for dir_path in dirs_to_check:
    status = '✅' if dir_path.exists() else '❌'
    print(f'  {status} {dir_path}')

print('\nEvaluation Plots:')
plots_dir = config.DOCS_DIR / 'evaluation_plots'
if plots_dir.exists():
    plots = list(plots_dir.glob('*.png'))
    for plot in plots:
        print(f'  ✅ {plot.name}')
else:
    print('  ❌ No evaluation plots directory')
"

# Quick validation
quick-validate:
	@echo "🔍 Quick validation..."
	@python -c "
import sys
from pathlib import Path
sys.path.append('src')
from config import Config
config = Config()

required_files = [
    config.MODEL_ARTIFACT_PATH,
    config.DOCS_DIR / 'evaluation_report.json'
]

all_good = True
for file_path in required_files:
    if not file_path.exists():
        print(f'❌ Missing: {file_path}')
        all_good = False
    else:
        print(f'✅ Found: {file_path}')

if all_good:
    print('\n🎉 All required files present!')
else:
    print('\n❌ Some files are missing. Run: make pipeline')
    exit(1)
"

# Development helpers
dev-setup:
	@echo "🛠️ Setting up development environment..."
	make install
	make pipeline-no-deploy
	@echo "✅ Development setup completed"

dev-test:
	@echo "🧪 Running development tests..."
	make test
	make quick-validate
	@echo "✅ Development tests completed"

# Production helpers
prod-deploy:
	@echo "🚀 Production deployment..."
	make docker-build
	make docker-run
	@echo "✅ Production deployment completed"

prod-test:
	@echo "🧪 Production testing..."
	make docker-test
	@echo "✅ Production tests completed"

# Monitoring
logs:
	@echo "📋 Showing service logs..."
	docker-compose logs -f

logs-streamlit:
	@echo "📋 Showing Streamlit logs..."
	docker-compose logs -f streamlit

logs-mlflow:
	@echo "📋 Showing MLflow logs..."
	docker-compose logs -f mlflow

# Health checks
health:
	@echo "🏥 Health check..."
	@curl -f http://localhost:8501/ > /dev/null 2>&1 && echo "✅ Streamlit: Healthy" || echo "❌ Streamlit: Unhealthy"
	@curl -f http://localhost:5000/health > /dev/null 2>&1 && echo "✅ MLflow: Healthy" || echo "❌ MLflow: Unhealthy"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "Documentation files:"
	@ls -la docs/
	@echo ""
	@echo "Usage guides:"
	@echo "  - docs/streamlit_usage.md"
	@echo "  - docs/docker_usage.md"
	@echo "  - docs/evaluation_summary.md"

# Full system test
full-test:
	@echo "🧪 Running full system test..."
	make clean
	make install
	make pipeline-no-deploy
	make validate
	make docker-build
	make docker-run
	sleep 10
	make health
	make docker-clean
	@echo "✅ Full system test completed"

# AWS EC2 Deployment
aws-deploy:
	@echo "☁️ Deploying to AWS EC2..."
	@echo "This will run the deployment script on your EC2 instance."
	@echo "Make sure you have:"
	@echo "  1. SSH access to your EC2 instance"
	@echo "  2. Security group configured for ports 22, 80, 443, 8501, 5000"
	@echo "  3. Internet access on the instance"
	@echo ""
	@echo "To deploy:"
	@echo "  1. Copy deploy_ec2.sh to your EC2 instance"
	@echo "  2. SSH into your instance"
	@echo "  3. Run: ./deploy_ec2.sh"
	@echo ""
	@echo "Or use the quick deployment:"
	@echo "  ssh -i your-key.pem ec2-user@your-ec2-ip 'curl -sSL https://raw.githubusercontent.com/Harish19102003/house-price-prediction-mlops/main/deploy_ec2.sh | bash'"

aws-setup:
	@echo "☁️ AWS EC2 Setup Instructions"
	@echo "============================"
	@echo ""
	@echo "1. Launch EC2 Instance:"
	@echo "   - Instance Type: t3.medium (2 vCPU, 4 GB RAM)"
	@echo "   - AMI: Amazon Linux 2023 or Ubuntu 22.04 LTS"
	@echo "   - Storage: 20 GB gp3"
	@echo ""
	@echo "2. Configure Security Group:"
	@echo "   - SSH (22): Your IP"
	@echo "   - HTTP (80): 0.0.0.0/0"
	@echo "   - HTTPS (443): 0.0.0.0/0"
	@echo "   - Custom TCP (8501): 0.0.0.0/0 (Streamlit)"
	@echo "   - Custom TCP (5000): 0.0.0.0/0 (MLflow)"
	@echo ""
	@echo "3. Connect to Instance:"
	@echo "   ssh -i your-key.pem ec2-user@your-ec2-public-ip"
	@echo ""
	@echo "4. Run Deployment:"
	@echo "   ./deploy_ec2.sh"
	@echo ""
	@echo "5. Access Application:"
	@echo "   - Streamlit: http://your-ec2-ip:8501"
	@echo "   - MLflow: http://your-ec2-ip:5000"
	@echo ""
	@echo "📖 For detailed instructions, see: docs/deployment_aws_ec2.md"

aws-clean:
	@echo "🧹 Cleaning AWS deployment artifacts..."
	@echo "This will clean local AWS-related files:"
	@rm -f .env.aws
	@rm -f aws-deployment.log
	@echo "✅ AWS cleanup completed"

aws-help:
	@echo "☁️ AWS EC2 Deployment Help"
	@echo "=========================="
	@echo ""
	@echo "Quick Start:"
	@echo "  1. Launch EC2 instance (t3.medium recommended)"
	@echo "  2. Configure security group for ports 22, 80, 443, 8501, 5000"
	@echo "  3. SSH into instance: ssh -i key.pem ec2-user@ip"
	@echo "  4. Run: ./deploy_ec2.sh"
	@echo ""
	@echo "Manual Deployment:"
	@echo "  git clone https://github.com/Harish19102003/house-price-prediction-mlops.git"
	@echo "  cd house-price-prediction-mlops"
	@echo "  docker-compose up -d --build"
	@echo ""
	@echo "Useful Commands:"
	@echo "  - Check status: docker-compose ps"
	@echo "  - View logs: docker-compose logs -f"
	@echo "  - Health check: ./health_check.sh"
	@echo "  - Backup: ./backup.sh"
	@echo ""
	@echo "Troubleshooting:"
	@echo "  - Check security group rules"
	@echo "  - Verify Docker installation"
	@echo "  - Check service logs"
	@echo "  - Run health check script"
	@echo ""
	@echo "📖 Complete guide: docs/deployment_aws_ec2.md"

# AWS deployment script info
aws-script-info:
	@echo "📜 AWS Deployment Script Information"
	@echo "===================================="
	@echo ""
	@echo "Script: deploy_ec2.sh"
	@echo "Purpose: Automated AWS EC2 deployment"
	@echo ""
	@echo "What it does:"
	@echo "  1. Updates system packages"
	@echo "  2. Installs Docker and Docker Compose"
	@echo "  3. Clones the project repository"
	@echo "  4. Configures environment variables"
	@echo "  5. Builds and deploys Docker services"
	@echo "  6. Runs the complete ML pipeline"
	@echo "  7. Creates monitoring scripts"
	@echo ""
	@echo "Usage:"
	@echo "  ./deploy_ec2.sh [OPTIONS]"
	@echo ""
	@echo "Options:"
	@echo "  -h, --help        Show help message"
	@echo "  --skip-pipeline   Skip running ML pipeline"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - AWS EC2 instance with internet access"
	@echo "  - Security group configured correctly"
	@echo "  - SSH access to instance"
	@echo ""
	@echo "Output:"
	@echo "  - Application URLs"
	@echo "  - Health check script"
	@echo "  - Backup script"
	@echo "  - Instance information" 