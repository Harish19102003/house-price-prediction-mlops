# Docker Setup for House Price Prediction MLOps

## 🐳 Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available

### 1. Build and Run Production Services

```bash
# Build and start MLflow + Streamlit
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f
```

### 2. Access Services

Once running, access the services at:
- **Streamlit App**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

## 🚀 Deployment Options

### Option 1: Production Deployment
```bash
# Build production images
docker-compose -f docker-compose.yml build

# Start production services
docker-compose up -d

# Check service status
docker-compose ps
```

### Option 2: Development Environment
```bash
# Start development environment with Jupyter
docker-compose --profile dev up -d

# Access development services:
# - Jupyter Lab: http://localhost:8888
# - Streamlit: http://localhost:8502
# - MLflow: http://localhost:5001
```

### Option 3: Individual Services
```bash
# Run only MLflow
docker-compose up mlflow

# Run only Streamlit
docker-compose up streamlit

# Run with specific build target
docker build --target development -t house-price-dev .
```

## 🏗️ Docker Architecture

### Multi-Stage Build
- **Base Stage**: Common dependencies and setup
- **Development Stage**: Additional dev tools (Jupyter, testing)
- **Production Stage**: Optimized for serving

### Services
1. **MLflow Tracking Server** (Port 5000)
   - Model registry and experiment tracking
   - Artifact storage
   - REST API for model serving

2. **Streamlit App** (Port 8501)
   - Web interface for predictions
   - Connected to MLflow for model loading
   - Auto-saves predictions to docs/

3. **Development Environment** (Optional)
   - Jupyter Lab for notebook development
   - Full source code mounting
   - Debug-friendly configuration

## 📁 Volume Mounts

### Persistent Data
```yaml
volumes:
  - ./data:/app/data          # Raw and processed data
  - ./models:/app/models      # Trained models
  - ./docs:/app/docs          # Documentation and results
  - ./mlruns:/app/mlruns      # MLflow tracking data
```

### Development Volumes
```yaml
volumes:
  - ./src:/app/src            # Live code reloading
  - ./:/app                   # Full project mount (dev only)
```

## 🔧 Configuration

### Environment Variables

#### MLflow Service
```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_SERVE_ARTIFACTS=true
MLFLOW_TRACKING_USERNAME=admin      # Development only
MLFLOW_TRACKING_PASSWORD=admin      # Development only
```

#### Streamlit Service
```bash
MLFLOW_TRACKING_URI=http://mlflow:5000
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_LOGGER_LEVEL=debug        # Development only
```

### Network Configuration
- **mlops-network**: Bridge network for service communication
- **Port Mapping**: 
  - 8501 → Streamlit
  - 5000 → MLflow
  - 8888 → Jupyter (dev only)

## 🛠️ Development Workflow

### 1. Initial Setup
```bash
# Clone and setup
git clone <repository>
cd house-price-prediction-mlops

# Build development environment
docker-compose --profile dev build

# Start development services
docker-compose --profile dev up -d
```

### 2. Training Pipeline
```bash
# Access Jupyter Lab
open http://localhost:8888

# Or run training directly
docker-compose exec dev python src/model_training.py
```

### 3. Model Evaluation
```bash
# Run evaluation
docker-compose exec dev python src/evaluation.py

# Deploy model to MLflow
docker-compose exec dev python src/deployment.py
```

### 4. Testing Streamlit
```bash
# Access Streamlit app
open http://localhost:8502

# Check logs
docker-compose logs streamlit
```

## 🔍 Monitoring and Debugging

### Health Checks
```bash
# Check service health
docker-compose ps

# View health check logs
docker-compose exec streamlit curl -f http://localhost:8501/
```

### Logs
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs streamlit
docker-compose logs mlflow

# Follow logs in real-time
docker-compose logs -f streamlit
```

### Debugging
```bash
# Access container shell
docker-compose exec streamlit bash
docker-compose exec mlflow bash

# Check container resources
docker stats

# Inspect container
docker-compose exec streamlit ls -la /app
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
lsof -i :8501
lsof -i :5000

# Kill process or change ports in docker-compose.yml
```

#### 2. Permission Issues
```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./data ./models ./docs ./mlruns

# Or run with proper user mapping
docker-compose run --user $(id -u):$(id -g) streamlit
```

#### 3. Memory Issues
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory: 8GB+

# Or use lighter base image
# Change FROM python:3.9-slim to python:3.9-alpine
```

#### 4. Model Not Found
```bash
# Ensure models are trained first
docker-compose exec dev python src/model_training.py

# Check model files exist
docker-compose exec streamlit ls -la /app/models/
```

#### 5. MLflow Connection Issues
```bash
# Check MLflow is running
docker-compose ps mlflow

# Test connection
curl http://localhost:5000/health

# Check network connectivity
docker-compose exec streamlit ping mlflow
```

### Performance Optimization

#### 1. Build Optimization
```bash
# Use build cache
docker-compose build --no-cache

# Parallel builds
docker-compose build --parallel
```

#### 2. Resource Limits
```yaml
# Add to docker-compose.yml services
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

#### 3. Volume Performance
```bash
# Use delegated mount for better performance (macOS)
volumes:
  - ./data:/app/data:delegated
```

## 🔒 Security Considerations

### Production Deployment
1. **Remove development overrides**:
   ```bash
   rm docker-compose.override.yml
   ```

2. **Use secrets management**:
   ```yaml
   secrets:
     mlflow_password:
       file: ./secrets/mlflow_password.txt
   ```

3. **Enable authentication**:
   ```bash
   # Set proper credentials
   export MLFLOW_TRACKING_USERNAME=your_username
   export MLFLOW_TRACKING_PASSWORD=your_secure_password
   ```

4. **Network security**:
   ```yaml
   # Restrict network access
   networks:
     mlops-network:
       driver: bridge
       internal: true  # No external access
   ```

### Container Security
- Non-root user in production stage
- Minimal base image (python:3.9-slim)
- Health checks for monitoring
- Resource limits to prevent abuse

## 📊 Monitoring

### Service Status
```bash
# Check all services
docker-compose ps

# Monitor resource usage
docker stats

# View service logs
docker-compose logs --tail=100
```

### Application Metrics
- **Streamlit**: Built-in metrics at http://localhost:8501/_stcore/metrics
- **MLflow**: REST API metrics
- **Custom**: Add Prometheus/Grafana for advanced monitoring

## 🚀 Production Deployment

### AWS EC2 Example
```bash
# Install Docker on EC2
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy
git clone <repository>
cd house-price-prediction-mlops
docker-compose up -d --build
```

### Environment Variables
```bash
# Production environment file
cat > .env << EOF
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=secure_password_here
STREAMLIT_SERVER_ENABLE_CORS=true
EOF

# Use with docker-compose
docker-compose --env-file .env up -d
``` 