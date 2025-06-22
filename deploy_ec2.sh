#!/bin/bash

# AWS EC2 Deployment Script for House Price Prediction MLOps
# This script automates the deployment process on a fresh EC2 instance

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Configuration
PROJECT_NAME="house-price-prediction-mlops"
PROJECT_URL="https://github.com/Harish19102003/house-price-prediction-mlops.git"
DOCKER_COMPOSE_VERSION="v2.20.0"

# Detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    else
        error "Cannot detect OS"
    fi
}

# Update system packages
update_system() {
    log "Updating system packages..."
    
    if [[ "$OS" == *"Amazon Linux"* ]]; then
        sudo yum update -y
        sudo yum install -y git curl wget unzip
    elif [[ "$OS" == *"Ubuntu"* ]]; then
        sudo apt update && sudo apt upgrade -y
        sudo apt install -y git curl wget unzip
    else
        error "Unsupported OS: $OS"
    fi
}

# Install Docker
install_docker() {
    log "Installing Docker..."
    
    if [[ "$OS" == *"Amazon Linux"* ]]; then
        sudo yum install -y docker
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -a -G docker ec2-user
        newgrp docker
    elif [[ "$OS" == *"Ubuntu"* ]]; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -a -G docker ubuntu
        sudo systemctl start docker
        sudo systemctl enable docker
        newgrp docker
    fi
    
    # Verify Docker installation
    if ! docker --version; then
        error "Docker installation failed"
    fi
    
    log "Docker installed successfully"
}

# Install Docker Compose
install_docker_compose() {
    log "Installing Docker Compose..."
    
    sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Verify Docker Compose installation
    if ! docker-compose --version; then
        error "Docker Compose installation failed"
    fi
    
    log "Docker Compose installed successfully"
}

# Clone and setup project
setup_project() {
    log "Setting up project..."
    
    # Remove existing project if it exists
    if [ -d "$PROJECT_NAME" ]; then
        warn "Removing existing project directory"
        rm -rf "$PROJECT_NAME"
    fi
    
    # Clone project
    git clone "$PROJECT_URL"
    cd "$PROJECT_NAME"
    
    # Create environment file
    cat > .env << EOF
# MLflow Configuration
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=admin123
MLFLOW_SERVE_ARTIFACTS=true

# Streamlit Configuration
STREAMLIT_SERVER_ENABLE_CORS=true
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Security Configuration
DOCKER_NETWORK_MODE=bridge
EOF
    
    # Set proper permissions
    chmod 600 .env
    
    log "Project setup completed"
}

# Build and run services
deploy_services() {
    log "Building and deploying services..."
    
    # Build Docker images
    docker-compose build
    
    # Run services in background
    docker-compose up -d
    
    # Wait for services to start
    log "Waiting for services to start..."
    sleep 30
    
    # Check service status
    docker-compose ps
    
    log "Services deployed successfully"
}

# Create health check script
create_health_check() {
    log "Creating health check script..."
    
    cat > health_check.sh << 'EOF'
#!/bin/bash
STREAMLIT_URL="http://localhost:8501"
MLFLOW_URL="http://localhost:5000/health"

echo "Checking Streamlit..."
if curl -f $STREAMLIT_URL > /dev/null 2>&1; then
    echo "✅ Streamlit: Healthy"
else
    echo "❌ Streamlit: Unhealthy"
fi

echo "Checking MLflow..."
if curl -f $MLFLOW_URL > /dev/null 2>&1; then
    echo "✅ MLflow: Healthy"
else
    echo "❌ MLflow: Unhealthy"
fi
EOF
    
    chmod +x health_check.sh
    
    log "Health check script created"
}

# Create backup script
create_backup_script() {
    log "Creating backup script..."
    
    cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups/house-price-mlops"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz data/

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/

# Backup docs
tar -czf $BACKUP_DIR/docs_$DATE.tar.gz docs/

# Backup MLflow data
tar -czf $BACKUP_DIR/mlruns_$DATE.tar.gz mlruns/

echo "Backup completed: $BACKUP_DIR"
EOF
    
    chmod +x backup.sh
    
    log "Backup script created"
}

# Get instance information
get_instance_info() {
    log "Getting instance information..."
    
    # Get public IP
    PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
    
    # Get instance ID
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    
    # Get instance type
    INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
    
    echo "Instance Information:"
    echo "  Instance ID: $INSTANCE_ID"
    echo "  Instance Type: $INSTANCE_TYPE"
    echo "  Public IP: $PUBLIC_IP"
    echo ""
    echo "Application URLs:"
    echo "  Streamlit App: http://$PUBLIC_IP:8501"
    echo "  MLflow UI: http://$PUBLIC_IP:5000"
    echo ""
    echo "Useful Commands:"
    echo "  Check status: docker-compose ps"
    echo "  View logs: docker-compose logs -f"
    echo "  Health check: ./health_check.sh"
    echo "  Backup: ./backup.sh"
    echo "  Stop services: docker-compose down"
    echo "  Restart services: docker-compose restart"
}

# Run pipeline
run_pipeline() {
    log "Running ML pipeline..."
    
    # Run the complete pipeline
    docker-compose exec streamlit python main.py --step all --skip-deployment
    
    log "Pipeline completed successfully"
}

# Main deployment function
main() {
    log "Starting AWS EC2 deployment for House Price Prediction MLOps..."
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        error "Please do not run this script as root"
    fi
    
    # Detect OS
    detect_os
    log "Detected OS: $OS $VER"
    
    # Update system
    update_system
    
    # Install Docker
    install_docker
    
    # Install Docker Compose
    install_docker_compose
    
    # Setup project
    setup_project
    
    # Deploy services
    deploy_services
    
    # Create utility scripts
    create_health_check
    create_backup_script
    
    # Run pipeline
    run_pipeline
    
    # Get instance information
    get_instance_info
    
    log "Deployment completed successfully! 🎉"
    log "Your House Price Prediction MLOps pipeline is now running on AWS EC2."
}

# Help function
show_help() {
    echo "AWS EC2 Deployment Script for House Price Prediction MLOps"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --skip-pipeline Skip running the ML pipeline"
    echo ""
    echo "This script will:"
    echo "  1. Update system packages"
    echo "  2. Install Docker and Docker Compose"
    echo "  3. Clone and setup the project"
    echo "  4. Deploy services using Docker Compose"
    echo "  5. Run the complete ML pipeline"
    echo "  6. Create utility scripts for monitoring"
    echo ""
    echo "Prerequisites:"
    echo "  - AWS EC2 instance with internet access"
    echo "  - Security group configured for ports 22, 80, 443, 8501, 5000"
    echo "  - SSH access to the instance"
}

# Parse command line arguments
SKIP_PIPELINE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --skip-pipeline)
            SKIP_PIPELINE=true
            shift
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Run main function
main 