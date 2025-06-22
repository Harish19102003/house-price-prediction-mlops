# AWS EC2 Deployment Guide for House Price Prediction MLOps

## 🚀 Overview

This guide provides step-by-step instructions for deploying the House Price Prediction MLOps pipeline on AWS EC2, including Docker setup, security configuration, and optional HTTPS with NGINX.

## 📋 Prerequisites

- AWS Account with EC2 access
- SSH client (PuTTY for Windows, Terminal for Mac/Linux)
- Basic knowledge of AWS EC2 and Linux commands
- Domain name (optional, for HTTPS)

## 🚀 Quick Deployment with CloudFormation

### Option 1: Automated Deployment (Recommended)

For the fastest deployment, use our CloudFormation template:

1. **Launch CloudFormation Stack:**
   ```bash
   # Using AWS CLI
   aws cloudformation create-stack \
     --stack-name house-price-mlops \
     --template-body file://aws-cloudformation.yml \
     --parameters ParameterKey=KeyName,ParameterValue=your-key-pair-name \
     --capabilities CAPABILITY_NAMED_IAM
   ```

2. **Or use AWS Console:**
   - Go to AWS CloudFormation Console
   - Click "Create stack" → "With new resources"
   - Upload `aws-cloudformation.yml`
   - Fill in parameters:
     - **KeyName**: Your EC2 key pair name
     - **InstanceType**: t3.medium (recommended)
     - **VolumeSize**: 20 GB
     - **AllowedIP**: Your IP address (e.g., 192.168.1.1/32)

3. **Wait for Deployment:**
   - Stack creation takes 5-10 minutes
   - Check the "Outputs" tab for your application URLs

4. **Access Your Application:**
   - **Status Page**: `http://<public-ip>`
   - **Streamlit App**: `http://<public-ip>:8501`
   - **MLflow UI**: `http://<public-ip>:5000`

### Option 2: Manual Deployment

If you prefer manual control, follow the step-by-step instructions below.

## 🏗️ Step 1: Launch EC2 Instance

### 1.1 Instance Configuration

**Recommended Instance Type:**
- **t3.medium** (2 vCPU, 4 GB RAM) - For development/testing
- **t3.large** (2 vCPU, 8 GB RAM) - For production workloads
- **c5.large** (2 vCPU, 4 GB RAM) - For compute-intensive tasks

**AMI Selection:**
- **Amazon Linux 2023** (recommended)
- **Ubuntu 22.04 LTS** (alternative)

### 1.2 Security Group Configuration

Create a security group with the following rules:

| Type | Protocol | Port Range | Source | Description |
|------|----------|------------|--------|-------------|
| SSH | TCP | 22 | Your IP | SSH access |
| HTTP | TCP | 80 | 0.0.0.0/0 | HTTP traffic |
| HTTPS | TCP | 443 | 0.0.0.0/0 | HTTPS traffic |
| Custom TCP | TCP | 8501 | 0.0.0.0/0 | Streamlit app |
| Custom TCP | TCP | 5000 | 0.0.0.0/0 | MLflow tracking |
| Custom TCP | TCP | 8888 | Your IP | Jupyter (optional) |

### 1.3 Launch Commands

```bash
# Using AWS CLI
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --count 1 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=house-price-mlops}]'
```

## 🔧 Step 2: Connect to EC2 Instance

### 2.1 SSH Connection

```bash
# Using SSH key
ssh -i "your-key.pem" ec2-user@your-ec2-public-ip

# For Ubuntu AMI
ssh -i "your-key.pem" ubuntu@your-ec2-public-ip
```

### 2.2 Initial Setup

```bash
# Update system packages
sudo yum update -y  # For Amazon Linux
# OR
sudo apt update && sudo apt upgrade -y  # For Ubuntu

# Install essential tools
sudo yum install -y git curl wget unzip  # For Amazon Linux
# OR
sudo apt install -y git curl wget unzip  # For Ubuntu
```

## 🐳 Step 3: Install Docker and Docker Compose

### 3.1 Install Docker

**For Amazon Linux 2023:**
```bash
# Install Docker
sudo yum install -y docker

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -a -G docker ec2-user

# Logout and login again, or run:
newgrp docker
```

**For Ubuntu 22.04:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -a -G docker ubuntu

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Logout and login again, or run:
newgrp docker
```

### 3.2 Install Docker Compose

```bash
# Install Docker Compose v2
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

### 3.3 Verify Docker Installation

```bash
# Test Docker
docker run hello-world

# Check Docker status
docker info
```

## 📦 Step 4: Deploy the ML Pipeline

### 4.1 Clone the Repository

```bash
# Clone the project
git clone https://github.com/yourusername/house-price-prediction-mlops.git
cd house-price-prediction-mlops

# Verify project structure
ls -la
```

### 4.2 Configure Environment

```bash
# Create environment file
cat > .env << EOF
# MLflow Configuration
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=your_secure_password_here
MLFLOW_SERVE_ARTIFACTS=true

# Streamlit Configuration
STREAMLIT_SERVER_ENABLE_CORS=true
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Security Configuration
DOCKER_NETWORK_MODE=bridge
EOF

# Set proper permissions
chmod 600 .env
```

### 4.3 Build and Run Services

```bash
# Build Docker images
docker-compose build

# Run services in background
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4.4 Verify Deployment

```bash
# Check if services are running
curl -f http://localhost:8501/ || echo "Streamlit not ready"
curl -f http://localhost:5000/health || echo "MLflow not ready"

# Check container health
docker-compose ps
```

## 🌐 Step 5: Access Your Application

### 5.1 Public URLs

Once deployed, access your application at:

- **Streamlit App**: `http://your-ec2-public-ip:8501`
- **MLflow UI**: `http://your-ec2-public-ip:5000`

### 5.2 Test the Pipeline

```bash
# Run the complete pipeline
docker-compose exec streamlit python main.py --step all --skip-deployment

# Test inference
docker-compose exec streamlit python src/inference.py --sample
```

## 🔒 Step 6: Security Configuration (Optional)

### 6.1 Install NGINX

```bash
# Install NGINX
sudo yum install -y nginx  # For Amazon Linux
# OR
sudo apt install -y nginx  # For Ubuntu

# Start and enable NGINX
sudo systemctl start nginx
sudo systemctl enable nginx
```

### 6.2 Configure NGINX Reverse Proxy

```bash
# Create NGINX configuration
sudo tee /etc/nginx/conf.d/house-price-mlops.conf > /dev/null << 'EOF'
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    # Streamlit App
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # MLflow UI
    location /mlflow/ {
        proxy_pass http://localhost:5000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Test NGINX configuration
sudo nginx -t

# Reload NGINX
sudo systemctl reload nginx
```

### 6.3 Install SSL Certificate with Certbot

```bash
# Install Certbot
sudo yum install -y certbot python3-certbot-nginx  # For Amazon Linux
# OR
sudo apt install -y certbot python3-certbot-nginx  # For Ubuntu

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Test automatic renewal
sudo certbot renew --dry-run
```

### 6.4 Update Security Group

After setting up NGINX, update your security group to only allow:
- Port 22 (SSH) from your IP
- Port 80 (HTTP) from anywhere
- Port 443 (HTTPS) from anywhere

## 📊 Step 7: Monitoring and Maintenance

### 7.1 Health Monitoring

```bash
# Create health check script
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

# Run health check
./health_check.sh
```

### 7.2 Log Management

```bash
# View service logs
docker-compose logs -f streamlit
docker-compose logs -f mlflow

# Set up log rotation
sudo tee /etc/logrotate.d/docker-compose << 'EOF'
/opt/house-price-prediction-mlops/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 root root
}
EOF
```

### 7.3 Backup Strategy

```bash
# Create backup script
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

# Run backup
./backup.sh
```

## 🔄 Step 8: Updates and Maintenance

### 8.1 Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart services
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Verify deployment
docker-compose ps
./health_check.sh
```

### 8.2 System Updates

```bash
# Update system packages
sudo yum update -y  # For Amazon Linux
# OR
sudo apt update && sudo apt upgrade -y  # For Ubuntu

# Update Docker
sudo yum update docker  # For Amazon Linux
# OR
sudo apt update docker.io  # For Ubuntu

# Restart Docker
sudo systemctl restart docker
```

### 8.3 Cleanup

```bash
# Clean unused Docker resources
docker system prune -f

# Clean old backups (keep last 7 days)
find /opt/backups/house-price-mlops -name "*.tar.gz" -mtime +7 -delete
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
sudo netstat -tulpn | grep :8501
sudo netstat -tulpn | grep :5000

# Kill process if needed
sudo kill -9 <PID>
```

#### 2. Docker Permission Issues
```bash
# Fix Docker permissions
sudo chmod 666 /var/run/docker.sock
sudo usermod -a -G docker $USER
newgrp docker
```

#### 3. Memory Issues
```bash
# Check memory usage
free -h
docker stats

# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Disk Space Issues
```bash
# Check disk usage
df -h

# Clean Docker
docker system prune -a -f

# Clean logs
sudo journalctl --vacuum-time=7d
```

### Debug Commands

```bash
# Check service status
docker-compose ps
docker-compose logs

# Check system resources
htop
df -h
free -h

# Check network connectivity
curl -v http://localhost:8501
curl -v http://localhost:5000/health

# Check security group
aws ec2 describe-security-groups --group-ids sg-xxxxxxxxx
```

## 📈 Performance Optimization

### 1. Resource Allocation

```bash
# Monitor resource usage
docker stats

# Adjust resource limits in docker-compose.yml
services:
  streamlit:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### 2. Caching Strategy

```bash
# Enable Docker build cache
docker-compose build --parallel

# Use volume mounts for persistent data
volumes:
  - ./data:/app/data:delegated
  - ./models:/app/models:delegated
```

### 3. Load Balancing (Optional)

For high-traffic scenarios, consider:
- AWS Application Load Balancer
- Multiple EC2 instances
- Auto Scaling Groups
- CloudFront CDN

## 💰 Cost Optimization

### 1. Instance Selection
- Use Spot Instances for non-critical workloads
- Right-size instances based on usage
- Consider Reserved Instances for production

### 2. Storage Optimization
- Use EBS gp3 volumes for better performance/cost ratio
- Implement lifecycle policies for backups
- Use S3 for long-term storage

### 3. Monitoring Costs
```bash
# Set up AWS Cost Explorer alerts
# Monitor CloudWatch metrics
# Use AWS Budgets for cost tracking
```

## 🔐 Security Best Practices

### 1. Network Security
- Use VPC with private subnets
- Implement Network ACLs
- Use AWS WAF for web application protection

### 2. Access Control
- Use IAM roles instead of access keys
- Implement least privilege access
- Enable AWS CloudTrail for audit logging

### 3. Data Protection
- Encrypt data at rest and in transit
- Use AWS KMS for key management
- Implement regular security updates

## 📞 Support and Resources

### Useful Commands
```bash
# Quick status check
docker-compose ps && ./health_check.sh

# View recent logs
docker-compose logs --tail=100

# Restart services
docker-compose restart

# Full system check
df -h && free -h && docker system df
```

### Documentation Links
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Docker Documentation](https://docs.docker.com/)
- [NGINX Documentation](https://nginx.org/en/docs/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)

### Getting Help
1. Check logs: `docker-compose logs -f`
2. Verify configuration: `docker-compose config`
3. Test connectivity: `curl -v http://localhost:8501`
4. Check system resources: `htop` and `df -h`

---

**🎉 Congratulations!** Your House Price Prediction MLOps pipeline is now deployed on AWS EC2 with production-ready configuration, security, and monitoring. 