# Multi-stage Dockerfile for House Price Prediction MLOps Pipeline
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY docs/ ./docs/
COPY notebooks/ ./notebooks/
COPY mlruns/ ./mlruns/

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/docs /app/mlruns

# Stage 2: Development stage (for training and development)
FROM base as development

# Install additional development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    black \
    flake8 \
    pytest

# Expose ports for development
EXPOSE 8888 8501 5000

# Default command for development
CMD ["bash"]

# Stage 3: Production stage (for serving)
FROM base as production

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# Expose ports
EXPOSE 8501 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/ || exit 1

# Default command for production
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 