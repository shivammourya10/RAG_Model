# Production Dockerfile for RAG Model
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for production
ENV RAG_RENDER_MODE=true
ENV RAG_FAST_STARTUP=true
ENV RAG_PRELOAD_MODELS=true
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/huggingface_cache
ENV TORCH_HOME=/tmp/torch_cache

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download models to reduce cold start time
RUN python -c "\
import os; \
from model_cache import ModelCache; \
try: \
    print('🔄 Pre-downloading models...'); \
    ModelCache.get_embedder('all-MiniLM-L6-v2'); \
    print('✅ Models pre-downloaded successfully'); \
except Exception as e: \
    print(f'⚠️ Model pre-download failed: {e}'); \
    print('Models will be downloaded on first request'); \
"

# Create cache directories
RUN mkdir -p /tmp/transformers_cache /tmp/huggingface_cache /tmp/torch_cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Start command
CMD ["python", "main.py"]
