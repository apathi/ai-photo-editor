# AI Photo Editor - Dockerfile
# Multi-stage build for optimized production image

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p examples/sample_images examples/output test_results

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run application
CMD ["python", "app.py"]


# ========================================
# GPU-enabled variant (uncomment to use)
# ========================================

# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu
#
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PIP_NO_CACHE_DIR=1 \
#     DEBIAN_FRONTEND=noninteractive
#
# RUN apt-get update && apt-get install -y \
#     python3.10 \
#     python3-pip \
#     git \
#     wget \
#     curl \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*
#
# WORKDIR /app
#
# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt
#
# COPY . .
#
# RUN mkdir -p examples/sample_images examples/output test_results
#
# EXPOSE 5000
#
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:5000/api/health || exit 1
#
# CMD ["python3", "app.py"]
