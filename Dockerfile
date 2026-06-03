# ============================================================
# Dockerfile for the Automated Trading System
# Multi-stage build for minimal production image
# ============================================================

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create directories for logs, data, reports, and state
RUN mkdir -p logs data reports/charts state data_cache

# Expose dashboard port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command: run backtesting mode
# Override with: docker run <image> python main.py --mode paper
CMD ["python", "main.py", "--mode", "backtest", "--visualize"]
