# Dockerfile.simple - Simple FastAPI VLM WebSocket Service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-fastapi.txt .
RUN pip install --no-cache-dir -r requirements-fastapi.txt

# Copy simple application modules
COPY models.py .
COPY websocket_manager.py .
COPY api.py .

# Create non-root user
RUN useradd --create-home --shell /bin/bash vlm && \
    chown -R vlm:vlm /app
USER vlm

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the simple FastAPI service
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]