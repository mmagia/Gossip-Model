FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (без torch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create fresh src directory
RUN mkdir -p ./src

# Generate gRPC code from proto file
COPY gossip.proto .
RUN python -m grpc_tools.protoc -I. --python_out=./src --grpc_python_out=./src gossip.proto

# Copy application code
COPY src/ ./src/
RUN python -m grpc_tools.protoc -I. --python_out=./src --grpc_python_out=./src gossip.proto

COPY tests/ ./tests/
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/experiments/results

ENV PYTHONPATH=/app/src

EXPOSE 50051-50055

CMD ["python", "-m", "main"]