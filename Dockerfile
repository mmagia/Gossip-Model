FROM python:3.10-slim
WORKDIR /app

# Install build-essential for C-extensions if needed by any dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies; PyTorch is included here for ML capabilities
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create separate directory for source code to maintain clean workspace
RUN mkdir -p ./src

# Generate gRPC code from proto file
COPY gossip.proto .
RUN python -m grpc_tools.protoc -I. --python_out=./src --grpc_python_out=./src gossip.proto

# Copy the rest of the application components
COPY src/ ./src/
COPY tests/ ./tests/
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/

# Initialize persistent storage points for data, logs, and experiment results
RUN mkdir -p /app/data /app/logs /app/experiments/results

# Set Python Path to ensure local modules are discoverable in the container
ENV PYTHONPATH=/app/src

# Default gRPC ports used by the gossip nodes
EXPOSE 50051-50055

CMD ["python", "-m", "main"]