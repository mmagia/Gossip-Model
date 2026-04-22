## 🐳 Docker Deployment (5 Nodes)

### Prerequisites
- Docker (≥ 20.10)
- Docker Compose (≥ 2.0)

### Quick Start

1. **Build and start all nodes:**
```bash
docker compose up --build
```
2. **After stopping the Docker Compose Containers containers plot the results in the diagrams:**
```bash
python scripts/docker_collect.py

