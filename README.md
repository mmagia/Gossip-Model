This project implements a peer-to-peer (P2P) training system where multiple nodes collaborate to train a shared machine learning model without any central server.

##  Key Features
* **Fully Decentralized:** No master node or central parameter server.
* **Gossip Protocol:** Parameters are shared between neighbors via asynchronous updates.
* **Non-IID Support:** Nodes can train on different data shards (e.g., specific MNIST digits) and still converge.
* **Divergence Tracking:** Built-in monitoring of model discrepancy using MSE metrics.


## Project Architecture

```text
Gossip-Model/
├── src/
│   ├── ml/             # ML Engine (Models, Trainers, Aggregation)
│   ├── config.py       # Global Configuration (LR, Alpha, Ports)
│   └── main.py         # P2P Network Logic & Integration
├── experiments/        # Result Analysis & Plots
├── tests/              # Unit & Integration Tests
└── data/               # Local Datasets (MNIST)
```


## Docker Deployment (5 Nodes)

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
```

---


## Running the Project

### Visualization
After training, generate convergence plots:
```bash
python experiments/plot_results.py
python experiments/analyze_divergence.py
```
*Graphs will be saved in `experiments/results/`.*
