This project implements a peer-to-peer (P2P) training system where multiple nodes collaborate to train a shared machine learning model without any central server. 5 nodes train locally to classify digits from 0-9, however each node learns only how to classify 2 digits and then exchange its knowledge by propagating weights through the network via gRPC protocol to other nodes.
Thus every peer is able to classify the whole dataset, and its accuracy converges. Moreover, network problems are sumilated to check the concept in real world scenario: network delays and artificial packet losses were introduced. We also implemented centralized logging and containerized each node via Docker.

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


## Running the Project

### Prerequisites
- Docker (≥ 20.10)
- Docker Compose (≥ 2.0)

### Quick Start

1. **Clone the repository:**

2. **Build and start all nodes:**
```bash
docker compose up --build
```
Once in a while centalized logging will happend, and you will see the batches of information provided by nodes in your terminal.
3. **After stopping the Docker Compose Containers containers plot the results in the diagrams:**
```bash
python scripts/docker_collect.py
```
*Graphs will be saved in `experiments/results/`.*
