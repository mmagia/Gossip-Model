This project implements a peer-to-peer (P2P) training system where multiple nodes collaborate to train a shared machine learning model without any central server. 5 nodes train locally to classify digits from 0-9, however each node learns only how to classify 2 digits and then exchange its knowledge by propagating weights through the network via gRPC protocol to other nodes.

Thus every peer is able to classify the whole dataset, and its accuracy converges. Moreover, network problems are sumilated to check the concept in real world scenario: network delays and artificial packet losses were introduced. We also implemented centralized logging and containerized each node via Docker.

##  Key Features
* **Fully Decentralized:** No master node or central parameter server.
* **Gossip Protocol:** Parameters are shared between neighbors via asynchronous updates.
* **Non-IID Support:** Nodes can train on different data shards (e.g., specific MNIST digits) and still converge.
* **Divergence Tracking:** Built-in monitoring of model discrepancy using MSE metrics.


## Network Simulation & Robustness
To validate the system under real-world conditions, the following parameters are simulated (see `src/config.py`):
- **Packet Loss:** Artificial drops of gossip messages to test eventual convergence.
- **Asynchronous Jitter:** Nodes communicate at slightly randomized intervals.
- **Straggler Simulation:** Specific nodes (e.g., Node 2) are intentionally delayed to test the network's resilience to slow peers.


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
```bash
git clone git@github.com:mmagia/Gossip-Model.git
```

2. **Build and start all nodes:**
```bash
docker compose up --build
```
Once in a while centalized logging will happen, and you will see the batches of information provided by nodes in your terminal.

3. **After stopping the Docker Compose Containers they run trained models on validation images. To plot the results in the diagrams, execute the following command in your terminal:**
```bash
python scripts/docker_collect.py
```
*Graphs will be saved in `experiments/results/`.*

## Note on Data Persistence
The following directories are used for data exchange between Docker and the Host machine:
- `/logs`: Temporary storage for raw JSON metrics (cleared on each run).
- `/experiments/results`: Target for generated plots and validation images (`.png`).
- `/data`: MNIST dataset cache to avoid re-downloading on each container start.

> **Note:** These directories are included in `.gitignore` to keep the repository clean. Results are meant to be generated locally after each experiment.
