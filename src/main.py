import os
import multiprocessing as mp
import time
from ml.trainer import DecentralizedTrainer
from network.node import Node
from ml.dataset import get_dataloader
from config import GOSSIP_INTERVAL

def run_node_process(node_id: int, port: str, peers: list, num_nodes: int = 5):
    """Launch a single node process with its own data, model, and gossip loop."""
    print(f"Starting Node {node_id} | port {port} | peers: {peers}")

    # Load Non-IID data splits (each node sees only subset of digit classes)
    train_loader = get_dataloader(node_id=node_id, num_nodes=num_nodes, is_train=True)
    test_loader = get_dataloader(node_id=node_id, num_nodes=num_nodes, is_train=False)

    # Initialize local model trainer
    ml_model = DecentralizedTrainer(
        node_id=node_id,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=0.01
    )

    # Create gossip node with gRPC server/client logic
    node = Node(
        node_id=node_id,
        port=port,
        peers=peers,
        ml_model=ml_model
    )

    # Start infinite training + gossip loop
    node.run(gossip_interval=GOSSIP_INTERVAL)

if __name__ == "__main__":
    # Force 'spawn' method for multiprocessing
    mp.set_start_method('spawn', force=True)

    # Read configuration from environment variables (set by Docker)
    node_id = int(os.environ.get("NODE_ID", 0))
    port = os.environ.get("PORT", "50051")
    peers_str = os.environ.get("PEERS", "")
    
    if peers_str:
        # Parse comma-separated peer list: "node1:50052,node2:50053" -> list
        peers = peers_str.split(",")
    else:
        # Fallback for local non-Docker execution
        all_ports = ["50051", "50052", "50053", "50054", "50055"]
        peers = [f"localhost:{p}" for p in all_ports if p != port]

    num_nodes = int(os.environ.get("NUM_NODES", 5))

    print(f"[Main] Node {node_id} starting with PORT={port}, PEERS={peers}")

    # Launch the node process
    run_node_process(node_id, port, peers, num_nodes)