import os
import multiprocessing as mp
import sys
import time
import signal
from ml.trainer import DecentralizedTrainer
from network.node import Node
from ml.dataset import get_dataloader
from config import GOSSIP_INTERVAL
from torchvision import datasets, transforms
from config import RESULTS_DIR


def _signal_handler(signum, frame):
    print(f"\n[Node] Received SIGTERM, initiating shutdown...", flush=True)
    raise SystemExit("Received SIGTERM")

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

    target_digit = (node_id * 2 + 3) % 10
    test_ds = datasets.MNIST('./data', train=False, download=True,
                             transform=transforms.Compose([transforms.ToTensor()]))

    example_img, example_label = None, None
    for img, lbl in test_ds:
        if lbl == target_digit:
            example_img, example_label = img, lbl
            break

    signal.signal(signal.SIGTERM, _signal_handler)

    # Start infinite training + gossip loop
    try:
        node.run(gossip_interval=GOSSIP_INTERVAL)
    except (KeyboardInterrupt, SystemExit):
        print(f"\n[Node {node_id}] Stopping training...", flush=True)
    finally:
        if example_img is not None:
            print(f"[Node {node_id}] Running FINAL EXAM for digit {target_digit}...", flush=True)
            prediction = ml_model.final_visual_check(
                example_img, example_label, target_digit, RESULTS_DIR
            )
            print(f"NODE {node_id} RESULT: Predicted {prediction} !!!", flush=True)

            import time
            time.sleep(1)

        node.stop_server()
        print(f"[Node {node_id}] Shutdown complete.", flush=True)
        sys.exit(0)

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