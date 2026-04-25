import os
import sys
import json
import glob
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_DIR, RESULTS_DIR

#Helper function to load nodes logs in json format
def load_logs(log_dir="logs"):
    log_files = glob.glob(os.path.join(log_dir, "*.json"))

    if not log_files:
        print(f"Directory {log_dir} is empty or does not exist")
        return None

    data = {}
    for file_path in log_files:
        # Extract node name from filename
        filename = os.path.basename(file_path)
        node_name = filename.replace("_metrics.json", "")

        with open(file_path, 'r', encoding='utf-8') as f:
            data[node_name] = json.load(f)

    return data

#Function to plot models accuracy
def plot_metrics(data, save_dir="experiments/results"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for node_name, metrics in data.items():
        steps = [entry["global_step"] for entry in metrics]
        accuracy = [entry["accuracy"] for entry in metrics]
        # Plot line for each node
        plt.plot(steps, accuracy, label=node_name, marker='o', markersize=3)

    plt.title("Model Convergence: Accuracy", fontsize=14)
    plt.xlabel("Global Training Step", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    acc_path_png = os.path.join(save_dir, "accuracy_convergence.png")
    plt.savefig(acc_path_png, dpi=300, bbox_inches='tight')
    print(f"Accuracy chart saved: {acc_path_png}")
    plt.close()

    plt.figure(figsize=(10, 6))
    for node_name, metrics in data.items():
        steps = [entry["global_step"] for entry in metrics]
        loss = [entry["loss"] for entry in metrics]
        plt.plot(steps, loss, label=node_name, marker='x', markersize=3)

    plt.title("Model Convergence: Loss", fontsize=14)
    plt.xlabel("Global Training Step", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    loss_path_png = os.path.join(save_dir, "loss_convergence.png")
    plt.savefig(loss_path_png, dpi=300, bbox_inches='tight')
    print(f"Loss chart saved: {loss_path_png}")
    plt.close()


if __name__ == "__main__":
    logs_data = load_logs(log_dir=LOG_DIR)
    if logs_data:
        plot_metrics(logs_data, save_dir=RESULTS_DIR)
        print("Visualization completed!")