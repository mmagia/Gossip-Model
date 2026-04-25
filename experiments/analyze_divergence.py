import os
import sys
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_DIR, RESULTS_DIR

#Helper function to load nodes logs in json format
def load_all_metrics(log_dir="logs"):
    log_files = glob.glob(os.path.join(log_dir, "*.json"))
    all_data = {}

    for file_path in log_files:
        node_name = os.path.basename(file_path).replace("_metrics.json", "")
        with open(file_path, 'r', encoding='utf-8') as f:
            all_data[node_name] = json.load(f)
    return all_data

#Function which plots models divergence accross all nodes
def plot_divergence(data, save_dir="experiments/results"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    all_steps = sorted(list(set(
        entry["global_step"] for metrics in data.values() for entry in metrics
    )))
    std_accuracy = []

    for step in all_steps:
        accs_at_step = []
        for node_metrics in data.values():
            for entry in node_metrics:
                if entry["global_step"] == step:
                    accs_at_step.append(entry["accuracy"])

        if accs_at_step:
            std_accuracy.append(np.std(accs_at_step))
        else:
            std_accuracy.append(0)

    plt.plot(all_steps, std_accuracy, color='purple', linewidth=2, label='Accuracy Variance')

    plt.title("Model Divergence Over Time (Standard Deviation of Accuracy)", fontsize=14)
    plt.xlabel("Global Training Step", fontsize=12)
    plt.ylabel("Standard Deviation (%)", fontsize=12)
    plt.fill_between(all_steps, std_accuracy, alpha=0.2, color='purple')
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = os.path.join(save_dir, "model_divergence.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Divergence analysis saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    data = load_all_metrics(log_dir=LOG_DIR)

    if data:
        plot_divergence(data, save_dir=RESULTS_DIR)