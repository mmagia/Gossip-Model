import json
import os
from datetime import datetime


# Centralized metric collection for system validation and performance analysis
# Used by DevOps to aggregate results from distributed nodes for plotting
class MetricsLogger:


    def __init__(self, node_id, log_dir="logs"):
        self.node_id = node_id
        self.log_dir = log_dir
        self.metrics = []

        # Ensure the target directory exists for volume mounting in Docker
        os.makedirs(self.log_dir, exist_ok=True)
        self.filename = os.path.join(self.log_dir, f"node_{node_id}_metrics.json")
        # Clean up old logs from previous runs to avoid data contamination
        if os.path.exists(self.filename):
            os.remove(self.filename)

    # Records a snapshot of the current training state
    def log_step(self, epoch, global_step, loss, accuracy):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "global_step": global_step,
            "loss": round(loss, 4),
            "accuracy": round(accuracy, 2)
        }
        self.metrics.append(entry)
        self._save()

    # Flushes metrics to disk in JSON format for external analysis tools
    def _save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=4)