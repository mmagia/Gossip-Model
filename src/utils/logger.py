import json
import os
from datetime import datetime


# A class for saving metrics (Loss, Accuracy) to a JSON file
# Useful for devOps to take these files and draw graphs
class MetricsLogger:


    def __init__(self, node_id, log_dir="logs"):
        self.node_id = node_id
        self.log_dir = log_dir
        self.metrics = []

        os.makedirs(self.log_dir, exist_ok=True)
        self.filename = os.path.join(self.log_dir, f"node_{node_id}_metrics.json")
        if os.path.exists(self.filename):
            os.remove(self.filename)

    # Adds a new entry and saves the file
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

    # Overwrites the JSON file with the updated list of metrics
    def _save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=4)