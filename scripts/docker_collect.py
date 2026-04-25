#!/usr/bin/env python3
"""
Post-experiment script: collect logs from all Docker containers
and generate convergence plots (accuracy, loss, divergence).
"""
import os
import sys
import glob
import sys

# Dynamically add 'src' to PYTHONPATH to access global config and paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Add 'experiments' to PYTHONPATH to import visualization utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments')))


from config import LOG_DIR, RESULTS_DIR
from plot_results import load_logs, plot_metrics
from analyze_divergence import plot_divergence

def collect_and_plot():
    """
    Main entry point for data aggregation. 
    Expects JSON logs to be present in the mounted LOG_DIR.
    """
    # Safety check: Ensure the log directory exists before processing
    if not os.path.exists(LOG_DIR):
        print(f"❌ Directory {LOG_DIR} not found. Run 'docker-compose up' first.")
        return

    # Identify all node metric files using glob patterns
    log_files = glob.glob(os.path.join(LOG_DIR, "node_*_metrics.json"))
    if not log_files:
        print(f"❌ No node_*_metrics.json files found in {LOG_DIR}")
        return

    print(f"✅ Found {len(log_files)} log files")
    
    # Load raw JSON data into memory
    data = load_logs(LOG_DIR)
    
    if data:
        print(f"Generating plots for {len(data)} nodes...")
        # Create accuracy over time plot
        plot_metrics(data, save_dir=RESULTS_DIR)
        # Create model divergence (standard deviation of accuracy) plot
        plot_divergence(data, save_dir=RESULTS_DIR)
        print(f"✅ Plots saved to {RESULTS_DIR}")
    else:
        print("❌ Failed to load metrics data")

if __name__ == "__main__":
    collect_and_plot()