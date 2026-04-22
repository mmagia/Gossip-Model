import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import Counter
import os
from config import BATCH_SIZE, NUM_NODES, DATA_PATH


def get_dataloader(node_id, num_nodes=NUM_NODES, batch_size=BATCH_SIZE, is_train=True, data_path=DATA_PATH):
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        dataset = datasets.MNIST(
            root=data_path, 
            train=is_train, 
            download=True, 
            transform=transform
        )
    except Exception:
        return None

    if not is_train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    labels = np.array(dataset.targets)
    num_classes = 10
    
    classes_per_node = max(1, num_classes // num_nodes)
    start_class = node_id * classes_per_node
    
    if node_id == num_nodes - 1:
        target_classes = list(range(start_class, num_classes))
    else:
        target_classes = list(range(start_class, start_class + classes_per_node))

    indices = np.where(np.isin(labels, target_classes))[0]
    node_subset = Subset(dataset, indices)

    print(f" [DATASET] Node {node_id} | Classes: {target_classes} | Samples: {len(node_subset)}")

    return DataLoader(node_subset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    loader = get_dataloader(node_id=0, num_nodes=5, batch_size=16)
    if loader:
        images, targets = next(iter(loader))
        print(f"Check: {targets.unique().tolist()}")