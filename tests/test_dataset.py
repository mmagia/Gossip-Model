import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ml.dataset import get_dataloader  

# unit tests for model datasets
def test_non_iid_distribution():
    num_nodes = 5
    batch_size = 100
    
    loader_node_0 = get_dataloader(node_id=0, num_nodes=num_nodes, batch_size=batch_size, is_train=True)
    loader_node_1 = get_dataloader(node_id=1, num_nodes=num_nodes, batch_size=batch_size, is_train=True)
    
    def get_labels_from_loader(loader, num_batches=5):
        all_labels = set()
        iterator = iter(loader)
        for _ in range(num_batches):
            _, labels = next(iterator)
            all_labels.update(labels.tolist())
        return all_labels

    labels_0 = get_labels_from_loader(loader_node_0)
    labels_1 = get_labels_from_loader(loader_node_1)
    
    intersection = labels_0.intersection(labels_1)
    
    assert len(intersection) == 0
    assert labels_0 == {0, 1}
    assert labels_1 == {2, 3}

def test_global_test_loader():
    test_loader = get_dataloader(node_id=0, num_nodes=5, batch_size=100, is_train=False)
    
    all_labels = set()
    iterator = iter(test_loader)
    for _ in range(20):
        _, labels = next(iterator)
        all_labels.update(labels.tolist())
    
    assert len(all_labels) == 10

if __name__ == "__main__":
    try:
        test_non_iid_distribution()
        test_global_test_loader()
        print("Success")
    except AssertionError as e:
        print(f"Failure: {e}")
    except Exception as e:
        print(f"Error: {e}")