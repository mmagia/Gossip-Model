import sys
import os
import torch
import unittest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ml.aggregator import aggregate, calculate_mse
from ml.model import SimpleCNN

#Unit tests for model aggregator
class TestAggregator(unittest.TestCase):
    def setUp(self):
        self.model_a = SimpleCNN()
        self.model_b = SimpleCNN()

        self.zeros_weights = {k: torch.zeros_like(v) for k, v in self.model_a.state_dict().items()}
        self.ones_weights = {k: torch.ones_like(v) for k, v in self.model_a.state_dict().items()}

    def test_simple_averaging(self):
        self.model_a.load_state_dict(self.ones_weights)
        new_weights = aggregate(self.model_a, self.zeros_weights, alpha=0.5)

        sample_layer = new_weights['conv1.weight']
        mean_val = torch.mean(sample_layer).item()

        self.assertAlmostEqual(mean_val, 0.5, places=5, msg="Averaging logic is broken!")

    def test_weighted_averaging(self):
        self.model_a.load_state_dict(self.ones_weights)

        # W_new = 1.0 * (1 - 0.2) + 0.0 * 0.2 = 0.8
        new_weights = aggregate(self.model_a, self.zeros_weights, alpha=0.2)

        sample_layer = new_weights['conv1.weight']
        mean_val = torch.mean(sample_layer).item()

        self.assertAlmostEqual(mean_val, 0.8, places=5)

    def test_mse_calculation(self):
        mse_val = calculate_mse(self.ones_weights, self.zeros_weights)
        self.assertAlmostEqual(mse_val, 1.0, places=5)

        mse_identical = calculate_mse(self.ones_weights, self.ones_weights)
        self.assertEqual(mse_identical, 0.0)

    def test_device_consistency(self):
        cpu_weights = self.ones_weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_a.to(device)

        try:
            new_weights = aggregate(self.model_a, cpu_weights, alpha=0.5)
            self.assertIsNotNone(new_weights)
        except Exception as e:
            self.fail(f"Aggregation failed during cross-device transfer: {e}")


if __name__ == "__main__":
    unittest.main()