import torch
import torch.nn as nn
import torch.optim as optim
from ml.model import SimpleCNN
from utils.logger import MetricsLogger
from ml.aggregator import aggregate, calculate_mse
from config import LEARNING_RATE, MOMENTUM, DEVICE, LOG_DIR
import matplotlib.pyplot as plt
import os


class DecentralizedTrainer:
    def __init__(self, node_id, train_loader, test_loader, lr=LEARNING_RATE, log_dir=LOG_DIR):
        self.device = DEVICE
        self.node_id = node_id

        # Initializing the model and optimizer
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=MOMENTUM)

        # Data loading
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Turns the dataset into a generator, from which we will extract data one batch at a time using the function
        self.train_iterator = iter(self.train_loader)

        # Import logger
        self.logger = MetricsLogger(node_id=node_id, log_dir=log_dir)

        self.global_step = 0
        self.current_epoch = 0
        self.current_accuracy = 0.0


    def final_visual_check(self, image_tensor, label, target_digit, results_dir):
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0))
            prediction = output.argmax(dim=1).item()

        plt.figure(figsize=(5, 5))
        plt.imshow(image_tensor.squeeze().numpy(), cmap='gray')

        color = 'green' if prediction == label else 'red'
        plt.title(f"Node ID: {self.node_id}\nTarget Digit: {target_digit}\nPredicted: {prediction}",
                  fontsize=12, color=color)
        plt.axis('off')

        save_path = os.path.join(results_dir, f"final_check_node_{self.node_id}.png")
        plt.savefig(save_path)
        plt.close()
        return prediction

    # Performs num_batches of training steps on local data
    def train_step(self, num_batches=1):
        self.model.train()
        running_loss = 0.0

        for _ in range(num_batches):
            try:
                data, target = next(self.train_iterator)
            except StopIteration:
                self.current_epoch += 1
                self.train_iterator = iter(self.train_loader)
                data, target = next(self.train_iterator)

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            self.global_step += 1

        avg_loss = running_loss / num_batches
        return avg_loss

    # Validates the model on the global test set & returns accuracy and average loss
    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        self.current_accuracy = accuracy

        # Write results in logger (for DevOps)
        self.logger.log_step(
            epoch=self.current_epoch,
            global_step=self.global_step,
            loss=avg_loss,
            accuracy=accuracy
        )

        return accuracy, avg_loss

    # Exporting scales for the network module
    def get_weights(self):
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def aggregate_weights(self, peer_weights, peer_acc, alpha=None):
        my_acc = self.current_accuracy

        if peer_acc > my_acc:
            alpha = 0.5
        else:
            alpha = max(0.05, 0.5 - (my_acc - peer_acc) / 100.0)

        peer_weights_on_device = {k: v.to(self.device) for k, v in peer_weights.items()}

        new_weights = aggregate(self.model, peer_weights_on_device, alpha=alpha)

        self.model.load_state_dict(new_weights)

        return {k: v.cpu().clone() for k, v in new_weights.items()}
