import torch
import torch.nn as nn
import random
import numpy as np
from src.config import SEED


# Lightweight convolutional neural network for image classification (MNIST)
# Specifically designed with a small number of parameters (~1 MB)
# to avoid overloading the network with frequent Gossip exchanges
class SimpleCNN(nn.Module):


    def __init__(self, num_classes=10, input_channels=1):
        super(SimpleCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After the previous step, we're left with 32 channels, each 7x7 pixels in size
        # To fit them into Linear, we need to multiply these numbers together.
        # The layer will transform these 1568 features into 128 abstract thoughts
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # The tensor after convolution has the shape [Batch, 32, 7, 7], meaning it is three-dimensional
        # The .view() transforms it into a vector [Batch, 1568]
        x = x.view(x.size(0), -1)

        # At the output we get a tensor of size [Batch, 10]
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x


# A utility for estimating the size of a model in megabytes (important for networks)
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


# Fixes random number generators to ensure reproducibility of experiments
# Call at the very beginning of main.py or when initializing the Trainer
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"🌱 Seed зафиксирован: {seed}")