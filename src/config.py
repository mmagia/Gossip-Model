import torch
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_EPOCHS = 5
ALPHA = 0.5

NUM_CLASSES = 10
INPUT_CHANNELS = 1
IMAGE_SIZE = 28

NUM_NODES = 5
BASE_PORT = 5000
HOST = "127.0.0.1"

DATA_PATH = os.path.join(ROOT_DIR, "data")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


GOSSIP_INTERVAL = 3.0

GOSSIP_JITTER_MIN = -0.4
GOSSIP_JITTER_MAX = 0.4

PACKET_LOSS_PROB = 0.05

OUTGOING_DELAY_MIN = 0.03
OUTGOING_DELAY_MAX = 0.20

ENABLE_SLOW_NODE = True
SLOW_NODE_ID = 2
SLOW_NODE_EXTRA_DELAY_MIN = 0.15
SLOW_NODE_EXTRA_DELAY_MAX = 0.45

GRPC_TIMEOUT = 5.0
