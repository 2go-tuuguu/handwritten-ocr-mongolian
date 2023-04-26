import torch

DATA_DIR = "data"
TRAIN_DIR = "train"
TEST_DIR = "small_test"
BATCH_SIZE = 64
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 300
NUM_WORKERS = 2
EPOCH_END = 260
EPOCH_START = 0
# apple m1
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# DEVICE = "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))
CHECKPOINT_PATH = "checkpoint_200.pth"