"""
Configuration and hyperparameters for IDSGAN.

Based on Section 4.1 of the paper:
- Batch size: 64
- Epochs: 100
- Learning rates: 0.0001 for both G and D
- Weight clipping: 0.01
- Noise dimension: 9
- Optimizer: RMSProp (WGAN)
"""

import os
import torch

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Dataset files
TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IDSGAN Hyperparameters (Section 4.1)
BATCH_SIZE = 64
EPOCHS = 100
LR_GENERATOR = 0.0001
LR_DISCRIMINATOR = 0.0001
CLIP_VALUE = 0.01          # Weight clipping threshold for WGAN
NOISE_DIM = 9              # Dimension of the noise vector
N_CRITIC = 5               # Discriminator steps per generator step (WGAN)

# Generator Architecture
# Input: original record (NUM_FEATURES) + noise (NOISE_DIM)
# 5 linear layers: input -> 256 -> 256 -> 256 -> 256 -> NUM_FEATURES
G_HIDDEN_DIM = 256
G_NUM_LAYERS = 5

# Discriminator Architecture
# Input: traffic record (NUM_FEATURES)
# Layers: NUM_FEATURES -> 256 -> 256 -> 1
D_HIDDEN_DIM = 256

# Feature dimensions (after one-hot encoding)
# Original: 41 features
# After encoding: 41 - 3 (categorical) + 3 (protocol) + 70 (service) + 11 (flag) = 122
NUM_ORIGINAL_FEATURES = 41
NUM_FEATURES = 122  # After one-hot encoding

# Binary feature threshold for post-processing
BINARY_THRESHOLD = 0.5

# Random seed for reproducibility
RANDOM_SEED = 42

# Attack categories
ATTACK_CATEGORIES = ["DoS", "Probe", "U2R", "R2L"]
# U2R and R2L are grouped together for training/evaluation (per paper Section 4.2)
ATTACK_GROUPS = {
    "DoS": ["DoS"],
    "U2R_R2L": ["U2R", "R2L"],
}

# IDS model names
IDS_MODELS = ["SVM", "NB", "MLP", "LR", "DT", "RF", "KNN"]

# Create directories
for d in [DATA_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)
