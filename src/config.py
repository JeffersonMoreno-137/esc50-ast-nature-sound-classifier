"""Configuration constants and hyperparameters for AST Nature Sound Classification."""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
DOCS_DIR = BASE_DIR / "docs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
OUTPUT_DIR = BASE_DIR / "results_ast_nature"

# Dataset settings
DATASET_NAME = "confit/esc50-demo"
DATASET_FOLD = "fold1"

# Audio extraction parameters
TARGET_SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
MAX_SPECTROGRAM_LENGTH = 1024
NORMALIZE_AUDIO = True
NORM_MEAN = 0.0
NORM_STD = 0.5

# Model configuration
MODEL_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
NUM_CLASSES = 4

# Class labels and mapping
TARGET_NAMES = ["Aves", "Agua", "Viento", "Mamíferos"]
TARGET_NAMES_EN = ["Birds", "Water", "Wind", "Mammals"]

NATURE_MAP = {
    # 0. Birds / Aves
    "chirping_birds": 0,
    "crow": 0,
    "rooster": 0,
    "hen": 0,
    # 1. Water / Agua
    "rain": 1,
    "sea_waves": 1,
    "water_drops": 1,
    "pouring_water": 1,
    # 2. Wind / Viento
    "wind": 2,
    "thunderstorm": 2,
    # 3. Mammals / Mamíferos
    "dog": 3,
    "cat": 3,
    "pig": 3,
    "cow": 3,
    "sheep": 3,
}

LABEL2ID = {name: i for i, name in enumerate(TARGET_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(TARGET_NAMES)}

# Class weights for cost-sensitive CrossEntropyLoss
# Compensates for sample distribution: Aves (128), Agua (128), Viento (64), Mamíferos (160)
CLASS_WEIGHTS = [1.0, 1.0, 2.0, 0.8]

# Training defaults
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 42

def get_device():
    """Detect available compute device adhering to PyTorch best practices."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return "cpu"
