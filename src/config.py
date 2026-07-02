"""
Global configuration for the Disaster Assessment System.
"""

import os
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
MODELS_DIR = "models"

### SIAMESE
IMAGE_SIZE_SIAMESE = 512
DILATION_KERNEL_SIZE = 5
SIAMESE_WEIGHTS_PATH = os.path.join(MODELS_DIR,"best_siamese_model.pth")

### CNN
IMAGE_SIZE_CNN = 224
PADDING_RATIO = 0.10
CLASS_NAMES = ["no-damage", "damaged", "destroyed",]
CLASS_TO_IDX = {"no-damage": 0, "damaged": 1, "destroyed": 2,}
MERGE_MAPPING = {"no-damage": "no-damage",  "minor-damage": "damaged", "major-damage": "damaged", "destroyed": "destroyed",}
IGNORED_CLASSES = {"un-classified"}
CLASSIFIER_WEIGHTS_PATH = os.path.join( MODELS_DIR,"best_classifier_model.pth")

### RAG
